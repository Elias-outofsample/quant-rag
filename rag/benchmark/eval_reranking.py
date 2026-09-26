"""Reclassement du pool dense par un cross-encodeur — l'instrument du chantier A.

Protocole pré-enregistré et publié **avant la première paire scorée** :
``rag/benchmark/PRE-ENREGISTREMENT-RERANKING-2026-09-07.md`` (commit ``187bf53``).

Ce que ce script mesure, et rien d'autre : les *B* premiers candidats du pool dense@50
sont réordonnés par ``Qwen/Qwen3-Reranker-0.6B``, les suivants gardent leur ordre dense
**derrière** eux. Le pool n'est jamais tronqué, seule sa tête est réordonnée.

**Le budget de décision est B = 10, et lui seul** — la borne de latence humaine (p95 ≤ 4,0 s)
n'admet que lui. B = 30 et B = 50 sont dérivés gratuitement des mêmes scores et publiés en
**sensibilité** : ils ne portent aucun verdict. Les laisser décider serait déplacer la borne
après avoir vu le résultat.

Hors ligne intégral, et c'est ce qui rend le chantier exécutable aujourd'hui :

  pool     ``.cache/router-retrievals-<signature>.json``  (classements figés de Qdrant)
  textes   ``.cache/chunk-index-v5-<signature>.pkl``      (ChunkIndex)
  garde    comptage sur la sélection de production        (zéro appel LLM)

**Qdrant n'est jamais ouvert** : les serveurs MCP n'ont pas à être arrêtés, aucun ``/mcp``
n'est demandé. Le pool provient du Qdrant **embarqué**, dont la recherche est **exacte** —
condition expérimentale signalée par le chantier B et inscrite au §6.4 du pré-enregistrement :
un backend HNSW approximatif rendrait un autre pool et invaliderait ces mesures.

Pourquoi un script de plus, et pas ``eval_rerank_budget.py`` : ce dernier reprend son cache
sous un nom **sans signature** (l. 61-62), et le fichier périmé du 3 septembre (corpus
``6c21d82412``) est sur le disque. Ici, **tout** porte la signature.

    .venv/bin/python rag/benchmark/eval_reranking.py --controle    # contrôle seul, 0 modèle
    .venv/bin/python rag/benchmark/eval_reranking.py               # ~40 min, reprise par question
    .venv/bin/python rag/benchmark/eval_reranking.py --limit 5     # rodage
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import corpus_overlay  # noqa: E402
import eval_rerankers  # noqa: E402
import experiment  # noqa: E402
import metrics  # noqa: E402
import pipeline  # noqa: E402
import quant_rag  # noqa: E402
from corpus import ChunkIndex  # noqa: E402

SIGNATURE = corpus_overlay.signature()
MODEL = "qwen3-0.6b"
#: Le nom porte la signature : un cache d'un autre état n'est pas rechargé par erreur,
#: il n'est simplement pas trouvé. C'est le défaut d'``eval_rerank_budget.py``, pas le nôtre.
SCORES = HERE / ".cache" / f"reranking-scores-{SIGNATURE}.json"
OUTPUT = HERE / f"results-reranking-{SIGNATURE}.json"

#: Pré-enregistrement §2. Le premier est le budget de décision ; les autres, la sensibilité.
BUDGET_DECISION = 10
BUDGETS = (10, 30, 50)
#: Pré-enregistrement §3. La convention maison (4000 / 20260901) est publiée en sensibilité.
DRAWS, SEED = 10_000, 20260907
MAISON_DRAWS, MAISON_SEED = 4_000, 20260901
#: Pré-enregistrement §4.2 — la sélection est celle de la production, pas celle du banc.
SERVIS, PER_DOCUMENT, MIN_CHARACTERS = 5, 2, quant_rag.MIN_CHARACTERS
#: Pré-enregistrement §4.2 — la garde échoue au plus petit b dont la borne haute de Wilson dépasse 15 %.
WILSON_PLAFOND = 0.15
Z = 1.96


# --------------------------------------------------------------------------- outillage

def wilson_haut(b: int, n: int) -> float:
    """Borne haute de l'IC95 de Wilson pour b/n."""
    if n == 0:
        return 1.0
    p = b / n
    d = 1 + Z * Z / n
    centre = (p + Z * Z / (2 * n)) / d
    rayon = (Z * ((p * (1 - p) / n + Z * Z / (4 * n * n)) ** 0.5)) / d
    return centre + rayon


def seuil_de_garde(n: int) -> int:
    """Le plus petit b dont la borne haute de Wilson de b/n dépasse 15 %."""
    b = 0
    while b <= n:
        if wilson_haut(b, n) > WILSON_PLAFOND:
            return b
        b += 1
    return n + 1


def pages_libres_mo() -> float | None:
    """L'état mémoire de la machine. Une latence publiée sans lui ne mesure rien : les
    28,8 s du 3 septembre 2026 étaient du swap (``eval_rerank_latency.py``)."""
    try:
        sortie = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=10).stdout
    except Exception:  # noqa: BLE001
        return None
    taille, libres = 4096, None
    for ligne in sortie.splitlines():
        if "page size of" in ligne:
            taille = int(ligne.split("page size of")[1].split("bytes")[0].strip())
        if ligne.startswith("Pages free"):
            libres = int(ligne.split(":")[1].strip().rstrip("."))
    return round(libres * taille / 1_048_576, 1) if libres is not None else None


def reordonner(rows: list[dict], scores: dict[str, float], budget: int) -> list[dict]:
    """Les ``budget`` premiers candidats réordonnés par score ; la queue garde l'ordre dense.

    Le départage est **explicite** — ``(-score, rang_dense, chunk_id)`` — et non laissé à la
    stabilité du tri de Python. ``quant_rag._rerank`` s'en remet aujourd'hui à cette stabilité ;
    l'écrire supprime une dépendance à un détail de langage.
    """
    tete, queue = rows[:budget], rows[budget:]
    classee = sorted(
        tete,
        key=lambda r: (-scores.get(r["chunk_id"], float("-inf")), r["rang_dense"], r["chunk_id"]),
    )
    sortie = []
    for position, row in enumerate(classee + queue, 1):
        sortie.append({**row, "rang_final": position,
                       "rerank_score": scores.get(row["chunk_id"]) if position <= len(tete) else None,
                       "reclasse": position <= len(tete)})
    return sortie


def servis(rows: list[dict]) -> list[dict]:
    """La sélection **de production** : ``_select`` de ``quant_rag``, pas ``build_context``.

    La garde porte sur ce que l'utilisateur reçoit : plafond de 2 par document, seuil de
    250 caractères, filtre d'en-têtes, suppression des quasi-doublons.
    """
    return quant_rag._select(rows, SERVIS, PER_DOCUMENT, True, MIN_CHARACTERS)


def or_servi(rows: list[dict], gold: set[str]) -> bool:
    return any(r["chunk_id"] in gold for r in servis(rows))


# --------------------------------------------------------------------------- exécution

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--controle", action="store_true", help="contrôle de ligne de base seul, aucun modèle chargé")
    parser.add_argument("--limit", type=int, default=None, help="sous-échantillon, pour un rodage")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--echantillon-latence", type=int, default=8)
    args = parser.parse_args()

    debut = time.perf_counter()
    print(f"corpus {SIGNATURE} · modèle {eval_rerankers.MODELS[MODEL]} · device {quant_rag.device()}")
    print(f"mémoire libre au départ : {pages_libres_mo()} Mo")

    index = ChunkIndex.load(verbose=False)
    items = experiment.load_items(index)
    cache = experiment.cached_rankings()

    # Le pool, avec texte ET content_type : la sélection de production filtre les en-têtes.
    pools: dict[str, list[dict]] = {}
    for item in items:
        rows = []
        for rang, (chunk_id, document_id, score) in enumerate(cache[item["key"]]["dense"], 1):
            src = index.get(chunk_id) or {}
            rows.append({"chunk_id": chunk_id, "document_id": document_id, "score": score,
                         "score_dense": score, "rang_dense": rang,
                         "text": src.get("text", ""), "content_type": src.get("content_type")})
        pools[item["key"]] = rows

    # ---- 1. contrôle de ligne de base, AVANT tout chargement de modèle (§11.1)
    reference = [{"bench": i["bench"], "key": i["key"], "qid": i["qid"], "kind": i.get("kind", "single"),
                  "none": experiment.measure(i, pools[i["key"]])} for i in items]
    ecarts = eval_rerankers.controle_ligne_de_base(
        experiment.summarise(reference, ["none"], reference="none"), ["none"])
    print("\n=== contrôle — le pool dense reproduit-il results-router-v3.json ? ===")
    if ecarts:
        for ligne in ecarts:
            print(f"  ÉCART  {ligne}")
        sys.exit("le pool a dérivé : on ne mesurerait pas un reranker, on mesurerait la dérive.")
    print("  les trois vues reproduisent la ligne de base à l'identique")
    if args.controle:
        return

    if args.limit:
        items = items[:args.limit]

    # ---- 2. scoring du pool complet, une seule passe, reprise par question (§11.2)
    scores: dict[str, dict[str, float]] = {}
    if SCORES.exists():
        scores = json.loads(SCORES.read_text(encoding="utf-8"))
        print(f"\nreprise : {len(scores)} questions déjà scorées dans {SCORES.name}")
    manquantes = [i for i in items if i["key"] not in scores]
    if manquantes:
        print(f"scoring de {len(manquantes)} questions (~{len(manquantes) * 15 / 60:.0f} min)")
        scorer = eval_rerankers.load(MODEL)
        for n, item in enumerate(manquantes, 1):
            rows = [r for r in pools[item["key"]] if r["text"]]
            textes = [r["text"][:eval_rerankers.TEXT_CHARACTERS] for r in rows]
            valeurs = scorer.score(pipeline.query_of(item), textes)
            scores[item["key"]] = {r["chunk_id"]: float(s) for r, s in zip(rows, valeurs)}
            SCORES.parent.mkdir(parents=True, exist_ok=True)
            SCORES.write_text(json.dumps(scores), encoding="utf-8")
            if n % 10 == 0 or n == len(manquantes):
                reste = (time.perf_counter() - debut) / n * (len(manquantes) - n) / 60
                print(f"  {n}/{len(manquantes)}  reste ~{reste:.0f} min  ·  {pages_libres_mo()} Mo libres")
        # Le modèle est relâché AVANT la phase de latence : la machine tourne à ~70 Mo de
        # pages libres pendant le scoring, et les 28,8 s du 3 septembre étaient du swap.
        del scorer
        import gc

        import torch

        gc.collect()
        if quant_rag.device() == "mps":
            torch.mps.empty_cache()

    # ---- 3. dérivation des budgets, et le bras saboté (§11.3, §11.5)
    alea = random.Random(SEED)
    sabotes: dict[str, dict[str, float]] = {}
    for item in items:
        cles = list(scores[item["key"]])
        valeurs = list(scores[item["key"]].values())
        alea.shuffle(valeurs)
        sabotes[item["key"]] = dict(zip(cles, valeurs))

    configs = ["none"] + [f"b{b}" for b in BUDGETS] + ["sabote"]
    par_question = []
    for item in items:
        rows = pools[item["key"]]
        ligne = {"bench": item["bench"], "key": item["key"], "qid": item["qid"],
                 "kind": item.get("kind", "single"), "none": experiment.measure(item, rows)}
        for b in BUDGETS:
            ligne[f"b{b}"] = experiment.measure(item, reordonner(rows, scores[item["key"]], b))
        ligne["sabote"] = experiment.measure(item, reordonner(rows, sabotes[item["key"]], BUDGET_DECISION))
        par_question.append(ligne)

    resultat = experiment.summarise(par_question, configs, reference="none")
    experiment.print_summary(resultat, configs, "none")

    # L'intervalle du pré-enregistrement, et celui de la convention maison en sensibilité.
    resultat["paired_preenregistre"] = {}
    resultat["paired_convention_maison"] = {}
    for bench in experiment.BENCHES:
        sous = [r for r in par_question if bench == "pooled" or r["bench"] == bench]
        if not sous:
            continue
        base = [r["none"]["ndcg"] for r in sous]
        resultat["paired_preenregistre"][bench] = {
            c: metrics.paired_delta(base, [r[c]["ndcg"] for r in sous], draws=DRAWS, seed=SEED)
            for c in configs if c != "none"}
        resultat["paired_convention_maison"][bench] = {
            c: metrics.paired_delta(base, [r[c]["ndcg"] for r in sous], draws=MAISON_DRAWS, seed=MAISON_SEED)
            for c in configs if c != "none"}

    # ---- 4. la garde, hors ligne, sur la sélection de production (§11.4)
    golds = {i["key"]: set(i["gold_chunks"]) for i in items}
    population, perdues, gagnees = [], [], []
    for item in items:
        cle = item["key"]
        rows = pools[cle]
        avant = or_servi(rows, golds[cle])
        apres = or_servi(reordonner(rows, scores[cle], BUDGET_DECISION), golds[cle])
        if avant:
            population.append(cle)
            if not apres:
                perdues.append(cle)
        elif apres:
            gagnees.append(cle)
    n, b = len(population), len(perdues)
    seuil = seuil_de_garde(n)
    garde = {"population_servie_dense": n, "population_preenregistree": 76,
             "perdues": perdues, "b": b, "gagnees": gagnees, "n_gagnees": len(gagnees),
             "seuil_echec": seuil, "seuil_preenregistre": 6,
             "wilson_haut_de_b": round(wilson_haut(b, n), 4),
             # À très petit n, aucun b ne passe (seuil 0) : la garde ne discrimine plus rien.
             # Elle n'est informative qu'à partir du n où b = 0 franchit la borne.
             "informative": seuil >= 1,
             "echoue": seuil >= 1 and b >= seuil}
    print(f"\n=== garde (hors ligne, sélection de production) ===")
    print(f"  population servie sous dense : {n} (pré-enregistré : 76)")
    print(f"  perdues b = {b}  ·  seuil d'échec b >= {seuil}  ·  borne haute Wilson {wilson_haut(b, n) * 100:.2f} %")
    print(f"  gagnées (or non servi qui le devient) : {len(gagnees)}")
    if not garde["informative"]:
        print(f"  population trop petite (n={n}) : la garde ne discrimine rien, verdict non lisible")
    else:
        print(f"  VERDICT GARDE : {'ÉCHEC' if garde['echoue'] else 'franchie'}")

    # ---- 5. la latence, à chaud, avec l'état mémoire (§11.6)
    latence = {"memoire_libre_mo_avant": pages_libres_mo()}
    if args.echantillon_latence:
        scorer = eval_rerankers.load(MODEL)
        echantillon = items[:args.echantillon_latence]
        rows0 = [r for r in pools[echantillon[0]["key"]] if r["text"]][:BUDGET_DECISION]
        scorer.score("réchauffement", [r["text"][:eval_rerankers.TEXT_CHARACTERS] for r in rows0])
        temps = []
        for item in echantillon:
            rows = [r for r in pools[item["key"]] if r["text"]][:BUDGET_DECISION]
            depart = time.perf_counter()
            scorer.score(item.get("retrieval_query") or item["question"],
                         [r["text"][:eval_rerankers.TEXT_CHARACTERS] for r in rows])
            temps.append(time.perf_counter() - depart)
        temps.sort()
        latence.update({"budget": BUDGET_DECISION, "n": len(temps),
                        "p50": round(statistics.median(temps), 2),
                        "p95": round(temps[min(len(temps) - 1, int(0.95 * len(temps)))], 2),
                        "max": round(temps[-1], 2), "moyenne": round(statistics.mean(temps), 2),
                        "memoire_libre_mo_apres": pages_libres_mo(),
                        "borne_preenregistree_p95": 4.0})
        latence["franchit_la_borne"] = latence["p95"] <= 4.0
        print(f"\n=== latence, budget {BUDGET_DECISION}, à chaud, n={len(temps)} ===")
        print(f"  p50 {latence['p50']} s · p95 {latence['p95']} s · max {latence['max']} s"
              f"  ·  borne 4,0 s : {'franchie' if latence['franchit_la_borne'] else 'DÉPASSÉE'}")
        print(f"  mémoire libre {latence['memoire_libre_mo_avant']} -> {latence['memoire_libre_mo_apres']} Mo")

    charge = {
        "version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "pre_enregistrement": "rag/benchmark/PRE-ENREGISTREMENT-RERANKING-2026-09-07.md",
        "corpus_state": corpus_overlay.describe(),
        "model": eval_rerankers.MODELS[MODEL],
        "settings": {"budgets": list(BUDGETS), "budget_decision": BUDGET_DECISION,
                     "pool_source": SCORES.name, "pool_cache": experiment.CACHE.name,
                     "max_length": eval_rerankers.MAX_LENGTH,
                     "text_characters": eval_rerankers.TEXT_CHARACTERS,
                     "batch": eval_rerankers.BATCH, "device": quant_rag.device(),
                     "draws": DRAWS, "seed": SEED,
                     "selection": {"passages": SERVIS, "per_document": PER_DOCUMENT,
                                   "min_characters": MIN_CHARACTERS, "dedupe": True,
                                   "source": "quant_rag._select (production)"},
                     "backend_dense": "qdrant embarqué, recherche exacte (condition expérimentale)"},
        "controle_ligne_de_base": ecarts,
        "garde": garde,
        "latence": latence,
        "resultat": resultat,
        "wins_losses": {c: experiment.wins_losses(par_question, c, "none")
                        for c in configs if c != "none"},
        "par_question": par_question,
        "wall_clock_s": round(time.perf_counter() - debut, 1),
    }
    args.output.write_text(json.dumps(charge, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nécrit : {args.output.name}  ({charge['wall_clock_s'] / 60:.0f} min)")


if __name__ == "__main__":
    main()
