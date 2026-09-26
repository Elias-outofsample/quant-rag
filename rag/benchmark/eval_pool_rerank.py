"""Phase A — l'union élargie, reclassée. Mesure shadow.

Suite conditionnelle de la Phase 0 (``eval_pool_recall.py``), qui a mesuré un **plafond de
rappel de pool** : à quel point la cible est *présente* dans un pool élargi. Elle n'a rien
dit de son **rang** une fois ce pool reclassé. C'est toute la question ici.

La réserve qui l'accompagne est mesurée, pas théorique : les 6 cibles que l'union à
profondeur 50 ajoutait, aucune des sept règles de fusion ne les remontait — R@30 ``pooled``
0,748 pour ``rrf20@50`` contre 0,774 pour le dense seul. Élargir le pool ajoute des cibles
*et* dégrade le classement. Ce chantier remplace la règle de fusion par le seul reclassement
validé en signal et porte les cibles récupérables de 6 à 12. **Il peut se conclure par un
NO-GO.**

Protocole pré-enregistré et commité avant la première paire scorée :
``rag/benchmark/RAPPORT-PHASE-A-2026-09-05.md``, Partie I.

Une seule passe de scoring. ``Qwen3-Reranker-0.6B`` score chaque paire ``(requête, texte)``
indépendamment — vérifié dans ``eval_rerankers.QwenReranker.score`` et ``quant_rag._rerank``,
pas supposé. On score donc l'union **une fois** et le classement de chacun de ses
sous-ensembles s'en déduit par tri. Cinq configurations pour le prix d'une, et une même paire
reçoit exactement le même score dans toutes : la comparaison est strictement contrôlée.

    .venv/bin/python rag/benchmark/eval_pool_rerank.py --step score --limit 10   # pilote
    .venv/bin/python rag/benchmark/eval_pool_rerank.py --step score              # ~100 min
    .venv/bin/python rag/benchmark/eval_pool_rerank.py --step report
    .venv/bin/python rag/benchmark/eval_pool_rerank.py --step report --sabotage  # A et B doivent hurler
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import corpus_overlay  # noqa: E402
import experiment  # noqa: E402
import pipeline  # noqa: E402
from corpus import ChunkIndex  # noqa: E402
from eval_rerankers import controle_ligne_de_base  # noqa: E402

SIGNATURE = corpus_overlay.signature()
CACHE = HERE / ".cache"
PARTIAL = CACHE / f"pool-rerank-partial-{SIGNATURE}.json"
OUTPUT = HERE / f"results-pool-rerank-{SIGNATURE}.json"
ARCHIVE_RERANK = HERE / "results-rerankers-dense-bb7bf33c37.json"
MODEL = "Qwen/Qwen3-Reranker-0.6B"

#: Seuil de qualité, écrit avant la mesure. Magnitude de décision de la maison (l'arbitrage
#: de la fusion exigeait +0,010 et a refusé +0,007), et le pool passe de 47 à 129 candidats.
SEUIL_NDCG = 0.010
#: Au-delà, un GO qualité devient un HOLD : gain réel, non déployable en l'état.
SEUIL_LATENCE_S = 30.0

#: Les cinq configurations. ``none`` porte le nom qu'attend ``controle_ligne_de_base``.
CONFIGS = ("none", "rr/dense50", "rr/dense100", "rr/dense50+bm25", "rr/dense100+bm25")
REFERENCE = "rr/dense50"
LIBELLE = {
    "none": "dense@50 (production)",
    "rr/dense50": "dense@50 + rr",
    "rr/dense100": "dense@100 + rr",
    "rr/dense50+bm25": "dense@50 ∪ bm25@50 + rr",
    "rr/dense100+bm25": "dense@100 ∪ bm25@50 + rr",
}


# --------------------------------------------------------------------------- les pools

def build_pools(index: ChunkIndex, items: list[dict], sabotage: bool = False) -> dict:
    """Pour chaque question : l'union à scorer, et l'appartenance de chaque configuration.

    L'ordre de l'union est délibéré — pool de référence d'abord, dans l'ordre du cache, puis
    les ajouts de profondeur, puis ceux de BM25. ``sorted`` est stable en Python : un
    sous-ensemble garde l'ordre relatif de l'union, donc le sous-ensemble ``dense@50`` a
    exactement l'ordre du cache, et les ex æquo se départagent comme dans le banc archivé.
    C'est ce qui rend le contrôle A exact plutôt qu'approximatif.
    """
    cache = experiment.cached_rankings()
    dense_atom = (json.loads((CACHE / f"pool-dense-{SIGNATURE}.json").read_text(encoding="utf-8"))
                  ["atomes"]["dense@100"])
    bm25_atom = (json.loads((CACHE / f"pool-offline-{SIGNATURE}.json").read_text(encoding="utf-8"))
                 ["atomes"]["bm25@50"])
    pools = {}
    for item in items:
        key = item["key"]
        reference = experiment.rows_from_cache(cache[key]["dense"])
        if sabotage:
            reference = reference[1:] + reference[:1]
        # L'appartenance de chaque générateur se lit dans SON atome, indépendamment de
        # l'ordre où l'union est bâtie. Un premier jet attribuait chaque chunk au premier
        # atome qui le réclamait — dense@100 passant avant BM25, les chunks trouvés par les
        # deux disparaissaient de l'appartenance « dense@50 ∪ bm25@50 ». Cela retirait
        # exactement v1/q02 et v3/m03 (l'intersection dense@100 ∩ bm25@50, mesurée à 2 en
        # Phase 0) et faisait dire à cette configuration 4 récupérations au lieu de 6.
        # Le candidat n'était pas touché — son union contient les deux atomes — mais une
        # ligne fausse dans le tableau reste une ligne fausse.
        #
        # L'ordre de l'union, lui, ne change pas : référence dans l'ordre du cache, puis
        # l'atome dense dans son ordre stocké, puis l'atome BM25. `sorted` est stable, donc
        # le sous-ensemble dense@50 garde l'ordre du cache et le contrôle A reste exact.
        # Passer par des ensembles ici rendrait le départage des ex æquo dépendant du
        # hachage des chaînes, donc non reproductible d'un processus à l'autre.
        dense50 = {row["chunk_id"] for row in reference}

        def atome_de(source_atome) -> list[str]:
            vu, garde = set(), []
            for chunk_id in source_atome.get(key, ()):
                if chunk_id in dense50 or chunk_id in vu or index.get(chunk_id) is None:
                    continue
                vu.add(chunk_id)
                garde.append(chunk_id)
            return garde

        ajouts = {"dense100": atome_de(dense_atom), "bm25": atome_de(bm25_atom)}

        vus = set(dense50)
        union = list(reference)
        for chunk_id in ajouts["dense100"] + ajouts["bm25"]:
            if chunk_id in vus:
                continue
            vus.add(chunk_id)
            union.append({"chunk_id": chunk_id,
                          "document_id": index.get(chunk_id)["document_id"], "score": 0.0})
        for row in union:
            source = index.get(row["chunk_id"])
            row["text"] = source["text"] if source else ""
        dense100 = dense50 | set(ajouts["dense100"])
        pools[key] = {
            "union": union,
            "reference_order": reference,
            "membres": {
                "rr/dense50": dense50,
                "rr/dense100": dense100,
                "rr/dense50+bm25": dense50 | set(ajouts["bm25"]),
                "rr/dense100+bm25": dense100 | set(ajouts["bm25"]),
            },
        }
    return pools


# --------------------------------------------------------------------- étape 1 : scoring

def step_score(index: ChunkIndex, items: list[dict], limit: int | None) -> None:
    from eval_rerankers import TEXT_CHARACTERS, QwenReranker

    pools = build_pools(index, items)
    if limit:
        items = items[:limit]
    partial = json.loads(PARTIAL.read_text(encoding="utf-8")) if PARTIAL.exists() else {}
    reste = [item for item in items if item["key"] not in partial]
    paires = sum(len([r for r in pools[i["key"]]["union"] if r["text"]]) for i in reste)
    print(f"\n  {len(reste)}/{len(items)} questions à scorer · {paires} paires "
          f"· {paires / max(len(reste), 1):.0f} paires/question")
    if not reste:
        print("  tout est déjà en cache")
        return
    print(f"  estimation à 0,29 s/paire : {paires * 0.29 / 60:.0f} min")

    scorer = QwenReranker(MODEL)
    latences = []
    started = time.perf_counter()
    for n, item in enumerate(reste, 1):
        rows = [r for r in pools[item["key"]]["union"] if r["text"]]
        t0 = time.perf_counter()
        scores = scorer.score(pipeline.query_of(item), [r["text"][:TEXT_CHARACTERS] for r in rows])
        ecoule = time.perf_counter() - t0
        latences.append({"key": item["key"], "pairs": len(rows), "seconds": round(ecoule, 2)})
        partial[item["key"]] = {r["chunk_id"]: s for r, s in zip(rows, scores)}
        # Écrit à chaque question : une interruption ne coûte que la question en cours.
        PARTIAL.write_text(json.dumps(partial, ensure_ascii=False), encoding="utf-8")
        (CACHE / f"pool-rerank-latence-{SIGNATURE}.json").write_text(
            json.dumps(_fusionne_latences(latences[-1:]), ensure_ascii=False), encoding="utf-8")
        reste_s = (time.perf_counter() - started) / n * (len(reste) - n)
        print(f"      {n}/{len(reste)}  {item['key']:<10} {len(rows):>3} paires  {ecoule:>6.1f} s "
              f"({ecoule / max(len(rows), 1):.3f} s/paire)  reste ~{reste_s / 60:.0f} min", flush=True)
    print(f"\n  {len(reste)} questions en {(time.perf_counter() - started) / 60:.1f} min")


def _fusionne_latences(nouvelles: list[dict]) -> list[dict]:
    path = CACHE / f"pool-rerank-latence-{SIGNATURE}.json"
    anciennes = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    par_cle = {row["key"]: row for row in anciennes}
    par_cle.update({row["key"]: row for row in nouvelles})
    return list(par_cle.values())


# ---------------------------------------------------------------------- étape 2 : rapport

def controle_a(per_question: list[dict], sabotage: bool) -> list[str]:
    """``dense@50+rr`` doit reproduire le banc reranker archivé, question par question.

    Aucune tolérance. Le banc archivé porte ``bb7bf33c37`` et le corpus est à ``10390927db``,
    mais ce qui a changé entre les deux est le **titre** servi à l'index lexical et à
    l'embedding — or le reranker ne voit que ``(requête, texte)``, et les deux vues corpus
    donnent la même empreinte sha256 sur la concaténation des 22 190 textes. Les scores
    doivent donc être identiques au bit près.
    """
    if not ARCHIVE_RERANK.exists():
        return [f"{ARCHIVE_RERANK.name} absent : impossible de vérifier le reclassement"]
    archive = json.loads(ARCHIVE_RERANK.read_text(encoding="utf-8"))
    attendu = {r["key"]: r["qwen3-0.6b"] for r in archive["pools"]["dense"]["per_question"]}
    ecarts = []
    for row in per_question:
        cible = attendu.get(row["key"])
        if cible is None:
            ecarts.append(f"{row['key']} : absent du banc archivé")
            continue
        obtenu = row[REFERENCE]
        if abs(obtenu["ndcg"] - cible["ndcg"]) > 1e-9:
            ecarts.append(f"{row['key']} : nDCG {obtenu['ndcg']:.6f} au lieu de {cible['ndcg']:.6f}")
        elif obtenu["first_rank"] != cible["first_rank"]:
            ecarts.append(f"{row['key']} : rang {obtenu['first_rank']} au lieu de {cible['first_rank']}")
    return ecarts


def step_report(index: ChunkIndex, items: list[dict], sabotage: bool) -> None:
    if not PARTIAL.exists():
        sys.exit("aucun score en cache : lance --step score")
    scores = json.loads(PARTIAL.read_text(encoding="utf-8"))
    if sabotage:
        # Décaler le pool d'un rang ne suffit PAS à faire hurler le contrôle A : une rotation
        # ne change pas l'*ensemble* des candidats, et le classement se fait par score — le
        # sous-ensemble reclassé serait identique. Un contrôle qu'on ne sait pas faire échouer
        # n'est pas un contrôle. On décale donc aussi l'attribution des scores : le
        # reclassement devient faux, et le contrôle A doit le voir.
        scores = {key: dict(zip(list(note), list(note.values())[1:] + list(note.values())[:1]))
                  for key, note in scores.items()}
    pools = build_pools(index, items, sabotage=sabotage)
    items = [item for item in items if item["key"] in scores]
    print(f"\n  {len(items)} questions scorées")

    per_question = []
    tailles = {config: {} for config in CONFIGS}
    for item in items:
        key = item["key"]
        note = scores[key]
        union = pools[key]["union"]
        record = {"bench": item["bench"], "key": key, "qid": item["qid"], "kind": item.get("kind", "single"),
                  "none": experiment.measure(item, pools[key]["reference_order"])}
        tailles["none"][key] = len(pools[key]["reference_order"])
        for config in CONFIGS[1:]:
            membres = pools[key]["membres"][config]
            sous_ensemble = [r for r in union if r["chunk_id"] in membres and r["text"] and r["chunk_id"] in note]
            classe = sorted(sous_ensemble, key=lambda r: -note[r["chunk_id"]])
            record[config] = experiment.measure(item, classe)
            tailles[config][key] = len(classe)
        per_question.append(record)

    resultat = experiment.summarise(per_question, list(CONFIGS), reference=REFERENCE)

    # --- contrôles, avant de lire quoi que ce soit
    ecarts_a = controle_a(per_question, sabotage)
    ecarts_b = controle_ligne_de_base(resultat, list(CONFIGS))
    print("\n  === contrôle A — dense@50+rr reproduit le banc reranker archivé ===")
    if ecarts_a:
        for ligne in ecarts_a[:8]:
            print(f"      ÉCART  {ligne}")
        print(f"      ({len(ecarts_a)} écarts au total sur {len(items)} questions)")
    else:
        print(f"      {len(items)}/{len(items)} questions : nDCG et rang identiques au bit près")
    print("  === contrôle B — dense@50 sans rerank reproduit results-router-v3.json ===")
    if ecarts_b:
        for ligne in ecarts_b[:8]:
            print(f"      ÉCART  {ligne}")
    else:
        print("      les trois vues reproduisent la ligne de base à l'identique")
    if sabotage:
        print("\n  --sabotage : les contrôles A et B ci-dessus DOIVENT signaler des écarts.")
        return
    if ecarts_a or ecarts_b:
        sys.exit("\nARRÊT : le pool ou le reclassement a dérivé, le banc mesurerait la dérive.")

    # --- contrôle C : le rappel de pool de l'union reproduit la Phase 0
    phase0 = json.loads((HERE / f"results-pool-recall-{SIGNATURE}.json").read_text(encoding="utf-8"))
    attendu_c = phase0["lines"]["dense@100"]["pool_recall_before"]
    ratés_union = sum(1 for r in per_question if r["rr/dense100+bm25"]["first_rank"] is None)
    rappel_union = round(1 - ratés_union / len(per_question), 3)
    print("  === contrôle C — le rappel de pool de l'union reproduit la Phase 0 ===")
    print(f"      pool de référence {1 - sum(1 for r in per_question if r['none']['first_rank'] is None) / len(per_question):.3f} "
          f"(Phase 0 : {attendu_c}) · union {rappel_union} (Phase 0 : 0.903)")

    experiment.print_summary(resultat, list(CONFIGS), reference=REFERENCE)

    # --- le nerf du chantier : les cibles que l'union ajoute, le reranker les remonte-t-il ?
    #
    # La Phase 0 a mesuré leur *présence* dans le pool. Un plafond de rappel n'est pas un gain :
    # les 6 cibles que l'union à profondeur 50 ajoutait, aucune des sept règles de fusion ne
    # les remontait. Ici la règle de fusion est remplacée par un reranker validé — et c'est
    # exactement cette substitution que ce bloc juge, question par question.
    recuperees = [r for r in per_question
                  if r["none"]["first_rank"] is None and r["rr/dense100+bm25"]["first_rank"] is not None]
    perdues = [r for r in per_question
               if r["none"]["first_rank"] is not None and r["rr/dense100+bm25"]["first_rank"] is None]
    print(f"\n=== les cibles que l'union ajoute — où le reclassement les place-t-il ? ===")
    print(f"  {len(recuperees)} questions entrent dans le pool ; {len(perdues)} en sortent")
    if recuperees:
        rangs = sorted(r["rr/dense100+bm25"]["first_rank"] for r in recuperees)
        print(f"  rangs après reclassement : {rangs}")
        for seuil in (1, 3, 5, 10, 30):
            print(f"      dans le top {seuil:<3} : {sum(1 for x in rangs if x <= seuil)}/{len(rangs)}")
        print(f"  gain de nDCG@10 apporté par ces {len(recuperees)} questions seules : "
              f"{sum(r['rr/dense100+bm25']['ndcg'] for r in recuperees) / len(per_question):+.3f}")
        for r in sorted(recuperees, key=lambda x: x["rr/dense100+bm25"]["first_rank"]):
            print(f"      {r['key']:<10} {r['kind']:<8} rang {r['rr/dense100+bm25']['first_rank']:>3}"
                  f"   nDCG {r['rr/dense100+bm25']['ndcg']:.3f}")

    # --- le prix de l'élargissement : ce que le pool plus large déclasse
    deplaces = [(r["key"], r[REFERENCE]["first_rank"], r["rr/dense100+bm25"]["first_rank"])
                for r in per_question
                if r[REFERENCE]["first_rank"] and r["rr/dense100+bm25"]["first_rank"]
                and r["rr/dense100+bm25"]["first_rank"] > r[REFERENCE]["first_rank"]]
    ameliores = [(r["key"], r[REFERENCE]["first_rank"], r["rr/dense100+bm25"]["first_rank"])
                 for r in per_question
                 if r[REFERENCE]["first_rank"] and r["rr/dense100+bm25"]["first_rank"]
                 and r["rr/dense100+bm25"]["first_rank"] < r[REFERENCE]["first_rank"]]
    print(f"\n=== dilution — sur les questions que les deux trouvent ===")
    print(f"  reculées : {len(deplaces)}  ·  avancées : {len(ameliores)}  "
          f"·  inchangées : {len(per_question) - len(deplaces) - len(ameliores) - len(recuperees) - len(perdues)}")
    if deplaces:
        pire = sorted(deplaces, key=lambda x: x[1] - x[2])[:5]
        print(f"  les cinq pires reculs : " + " · ".join(f"{k} {a}→{b}" for k, a, b in pire))

    # --- latence
    latences = json.loads((CACHE / f"pool-rerank-latence-{SIGNATURE}.json").read_text(encoding="utf-8"))
    #: Cadence **propre à chaque question** : les passages n'ont pas tous la même longueur,
    #: et une moyenne globale écraserait cette dispersion — or c'est elle qui fait le p95.
    cadence = {row["key"]: row["seconds"] / row["pairs"] for row in latences if row["pairs"]}
    mesure_union = {row["key"]: row["seconds"] for row in latences}
    s_paire = statistics.median(cadence.values())

    def centiles(valeurs: list[float]) -> tuple[float, float]:
        rang = sorted(valeurs)
        return (statistics.median(rang), rang[min(len(rang) - 1, int(len(rang) * 0.95))])

    print(f"\n=== latence — {len(latences)} questions mesurées, {s_paire:.3f} s/paire (médiane) ===")
    print(f"{'configuration':<28}{'pool moyen':>11}{'p50':>9}{'p95':>9}{'max':>9}   statut")
    latence_config = {}
    for config in CONFIGS:
        taille = statistics.mean(tailles[config].values())
        if config == "none":
            latence_config[config] = {"pool_moyen": round(taille, 1), "latence_p50_s": 0.0,
                                      "latence_p95_s": 0.0, "latence_max_s": 0.0, "mesuree": False}
            print(f"{LIBELLE[config]:<28}{taille:>11.1f}{'—':>9}{'—':>9}{'—':>9}   pas de reclassement")
            continue
        if config == "rr/dense100+bm25":
            # La seule configuration dont le pool est exactement ce qui a été scoré : sa
            # latence est mesurée de bout en bout, pas dérivée.
            par_question = [mesure_union[k] for k in tailles[config] if k in mesure_union]
            statut = "mesurée"
        else:
            par_question = [tailles[config][k] * cadence[k] for k in tailles[config] if k in cadence]
            statut = "dérivée (pool × cadence de la question)"
        p50_c, p95_c = centiles(par_question)
        latence_config[config] = {"pool_moyen": round(taille, 1), "latence_p50_s": round(p50_c, 1),
                                  "latence_p95_s": round(p95_c, 1), "latence_max_s": round(max(par_question), 1),
                                  "mesuree": config == "rr/dense100+bm25"}
        print(f"{LIBELLE[config]:<28}{taille:>11.1f}{p50_c:>7.1f} s{p95_c:>7.1f} s"
              f"{max(par_question):>7.1f} s   {statut}")
    p50 = latence_config["rr/dense100+bm25"]["latence_p50_s"]
    p95 = latence_config["rr/dense100+bm25"]["latence_p95_s"]

    # --- verdict, selon les seuils pré-enregistrés
    #
    # Le seuil s'applique à chacune des configurations reclassées, toutes pré-enregistrées en
    # Partie I. Une seule porte le verdict du chantier — ``rr/dense100+bm25``, le candidat que
    # la Phase 0 désignait ; les autres sont des comparaisons secondaires. Elles partagent le
    # même seuil, donc la probabilité qu'au moins une passe par chance est plus grande que
    # celle de la seule candidate : c'est dit ici plutôt que tu dans le tableau.
    def juger(config: str) -> dict:
        d = resultat["paired"]["pooled"][config]
        taille = latence_config[config]["latence_p50_s"]
        qual = d["delta"] >= SEUIL_NDCG and d["significant"]
        lat = taille <= SEUIL_LATENCE_S
        return {"delta": d, "qualite": qual, "latence_p50_s": taille, "latence_ok": lat,
                "verdict": "GO" if (qual and lat) else ("HOLD" if qual else "NO-GO")}

    jugements = {c: juger(c) for c in CONFIGS if c not in ("none", REFERENCE)}
    print(f"\n=== verdicts — seuils pré-enregistrés "
          f"(Δ nDCG@10 pooled vs {REFERENCE} ≥ +{SEUIL_NDCG:.3f}, IC95 hors zéro, p50 ≤ {SEUIL_LATENCE_S:.0f} s) ===")
    print(f"{'configuration':<28}{'Δ nDCG@10':>11}{'IC95':>20}{'p50':>9}{'verdict':>10}")
    for config, j in jugements.items():
        d = j["delta"]
        ci = f"[{d['ci95'][0]:+.3f} ; {d['ci95'][1]:+.3f}]"
        marque = "  ← candidat" if config == "rr/dense100+bm25" else ""
        print(f"{LIBELLE[config]:<28}{d['delta']:>+11.3f}{ci:>20}{j['latence_p50_s']:>7.0f} s{j['verdict']:>10}{marque}")
    verdict = jugements["rr/dense100+bm25"]["verdict"]
    delta = jugements["rr/dense100+bm25"]["delta"]
    qualite = jugements["rr/dense100+bm25"]["qualite"]
    latence_ok = jugements["rr/dense100+bm25"]["latence_ok"]
    print(f"\n  >>> verdict du chantier (candidat pré-enregistré) : {verdict}")
    autres = [LIBELLE[c] for c, j in jugements.items() if j["verdict"] == "GO" and c != "rr/dense100+bm25"]
    if autres:
        print(f"      configurations secondaires en GO : {', '.join(autres)}")

    payload = {
        "corpus": {**corpus_overlay.describe(), "chunks": len(index.chunks),
                   "documents": len(index.documents)},
        "protocole": {
            "reranker": MODEL, "reference": REFERENCE, "n_questions": len(items),
            "seuils": {"ndcg_pooled": SEUIL_NDCG, "latence_p50_s": SEUIL_LATENCE_S},
            "scoring": "une passe sur l'union, sous-ensembles dérivés par tri (paires indépendantes)",
            "bootstrap": {"draws": 4000, "seed": 20260901},
            "external_llm_calls": 0, "shadow": True,
        },
        "controles": {"A_rerank_archive": ecarts_a or "reproduit au bit près",
                      "B_ligne_de_base": ecarts_b or "reproduit",
                      "C_rappel_de_pool": {"reference": 0.826, "union": rappel_union, "phase0": 0.903}},
        "latence": {"union_p50_s": round(p50, 1), "union_p95_s": round(p95, 1),
                    "s_par_paire": round(s_paire, 4), "par_configuration": latence_config},
        "tailles_de_pool": {c: round(statistics.mean(v.values()), 1) for c, v in tailles.items()},
        "tailles_de_pool_par_question": tailles,
        **resultat,
        "gains_pertes": experiment.wins_losses(per_question, "rr/dense100+bm25", REFERENCE),
        "cibles_ajoutees_par_lunion": [
            {"key": r["key"], "kind": r["kind"], "rang": r["rr/dense100+bm25"]["first_rank"],
             "ndcg": round(r["rr/dense100+bm25"]["ndcg"], 4)} for r in recuperees],
        "cibles_sorties_du_pool": [r["key"] for r in perdues],
        "dilution": {"reculees": len(deplaces), "avancees": len(ameliores)},
        "per_question": per_question,
        "verdict": {"candidat": "rr/dense100+bm25", "qualite": qualite, "latence_ok": latence_ok,
                    "verdict": verdict, "delta_pooled": delta,
                    "par_configuration": jugements,
                    "note": "Le seuil s'applique aux quatre configurations reclassées, toutes "
                            "pré-enregistrées ; une seule porte le verdict du chantier. Quatre "
                            "comparaisons partagent un seuil : la probabilité qu'au moins une "
                            "passe par chance est plus grande que celle de la seule candidate."},
    }
    OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n  écrit  {OUTPUT.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--step", required=True, choices=("score", "report"))
    parser.add_argument("--limit", type=int, help="pilote : n premières questions")
    parser.add_argument("--sabotage", action="store_true",
                        help="décale le pool ET l'attribution des scores : A et B doivent refuser")
    args = parser.parse_args()

    index = ChunkIndex.load(verbose=False)
    items = experiment.load_items(index)
    print(f"  corpus {len(index.chunks)} chunks · {len(index.documents)} documents · signature {SIGNATURE}")
    print(f"  {len(items)} questions · reranker {MODEL}")
    if args.step == "score":
        step_score(index, items, args.limit)
    else:
        step_report(index, items, args.sabotage)


if __name__ == "__main__":
    main()
