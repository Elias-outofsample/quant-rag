"""Le reclassement sélectif — sept variantes dérivées d'un artefact figé, zéro seconde de GPU.

Protocole pré-enregistré et publié **avant d'avoir dérivé une seule variante** :
``rag/benchmark/PRE-ENREGISTREMENT-RECLASSEMENT-SELECTIF-2026-09-07.md`` (commit ``3a792e2``).

La question, unique : le HOLD de ``fc34604`` a montré que les 21 gains venaient des rangs
dense 3-10 et que les 7 pertes avaient toutes leur or aux rangs 1-3. Existe-t-il, dans les
scores **déjà calculés**, une règle qui garde les gains sans produire les pertes ?

La famille est **close** — trois gels de tête, une fusion RRF, trois seuils de marge — et
elle est énumérée dans le pré-enregistrement. Aucune huitième variante ne sera dérivée,
quelle que soit la lecture des sept : l'artefact étant figé, le p-hacking serait gratuit,
donc c'est la discipline qui le retient et rien d'autre.

Quatre conditions de survie, toutes ensemble (§4 du pré-enregistrement) : gain poolé
≥ +0,010 avec borne basse > 0 **sur un intervalle corrigé de Bonferroni à 99,29 %** ; Δ ≥ 0
sur v1 et v3 pris séparément ; ``b`` ≤ 5 sur la garde stricte du chantier précédent, même
instrument, même sélection de production ; et un sabotage nettement en dessous.

Tout est dérivé de ``.cache/reranking-scores-5530cba145.json`` (versionné) et du cache de
classements. **Qdrant n'est jamais ouvert, aucun appel LLM, aucun score nouveau calculé.**

    .venv/bin/python rag/benchmark/eval_reclassement_selectif.py
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import corpus_overlay  # noqa: E402
import eval_reranking  # noqa: E402
import experiment  # noqa: E402
import metrics  # noqa: E402
import quant_rag  # noqa: E402
from corpus import ChunkIndex  # noqa: E402

SIGNATURE = corpus_overlay.signature()
SCORES = HERE / ".cache" / f"reranking-scores-{SIGNATURE}.json"
OUTPUT = HERE / f"results-reclassement-selectif-{SIGNATURE}.json"

TETE = 10                     # budget de décision, inchangé
RRF_K = 60                    # convention du dépôt
SEED = 20260907
DRAWS = 10_000
#: §5 — Bonferroni sur les sept variantes de la famille close.
N_VARIANTES = 7
NIVEAU_CORRIGE = 1 - 0.05 / N_VARIANTES
SEUIL_GAIN = 0.010
SEUIL_GARDE = 5               # §4.3 — b <= 5, la garde stricte du chantier précédent

MOINS_INF = float("-inf")


# --------------------------------------------------------------------------- les sept règles

def _cle(row: dict, scores: dict) -> tuple:
    return (-scores.get(row["chunk_id"], MOINS_INF), row["rang_dense"], row["chunk_id"])


def ordre_b10(tete: list[dict], scores: dict) -> list[dict]:
    """Le reclassement plein de la tête — le point de comparaison, déjà HOLD, pas candidat."""
    return sorted(tete, key=lambda r: _cle(r, scores))


def ordre_gel(tete: list[dict], scores: dict, k: int) -> list[dict]:
    """Les k premiers rangs dense sont figés ; le reste de la tête est reclassé."""
    return tete[:k] + sorted(tete[k:], key=lambda r: _cle(r, scores))


def ordre_rrf(tete: list[dict], scores: dict) -> list[dict]:
    """Fusion réciproque du rang dense et du rang reranker, sur la même tête."""
    par_rerank = {r["chunk_id"]: i for i, r in enumerate(ordre_b10(tete, scores), 1)}
    def note(r):
        return 1 / (RRF_K + r["rang_dense"]) + 1 / (RRF_K + par_rerank[r["chunk_id"]])
    return sorted(tete, key=lambda r: (-note(r), r["rang_dense"], r["chunk_id"]))


def ordre_marge(tete: list[dict], scores: dict, delta: float) -> list[dict]:
    """Un candidat ne dépasse un candidat mieux classé en dense que s'il le bat de ``delta``.

    Réalisation déterministe : un tri à bulles sur la tête, dont la seule permutation permise
    est l'échange de deux voisins ``(a, b)`` — ``a`` mieux classé en dense — lorsque
    ``score(b) - score(a) >= delta``. Le résultat ne dépend pas de l'ordre des passes : la
    règle d'échange est locale, stricte, et la tête ne fait que dix éléments.
    """
    ordre = list(tete)
    for _ in range(len(ordre)):
        echange = False
        for i in range(len(ordre) - 1):
            a, b = ordre[i], ordre[i + 1]
            sa = scores.get(a["chunk_id"], MOINS_INF)
            sb = scores.get(b["chunk_id"], MOINS_INF)
            if sb - sa >= delta:
                ordre[i], ordre[i + 1] = b, a
                echange = True
        if not echange:
            break
    return ordre


VARIANTES = {
    "gel1": lambda t, s: ordre_gel(t, s, 1),
    "gel2": lambda t, s: ordre_gel(t, s, 2),
    "gel3": lambda t, s: ordre_gel(t, s, 3),
    "rrf": ordre_rrf,
    "marge01": lambda t, s: ordre_marge(t, s, 0.01),
    "marge05": lambda t, s: ordre_marge(t, s, 0.05),
    "marge10": lambda t, s: ordre_marge(t, s, 0.10),
}
assert len(VARIANTES) == N_VARIANTES, "la famille est close : sept variantes, ni plus ni moins"


def appliquer(rows: list[dict], scores: dict, regle) -> list[dict]:
    """La règle réordonne la tête ; la queue garde son ordre dense."""
    ordonnee = regle(rows[:TETE], scores) + rows[TETE:]
    return [{**r, "rang_final": i} for i, r in enumerate(ordonnee, 1)]


# --------------------------------------------------------------------------- exécution

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()

    if not SCORES.exists():
        sys.exit(f"{SCORES.name} absent — il vaut 38 min de M4 et il est versionné depuis f395541.")
    scores_par_question = json.loads(SCORES.read_text(encoding="utf-8"))

    index = ChunkIndex.load(verbose=False)
    items = experiment.load_items(index)
    cache = experiment.cached_rankings()
    print(f"corpus {SIGNATURE} · {len(items)} questions · {sum(len(v) for v in scores_par_question.values())} candidats scorés")
    print("aucun score calculé, aucun appel LLM, Qdrant jamais ouvert\n")

    pools = {}
    for item in items:
        rows = []
        for rang, (chunk_id, document_id, score) in enumerate(cache[item["key"]]["dense"], 1):
            src = index.get(chunk_id) or {}
            rows.append({"chunk_id": chunk_id, "document_id": document_id, "score": score,
                         "rang_dense": rang, "text": src.get("text", ""),
                         "content_type": src.get("content_type")})
        pools[item["key"]] = rows

    # Le bras saboté : mêmes scores, appariement au texte détruit. Graine du pré-enregistrement.
    alea = random.Random(SEED)
    sabotes = {}
    for item in items:
        cles = list(scores_par_question[item["key"]])
        valeurs = list(scores_par_question[item["key"]].values())
        alea.shuffle(valeurs)
        sabotes[item["key"]] = dict(zip(cles, valeurs))

    noms = ["b10"] + list(VARIANTES)
    regles = {"b10": ordre_b10, **VARIANTES}

    par_question = []
    for item in items:
        rows, sc = pools[item["key"]], scores_par_question[item["key"]]
        ligne = {"bench": item["bench"], "key": item["key"], "qid": item["qid"],
                 "kind": item.get("kind", "single"), "none": experiment.measure(item, rows)}
        for nom in noms:
            ligne[nom] = experiment.measure(item, appliquer(rows, sc, regles[nom]))
            ligne[f"{nom}@sabote"] = experiment.measure(
                item, appliquer(rows, sabotes[item["key"]], regles[nom]))
        par_question.append(ligne)

    # ---- gain, avec l'intervalle corrigé qui décide et l'intervalle à 95 % en diagnostic
    golds = {i["key"]: set(i["gold_chunks"]) for i in items}
    verdicts = {}
    for nom in noms:
        bloc = {}
        for banc in experiment.BENCHES:
            sous = [r for r in par_question if banc == "pooled" or r["bench"] == banc]
            deltas = [r[nom]["ndcg"] - r["none"]["ndcg"] for r in sous]
            point = round(sum(deltas) / len(deltas), 4)
            bloc[banc] = {
                "n": len(sous), "delta": point,
                "ic95": list(metrics.bootstrap_ci(deltas, draws=DRAWS, level=0.95, seed=SEED)),
                "ic_bonferroni": list(metrics.bootstrap_ci(deltas, draws=DRAWS,
                                                           level=NIVEAU_CORRIGE, seed=SEED)),
            }

        # ---- garde et rotation, sélection de production, identiques au chantier précédent
        n_pop = perdues = gagnees = 0
        liste_perdues = []
        for item in items:
            cle = item["key"]
            avant = eval_reranking.or_servi(pools[cle], golds[cle])
            apres = eval_reranking.or_servi(
                appliquer(pools[cle], scores_par_question[cle], regles[nom]), golds[cle])
            if avant:
                n_pop += 1
                if not apres:
                    perdues += 1
                    liste_perdues.append(cle)
            elif apres:
                gagnees += 1
        rotation = perdues + gagnees

        # ---- sabotage de la même règle
        d_sab = [r[f"{nom}@sabote"]["ndcg"] - r["none"]["ndcg"] for r in par_question]
        delta_sabote = round(sum(d_sab) / len(d_sab), 4)

        poole = bloc["pooled"]
        conditions = {
            "gain": poole["delta"] >= SEUIL_GAIN and poole["ic_bonferroni"][0] > 0,
            "v1_v3_separement": bloc["v1"]["delta"] >= 0 and bloc["v3"]["delta"] >= 0,
            "garde": perdues <= SEUIL_GARDE,
            "sabotage": delta_sabote < poole["delta"],
        }
        verdicts[nom] = {
            "delta": bloc, "garde": {"n": n_pop, "b": perdues, "perdues": liste_perdues,
                                     "gagnees": gagnees, "rotation": rotation,
                                     "seuil": SEUIL_GARDE},
            "sabotage_delta": delta_sabote,
            "conditions": conditions,
            "survit": all(conditions.values()),
        }

    # ---- la règle de sélection du §6, écrite avant de regarder
    candidats = [n for n in VARIANTES if verdicts[n]["survit"]]
    gagnant = None
    if candidats:
        gagnant = sorted(candidats, key=lambda n: (
            -verdicts[n]["delta"]["pooled"]["delta"],
            verdicts[n]["garde"]["rotation"],
            list(VARIANTES).index(n),
        ))[0]

    largeur = 11
    print(f"{'variante':<10}{'Δ poolé':>9}{'IC99,29 %':>20}{'Δ v1':>8}{'Δ v3':>8}"
          f"{'b':>4}{'rot.':>6}{'sabot.':>9}{'survit':>8}")
    for nom in noms:
        v = verdicts[nom]
        p, ic = v["delta"]["pooled"], v["delta"]["pooled"]["ic_bonferroni"]
        marque = "—" if nom == "b10" else ("OUI" if v["survit"] else "non")
        print(f"{nom:<10}{p['delta']:>+9.4f}{f'[{ic[0]:+.3f} ; {ic[1]:+.3f}]':>20}"
              f"{v['delta']['v1']['delta']:>+8.3f}{v['delta']['v3']['delta']:>+8.3f}"
              f"{v['garde']['b']:>4}{v['garde']['rotation']:>6}{v['sabotage_delta']:>+9.4f}{marque:>8}")

    print(f"\nconditions non satisfaites, variante par variante :")
    for nom in VARIANTES:
        manquantes = [k for k, ok in verdicts[nom]["conditions"].items() if not ok]
        print(f"  {nom:<10}{'toutes satisfaites' if not manquantes else ', '.join(manquantes)}")

    print()
    if gagnant:
        print(f"=== GO EXPÉRIMENTAL — gagnante : {gagnant} ===")
        print(f"  règle du §6 appliquée à {len(candidats)} survivante(s) : {', '.join(candidats)}")
    else:
        print("=== AUCUNE SURVIVANTE — l'avenue se ferme par épuisement sur artefact figé ===")
        print("  la famille est close (§7) : on n'en dérive pas une huitième.")

    charge = {
        "version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "pre_enregistrement": "rag/benchmark/PRE-ENREGISTREMENT-RECLASSEMENT-SELECTIF-2026-09-07.md",
        "corpus_state": corpus_overlay.describe(),
        "settings": {"tete": TETE, "rrf_k": RRF_K, "draws": DRAWS, "seed": SEED,
                     "niveau_corrige": round(NIVEAU_CORRIGE, 6), "n_variantes": N_VARIANTES,
                     "seuil_gain": SEUIL_GAIN, "seuil_garde": SEUIL_GARDE,
                     "source_scores": SCORES.name, "appels_llm": 0, "qdrant": "jamais ouvert"},
        "verdicts": verdicts,
        "survivantes": candidats,
        "gagnant": gagnant,
        "par_question": par_question,
    }
    args.output.write_text(json.dumps(charge, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nécrit : {args.output.name}")


if __name__ == "__main__":
    main()
