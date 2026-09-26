"""La surface « recherche approfondie » — B = 30, et la décision se prend sur l'or servi.

Protocole pré-enregistré et publié **avant la première mesure de ce fil** :
``rag/benchmark/PRE-ENREGISTREMENT-RECHERCHE-APPROFONDIE-2026-09-07.md`` (commit ``7e4fa0e``).

Second fil du chantier reranking. Le premier avait disqualifié B = 30 et B = 50 sur une borne
de latence de 4,0 s — mais cette borne valait pour le **chemin par défaut**. Une surface
demandée explicitement par l'appelant est un autre produit, et l'arrêt était donc de raison et
non de preuve. La borne de ce fil, **10,0 s p95**, vient du coude du rendement marginal
(facteur 2,8 entre B = 30 et B = 50), calculé sur des chiffres antérieurs à ce fil.

**La métrique de décision change, et c'est déclaré** : le §15 du chantier précédent a montré
que le nDCG@10 et le comptage d'or réellement servi classent les variantes dans des sens
opposés. Ici c'est **l'or servi** qui décide — nombre de questions dont un chunk d'or est dans
les cinq passages rendus par la sélection de production — et le nDCG@10 qui sert de diagnostic.

Deux variantes, famille close : ``b30`` (reclassement plein de la tête de 30) et ``rrf30``
(fusion réciproque). ``b50`` et ``rrf50`` sont dérivés en sensibilité et **hors borne**.

Tout dérive de ``.cache/reranking-scores-5530cba145.json``, qui couvre le pool entier.
**Aucun score calculé, aucun appel LLM, Qdrant jamais ouvert.**

    .venv/bin/python rag/benchmark/eval_recherche_approfondie.py
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
import eval_reclassement_selectif as selectif  # noqa: E402
import eval_reranking  # noqa: E402
import experiment  # noqa: E402
import metrics  # noqa: E402
from corpus import ChunkIndex  # noqa: E402

SIGNATURE = corpus_overlay.signature()
SCORES = HERE / ".cache" / f"reranking-scores-{SIGNATURE}.json"
OUTPUT = HERE / f"results-recherche-approfondie-{SIGNATURE}.json"

SEED, DRAWS = 20260907, 10_000
#: §5 — Bonferroni sur les deux variantes qui décident (b50/rrf50 sont hors borne).
N_DECISIVES = 2
NIVEAU_CORRIGE = 1 - 0.05 / N_DECISIVES
#: §5.1 — le meilleur net du chemin par défaut vaut +14 ; la surface approfondie doit le battre.
GAIN_MINIMUM = 15
REFERENCE_SERVIE = 76
SEUIL_GARDE = 5

#: Les variantes qui décident, et celles qui ne font que renseigner.
DECISIVES = {"b30": (30, "plein"), "rrf30": (30, "rrf")}
SENSIBILITE = {"b50": (50, "plein"), "rrf50": (50, "rrf")}


def appliquer(rows: list[dict], scores: dict, tete: int, regle: str) -> list[dict]:
    """La règle réordonne la tête de ``tete`` ; la queue garde son ordre dense."""
    haut = rows[:tete]
    ordonnee = (selectif.ordre_b10(haut, scores) if regle == "plein"
                else selectif.ordre_rrf(haut, scores))
    return ordonnee + rows[tete:]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()

    if not SCORES.exists():
        sys.exit(f"{SCORES.name} absent — versionné depuis f395541.")
    scores_par_question = json.loads(SCORES.read_text(encoding="utf-8"))

    index = ChunkIndex.load(verbose=False)
    items = experiment.load_items(index)
    cache = experiment.cached_rankings()
    print(f"corpus {SIGNATURE} · {len(items)} questions · métrique de décision : l'or servi")
    print("aucun score calculé, aucun appel LLM, Qdrant jamais ouvert\n")

    pools, golds = {}, {}
    for item in items:
        rows = []
        for rang, (chunk_id, document_id, score) in enumerate(cache[item["key"]]["dense"], 1):
            src = index.get(chunk_id) or {}
            rows.append({"chunk_id": chunk_id, "document_id": document_id, "score": score,
                         "rang_dense": rang, "text": src.get("text", ""),
                         "content_type": src.get("content_type")})
        pools[item["key"]] = rows
        golds[item["key"]] = set(item["gold_chunks"])

    alea = random.Random(SEED)
    sabotes = {}
    for item in items:
        cles = list(scores_par_question[item["key"]])
        valeurs = list(scores_par_question[item["key"]].values())
        alea.shuffle(valeurs)
        sabotes[item["key"]] = dict(zip(cles, valeurs))

    # Référence : l'or servi sous le classement dense, par question.
    servi_dense = {i["key"]: eval_reranking.or_servi(pools[i["key"]], golds[i["key"]]) for i in items}
    n_ref = sum(servi_dense.values())
    print(f"référence — or servi sous dense : {n_ref}/{len(items)}"
          f"  (pré-enregistré : {REFERENCE_SERVIE})\n")

    toutes = {**DECISIVES, **SENSIBILITE}
    verdicts = {}
    for nom, (tete, regle) in toutes.items():
        servi, servi_sab = {}, {}
        for item in items:
            cle = item["key"]
            servi[cle] = eval_reranking.or_servi(
                appliquer(pools[cle], scores_par_question[cle], tete, regle), golds[cle])
            servi_sab[cle] = eval_reranking.or_servi(
                appliquer(pools[cle], sabotes[cle], tete, regle), golds[cle])

        perdues = [k for k in servi if servi_dense[k] and not servi[k]]
        gagnees = [k for k in servi if not servi_dense[k] and servi[k]]
        total = sum(servi.values())

        deltas = [int(servi[i["key"]]) - int(servi_dense[i["key"]]) for i in items]
        ic = metrics.bootstrap_ci(deltas, draws=DRAWS, level=NIVEAU_CORRIGE, seed=SEED)
        par_banc = {}
        for banc in ("v1", "v3"):
            sous = [i for i in items if i["bench"] == banc]
            par_banc[banc] = {"dense": sum(servi_dense[i["key"]] for i in sous),
                              "candidat": sum(servi[i["key"]] for i in sous)}

        # nDCG@10 en diagnostic — il ne décide rien dans ce fil.
        ndcg = [experiment.measure(i, appliquer(pools[i["key"]], scores_par_question[i["key"]],
                                                tete, regle))["ndcg"] for i in items]
        ndcg_ref = [experiment.measure(i, pools[i["key"]])["ndcg"] for i in items]
        delta_ndcg = round(sum(a - b for a, b in zip(ndcg, ndcg_ref)) / len(items), 4)

        decisive = nom in DECISIVES
        conditions = {
            "or_servi": total >= REFERENCE_SERVIE + GAIN_MINIMUM and ic[0] > 0,
            "v1_v3_sans_regression": all(par_banc[b]["candidat"] >= par_banc[b]["dense"]
                                         for b in ("v1", "v3")),
            "garde": len(perdues) <= SEUIL_GARDE,
            "sabotage": sum(servi_sab.values()) < total,
        }
        verdicts[nom] = {
            "decisive": decisive, "tete": tete, "regle": regle,
            "or_servi": total, "net": total - n_ref,
            "gagnees": len(gagnees), "perdues": len(perdues), "liste_perdues": perdues,
            "rotation": len(gagnees) + len(perdues),
            "ic_bonferroni": list(ic), "par_banc": par_banc,
            "or_servi_sabote": sum(servi_sab.values()),
            "delta_ndcg_diagnostic": delta_ndcg,
            "conditions": conditions,
            "survit": decisive and all(conditions.values()),
        }

    print(f"{'variante':<9}{'tête':>5}{'or servi':>10}{'net':>6}{'gagn.':>7}{'perd.':>7}"
          f"{'IC97,5 %':>18}{'sabot.':>8}{'ΔnDCG':>9}{'survit':>8}")
    for nom in list(DECISIVES) + list(SENSIBILITE):
        v = verdicts[nom]
        ic = v["ic_bonferroni"]
        marque = ("OUI" if v["survit"] else "non") if v["decisive"] else "hors borne"
        print(f"{nom:<9}{v['tete']:>5}{v['or_servi']:>10}{v['net']:>+6}{v['gagnees']:>7}"
              f"{v['perdues']:>7}{f'[{ic[0]:+.3f} ; {ic[1]:+.3f}]':>18}"
              f"{v['or_servi_sabote']:>8}{v['delta_ndcg_diagnostic']:>+9.4f}{marque:>8}")

    print("\nconditions non satisfaites (variantes décisives) :")
    for nom in DECISIVES:
        manque = [k for k, ok in verdicts[nom]["conditions"].items() if not ok]
        print(f"  {nom:<9}{'toutes satisfaites' if not manque else ', '.join(manque)}")

    survivantes = [n for n in DECISIVES if verdicts[n]["survit"]]
    gagnant = None
    if survivantes:
        gagnant = sorted(survivantes, key=lambda n: (-verdicts[n]["or_servi"],
                                                     verdicts[n]["perdues"],
                                                     0 if n == "rrf30" else 1))[0]
    print()
    if gagnant:
        print(f"=== GO EXPÉRIMENTAL sur la surface approfondie — gagnante : {gagnant} ===")
    else:
        print("=== AUCUNE SURVIVANTE — le fil se ferme, et l'arrêt devient prouvé ===")
        print("  on ne descend pas à B = 20 pour se rattraper (§7).")

    charge = {
        "version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "pre_enregistrement": "rag/benchmark/PRE-ENREGISTREMENT-RECHERCHE-APPROFONDIE-2026-09-07.md",
        "corpus_state": corpus_overlay.describe(),
        "settings": {"metrique_de_decision": "or servi dans les 5 passages de production",
                     "reference_servie": n_ref, "gain_minimum": GAIN_MINIMUM,
                     "seuil_garde": SEUIL_GARDE, "niveau_corrige": round(NIVEAU_CORRIGE, 6),
                     "borne_latence_p95": 10.0, "draws": DRAWS, "seed": SEED,
                     "source_scores": SCORES.name, "appels_llm": 0, "qdrant": "jamais ouvert"},
        "verdicts": verdicts, "survivantes": survivantes, "gagnant": gagnant,
    }
    args.output.write_text(json.dumps(charge, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nécrit : {args.output.name}")


if __name__ == "__main__":
    main()
