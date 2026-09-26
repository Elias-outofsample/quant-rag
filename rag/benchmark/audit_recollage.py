"""Recensement de la population et **audit du danger** de la transformation de recollage.

La transformation elle-même est dans `recollage.py` — une seule définition, partagée par
cet audit, la construction des bras et la mesure. Ce script ne la redéfinit pas : il la
met à l'épreuve.

Ce qu'il établit, avant le premier vecteur :

1. la **population** — les passages dont la chaîne plongée change réellement ;
2. ce que la **garde** accepte et refuse, avec des exemples des deux côtés ;
3. les **trois invariants** (I1 blanc seul, I2 prose intacte, I3 jetons LaTeX intacts) sur
   les 26 120 passages — pas sur un échantillon — pour la transformation de ce lot, pour
   la même sans garde, et pour le recollage atome-par-atome de la sonde `e11eac6` ;
4. ce que la transformation touche de **l'or du banc**.

    .venv/bin/python rag/benchmark/audit_recollage.py
"""
from __future__ import annotations

import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "titles"))

import familles_v4  # noqa: E402
import recollage as R  # noqa: E402
import reembed_titles as rt  # noqa: E402
from compare_v1_v2 import load_bench, load_v1  # noqa: E402
from corpus import ChunkIndex  # noqa: E402

RESULT = HERE / "audit-recollage.json"
SEED = 20260910

#: Atome de la sonde `sonde_latex.py` (commit `e11eac6`) — repris pour chiffrer son écart.
_ATOME = re.compile(r"\\[A-Za-z]+|\S")


def recoller_e11eac6(expression: str) -> str:
    """Le recollage de la sonde, atome par atome, repris à l'identique pour le comparer."""
    atomes = _ATOME.findall(expression)
    sortie: list[str] = []
    for atome in atomes:
        if sortie:
            precedent = sortie[-1]
            risque = precedent.startswith("\\") and precedent[1:].isalpha() and atome[:1].isalpha()
            sortie.append((" " if risque else "") + atome)
        else:
            sortie.append(atome)
    return "".join(sortie)


def sans_garde(texte: str) -> str:
    """La transformation de ce lot **sans** sa garde — pour chiffrer ce que la garde évite."""
    return R.MATHS.sub(lambda m: R.recoller(m.group()), texte)


def main() -> None:
    rows, export_title = rt.load_rows()
    textes = {r["chunk_id"]: r["text"] for r in rows}
    print(f"corpus : {len(rows)} passages")

    porteurs_dollar = {c for c, t in textes.items() if "$" in t}
    porteurs_double = {c for c, t in textes.items() if "$$" in t}
    porteurs_region = {c for c, t in textes.items() if R.MATHS.search(t)}
    print(f"porteurs de « $ »      : {len(porteurs_dollar):6d}  ({100 * len(porteurs_dollar) / len(rows):.1f} %)")
    print(f"porteurs de « $$ »     : {len(porteurs_double):6d}  ({100 * len(porteurs_double) / len(rows):.1f} %)")
    print(f"porteurs d'une région  : {len(porteurs_region):6d}  ({100 * len(porteurs_region) / len(rows):.1f} %)")

    # ------------------------------------------------------------------ la garde
    regions = [m.group() for c in sorted(porteurs_region) for m in R.MATHS.finditer(textes[c])]
    acceptees = [r for r in regions if R.est_mathematique(r)]
    refus_signature = [r for r in regions
                       if not (R.COMMANDE.search(r) or R.STRUCTURE.search(r))]
    refus_prose = [r for r in regions
                   if (R.COMMANDE.search(r) or R.STRUCTURE.search(r)) and R.mots_de_prose(r)]
    print(f"\nrégions mathématiques candidates : {len(regions)}")
    print(f"  acceptées par la garde          : {len(acceptees)} ({100 * len(acceptees) / len(regions):.1f} %)")
    print(f"  refusées faute de signature     : {len(refus_signature)}")
    print(f"  refusées par le test de prose   : {len(refus_prose)}")

    # ------------------------------------------------------------------ invariants, sur tout le corpus
    population: list[str] = []
    blancs = 0
    i1, i2, i3 = [], [], 0
    i1_sg, i2_sg = [], []
    longueurs_avant, longueurs_apres = [], []
    for chunk_id in sorted(textes):
        avant = textes[chunk_id]
        apres = R.recoller_passage(avant)
        if apres != avant:
            population.append(chunk_id)
            blancs += len(avant) - len(apres)
            longueurs_avant.append(len(avant))
            longueurs_apres.append(len(apres))
            v = R.violations(avant, apres)
            if v["I1"]:
                i1.append(chunk_id)
            if v["I2"]:
                i2.append({"chunk_id": chunk_id, "mots_detruits": v["I2"]})
            i3 += int(v["I3"])
        brut = sans_garde(avant)
        if brut != avant:
            if R.sans_blanc(avant) != R.sans_blanc(brut):
                i1_sg.append(chunk_id)
            manquants = sorted((Counter(R.MOT.findall(avant)) - Counter(R.MOT.findall(brut))).elements())
            if manquants:
                i2_sg.append({"chunk_id": chunk_id, "mots_detruits": manquants[:8]})
    i3_sonde = sum(1 for r in regions if R.jetons(recoller_e11eac6(r)) != R.jetons(r))

    import statistics
    print(f"\npopulation (chaîne plongée réellement changée) : {len(population)} passages "
          f"({100 * len(population) / len(rows):.1f} %)")
    print(f"  blancs retirés : {blancs} (moyenne {blancs / max(len(population), 1):.0f}/passage)")
    print(f"  longueur médiane : {int(statistics.median(longueurs_avant))} -> "
          f"{int(statistics.median(longueurs_apres))} caractères")
    print(f"\nINVARIANTS")
    print(f"  transformation de ce lot   : I1 {len(i1)} · I2 {len(i2)} · I3 {i3}")
    print(f"  la même sans garde         : I1 {len(i1_sg)} · I2 {len(i2_sg)}  <- ce que la garde évite")
    print(f"  recollage e11eac6 (atomes) : I3 {i3_sonde} régions sur {len(regions)}")
    if i2_sg:
        print("  exemples de mots que la transformation sans garde détruirait :")
        for ex in i2_sg[:4]:
            print(f"    {ex['chunk_id']} : {ex['mots_detruits']}")
    if i1 or i2 or i3:
        print("  ATTENTION — un invariant est violé MALGRÉ la garde ; ne pas mesurer dessus")

    # ------------------------------------------------------------------ l'or du banc
    index = ChunkIndex.load(verbose=False)
    items = ([("v1", it) for it in load_v1(index)]
             + [("v3", it) for it in load_bench(HERE / "questions-v3.jsonl")]
             + [("v4_formula", it) for it in familles_v4.lire_jsonl(HERE / "questions-v4-formula.jsonl")]
             + [("v4_table_cell", it) for it in familles_v4.lire_jsonl(HERE / "questions-v4-table-cell.jsonl")])
    gold = {f"{b}/{it['qid']}": set(it.get("gold_chunks") or []) for b, it in items}
    tous_ors = set().union(*gold.values()) if gold else set()
    dans_population = set(population)
    q_touchees = sorted(k for k, ors in gold.items() if ors & dans_population)
    print(f"\nquestions : {len(items)} ({dict(Counter(b for b, _ in items))}) · ors distincts {len(tous_ors)}")
    print(f"  ors dont la chaîne change            : {len(tous_ors & dans_population)}/{len(tous_ors)}")
    print(f"  questions dont au moins un or change : {len(q_touchees)}/{len(items)} "
          f"({dict(Counter(k.split('/')[0] for k in q_touchees))})")

    rng = random.Random(SEED)
    resultat = {
        "graine": SEED,
        "transformation": "rag/benchmark/recollage.py (définition unique, partagée)",
        "corpus": {"passages": len(rows), "porteurs_dollar": len(porteurs_dollar),
                   "porteurs_double_dollar": len(porteurs_double),
                   "porteurs_region_maths": len(porteurs_region)},
        "garde": {"regions_candidates": len(regions), "acceptees": len(acceptees),
                  "refusees_faute_de_signature": len(refus_signature),
                  "refusees_par_le_test_de_prose": len(refus_prose)},
        "invariants": {
            "ce_lot": {"I1": len(i1), "I2": len(i2), "I3": i3},
            "sans_garde": {"I1": len(i1_sg), "I2": len(i2_sg), "I2_exemples": i2_sg[:10]},
            "recollage_e11eac6_atomes": {"I3_regions": i3_sonde, "sur": len(regions)},
        },
        "population": {"n": len(population), "part_du_corpus": round(len(population) / len(rows), 4),
                       "blancs_retires": blancs,
                       "longueur_mediane_avant": int(statistics.median(longueurs_avant)),
                       "longueur_mediane_apres": int(statistics.median(longueurs_apres))},
        "banc": {"questions": len(items), "par_famille": dict(Counter(b for b, _ in items)),
                 "ors_distincts": len(tous_ors),
                 "ors_dont_la_chaine_change": len(tous_ors & dans_population),
                 "questions_dont_un_or_change": len(q_touchees),
                 "par_famille_questions_touchees": dict(Counter(k.split("/")[0] for k in q_touchees))},
        "exemples": {
            "acceptees": [{"region": r[:260], "recolle": R.recoller(r)[:260]}
                          for r in rng.sample(acceptees, min(12, len(acceptees)))],
            "refusees_prose": [{"region": r[:260], "mots": R.mots_de_prose(r)[:6],
                                "si_on_forcait": R.recoller(r)[:260]}
                               for r in rng.sample(refus_prose, min(12, len(refus_prose)))],
            "refusees_signature": [{"region": r[:200], "si_on_forcait": R.recoller(r)[:200]}
                                   for r in rng.sample(refus_signature, min(8, len(refus_signature)))],
        },
        "appels_llm": 0,
    }
    RESULT.write_text(json.dumps(resultat, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n-> {RESULT}")
    print("\n----- 4 régions acceptées -----")
    for ex in resultat["exemples"]["acceptees"][:4]:
        print(f"  AVANT {ex['region'][:140]!r}\n  APRÈS {ex['recolle'][:140]!r}")
    print("\n----- 6 régions refusées par le test de prose (ce qu'on aurait corrompu) -----")
    for ex in resultat["exemples"]["refusees_prose"][:6]:
        print(f"  {ex['mots']} :: {ex['region'][:120]!r}")


if __name__ == "__main__":
    main()
