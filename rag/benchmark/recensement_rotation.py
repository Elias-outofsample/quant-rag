"""Le recensement nominatif de la rotation — qui perd, qui gagne, et quoi.

**Pourquoi ce module existe.** Le §21 ter a mesuré qu'aucune garde statistique ne peut trancher
sur ce banc : `gel1` demanderait un banc de **284** questions, `b30` de **1 458**, et il en
compte **155**. La seule sortie qui ne demande ni un banc dix fois plus gros ni un renoncement à
toute garantie est de décider **nominativement** — lire les questions que la règle casse et
celles qu'elle répare, une par une. Elles sont peu nombreuses : `gel1` en casse **5** et en
répare **19**.

Ce module met ce recensement sous les yeux de qui doit trancher. **Il ne tranche rien lui-même**,
ne produit aucun verdict et n'a aucune règle de décision : il **nomme**. Zéro appel LLM, Qdrant
jamais ouvert, aucune écriture en production.

Ce qu'il montre, pour chaque question qui bascule
-------------------------------------------------
son banc et sa famille, son texte, le **rang de son or** avant et après la règle, s'il est
**servi** dans chacun des deux bras, et la **longueur de son chunk d'or** — parce que le §21 bis
a montré que la longueur décide ailleurs dans la chaîne et qu'il faut savoir si les questions
cassées lui ressemblent.

Tout est dérivé de `.cache/reranking-scores-<signature>.json` et du cache de classements, par
les fonctions de l'instrument d'origine (`eval_reclassement_selectif.appliquer`,
`eval_reranking.servis`) — jamais une copie.

    .venv/bin/python rag/benchmark/recensement_rotation.py --variante gel1
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import corpus_overlay  # noqa: E402
import eval_reclassement_selectif as ers  # noqa: E402
import eval_reranking  # noqa: E402
import experiment  # noqa: E402
from corpus import ChunkIndex  # noqa: E402

SIGNATURE = corpus_overlay.signature()


def recenser(variante: str) -> dict:
    if not ers.SCORES.exists():
        sys.exit(f"{ers.SCORES.name} absent.")
    scores = json.loads(ers.SCORES.read_text(encoding="utf-8"))
    index = ChunkIndex.load(verbose=False)
    items = experiment.load_items(index)
    cache = experiment.cached_rankings()
    regles = {"b10": ers.ordre_b10, **ers.VARIANTES}
    if variante not in regles:
        sys.exit(f"variante inconnue : {variante} (attendu {sorted(regles)})")

    perdues, gagnees, inchangees = [], [], 0
    for item in items:
        cle = item["key"]
        gold = set(item["gold_chunks"])
        if not gold:
            continue
        rows = []
        for rang, (chunk_id, document_id, score) in enumerate(cache[cle]["dense"], 1):
            src = index.get(chunk_id) or {}
            rows.append({"chunk_id": chunk_id, "document_id": document_id, "score": score,
                         "rang_dense": rang, "text": src.get("text", ""),
                         "content_type": src.get("content_type")})
        apres_rows = ers.appliquer(rows, scores[cle], regles[variante])
        avant, apres = eval_reranking.or_servi(rows, gold), eval_reranking.or_servi(apres_rows, gold)
        if avant == apres:
            inchangees += 1
            continue

        def rang_de(liste):
            for r, ligne in enumerate(liste, 1):
                if ligne["chunk_id"] in gold:
                    return r
            return None

        longueurs = [len(index.get(c)["text"]) for c in gold if index.get(c)]
        fiche = {
            "cle": cle, "banc": item["bench"], "kind": item.get("kind", "single"),
            "question": item["question"],
            "rang_or_avant": rang_de(rows), "rang_or_apres": rang_de(apres_rows),
            "servis_avant": [l["chunk_id"] for l in eval_reranking.servis(rows)],
            "servis_apres": [l["chunk_id"] for l in eval_reranking.servis(apres_rows)],
            "or": sorted(gold),
            "longueur_or_max": max(longueurs) if longueurs else None,
        }
        (perdues if avant else gagnees).append(fiche)

    toutes = [f["longueur_or_max"] for f in perdues + gagnees if f["longueur_or_max"]]
    lp = [f["longueur_or_max"] for f in perdues if f["longueur_or_max"]]
    lg = [f["longueur_or_max"] for f in gagnees if f["longueur_or_max"]]
    return {
        "signature": SIGNATURE, "variante": variante, "appels_llm": 0, "qdrant": "jamais ouvert",
        "b": len(perdues), "g": len(gagnees), "rotation": len(perdues) + len(gagnees),
        "inchangees": inchangees,
        "longueur_or": {
            "perdues_mediane": statistics.median(lp) if lp else None,
            "gagnees_mediane": statistics.median(lg) if lg else None,
            "toutes_mediane": statistics.median(toutes) if toutes else None,
        },
        "perdues": perdues, "gagnees": gagnees,
    }


def imprimer(r: dict) -> None:
    print(f"\nvariante {r['variante']} · corpus {r['signature']} · "
          f"b = {r['b']}, g = {r['g']}, rotation = {r['rotation']} · appels LLM {r['appels_llm']}")
    lo = r["longueur_or"]
    print(f"longueur du chunk d'or — médiane des cassées {lo['perdues_mediane']}, "
          f"des réparées {lo['gagnees_mediane']}")
    for titre, cles in (("CASSÉES — elles avaient leur or servi et ne l'ont plus", "perdues"),
                        ("RÉPARÉES — elles n'avaient pas leur or servi et l'ont", "gagnees")):
        print(f"\n{titre}  ({len(r[cles])})")
        for f in r[cles]:
            print(f"  {f['cle']:10s} {f['kind']:8s} rang de l'or {str(f['rang_or_avant']):>4s} → "
                  f"{str(f['rang_or_apres']):>4s}   or de {f['longueur_or_max']} c.")
            print(f"     {f['question'][:150]}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--variante", default="gel1")
    a = p.parse_args()
    r = recenser(a.variante)
    (HERE / f"recensement-rotation-{a.variante}-{SIGNATURE}.json").write_text(
        json.dumps(r, ensure_ascii=False, indent=1), encoding="utf-8")
    imprimer(r)
    print(f"\nécrit : rag/benchmark/recensement-rotation-{a.variante}-{SIGNATURE}.json")


if __name__ == "__main__":
    main()
