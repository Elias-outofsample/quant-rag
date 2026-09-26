"""La garde de non-régression du §11 — et elle se décide **hors ligne**, sans un appel LLM.

Le §11 du pré-enregistrement lui donne un titre — « le niveau réponse » — et deux composantes,
dont une seule est décisive :

**Le critère d'échec est un comptage.** *« Population : les questions v3 dont un chunk d'or est
dans les cinq passages servis à ``5530cba145``… La garde échoue au plus petit ``b`` dont la
borne haute de l'IC95 de Wilson de ``b/n`` dépasse 15 %. »* Une perte est une question qui
avait son or dans les cinq passages servis et ne l'a plus. Cela se compte sur les classements,
**avant tout appel LLM** — le §6 exige d'ailleurs cet ordre, et c'est la leçon du chantier
routage, où l'ordre pré-enregistré avait été enfreint par un ``--limit 0`` falsy.

**Le témoin ≥ −0,15 est une condition d'instrument**, et il demande des appels. Il ne peut être
lu que si le comptage passe : une garde déjà échouée ne se rachète pas au niveau réponse.

Ce que ce module ne fait pas
----------------------------
Il ne juge rien, ne génère rien, n'appelle aucun modèle et n'écrit qu'un verdict JSON. Il lit
Qdrant pour reproduire la sélection **de production** — ``limit=5``, ``per_document=2``,
``MIN_CHARACTERS=250``, déduplication — parce que la garde porte sur ce que l'utilisateur
reçoit, pas sur le classement brut.

    .venv/bin/python rag/benchmark/garde_reponse.py --candidat 8d4ee77f1f
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(ROOT / "rag" / "ingestion"))

import pipeline  # noqa: E402
import quant_rag  # noqa: E402
from build_collection_candidat import nom_collection  # noqa: E402
from compare_v1_v2 import load_bench  # noqa: E402
from corpus import ChunkIndex  # noqa: E402
from eval_dense_candidat import index_candidat  # noqa: E402

#: Part de pertes au-delà de laquelle la garde échoue, lue sur la borne HAUTE de l'IC95.
PLAFOND = 0.15
PASSAGES = 5


def wilson_haut(b: int, n: int, z: float = 1.959963985) -> float:
    if not n:
        return 1.0
    p = b / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    return centre + z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d


def seuil_echec(n: int) -> int:
    """Plus petit ``b`` dont la borne haute dépasse le plafond."""
    return next(b for b in range(n + 1) if wilson_haut(b, n) > PLAFOND)


def or_servi(collection: str, index, fichier_v3: pathlib.Path) -> dict[str, bool]:
    """Pour chaque question v3 positive : son or est-il dans les cinq passages **servis** ?"""
    precedente, quant_rag.COLLECTION = quant_rag.COLLECTION, collection
    try:
        out = {}
        for item in load_bench(fichier_v3):
            filtres = {k: v for k, v in (pipeline.filters_of(item) or {}).items()
                       if k in pipeline.FILTER_KEYS and v is not None}
            rows = quant_rag.search(pipeline.query_of(item), limit=PASSAGES, mode="dense",
                                    rerank=False, auto_period=False, log=False, **filtres)
            out[item["qid"]] = bool({r["chunk_id"] for r in rows} & set(item["gold_chunks"]))
        return out
    finally:
        quant_rag.COLLECTION = precedente


def garde(signature: str) -> dict:
    reference = or_servi(quant_rag.COLLECTION, ChunkIndex.load(verbose=False),
                         HERE / "questions-v3.jsonl")
    population = sorted(q for q, servi in reference.items() if servi)
    n = len(population)
    b_echec = seuil_echec(n)
    index = index_candidat(signature)
    bras = {}
    for nom in ("c1", "c2"):
        candidat = or_servi(nom_collection(signature, nom), index,
                            HERE / f"questions-v3-{signature}.jsonl")
        pertes = sorted(q for q in population if not candidat.get(q))
        gains = sorted(q for q, servi in candidat.items() if servi and not reference.get(q))
        b = len(pertes)
        bras[nom] = {"b": b, "pertes": pertes, "g": len(gains), "gains": gains,
                     "part": round(b / n, 4), "wilson_haut": round(wilson_haut(b, n), 4),
                     "tenue": b < b_echec}
    return {
        "signature_candidate": signature,
        "regle": ("§11 du pré-enregistrement — la garde échoue au plus petit b dont la borne "
                  "haute de l'IC95 de Wilson de b/n dépasse 15 %"),
        "population": {"n": n, "definition": "questions v3 dont l'or est dans les 5 passages "
                                             "servis à la référence 5530cba145", "qids": population},
        "seuil_echec": b_echec,
        "bras": bras,
        "appels_llm": 0,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--candidat", required=True, metavar="SIG")
    a = p.parse_args()
    r = garde(a.candidat)
    (HERE / f"garde-{a.candidat}.json").write_text(
        json.dumps(r, ensure_ascii=False, indent=1), encoding="utf-8")
    n, seuil = r["population"]["n"], r["seuil_echec"]
    print(f"population n = {n}   seuil d'échec : b >= {seuil}   (aucun appel LLM)")
    for nom, v in r["bras"].items():
        print(f"  {nom.upper():<3} b={v['b']:<3} g={v['g']:<3} b/n={v['part']:.3f}  "
              f"Wilson haut {100 * v['wilson_haut']:5.1f} %  ->  "
              f"{'TENUE' if v['tenue'] else 'ÉCHOUÉE'}")
        print(f"      pertes : {v['pertes']}")


if __name__ == "__main__":
    main()
