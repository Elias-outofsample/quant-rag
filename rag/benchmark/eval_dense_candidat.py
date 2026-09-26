"""Métrique de décision du chantier de représentation — **configuration `dense`, et elle seule**.

Phase 5 du pré-enregistrement. Aucun appel LLM. La collection servie
``quant_rag_ingested_all_qwen3_06b`` n'est jamais écrite ; elle n'est lue que par le contrôle
d'équivalence ci-dessous.

Pourquoi ce module plutôt que `calibrate_router` + `eval_router`
-----------------------------------------------------------------
Le §3.1 du pré-enregistrement fixe que **la décision se lit sur la configuration `dense`, et
que les lignes `rrf_rerank` et `router` sont diagnostiques, jamais décisionnelles** : la
correction commune change ``title_path`` pour 12 476 chunks, donc l'index BM25 par rapport à
la ligne de base, et toute configuration lexicale mêlerait un effet dense à un effet lexical.

Faire tourner `calibrate_router` sur les trois bras aurait demandé de rendre
``corpus_overlay.signature()``, ``TABLES`` et ``quant_rag.COLLECTION`` surchargeables par
l'environnement — c'est-à-dire d'ajouter à la production trois interrupteurs dont le mode de
défaillance est exactement celui que ce dossier a déjà payé deux fois : *un index juste sous
un nom qui ment*. Ce module fait l'inverse : il n'ajoute rien à la production, il emprunte le
**chemin de récupération de la production** (``pipeline.retrieve_item(..., "dense")``, qui
appelle ``quant_rag.search(mode="dense", rerank=False)``) et la **métrique du dossier**
(``metrics.ndcg`` avec ``gold_poids``).

Et il prouve qu'il est fidèle plutôt que de l'affirmer : ``--controle`` le fait tourner sur la
collection **servie** avec les questions **d'origine**, et exige de retrouver la ligne de base
publiée — v1 **0,897**, v3 **0,542**, pooled **0,599** (``results-router-v3.json``). Une
divergence, et rien de ce qui suit ne vaut.

Ce qui est calculé, et rien d'autre
------------------------------------
- population **fixe** : 155 questions (25 v1 + 130 v3 positives — ``load_bench`` écarte les
  20 négatives, ``kind != "negative"``) ;
- les questions **non comparables** restent au dénominateur et valent **0** ;
- ``gold_poids`` est lu quand il existe : sans lui l'or pondéré serait décoratif et le piège
  de comptage de ``metrics.ndcg`` se rouvrirait.

    .venv/bin/python rag/benchmark/eval_dense_candidat.py --controle
    .venv/bin/python rag/benchmark/eval_dense_candidat.py --candidat 8d4ee77f1f --bras c1
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(ROOT / "rag" / "ingestion"))

import corpus_overlay  # noqa: E402
import gel_corpus  # noqa: E402
import metrics  # noqa: E402
import pipeline  # noqa: E402
import quant_rag  # noqa: E402
from build_collection_candidat import nom_collection  # noqa: E402
from compare_v1_v2 import load_bench, load_v1  # noqa: E402
from corpus import ChunkIndex  # noqa: E402

PROCESSED = ROOT / "data" / "processed"
#: La ligne de base publiée, telle que results-router-v3.json la porte. Le contrôle
#: d'équivalence les exige au millième.
BASELINE = {"v1": 0.897, "v3": 0.542, "pooled": 0.599}


def index_candidat(signature: str) -> ChunkIndex:
    """La vue corpus du candidat, en mémoire — jamais un cache, jamais un nom partagé.

    ``ChunkIndex.load()`` nomme son cache par la signature **vivante** : l'utiliser pour un
    candidat écrirait un fichier qui ment sur son contenu. On construit donc la vue à la main,
    à partir des artefacts du candidat, et elle ne survit pas au processus.
    """
    racine = PROCESSED / f"candidat-{signature}"
    servis = {r["chunk_id"]: r["text"] for r in
              (json.loads(l) for l in (racine / "chunks-servis.jsonl").open(encoding="utf-8") if l.strip())}
    chunks, documents = {}, {}
    for row in (json.loads(l) for l in (racine / "rows.jsonl").open(encoding="utf-8") if l.strip()):
        chunk, document = row["chunk"], row["document"]
        chunks[chunk["chunk_id"]] = {
            "chunk_id": chunk["chunk_id"], "document_id": chunk["document_id"],
            "parent_id": chunk.get("parent_id"),
            "section": chunk.get("section") or chunk.get("title_path") or "",
            "page_start": chunk.get("page_start"), "content_type": chunk.get("content_type") or "",
            "token_count": chunk.get("token_count"), "text": servis[chunk["chunk_id"]]}
        documents.setdefault(chunk["document_id"], {
            "document_id": chunk["document_id"], "title": document.get("title") or "",
            "filename": document.get("filename") or "", "chunks": 0})
        documents[chunk["document_id"]]["chunks"] += 1
    return ChunkIndex(chunks, documents, Counter(), len(chunks))


def mesurer(item: dict, ranked: list) -> dict:
    gold_chunks, gold_documents = set(item["gold_chunks"]), set(item["gold_documents"])
    poids = item.get("gold_poids") or None
    rows = [{"chunk_id": r[0], "document_id": r[1]} if isinstance(r, (list, tuple)) else r for r in ranked]
    return {"first_rank": metrics.rank_of(rows, "chunk_id", gold_chunks),
            "all_found_at": None, "gold_count": len(gold_chunks),
            "comparable": item.get("comparable", True),
            "ndcg": metrics.ndcg(rows, gold_chunks, gold_documents, poids=poids),
            "doc_ndcg": metrics.ndcg_documents(rows, gold_documents),
            "doc_rank": metrics.rank_of(rows, "document_id", gold_documents)}


def evaluer(collection: str, fichier_v1: Path, fichier_v3: Path, index) -> list[dict]:
    """Le classement dense de chaque question, sur la collection demandée."""
    precedente, quant_rag.COLLECTION = quant_rag.COLLECTION, collection
    try:
        resultats = []
        for banc, items in (("v1", load_v1(index, fichier_v1)), ("v3", load_bench(fichier_v3))):
            for n, item in enumerate(items, 1):
                ranked = pipeline.retrieve_item(item, "dense", index, limit=pipeline.POOL)
                resultats.append({"bench": banc, "qid": item["qid"], "kind": item.get("kind", "single"),
                                  **mesurer(item, ranked)})
                if n % 25 == 0:
                    print(f"    {banc} {n}/{len(items)}", end="\r", flush=True)
        print()
        return resultats
    finally:
        quant_rag.COLLECTION = precedente


def resume(resultats: list[dict]) -> dict:
    def moyenne(sous: list[dict]) -> dict:
        return {"n": len(sous),
                "nDCG@10": round(statistics.mean(r["ndcg"] for r in sous), 4) if sous else None,
                "comparables": sum(1 for r in sous if r["comparable"]),
                "ratés": sum(1 for r in sous if r["first_rank"] is None)}
    return {"v1": moyenne([r for r in resultats if r["bench"] == "v1"]),
            "v3": moyenne([r for r in resultats if r["bench"] == "v3"]),
            "pooled": moyenne(resultats)}


def controle() -> int:
    """La collection SERVIE, les questions d'ORIGINE : la ligne de base doit revenir."""
    index = ChunkIndex.load(verbose=False)
    resultats = evaluer(quant_rag.COLLECTION, HERE / "questions-v1.jsonl",
                        HERE / "questions-v3.jsonl", index)
    r = resume(resultats)
    echecs = 0
    print(f"  contrôle d'équivalence — collection servie, questions d'origine, signature "
          f"{corpus_overlay.signature()}")
    for vue, attendu in BASELINE.items():
        obtenu = r[vue]["nDCG@10"]
        ok = abs(obtenu - attendu) < 5e-4
        print(f"    {vue:<7} n={r[vue]['n']:<4} nDCG@10 {obtenu:.4f}   publié {attendu:.3f}   "
              f"{'OK' if ok else 'ÉCART'}")
        echecs += not ok
    # Le nom suit la signature VIVANTE. Il a été codé en dur sur 5530cba145 jusqu'au
    # 9 septembre 2026 : un contrôle joué sur un autre corpus s'écrivait alors sous le nom du
    # précédent — mesure juste, nom faux — et écrasait la référence que lit
    # verdict_representation. ``audit_pertes`` construisait déjà le nom de cette façon.
    (HERE / f"results-dense-controle-{corpus_overlay.signature()}.json").write_text(
        json.dumps({"corpus": corpus_overlay.describe(), "summary": r, "per_question": resultats},
                   ensure_ascii=False, indent=1), encoding="utf-8")
    return 1 if echecs else 0


def bras(signature: str, nom_bras: str) -> dict:
    collection = nom_collection(signature, nom_bras)
    if not quant_rag.client().collection_exists(collection):
        sys.exit(f"collection absente : {collection}")
    index = index_candidat(signature)
    fichier_v1 = HERE / f"questions-v1-{signature}.jsonl"
    fichier_v3 = HERE / f"questions-v3-{signature}.jsonl"
    for chemin in (fichier_v1, fichier_v3):
        if not chemin.exists():
            sys.exit(f"banc ré-oré introuvable : {chemin}")
    resultats = evaluer(collection, fichier_v1, fichier_v3, index)
    r = resume(resultats)
    sortie = {"bras": nom_bras, "signature_candidate": signature, "collection": collection,
              "signature_vivante": corpus_overlay.signature(), "gel": gel_corpus.etat(),
              "configuration": "dense (limit=POOL=50, rerank=False, dedupe=False, per_document=0)",
              "summary": r, "per_question": resultats}
    chemin = HERE / f"results-dense-{signature}-{nom_bras}.json"
    chemin.write_text(json.dumps(sortie, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"  {nom_bras}  v1 {r['v1']['nDCG@10']}  v3 {r['v3']['nDCG@10']}  "
          f"pooled {r['pooled']['nDCG@10']}  ({chemin.name})")
    return sortie


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--controle", action="store_true",
                   help="collection servie + questions d'origine : la ligne de base doit revenir")
    p.add_argument("--candidat", metavar="SIG")
    p.add_argument("--bras", choices=("c1", "c2", "sabote"))
    a = p.parse_args()
    if a.controle:
        code = controle()
        if code:
            sys.exit("ÉCART sur la ligne de base : l'évaluateur ne reproduit pas le publié, on s'arrête")
        if not a.candidat:
            return
    if not (a.candidat and a.bras):
        p.print_help()
        return
    bras(a.candidat, a.bras)


if __name__ == "__main__":
    main()
