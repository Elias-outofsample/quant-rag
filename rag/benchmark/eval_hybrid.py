"""Compare dense seul / BM25 seul / hybride RRF sur le benchmark known-item.

L'index BM25 est celui de la production (``quant_rag.bm25()``), bâti sur l'état
courant du corpus et nommé par sa signature (l'amont ne l'avait construit que
sur 941 et 6 624 chunks d'anciens corpus), en réutilisant src/retrieval/.
"""
from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(ROOT / "src"))

import quant_rag  # noqa: E402
from retrieval.hybrid import reciprocal_rank_fusion  # noqa: E402

CUTOFFS = (1, 3, 5, 10)
POOL = 50


def build_bm25():
    """L'index de production, nommé par la signature de l'état du corpus."""
    started = time.perf_counter()
    index = quant_rag.bm25()
    print(f"  BM25 {quant_rag.BM25_PATH.name} : {index.doc_count} chunks ({time.perf_counter() - started:.0f} s)")
    return index


def summarise(ranks):
    hits = [r for r in ranks if r is not None]
    out = {f"recall@{k}": sum(1 for r in hits if r <= k) / len(ranks) for k in CUTOFFS}
    out["MRR"] = sum(1 / r for r in hits) / len(ranks)
    out["misses"] = len(ranks) - len(hits)
    return out


def with_text(row):
    """Complète un candidat BM25 avec son texte, lu dans le payload Qdrant."""
    from qdrant_client import models

    found, _ = quant_rag.client().scroll(
        collection_name=quant_rag.COLLECTION, limit=1, with_payload=True,
        scroll_filter=models.Filter(must=[models.FieldCondition(
            key="chunk_id", match=models.MatchValue(value=row["chunk_id"]))]),
    )
    if found:
        row = {**row, "text": found[0].payload.get("text", ""),
               "title": found[0].payload.get("title")}
    return row


def rank_of(rows, key, value):
    return next((i for i, row in enumerate(rows, 1) if row.get(key) == value), None)


#: L'export initial **et** le relais des chunks entrés par livraison. Le premier ne bouge
#: plus : seul, il ignore 62 des 318 documents actifs. Aucune question du banc known-item ne
#: vise aujourd'hui un chunk importé — mais une question qui en viserait un ferait sortir ce
#: script sur un KeyError, ce qui est la bonne panne pour la mauvaise raison.
SOURCES_ROWS = (ROOT / "data/embeddings/ingested-all-qwen3-06b/rows.jsonl",
                ROOT / "rag/ingestion/imported-rows.jsonl")


def document_of_chunk(chunk_ids: set[str] | None = None) -> dict[str, str]:
    """``chunk_id`` → ``document_id`` sur tout le corpus servi, export et imports réunis."""
    doc_of: dict[str, str] = {}
    for source in SOURCES_ROWS:
        if not source.exists():
            continue
        for line in source.open(encoding="utf-8"):
            if not line.strip():
                continue
            chunk = json.loads(line)["chunk"]
            if chunk_ids is not None and chunk["chunk_id"] not in chunk_ids:
                continue
            doc_of.setdefault(chunk["chunk_id"], chunk["document_id"])
    return doc_of


def main():
    bm25 = build_bm25()
    questions = [json.loads(l) for l in (HERE / "questions-v1.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    doc_of = document_of_chunk({q["target_chunk"] for q in questions})
    manquantes = [q["qid"] for q in questions if q["target_chunk"] not in doc_of]
    if manquantes:
        sys.exit(f"ÉCHEC : {len(manquantes)} question(s) visent un chunk introuvable dans le corpus servi : "
                 f"{', '.join(manquantes[:5])}")

    modes = ("dense", "bm25", "hybride RRF", "hybride RRF + rerank")
    results = {m: {"chunk": [], "doc": []} for m in modes}

    for q in questions:
        target, target_doc = q["target_chunk"], doc_of[q["target_chunk"]]
        dense = quant_rag.search(q["question"], limit=POOL, pool=POOL, mode="dense", rerank=False,
                                 dedupe=False, per_document=0, min_characters=0, log=False)
        lex = [{"chunk_id": r["chunk_id"], "document_id": r["document_id"], "score": s}
               for r, s in bm25.search(q["question"], limit=POOL)]
        fused = reciprocal_rank_fusion(dense, lex)
        by_id = {r["chunk_id"]: r for r in dense}
        # Les candidats issus uniquement de BM25 n'ont pas de texte : on va le chercher
        # dans le payload Qdrant, sinon le reranker les ignorerait silencieusement.
        fused_full = [by_id.get(r["chunk_id"]) or with_text(r) for r in fused]
        reranked = quant_rag._rerank(q["question"], [r for r in fused_full if r.get("text")][:POOL])

        for name, rows in zip(modes, (dense, lex, fused, reranked)):
            results[name]["chunk"].append(rank_of(rows, "chunk_id", target))
            results[name]["doc"].append(rank_of(rows, "document_id", target_doc))
        print(f"  {q['qid']} " + "  ".join(
            f"{n.split()[0][:6]}={str(results[n]['chunk'][-1] or '-'):>3}" for n in modes))

    print(f"\n{len(questions)} questions, pool={POOL}\n")
    for level, label in (("chunk", "CHUNK EXACT"), ("doc", "DOCUMENT")):
        print(f"--- {label}")
        print(f"{'':<24}" + "".join(f"{f'R@{k}':>8}" for k in CUTOFFS) + f"{'MRR':>9}{'ratés':>8}")
        for m in modes:
            s = summarise(results[m][level])
            print(f"{m:<24}" + "".join(f"{s[f'recall@{k}']:>8.2f}" for k in CUTOFFS) + f"{s['MRR']:>9.3f}{s['misses']:>8}")
        print()
    (HERE / "results-hybrid-v1.json").write_text(json.dumps(
        {m: {lv: summarise(results[m][lv]) for lv in ("chunk", "doc")} for m in modes}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
