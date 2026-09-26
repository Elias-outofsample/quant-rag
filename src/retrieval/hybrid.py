"""Rank-based fusion shared by hybrid retrieval callers."""

def reciprocal_rank_fusion(dense_hits, lexical_hits, rrf_k=60, dense_weight=1.0, lexical_weight=1.0):
    candidates = {}
    for source, hits, weight in (("dense", dense_hits, dense_weight), ("bm25", lexical_hits, lexical_weight)):
        for rank, hit in enumerate(hits, 1):
            chunk_id = hit["chunk_id"]
            candidate = candidates.setdefault(chunk_id, {"chunk_id": chunk_id, **hit, "dense_rank": None, "dense_score": None, "bm25_rank": None, "bm25_score": None, "rrf_score": 0.0})
            candidate[f"{source}_rank"] = rank
            candidate[f"{source}_score"] = hit["score"]
            candidate["rrf_score"] += weight / (rrf_k + rank)
    return sorted(candidates.values(), key=lambda item: (-item["rrf_score"], min(rank for rank in (item["dense_rank"], item["bm25_rank"]) if rank is not None), item["chunk_id"]))
