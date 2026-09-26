"""Outillage commun des expériences de retrieval (chantiers titres, réécriture, rerankers, protocole).

Ce que chaque expérience partage, et qui ne doit exister qu'une fois :

  - les 155 questions des deux bancs (v1 known-item, v3 ouvert — positives seulement),
    avec le texte soumis au retrieval et la portée bibliographique des questions datées ;
  - la mesure par question (rang du premier chunk d'or, nDCG@10 chunk et document) ;
  - les synthèses par banc et par famille, les écarts appariés avec IC95 bootstrap
    contre une configuration de référence, et la liste des gains / pertes nets.

Rien ici n'appelle un LLM ni Qdrant : les classements viennent de ``dense_matrix``
(prouvé identique à Qdrant sur les 155 questions) ou du cache de calibration.
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import metrics  # noqa: E402
import pipeline  # noqa: E402
import corpus_overlay  # noqa: E402
import quant_rag  # noqa: E402
from compare_v1_v2 import load_bench, load_v1  # noqa: E402
from corpus import ChunkIndex  # noqa: E402

#: Le nom porte la signature de l'état du corpus : un cache de classements calculé
#: sur un autre corpus n'est pas rechargé, il n'est pas trouvé. Il le fallait — un
#: import ajoute des chunks qui concourent contre l'or sans que l'or bouge.
CACHE = HERE / ".cache" / f"router-retrievals-{corpus_overlay.signature()}.json"
KINDS = ("single", "table", "multi", "dated", "exact")
BENCHES = ("v1", "v3", "pooled")


def load_items(index: ChunkIndex | None = None) -> list[dict]:
    """Les questions positives des deux bancs, chacune avec ``bench`` et ``key``."""
    index = index or ChunkIndex.load(verbose=False)
    items = []
    for item in load_v1(index):
        items.append({**item, "bench": "v1", "key": f"v1/{item['qid']}"})
    for item in load_bench(HERE / "questions-v3.jsonl"):
        items.append({**item, "bench": "v3", "key": f"v3/{item['qid']}", "kind": item.get("kind", "single")})
    return items


def scope_of(item: dict) -> list[str] | None:
    filters = pipeline.filters_of(item)
    return quant_rag.document_scope(None, **filters) if filters else None


def cached_rankings() -> dict:
    """Classements Qdrant du cache de calibration : {key: {config: [(chunk, doc, score), ...]}}."""
    if not CACHE.exists():
        sys.exit("cache de calibration absent : lance d'abord calibrate_router.py")
    return json.loads(CACHE.read_text(encoding="utf-8"))


def rows_from_cache(entry: list) -> list[dict]:
    return [{"chunk_id": c, "document_id": d, "score": s} for c, d, s in entry]


def measure(item: dict, rows: list[dict]) -> dict:
    gold_chunks, gold_documents = set(item["gold_chunks"]), set(item["gold_documents"])
    ranks = metrics.ranks_of_all(rows, "chunk_id", gold_chunks)
    found = [r for r in ranks.values() if r is not None]
    return {"first_rank": min(found) if found else None,
            "all_found_at": max(ranks.values()) if all(r is not None for r in ranks.values()) else None,
            "gold_count": len(gold_chunks),
            "ndcg": metrics.ndcg(rows, gold_chunks, gold_documents),
            "doc_ndcg": metrics.ndcg_documents(rows, gold_documents),
            "doc_rank": metrics.rank_of(rows, "document_id", gold_documents)}


def rrf(rankings: list[list[dict]], k: int = 60, weights: list[float] | None = None) -> list[dict]:
    """Fusion réciproque de n classements (même constante k=60 que ``src/retrieval/hybrid.py``)."""
    weights = weights or [1.0] * len(rankings)
    fused: dict[str, dict] = {}
    for weight, ranking in zip(weights, rankings):
        for rank, row in enumerate(ranking, 1):
            entry = fused.setdefault(row["chunk_id"], {"chunk_id": row["chunk_id"], "document_id": row["document_id"],
                                                        "score": 0.0, "best_rank": rank, "score_kind": "rrf"})
            entry["score"] += weight / (k + rank)
            entry["best_rank"] = min(entry["best_rank"], rank)
    return sorted(fused.values(), key=lambda r: (-r["score"], r["best_rank"], r["chunk_id"]))


# ---------------------------------------------------------------------- synthèse

def summarise(per_question: list[dict], configs: list[str], reference: str) -> dict:
    """``per_question`` : [{bench, qid, kind, <config>: record, ...}]."""
    out: dict = {"summary": {}, "paired": {}, "by_kind": {}}
    for bench in BENCHES:
        subset = [r for r in per_question if bench == "pooled" or r["bench"] == bench]
        if not subset:
            continue
        out["summary"][bench] = {}
        for config in configs:
            recs = [r[config] for r in subset]
            s = metrics.summarise(recs)
            s["doc_nDCG@10"] = round(statistics.mean(r["doc_ndcg"] for r in recs), 3)
            s["doc_recall@10"] = round(sum(1 for r in recs if r["doc_rank"] and r["doc_rank"] <= 10) / len(recs), 3)
            out["summary"][bench][config] = s
        out["paired"][bench] = {config: metrics.paired_delta([r[reference]["ndcg"] for r in subset], [r[config]["ndcg"] for r in subset])
                                for config in configs if config != reference}
    kinds = [k for k in KINDS if any(r["bench"] == "v3" and r["kind"] == k for r in per_question)]
    for kind in kinds:
        sub = [r for r in per_question if r["bench"] == "v3" and r["kind"] == kind]
        out["by_kind"][kind] = {"n": len(sub),
                                **{config: round(statistics.mean(r[config]["ndcg"] for r in sub), 3) for config in configs},
                                "paired_vs_reference": {config: metrics.paired_delta([r[reference]["ndcg"] for r in sub], [r[config]["ndcg"] for r in sub])
                                                        for config in configs if config != reference}}
    return out


def wins_losses(per_question: list[dict], config: str, reference: str, threshold: float = 0.15) -> dict:
    wins, losses = [], []
    for r in per_question:
        delta = r[config]["ndcg"] - r[reference]["ndcg"]
        entry = {"key": r["key"], "kind": r["kind"], "rank_reference": r[reference]["first_rank"],
                 "rank_config": r[config]["first_rank"], "delta_ndcg": round(delta, 3)}
        if delta >= threshold:
            wins.append(entry)
        elif delta <= -threshold:
            losses.append(entry)
    wins.sort(key=lambda e: -e["delta_ndcg"])
    losses.sort(key=lambda e: e["delta_ndcg"])
    return {"threshold": threshold, "wins": wins, "losses": losses, "n_wins": len(wins), "n_losses": len(losses)}


def print_summary(result: dict, configs: list[str], reference: str) -> None:
    print(f"\n{'':<10}{'config':<18}{'n':>5}{'nDCG':>7}{'R@1':>6}{'R@5':>6}{'R@10':>6}{'MRR':>7}{'docNDCG':>9}{'ratés':>7}{'Δ vs ' + reference + ' [IC95]':>26}")
    for bench in BENCHES:
        if bench not in result["summary"]:
            continue
        for config in configs:
            s = result["summary"][bench][config]
            pd = result["paired"][bench].get(config)
            cell = f"{pd['delta']:+.3f} [{pd['ci95'][0]:+.3f},{pd['ci95'][1]:+.3f}]{'*' if pd['significant'] else ' '}" if pd else ""
            print(f"{bench:<10}{config:<18}{s['n']:>5}{s['nDCG@10']:>7.3f}{s['recall@1']:>6.2f}{s['recall@5']:>6.2f}{s['recall@10']:>6.2f}"
                  f"{s['MRR']:>7.3f}{s['doc_nDCG@10']:>9.3f}{s['misses']:>7}{cell:>26}")
    print(f"\n=== v3 par famille (nDCG@10 chunk) ===")
    print(f"{'famille':<9}{'n':>4}" + "".join(f"{c[:14]:>15}" for c in configs))
    for kind, block in result["by_kind"].items():
        print(f"{kind:<9}{block['n']:>4}" + "".join(f"{block[c]:>15.3f}" for c in configs))
    for config in configs:
        if config == reference:
            continue
        cells = "  ".join(f"{k} {b['paired_vs_reference'][config]['delta']:+.3f}"
                          f"[{b['paired_vs_reference'][config]['ci95'][0]:+.2f},{b['paired_vs_reference'][config]['ci95'][1]:+.2f}]"
                          f"{'*' if b['paired_vs_reference'][config]['significant'] else ''}" for k, b in result["by_kind"].items())
        print(f"  {config:<18} {cells}")
