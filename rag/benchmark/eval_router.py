"""Évalue la configuration ``router`` sur les deux bancs, et vérifie sa cohérence.

Le routeur ne fait rien de neuf : il *choisit* entre deux chemins déjà mesurés
(dense, rrf_rerank). Sur chaque question, son classement doit donc être identique
à celui du chemin choisi — même code, même pool. Toute divergence signale un bug de
plomberie (texte manquant, filtre parasite), pas une différence de qualité.

Les classements dense et rrf_rerank sont repris du cache de calibration
(``.cache/router-retrievals.json``) ; seul le routeur est exécuté ici.

    .venv/bin/python rag/benchmark/eval_router.py               # v1 + v3 -> results-router-v3.json
    .venv/bin/python rag/benchmark/eval_router.py --bench v2    # v1 + v2 -> results-router-v1.json (historique)
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))   # rag/ : corpus_overlay

import argparse  # noqa: E402

import metrics  # noqa: E402
import pipeline  # noqa: E402
import corpus_overlay  # noqa: E402
import gel_corpus  # noqa: E402  — la signature gelée doit figurer dans tout résultat
import quant_rag  # noqa: E402
from compare_v1_v2 import load_bench, load_v1  # noqa: E402
from corpus import ChunkIndex  # noqa: E402

#: Le nom porte la signature de l'état du corpus : un cache de classements calculé
#: sur un autre corpus n'est pas rechargé, il n'est pas trouvé. Il le fallait — un
#: import ajoute des chunks qui concourent contre l'or sans que l'or bouge.
CACHE = HERE / ".cache" / f"router-retrievals-{corpus_overlay.signature()}.json"
BASELINES = ("dense", "rrf_rerank")
OPEN_BENCH = "v3"


def _collection_documents() -> list[dict]:
    """document_id de chaque point servi — le compte de documents du corpus mesuré."""
    rows, offset = [], None
    while True:
        points, offset = quant_rag.client().scroll(quant_rag.COLLECTION, limit=8192, offset=offset,
                                                   with_payload=["document_id"], with_vectors=False)
        rows.extend(p.payload for p in points)
        if offset is None:
            break
    return rows


def output_path(bench: str) -> Path:
    return HERE / ("results-router-v1.json" if bench == "v2" else f"results-router-{bench}.json")


def measure(item: dict, ranked: list) -> dict:
    gold_chunks, gold_documents = set(item["gold_chunks"]), set(item["gold_documents"])
    # ``gold_poids`` n'existe que sur un banc ré-oré (gold_ancrage --appliquer). Absent, le
    # calcul est celui d'avant, au bit près. Présent, il DOIT être lu : sinon l'or pondéré
    # serait décoratif et le piège de comptage de metrics.ndcg se rouvrirait.
    poids = item.get("gold_poids") or None
    rows = [{"chunk_id": r[0], "document_id": r[1]} if isinstance(r, (list, tuple)) else r for r in ranked]
    rank = metrics.rank_of(rows, "chunk_id", gold_chunks)
    return {"first_rank": rank, "all_found_at": None, "gold_count": len(gold_chunks),
            "comparable": item.get("comparable", True),
            "ndcg": metrics.ndcg(rows, gold_chunks, gold_documents, poids=poids),
            "doc_ndcg": metrics.ndcg_documents(rows, gold_documents),
            "doc_rank": metrics.rank_of(rows, "document_id", gold_documents)}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--bench", default=OPEN_BENCH, choices=("v2", "v3"))
    parser.add_argument("--gold-signature", metavar="SIG",
                        help="joue le banc sur les questions ré-orées questions-<banc>-SIG.jsonl "
                             "(gold_ancrage --appliquer). La population reste celle d'origine : "
                             "les questions non comparables y sont présentes et valent 0.")
    args = parser.parse_args()
    open_bench = args.bench
    suffixe = f"-{args.gold_signature}" if args.gold_signature else ""
    fichier_v1 = HERE / f"questions-v1{suffixe}.jsonl"
    fichier_ouvert = HERE / f"questions-{open_bench}{suffixe}.jsonl"
    for chemin in (fichier_v1, fichier_ouvert):
        if not chemin.exists():
            sys.exit(f"banc introuvable : {chemin}")
    benches = ("v1", open_bench, "pooled")
    if not CACHE.exists():
        sys.exit("cache de calibration absent : lance d'abord calibrate_router.py")
    cache = json.loads(CACHE.read_text(encoding="utf-8"))
    index = ChunkIndex.load(verbose=False)
    v2_qids = {item["qid"] for item in load_bench(HERE / "questions-v2.jsonl")} if open_bench == "v3" else set()

    per_question, mismatches = [], []
    for bench, items in (("v1", load_v1(index, fichier_v1)), (open_bench, load_bench(fichier_ouvert))):
        print(f"{bench} : {len(items)} questions")
        for item in items:
            key = f"{bench}/{item['qid']}"
            if key not in cache:
                sys.exit(f"{key} absent du cache de calibration : lance d'abord calibrate_router.py --bench {open_bench}")
            decision = quant_rag.route(pipeline.query_of(item))
            ranked = pipeline.retrieve_item(item, "router", index, limit=pipeline.POOL)
            got = measure(item, ranked)
            expected_config = "rrf_rerank" if decision["mode"] == "hybrid" else "dense"
            expected = measure(item, cache[key][expected_config])
            consistent = abs(got["ndcg"] - expected["ndcg"]) < 1e-9 and got["first_rank"] == expected["first_rank"]
            if not consistent:
                mismatches.append({"key": key, "mode": decision["mode"], "router": got, "expected": expected})
            per_question.append({
                "bench": bench, "qid": item["qid"], "kind": item.get("kind", "single"),
                "in_v2": bench == "v1" or item["qid"] in v2_qids or open_bench == "v2",
                "filters": pipeline.filters_of(item) or None,
                "mode": decision["mode"], "n_exact": decision["exact_tokens"]["n_exact"],
                "exact_tokens": decision["exact_tokens"], "reason": decision["reason"],
                "router": got, "consistent": consistent,
                **{b: measure(item, cache[key][b]) for b in BASELINES},
            })
            print(f"  {item['qid']:<4} {decision['mode']:<7} chunk@{str(got['first_rank'] or '-'):>3}"
                  f"  nDCG {got['ndcg']:.2f} (dense {per_question[-1]['dense']['ndcg']:.2f},"
                  f" rrf_rerank {per_question[-1]['rrf_rerank']['ndcg']:.2f}){'' if consistent else '   !! INCOHÉRENT'}")

    # --- synthèse (le sous-ensemble v2 de v3 est rapporté aussi, pour la continuité)
    summary, paired = {}, {}
    views = list(benches) + (["v2-subset"] if open_bench == "v3" else [])
    for bench in views:
        if bench == "v2-subset":
            subset = [r for r in per_question if r["bench"] == open_bench and r["in_v2"]]
        else:
            subset = [r for r in per_question if bench == "pooled" or r["bench"] == bench]
        summary[bench] = {}
        for config in BASELINES + ("router",):
            recs = [r[config] for r in subset]
            s = metrics.summarise(recs)
            s["doc_nDCG@10"] = round(statistics.mean(r["doc_ndcg"] for r in recs), 3)
            s["doc_recall@10"] = round(sum(1 for r in recs if r["doc_rank"] and r["doc_rank"] <= 10) / len(recs), 3)
            summary[bench][config] = s
        summary[bench]["hybrid_share"] = round(sum(1 for r in subset if r["mode"] == "hybrid") / len(subset), 3)
        paired[bench] = {
            "router vs dense": metrics.paired_delta([r["dense"]["ndcg"] for r in subset], [r["router"]["ndcg"] for r in subset]),
            "router vs rrf_rerank": metrics.paired_delta([r["rrf_rerank"]["ndcg"] for r in subset], [r["router"]["ndcg"] for r in subset]),
        }

    print(f"\n=== cohérence : {len(mismatches)} divergence(s) sur {len(per_question)} questions ===")
    for m in mismatches:
        print(f"  {m['key']} ({m['mode']}) : routeur nDCG {m['router']['ndcg']:.3f} rang {m['router']['first_rank']}"
              f" | attendu {m['expected']['ndcg']:.3f} rang {m['expected']['first_rank']}")

    print("\n=== nDCG@10 chunk · R@1 chunk · R@10 chunk · nDCG@10 doc ===")
    print(f"{'':<10}{'config':<12}{'n':>5}{'nDCG':>7}{'R@1':>7}{'R@10':>7}{'docNDCG':>9}{'MRR':>7}{'ratés':>7}")
    for bench in views:
        for config in BASELINES + ("router",):
            s = summary[bench][config]
            print(f"{bench:<10}{config:<12}{s['n']:>5}{s['nDCG@10']:>7.3f}{s['recall@1']:>7.2f}{s['recall@10']:>7.2f}"
                  f"{s['doc_nDCG@10']:>9.3f}{s['MRR']:>7.3f}{s['misses']:>7}")
        print(f"{'':<10}part hybride : {summary[bench]['hybrid_share']:.0%}")
    print("\n=== écarts appariés nDCG@10 chunk, IC95 bootstrap ===")
    for bench in views:
        for label, pd in paired[bench].items():
            print(f"  {bench:<10}{label:<22}{pd['delta']:+.3f}  [{pd['ci95'][0]:+.3f}, {pd['ci95'][1]:+.3f}]{'  *' if pd['significant'] else ''}")

    # --- par famille du banc ouvert
    by_kind = {}
    kinds = sorted({r["kind"] for r in per_question if r["bench"] == open_bench})
    print(f"\n=== {open_bench} par famille (nDCG@10 chunk) ===")
    print(f"{'famille':<10}{'n':>4}{'dense':>8}{'rrf_rr':>8}{'router':>8}{'part hyb':>10}{'router−dense [IC95]':>26}")
    for k in kinds:
        sub = [r for r in per_question if r["bench"] == open_bench and r["kind"] == k]
        pd = metrics.paired_delta([r["dense"]["ndcg"] for r in sub], [r["router"]["ndcg"] for r in sub])
        by_kind[k] = {"n": len(sub), **{c: round(statistics.mean(r[c]["ndcg"] for r in sub), 3) for c in BASELINES + ("router",)},
                      "hybrid_share": round(sum(1 for r in sub if r["mode"] == "hybrid") / len(sub), 3), "router_vs_dense": pd}
        print(f"{k:<10}{len(sub):>4}{by_kind[k]['dense']:>8.3f}{by_kind[k]['rrf_rerank']:>8.3f}{by_kind[k]['router']:>8.3f}"
              f"{by_kind[k]['hybrid_share']:>10.0%}{pd['delta']:>+10.3f} [{pd['ci95'][0]:+.3f},{pd['ci95'][1]:+.3f}]{'*' if pd['significant'] else ''}")

    OUTPUT = output_path(open_bench)
    OUTPUT.write_text(json.dumps({
        # De quoi savoir, dans six mois, sur quel corpus ces chiffres ont été mesurés.
        # Ajouter des documents change les scores sans changer l'or : les nouveaux chunks
        # concourent contre lui. Deux fichiers de résultats ne sont comparables que si ces
        # trois valeurs sont identiques.
        "gel": gel_corpus.etat(),
        "corpus": {**corpus_overlay.describe(),
                   "documents": len({r["document_id"] for r in _collection_documents()}),
                   "chunks": quant_rag.client().count(quant_rag.COLLECTION, exact=True).count},
        "benches": {"known_item": "v1", "open": open_bench},
        "rule": {"hybrid_if": (f"n_exact >= {quant_rag.EXACT_TOKENS_FOR_HYBRID}" if quant_rag.EXACT_TOKENS_FOR_HYBRID is not None
                               else "never (auto = dense)"), "hybrid_path": "rrf_rerank",
                 "dense_path": "dense", "pool": pipeline.POOL},
        "summary": summary, "paired": paired, "by_kind": by_kind,
        "consistency": {"mismatches": mismatches, "checked": len(per_question)},
        "per_question": per_question,
    }, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\n-> {OUTPUT}")


if __name__ == "__main__":
    main()
