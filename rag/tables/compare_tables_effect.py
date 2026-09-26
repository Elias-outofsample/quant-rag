"""Avant / après la conversion des tableaux : mêmes questions, même corpus par ailleurs.

Compare deux exécutions de ``benchmark/run_benchmark.py`` (end-to-end v2, configs
``dense`` et ``router``) et deux ``results-router-v1.json`` (retrieval v1 + v2), la
paire « avant » ayant été produite sur le corpus dédupliqué, la paire « après » sur le
même corpus avec les chunks-tableaux convertis et ré-embarqués.

Sortie : ``results-tables-before-after-v1.json`` — par question-tableau (t01–t10) rang et
nDCG du chunk d'or, chunk d'or montré, abstention, couverture ; agrégats par famille de
questions ; écarts appariés avec IC95 bootstrap sur les 40 questions positives.

    python rag/tables/compare_tables_effect.py \\
        --e2e-before benchmark/results-e2e-dedup-v1.json --e2e-after benchmark/results-e2e-tables-v1.json \\
        --router-before .cache/results-router-dedup.json --router-after benchmark/results-router-v1.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "benchmark"))
import metrics  # noqa: E402

OUT = HERE / "results-tables-before-after-v1.json"


def rows_by(e2e: dict) -> dict:
    return {(r["config"], r["qid"]): r for r in e2e["per_row"]}


def coverage(row: dict):
    judge = row.get("judge") or {}
    for key in ("coverage", "fact_coverage", "facts_coverage"):
        if key in judge:
            return judge[key]
    if "facts_covered" in judge and "facts_total" in judge and judge["facts_total"]:
        return judge["facts_covered"] / judge["facts_total"]
    return None


def summarise(rows_before: dict, rows_after: dict, config: str, kinds: tuple[str, ...]) -> dict:
    keys = [k for k in rows_after if k[0] == config and rows_after[k]["kind"] in kinds and k in rows_before]
    def agg(rows):
        got = [rows[k] for k in keys]
        nd = [r["retrieval"]["chunk"]["ndcg"] for r in got if r.get("retrieval")]
        cov = [c for c in (coverage(r) for r in got) if c is not None]
        return {"n": len(got), "ndcg10_chunk": round(statistics.mean(nd), 3) if nd else None,
                "recall@10": round(sum(1 for r in got if (r["retrieval"]["chunk"]["first_rank"] or 99) <= 10) / len(got), 3) if got else None,
                "gold_in_context": round(sum(1 for r in got if r.get("gold_in_context")) / len(got), 3) if got else None,
                "abstention": round(sum(1 for r in got if r.get("abstained")) / len(got), 3) if got else None,
                "coverage": round(statistics.mean(cov), 3) if cov else None}
    before, after = agg(rows_before), agg(rows_after)
    nd_b = [rows_before[k]["retrieval"]["chunk"]["ndcg"] for k in keys]
    nd_a = [rows_after[k]["retrieval"]["chunk"]["ndcg"] for k in keys]
    cov_b = [coverage(rows_before[k]) for k in keys]
    cov_a = [coverage(rows_after[k]) for k in keys]
    paired = {"ndcg10_chunk": metrics.paired_delta(nd_b, nd_a)}
    if all(c is not None for c in cov_b + cov_a) and keys:
        paired["coverage"] = metrics.paired_delta(cov_b, cov_a)
    return {"before": before, "after": after, "paired_delta_after_minus_before": paired}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--e2e-before", type=Path, required=True)
    parser.add_argument("--e2e-after", type=Path, required=True)
    parser.add_argument("--router-before", type=Path, required=True)
    parser.add_argument("--router-after", type=Path, required=True)
    args = parser.parse_args()

    e2e_b, e2e_a = (json.loads(p.read_text(encoding="utf-8")) for p in (args.e2e_before, args.e2e_after))
    rb, ra = rows_by(e2e_b), rows_by(e2e_a)
    sample = next(iter(ra.values()))
    if coverage(sample) is None:
        print("champs juge disponibles :", list((sample.get("judge") or {}).keys()))

    per_table_question = []
    for qid in sorted({k[1] for k in ra if ra[k]["kind"] == "table"}):
        entry = {"qid": qid}
        for config in ("dense", "router"):
            b, a = rb.get((config, qid)), ra.get((config, qid))
            if not (a and b):
                continue
            entry[config] = {
                "rank_before": b["retrieval"]["chunk"]["first_rank"], "rank_after": a["retrieval"]["chunk"]["first_rank"],
                "ndcg_before": round(b["retrieval"]["chunk"]["ndcg"], 3), "ndcg_after": round(a["retrieval"]["chunk"]["ndcg"], 3),
                "gold_shown_before": b.get("gold_in_context"), "gold_shown_after": a.get("gold_in_context"),
                "abstained_before": b.get("abstained"), "abstained_after": a.get("abstained"),
                "coverage_before": coverage(b), "coverage_after": coverage(a),
            }
        per_table_question.append(entry)

    families = {}
    for config in ("dense", "router"):
        families[config] = {"table": summarise(rb, ra, config, ("table",)),
                            "non_table": summarise(rb, ra, config, ("single", "multi")),
                            "all_positive": summarise(rb, ra, config, ("single", "multi", "table"))}
        neg_b = [rb[k] for k in rb if k[0] == config and rb[k]["kind"] == "negative"]
        neg_a = [ra[k] for k in ra if k[0] == config and ra[k]["kind"] == "negative"]
        families[config]["negative_abstention"] = {
            "before": round(sum(1 for r in neg_b if r.get("abstained")) / len(neg_b), 3) if neg_b else None,
            "after": round(sum(1 for r in neg_a if r.get("abstained")) / len(neg_a), 3) if neg_a else None}

    router_b, router_a = (json.loads(p.read_text(encoding="utf-8")) for p in (args.router_before, args.router_after))
    retrieval = {"before": router_b["summary"], "after": router_a["summary"],
                 "paired_before": router_b.get("paired"), "paired_after": router_a.get("paired")}
    changed = []
    pb = {(r["bench"], r["qid"]): r for r in router_b["per_question"]}
    for r in router_a["per_question"]:
        o = pb.get((r["bench"], r["qid"]))
        if o and any(abs(o[c]["ndcg"] - r[c]["ndcg"]) > 1e-9 for c in ("router", "dense", "rrf_rerank")):
            changed.append({"bench": r["bench"], "qid": r["qid"], "kind": r.get("kind"),
                            **{c: {"before": (o[c]["first_rank"], round(o[c]["ndcg"], 3)), "after": (r[c]["first_rank"], round(r[c]["ndcg"], 3))}
                               for c in ("router", "dense", "rrf_rerank")}})

    out = {"version": "tables-before-after-v1", "e2e": {"before": args.e2e_before.name, "after": args.e2e_after.name},
           "router_results": {"before": str(args.router_before), "after": str(args.router_after)},
           "table_questions": per_table_question, "families": families,
           "retrieval_bench_v1_v2": retrieval, "retrieval_questions_changed": changed}
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")

    print("=== questions-tableaux t01–t10 (rang du chunk d'or, dense | router) ===")
    for e in per_table_question:
        d, r = e.get("dense", {}), e.get("router", {})
        print(f"  {e['qid']}  dense {str(d.get('rank_before')):>4} → {str(d.get('rank_after')):<4} "
              f"router {str(r.get('rank_before')):>4} → {str(r.get('rank_after')):<4}  "
              f"montré {int(bool(r.get('gold_shown_before')))}→{int(bool(r.get('gold_shown_after')))}  "
              f"couverture {r.get('coverage_before')}→{r.get('coverage_after')}")
    for config in ("dense", "router"):
        print(f"\n=== {config} ===")
        for fam, v in families[config].items():
            if fam == "negative_abstention":
                print(f"  négatives abstention {v['before']} → {v['after']}")
                continue
            b, a, p = v["before"], v["after"], v["paired_delta_after_minus_before"]
            print(f"  {fam:12} n={a['n']:2}  nDCG {b['ndcg10_chunk']} → {a['ndcg10_chunk']}  R@10 {b['recall@10']} → {a['recall@10']}  "
                  f"montré {b['gold_in_context']} → {a['gold_in_context']}  abst. {b['abstention']} → {a['abstention']}  "
                  f"couverture {b['coverage']} → {a['coverage']}  Δ nDCG {p['ndcg10_chunk']}"
                  + (f"  Δ couv. {p['coverage']}" if "coverage" in p else ""))
    print(f"\n=== retrieval v1+v2 (results-router) : {len(changed)} questions changent ===")
    for c in changed[:20]:
        print(f"  {c['bench']}/{c['qid']:4} {str(c.get('kind')):7} router {c['router']['before']} → {c['router']['after']}  rrf_rerank {c['rrf_rerank']['before']} → {c['rrf_rerank']['after']}")
    print("->", OUT)


if __name__ == "__main__":
    main()
