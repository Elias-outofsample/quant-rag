"""Abstention à tort, mesurée sans juge : l'or était sous les yeux du générateur, il s'est tu.

``run_benchmark.py`` publie déjà ``false_abstention_when_gold_shown`` par configuration.
Ce script en fait une métrique lisible et actionnable, à partir du fichier de résultats
existant (aucun appel LLM, aucun accès à l'index) :

  gold_present_but_abstained   part des questions positives où (a) un chunk d'or figurait
                               dans les passages montrés et (b) la réponse commence par
                               INSUFFICIENT_EVIDENCE — deux tests d'appartenance, zéro jugement ;
  document_present_but_abstained   variante plus faible : un chunk du *document* d'or était
                               montré (la réponse peut chevaucher une frontière de découpage) ;
  par famille, par rang de l'or dans le contexte (montré au rang 1 ou plus loin), et la
  liste nominative des cas, avec la phrase d'abstention — c'est elle qu'il faut lire avant
  de toucher au prompt de réponse (``pipeline.ANSWER_SYSTEM``).

Lecture : une abstention avec l'or au rang 1 est un échec de *génération* (ou une
question dont le passage d'or ne répond pas aussi bien que le rédacteur l'a cru) ; le
juge n'y est pour rien. Le complément — répondu quand l'or est montré — est la
couverture conditionnelle déjà publiée.

    .venv/bin/python rag/benchmark/abstention.py                       # results-e2e-v3.json, config dense
    .venv/bin/python rag/benchmark/abstention.py --results X.json --config dense --compare Y.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import metrics  # noqa: E402
import pipeline  # noqa: E402

RESULTS = HERE / "results-e2e-v3.json"
OUTPUT = HERE / "results-abstention-v3.json"
KINDS = ("single", "table", "multi", "dated", "exact")


def load_rows(path: Path, config: str, questions: dict | None = None) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = [r for r in data["per_row"] if r["config"] == config and r.get("answer") is not None]
    return rows


def gold_rank_in_context(row: dict, gold_chunks: set, gold_documents: set) -> tuple[int | None, int | None]:
    """Rang (1..5) du premier chunk d'or, et du premier chunk du document d'or, dans les passages montrés."""
    chunk_rank = document_rank = None
    for position, passage in enumerate(row.get("context") or [], 1):
        if chunk_rank is None and passage.get("chunk_id") in gold_chunks:
            chunk_rank = position
        if document_rank is None and passage.get("document_id") in gold_documents:
            document_rank = position
    return chunk_rank, document_rank


def measure(rows: list[dict], by_qid: dict) -> dict:
    positives = [r for r in rows if r["kind"] != "negative"]
    cases = []
    for r in positives:
        item = by_qid[r["qid"]]
        chunk_rank, document_rank = gold_rank_in_context(r, set(item["gold_chunks"]), set(item["gold_documents"]))
        abstained = pipeline.abstained(r.get("answer"))
        cases.append({"qid": r["qid"], "kind": r["kind"], "abstained": abstained,
                      "gold_chunk_rank_in_context": chunk_rank, "gold_document_rank_in_context": document_rank,
                      "gold_retrieval_rank": (r.get("retrieval") or {}).get("chunk", {}).get("first_rank"),
                      "coverage": (r.get("judge") or {}).get("coverage"),
                      "answer_head": (r.get("answer") or "").strip().split("\n")[0][:220]})

    def rate(subset, key):
        shown = [c for c in subset if c[key] is not None]
        return {"shown": len(shown), "abstained": sum(1 for c in shown if c["abstained"]),
                "rate": round(sum(1 for c in shown if c["abstained"]) / len(shown), 3) if shown else None}

    out = {
        "positives": len(positives),
        "abstained_total": sum(1 for c in cases if c["abstained"]),
        "false_abstention_rate": round(sum(1 for c in cases if c["abstained"]) / len(positives), 3) if positives else None,
        "gold_present_but_abstained": rate(cases, "gold_chunk_rank_in_context"),
        "document_present_but_abstained": rate(cases, "gold_document_rank_in_context"),
        "by_kind": {k: rate([c for c in cases if c["kind"] == k], "gold_chunk_rank_in_context") for k in KINDS
                    if any(c["kind"] == k for c in cases)},
        "by_gold_rank_in_context": {str(rank): rate([c for c in cases if c["gold_chunk_rank_in_context"] == rank],
                                                    "gold_chunk_rank_in_context") for rank in range(1, pipeline.CONTEXT_PASSAGES + 1)},
        # l'or absent du contexte : l'abstention y est *correcte* du point de vue du générateur
        "gold_absent": {"n": sum(1 for c in cases if c["gold_chunk_rank_in_context"] is None),
                        "abstained": sum(1 for c in cases if c["gold_chunk_rank_in_context"] is None and c["abstained"]),
                        "answered_anyway": sum(1 for c in cases if c["gold_chunk_rank_in_context"] is None and not c["abstained"])},
        "coverage_when_gold_shown_and_answered": (round(statistics.mean(
            c["coverage"] for c in cases if c["gold_chunk_rank_in_context"] is not None and not c["abstained"] and c["coverage"] is not None), 3)
            if any(c["gold_chunk_rank_in_context"] is not None and not c["abstained"] and c["coverage"] is not None for c in cases) else None),
        "cases_gold_present_but_abstained": [c for c in cases if c["abstained"] and c["gold_chunk_rank_in_context"] is not None],
    }
    return out


def negatives(rows: list[dict]) -> dict:
    neg = [r for r in rows if r["kind"] == "negative"]
    if not neg:
        return {}
    return {"n": len(neg),
            "correct_abstention_rate": round(sum(1 for r in neg if pipeline.abstained(r.get("answer"))) / len(neg), 3),
            "fabrication_rate": round(sum(1 for r in neg if (r.get("judge") or {}).get("fabricated")) / len(neg), 3)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--results", type=Path, default=RESULTS)
    parser.add_argument("--config", default="dense")
    parser.add_argument("--compare", type=Path, help="second fichier de résultats (même banc) : écart apparié")
    parser.add_argument("--questions", type=Path, default=HERE / "questions-v3.jsonl")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()

    by_qid = {json.loads(l)["qid"]: json.loads(l) for l in args.questions.read_text(encoding="utf-8").splitlines() if l.strip()}
    rows = load_rows(args.results, args.config)
    result = {"results": args.results.name, "config": args.config, "questions": args.questions.name,
              "measure": measure(rows, by_qid), "negatives": negatives(rows)}
    m = result["measure"]
    print(f"{args.results.name} · {args.config} · {m['positives']} questions positives")
    print(f"  abstention à tort (toutes)                : {m['abstained_total']}/{m['positives']} = {m['false_abstention_rate']}")
    g = m["gold_present_but_abstained"]
    print(f"  gold_present_but_abstained (chunk d'or montré) : {g['abstained']}/{g['shown']} = {g['rate']}")
    d = m["document_present_but_abstained"]
    print(f"  document_present_but_abstained             : {d['abstained']}/{d['shown']} = {d['rate']}")
    print(f"  or absent du contexte : {m['gold_absent']['n']} questions, abstention {m['gold_absent']['abstained']}, "
          f"réponse quand même {m['gold_absent']['answered_anyway']}")
    print("  par famille (or montré → abstenu) : " + ", ".join(f"{k} {v['abstained']}/{v['shown']}" for k, v in m["by_kind"].items()))
    print("  par rang de l'or dans le contexte : " + ", ".join(f"rang {k} {v['abstained']}/{v['shown']}" for k, v in m["by_gold_rank_in_context"].items()))
    if result["negatives"]:
        print(f"  négatives : abstention correcte {result['negatives']['correct_abstention_rate']}, fabrication {result['negatives']['fabrication_rate']}")
    print("\n  cas (qid, famille, rang de l'or dans le contexte, début de la réponse) :")
    for c in m["cases_gold_present_but_abstained"]:
        print(f"    {c['qid']:<4} {c['kind']:<7} ctx@{c['gold_chunk_rank_in_context']}  {c['answer_head'][:110]}")

    if args.compare:
        other = measure(load_rows(args.compare, args.config), by_qid)
        left = {c["qid"]: c for c in m["cases_gold_present_but_abstained"]}
        # apparié sur les questions où l'or est montré dans les DEUX exécutions
        a_cases = {c["qid"]: c for c in measure(rows, by_qid)["cases_gold_present_but_abstained"]}
        rows_b = load_rows(args.compare, args.config)
        shown_a = {r["qid"] for r in rows if r["kind"] != "negative" and gold_rank_in_context(r, set(by_qid[r["qid"]]["gold_chunks"]), set(by_qid[r["qid"]]["gold_documents"]))[0]}
        shown_b = {r["qid"] for r in rows_b if r["kind"] != "negative" and gold_rank_in_context(r, set(by_qid[r["qid"]]["gold_chunks"]), set(by_qid[r["qid"]]["gold_documents"]))[0]}
        common = sorted(shown_a & shown_b)
        abst_a = {r["qid"]: pipeline.abstained(r.get("answer")) for r in rows}
        abst_b = {r["qid"]: pipeline.abstained(r.get("answer")) for r in rows_b}
        paired = metrics.paired_delta([float(abst_a[q]) for q in common], [float(abst_b[q]) for q in common])
        result["comparison"] = {"other": args.compare.name, "other_measure": {k: v for k, v in other.items() if k != "cases_gold_present_but_abstained"},
                                "common_gold_shown": len(common),
                                "abstention_rate_a": round(sum(abst_a[q] for q in common) / len(common), 3) if common else None,
                                "abstention_rate_b": round(sum(abst_b[q] for q in common) / len(common), 3) if common else None,
                                "paired_delta_b_minus_a": paired,
                                "flipped": {"now_answers": [q for q in common if abst_a[q] and not abst_b[q]],
                                            "now_abstains": [q for q in common if not abst_a[q] and abst_b[q]]},
                                "negatives_b": negatives(rows_b)}
        print(f"\n  comparaison avec {args.compare.name} sur {len(common)} questions à or montré dans les deux :")
        print(f"    abstention {result['comparison']['abstention_rate_a']} → {result['comparison']['abstention_rate_b']}  "
              f"Δ {paired.get('delta'):+.3f} [{paired.get('ci95', [0, 0])[0]:+.3f}, {paired.get('ci95', [0, 0])[1]:+.3f}]")
        print(f"    répond désormais : {result['comparison']['flipped']['now_answers']}  ·  s'abstient désormais : {result['comparison']['flipped']['now_abstains']}")
        print(f"    négatives après : {result['comparison']['negatives_b']}")

    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n-> {args.output}")


if __name__ == "__main__":
    main()
