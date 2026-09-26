"""Titres propres dans l'embedding : avant / après, mêmes questions, même modèle — verdict du chantier B.

Compare deux matrices de vecteurs sur les 155 questions des deux bancs (v1 known-item,
v3 ouvert) : la collection courante (titres d'export) et l'overlay ``vectors-clean-titles-v1.npz``
(titres consolidés, ``reembed_titles.py``). Requêtes identiques, même encodeur, même
recette ; seul le titre en tête de chaque chunk diffère. Écarts appariés avec IC95, par
banc et par famille ; gains et pertes nets ; et la vue par document, pour voir si l'effet
vient des 21 documents au titre d'export aberrant (HTML, > 200 caractères) ou de tous.

    .venv/bin/python rag/titles/eval_titles.py
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "rag"))
sys.path.insert(0, str(ROOT / "rag" / "benchmark"))
import corpus_overlay  # noqa: E402
import experiment  # noqa: E402
import metrics  # noqa: E402
import pipeline  # noqa: E402
import quant_rag  # noqa: E402
from dense_matrix import Matrix  # noqa: E402

VECTORS = HERE / ".cache" / "vectors-clean-titles-v1.npz"
TITLES = HERE / "titles-clean-v1.json"
OUTPUT = HERE / "results-titles-v1.json"
CONFIGS = ("export_titles", "clean_titles")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--vectors", type=Path, default=VECTORS)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if not args.vectors.exists():
        sys.exit(f"{args.vectors} absent : lance d'abord reembed_titles.py")

    started = time.perf_counter()
    items = experiment.load_items()
    before = Matrix.load(label="export_titles", titles=False)
    after = Matrix.load(vectors=args.vectors, label="clean_titles", titles=False)
    titles = json.loads(TITLES.read_text(encoding="utf-8"))["documents"] if TITLES.exists() else {}
    aberrant = {d for d, t in titles.items() if len(t["export_title"]) > 200 or ("<" in t["export_title"] and ">" in t["export_title"])}
    print(f"{len(items)} questions · {len(before.chunk_ids)} chunks · overlays avant {before.applied} / après {after.applied}")

    per_question = []
    for position, item in enumerate(items, 1):
        query = pipeline.query_of(item)
        scope = experiment.scope_of(item)
        vector = quant_rag.encode_query(query)
        row = {"bench": item["bench"], "key": item["key"], "qid": item["qid"], "kind": item.get("kind", "single"),
               "gold_documents": item["gold_documents"],
               "gold_in_aberrant_title_document": any(d in aberrant for d in item["gold_documents"]),
               "export_titles": experiment.measure(item, before.search(vector, scope=scope)),
               "clean_titles": experiment.measure(item, after.search(vector, scope=scope))}
        per_question.append(row)
        print(f"  [{position}/{len(items)}] {item['key']:<8} export@{str(row['export_titles']['first_rank'] or '-'):>3}"
              f" → clean@{str(row['clean_titles']['first_rank'] or '-'):>3}", end="\r", flush=True)
    print()

    result = experiment.summarise(per_question, list(CONFIGS), reference="export_titles")
    result["wins_losses"] = experiment.wins_losses(per_question, "clean_titles", "export_titles")
    sub = [r for r in per_question if r["gold_in_aberrant_title_document"]]
    rest = [r for r in per_question if not r["gold_in_aberrant_title_document"]]
    result["by_title_quality"] = {
        "gold_in_aberrant_title_document": {"n": len(sub), "documents": len(aberrant),
                                            **({c: round(statistics.mean(r[c]["ndcg"] for r in sub), 3) for c in CONFIGS} if sub else {}),
                                            "paired": metrics.paired_delta([r["export_titles"]["ndcg"] for r in sub], [r["clean_titles"]["ndcg"] for r in sub]) if sub else None},
        "other": {"n": len(rest), **{c: round(statistics.mean(r[c]["ndcg"] for r in rest), 3) for c in CONFIGS},
                  "paired": metrics.paired_delta([r["export_titles"]["ndcg"] for r in rest], [r["clean_titles"]["ndcg"] for r in rest])},
    }
    experiment.print_summary(result, list(CONFIGS), "export_titles")
    wl = result["wins_losses"]
    print(f"\n=== gains / pertes nets (|Δ nDCG| ≥ {wl['threshold']}) : {wl['n_wins']} / {wl['n_losses']} ===")
    for e in wl["wins"][:8]:
        print(f"  + {e['key']:<8} {e['kind']:<7} rang {e['rank_reference']} → {e['rank_config']}  Δ {e['delta_ndcg']:+.3f}")
    for e in wl["losses"][:8]:
        print(f"  - {e['key']:<8} {e['kind']:<7} rang {e['rank_reference']} → {e['rank_config']}  Δ {e['delta_ndcg']:+.3f}")
    q = result["by_title_quality"]
    print(f"\n=== selon le titre d'export du document d'or ===")
    for name, block in q.items():
        if block.get("paired"):
            print(f"  {name:<36} n={block['n']:>3}  export {block['export_titles']:.3f} → clean {block['clean_titles']:.3f}"
                  f"  Δ {block['paired']['delta']:+.3f} [{block['paired']['ci95'][0]:+.3f},{block['paired']['ci95'][1]:+.3f}]{'*' if block['paired']['significant'] else ''}")

    payload = {"version": "titles-v1", "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "corpus_state": corpus_overlay.describe(), "vectors": str(args.vectors.relative_to(ROOT)),
               "configs": list(CONFIGS), "reference": "export_titles", **result,
               "wall_clock_s": round(time.perf_counter() - started), "per_question": per_question}
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n-> {args.output}  ({payload['wall_clock_s']} s)")


if __name__ == "__main__":
    main()
