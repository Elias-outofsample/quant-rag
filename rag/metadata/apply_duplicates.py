"""Retire de l'index les documents listés dans ``duplicates-v1.json`` et remet tout en cohérence.

1. suppression des points Qdrant des documents retirés (``client.delete`` par ``document_id``),
   comptages avant/après vérifiés contre le nombre de chunks attendu ;
2. reconstruction du BM25 depuis la collection (``quant_rag.rebuild_bm25``) — le chemin
   lexical et le banc lisent le même fichier ;
3. re-ciblage des questions du banc dont le chunk d'or était dans un document retiré,
   vers son jumeau dans le document conservé (``benchmark_remap`` du fichier de décision),
   avec provenance ``gold_remapped`` dans la question — idempotent.

    python rag/metadata/apply_duplicates.py            # applique
    python rag/metadata/apply_duplicates.py --dry-run  # compte, n'écrit rien
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import corpus_overlay  # noqa: E402
import quant_rag  # noqa: E402

HERE = Path(__file__).resolve().parent
BENCH = HERE.parent / "benchmark"
RESULT = HERE / "results-apply-duplicates-v1.json"


def remap_questions(decision: dict, dry_run: bool) -> list[dict]:
    done = []
    for bench in ("v1", "v2"):
        path = BENCH / f"questions-{bench}.jsonl"
        lines = path.read_text(encoding="utf-8").splitlines()
        out, changed = [], False
        for line in lines:
            if not line.strip():
                out.append(line)
                continue
            q = json.loads(line)
            for m in decision["benchmark_remap"]:
                if m["bench"] != bench or m["qid"] != q["qid"]:
                    continue
                if bench == "v1" and q.get("target_chunk") == m["from_chunk"]:
                    q["target_chunk"] = m["to_chunk"]
                    q.setdefault("gold_remapped", []).append({**m, "decision": decision["version"]})
                    changed = True
                elif bench == "v2" and m["from_chunk"] in q.get("gold_chunks", []):
                    q["gold_chunks"] = [m["to_chunk"] if c == m["from_chunk"] else c for c in q["gold_chunks"]]
                    removed_doc = next(r for r in decision["remove"] if r["document_id"] in q.get("gold_documents", []))
                    q["gold_documents"] = [removed_doc["kept"]["document_id"] if d == removed_doc["document_id"] else d
                                           for d in q["gold_documents"]]
                    q.setdefault("gold_remapped", []).append({**m, "from_document": removed_doc["document_id"],
                                                              "to_document": removed_doc["kept"]["document_id"],
                                                              "decision": decision["version"]})
                    changed = True
                if changed:
                    done.append({"bench": bench, "qid": q["qid"], "to_chunk": m["to_chunk"]})
            out.append(json.dumps(q, ensure_ascii=False))
        if changed and not dry_run:
            path.write_text("\n".join(out) + "\n", encoding="utf-8")
    return done


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    from qdrant_client import models

    decision = json.loads(corpus_overlay.DUPLICATES.read_text(encoding="utf-8"))
    client = quant_rag.client()
    before = client.count(quant_rag.COLLECTION, exact=True).count
    per_doc = {}
    for entry in decision["remove"]:
        flt = models.Filter(must=[models.FieldCondition(key="document_id", match=models.MatchValue(value=entry["document_id"]))])
        per_doc[entry["document_id"]] = client.count(quant_rag.COLLECTION, exact=True, count_filter=flt).count
    expected = sum(e["chunks"] for e in decision["remove"])
    print(f"points {before} ; à retirer {per_doc} (attendu {expected})")
    if args.dry_run:
        print(json.dumps(remap_questions(decision, dry_run=True), ensure_ascii=False))
        return

    started = time.perf_counter()
    for entry in decision["remove"]:
        client.delete(quant_rag.COLLECTION, wait=True, points_selector=models.FilterSelector(filter=models.Filter(
            must=[models.FieldCondition(key="document_id", match=models.MatchValue(value=entry["document_id"]))])))
    after = client.count(quant_rag.COLLECTION, exact=True).count
    bm25 = quant_rag.rebuild_bm25()
    remapped = remap_questions(decision, dry_run=False)
    summary = {
        "decision": decision["version"], "points_before": before, "points_after": after,
        "points_removed": before - after, "expected_removed": expected,
        "removed": [{"document_id": e["document_id"], "short_ref": e["short_ref"], "chunks": per_doc[e["document_id"]]}
                    for e in decision["remove"]],
        "bm25_chunks": bm25.doc_count, "benchmark_remapped": remapped,
        "seconds": round(time.perf_counter() - started, 1),
    }
    RESULT.write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    if before - after != expected or bm25.doc_count != after:
        sys.exit("ÉCHEC : comptages incohérents")


if __name__ == "__main__":
    main()
