"""Vérifie que l'index BM25 servi est celui de l'état courant du corpus — et par qui.

Le problème que ce script règle : après la déduplication (19 443 → 18 636 chunks)
et la conversion des tableaux, l'index BM25 a été rebâti *sous le même nom* que la
référence livrée. Le fichier était juste, son nom mentait, et deux scripts du banc
gardaient un repli qui aurait rebâti 19 443 chunks sans overlay si le fichier avait
disparu. Désormais l'index porte la signature de l'overlay dans son nom
(``quant_rag.BM25_PATH``), il est construit à un seul endroit
(``quant_rag.rebuild_bm25``) et ce script prouve, sans rien supposer :

  1. reconstruction depuis la collection Qdrant, et identité record par record avec
     l'index précédent s'il en existe un (même nombre de chunks) ;
  2. l'ensemble des chunks indexés est exactement celui de la collection ;
  3. aucun chunk d'un document retiré ; les chunks-tableaux sont tokenisés depuis
     le Markdown, pas depuis le HTML ;
  4. chaque point d'entrée du banc (pipeline, eval_hybrid, calibrate_router) résout
     le même objet que la production.

    .venv/bin/python rag/benchmark/check_bm25.py [--restore-reference]

``--restore-reference`` rebâtit en plus la référence livrée (19 443 chunks, sans
overlay, depuis rows.jsonl) sous son nom d'origine ``bm25-ingested-all-v1.json``,
pour que ce nom redise ce qu'il contient.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(ROOT / "src"))

import corpus_overlay  # noqa: E402
import eval_hybrid  # noqa: E402
import pipeline  # noqa: E402
import quant_rag  # noqa: E402
from retrieval.lexical import BM25Index  # noqa: E402

import re  # noqa: E402

REFERENCE = quant_rag.LEXICAL_DIR / "bm25-ingested-all-v1.json"
_TAG = re.compile(r"<\s*/?\s*(td|tr|th|table|tbody|thead)\b", re.I)
OUTPUT = HERE / f"results-bm25-{corpus_overlay.LABEL}.json"


def collection_ids() -> tuple[set[str], set[str]]:
    chunks, documents = set(), set()
    offset = None
    while True:
        points, offset = quant_rag.client().scroll(collection_name=quant_rag.COLLECTION, limit=4096,
                                                   offset=offset, with_payload=["chunk_id", "document_id"])
        for point in points:
            chunks.add(point.payload["chunk_id"])
            documents.add(point.payload["document_id"])
        if offset is None:
            break
    return chunks, documents


def records_equal(a: BM25Index, b: BM25Index) -> dict:
    left = {r["chunk_id"]: r["term_counts"] for r in a.records}
    right = {r["chunk_id"]: r["term_counts"] for r in b.records}
    common = left.keys() & right.keys()
    return {"left": len(left), "right": len(right), "common": len(common),
            "identical_term_counts": sum(1 for c in common if left[c] == right[c]),
            "only_left": len(left.keys() - right.keys()), "only_right": len(right.keys() - left.keys())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--restore-reference", action="store_true")
    args = parser.parse_args()
    report: dict = {"corpus": corpus_overlay.describe(), "index": quant_rag.BM25_PATH.name,
                    "checked_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}

    # Ce qu'il y avait avant, sous quelque nom que ce soit.
    previous = None
    for candidate in (quant_rag.BM25_PATH, REFERENCE):
        if candidate.exists():
            previous = (candidate, BM25Index.load(candidate))
            print(f"index existant : {candidate.name} ({previous[1].doc_count} chunks)")
            break

    print("1. reconstruction depuis la collection…")
    started = time.perf_counter()
    fresh = quant_rag.rebuild_bm25()
    report["rebuild"] = {"seconds": round(time.perf_counter() - started, 1), "records": fresh.doc_count,
                         "terms": len(fresh.document_frequency),
                         "manifest": json.loads(quant_rag.BM25_MANIFEST.read_text(encoding="utf-8"))}
    print(f"   {fresh.doc_count} chunks, {len(fresh.document_frequency)} termes, {report['rebuild']['seconds']} s")
    if previous is not None:
        comparison = records_equal(previous[1], fresh)
        comparison["previous_file"] = previous[0].name
        report["identical_to_previous"] = comparison
        print(f"   vs {previous[0].name} : {comparison['identical_term_counts']}/{comparison['common']} records identiques, "
              f"{comparison['only_left']} en trop avant, {comparison['only_right']} nouveaux")

    print("2. ensemble des chunks = collection…")
    chunks, documents = collection_ids()
    indexed = {r["chunk_id"] for r in fresh.records}
    report["coverage"] = {"collection_points": len(chunks), "collection_documents": len(documents),
                          "indexed": len(indexed), "missing_from_index": len(chunks - indexed),
                          "not_in_collection": len(indexed - chunks)}
    print(f"   collection {len(chunks)} points / {len(documents)} documents ; index {len(indexed)} ; "
          f"manquants {len(chunks - indexed)} ; fantômes {len(indexed - chunks)}")

    print("3. overlays appliqués…")
    removed = corpus_overlay.removed_documents()
    overrides = corpus_overlay.text_overrides()
    from_removed = sum(1 for r in fresh.records if r["document_id"] in removed)
    table_records = [r for r in fresh.records if r["chunk_id"] in overrides]
    # Le texte indexé est celui de la collection ; l'identité collection = overlay a
    # été établie à la conversion (tables/results-convert-tables-v1.json). On teste
    # donc l'overlay lui-même : de vraies balises, pas des « < » d'inégalités
    # (« VIX < 20 », « p < 0,05 » : 251 tableaux en contiennent, légitimement).
    html_text = sum(1 for r in table_records if _TAG.search(overrides[r["chunk_id"]]))
    markdown_like = sum(1 for r in table_records if "|" in r["term_counts"])
    # Le titre d'export entre dans le texte lexical de chaque chunk (recette d'origine,
    # conservée). Un document a pour titre d'export… un tableau HTML entier : ses
    # 94 chunks portent des jetons « td » qui ne viennent pas de leur texte.
    title_html = set()
    for line in (quant_rag.EXPORT / "payloads.jsonl").open(encoding="utf-8"):
        payload = json.loads(line)["payload"]
        if payload["document_id"] not in title_html and _TAG.search(payload.get("title") or ""):
            title_html.add(payload["document_id"])
    report["overlays"] = {"records_from_removed_documents": from_removed, "table_records": len(table_records),
                          "table_texts_with_html_tags": html_text, "table_records_markdown_like": markdown_like,
                          "documents_whose_export_title_is_html": sorted(title_html)}
    print(f"   documents retirés : {from_removed} record(s) ; tableaux : {len(table_records)} records, "
          f"{markdown_like} en Markdown, {html_text} texte(s) avec balises HTML ; "
          f"titre d'export HTML : {len(title_html)} document(s) {sorted(title_html)}")

    print("4. points d'entrée…")
    production = quant_rag.bm25()
    entry_points = {"quant_rag.bm25": production, "pipeline.bm25": pipeline.bm25(),
                    "eval_hybrid.build_bm25": eval_hybrid.build_bm25()}
    report["entry_points"] = {name: {"same_object_as_production": obj is production, "records": obj.doc_count}
                              for name, obj in entry_points.items()}
    report["entry_points"]["calibrate_router"] = "pipeline.bm25 (import)"
    report["entry_points"]["path"] = str(quant_rag.BM25_PATH.relative_to(ROOT))
    for name, entry in report["entry_points"].items():
        if isinstance(entry, dict):
            print(f"   {name:<24} {'même objet' if entry['same_object_as_production'] else 'OBJET DISTINCT'}  "
                  f"({entry['records']} chunks)")

    if args.restore_reference:
        print("5. référence livrée (19 443 chunks, sans overlay) sous son nom d'origine…")
        started = time.perf_counter()
        rows = []
        for line in quant_rag.ROWS_PATH.open(encoding="utf-8"):
            row = json.loads(line)
            rows.append((row["document"], row["chunk"]))
        reference = BM25Index.from_corpus(rows)
        reference.dump(REFERENCE)
        report["reference"] = {"file": REFERENCE.name, "records": reference.doc_count,
                               "seconds": round(time.perf_counter() - started, 1)}
        print(f"   {REFERENCE.name} : {reference.doc_count} chunks ({report['reference']['seconds']} s)")

    ok = (report["coverage"]["missing_from_index"] == 0 and report["coverage"]["not_in_collection"] == 0
          and from_removed == 0 and html_text == 0
          and all(e["same_object_as_production"] for e in report["entry_points"].values() if isinstance(e, dict)))
    report["ok"] = ok
    OUTPUT.write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\n{'OK' if ok else 'ÉCHEC'} -> {OUTPUT.relative_to(ROOT)}")
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
