"""Construit ``data/graph/graph-lite.json`` depuis ``results.jsonl`` (GLiNER2).

Adaptation de ``scripts/build_gliner_ingested_all_graph.py`` (amont) : chemins locaux,
et l'état du corpus (signature des overlays, rapport d'extraction) entre dans le rapport.
Le **format du graphe est inchangé** — mêmes nœuds (``document``, ``chunk``, ``entity``),
mêmes arêtes (``contains_chunk``, ``mentions``, relations sémantiques), mêmes identifiants
``entity:<label>:<clé normalisée>`` — pour que les scripts amont et ``graph_search.py``
lisent l'un comme l'autre.

Le graphe est **brut** : tout ce que GLiNER a extrait y est, y compris les entités d'un
caractère (« x », « S ») et les commandes LaTeX. Le tri se fait à la lecture
(``rag/graph_search.py``), où il est réversible et mesurable ; ici on conserve la
sortie du modèle telle quelle, comparable d'un état du corpus à l'autre.

    .venv/bin/python rag/graph/build_graph.py            # ~30 s, aucune dépendance
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "rag"))
import corpus_overlay  # noqa: E402

RESULTS_DIR = ROOT / "data" / "graph" / "gliner-results"
GRAPH = ROOT / "data" / "graph" / "graph-lite.json"
REPORT = ROOT / "data" / "graph" / "graph-lite-report.json"
STRUCTURAL = {"mentions", "contains_chunk"}


def key(text: str) -> str:
    """Clé d'entité, comme en amont : NFKC, casse pliée, espaces réduits."""
    text = unicodedata.normalize("NFKC", text).casefold().strip()
    return re.sub(r"\s+", " ", text)


def build(rows: list[dict]) -> tuple[dict, list[dict]]:
    nodes: dict[str, dict] = {}
    edges: list[dict] = []
    chunks_by_entity: dict[str, set] = defaultdict(set)
    lookup: dict[str, str] = {}
    for row in rows:
        cid, did = row["chunk_id"], row["document_id"]
        cn, dn = f"chunk:{cid}", f"document:{did}"
        nodes.setdefault(dn, {"id": dn, "type": "document", "document_id": did})
        nodes.setdefault(cn, {"id": cn, "type": "chunk", "chunk_id": cid, "page_start": row.get("page_start")})
        edges.append({"source": dn, "target": cn, "relation": "contains_chunk", "source_chunk_id": cid})
        for label, mentions in (row.get("entities") or {}).items():
            for mention in mentions or []:
                text = (mention.get("text") or "").strip()
                if not text:
                    continue
                eid = f"entity:{label}:{key(text)}"
                node = nodes.setdefault(eid, {"id": eid, "type": "entity", "label": label, "name": text,
                                              "mentions": 0, "source_chunks": []})
                lookup.setdefault(key(text), eid)
                node["mentions"] += 1
                chunks_by_entity[eid].add(cid)
                edges.append({"source": cn, "target": eid, "relation": "mentions", "source_chunk_id": cid})
        values = row.get("relations") or {}
        if isinstance(values, dict) and "relation_extraction" in values:
            values = values["relation_extraction"]
        if isinstance(values, dict):
            items = ((rel, item) for rel, vals in values.items() for item in (vals or []))
        else:
            items = ((item.get("relation", "related_to"), item) for item in values)
        for rel, item in items:
            head, tail = item.get("head") or {}, item.get("tail") or {}
            ht, tt = (head.get("text") or "").strip(), (tail.get("text") or "").strip()
            if not ht or not tt:
                continue
            hid = lookup.get(key(ht), f"entity:untyped:{key(ht)}")
            tid = lookup.get(key(tt), f"entity:untyped:{key(tt)}")
            for eid, name in ((hid, ht), (tid, tt)):
                nodes.setdefault(eid, {"id": eid, "type": "entity", "label": "untyped", "name": name,
                                       "mentions": 0, "source_chunks": []})
            edges.append({"source": hid, "target": tid, "relation": rel, "source_chunk_id": cid,
                          "confidence": item.get("confidence", item.get("score"))})
    for eid, chunk_ids in chunks_by_entity.items():
        nodes[eid]["source_chunks"] = sorted(chunk_ids)
    return nodes, edges


def main() -> None:
    parser = argparse.ArgumentParser(description="results.jsonl (GLiNER2) -> graph-lite.json")
    parser.add_argument("--input", type=Path, default=RESULTS_DIR, help="dossier contenant results.jsonl et report.json")
    parser.add_argument("--output", type=Path, default=GRAPH)
    parser.add_argument("--report", type=Path, default=None, help="défaut : graph-lite-report.json à côté de la sortie")
    args = parser.parse_args()
    results = args.input / "results.jsonl"
    if not results.exists():
        sys.exit(f"absent : {results} — lance d'abord rag/graph/extract_entities.py")
    started = time.perf_counter()
    rows = [json.loads(line) for line in results.read_text(encoding="utf-8").splitlines() if line.strip()]
    # ``results.jsonl`` s'écrit en ajout et ne se relit jamais : il garde les extractions des
    # documents retirés depuis. Sans ce filtre, le graphe continue de servir un document que
    # la collection, le registre, le relais et BM25 ont tous cessé de servir — et personne ne
    # le dit. Le 5 septembre 2026, CST2010 est ainsi resté un nœud du graphe après avoir été
    # retiré partout ailleurs.
    retires = corpus_overlay.removed_documents()
    ecartes = [r for r in rows if r["document_id"] in retires]
    rows = [r for r in rows if r["document_id"] not in retires]
    if ecartes:
        print(f"  {len(ecartes)} chunk(s) de {len({r['document_id'] for r in ecartes})} document(s) "
              f"retiré(s) écartés du graphe", flush=True)
    nodes, edges = build(rows)
    extraction = json.loads((args.input / "report.json").read_text(encoding="utf-8")) if (args.input / "report.json").exists() else {}
    report = {
        "status": "COMPLETED", "mode": "graph-lite",
        "input_results": str(results.relative_to(ROOT)) if results.is_relative_to(ROOT) else str(results),
        "corpus": corpus_overlay.describe(),
        "extraction": {k: extraction.get(k) for k in ("status", "model", "device", "batch_size", "completed", "total",
                                                       "chunks_with_error", "chunks_per_second", "finished_at")},
        "sample_chunks": len(rows), "documents": len({r["document_id"] for r in rows}),
        "removed_documents_skipped": {"chunks": len(ecartes),
                                      "documents": sorted({r["document_id"] for r in ecartes})},
        "chunks_with_error": sum("error" in r for r in rows),
        "nodes": len(nodes), "entity_nodes": sum(n["type"] == "entity" for n in nodes.values()),
        "entity_labels": dict(Counter(n["label"] for n in nodes.values() if n["type"] == "entity").most_common()),
        "edges": len(edges), "mention_edges": sum(e["relation"] == "mentions" for e in edges),
        "relation_edges": sum(e["relation"] not in STRUCTURAL for e in edges),
        "relation_types": dict(Counter(e["relation"] for e in edges if e["relation"] not in STRUCTURAL).most_common()),
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seconds": round(time.perf_counter() - started, 1),
        "production_untouched": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"report": report, "nodes": list(nodes.values()), "edges": edges},
                                      ensure_ascii=False), encoding="utf-8")
    report_path = args.report or args.output.with_name(args.output.stem + "-report.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
