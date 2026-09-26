"""Extraction d'entités et de relations (GLiNER2) sur l'état courant du corpus.

Adaptation macOS de ``scripts/extract_gliner_ingested_all.py`` (amont, Windows/CUDA) :

  - lit ``rows.jsonl`` (les chunks avec leur texte, pas les PDF) **en appliquant les
    overlays** de ``rag/corpus_overlay.py`` : documents retirés, chunks-tableaux en
    Markdown. Le graphe livré avait été extrait sur les tableaux HTML bruts, et GLiNER
    y avait pris les balises pour des entités (« rowspan » : 5 303 mentions, entité n°1) ;
  - device ``mps`` / ``cpu`` (``cuda`` s'il existe), sans quantification ;
  - même modèle, même schéma d'entités, mêmes relations et mêmes seuils que l'amont,
    pour que le graphe reste comparable ; le format de ``results.jsonl`` est conservé
    (``build_graph.py`` et les scripts amont le lisent tel quel) ;
  - reprise : ``results.jsonl`` est relu au démarrage, seuls les chunks absents sont
    traités. On peut couper et relancer sans rien perdre ;
  - ``--sample N`` : mesure de vitesse sur un échantillon stable de N chunks, écrit
    ailleurs (``--output``), avant de lancer les 18 636 ;
  - ``--shard i/n`` : une partie du corpus (chunks d'indice ≡ i mod n), pour faire
    tourner CPU et MPS en parallèle dans deux dossiers ; ``--merge-from`` rapatrie
    ensuite les résultats des shards dans le dossier principal.

Sortie : ``data/graph/gliner-results/results.jsonl`` (une ligne par chunk : entités
par type avec positions, relations tête→queue), ``progress.json``, ``report.json``
(modèle, device, état du corpus — signature des overlays —, totaux, vitesse).

    .venv-gliner/bin/python rag/graph/extract_entities.py --sample 100 --device mps --output /tmp/x
    .venv-gliner/bin/python rag/graph/extract_entities.py                 # tout, reprise automatique
    .venv-gliner/bin/python rag/graph/extract_entities.py --shard 0/2 --device mps --output /tmp/s0 &
    .venv-gliner/bin/python rag/graph/extract_entities.py --shard 1/2 --device cpu --output /tmp/s1
    .venv-gliner/bin/python rag/graph/extract_entities.py --merge-from /tmp/s0 /tmp/s1   # puis les manquants

GLiNER2 vit dans un venv séparé (``.venv-gliner``) : ``gliner2[local]`` impose
``transformers < 5``, et rétrograder ``transformers`` dans ``.venv`` mettrait en danger
la reproduction des vecteurs Qwen3 et du reranker.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "rag"))
import corpus_overlay  # noqa: E402

ROWS = ROOT / "data" / "embeddings" / "ingested-all-qwen3-06b" / "rows.jsonl"
IMPORTED_ROWS = ROOT / "rag" / "ingestion" / "imported-rows.jsonl"
OUT_DIR = ROOT / "data" / "graph" / "gliner-results"
MODEL = "fastino/gliner2.5-multi-v1"
#: Identique à l'amont — changer le schéma, c'est changer le graphe.
ENTITY_SCHEMA = {
    "person": "People, authors, researchers, traders, investors",
    "organization": "Companies, institutions, exchanges, regulators",
    "financial_instrument": "Stocks, bonds, options, futures, currencies, indices",
    "market_concept": "Financial concepts, models, theories, strategies, risks",
    "measure": "Prices, returns, volatility, rates, Greeks, statistical measures",
    "method": "Econometric, statistical, mathematical or trading methods",
    "dataset": "Named datasets, indexes, samples or time series",
}
RELATIONS = [
    "measures", "predicts", "causes", "depends_on", "correlates_with",
    "applies_to", "uses_method", "compares_with", "is_a", "part_of",
]
RELATION_THRESHOLD = 0.5
MAX_CHARACTERS = 12000  # comme l'amont


def load_rows() -> list[dict]:
    """Les chunks de l'état courant du corpus : overlays appliqués, ordre de ``rows.jsonl``.

    ``rows.jsonl`` est l'artefact amont : il porte les 19 443 chunks livrés, et **rien de ce
    qui est entré par une livraison locale**. S'en tenir à lui donnait un graphe aveugle aux
    documents importés — la forme d'incomplétude silencieuse que tout ce chantier cherche à
    rendre impossible. On le lit donc tel quel, pour que l'ordre et le texte des 18 636
    chunks d'origine restent identiques au bit près, puis on **ajoute en queue** ce que la
    collection contient et que lui ne connaît pas.
    """
    rows, seen = [], set()
    with ROWS.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            chunk = json.loads(line)["chunk"]
            text = corpus_overlay.apply(chunk["document_id"], chunk["chunk_id"], chunk.get("text") or "")
            if text is None:  # document retiré (duplicates-v1.json)
                continue
            seen.add(chunk["chunk_id"])
            rows.append({"chunk_id": chunk["chunk_id"], "document_id": chunk["document_id"],
                         "page_start": chunk.get("page_start"), "page_end": chunk.get("page_end"),
                         "text": text.strip()[:MAX_CHARACTERS]})
    rows.extend(imported_rows(seen))
    return rows


def imported_rows(known: set[str]) -> list[dict]:
    """Chunks entrés par une livraison (``rag/ingestion/imported-rows.jsonl``).

    Un simple fichier, au même format que ``rows.jsonl`` : lisible depuis ``.venv-gliner``,
    qui n'a pas ``qdrant_client`` — l'isolement des environnements est délibéré
    (``gliner2[local]`` impose ``transformers < 5``).
    """
    if not IMPORTED_ROWS.exists():
        return []
    out = []
    for line in IMPORTED_ROWS.open(encoding="utf-8"):
        if not line.strip():
            continue
        chunk = json.loads(line)["chunk"]
        if chunk["chunk_id"] in known:
            continue
        text = corpus_overlay.apply(chunk["document_id"], chunk["chunk_id"], chunk.get("text") or "")
        if text is None:
            continue
        out.append({"chunk_id": chunk["chunk_id"], "document_id": chunk["document_id"],
                    "page_start": chunk.get("page_start"), "page_end": chunk.get("page_end"),
                    "text": text.strip()[:MAX_CHARACTERS]})
    out.sort(key=lambda row: row["chunk_id"])
    if out:
        print(f"  {len(out)} chunks importés ajoutés au corpus du graphe "
              f"({len({r['document_id'] for r in out})} document(s))")
    return out


def stable_sample(rows: list[dict], count: int) -> list[dict]:
    """N chunks tirés sur tout le corpus, toujours les mêmes (ordre sha256 des chunk_id)."""
    ranked = sorted(rows, key=lambda row: hashlib.sha256(row["chunk_id"].encode()).hexdigest())
    return ranked[:count]


def pick_device(requested: str) -> str:
    import torch

    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(device: str):
    from gliner2 import AutoExtractor

    model = AutoExtractor.from_pretrained(MODEL, map_location=device, quantize=False)
    schema = model.create_schema().entities(ENTITY_SCHEMA)
    schema = schema.relations({relation: {"threshold": RELATION_THRESHOLD} for relation in RELATIONS})
    return model, schema


def to_item(row: dict, payload: dict | None, error: str | None = None) -> dict:
    payload = payload or {}
    relations = payload.get("relations", payload.get("relation_extraction", {}))
    if isinstance(relations, dict):
        relation_count = sum(len(items or []) for items in relations.values())
    else:
        relation_count = len(relations or [])
    item = {
        "chunk_id": row["chunk_id"], "document_id": row["document_id"],
        "page_start": row.get("page_start"), "page_end": row.get("page_end"),
        "text_chars": len(row["text"]),
        "entities": payload.get("entities", {}), "relations": relations, "relation_count": relation_count,
    }
    if error:
        item["error"] = error
    return item


def extract(model, schema, batch: list[dict], batch_size: int) -> list[dict]:
    """Un lot ; en cas d'échec, chunk par chunk pour isoler le fautif (l'erreur est conservée)."""
    texts = [row["text"] for row in batch]
    try:
        values = model.batch_extract(texts, schema, batch_size=batch_size, include_spans=True)
        return [to_item(row, value.to_dict() if hasattr(value, "to_dict") else value) for row, value in zip(batch, values)]
    except Exception as exc:  # noqa: BLE001
        if len(batch) == 1:
            return [to_item(batch[0], None, repr(exc))]
        return [item for row in batch for item in extract(model, schema, [row], 1)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Extraction GLiNER2 sur l'état courant du corpus")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "mps", "cuda"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int, help="ne traiter que les N premiers chunks en attente")
    parser.add_argument("--sample", type=int, help="échantillon stable de N chunks (mesure de vitesse)")
    parser.add_argument("--shard", help="i/n : ne traiter que les chunks d'indice ≡ i (mod n)")
    parser.add_argument("--merge-from", type=Path, nargs="*", default=[],
                        help="dossiers de shards dont results.jsonl est rapatrié avant de traiter le reste")
    parser.add_argument("--output", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    rows = load_rows()
    if args.sample:
        rows = stable_sample(rows, args.sample)
    if args.shard:
        part, total = (int(x) for x in args.shard.split("/"))
        rows = [row for i, row in enumerate(rows) if i % total == part]
    out_dir = args.output
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path, progress_path = out_dir / "results.jsonl", out_dir / "progress.json"

    done: dict[str, dict] = {}
    if results_path.exists():
        for line in results_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                item = json.loads(line)
                done[item["chunk_id"]] = item
    wanted = {row["chunk_id"] for row in rows}
    merged = 0
    with results_path.open("a", encoding="utf-8") as output:
        for shard_dir in args.merge_from:
            for line in (shard_dir / "results.jsonl").read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                item = json.loads(line)
                if item["chunk_id"] in wanted and item["chunk_id"] not in done:
                    output.write(json.dumps(item, ensure_ascii=False) + "\n")
                    done[item["chunk_id"]] = item
                    merged += 1
    pending = [row for row in rows if row["chunk_id"] not in done]
    if args.limit:
        pending = pending[:args.limit]
    device = pick_device(args.device)
    print(json.dumps({"corpus": corpus_overlay.describe(), "chunks": len(rows), "already_done": len(done), "merged": merged,
                      "pending": len(pending), "device": device, "batch_size": args.batch_size, "shard": args.shard},
                     ensure_ascii=False), flush=True)

    started = time.perf_counter()
    model, schema = load_model(device) if pending else (None, None)
    load_seconds = round(time.perf_counter() - started, 1)
    started = time.perf_counter()
    processed = errors = 0
    with results_path.open("a", encoding="utf-8") as output:
        for start in range(0, len(pending), args.batch_size):
            batch = pending[start:start + args.batch_size]
            for item in extract(model, schema, batch, args.batch_size):
                output.write(json.dumps(item, ensure_ascii=False) + "\n")
                done[item["chunk_id"]] = item
                processed += 1
                errors += "error" in item
            output.flush()
            elapsed = time.perf_counter() - started
            rate = processed / elapsed if elapsed else 0.0
            progress = {"status": "RUNNING", "completed": len(done), "total": len(rows), "processed": processed,
                        "errors": errors, "elapsed_seconds": round(elapsed, 1), "chunks_per_second": round(rate, 3),
                        "eta_minutes": round((len(pending) - processed) / rate / 60, 1) if rate else None}
            progress_path.write_text(json.dumps(progress, indent=2), encoding="utf-8")
            if processed % (args.batch_size * 25) < args.batch_size or start + args.batch_size >= len(pending):
                print(json.dumps(progress), flush=True)

    elapsed = round(time.perf_counter() - started, 1)
    items = list(done.values())
    report = {
        "status": "COMPLETED" if len(done) >= len(rows) else "PARTIAL",
        "model": MODEL, "device": device, "quantize": False, "batch_size": args.batch_size, "shard": args.shard,
        "merged_from": [str(d) for d in args.merge_from], "merged_items": merged,
        "entity_schema": ENTITY_SCHEMA, "relations": RELATIONS, "relation_threshold": RELATION_THRESHOLD,
        "max_characters": MAX_CHARACTERS,
        "corpus": corpus_overlay.describe(), "rows": str(ROWS.relative_to(ROOT)),
        "total": len(rows), "completed": len(done), "documents": len({item["document_id"] for item in items}),
        "processed_this_run": processed, "errors_this_run": errors,
        "chunks_with_error": sum("error" in item for item in items),
        "mentions_total": sum(len(spans or []) for item in items for spans in (item.get("entities") or {}).values()),
        "relations_total": sum(item.get("relation_count", 0) for item in items),
        "model_load_seconds": load_seconds, "elapsed_seconds": elapsed,
        "chunks_per_second": round(processed / elapsed, 3) if elapsed and processed else None,
        "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    progress_path.write_text(json.dumps({"status": report["status"], **{k: report[k] for k in ("completed", "total")}}, indent=2),
                             encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in ("entity_schema", "relations", "corpus")},
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
