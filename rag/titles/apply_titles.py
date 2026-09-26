"""Applique les vecteurs « titres propres » à la collection Qdrant en place — l'adoption du chantier B.

``reembed_titles.py`` a produit ``.cache/vectors-clean-titles-v1.npz`` (18 636 chunks) et
``eval_titles.py`` a rendu le verdict (``results-titles-v1.json``). Ce script remplace les
vecteurs de la collection courante par ceux de l'overlay (``update_vectors``, par lots),
sans toucher aux payloads, puis vérifie sur un échantillon que ce qui est stocké est bien
ce qui a été écrit. ``build_index.py`` applique le même overlay à chaque reconstruction.

Après cette étape, tout cache de classements dense calculé sur l'ancienne collection est
faux : ``benchmark/.cache/router-retrievals.json`` est renommé ``…-export-titles.json`` et
``calibrate_router.py`` puis ``eval_router.py`` doivent être rejoués.

    .venv/bin/python rag/titles/apply_titles.py            # ~1 min ; refuse si le fichier de résultats manque
"""
from __future__ import annotations

import json
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "rag"))
import corpus_overlay  # noqa: E402
import quant_rag  # noqa: E402

VECTORS = corpus_overlay.TITLE_VECTORS
VERDICT = HERE / "results-titles-v1.json"
RESULT = HERE / "results-apply-titles-v1.json"
RETRIEVAL_CACHE = ROOT / "rag" / "benchmark" / ".cache" / "router-retrievals.json"
BATCH = 512


def main() -> None:
    if not VECTORS.exists():
        sys.exit(f"{VECTORS} absent : lance reembed_titles.py")
    if not VERDICT.exists():
        sys.exit(f"{VERDICT} absent : lance eval_titles.py — on n'applique pas un overlay non mesuré")
    from qdrant_client import models

    blob = np.load(VECTORS)
    chunk_ids, vectors = blob["chunk_ids"].tolist(), np.asarray(blob["vectors"], dtype=np.float32)
    client = quant_rag.client()
    started = time.perf_counter()

    # point id <- chunk_id, depuis la collection (les ids Qdrant sont ceux de l'export)
    point_of: dict[str, int] = {}
    offset = None
    while True:
        points, offset = client.scroll(collection_name=quant_rag.COLLECTION, limit=4096, offset=offset, with_payload=["chunk_id"])
        for p in points:
            point_of[p.payload["chunk_id"]] = p.id
        if offset is None:
            break
    missing = [c for c in chunk_ids if c not in point_of]
    if missing:
        sys.exit(f"{len(missing)} chunks de l'overlay absents de la collection (ex. {missing[:3]}) — état du corpus différent ?")
    if len(point_of) != len(chunk_ids):
        sys.exit(f"la collection a {len(point_of)} points, l'overlay {len(chunk_ids)} : état du corpus différent")

    written = 0
    for start in range(0, len(chunk_ids), BATCH):
        block = chunk_ids[start:start + BATCH]
        client.update_vectors(quant_rag.COLLECTION, wait=True, points=[
            models.PointVectors(id=point_of[c], vector=vectors[start + k].tolist()) for k, c in enumerate(block)])
        written += len(block)
        print(f"  {written}/{len(chunk_ids)}", end="\r", flush=True)
    print()

    # vérification : 200 points relus, cosinus contre l'overlay
    sample = list(range(0, len(chunk_ids), max(len(chunk_ids) // 200, 1)))[:200]
    got = client.retrieve(quant_rag.COLLECTION, ids=[point_of[chunk_ids[i]] for i in sample], with_vectors=True)
    by_id = {p.id: np.asarray(p.vector, dtype=np.float32) for p in got}
    cos = [float(by_id[point_of[chunk_ids[i]]] @ vectors[i] / (np.linalg.norm(by_id[point_of[chunk_ids[i]]]) * np.linalg.norm(vectors[i]))) for i in sample]

    renamed = None
    if RETRIEVAL_CACHE.exists():
        renamed = RETRIEVAL_CACHE.with_name("router-retrievals-export-titles.json")
        shutil.move(RETRIEVAL_CACHE, renamed)

    summary = {"status": "COMPLETED", "collection": quant_rag.COLLECTION, "vectors_written": written,
               "verification": {"points": len(sample), "cosine_min": min(cos), "cosine_mean": float(np.mean(cos))},
               "retrieval_cache_renamed": str(renamed.relative_to(ROOT)) if renamed else None,
               "corpus": corpus_overlay.describe(), "seconds": round(time.perf_counter() - started, 1),
               "applied_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    RESULT.write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    if min(cos) < 0.9999:
        sys.exit("ÉCHEC : un vecteur relu ne correspond pas à l'overlay")


if __name__ == "__main__":
    main()
