"""Tableaux HTML → Markdown dans l'index, avec ré-embedding des seuls chunks-tableaux.

Pourquoi ici et pas à la source : la conversion à la source (``mineru_adapter._entry_text``)
suppose de relancer MinerU, absent de cette machine, et d'adapter le découpage du chunker
(qui coupe sur ``<tr>``). Les chunks existent ; on convertit leur texte et on remplace
leur vecteur. Les 15 485 autres chunks ne bougent pas.

Étapes, toutes vérifiées :
  1. **reproduction de l'embedding** : 40 chunks *non* tableaux ré-embarqués avec la même
     recette que le corpus (``Document: <titre d'export>\\nPath: <title_path>\\n\\n<texte>``,
     Qwen3-Embedding-0.6B, max_length 1024, fp16, normalisé) et comparés aux vecteurs
     stockés — cosinus minimal rapporté ; on n'écrit rien si < 0,99 ;
  2. conversion (``src/parsing/table_markdown.py``) ; les fragments d'un tableau découpé
     reçoivent l'en-tête de leur première partie (chaîne ``previous_chunk_id`` de
     ``data/processed/ingested/*/chunks.jsonl``) ;
  3. ré-embedding des chunks convertis (MPS), ``update_vectors`` + ``set_payload``
     (``text`` = Markdown, ``text_html`` = original conservé) ;
  4. ``tables-markdown-v1.json`` (overlay texte, versionné) et ``.cache/vectors-…npz``
     (overlay vecteurs, pour ``build_index.py``) ; BM25 reconstruit depuis l'index.

    python rag/tables/convert_tables.py --check     # étape 1 seule
    python rag/tables/convert_tables.py             # tout
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "rag"))
sys.path.insert(0, str(ROOT / "src"))
import corpus_overlay  # noqa: E402
import quant_rag  # noqa: E402
from parsing.table_markdown import header_of, html_table_to_markdown  # noqa: E402

OVERLAY = HERE / "tables-markdown-v1.json"
VECTORS = HERE / ".cache" / "vectors-tables-markdown-v1.npz"
RESULT = HERE / "results-convert-tables-v1.json"
PROCESSED = ROOT / "data" / "processed" / "ingested"
BATCH = 8


def embedding_text(title: str, title_path: str | None, text: str) -> str:
    """Même recette que ``src/embeddings/base.py::embedding_text`` (titre = celui de l'export)."""
    return f"Document: {title}\nPath: {title_path or title}\n\n{text.strip()}"


def original_titles() -> dict[str, str]:
    """Titre vu par l'embedding des vecteurs **actuellement stockés**.

    C'était le titre d'export. Depuis l'adoption des titres propres (3 septembre 2026,
    ``titles/apply_titles.py``), les vecteurs de la collection ont été calculés avec le
    titre consolidé : reproduire avec le titre d'export rendait un cosinus de 0,466 et la
    garde de ``main()`` interrompait le script. Ce n'est pas la mesure du 2 septembre qui
    était fausse, c'est la garde qui a cessé de décrire la recette en vigueur.

    On lit donc l'overlay des titres quand il est appliqué, et le titre d'export sinon.
    """
    if corpus_overlay.TITLES.exists():
        table = json.loads(corpus_overlay.TITLES.read_text(encoding="utf-8"))["documents"]
        return {document: entry["clean_title"] for document, entry in table.items()}
    titles = {}
    for line in (quant_rag.EXPORT / "payloads.jsonl").open(encoding="utf-8"):
        p = json.loads(line)["payload"]
        titles.setdefault(p["document_id"], p.get("title"))
    return titles


def embed(texts: list[str]) -> np.ndarray:
    model = quant_rag.embedder()
    out = []
    for start in range(0, len(texts), BATCH):
        out.append(model.encode(texts[start:start + BATCH], normalize_embeddings=True, show_progress_bar=False,
                                batch_size=BATCH))
        print(f"  embed {min(start + BATCH, len(texts))}/{len(texts)}", end="\r", flush=True)
    print()
    return np.vstack(out).astype(np.float32)


def scroll(client, flt=None, with_vectors=False):
    from qdrant_client import models  # noqa: F401

    points, offset = [], None
    while True:
        batch, offset = client.scroll(quant_rag.COLLECTION, limit=2048, offset=offset, scroll_filter=flt,
                                      with_payload=True, with_vectors=with_vectors)
        points.extend(batch)
        if offset is None:
            break
    return points


def check_reproduction(client, titles: dict, n: int = 40) -> dict:
    from qdrant_client import models

    flt = models.Filter(must_not=[models.FieldCondition(key="content_type", match=models.MatchValue(value="table"))])
    points = scroll(client, flt, with_vectors=True)
    sample = points[:: max(len(points) // n, 1)][:n]
    texts = [embedding_text(titles[p.payload["document_id"]], p.payload.get("title_path"), p.payload["text"]) for p in sample]
    new = embed(texts)
    stored = np.asarray([p.vector for p in sample], dtype=np.float32)
    cos = (new * stored).sum(axis=1)
    return {"chunks": len(sample), "cosine_min": float(cos.min()), "cosine_mean": float(cos.mean()),
            "cosine_median": float(np.median(cos))}


def part_chains(chunks: list[dict] | None = None) -> dict[str, str | None]:
    """chunk_id -> previous_chunk_id, pour retrouver la première partie d'un tableau découpé.

    ``chunks`` permet de lire la chaîne sur une **livraison en vol**, dont les fichiers ne
    sont pas encore dans ``data/processed/ingested``. Sans lui, ce module ne connaît que le
    corpus **promu** : c'est le comportement d'origine, et il est conservé tel quel pour la
    passe de conversion du corpus. Argument ajouté le 9 septembre 2026 par le chantier
    ``reprise-ingestion`` — voir ``apply_delivery.convert_tables_for``, où son absence
    faisait perdre l'en-tête de tout tableau fragmenté d'un document importé.
    """
    if chunks is not None:
        return {row["chunk_id"]: row.get("previous_chunk_id")
                for row in chunks if row.get("content_type") == "table"}
    previous = {}
    for folder in PROCESSED.iterdir():
        if not folder.is_dir():
            continue
        for line in (folder / "chunks.jsonl").open(encoding="utf-8"):
            row = json.loads(line)
            if row.get("content_type") == "table":
                previous[row["chunk_id"]] = row.get("previous_chunk_id")
    return previous


def convert_all(points, previous: dict[str, str | None] | None = None) -> tuple[dict[str, str], dict]:
    """``previous`` : la chaîne des fragments. ``None`` = la lire sur le corpus promu, ce qui
    est le comportement d'origine et celui de la passe du corpus."""
    by_id = {p.payload["chunk_id"]: p for p in points}
    previous = part_chains() if previous is None else previous
    # première partie = remonter la chaîne tant que le précédent est un tableau du même parent
    def first_part(cid: str) -> str:
        seen = {cid}
        while True:
            prev = previous.get(cid)
            if not prev or prev in seen or prev not in by_id or by_id[prev].payload.get("parent_id") != by_id[cid].payload.get("parent_id"):
                return cid
            seen.add(prev)
            cid = prev
    headers, converted, stats = {}, {}, defaultdict(int)
    for cid, p in by_id.items():
        head = first_part(cid)
        if head not in headers:
            headers[head] = header_of(by_id[head].payload["text"])
        header = headers[head] if head != cid else None
        md = html_table_to_markdown(p.payload["text"], header=header)
        if md == p.payload["text"]:
            stats["unchanged"] += 1
            continue
        converted[cid] = md
        stats["converted"] += 1
        stats["fragments_with_inherited_header"] += int(header is not None)
        stats["chars_html"] += len(p.payload["text"])
        stats["chars_markdown"] += len(md)
    return converted, dict(stats)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="reproduction de l'embedding seulement")
    args = parser.parse_args()
    from qdrant_client import models

    client = quant_rag.client()
    titles = original_titles()
    started = time.perf_counter()
    reproduction = check_reproduction(client, titles)
    print("reproduction de l'embedding :", json.dumps(reproduction))
    if args.check:
        return
    if reproduction["cosine_min"] < 0.99:
        sys.exit("ÉCHEC : l'embedding local ne reproduit pas les vecteurs du corpus — ne rien écrire")

    flt = models.Filter(must=[models.FieldCondition(key="content_type", match=models.MatchValue(value="table"))])
    points = scroll(client, flt)
    print(f"{len(points)} chunks-tableaux dans l'index")
    converted, stats = convert_all(points)
    print("conversion :", json.dumps(stats))

    ids = list(converted)
    by_id = {p.payload["chunk_id"]: p for p in points}
    texts = [embedding_text(titles[by_id[c].payload["document_id"]], by_id[c].payload.get("title_path"), converted[c]) for c in ids]
    t0 = time.perf_counter()
    vectors = embed(texts)
    embed_seconds = time.perf_counter() - t0
    shift = float(np.mean([1 - float(np.dot(vectors[i], np.asarray(client.retrieve(
        quant_rag.COLLECTION, ids=[by_id[c].id], with_vectors=True)[0].vector, dtype=np.float32)))
        for i, c in list(enumerate(ids))[:: max(len(ids) // 60, 1)]]))

    for start in range(0, len(ids), 256):
        block = ids[start:start + 256]
        client.update_vectors(quant_rag.COLLECTION, wait=True, points=[
            models.PointVectors(id=by_id[c].id, vector=vectors[ids.index(c)].tolist()) for c in block])
        for c in block:
            client.set_payload(quant_rag.COLLECTION, wait=True, points=[by_id[c].id],
                               payload={"text": converted[c], "text_html": by_id[c].payload["text"]})
        print(f"  écrit {min(start + 256, len(ids))}/{len(ids)}", end="\r", flush=True)
    print()
    VECTORS.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(VECTORS, chunk_ids=np.asarray(ids), vectors=vectors)
    OVERLAY.write_text(json.dumps({"version": "tables-markdown-v1",
                                   "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                   "converter": "src/parsing/table_markdown.py", "stats": stats,
                                   "chunks": converted}, ensure_ascii=False), encoding="utf-8")
    bm25 = quant_rag.rebuild_bm25()

    summary = {"table_chunks": len(points), **stats, "embedding_reproduction": reproduction,
               "mean_cosine_shift_html_to_markdown": round(shift, 4),
               "embed_seconds": round(embed_seconds, 1), "chunks_per_second": round(len(ids) / embed_seconds, 2),
               "bm25_chunks": bm25.doc_count, "seconds_total": round(time.perf_counter() - started, 1),
               "overlay": OVERLAY.name, "vectors_overlay": str(VECTORS.relative_to(ROOT))}
    RESULT.write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
