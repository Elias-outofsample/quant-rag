"""Construit un index BM25 **de l'ombre** avec les titres du registre, à côté de la production.

Le texte lexical porte le titre du document (``retrieval.lexical.lexical_text`` :
``[document["title"], title_path, text]``). Or l'index servi porte le titre *de l'export* —
c'est-à-dire un dérivé du nom de fichier : ``EngleGranger1987``, ``2026 08 03 Jacquier
RoughBergomi turns grey``. Les 319 documents actifs ont un titre d'export différent de leur
titre consolidé. Personne n'a jamais mesuré ce que BM25 vaut avec les vrais titres.

Ce script ne réécrit rien. Il produit deux index en mémoire et n'en écrit qu'un :

    export     reconstruit à l'identique de la production — c'est le **contrôle** :
               s'il ne redonne pas le sha256 de l'index servi, alors mon parcours de la
               collection diffère de ``quant_rag.rebuild_bm25`` pour une autre raison que
               les titres, et la comparaison qui suit ne mesure pas ce qu'elle prétend.
    registry   le même, titres consolidés (``documents-metadata-v1.json``), écrit dans
               ``.cache/`` — gitignoré, hors de ``data/lexical/``, hors de la production.

    .venv/bin/python rag/benchmark/build_bm25_titles.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(ROOT / "src"))

import corpus_overlay  # noqa: E402
import quant_rag  # noqa: E402
from retrieval.lexical import BM25Index  # noqa: E402

CACHE = HERE / ".cache"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def scroll_points() -> list[dict]:
    """Les payloads servis, dans l'ordre où ``rebuild_bm25`` les lit."""
    rows, offset = [], None
    while True:
        points, offset = quant_rag.client().scroll(
            collection_name=quant_rag.COLLECTION, limit=4096, offset=offset,
            with_payload=["chunk_id", "document_id", "text", "title", "title_path", "content_type"])
        rows.extend(p.payload for p in points)
        if offset is None:
            return rows


def titles(source: str, payloads: list[dict]) -> dict[str, str]:
    """``document_id`` -> titre porté par le texte lexical, selon la source demandée."""
    if source == "registry":
        meta = quant_rag.document_metadata()
        return {p["document_id"]: (meta.get(p["document_id"], {}).get("title") or p.get("title"))
                for p in payloads}
    original = {}
    for line in (quant_rag.EXPORT / "payloads.jsonl").open(encoding="utf-8"):
        payload = json.loads(line)["payload"]
        original.setdefault(payload["document_id"], payload.get("title"))
    return {p["document_id"]: (original.get(p["document_id"]) or p.get("title")) for p in payloads}


def build(source: str, payloads: list[dict]) -> BM25Index:
    title_of = titles(source, payloads)
    rows = [({"document_id": p["document_id"], "title": title_of[p["document_id"]]},
             {"chunk_id": p["chunk_id"], "document_id": p["document_id"], "text": p.get("text", ""),
              "title_path": p.get("title_path"), "content_type": p.get("content_type")})
            for p in payloads]
    return BM25Index.from_corpus(rows)


def main():
    signature = corpus_overlay.signature()
    payloads = scroll_points()
    print(f"collection : {len(payloads)} points, {len({p['document_id'] for p in payloads})} documents")

    CACHE.mkdir(parents=True, exist_ok=True)
    temoin = CACHE / f"bm25-controle-export-{signature}.json"
    build("export", payloads).dump(temoin)
    attendu = sha256(quant_rag.bm25_path())
    obtenu = sha256(temoin)
    temoin.unlink()
    print(f"contrôle : reconstruction à titres d'export -> {obtenu[:16]}…")
    print(f"           index servi                      -> {attendu[:16]}…")
    if obtenu != attendu:
        sys.exit("ARRÊT : la reconstruction ne redonne pas l'index servi. L'écart mesuré ensuite "
                 "ne serait pas imputable aux seuls titres.")
    print("           identiques — l'écart qui suit ne tiendra qu'aux titres")

    export_titles, registry_titles = titles("export", payloads), titles("registry", payloads)
    changes = sum(1 for d in registry_titles if (registry_titles[d] or "") != (export_titles[d] or ""))
    index = build("registry", payloads)
    sortie = CACHE / f"bm25-titles-registry-{signature}.json"
    index.dump(sortie)
    (CACHE / f"bm25-titles-registry-{signature}.manifest.json").write_text(json.dumps({
        "index": sortie.name, "title_source": "registry (rag/metadata/documents-metadata-v1.json)",
        "shadow": True, "production_index": quant_rag.bm25_path().name,
        "production_sha256": attendu, "sha256": sha256(sortie),
        "built_from": {"collection": quant_rag.COLLECTION, "points": len(payloads),
                       "documents": len({p["document_id"] for p in payloads})},
        "documents_dont_le_titre_change": changes,
        "records": index.doc_count, "terms": len(index.document_frequency),
        "k1": index.k1, "b": index.b, "corpus_signature": signature,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\nindex de l'ombre : {sortie}")
    print(f"  {index.doc_count} records, {len(index.document_frequency)} termes "
          f"(servi : {len(quant_rag.bm25().document_frequency)} termes)")
    print(f"  titres changés : {changes} document(s) sur {len(registry_titles)}")
    print(f"  sha256 : {sha256(sortie)[:16]}…")


if __name__ == "__main__":
    main()
