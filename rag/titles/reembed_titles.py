"""Ré-embedding du corpus avec le titre *propre* de chaque document — expérience isolée.

Le texte embarqué de chaque chunk commence par ``Document: <titre>`` (recette de
``src/embeddings/base.py``), et ce titre est celui de l'export : dérivé du nom de
fichier (« ssrn 1404905 », « 2026 08 03 Lin FactorEngine AuditableFactorMining »), un
nom de fichier de 200 caractères, ou — pour un document — un tableau HTML de
2 952 caractères qui précède chacun de ses 151 chunks. Les métadonnées consolidées
(``metadata/documents-metadata-v1.json``) donnent un titre lisible pour 256/256
documents. Question posée : le retrieval dense s'améliore-t-il si l'embedding voit ce
titre-là ? Rien d'autre ne change : même modèle, même recette, même texte, même
``Path:``.

Ce script ne touche ni à la collection ni à l'export. Il écrit un overlay de vecteurs
(``.cache/vectors-clean-titles-v1.npz``, tous les chunks) et la table des titres
(``titles-clean-v1.json``) ; ``benchmark/dense_matrix.py`` mesure la variante sans
Qdrant, et ``eval_titles.py`` rend le verdict. Si l'expérience est retenue,
``build_index.py`` appliquera l'overlay comme celui des tableaux.

Étapes :
  --audit   titres d'export contre titres propres (longueurs, HTML, exemples) — aucun modèle
  --check   reproduction : 40 chunks hors tableaux ré-embarqués avec le titre d'export et
            comparés aux vecteurs stockés (cosinus minimal ; on n'écrit rien sous 0,99)
  (défaut)  audit + check + ré-embedding complet, reprise sur interruption

    .venv/bin/python rag/titles/reembed_titles.py --audit
    .venv/bin/python rag/titles/reembed_titles.py --check
    .venv/bin/python rag/titles/reembed_titles.py            # ~100 min sur M4 (MPS)
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "rag"))
sys.path.insert(0, str(ROOT / "rag" / "benchmark"))
import corpus_overlay  # noqa: E402
import quant_rag  # noqa: E402

ROWS = ROOT / "data" / "embeddings" / "ingested-all-qwen3-06b" / "rows.jsonl"
#: Le relais des chunks entrés par livraison. ``rows.jsonl`` est l'export initial et ne
#: bouge plus : sans ce second fichier, ce script raisonnerait sur 256 documents quand le
#: corpus en sert 318, et l'audit des titres annoncerait « 256/256 » en toute confiance.
IMPORTED_ROWS = ROOT / "rag" / "ingestion" / "imported-rows.jsonl"
TITLES = HERE / "titles-clean-v1.json"
VECTORS = HERE / ".cache" / "vectors-clean-titles-v1.npz"
PARTIAL = HERE / ".cache" / "vectors-clean-titles-v1.partial.npz"
RESULT = HERE / "results-reembed-titles-v1.json"
BATCH = 8
BLOCK = 256          # chunks par sauvegarde partielle
MAX_TITLE = 200      # garde-fou : un titre propre plus long est tronqué au dernier mot

_TAGS = re.compile(r"<[^>]+>")


def embedding_text(title: str, title_path: str | None, text: str) -> str:
    """Même recette que ``src/embeddings/base.py::embedding_text`` et ``tables/convert_tables.py``."""
    return f"Document: {title}\nPath: {title_path or title}\n\n{text.strip()}"


def clean_title(export_title: str, record: dict | None) -> str:
    """Titre propre, conservateur : le titre consolidé, sinon le titre d'export nettoyé.

    - titre consolidé (métadonnées), espaces réduits ; c'est déjà ce que ``source`` affiche ;
    - repli : titre d'export sans balises HTML, espaces réduits ;
    - au plus ``MAX_TITLE`` caractères, coupé au dernier mot — un titre n'est pas un résumé.
    Ni auteurs ni année : ce serait une autre expérience (identité vs. contexte bibliographique).
    """
    candidate = (record or {}).get("title") or ""
    candidate = " ".join(candidate.split())
    if not candidate:
        candidate = " ".join(_TAGS.sub(" ", export_title or "").split())
    if len(candidate) > MAX_TITLE:
        candidate = candidate[:MAX_TITLE].rsplit(" ", 1)[0]
    return candidate or (export_title or "")


def _lire(fichier: Path, rows: list[dict], export_title: dict[str, str], vus: set[str]) -> None:
    """Un fichier au format ``rows.jsonl`` — l'export initial ou le relais des imports."""
    if not fichier.exists():
        return
    for line in fichier.open(encoding="utf-8"):
        if not line.strip():
            continue
        row = json.loads(line)
        document, chunk = row["document"], row["chunk"]
        if chunk["chunk_id"] in vus:
            continue
        text = corpus_overlay.apply(chunk["document_id"], chunk["chunk_id"], chunk.get("text") or "")
        if text is None:
            continue
        vus.add(chunk["chunk_id"])
        export_title.setdefault(chunk["document_id"], document.get("title") or "")
        rows.append({"chunk_id": chunk["chunk_id"], "document_id": chunk["document_id"],
                     "title_path": chunk.get("title_path"), "content_type": chunk.get("content_type"), "text": text})


def load_rows() -> tuple[list[dict], dict[str, str]]:
    """Chunks de l'état courant du corpus (overlays appliqués) et titre d'export par document.

    Deux sources, dans cet ordre : l'export initial, puis le relais des chunks entrés par
    livraison. ``chunk_id`` dédoublonne — un chunk réimporté n'est compté qu'une fois, et
    c'est la version de l'export qui prime, comme partout ailleurs dans le dépôt.
    """
    rows: list[dict] = []
    export_title: dict[str, str] = {}
    vus: set[str] = set()
    _lire(ROWS, rows, export_title, vus)
    _lire(IMPORTED_ROWS, rows, export_title, vus)
    return rows, export_title


def titles_table(export_title: dict[str, str]) -> dict[str, dict]:
    meta = quant_rag.document_metadata()
    return {doc: {"export_title": export_title[doc], "clean_title": clean_title(export_title[doc], meta.get(doc))}
            for doc in export_title}


def audit(rows: list[dict], titles: dict[str, dict]) -> dict:
    per_doc = {}
    for row in rows:
        per_doc[row["document_id"]] = per_doc.get(row["document_id"], 0) + 1
    export_lengths = sorted(len(t["export_title"]) for t in titles.values())
    clean_lengths = sorted(len(t["clean_title"]) for t in titles.values())
    html = [d for d, t in titles.items() if "<" in t["export_title"] and ">" in t["export_title"]]
    long = [d for d, t in titles.items() if len(t["export_title"]) > 200]
    no_path = sum(1 for r in rows if not r["title_path"])
    return {
        "documents": len(titles), "chunks": len(rows),
        "export_title_length": {"median": export_lengths[len(export_lengths) // 2],
                                "p90": export_lengths[int(0.9 * len(export_lengths))], "max": export_lengths[-1]},
        "clean_title_length": {"median": clean_lengths[len(clean_lengths) // 2], "max": clean_lengths[-1]},
        "export_titles_with_html": {"documents": len(html), "chunks": sum(per_doc[d] for d in html),
                                    "longest_characters": max((len(titles[d]["export_title"]) for d in html), default=0)},
        "export_titles_over_200": {"documents": len(long), "chunks": sum(per_doc[d] for d in long)},
        "identical_after_casefold": sum(1 for t in titles.values()
                                        if t["export_title"].strip().casefold() == t["clean_title"].strip().casefold()),
        "chunks_without_title_path": no_path,  # ``Path:`` retombe alors sur le titre — propre lui aussi désormais
        "examples": [{"document_id": d, "export": t["export_title"][:120], "clean": t["clean_title"]}
                     for d, t in list(titles.items())[:8]],
    }


def embed(texts: list[str]) -> np.ndarray:
    model = quant_rag.embedder()
    return np.asarray(model.encode(texts, normalize_embeddings=True, show_progress_bar=False, batch_size=BATCH),
                      dtype=np.float32)


def check_reproduction(rows: list[dict], titles: dict[str, dict], n: int = 40) -> dict:
    """Le modèle local reproduit-il les vecteurs stockés, titre d'export inchangé ?"""
    from dense_matrix import Matrix

    matrix = Matrix.load()
    sample = [r for r in rows if r["content_type"] != "table"]
    sample = sample[:: max(len(sample) // n, 1)][:n]
    texts = [embedding_text(titles[r["document_id"]]["export_title"], r["title_path"], r["text"]) for r in sample]
    started = time.perf_counter()
    new = embed(texts)
    seconds = time.perf_counter() - started
    stored = np.vstack([matrix.vectors[matrix.position_of(r["chunk_id"])] for r in sample])
    cos = (new * stored).sum(axis=1)
    # et le déplacement produit par le titre propre, sur les mêmes chunks
    clean = embed([embedding_text(titles[r["document_id"]]["clean_title"], r["title_path"], r["text"]) for r in sample])
    shift = (clean * stored).sum(axis=1)
    return {"chunks": len(sample), "cosine_min": float(cos.min()), "cosine_mean": float(cos.mean()),
            "cosine_median": float(np.median(cos)), "chunks_per_second": round(len(sample) / seconds, 2),
            "clean_title_cosine_to_stored": {"min": float(shift.min()), "mean": float(shift.mean()), "max": float(shift.max())}}


def reembed(rows: list[dict], titles: dict[str, dict]) -> dict:
    done: dict[str, np.ndarray] = {}
    if PARTIAL.exists():
        blob = np.load(PARTIAL)
        done = dict(zip(blob["chunk_ids"].tolist(), blob["vectors"]))
        print(f"reprise : {len(done)} chunks déjà embarqués")
    todo = [r for r in rows if r["chunk_id"] not in done]
    # les textes courts ensemble, les longs ensemble : moins de rembourrage par lot
    todo.sort(key=lambda r: len(r["text"]))
    started = time.perf_counter()
    for start in range(0, len(todo), BLOCK):
        block = todo[start:start + BLOCK]
        vectors = embed([embedding_text(titles[r["document_id"]]["clean_title"], r["title_path"], r["text"]) for r in block])
        for r, v in zip(block, vectors):
            done[r["chunk_id"]] = v
        PARTIAL.parent.mkdir(parents=True, exist_ok=True)
        np.savez(PARTIAL, chunk_ids=np.asarray(list(done)), vectors=np.vstack(list(done.values())))
        elapsed = time.perf_counter() - started
        rate = (start + len(block)) / elapsed
        print(f"  {len(done)}/{len(rows)}  {rate:.2f} chunks/s  reste ~{(len(todo) - start - len(block)) / max(rate, 1e-6) / 60:.0f} min",
              flush=True)
    ordered = [r["chunk_id"] for r in rows]
    np.savez_compressed(VECTORS, chunk_ids=np.asarray(ordered), vectors=np.vstack([done[c] for c in ordered]))
    PARTIAL.unlink(missing_ok=True)
    seconds = time.perf_counter() - started
    return {"chunks": len(rows), "embedded_now": len(todo), "seconds": round(seconds, 1),
            "chunks_per_second": round(len(todo) / seconds, 2) if seconds else None, "vectors": str(VECTORS.relative_to(ROOT))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--audit", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    rows, export_title = load_rows()
    titles = titles_table(export_title)
    report = {"version": "clean-titles-v1", "corpus": corpus_overlay.describe(),
              "recipe": "Document: <clean title>\\nPath: <title_path or clean title>\\n\\n<text>",
              "model": quant_rag.MODEL_ID, "device": quant_rag.device(), "max_length": quant_rag.MAX_LENGTH,
              "audit": audit(rows, titles)}
    print("audit :", json.dumps({k: v for k, v in report["audit"].items() if k != "examples"}, ensure_ascii=False))
    TITLES.write_text(json.dumps({"version": "clean-titles-v1", "rule": clean_title.__doc__.strip().split("\n")[0],
                                  "max_title": MAX_TITLE, "documents": titles}, ensure_ascii=False, indent=1), encoding="utf-8")
    if args.audit:
        RESULT.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
        return

    report["reproduction"] = check_reproduction(rows, titles)
    print("reproduction :", json.dumps(report["reproduction"]))
    if report["reproduction"]["cosine_min"] < 0.99:
        RESULT.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
        sys.exit("ÉCHEC : l'embedding local ne reproduit pas les vecteurs du corpus — rien n'est écrit")
    if args.check:
        RESULT.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
        return

    report["reembedding"] = reembed(rows, titles)
    report["finished_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    RESULT.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(report["reembedding"], indent=1))


if __name__ == "__main__":
    main()
