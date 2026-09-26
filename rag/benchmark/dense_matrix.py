"""Retrieval dense *sans Qdrant* : produit scalaire sur la matrice des vecteurs du corpus.

Pourquoi : Qdrant embarqué n'accepte qu'un processus à la fois sur ``qdrant_storage_local/``.
Les expériences de retrieval (titres propres, réécriture de requête, rerankers) n'ont
besoin que des vecteurs et des identifiants — tous deux sur le disque. Ce module
reconstruit exactement ce que la collection contient et cherche par force brute :
26 120 × 1 024 en float32, ~30 ms.

**Il a servi un corpus partiel pendant trois jours, et rien ne le disait.** L'export
``data/qdrant-export/`` est l'instantané *daté* des 258 documents livrés par l'amont ;
il n'est pas régénéré à l'ingestion, et il ne le sera jamais — ``build_index.py`` §12
explique pourquoi. Le 7 septembre 2026 il portait **18 636 chunks / 256 documents**
quand la collection en servait **26 120 / 418** : 7 484 passages, 162 documents entiers
manquaient, silencieusement. Aucun vecteur étranger, aucun périmé — un sous-ensemble
strict, ce qui est le mode de panne le plus difficile à voir : les résultats restent
plausibles, ils sont seulement calculés sur 71 % du corpus.

Deux corrections en découlent, et elles vont ensemble :

1. **La matrice a désormais les deux sources de la collection**, comme
   ``build_index.py`` : l'export *et* les documents entrés depuis, dont les vecteurs
   sont dans ``rag/ingestion/.cache/vectors-<livraison>.npz``. La recette n'est pas
   réécrite ici — ``build_index.imported_points`` est appelée telle quelle, pour qu'un
   changement de recette ne puisse pas faire diverger l'instrument de la production.
2. **Le chargement se prouve ou refuse.** ``Matrix.load()`` compare sa cardinalité au
   ``built_from`` du manifeste de l'index BM25 de la signature courante — le seul
   témoin hors ligne de ce que la collection contient, lu dans Qdrant au moment de sa
   construction. Sans manifeste, ou en désaccord, le chargement échoue au lieu de
   rendre un chiffre. ``allow_partial=True`` lève la garde et **étiquette** la matrice.

``--check`` reste la preuve de bout en bout : il rejoue les 155 questions des deux bancs
et compare le top-50 au classement dense que Qdrant a rendu
(``.cache/router-retrievals-<signature>.json``, écrit par ``calibrate_router.py``).

    .venv/bin/python rag/benchmark/dense_matrix.py --check

Usage dans un script :

    matrix = Matrix.load()                       # vecteurs de la collection courante
    rows = matrix.search(quant_rag.encode_query(q), pool=50, scope=None)
    other = Matrix.load(vectors=Path(".../vectors-clean-titles-v1.npz"))  # variante à comparer
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import corpus_overlay  # noqa: E402
import quant_rag  # noqa: E402

EXPORT = ROOT / "data" / "qdrant-export"
TABLE_VECTORS = ROOT / "rag" / "tables" / ".cache" / "vectors-tables-markdown-v1.npz"
TITLE_VECTORS = corpus_overlay.TITLE_VECTORS   # titres propres (3 septembre 2026), après les tableaux, s'il existe
#: Le nom porte la signature de l'état du corpus : un cache de classements calculé
#: sur un autre corpus n'est pas rechargé, il n'est pas trouvé. Il le fallait — un
#: import ajoute des chunks qui concourent contre l'or sans que l'or bouge.
CACHE = HERE / ".cache" / f"router-retrievals-{corpus_overlay.signature()}.json"
#: Manifeste de l'index BM25 de la signature courante. Son bloc ``built_from`` est écrit
#: en lisant la collection Qdrant elle-même : c'est le seul témoin hors ligne, et
#: indépendant de ce module, de ce que la collection servie contient.
LEXICAL = ROOT / "data" / "lexical"


def served_counts() -> dict | None:
    """Ce que la collection servie contient — ``None`` si aucun manifeste ne le dit.

    Rendre ``None`` plutôt qu'un chiffre supposé est délibéré : c'est la règle que
    ``mcp_server._comptes`` applique déjà — sans artefact, se taire au lieu de mentir.
    L'appelant décide, et ``Matrix.load`` décide de refuser.
    """
    path = LEXICAL / f"bm25-{corpus_overlay.LABEL}-{corpus_overlay.signature()}.manifest.json"
    if not path.exists():
        return None
    built = json.loads(path.read_text(encoding="utf-8")).get("built_from")
    return built if isinstance(built, dict) and built.get("points") else None


def imported_arrays() -> tuple[list[str], list[str], list[str], list, list[str]]:
    """Chunks entrés depuis l'export : identifiants, documents, types, vecteurs, alertes.

    **La recette n'est pas réimplémentée.** ``build_index.imported_points`` est appelée
    telle quelle et on ne garde que ce dont la matrice a besoin. Si la production change
    sa façon de composer un point importé — un filtre ``rag_eligible``, un overlay de
    plus —, l'instrument suit sans qu'on y pense. Une copie divergerait en silence, et
    c'est exactement le genre d'écart que ce module vient de payer.
    """
    sys.path.insert(0, str(ROOT / "rag"))
    import build_index  # noqa: PLC0415  — import tardif : il tire qdrant_client

    metadata = {}
    if build_index.METADATA.exists():
        records = json.loads(build_index.METADATA.read_text(encoding="utf-8"))["documents"]
        metadata = {r["document_id"]: {k: r.get(k) for k in build_index.PAYLOAD_FIELDS} for r in records}
    points, warnings = build_index.imported_points(metadata)
    chunk_ids = [p.payload["chunk_id"] for p in points]
    document_ids = [p.payload["document_id"] for p in points]
    content_types = [p.payload.get("content_type") or "" for p in points]
    vectors = [p.vector for p in points]
    return chunk_ids, document_ids, content_types, vectors, warnings


class Matrix:
    """Vecteurs normalisés de la collection courante, alignés sur ``chunk_ids``."""

    def __init__(self, chunk_ids: np.ndarray, document_ids: np.ndarray, vectors: np.ndarray, label: str,
                 content_types: np.ndarray | None = None):
        self.chunk_ids = chunk_ids
        self.document_ids = document_ids
        self.content_types = content_types if content_types is not None else np.asarray([""] * len(chunk_ids))
        self.vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        self.label = label
        self._position = {cid: i for i, cid in enumerate(chunk_ids.tolist())}
        self._by_document: dict[str, list[int]] = {}
        for i, doc in enumerate(document_ids.tolist()):
            self._by_document.setdefault(doc, []).append(i)

    # ------------------------------------------------------------------ construction

    @classmethod
    def load(cls, vectors: Path | None = None, label: str | None = None, titles: bool = True,
             imported: bool = True, allow_partial: bool = False) -> "Matrix":
        """La collection courante (export + imports + overlays), ou une variante de vecteurs.

        ``vectors`` : un ``.npz`` avec ``chunk_ids`` et ``vectors`` couvrant tout ou partie
        des chunks ; les chunks absents gardent le vecteur de la collection. C'est le
        même mécanisme que l'overlay des tableaux, qui est appliqué en premier, puis celui
        des titres propres s'il existe (``titles=False`` pour l'état « titres d'export »).

        ``imported=False`` rend l'état « export seul », c'est-à-dire ce que ce module
        chargeait avant le 7 septembre 2026 : utile pour *reproduire* une mesure ancienne,
        jamais pour en produire une nouvelle. Il impose ``allow_partial=True``.
        """
        ids = np.load(EXPORT / "ids.npy")
        stored = np.load(EXPORT / "vectors.npy", mmap_mode="r")
        payloads = [json.loads(line)["payload"] for line in (EXPORT / "payloads.jsonl").open(encoding="utf-8") if line.strip()]
        if not len(ids) == len(stored) == len(payloads):
            sys.exit(f"export incohérent : ids={len(ids)} vectors={len(stored)} payloads={len(payloads)}")
        removed = corpus_overlay.removed_documents()
        keep = [i for i, p in enumerate(payloads) if p.get("document_id") not in removed]
        chunk_ids = [payloads[i]["chunk_id"] for i in keep]
        document_ids = [payloads[i]["document_id"] for i in keep]
        content_types = [payloads[i].get("content_type") or "" for i in keep]
        matrix = np.asarray(stored[keep], dtype=np.float32) if keep else np.zeros((0, stored.shape[1]), np.float32)
        position = {cid: k for k, cid in enumerate(chunk_ids)}
        applied, alerts = [], []
        # Les overlays de l'amont ne portent que des chunks de l'export : ils s'appliquent
        # au bloc de l'export, exactement comme dans ``build_index.main``. Un point importé
        # tient son vecteur de sa livraison, et ni les tableaux ni les titres ne le touchent.
        for path in (TABLE_VECTORS, TITLE_VECTORS if titles and corpus_overlay.TITLES.exists() else None):
            if path is None:
                continue
            if not path.exists():
                if path is TABLE_VECTORS and corpus_overlay.text_overrides():
                    sys.exit(f"overlay des tableaux sans vecteurs : {path} absent")
                continue
            blob = np.load(path)
            n = 0
            for cid, vec in zip(blob["chunk_ids"].tolist(), blob["vectors"]):
                k = position.get(cid)
                if k is not None:
                    matrix[k] = vec
                    n += 1
            applied.append((path.name, n))

        if imported:
            more_ids, more_docs, more_types, more_vectors, alerts = imported_arrays()
            if more_ids:
                chunk_ids += more_ids
                document_ids += more_docs
                content_types += more_types
                matrix = np.concatenate([matrix, np.asarray(more_vectors, dtype=np.float32)])
                position = {cid: k for k, cid in enumerate(chunk_ids)}
                applied.append(("registre des imports", len(more_ids)))

        if vectors is not None:                       # la variante expérimentale, sur tout
            if vectors.exists():
                blob = np.load(vectors)
                n = 0
                for cid, vec in zip(blob["chunk_ids"].tolist(), blob["vectors"]):
                    k = position.get(cid)
                    if k is not None:
                        matrix[k] = vec
                        n += 1
                applied.append((vectors.name, n))

        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        matrix = matrix / np.maximum(norms, 1e-12)
        out = cls(np.asarray(chunk_ids), np.asarray(document_ids), matrix,
                  label or (vectors.stem if vectors else "collection"), np.asarray(content_types))
        out.applied = applied
        out.warnings = alerts
        out.signature = corpus_overlay.signature()
        out.partial = bool(allow_partial or not imported)
        out.verify(allow_partial=out.partial)
        return out

    # ------------------------------------------------------------------ preuve de couverture

    def verify(self, allow_partial: bool = False) -> dict:
        """La matrice couvre-t-elle la collection servie ? Sinon, on n'en fait rien.

        Le défaut du 7 septembre n'est pas qu'un export ait vieilli — c'était prévu — mais
        qu'un chargement incomplet **rende un résultat**. Un instrument qui mesure sur 71 %
        du corpus sans le dire est pire qu'un instrument absent : ses chiffres sont
        plausibles. La garde est donc au chargement, pas dans un ``--check`` que personne
        n'est obligé de lancer.
        """
        expected = served_counts()
        got = {"points": int(len(self.chunk_ids)), "documents": int(len(self._by_document))}
        self.coverage = {"attendu": expected, "obtenu": got, "signature": self.signature}
        if expected is None:
            message = (f"aucun manifeste d'index BM25 pour la signature {self.signature} : "
                       "ce que la collection contient n'est pas prouvable hors ligne. "
                       "Lance rag/benchmark/check_bm25.py, ou passe allow_partial=True "
                       "en sachant que la matrice n'est plus comparable au chemin servi.")
        elif got["points"] != expected.get("points") or got["documents"] != expected.get("documents"):
            manque = expected.get("points", 0) - got["points"]
            message = (f"matrice incomplète : {got['points']} chunks / {got['documents']} documents "
                       f"contre {expected.get('points')} / {expected.get('documents')} servis "
                       f"(signature {self.signature}) — il manque {manque} passages. "
                       "L'export data/qdrant-export/ est un instantané figé de l'amont ; les "
                       "documents entrés depuis viennent du registre et de "
                       "rag/ingestion/.cache/vectors-<livraison>.npz. Vérifie les alertes "
                       "de imported_arrays(), ou passe allow_partial=True pour une mesure "
                       "explicitement partielle.")
        else:
            self.coverage["complete"] = True
            return self.coverage
        self.coverage["complete"] = False
        if not allow_partial:
            sys.exit(f"dense_matrix : {message}")
        self.label = f"{self.label} (PARTIELLE)"
        print(f"  ATTENTION  dense_matrix : {message}", file=sys.stderr)
        return self.coverage

    # ------------------------------------------------------------------ requête

    def position_of(self, chunk_id: str) -> int | None:
        return self._position.get(chunk_id)

    def mask(self, scope: list[str] | None) -> np.ndarray | None:
        if scope is None:
            return None
        allowed = np.zeros(len(self.chunk_ids), dtype=bool)
        for doc in scope:
            for i in self._by_document.get(doc, ()):
                allowed[i] = True
        return allowed

    def scores(self, vector) -> np.ndarray:
        q = np.asarray(vector, dtype=np.float32)
        q = q / max(float(np.linalg.norm(q)), 1e-12)
        return self.vectors @ q

    def search(self, vector, pool: int = quant_rag.POOL, scope: list[str] | None = None,
               skip_headings: bool = True) -> list[dict]:
        """Top-``pool`` par cosinus, restreint aux documents de ``scope`` s'il est donné.

        Même contrat de sortie que le chemin dense du banc (``pipeline.retrieve``) : le
        top-``pool`` de Qdrant, dont ``quant_rag._select`` retire ensuite les chunks
        ``content_type == "heading"`` *même* quand tout autre filtre est désactivé —
        d'où des listes de 47 à 50 entrées dans le cache. ``skip_headings=True`` reproduit
        ce comportement ; sans lui, 43 des 155 top-10 diffèrent.
        """
        scores = self.scores(vector)
        allowed = self.mask(scope)
        if allowed is not None:
            scores = np.where(allowed, scores, -np.inf)
        pool = min(pool, len(scores))
        top = np.argpartition(-scores, pool - 1)[:pool] if pool < len(scores) else np.arange(len(scores))
        # tri stable : score décroissant, puis chunk_id — Qdrant départage les ex æquo par
        # identifiant de point ; les ex æquo exacts sont rarissimes en float32.
        top = sorted(top.tolist(), key=lambda i: (-float(scores[i]), self.chunk_ids[i]))
        return [{"chunk_id": str(self.chunk_ids[i]), "document_id": str(self.document_ids[i]),
                 "score": float(scores[i]), "score_kind": "cosine", "content_type": str(self.content_types[i])}
                for i in top if np.isfinite(scores[i]) and not (skip_headings and self.content_types[i] == "heading")]

    def search_text(self, query: str, pool: int = quant_rag.POOL, scope: list[str] | None = None,
                    skip_headings: bool = True) -> list[dict]:
        return self.search(quant_rag.encode_query(query), pool=pool, scope=scope, skip_headings=skip_headings)


# ---------------------------------------------------------------------- preuve

def check(matrix: Matrix) -> dict:
    """Le top-50 de la matrice est-il celui que Qdrant a rendu ? (155 questions, en cache)."""
    import pipeline
    from compare_v1_v2 import load_bench, load_v1
    from corpus import ChunkIndex

    if not CACHE.exists():
        sys.exit("cache de calibration absent : lance d'abord calibrate_router.py")
    cache = json.loads(CACHE.read_text(encoding="utf-8"))
    index = ChunkIndex.load(verbose=False)
    items = [("v1", it) for it in load_v1(index)] + [("v3", it) for it in load_bench(HERE / "questions-v3.jsonl")]
    identical_50, identical_10, checked, worst = 0, 0, 0, []
    for bench, item in items:
        key = f"{bench}/{item['qid']}"
        expected = cache.get(key, {}).get("dense")
        if expected is None:
            continue
        filters = pipeline.filters_of(item)
        scope = quant_rag.document_scope(None, **filters) if filters else None
        got = matrix.search_text(pipeline.query_of(item), pool=pipeline.POOL, scope=scope)
        got_ids = [r["chunk_id"] for r in got]
        exp_ids = [c for c, _, _ in expected]
        checked += 1
        identical_50 += got_ids == exp_ids
        identical_10 += got_ids[:10] == exp_ids[:10]
        if got_ids[:10] != exp_ids[:10]:
            # première position qui diffère, et l'écart de score correspondant
            first = next(i for i in range(min(len(got_ids), len(exp_ids))) if got_ids[i] != exp_ids[i])
            exp_scores = {c: s for c, _, s in expected}
            worst.append({"key": key, "first_difference_at": first + 1,
                          "matrix": got_ids[first], "qdrant": exp_ids[first],
                          "score_gap": round(abs(got[first]["score"] - exp_scores.get(exp_ids[first], got[first]["score"])), 6)})
        print(f"  {key:<8} top10 {'=' if got_ids[:10] == exp_ids[:10] else '≠'}  top50 {'=' if got_ids == exp_ids else '≠'}", end="\r", flush=True)
    print()
    return {"checked": checked, "identical_top50": identical_50, "identical_top10": identical_10,
            "differences": worst[:20], "vectors_applied": matrix.applied,
            "chunks": int(len(matrix.chunk_ids)), "couverture": matrix.coverage}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--check", action="store_true", help="compare le top-50 aux classements Qdrant en cache")
    parser.add_argument("--vectors", type=Path, help="variante de vecteurs (.npz chunk_ids/vectors)")
    parser.add_argument("--export-seul", action="store_true",
                        help="état d'avant le 7 septembre 2026 : export sans les imports (mesure partielle)")
    parser.add_argument("query", nargs="?")
    args = parser.parse_args()
    m = Matrix.load(vectors=args.vectors, imported=not args.export_seul)
    print(f"matrice : {len(m.chunk_ids)} chunks, {len(m._by_document)} documents, overlays {m.applied}")
    for alert in getattr(m, "warnings", []):
        print(f"  ATTENTION  {alert}", file=sys.stderr)
    if args.check:
        result = check(m)
        print(json.dumps(result, indent=1, ensure_ascii=False))
    elif args.query:
        for i, row in enumerate(m.search_text(args.query, pool=10), 1):
            print(f"[{i}] {row['score']:.4f} {row['chunk_id']} {row['document_id']}")
