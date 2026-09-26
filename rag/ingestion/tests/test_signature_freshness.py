"""La signature et le nom de l'index BM25 ne doivent jamais être périmés dans un processus.

Ce test garde le mode de défaillance que ce dépôt a déjà payé deux fois : un index dont le
contenu est juste et le nom ment. Il tient à deux choses, toutes deux faciles à casser sans
s'en apercevoir :

  - ``corpus_overlay.signature()`` et ``registry_digest()`` **ne doivent pas être mémorisées** :
    un processus qui modifie un overlay ou le registre en cours d'exécution — ce que fera
    ``apply_delivery.py``, comme le font déjà ``convert_tables.py`` et ``apply_duplicates.py`` —
    doit voir la nouvelle valeur immédiatement ;
  - ``quant_rag.BM25_PATH`` **ne doit pas être une constante de module** : figée à l'import,
    elle ferait écrire ``rebuild_bm25()`` sous l'ancien nom.

Aucune écriture dans le corpus : chaque cas restaure l'état d'origine.

    .venv/bin/python -m pytest rag/ingestion/tests/test_signature_freshness.py -q
"""
from __future__ import annotations

import json
import sys
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "rag"))
sys.path.insert(0, str(ROOT / "src"))

import corpus_overlay  # noqa: E402
import quant_rag  # noqa: E402


@contextmanager
def temporarily(path: Path, content: bytes):
    """Remplace le contenu d'un fichier, puis le restaure quoi qu'il arrive."""
    existed = path.exists()
    backup = path.read_bytes() if existed else None
    path.write_bytes(content)
    try:
        yield
    finally:
        if existed:
            path.write_bytes(backup)
        else:
            path.unlink(missing_ok=True)
        corpus_overlay.invalidate()


def _registry_with_extra_document() -> bytes:
    data = json.loads(corpus_overlay.REGISTRY.read_text(encoding="utf-8"))
    data["documents"].append({"sha256": "f" * 64, "document_id": "doc-test-eph",
                              "status": "active", "files": {"chunks.jsonl": "a" * 64}})
    return json.dumps(data).encode()


def test_signature_suit_le_registre_dans_le_meme_processus():
    before = corpus_overlay.signature()
    with temporarily(corpus_overlay.REGISTRY, _registry_with_extra_document()):
        assert corpus_overlay.signature() != before, (
            "signature() est mémorisée : ajouter un document ne changerait pas le nom de "
            "l'index BM25, qui serait rechargé incomplet")
    assert corpus_overlay.signature() == before


def test_signature_suit_les_overlays_dans_le_meme_processus():
    before = corpus_overlay.signature()
    payload = json.loads(corpus_overlay.DUPLICATES.read_text(encoding="utf-8"))
    payload["remove"] = list(payload.get("remove", [])) + [
        {"document_id": "doc-test-eph", "chunks": 1, "reason": "test", "kept": {}}]
    with temporarily(corpus_overlay.DUPLICATES, json.dumps(payload).encode()):
        corpus_overlay.invalidate()
        assert corpus_overlay.signature() != before
    assert corpus_overlay.signature() == before


def test_bm25_path_nest_pas_fige_a_limport():
    before = quant_rag.BM25_PATH
    with temporarily(corpus_overlay.REGISTRY, _registry_with_extra_document()):
        assert quant_rag.BM25_PATH != before, (
            "BM25_PATH est une constante de module : rebuild_bm25() écrirait sous l'ancien "
            "nom, et le fichier mentirait sur ce qu'il contient")
        assert quant_rag.BM25_PATH.name.startswith(f"bm25-{corpus_overlay.LABEL}-")
        assert quant_rag.BM25_MANIFEST.name.endswith(".manifest.json")
    assert quant_rag.BM25_PATH == before


def test_empreinte_du_registre_ignore_les_horodatages():
    """Reconstruire le registre sans changer le corpus ne doit pas invalider les index."""
    data = json.loads(corpus_overlay.REGISTRY.read_text(encoding="utf-8"))
    before = corpus_overlay.registry_digest()
    data["built_at"] = "1999-01-01T00:00:00+00:00"
    for entry in data["documents"]:
        entry["delivery"] = dict(entry.get("delivery", {}), received_at="1999-01-01T00:00:00+00:00")
    with temporarily(corpus_overlay.REGISTRY, json.dumps(data).encode()):
        assert corpus_overlay.registry_digest() == before, (
            "un horodatage entre dans l'empreinte : reconstruire le registre invaliderait "
            "l'index BM25 et les caches du banc pour rien")


def test_invalidate_purge_les_overlays_memorises():
    """``invalidate()`` doit vider les overlays mémorisés, et l'état d'origine doit revenir.

    Le compte de documents retirés est **relevé, pas codé en dur** : il a valu 2, puis 3
    après l'échange CST2010 → preprint SSRN (``b6bc88a``), et le test échouait depuis sans
    que rien ne soit cassé. Un test qui suit le corpus au lieu de vérifier la propriété
    qu'il garde finit par crier pour une raison qui n'est pas la sienne — et on l'ignore.
    """
    avant = dict(corpus_overlay.removed_documents())
    corpus_overlay.text_overrides()
    assert avant, "l'overlay de déduplication est vide : le test ne garde plus rien"
    payload = {"version": "test", "remove": []}
    with temporarily(corpus_overlay.DUPLICATES, json.dumps(payload).encode()):
        corpus_overlay.invalidate()
        assert corpus_overlay.removed_documents() == {}, (
            "invalidate() n'a pas purgé les documents retirés mémorisés")
    corpus_overlay.invalidate()
    assert corpus_overlay.removed_documents() == avant, (
        "l'état d'origine n'est pas revenu après restauration de l'overlay")
