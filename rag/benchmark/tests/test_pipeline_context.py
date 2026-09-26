"""Le banc doit montrer au générateur ce que la production lui montre — titre compris.

Le défaut que ce test garde. ``pipeline._with_text`` complète les candidats venus de BM25
(qui n'ont pas de texte) en lisant l'index corpus. Il prenait le titre dans
``ChunkIndex.title_of``, c'est-à-dire le titre de **l'export** — pour les 319 documents
actifs, presque toujours dérivé du nom de fichier : « ssrn 4906546 », « 2026 08 03 Jacquier
RoughBergomi turns grey ». La production, elle, sert le titre **consolidé** :
``quant_rag._payload_row`` fait ``meta.get("title") or payload.get("title")``, et
``_fill_text`` passe par lui.

Conséquence : le banc montrait au générateur de moins bons titres que la production, et
aucun ``short_ref`` — donc aucune citation — pour les candidats venus de BM25.
``format_passages`` met justement ``short_ref`` en tête de chaque passage. Le banc
sous-estimait la production sur toutes les configurations qui passent par le lexical
(``bm25``, ``rrf``, ``rrf_rerank``).

Ce que ce test **ne** garde pas, et c'est délibéré : aucune métrique de retrieval n'est
concernée. ``metrics`` ne lit que ``chunk_id`` et ``document_id``, ``quant_rag._rerank`` ne
lit que ``text``. Seul change ce que le générateur a sous les yeux — donc aucune ligne de
base de retrieval n'est invalidée par le correctif.

Rien n'est chargé : l'index est un mannequin. Aucun Qdrant, aucun modèle, aucun corpus.

    .venv/bin/python -m pytest rag/benchmark/tests/test_pipeline_context.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
for chemin in (ROOT / "rag" / "benchmark", ROOT / "rag", ROOT / "src"):
    sys.path.insert(0, str(chemin))

import pipeline  # noqa: E402

CHUNK = "chunk-0000000000000001"
DOCUMENT = "doc-000000000000000a"


class IndexMannequin:
    """Le strict nécessaire de ``ChunkIndex`` pour ``_with_text``."""

    def __init__(self, fiche: dict | None):
        self.metadata = {DOCUMENT: fiche} if fiche else {}

    def get(self, chunk_id):
        if chunk_id != CHUNK:
            return None
        return {"chunk_id": CHUNK, "document_id": DOCUMENT, "text": "le texte du passage",
                "section": "3. Résultats", "page_start": 12}

    def title_of(self, document_id):
        return "ssrn 4906546"      # le titre de l'export, dérivé du nom de fichier


def complete(fiche: dict | None) -> dict:
    ligne = pipeline._with_text([{"chunk_id": CHUNK, "document_id": DOCUMENT, "score": 1.0}],
                                IndexMannequin(fiche))
    assert len(ligne) == 1
    return ligne[0]


def test_le_titre_consolide_prime_sur_celui_de_lexport():
    ligne = complete({"title": "Multivariate Cointegration in Statistical Arbitrage",
                      "short_ref": "Jungblut (2024)"})
    assert ligne["title"] == "Multivariate Cointegration in Statistical Arbitrage"
    assert ligne["title"] != "ssrn 4906546"


def test_la_citation_accompagne_le_titre():
    """``format_passages`` met ``short_ref`` en tête : sans lui, le passage arrive sans citation."""
    ligne = complete({"title": "Un titre", "short_ref": "Jungblut (2024)"})
    assert ligne["short_ref"] == "Jungblut (2024)"
    rendu = pipeline.format_passages([ligne], year=True)
    assert rendu.startswith("[1] Jungblut (2024) — Un titre — 3. Résultats")


def test_sans_fiche_consolidee_on_retombe_sur_lexport():
    """Un document sans fiche ne doit pas perdre son titre — le repli reste l'export."""
    ligne = complete(None)
    assert ligne["title"] == "ssrn 4906546"
    assert ligne["short_ref"] is None


def test_une_fiche_au_titre_vide_ne_masque_pas_lexport():
    ligne = complete({"title": "", "short_ref": None})
    assert ligne["title"] == "ssrn 4906546"


def test_le_texte_et_la_section_sont_toujours_completes():
    """La raison d'être de ``_with_text`` : un candidat sans texte est ignoré par le reranker."""
    ligne = complete({"title": "Un titre"})
    assert ligne["text"] == "le texte du passage"
    assert ligne["section"] == "3. Résultats"
    assert ligne["page_start"] == 12
