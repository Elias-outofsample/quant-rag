"""``pipeline.retrieve`` doit interroger la profondeur qu'on lui demande, pas ``POOL``.

Le défaut que ce test garde. ``retrieve`` codait en dur ``limit=POOL, pool=POOL`` dans son
appel à Qdrant et n'appliquait le ``limit`` demandé qu'en **troncature après** la requête.
Appelé avec ``limit=100``, il rendait 47 candidats en laissant croire qu'il en rendait 100 —
le filtre d'en-têtes de ``quant_rag._select`` rogne le top-50 de Qdrant. Trouvé par la
Phase 0 du chantier « rappel du pool », qui avait précisément besoin d'une profondeur 100 ;
un banc bâti dessus aurait mesuré la profondeur 50 en croyant mesurer la 100.

Ce qui doit rester vrai, et que ce test vérifie dans les deux sens :

  - **aucun appelant existant n'est déplacé.** Tous demandent ``limit=pipeline.POOL`` ou
    moins, et ``max(limit, POOL) == POOL`` y est une identité : la profondeur interrogée
    reste 50, donc la ligne de base ne bouge pas. Ce n'est pas une espérance, c'est le même
    appel avec les mêmes arguments ;
  - **au-delà de 50, la profondeur demandée est honorée**, côté dense comme côté lexical.

Rien n'est chargé : ``quant_rag.search`` et ``quant_rag._lexical`` sont remplacés par des
mouchards qui enregistrent leurs arguments. Aucun modèle, aucun Qdrant, aucune écriture —
et le test reste exécutable pendant qu'un banc tourne.

    .venv/bin/python -m pytest rag/benchmark/tests/test_pipeline_depth.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
for chemin in (ROOT / "rag" / "benchmark", ROOT / "rag", ROOT / "src"):
    sys.path.insert(0, str(chemin))

import pipeline  # noqa: E402
import quant_rag  # noqa: E402


class Mouchard:
    """Remplace les deux points d'entrée de recherche et retient ce qu'on leur a demandé."""

    def __init__(self):
        self.dense: list[dict] = []
        self.lexical: list[int] = []
        self._search, self._lexical_fn = quant_rag.search, quant_rag._lexical

    def __enter__(self):
        quant_rag.search = lambda question, **kw: (self.dense.append(kw), [])[1]
        quant_rag._lexical = lambda question, pool, scope: (self.lexical.append(pool), [])[1]
        return self

    def __exit__(self, *_):
        quant_rag.search, quant_rag._lexical = self._search, self._lexical_fn
        return False


def profondeur_dense(limit: int) -> dict:
    with Mouchard() as espion:
        pipeline.retrieve("question", "dense", None, limit=limit)
    return espion.dense[0]


def test_les_appelants_existants_ne_sont_pas_deplaces():
    """Tout appel à ``limit <= POOL`` interroge exactement la profondeur d'avant."""
    for limit in (1, 5, 6, 25, pipeline.POOL):
        appel = profondeur_dense(limit)
        assert appel["pool"] == pipeline.POOL, f"limit={limit} : pool {appel['pool']} au lieu de {pipeline.POOL}"
        assert appel["limit"] == pipeline.POOL, f"limit={limit} : limit {appel['limit']} au lieu de {pipeline.POOL}"


def test_une_profondeur_superieure_est_honoree():
    for limit in (51, 100, 150):
        appel = profondeur_dense(limit)
        assert appel["pool"] == limit, f"limit={limit} : Qdrant interrogé à {appel['pool']}"
        assert appel["limit"] == limit


def test_le_lexical_suit_la_meme_profondeur():
    for limit, attendu in ((10, pipeline.POOL), (pipeline.POOL, pipeline.POOL), (100, 100)):
        with Mouchard() as espion:
            pipeline.retrieve("question", "bm25", None, limit=limit)
        assert espion.lexical == [attendu], f"limit={limit} : BM25 interrogé à {espion.lexical}"


def test_le_mode_dense_reste_impose():
    """Une ligne de base qui route n'est plus une ligne de base — le mode ne doit pas dériver."""
    appel = profondeur_dense(100)
    assert appel["mode"] == "dense"
    assert appel["rerank"] is False
    assert appel["auto_period"] is False
    assert appel["dedupe"] is False
    assert appel["per_document"] == 0
    assert appel["min_characters"] == 0
