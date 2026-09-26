"""``describe()`` ne doit pas lever quand un overlay sort de l'arbre du module.

Le défaut, §11.3 du rapport ``reprise-ingestion``. ``describe()`` exprimait chaque chemin
en relatif de ``HERE.parent`` ; ``Path.relative_to`` **lève** ``ValueError`` dès qu'on sort
de cet arbre. Ce n'est pas une hypothèse d'école : ``describe()`` est appelée par
``quant_rag.rebuild_bm25``, donc à l'**étape 10** d'un import — après l'upsert et après la
promotion, pile dans la fenêtre que le journal existe pour décrire. Un import mourait donc
au pire moment, pour une raison qui n'avait rien à voir avec lui.

Ce test est direct : il n'a besoin ni de Qdrant, ni de corpus, ni de livraison.
"""
from __future__ import annotations

import sys
from pathlib import Path

MACOS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(MACOS))

import corpus_overlay  # noqa: E402


def test_un_overlay_hors_de_l_arbre_du_module_ne_fait_plus_lever(tmp_path, monkeypatch):
    """Le cas exact : un overlay déplacé ailleurs, comme dans un arbre de test."""
    ailleurs = tmp_path / "ailleurs"
    ailleurs.mkdir()
    for nom in ("DUPLICATES", "TABLES", "TITLES", "REGISTRY", "METADATA"):
        monkeypatch.setattr(corpus_overlay, nom, ailleurs / f"{nom.lower()}.json")
    monkeypatch.setattr(corpus_overlay, "TITLE_VECTORS", ailleurs / "vecteurs.npz")
    corpus_overlay.invalidate()

    etat = corpus_overlay.describe()               # levait ValueError avant le 9 sept. 2026

    assert etat["label"] == corpus_overlay.LABEL
    assert etat["overlays"][0]["file"].endswith("duplicates.json")
    assert etat["registry"]["present"] is False
    assert isinstance(etat["signature"], str) and etat["signature"]


def test_le_chemin_reste_relatif_quand_l_overlay_est_dans_l_arbre(monkeypatch):
    """La correction ne doit pas rendre tous les chemins absolus : un manifeste relu dans
    six mois se lit mieux en relatif, et les manifestes déjà écrits le sont."""
    corpus_overlay.invalidate()
    etat = corpus_overlay.describe()
    assert etat["overlays"][0]["file"] == "rag/metadata/duplicates-v1.json"
    assert etat["registry"]["file"] == "rag/ingestion/registry-v1.json"
    assert not Path(etat["titles"]["file"]).is_absolute()
