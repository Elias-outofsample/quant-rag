"""``--sans-graphe`` : sauter l'étape du graphe, et le dire.

Pourquoi le drapeau existe
---------------------------
L'extraction GLiNER coûte ~95 s par document, et le graphe est mesuré **sans apport** — ni
au rang, ni au niveau réponse (§12 de ``docs/STRATEGIE.md``). Aucun drapeau ne permettait
de la sauter : un lot qui n'avait pas besoin du graphe le payait quand même.

Ce que ces tests figent
------------------------
1. le **défaut ne bouge pas** : sans le drapeau, ``SANS_GRAPHE`` est faux et l'étape court ;
2. avec le drapeau, l'étape rend ``True`` sans lancer un seul sous-processus ;
3. l'état du lot porte ``sautée`` **et son motif** — un « graphe » absent du rapport se
   lirait comme un graphe à jour, ce qui est le contraire de la vérité ;
4. le message dit ce qu'on perd, en toutes lettres.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "rag" / "ingestion"))
sys.path.insert(0, str(ROOT / "rag"))

import batch_driver  # noqa: E402


@pytest.fixture
def doc() -> dict:
    return {"pdf": "essai.pdf", "mesures": {}}


def test_le_defaut_n_a_pas_bouge():
    """Le drapeau ajoute une possibilité ; il ne change pas ce que fait un lot ordinaire."""
    assert batch_driver.SANS_GRAPHE is False


def test_avec_le_drapeau_l_etape_rend_vrai_sans_lancer_de_sous_processus(doc, monkeypatch):
    """La preuve qu'elle est bien sautée : ``courir`` est piégé et ne doit jamais être appelé."""
    appels = []
    monkeypatch.setattr(batch_driver, "courir",
                        lambda *a, **k: appels.append(a) or (_ for _ in ()).throw(
                            AssertionError("l'étape du graphe a lancé un sous-processus")))
    monkeypatch.setattr(batch_driver, "SANS_GRAPHE", True)

    ok, message = batch_driver.etape_graphe(doc, ROOT / "journal-inexistant.log", False)
    assert ok is True
    assert appels == []


def test_l_etat_du_lot_porte_sautee_et_son_motif(doc, monkeypatch):
    """Un rapport de lot doit pouvoir dire, six mois plus tard, que ce document n'a pas de
    graphe — et pourquoi."""
    monkeypatch.setattr(batch_driver, "SANS_GRAPHE", True)
    batch_driver.etape_graphe(doc, ROOT / "journal-inexistant.log", False)

    graphe = doc["mesures"]["graphe"]
    assert graphe["etape"] == "sautée"
    assert graphe["motif"] == "--sans-graphe"
    assert graphe["secondes"] == 0.0
    for outil in ("search_graph", "expand_entity", "connect_entities"):
        assert outil in graphe["consequence"], (
            "l'état doit nommer les outils qui cessent de refléter le document")


def test_le_message_dit_ce_qu_on_perd(doc, monkeypatch):
    monkeypatch.setattr(batch_driver, "SANS_GRAPHE", True)
    _, message = batch_driver.etape_graphe(doc, ROOT / "journal-inexistant.log", False)
    assert "sautée" in message and "--sans-graphe" in message
    assert "rattrapé" in message, "le message doit dire que le retard est rattrapable"


def test_le_drapeau_est_offert_en_ligne_de_commande():
    """Un drapeau que l'analyseur ne connaît pas est un drapeau qui n'existe pas."""
    source = (ROOT / "rag" / "ingestion" / "batch_driver.py").read_text(encoding="utf-8")
    assert '"--sans-graphe"' in source
    assert "SANS_GRAPHE = arguments.sans_graphe" in source
    # et il s'annonce : un lot dont le graphe est sauté ne doit pas pouvoir passer inaperçu
    assert "ATTENTION --sans-graphe" in source


def test_sans_le_drapeau_l_etape_tente_bien_l_extraction(doc, monkeypatch):
    """Le pendant du test 2 : le chemin par défaut appelle ``courir``, il n'est pas court-circuité."""
    appels = []

    def courir(commande, journal):
        appels.append(commande)
        return 1, "panne simulée", 0.0            # code != 0 : l'étape échoue, c'est voulu

    monkeypatch.setattr(batch_driver, "courir", courir)
    monkeypatch.setattr(batch_driver, "SANS_GRAPHE", False)

    ok, message = batch_driver.etape_graphe(doc, ROOT / "journal-inexistant.log", False)
    assert ok is False and "extract_entities" in message
    assert len(appels) == 1, "le chemin par défaut doit lancer l'extraction"
