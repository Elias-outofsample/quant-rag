"""Gardes de ``verify_citation`` : ce qu'il doit trouver, et ce qu'il doit refuser.

Le contrôle complet — 50 extraits réels, 50 altérés d'un mot — vit dans
``rag/benchmark/controle_citation.py`` et demande la collection Qdrant. Ces tests-ci ne
demandent que le disque, tournent en moins d'une seconde, et couvrent ce qui doit tenir à
chaque commit : la normalisation, la règle de décision, et le refus.

Les fixtures sont trois **documents réels** du corpus servi (``fixtures-citation.json``),
choisis petits pour que la lecture reste rapide.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

BENCH = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BENCH))
sys.path.insert(0, str(BENCH.parent))

import pytest

import ancrage
import citation

CAS = json.loads((Path(__file__).parent / "fixtures-citation.json").read_text(encoding="utf-8"))


@pytest.fixture(autouse=True)
def sans_repli_qdrant(monkeypatch):
    """Ces tests ne touchent **pas** Qdrant, et c'est une garde, pas une commodité.

    Le repli « texte servi » de ``verify_citation`` ouvre la collection, et le client embarqué
    est mono-processus : le laisser s'ouvrir ici prenait le verrou pour toute la session pytest
    et faisait échouer ``test_dense_matrix_couverture`` plus loin — un test vert transformé en
    rouge par un voisin, ce qui est le pire genre de dépendance entre tests.

    Le repli lui-même est couvert par ``rag/benchmark/controle_citation.py``, qui a la
    collection et qui mesure les 100 citations.
    """
    monkeypatch.setattr(citation, "_chercher_dans_le_servi", lambda *a, **k: None)


# --------------------------------------------------------------------------- normalisation

def test_la_normalisation_rend_une_table_d_origine_de_meme_longueur():
    """L'invariant sans lequel un offset rendu serait faux : un index par caractère produit."""
    for texte in ("Un texte simple.", "x<sup>2</sup> et  des   blancs", "« citation » — tiret",
                  "ﬁn de ligature", "", "   ", "a\n\nb"):
        normalise, origine = citation.normaliser(texte)
        assert len(normalise) == len(origine), repr(texte)
        assert all(0 <= i < max(len(texte), 1) for i in origine), repr(texte)


def test_la_normalisation_efface_ce_qui_ne_doit_pas_faire_echouer_une_citation():
    # La **balise** disparaît, pas son contenu : « x<sup>2</sup> » est « x2 », c'est-à-dire x²,
    # et l'exposant est de l'information. Effacer le 2 rendrait la citation fausse.
    assert citation.normaliser("x<sup>2</sup>y")[0] == "x2y"
    assert citation.normaliser("a   b")[0] == "a b"
    assert citation.normaliser("un—tiret")[0] == "un-tiret"
    assert citation.normaliser("l’apostrophe")[0] == "l'apostrophe"
    assert citation.normaliser("ﬁn")[0] == "fin"
    assert citation.normaliser("MAJUSCULES")[0] == "majuscules"


def test_la_normalisation_ne_laisse_ni_blanc_de_tete_ni_blanc_de_queue():
    normalise, origine = citation.normaliser("   du texte   ")
    assert normalise == "du texte"
    assert len(normalise) == len(origine)


# --------------------------------------------------------------------------- la règle de décision

@pytest.mark.parametrize("cas", CAS, ids=[c["document_id"] for c in CAS])
def test_une_citation_reelle_est_retrouvee_avec_ses_offsets(cas):
    verdict = citation.verify_citation(cas["document_id"], cas["citation"])
    assert verdict["trouve"] is True
    debut, fin = verdict["offsets"]
    assert 0 <= debut < fin <= cas["caracteres"]
    assert verdict["doc_text_sha256"]
    assert verdict["distance"] == 0


@pytest.mark.parametrize("cas", CAS, ids=[c["document_id"] for c in CAS])
def test_les_offsets_rendus_designent_bien_la_citation(cas):
    """La preuve que l'offset est **opposable** : on relit le document à cette position."""
    verdict = citation.verify_citation(cas["document_id"], cas["citation"])
    texte = ancrage.texte_canonique_du_document(cas["document_id"])
    debut, fin = verdict["offsets"]
    attendu, _ = citation.normaliser(cas["citation"])
    relu, _ = citation.normaliser(texte[debut:fin])
    assert relu == attendu


@pytest.mark.parametrize("cas", CAS, ids=[c["document_id"] for c in CAS])
def test_une_citation_alteree_d_un_mot_est_refusee(cas):
    """Le sens qui compte. Un vérificateur qui accepte une déformation garantit une fausseté."""
    mots = cas["citation"].split()
    mots[len(mots) // 2] = "kurtosis"
    verdict = citation.verify_citation(cas["document_id"], " ".join(mots))
    assert verdict["trouve"] is False
    assert verdict["plus_proche"], "un refus doit dire où la citation a dérivé"
    assert verdict["plus_proche"]["mots_differents"] >= 1


def test_une_citation_inventee_est_refusee_sans_voisinage_trompeur():
    cas = CAS[0]
    verdict = citation.verify_citation(
        cas["document_id"],
        "the estimator achieves quadratic convergence under bounded liquidity shocks")
    assert verdict["trouve"] is False
    proche = verdict.get("plus_proche")
    assert proche is None or proche["ressemblance"] < 0.9


def test_un_document_inconnu_le_dit_au_lieu_de_deviner():
    verdict = citation.verify_citation("doc-inexistant", "quoi que ce soit")
    assert verdict["trouve"] is False
    assert "inconnu" in verdict["raison"]


def test_une_citation_vide_est_refusee():
    assert citation.verify_citation(CAS[0]["document_id"], "")["trouve"] is False
    assert citation.verify_citation(CAS[0]["document_id"], "   ")["trouve"] is False


def test_le_lot_rend_autant_de_verdicts_que_de_demandes():
    demandes = [{"document_id": c["document_id"], "quote": c["citation"]} for c in CAS]
    demandes.append({"document_id": CAS[0]["document_id"], "quote": "phrase absente inventée ici"})
    verdicts = citation.verify_citations(demandes)
    assert len(verdicts) == len(demandes)
    assert [v["trouve"] for v in verdicts] == [True] * len(CAS) + [False]


# --------------------------------------------------------------------------- l'ancrage

def test_l_ancrage_couvre_les_documents_des_fixtures():
    index = ancrage.charger()
    assert index.get("format") == ancrage.FORMAT
    for cas in CAS:
        assert cas["document_id"] in index["documents"]


def test_la_fusion_des_intervalles_reunit_les_blocs_contigus():
    """Deux blocs voisins sont séparés par le séparateur, deux caractères, et rien d'autre."""
    assert ancrage.fusionner([["b1", 0, 10], ["b2", 12, 20]]) == [[0, 20]]
    assert ancrage.fusionner([["b1", 0, 10], ["b2", 40, 50]]) == [[0, 10], [40, 50]]
    assert ancrage.fusionner([]) == []


def test_la_fusion_ne_depend_pas_de_l_ordre_recu():
    assert ancrage.fusionner([["b2", 12, 20], ["b1", 0, 10]]) == [[0, 20]]
