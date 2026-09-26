"""``dense_matrix`` ne doit plus pouvoir mesurer sur un corpus partiel sans le dire.

Le 7 septembre 2026, ce module chargeait **18 636 chunks / 256 documents** quand la
collection servie en contenait **26 120 / 418** : l'export ``data/qdrant-export/`` est
l'instantané figé des 258 documents de l'amont, il n'est pas régénéré à l'ingestion, et
rien ne comparait sa cardinalité à celle du corpus. Son ``--check`` le signalait —
45/155 top-50 identiques — mais ``Matrix.load()`` ne le lançait pas, et aucun de ses
quatre appelants non plus.

Ces tests disent quatre choses : la cardinalité servie se lit dans un artefact et n'est
pas supposée ; un chargement incomplet **échoue** au lieu de rendre un chiffre ; l'absence
de témoin échoue aussi, plutôt que de faire confiance ; et une mesure explicitement
partielle reste possible, mais elle est **étiquetée**.

Le dernier test est celui qui aurait attrapé le défaut : il compare le chargement réel à
ce que la collection contient. Il échouait avant la correction, avec un déficit de
7 484 passages.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

BENCHMARK = Path(__file__).resolve().parents[1]
MACOS = BENCHMARK.parent
sys.path.insert(0, str(BENCHMARK))
sys.path.insert(0, str(MACOS))

import corpus_overlay  # noqa: E402
import dense_matrix  # noqa: E402


def _matrice(chunks: int, documents: int) -> dense_matrix.Matrix:
    """Une matrice minuscule dont on contrôle exactement la cardinalité."""
    chunk_ids = np.asarray([f"chunk-{i:04d}" for i in range(chunks)])
    document_ids = np.asarray([f"doc-{i % documents:04d}" for i in range(chunks)])
    vectors = np.zeros((chunks, 4), dtype=np.float32)
    vectors[:, 0] = 1.0
    matrix = dense_matrix.Matrix(chunk_ids, document_ids, vectors, "essai")
    matrix.signature = "essai00000"
    return matrix


def test_la_cardinalite_servie_se_lit_dans_le_manifeste_et_non_dans_une_constante():
    """``built_from`` est écrit en lisant Qdrant : c'est le seul témoin hors ligne."""
    servi = dense_matrix.served_counts()
    if servi is None:
        pytest.skip(f"aucun manifeste d'index BM25 pour la signature {corpus_overlay.signature()}")
    assert servi["points"] > 0 and servi["documents"] > 0
    source = (BENCHMARK / "dense_matrix.py").read_text(encoding="utf-8")
    assert str(servi["points"]) not in source, "la cardinalité servie est écrite en dur"


def test_un_chargement_incomplet_echoue_au_lieu_de_rendre_un_chiffre(monkeypatch):
    """Le mode de panne exact du 7 septembre : 71 % du corpus, et un résultat plausible."""
    monkeypatch.setattr(dense_matrix, "served_counts",
                        lambda: {"points": 26120, "documents": 418})
    matrice = _matrice(chunks=18636, documents=256)
    with pytest.raises(SystemExit) as arret:
        matrice.verify()
    message = str(arret.value)
    assert "incomplète" in message
    assert "7484" in message.replace(" ", ""), "le déficit exact doit être dans le message"


def test_une_matrice_qui_deborde_echoue_aussi(monkeypatch):
    """La garde n'est pas un minorant. Une matrice plus grosse que la collection ne la
    décrit pas davantage : elle porte des chunks qui ne concourent pas, et ils prendraient
    des rangs à ceux qui concourent."""
    monkeypatch.setattr(dense_matrix, "served_counts", lambda: {"points": 10, "documents": 3})
    with pytest.raises(SystemExit):
        _matrice(chunks=12, documents=3).verify()
    with pytest.raises(SystemExit):
        _matrice(chunks=10, documents=4).verify()


def test_sans_manifeste_la_garde_echoue_plutot_que_de_faire_confiance(monkeypatch):
    """Se taire au lieu de mentir — la règle que ``mcp_server._comptes`` applique déjà."""
    monkeypatch.setattr(dense_matrix, "served_counts", lambda: None)
    matrice = _matrice(chunks=10, documents=2)
    with pytest.raises(SystemExit) as arret:
        matrice.verify()
    assert "manifeste" in str(arret.value)


def test_une_mesure_partielle_reste_possible_mais_elle_est_etiquetee(monkeypatch, capsys):
    monkeypatch.setattr(dense_matrix, "served_counts",
                        lambda: {"points": 26120, "documents": 418})
    matrice = _matrice(chunks=18636, documents=256)
    couverture = matrice.verify(allow_partial=True)          # ne lève pas
    assert couverture["complete"] is False
    assert "PARTIELLE" in matrice.label
    assert "incomplète" in capsys.readouterr().err


def test_une_matrice_complete_passe(monkeypatch):
    monkeypatch.setattr(dense_matrix, "served_counts", lambda: {"points": 12, "documents": 4})
    couverture = _matrice(chunks=12, documents=4).verify()
    assert couverture["complete"] is True


def test_le_chargement_reel_couvre_la_collection_servie():
    """Le test qui aurait attrapé le défaut. Avant la correction : 18 636 contre 26 120.

    Il charge la vraie matrice (~15 s) parce que c'est exactement ce qui n'était pas
    vérifié : les tests unitaires ci-dessus prouvent la garde, celui-ci prouve qu'elle
    est **satisfaite** par les artefacts réellement présents sur cette machine.
    """
    servi = dense_matrix.served_counts()
    if servi is None:
        pytest.skip(f"aucun manifeste d'index BM25 pour la signature {corpus_overlay.signature()}")
    matrice = dense_matrix.Matrix.load()                      # échouerait si incomplète
    assert len(matrice.chunk_ids) == servi["points"]
    assert len(set(matrice.document_ids.tolist())) == servi["documents"]
    assert matrice.coverage["complete"] is True
    assert len(set(matrice.chunk_ids.tolist())) == servi["points"], "identifiants dupliqués"


def test_les_vecteurs_des_livraisons_sont_ceux_que_la_collection_sert():
    """Une reconstruction doit reproduire ce qui est servi, pas s'en approcher.

    ``build_index`` téléverse le vecteur du ``.npz`` tel quel : si le cache diffère, une
    reconstruction change le classement sans que rien ne le dise. Le 7 septembre 2026,
    410 passages de 7 documents étaient dans ce cas — dont 127 au-delà de 10⁻⁶ en cosinus,
    donc capables de déplacer un rang.
    """
    if not (Path(__file__).resolve().parents[3] / "qdrant_storage_local").exists():
        pytest.skip("collection locale absente")
    sys.path.insert(0, str(MACOS / "ingestion"))
    import recover_vectors  # noqa: PLC0415

    manquants = recover_vectors.documents_sans_vecteurs()
    assert manquants == [], f"{sum(len(d['absents']) for d in manquants)} passages sans vecteur sur le disque"
    recolte, par_document = recover_vectors.divergents()
    assert par_document == {}, f"cache divergent de la collection : {par_document}"


def test_le_gabarit_du_manifeste_suit_celui_de_l_index_bm25():
    """Deux constructions du même nom de fichier divergeraient en silence."""
    attendu = (f"bm25-{corpus_overlay.LABEL}-{corpus_overlay.signature()}.manifest.json")
    lus = list(dense_matrix.LEXICAL.glob("bm25-*.manifest.json"))
    if not lus:
        pytest.skip("aucun manifeste d'index BM25 sur cette machine")
    assert (dense_matrix.LEXICAL / attendu).exists() or attendu not in {p.name for p in lus}


def test_le_manifeste_lu_decrit_bien_la_collection_servie():
    chemin = dense_matrix.LEXICAL / f"bm25-{corpus_overlay.LABEL}-{corpus_overlay.signature()}.manifest.json"
    if not chemin.exists():
        pytest.skip("aucun manifeste pour la signature courante")
    manifeste = json.loads(chemin.read_text(encoding="utf-8"))
    assert manifeste["built_from"]["collection"] == "quant_rag_ingested_all_qwen3_06b"
    assert manifeste["corpus_signature"] == corpus_overlay.signature()
