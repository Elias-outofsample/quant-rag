"""Le chemin de mesure du lot `latex-recolle`, éprouvé sur une matrice synthétique.

Ces tests existent pour une raison précise : les trois bras coûtent ~3 h de MPS, et
découvrir une erreur de signe, de dtype ou de projection **après** cet encodage serait trois
heures perdues. Ils n'appellent ni le modèle, ni Qdrant, ni la collection servie — ils
construisent une `Matrix` de quatre passages et vérifient que chaque brique rend ce qu'elle
promet.

    .venv/bin/python -m pytest rag/benchmark/tests/test_bancs_latex.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

BENCHMARK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCHMARK))
sys.path.insert(0, str(BENCHMARK.parent))

import bancs_latex as B  # noqa: E402
from dense_matrix import Matrix  # noqa: E402


def matrice_jouet() -> Matrix:
    """Quatre passages dans deux documents, vecteurs unitaires en dimension 3."""
    vecteurs = np.asarray([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0.8, 0.6, 0]], dtype=np.float32)
    m = Matrix(np.asarray(["a", "b", "c", "d"]), np.asarray(["doc1", "doc1", "doc2", "doc2"]),
               vecteurs, "jouet", np.asarray(["", "", "", ""]))
    m.signature = "jouet"
    return m


# --------------------------------------------------------------------------- patcher

def test_patcher_normalise_et_rend_le_cosinus_contre_la_reference():
    m = matrice_jouet()
    patchee, absents, cos = B.patcher(m, {"a": np.asarray([2, 0, 0], dtype=np.float32)}, "test")
    assert absents == []
    assert np.allclose(patchee.vectors[0], [1, 0, 0])          # renormalisé, pas de norme 2
    assert cos.shape == (1,) and cos[0] == pytest.approx(1.0)   # même direction -> cosinus 1
    assert np.allclose(patchee.vectors[1:], m.vectors[1:])      # les autres sont intacts


def test_patcher_signale_un_chunk_absent_au_lieu_de_le_perdre():
    m = matrice_jouet()
    _, absents, cos = B.patcher(m, {"inconnu": np.asarray([1, 0, 0], dtype=np.float32)}, "test")
    assert absents == ["inconnu"] and len(cos) == 0


def test_patcher_ne_modifie_pas_la_matrice_de_reference():
    """Un bras qui abîmerait la référence ferait dériver tous les bras suivants."""
    m = matrice_jouet()
    avant = m.vectors.copy()
    B.patcher(m, {"a": np.asarray([0, 0, 1], dtype=np.float32)}, "test")
    assert np.array_equal(m.vectors, avant)


# --------------------------------------------------------------------------- or servi et rang

def test_rang_de_l_or_rend_le_premier_rang_et_None_hors_pool():
    pool = [{"chunk_id": "b"}, {"chunk_id": "a"}, {"chunk_id": "c"}]
    assert B.rang_de_l_or(pool, {"a"}) == 2
    assert B.rang_de_l_or(pool, {"a", "c"}) == 2        # le premier rencontré
    assert B.rang_de_l_or(pool, {"d"}) is None


def test_servi_applique_bien_le_plancher_de_caracteres_de_production():
    court = {"chunk_id": "a", "document_id": "doc1", "text": "trop court"}
    long = {"chunk_id": "b", "document_id": "doc1", "text": "x" * 300}
    assert B.servi([court, long]) == {"b"}


# --------------------------------------------------------------------------- gains et pertes

def test_gagnes_perdus_separe_les_deux_sens_et_filtre_par_famille():
    ref = {"v1/q1": False, "v1/q2": True, "v4_formula/q3": False, "v4_formula/q4": True}
    bras = {"v1/q1": True, "v1/q2": False, "v4_formula/q3": True, "v4_formula/q4": True}
    tous = B.gagnes_perdus(ref, bras)
    assert (tous["n_gagnes"], tous["n_perdus"], tous["net"]) == (2, 1, 1)
    formula = B.gagnes_perdus(ref, bras, prefixe="v4_formula")
    assert (formula["n"], formula["n_gagnes"], formula["n_perdus"], formula["net"]) == (2, 1, 0, 1)
    v1 = B.gagnes_perdus(ref, bras, prefixe="v1")
    assert v1["net"] == 0 and v1["n_gagnes"] == 1 and v1["n_perdus"] == 1


def test_le_prefixe_de_famille_ne_mord_pas_sur_une_famille_voisine():
    """`v4_formula` ne doit pas capturer `v4_formula_bis` ni l'inverse — d'où le `/`."""
    ref = {"v4_formula/q1": False, "v4_formula_bis/q2": False}
    bras = {"v4_formula/q1": True, "v4_formula_bis/q2": True}
    assert B.gagnes_perdus(ref, bras, prefixe="v4_formula")["n"] == 1


# --------------------------------------------------------------------------- statistiques secondaires

def test_mcnemar_est_le_test_binomial_sur_les_discordances():
    assert B.mcnemar_exact(0, 0) == 1.0
    assert B.mcnemar_exact(5, 0) == pytest.approx(2 * 0.5 ** 5)
    assert B.mcnemar_exact(1, 1) == pytest.approx(1.0)
    assert B.mcnemar_exact(3, 0) == B.mcnemar_exact(0, 3)      # bilatéral, donc symétrique


def test_le_bootstrap_encadre_le_net_observe_et_est_reproductible():
    deltas = {f"q{i}": (1 if i < 8 else (-1 if i < 11 else 0)) for i in range(300)}
    a = B.bootstrap_net(deltas, tirages=2000)
    assert a["net"] == 5
    assert a["ic95"][0] <= 5 <= a["ic95"][1]
    assert a == B.bootstrap_net(deltas, tirages=2000)           # graine fixée


def test_le_contraste_est_bootstrape_apparie_pas_comme_deux_bras_independants():
    """Deux bras identiques doivent rendre un contraste nul **sans incertitude** : c'est la
    signature de l'appariement. Bootstrapés séparément, ils rendraient un intervalle large."""
    deltas = {f"q{i}": (1 if i < 20 else 0) for i in range(300)}
    apparie = B.bootstrap_contraste(deltas, deltas, tirages=500)
    assert apparie["contraste"] == 0 and apparie["ic95"] == [0, 0]


# --------------------------------------------------------------------------- bras de bruit

def test_le_bruit_isotrope_respecte_le_cosinus_demande():
    m = matrice_jouet()
    vus: list[float] = []

    def mesurer(bruitee: Matrix) -> tuple[int, int]:
        vus.extend(float(np.dot(m.vectors[i], bruitee.vectors[i])) for i in range(len(m.chunk_ids)))
        return 0, 0

    B.bras_bruit(m, ["a", "b"], np.asarray([0.99], dtype=np.float32), 3, mesurer)
    patches = [vus[i] for t in range(3) for i in (4 * t, 4 * t + 1)]
    intacts = [vus[i] for t in range(3) for i in (4 * t + 2, 4 * t + 3)]
    assert all(c == pytest.approx(0.99, abs=2e-3) for c in patches)
    assert all(c == pytest.approx(1.0, abs=1e-6) for c in intacts)   # hors population : intacts


def test_le_bruit_rend_le_taux_de_fausse_alarme_et_sa_distribution():
    m = matrice_jouet()
    nets = iter([0, 3, -4, 3, 1])

    def mesurer(_: Matrix) -> tuple[int, int]:
        return next(nets), 0

    r = B.bras_bruit(m, ["a"], np.asarray([0.999], dtype=np.float32), 5, mesurer)
    assert r["tous_les_nets"] == [0, 3, -4, 3, 1]
    assert r["taux_de_fausse_alarme_du_seuil_3"] == pytest.approx(2 / 5)
    assert r["taux_de_fausse_alarme_bilateral_3"] == pytest.approx(3 / 5)
    assert r["net"]["min"] == -4 and r["net"]["max"] == 3


def test_le_bruit_est_reproductible_a_graine_egale():
    m = matrice_jouet()
    vus: dict[int, list[float]] = {0: [], 1: []}
    for passage in (0, 1):
        def mesurer(bruitee: Matrix, p=passage) -> tuple[int, int]:
            vus[p].append(float(bruitee.vectors[0][0]))
            return 0, 0
        B.bras_bruit(m, ["a"], np.asarray([0.95], dtype=np.float32), 2, mesurer)
    assert vus[0] == vus[1]
