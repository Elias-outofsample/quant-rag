"""La garde de réemploi de vecteurs mesurait l'erreur de norme, pas un écart angulaire.

Le défaut, mesuré le 10 septembre 2026 par le lot B (§14.3 de ``docs/STRATEGIE.md``)
-------------------------------------------------------------------------------------
``reemploi_candidat`` prouve qu'un vecteur réemployé est bien celui qu'on croit, en le
comparant au même texte ré-embarqué aujourd'hui. La comparaison était un produit scalaire
brut, en float32, **sans renormaliser**.

Or ``normalize_embeddings=True`` ne rend pas des vecteurs unitaires : sur le corpus servi
les normes vont de ``0,99975586`` à ``1,00047958``. Un produit scalaire brut entre deux
vecteurs identiques rend ``‖v‖²`` — soit un écart à 1 de **deux fois** l'erreur de norme.
Le chiffre mesuré : sur 14 597 vecteurs identiques au bit près, **3 625 tombaient sous
0,9999**.

La garde tenait, parce que ``COSINUS_REFUS`` vaut 0,99, vingt fois plus large que
l'artefact. Mais c'était une garde qui ne savait pas ce qu'elle mesurait : toute tentative
de la resserrer aurait fait refuser des vecteurs identiques. **Le seuil n'est pas touché
ici** — le resserrer est une décision de mesure, pas de clôture.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "rag" / "ingestion"))
sys.path.insert(0, str(ROOT / "rag"))

import reemploi_candidat  # noqa: E402

#: Les bornes réellement observées sur les vecteurs du corpus servi. Un test qui fabriquerait
#: des vecteurs parfaitement unitaires ne prouverait rien : le défaut vit exactement ici.
NORME_MIN, NORME_MAX = 0.99975586, 1.00047958


def _vecteurs_comme_le_modele_en_produit(n: int = 200, d: int = 1024) -> np.ndarray:
    """``n`` vecteurs float32 dont les normes couvrent la plage observée sur le corpus.

    Graine fixe : un test de garde qui échoue une fois sur dix n'est pas une garde.
    """
    rng = np.random.default_rng(20260910)
    brut = rng.standard_normal((n, d))
    unitaires = brut / np.linalg.norm(brut, axis=1, keepdims=True)
    normes = np.linspace(NORME_MIN, NORME_MAX, n)
    return (unitaires * normes[:, None]).astype(np.float32)


def test_deux_cents_vecteurs_compares_a_eux_memes_rendent_exactement_1():
    """Le contrat de la fonction : un vecteur comparé à lui-même a un cosinus de 1,0.

    Une mutation qui retire la renormalisation fait tomber ce test — c'est vérifié plus bas
    par ``test_sans_renormalisation_la_garde_mesure_autre_chose``, qui reproduit le calcul
    d'avant sur le même échantillon.
    """
    v = _vecteurs_comme_le_modele_en_produit()
    cos = reemploi_candidat.cosinus(v, v)
    assert cos.shape == (200,)
    assert cos.dtype == np.float64, "l'accumulation doit se faire en float64"
    assert cos.min() >= 1.0 - 1e-12, f"cosinus min {cos.min():.15f} — un vecteur n'est pas colinéaire à lui-même"
    assert cos.max() <= 1.0 + 1e-12


def test_sans_renormalisation_la_garde_mesure_autre_chose():
    """Le défaut, reproduit sur le même échantillon — et l'écart chiffré.

    C'est la mutation, écrite en clair : si ``cosinus()`` redevenait ce calcul-là, le test
    précédent tomberait. On mesure ici de combien.
    """
    v = _vecteurs_comme_le_modele_en_produit()
    ancien = (v.astype(np.float32) * v.astype(np.float32)).sum(axis=1)   # le calcul d'avant
    nouveau = reemploi_candidat.cosinus(v, v)

    assert ancien.min() < 1.0 - 1e-6, "l'échantillon doit reproduire le défaut, sinon il ne prouve rien"
    assert nouveau.min() > ancien.min()
    # ‖v‖² : l'écart à 1 vaut deux fois l'erreur de norme, au premier ordre
    assert ancien.min() == pytest.approx(NORME_MIN ** 2, abs=1e-6)
    assert ancien.max() == pytest.approx(NORME_MAX ** 2, abs=1e-6)


def test_le_seuil_de_refus_n_a_pas_bouge():
    """Corriger le calcul n'autorise pas à resserrer le seuil dans le même geste."""
    assert reemploi_candidat.COSINUS_REFUS == 0.99


def test_un_vrai_ecart_angulaire_reste_vu():
    """La correction ne doit pas rendre la garde aveugle : deux directions différentes
    doivent toujours donner un cosinus franchement inférieur à 1."""
    v = _vecteurs_comme_le_modele_en_produit(n=50)
    autre = _vecteurs_comme_le_modele_en_produit(n=50)[::-1]
    cos = reemploi_candidat.cosinus(v, autre)
    assert cos.max() < 0.5, "des directions tirées au hasard en dimension 1024 sont quasi orthogonales"


def test_un_vecteur_nul_ne_produit_pas_de_nan():
    """``min()`` sur un tableau qui contient un NaN rend NaN, et ``NaN < seuil`` est faux :
    la garde laisserait passer au lieu de refuser. Le cas est traité explicitement."""
    v = _vecteurs_comme_le_modele_en_produit(n=3)
    v[1] = 0.0
    cos = reemploi_candidat.cosinus(v, v)
    assert not np.isnan(cos).any()
    assert cos[1] == 0.0
    assert cos[0] >= 1.0 - 1e-12 and cos[2] >= 1.0 - 1e-12


def test_le_module_n_utilise_plus_le_produit_scalaire_brut():
    """Garde-fou de source : l'ancien calcul ne doit pas revenir par une autre porte."""
    source = (ROOT / "rag" / "ingestion" / "reemploi_candidat.py").read_text(encoding="utf-8")
    assert "cos = cosinus(calcules, stockes)" in source
    assert "(calcules * stockes).sum(axis=1)" not in source
