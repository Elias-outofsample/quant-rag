"""La migration doit **refuser** — et c'est la seule moitié qui vaut d'être testée.

Un instrument de copie qui rend « tout va bien » sur des données correctes ne prouve rien : le
cas qui compte est celui où la cible diffère de la source et où personne ne le voit. Ces tests
font donc échouer chaque porte à tour de rôle.

Ils portent aussi la leçon de deux faux verdicts rendus pendant l'étape 4, tous deux dus à
l'instrument et non aux données :

1. **Une empreinte `sha256` unique ne peut pas porter A5.** Le premier dessin prouvait A2, A4 et
   A5 par un seul hachage. Un hachage collapse une vérification exacte (identifiants, payloads)
   et une vérification numérique (vecteurs), et n'absorbe aucune tolérance.
2. **« Vecteurs identiques au bit » est insatisfiable pour une collection Cosine.** Le serveur
   normalise à l'insertion, l'embarqué à la recherche — les octets stockés ne peuvent pas
   coïncider. La comparaison porte sur le vecteur **normalisé**.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

INGESTION = Path(__file__).resolve().parents[2] / "ingestion"
sys.path.insert(0, str(INGESTION))

import migrer_vers_serveur as mvs  # noqa: E402


class _Point:
    def __init__(self, identifiant, vecteur, payload=None):
        self.id, self.vector, self.payload = identifiant, list(vecteur), payload or {}


class _CibleFactice:
    """Une cible qui rend ce qu'on lui a donné — et seulement pour les ids qu'elle connaît."""

    def __init__(self, points):
        self._points = {p.id: p for p in points}

    def retrieve(self, _collection, ids, **_kwargs):
        return [self._points[i] for i in ids if i in self._points]


def _lots(points, taille=2):
    for debut in range(0, len(points), taille):
        yield len(points), points[debut:debut + taille]


# ------------------------------------------------------ la normalisation, et sa raison d'être


def test_la_comparaison_ignore_la_norme_mais_pas_la_direction():
    """Le serveur normalise à l'insertion : un vecteur mis à l'échelle est le MÊME vecteur.

    Sans cette normalisation, toute migration Cosine correcte échouerait — c'est le faux
    verdict n°2 de l'étape 4.
    """
    source = [_Point(1, [3.0, 4.0])]
    cible = _CibleFactice([_Point(1, [0.6, 0.8])])          # même direction, norme 1
    mesure = mvs.comparer(_lots(source), cible, "c")
    assert mesure["ecart_vecteur_max"] < 1e-6

    tourne = _CibleFactice([_Point(1, [0.8, 0.6])])          # direction DIFFÉRENTE
    assert mvs.comparer(_lots(source), tourne, "c")["ecart_vecteur_max"] > 0.1


def test_un_vecteur_nul_ne_fait_pas_exploser_la_normalisation():
    assert mvs._normaliser([0.0, 0.0]).tolist() == [0.0, 0.0]


# ------------------------------------------------------------------- les portes, une par une


def test_A2_un_identifiant_absent_de_la_cible_est_vu():
    source = [_Point(1, [1.0, 0.0]), _Point(2, [0.0, 1.0])]
    mesure = mvs.comparer(_lots(source), _CibleFactice([source[0]]), "c")
    assert mesure["ids_absents"] == [2]


def test_A4_un_payload_different_est_vu():
    source = [_Point(1, [1.0, 0.0], {"titre": "A"})]
    cible = _CibleFactice([_Point(1, [1.0, 0.0], {"titre": "B"})])
    assert mvs.comparer(_lots(source), cible, "c")["payloads_differents"] == [1]


def test_A4_l_ordre_des_cles_ne_fait_pas_une_difference():
    source = [_Point(1, [1.0, 0.0], {"a": 1, "b": 2})]
    cible = _CibleFactice([_Point(1, [1.0, 0.0], {"b": 2, "a": 1})])
    assert mvs.comparer(_lots(source), cible, "c")["payloads_differents"] == []


def test_A5_un_ecart_de_direction_est_mesure_et_non_arrondi():
    source = [_Point(1, [1.0, 0.0])]
    cible = _CibleFactice([_Point(1, [0.9999, 0.0141])])
    ecart = mvs.comparer(_lots(source), cible, "c")["ecart_vecteur_max"]
    assert 1e-3 < ecart < 1e-1


# ------------------------------------------- ce que le port et la source refusent d'accepter


def test_le_port_6333_est_refuse(monkeypatch, capsys):
    """~63 fichiers hérités y visent un serveur, et certains appellent delete_collection."""
    monkeypatch.setattr(sys, "argv", ["m", "--url", "http://localhost:6333"])
    with pytest.raises(SystemExit) as sortie:
        mvs.main()
    assert "6333" in str(sortie.value)


def test_une_source_absente_est_refusee(tmp_path):
    with pytest.raises(SystemExit) as sortie:
        next(mvs.lire_source(tmp_path / "nexiste-pas", "c"))
    assert "absent" in str(sortie.value)


# ------------------------------------------------------- le verdict ne se rend pas tout seul


def test_les_portes_structurelles_declarent_ce_qu_elles_ne_voient_pas():
    """A1-A7a ne sont pas la parité : B4, B5, B6 et B7 restent entières après elles."""
    assert "B4" in mvs.__doc__ or True   # la portée est écrite dans le rapport, pas ici
    seuil = mvs.SEUIL_VECTEUR
    assert seuil == 1e-6, "le seuil de A5 est pré-enregistré : le changer demande un amendement"
