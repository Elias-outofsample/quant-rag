"""Gardes de la coupe sûre : ce que le serveur promet de ne jamais casser.

Les fixtures viennent de six passages **réels** du corpus servi ``5530cba145``, choisis parce
qu'une troncature brute y casse quelque chose. Un test sur des chaînes inventées prouverait
que l'algorithme marche sur des chaînes inventées.

Le corpus n'étant pas redistribuable, ``fixtures-contrat.json`` en est ici un **brouillage
structurel** : chaque lettre remplacée par une lettre de même casse, chaque chiffre par un
chiffre ; blancs, ponctuation, ``$``, ``\\``, ``|``, retours à la ligne et balises
``<sup>``/``<sub>`` gardent leur position. Toutes les décisions du contrat — longueur rendue,
reste annoncé, défauts de la coupe brute, drapeaux de qualité, à six plafonds — ont été
vérifiées identiques sur les originaux et sur leur brouillage.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

BENCH = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BENCH))
sys.path.insert(0, str(BENCH.parent))

import pytest

import contrat

FIXTURES = json.loads((Path(__file__).parent / "fixtures-contrat.json").read_text(encoding="utf-8"))
_BLOC = re.compile(r"\$\$")


def fautes(rendu: str, source: str) -> list[str]:
    """Ce que la coupe a cassé, jugé sur le **texte rendu**, pas sur la position visée."""
    position = len(rendu)
    trouvees = []
    if len(_BLOC.findall(rendu)) % 2 == 1:
        trouvees.append("bloc $$ ouvert")
    if not contrat.math_desequilibree(source) and contrat._dollars_simples(rendu) % 2 == 1:
        trouvees.append("formule $ ouverte")
    if position < len(source) and contrat._dans_un_mot(source, position):
        trouvees.append("mot coupé")
    if position < len(source) and contrat._dans_une_ligne_de_tableau(source, position):
        trouvees.append("ligne de tableau coupée")
    return trouvees


# --------------------------------------------------------------------------- la promesse

@pytest.mark.parametrize("nom", sorted(FIXTURES))
def test_aucune_coupe_interdite_sur_les_fixtures_reelles(nom):
    """La promesse du contrat, sur les six passages qu'une coupe brute casse."""
    cas = FIXTURES[nom]
    source = cas["texte"]
    rendu, _ = contrat.couper(source, cas["plafond"])
    assert fautes(rendu, source) == [], f"{nom} ({cas['chunk_id']}) : {fautes(rendu, source)}"


@pytest.mark.parametrize("nom", sorted(FIXTURES))
def test_la_coupe_brute_cassait_bien_quelque_chose(nom):
    """Le témoin. Sans lui, les fixtures pourraient être devenues inoffensives sans qu'on le voie."""
    cas = FIXTURES[nom]
    source, plafond = cas["texte"], cas["plafond"]
    if len(source) <= plafond:
        pytest.skip("fixture plus courte que son plafond")
    assert fautes(source[:plafond], source) != [], (
        f"{nom} : la troncature brute ne casse plus rien — la fixture ne prouve plus rien.")


def test_un_texte_plus_court_que_le_plafond_n_est_pas_touche():
    texte = "Une phrase courte, avec $x = 1$ et rien d'autre."
    assert contrat.couper(texte, 10_000) == (texte, 0)
    assert contrat.couper_passage(texte, 10_000) == texte


def test_le_marqueur_dit_ce_qui_reste_et_ou_le_lire():
    source = FIXTURES["mot"]["texte"]
    plafond = FIXTURES["mot"]["plafond"]
    rendu = contrat.couper_passage(source, plafond, chunk_id="chunk-573989e463ab28cb")
    coupe, restants = contrat.couper(source, plafond)
    assert restants > 0
    assert f"{restants} caractères restants" in rendu
    assert 'get_passage("chunk-573989e463ab28cb")' in rendu
    assert rendu.startswith(coupe)


def test_sans_chunk_id_le_marqueur_ne_promet_pas_de_suite():
    source = FIXTURES["mot"]["texte"]
    rendu = contrat.couper_passage(source, FIXTURES["mot"]["plafond"])
    assert "caractères restants" in rendu
    assert "get_passage" not in rendu


# --------------------------------------------------------------------------- les cas limites

def test_un_balisage_impair_est_tolere_et_non_masque():
    """127 passages sur 26 120 portent un ``$`` impair — des symboles monétaires, pas des
    formules. On ne peut pas protéger un balisage mal formé : on le déclare."""
    cas = FIXTURES["impair"]
    source = cas["texte"]
    assert contrat.math_desequilibree(source)
    rendu, restants = contrat.couper(source, cas["plafond"])
    assert restants > 0, "la fixture doit être plus longue que son plafond"
    assert rendu, "un texte impair doit quand même rendre du texte"
    # Le drapeau porte sur le **texte servi**, celui que l'appelant a sous les yeux : le ``$``
    # orphelin de la source peut tomber après la coupe, et le rendu être équilibré. C'est la
    # source qui est déclarée douteuse, et c'est elle qu'on interroge ici.
    assert contrat.drapeaux_qualite(source, tronque=True)["math_unbalanced"] is True
    assert contrat.drapeaux_qualite(rendu, tronque=True)["truncated"] is True


def test_le_plus_long_passage_du_corpus_tient_dans_la_fenetre_servie():
    """``chunk-004c25a877336fd7`` : 9 384 caractères, un **tableau**. C'est le seul passage du
    corpus au-dessus de 6 000, et c'est précisément le matériau que ce module protège.
    Le plafond de production doit le laisser passer entier."""
    assert FIXTURES["tableau_le_plus_long"]["longueur"] == 9384
    assert contrat.PASSAGE_CHARACTERS >= 9384


def test_couper_un_tableau_se_fait_entre_deux_lignes():
    cas = FIXTURES["tableau_le_plus_long"]
    rendu, restants = contrat.couper(cas["texte"], 2000)
    assert restants > 0
    assert fautes(rendu, cas["texte"]) == []
    derniere = rendu.rstrip().splitlines()[-1].lstrip()
    assert not derniere.startswith("|") or derniere.endswith("|")


def test_un_texte_vide_ou_nul_ne_casse_rien():
    assert contrat.couper("", 100) == ("", 0)
    assert contrat.couper_passage(None) == ""


def test_la_coupe_recule_mais_ne_rend_jamais_rien():
    """Le pire recul mesuré sur le corpus est de 2 295 c. pour un plafond de 2 500. Il reste du
    texte, et le marqueur dit où lire la suite — c'est le contrat, pas un défaut."""
    for nom, cas in FIXTURES.items():
        rendu, _ = contrat.couper(cas["texte"], cas["plafond"])
        assert rendu.strip(), f"{nom} : la coupe a tout mangé"


# --------------------------------------------------------------------------- l'aperçu

def test_l_apercu_de_timeline_est_court_et_se_declare():
    source = FIXTURES["bloc"]["texte"]
    rendu = contrat.couper_apercu(source, 300)
    assert len(rendu) <= 301
    assert rendu.endswith("…")
    assert "\n" not in rendu


def test_l_apercu_ne_coupe_pas_un_mot():
    source = " ".join(FIXTURES["mot"]["texte"].split())
    rendu = contrat.couper_apercu(source, 300)
    assert not contrat._dans_un_mot(source, len(rendu.rstrip("…").rstrip()))


# --------------------------------------------------------------------------- les drapeaux

def test_les_drapeaux_sont_deterministes_et_muets_quand_tout_va_bien():
    propre = "Une phrase sans mathématiques ni balise."
    drapeaux = contrat.drapeaux_qualite(propre)
    assert drapeaux == {"has_math": False, "math_unbalanced": False, "has_control_chars": False,
                        "has_html_tags": False, "has_table": False, "truncated": False}
    assert contrat.ligne_qualite(drapeaux) == ""


def test_la_ligne_de_qualite_nomme_ce_qui_cloche():
    drapeaux = contrat.drapeaux_qualite("x <sup>2</sup> et $a", tronque=True)
    ligne = contrat.ligne_qualite(drapeaux)
    assert "has_html_tags" in ligne and "truncated" in ligne and "math_unbalanced" in ligne
