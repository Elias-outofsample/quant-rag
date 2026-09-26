"""La transformation de `latex-recolle` doit **réparer une formule sans jamais abîmer un texte**.

Ces tests ne vérifient pas « les cas faciles » : chacun correspond à un défaut qui a été
mesuré sur le corpus avant d'être corrigé, ou à une corruption que la transformation
pourrait commettre et que les invariants interdisent.

- `\\ P` -> `\\P` et `\\ {` -> `\\{` : **4 022 des 147 989 régions** du corpus changeaient de
  suite de jetons LaTeX sous le recollage atome-par-atome de la sonde `e11eac6`.
- `$0.02 a day, and …$` : le corpus est financier, `\\$[^$\\n]+\\$` apparie deux montants
  avec la prose entre eux — **500 passages** en portent au moins un.
- `\\begin{array}` : sans retrait des noms d'environnements, **378 régions** de matrices
  sont prises pour de la prose et refusées.

    .venv/bin/python -m pytest rag/benchmark/tests/test_recollage.py -q
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

BENCHMARK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCHMARK))
sys.path.insert(0, str(BENCHMARK.parent))

import recollage as R  # noqa: E402


# --------------------------------------------------------------------------- le cas canonique

def test_exemple_canonique_de_latex_norme():
    """L'exemple même de `latex_norme.py`, qui motive tout le lot."""
    eclate = (r"V _ { i , t } ^ { \mathrm { b i d } } = \frac { 1 } { N } "
              r"\sum _ { s \in t } V _ { i , s }")
    assert R.recoller(eclate) == r"V_{i,t}^{\mathrm{bid}}=\frac{1}{N}\sum_{s\in t}V_{i,s}"


def test_une_commande_nommee_ne_se_prolonge_pas():
    """``\\in`` suivi de ``t`` ne doit jamais devenir ``\\int`` — une autre commande."""
    assert R.recoller(r"\in t") == r"\in t"
    assert R.jetons(R.recoller(r"\in t")) == [r"\in", "t"]


# --------------------------------------------------------------------------- ce que la sonde cassait

def test_une_espace_de_controle_garde_son_espace():
    """``\\ `` est l'espace de contrôle ; la coller au jeton suivant invente ``\\(`` ou ``\\P``."""
    for avant in (r"0 \ P", r"\ { \mathcal E }", r"} \ ( k _ { 1 } = 5 0 )"):
        apres = R.recoller(avant)
        assert R.jetons(apres) == R.jetons(avant), avant


def test_deux_contre_obliques_restent_un_retour_a_la_ligne():
    """``\\\\`` est un retour à la ligne LaTeX : il ne doit pas se desserrer en ``\\ \\``."""
    avant = r"{ a } \\ { b }"
    assert R.jetons(R.recoller(avant)) == R.jetons(avant)
    assert r"\\" in R.recoller(avant)


# --------------------------------------------------------------------------- la garde

def test_un_montant_en_dollars_nest_pas_une_formule():
    region = r"$0.02 a day, and a put writer would theoretically make \$"
    assert not R.est_mathematique(region)
    assert R.recoller_passage(f"Il gagne {region} de plus.") == f"Il gagne {region} de plus."


def test_une_matrice_est_une_formule_malgre_le_nom_de_son_environnement():
    """« array » est de la syntaxe, pas de la prose — 378 régions du corpus en dépendent."""
    region = r"$\begin{array} { r } { a _ { 1 } = \frac { v } { \Gamma ( \alpha ) } } \end{array}$"
    assert R.mots_de_prose(region) == []
    assert R.est_mathematique(region)
    assert R.recoller_passage(region) == r"$\begin{array}{r}{a_{1}=\frac{v}{\Gamma(\alpha)}}\end{array}$"


def test_une_region_sans_signature_latex_est_refusee():
    for region in ("$12 per trade for market orders $", "$N = 1$", "$5 billion$"):
        assert not R.est_mathematique(region), region


def test_une_formule_dont_lamont_a_perdu_la_contre_oblique_est_refusee():
    """`{ lefteqn { … } }` : un dégât d'OCR, pas une formule. 45 régions du corpus."""
    assert not R.est_mathematique(r"$\begin{array} { r l } { lefteqn { \frac { 1 } { 2 } } } \end{array}$")


def test_la_prose_autour_dune_formule_nest_pas_touchee():
    passage = "Le rendement est donné par $r _ { t } = \\mu$ dans ce modèle, à l'équilibre."
    assert R.recoller_passage(passage) == (
        "Le rendement est donné par $r_{t}=\\mu$ dans ce modèle, à l'équilibre.")


def test_un_bloc_affiche_nest_pas_coupe_en_deux_par_la_regex_en_ligne():
    passage = "$$\na = 1\n$$ et $$\nb = 2\n$$"
    regions = [m.group() for m in R.MATHS.finditer(passage)]
    assert regions == ["$$\na = 1\n$$", "$$\nb = 2\n$$"]


# --------------------------------------------------------------------------- les invariants

def test_les_trois_invariants_tiennent_sur_un_melange_de_prose_et_de_formules():
    passage = ("Le prix passe de $1.50 a share to $3.00 a share, and the payoff is "
               "$$\nV _ { i , t } ^ { \\mathrm { b i d } } = \\frac { 1 } { N } \\sum _ { s \\in t } V _ { i , s }\n$$ "
               "ce qui vaut $\\sigma ^ { 2 } \\mathbb { I }$ à l'équilibre.")
    apres = R.recoller_passage(passage)
    v = R.violations(passage, apres)
    assert v["I1"] is False and v["I2"] == [] and v["I3"] == 0
    assert "bid" in apres and "a share to" in apres     # réparé d'un côté, intact de l'autre


def test_I2_voit_la_prose_collee_que_I1_ne_voit_pas():
    """Une corruption qui ne retire que du blanc passe I1 ; c'est I2 qui l'attrape."""
    avant = "and a put writer"
    apres = "andaputwriter"
    assert R.sans_blanc(avant) == R.sans_blanc(apres)       # I1 ne voit rien
    assert R.violations(avant, apres)["I2"]                  # I2 voit la perte


def test_creer_un_mot_est_permis_en_detruire_un_ne_lest_pas():
    assert R.violations("$\\mathrm { b i d }$", "$\\mathrm{bid}$")["I2"] == []


# --------------------------------------------------------------------------- le placebo

def test_le_placebo_est_apparie_en_longueur_au_candidat():
    rng = random.Random(20260910)
    passage = ("Un texte avec de la prose et une formule "
               "$$\n\\alpha _ { t } = \\beta _ { t } + \\gamma\n$$ puis encore de la prose ici.")
    candidat = R.recoller_passage(passage)
    placebo = R.placebo_passage(passage, len(candidat), rng)
    assert len(placebo) == len(candidat) < len(passage)


def test_le_placebo_ne_recolle_aucune_paire_latex():
    """Il retire autant de blancs, ailleurs : la formule reste éclatée."""
    rng = random.Random(1)
    passage = "$$\n\\alpha _ { t } = \\beta _ { t }\n$$ " + "mot " * 60
    candidat = R.recoller_passage(passage)
    placebo = R.placebo_passage(passage, len(candidat), rng)
    assert "\\alpha_{t}" not in placebo
    assert "\\alpha_{t}" in candidat


def test_le_placebo_ne_touche_que_le_blanc():
    rng = random.Random(2)
    passage = "$$\n\\alpha _ { t } = 1\n$$ " + "mot " * 40
    placebo = R.placebo_passage(passage, len(R.recoller_passage(passage)), rng)
    assert R.sans_blanc(placebo) == R.sans_blanc(passage)


def test_le_placebo_est_reproductible_a_graine_egale():
    passage = "$$\n\\alpha _ { t } = 1\n$$ " + "mot " * 40
    cible = len(R.recoller_passage(passage))
    assert (R.placebo_passage(passage, cible, random.Random(7))
            == R.placebo_passage(passage, cible, random.Random(7)))


# --------------------------------------------------------------------------- idempotence

def test_le_recollage_est_idempotent():
    """Appliqué deux fois, il ne retire rien de plus — sinon la transformation ne serait
    pas une fonction de la chaîne mais de son histoire."""
    passage = ("$$\nV _ { i , t } ^ { \\mathrm { b i d } } = \\frac { 1 } { N }\n$$ et "
               "$x _ { t } \\in \\mathcal { F }$ avec $1.50 a share to $3.00 a share$")
    une = R.recoller_passage(passage)
    assert R.recoller_passage(une) == une
