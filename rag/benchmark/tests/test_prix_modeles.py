"""Le barème de prix, et le fait qu'il n'y en ait plus qu'un.

Ce fichier est né d'un conflit de fusion. ``prix-modeles.json`` a été écrit deux fois le
8 septembre 2026 — par ``contrat-de-sortie`` pour ``llm.py`` (clés ``entree``/``sortie``,
taux BCE daté) et par ``instrument-v4`` pour ``banc_v4.py`` (clés ``entree_par_million``/
``sortie_par_million``, 1 USD compté pour 1 EUR). Les deux fichiers étaient justes ; ensemble
ils étaient un piège, parce que **chaque lecteur ignore silencieusement le schéma de l'autre** :
``banc_v4.chiffrer`` lisait ses clés avec un défaut à ``0.0`` et aurait publié un coût de
0,0000 USD en affichant ``tarif_connu: true``.

Les tests ci-dessous tiennent les deux bouts : un test par lecteur, plus un test qui interdit
au second schéma de revenir.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

BENCH = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BENCH))

import banc_v4 as B  # noqa: E402
import llm  # noqa: E402

BAREME = BENCH / "prix-modeles.json"
TAUX = 1.1622  # BCE, 2026-09-05 — nombre de dollars pour un euro


def _brut() -> dict:
    return json.loads(BAREME.read_text(encoding="utf-8"))


# ------------------------------------------------------------------ le fichier lui-même

def test_la_fusion_n_a_perdu_aucun_des_deux_cotes():
    """Chaque côté du conflit avait des modèles que l'autre n'avait pas."""
    modeles = _brut()["modeles"]
    assert "ministral-8b-latest" in modeles, "modèle apporté par instrument-v4, perdu à la fusion"
    assert "gemini-3.8-flash" in modeles, "modèle apporté par contrat-de-sortie, perdu à la fusion"
    assert modeles["gemini-3.1-flash-lite"]["palier_gratuit_detail"]["plafond_journalier_requetes"] == 500
    assert _brut()["taux_usd_par_eur"]["valeur"] == TAUX


def test_il_ne_reste_qu_un_seul_schema():
    """Le retour de ``entree_par_million`` ferait revenir deux lecteurs qui s'ignorent.

    Le contrôle porte sur les **clés**, pas sur le texte : la note de fusion cite les anciens
    noms pour expliquer d'où l'on vient, et doit pouvoir le faire.
    """
    for nom, tarif in _brut()["modeles"].items():
        assert "entree" in tarif and "sortie" in tarif, f"{nom} n'a pas de tarif lisible"
        residus = [cle for cle in tarif if cle.endswith("_par_million")]
        assert not residus, f"{nom} porte encore l'ancien schéma : {residus}"


# ------------------------------------------------------------------ lecteur 1 : llm.py

def test_llm_chiffre_en_dollars_et_en_euros_au_taux_date():
    assert llm.cout_usd("mistral-small-latest", 1_000_000, 0) == 0.15
    assert llm.cout_usd("mistral-medium-latest", 1_000_000, 0) == 1.5
    assert llm.cout_eur("mistral-small-latest", 1_000_000, 0) == 0.15 / TAUX


def test_llm_rend_None_et_jamais_zero_quand_il_ne_sait_pas():
    """``None`` dit « je ne sais pas » ; ``0.0`` dirait « c'était gratuit »."""
    assert llm.cout_usd("modele-jamais-releve", 1_000_000, 1_000_000) is None
    assert llm.cout_eur("modele-jamais-releve", 1_000_000, 1_000_000) is None


def test_llm_rend_None_pour_un_modele_present_mais_sans_prix(monkeypatch):
    """Le cas exact que la fusion aurait créé : le modèle est là, ses clés de prix ne le sont pas."""
    monkeypatch.setattr(llm, "_prix", {"taux_usd_par_eur": {"valeur": TAUX},
                                       "modeles": {"modele-muet": {"role": "sans tarif"}}})
    assert llm.cout_usd("modele-muet", 1_000_000, 0) is None
    assert llm.cout_eur("modele-muet", 1_000_000, 0) is None


# ------------------------------------------------------------------ lecteur 2 : banc_v4.py

def test_le_banc_lit_le_meme_bareme_que_llm():
    """Un seul analyseur : le total du banc doit être le chiffre de ``llm``, pas un parallèle."""
    registre = {"mistral-medium-latest": {"appels": 3, "cache": 0,
                                          "entree": 2_000_000, "sortie": 100_000}}
    sortie = B.chiffrer(registre)
    attendu = llm.cout_usd("mistral-medium-latest", 2_000_000, 100_000)
    assert sortie["total_usd"] == round(attendu, 4)
    assert sortie["total_eur"] == round(attendu / TAUX, 4)
    assert sortie["total_eur_majore"] == sortie["total_usd"]


def test_le_banc_exclut_du_total_ce_qu_il_ne_sait_pas_chiffrer(monkeypatch):
    """Un total muet sur ce qu'il ignore a l'air d'un total — c'est le défaut qu'on ferme."""
    monkeypatch.setattr(llm, "_prix", {"taux_usd_par_eur": {"valeur": TAUX},
                                       "modeles": {"connu": {"entree": 1.0, "sortie": 1.0},
                                                   "muet": {"role": "sans tarif"}}})
    sortie = B.chiffrer({"connu": {"appels": 1, "cache": 0, "entree": 1_000_000, "sortie": 0},
                         "muet": {"appels": 1, "cache": 0, "entree": 9_000_000, "sortie": 0}})
    assert sortie["par_modele"]["muet"]["tarif_connu"] is False
    assert sortie["par_modele"]["muet"]["cout_usd"] is None
    assert sortie["modeles_sans_tarif"] == ["muet"]
    assert sortie["total_usd"] == 1.0, "les 9 M de jetons non tarifés ont été comptés zéro"
