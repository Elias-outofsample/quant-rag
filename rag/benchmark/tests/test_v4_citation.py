"""La métrique de citation : ce qu'elle crédite, ce qu'elle refuse, et où elle s'arrête.

Cette métrique répond sans juge à la question que le produit promet : *les affirmations
chiffrées ou formulées sont-elles citées, et la citation pointe-t-elle un passage qui contient
réellement ce qui est affirmé ?*

Le test qui compte le plus est le dernier de la première section : **la vérification porte sur
la fenêtre servie**, 2 500 caractères. Créditer une citation contre le texte entier du chunk
créditerait ce que le générateur ne pouvait pas lire — et ferait mentir la métrique
exactement là où le banc vient d'être réparé (``4eee997``).
"""
from __future__ import annotations

import sys
from pathlib import Path

BENCH = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BENCH))
sys.path.insert(0, str(BENCH.parent))

import score_citation as sc  # noqa: E402


def passage(numero: int, texte: str) -> dict:
    return {"chunk_id": f"chunk-test-{numero}", "document_id": f"doc-test-{numero}",
            "text": texte}


CONTEXTE = [
    passage(1, "The realised Sharpe ratio of the strategy is 1.42 over the full sample, "
               "and the annualised turnover reaches 0.096 per month."),
    passage(2, "Transaction costs are modelled as a fixed 5 basis point charge. "
               "The maximum drawdown observed is 0.318."),
    passage(3, "We estimate the local volatility surface by the formula "
               "$$\\sigma _ { l o c } ^ { 2 } ( t , k ) = E [ V _ { t } | X _ { t } = k ]$$ "
               "for every maturity."),
]


# ------------------------------------------------------------------ marqueurs

def test_les_marqueurs_simples_sont_lus():
    assert sc.marqueurs("The value is 1.42 [1] and the cost is 5 bp [2].") == [1, 2]


def test_les_marqueurs_groupes_sont_lus():
    assert sc.marqueurs("Both sources agree [1, 3].") == [1, 3]


def test_les_marqueurs_accoles_sont_lus():
    assert sc.marqueurs("Both sources agree [1][3].") == [1, 3]


def test_une_reponse_sans_marqueur_n_en_rend_aucun():
    assert sc.marqueurs("The value is 1.42.") == []


# ------------------------------------------------------------------ classement des affirmations

def test_une_affirmation_appuyee_est_creditee():
    resultat = sc.verifier("The realised Sharpe ratio is 1.42 [1].", CONTEXTE)
    assert resultat["comptes"]["appuyee"] == 1
    assert resultat["precision_citations"] == 1.0


def test_un_chiffre_absent_de_tout_passage_est_absent():
    resultat = sc.verifier("The realised Sharpe ratio is 2.71 [1].", CONTEXTE)
    assert resultat["comptes"]["absente"] == 1
    assert resultat["precision_citations"] == 0.0


def test_citer_le_mauvais_passage_est_distingue_d_une_invention():
    """Le chiffre existe, mais pas là où la réponse dit qu'il est. Ce n'est pas la même faute."""
    resultat = sc.verifier("The maximum drawdown is 0.318 [1].", CONTEXTE)
    assert resultat["comptes"]["mauvais_passage"] == 1
    assert resultat["comptes"]["absente"] == 0


def test_une_affirmation_sans_citation_est_comptee_a_part():
    resultat = sc.verifier("The realised Sharpe ratio is 1.42.", CONTEXTE)
    assert resultat["comptes"]["sans_citation"] == 1
    assert resultat["taux_sans_citation"] == 1.0
    # Sans affirmation notée, la précision est indéfinie — pas zéro. Un zéro se moyennerait.
    assert resultat["precision_citations"] is None


def test_un_marqueur_hors_bornes_est_signale():
    resultat = sc.verifier("The ratio is 1.42 [7].", CONTEXTE)
    assert resultat["marqueurs_hors_bornes"] == [7]
    assert resultat["comptes"]["hors_bornes"] == 1


def test_chaque_affirmation_tombe_dans_exactement_une_categorie():
    texte = ("The Sharpe ratio is 1.42 [1]. The drawdown is 0.318 [1]. "
             "The alpha is 2.71 [2]. The turnover is 0.096. The beta is 3.30 [9].")
    resultat = sc.verifier(texte, CONTEXTE)
    assert sum(resultat["comptes"].values()) == resultat["n_affirmations"]
    assert set(resultat["comptes"]) == set(sc.CATEGORIES)


def test_la_verification_porte_sur_la_fenetre_servie_pas_sur_le_chunk_entier():
    """Le test qui garde la métrique honnête.

    Un chiffre placé au caractère 3 000 n'a **jamais** été montré au générateur : le créditer
    reviendrait à noter une citation contre un texte que le système n'a pas vu. C'est
    exactement le défaut que la réparation du banc (1 600 -> 2 500 caractères) a corrigé
    ailleurs ; il n'a pas à revenir ici.
    """
    long_passage = [passage(1, "Filler. " * 380 + " The hidden coefficient is 9.87.")]
    assert len(long_passage[0]["text"]) > 3000

    creditee = sc.verifier("The coefficient is 9.87 [1].", long_passage, fenetre=4000)
    assert creditee["comptes"]["appuyee"] == 1

    tronquee = sc.verifier("The coefficient is 9.87 [1].", long_passage, fenetre=2500)
    assert tronquee["comptes"]["appuyee"] == 0
    assert tronquee["comptes"]["absente"] == 1


def test_les_marqueurs_ne_sont_pas_pris_pour_des_affirmations():
    """Sans cette précaution, « [1] » serait une affirmation chiffrée dans chaque phrase."""
    resultat = sc.verifier("This follows from the cited work [1][2][3].", CONTEXTE)
    assert resultat["n_affirmations"] == 0


def test_une_formule_restituee_est_creditee():
    """Une expression mathématique est une affirmation vérifiable au même titre qu'un chiffre."""
    resultat = sc.verifier(
        "The local variance is $\\sigma_{loc}^2(t,k) = E[V_t | X_t = k]$ [3].", CONTEXTE)
    assert resultat["comptes"]["appuyee"] == 1


def test_une_formule_alteree_n_est_pas_creditee():
    resultat = sc.verifier(
        "The local variance is $\\sigma_{loc}^2(t,k) = E[W_t | X_t = k]$ [3].", CONTEXTE)
    assert resultat["comptes"]["appuyee"] == 0


def test_le_pourcentage_ne_se_confond_pas_avec_sa_fraction():
    contexte = [passage(1, "The hit rate is 84% across the sample.")]
    assert sc.verifier("The hit rate is 84% [1].", contexte)["comptes"]["appuyee"] == 1
    assert sc.verifier("The hit rate is 0.84 [1].", contexte)["comptes"]["appuyee"] == 0


def test_un_contexte_vide_ne_fait_pas_tomber_la_metrique():
    resultat = sc.verifier("The ratio is 1.42 [1].", [])
    assert resultat["n_passages"] == 0
    assert resultat["comptes"]["hors_bornes"] == 1


def test_une_reponse_vide_est_lisible():
    resultat = sc.verifier("", CONTEXTE)
    assert resultat["n_affirmations"] == 0
    assert resultat["precision_citations"] is None


def test_la_metrique_ne_fait_aucun_appel_reseau():
    assert sc.verifier("The ratio is 1.42 [1].", CONTEXTE)["appels_llm"] == 0


# ------------------------------------------------------------------ extraits littéraux

REFERENCE = ("We ﬁnd that the eﬀect of realised volatility on premiums is "
             "statistically reliable at the 5% level.")


def test_un_extrait_litteral_est_retrouve():
    lus = sc.extraits('The paper states that "the effect of realised volatility on premiums '
                      'is statistically reliable" here.')
    assert len(lus) == 1
    verdicts = sc.verifier_extraits(lus, REFERENCE)
    assert verdicts[0]["retrouve"] is True


def test_une_paraphrase_entre_guillemets_n_est_pas_retrouvee():
    lus = sc.extraits('The paper states that "realised volatility reliably moves the premium '
                      'at conventional levels" here.')
    verdicts = sc.verifier_extraits(lus, REFERENCE)
    assert verdicts[0]["retrouve"] is False


def test_les_ligatures_perdues_sont_absorbees():
    """24,2 % des chunks du corpus ont perdu leurs ligatures ﬁ/ﬀ. Ce n'est pas une différence."""
    assert sc.normaliser_texte("we ﬁnd the eﬀect") == sc.normaliser_texte("we find the effect")


def test_les_guillemets_et_tirets_sont_unifies():
    assert sc.normaliser_texte("l'écart — mesuré") == sc.normaliser_texte("l'écart - mesuré")


def test_les_balises_sup_et_sub_sont_retirees():
    """16,5 % des chunks portent des <sup>/<sub> que MinerU n'a pas convertis."""
    assert sc.normaliser_texte("R<sup>2</sup> of 0.9") == sc.normaliser_texte("R2 of 0.9")


def test_un_extrait_trop_court_n_est_pas_une_citation():
    """Trois mots entre guillemets sont un mot mis en relief, pas une citation vérifiable."""
    assert sc.extraits('the so-called "risk premium" of the asset') == []


def test_les_guillemets_courbes_et_francais_sont_lus():
    droits = sc.extraits('He wrote "the effect of realised volatility on premiums is real".')
    courbes = sc.extraits('He wrote “the effect of realised volatility on premiums is real”.')
    francais = sc.extraits('Il écrit « the effect of realised volatility on premiums is real ».')
    assert len(droits) == len(courbes) == len(francais) == 1
