"""La parité de backend se prouve par groupes de score, jamais position à position.

Le mode local trie par ``np.argsort(scores)[::-1]`` sans clé de départage secondaire
(``local_collection.LocalCollection``) : deux chunks à score strictement égal sortent dans un ordre que
rien ne spécifie, et l'inversion du tri renverse en plus leur ordre relatif. Un serveur Qdrant
applique ses propres règles. Exiger l'égalité position à position reviendrait donc à exiger
d'un serveur qu'il reproduise un détail d'implémentation de numpy — un test qu'aucune migration
correcte ne pourrait passer, et qui échouerait pour la mauvaise raison.

La comparaison retenue découpe chaque classement en **groupes de score** et compare la suite
des ensembles de ``chunk_id``. Elle accepte une permutation à l'intérieur d'un groupe d'ex æquo,
et **rien d'autre**. Ces tests vérifient les deux moitiés de cette phrase : qu'elle accepte ce
qu'elle doit accepter, et surtout qu'elle refuse tout le reste — un vrai réordonnancement, une
question manquante, une question en trop, un chunk substitué à score identique.

Le recensement des ex æquo de la référence courante (`--egalites`) nomme **2** questions,
`v3/t03` et `v3/s33`. Un test le reproduit sur l'artefact réel s'il est présent : une référence
qui gagnerait des ex æquo sans qu'on le sache invaliderait le pré-enregistrement, qui les
énumère.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

BANC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BANC))

import parite_backend as pb  # noqa: E402


def _classement(paires):
    """[(chunk, score)] -> le format de la référence, document_id sans importance ici."""
    return [[c, "doc-x", s] for c, s in paires]


def _reference(classements):
    return {"signature_corpus": "test", "branche": "dense", "classements": classements}


# ------------------------------------------------------------------ groupes de score


def test_les_scores_distincts_font_des_groupes_distincts():
    groupes = pb._groupes(_classement([("a", 0.9), ("b", 0.8), ("c", 0.7)]), 0.0)
    assert [sorted(m) for _s, m in groupes] == [["a"], ["b"], ["c"]]


def test_les_ex_aequo_stricts_se_regroupent():
    groupes = pb._groupes(_classement([("a", 0.9), ("b", 0.9), ("c", 0.7)]), 0.0)
    assert [sorted(m) for _s, m in groupes] == [["a", "b"], ["c"]]


def test_le_groupe_se_referme_sur_son_ouverture_pas_sur_son_precedent():
    """Sans cette règle, une chaîne d'écarts minuscules fusionnerait des scores distincts.

    Trois scores espacés de 0,6 avec une tolérance de 1,0 : de proche en proche ils se
    toucheraient tous, alors que le premier et le dernier sont à 1,2 l'un de l'autre.
    """
    groupes = pb._groupes(_classement([("a", 3.0), ("b", 2.4), ("c", 1.8)]), 1.0)
    assert [sorted(m) for _s, m in groupes] == [["a", "b"], ["c"]]


# ----------------------------------------------------------------------- comparaison


def test_deux_classements_identiques_sont_en_parite():
    ref = _reference({"q1": _classement([("a", 0.9), ("b", 0.8)])})
    rapport = pb.comparer(ref, {"classements": {"q1": _classement([("a", 0.9), ("b", 0.8)])}}, 0.0)
    assert rapport["PARITE"] is True
    assert rapport["verdict_partiel"] == "PARITE STRICTE"
    assert rapport["B1b_ordre_interne_DIAGNOSTIC"]["identiques_position_a_position"] == 1


def test_une_permutation_dex_aequo_est_acceptee_et_comptee():
    """Elle passe la parité — mais elle est comptée à part, jamais confondue avec l'identité."""
    ref = _reference({"q1": _classement([("a", 0.9), ("b", 0.9), ("c", 0.7)])})
    candidat = {"classements": {"q1": _classement([("b", 0.9), ("a", 0.9), ("c", 0.7)])}}
    rapport = pb.comparer(ref, candidat, 0.0)
    assert rapport["PARITE"] is True
    diagnostic = rapport["B1b_ordre_interne_DIAGNOSTIC"]
    assert diagnostic["identiques_position_a_position"] == 0
    assert diagnostic["equivalentes_par_permutation_dex_aequo"]["qids"] == ["q1"]


def test_un_vrai_reordonnancement_echoue():
    ref = _reference({"q1": _classement([("a", 0.9), ("b", 0.8)])})
    candidat = {"classements": {"q1": _classement([("b", 0.8), ("a", 0.9)])}}
    rapport = pb.comparer(ref, candidat, 0.0)
    assert rapport["PARITE"] is False
    assert rapport["B1b_ordre_interne_DIAGNOSTIC"]["divergentes"]["n"] == 1


# ------------------------------------------------- B1a : la composition est la porte dure


def _long(n, depart=1.0, pas=0.001):
    return [(f"c{i:03d}", depart - i * pas) for i in range(n)]


def test_b1a_attrape_un_echange_a_la_frontiere_du_top5():
    """Le cas pour lequel l'amendement n°1 existe.

    Un échange entre les rangs 5 et 6 ne touche PAS l'ensemble du top-50 — il ne fait que
    permuter deux membres à l'intérieur. Mais il change l'ensemble du top-5, donc ce qui est
    servi. Tester la composition à la seule profondeur 50 le laisserait passer.
    """
    base = _long(50)
    permute = base[:4] + [base[5], base[4]] + base[6:]
    rapport = pb.comparer(_reference({"q1": _classement(base)}),
                          {"classements": {"q1": _classement(permute)}}, 0.0)
    composition = rapport["B1a_composition_PORTE_DURE"]
    assert composition["profondeurs"]["top5"]["divergentes"] == ["q1"]
    assert composition["profondeurs"]["top50"]["divergentes"] == []   # invisible à cette profondeur
    assert composition["tenue"] is False
    assert rapport["verdict_partiel"] == "REFUS"


def test_b1a_tient_sur_une_permutation_de_rang_profond():
    """Le cas que l'amendement refuse de refuser.

    Deux rangs profonds échangés : aucun ensemble ne bouge à 5, 10 ni 50. La porte tient ;
    seul le diagnostic B1b le signale. Un reranker reclasse ce qu'il reçoit, et l'utilisateur
    ne voit pas les rangs 30 et 31.
    """
    base = _long(50)
    permute = base[:30] + [base[31], base[30]] + base[32:]
    rapport = pb.comparer(_reference({"q1": _classement(base)}),
                          {"classements": {"q1": _classement(permute)}}, 0.0)
    assert rapport["B1a_composition_PORTE_DURE"]["tenue"] is True
    assert rapport["B1b_ordre_interne_DIAGNOSTIC"]["divergentes"]["n"] == 1
    assert rapport["PARITE"] is False          # B1b non vert à tau = 0 : ce n'est pas une parité stricte


def test_b1a_attrape_un_candidat_substitue_en_fond_de_pool():
    """Un chunk qui entre au rang 49 en chasse un autre : le reranker reçoit un autre pool."""
    base = _long(50)
    substitue = base[:49] + [("INTRUS", base[49][1])]
    rapport = pb.comparer(_reference({"q1": _classement(base)}),
                          {"classements": {"q1": _classement(substitue)}}, 0.0)
    assert rapport["B1a_composition_PORTE_DURE"]["profondeurs"]["top50"]["divergentes"] == ["q1"]
    assert rapport["PARITE"] is False


def test_le_verdict_partiel_dit_ce_quil_ne_voit_pas():
    """B6 et B7 sont des portes dures hors de portée de ce module : il doit le déclarer."""
    ref = _reference({"q1": _classement([("a", 0.9)])})
    rapport = pb.comparer(ref, {"classements": {"q1": _classement([("a", 0.9)])}}, 0.0)
    assert "B6" in rapport["portee"] and "B7" in rapport["portee"]


def test_un_chunk_substitue_a_score_identique_echoue():
    """Le piège : les scores concordent parfaitement, seul le contenu a changé.

    Une comparaison qui ne regarderait que les scores — ou que le nDCG — laisserait passer une
    collection servant d'autres passages sous les mêmes valeurs.
    """
    ref = _reference({"q1": _classement([("a", 0.9), ("b", 0.8)])})
    candidat = {"classements": {"q1": _classement([("a", 0.9), ("ZZZ", 0.8)])}}
    assert pb.comparer(ref, candidat, 0.0)["PARITE"] is False


def test_une_question_manquante_echoue():
    ref = _reference({"q1": _classement([("a", 0.9)]), "q2": _classement([("b", 0.9)])})
    rapport = pb.comparer(ref, {"classements": {"q1": _classement([("a", 0.9)])}}, 0.0)
    assert rapport["PARITE"] is False
    assert rapport["questions_manquantes"] == ["q2"]


def test_une_question_en_trop_echoue():
    ref = _reference({"q1": _classement([("a", 0.9)])})
    candidat = {"classements": {"q1": _classement([("a", 0.9)]), "q9": _classement([("z", 0.5)])}}
    rapport = pb.comparer(ref, candidat, 0.0)
    assert rapport["PARITE"] is False
    assert rapport["questions_en_trop"] == ["q9"]


def test_un_classement_tronque_echoue():
    """Un serveur qui rendrait moins de candidats doit échouer, pas passer sur un préfixe."""
    ref = _reference({"q1": _classement([("a", 0.9), ("b", 0.8), ("c", 0.7)])})
    candidat = {"classements": {"q1": _classement([("a", 0.9), ("b", 0.8)])}}
    assert pb.comparer(ref, candidat, 0.0)["PARITE"] is False


def test_la_tolerance_leve_une_divergence_de_dernier_bit_mais_pas_un_vrai_ecart():
    ref = _reference({"q1": _classement([("a", 0.9), ("b", 0.9 - 1e-9)])})
    candidat = {"classements": {"q1": _classement([("b", 0.9 - 1e-9), ("a", 0.9)])}}
    assert pb.comparer(ref, candidat, 0.0)["PARITE"] is False        # τ = 0 : c'est un écart
    assert pb.comparer(ref, candidat, 1e-6)["PARITE"] is True        # τ large : indiscernable

    net = _reference({"q1": _classement([("a", 0.9), ("b", 0.5)])})
    inverse = {"classements": {"q1": _classement([("b", 0.5), ("a", 0.9)])}}
    assert pb.comparer(net, inverse, 1e-6)["PARITE"] is False        # la tolérance ne couvre pas tout


# ------------------------------------------------------------------- recensement réel


def test_le_recensement_compte_les_ex_aequo_et_la_fragilite():
    ref = _reference({
        "q1": _classement([("a", 0.9), ("b", 0.9), ("c", 0.7)]),   # un ex æquo strict
        "q2": _classement([("d", 0.9), ("e", 0.5)]),               # écart net
    })
    rapport = pb.egalites(ref)
    assert rapport["ex_aequo_stricts"]["questions_touchees"] == 1
    assert rapport["ex_aequo_stricts"]["rangs_permutables"] == 1
    assert rapport["questions_fragiles_par_amplitude_de_perturbation"]["0"] == 1


@pytest.mark.skipif(not (BANC / "parite-reference-dense-5530cba145.json").exists(),
                    reason="référence de parité absente")
def test_la_reference_gelee_porte_exactement_les_deux_ex_aequo_pre_enregistres():
    """Le pré-enregistrement les nomme : `v3/t03` et `v3/s33`. S'il y en avait d'autres,
    la règle B2 — « aucune autre permutation acceptée » — perdrait son sens."""
    reference = json.loads((BANC / "parite-reference-dense-5530cba145.json").read_text())
    detail = pb.egalites(reference)["ex_aequo_stricts"]["detail"]
    assert sorted(detail) == ["v3/s33", "v3/t03"]


@pytest.mark.skipif(not (BANC / "parite-reference-dense-5530cba145.json").exists(),
                    reason="référence de parité absente")
def test_la_reference_gelee_couvre_la_population_fixe():
    reference = json.loads((BANC / "parite-reference-dense-5530cba145.json").read_text())
    assert reference["questions"] == 155
    assert reference["branche"] == "dense"
    assert pb.comparer(reference, reference, 0.0)["PARITE"] is True
