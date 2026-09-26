"""Tests du validateur ``human-v1``.

Aucun accès au corpus : les contrôles portent sur la logique, pas sur les données. Un
test qui aurait besoin des 26 120 chunks pour dire si un schéma est cohérent serait un
test qu'on finit par ne plus lancer.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
MODULE = HERE.parent / "human-v1" / "valider.py"

# Le répertoire ``human-v1`` porte un tiret : il n'est pas importable comme paquet.
_spec = importlib.util.spec_from_file_location("human_v1_valider", MODULE)
valider = importlib.util.module_from_spec(_spec)
sys.modules["human_v1_valider"] = valider
_spec.loader.exec_module(valider)


# ------------------------------------------------------------------ ancrage


def test_normalisation_plie_casse_ponctuation_et_balises():
    """Les balises partent, leur contenu reste — comme dans ``gold_ancrage``.

    Un exposant de note de bas de page (« <sup>3</sup> ») laisse donc un « 3 » dans le
    texte normalisé. C'est voulu : le chiffre appartient au texte tel qu'il est indexé, et
    les deux côtés de la comparaison le portent de la même façon.
    """
    assert valider.normaliser("The <sup>3</sup> Price, Impact!") == "the 3 price impact"


def test_ancre_plus_courte_que_le_gramme_reste_appariable():
    """Une ancre de moins de 5 mots doit pouvoir s'apparier à un candidat plus long.

    C'est le défaut latent corrigé dans ``gold_ancrage`` le 6 septembre 2026 : sans ``k``
    effectif commun aux deux côtés, une ancre courte ne produit aucun gramme partagé.
    """
    candidats = [{"chunk_id": "c1", "text": "alpha beta gamma delta epsilon zeta",
                  "blocs": [], "page_start": 1, "page_end": 1, "content_type": "text"}]
    situation = valider.situer("beta gamma delta", candidats)
    assert situation["couverture"] == 1.0


def test_situer_trouve_le_bon_chunk_et_mesure_la_marge():
    candidats = [
        {"chunk_id": "c1", "text": "the permanent component affects all future trades equally",
         "blocs": ["b1"], "page_start": 3, "page_end": 3, "content_type": "text"},
        {"chunk_id": "c2", "text": "an unrelated passage about calendar spread arbitrage conditions",
         "blocs": ["b2"], "page_start": 9, "page_end": 9, "content_type": "text"},
    ]
    situation = valider.situer("the permanent component affects all future trades equally", candidats)
    assert situation["chunk"] == "c1"
    assert situation["couverture"] == 1.0
    assert situation["marge"] == 1.0
    assert situation["page"] == 3


def test_texte_duplique_dans_le_document_donne_une_marge_nulle():
    """Le cas mesuré sur 18,7 % des ancres de tableau : deux endroits également plausibles."""
    texte = "total return net of fees over the sample period considered here"
    candidats = [
        {"chunk_id": "c1", "text": texte, "blocs": ["b1"], "page_start": 1, "page_end": 1,
         "content_type": "table"},
        {"chunk_id": "c2", "text": texte, "blocs": ["b2"], "page_start": 7, "page_end": 7,
         "content_type": "table"},
    ]
    situation = valider.situer(texte, candidats)
    assert situation["marge"] == 0.0
    assert situation["ex_aequo"] == 2


# ------------------------------------------------------------------ re-découpage


def test_redecoupage_propage_les_blocs():
    chunks = [
        {"chunk_id": "c1", "text": "Première phrase. Deuxième phrase.", "blocs": ["b1", "b2"],
         "page_start": 1, "page_end": 1, "content_type": "text"},
        {"chunk_id": "c2", "text": "Troisième phrase.", "blocs": ["b3"],
         "page_start": 2, "page_end": 2, "content_type": "text"},
    ]
    neufs = valider.redecouper(chunks, cible=10_000)
    assert len(neufs) == 1
    assert set(neufs[0]["blocs"]) == {"b1", "b2", "b3"}


# ------------------------------------------------------------------ schéma


def item_minimal(**surcharge):
    base = {
        "id": "H900", "question": "Une question de test suffisamment longue ?",
        "repondable": "oui", "famille": "factuelle_localisee", "domaine": "options",
        "statut": "pilote", "gold_version": 1,
        # Une ancre citée par un fait doit être citable : document, page, extrait, offsets
        # opposables. Le contrat l'exige depuis l'audit du 7 septembre 2026 — voir
        # ``test_human_v1_contrat.py`` P6.
        "appuis": [{"ancre_id": "A1", "role": "porte_le_fait", "document_id": "doc-1",
                    "page": 3, "texte": "un texte d'ancre",
                    "offsets": {"debut": 10, "fin": 40, "doc_text_sha256": "abc123"}}],
        "faits_attendus": [{"texte": "un fait", "ancres": ["A1"], "exigence": "obligatoire",
                            "statut_fait": "gold", "introduit_en": 1}],
        "conditions": [],
        "citation_minimale": {"documents_distincts": 1},
    }
    base.update(surcharge)
    return base


def test_item_minimal_valide():
    assert valider.valider_schema(item_minimal()) == []


def test_fait_sans_ancre_est_refuse():
    item = item_minimal(faits_attendus=[{"texte": "un fait", "ancres": [], "exigence": "obligatoire"}])
    erreurs = valider.valider_schema(item)
    assert any("n'est rattaché à aucune ancre" in e for e in erreurs)


def test_fait_renvoyant_a_une_ancre_inexistante_est_refuse():
    item = item_minimal(faits_attendus=[{"texte": "un fait", "ancres": ["A7"],
                                         "exigence": "obligatoire"}])
    assert any("ancres inexistantes" in e for e in valider.valider_schema(item))


def test_item_non_repondable_ne_peut_pas_porter_d_ancres():
    item = item_minimal(repondable="non", abstention_attendue=True,
                        absence={"porte": "x", "preuve": "y", "force_de_la_preuve": "faible"})
    assert any("ne peut pas porter d'ancres" in e for e in valider.valider_schema(item))


def test_item_non_repondable_doit_declarer_la_force_de_sa_preuve():
    """Une absence non prouvée et une absence prouvée ne valent pas la même chose.

    Le banc v3 prouve l'absence par balayage littéral de toutes les graphies d'une source
    (``corpus.phrase_absent``). Un item human-v1 dont l'absence repose sur une lecture des
    meilleurs passages doit le dire, sous peine d'être lu comme une preuve du même ordre.
    """
    item = item_minimal(repondable="non", abstention_attendue=True, appuis=[],
                        faits_attendus=[], absence={"porte": "x", "preuve": "y"})
    assert any("force_de_la_preuve" in e for e in valider.valider_schema(item))


def test_citation_minimale_ne_peut_exiger_plus_de_documents_que_l_item_n_en_ancre():
    item = item_minimal(citation_minimale={"documents_distincts": 3})
    assert any("mais l'item n'en ancre que" in e for e in valider.valider_schema(item))


def test_condition_doit_dire_ce_qu_une_omission_coute():
    """« Réponse fausse » et « réponse incomplète » ne se notent pas pareil.

    Sans ce champ, un barème ne peut pas distinguer une réponse qui oublie une nuance
    d'une réponse qui affirme le contraire de la source.
    """
    item = item_minimal(conditions=[{"texte": "une condition", "ancres": ["A1"]}])
    assert any("consequence_si_omis" in e for e in valider.valider_schema(item))


@pytest.mark.parametrize("champ,valeur", [
    ("repondable", "peut-être"), ("famille", "inventee"),
    ("domaine", "astrologie"), ("statut", "definitif"),
])
def test_vocabulaires_fermes(champ, valeur):
    assert valider.valider_schema(item_minimal(**{champ: valeur}))


# ------------------------------------------------------------------ ancres, contrôles


def test_ancre_trop_courte_est_refusee():
    corpus = {"doc-1": [{"chunk_id": "c1", "text": "un texte d'ancre quelconque",
                         "blocs": [], "page_start": 1, "page_end": 1, "content_type": "text"}]}
    erreurs, _, _ = valider.valider_ancres(item_minimal(), corpus)
    assert any("trop courte" in e for e in erreurs)


def test_ancre_absente_du_document_est_refusee():
    longue = " ".join(f"mot{i}" for i in range(30))
    item = item_minimal(appuis=[{"ancre_id": "A1", "role": "porte_le_fait",
                                 "document_id": "doc-1", "texte": longue}])
    corpus = {"doc-1": [{"chunk_id": "c1", "text": "rien à voir avec l'ancre demandée ici",
                         "blocs": [], "page_start": 1, "page_end": 1, "content_type": "text"}]}
    erreurs, _, _ = valider.valider_ancres(item, corpus)
    assert any("introuvable" in e for e in erreurs)


def test_ancre_presente_et_unique_ne_leve_rien():
    longue = " ".join(f"mot{i}" for i in range(30))
    item = item_minimal(appuis=[{"ancre_id": "A1", "role": "porte_le_fait",
                                 "document_id": "doc-1", "texte": longue}])
    corpus = {"doc-1": [
        {"chunk_id": "c1", "text": f"avant {longue} après", "blocs": ["b1"],
         "page_start": 4, "page_end": 4, "content_type": "text"},
        {"chunk_id": "c2", "text": "un passage sans rapport du tout avec le précédent",
         "blocs": ["b2"], "page_start": 8, "page_end": 8, "content_type": "text"}]}
    erreurs, alertes, mesures = valider.valider_ancres(item, corpus)
    assert erreurs == []
    assert alertes == []
    assert mesures[0]["blocs"] == ["b1"]
    assert mesures[0]["page"] == 4


# ------------------------------------------------------------------ gold versionné


def fait(texte, statut="gold", **surcharge):
    base = {"texte": texte, "ancres": ["A1"], "exigence": "obligatoire", "statut_fait": statut}
    if statut == "gold":
        base["introduit_en"] = 1
    elif statut == "gold_candidate":
        base.update({"propose_en": 1, "origine": "revue_corpus"})
    base.update(surcharge)
    return base


def test_un_candidat_ne_compte_dans_aucun_score():
    item = item_minimal(faits_attendus=[fait("compte"), fait("ne compte pas", "gold_candidate")])
    retenus = valider.gold_a_la_version(item, 1)
    assert [f["texte"] for f in retenus] == ["compte"]


def test_gold_reconstructible_a_une_version_anterieure():
    """Le contrôle qui rend un score interprétable après une évolution du gold.

    Un fait introduit en v2 ne doit pas contaminer le score de v1 : sans cela, publier une
    v2 réécrirait rétroactivement tous les chiffres antérieurs.
    """
    item = item_minimal(gold_version=2, faits_attendus=[
        fait("présent dès v1"),
        fait("ajouté en v2", introduit_en=2),
    ])
    assert [f["texte"] for f in valider.gold_a_la_version(item, 1)] == ["présent dès v1"]
    assert len(valider.gold_a_la_version(item, 2)) == 2


def test_un_fait_retire_disparait_du_gold_a_partir_de_sa_version():
    item = item_minimal(gold_version=3, faits_attendus=[
        fait("faux, retiré en v3", "gold_retire", introduit_en=1, retire_en=3,
             motif_retrait="la source ne soutient pas ce fait"),
    ])
    assert len(valider.gold_a_la_version(item, 2)) == 1   # comptait encore en v2
    assert valider.gold_a_la_version(item, 3) == []       # plus en v3


def test_promouvoir_un_candidat_ne_change_pas_les_scores_anterieurs():
    """La propriété demandée : une v2 n'invalide pas les chiffres publiés en v1."""
    avant = item_minimal(gold_version=1, faits_attendus=[
        fait("origine"), fait("proposé", "gold_candidate", origine="execution",
                              decouvert_par="dense@5")])
    apres = item_minimal(gold_version=2, faits_attendus=[
        fait("origine"), fait("proposé", introduit_en=2, propose_en=1,
                              origine="execution", decouvert_par="dense@5")])
    assert len(valider.gold_a_la_version(avant, 1)) == 1
    assert len(valider.gold_a_la_version(apres, 1)) == 1   # v1 inchangée après promotion
    assert len(valider.gold_a_la_version(apres, 2)) == 2


def test_une_comparaison_peut_ecarter_les_faits_decouverts_par_un_des_comparands():
    """Un gold enrichi par les sorties de A favorise A, et cela ne se voit pas dans le score."""
    item = item_minimal(gold_version=2, faits_attendus=[
        fait("neutre"),
        fait("trouvé par A", introduit_en=2, propose_en=1, origine="execution",
             decouvert_par="dense@5"),
    ])
    assert len(valider.gold_a_la_version(item, 2)) == 2
    assert [f["texte"] for f in valider.gold_a_la_version(item, 2, {"dense@5"})] == ["neutre"]


def test_candidat_issu_d_une_execution_doit_nommer_sa_configuration():
    item = item_minimal(faits_attendus=[fait("x", "gold_candidate", origine="execution")])
    assert any("decouvert_par" in e for e in valider.valider_schema(item))


def test_candidat_ne_peut_pas_porter_introduit_en():
    item = item_minimal(faits_attendus=[fait("x", "gold_candidate", introduit_en=1)])
    assert any("ne peut pas porter introduit_en" in e for e in valider.valider_schema(item))


def test_fait_retire_sans_motif_est_refuse():
    """Retirer un fait parce qu'il fait baisser un score est exactement ce qu'on interdit."""
    item = item_minimal(gold_version=2, faits_attendus=[
        fait("x", "gold_retire", introduit_en=1, retire_en=2)])
    assert any("motif" in e for e in valider.valider_schema(item))


def test_version_de_fait_ne_peut_depasser_celle_du_lot():
    item = item_minimal(gold_version=1, faits_attendus=[fait("x", introduit_en=4)])
    assert any("dépasse la gold_version du lot" in e for e in valider.valider_schema(item))


def test_gold_version_obligatoire():
    item = item_minimal()
    del item["gold_version"]
    assert any("gold_version" in e for e in valider.valider_schema(item))


def test_lot_a_une_seule_version_de_gold():
    lot = [item_minimal(id="H001"), item_minimal(id="H002", gold_version=2)]
    assert any("un lot porte un seul contrat" in e for e in valider.valider_lot(lot))


def test_lot_refuse_les_identifiants_en_double():
    assert any("double" in e for e in valider.valider_lot([item_minimal(), item_minimal()]))


def test_resume_gold_separe_ce_qui_compte_de_ce_qui_ne_compte_pas():
    lot = [item_minimal(faits_attendus=[
        fait("a"), fait("b", "gold_candidate", origine="execution", decouvert_par="dense@5")])]
    resume = valider.resume_gold(lot)
    assert resume["par_statut"]["gold"] == 1
    assert resume["par_statut"]["gold_candidate"] == 1
    assert resume["candidats_par_origine"] == {"dense@5": 1}


def test_seuils_alignes_sur_gold_ancrage():
    """Deux instruments qui décident sur la même grandeur doivent décider au même endroit."""
    ga = valider.charger_gold_ancrage()
    if ga is None:
        pytest.skip("gold_ancrage indisponible")
    assert valider.SEUIL_COUVERTURE == ga.SEUIL_ANCRE
    assert valider.K_GRAMME == ga.K_GRAMME
    assert valider.normaliser("The Price, Impact!") == ga.normaliser("The Price, Impact!")


# ---------------------------------------------------------------- ce que le contrat exigeait
# encore, et qui manquait : l'auteur d'une décision, la trace d'une promotion, et la
# séparation entre « contre quel gold » et « quand ».


def test_un_retrait_doit_nommer_son_auteur_et_pas_seulement_son_motif():
    """Un motif sans auteur est une décision que personne n'a prise."""
    item = item_minimal(gold_version=2, faits_attendus=[
        fait("x", "gold_retire", introduit_en=1, retire_en=2,
             motif_retrait="la source ne le soutient pas")])
    assert any("retire_par" in e for e in valider.valider_schema(item))
    item["faits_attendus"][0]["retire_par"] = "annotateur humain"
    assert not [e for e in valider.valider_schema(item) if "retire_par" in e]


def test_une_promotion_doit_creer_une_version_nouvelle():
    """Promouvoir dans la version où le fait a été proposé, c'est promouvoir sans version."""
    item = item_minimal(gold_version=1, faits_attendus=[
        fait("x", introduit_en=1, propose_en=1, promu_par="humain", revue_source="lue p. 4")])
    assert any("crée une version nouvelle" in e for e in valider.valider_schema(item))


def test_une_promotion_doit_porter_son_auteur_et_sa_revue_source():
    item = item_minimal(gold_version=2, faits_attendus=[fait("x", introduit_en=2, propose_en=1)])
    erreurs = valider.valider_schema(item)
    assert any("promu_par" in e for e in erreurs)
    assert any("revue_source" in e for e in erreurs)


def test_un_fait_d_origine_n_a_rien_a_prouver():
    """La trace de promotion n'est exigée que d'un fait qui a été candidat."""
    item = item_minimal(gold_version=2, faits_attendus=[fait("x", introduit_en=1)])
    assert valider.valider_schema(item) == []


def test_le_gold_ne_peut_pas_changer_pendant_une_campagne():
    """L'invariant temporel : les premiers items seraient notés contre un gold plus pauvre."""
    lot = [item_minimal(gold_version=1), item_minimal(gold_version=2)]
    lot[0]["id"], lot[1]["id"] = "H001", "H002"
    for it in lot:
        it["version_campagne"] = 1
    assert any("pendant l'annotation" in e for e in valider.valider_lot(lot))


def test_deux_campagnes_peuvent_porter_deux_golds():
    lot = [item_minimal(gold_version=1), item_minimal(gold_version=2)]
    lot[0]["id"], lot[1]["id"] = "H001", "H002"
    lot[0]["version_campagne"], lot[1]["version_campagne"] = 1, 2
    assert not [e for e in valider.valider_lot(lot) if "pendant l'annotation" in e]


def test_des_offsets_absents_exigent_un_motif():
    """Un offset manquant est une information ; un offset manquant en silence est un trou.

    Depuis l'audit du contrat, le motif ne suffit plus pour une ancre **citée** : celle-là
    est refusée tout court (P6). Le motif reste la voie pour une ancre de contexte, qu'on
    peut vouloir garder sans pouvoir la localiser.
    """
    item = item_minimal()
    item["appuis"].append({"ancre_id": "A2", "role": "contexte", "document_id": "doc-1",
                           "page": 4, "texte": "un texte de contexte", "offsets": None})
    assert any("sans motif déclaré" in e for e in valider.valider_schema(item))
    item["appuis"][1]["offsets_motif"] = "texte_introuvable"
    assert valider.valider_schema(item) == []

    # et la même ancre, si un fait la cite, n'a plus droit au motif
    item["faits_attendus"][0]["ancres"] = ["A1", "A2"]
    assert any("sans offsets" in e for e in valider.valider_schema(item))


def test_un_offset_sans_empreinte_du_texte_n_est_pas_opposable():
    item = item_minimal()
    item["appuis"][0]["offsets"] = {"debut": 10, "fin": 40}
    assert any("doc_text_sha256" in e for e in valider.valider_schema(item))


def test_un_intervalle_inverse_est_refuse():
    item = item_minimal()
    item["appuis"][0]["offsets"] = {"debut": 40, "fin": 10, "doc_text_sha256": "abc"}
    assert any("inversé" in e for e in valider.valider_schema(item))
