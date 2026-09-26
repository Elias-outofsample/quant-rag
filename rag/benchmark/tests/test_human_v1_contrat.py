"""Audit du contrat `human-v1` — les huit propriétés exigées avant la cohorte 1.

Ce fichier est un **audit**, pas un complément de couverture : chaque test y nomme une
propriété que le contrat doit garantir avant qu'un humain n'y consacre du temps, et chaque
propriété a une panne concrète derrière elle. Une règle sans panne nommable n'entre pas ici.

Les huit, dans l'ordre du cahier des charges :

    P1  un `gold` ne compte dans une campagne que s'il est gelé avant elle
    P2  un `gold_candidate` ne compte jamais dans la campagne qui l'a découvert
    P3  une promotion porte version postérieure, date, auteur **humain**, motif, revue source
    P4  un `gold_retire` conserve le fait, le motif, l'auteur et la version
    P5  les scores d'une version restent reconstructibles avec le gold de cette version
    P6  une citation utilisateur porte document, page, offsets, extrait, `doc_text_sha256`
    P7  les `block_ids` restent internes — jamais une citation
    P8  un lot présenté comme humain est refusé si aucun humain n'a validé

Aucun accès au corpus : la logique se teste sans les 26 120 chunks.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODULE = HERE.parent / "human-v1" / "valider.py"

_spec = importlib.util.spec_from_file_location("human_v1_valider_contrat", MODULE)
valider = importlib.util.module_from_spec(_spec)
sys.modules["human_v1_valider_contrat"] = valider
_spec.loader.exec_module(valider)


def item(**surcharge):
    base = {
        "id": "H900", "question": "Une question de test suffisamment longue ?",
        "repondable": "oui", "famille": "factuelle_localisee", "domaine": "options",
        "statut": "pilote", "gold_version": 1, "version_campagne": 1,
        "appuis": [{"ancre_id": "A1", "role": "porte_le_fait", "document_id": "doc-1",
                    "page": 3, "texte": "un texte d'ancre",
                    "offsets": {"debut": 10, "fin": 40, "doc_text_sha256": "abc123"}}],
        "faits_attendus": [{"texte": "un fait", "ancres": ["A1"], "exigence": "obligatoire",
                            "statut_fait": "gold", "introduit_en": 1}],
        "conditions": [], "citation_minimale": {"documents_distincts": 1},
    }
    base.update(surcharge)
    return base


def promu(**surcharge):
    """Un fait candidat promu en v2 — la forme complète que P3 exige."""
    base = {"texte": "promu", "ancres": ["A1"], "exigence": "obligatoire",
            "statut_fait": "gold", "propose_en": 1, "introduit_en": 2,
            "origine": "revue_corpus", "promu_par": "relecteur (humain)",
            "promu_le": "2026-09-08", "motif_promotion": "le fait est central à la question",
            "revue_source": "doc-1 p. 3, lu et confirmé"}
    base.update(surcharge)
    return base


# ------------------------------------------------------------------ P1 et P5


def test_P1_un_gold_ne_compte_que_dans_les_versions_posterieures_a_son_introduction():
    """Panne évitée : un fait ajouté après coup gonfle rétroactivement un score publié."""
    sujet = item(gold_version=2, faits_attendus=[
        {"texte": "dès v1", "ancres": ["A1"], "exigence": "obligatoire",
         "statut_fait": "gold", "introduit_en": 1},
        {"texte": "ajouté en v2", "ancres": ["A1"], "exigence": "obligatoire",
         "statut_fait": "gold", "introduit_en": 2}])
    assert [f["texte"] for f in valider.gold_a_la_version(sujet, 1)] == ["dès v1"]
    assert len(valider.gold_a_la_version(sujet, 2)) == 2


def test_P1_le_gold_ne_recule_pas_d_une_campagne_a_la_suivante():
    """Panne évitée : rejouer une campagne contre un gold antérieur choisi après coup.

    Le lot interdit déjà deux `gold_version` dans une même campagne. Il manquait le sens
    inverse : une campagne 2 notée contre la v1 alors que la campagne 1 l'était contre la
    v2 permettrait de choisir, après résultat, la version qui flatte.
    """
    lot = [item(id="H001", version_campagne=1, gold_version=2),
           item(id="H002", version_campagne=2, gold_version=1)]
    assert any("recule" in e for e in valider.valider_lot(lot))


def test_P5_une_promotion_ne_change_pas_les_scores_deja_publies():
    avant = item(gold_version=1, faits_attendus=[
        {"texte": "origine", "ancres": ["A1"], "exigence": "obligatoire",
         "statut_fait": "gold", "introduit_en": 1},
        {"texte": "proposé", "ancres": ["A1"], "exigence": "obligatoire",
         "statut_fait": "gold_candidate", "propose_en": 1, "origine": "revue_corpus"}])
    apres = item(gold_version=2, faits_attendus=[
        {"texte": "origine", "ancres": ["A1"], "exigence": "obligatoire",
         "statut_fait": "gold", "introduit_en": 1},
        promu(texte="proposé")])
    assert len(valider.gold_a_la_version(avant, 1)) == 1
    assert len(valider.gold_a_la_version(apres, 1)) == 1        # v1 inchangée
    assert len(valider.gold_a_la_version(apres, 2)) == 2


# ------------------------------------------------------------------ P2


def test_P2_un_candidat_ne_compte_dans_aucune_version():
    """Panne évitée : une sortie du système entre dans le gold de la campagne qui l'a produite."""
    sujet = item(gold_version=3, faits_attendus=[
        {"texte": "candidat", "ancres": ["A1"], "exigence": "obligatoire",
         "statut_fait": "gold_candidate", "propose_en": 1, "origine": "execution",
         "decouvert_par": "dense@5"}])
    assert all(valider.gold_a_la_version(sujet, k) == [] for k in (1, 2, 3))


def test_P2_un_candidat_issu_d_une_execution_nomme_la_configuration():
    sujet = item(faits_attendus=[
        {"texte": "x", "ancres": ["A1"], "exigence": "obligatoire",
         "statut_fait": "gold_candidate", "propose_en": 1, "origine": "execution"}])
    assert any("decouvert_par" in e for e in valider.valider_schema(sujet))


def test_P2_une_comparaison_peut_ecarter_les_faits_decouverts_par_un_comparand():
    sujet = item(gold_version=2, faits_attendus=[
        {"texte": "neutre", "ancres": ["A1"], "exigence": "obligatoire",
         "statut_fait": "gold", "introduit_en": 1},
        promu(texte="trouvé par A", origine="execution", decouvert_par="dense@5")])
    assert len(valider.gold_a_la_version(sujet, 2)) == 2
    assert [f["texte"] for f in valider.gold_a_la_version(sujet, 2, {"dense@5"})] == ["neutre"]


# ------------------------------------------------------------------ P3


def test_P3_une_promotion_complete_est_acceptee():
    assert valider.valider_schema(item(gold_version=2, faits_attendus=[promu()])) == []


def test_P3_une_promotion_exige_une_version_posterieure():
    sujet = item(gold_version=1, faits_attendus=[promu(introduit_en=1)])
    assert any("version nouvelle" in e for e in valider.valider_schema(sujet))


def test_P3_une_promotion_exige_une_date():
    """Panne évitée : une promotion sans date ne peut pas être située par rapport à une campagne."""
    fait = promu()
    del fait["promu_le"]
    assert any("promu_le" in e for e in valider.valider_schema(item(gold_version=2,
                                                                   faits_attendus=[fait])))


def test_P3_une_promotion_exige_un_motif_distinct_de_la_revue_source():
    """Le motif dit *pourquoi promouvoir*, la revue dit *où c'est écrit*. Les deux."""
    sans_motif = promu()
    del sans_motif["motif_promotion"]
    sans_revue = promu()
    del sans_revue["revue_source"]
    assert any("motif_promotion" in e
               for e in valider.valider_schema(item(gold_version=2, faits_attendus=[sans_motif])))
    assert any("revue_source" in e
               for e in valider.valider_schema(item(gold_version=2, faits_attendus=[sans_revue])))


def test_P3_un_modele_ne_peut_pas_promouvoir_son_propre_candidat():
    """**La panne la plus dangereuse du dossier, et elle s'est déjà produite.**

    `H004` du lot pilote portait un fait apparu dans une sortie du système, promu d'emblée
    en gold. Sans cette règle, le contrat d'évaluation se met à suivre les sorties qu'il est
    censé juger, et le biais ne se voit dans aucun score.
    """
    for auteur in ("claude-opus-5 (session B)", "GPT-5", "gemini-3.1-flash-lite", "mistral"):
        sujet = item(gold_version=2, faits_attendus=[promu(promu_par=auteur)])
        assert any("humain" in e for e in valider.valider_schema(sujet)), auteur
    assert valider.valider_schema(item(gold_version=2,
                                       faits_attendus=[promu(promu_par="relecteur")])) == []


# ------------------------------------------------------------------ P4


def test_P4_un_retrait_conserve_le_fait_son_motif_son_auteur_et_sa_version():
    retire = {"texte": "faux", "ancres": ["A1"], "exigence": "obligatoire",
              "statut_fait": "gold_retire", "introduit_en": 1, "retire_en": 3,
              "motif_retrait": "la source ne soutient pas ce fait",
              "retire_par": "relecteur",
              "source_correction": "doc-1 p. 5, le passage dit le contraire",
              "adjudication": "relecture de la source, fait invalidé"}
    sujet = item(gold_version=3, faits_attendus=[retire])
    assert valider.valider_schema(sujet) == []
    assert len(valider.gold_a_la_version(sujet, 2)) == 1        # comptait encore en v2
    assert valider.gold_a_la_version(sujet, 3) == []            # plus en v3
    assert sujet["faits_attendus"][0]["texte"] == "faux"        # le fait n'est pas supprimé
    for champ in ("motif_retrait", "retire_par"):
        ampute = dict(retire)
        del ampute[champ]
        assert any(champ.split("_")[0] in e
                   for e in valider.valider_schema(item(gold_version=3, faits_attendus=[ampute])))


def test_P4_un_modele_ne_peut_pas_retirer_un_fait():
    """Panne évitée : retirer un fait parce qu'il fait baisser un score."""
    retire = {"texte": "faux", "ancres": ["A1"], "exigence": "obligatoire",
              "statut_fait": "gold_retire", "introduit_en": 1, "retire_en": 2,
              "motif_retrait": "gêne le score", "retire_par": "claude-opus-5",
              "source_correction": "aucune", "adjudication": "aucune"}
    assert any("humain" in e
               for e in valider.valider_schema(item(gold_version=2, faits_attendus=[retire])))


def test_P4_un_retrait_adjudique_par_une_revue_contradictoire_est_recevable():
    """L'ouverture est délibérée, et elle est bornée : ce n'est pas un modèle **seul**.

    Une campagne AI-reviewed doit pouvoir retirer un fait qu'elle a elle-même promu, sinon
    une erreur constatée resterait au gold. Ce qui remplace la signature humaine n'est pas
    un nom : c'est la revue tracée, sa source et sa décision.
    """
    retire = {"texte": "faux", "ancres": ["A1"], "exigence": "obligatoire",
              "statut_fait": "gold_retire", "introduit_en": 1, "retire_en": 2,
              "motif_retrait": "l'extrait ne soutient pas ce fait",
              "retire_par": "adjudicateur (claude-opus-5)",
              "agents": ["A:sonnet", "B:opus", "C:haiku"],
              "source_correction": "doc-1 p. 4, offsets 90-140",
              "adjudication": "objection du contradicteur retenue, aucun agent ne la lève"}
    assert valider.valider_schema(item(gold_version=2, faits_attendus=[retire])) == []


# ------------------------------------------------------------------ P6 et P7


def test_P6_une_ancre_qui_porte_un_fait_doit_etre_citable():
    """Panne évitée : une citation servie sans offsets retombe sur `chunk_id`, qui ne
    survit pas à un re-découpage — c'est le trou de produit du §4.2 de `STRATEGIE.md`."""
    for champ in ("document_id", "page", "texte", "offsets"):
        sujet = item()
        del sujet["appuis"][0][champ]
        erreurs = valider.valider_schema(sujet)
        assert erreurs, f"une ancre sans {champ} est acceptée"


def test_P6_un_offset_sans_empreinte_du_texte_n_est_pas_opposable():
    sujet = item()
    sujet["appuis"][0]["offsets"] = {"debut": 10, "fin": 40}
    assert any("doc_text_sha256" in e for e in valider.valider_schema(sujet))


def test_P6_le_motif_d_offsets_absents_ne_vaut_que_pour_une_ancre_non_citee():
    """Une ancre de contexte peut ne pas être localisable ; une ancre citée, non."""
    citee = item()
    citee["appuis"][0]["offsets"] = None
    citee["appuis"][0]["offsets_motif"] = "texte_introuvable"
    assert any("porte_le_fait" in e for e in valider.valider_schema(citee))

    contexte = item()
    contexte["appuis"][0]["role"] = "contexte"
    contexte["appuis"][0]["offsets"] = None
    contexte["appuis"][0]["offsets_motif"] = "texte_introuvable"
    contexte["appuis"].append({"ancre_id": "A2", "role": "porte_le_fait",
                               "document_id": "doc-1", "page": 3, "texte": "autre",
                               "offsets": {"debut": 1, "fin": 9, "doc_text_sha256": "abc"}})
    contexte["faits_attendus"][0]["ancres"] = ["A2"]
    assert valider.valider_schema(contexte) == []


def test_P7_les_block_ids_ne_sont_jamais_exiges_pour_citer():
    """Ils aident en interne ; une citation complète ne doit pas en dépendre."""
    sujet = item()
    assert "blocs" not in sujet["appuis"][0]
    assert valider.valider_schema(sujet) == []


# ------------------------------------------------------------------ P8


def test_P8_un_lot_declare_valide_sans_relecteur_humain_est_refuse():
    """Panne évitée : un lot écrit par un modèle est publié comme banc humain."""
    lot = [item(statut="valide", auteur="claude-opus-5")]
    erreurs = valider.valider_lot(lot) + valider.valider_schema(lot[0])
    assert any("valide_par" in e for e in erreurs)


def test_P8_un_lot_valide_par_un_humain_date_est_accepte():
    sujet = item(statut="valide", auteur="claude-opus-5 (proposition)",
                 valide_par="relecteur", valide_le="2026-09-08")
    assert valider.valider_schema(sujet) == []
    assert valider.valider_lot([sujet]) == []


def test_P8_un_lot_pilote_n_exige_pas_de_relecteur_mais_ne_pretend_rien():
    assert valider.valider_schema(item(statut="pilote")) == []
