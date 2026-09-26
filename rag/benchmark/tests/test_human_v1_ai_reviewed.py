"""Audit du contrat avant d'y ajouter un statut AI-reviewed — les 14 propriétés exigées.

Ce fichier est écrit **avant** toute modification du schéma, pour savoir ce qui tient déjà.
La frontière qu'il protège est une seule : *un lot validé par des modèles ne doit jamais
pouvoir se faire passer pour un lot validé par une personne.* Tout le reste en découle —
si la frontière est franchissable, la campagne AI-reviewed devient un blanchiment de gold.

    P1   un `gold_candidate` ne compte dans aucune campagne
    P2   un fait AI-reviewed ne compte que dans une campagne AI-reviewed
    P3   aucun score passé n'est recalculé par un gold ajouté depuis
    P4   une promotion porte version, date, motif, source, **agents**, **adjudication**, **désaccords**
    P5   un retrait porte trace, motif, version, **source de la correction**, **adjudication**
    P6   une citation ne dépend pas d'un `chunk_id`
    P7   une citation porte document, référence, auteur, année, page **ou plage qualifiée**,
         offsets, extrait, empreinte
    P8   les offsets sont vérifiés contre le texte canonique
    P9   un tableau servi en Markdown reste relié à sa source canonique
    P10  un fait ne peut être promu que si son ancre porte réellement une source citable
    P11  une réponse partielle ne devient pas complète par défaut
    P12  une inférence ne se confond pas avec un fait explicitement écrit
    P13  une abstention ne se promeut pas sur une absence purement lexicale
    P14  un lot AI-reviewed ne peut pas s'appeler human-reviewed
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
HUMAN = HERE.parent / "human-v1"

_spec = importlib.util.spec_from_file_location("hv1_ai", HUMAN / "valider.py")
valider = importlib.util.module_from_spec(_spec)
sys.modules["hv1_ai"] = valider
_spec.loader.exec_module(valider)


def item(**surcharge):
    base = {
        "id": "X900", "question": "Une question de test suffisamment longue ?",
        "repondable": "oui", "famille": "factuelle_localisee", "domaine": "options",
        "statut": "candidat", "gold_version": 1, "version_campagne": 1,
        "appuis": [{"ancre_id": "A1", "role": "porte_le_fait", "document_id": "doc-1",
                    "page": 3, "pages_couvertes": [3], "texte": "un texte d'ancre",
                    "offsets": {"debut": 10, "fin": 40, "doc_text_sha256": "abc123"}}],
        "faits_attendus": [], "conditions": [],
        "citation_minimale": {"documents_distincts": 1},
    }
    base.update(surcharge)
    return base


def fait_ai(**surcharge):
    """La forme complète que P4 doit exiger d'un fait promu par revue multi-agent."""
    base = {"texte": "un fait vérifié", "ancres": ["A1"], "exigence": "obligatoire",
            "statut_fait": "gold_ai_reviewed", "propose_en": 1, "introduit_en": 2,
            "origine": "revue_corpus", "nature": "explicite",
            "promu_par": "adjudicateur (claude-opus-5)", "promu_le": "2026-09-08",
            "motif_promotion": "soutenu explicitement par l'extrait A1",
            "revue_source": "doc-1 p. 3, offsets 10-40",
            "agents": ["A:sonnet", "B:opus", "C:haiku", "D:opus", "E:sonnet"],
            "adjudication": "cinq revues indépendantes, aucune objection matérielle",
            "desaccords": []}
    base.update(surcharge)
    return base


# ------------------------------------------------------------------ P1, P2, P3


def test_P1_un_candidat_ne_compte_dans_aucune_campagne():
    sujet = item(faits_attendus=[{"texte": "x", "ancres": ["A1"], "exigence": "obligatoire",
                                  "statut_fait": "gold_candidate", "propose_en": 1,
                                  "origine": "revue_corpus"}])
    for mode in (None, "humain", "multi_agent_ai"):
        assert valider.gold_a_la_version(sujet, 2, review_mode=mode) == []


def test_P2_un_fait_ai_ne_compte_que_dans_une_campagne_ai():
    """Panne évitée : un gold validé par des modèles gonfle un score annoncé comme humain."""
    sujet = item(gold_version=2, faits_attendus=[fait_ai()])
    assert len(valider.gold_a_la_version(sujet, 2, review_mode="multi_agent_ai")) == 1
    assert valider.gold_a_la_version(sujet, 2, review_mode="humain") == []
    assert valider.gold_a_la_version(sujet, 2) == [], "le défaut doit être le mode humain"


def test_P2_un_fait_humain_ne_compte_pas_dans_une_campagne_ai():
    """La frontière est symétrique : une campagne AI ne s'attribue pas le travail humain."""
    sujet = item(gold_version=2, faits_attendus=[
        {"texte": "x", "ancres": ["A1"], "exigence": "obligatoire", "statut_fait": "gold",
         "introduit_en": 1}])
    assert len(valider.gold_a_la_version(sujet, 2, review_mode="humain")) == 1
    assert valider.gold_a_la_version(sujet, 2, review_mode="multi_agent_ai") == []


def test_P3_une_promotion_ne_change_pas_un_score_publie():
    sujet = item(gold_version=2, faits_attendus=[fait_ai()])
    assert valider.gold_a_la_version(sujet, 1, review_mode="multi_agent_ai") == []
    assert len(valider.gold_a_la_version(sujet, 2, review_mode="multi_agent_ai")) == 1


# ------------------------------------------------------------------ P4, P5


def test_P4_une_promotion_ai_complete_est_acceptee():
    assert valider.valider_schema(item(gold_version=2, faits_attendus=[fait_ai()])) == []


@pytest.mark.parametrize("champ", ["promu_le", "motif_promotion", "revue_source",
                                   "agents", "adjudication", "desaccords"])
def test_P4_une_promotion_ai_exige_sa_tracabilite(champ):
    """Sans agents ni adjudication, « validé par revue contradictoire » est une affirmation nue."""
    f = fait_ai()
    del f[champ]
    erreurs = valider.valider_schema(item(gold_version=2, faits_attendus=[f]))
    assert any(champ in e for e in erreurs), f"{champ} n'est pas exigé"


def test_P4_les_agents_doivent_etre_identifiables_et_plusieurs():
    """Une « revue contradictoire » à un seul agent n'est pas contradictoire."""
    for agents in ([], ["A"], "sonnet"):
        f = fait_ai(agents=agents)
        assert any("agents" in e for e in valider.valider_schema(
            item(gold_version=2, faits_attendus=[f])))


def test_P5_un_retrait_ai_porte_sa_source_de_correction_et_son_adjudication():
    retire = {"texte": "faux", "ancres": ["A1"], "exigence": "obligatoire",
              "statut_fait": "gold_retire", "introduit_en": 1, "retire_en": 2,
              "motif_retrait": "la source dit le contraire", "retire_par": "adjudicateur (IA)",
              "source_correction": "doc-1 p. 4, offsets 90-140",
              "adjudication": "objection du contradicteur retenue, non résolue"}
    assert valider.valider_schema(item(gold_version=2, faits_attendus=[retire])) == []
    for champ in ("source_correction", "adjudication"):
        ampute = dict(retire)
        del ampute[champ]
        assert any(champ in e for e in valider.valider_schema(
            item(gold_version=2, faits_attendus=[ampute]))), champ


# ------------------------------------------------------------------ P7, P10


def test_P7_une_plage_de_pages_est_qualifiee_et_non_reduite_a_une_page():
    """Panne évitée : annoncer « p. 30 » quand l'extrait court sur 30-31.

    Le système n'a pas de pointeur de phrase à la réponse ; annoncer une page exacte pour un
    passage qui en couvre deux est une précision qu'il ne possède pas.
    """
    sujet = item()
    sujet["appuis"][0]["pages_couvertes"] = [30, 31]
    sujet["appuis"][0]["page"] = 30
    sujet["faits_attendus"] = [fait_ai()]
    sujet["gold_version"] = 2
    erreurs = valider.valider_schema(sujet)
    assert any("plage" in e or "pages_couvertes" in e for e in erreurs), \
        "une ancre multi-page doit déclarer sa plage, pas une page unique"


def test_P10_un_fait_ne_se_promeut_pas_sur_une_ancre_non_citable():
    sujet = item(gold_version=2, faits_attendus=[fait_ai()])
    sujet["appuis"][0]["offsets"] = None
    sujet["appuis"][0]["offsets_motif"] = "texte_introuvable"
    assert valider.valider_schema(sujet), "une ancre sans offsets ne peut pas porter un gold"


# ------------------------------------------------------------------ P11, P12, P13


def test_P11_un_item_partiel_declare_ce_qui_n_est_pas_couvert():
    """Panne évitée : un item partiel noté comme s'il était complet."""
    sujet = item(repondable="partiel", gold_version=2, faits_attendus=[fait_ai()])
    assert any("couverture_partielle" in e for e in valider.valider_schema(sujet))
    sujet["couverture_partielle"] = {"couvert": "a", "non_couvert": "b", "preuve": "c"}
    assert any("reponse_fausse" in e for e in valider.valider_schema(sujet)), \
        "un item partiel doit porter la condition qui rend fausse l'invention de la part manquante"


def test_P12_une_inference_se_declare_et_montre_ses_premisses():
    """Panne évitée : une inférence promue comme un fait textuellement écrit."""
    sujet = item(gold_version=2, faits_attendus=[fait_ai(nature="inference")])
    erreurs = valider.valider_schema(sujet)
    assert any("premisses" in e for e in erreurs)
    assert any("hypotheses" in e for e in erreurs)
    complet = fait_ai(nature="inference", premisses=["A1"],
                      hypotheses=["le profil de perte d'un put diffère d'une mise totale"])
    assert valider.valider_schema(item(gold_version=2, faits_attendus=[complet])) == []


def test_P12_une_nature_inconnue_est_refusee():
    sujet = item(gold_version=2, faits_attendus=[fait_ai(nature="peut-etre")])
    assert any("nature" in e for e in valider.valider_schema(sujet))


def test_P13_une_abstention_exige_une_recherche_semantique_pas_un_grep():
    """Panne évitée, et elle s'est produite : « SEC Rule 605 » absent à 0 occurrence sur
    26 120 passages, alors que le corpus porte les règles 11Ac1-5 et 11Ac1-6, renumérotées
    depuis. Une absence lexicale n'est pas une absence."""
    sujet = item(repondable="non", appuis=[], faits_attendus=[],
                 citation_minimale={"documents_distincts": 0},
                 abstention_attendue="refuser en nommant ce qui manque",
                 absence={"porte": "x", "preuve": "0 occurrence de « medallion »",
                          "force_de_la_preuve": "forte"})
    erreurs = valider.valider_schema(sujet)
    assert any("recherche_semantique" in e for e in erreurs)
    sujet["absence"]["recherche_semantique"] = [
        {"variante": "medallion", "occurrences": 0},
        {"variante": "renaissance technolog", "occurrences": 1,
         "lecture": "mention de la fondation en 1982, aucun chiffre de performance"},
    ]
    assert valider.valider_schema(sujet) == []


def test_P13_une_preuve_forte_exige_plusieurs_variantes():
    sujet = item(repondable="non", appuis=[], faits_attendus=[],
                 citation_minimale={"documents_distincts": 0},
                 abstention_attendue="refuser",
                 absence={"porte": "x", "preuve": "y", "force_de_la_preuve": "forte",
                          "recherche_semantique": [{"variante": "medallion", "occurrences": 0}]})
    assert any("variante" in e for e in valider.valider_schema(sujet))


# ------------------------------------------------------------------ P14


def test_P14_un_lot_ai_reviewed_ne_peut_pas_se_dire_humain():
    """La frontière, et elle est la raison d'être de tout ce fichier."""
    lot = [item(statut="valide", valide_par="adjudicateur (claude-opus-5)",
                valide_le="2026-09-08", gold_version=2, faits_attendus=[fait_ai()])]
    erreurs = valider.valider_lot(lot) + valider.valider_schema(lot[0])
    assert any("modèle" in e or "humain" in e for e in erreurs)


def test_P14_un_lot_ai_reviewed_a_son_propre_statut():
    lot = [item(statut="ai_reviewed", review_mode="multi_agent_ai",
                valide_par="adjudicateur (claude-opus-5)", valide_le="2026-09-08",
                gold_version=2, faits_attendus=[fait_ai()])]
    assert valider.valider_lot(lot) == []
    assert valider.valider_schema(lot[0]) == []


def test_P14_un_statut_ai_reviewed_exige_son_review_mode():
    lot = [item(statut="ai_reviewed", valide_par="adjudicateur (IA)", valide_le="2026-09-08",
                gold_version=2, faits_attendus=[fait_ai()])]
    assert any("review_mode" in e for e in valider.valider_schema(lot[0]))


# ------------------------------------------------------------------ la campagne réelle


def test_la_campagne_assemblee_ne_compte_rien_en_mode_humain():
    """La garantie qui compte : `ai-reviewed-v1` ne peut pas gonfler un score humain.

    Ce test lit le fichier de campagne réellement produit, pas une fixture. S'il tombe un
    jour, c'est que la frontière a été franchie par un artefact, pas par une théorie.
    """
    import json

    chemin = HUMAN / "ai-reviewed-v1" / "items-ai-reviewed-v1.jsonl"
    if not chemin.exists():
        pytest.skip("campagne non assemblée")
    items = [json.loads(l) for l in chemin.read_text(encoding="utf-8").splitlines() if l.strip()]
    humain = sum(len(valider.gold_a_la_version(i, 2, review_mode="humain")) for i in items)
    ia = sum(len(valider.gold_a_la_version(i, 2, review_mode="multi_agent_ai")) for i in items)
    assert humain == 0, "un fait AI-reviewed compte dans une campagne humaine"
    assert ia > 0, "la campagne IA ne compte rien : l'assemblage a échoué"
    assert all(i.get("review_mode") == "multi_agent_ai" for i in items)
    assert all(i.get("statut") == "ai_reviewed" for i in items)


def test_la_campagne_ne_se_dit_nulle_part_validee_par_un_humain():
    """Panne évitée : une phrase de manifeste qui transforme une mesure utile en fausse
    revendication. Le mot « humain » ne doit apparaître dans la campagne que pour dire ce
    qu'elle **n'est pas**."""
    import json

    chemin = HUMAN / "ai-reviewed-v1" / "MANIFESTE.json"
    if not chemin.exists():
        pytest.skip("manifeste absent")
    manifeste = json.loads(chemin.read_text(encoding="utf-8"))
    assert manifeste["review_mode"] == "multi_agent_ai"
    assert "PAS un gold validé par une personne" in manifeste["avertissement"]
    for champ in ("adjudicateur",):
        assert not valider.signature_humaine(manifeste[champ]), \
            "l'adjudicateur d'une campagne IA ne doit pas se présenter comme humain"


def test_un_fait_promu_porte_toute_sa_tracabilite():
    import json

    chemin = HUMAN / "ai-reviewed-v1" / "items-ai-reviewed-v1.jsonl"
    if not chemin.exists():
        pytest.skip("campagne non assemblée")
    items = [json.loads(l) for l in chemin.read_text(encoding="utf-8").splitlines() if l.strip()]
    promus = [f for i in items for f in i["faits_attendus"]
              if f["statut_fait"] == "gold_ai_reviewed"]
    assert promus, "aucun fait promu"
    for f in promus:
        for champ in ("promu_par", "promu_le", "motif_promotion", "revue_source",
                      "adjudication", "agents", "desaccords", "nature"):
            assert champ in f, f"{f['texte'][:40]} — {champ} manquant"
        assert len(f["agents"]) >= 2


def test_un_candidat_refuse_se_distingue_d_un_candidat_non_traite():
    """Sans cette distinction, une campagne suivante repromouvrait un fait réfuté."""
    import json

    chemin = HUMAN / "ai-reviewed-v1" / "items-ai-reviewed-v1.jsonl"
    if not chemin.exists():
        pytest.skip("campagne non assemblée")
    items = [json.loads(l) for l in chemin.read_text(encoding="utf-8").splitlines() if l.strip()]
    verdicts = [f["revue_ai"]["verdict_revue"] for i in items for f in i["faits_attendus"]
                if f.get("revue_ai")]
    assert verdicts, "aucun verdict de revue enregistré"
    assert set(verdicts) <= {"refuse", "insuffisamment_source", "ambigu", "non_adjuge"}
    assert "refuse" in verdicts, "aucun fait réfuté : la revue n'aurait rien trouvé"


# ------------------------------------------- P15 : l'aptitude à noter, distincte de la validité


def test_P15_un_fait_exige_mais_refuse_rend_la_note_arbitraire():
    """Le défaut que ni le schéma ni les ancres ne pouvaient voir.

    Un item peut être parfaitement formé — offsets exacts, empreintes justes, statuts
    cohérents — et exiger de la réponse un fait que sa propre revue déclare faux. La note
    devient alors indécidable : l'énoncer coûte d'être faux, l'omettre coûte la couverture.
    Trouvé sur huit faits de `ai-reviewed-v1` par le rôle D, le 7 septembre 2026.
    """
    contradictoire = item(faits_attendus=[
        {"texte": "un fait que la revue refuse", "ancres": ["A1"], "exigence": "obligatoire",
         "statut_fait": "gold_candidate", "propose_en": 1, "origine": "revue_corpus",
         "revue_ai": {"verdict_revue": "refuse", "motif": "réfuté par sa source"}},
    ])
    empechements = valider.aptitude_a_noter(contradictoire)
    assert any("sa revue le refuse" in e for e in empechements), empechements


def test_P15_un_item_sans_aucun_fait_obligatoire_promu_ne_mesure_rien():
    creux = item(faits_attendus=[
        {"texte": "un candidat non traité", "ancres": ["A1"], "exigence": "obligatoire",
         "statut_fait": "gold_candidate", "propose_en": 1, "origine": "revue_corpus"},
    ])
    assert any("ne mesure rien" in e for e in valider.aptitude_a_noter(creux))


def test_P15_un_item_d_abstention_n_a_rien_a_exiger():
    """Un item sans fait obligatoire est apte : c'est sa forme normale, pas un manque."""
    assert valider.aptitude_a_noter(item(repondable="non", faits_attendus=[])) == []


def test_P15_un_item_dont_les_faits_obligatoires_sont_promus_est_apte():
    sain = item(faits_attendus=[
        {"texte": "un fait promu", "ancres": ["A1"], "exigence": "obligatoire",
         "statut_fait": "gold_ai_reviewed", "introduit_en": 1, "nature": "explicite",
         "promu_par": "adjudicateur (claude-opus-5)", "promu_le": "2026-09-07",
         "motif_promotion": "porté mot pour mot", "revue_source": "A1", "adjudication": "…",
         "agents": ["A:sonnet", "B:opus"], "desaccords": []},
    ])
    assert valider.aptitude_a_noter(sain) == []


# ------------------------------------------- P16 : un chiffre de tableau sans son en-tête


def test_P16_un_chiffre_de_tableau_ne_devient_pas_gold_sans_sa_ligne_d_entete():
    """`C05/A1` citait les trois lignes de données et coupait juste après l'en-tête : rien,
    dans l'intervalle cité, ne disait que 0,27 est SMB."""
    sans_entete = item(
        appuis=[{"ancre_id": "A1", "role": "porte_le_fait", "document_id": "doc-1", "page": 31,
                 "pages_couvertes": [31], "content_type": "table",
                 "texte": "<tr><td>Mean</td><td>0.51</td><td>0.27</td></tr>",
                 "offsets": {"debut": 10, "fin": 40, "doc_text_sha256": "abc"}}],
        faits_attendus=[{"texte": "Les moyennes sont 0,51 et 0,27.", "ancres": ["A1"],
                         "exigence": "obligatoire", "statut_fait": "gold_ai_reviewed",
                         "introduit_en": 1, "nature": "explicite", "promu_le": "2026-09-07",
                         "promu_par": "adjudicateur (claude-opus-5)", "motif_promotion": "…",
                         "revue_source": "A1", "adjudication": "…",
                         "agents": ["A:sonnet", "B:opus"], "desaccords": []}])
    erreurs = valider.valider_lecture_des_tableaux(sans_entete)
    assert any("ligne d'en-tête" in e for e in erreurs), erreurs

    avec_entete = __import__("json").loads(__import__("json").dumps(sans_entete))
    avec_entete["appuis"][0]["texte"] = (
        "<table><tr><td></td><td>SMB</td></tr><tr><td>Mean</td><td>0.51</td><td>0.27</td></tr>")
    assert valider.valider_lecture_des_tableaux(avec_entete) == []


def test_P16_la_regle_ne_bloque_ni_un_candidat_ni_un_fait_sans_chiffre():
    """Elle bloque la **promotion**, pas la proposition : un candidat ne compte nulle part."""
    candidat = item(
        appuis=[{"ancre_id": "A1", "role": "porte_le_fait", "document_id": "doc-1", "page": 31,
                 "content_type": "table", "texte": "<tr><td>0.51</td></tr>",
                 "offsets": {"debut": 10, "fin": 40, "doc_text_sha256": "abc"}}],
        faits_attendus=[{"texte": "La moyenne est 0,51.", "ancres": ["A1"],
                         "exigence": "obligatoire", "statut_fait": "gold_candidate",
                         "propose_en": 1, "origine": "revue_corpus"}])
    assert valider.valider_lecture_des_tableaux(candidat) == []


# ------------------------------------------- la campagne réelle, après les rôles C et D


def _campagne():
    import json

    chemin = HUMAN / "ai-reviewed-v1" / "items-ai-reviewed-v1.jsonl"
    if not chemin.exists():
        pytest.skip("campagne non assemblée")
    return [json.loads(l) for l in chemin.read_text(encoding="utf-8").splitlines() if l.strip()]


def test_l_ancre_de_tableau_de_la_campagne_porte_sa_ligne_d_entete():
    for item_reel in _campagne():
        assert valider.valider_lecture_des_tableaux(item_reel) == [], item_reel["id"]


def test_chaque_ancre_corrigee_conserve_ce_qu_elle_disait_avant():
    """Une correction d'ancre qui effacerait l'ancienne serait une réécriture silencieuse."""
    corrigees = [(i["id"], a) for i in _campagne() for a in i["appuis"] if a.get("corrections")]
    assert corrigees, "aucune correction d'ancre enregistrée"
    for ident, appui in corrigees:
        for c in appui["corrections"]:
            for champ in ("le", "campagne", "offsets_precedents", "texte_precedent",
                          "motif", "decide_par"):
                assert c.get(champ), f"{ident}/{appui['ancre_id']} — {champ} manquant"
            assert c["offsets_precedents"] != {"debut": appui["offsets"]["debut"],
                                               "fin": appui["offsets"]["fin"]}


def test_la_campagne_dit_combien_de_ses_items_peuvent_reellement_noter():
    """Le résultat que la campagne doit porter, et qu'aucun contrôle mécanique ne donnait."""
    items = _campagne()
    aptes = [i["id"] for i in items if not valider.aptitude_a_noter(i)]
    assert len(aptes) < len(items), (
        "tous les items seraient aptes : la règle d'aptitude ne mesure plus rien")
    assert aptes, "aucun item apte : le lot ne porterait plus aucun barème"


def test_un_fait_retrograde_ne_peut_pas_etre_repromu_par_inadvertance():
    """`C08/F1` a été promu puis rétrogradé. Un fait rétrogradé qui garderait `introduit_en`
    recompterait au premier assemblage suivant."""
    for item_reel in _campagne():
        for numero, fait in enumerate(item_reel["faits_attendus"], 1):
            if fait["statut_fait"] != "gold_candidate":
                continue
            assert fait.get("introduit_en") is None, f"{item_reel['id']}/F{numero}"
            assert valider.valider_fait_versionne(item_reel["id"], numero, fait, 2) == []


# ------------------------------------------- le contrôle à livre fermé et les rôles C et D


def test_le_controle_a_livre_ferme_declare_l_instrument_qu_il_a_reellement_utilise():
    """La substitution est légitime ; la taire ne le serait pas.

    Le contrôle devait tourner sur le générateur du banc. Les deux fournisseurs étant fermés,
    il a tourné sur `claude-opus-5`. Ce test garde la seule chose qui rend le résultat
    lisible : que l'artefact dise quel instrument a servi, et que sa conclusion soit
    déclarée asymétrique — une couverture nulle se transfère, une couverture pleine non.
    """
    import json

    chemin = HUMAN / "ai-reviewed-v1" / "controle-livre-ferme.json"
    if not chemin.exists():
        pytest.skip("contrôle non exécuté")
    art = json.loads(chemin.read_text(encoding="utf-8"))
    utilise = art["instrument"]["utilise"]
    assert utilise["substitution"] == "declaree"
    assert utilise["generateur"] != art["instrument"]["prevu"]["generateur"]
    assert art["instrument"]["prevu"]["etat"] == "indisponible"
    assert art["instrument"]["prevu"]["cause"], "une indisponibilité doit porter sa cause"
    assert utilise["ce_qui_se_transfere"] and utilise["ce_qui_ne_se_transfere_pas"]


def test_le_controle_a_livre_ferme_couvre_tout_le_lot():
    import json

    chemin = HUMAN / "ai-reviewed-v1" / "controle-livre-ferme.json"
    if not chemin.exists():
        pytest.skip("contrôle non exécuté")
    art = json.loads(chemin.read_text(encoding="utf-8"))
    mesures = art["runs"]["substitut"]["items"]
    assert sorted(mesures) == sorted(i["id"] for i in _campagne()), "des items manquent"
    for ident, v in mesures.items():
        assert v["verdict_discrimination"] in {
            "ne_discrimine_pas", "discrimine_mal", "discrimine", "sans_objet_abstention"}, ident
        for condition in ("nu", "prod"):
            assert v[condition]["coverage"] in (0, 1, 2), ident
            assert v[condition]["reponse"], f"{ident}/{condition} — réponse brute non publiée"


def test_un_run_ne_peut_pas_ecraser_l_autre():
    """Deux instruments, deux blocs. Écraser le substitut effacerait la seule mesure existante
    et, avec elle, la possibilité de chiffrer ce que la substitution a coûté."""
    source = (HUMAN / "ai-reviewed-v1" / "controle_livre_ferme.py").read_text(encoding="utf-8")
    assert 'runs.setdefault' in source or 'setdefault("runs"' in source
    assert "n'écrase pas" in source or "écraser la campagne substitut" in source


@pytest.mark.parametrize("role", ["A", "B", "C", "D"])
def test_chaque_role_a_rendu_un_rapport_par_item(role):
    dossier = HUMAN / "ai-reviewed-v1" / "rapports" / role
    if not (HUMAN / "ai-reviewed-v1" / "rapports").exists():
        pytest.skip("campagne non assemblée")
    rendus = sorted(p.stem for p in dossier.glob("C*.md"))
    assert rendus == sorted(i["id"] for i in _campagne()), f"rôle {role} : {rendus}"


def test_les_rapports_C_et_D_disent_avoir_travaille_sans_lire_les_autres():
    for role in ("C", "D"):
        for rapport in (HUMAN / "ai-reviewed-v1" / "rapports" / role).glob("C*.md"):
            texte = rapport.read_text(encoding="utf-8")
            assert "sans" in texte.lower() and "rapports" in texte.lower(), rapport.name
