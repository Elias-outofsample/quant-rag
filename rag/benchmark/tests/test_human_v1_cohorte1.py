"""La cohorte 1 et son paquet de revue — ce qu'un humain doit pouvoir croire sans lire le code.

Trois choses sont testées ici, et chacune protège une promesse faite au relecteur :

    le lot         dix propositions, toutes candidates, aucune ne peut entrer dans un score
    la page        la page citée est celle d'un lecteur, pas le `page_start` 0-basé du chunk
    la fiche       aucune clé interne hors de l'annexe technique — un `chunk_id` n'est pas
                   une citation, et une fiche qui en montre une apprend au relecteur à en
                   accepter une
"""
from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
HUMAN = HERE.parent / "human-v1"
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parents[1]))

LOT = HUMAN / "items-cohorte1-v0.jsonl"
FICHE = HUMAN / "REVUE-COHORTE1.md"


def _module(nom: str, chemin: Path):
    spec = importlib.util.spec_from_file_location(nom, chemin)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[nom] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def items():
    if not LOT.exists():
        pytest.skip("cohorte 1 non construite")
    return [json.loads(l) for l in LOT.read_text(encoding="utf-8").splitlines() if l.strip()]


@pytest.fixture(scope="module")
def verifier():
    return _module("hv1_verifier", HUMAN / "verifier_cohorte1.py")


# ------------------------------------------------------------------ le lot


def test_le_lot_compte_dix_items_tous_candidats(items):
    assert len(items) == 10
    assert {i["statut"] for i in items} == {"candidat"}
    assert all(i.get("relecture_humaine") is None for i in items)
    assert all(f["statut_fait"] == "gold_candidate"
               for i in items for f in i["faits_attendus"])


def test_aucun_fait_ne_peut_entrer_dans_un_score(items):
    """La garantie centrale : un lot non relu ne pèse sur rien."""
    valider = _module("hv1_valider_c1", HUMAN / "valider.py")
    for item in items:
        for version in (1, 2, 3, 10):
            assert valider.gold_a_la_version(item, version) == []


def test_les_dix_situations_exigees_sont_couvertes(items, verifier):
    couvertes = {s for i in items for s in i.get("situations") or []}
    assert verifier.SITUATIONS <= couvertes, verifier.SITUATIONS - couvertes


def test_le_lot_porte_un_cas_d_abstention_et_un_cas_partiel(items):
    assert sum(1 for i in items if i["repondable"] == "non") >= 1
    assert sum(1 for i in items if i["repondable"] == "partiel") >= 1


def test_un_cas_d_abstention_porte_sa_preuve_d_absence(items):
    for item in (i for i in items if i["repondable"] == "non"):
        absence = item.get("absence") or {}
        assert absence.get("preuve"), f"{item['id']} : absence sans preuve"
        assert absence.get("force_de_la_preuve") in {"faible", "moyenne", "forte"}
        assert not item["appuis"], "un item non répondable ne cite rien"


def test_les_questions_ne_reprennent_pas_le_banc_v3(items, verifier):
    erreurs, pires = verifier.controle_nouveaute(items)
    assert erreurs == []
    assert max(p["jaccard_max"] for p in pires.values()) < 0.5


# ------------------------------------------------------------------ la promesse multi-passage


def test_un_item_multi_passage_declare_ne_peut_pas_se_resoudre_dans_un_seul_passage(verifier):
    """Le contrôle doit **pouvoir** échouer, et il doit échouer sur le bon motif.

    Panne évitée, et déjà survenue : la première version comptait *toutes* les ancres de
    l'item, faits ``souhaitable`` compris. Un item dont les faits obligatoires tiennent
    dans un passage et dont un fait facultatif pointe ailleurs passait pour multi-passage.
    """
    faux = {
        "id": "X01",
        "situations": ["deux_passages_meme_document"],
        "citation_minimale": {"documents_distincts": 1},
        "appuis": [
            {"ancre_id": "A1", "document_id": "d1", "chunk_id_origine": "c1"},
            {"ancre_id": "A2", "document_id": "d1", "chunk_id_origine": "c1"},
            {"ancre_id": "A3", "document_id": "d1", "chunk_id_origine": "c2"},
        ],
        "faits_attendus": [
            {"exigence": "obligatoire", "ancres": ["A1"]},
            {"exigence": "obligatoire", "ancres": ["A2"]},
            {"exigence": "souhaitable", "ancres": ["A3"]},
        ],
    }
    erreurs, _ = verifier.controle_multi_passage([faux])
    assert any("un seul passage servi" in e for e in erreurs), erreurs

    vrai = json.loads(json.dumps(faux))
    vrai["faits_attendus"][1]["ancres"] = ["A3"]
    erreurs, resume = verifier.controle_multi_passage([vrai])
    assert not [e for e in erreurs if "X01" in e], erreurs
    assert resume["par_item"]["X01"]["chunks"] == ["c1", "c2"]


def test_la_promesse_de_documents_distincts_se_mesure_sur_les_faits_obligatoires(verifier):
    faux = {
        "id": "X02",
        "situations": ["plusieurs_documents"],
        "citation_minimale": {"documents_distincts": 2},
        "appuis": [
            {"ancre_id": "A1", "document_id": "d1", "chunk_id_origine": "c1"},
            {"ancre_id": "A2", "document_id": "d2", "chunk_id_origine": "c2"},
        ],
        "faits_attendus": [
            {"exigence": "obligatoire", "ancres": ["A1"]},
            {"exigence": "souhaitable", "ancres": ["A2"]},
        ],
    }
    erreurs, _ = verifier.controle_multi_passage([faux])
    assert any("plusieurs documents" in e for e in erreurs), erreurs
    assert any("documents distincts en citation" in e for e in erreurs), erreurs


def test_un_item_d_abstention_ne_promet_aucun_passage(verifier):
    erreurs, resume = verifier.controle_multi_passage(
        [{"id": "X03", "situations": ["abstention_attendue"], "appuis": [], "faits_attendus": []}])
    assert not [e for e in erreurs if "X03" in e], erreurs
    assert resume["par_item"]["X03"]["faits_obligatoires"] == 0


# ------------------------------------------------------------------ la page


def test_la_page_citee_est_celle_d_un_lecteur(items):
    """Le `page_start` d'un chunk est faux pour une citation, et pour **deux** raisons.

    D'abord il est 0-basé : le bloc de titre porte `page_idx` 0, donc la page d'un lecteur
    vaut `page_idx + 1`. Ensuite il désigne la page où le **chunk** commence, pas celle où
    se trouve la **phrase citée** — un chunk qui enjambe une coupure de page les met sur
    deux pages différentes. Mesuré sur les deux lots, 49 ancres : 42 à un écart d'une page,
    5 de deux, 2 de trois. La seconde raison ne se corrige donc pas par un « +1 ».

    Quand les deux valeurs diffèrent, l'item conserve celle du chunk sous un nom explicite :
    l'écart est un défaut du chemin servi, et l'effacer le rendrait invisible.
    """
    ecarts = 0
    for item in items:
        for appui in item["appuis"]:
            couvertes = appui.get("pages_couvertes")
            assert couvertes, f"{item['id']}/{appui['ancre_id']} : aucune page couverte"
            assert appui["page"] in couvertes
            assert appui["page"] == min(couvertes)
            if "page_declaree_par_le_chunk" in appui:
                ecarts += 1
                # Le chunk commence avant l'ancre, et sa page est 0-basée : elle est donc
                # toujours **strictement inférieure** à la page citée. Exiger exactement −1
                # supposerait qu'aucun chunk n'enjambe une page, ce qui est faux.
                assert appui["page_declaree_par_le_chunk"] < appui["page"]
    assert ecarts > 0, "aucun écart : le défaut de page aurait donc disparu sans qu'on le sache"


def test_les_offsets_et_les_empreintes_tiennent(items):
    offsets = _module("hv1_offsets_c1", HUMAN / "offsets.py")
    assert offsets.verifier(items) == []


def test_la_normalisation_des_espaces_preserve_la_longueur():
    offsets = _module("hv1_offsets_len", HUMAN / "offsets.py")
    for texte in ("Lesmond et\xa0 al. 2004", "a b c\td", "rien à changer"):
        assert len(offsets.normaliser_espaces(texte)) == len(texte)


# ------------------------------------------------------------------ la fiche


def test_la_fiche_ne_montre_aucune_cle_interne_hors_annexe():
    """Un `chunk_id` dans une citation apprend au relecteur à en accepter une."""
    if not FICHE.exists():
        pytest.skip("paquet de revue non généré")
    texte = FICHE.read_text(encoding="utf-8")
    corps, _, annexe = texte.partition("## Annexe technique")
    assert annexe, "l'annexe technique doit exister et être séparée"
    fuites = re.findall(r"chunk-[0-9a-f]{8,}|block-[0-9a-f]{8,}", corps)
    assert fuites == [], f"clés internes dans le corps de la fiche : {sorted(set(fuites))[:5]}"


def test_la_fiche_porte_les_huit_rubriques_pour_chaque_item(items):
    if not FICHE.exists():
        pytest.skip("paquet de revue non généré")
    texte = FICHE.read_text(encoding="utf-8")
    assert texte.count("### 8. Décision du relecteur") == len(items)
    for attendu in ("relecteur (nom ou identifiant)", "date (AAAA-MM-JJ)",
                    "version de campagne", "temps réellement passé",
                    "justification de la décision", "désaccords"):
        assert texte.count(attendu) >= len(items), attendu
    for decision in ("VALIDER", "CORRIGER", "AMBIGU", "GARDER EN CANDIDAT", "RETIRER",
                     "DEMANDER UNE AUTRE SOURCE"):
        assert texte.count(decision) >= len(items), decision


def test_la_fiche_dit_ce_que_le_lot_est(items):
    if not FICHE.exists():
        pytest.skip("paquet de revue non généré")
    tete = FICHE.read_text(encoding="utf-8")[:1200]
    assert "écrites par un modèle" in tete
    assert "Aucune n'a été relue" in tete
    assert "rien n'est publiable" in tete
