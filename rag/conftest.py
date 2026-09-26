"""Test-suite guard for the parts that read the private corpus.

The repository ships code only. The corpus (418 research documents) is third-party
copyrighted material; its Qdrant export, the consolidated bibliographic metadata, the
ingestion registry, the benchmark question sets and the delivery fixtures are derived from
it. All of them stay private. The tests listed here read one of them: without it they are
reported as *skipped*, with the missing path, instead of failing. With the corpus on disk
they run unchanged. Every other test (the large majority) runs everywhere, CI included.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

#: What "the corpus is on disk" means. All of it, or the listed tests are skipped.
CORPUS = (
    "data/embeddings/ingested-all-qwen3-06b/rows.jsonl",
    "data/processed/ingested",
    "rag/metadata/documents-metadata-v1.json",
    "rag/metadata/duplicates-v1.json",
    "rag/ingestion/registry-v1.json",
    "rag/benchmark/questions-v3.jsonl",
    "rag/ingestion/tests/fixtures",
)

#: module -> None (every test in it) or the names of the tests that read the corpus.
NEEDS_CORPUS: dict[str, set[str] | None] = {
    "rag/benchmark/test_period_bounds.py": {
        "test_every_bench_question",
    },
    "rag/benchmark/tests/test_contrat_citation.py": {
        "test_l_ancrage_couvre_les_documents_des_fixtures",
        "test_le_lot_rend_autant_de_verdicts_que_de_demandes",
        "test_les_offsets_rendus_designent_bien_la_citation",
        "test_un_document_inconnu_le_dit_au_lieu_de_deviner",
        "test_une_citation_alteree_d_un_mot_est_refusee",
        "test_une_citation_inventee_est_refusee_sans_voisinage_trompeur",
        "test_une_citation_reelle_est_retrouvee_avec_ses_offsets",
    },
    "rag/benchmark/tests/test_garde_derive.py": {
        "test_le_manifeste_du_depot_dit_ce_que_la_collection_contient",
    },
    "rag/benchmark/tests/test_instrument_repare.py": {
        "test_la_population_ne_charge_aucun_index",
    },
    "rag/benchmark/tests/test_mcp_comptes.py": {
        "test_le_nombre_de_documents_sans_annee_exclut_les_documents_retires",
    },
    "rag/benchmark/tests/test_source_class.py": {
        "test_deux_appels_rendent_la_meme_chose",
        "test_la_regle_s_accorde_avec_les_provenances_declarees_a_la_main",
        "test_le_champ_est_inscrit_pour_chaque_document",
        "test_le_champ_inscrit_est_bien_celui_que_la_regle_derive",
        "test_le_registre_porte_la_meme_classe_que_les_metadonnees",
        "test_tous_les_documents_du_corpus_sont_classes",
    },
    "rag/benchmark/tests/test_v4_formule.py": {
        "test_aucune_accolade_orpheline_ne_subsiste",
        "test_l_or_est_retrouve_dans_un_passage_tronque_a_la_fenetre_servie",
        "test_la_compaction_naive_est_refusee_a_juste_titre",
        "test_la_population_de_l_etude_est_reelle_et_suffisante",
        "test_le_corpus_ecrit_bien_ses_mathematiques_caractere_par_caractere",
        "test_le_recouvrement_vaut_un_quand_la_formule_est_bien_la",
        "test_leurres_du_meme_document_le_controle_serre",
        "test_leurres_une_formule_d_un_autre_document_n_est_jamais_trouvee",
        "test_negatifs_chaque_alteration_change_vraiment_la_forme_canonique",
        "test_negatifs_les_quatre_familles_d_alteration_sont_toutes_exercees",
        "test_negatifs_un_seul_symbole_altere_suffit_a_faire_dire_faux",
        "test_positifs_le_scoreur_absorbe_la_reecriture",
        "test_positifs_police_retiree",
    },
    "rag/benchmark/tests/test_v4_tableau.py": {
        "test_chaque_or_est_une_valeur_lisible_et_non_parenthesee",
        "test_chaque_or_est_unique_dans_son_tableau",
        "test_chaque_or_tient_dans_la_fenetre_servie",
        "test_la_legende_est_lue_telle_que_l_overlay_l_ecrit",
        "test_la_selection_est_reproductible",
        "test_la_selection_rend_des_candidats_et_un_registre_de_rejets",
        "test_le_quota_par_document_est_respecte",
        "test_le_tableau_propre_du_corpus_est_analysable",
        "test_les_entetes_et_les_libelles_sont_distincts",
        "test_un_entete_multiniveau_est_refuse",
        "test_un_fragment_sans_ligne_de_donnees_est_refuse",
        "test_un_vrai_candidat_du_corpus_se_note_juste_sur_sa_propre_cellule",
    },
    "rag/ingestion/tests/test_aller_retour.py": {
        "test_aucun_index_bm25_de_signature_morte_ne_subsiste",
        "test_chaque_fichier_reecrit_garde_sa_forme",
        "test_l_arborescence_revient_a_l_identique",
        "test_l_inventaire_lexical_protege_le_vivant_et_le_gele",
        "test_l_overlay_des_tableaux_revient_octet_pour_octet",
        "test_la_purge_ne_supprime_rien_sans_vraiment",
        "test_la_purge_ne_touche_jamais_a_l_index_servi",
        "test_le_npz_partiel_ne_survit_pas_a_une_annulation",
        "test_le_staging_ne_survit_pas_a_un_echec_precoce",
        "test_le_verrou_tenu_se_dit_comme_le_reste_du_depot",
    },
    "rag/ingestion/tests/test_apply_delivery.py": None,
    "rag/ingestion/tests/test_signature_freshness.py": None,
}

#: Tests that exercise the real model loader (torch + a downloaded checkpoint).
NEEDS_TORCH: dict[str, set[str]] = {
    "rag/benchmark/tests/test_reclassement_selectif.py": {
        "test_le_repli_tient_avec_le_vrai_chargeur_pas_seulement_sous_mock",
    },
}


def _missing_corpus() -> str | None:
    for rel in CORPUS:
        if not (ROOT / rel).exists():
            return rel
    return None


def _selected(table: dict, rel: str, name: str) -> bool:
    if rel not in table:
        return False
    names = table[rel]
    return names is None or name in names


def pytest_collection_modifyitems(config, items):
    missing = _missing_corpus()
    has_torch = importlib.util.find_spec("torch") is not None
    for item in items:
        rel = Path(str(item.fspath)).resolve().relative_to(ROOT).as_posix()
        name = item.originalname or item.name
        if missing and _selected(NEEDS_CORPUS, rel, name):
            item.add_marker(pytest.mark.skip(
                reason=f"needs the private corpus ({missing} not found)"))
        elif not has_torch and _selected(NEEDS_TORCH, rel, name):
            item.add_marker(pytest.mark.skip(reason="needs torch and the reranker checkpoint"))
