"""Le banc v4 : son refus de comparer ce qui n'est pas comparable, et sa facturation.

Deux garanties que ce fichier transforme en tests, parce qu'elles ont chacune une histoire.

**Le refus de comparer.** Cinq champs d'en-tête gouvernent : signature, fenêtre servie, fenêtre
du **juge**, modèles, version des questions. Le piège que ce banc existe pour éviter est
double : le fil `characters` a jugé à fenêtre fixe 4 500 quand ``judge.grade`` a désormais pour
défaut 2 500 — un contrôle qui ne surveillerait que la fenêtre *servie* ne verrait rien. Et la
suite du 9 septembre fait passer la population de 109 à sa taille finale : les deux runs
portent des questions différentes, et rien ne doit permettre de les comparer par mégarde.

**La facturation.** ``_facturer`` tient ses comptes en mémoire et ne les écrit qu'à la fin.
La campagne du graphe, interrompue puis reprise, a rapporté **0,6319 USD** pour une dépense
réelle de **0,9094** : les appels du premier run n'étaient dans aucun fichier. Le cache, lui,
garde tout. Le critère de recette fixé au pré-enregistrement est *un run interrompu puis repris
rend le même coût qu'un run d'un seul tenant* — c'est ce que le dernier test vérifie, sur la
campagne réellement interrompue.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

BENCH = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BENCH))
sys.path.insert(0, str(BENCH.parent))

import banc_v4 as B  # noqa: E402


def en_tete(**remplacements):
    """Un en-tête plausible, modifiable champ par champ."""
    base = {
        "signature": "5530cba145",
        "fenetre_servie": 2500,
        "fenetre_du_juge": 2500,
        "modeles": {"generateur": "mistral-small-latest", "juge": "mistral-medium-latest",
                    "auteur": "gemini-3.1-flash-lite"},
        "version_questions": {"formula": "aaaa1111", "table_cell": "bbbb2222",
                              "negative_voisine": "cccc3333", "negative_v3": "dddd4444"},
    }
    base.update(remplacements)
    return base


def run(entete, scores):
    """Un résultat de run minimal : son en-tête et un détail par question."""
    return {**entete, "date": "2026-09-09",
            "detail": {cle: {"famille": "formula", "qid": cle.split("/")[1], "score": note,
                             "abstenue": False, "couverture_jugee": 2 * note}
                       for cle, note in scores.items()}}


# ------------------------------------------------------------------ les cinq champs gouvernants

def test_deux_en_tetes_identiques_ne_divergent_pas():
    assert B.divergences(en_tete(), en_tete()) == []


@pytest.mark.parametrize("champ,valeur", [
    ("signature", "0000000000"),
    ("fenetre_servie", 4500),
    ("fenetre_du_juge", 4500),
])
def test_chaque_champ_gouvernant_est_surveille(champ, valeur):
    assert B.divergences(en_tete(), en_tete(**{champ: valeur})) == [champ]


def test_la_fenetre_du_juge_est_surveillee_independamment_de_la_fenetre_servie():
    """Le piège nommé : deux runs qui servent 2 500 mais jugent 2 500 et 4 500.

    C'est exactement l'écart entre le banc v4 et le cache `characters`. Un contrôle qui ne
    regarderait que ce que voit le générateur les déclarerait comparables.
    """
    a = en_tete(fenetre_servie=2500, fenetre_du_juge=2500)
    b = en_tete(fenetre_servie=2500, fenetre_du_juge=4500)
    assert B.divergences(a, b) == ["fenetre_du_juge"]


def test_un_changement_de_modele_est_surveille():
    autre = {"generateur": "mistral-small-latest", "juge": "mistral-medium-latest",
             "auteur": "ministral-8b-latest"}
    assert B.divergences(en_tete(), en_tete(modeles=autre)) == ["modeles"]


def test_une_population_differente_est_surveillee():
    """Le cas de la suite du 9 septembre : la population grandit, les sha256 changent."""
    grandie = {"formula": "9999ffff", "table_cell": "8888eeee",
               "negative_voisine": "7777dddd", "negative_v3": "dddd4444"}
    assert B.divergences(en_tete(), en_tete(version_questions=grandie)) == ["version_questions"]


# ------------------------------------------------------------------ le refus, et le drapeau

def test_comparer_refuse_deux_populations_differentes():
    """Sans ``--drapeau``, la comparaison s'arrête. Elle ne rend pas un chiffre douteux."""
    ancien = run(en_tete(), {"formula/f001": 1, "formula/f002": 0})
    nouveau = run(en_tete(version_questions={"formula": "9999ffff", "table_cell": "8888eeee",
                                             "negative_voisine": "7777dddd",
                                             "negative_v3": "dddd4444"}),
                  {"formula/f001": 1, "formula/f002": 1})
    with pytest.raises(SystemExit) as arret:
        B.comparer(nouveau, ancien, drapeau=False)
    assert "version_questions" in str(arret.value)


def test_le_drapeau_autorise_la_comparaison_et_ECRIT_l_ecart():
    """Le drapeau ne rend pas la comparaison légitime : il la rend **traçable**.

    Ce qui compte n'est pas qu'elle passe, c'est que le JSON porte *ce qui* diffère — sans
    quoi un lecteur trouverait un Δ sans savoir qu'il enjambe deux instruments.
    """
    ancien = run(en_tete(), {"formula/f001": 1, "formula/f002": 0})
    nouveau = run(en_tete(version_questions={"formula": "9999ffff", "table_cell": "8888eeee",
                                             "negative_voisine": "7777dddd",
                                             "negative_v3": "dddd4444"}),
                  {"formula/f001": 1, "formula/f002": 1})
    sortie = B.comparer(nouveau, ancien, drapeau=True)
    assert sortie["divergences_d_en_tete"] == ["version_questions"]
    assert sortie["drapeau_leve"] is True
    assert sortie["par_famille"]["formula"]["n"] == 2
    assert sortie["par_famille"]["formula"]["delta"] == 0.5


def test_deux_runs_du_meme_instrument_se_comparent_sans_drapeau():
    ancien = run(en_tete(), {"formula/f001": 1, "formula/f002": 0})
    nouveau = run(en_tete(), {"formula/f001": 1, "formula/f002": 1})
    sortie = B.comparer(nouveau, ancien, drapeau=False)
    assert sortie["divergences_d_en_tete"] == []
    assert sortie["drapeau_leve"] is False
    assert sortie["par_famille"]["formula"]["delta"] == 0.5


def test_la_comparaison_est_appariee_sur_les_seules_questions_communes():
    """Une question qui n'existe que d'un côté n'entre pas dans le Δ. Sinon il ne serait pas apparié."""
    ancien = run(en_tete(), {"formula/f001": 0, "formula/f002": 0})
    nouveau = run(en_tete(), {"formula/f001": 1, "formula/f003": 1})
    sortie = B.comparer(nouveau, ancien, drapeau=False)
    assert sortie["par_famille"]["formula"]["n"] == 1


# ------------------------------------------------------------------ la facturation

def test_le_chiffrage_impute_au_bon_modele():
    """Multiplier un total de jetons par un seul prix se trompe d'un facteur dix ici."""
    petit = B.chiffrer({"mistral-small-latest": {"appels": 1, "cache": 0,
                                                 "entree": 1_000_000, "sortie": 0}})
    grand = B.chiffrer({"mistral-medium-latest": {"appels": 1, "cache": 0,
                                                  "entree": 1_000_000, "sortie": 0}})
    assert petit["total_usd"] == 0.15
    assert grand["total_usd"] == 1.5


def test_un_modele_sans_tarif_est_signale_et_non_compte_a_zero_en_silence():
    sortie = B.chiffrer({"modele-inconnu": {"appels": 1, "cache": 0, "entree": 1_000, "sortie": 1}})
    assert sortie["par_modele"]["modele-inconnu"]["tarif_connu"] is False


# ------------------------------------------------- la fenêtre du juge fait partie de l'identité

ITEM_POSITIF = {"kind": "positive", "famille": "formula", "qid": "f001",
                "question": "Quelle est la variance du modèle ?",
                "answer_facts": ["sigma^2 = 0,04"]}
CONTEXTE_LONG = [{"title": "T", "section": "S", "text": "x" * 9_000}]


def test_la_cle_du_juge_change_avec_sa_fenetre():
    """Le défaut que la re-notation aurait rencontré, réduit à son noyau.

    ``judge.grade`` reçoit ``characters``. Deux fenêtres du juge sur un passage plus long
    qu'elles produisent deux prompts différents, donc deux clés de cache différentes. Le cache
    des verdicts, lui, était indexé par ``famille/qid`` seul : re-noter à 10 000 aurait retrouvé
    le verdict rendu à 2 500 et sauté l'appel.
    """
    etroite = B._cle_juge(ITEM_POSITIF, "réponse", CONTEXTE_LONG, "v4-formula/f001", 2_500, B.JUGE)
    large = B._cle_juge(ITEM_POSITIF, "réponse", CONTEXTE_LONG, "v4-formula/f001", 10_000, B.JUGE)
    assert etroite != large, "la fenêtre du juge ne change pas la clé — le cache est aveugle"


def test_un_verdict_rendu_a_une_autre_fenetre_n_appartient_pas_au_run():
    """Aucun fichier de cache n'existe pour ce prompt : le verdict doit être refait."""
    assert B._verdict_appartient(ITEM_POSITIF, CONTEXTE_LONG, "réponse jamais notée",
                                 "v4-formula/f001", 10_000) is False


def test_un_verdict_est_reconnu_quand_son_prompt_a_bien_ete_paye():
    """Et le chemin inverse : si l'appel a eu lieu à cette fenêtre, on ne le repaie pas.

    Le fichier est écrit dans le vrai cache puis retiré : son prompt est synthétique, aucune
    entrée réelle ne porte cette empreinte.
    """
    chemin = B._cle_juge(ITEM_POSITIF, "réponse", CONTEXTE_LONG, "v4-formula/f001", 10_000, B.JUGE)
    chemin.parent.mkdir(parents=True, exist_ok=True)
    chemin.write_text(json.dumps({"content": "{}"}), encoding="utf-8")
    try:
        assert B._verdict_appartient(ITEM_POSITIF, CONTEXTE_LONG, "réponse",
                                     "v4-formula/f001", 10_000) is True
    finally:
        chemin.unlink()


def test_facturer_depuis_le_cache_compte_les_manquants():
    """Un appel introuvable doit être **compté**, jamais ignoré.

    Un silence rendrait un coût sous-estimé qui a l'air d'un coût — c'est-à-dire le défaut
    qu'on répare, sous une autre forme.
    """
    inexistant = BENCH / ".cache" / "llm" / ("0" * 32 + ".json")
    sortie = B.facturer_depuis_le_cache([("mistral-small-latest", inexistant)])
    assert sortie["manquants"] == 1
    assert sortie["registre"] == {}


CAMPAGNE_GRAPHE = BENCH / ".cache" / "graphe-verdicts-5530cba145.json"


@pytest.mark.skipif(not CAMPAGNE_GRAPHE.exists(),
                    reason="le cache de la campagne du graphe n'est pas présent")
def test_recette_un_run_interrompu_rend_le_meme_cout_qu_un_run_d_un_seul_tenant():
    """Le critère de recette de l'amendement n° 8, sur la campagne réellement interrompue.

    La campagne du graphe a été tuée à mi-course puis reprise. Son registre en mémoire ne
    porte que la reprise — **0,6319 USD**. Reconstruit depuis le cache, le coût des 240 appels
    est **0,9094 USD**, et aucun appel ne manque. C'est l'écart que ce test fixe : la
    facturation par le cache ne dépend pas du découpage du run.
    """
    import apport_graphe as A  # noqa: E402
    import judge  # noqa: E402
    import pipeline  # noqa: E402

    population = json.loads((BENCH / ".cache" / f"graphe-population-{B.SIGNATURE}.json")
                            .read_text(encoding="utf-8"))
    reponses = json.loads((BENCH / ".cache" / f"graphe-reponses-{B.SIGNATURE}.json")
                          .read_text(encoding="utf-8"))

    def appels():
        for qid in sorted(population["contextes"]):
            for bras in A.BRAS:
                contexte = population["contextes"][qid][bras]
                question = population["contextes"][qid]["question"]
                yield A.GENERATEUR, B._cle_generation(question, contexte, A.FENETRE_SERVIE,
                                                      A.GENERATEUR)
                yield A.JUGE, B._cle_juge(population["items"][qid],
                                          reponses[f"{qid}/{bras}"]["answer"], contexte,
                                          f"graphe-{qid}/{bras}", A.FENETRE_JUGE, A.JUGE)

    sortie = B.facturer_depuis_le_cache(appels())
    assert sortie["manquants"] == 0, "la reconstruction des clés a dérivé du code appelant"
    total = B.chiffrer(sortie["registre"])["total_usd"]
    assert total == pytest.approx(0.9094, abs=0.0005), (
        f"coût reconstruit {total} — le registre en mémoire disait 0,6319")
    assert sum(l["appels"] for l in sortie["registre"].values()) == 240


LIGNE_DE_BASE = BENCH / f"results-v4-{B.SIGNATURE}.json"


@pytest.mark.skipif(not LIGNE_DE_BASE.exists(), reason="la ligne de base n'est pas présente")
def test_la_ligne_de_base_publie_le_cout_DU_CACHE_et_non_du_registre():
    """Écrire la fonction ne suffit pas : il faut qu'elle soit **branchée** sur ce qu'on publie.

    `facturer_depuis_le_cache` existait déjà et était testée, mais `verdict` publiait encore
    `chiffrage`, c'est-à-dire le registre en mémoire — celui qui a menti de 0,6319 contre
    0,9094. Une correction non branchée est une correction qui n'a pas eu lieu ; ce test la
    tient sur le chemin qui produit l'artefact.

    Il vérifie aussi que les deux chiffres restent **nommés séparément**. Ils ne mesurent pas
    la même chose : le cache dit ce que coûte de produire le run entier à froid, le registre
    ce que ce processus-ci a payé. Les additionner ou les confondre serait une faute de plus.
    """
    cout = json.loads(LIGNE_DE_BASE.read_text(encoding="utf-8"))["cout"]
    assert "cache" in cout["source"], "le coût publié ne vient pas du cache disque"
    assert cout["appels_manquants"] == 0, (
        f"{cout['appels_manquants']} appels du run introuvables dans le cache — "
        "le coût publié est sous-estimé")
    assert cout["marginal_de_ce_processus_usd"] <= cout["total_usd"], (
        "ce qu'un processus a payé ne peut pas dépasser le coût du run entier")
    assert cout["deja_en_cache_usd"] == pytest.approx(
        cout["total_usd"] - cout["marginal_de_ce_processus_usd"], abs=0.0002)


# ------------------------------------------------------------------ le cache appartient-il à la question ?

#: Le contrat de réponse sous lequel ces tests écrivent et relisent leur cache. Il est
#: **nommé** : la version du prompt entre dans la clé de cache depuis le chantier
#: `contrat-de-reponse`, et un test qui laisserait le défaut décider passerait à côté
#: du jour où ce défaut change.
CONTRAT = "v2"

CONTEXTE = [{"chunk_id": "c1", "document_id": "d1", "title": "T", "section": "s",
             "short_ref": "A 2020", "text": "The realised Sharpe ratio is 1.42."}]


def _ecrire_cache_generation(monkeypatch, tmp_path, question, contexte, reponse,
                             prompt=CONTRAT):
    """Écrit dans un faux cache LLM la réponse que ``pipeline.answer`` y aurait mise."""
    import llm  # noqa: E402
    monkeypatch.setattr(llm, "CACHE", tmp_path)
    chemin = B._cle_generation(question, contexte, 2500, B.GENERATEUR, prompt)
    chemin.parent.mkdir(parents=True, exist_ok=True)
    chemin.write_text(json.dumps({"model": B.GENERATEUR, "content": reponse, "usage": {}}),
                      encoding="utf-8")
    return chemin


def test_une_reponse_appartient_a_la_question_qui_l_a_produite(monkeypatch, tmp_path):
    question = "What is the realised Sharpe ratio of the strategy?"
    _ecrire_cache_generation(monkeypatch, tmp_path, question, CONTEXTE, "It is 1.42 [1].")
    item = {"question": question}
    assert B._reponse_appartient(item, CONTEXTE, {"answer": "It is 1.42 [1]."}, 2500, CONTRAT) is True


def test_une_reponse_n_appartient_PAS_a_une_autre_question(monkeypatch, tmp_path):
    """Le cas qui a motivé le contrôle : une famille renumérotée.

    La question n° 7 d'hier devient la n° 9 d'aujourd'hui ; le cache, indexé par le numéro,
    rendrait la réponse de l'autre. Mesuré sur la population du 9 septembre : **20 entrées**
    dans ce cas. Sans ce contrôle, vingt questions auraient été notées avec la réponse d'une
    autre — et rien, dans le JSON de sortie, ne l'aurait laissé voir.
    """
    ancienne = "What is the realised Sharpe ratio of the strategy?"
    nouvelle = "What is the maximum drawdown of the strategy?"
    _ecrire_cache_generation(monkeypatch, tmp_path, ancienne, CONTEXTE, "It is 1.42 [1].")
    item = {"question": nouvelle}
    assert B._reponse_appartient(item, CONTEXTE, {"answer": "It is 1.42 [1]."}, 2500, CONTRAT) is False


def test_une_reponse_sans_trace_dans_le_cache_n_est_pas_creditee(monkeypatch, tmp_path):
    """Faute de preuve, on refait. Le doute ne se résout pas en faveur de l'économie."""
    import llm  # noqa: E402
    monkeypatch.setattr(llm, "CACHE", tmp_path)
    item = {"question": "Anything at all, never generated?"}
    assert B._reponse_appartient(item, CONTEXTE, {"answer": "whatever"}, 2500, CONTRAT) is False


def test_un_contexte_different_invalide_la_reponse(monkeypatch, tmp_path):
    """Même question, autres passages : ce n'est pas la même mesure.

    C'est le cas du chantier 1 quand la fenêtre servie change, et celui d'une famille dont le
    retrieval a bougé : la réponse gardée a été produite sur d'autres passages.
    """
    question = "What is the realised Sharpe ratio of the strategy?"
    _ecrire_cache_generation(monkeypatch, tmp_path, question, CONTEXTE, "It is 1.42 [1].")
    autre = [{**CONTEXTE[0], "text": "The maximum drawdown observed is 0.318."}]
    item = {"question": question}
    assert B._reponse_appartient(item, autre, {"answer": "It is 1.42 [1]."}, 2500, CONTRAT) is False


# ------------------------------------------------------------------ survie à un changement de corpus

def test_un_tableau_dont_l_en_tete_se_vide_ne_survit_pas():
    """Le contrôle qui a vu la régression des en-têtes, sur un cas minimal.

    La valeur d'or reste présente dans les deux textes : un contrôle qui se contente de
    chercher la valeur passerait. Ce qui meurt, c'est la **colonne** — donc la question, qui
    demande une cellule à l'intersection d'une ligne et d'un en-tête nommé. C'est la
    distinction que `survie_questions` tient et qu'un balayage littéral manque.
    """
    import survie_questions as S  # noqa: E402

    lignes = "| 9:11 | 100 | 200 | 73800 |\n| 9:12 | 101 | 201 | 73900 |\n| 9:13 | 102 | 202 | 74000 |\n"
    avant = f"Table: T\n\n| time | pb3 | pb4 | pb5 |\n|---|---|---|---|\n{lignes}"
    apres = f"Table: T\n\n| time | price | size |   |\n|---|---|---|---|\n{lignes}"
    assert "73800" in avant and "73800" in apres, "la valeur survit — c'est tout le piège"
    assert S._tableau(avant) == (True, "utilisable")
    assert S._tableau(apres) == (False, "entetes_vides")


ARTEFACT_SURVIE = BENCH / "survie-questions-5530cba145-vers-e1bdf36e2e.json"


@pytest.mark.skipif(not ARTEFACT_SURVIE.exists(), reason="le rapport de survie n'est pas présent")
def test_la_survie_des_questions_au_corpus_e1bdf36e2e_est_figee():
    """Fige le verdict du 9 septembre, pour qu'une dérive silencieuse se voie.

    `formula` et `negative_voisine` passent intactes ; cinq `table_cell` tombent. Si ce
    fichier change sans qu'on l'ait décidé, c'est que le corpus ou les questions ont bougé.
    """
    rapport = json.loads(ARTEFACT_SURVIE.read_text(encoding="utf-8"))
    assert rapport["corpus"]["identifiants_disparus"] == 0
    assert rapport["corpus"]["identifiants_apparus"] == 0
    assert rapport["formula"] == {"n": 45, "survivantes": 45}
    assert rapport["negative_voisine"] == {"n": 34, "survivantes": 34}
    assert rapport["table_cell"] == {"n": 100, "survivantes": 95}
    assert [m["qid"] for m in rapport["mortes"]["table_cell"]] == [
        "t036", "t056", "t081", "t088", "t092"]
    # La correction reste largement bénéfique : c'est la moitié du verdict, et l'omettre
    # ferait lire « 29 tableaux cassés » comme un réquisitoire.
    assert rapport["corpus"]["tableaux_gagnes"] > rapport["corpus"]["tableaux_perdus"]


# ------------------------------------------------------------------ la version du contrat, dans la clé

def test_deux_contrats_ne_partagent_pas_une_cle_de_cache(monkeypatch, tmp_path):
    """Le piège que le chantier ``contrat-de-reponse`` a failli laisser ouvert.

    ``_cle_generation`` lisait ``pipeline.DEFAULT_PROMPT`` en dur. Un run mené sous ``v3``
    aurait donc reconstruit la clé du prompt **par défaut**, déclaré périmée chacune des
    199 réponses qu'il venait d'écrire, **jeté leurs contextes avec elles**, et repayé le run
    entier à chaque relance — sans qu'aucun message ne dise autre chose que « 199 réponses
    invalidées ». La panne aurait été silencieuse *et* coûteuse, la pire combinaison.
    """
    import llm  # noqa: PLC0415
    monkeypatch.setattr(llm, "CACHE", tmp_path)
    question = "What is the realised Sharpe ratio of the strategy?"
    cles = {p: B._cle_generation(question, CONTEXTE, 2500, B.GENERATEUR, p)
            for p in ("v1", "v2", "v3")}
    assert len({c.name for c in cles.values()}) == 3


def test_une_reponse_ecrite_sous_un_contrat_n_appartient_pas_a_un_autre(monkeypatch, tmp_path):
    """Le corollaire, du côté qui décide de repayer : la réponse v2 n'est pas la réponse v3."""
    question = "What is the realised Sharpe ratio of the strategy?"
    _ecrire_cache_generation(monkeypatch, tmp_path, question, CONTEXTE, "It is 1.42 [1].",
                             prompt="v2")
    item = {"question": question}
    assert B._reponse_appartient(item, CONTEXTE, {"answer": "It is 1.42 [1]."}, 2500, "v2") is True
    assert B._reponse_appartient(item, CONTEXTE, {"answer": "It is 1.42 [1]."}, 2500, "v3") is False


def test_la_version_du_prompt_gouverne_la_comparabilite():
    """Deux contrats ne se comparent pas en silence : ``divergences`` doit le dire.

    Les cinq champs d'origine sont identiques entre les deux bras — même corpus, mêmes fenêtres,
    mêmes modèles, mêmes questions. Sans le sixième, le banc aurait comparé v2 et v3 sans un mot,
    alors que le texte du contrat est précisément la variable mesurée.
    """
    assert "version_prompt" in B.CHAMPS_GOUVERNANTS
    a = B.en_tete(10000, 10000, "v2")
    b = B.en_tete(10000, 10000, "v3")
    assert B.divergences(a, b) == ["version_prompt"]
    assert B.divergences(a, a) == []


def test_l_empreinte_du_prompt_porte_sur_le_texte_et_non_sur_son_nom():
    """Renommer une version ne trompe personne ; en réécrire une phrase, si."""
    import pipeline  # noqa: PLC0415
    assert (B.en_tete(10000, 10000, "v3")["version_prompt"]["sha256_16"]
            == pipeline.empreinte_prompt("v3"))
    assert (pipeline.empreinte_prompt("v2") != pipeline.empreinte_prompt("v3"))
