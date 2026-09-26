"""L'instrument réparé : bras appariés dans un même run, plancher obligatoire, gardes orientées.

Ce que ces tests ferment, et ce que ça avait coûté
---------------------------------------------------
Trois chantiers consécutifs (`integration`, `representation`, `contrat-de-reponse`) ont publié
des Δ au niveau réponse avant qu'on sache que **le générateur n'est pas reproductible** :
``mistral-small-latest`` à ``temperature=0`` rend 51/130, 54/130 puis 110/199 réponses
identiques à prompt identique au bit près. Le bras placebo du lot 6 a ensuite reproduit, **à
variable nulle**, exactement le seul Δ significatif publié : ``+0,1176 [+0,029 ; +0,235]`` sur
les fabrications. Autrement dit : le banc ne pouvait pas distinguer un effet de son propre
bruit, et rien dans ses fichiers ne le disait.

Chacun de ces tests échoue sur le banc du 9 septembre 2026 — soit parce que la fonction
n'existe pas (``analyser_bras``, ``plancher_requis``, ``_test_exact``), soit parce que le
comportement était l'inverse (la garde ``a`` sur les négatives). Ils tournent hors ligne, sans
corpus, sans réseau et sans un centime d'appel.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

BENCHMARK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCHMARK))
sys.path.insert(0, str(BENCHMARK.parent))

import banc_v4  # noqa: E402
import llm  # noqa: E402
import pipeline  # noqa: E402


# --------------------------------------------------------------------------- §4a : les bras

def test_le_premier_bras_est_la_reference_et_les_autres_des_variantes():
    bras = banc_v4.analyser_bras("a1:fenetre=2500,a2:fenetre=10000", 2500)
    assert [b.nom for b in bras] == ["a1", "a2"]
    assert [b.role for b in bras] == ["reference", "variante"]
    assert [b.fenetre_servie for b in bras] == [2500, 10000]
    assert all(b.prompt == pipeline.DEFAULT_PROMPT for b in bras)


def test_un_bras_nomme_comme_un_contrat_prend_ce_contrat():
    """`--bras v2,v4` doit valoir `--prompt v2` contre `--prompt v4`, sans le dire deux fois."""
    bras = banc_v4.analyser_bras("v2,v4", 2500)
    assert [b.prompt for b in bras] == ["v2", "v4"]


def test_le_placebo_herite_du_contrat_ET_de_la_fenetre_de_la_reference():
    bras = banc_v4.analyser_bras("ref:prompt=v2:fenetre=7000,plancher:placebo", 2500)
    reference, placebo = bras
    assert placebo.prompt == reference.prompt == "v2"
    assert placebo.fenetre_servie == reference.fenetre_servie == 7000
    assert placebo.role == "placebo" and placebo.jetable is True
    assert reference.jetable is False


def test_un_placebo_qui_porte_une_variable_est_refuse():
    """Le cœur du §4b : un « placebo » qui diffère par quelque chose n'est pas un placebo.

    Et l'erreur serait **rassurante**, ce qui la rend pire : un plancher élargi par une vraie
    variable rendrait tous les Δ non concluants, donc silencieux. Une faute de frappe suffirait.
    """
    with pytest.raises(SystemExit) as echec:
        banc_v4.analyser_bras("ref:fenetre=2500,faux:placebo:fenetre=10000", 2500)
    assert "ne diffère PAR RIEN" in str(echec.value)


def test_le_placebo_ne_peut_pas_etre_le_premier_bras_D_UN_RUN_A_PLUSIEURS_BRAS():
    with pytest.raises(SystemExit):
        banc_v4.analyser_bras("plancher:placebo,ref", 2500)


def test_un_bras_SEUL_peut_se_declarer_placebo():
    """Au verdict il n'y a qu'un bras à la fois : c'est ainsi qu'on lui dit qu'il EST le plancher.

    Sans cette porte, le placebo du 9 septembre (``v2placebo``, tiré avant que le manifeste de
    run existe) se verrait réclamer un plancher à lui — et le seul moyen de le publier serait
    ``--sans-plancher``, qui le marquerait « non concluant » alors qu'il est la mesure du bruit.
    """
    seul, = banc_v4.analyser_bras("v2placebo:placebo", 2500)
    assert seul.role == "placebo"
    assert banc_v4.plancher_requis(seul, {}, plancher=None, sans_plancher=False) is False


def test_deux_bras_ne_peuvent_pas_porter_le_meme_nom():
    """Ils partageraient leurs fichiers d'état, et le second écraserait le premier."""
    with pytest.raises(SystemExit) as echec:
        banc_v4.analyser_bras("a,a:fenetre=10000", 2500)
    assert "même nom" in str(echec.value)


def test_un_seul_plancher_par_run():
    with pytest.raises(SystemExit):
        banc_v4.analyser_bras("ref,p1:placebo,p2:placebo", 2500)


def test_un_contrat_inconnu_est_refuse_avant_toute_depense():
    with pytest.raises(SystemExit):
        banc_v4.analyser_bras("ref:prompt=v99", 2500)


def test_les_fichiers_d_etat_sont_nommes_par_le_BRAS_et_non_par_le_contrat():
    """Sans quoi un placebo — même contrat que sa référence — écraserait la référence."""
    bras = banc_v4.analyser_bras("ref:prompt=v4,plancher:placebo", 2500)
    chemins = {b.nom: banc_v4._cache_reponses(b.nom) for b in bras}
    assert chemins["ref"] != chemins["plancher"]
    assert chemins["ref"].name.endswith("-ref.json")
    assert chemins["plancher"].name.endswith("-plancher.json")


def test_un_bras_nomme_par_son_contrat_garde_le_chemin_d_avant_l_option():
    """Compatibilité : les artefacts publiés le 9 septembre doivent rester lus."""
    assert banc_v4._cache_reponses("v2").name == f"v4-reponses-{banc_v4.SIGNATURE}-v2.json"


def test_le_cache_d_appels_est_detourne_puis_remis():
    """La propriété qui rend un placebo possible : ni servi par le cache du dépôt, ni écrit dedans."""
    avant = llm.CACHE
    ailleurs = Path("/tmp/cache-de-test-du-banc")
    with banc_v4.cache_d_appels(ailleurs):
        assert llm.CACHE == ailleurs
        interne = llm._cache_path({"model": "m", "messages": []})
        assert interne.parent == ailleurs
    assert llm.CACHE == avant
    assert llm._cache_path({"model": "m", "messages": []}).parent == avant


def test_le_cache_d_appels_est_remis_meme_si_le_bras_echoue():
    avant = llm.CACHE
    with pytest.raises(ValueError):
        with banc_v4.cache_d_appels(Path("/tmp/peu-importe")):
            raise ValueError("le bras a explosé")
    assert llm.CACHE == avant


# ------------------------------------------------- §4a : l'entrelacement, mesuré sur l'ordre

def _banc_hors_ligne(monkeypatch, tmp_path, reponses_par_appel):
    """Un banc sans corpus, sans Qdrant et sans réseau : seul l'ORDRE des appels nous intéresse."""
    monkeypatch.setattr(banc_v4, "CACHE", tmp_path)
    (tmp_path / f"v4-temoins-{banc_v4.SIGNATURE}.json").write_text(
        json.dumps({"lisible": True, "accuracy": 1.0}), encoding="utf-8")
    items = [{"qid": f"q{n}", "qid_banc": f"table_cell/q{n}", "famille": "table_cell",
              "question": f"question {n}", "or_valeur": "1", "gold_chunks": ["c1"]}
             for n in range(3)]
    monkeypatch.setattr(banc_v4, "_index", lambda: None)
    monkeypatch.setattr(banc_v4, "population", lambda index=None: items)
    monkeypatch.setattr(banc_v4, "servis", lambda question: [
        {"chunk_id": "c1", "document_id": "d1", "title": "t", "short_ref": "r", "section": "s",
         "page_start": 1, "text": "1", "content_type": "table", "score": 1.0}])
    monkeypatch.setattr(pipeline, "answer", reponses_par_appel)
    return items


def test_les_bras_sont_entrelaces_PAR_QUESTION_et_non_joues_l_un_apres_l_autre(monkeypatch, tmp_path):
    """La réparation centrale du §4a.

    Jouer le bras A en entier puis le bras B en entier compare deux **moments** : deux états du
    service distant, deux files d'attente, deux tirages d'un générateur non reproductible.
    L'ordre attendu est donc (q1,A) (q1,B) (q2,A) (q2,B)… et non (q1,A) (q2,A) … (q1,B) (q2,B).
    """
    ordre = []

    def repondre(question, contexte, model=None, prompt=None, characters=None, **_):
        ordre.append((question, characters))
        return {"answer": "1", "abstained": False}

    _banc_hors_ligne(monkeypatch, tmp_path, repondre)
    bras = banc_v4.analyser_bras("a:fenetre=2500,b:fenetre=9000", 2500)
    banc_v4.mesure(bras, fenetre_juge=2500)

    assert ordre == [("question 0", 2500), ("question 0", 9000),
                     ("question 1", 2500), ("question 1", 9000),
                     ("question 2", 2500), ("question 2", 9000)]


def test_les_bras_partagent_le_MEME_contexte_question_par_question(monkeypatch, tmp_path):
    """C'est la condition de l'appariement : la récupération est faite une fois, pas une par bras."""
    recuperations = []
    monkeypatch.setattr(banc_v4, "CACHE", tmp_path)
    (tmp_path / f"v4-temoins-{banc_v4.SIGNATURE}.json").write_text(
        json.dumps({"lisible": True}), encoding="utf-8")
    items = [{"qid": "q0", "qid_banc": "table_cell/q0", "famille": "table_cell",
              "question": "q", "or_valeur": "1", "gold_chunks": ["c1"]}]
    monkeypatch.setattr(banc_v4, "_index", lambda: None)
    monkeypatch.setattr(banc_v4, "population", lambda index=None: items)

    def recuperer(question):
        recuperations.append(question)
        return [{"chunk_id": "c1", "document_id": "d1", "title": "t", "short_ref": "r",
                 "section": "s", "page_start": 1, "text": "1", "content_type": "table",
                 "score": 1.0}]

    monkeypatch.setattr(banc_v4, "servis", recuperer)
    monkeypatch.setattr(pipeline, "answer",
                        lambda *a, **k: {"answer": "1", "abstained": False})
    banc_v4.mesure(banc_v4.analyser_bras("a,b:fenetre=9000,p:placebo", 2500), fenetre_juge=2500)
    assert recuperations == ["q"], "la récupération doit être faite UNE fois pour les trois bras"


def test_un_bras_jetable_n_ecrit_rien_dans_le_cache_du_depot(monkeypatch, tmp_path):
    """La leçon du lot 6 : un placebo qui pollue le cache détruit ce qu'il mesure."""
    vus = []

    def repondre(question, contexte, model=None, prompt=None, characters=None, **_):
        vus.append(llm.CACHE)
        return {"answer": "1", "abstained": False}

    _banc_hors_ligne(monkeypatch, tmp_path, repondre)
    cache_du_depot = llm.CACHE
    banc_v4.mesure(banc_v4.analyser_bras("ref,plancher:placebo", 2500), fenetre_juge=2500)

    reference = [c for c in vus[0::2]]
    placebo = [c for c in vus[1::2]]
    assert all(c == cache_du_depot for c in reference)
    assert all(c != cache_du_depot for c in placebo)
    assert len(set(placebo)) == 1, "un seul cache temporaire pour tout le bras placebo"
    assert llm.CACHE == cache_du_depot, "le cache du dépôt doit être remis à la fin"


def test_un_bras_jetable_ecrit_son_etat_AU_FIL_DE_L_EAU(monkeypatch, tmp_path):
    """Trouvé par une interruption réelle, le 10 septembre 2026.

    Un run de trois bras a été tué à 76/199 par une pression mémoire venue d'une autre session.
    Les deux bras ordinaires ont survécu — leur état est écrit à chaque question. **Le placebo
    avait tout perdu** : son état n'était écrit qu'à la fin, alors que ses 121 appels étaient
    payés. « Jetable » qualifie son cache d'appels, pas son travail.
    """
    appels = []

    def repondre(question, contexte, model=None, prompt=None, characters=None, **_):
        appels.append(question)
        if len(appels) > 4:                      # interruption au milieu de la 3e question
            raise KeyboardInterrupt("tué par le système")
        return {"answer": "1", "abstained": False}

    _banc_hors_ligne(monkeypatch, tmp_path, repondre)
    with pytest.raises(KeyboardInterrupt):
        banc_v4.mesure(banc_v4.analyser_bras("ref,plancher:placebo", 2500), fenetre_juge=2500)

    survivant = json.loads(banc_v4._cache_reponses("plancher").read_text(encoding="utf-8"))
    assert len(survivant) == 2, "le placebo doit avoir gardé les questions déjà payées"


def test_une_reprise_de_plancher_ne_remet_PAS_l_etat_a_zero(monkeypatch, tmp_path):
    """`--plancher-cache DIR` est la seule porte de sortie de « un placebo repart de zéro ».

    Elle est explicite par construction : sans l'option, le comportement ne change pas d'un
    octet. Avec elle, le placebo reprend sur SES propres tirages — la seule reprise correcte,
    puisqu'un placebo tiré un autre jour ne mesure pas le bruit de celui-ci.
    """
    reprise = tmp_path / "cache-repris"
    (reprise / "plancher").mkdir(parents=True)
    monkeypatch.setattr(llm, "CACHE", tmp_path / "llm-du-depot")
    appels = []

    def repondre(question, contexte, model=None, prompt=None, characters=None, **_):
        # Le faux générateur écrit son entrée de cache, comme le vrai. Sans cela, le contrôle de
        # péremption de l'étage 0 ne trouve rien et déclare TOUTES les réponses périmées — le
        # banc d'essai ne pourrait alors exercer aucune reprise, ni celle-ci ni celle des bras
        # ordinaires.
        appels.append(question)
        chemin = banc_v4._cle_generation(question, contexte, characters, banc_v4.GENERATEUR,
                                         prompt)
        chemin.parent.mkdir(parents=True, exist_ok=True)
        chemin.write_text(json.dumps(
            {"content": "1", "usage": {"prompt_tokens": 10, "completion_tokens": 2}}),
            encoding="utf-8")
        return {"answer": "1", "abstained": False}

    _banc_hors_ligne(monkeypatch, tmp_path, repondre)

    # 1er run : il remplit le cache repris et l'état du placebo.
    banc_v4.mesure(banc_v4.analyser_bras("ref,plancher:placebo", 2500), fenetre_juge=2500,
                   cache_jetable_repris=reprise)
    assert len(appels) == 6, "3 questions x 2 bras"
    assert len(json.loads(banc_v4._cache_reponses("plancher").read_text("utf-8"))) == 3

    # 2e run AVEC reprise : le placebo retrouve ses tirages, il n'en paie aucun.
    appels.clear()
    banc_v4.mesure(banc_v4.analyser_bras("ref,plancher:placebo", 2500), fenetre_juge=2500,
                   cache_jetable_repris=reprise)
    assert appels == [], "rien ne doit être repayé : les deux bras reprennent sur leur état"
    assert reprise.exists(), "un répertoire repris appartient à l'opérateur, on ne l'efface pas"

    # 3e run SANS reprise : le placebo repart de zéro, la référence non.
    appels.clear()
    banc_v4.mesure(banc_v4.analyser_bras("ref,plancher:placebo", 2500), fenetre_juge=2500)
    assert len(appels) == 3, "le placebo seul est retiré, et il l'est en entier"


def test_le_manifeste_du_run_nomme_la_reference_et_le_placebo(monkeypatch, tmp_path):
    _banc_hors_ligne(monkeypatch, tmp_path,
                     lambda *a, **k: {"answer": "1", "abstained": False})
    manifeste = banc_v4.mesure(banc_v4.analyser_bras("ref,cand:fenetre=9000,p:placebo", 2500),
                               fenetre_juge=2500)
    assert manifeste["reference"] == "ref"
    assert manifeste["placebo"] == "p"
    assert manifeste["run_id"]
    assert json.loads(banc_v4._fichier_run().read_text(encoding="utf-8")) == manifeste


# ------------------------------------------------------------------- §4c : les tirages k > 1

def test_les_tirages_2_a_k_passent_par_un_cache_NEUF_par_bras(monkeypatch, tmp_path):
    """Sinon la variance intra-question vaudrait zéro par construction.

    À température 0 le cache est indexé par l'empreinte du payload : un second tirage servi par
    le cache rendrait la première réponse. Et le cache doit être propre à **chaque bras** — un
    placebo porte le même payload que sa référence, un cache partagé au rang t lui servirait
    donc la réponse que la référence vient d'y écrire.
    """
    caches = []

    def repondre(question, contexte, model=None, prompt=None, characters=None, **_):
        caches.append(llm.CACHE)
        return {"answer": f"{len(caches)}", "abstained": False}

    _banc_hors_ligne(monkeypatch, tmp_path, repondre)
    depot = llm.CACHE
    banc_v4.mesure(banc_v4.analyser_bras("ref,plancher:placebo", 2500),
                   fenetre_juge=2500, tirages=3)

    # 3 questions × 2 bras × 3 tirages = 18 appels ; 6 seulement au cache du dépôt (bras `ref`,
    # tirage 1), et aucun cache de tirage n'est partagé entre les deux bras.
    assert len(caches) == 18
    assert sum(1 for c in caches if c == depot) == 3
    assert len({c for c in caches if c != depot}) == 5  # placebo + 2 rangs × 2 bras


def test_la_variance_intra_question_est_publiee_et_dit_ce_qu_elle_ne_couvre_pas(monkeypatch, tmp_path):
    reponses = iter(["1", "1", "2"] * 10)
    _banc_hors_ligne(monkeypatch, tmp_path,
                     lambda *a, **k: {"answer": next(reponses), "abstained": False})
    banc_v4.mesure(banc_v4.analyser_bras("ref", 2500), fenetre_juge=2500, tirages=3)
    tirages = json.loads(banc_v4._cache_tirages("ref").read_text(encoding="utf-8"))
    assert len(tirages) == 3 and all(len(v) == 2 for v in tirages.values())


# --------------------------------------------------------- §4b : le plancher, et l'obligation

def test_un_delta_sans_plancher_est_refuse():
    """Le refus qui n'existait pas, et sans lequel trois chantiers ont publié du bruit."""
    bras = banc_v4.Bras(nom="cand", prompt="v4", fenetre_servie=2500, role="variante")
    assert banc_v4.plancher_requis(bras, {}, plancher=None, sans_plancher=False) is True


def test_le_placebo_lui_meme_n_a_pas_besoin_d_un_plancher():
    """Son Δ EST le plancher — lui en demander un serait une régression à l'infini."""
    placebo = banc_v4.Bras(nom="p", prompt="v4", fenetre_servie=2500, role="placebo",
                           jetable=True)
    assert banc_v4.plancher_requis(placebo, {}, plancher=None, sans_plancher=False) is False
    variante = banc_v4.Bras(nom="p", prompt="v4", fenetre_servie=2500, role="variante")
    assert banc_v4.plancher_requis(variante, {"placebo": "p"}, None, False) is False


def test_sans_plancher_leve_le_refus_mais_s_ecrit():
    bras = banc_v4.Bras(nom="cand", prompt="v4", fenetre_servie=2500, role="variante")
    assert banc_v4.plancher_requis(bras, {}, plancher=None, sans_plancher=True) is False
    assert banc_v4.plancher_requis(bras, {}, plancher=Path("p.json"), sans_plancher=False) is False


def _comparaison(delta, ic, a=0, b=0, famille="negative_voisine"):
    return {"par_famille": {famille: {"n": 34, "delta": delta, "ic95": ic, "a": a, "b": b}}}


def test_un_GAIN_dans_le_bruit_est_declare_non_concluant():
    """La moitié de la règle qu'on est tenté d'oublier — c'est celle qui fait plaisir."""
    resultat = banc_v4._conclure_contre_le_plancher(
        _comparaison(+0.1176, [0.029, 0.235]), _comparaison(+0.1176, [0.029, 0.235]))
    bloc = resultat["par_famille"]["negative_voisine"]
    assert bloc["conclusion"] == "NON_CONCLUANT"
    assert bloc["ic_recouvre_le_plancher"] is True


def test_une_PERTE_dans_le_bruit_est_declaree_non_concluante():
    """L'autre moitié : un instrument qui ne saurait invalider que les bonnes nouvelles serait
    un instrument orienté."""
    resultat = banc_v4._conclure_contre_le_plancher(
        _comparaison(-0.10, [-0.25, +0.02]), _comparaison(+0.02, [-0.18, +0.20]))
    assert resultat["par_famille"]["negative_voisine"]["conclusion"] == "NON_CONCLUANT"


def test_un_delta_hors_du_bruit_reste_concluant():
    resultat = banc_v4._conclure_contre_le_plancher(
        _comparaison(+0.40, [0.30, 0.50]), _comparaison(+0.01, [-0.05, 0.07]))
    bloc = resultat["par_famille"]["negative_voisine"]
    assert bloc["conclusion"] == "CONCLUANT"
    assert bloc["ic_recouvre_le_plancher"] is False


def test_sans_plancher_aucune_famille_n_est_concluante():
    resultat = banc_v4._conclure_contre_le_plancher(_comparaison(+0.40, [0.30, 0.50]), None)
    assert resultat["par_famille"]["negative_voisine"]["conclusion"] == "SANS_PLANCHER"
    assert resultat["plancher"]["present"] is False


def test_le_seuil_de_la_garde_a_monte_au_niveau_du_bruit_quand_le_bruit_le_depasse():
    """`a <= 2` a été fixé sur un générateur supposé déterministe ; le placebo rend a = +3 à +10.

    Un seuil sous le plancher de bruit de l'instrument qu'il surveille n'est pas un seuil : il
    déclare une régression à système inchangé. Le seuil retenu est donc le plus exigeant des
    deux **tenables**.
    """
    resultat = banc_v4._conclure_contre_le_plancher(
        _comparaison(0.0, [-0.1, 0.1], a=5), _comparaison(0.0, [-0.1, 0.1], a=7))
    bloc = resultat["par_famille"]["negative_voisine"]
    assert bloc["garde_abstention_seuil_retenu"] == 7
    assert bloc["garde_abstention"] is True
    assert bloc["seuil_pre_enregistre_sous_le_bruit"] is True


def test_un_bruit_faible_laisse_le_seuil_pre_enregistre_en_place():
    resultat = banc_v4._conclure_contre_le_plancher(
        _comparaison(0.0, [-0.1, 0.1], a=4), _comparaison(0.0, [-0.1, 0.1], a=1))
    bloc = resultat["par_famille"]["negative_voisine"]
    assert bloc["garde_abstention_seuil_retenu"] == banc_v4.GARDE_ABSTENTION == 2
    assert bloc["garde_abstention"] is False
    assert bloc["seuil_pre_enregistre_sous_le_bruit"] is False


# --------------------------------------------------- §4d : la garde `a`, orientée par famille

def _paire(famille, avant_abstenue, apres_abstenue):
    return ({"famille": famille, "score": int(apres_abstenue), "abstenue": apres_abstenue},
            {"famille": famille, "score": int(avant_abstenue), "abstenue": avant_abstenue})


def _comparer_une_famille(famille, avant, apres):
    courant = {"detail": {}, **banc_v4.en_tete(2500, 2500, "v4")}
    reference = {"detail": {}, **banc_v4.en_tete(2500, 2500, "v4")}
    for n, (av, ap) in enumerate(zip(avant, apres)):
        a, r = _paire(famille, av, ap)
        courant["detail"][f"{famille}/q{n}"] = a
        reference["detail"][f"{famille}/q{n}"] = r
    return banc_v4.comparer(courant, reference, drapeau=False)["par_famille"][famille]


def test_sur_une_NEGATIVE_une_abstention_nouvelle_est_un_GAIN():
    """Le premier des deux tests du §4d.

    Sur une négative, l'abstention est la bonne réponse — c'est même le score de la famille.
    La garde d'origine, reprise d'`eval_characters` qui ne mesurait que des positives, comptait
    ces abstentions comme des dégradations. Elle ne se trompait pas d'un peu : de signe.
    """
    bloc = _comparer_une_famille("negative_voisine",
                                 avant=[False, False, False, False],
                                 apres=[True, True, True, True])
    assert bloc["abstentions_nouvelles"] == 4 and bloc["abstentions_disparues"] == 0
    assert bloc["a_brut"] == +4, "le chiffre d'hier, conservé à côté du corrigé"
    assert bloc["a"] == -4, "orienté dégradation : quatre gains, donc a négatif"
    assert bloc["garde_abstention_seuil_pre_enregistre"] is True
    assert bloc["delta"] == 1.0, "le score de la famille monte, lui aussi"


def test_sur_une_NEGATIVE_une_abstention_qui_DISPARAIT_est_la_degradation():
    """Le mode d'échec le plus grave du produit : il se met à répondre là où le corpus ne le
    permet pas. La garde d'origine le comptait comme un progrès."""
    bloc = _comparer_une_famille("negative_voisine",
                                 avant=[True, True, True, True],
                                 apres=[False, False, False, False])
    assert bloc["a_brut"] == -4
    assert bloc["a"] == +4
    assert bloc["garde_abstention_seuil_pre_enregistre"] is False


def test_sur_une_POSITIVE_une_abstention_nouvelle_reste_la_degradation():
    """Le second test du §4d : sur une positive, le sens d'origine était le bon, et il ne bouge pas."""
    bloc = _comparer_une_famille("table_cell",
                                 avant=[False, False, False, False],
                                 apres=[True, True, True, True])
    assert bloc["a_brut"] == +4 and bloc["a"] == +4
    assert bloc["garde_abstention_seuil_pre_enregistre"] is False
    assert "positive" in bloc["orientation_de_a"]


def test_l_orientation_est_ecrite_dans_le_fichier_et_pas_seulement_dans_le_code():
    bloc = _comparer_une_famille("negative_v3", avant=[False], apres=[True])
    assert "négative" in bloc["orientation_de_a"]


# ------------------------------------------------------- §4e : le test exact, à petit effectif

def test_le_bootstrap_degenere_est_signale_et_double_d_un_test_exact():
    """Le lot 6 a publié un IC `[0 ; 0]` sur 20 items — « Δ nul et parfaitement mesuré ».

    La vérité était « aucune paire discordante, donc aucune information ». Le test exact le dit,
    le bootstrap ne le pouvait pas.
    """
    bloc = _comparer_une_famille("negative_v3", avant=[True] * 20, apres=[True] * 20)
    assert bloc["ic95"] == [0.0, 0.0]
    assert bloc["bootstrap_degenere"] is True
    assert bloc["test_exact"]["paires_discordantes"] == 0
    assert bloc["test_exact"]["p"] == 1.0
    assert "aucune information" in bloc["test_exact"]["lecture"]


def test_le_test_exact_conclut_la_ou_l_effectif_le_permet():
    bloc = _comparer_une_famille("negative_v3", avant=[True] * 20,
                                 apres=[False] * 12 + [True] * 8)
    assert bloc["test_exact"]["paires_discordantes"] == 12
    assert bloc["test_exact"]["significatif_5pct"] is True
    assert bloc["test_exact"]["p"] < 0.001


def test_le_test_exact_ne_conclut_PAS_sur_deux_paires_discordantes():
    """Douze basculements sur vingt concluent ; deux ne concluent pas. C'est toute la différence
    qu'un effectif fait, et le bootstrap la rendait invisible."""
    bloc = _comparer_une_famille("negative_v3", avant=[True] * 20,
                                 apres=[False] * 2 + [True] * 18)
    assert bloc["test_exact"]["paires_discordantes"] == 2
    assert bloc["test_exact"]["significatif_5pct"] is False


def test_le_test_exact_refuse_les_scores_non_binaires():
    """Rendre None plutôt qu'un chiffre qui n'aurait pas de sens."""
    assert banc_v4._test_exact([0.5, -0.25]) is None
    assert banc_v4._test_exact([]) is None


def test_un_verdict_a_population_nulle_est_refuse_et_n_ecrase_rien(monkeypatch, tmp_path):
    """Trouvé en exécutant l'instrument réparé, le 10 septembre 2026.

    Un bras dont l'état vit dans un AUTRE worktree rend une population nulle. Le banc écrivait
    alors un ``results-v4-*.json`` d'apparence normale — en-têtes justes, coûts à zéro, familles
    absentes — **par-dessus l'artefact publié du même nom**, en une seconde et sans un mot. Un
    fichier vide qui a l'air d'un résultat est pire qu'une erreur.
    """
    monkeypatch.setattr(banc_v4, "CACHE", tmp_path)
    monkeypatch.setattr(banc_v4, "_index", lambda: None)
    monkeypatch.setattr(banc_v4, "population", lambda index=None: [
        {"qid": "q0", "qid_banc": "table_cell/q0", "famille": "table_cell", "question": "q",
         "or_valeur": "1", "gold_chunks": ["c1"]}])
    publie = banc_v4.HERE / f"results-v4-{banc_v4.SIGNATURE}-bras-de-test.json"
    assert not publie.exists(), "ce test ne doit jamais toucher un artefact réel"
    bras = banc_v4.Bras(nom="bras-de-test", prompt="v4", fenetre_servie=2500, role="reference")
    with pytest.raises(SystemExit) as echec:
        banc_v4.verdict(bras, 2500, reference=None, drapeau=False)
    assert "population NULLE" in str(echec.value)
    assert not publie.exists(), "rien ne doit avoir été écrit"


def test_la_population_ne_charge_aucun_index(monkeypatch):
    """184 Mo de mémoire résidente, chargés pour rien à chaque étape.

    `population()` reçoit un `index` **qu'elle n'a jamais lu**. Les étapes `mesure`,
    `contextes`, `verdict` le chargeaient quand même. Sur une machine où une seconde session
    mesurait en parallèle, c'est ce chargement inutile qui a fait la différence entre un run qui
    tient et un run tué par le système.
    """
    def refuser(*a, **k):
        raise AssertionError("aucune étape ne doit charger ChunkIndex pour bâtir la population")

    monkeypatch.setattr(banc_v4.ChunkIndex, "load", staticmethod(refuser))
    items = banc_v4.population()
    assert len(items) == 199
    assert {i["famille"] for i in items} == set(banc_v4.FAMILLES)


def test_deux_intervalles_qui_se_touchent_se_recouvrent():
    assert banc_v4._recouvrent([0.0, 0.1], [0.1, 0.2]) is True
    assert banc_v4._recouvrent([0.0, 0.1], [0.11, 0.2]) is False
    assert banc_v4._recouvrent(None, [0.1, 0.2]) is None
