"""Le contrat de réponse v3 : un seul texte, deux lecteurs, et rien qui puisse diverger.

Le risque que ces tests ferment
--------------------------------
Le banc mesure un prompt ; le produit en applique un autre. C'était l'état du 9 septembre 2026 :
``pipeline.ANSWER_SYSTEM_V2`` vivait dans le module du banc, et ``rag/agents/rag-scout.md``
portait une paraphrase française de deux de ses règles. Les chiffres publiés — abstention,
couverture, fabrication — décrivaient alors un système que l'utilisateur n'employait pas.

Le contrat v3 est donc un **fichier**, lu par le banc et recopié dans l'agent, et ces tests
refusent que les deux copies s'éloignent d'un caractère. C'est la seule chose qui rend le proxy
``mistral-small`` défendable : ce n'est pas le même modèle, mais c'est le même contrat.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

BENCHMARK = Path(__file__).resolve().parents[1]
MACOS = BENCHMARK.parent
sys.path.insert(0, str(BENCHMARK))
sys.path.insert(0, str(MACOS))

import pipeline  # noqa: E402
import score_citation  # noqa: E402

AGENT = MACOS / "agents" / "rag-scout.md"


def _bloc_du_contrat(markdown: str) -> str:
    """Le bloc de code de ``rag-scout.md`` qui recopie le contrat, sans ses clôtures."""
    blocs = re.findall(r"^```\n(.*?)^```$", markdown, re.MULTILINE | re.DOTALL)
    portants = [b for b in blocs if "Rules, in order of priority" in b]
    assert len(portants) == 1, f"{len(portants)} bloc(s) de contrat dans rag-scout.md, attendu 1"
    return portants[0]


def test_l_agent_porte_le_contrat_mot_pour_mot():
    """Le texte de l'agent est celui du fichier. Pas « équivalent » : identique."""
    assert (_bloc_du_contrat(AGENT.read_text(encoding="utf-8")).strip()
            == pipeline.ANSWER_PROMPTS[pipeline.DEFAULT_PROMPT]), (
        "l'agent doit porter le contrat PAR DÉFAUT du banc — celui dont les chiffres sont publiés")


def test_le_contrat_vient_d_un_fichier_et_non_du_module():
    """Un contrat recopié dans le code redeviendrait deux contrats à la première retouche."""
    assert pipeline.CONTRAT_V3.exists()
    assert pipeline.ANSWER_SYSTEM_V3 not in BENCHMARK.joinpath("pipeline.py").read_text("utf-8")


def test_les_versions_coexistent_et_le_defaut_est_verrouille():
    """La bascule du défaut est une décision de l'utilisateur, pas un effet de bord de code.

    Ce test est **fait pour échouer** le jour où quelqu'un change `DEFAULT_PROMPT` : la ligne
    ci-dessous doit alors être modifiée sciemment, dans le même commit, avec la mesure qui la
    justifie. Un défaut qui bascule sans que rien ne casse est un défaut qui bascule sans qu'on
    le sache — et c'est ainsi que le banc et le produit se mettent à mesurer deux systèmes.
    """
    assert {"v1", "v2", "v3", "v4"} <= set(pipeline.ANSWER_PROMPTS)
    assert pipeline.DEFAULT_PROMPT == "v4"


def test_le_plancher_de_mots_du_contrat_est_celui_du_scoreur():
    """Un contrat qui autoriserait des citations plus courtes que ce que l'instrument sait
    vérifier produirait des extraits invérifiables — et un taux flatteur calculé sur le reste."""
    for nom in ("v3", "v4"):
        assert f"{score_citation.MOTS_MINIMUM} to 25 words" in pipeline.ANSWER_PROMPTS[nom]


def test_le_signal_d_absence_se_lit_ligne_par_ligne():
    reponse = ("The passages give the ex-ante volatility [1] \"the ex-ante volatility is 19.6 %\".\n"
               "NOT_IN_SOURCES: the realised Sharpe ratio\n")
    assert pipeline.non_trouve(reponse) == ["the realised Sharpe ratio"]
    # Il n'est pas une abstention, et le confondre avec elle est exactement ce qui rendait
    # `negative_voisine` illisible : le produit doit pouvoir dire « j'ai le sujet, pas la
    # grandeur » sans que cela compte comme « je n'ai rien ».
    assert pipeline.abstained(reponse) is False


def test_une_mention_dans_la_prose_n_est_pas_le_signal():
    """Sinon une réponse qui *parle* du jeton serait comptée comme l'ayant émis."""
    assert pipeline.non_trouve("the system would answer NOT_IN_SOURCES: x in that case") == []


def test_le_signal_tolere_l_indentation_mais_pas_le_saut_de_ligne_avale():
    assert pipeline.non_trouve("  NOT_IN_SOURCES:  vol of vol  ") == ["vol of vol"]
    assert pipeline.non_trouve("a\nNOT_IN_SOURCES: b\nc") == ["b"]


def test_une_citation_de_formule_est_verifiee_par_latex_et_non_par_la_chaine():
    """La correction d'instrument du 9 septembre 2026, sur le cas réel qui l'a motivée.

    Le générateur écrit la forme compacte, le passage porte la forme éclatée de MinerU. Une
    comparaison littérale déclare la citation fausse — elle mesure le rendu du convertisseur,
    ce que ``score_citation`` refuse explicitement de faire ailleurs dans le même module.
    """
    passage = ("the loss is $\\mathcal { L } = \\frac { 1 } { N } \\sum _ { i } "
               "\\left( x _ { i } - y _ { i } \\right) ^ { 2 }$ over the surface")
    citee = "$\\mathcal{L} = \\frac{1}{N} \\sum_i \\left( x_i - y_i \\right)^2$"
    [fiche] = score_citation.verifier_extraits([citee], passage)
    assert fiche["retrouve"] is True
    assert fiche["voie"] == "latex"


def test_une_formule_alteree_reste_introuvable():
    """La tolérance s'arrête à la ré-écriture : un symbole changé est une fabrication."""
    passage = "$\\mathcal { L } = \\frac { 1 } { N } \\sum _ { i } x _ { i }$"
    [fiche] = score_citation.verifier_extraits(["$\\mathcal{L} = \\frac{1}{M} \\sum_i x_i$"],
                                               passage)
    assert fiche["retrouve"] is False


def test_une_formule_exacte_dans_une_phrase_inventee_reste_introuvable():
    """Les deux conditions, pas l'une ou l'autre — sinon une formule juste blanchirait sa phrase."""
    passage = "the loss is $\\mathcal { L } = \\frac { 1 } { N }$ over the surface"
    citee = "the authors report a Sharpe ratio of $\\mathcal{L} = \\frac{1}{N}$ across all regimes"
    [fiche] = score_citation.verifier_extraits([citee], passage)
    assert fiche["retrouve"] is False


def test_une_citation_de_prose_reste_verifiee_litteralement():
    passage = "the ex-ante volatility is 19.6 % on average over the period"
    [exact] = score_citation.verifier_extraits(["the ex-ante volatility is 19.6 %"], passage)
    [faux] = score_citation.verifier_extraits(["the ex-ante volatility is 19.7 %"], passage)
    assert (exact["retrouve"], exact["voie"]) == (True, "litteral")
    assert faux["retrouve"] is False


# ------------------------------------------------------------------ v4 : un assemblage, pas un texte neuf

def _regles(contrat: str) -> dict[int, str]:
    """Les clauses numérotées d'un contrat, par leur numéro."""
    return {int(m.group(1)): m.group(2)
            for m in re.finditer(r"^(\d+)\. (.+)$", contrat, re.MULTILINE)}


#: D'où vient chaque clause de v4 : ``numéro dans v4 -> (contrat source, numéro dans la source)``.
#: C'est la table qui rend les deux contrastes lisibles — v4 contre v2 isole les clauses de
#: citation, v4 contre v3 isole le bloc d'abstention — et elle ne vaut que si **aucun mot** n'a
#: été réécrit au passage.
ORIGINE_V4 = {1: ("v2", 1), 2: ("v2", 2), 3: ("v2", 3),
              4: ("v3", 4), 5: ("v3", 5), 6: ("v2", 4), 7: ("v3", 8)}


def test_chaque_clause_de_v4_vient_mot_pour_mot_de_v2_ou_de_v3():
    """Le contraste n'isole ce qu'il prétend isoler que si rien n'a été récrit en chemin.

    Une clause « équivalente » mais reformulée ferait de v4 un troisième texte, et les deux
    contrastes mesureraient chacun deux choses à la fois — exactement ce que le §22 du
    9 septembre a montré capable d'accuser la mauvaise cause.
    """
    regles = {nom: _regles(pipeline.ANSWER_PROMPTS[nom]) for nom in ("v2", "v3", "v4")}
    for numero, (source, origine) in ORIGINE_V4.items():
        assert regles["v4"][numero] == regles[source][origine], (
            f"la clause {numero} de v4 ne vient pas mot pour mot de {source}/{origine}")
    assert len(regles["v4"]) == len(ORIGINE_V4), "v4 porte une clause dont l'origine n'est pas déclarée"


def test_v4_ne_porte_pas_le_bloc_d_abstention_de_v3():
    """La variable du troisième bras : le signal, retiré, et lui seul."""
    assert pipeline.NON_TROUVE in pipeline.ANSWER_SYSTEM_V3
    assert pipeline.NON_TROUVE not in pipeline.ANSWER_SYSTEM_V4
    assert pipeline.non_trouve("NOT_IN_SOURCES: x") == ["x"], "le lecteur reste, la clause part"


def test_v4_garde_les_clauses_de_citation_de_v3():
    """Ce que v4 conserve : la citation double, le LaTeX, le plafond élargi."""
    assert f"{score_citation.MOTS_MINIMUM} to 25 words" in pipeline.ANSWER_SYSTEM_V4
    assert "LaTeX between dollar signs" in pipeline.ANSWER_SYSTEM_V4
    assert "200 if your reply contains a formula" in pipeline.ANSWER_SYSTEM_V4
