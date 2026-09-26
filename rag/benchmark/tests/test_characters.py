"""Gardes de l'instrument `characters` et de sa lecture ré-ancrée.

Ces tests ne mesurent rien : ils protègent les invariants du pré-enregistrement
``PRE-ENREGISTREMENT-CHARACTERS-2026-09-08.md`` contre une modification silencieuse.
"""
from __future__ import annotations

import sys
from pathlib import Path

BENCH = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BENCH))
sys.path.insert(0, str(BENCH.parent))

import eval_characters as ec
import reancrage_characters as rc


def test_grille_et_reference_figees():
    """§2 : quatre paliers, aucun ajouté ni retiré, la référence est le servi du banc."""
    assert ec.GRILLE == (1600, 2400, 3200, 4500)
    assert ec.REFERENCE == 1600


def test_fenetre_du_juge_fixe_au_plus_grand_palier():
    """§4 : l'instrument ne doit pas changer avec le bras qu'il mesure."""
    assert ec.FENETRE_JUGE == max(ec.GRILLE) == 4500


def test_seuils_des_gardes_conformes_a_wilson():
    """§5 : `a <= 2` tient et `a >= 3` échoue à 6,25 % ; `b <= 6` tient à 10 %."""
    from garde_reponse import wilson_haut

    assert wilson_haut(2, 130) <= 0.0625 < wilson_haut(3, 130)
    assert wilson_haut(6, 130) <= 0.10 < wilson_haut(7, 130)
    assert (ec.GARDE_ABSTENTION, ec.GARDE_DEGRADATION) == (2, 6)


def test_condition_de_lisibilite():
    """§4 : la famille bloquante est celle qui garde contre « plus fluide, moins fondé »."""
    assert ec.FAMILLE_BLOQUANTE == "unsupported_fluent"
    assert ec.SEUIL_FAMILLE == 0.75


def test_ancre_produit_est_le_palier_de_la_production():
    """§2 bis : la production sert 2 500 c. ; 2 400 est le palier qui la reproduit."""
    assert rc.ANCRE_PRODUIT == 2400
    assert rc.ANCRE_PRODUIT in ec.GRILLE


#: **Toutes** les troncatures de passage de la surface servie, relevées le 8 septembre 2026.
#: Il n'y en avait pas une, il y en avait **cinq**, et elles ne s'accordaient pas :
#:
#: - ``search_documents`` (``mcp_server.py``) et ``search_graph`` (``graph_search.py``) rendaient
#:   **2 500** ; ``connect_entities`` **1 200**, moins de la moitié ; ``timeline`` **300** ;
#: - et ``get_passage``, via ``quant_rag.joindre_passages``, **6 000** — celle-là ne s'écrivait
#:   pas ``row['text'][:6000]`` mais ``[:max_characters]``, et le balayage ci-dessous, tel qu'il
#:   était écrit, **ne la voyait pas**. Elle coupait 30,2 % des fenêtres de ``get_passage``.
#:
#: Elles sont désormais **zéro** : les cinq sites passent par ``contrat``, source unique de la
#: fenêtre servie. Le balayage reste — c'est la seule chose qui voit une troncature littérale
#: réintroduite, et une assertion sur l'import ne la verrait pas : un ``row['text'][:2500]``
#: réécrit coexisterait sans bruit avec un ``import contrat``, test vert, défaut invisible.
#: L'assertion sur l'import est **ajoutée** au balayage, jamais substituée.
#: Ce qui reste, et **pourquoi chacune a le droit de rester** :
#:
#: - ``quant_rag.py`` ``[:2000]`` — l'entrée du **reranker**, pas un texte servi. C'est ce que
#:   le cross-encodeur voit pour scorer ; la changer changerait le **classement**, ce que ce
#:   chantier a interdiction de faire.
#: - ``quant_rag.py`` ``[:900]`` — l'affichage de la **CLI** ``__main__``, pas la surface MCP.
#:
#: Toute autre valeur qui apparaît ici est une fenêtre servie réintroduite en dur.
TRONCATURES_SERVIES = {
    "mcp_server.py": [],
    "graph_search.py": [],
    "quant_rag.py": ["900", "2000"],
}

#: Le balayage. Trois formes, parce que la cinquième troncature a échappé à la première :
#: un littéral (``[:2500]``), une variable de fenêtre (``[:max_characters]``), une constante
#: en majuscules (``[:MAX_CHARS]``). Toutes appliquées à quelque chose qui s'appelle « texte ».
MOTIF_TRONCATURE = (r"(?:row\['text'\]|row\[\"text\"\]|\btexte?\b|\.text)"
                    r"\s*\[\s*:\s*(\d+|max_characters|[A-Z_]{4,})\s*\]")

#: Les fichiers qui rendent un passage servi, et le nom du module qui doit porter leur fenêtre.
SURFACE_SERVIE = ("mcp_server.py", "graph_search.py", "quant_rag.py")


def code_seul(chemin) -> str:
    """La source **sans ses commentaires ni ses docstrings**, mais avec toutes ses chaînes.

    Sans cela le balayage se mordrait la queue : la docstring de ``joindre_passages`` cite
    ``row['text'][:6000]`` pour raconter le défaut qu'elle vient de réparer, et un test qui
    compte les mentions au lieu des appels transformerait toute documentation honnête en échec.

    Effacer **toutes** les chaînes serait le défaut symétrique, et plus grave : ``row['text']``
    contient une chaîne, si bien qu'un balayage aveugle aux chaînes ne verrait plus aucune
    troncature servie — un test toujours vert, qui ne garde plus rien.
    """
    import ast
    import tokenize

    source = chemin.read_text(encoding="utf-8")
    lignes = source.splitlines()
    a_effacer: set[int] = set()
    for noeud in ast.walk(ast.parse(source)):
        if isinstance(noeud, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            if ast.get_docstring(noeud, clean=False) is not None:
                corps = noeud.body[0]
                a_effacer.update(range(corps.lineno, (corps.end_lineno or corps.lineno) + 1))
    # Un commentaire de fin de ligne ne fait taire que ce qui le suit : effacer la ligne
    # entière masquerait le code qui le précède, et donc une troncature commentée.
    coupures: dict[int, int] = {}
    with chemin.open("rb") as handle:
        for jeton in tokenize.tokenize(handle.readline):
            if jeton.type == tokenize.COMMENT:
                numero, colonne = jeton.start
                coupures[numero] = min(coupures.get(numero, colonne), colonne)
    rendu = []
    for numero, ligne in enumerate(lignes, 1):
        if numero in a_effacer:
            rendu.append("")
        elif numero in coupures:
            rendu.append(ligne[:coupures[numero]])
        else:
            rendu.append(ligne)
    return "\n".join(rendu)


def test_la_surface_servie_n_a_plus_aucune_troncature_litterale():
    """Le défaut du §2 bis coûtait 900 caractères et personne ne l'avait vu. Il y en avait cinq."""
    import re

    for fichier, attendues in TRONCATURES_SERVIES.items():
        trouvees = sorted(re.findall(MOTIF_TRONCATURE, code_seul(BENCH.parent / fichier)),
                          key=lambda v: (not v.isdigit(), int(v) if v.isdigit() else v))
        assert trouvees == attendues, (
            f"{fichier} : troncatures servies {trouvees}, attendues {attendues}. "
            "Une fenêtre a été réintroduite en dur sur le chemin servi. Elle doit passer par "
            "``contrat.couper_passage`` — sans quoi le banc et la production divergent de "
            "nouveau en silence, et la note de non-comparabilité du README §5 devient fausse."
        )


def test_chaque_site_servi_passe_par_le_contrat():
    """L'ajout au balayage, pas son remplacement : zéro littéral **et** la source unique lue."""
    import contrat

    for fichier in SURFACE_SERVIE:
        source = (BENCH.parent / fichier).read_text(encoding="utf-8")
        assert "import contrat" in source, f"{fichier} ne lit pas la fenêtre servie"
    rendus = (BENCH.parent / "mcp_server.py").read_text(encoding="utf-8")
    assert "contrat.couper_passage" in rendus, "search_documents ne coupe plus par le contrat"
    assert "contrat.couper_apercu" in rendus, "timeline ne coupe plus par le contrat"
    graphe = (BENCH.parent / "graph_search.py").read_text(encoding="utf-8")
    assert graphe.count("contrat.couper_passage") == 2, (
        "search_graph et connect_entities doivent couper par le contrat, tous les deux : "
        "c'est l'écart 2 500 / 1 200 qui est réparé ici.")
    assert "contrat.couper(" in (BENCH.parent / "quant_rag.py").read_text(encoding="utf-8"), (
        "joindre_passages — la cinquième troncature — doit couper par le contrat")
    assert contrat.PASSAGE_CHARACTERS >= 9384, (
        "le plus long passage du corpus (chunk-004c25a877336fd7, un tableau) doit tenir entier")


def test_le_banc_rend_exactement_ce_que_la_production_sert():
    """**Le défaut du §2 bis, transformé en garde.**

    Le banc a rendu 1 600 caractères par passage pendant que la production en servait 2 500,
    et rien ne comparait les deux nombres. Ce test est ce qui manquait. La fenêtre n'est plus
    un nombre recopié de part et d'autre : c'est **le même objet**, lu au même endroit.
    """
    import inspect

    import contrat
    import judge
    import pipeline

    assert pipeline.CARACTERES_SERVIS == contrat.PASSAGE_CHARACTERS
    assert (inspect.signature(pipeline.format_passages).parameters["characters"].default
            == pipeline.CARACTERES_SERVIS)
    # Le juge doit voir au moins ce que le générateur a vu, sans quoi il note « non ancrée »
    # une réponse tirée d'un matériau qu'elle avait légitimement sous les yeux.
    assert (inspect.signature(judge.grade).parameters["characters"].default
            >= pipeline.CARACTERES_SERVIS)


def test_la_fenetre_du_juge_de_characters_reste_nommee():
    """Le cache de ``characters`` a été jugé à fenêtre **fixe 4 500**. Toute mesure comparée à
    lui doit régler le juge à 4 500, ou déclarer l'écart : le défaut du juge, lui, a changé."""
    import contrat

    assert ec.FENETRE_JUGE == 4500
    assert contrat.PASSAGE_CHARACTERS != ec.FENETRE_JUGE, (
        "si les deux coïncident un jour, ce test perd son objet — mais pas la règle : "
        "un verdict rendu à 4 500 ne se compare qu'à un verdict rendu à 4 500.")


def test_test_des_signes():
    """McNemar exact : symétrique, `p = 1` sans discordance, strict quand tout va d'un côté."""
    assert rc._signe(0, 0) == 1.0
    assert rc._signe(8, 5) == rc._signe(5, 8)
    assert rc._signe(8, 5) > 0.05          # le contraste 1 600 -> 2 400 : indistinguable
    assert rc._signe(10, 0) < 0.01         # dix contre zéro : lui ne l'est pas
    assert rc._signe(1, 0) == 1.0
