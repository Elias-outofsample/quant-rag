"""Notation d'une réponse de la famille `table_cell` — et le diagnostic qui dit *pourquoi* elle rate.

Ce que « juste » veut dire ici
------------------------------
La question désigne une cellule ; la réponse est juste si elle **énonce la valeur de cette
cellule**. Rien d'autre n'est exigé : ni la phrase, ni l'unité rédigée, ni l'ordre des mots.
La comparaison est donc déléguée entière à ``norme_valeurs`` — même table de normalisation
que la métrique de citation, pour la raison que son docstring donne : deux implémentations,
ce sont deux réponses différentes sur ``0.250`` contre ``0.25``, et une divergence qu'aucun
test n'attrape.

L'égalité se juge **à la précision de l'or**. Une cellule ``0.250`` affirme trois décimales,
et ``0.25`` dit la même chose : juste. ``0.26`` dit autre chose : faux. Ce n'est pas une
tolérance de confort, c'est la seule lecture qui ne punisse pas un générateur pour avoir
écrit un nombre comme un humain l'écrit.

Le diagnostic, et pourquoi il vaut le score
--------------------------------------------
Un échec de cette famille a deux causes très différentes, et le booléen ne les sépare pas :
le système n'a pas trouvé le tableau, ou il l'a trouvé et **a lu la mauvaise cellule**. La
seconde est une erreur de lecture de grille — la ligne d'à côté, la colonne d'à côté — et
c'est précisément ce que la famille est censée mesurer. En passant ``autres_valeurs`` (les
autres cellules du même tableau, que ``selection_tableaux`` fournit dans chaque candidat),
le scoreur distingue « aucune valeur du tableau » de « une valeur du tableau, mais pas
celle-là ».

L'abstention n'est pas une réponse juste
-----------------------------------------
Le pipeline demande au générateur, quand les passages ne portent pas la réponse, de
commencer sa réponse par le jeton ``INSUFFICIENT_EVIDENCE``. Une abstention qui cite malgré
tout un nombre — « INSUFFICIENT_EVIDENCE, les passages ne donnent que 8.59 pour… » — ne doit
pas être comptée juste parce que le bon chiffre traîne dans la phrase : le système a refusé
de répondre, on enregistre qu'il a refusé. Le jeton est **redéfini ici** plutôt qu'importé
de ``pipeline`` : ce module doit rester chargeable sans réseau ni clé d'API.

Ce que ce module ne fait pas
-----------------------------
Aucun appel LLM, aucun juge, aucune écriture. Il ne vérifie pas non plus que la réponse cite
le bon passage — c'est une autre mesure, et les mêler rendrait les deux illisibles.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import norme_valeurs  # noqa: E402
from selection_tableaux import canonique, confondables  # noqa: E402

#: Le jeton d'abstention du pipeline. Recopié, jamais importé — voir le docstring.
JETON_ABSTENTION = "INSUFFICIENT_EVIDENCE"

#: Décorations que les générateurs mettent devant le jeton. Le pipeline exige que la réponse
#: *commence* par lui ; un ``**INSUFFICIENT_EVIDENCE**`` est une abstention rédigée en gras,
#: pas une réponse.
_ORNEMENTS = " \t\n\r*_`\"'«»[](){}#>-–—:"


def abstention(reponse_texte: str) -> bool:
    """La réponse commence-t-elle par le jeton d'abstention ?

    On exige bien le **début** : un jeton cité au milieu d'une phrase (« je n'ai pas répondu
    INSUFFICIENT_EVIDENCE parce que… ») est un commentaire du générateur sur sa consigne,
    pas un refus de répondre.
    """
    return (reponse_texte or "").lstrip(_ORNEMENTS).startswith(JETON_ABSTENTION)


def score(reponse_texte: str, valeur_or_brute: str, autres_valeurs=None) -> dict:
    """Note une réponse contre la cellule d'or, et diagnostique l'échec.

    ``valeur_or_brute`` est la cellule telle qu'elle est écrite dans le chunk
    (``candidat['valeur_brute']``) ; ``autres_valeurs`` les autres cellules du même tableau
    (``candidat['autres_valeurs']``), facultatives — sans elles, le diagnostic de mauvaise
    cellule est simplement absent, jamais faux.

    Lève ``ValueError`` si l'or n'est pas une valeur lisible : une question dont l'or est
    illisible ne doit pas se noter « fausse » en silence, elle doit sauter aux yeux.
    """
    or_valeur = norme_valeurs.lire(valeur_or_brute)
    if or_valeur is None:
        raise ValueError(f"valeur d'or illisible : {valeur_or_brute!r}")

    texte = reponse_texte or ""
    refus = abstention(texte)
    lues = norme_valeurs.nombres(texte)

    trouvee = next((valeur for valeur in lues if norme_valeurs.egales(or_valeur, valeur)), None)

    autres_trouvees = []
    for brute in (autres_valeurs or []):
        autre = norme_valeurs.lire(brute)
        if autre is None or confondables(or_valeur, autre):
            # Une autre cellule qu'une même réponse satisferait n'est pas un indice de
            # mauvaise lecture : elle est indiscernable de l'or. (La sélection interdit ce
            # cas ; le scoreur ne suppose pas que son entrée vient de la sélection.)
            continue
        if any(norme_valeurs.egales(autre, valeur) for valeur in lues):
            autres_trouvees.append(brute)

    juste = trouvee is not None and not refus
    return {
        "juste": juste,
        "valeur_trouvee": trouvee.source if trouvee is not None else None,
        "valeur_or_canonique": canonique(or_valeur),
        "abstention": refus,
        "autres_cellules": autres_trouvees,
        # La signature d'une lecture de la mauvaise cellule : pas la bonne valeur, mais une
        # valeur du tableau. Le système avait le passage sous les yeux.
        "mauvaise_cellule": not juste and not refus and bool(autres_trouvees),
        # Le bon chiffre, la mauvaise unité : « 84 » pour une cellule « 84% ». C'est un
        # échec au sens de norme_valeurs — changer d'unité sans le dire n'est pas répondre —
        # mais ce n'est pas la même faute qu'un chiffre faux, et le confondre avec elle
        # ferait passer un défaut de rédaction pour un défaut de lecture.
        "unite_divergente": trouvee is None and any(
            valeur.decimal == or_valeur.decimal and valeur.pourcentage != or_valeur.pourcentage
            for valeur in lues),
        "chiffres_lus": len(lues),
    }
