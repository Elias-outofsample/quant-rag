"""Forme canonique d'une valeur chiffrée — partagée par la famille `table_cell` et la métrique de citation.

Pourquoi une seule table de normalisation, et pas une par scoreur
-----------------------------------------------------------------
Le scoreur de cellule et la métrique de citation posent la même question — « ce nombre-ci
est-il ce nombre-là ? » — sur les mêmes tableaux. Deux implémentations, ce sont deux
réponses différentes sur ``0.250`` contre ``0.25``, et une divergence qu'aucun test
n'attraperait puisque chacune passerait la sienne.

Les décisions, et le fait de corpus qui les commande
-----------------------------------------------------
**Les parenthèses ne sont pas un signe.** En finance de marché ``(1 234)`` vaut −1 234 ;
dans les tableaux de régression de ce corpus — la forme dominante — ``(0.089)`` est un
écart-type sous son coefficient. Trancher dans un sens ou dans l'autre fabriquerait un or
faux une fois sur deux. On ne tranche donc pas : la valeur est rendue **positive**, la
parenthèse est **signalée**, et la sélection des cellules d'or **refuse** une cellule
parenthésée. Ce qui est ambigu ne devient pas de l'or.

**Les étoiles de significativité ne sont pas des chiffres.** ``0.077***`` et ``0.077`` sont
la même mesure ; le nombre d'étoiles dit un seuil, pas une valeur.

**La virgule est ambiguë et se tranche par la forme.** Suivie d'exactement trois chiffres
qui ne finissent pas le nombre, c'est un séparateur de milliers (``1,234.5``) ; suivie
d'un ou deux chiffres en fin de nombre, c'est une virgule décimale (``1,5``). Les deux
existent dans un corpus qui mêle sources anglophones et francophones.

**Le pourcentage fait partie de la valeur.** ``5 %`` et ``0.05`` sont numériquement liés
mais ne s'écrivent pas l'un pour l'autre : exiger l'égalité du drapeau évite de compter
juste une réponse qui a changé d'unité sans le dire.

**L'égalité se juge à la précision de l'or, pas à la virgule près.** Une cellule écrite
``0.250`` affirme trois décimales ; une réponse ``0.25`` dit la même chose. Une cellule
``0.25`` n'affirme rien au-delà de deux décimales, et ``0.2512`` s'y confond. La règle est
donc : égal si l'écart ne dépasse pas une demi-unité du dernier rang **de l'or**.
"""
from __future__ import annotations

import re
import unicodedata
from decimal import Decimal, InvalidOperation

#: Marques de significativité, croix de note de bas de tableau, symboles monétaires.
_DECORATIONS = "*∗×✳†‡§¶$€£¥%‰ \t   "
#: Toutes les graphies du signe moins qu'un PDF peut produire.
_MOINS = "−‒–—―➖"

_NOMBRE = re.compile(r"[-+]?\d[\d,.    ]*\d|[-+]?\d")


class Valeur:
    """Une valeur chiffrée canonique : un nombre, un drapeau pourcentage, une précision."""

    __slots__ = ("decimal", "pourcentage", "parenthesee", "decimales", "source")

    def __init__(self, decimal: Decimal, pourcentage: bool, parenthesee: bool,
                 decimales: int, source: str):
        self.decimal = decimal
        self.pourcentage = pourcentage
        self.parenthesee = parenthesee
        self.decimales = decimales
        self.source = source

    def __repr__(self) -> str:  # pragma: no cover - confort de mise au point
        return (f"Valeur({self.decimal}, pourcentage={self.pourcentage}, "
                f"parenthesee={self.parenthesee}, decimales={self.decimales})")

    def __eq__(self, autre) -> bool:
        return isinstance(autre, Valeur) and egales(self, autre)

    def __hash__(self) -> int:
        return hash((self.decimal, self.pourcentage))


def _plier(texte: str) -> str:
    texte = unicodedata.normalize("NFKC", texte or "")
    for signe in _MOINS:
        texte = texte.replace(signe, "-")
    return texte


def _virgules(brut: str) -> str:
    """Sépare le rôle des virgules par la forme du nombre, jamais par une préférence de locale."""
    for espace in (" ", " ", " ", " "):
        brut = brut.replace(espace, "")
    if "," not in brut:
        return brut
    if "." in brut:                     # 1,234.5 — la virgule ne peut être que le millier
        return brut.replace(",", "")
    morceaux = brut.split(",")
    queue = morceaux[1:]
    if all(len(m) == 3 for m in queue) and len(morceaux) > 1 and len(morceaux[0]) <= 3:
        return brut.replace(",", "")    # 1,234 / 12,345,678
    if len(queue) == 1 and 1 <= len(queue[0]) <= 2:
        return brut.replace(",", ".")   # 1,5 — virgule décimale
    return brut.replace(",", "")


def lire(texte: str) -> Valeur | None:
    """Valeur canonique d'une cellule ou d'un fragment, ou ``None`` si ce n'en est pas une."""
    if texte is None:
        return None
    source = str(texte)
    plie = _plier(source).strip()
    if not plie:
        return None
    parenthesee = bool(re.fullmatch(r"[\(\[]\s*[^()\[\]]*\s*[\)\]]", plie))
    if parenthesee:
        plie = plie[1:-1].strip()
    pourcentage = "%" in plie or "‰" in plie
    noyau = plie.strip(_DECORATIONS).strip()
    noyau = re.sub(r"[*∗†‡]+$", "", noyau).strip()
    if not noyau:
        return None
    signe = ""
    while noyau[:1] in "+-":
        if noyau[0] == "-":
            signe = "" if signe == "-" else "-"
        noyau = noyau[1:].strip()
    if not noyau or not re.fullmatch(r"[\d,.    ]+", noyau):
        return None
    noyau = _virgules(noyau)
    if not re.fullmatch(r"\d*\.?\d+|\d+\.?", noyau):
        return None
    try:
        valeur = Decimal(signe + noyau.rstrip("."))
    except InvalidOperation:
        return None
    fraction = noyau.split(".", 1)
    decimales = len(fraction[1]) if len(fraction) == 2 else 0
    return Valeur(valeur, pourcentage, parenthesee, decimales, source)


def egales(reference: Valeur | None, candidate: Valeur | None) -> bool:
    """Égalité **à la précision de la référence** — voir le module pour la règle et son motif."""
    if reference is None or candidate is None:
        return False
    if reference.pourcentage != candidate.pourcentage:
        return False
    tolerance = Decimal(1).scaleb(-reference.decimales) / 2
    return abs(reference.decimal - candidate.decimal) <= tolerance


def nombres(texte: str) -> list[Valeur]:
    """Toutes les valeurs chiffrées d'un texte libre, dans l'ordre.

    Les marqueurs de citation ``[1]`` sont retirés d'abord : ce sont des numéros de passage,
    pas des affirmations chiffrées, et les compter ferait passer toute réponse citée pour
    une réponse chiffrée.
    """
    plie = _plier(texte or "")
    plie = re.sub(r"\[\s*\d+(?:\s*[,;]\s*\d+)*\s*\]", " ", plie)
    trouvees = []
    for correspondance in _NOMBRE.finditer(plie):
        debut, fin = correspondance.span()
        etendu = correspondance.group()
        if plie[fin:fin + 1] == "%" or plie[fin:fin + 2] == " %":
            etendu += "%"
        if plie[max(debut - 1, 0):debut] == "-":
            etendu = "-" + etendu
        valeur = lire(etendu)
        if valeur is not None:
            trouvees.append(valeur)
    return trouvees


def contient(texte: str, reference: Valeur | None) -> bool:
    """La valeur ``reference`` figure-t-elle dans ``texte`` ?"""
    if reference is None:
        return False
    return any(egales(reference, valeur) for valeur in nombres(texte))
