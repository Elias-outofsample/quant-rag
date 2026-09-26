"""La citation, mesurée sans juge — « [1] » pointe-t-il un passage qui dit vraiment ce qui est affirmé ?

Pourquoi cette métrique existe
------------------------------
La consigne du générateur est explicite — *« Cite the passages you rely on inline as [1], [3] »* —
et le produit promet des citations exactes. Rien, jusqu'ici, ne vérifiait la promesse : le juge
lit la réponse et la question, jamais la correspondance entre un marqueur et son passage. Une
réponse peut donc être notée juste tout en citant [2] pour un chiffre qui n'est que dans [1], ou
en citant [7] quand cinq passages ont été servis. Ces deux fautes sont invisibles au juge et
visibles ici, **sans un appel LLM** : ce sont des faits de texte, pas des jugements.

La fenêtre, et pourquoi elle est le cœur du module
--------------------------------------------------
``pipeline.format_passages`` tronque chaque passage à ``CARACTERES_SERVIS`` = 2 500 caractères.
Mesuré sur l'index ``5530cba145`` (26 120 chunks, 418 documents) : **25,8 % des chunks dépassent
2 500 caractères**. Vérifier une citation contre le texte *entier* du chunk créditerait donc, une
fois sur quatre environ, une citation que le générateur ne pouvait pas lire. Exemple réel —
``chunk-a168f9e0446df729`` (3 546 caractères) porte son équation de régression Fama-French, et
le coefficient ``- 0 . 0 1 0 5``, au caractère 3 409 : hors de la fenêtre servie. Une réponse qui
l'annonce ne l'a pas lue dans ce passage. ``verifier`` tronque **avant** de vérifier.

Le nombre 2 500 est recopié ici et non importé : ``pipeline`` importe ``llm``, et ce module doit
rester importable et exécutable hors ligne. ``tests/test_v4_citation.py`` lit ``pipeline.py``
comme du texte et refuse que les deux nombres divergent.

Ce que « se retrouver » veut dire, et le fait de corpus qui l'impose
--------------------------------------------------------------------
MinerU rend les mathématiques **éclatées caractère par caractère** —
``V _ { i , t } ^ { \\mathrm { b i d } }`` — et 59,3 % des chunks portent du ``$``. Une
comparaison de chaînes mesurerait le rendu de MinerU. On délègue donc, sans réimplémenter :

  - une **valeur chiffrée** est retrouvée par ``norme_valeurs.contient`` — égalité à la précision
    de la référence, drapeau pourcentage exigé (``19.6`` n'atteste pas ``19.6 %``) ;
  - une **expression** est retrouvée par ``latex_norme.inclus`` (sous-suite d'atomes) **ou** par
    ``latex_norme.recouvrement_local >= 0.9``. Le seuil se justifie par ce que la fonction
    calcule : à fenêtre de la taille de la référence, c'est ``|intersection| / |référence|``.
    0,9 tolère **un atome sur dix** — un ``\\left`` perdu, une virgule d'indice que le générateur
    n'a pas recopiée — et refuse une formule qui ne partagerait que ses variables : sur les
    formules courtes du corpus (8 à 20 atomes), 0,9 n'autorise qu'un seul atome d'écart.

Ce que ce module REFUSE de faire
---------------------------------
**Il ne juge pas le sens.** Une phrase sans chiffre et sans formule — « the strategy is robust
across regimes [1] » — n'est pas une affirmation vérifiable ici, et n'entre pas dans les comptes.
La métrique ne couvre donc *pas* toutes les affirmations d'une réponse, seulement celles qu'un
programme peut confronter au texte. Le prétendre serait faux.

**Il ne répare pas le corpus.** ``normaliser_texte`` **absorbe** les artefacts — elle ne corrige
rien et n'écrit rien. Mesuré : 43,3 % des chunks portent au moins un artefact qu'elle absorbe
(25,8 % des guillemets ou apostrophes typographiques, 20,3 % un tiret non ASCII, 13,4 % une
balise ``<sup>``/``<sub>``). Sur la ligature littérale la mesure **contredit la note de mission** :
``ﬁ ﬀ ﬂ ﬃ ﬄ`` n'apparaissent dans **aucun** chunk (0 sur 26 120) — le pli ligature reste, parce
qu'il porte sur la réponse du générateur autant que sur le passage, mais il ne faut pas lui
attribuer un effet qu'il n'a pas de ce côté-ci.

**Il ne distingue pas un chiffre-affirmation d'un chiffre-décor.** ``norme_valeurs.nombres``
retire les marqueurs ``[1]`` — c'est acquis. Restent comptés **en trop** : une année (« since
2019 »), un numéro de page, un numéro d'équation ou de table (« Table 3 »), un effectif d'année
dans une plage. Deux garde-fous seulement, et ils sont partiels : les chiffres **internes à une
région mathématique** sont retirés de la liste des valeurs (le ``1`` et le ``2`` de
``\\frac{1}{2}`` sont de la structure, et l'expression est déjà vérifiée comme un tout), et un
groupe entre crochets dont un nombre dépasse ``PLAFOND_MARQUEUR`` n'est pas lu comme un marqueur
— ``[2023]`` derrière « Sabino da Silva et al. » est une référence bibliographique, pas le
passage n° 2023. Le biais résiduel est **optimiste** : une année citée se retrouve presque
toujours dans un passage servi, donc elle gonfle ``appuyee``. Un lecteur de ces chiffres doit le
savoir.

**Il ne rattrape pas le signe d'un nombre éclaté.** Mesuré en construisant ce module :
``norme_valeurs.nombres("- 0 . 0 1 0 5")`` rend ``0.0105``, positif — le moins est séparé du
premier chiffre par une espace, et la règle de recollement ne regarde que le caractère
immédiatement précédent. Conséquence exacte, et elle est **pessimiste** : une réponse qui
annonce ``-0.0105`` n'est pas créditée par le passage qui porte le coefficient sous sa forme
éclatée. La voie sûre pour un coefficient négatif est donc l'expression, où ``latex_norme``
garde le signe comme un atome. On le signale plutôt que de contourner : corriger
``norme_valeurs`` déplacerait la frontière des scoreurs de tableaux, qui ne l'a pas demandé.
"""
from __future__ import annotations

import re
import sys
import unicodedata
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import latex_norme  # noqa: E402
import norme_valeurs  # noqa: E402

#: Miroir de ``pipeline.CARACTERES_SERVIS``. Recopié, pas importé — voir le module.
FENETRE_SERVIE = 2500

#: Il n'y a **pas** de seuil de recouvrement : une expression est retrouvée par inclusion
#: exacte, ou elle ne l'est pas. Voir ``_porte_l_expression`` pour la mesure qui a fait
#: retirer le seuil de 0,9 qui tenait cette place.
SEUIL_RECOUVREMENT = None

#: Longueur minimale d'une citation littérale, en mots. En deçà, une coïncidence de vocabulaire
#: suffirait à « retrouver » l'extrait dans n'importe quel passage du corpus.
MOTS_MINIMUM = 5

#: Au-delà, un nombre entre crochets n'est plus un numéro de passage. La production en sert cinq ;
#: 99 laisse toute la marge imaginable à un banc, et écarte les années — ``[2023]``, ``[1993]`` —
#: que le corpus emploie pour ses références bibliographiques.
PLAFOND_MARQUEUR = 99

#: Les cinq classes, exclusives et exhaustives. L'ordre est celui de l'évaluation.
CATEGORIES = ("sans_citation", "hors_bornes", "appuyee", "mauvais_passage", "absente")

# --------------------------------------------------------------------------- marqueurs

#: Un groupe entre crochets fait de nombres, de séparateurs de liste et de tirets de plage.
_MARQUEUR = re.compile(r"\[\s*(\d{1,4}(?:\s*[-–—,;]\s*\d{1,4})*)\s*\]")
_SEPARATEUR = re.compile(r"\s*([-–—,;])\s*")


def _developper(contenu: str) -> list[int] | None:
    """Numéros d'un groupe entre crochets, ou ``None`` si ce n'est pas une citation de passage.

    ``[1-3]`` est développé en 1, 2, 3 : le générateur écrit des plages, et les ignorer
    ferait passer la phrase pour non citée — une **panne silencieuse**, pire qu'une erreur
    déclarée. Le développement ne s'applique qu'à une plage croissante ; ``[3-1]`` est lu comme
    deux numéros, parce qu'aucune interprétation de plage n'y est défendable.
    """
    morceaux = _SEPARATEUR.split(contenu.strip())
    try:
        lus = [int(morceaux[i]) for i in range(0, len(morceaux), 2)]
    except (ValueError, IndexError):
        return None
    if not lus or any(n < 1 or n > PLAFOND_MARQUEUR for n in lus):
        return None
    sortie = [lus[0]]
    for rang in range(1, len(lus)):
        if morceaux[2 * rang - 1] in "-–—" and lus[rang] > lus[rang - 1]:
            sortie.extend(range(lus[rang - 1] + 1, lus[rang] + 1))
        else:
            sortie.append(lus[rang])
    return sortie


def marqueurs(texte: str) -> list[int]:
    """Numéros de passage cités, **dans l'ordre d'apparition**, sans répétition.

    Gère ``[1]``, ``[1, 3]``, ``[1; 3]``, ``[1][3]`` et ``[1-3]``. Une répétition ne dit rien de
    plus qu'une occurrence — la question est « ce passage est-il invoqué ? » — et la dédoublonner
    évite qu'une réponse bavarde paraisse citer davantage de sources.
    """
    trouves: list[int] = []
    for correspondance in _MARQUEUR.finditer(texte or ""):
        numeros = _developper(correspondance.group(1))
        if numeros is None:
            continue
        for numero in numeros:
            if numero not in trouves:
                trouves.append(numero)
    return trouves


# --------------------------------------------------------------------------- normalisation

#: Le pli ligature. NFKC le fait déjà (``ﬁ`` → ``fi``) ; la table reste **explicite** pour que la
#: règle soit lisible sans connaître les tables Unicode, et pour couvrir ``ﬅ``/``ﬆ`` que le corpus
#: pourrait porter. Mesuré : zéro chunk sur 26 120 en porte une littérale.
_LIGATURES = {"ﬁ": "fi", "ﬀ": "ff", "ﬂ": "fl", "ﬃ": "ffi", "ﬄ": "ffl", "ﬅ": "ft", "ﬆ": "st"}

#: Guillemets et apostrophes ramenés aux formes ASCII. 25,8 % des chunks en portent une variante,
#: et le générateur, lui, écrit presque toujours droit : sans ce pli, une citation littérale
#: exacte serait déclarée introuvable pour un seul glyphe.
_GUILLEMETS = {"“": '"', "”": '"', "„": '"', "«": '"', "»": '"', "″": '"', "‟": '"',
               "‘": "'", "’": "'", "‚": "'", "‛": "'", "′": "'", "‹": "'", "›": "'",
               "`": "'", "´": "'"}

#: Toutes les graphies du tiret, moins Unicode compris. 20,3 % des chunks en portent une.
_TIRETS = "‐‑‒–—―−﹘﹣－"

#: **La balise seule, jamais son contenu.** Une règle « balise ET contenu » a d'abord été
#: écrite, sur l'argument — mesuré, et vrai — que 84,9 % des 38 589 balises du corpus ont un
#: contenu d'au plus quatre caractères, souvent un appel de note. L'argument portait sur la
#: fréquence et manquait le seul cas qui compte : ``R<sup>2</sup>``. Le NFKC rend ``R²`` en
#: ``R2``, un générateur écrit ``R2``, et jeter le contenu rendait ``R`` — la normalisation
#: censée réconcilier trois graphies en fabriquait une quatrième, et un extrait littéralement
#: exact devenait introuvable. L'exposant est du contenu ; seule la balise est du rendu.
#: Le motif attrape aussi la balise orpheline : la fenêtre de 2 500 caractères coupe des paires.
_BALISE_SEULE = re.compile(r"</?(?:sup|sub)>")

_ESPACES = re.compile(r"\s+")


def normaliser_texte(texte: str) -> str:
    """Forme sous laquelle on cherche un extrait littéral. Elle **absorbe** les artefacts.

    NFKC, ligatures, guillemets et apostrophes unifiés, tirets unifiés, balises ``<sup>``/
    ``<sub>`` retirées **en gardant leur contenu**, caractères de contrôle changés en espace
    (1,2 % des chunks en portent, résidus de polices à symboles), espaces réduits, casse pliée.

    Le contenu des balises est gardé, et c'est une correction, pas un détail : jeter la paire
    entière transformait ``R<sup>2</sup>`` en ``R``, quand le NFKC rend ``R²`` en ``R2`` et
    qu'un générateur écrit ``R2``. La normalisation censée réconcilier les trois graphies en
    fabriquait une quatrième, et un extrait littéral parfaitement exact devenait introuvable.
    L'exposant est du **contenu** ; seule la balise est du rendu.

    Elle ne corrige rien dans le corpus : celui-ci reste ce qu'il est, on cesse seulement de
    compter ses accidents de rendu comme des différences de contenu.
    """
    plie = unicodedata.normalize("NFKC", texte or "")
    for avant, apres in _LIGATURES.items():
        plie = plie.replace(avant, apres)
    plie = _BALISE_SEULE.sub("", plie)
    for signe in _TIRETS:
        plie = plie.replace(signe, "-")
    for avant, apres in _GUILLEMETS.items():
        plie = plie.replace(avant, apres)
    plie = "".join(c if ord(c) >= 32 or c in "\n\t\r" else " " for c in plie)
    return _ESPACES.sub(" ", plie).strip().casefold()


# --------------------------------------------------------------------------- découpe en phrases

#: Abréviations dont le point ne finit pas une phrase. ``et al.`` est la plus coûteuse : sans
#: elle, « Sabino da Silva et al. [2023] » se coupe en deux, et le ``[2023]`` orphelin ouvre une
#: phrase qu'il ne cite pas.
_ABREVIATIONS = ("al.", "e.g.", "i.e.", "cf.", "etc.", "vs.", "viz.", "ibid.", "resp.",
                 "approx.", "est.", "fig.", "eq.", "eqs.", "tab.", "sec.", "ref.", "refs.",
                 "no.", "nos.", "vol.", "ch.", "pp.", "p.", "dr.", "prof.", "mr.", "mrs.",
                 "ms.", "st.", "inc.", "ltd.", "univ.")

#: Fin de phrase candidate : une ponctuation terminale suivie de blanc ou de fin, ou un saut de
#: ligne — le générateur répond souvent en liste à puces, sans le moindre point.
_FIN_CANDIDATE = re.compile(r"[.!?…]+(?=\s|$)|\n+")


def _zones_math(texte: str) -> list[tuple[int, int]]:
    """Bornes des régions mathématiques, telles que ``latex_norme`` les définit.

    On repère les régions en cherchant les contenus que ``latex_norme.expressions`` rend, plutôt
    qu'en recopiant ses délimiteurs : une seconde définition des régions ``$…$`` finirait par
    diverger de la première sans qu'aucun test ne le voie. Imprécision assumée et bornée — si le
    contenu d'une formule apparaît aussi en clair plus tôt dans le texte, c'est cette occurrence
    qui est protégée ; elle protège alors une région de trop, jamais une de moins.
    """
    zones, curseur = [], 0
    for expression in latex_norme.expressions(texte):
        debut = texte.find(expression, curseur)
        if debut < 0:
            continue
        zones.append((debut, debut + len(expression)))
        curseur = debut + len(expression)
    return zones


def _dans_une_zone(position: int, zones: list[tuple[int, int]]) -> bool:
    return any(debut <= position < fin for debut, fin in zones)


def _phrases(texte: str) -> list[str]:
    """Découpe en phrases, taillée sur ce que le corpus et le générateur écrivent vraiment.

    Trois pièges nommés, et ce qui les évite :

    1. **Le point d'un nombre éclaté.** ``- 0 . 0 1 0 5`` — la forme que MinerU rend — offre un
       point suivi d'un blanc à chaque chiffre. Une coupe n'est retenue que si la suite commence
       par autre chose qu'une minuscule ou un chiffre.
    2. **Le point d'une formule.** ``$$ … \\times C M A . $$`` clôt son équation par un point de
       phrase, à l'intérieur des délimiteurs. Les régions mathématiques ne se coupent pas.
    3. **Le point d'une abréviation.** ``et al.``, ``e.g.`` — voir ``_ABREVIATIONS``.
    """
    contenu = texte or ""
    zones = _zones_math(contenu)
    coupes = []
    for correspondance in _FIN_CANDIDATE.finditer(contenu):
        debut, fin = correspondance.span()
        if _dans_une_zone(debut, zones):
            continue
        if correspondance.group().startswith("\n"):
            coupes.append((debut, fin))
            continue
        avant = contenu[:fin].lower()
        if any(avant.endswith(abreviation) for abreviation in _ABREVIATIONS):
            continue
        suite = contenu[fin:].lstrip()
        if suite[:1].islower() or suite[:1].isdigit():
            continue
        coupes.append((debut, fin))
    phrases, curseur = [], 0
    for debut, fin in coupes:
        # La ponctuation terminale appartient à la phrase qu'elle ferme ; un saut de ligne, non.
        borne = debut if contenu[debut:fin].startswith("\n") else fin
        morceau = contenu[curseur:borne].strip()
        if morceau:
            phrases.append(morceau)
        curseur = fin
    reste = contenu[curseur:].strip()
    if reste:
        phrases.append(reste)
    return phrases


# --------------------------------------------------------------------------- affirmations

_PONCTUATION_SEULE = re.compile(r"^[\s\W_]*$")


def _hors_math(phrase: str) -> str:
    """La phrase, régions mathématiques blanchies — leurs chiffres sont de la structure."""
    caracteres = list(phrase)
    for debut, fin in _zones_math(phrase):
        for position in range(debut, min(fin, len(caracteres))):
            caracteres[position] = " "
    return "".join(caracteres)


def _sans_marqueurs(phrase: str) -> str:
    return _MARQUEUR.sub(" ", phrase)


def affirmations(texte: str) -> list[dict]:
    """Une entrée par phrase portant une affirmation **vérifiable** : un chiffre ou une formule.

    Clés : ``phrase``, ``valeurs`` (objets ``norme_valeurs.Valeur``), ``expressions`` (sources
    LaTeX), ``marqueurs`` (numéros cités **dans cette phrase**), ``rang`` (ordre d'apparition).

    Piège traité : le générateur pose parfois ses marqueurs **après** le point — « … is 19.6 %.
    [1] ». Une phrase qui ne contient que des marqueurs n'affirme rien ; ses marqueurs sont
    rendus à la phrase précédente, faute de quoi l'affirmation passerait pour non citée et la
    citation pour hors sujet. Deux fautes fabriquées par la découpe, pas par la réponse.
    """
    brutes = []
    for rang, phrase in enumerate(_phrases(texte)):
        expressions_lues = latex_norme.expressions(phrase)
        valeurs_lues = norme_valeurs.nombres(_hors_math(phrase))
        brutes.append({"rang": rang, "phrase": phrase, "valeurs": valeurs_lues,
                       "expressions": expressions_lues, "marqueurs": marqueurs(phrase)})
    for position in range(len(brutes) - 1, 0, -1):
        fiche = brutes[position]
        if fiche["valeurs"] or fiche["expressions"] or not fiche["marqueurs"]:
            continue
        if not _PONCTUATION_SEULE.match(_sans_marqueurs(fiche["phrase"])):
            continue
        precedente = brutes[position - 1]
        for numero in fiche["marqueurs"]:
            if numero not in precedente["marqueurs"]:
                precedente["marqueurs"].append(numero)
        fiche["marqueurs"] = []
    return [f for f in brutes if f["valeurs"] or f["expressions"]]


# --------------------------------------------------------------------------- extraits littéraux

#: Guillemets droits, français et anglais courbes. Le guillemet **simple** courbe est exclu à
#: dessein : ``’`` est l'apostrophe de 19,6 % des chunks (« the portfolio’s annualized return »),
#: et l'accepter comme fermante ouvrirait un extrait à chaque possessif anglais.
_EXTRAIT = re.compile(r'"([^"\n]{1,600}?)"|«\s*([^»\n]{1,600}?)\s*»|“([^”\n]{1,600}?)”')

_BORDS = " \t\r\n.…"


def extraits(texte: str) -> list[str]:
    """Citations littérales d'au moins ``MOTS_MINIMUM`` mots, telles qu'écrites, sans répétition.

    Le saut de ligne est interdit à l'intérieur d'un extrait : un guillemet dépareillé — le
    générateur en produit — avalerait sinon la moitié de la réponse et se déclarerait introuvable,
    ce qui est un faux négatif spectaculaire pour une faute de frappe.
    """
    trouves, vus = [], set()
    for correspondance in _EXTRAIT.finditer(texte or ""):
        brut = next(g for g in correspondance.groups() if g is not None).strip()
        empreinte = normaliser_texte(brut)
        if len(empreinte.split()) < MOTS_MINIMUM or empreinte in vus:
            continue
        vus.add(empreinte)
        trouves.append(brut)
    return trouves


def verifier_extraits(extraits_lus, texte_de_reference: str) -> list[dict]:
    """Chaque extrait est-il retrouvé dans le texte de référence — prose **et** formules ?

    Les points de suspension de bord sont retirés de l'aiguille : « … the company is added … »
    est une élision de citation, pas trois caractères que le passage devrait contenir.

    **Une citation de formule ne se vérifie pas comme de la prose**, et le faire mesure le rendu
    de MinerU au lieu de la fidélité du générateur — ce que l'en-tête de ce module refuse
    explicitement, et que la comparaison littérale faisait ici. Mesuré sur le test de format du
    contrat v3 : la réponse écrit ``$\\mathcal{L}_{\\text{VAE}} = \\frac{1}{N}…$``, le passage
    porte la même formule sous la forme éclatée ``\\mathcal { L } _ { \\text { V A E } } =
    \\frac { 1 } { N }…``, et la chaîne ne s'y retrouve pas. Aucune des deux n'est fausse : elles
    diffèrent d'un espacement que ``latex_norme`` absorbe déjà à 100 % sur les 312 formules
    réelles de ``tests/test_v4_formule.py``.

    Un extrait qui porte une région mathématique est donc retrouvé si **toutes** ses expressions
    sont incluses dans la référence au sens de ``latex_norme.inclus`` — le même prédicat, sans
    tolérance de recouvrement, que ``_porte_l_expression`` — **et** si la prose qui les entoure
    s'y retrouve littéralement. Les deux, pas l'une ou l'autre : une formule exacte noyée dans
    une phrase inventée reste une citation inventée.

    ``voie`` dit par quel chemin l'extrait a été jugé. Un taux agrégé qui mélange deux méthodes
    de vérification doit pouvoir être défait par celui qui le lit.
    """
    reference = normaliser_texte(texte_de_reference)
    sortie = []
    for extrait in extraits_lus or []:
        aiguille = normaliser_texte(extrait).strip(_BORDS)
        expressions_lues = latex_norme.expressions(extrait)
        retrouve = bool(aiguille) and aiguille in reference
        voie = "litteral"
        if not retrouve and expressions_lues:
            formules_ok = all(latex_norme.inclus(expression, texte_de_reference)
                              for expression in expressions_lues)
            prose = normaliser_texte(_hors_math(extrait)).strip(_BORDS)
            # La prose résiduelle est souvent vide — l'extrait *est* la formule — ou réduite à
            # quelques mots de liaison ; sous le plancher de mots elle ne discrimine rien, et
            # exiger de la retrouver ferait échouer une citation de formule parfaitement exacte.
            prose_ok = len(prose.split()) < MOTS_MINIMUM or prose in reference
            retrouve = formules_ok and prose_ok
            voie = "latex" if retrouve else "litteral"
        sortie.append({"extrait": extrait,
                       "mots": len(aiguille.split()),
                       "formules": len(expressions_lues),
                       "voie": voie,
                       "retrouve": retrouve})
    return sortie


# --------------------------------------------------------------------------- vérification

def _passages_servis(contexte, fenetre: int) -> list[dict]:
    """Les passages **tels que le générateur les a vus** : numérotés depuis 1, tronqués.

    La troncature reproduit ``pipeline.format_passages`` au caractère près, ``(text or '')[:n]``.
    """
    servis = []
    for numero, ligne in enumerate(contexte or [], 1):
        servis.append({"numero": numero,
                       "text": (ligne.get("text") or "")[:fenetre],
                       "chunk_id": ligne.get("chunk_id"),
                       "document_id": ligne.get("document_id")})
    return servis


def _valeur_lisible(valeur) -> str:
    return f"{valeur.decimal}{' %' if valeur.pourcentage else ''}"


def _porte_la_valeur(passage: dict, valeur) -> bool:
    return norme_valeurs.contient(passage["text"], valeur)


def _porte_l_expression(passage: dict, expression: str) -> bool:
    """Le passage porte-t-il cette expression ? **Inclusion exacte, rien d'autre.**

    Un seuil de recouvrement a été essayé, à 0,9, et retiré : mesuré, il créditait
    ``E[W_t | X_t = k]`` contre un passage qui écrit ``E[V_t | X_t = k]``. Une formule
    altérée d'un seul symbole sur vingt donne un recouvrement de 0,95 — au-dessus de tout
    seuil qu'on oserait poser —, et **un seul symbole est exactement à quoi ressemble une
    fabrication**. Ici on ne mesure pas la ressemblance : on mesure si le passage cité *dit*
    ce que la réponse lui fait dire. La tolérance jouait contre l'objet même de la métrique.

    Elle ne coûtait d'ailleurs rien à retirer : ``latex_norme`` absorbe déjà les ré-écritures
    d'un générateur — forme compacte, police retirée, ``\\left``/``\\right``, accolades d'un
    atome, variantes grecques —, à **100 %** sur les 312 formules réelles de
    ``tests/test_v4_formule.py``. Ce que le seuil rattrapait n'était pas de la ré-écriture,
    c'était de l'erreur.

    ``recouvrement_local`` reste publié dans la fiche de chaque affirmation, en diagnostic :
    il dit *de combien* la formule est abîmée quand elle l'est. Il ne vote pas.
    """
    return latex_norme.inclus(expression, passage["text"])


def _items(affirmation: dict, passages: list[dict]) -> list[dict]:
    """Chaque élément vérifiable de la phrase, avec les numéros des passages qui le portent.

    Le relevé est fait sur **tous** les passages servis, même quand la phrase ne cite rien :
    c'est ce qui permet de dire « le chiffre existait, mais dans un autre passage » plutôt que
    « introuvable », et c'est toute la différence entre une faute de citation et une invention.
    """
    releve = []
    for valeur in affirmation["valeurs"]:
        releve.append({"genre": "valeur", "texte": _valeur_lisible(valeur),
                       "passages": [p["numero"] for p in passages if _porte_la_valeur(p, valeur)]})
    for expression in affirmation["expressions"]:
        releve.append({"genre": "expression", "texte": latex_norme.normaliser(expression),
                       "passages": [p["numero"] for p in passages
                                    if _porte_l_expression(p, expression)]})
    return releve


def _classer(affirmation: dict, passages: list[dict]) -> dict:
    """Range une affirmation dans **exactement une** des cinq classes.

    L'ordre décide, et il est délibéré. Un marqueur hors bornes est une citation cassée même si
    la phrase est par ailleurs soutenue : ``[7]`` quand cinq passages ont été servis désigne un
    passage qui n'existe pas, et le lecteur ne peut rien en faire. De même, ``absente`` l'emporte
    sur ``mauvais_passage`` quand la phrase mêle les deux : une phrase dont un élément n'est nulle
    part est une phrase non fondée, quoi qu'il en soit du reste.
    """
    cites = affirmation["marqueurs"]
    releve = _items(affirmation, passages)
    fiche = {"rang": affirmation["rang"], "phrase": affirmation["phrase"],
             "marqueurs": list(cites), "items": releve}
    if not cites:
        fiche["categorie"] = "sans_citation"
        return fiche
    hors = [n for n in cites if n < 1 or n > len(passages)]
    if hors:
        fiche["categorie"] = "hors_bornes"
        fiche["marqueurs_hors_bornes"] = hors
        return fiche
    manquants = [item for item in releve if not set(item["passages"]) & set(cites)]
    if not manquants:
        fiche["categorie"] = "appuyee"
    elif any(not item["passages"] for item in manquants):
        fiche["categorie"] = "absente"
    else:
        fiche["categorie"] = "mauvais_passage"
    return fiche


def verifier(reponse_texte: str, contexte, fenetre: int = FENETRE_SERVIE) -> dict:
    """Vérifie les citations d'une réponse contre les passages servis. Aucun appel réseau.

    ``contexte`` : les passages dans l'ordre où ils ont été numérotés pour le générateur — des
    dicts portant ``text``, ``chunk_id``, ``document_id``. ``fenetre`` tronque chaque texte
    **avant** vérification : c'est ce que le générateur a réellement lu.

    Le résultat est entièrement sérialisable en JSON — les valeurs y sont rendues lisibles, pas
    en objets ``Valeur`` — parce que la place de ces chiffres est un ``results-*.json`` relu dans
    six mois, pas une session Python.
    """
    passages = _passages_servis(contexte, fenetre)
    fiches = [_classer(a, passages) for a in affirmations(reponse_texte)]
    comptes = {categorie: 0 for categorie in CATEGORIES}
    for fiche in fiches:
        comptes[fiche["categorie"]] += 1
    total = len(fiches)
    notees = comptes["appuyee"] + comptes["mauvais_passage"] + comptes["absente"]
    cites = marqueurs(reponse_texte)
    reference = "\n\n".join(p["text"] for p in passages)
    return {
        "fenetre": fenetre,
        "n_passages": len(passages),
        "n_affirmations": total,
        "comptes": comptes,
        # Indéfini plutôt que zéro : une réponse sans affirmation notable n'a pas une précision
        # de 0 %, elle n'en a pas. Un zéro se moyennerait, et mentirait.
        "precision_citations": round(comptes["appuyee"] / notees, 4) if notees else None,
        "taux_sans_citation": round(comptes["sans_citation"] / total, 4) if total else None,
        "marqueurs": cites,
        "marqueurs_hors_bornes": [n for n in cites if n < 1 or n > len(passages)],
        "passages_cites": [n for n in cites if 1 <= n <= len(passages)],
        "extraits": verifier_extraits(extraits(reponse_texte), reference),
        "detail": fiches,
        "appels_llm": 0,
    }
