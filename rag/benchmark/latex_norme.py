"""Forme canonique d'une expression LaTeX — la brique sur laquelle repose la famille `formula`.

Pourquoi ce module existe, et pourquoi il ne pouvait pas être une comparaison de chaînes
----------------------------------------------------------------------------------------
MinerU rend les mathématiques du corpus **éclatées caractère par caractère** :

    V _ { i , t } ^ { \\mathrm { b i d } } = \\frac { 1 } { N } \\sum _ { s \\in t } V _ { i , s }

Un générateur qui restitue la même formule l'écrit presque toujours compacte —
``V_{i,t}^{\\mathrm{bid}}`` — et souvent sans la police (``V_{i,t}^{bid}``). Ces trois
graphies sont *la même formule* pour n'importe quel lecteur, et trois chaînes différentes
pour ``==``. Une mesure de « restitution fidèle d'une formule » qui compte les espaces
mesurerait le rendu de MinerU, pas la fidélité du système.

La forme canonique est donc une **suite d'atomes**, pas une chaîne :

  - une commande ``\\[a-zA-Z]+`` est **un** atome (``\\frac``, ``\\alpha``) ;
  - tout autre caractère non blanc est **un** atome (``{``, ``_``, ``b``, ``,``).

C'est ce découpage — et lui seul — qui rend ``\\mathrm { b i d }`` et ``\\mathrm{bid}``
identiques (atomes ``\\mathrm { b i d }``) tout en gardant ``\\alpha \\beta`` distinct de
``\\alphabeta``. Un simple ``"".join(texte.split())`` aurait réussi le premier cas et
fabriqué une collision sur le second.

Les deux niveaux, et lequel décide
-----------------------------------
``NIVEAU_SOUPLE`` (celui qui **note**) efface les commandes de police : ``\\mathbb{E}[X]``
et ``E[X]`` sont la même formule pour un quant qui la lit. ``NIVEAU_STRICT`` les conserve,
unifiées par famille. Le strict est publié **à côté** du souple, en diagnostic : l'écart
entre les deux dit combien de restitutions ne diffèrent que par la police.

Ce que ce module ne fait pas
-----------------------------
Il ne corrige rien dans le corpus (règle du §2 du handoff : absorber les artefacts, pas les
réparer) et il n'évalue pas les environnements multi-lignes — ``\\begin{array}``,
``aligned``, ``cases``. Leurs ``&`` et ``\\\\`` sont des conventions d'alignement qu'un
générateur n'a aucune raison de reproduire ; une question dont l'or en dépend mesurerait la
typographie. ``environnement_multiligne`` les signale pour que la sélection les écarte.
"""
from __future__ import annotations

import re
import unicodedata

# --------------------------------------------------------------------------- atomes

#: Une commande LaTeX, ou n'importe quel autre caractère non blanc.
_ATOME = re.compile(r"\\[a-zA-Z]+|\\.|\S")

# --------------------------------------------------------------------------- tables

#: Effacées : elles ne changent pas ce que la formule dit, seulement sa taille à l'écran.
_TAILLE = {
    "\\left", "\\right", "\\big", "\\Big", "\\bigg", "\\Bigg",
    "\\bigl", "\\bigr", "\\Bigl", "\\Bigr", "\\biggl", "\\biggr", "\\Biggl", "\\Biggr",
    "\\bigm", "\\Bigm", "\\biggm", "\\Biggm", "\\middle",
    "\\displaystyle", "\\textstyle", "\\scriptstyle", "\\scriptscriptstyle",
    "\\limits", "\\nolimits", "\\mathstrut", "\\strut",
}

#: Espacements explicites. Ils sont au rendu ce que l'indentation est au code.
_ESPACES = {
    "\\,", "\\;", "\\:", "\\!", "\\ ", "\\quad", "\\qquad",
    "\\thinspace", "\\medspace", "\\thickspace", "\\negthinspace", "\\enspace",
}

#: Commandes à argument dont l'argument lui-même est du blanc : on jette commande ET argument.
_ESPACES_ARG = {"\\hspace", "\\vspace", "\\kern", "\\mskip", "\\hskip", "\\phantom",
                "\\hphantom", "\\vphantom"}

#: Familles de police. En souple, la commande disparaît et son argument reste.
_POLICES = {
    "\\mathrm", "\\mathbf", "\\mathbb", "\\mathcal", "\\mathscr", "\\mathsf",
    "\\mathfrak", "\\mathit", "\\mathtt", "\\mathnormal", "\\text", "\\textrm",
    "\\textbf", "\\textit", "\\textsf", "\\texttt", "\\operatorname", "\\operatorname*",
    "\\pmb", "\\boldsymbol", "\\bm", "\\bf", "\\rm", "\\it", "\\sf", "\\tt", "\\cal",
    "\\mathbfit", "\\symbf", "\\mathbfcal",
}

#: En strict, chaque police est ramenée au représentant de sa famille.
_POLICE_CANONIQUE = {
    "\\mathbf": "\\mathbf", "\\pmb": "\\mathbf", "\\boldsymbol": "\\mathbf",
    "\\bm": "\\mathbf", "\\bf": "\\mathbf", "\\textbf": "\\mathbf",
    "\\mathbfit": "\\mathbf", "\\symbf": "\\mathbf",
    "\\mathrm": "\\mathrm", "\\text": "\\mathrm", "\\textrm": "\\mathrm",
    "\\rm": "\\mathrm", "\\operatorname": "\\mathrm", "\\operatorname*": "\\mathrm",
    "\\mathit": "\\mathit", "\\textit": "\\mathit", "\\it": "\\mathit",
    "\\mathsf": "\\mathsf", "\\textsf": "\\mathsf", "\\sf": "\\mathsf",
    "\\mathtt": "\\mathtt", "\\texttt": "\\mathtt", "\\tt": "\\mathtt",
    "\\mathcal": "\\mathcal", "\\cal": "\\mathcal", "\\mathbfcal": "\\mathcal",
    "\\mathbb": "\\mathbb", "\\mathscr": "\\mathscr", "\\mathfrak": "\\mathfrak",
    "\\mathnormal": "\\mathnormal",
}

#: Variantes grecques et relationnelles unifiées — **unification déclarée**, exigée au §4 du
#: handoff. ``\\varphi``/``\\phi`` sont deux glyphes du même symbole : les auteurs les mêlent
#: à l'intérieur d'un même article, et aucun lecteur ne lit une formule différente.
_SYNONYMES = {
    "\\varphi": "\\phi", "\\varepsilon": "\\epsilon", "\\vartheta": "\\theta",
    "\\varrho": "\\rho", "\\varsigma": "\\sigma", "\\varpi": "\\pi",
    "\\le": "\\leq", "\\ge": "\\geq", "\\ne": "\\neq",
    "\\rightarrow": "\\to", "\\longrightarrow": "\\to", "\\Rightarrow": "\\implies",
    "\\leftarrow": "\\gets", "\\longleftarrow": "\\gets",
    "\\vert": "|", "\\lvert": "|", "\\rvert": "|",
    "\\Vert": "\\|", "\\lVert": "\\|", "\\rVert": "\\|",
    "\\ast": "*", "\\centerdot": "\\cdot", "\\dotsc": "\\dots", "\\dotsb": "\\dots",
    "\\ldots": "\\dots", "\\cdots": "\\dots",
    "\\intop": "\\int", "\\sumop": "\\sum",
    "\\lbrace": "\\{", "\\rbrace": "\\}", "\\lbrack": "[", "\\rbrack": "]",
    "\\colon": ":", "\\mid": "|",
}

#: Enveloppes de mise en page dont l'ouverture ET la fermeture sont effacées, avec le
#: spécificateur de colonnes de ``array``.
_ENVIRONNEMENTS_MULTILIGNE = {
    "array", "aligned", "align", "align*", "cases", "gathered", "gather", "gather*",
    "matrix", "pmatrix", "bmatrix", "vmatrix", "Vmatrix", "Bmatrix", "smallmatrix",
    "split", "eqnarray", "eqnarray*", "multline", "multline*", "alignat", "alignat*",
    "subequations", "table", "tabular",
}

#: **Mathématiques Unicode → LaTeX.** Trou trouvé en lisant les réponses réelles du banc, et
#: qu'aucun test ne pouvait voir : les ré-écritures d'attaque étaient toutes **dérivées du
#: corpus**, donc toujours en LaTeX. Un générateur, lui, répond volontiers
#: ``S_r = { M ∈ {0,1}^{r×r} : ∑_{j=1}^r M_{ij} = 1 }`` là où le papier écrit
#: ``\\mathcal{S}_r = \\left\\{ M \\in \\{0,1\\}^{r \\times r} : \\sum ... \\right\\}``. C'est la
#: **même formule**, et le scoreur la comptait fausse — donc il notait la police de sortie du
#: générateur, pas sa fidélité. Mesuré sur la ligne de base : 1 des 3 échecs à or servi.
_UNICODE_MATHS = {
    "α": "\\alpha", "β": "\\beta", "γ": "\\gamma", "δ": "\\delta", "ε": "\\epsilon",
    "ζ": "\\zeta", "η": "\\eta", "θ": "\\theta", "ι": "\\iota", "κ": "\\kappa",
    "λ": "\\lambda", "μ": "\\mu", "ν": "\\nu", "ξ": "\\xi", "π": "\\pi", "ρ": "\\rho",
    "σ": "\\sigma", "ς": "\\sigma", "τ": "\\tau", "υ": "\\upsilon", "φ": "\\phi",
    "ϕ": "\\phi", "χ": "\\chi", "ψ": "\\psi", "ω": "\\omega", "ϑ": "\\theta",
    "ϱ": "\\rho", "ϖ": "\\pi",
    "Γ": "\\Gamma", "Δ": "\\Delta", "Θ": "\\Theta", "Λ": "\\Lambda", "Ξ": "\\Xi",
    "Π": "\\Pi", "Σ": "\\Sigma", "Υ": "\\Upsilon", "Φ": "\\Phi", "Ψ": "\\Psi",
    "Ω": "\\Omega",
    "∈": "\\in", "∉": "\\notin", "∋": "\\ni", "⊂": "\\subset", "⊆": "\\subseteq",
    "⊃": "\\supset", "⊇": "\\supseteq", "∪": "\\cup", "∩": "\\cap", "∅": "\\emptyset",
    "∑": "\\sum", "∏": "\\prod", "∫": "\\int", "∬": "\\iint", "∮": "\\oint",
    "√": "\\sqrt", "∞": "\\infty", "∂": "\\partial", "∇": "\\nabla",
    "×": "\\times", "÷": "\\div", "±": "\\pm", "∓": "\\mp", "·": "\\cdot",
    "≤": "\\leq", "≥": "\\geq", "≠": "\\neq", "≈": "\\approx", "≃": "\\simeq",
    "≅": "\\cong", "≡": "\\equiv", "∼": "\\sim", "∝": "\\propto",
    "→": "\\to", "←": "\\gets", "↦": "\\mapsto", "⇒": "\\implies", "⇔": "\\iff",
    "∀": "\\forall", "∃": "\\exists", "¬": "\\neg", "∧": "\\wedge", "∨": "\\vee",
    "ℝ": "\\mathbb R", "ℕ": "\\mathbb N", "ℤ": "\\mathbb Z", "ℚ": "\\mathbb Q",
    "ℂ": "\\mathbb C", "ℙ": "\\mathbb P", "𝔼": "\\mathbb E", "ℓ": "\\ell",
    "…": "\\dots", "⋯": "\\dots", "′": "\\prime", "″": "\\prime\\prime",
    "∥": "\\|", "⟨": "\\langle", "⟩": "\\rangle", "⌊": "\\lfloor", "⌋": "\\rfloor",
}

_ENV = re.compile(r"\\(?:begin|end)\s*\{\s*([a-zA-Z*]+)\s*\}")
_TAG = re.compile(r"\\(?:tag|label|ref|eqref|nonumber|notag|numberwithin)\s*(\*?)\s*(\{[^{}]*\})?")
_ARG_JETABLE = re.compile(
    r"\\(?:hspace|vspace|kern|mskip|hskip|phantom|hphantom|vphantom)\s*\*?\s*\{[^{}]*\}")
_SUPSUB = (("<sup>", "^{"), ("</sup>", "}"), ("<sub>", "_{"), ("</sub>", "}"))

NIVEAU_STRICT, NIVEAU_SOUPLE = "strict", "souple"


# --------------------------------------------------------------------------- outils

def environnement_multiligne(source: str) -> str | None:
    """Nom du premier environnement multi-ligne rencontré, ou ``None``.

    La sélection s'en sert pour écarter les formules dont l'or dépendrait d'une convention
    d'alignement (``&``, ``\\\\``) que le générateur n'a aucune raison de reproduire.
    """
    for nom in _ENV.findall(source or ""):
        if nom in _ENVIRONNEMENTS_MULTILIGNE:
            return nom
    return None


def _pre_nettoyage(source: str) -> str:
    texte = unicodedata.normalize("NFKC", source or "")
    texte = "".join(c for c in texte if ord(c) >= 32 or c in "\n\t\r")
    # Les symboles Unicode deviennent leur commande LaTeX, entourée d'espaces : sans les
    # espaces, « ∑x » donnerait l'atome « \sumx » au lieu de « \sum » puis « x ».
    if any(c in _UNICODE_MATHS for c in texte):
        texte = "".join(f" {_UNICODE_MATHS[c]} " if c in _UNICODE_MATHS else c for c in texte)
    for avant, apres in _SUPSUB:
        texte = texte.replace(avant, apres)
    texte = _TAG.sub(" ", texte)
    texte = _ARG_JETABLE.sub(" ", texte)
    texte = _ENV.sub(lambda m: " " if m.group(1) in _ENVIRONNEMENTS_MULTILIGNE else m.group(0), texte)
    return texte


def _atomiser(texte: str) -> list[str]:
    return _ATOME.findall(texte)


def _fermeture(atomes: list[str], depart: int) -> int:
    """Index de la fermante appariée à ``atomes[depart] == '{'``, ou **-1** si elle manque.

    Le ``-1`` est le point important, et il a coûté un plantage. Cette fonction rendait
    ``len - 1`` quand l'accolade n'était pas refermée — un index plausible, jamais faux au
    typage, et faux au sens. ``inclus`` est appliqué à des **passages tronqués à 2 500
    caractères** : la coupe casse des paires, et les appelants pliaient alors un groupe qui
    n'existait pas, perdant du contenu en silence (``a + \\mathrm { c o v`` rendait
    ``a + c o``) avant de déborder de la liste. Une fermante absente est un fait ; le dire
    laisse chaque appelant traiter l'accolade comme l'atome littéral qu'elle est.
    """
    if depart >= len(atomes) or atomes[depart] != "{":
        return -1
    profondeur = 0
    for i in range(depart, len(atomes)):
        if atomes[i] == "{":
            profondeur += 1
        elif atomes[i] == "}":
            profondeur -= 1
            if profondeur == 0:
                return i
    return -1


def _effacer_specificateur_array(atomes: list[str]) -> list[str]:
    """``\\begin{array}`` a disparu au pré-nettoyage, son ``{ r c l }` reste : il n'a pas de sens.

    On ne peut le repérer qu'ici : c'est un groupe qui ne contient que des lettres de
    colonnes (``l``, ``c``, ``r``, ``|``, ``@``) et qui suivait l'environnement effacé.
    Prudence assumée : on n'efface qu'en tête d'expression, là où il ne peut être qu'un
    spécificateur — jamais au milieu, où ``{ r c l }` pourrait être une vraie expression.
    """
    i = 0
    while i < len(atomes) and atomes[i] == "{":
        fin = _fermeture(atomes, i)
        if fin < 0:
            break
        contenu = atomes[i + 1:fin]
        if contenu and all(a in {"l", "c", "r", "|", "@", "p", "s"} for a in contenu):
            atomes = atomes[:i] + atomes[fin + 1:]
            continue
        break
    return atomes


def _appliquer_polices(atomes: list[str], niveau: str) -> list[str]:
    """Polices et synonymes. En souple, une police emporte **les accolades de son argument**.

    Le défaut que cette règle répare, trouvé par l'attaque adversariale du scoreur de formule
    et mesuré sur 312 blocs réels : effacer la seule commande laissait un groupe orphelin dès
    que l'argument faisait plus d'un atome. ``a + \\mathrm{cov}(x,y)`` devenait
    ``a + { c o v } ( x , y )`` là où le générateur écrit ``a + c o v ( x , y )`` — **60 des
    312 formules échouaient sur cette seule ré-écriture**, et toutes pour ce seul mécanisme.
    Les deux replis d'accolades ne pouvaient pas l'attraper : le premier ne plie qu'un groupe
    d'**un** atome, le second qu'un groupe **dans** un groupe (le cas ``^ { \\mathrm{bid} }``,
    qui, lui, passait déjà — d'où l'illusion que le problème était réglé).

    Retirer les accolades avec la commande est le geste juste et non l'astuce commode :
    en souple, ``\\mathrm{ab}`` *est* ``ab``, et ``x ^ { \\mathrm { a b } }`` garde bien son
    groupe extérieur, donc ``x ^ { a b }``. Le sens est préservé des deux côtés.

    L'étoile de ``\\operatorname*`` est mangée ici pour la même raison : ``_ATOME`` coupe la
    commande avant l'étoile, si bien que l'entrée ``\\operatorname*`` de ``_POLICES`` était du
    code mort et que l'étoile survivait — en multiplication.
    """
    sortie: list[str] = []
    i = 0
    while i < len(atomes):
        atome = atomes[i]
        if atome in _POLICES:
            if niveau != NIVEAU_SOUPLE:
                sortie.append(_POLICE_CANONIQUE.get(atome, atome))
                i += 1
                if i < len(atomes) and atomes[i] == "*":
                    i += 1
                continue
            i += 1
            if i < len(atomes) and atomes[i] == "*":
                i += 1
            if i < len(atomes) and atomes[i] == "{":
                fin = _fermeture(atomes, i)
                if fin < 0:
                    # Groupe coupé par la troncature : la police disparaît, son argument
                    # reste, accolade comprise. On ne devine pas où le groupe se fermait.
                    continue
                sortie.extend(_appliquer_polices(atomes[i + 1:fin], niveau))
                i = fin + 1
            continue
        sortie.append(_SYNONYMES.get(atome, atome))
        i += 1
    return sortie


def _replier_accolades(atomes: list[str]) -> list[str]:
    """Trois replis, tous neutres pour le sens, répétés jusqu'au point fixe.

    1. ``{ x }`` -> ``x`` quand le groupe tient en **un** atome. C'est la règle qui
       réconcilie le ``X _ { t }`` du corpus et le ``X_t`` d'un générateur. Elle ne touche
       pas ``{ t + 1 }`` : trois atomes, le groupe est porteur — et elle ne touche surtout
       pas le premier argument de ``\\frac { a + b } { c }``, qu'elle casserait.
    2. ``{ { … } }`` -> ``{ … }``. Sans elle, effacer une police en souple laisse une
       accolade orpheline : ``^ { \\mathrm { b i d } }`` devenait ``^ { { b i d } }`` quand
       le générateur, lui, écrit ``^ { b i d }``. Mesuré : c'est le cas qui faisait échouer
       la restitution la plus fidèle possible d'une formule du corpus.
    3. Un groupe qui enveloppe toute l'expression est retiré : il ne dit rien.
    """
    for _ in range(12):
        sortie, i, change = [], 0, False
        while i < len(atomes):
            if atomes[i] == "{" and i + 1 < len(atomes):
                # La borne ``i + 1 < len`` n'est pas une précaution de style. ``inclus`` est
                # appliqué à des **passages tronqués à 2 500 caractères** : la coupe casse des
                # paires d'accolades, ``_fermeture`` rend alors le dernier index faute de
                # fermante, et le repli lisait au-delà de la liste. Trouvé en production, sur
                # le premier candidat du corpus dont la fenêtre servie finit par une accolade
                # ouvrante.
                fin = _fermeture(atomes, i)
                if fin < 0:       # fermante absente : l'accolade est un atome, pas un groupe
                    sortie.append(atomes[i])
                    i += 1
                    continue
                if fin == i + 1:  # groupe vide
                    i = fin + 1
                    change = True
                    continue
                if fin == i + 2 and atomes[i + 1] not in {"{", "}"}:
                    sortie.append(atomes[i + 1])
                    i = fin + 1
                    change = True
                    continue
                if atomes[i + 1] == "{" and _fermeture(atomes, i + 1) == fin - 1:
                    sortie.extend(atomes[i + 1:fin])
                    i = fin + 1
                    change = True
                    continue
            sortie.append(atomes[i])
            i += 1
        atomes = sortie
        if not change:
            break
    if len(atomes) > 2 and atomes[0] == "{" and _fermeture(atomes, 0) == len(atomes) - 1 >= 0:
        atomes = atomes[1:-1]
    return atomes


def _elaguer_bords(atomes: list[str]) -> list[str]:
    """Ponctuation terminale et séparateurs d'alignement résiduels.

    Le corpus clôt ses équations par ``.`` ou ``,`` — c'est de la ponctuation de phrase,
    pas de la mathématique. Un générateur ne la reproduit pas.
    """
    atomes = [a for a in atomes if a not in {"&", "\\\\", "\\cr", "~"}]
    while atomes and atomes[-1] in {".", ",", ";", ":"}:
        atomes.pop()
    while atomes and atomes[0] in {".", ",", ";", ":"}:
        atomes.pop(0)
    return atomes


# --------------------------------------------------------------------------- interface

def atomes(source: str, niveau: str = NIVEAU_SOUPLE) -> list[str]:
    """Suite d'atomes canoniques d'une expression LaTeX."""
    liste = _atomiser(_pre_nettoyage(source))
    # ``\left.`` et ``\right.`` : le point est un **délimiteur nul**, pas une ponctuation.
    # Effacer ``\left`` en laissant le ``.`` fabriquait un atome que nul générateur n'écrit —
    # et c'est ce point orphelin qui faisait échouer la restitution la plus fidèle de la
    # ligne de base (f001, recouvrement 0,984 et pourtant « faux »).
    sans_taille, saute = [], False
    for atome in liste:
        if saute and atome == ".":
            saute = False
            continue
        saute = atome in _TAILLE
        if atome in _TAILLE or atome in _ESPACES or atome in _ESPACES_ARG:
            continue
        sans_taille.append(atome)
    liste = sans_taille
    liste = _effacer_specificateur_array(liste)
    liste = _appliquer_polices(liste, niveau)
    liste = _replier_accolades(liste)
    return _elaguer_bords(liste)


def normaliser(source: str, niveau: str = NIVEAU_SOUPLE) -> str:
    """Forme canonique lisible : les atomes joints par une espace.

    Le séparateur est indispensable — ``\\alpha`` suivi de ``\\beta`` doit rester distinct
    de la commande ``\\alphabeta``.
    """
    return " ".join(atomes(source, niveau))


def sous_suite(aiguille: list[str], meule: list[str]) -> int:
    """Index de la première occurrence contiguë de ``aiguille`` dans ``meule``, ou ``-1``."""
    n, m = len(aiguille), len(meule)
    if not n or n > m:
        return -1
    premier = aiguille[0]
    for depart in range(m - n + 1):
        if meule[depart] == premier and meule[depart:depart + n] == aiguille:
            return depart
    return -1


def inclus(reference: str, candidat: str, niveau: str = NIVEAU_SOUPLE) -> bool:
    """``reference`` apparaît-elle telle quelle dans ``candidat`` ? Le score principal."""
    a_ref = atomes(reference, niveau)
    return bool(a_ref) and sous_suite(a_ref, atomes(candidat, niveau)) >= 0


def recouvrement_local(reference: str, candidat: str, niveau: str = NIVEAU_SOUPLE) -> float:
    """Meilleur recouvrement d'atomes sur une fenêtre de la taille de la référence.

    **Diagnostic, jamais décision.** Un F1 global punirait une réponse longue pour sa
    longueur ; ce qu'on veut savoir, c'est si la formule est *quelque part*, et à quel point
    elle est abîmée. On glisse donc une fenêtre de la taille exacte de l'or et on garde le
    meilleur F1 — à tailles égales, il vaut ``|intersection| / |or|``.
    """
    a_ref = atomes(reference, niveau)
    a_cand = atomes(candidat, niveau)
    n = len(a_ref)
    if not n or not a_cand:
        return 0.0
    from collections import Counter

    besoin = Counter(a_ref)
    if len(a_cand) <= n:
        commun = sum((besoin & Counter(a_cand)).values())
        return round(2 * commun / (n + len(a_cand)), 4)
    fenetre = Counter(a_cand[:n])
    commun = sum((besoin & fenetre).values())
    meilleur = commun
    for i in range(n, len(a_cand)):
        fenetre[a_cand[i]] += 1
        fenetre[a_cand[i - n]] -= 1
        if fenetre[a_cand[i - n]] == 0:
            del fenetre[a_cand[i - n]]
        commun = sum((besoin & fenetre).values())
        meilleur = max(meilleur, commun)
    return round(meilleur / n, 4)


#: Expressions mathématiques d'un texte libre : ``$$…$$``, ``\\[…\\]``, ``$…$``, ``\\(…\\)``.
_MATHS = re.compile(r"\$\$(.+?)\$\$|\\\[(.+?)\\\]|\$([^$]+?)\$|\\\((.+?)\\\)", re.DOTALL)


def expressions(texte: str) -> list[str]:
    """Les régions mathématiques d'une réponse. Vide si le générateur n'a pas balisé."""
    return [next(g for g in m.groups() if g is not None) for m in _MATHS.finditer(texte or "")]
