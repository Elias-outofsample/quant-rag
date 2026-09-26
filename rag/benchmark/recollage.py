"""La transformation du lot `latex-recolle` : **une seule définition**, partagée.

L'audit, la construction des bras et la mesure importent tous ce module. Une copie de la
transformation dans chacun d'eux est exactement l'écart qui fait dériver un instrument de
ce qu'il prétend mesurer — ``dense_matrix`` a payé cette leçon le 7 septembre 2026 et
appelle désormais ``build_index.imported_points`` telle quelle plutôt que de la
réimplémenter.

Ce qu'elle fait, et les trois invariants qui la prouvent : voir `audit_recollage.py`.

Héritage et écart déclaré par rapport à la sonde `sonde_latex.py` (commit `e11eac6`)
------------------------------------------------------------------------------------
La sonde recollait **atome par atome** (``\\[A-Za-z]+|\\S``). Sur les 147 989 régions du
corpus, cela **change la suite de jetons LaTeX de 4 022 d'entre elles** — ``\\ P``
devient ``\\P`` (une autre commande), ``\\ {`` devient ``\\{`` (une accolade échappée là
où il y avait une espace de contrôle puis un groupe). Le garde-fou de la sonde ne voyait
que le cas ``\\in`` + ``t`` -> ``\\int``, parce qu'il exigeait ``precedent[1:].isalpha()``
— ce qui est faux pour une contre-oblique nue.

Ce module recolle donc **jeton par jeton** selon la lecture LaTeX (mot de contrôle,
symbole de contrôle, caractère), émet chaque jeton tel quel — une espace de contrôle
``\\ `` garde son espace — et n'insère une espace que là où la jointure changerait la
lecture. Résultat : **0 région sur 147 989** change de suite de jetons (invariant I3).
"""
from __future__ import annotations

import random
import re
from collections import Counter

#: Régions mathématiques candidates. ``$$…$$`` d'abord, sans quoi ``$…$`` couperait un
#: bloc affiché en deux. Même regex que `diagnostic_formules._MATHS` et `sonde_latex`.
MATHS = re.compile(r"\$\$.+?\$\$|\$[^$\n]+\$", re.DOTALL)
#: Jeton LaTeX : mot de contrôle (``\`` + lettres), symbole de contrôle (``\`` + un
#: caractère, l'espace comprise), sinon un caractère non blanc.
JETON = re.compile(r"\\[A-Za-z]+|\\.|\S", re.DOTALL)
#: Déclaration d'environnement : son nom est de la syntaxe, pas de la prose.
ENVIRONNEMENT = re.compile(r"\\(?:begin|end)\s*\{\s*[A-Za-z*]+\s*\}")
COMMANDE = re.compile(r"\\[A-Za-z]+")
#: Structure qu'une formule porte et qu'une phrase ne porte pas.
STRUCTURE = re.compile(r"[_^{}]")
#: Mot : trois lettres d'affilée ou plus. Une formule éclatée n'en a aucun une fois sa
#: syntaxe retirée — ses lettres sont séparées par des espaces.
MOT = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ]{3,}")
BLANC = re.compile(r"\s+")


def jetons(expression: str) -> list[str]:
    return JETON.findall(expression)


def recoller(expression: str) -> str:
    """Resserre une région mathématique sans changer sa lecture LaTeX.

    ``V _ { i , t } ^ { \\mathrm { b i d } }`` -> ``V_{i,t}^{\\mathrm{bid}}``.
    """
    sortie: list[str] = []
    for jeton in jetons(expression):
        if sortie:
            precedent = sortie[-1]
            mot_de_controle = (precedent.startswith("\\") and len(precedent) > 1
                               and precedent[1:].isalpha())
            if mot_de_controle and jeton[:1].isalpha():
                sortie.append(" ")      # ``\in`` + ``t`` ne doit pas devenir ``\int``
        sortie.append(jeton)
    return "".join(sortie)


def mots_de_prose(region: str) -> list[str]:
    """Mots de la région, **syntaxe LaTeX retirée** : noms d'environnements puis commandes.

    Sans le retrait des noms d'environnements, les 378 régions qui portent plusieurs
    ``\\begin{array}`` — des matrices, le cœur du produit — seraient prises pour de la
    prose et refusées. C'est la correction que l'audit a imposée à une première version
    de cette garde.
    """
    return MOT.findall(COMMANDE.sub(" ", ENVIRONNEMENT.sub(" ", region)))


def est_mathematique(region: str) -> bool:
    """La garde. Une région n'est recollée que si **les deux** critères le disent.

    1. elle porte une signature LaTeX — une commande nommée, ou l'un de ``_^{}`` ;
    2. il n'y reste aucun mot de 3 lettres ou plus, syntaxe retirée.

    Le corpus est financier : ``$`` y est aussi le signe du dollar, et ``\\$[^$\\n]+\\$``
    apparie volontiers deux montants avec la prose entre eux — ``$0.02 a day, and a put
    writer would make \\$``. Recoller celle-là donnerait ``$0.02aday,and…``. **500
    passages** du corpus portent au moins une région de cette sorte ; la garde les écarte.

    Conjonctive et conservatrice : le coût de refuser à tort une formule est nul (son
    vecteur ne bouge pas), celui d'accepter à tort une phrase est une chaîne plongée
    corrompue.
    """
    if not (COMMANDE.search(region) or STRUCTURE.search(region)):
        return False
    return not mots_de_prose(region)


def recoller_passage(texte: str) -> str:
    """Recolle les régions que la garde reconnaît ; la prose et les montants restent intacts."""
    return MATHS.sub(lambda m: recoller(m.group()) if est_mathematique(m.group()) else m.group(), texte)


# --------------------------------------------------------------------------- placebo

def placebo_passage(texte: str, cible: int, rng: random.Random) -> str:
    """Retire des caractères blancs **au hasard** jusqu'à la longueur ``cible``.

    Le bras placebo de ce lot est **apparié en longueur au candidat** (§2 et §4.1 du plan) :
    c'est la limite déclarée de la sonde `sonde_latex.py`, dont le placebo gardait la
    longueur de l'ORIGINAL quand le recollage la raccourcit. Ici le placebo retire le même
    *nombre* de caractères blancs que le recollage, mais à des positions tirées au hasard
    sur tout le passage au lieu des jointures d'atomes LaTeX. Même longueur, même nombre de
    blancs ôtés, même sorte d'édition — seul le *lieu* change, donc seule la justesse change.

    Sa destructivité est connue et mesurée (cf. le rapport) : le recollage ôte presque tous
    les blancs des régions mathématiques, donc ``cible`` est grande, donc le placebo mange
    aussi des espaces de prose et colle des mots. C'est pourquoi la règle de décision ne
    repose pas seulement sur « battre le placebo » mais aussi sur un seuil absolu.
    """
    a_retirer = len(texte) - cible
    if a_retirer <= 0:
        return texte
    positions = [i for i, c in enumerate(texte) if c.isspace()]
    a_retirer = min(a_retirer, len(positions))
    retires = set(rng.sample(positions, a_retirer))
    return "".join(c for i, c in enumerate(texte) if i not in retires)


# --------------------------------------------------------------------------- invariants

def sans_blanc(texte: str) -> str:
    return BLANC.sub("", texte)


def violations(avant: str, apres: str) -> dict[str, object]:
    """I1, I2, I3 pour un passage — la preuve que la transformation ne corrompt rien.

    - **I1** ``sans_blanc(avant) == sans_blanc(après)`` : seul le blanc bouge.
    - **I2** aucun mot de 3+ lettres du texte d'avant n'est détruit (en créer est permis :
      ``b i d`` -> ``bid`` est la réparation attendue ; en détruire serait une corruption
      de prose, que I1 ne voit pas puisqu'elle ne retire que du blanc).
    - **I3** la suite de jetons LaTeX de chaque région recollée est inchangée.
    """
    manquants = sorted((Counter(MOT.findall(avant)) - Counter(MOT.findall(apres))).elements())
    i3 = [m.group() for m in MATHS.finditer(avant)
          if est_mathematique(m.group()) and jetons(recoller(m.group())) != jetons(m.group())]
    return {"I1": sans_blanc(avant) != sans_blanc(apres),
            "I2": manquants[:8], "I3": len(i3)}
