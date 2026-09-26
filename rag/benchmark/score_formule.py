"""Score de la famille `formula` — mince par construction, et il l'assume.

Pourquoi ce module tient en trois questions et pas en un algorithme
-------------------------------------------------------------------
Toute la difficulté d'une formule est déjà tranchée par ``latex_norme`` : MinerU rend les
mathématiques du corpus éclatées caractère par caractère — sur 312 blocs ``$$ … \\tag{n} … $$``
prélevés un par document, la médiane fait 43 atomes et le maximum 186 — et un générateur les
réécrit compactes, sans police, sans ``\\left``. Comparer des chaînes mesurerait le rendu de
MinerU. Ce module n'ajoute donc **aucune** normalisation : il importe la forme canonique et
se contente de dire *où* il a cherché, *ce* qu'il a trouvé, et *ce qu'il refuse de décider*.

Les trois décisions, et leur motif mesuré
------------------------------------------
**Le score principal est une inclusion d'atomes au NIVEAU_SOUPLE**, jamais un recouvrement.
``recouvrement`` est publié à côté et n'a aucun droit de vote : mesuré sur ``a + b = c``
contre ``a - b = c``, il vaut 0,80 alors que la formule est fausse — un seuil sur ce
chiffre aurait compté juste un changement de signe. Il sert à trier les échecs (« abîmée »
contre « absente »), pas à les prononcer.

**On cherche d'abord dans les régions balisées, puis dans le texte entier.** Un générateur
ne balise pas toujours : refuser une restitution correcte parce qu'elle est écrite sans
``$$`` mesurerait le formatage. Le risque de ce repli est réel et il est nommé : hors
balise, la prose est atomisée lettre à lettre, donc un or court se retrouve *dans un mot* —
``inclus("V a R", "la VaR du portefeuille")`` est vrai. Il n'est pas conjuré par une
astuce, il est **rendu mesurable** : ``voie`` dit par quel chemin le « juste » a été
obtenu, et ``or_court`` signale un or trop maigre pour que ce chemin soit sûr. Une campagne
qui trouve beaucoup de ``voie == "texte_entier"`` sur des ors courts sait quoi relire.
Contrôle : sur 1 248 paires de documents distincts, le repli n'a produit aucune collision
avec des ors de 12 atomes ou plus.

**Un leurre qui EST l'or n'est pas un leurre.** Le contrôle négatif compare la réponse aux
formules des autres chunks du même document. Mais le corpus répète ses équations : sur 279
documents pourvus d'au moins un leurre interne, 7 portaient la même équation dans deux
chunks sous deux ``\\tag`` différents (``A.9a`` et ``7a`` pour ``chunk-08a977bdd7b6022f``,
``19`` et ``20`` pour ``chunk-043d74bae132f519``) — 8 leurres en tout sur 8 752 comparés.
Compter ces répétitions comme des leurres touchés ferait
passer un scoreur juste pour un scoreur lâche. Ils sont donc écartés — et comptés, dans
``leurres_ecartes``, pour que l'écart ne soit pas silencieux.

Ce que ce module refuse de faire
---------------------------------
Il n'appelle rien — ni réseau, ni modèle, ni index : ``score`` est une fonction de deux
chaînes. Il ne répare pas le corpus, ne corrige pas ``latex_norme`` et ne juge pas la
prose autour de la formule. Il ne tranche pas non plus le cas des environnements
multi-lignes : ``or_multiligne`` les signale — leurs ``&`` et ``\\\\`` sont une convention
d'alignement qu'un générateur n'a aucune raison de reproduire — mais c'est à la sélection
des ors de les écarter, pas au scoreur de les repêcher.

Défaut connu de la brique importée, mesuré ici et **non corrigé** (le fichier ne nous
appartient pas) : au NIVEAU_SOUPLE, effacer une commande de police laisse ses accolades
orphelines quand l'argument fait plus d'un atome. ``a + \\mathrm{cov}(x,y)`` devient
``a + { c o v } ( x , y )`` là où le générateur écrit ``a + c o v ( x , y )``. C'est 60 des
312 formules de l'étude (19,2 %) qui échouent sur la seule réécriture « police retirée ».
Voir ``tests/test_v4_formule.py``, cas marqués ``xfail``.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import latex_norme  # noqa: E402

#: Redéfini ici, et pas importé de ``pipeline`` : ce scoreur doit rester chargeable hors
#: ligne, sans traîner derrière lui la chaîne de génération. La convention est celle de
#: ``pipeline.abstained`` — le jeton n'importe où, sur une frontière de mot.
JETON_ABSTENTION = "INSUFFICIENT_EVIDENCE"
_ABSTENTION = re.compile(rf"\b{JETON_ABSTENTION}\b")

#: En deçà, un or n'est plus discriminant hors balise : la prose atomisée lettre à lettre
#: le contient par accident. Ne bloque rien — lève ``or_court``, que la campagne recoupe
#: avec ``voie``. Le chiffre est celui du plancher de l'étude adversariale (312 formules,
#: aucune collision inter-documents à partir de 12 atomes).
ATOMES_SUSPECTS = 12


def abstenue(texte: str) -> bool:
    """La réponse porte-t-elle le jeton d'abstention ?"""
    return bool(_ABSTENTION.search(texte or ""))


def _sources(texte: str) -> tuple[list[str], list[str]]:
    """Les régions balisées d'une réponse, puis le repli — le texte entier.

    Les régions sont éprouvées **une par une**, jamais concaténées : recoller ``$a=$`` et
    ``$b$`` fabriquerait une contiguïté d'atomes que la réponse n'affirme pas.
    """
    regions = latex_norme.expressions(texte or "")
    return regions, [texte or ""]


def _premiere_inclusion(formule_or: str, regions: list[str], repli: list[str],
                        niveau: str) -> str | None:
    """Nom de la voie par laquelle l'or a été trouvé, ou ``None``."""
    if any(latex_norme.inclus(formule_or, source, niveau) for source in regions):
        return "balisee"
    if any(latex_norme.inclus(formule_or, source, niveau) for source in repli):
        return "texte_entier"
    return None


def _leurres_utiles(formule_or: str, leurres) -> tuple[list[str], int]:
    """Écarte les leurres qui ne discriminent rien, et compte combien ont été écartés.

    Un leurre est inutile s'il ne dit pas autre chose que l'or : même suite d'atomes (la
    même équation répétée sous un autre ``\\tag``, cas mesuré 3 fois sur 288 documents),
    ou l'un inclus dans l'autre — auquel cas le trouver ne prouve rien sur la lâcheté du
    scoreur, seulement que le corpus se répète.
    """
    atomes_or = latex_norme.atomes(formule_or or "")
    retenus, ecartes = [], 0
    for leurre in leurres or ():
        atomes_leurre = latex_norme.atomes(leurre or "")
        if not atomes_leurre:
            ecartes += 1
            continue
        if (atomes_leurre == atomes_or
                or latex_norme.sous_suite(atomes_or, atomes_leurre) >= 0
                or latex_norme.sous_suite(atomes_leurre, atomes_or) >= 0):
            ecartes += 1
            continue
        retenus.append(leurre)
    return retenus, ecartes


def score(reponse_texte: str, formule_or: str, leurres=None) -> dict:
    """Restitution d'une formule : un verdict, et tout ce qui permet de le contester.

    ``juste`` est le seul chiffre décisionnel. ``juste_strict``, ``recouvrement``,
    ``balisee``, ``voie`` et ``or_court`` sont des diagnostics : ils disent *pourquoi* le
    verdict est tombé, et rien de plus. ``leurre_inclus`` est le contrôle négatif — vrai
    signale un scoreur (ou une réponse) qui confond deux formules du même document.
    """
    texte = reponse_texte or ""
    atomes_or = latex_norme.atomes(formule_or or "")
    regions, repli = _sources(texte)

    voie = _premiere_inclusion(formule_or, regions, repli, latex_norme.NIVEAU_SOUPLE) if atomes_or else None
    voie_stricte = _premiere_inclusion(formule_or, regions, repli, latex_norme.NIVEAU_STRICT) if atomes_or else None

    # Borne supérieure : le meilleur recouvrement sur toutes les sources envisagées. Sur le
    # texte entier, la fenêtre glissante peut chevaucher de la prose — raison de plus pour
    # que ce chiffre ne décide jamais.
    recouvrement = max(
        (latex_norme.recouvrement_local(formule_or, source, latex_norme.NIVEAU_SOUPLE)
         for source in regions + repli), default=0.0) if atomes_or else 0.0

    retenus, ecartes = _leurres_utiles(formule_or, leurres)
    if leurres is None:
        leurre_inclus = None
        leurres_touches = []
    else:
        leurres_touches = [leurre for leurre in retenus
                           if _premiere_inclusion(leurre, regions, repli,
                                                  latex_norme.NIVEAU_SOUPLE) is not None]
        leurre_inclus = bool(leurres_touches)

    return {
        "juste": voie is not None,
        "juste_strict": voie_stricte is not None,
        "recouvrement": recouvrement,
        "balisee": bool(regions),
        "abstenue": abstenue(texte),
        "leurre_inclus": leurre_inclus,
        # --- diagnostics : aucun n'a de droit de vote
        "voie": voie,
        "atomes_or": len(atomes_or),
        "or_court": 0 < len(atomes_or) < ATOMES_SUSPECTS,
        "or_multiligne": latex_norme.environnement_multiligne(formule_or or ""),
        "leurres_retenus": len(retenus),
        "leurres_ecartes": ecartes,
        "leurres_touches": leurres_touches,
        "normalisee_or": latex_norme.normaliser(formule_or or ""),
    }
