"""Sélection des cellules d'or de la famille `table_cell` — et surtout, tout ce qu'elle refuse.

Ce que la question demande, et ce qui la rend mesurable
-------------------------------------------------------
« Dans le tableau intitulé X du document Y, quelle valeur est rapportée pour la ligne L
dans la colonne C ? » Une telle question ne mesure la **lecture d'un tableau** que si
personne — ni un lecteur, ni un correcteur — ne peut justifier une autre réponse que la
cellule visée. Tout ce module est là pour écarter les tableaux où ce n'est pas le cas.
Il ne fabrique pas de questions : il rend des cellules dont la lecture est sans ambiguïté,
et il compte, motif par motif, ce qu'il a refusé.

Les faits de corpus qui commandent chaque refus (mesurés sur 26 120 chunks, 418 documents,
signature 5530cba145)
-----------------------------------------------------------------------------------------
**MinerU fragmente les grands tableaux, en répétant légende et en-tête.** Le document
``doc-4cd89168f8dede32`` porte deux chunks intitulés ``Table: Theta per 1mn Cash Gamma``,
mêmes colonnes, **valeurs différentes** : à la ligne « 80 % », colonne « 4 Years », l'un dit
``-8,338`` et l'autre ``-10,045``. Une question posée sur ce tableau n'a pas de bonne
réponse. 126 couples de fragments partagent ainsi une légende. D'où l'exigence : la légende
doit être portée par **un seul** chunk du document.

**La production sert 2 500 caractères par passage** (``pipeline.CARACTERES_SERVIS``). Un
chunk plus long verrait sa fin coupée : une question dont la cellule tombe hors de la
fenêtre mesurerait la troncature, pas la lecture. Le chunk doit donc tenir entier dans la
fenêtre — 5 622 des 5 648 chunks de type ``table`` y tiennent, l'exigence ne coûte presque
rien.

**Les en-têtes à deux niveaux ne survivent pas au Markdown.** ``chunk-18e44d872a932d47``
rend ``|   | Short Position |   | Hedge Portfolio |   |`` : les colonnes fusionnées
deviennent des cellules vides, et il devient impossible de nommer une colonne sans mentir.
Un en-tête de données vide, répété, ou numérique fait donc rejeter le tableau entier.

**Une valeur qui figure deux fois n'est plus une preuve de lecture.** Dans le même
``chunk-18e44d872a932d47``, les colonnes « Short » et « Hedge » sont identiques ligne à
ligne : répondre ``0.250`` ne dit pas laquelle a été lue. L'or doit donc être **unique**
dans son tableau — et la comparaison ignore le drapeau pourcentage, parce qu'un lecteur qui
laisse tomber l'unité confondrait quand même ``5`` et ``5 %``.

**Une parenthèse n'est pas un signe.** ``(0.089)`` est un écart-type dans un tableau de
régression et un nombre négatif dans un tableau de flux ; ``norme_valeurs`` explique
pourquoi il n'est pas question de trancher. Ce qui est ambigu ne devient pas de l'or.

Ce que ce module ne fait pas
-----------------------------
Aucun appel réseau, aucun LLM, aucune écriture : il lit le corpus et rend des dicts. Il ne
répare rien dans le corpus (règle du §2 du handoff : absorber les artefacts, pas les
réparer) — un tableau mal rendu par MinerU est **rejeté**, jamais recousu. Il ne juge pas
non plus une réponse : c'est le rôle de ``score_tableau``.

    .venv/bin/python rag/benchmark/selection_tableaux.py --tete 5
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import norme_valeurs  # noqa: E402

#: Ce que la production sert par passage. Recopié plutôt qu'importé de ``pipeline`` :
#: importer le pipeline tirerait ``llm.py`` et ses clés d'API dans un module qui doit rester
#: exécutable hors ligne. ``tests/test_v4_tableau.py`` relit la constante dans le source de
#: ``pipeline.py`` pour que la copie ne puisse pas dériver en silence.
FENETRE_SERVIE = 2500

#: Une légende MinerU. Le corpus n'en connaît qu'une graphie : la ligne commence par ce
#: préfixe, y compris quand ce qui suit est « Figure 192 » — MinerU range aussi les captures
#: de figures tabulaires ici.
PREFIXE_LEGENDE = "Table:"

#: La ligne ``|---|---|`` qui sépare l'en-tête des données. Les deux-points d'alignement
#: (``:---:``) sont tolérés : ils ne changent rien au contenu.
_SEPARATRICE = re.compile(r"\|(?:\s*:?-+:?\s*\|)+")

MINIMUM_LIGNES = 3
MINIMUM_COLONNES = 2
QUOTA_DOCUMENT = 2


class Cellule:
    """Une cellule et sa position **absolue** dans le texte du chunk.

    La position n'est pas un confort : elle permet de vérifier qu'une cellule d'or est bien
    dans les 2 500 caractères servis, et de relire l'or dans le chunk sans refaire l'analyse.
    """

    __slots__ = ("texte", "debut")

    def __init__(self, texte: str, debut: int):
        self.texte = texte
        self.debut = debut

    def __repr__(self) -> str:  # pragma: no cover - confort de mise au point
        return f"Cellule({self.texte!r}, debut={self.debut})"


class Ligne:
    """Un libellé de ligne et ses cellules de **données** — le libellé n'en fait pas partie."""

    __slots__ = ("libelle", "cellules")

    def __init__(self, libelle: Cellule, cellules: list):
        self.libelle = libelle
        self.cellules = cellules

    def __repr__(self) -> str:  # pragma: no cover - confort de mise au point
        return f"Ligne({self.libelle.texte!r}, {len(self.cellules)} cellules)"


class Tableau:
    """Un tableau Markdown analysé, ou la raison pour laquelle il est inutilisable.

    ``motif`` vaut ``None`` quand le tableau est exploitable. Sinon il nomme le **premier**
    critère enfreint : on ne cumule pas les motifs, sans quoi le recensement compterait
    plusieurs fois le même tableau et personne ne saurait plus ce qui coûte le plus cher.

    ``entetes`` et ``lignes[i].cellules`` sont alignés index par index. La cellule de coin
    (en haut à gauche) est mise à part dans ``coin`` : elle nomme la colonne des libellés,
    que la question ne cite jamais, et elle est vide dans la moitié des tableaux du corpus.
    L'exiger non vide reviendrait à rejeter des tableaux parfaitement lisibles.
    """

    __slots__ = ("legende", "coin", "entetes", "lignes", "motif")

    def __init__(self, legende=None, coin=None, entetes=None, lignes=None, motif=None):
        self.legende = legende
        self.coin = coin
        self.entetes = entetes or []
        self.lignes = lignes or []
        self.motif = motif

    @property
    def utilisable(self) -> bool:
        return self.motif is None

    def toutes_cellules(self) -> list:
        """Libellés **et** données. C'est la population sur laquelle l'unicité se juge :
        un libellé de ligne numérique (« 30 », « 1080 ») est une valeur lisible comme une
        autre, et une réponse qui le recopie doit pouvoir être distinguée de l'or."""
        return [c for ligne in self.lignes for c in [ligne.libelle] + list(ligne.cellules)]

    def __repr__(self) -> str:  # pragma: no cover - confort de mise au point
        if self.motif:
            return f"Tableau(rejeté : {self.motif})"
        return f"Tableau({len(self.lignes)}x{len(self.entetes)}, {self.legende!r})"


# --------------------------------------------------------------------------- analyse

def _cellules_de_ligne(ligne: str, base: int) -> list:
    """Les cellules d'une ligne à pipes, avec leur position absolue.

    On découpe entre barres consécutives plutôt qu'avec ``split('|')`` pour garder les
    décalages : sans eux, impossible de dire *où* est l'or dans le chunk.
    """
    depouillee = ligne.rstrip()
    if not depouillee.lstrip().startswith("|") or not depouillee.endswith("|"):
        return []
    barres = [i for i, caractere in enumerate(depouillee) if caractere == "|"]
    cellules = []
    for gauche, droite in zip(barres, barres[1:]):
        brut = depouillee[gauche + 1:droite]
        texte = brut.strip()
        decalage = len(brut) - len(brut.lstrip())
        cellules.append(Cellule(texte, base + gauche + 1 + decalage))
    return cellules


def _est_separatrice(ligne: str) -> bool:
    depouillee = ligne.strip()
    return bool(depouillee) and bool(_SEPARATRICE.fullmatch(depouillee))


def _lignes_positionnees(texte: str):
    """(ligne, position absolue du premier caractère) pour tout le texte du chunk."""
    position = 0
    for ligne in (texte or "").split("\n"):
        yield ligne, position
        position += len(ligne) + 1


def analyser_tableau(texte: str) -> Tableau:
    """Analyse un chunk de type ``table`` : légende, en-têtes, lignes — ou le motif du rejet.

    Rend toujours un ``Tableau`` : un rejet est une information, pas une absence. Les
    critères sont éprouvés dans l'ordre du moins cher au plus cher, et le premier qui
    échoue arrête l'analyse.
    """
    lignes_brutes = list(_lignes_positionnees(texte))

    # 1. La légende, en tête. Elle est ce que la question citera pour désigner le tableau ;
    #    sans elle, la question devrait décrire le tableau, et décrire c'est déjà répondre.
    legende = None
    index_legende = None
    for numero, (ligne, _) in enumerate(lignes_brutes):
        if not ligne.strip():
            continue
        if ligne.strip().startswith(PREFIXE_LEGENDE):
            legende = ligne.strip()[len(PREFIXE_LEGENDE):].strip()
            index_legende = numero
        break
    if not legende:
        return Tableau(motif="legende_absente")

    # 2. Un seul tableau par chunk. Deux légendes ou deux blocs de pipes séparés, et
    #    « le tableau intitulé X » ne désigne plus une seule grille.
    numeros_pipes = [n for n, (ligne, _) in enumerate(lignes_brutes)
                     if ligne.strip().startswith("|")]
    if not numeros_pipes:
        return Tableau(legende=legende, motif="table_illisible")
    debut, fin = numeros_pipes[0], numeros_pipes[-1]
    if len(numeros_pipes) != fin - debut + 1:
        return Tableau(legende=legende, motif="plusieurs_tableaux")
    autres_legendes = [n for n, (ligne, _) in enumerate(lignes_brutes)
                       if n != index_legende and ligne.strip().startswith(PREFIXE_LEGENDE)]
    if autres_legendes:
        return Tableau(legende=legende, motif="plusieurs_tableaux")

    # 3. En-tête + séparatrice + données. Sans séparatrice, la première ligne de données
    #    passerait pour un en-tête et toutes les colonnes seraient mal nommées.
    if fin - debut < 2 or not _est_separatrice(lignes_brutes[debut + 1][0]):
        return Tableau(legende=legende, motif="table_illisible")
    entete = _cellules_de_ligne(*lignes_brutes[debut])
    if not entete:
        return Tableau(legende=legende, motif="table_illisible")
    if len(entete) - 1 < MINIMUM_COLONNES:
        return Tableau(legende=legende, motif="colonnes_insuffisantes")

    lignes = []
    for ligne, base in lignes_brutes[debut + 2:fin + 1]:
        cellules = _cellules_de_ligne(ligne, base)
        # Une ligne de largeur différente de l'en-tête est un tableau mal rendu : la
        # correspondance cellule -> colonne est perdue, on ne la devine pas.
        if len(cellules) != len(entete):
            return Tableau(legende=legende, motif="colonnes_irregulieres")
        lignes.append(Ligne(cellules[0], cellules[1:]))
    if len(lignes) < MINIMUM_LIGNES:
        return Tableau(legende=legende, motif="lignes_insuffisantes")

    coin, entetes = entete[0], entete[1:]
    tableau = Tableau(legende=legende, coin=coin, entetes=entetes, lignes=lignes)

    # 4. Nommer une colonne et une ligne sans ambiguïté.
    titres = [cellule.texte for cellule in entetes]
    if any(not titre for titre in titres):
        return Tableau(legende=legende, coin=coin, entetes=entetes, lignes=lignes,
                       motif="entetes_vides")
    if len(set(titres)) != len(titres):
        return Tableau(legende=legende, coin=coin, entetes=entetes, lignes=lignes,
                       motif="entetes_dupliques")
    # Un en-tête numérique se confondrait avec une valeur dans la question comme dans la
    # réponse : « la colonne 2019 » et « la valeur 2019 » ne se distinguent plus.
    if any(valeur_de(titre) is not None for titre in titres):
        return Tableau(legende=legende, coin=coin, entetes=entetes, lignes=lignes,
                       motif="entetes_numeriques")

    libelles = [ligne.libelle.texte for ligne in lignes]
    if any(not libelle for libelle in libelles):
        return Tableau(legende=legende, coin=coin, entetes=entetes, lignes=lignes,
                       motif="libelles_vides")
    if len(set(libelles)) != len(libelles):
        return Tableau(legende=legende, coin=coin, entetes=entetes, lignes=lignes,
                       motif="libelles_dupliques")

    # 5. Les deux signatures d'un en-tête à plusieurs niveaux aplati par MinerU.
    #    a) Le libellé de coin réapparaît en libellé de ligne : la « ligne » est en fait le
    #       second niveau d'en-tête. ``chunk-01a5ca6c677898c2`` (« Method » deux fois) et
    #       ``chunk-7291ca40daea0f82`` (« Strike » deux fois) sont dans ce cas.
    if coin.texte and any(libelle.casefold() == coin.texte.casefold() for libelle in libelles):
        return Tableau(legende=legende, coin=coin, entetes=entetes, lignes=lignes,
                       motif="entete_repetee")
    #    b) Une colonne entièrement vide : la grille rendue a plus de colonnes que le
    #       tableau n'en a, donc l'appariement cellule -> en-tête n'est plus sûr, y compris
    #       pour les colonnes qui, elles, sont remplies.
    if any(all(not ligne.cellules[colonne].texte for ligne in lignes)
           for colonne in range(len(entetes))):
        return Tableau(legende=legende, coin=coin, entetes=entetes, lignes=lignes,
                       motif="colonne_vide")
    #    c) Une première ligne de données sans le moindre nombre, alors que les suivantes en
    #       ont : c'est un second niveau d'en-tête tombé dans le corps du tableau
    #       (``chunk-47fcb31e0bd1c8b9`` : « Parameter | $\\beta_1$ | … » sous une ligne
    #       d'en-tête déjà pleine). Les colonnes portent alors deux noms concurrents et la
    #       question en citerait un pour une valeur rangée sous l'autre.
    if (all(valeur_de(cellule.texte) is None for cellule in lignes[0].cellules)
            and any(valeur_de(cellule.texte) is not None
                    for ligne in lignes[1:] for cellule in ligne.cellules)):
        return Tableau(legende=legende, coin=coin, entetes=entetes, lignes=lignes,
                       motif="entete_secondaire")
    return tableau


# --------------------------------------------------------------------------- cellules

def canonique(valeur) -> str:
    """Écriture stable d'une valeur, précision comprise : ``0.250`` reste ``0.250``.

    Le nombre de décimales n'est pas cosmétique — c'est lui qui fixe la tolérance de
    ``norme_valeurs.egales``. L'écrire en notation scientifique perdrait cette information.
    """
    if valeur is None:
        return ""
    return format(valeur.decimal, "f") + ("%" if valeur.pourcentage else "")


def valeur_de(texte):
    """``norme_valeurs.lire`` protégé — une cellule sans le moindre chiffre n'est pas une valeur.

    Le garde n'est pas une précaution de style. ``norme_valeurs.lire('-')`` **lève un
    IndexError** : la boucle qui consomme les signes de tête teste ``noyau[:1] in '+-'``, et
    la chaîne vide est sous-chaîne de tout, donc elle ne s'arrête jamais. Or le tiret seul
    est la façon dont ce corpus écrit une case manquante — il y en a dans des centaines de
    tableaux. ``norme_valeurs`` est un module partagé, gelé pour ce chantier : on ne le
    corrige pas ici, on ne lui donne pas une entrée qu'il ne sait pas lire. Le défaut est
    signalé au chantier qui possède le fichier.
    """
    if not any(caractere.isdigit() for caractere in texte or ""):
        return None
    return norme_valeurs.lire(texte)


def _sans_unite(valeur):
    """La même valeur, drapeau pourcentage effacé.

    Une sélection qui laisserait ``5`` et ``5 %`` cohabiter dans un tableau parierait sur
    un lecteur qui n'oublie jamais l'unité. On ne parie pas là-dessus.
    """
    return norme_valeurs.Valeur(valeur.decimal, False, False, valeur.decimales, valeur.source)


def acceptable_pour(reference, candidate) -> bool:
    """``candidate`` serait-elle acceptée par un correcteur dont l'or est ``reference`` ?

    C'est ``norme_valeurs.egales`` — donc la tolérance de la **référence**, pas celle du
    candidat — l'unité en moins. Le sens compte : une légende « TABLE 1. » ne rend pas
    ambiguë une cellule ``1.05`` (personne ne lit ``1.05`` dans « TABLE 1 »), alors qu'une
    légende qui cite ``0.936`` rend ambiguë la cellule ``0.936``.
    """
    if reference is None or candidate is None:
        return False
    return norme_valeurs.egales(_sans_unite(reference), _sans_unite(candidate))


def confondables(premiere, seconde) -> bool:
    """Deux valeurs qu'une même réponse pourrait satisfaire, **dans un sens ou dans l'autre**.

    L'unicité dans un tableau se juge ainsi, et non dans le seul sens or -> autre cellule :
    si l'or est ``4.6`` et la cellule voisine ``4.57``, une réponse recopiée de la voisine
    tombe dans la demi-unité du dernier rang de l'or et serait comptée juste. Le tableau est
    alors ambigu même si l'inverse ne l'est pas. Deux sens, donc — la sélection confond ici
    plus qu'il ne serait strictement nécessaire, et c'est le sens sûr de l'erreur.
    """
    return acceptable_pour(premiere, seconde) or acceptable_pour(seconde, premiere)


def cellules_eligibles(tableau: Tableau, motifs: Counter | None = None) -> list:
    """Les cellules de données qui peuvent devenir un or, dans l'ordre de lecture.

    Chaque rejet est compté : c'est le seul moyen de savoir si la famille échoue sur la
    structure des tableaux ou sur la rareté des valeurs uniques.
    """
    compte = motifs if motifs is not None else Counter()
    if not tableau.utilisable:
        return []
    toutes = tableau.toutes_cellules()
    valeurs = {id(cellule): valeur_de(cellule.texte) for cellule in toutes}
    valeurs_legende = norme_valeurs.nombres(tableau.legende or "")

    eligibles = []
    for numero_ligne, ligne in enumerate(tableau.lignes):
        for numero_colonne, cellule in enumerate(ligne.cellules):
            valeur = valeurs[id(cellule)]
            if valeur is None:
                compte["cellule_non_numerique"] += 1
                continue
            if valeur.parenthesee:
                compte["cellule_parenthesee"] += 1
                continue
            if any(acceptable_pour(valeur, autre) for autre in valeurs_legende):
                # La légende « Table 3 » donnerait la réponse « 3 » sans lire la grille — et
                # les légendes de ce corpus citent volontiers leurs propres chiffres
                # (« The SPY value of 0.936 is in the right tail »).
                compte["valeur_dans_legende"] += 1
                continue
            if any(cellule is not autre and confondables(valeur, valeurs[id(autre)])
                   for autre in toutes):
                compte["valeur_non_unique"] += 1
                continue
            eligibles.append({
                "ligne": numero_ligne,
                "colonne": numero_colonne,
                "libelle_ligne": ligne.libelle.texte,
                "entete_colonne": tableau.entetes[numero_colonne].texte,
                "valeur_brute": cellule.texte,
                "valeur_canonique": canonique(valeur),
                "position": cellule.debut,
            })
    return eligibles


# --------------------------------------------------------------------------- corpus

def _chunks_par_document(index) -> dict:
    par_document = defaultdict(list)
    for chunk_id, chunk in index.chunks.items():
        par_document[chunk.get("document_id")].append(chunk_id)
    return par_document


def _legende_partagee(index, voisins: list, chunk_id: str, legende: str) -> bool:
    """La légende figure-t-elle dans un **autre** chunk du même document ?

    On cherche la légende dans le texte entier des voisins, pas seulement dans leur
    première ligne, et sans filtrer sur ``content_type`` : un fragment de suite peut
    arriver en ``mixed``, et il rendrait la question tout aussi ambiguë.
    """
    for voisin in voisins:
        if voisin == chunk_id:
            continue
        if legende in (index.chunks[voisin].get("text") or ""):
            return True
    return False


def candidats(index, quota_document: int = QUOTA_DOCUMENT,
              motifs: Counter | None = None, limite: int | None = None) -> list:
    """Les cellules d'or du corpus, prêtes à devenir des questions ``table_cell``.

    Au plus **une** cellule par tableau — deux cellules du même tableau mesureraient deux
    fois la même chose — et au plus ``quota_document`` par document, pour qu'une famille de
    trente questions ne soit pas la lecture d'un seul livre.

    ``motifs`` reçoit le décompte des rejets si on le fournit ; un chunk n'est compté que
    dans le **premier** motif qui le rejette. Parcours trié par ``chunk_id`` : deux
    exécutions rendent la même liste, dans le même ordre.
    """
    compte = motifs if motifs is not None else Counter()
    par_document = _chunks_par_document(index)
    retenus_par_document: Counter = Counter()
    sortie = []

    for chunk_id in sorted(index.chunks):
        chunk = index.chunks[chunk_id]
        if chunk.get("content_type") != "table":
            compte["type_non_table"] += 1
            continue
        texte = chunk.get("text") or ""
        if len(texte) > FENETRE_SERVIE:
            compte["chunk_hors_fenetre"] += 1
            continue
        tableau = analyser_tableau(texte)
        if not tableau.utilisable:
            compte[tableau.motif] += 1
            continue
        document_id = chunk.get("document_id")
        # Éprouvé après l'analyse : il faut la légende pour la chercher ailleurs.
        if _legende_partagee(index, par_document[document_id], chunk_id, tableau.legende):
            compte["legende_partagee"] += 1
            continue
        if retenus_par_document[document_id] >= quota_document:
            compte["quota_document"] += 1
            continue
        eligibles = cellules_eligibles(tableau, compte)
        if not eligibles:
            compte["aucune_cellule_eligible"] += 1
            continue

        cellule = eligibles[0]
        autres = [c.texte for c in tableau.toutes_cellules()
                  if valeur_de(c.texte) is not None
                  and c.debut != cellule["position"]]
        sortie.append({
            "chunk_id": chunk_id,
            "document_id": document_id,
            "section": chunk.get("section"),
            "page_start": chunk.get("page_start"),
            "legende": tableau.legende,
            "libelle_ligne": cellule["libelle_ligne"],
            "entete_colonne": cellule["entete_colonne"],
            "valeur_brute": cellule["valeur_brute"],
            "valeur_canonique": cellule["valeur_canonique"],
            "position": cellule["position"],
            "ligne": cellule["ligne"],
            "colonne": cellule["colonne"],
            "longueur_chunk": len(texte),
            "autres_valeurs": autres,
            # Signalé, pas rejeté : 35 candidats sur 220 nomment leur ligne ou leur colonne
            # par une formule que MinerU a éclatée (« $N _ { T }$ »). La cellule reste
            # parfaitement identifiée, mais la question serait illisible pour un relecteur
            # humain. Le choix appartient à qui rédige les questions, pas à la sélection.
            "nom_latex": "$" in cellule["entete_colonne"] or "$" in cellule["libelle_ligne"],
        })
        retenus_par_document[document_id] += 1
        compte["retenu"] += 1
        if limite is not None and len(sortie) >= limite:
            break
    return sortie


def main() -> None:  # pragma: no cover - outil de mise au point
    from corpus import ChunkIndex

    analyseur = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    analyseur.add_argument("--tete", type=int, default=3, help="candidats à afficher")
    arguments = analyseur.parse_args()

    motifs: Counter = Counter()
    trouves = candidats(ChunkIndex.load(verbose=False), motifs=motifs)
    print(f"{len(trouves)} cellules d'or retenues (aucun appel LLM)")
    for motif, nombre in motifs.most_common():
        print(f"  {motif:<26} {nombre}")
    for candidat in trouves[:arguments.tete]:
        print(f"\n{candidat['chunk_id']}  {candidat['legende'][:70]}")
        print(f"  ligne « {candidat['libelle_ligne']} »  colonne « {candidat['entete_colonne']} »"
              f"  ->  {candidat['valeur_brute']}")


if __name__ == "__main__":
    main()
