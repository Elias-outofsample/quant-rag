"""Sélection programmatique des formules candidates de la famille `formula`.

Ce que cette sélection refuse, et pourquoi chaque refus est une mesure sauvée
-----------------------------------------------------------------------------
**Une formule dont l'or tombe hors de la fenêtre servie.** La production tronque chaque
passage à 2 500 caractères (``pipeline.CARACTERES_SERVIS``, aligné le 8 septembre 2026 sur
``mcp_server.py``). Une équation qui commence au caractère 2 700 n'est **jamais montrée** au
générateur : la question mesurerait la troncature, pas la fidélité, et son score basculerait
au seul changement de fenêtre du chantier 1 — confondant exactement le « avant » et le
« après » que ce chantier existe pour produire. Mesuré sur le corpus : **1 110 des 12 210
occurrences de ``\\tag{`` (9,1 %) sont au-delà du caractère 2 500 de leur chunk**, et
1 998 des 5 638 chunks à ``\\tag`` (35,4 %) dépassent cette longueur. On ne retient que les
blocs dont la **fin** tient dans la fenêtre — 11 080 sur 12 210.

**Un environnement multi-ligne.** ``\\begin{array}``, ``aligned``, ``cases`` portent des
``&`` et des ``\\\\`` qui sont des conventions d'alignement. Aucun générateur n'a de raison
de les reproduire, et un or qui en dépend mesurerait la typographie. 2 276 blocs sur 11 080
en portent : ils sortent.

**Une formule qui n'est pas unique dans le corpus.** Si la même équation normalisée figure
dans deux chunks, la question a deux ors et l'inclusion peut être satisfaite par un passage
que le système n'a pas trouvé. L'unicité se vérifie sur la forme canonique, pas sur la
chaîne : c'est la seule façon de voir que deux graphies MinerU différentes sont la même
équation.

**Une formule trop courte ou trop longue.** Trop courte (``x = 1``), l'inclusion est
satisfaite par hasard. Trop longue, elle ne peut pas tenir dans les 130 mots que le contrat
de sortie accorde au générateur, et la question mesurerait le plafond de mots.

**Une formule sans phrase définitoire devant elle.** L'auteur de la question doit pouvoir
demander *ce que la formule dit* sans la recopier. S'il n'y a que des équations autour, il
n'y a rien à décrire, et la question ne pourra être qu'une paraphrase de la notation.

Ce module ne fait aucun appel réseau : il lit le corpus et rend des candidats.
"""
from __future__ import annotations

import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import latex_norme  # noqa: E402
from corpus import looks_like_toc  # noqa: E402

#: Fenêtre servie par la production. Importée en dur plutôt que de ``pipeline`` pour que la
#: sélection reste lisible hors ligne ; ``tests/test_v4_formule_selection.py`` vérifie l'égalité.
FENETRE_SERVIE = 2500

_BLOC = re.compile(r"\$\$(.+?)\$\$", re.DOTALL)
_TAG = re.compile(r"\\tag\s*\{([^{}]*)\}")

#: Bornes sur le nombre d'atomes canoniques de l'or.
#: 14 atomes, c'est environ ``X _ { t } = \alpha + \beta Y _ { t }`` : en dessous, une
#: inclusion fortuite devient plausible. 110 atomes, c'est ce qu'un générateur peut encore
#: restituer dans les 130 mots du contrat de sortie ; au-delà on mesurerait le plafond de mots.
ATOMES_MIN, ATOMES_MAX = 14, 110

#: Indices qu'une phrase *dit* quelque chose de la formule plutôt que de l'annoncer.
_INDICES = (
    "where", "denote", "denotes", "denoting", "defined", "define", "defines", "definition",
    "given by", "is the", "are the", "we let", "let ", "satisfies", "satisfy", "follows",
    "evolves", "solves", "solution", "assume", "assumption", "condition", "such that",
    "we obtain", "yields", "implies", "expressed", "written as", "reads", "with", "denoted",
)
#: Longueur de texte examinée devant le bloc pour y chercher une phrase définitoire.
_AVANT = 900
#: Une phrase, c'est au moins ce nombre de mots alphabétiques — sinon c'est de la notation.
_MOTS_MIN = 10

_MOT = re.compile(r"[A-Za-z]{2,}")


def blocs_du_chunk(texte: str, fenetre: int = FENETRE_SERVIE) -> list[dict]:
    """Les blocs ``$$ … \\tag{n} … $$`` d'un chunk, avec leur position et leur numéro."""
    trouves = []
    for correspondance in _BLOC.finditer(texte or ""):
        corps = correspondance.group(1)
        etiquette = _TAG.search(corps)
        if not etiquette:
            continue
        trouves.append({
            "corps": corps.strip(),
            "numero": etiquette.group(1).strip(),
            "debut": correspondance.start(),
            "fin": correspondance.end(),
            "dans_fenetre": correspondance.end() <= fenetre,
        })
    return trouves


def _phrase_definitoire(texte: str, debut: int) -> str | None:
    """La dernière phrase de prose devant le bloc, si elle en est une."""
    amont = texte[max(0, debut - _AVANT):debut]
    amont = _BLOC.sub(" ", amont)          # les équations voisines ne sont pas de la prose
    amont = re.sub(r"\$[^$]*\$", " ", amont)
    for morceau in reversed(re.split(r"(?<=[.:;])\s+|\n\n+", amont)):
        morceau = " ".join(morceau.split())
        # Un fragment qui porte encore de la notation n'est pas une phrase : c'est la queue
        # d'une équation qu'un ``$$`` non refermé — MinerU en produit — a laissée passer.
        # Sans ce refus, « repsilon \to 0 } \mathbb { E } [ … » comptait dix « mots ».
        if any(marque in morceau for marque in ("\\", "{", "}", "^", "$")):
            continue
        if len(_MOT.findall(morceau)) < _MOTS_MIN:
            continue
        minuscule = morceau.casefold()
        if any(indice in minuscule for indice in _INDICES):
            return morceau[-400:]
    return None


def _forme_canonique(corps: str) -> tuple[str, list[str]]:
    atomes = latex_norme.atomes(corps, latex_norme.NIVEAU_SOUPLE)
    return " ".join(atomes), atomes


def recenser(index, fenetre: int = FENETRE_SERVIE) -> tuple[dict, Counter]:
    """Tous les blocs du corpus, indexés par forme canonique — la base de l'unicité.

    Le recensement porte sur **tout** le corpus, y compris les blocs hors fenêtre et les
    environnements multi-lignes : une formule est unique ou ne l'est pas indépendamment de
    ce que la sélection accepte. Restreindre le recensement aux candidats déclarerait
    « unique » une équation qui figure ailleurs sous une forme écartée.
    """
    par_forme: dict[str, set] = defaultdict(set)
    motifs = Counter()
    for chunk_id, ligne in index.chunks.items():
        for bloc in blocs_du_chunk(ligne["text"], fenetre):
            motifs["blocs_tag"] += 1
            forme, _ = _forme_canonique(bloc["corps"])
            if forme:
                par_forme[forme].add(chunk_id)
    return par_forme, motifs


def candidats(index, fenetre: int = FENETRE_SERVIE, par_document: int = 2,
              par_chunk: int = 1) -> tuple[list[dict], Counter]:
    """Formules candidates, et le compte des rejets par motif.

    Le compte des rejets n'est pas décoratif : la règle d'arrêt du pré-enregistrement se lit
    dessus, et une famille dont on ne sait pas ce qu'elle a écarté n'est pas reproductible.
    """
    par_forme, motifs = recenser(index, fenetre)
    retenus: list[dict] = []
    pris_par_document: Counter = Counter()

    for chunk_id in sorted(index.chunks):
        ligne = index.chunks[chunk_id]
        texte, document = ligne["text"], ligne["document_id"]
        blocs = blocs_du_chunk(texte, fenetre)
        if not blocs:
            continue
        if looks_like_toc(texte):
            motifs["rejet_table_des_matieres"] += len(blocs)
            continue
        pris_ici = 0
        for bloc in blocs:
            # Les filtres de QUALITÉ passent avant les quotas. L'ordre inverse — celui qui
            # venait naturellement — faisait compter 9 981 blocs en « quota du document »
            # alors que la plupart auraient été refusés pour une raison intrinsèque : le
            # tableau des motifs disait alors combien de blocs le quota avait touchés, pas
            # ce que la sélection écarte. La règle d'arrêt du pré-enregistrement se lit sur
            # ce tableau ; il doit décrire la famille, pas l'ordre des ``if``.
            if not bloc["dans_fenetre"]:
                motifs["rejet_hors_fenetre_servie"] += 1
                continue
            environnement = latex_norme.environnement_multiligne(bloc["corps"])
            if environnement:
                motifs["rejet_environnement_multiligne"] += 1
                continue
            forme, atomes = _forme_canonique(bloc["corps"])
            if not forme:
                motifs["rejet_forme_vide"] += 1
                continue
            if len(atomes) < ATOMES_MIN:
                motifs["rejet_trop_courte"] += 1
                continue
            if len(atomes) > ATOMES_MAX:
                motifs["rejet_trop_longue"] += 1
                continue
            if "=" not in atomes and "\\sim" not in atomes and "\\propto" not in atomes:
                motifs["rejet_sans_relation"] += 1
                continue
            porteurs = par_forme.get(forme, set())
            if len(porteurs) != 1:
                motifs["rejet_forme_non_unique"] += 1
                continue
            phrase = _phrase_definitoire(texte, bloc["debut"])
            if not phrase:
                motifs["rejet_sans_phrase_definitoire"] += 1
                continue
            motifs["eligible"] += 1
            if pris_ici >= par_chunk:
                motifs["quota_chunk"] += 1
                continue
            if pris_par_document[document] >= par_document:
                motifs["quota_document"] += 1
                continue
            motifs["retenu"] += 1
            pris_ici += 1
            pris_par_document[document] += 1
            retenus.append({
                "chunk_id": chunk_id,
                "document_id": document,
                "section": ligne.get("section") or "",
                "page_start": ligne.get("page_start"),
                "formule_brute": bloc["corps"],
                "formule_canonique": forme,
                "atomes": len(atomes),
                "numero_equation": bloc["numero"],
                "phrase_definitoire": phrase,
                "position_bloc": [bloc["debut"], bloc["fin"]],
                "longueur_chunk": len(texte),
                "or_hors_fenetre": False,
            })
    return retenus, motifs


def leurres(index, candidat: dict, fenetre: int = FENETRE_SERVIE, combien: int = 3) -> list[str]:
    """Formules d'AUTRES chunks du même document — le contrôle négatif du scoreur.

    Si une réponse « juste » contient aussi un leurre, l'inclusion est satisfaite trop
    facilement et le scoreur est à revoir. Le contrôle ne coûte rien et se lit à chaque run.
    """
    sortie = []
    for chunk_id in index.chunks_of(candidat["document_id"]):
        if chunk_id == candidat["chunk_id"]:
            continue
        for bloc in blocs_du_chunk(index.chunks[chunk_id]["text"], fenetre):
            forme, atomes = _forme_canonique(bloc["corps"])
            if forme and forme != candidat["formule_canonique"] and len(atomes) >= ATOMES_MIN:
                sortie.append(bloc["corps"])
                if len(sortie) >= combien:
                    return sortie
    return sortie
