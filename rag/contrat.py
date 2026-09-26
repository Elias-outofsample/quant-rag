"""Le contrat de sortie du serveur : ce qui est servi, comment il est coupé, ce qui le trace.

Ce module n'a qu'une raison d'être : **il n'y a qu'un seul endroit où la fenêtre servie est
écrite**. Avant lui il y en avait cinq, et elles ne s'accordaient pas — relevé le 8 septembre
2026 sur la pointe de ``amelioration-retrieval`` :

===========================  =========================  ==========  ================================
outil MCP                    site                       fenêtre     ce qu'il sert
===========================  =========================  ==========  ================================
``search_documents``         ``mcp_server.py`` l. 223   2 500       un passage de recherche
``search_graph``             ``graph_search.py`` l. 366 2 500       le **même** passage
``connect_entities``         ``graph_search.py`` l. 406 1 200       le **même** passage, en deux fois moins
``timeline``                 ``mcp_server.py`` l. 286   300         un aperçu d'une ligne
``get_passage``              ``quant_rag.py`` l. 576    6 000       le passage et ses voisins, joints
===========================  =========================  ==========  ================================

Les quatre premières étaient connues (``tests/test_characters.py``). **La cinquième ne
l'était pas** : elle ne s'écrit pas ``row['text'][:6000]`` mais ``[:max_characters]`` dans
``joindre_passages``, et le balayage par expression régulière qui garde les autres ne la voit
pas. Elle est nommée ici pour la même raison que les quatre autres.

Ce que le contrat garantit
--------------------------
Sur les 26 120 passages du corpus ``5530cba145``, **aucun passage servi n'est coupé à
l'intérieur d'une formule, d'une ligne de tableau ou d'un mot.** C'est le seul critère de ce
module, et il ne se confond pas avec la couverture des réponses : le fil ``characters`` a
mesuré le 8 septembre 2026 qu'élargir la fenêtre **ne fait pas gagner de couverture** au-delà
de ce que la production sert déjà (2 400 → 4 500 : +0,031, IC95 [−0,061 ; +0,123], non
significatif, et 3 questions tombent de 2 à 0). Ce module ne revendique aucun gain de
couverture. Il répare un autre défaut, que ce fil n'a pas mesuré : **ce qui arrive entier**.

Ce que coûtait la coupe brute, mesuré sur les 26 120 passages servis
--------------------------------------------------------------------
À 2 500 caractères, 6 729 passages (25,8 %) sont tronqués, et **4 883 d'entre eux — 72,6 % —
subissent au moins une coupe interdite** :

- 1 179 coupés à l'intérieur d'un bloc ``$$…$$`` ouvert,
- 277 à l'intérieur d'une formule ``$…$``,
- 29 au milieu d'une ligne de tableau Markdown,
- 3 792 en plein mot.

Ce que coûte le contrat, mesuré sur la même population
-------------------------------------------------------
Le passage servi moyen passe de **1 550,2 à 1 734,9 caractères, soit +184,7 c. (+11,9 %)**.
À ``limit=5``, cela fait **+924 caractères par requête, environ +237 jetons d'entrée**
(compté à 3,9 caractères par jeton). C'est le prix de la fidélité, et c'est le seul argument
qui soutient ce changement.
"""
from __future__ import annotations

import hashlib
import json
import re
import unicodedata
import uuid

#: Version du **contrat de sortie**, distincte de celle du serveur MCP (``mcp_server.mcp``).
#: Elle change quand ce que l'appelant reçoit change : fenêtre, forme des lignes servies,
#: champs d'ancrage. Elle est journalisée et servie, pour qu'une trace ancienne reste lisible.
VERSION = "1.0.0"

#: Fenêtre d'**un passage de recherche** — ``search_documents``, ``search_graph``,
#: ``connect_entities``, et la fenêtre du banc.
#:
#: **Pourquoi 10 000 et pas 6 000.** C'est un plafond de *politique* — « aucun passage ne doit
#: occuper plus de 10 000 caractères du contexte de l'appelant » — et non la mesure du corpus.
#: Il se trouve qu'il ne coupe rien aujourd'hui : le plus long passage servi fait 9 384 c.
#: (``chunk-004c25a877336fd7``, un **tableau** rendu par l'overlay Markdown). Le choix entre
#: 6 000 et 10 000 a été tranché par la mesure, pas par le goût : le passage servi moyen vaut
#: 1 734,7 c. à 6 000 et 1 734,9 c. sans plafond du tout. **L'écart est de deux dixièmes de
#: caractère par passage.** Payer 0,2 c. pour ne jamais couper le seul tableau de 9 384 c. du
#: corpus — précisément le matériau que ce module existe pour protéger — est le bon échange.
#:
#: Le plafond reste un vrai garde-fou pour un corpus futur (cible : 3 000 documents) : au-delà,
#: ``couper_passage`` coupe, proprement, et le dit.
PASSAGE_CHARACTERS = 10_000

#: Fenêtre de ``get_passage``, qui joint le chunk demandé **et un voisin de chaque côté**.
#: Mesuré sur les 26 120 fenêtres du corpus : médiane 4 520 c., p95 8 850, **max 13 982**.
#: Le défaut historique de 6 000 en tronquait **30,2 %** — un outil dont le nom promet le
#: passage entier rendait moins que le passage entier une fois sur trois.
PASSAGE_ENTIER_CHARACTERS = 15_000

#: Fenêtre de ``timeline``. Ce n'est **pas** un passage servi : c'est un aperçu d'une ligne par
#: document, sur un outil qui en aligne vingt. Il garde donc sa fenêtre courte, et il est le
#: seul du contrat auquel la garantie « aucune formule coupée » ne s'applique pas — sur un
#: corpus dont 59,3 % des passages portent du LaTeX, un aperçu de 300 c. tomberait presque
#: toujours dans une formule, et reculer jusqu'à un point sûr ne rendrait plus rien.
#: ``couper_apercu`` recule quand même jusqu'à une frontière de mot et marque la coupe.
APERCU_CHARACTERS = 300

#: Tolérance de recul pour préférer une frontière de phrase ou de paragraphe à une frontière de
#: mot. Au-delà, on garde le texte : perdre 400 caractères pour finir sur un point est un
#: mauvais échange, et le plafond ne mord de toute façon sur aucun passage du corpus actuel.
#:
#: **Plafonnée à un dixième de la fenêtre**, et c'est nécessaire : une tolérance absolue de 400
#: sur un aperçu de 300 caractères reculerait jusqu'au début du texte pour attraper un saut de
#: paragraphe, et rendrait presque rien. 4 % de recul pour finir sur un point est un bon
#: échange ; 33 % n'en est pas un.
TOLERANCE_FRONTIERE = 400

_BLOC = re.compile(r"\$\$")
_FIN_PHRASE = re.compile(r"[.!?][\"'»)\]]*\s")
_MOT = re.compile(r"\w", re.UNICODE)


# --------------------------------------------------------------------------- coupe sûre

def _dollars_simples(texte: str, fin: int | None = None) -> int:
    """Nombre de ``$`` isolés — ni ``$$``, ni ``\\$`` échappé — avant ``fin``."""
    limite = len(texte) if fin is None else min(fin, len(texte))
    total, i = 0, 0
    while i < limite:
        caractere = texte[i]
        if caractere == "\\":
            i += 2
            continue
        if caractere == "$":
            if i + 1 < len(texte) and texte[i + 1] == "$":
                i += 2
                continue
            total += 1
        i += 1
    return total


def math_desequilibree(texte: str) -> bool:
    """Le texte lui-même porte-t-il un balisage mathématique impair ?

    Mesuré sur les 26 120 passages servis : **0** ont un ``$$`` impair, **127 (0,49 %)** ont un
    ``$`` simple impair — et l'inspection montre que ce sont des symboles monétaires dans des
    tableaux (« Average Realized Profit ($) »), pas des formules ouvertes.

    La conséquence est une règle : sur un texte déjà impair, le critère « ne pas couper dans une
    formule ``$…$`` » n'a plus de sens et il est **désactivé** — on ne protège pas un balisage
    qui n'est pas bien formé à la source. Le drapeau ``math_unbalanced`` le dit à l'appelant.
    """
    return (len(_BLOC.findall(texte)) % 2 == 1) or (_dollars_simples(texte) % 2 == 1)


def _dans_un_mot(texte: str, position: int) -> bool:
    if position <= 0 or position >= len(texte):
        return False
    precedent = texte[position - 1]
    return bool(_MOT.match(precedent) or precedent == "\\") and bool(_MOT.match(texte[position]))


def _dans_une_ligne_de_tableau(texte: str, position: int) -> bool:
    """Vrai si couper ici laisserait une ligne de tableau Markdown à moitié rendue."""
    if position >= len(texte):
        return False
    debut = texte.rfind("\n", 0, position) + 1
    fin = texte.find("\n", position)
    fin = len(texte) if fin == -1 else fin
    return texte[debut:fin].lstrip().startswith("|") and position < fin


def coupe_sure(texte: str, position: int, tolerer_inline: bool = False) -> bool:
    """Couper ``texte`` à ``position`` laisse-t-il un matériau intact ?

    Quatre conditions, toutes nécessaires : hors d'un bloc ``$$…$$`` ouvert, hors d'une formule
    ``$…$`` ouverte, hors d'une ligne de tableau, et hors d'un mot.
    """
    if position <= 0 or position >= len(texte):
        return position >= len(texte)
    if _dans_un_mot(texte, position):
        return False
    if _dans_une_ligne_de_tableau(texte, position):
        return False
    if len(_BLOC.findall(texte[:position])) % 2 == 1:
        return False
    if not tolerer_inline and _dollars_simples(texte, position) % 2 == 1:
        return False
    return True


def _parites(texte: str, jusqu_a: int) -> tuple[list[int], list[int]]:
    """Parité cumulée de ``$$`` et de ``$`` simple pour toute position ``<= jusqu_a``.

    Calculée en une passe. Sans elle, ``point_de_coupe`` serait quadratique : il testerait
    jusqu'à 10 000 positions en rescannant le préfixe à chaque fois, soit 10⁸ opérations sur un
    seul passage. Mesuré avant correction : plusieurs secondes par passage.
    """
    bloc = [0] * (jusqu_a + 1)
    inline = [0] * (jusqu_a + 1)
    b = s = i = 0
    while i < jusqu_a:
        caractere = texte[i]
        if caractere == "\\":
            for _ in range(min(2, jusqu_a - i)):
                bloc[i + 1] = b
                inline[i + 1] = s
                i += 1
            continue
        if caractere == "$" and i + 1 < len(texte) and texte[i + 1] == "$":
            b += 1
            for _ in range(min(2, jusqu_a - i)):
                bloc[i + 1] = b
                inline[i + 1] = s
                i += 1
            continue
        if caractere == "$":
            s += 1
        bloc[i + 1] = b
        inline[i + 1] = s
        i += 1
    return bloc, inline


def point_de_coupe(texte: str, plafond: int) -> int:
    """La plus grande position ``p <= plafond`` où couper ne casse rien.

    On prend la plus grande frontière **sûre**, puis on remonte à une fin de phrase ou de
    paragraphe si elle se trouve à moins de ``TOLERANCE_FRONTIERE`` en arrière — finir sur un
    point vaut mieux que finir au milieu d'une phrase, mais pas au prix de 400 caractères.

    Quand aucune position n'est sûre — un texte dont le balisage est impair dès le début —, on
    recule jusqu'à la dernière frontière de mot et ``drapeaux_qualite`` le signale. Rendre zéro
    caractère serait pire que rendre un texte marqué comme imparfait.
    """
    if plafond is None or len(texte) <= plafond:
        return len(texte)
    tolerer = math_desequilibree(texte)
    bloc, inline = _parites(texte, plafond)

    def sure(position: int) -> bool:
        if position <= 0:
            return False
        if bloc[position] % 2 == 1:
            return False
        if not tolerer and inline[position] % 2 == 1:
            return False
        if _dans_un_mot(texte, position):
            return False
        return not _dans_une_ligne_de_tableau(texte, position)

    meilleur = 0
    for position in range(plafond, 0, -1):
        if sure(position):
            meilleur = position
            break
    if not meilleur:
        # Dernier recours : la dernière frontière de mot, même imparfaite.
        for position in range(plafond, 0, -1):
            if not _dans_un_mot(texte, position):
                return position
        return plafond
    plancher = max(meilleur - min(TOLERANCE_FRONTIERE, plafond // 10), 0)
    fenetre = texte[plancher:meilleur]
    paragraphe = fenetre.rfind("\n\n")
    if paragraphe != -1:
        candidat = plancher + paragraphe
        if sure(candidat):
            return candidat
    for correspondance in reversed(list(_FIN_PHRASE.finditer(fenetre))):
        candidat = plancher + correspondance.end()
        if sure(candidat):
            return candidat
    return meilleur


def couper(texte: str, plafond: int) -> tuple[str, int]:
    """``(texte coupé, caractères restants)``. Sans marqueur : c'est la brique testable."""
    texte = texte or ""
    if plafond is None or len(texte) <= plafond:
        return texte, 0
    position = point_de_coupe(texte, plafond)
    return texte[:position].rstrip(), len(texte) - position


def marqueur(restants: int, chunk_id: str | None = None) -> str:
    suite = f", get_passage(\"{chunk_id}\") pour la suite" if chunk_id else ""
    return f"\n\n[… tronqué : {restants} caractères restants{suite}]"


def couper_passage(texte: str, plafond: int | None = None, chunk_id: str | None = None) -> str:
    """Le passage servi, coupé au plafond **sans rien casser**, et marqué s'il l'a été.

    C'est le seul point de rendu d'un passage de recherche. ``mcp_server.search_documents``,
    ``graph_search.format_search``, ``graph_search.format_connect`` et
    ``benchmark.pipeline.format_passages`` passent tous par ici.
    """
    plafond = PASSAGE_CHARACTERS if plafond is None else plafond
    coupe, restants = couper(texte, plafond)
    return coupe + marqueur(restants, chunk_id) if restants else coupe


def couper_apercu(texte: str, plafond: int | None = None) -> str:
    """L'aperçu d'une ligne de ``timeline``. Une frontière de mot, et le dire.

    La garantie « aucune formule coupée » **ne s'applique pas ici** et c'est délibéré : voir
    ``APERCU_CHARACTERS``. Ce qui est garanti, c'est qu'un aperçu ne se fait jamais passer pour
    un passage entier — il porte toujours son ``…``.
    """
    plafond = APERCU_CHARACTERS if plafond is None else plafond
    texte = " ".join((texte or "").split())
    if len(texte) <= plafond:
        return texte
    position = plafond
    while position > 0 and _dans_un_mot(texte, position):
        position -= 1
    return texte[:position].rstrip() + "…"


# --------------------------------------------------------------------------- qualité servie

def drapeaux_qualite(texte: str, tronque: bool = False) -> dict:
    """Ce que le passage servi porte de douteux, de façon **déterministe** — aucun jugement.

    Les défauts sont amont (parse, overlays) et ne se réparent pas ici : les réparer changerait
    le texte des chunks, donc la signature du corpus, qui est gelée. Ce qui est en notre pouvoir
    est de ne pas les taire.
    """
    texte = texte or ""
    return {
        "has_math": "$" in texte,
        "math_unbalanced": math_desequilibree(texte),
        "has_control_chars": any(unicodedata.category(c) == "Cc" and c not in "\n\t\r" for c in texte),
        "has_html_tags": "<sup>" in texte or "<sub>" in texte,
        "has_table": any(l.lstrip().startswith("|") for l in texte.splitlines()),
        "truncated": bool(tronque),
    }


def ligne_ancre(row: dict) -> str:
    """« ancre : … » — où le passage vit dans le texte canonique, et l'empreinte de ce texte.

    Rendue vide quand la collection ne porte pas d'ancrage : un serveur qui tourne sur une
    collection bâtie avant le contrat de sortie se tait sur ce qu'il ne sait pas, plutôt que
    d'afficher une ligne creuse. C'est la même règle que ``_comptes`` applique aux comptes.

    La **granularité** est dite, et elle compte : ``exacte`` (86,9 % des passages servis)
    signifie que les offsets désignent exactement le texte du passage ; ``bloc`` (13,1 %)
    signifie qu'ils désignent le ou les blocs **dont il est issu** — un tableau réassemblé, des
    espaces normalisés — et qu'une recherche littérale dans cette plage peut échouer.
    """
    intervalles = row.get("ancrage_intervalles")
    sha = row.get("doc_text_sha256")
    if not intervalles or not sha:
        return ""
    plages = ", ".join(f"{debut}–{fin}" for debut, fin in intervalles[:4])
    if len(intervalles) > 4:
        plages += f", … ({len(intervalles)} plages)"
    return (f"{row.get('document_id')} · caractères {plages} "
            f"({row.get('ancrage_granularite')}) · sha256:{sha[:12]}")


def ligne_qualite(drapeaux: dict) -> str:
    """La ligne « qualité : » servie. Vide quand il n'y a rien à signaler — ne pas bavarder."""
    dits = [nom for nom in ("math_unbalanced", "has_control_chars", "has_html_tags", "truncated")
            if drapeaux.get(nom)]
    return ", ".join(dits)


# --------------------------------------------------------------------------- traçabilité

def request_id() -> str:
    """Un identifiant par requête servie. Sans lui, une ligne de journal et une réponse ne se
    recollent pas — et c'est exactement ce qu'il a fallu faire à la main tout le 8 septembre."""
    return uuid.uuid4().hex[:16]


def appelant(explicite: str | None = None) -> str:
    """Qui a demandé. Déduit du point d'entrée quand personne ne le dit.

    Le déduire plutôt que l'exiger est délibéré : sans cela, il aurait fallu ajouter un
    argument à des appels qui appartiennent à d'autres chantiers (``benchmark/pipeline.py``
    est au chantier « instrument »), et un champ qu'on oublie de passer vaut moins qu'un champ
    deviné correctement. Un point d'entrée inattendu rend son propre nom — jamais « cli » par
    défaut, qui serait une affirmation fausse.
    """
    if explicite:
        return explicite
    if APPELANT_DECLARE:
        return APPELANT_DECLARE
    import sys
    from pathlib import Path

    brut = sys.argv[0] if sys.argv else ""
    if not brut:
        return "inconnu"
    nom = Path(brut).name
    chemin = str(Path(brut).absolute()).replace("\\", "/")
    if nom == "mcp_server.py":
        return "mcp"
    if "/benchmark/" in chemin:
        return "bench"
    if nom in ("quant_rag.py", "graph_search.py"):
        return "cli"
    return nom


def configuration_servie() -> dict:
    """La configuration **effectivement** servie, lue sur le code, jamais recopiée.

    Les valeurs sont lues par réflexion (``inspect``) sur ``quant_rag`` plutôt que redéclarées :
    un hash qui suit une copie ne dit rien. ``config/embedding.yaml`` et
    ``config/retrieval.yaml`` ne sont **pas** lus — aucun fichier de ``rag/`` ne les lit, et
    ils contredisent la production (``device: cuda``, ``max_length: 2048``).
    """
    import inspect

    import quant_rag

    selection = inspect.signature(quant_rag.search_explained).parameters
    doublon = inspect.signature(quant_rag._near_duplicate).parameters
    bardeaux = inspect.signature(quant_rag._shingles).parameters
    return {
        "contrat_version": VERSION,
        "passage_characters": PASSAGE_CHARACTERS,
        "passage_entier_characters": PASSAGE_ENTIER_CHARACTERS,
        "apercu_characters": APERCU_CHARACTERS,
        "collection": quant_rag.COLLECTION,
        "model_id": quant_rag.MODEL_ID,
        "max_length": quant_rag.MAX_LENGTH,
        "query_instruction": quant_rag.QUERY_INSTRUCTION,
        "pool": quant_rag.POOL,
        "min_characters": quant_rag.MIN_CHARACTERS,
        "limit": selection["limit"].default,
        "per_document": selection["per_document"].default,
        "dedupe": selection["dedupe"].default,
        "dedupe_threshold": doublon["threshold"].default,
        "dedupe_shingle": bardeaux["size"].default,
        "exact_tokens_for_hybrid": quant_rag.EXACT_TOKENS_FOR_HYBRID,
        "reranker_id": quant_rag.RERANKER_ID,
        "reclassement_id": quant_rag.RECLASSEMENT_ID,
        "decalage_page_utilisateur": quant_rag.DECALAGE_PAGE_UTILISATEUR,
    }


_HASH_MEMOIRE: dict[str, str] = {}


def config_hash(configuration: dict | None = None) -> str:
    """Douze hexadécimaux qui résument la configuration servie. Change dès qu'elle change.

    Mémorisé pour l'appel sans argument : la configuration servie est faite de constantes de
    module et de valeurs par défaut de signatures — elle ne peut pas bouger dans un processus
    vivant. La **signature du corpus**, elle, le peut, et n'est délibérément pas mémorisée
    (``entete_tracabilite`` la relit à chaque requête).
    """
    if configuration is None:
        if "defaut" not in _HASH_MEMOIRE:
            _HASH_MEMOIRE["defaut"] = config_hash(configuration_servie())
        return _HASH_MEMOIRE["defaut"]
    empreinte = json.dumps(configuration, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(empreinte.encode("utf-8")).hexdigest()[:12]


#: Version du serveur MCP, déclarée par lui à l'import (``contrat.enregistrer_serveur``). Reste
#: ``None`` quand l'appel ne passe pas par le serveur — un banc, la CLI —, et le champ servi le
#: dit alors franchement plutôt que d'inventer une version qui n'a rien servi.
SERVEUR_VERSION: str | None = None

#: L'appelant déclaré par le point d'entrée, quand il se nomme lui-même.
APPELANT_DECLARE: str | None = None


def enregistrer_serveur(version: str, caller: str = "mcp") -> None:
    """Le point d'entrée déclare sa version et son nom au contrat. Appelé une fois, à l'import.

    C'est une déclaration et non un paramètre de ``search_explained`` — délibérément. Ajouter
    un argument à cette signature aurait cassé la garde
    ``test_reclassement_selectif.test_le_parametre_est_en_fin_de_signature``, qui protège des
    appelants passant leurs arguments **positionnellement**, et un chantier de traçabilité n'a
    aucune raison de fragiliser le chemin servi pour se nommer.
    """
    global SERVEUR_VERSION, APPELANT_DECLARE
    SERVEUR_VERSION = version
    APPELANT_DECLARE = caller


def entete_tracabilite(caller: str | None = None, identifiant: str | None = None,
                       server_version: str | None = None) -> dict:
    """Les champs que toute réponse servie et toute ligne de journal doivent porter.

    Sans ``request_id``, une ligne de ``router-decisions.jsonl`` et la réponse qu'elle a servie
    ne se recollent pas — il a fallu le faire à la main pendant tout le 8 septembre 2026.
    Sans ``config_hash``, deux mesures prises à deux jours d'intervalle ne peuvent pas prouver
    qu'elles portent sur la même configuration.
    """
    import corpus_overlay

    return {
        "request_id": identifiant or request_id(),
        "server_version": server_version or SERVEUR_VERSION,
        "contrat_version": VERSION,
        "corpus_signature": corpus_overlay.signature(),
        "config_hash": config_hash(),
        "caller": appelant(caller),
    }
