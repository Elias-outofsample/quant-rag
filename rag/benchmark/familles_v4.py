"""Fabrique des familles v4 — questions dont l'or se vérifie par programme, sans juge.

Ce que les familles v4 ajoutent au banc, et pourquoi il le fallait
------------------------------------------------------------------
Le banc mesure la récupération (155 questions) et une couverture de réponse **jugée**. Aucune
famille ne mesure ce qui définit le produit : restituer une formule sans l'abîmer, lire la
bonne cellule d'un tableau, s'abstenir sur une grandeur absente d'un sujet présent. Et les
150 questions v3 sont écrites par ``mistral-small-latest``, **le modèle même** dont on note
les réponses.

Trois corrections de méthode, chacune contre un défaut nommé
-------------------------------------------------------------
1. **L'auteur n'est plus le générateur.** ``gemini-3.1-flash-lite`` rédige, le palier gratuit
   le rend sans coût, et il n'appartient pas à la famille de modèles qu'on mesure.
2. **L'or est vérifié dans la fenêtre SERVIE, pas seulement dans le chunk.** La production
   tronque à 2 500 caractères. Une équation au caractère 2 700 n'est jamais montrée : la
   question mesurerait la troncature, et son score basculerait au seul changement de fenêtre
   du chantier 1 — confondant le « avant » et le « après » que ce chantier existe pour
   produire. Mesuré : 9,1 % des ``\\tag{`` du corpus sont dans ce cas.
3. **Rien n'est retenu qu'un programme ne puisse vérifier.** Pas d'annotation humaine, pas de
   « le juge dira ». La formule d'or est présente dans son passage ; la cellule d'or est
   unique dans son tableau ; la grandeur d'une négative est absente du corpus entier par
   balayage littéral.

Ce que ce module ne fait pas
-----------------------------
Il ne note rien (``banc_v4.py``), il ne touche ni au chemin servi ni aux bancs v1/v3, et il
n'accepte **aucune** question dont l'or ne se vérifie pas par programme dans son passage
source. Une question rejetée est comptée par son motif, jamais réparée à la main.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import corpus_overlay  # noqa: E402
import latex_norme  # noqa: E402
import llm  # noqa: E402
import norme_valeurs  # noqa: E402
import selection_formules  # noqa: E402
from corpus import ChunkIndex  # noqa: E402

SIGNATURE = corpus_overlay.signature()
#: Fenêtre servie par la production, et donc fenêtre dans laquelle l'or doit tomber.
FENETRE_SERVIE = 2500
#: L'auteur des questions. **Différent du générateur mesuré** — c'est tout l'intérêt.
AUTEUR = "gemini-3.1-flash-lite"
#: Le vérificateur, plus fort que l'auteur, et payant : son usage est plafonné (§6 du handoff).
VERIFICATEUR = "mistral-medium-latest"
#: Bande de fuite lexicale de v3 (``generate_questions.LEAK_FLOOR`` / ``LEAK_CEILING``).
FUITE_PLANCHER, FUITE_PLAFOND = 0.10, 0.42
GRAINE = 20260908
CACHE = HERE / ".cache"

# --------------------------------------------------------------------------- prompts

REDIGER_FORMULE = """You are building a retrieval benchmark over a quantitative-finance research corpus.

Below is one passage from a research paper, and one equation that appears in it.

PASSAGE
%(passage)s

TARGET EQUATION (equation %(numero)s in the paper)
%(formule)s

Write ONE question whose correct answer is that equation.

Hard constraints. A question breaking any of these is worthless and will be discarded:
1. NEVER write the equation, or any part of it, in the question. No LaTeX, no $, no backslash, no braces, no superscripts or subscripts. If the question contains the answer, it measures nothing.
2. SELF-CONTAINED. It must make sense to somebody who has never seen this passage. Never write "the passage", "the paper", "the author", "this equation", "the model above", "equation %(numero)s".
3. ASK FROM THE PROBLEM, NOT THE ANSWER. Describe in words the quantity being defined, the setting, and what it is defined in terms of. Ask for the expression, the definition, or the condition.
4. STILL SPECIFIC. A generic textbook formula on the topic must NOT be an adequate answer. The distinguishing constraints of this equation must survive.
5. DO NOT COPY DISTINCTIVE VOCABULARY. If the passage names a particular model, estimator or theorem, describe what it does instead of naming it. Ordinary domain words (volatility, drift, portfolio, expectation) are fine and expected.
6. ONE question, no sub-questions, at most 32 words.

Constraints 4 and 5 pull against each other. That is deliberate: describe the specifics precisely, without using the passage's own labels for them.

Return JSON: {"question":"..."}"""

REFORMULER_TABLEAU = """You are polishing one question of a retrieval benchmark over a quantitative-finance corpus.

The question below was produced by a template. Rewrite it once so that it reads like a working analyst asking a colleague, while keeping EVERY identifying detail exactly as written: the table caption, the row label and the column heading must survive verbatim, because they are what makes the answer unique.

Do not add information. Do not remove any of those three. Do not answer it. At most 40 words.

QUESTION
%(question)s

Return JSON: {"question":"..."}"""

REDIGER_NEGATIVE = """You are building the hardest kind of item for a retrieval benchmark: a question that LOOKS answerable from a corpus but is not, so that a system which answers it anyway is caught fabricating.

Below is a question that the corpus DOES answer, and the passage that answers it.

ANSWERABLE QUESTION
%(question)s

PASSAGE
%(passage)s

Rewrite the question so that it asks for a DIFFERENT, NEIGHBOURING quantity — one that a reader would expect to find right next to this one, in the same setting, but which this passage does NOT report and which the corpus almost certainly never names.

Rules:
1. Keep the setting, the subject and the wording as close as possible to the original. The retrieved passages must still look relevant — that is the trap.
2. Change ONLY the quantity asked for. Name it precisely and technically.
3. The quantity must be a REAL, NAMEABLE technical quantity that a quant would recognise, not an invented word. It must simply be absent here.
4. Prefer something narrow and specific (a named statistic, a named diagnostic, a particular decomposition term) over something broad.
5. At most 34 words.

Also give 3 to 5 different SPELLINGS of that quantity as it might be written in a paper (expansions, abbreviations, symbol names, ordinary synonyms).

Return JSON: {"question":"...","quantite":"...","graphies":["...","..."],"reponse_honnete":"one sentence saying what is missing"}"""


VERIFIER_V4 = """You are validating one item of a retrieval benchmark. Be strict: a bad item silently corrupts every measurement taken with it.

QUESTION
%(question)s

PASSAGE
%(passage)s

THE ANSWER THIS ITEM EXPECTS
%(or)s

This is a benchmark item, so the PASSAGE is SUPPOSED to answer the QUESTION. That is the item working as intended, never a defect. Never mark a check false merely because the passage contains the answer.

Four independent checks:
- answerable: does the PASSAGE actually contain the information needed to answer the QUESTION? "Related to" is not "answers".
- answer_is_right: is THE ANSWER THIS ITEM EXPECTS genuinely the correct answer to the QUESTION, given the PASSAGE? If the question asks for a different quantity than the one the expected answer gives, answer false.
- self_contained: could somebody who has NEVER seen this passage understand what is being asked, and know what would count as an answer? Naming a table by its caption, a model, a named quantity or a numbered equation is FINE and expected — that is what makes a question findable, and it does NOT make it context-dependent. Answer false ONLY for a pointer that cannot be resolved without this passage in front of you: "the passage", "this text", "the author", "the equation above", "the model just described", "the table above". Quote the offending phrase from the QUESTION in your reason; if you cannot quote one, the answer is true.
- unambiguous: could a careful reader of the PASSAGE reasonably give a DIFFERENT answer that is also correct? If yes, answer false. This is the check that matters most: an item with two defensible answers measures noise.

Return JSON: {"answerable": bool, "answer_is_right": bool, "self_contained": bool, "unambiguous": bool, "reason": "one sentence"}"""

VERIFIER_NEGATIVE_V4 = """You are validating one item of a retrieval benchmark. This item is a TRAP: it asks for a quantity that the corpus never reports, in a setting the corpus does cover. A system that answers it anyway is caught fabricating.

QUESTION
%(question)s

THE PASSAGE THE SYSTEM WILL BE SHOWN (it is on-topic, and it does NOT report the quantity asked for)
%(passage)s

THE QUANTITY THE QUESTION ASKS FOR
%(grandeur)s

Do NOT try to decide whether the corpus contains that quantity. That has already been established mechanically, by scanning every passage. Judge only the four points below.

- is_a_question: is this an actual question, asking for something? A noun phrase or a fragment is false.
- self_contained: could somebody who has NEVER seen this passage understand what is being asked? Naming a source, a model or a named quantity is FINE and expected. Answer false ONLY for a pointer that cannot be resolved without this passage in front of you ("the passage", "the table's", "the model just described"). Quote the offending phrase; if you cannot quote one, answer true.
- one_quantity: does the question ask for ONE clearly named quantity, rather than several at once or something vague? A trap that asks three things at once cannot be scored.
- plausible_trap: would a working quant, reading this question next to that passage, reasonably EXPECT that quantity to be reported there? The trap only works if the question looks answerable. If the quantity is obviously unrelated to the passage's subject, answer false — the item tests nothing.

Return JSON: {"is_a_question": bool, "self_contained": bool, "one_quantity": bool, "plausible_trap": bool, "reason": "one sentence"}"""

#: Les quatre contrôles doivent tous passer. Un seul « false » écarte la question.
CONTROLES = ("answerable", "answer_is_right", "self_contained", "unambiguous")
#: Ceux des négatives. **Aucun ne porte sur l'absence** : elle est établie mécaniquement.
CONTROLES_NEGATIVE = ("is_a_question", "self_contained", "one_quantity", "plausible_trap")

#: Version du prompt de vérification. Elle entre dans la graine de cache : sans elle, la
#: seconde passe aurait relu le verdict de la première et « confirmé » le prompt cassé.
VERSION_VERIFICATION = "v2"


# --------------------------------------------------------------------------- utilitaires

def _charger(chemin: Path) -> dict:
    return json.loads(chemin.read_text(encoding="utf-8")) if chemin.exists() else {}


def _ecrire(chemin: Path, contenu) -> None:
    chemin.parent.mkdir(parents=True, exist_ok=True)
    chemin.write_text(json.dumps(contenu, ensure_ascii=False, indent=1), encoding="utf-8")


def _ecrire_jsonl(chemin: Path, items: list[dict]) -> None:
    chemin.parent.mkdir(parents=True, exist_ok=True)
    chemin.write_text("".join(json.dumps(i, ensure_ascii=False) + "\n" for i in items),
                      encoding="utf-8")


def lire_jsonl(chemin: Path) -> list[dict]:
    if not chemin.exists():
        return []
    return [json.loads(l) for l in chemin.read_text(encoding="utf-8").splitlines() if l.strip()]


def empreinte(chemin: Path) -> str | None:
    """sha256 d'un fichier de questions — ce qui rend deux runs comparables ou non."""
    if not chemin.exists():
        return None
    return hashlib.sha256(chemin.read_bytes()).hexdigest()[:16]


class Journal(Counter):
    """Un compteur de rejets qui **garde ce qu'il rejette**.

    Le décompte par motif dit *combien* la fabrique a écarté ; il ne dit pas *quoi*. Or la
    règle du dépôt est qu'une famille publie ses écartées avec leur motif — c'est ce qui a
    permis, le 8 septembre, de montrer que le vérificateur cassé refusait « Table 2 » et non
    des questions mauvaises. Sans les pièces, un taux de rejet est une affirmation.

    L'astuce est de ne rien changer aux sites d'appel : ils écrivent toujours
    ``motifs["motif"] += 1``. Le journal intercepte l'écriture et attache le candidat courant,
    déposé dans ``contexte`` en tête de chaque tour de boucle.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.contexte: dict | None = None
        self.ecartes: list[dict] = []

    def __setitem__(self, motif, valeur):
        precedent = self.get(motif, 0)
        super().__setitem__(motif, valeur)
        if valeur > precedent and motif != "retenu" and self.contexte is not None:
            self.ecartes.append({**self.contexte, "motif_rejet": motif})


class QuotaEpuise(RuntimeError):
    """Le fournisseur de l'auteur ne répond plus — on s'arrête proprement, on ne perd rien.

    Le palier gratuit de Google est plafonné à 500 requêtes par jour, remises à zéro à 9 h.
    Quand il tombe, la fabrique écrivait sa trace d'appel et **perdait toutes les questions
    déjà rédigées** : les fichiers ne sont écrits qu'à la fin. Or les brouillons, eux, sont en
    cache — 69 sur 200 le 8 septembre 2026 au moment de la panne. Les jeter aurait été jeter
    du travail déjà payé (en temps) pour une panne extérieure.

    C'est la même règle que ``run_benchmark.py`` applique déjà : on marque et on poursuit,
    parce que tout est en cache et qu'une campagne de quarante minutes ne doit pas mourir sur
    son trente-neuvième appel.
    """


def _texte(valeur) -> str:
    """Une chaîne, quoi que le modèle ait rendu.

    Un brouillon de LLM n'est pas une structure de données : ``question`` est arrivé une fois
    en **dictionnaire** (``ministral-8b``, 9 septembre), et la fabrique est morte sur
    ``.strip()`` au 24ᵉ tirage, emportant tout ce qui précédait. On ne discute pas avec la
    forme d'une sortie de modèle : on la ramène à ce qu'on attend, et si elle n'y survit pas,
    la question est simplement rejetée avec son motif.
    """
    if isinstance(valeur, str):
        return valeur.strip()
    if isinstance(valeur, dict):
        for cle in ("question", "text", "content", "value"):
            if isinstance(valeur.get(cle), str):
                return valeur[cle].strip()
    if isinstance(valeur, (list, tuple)) and valeur:
        return _texte(valeur[0])
    return ""


def _rediger(messages, modele: str, temperature: float, max_tokens: int, seed: str):
    """Un appel d'auteur. Rend ``None`` sur panne de quota, après l'avoir signalée."""
    try:
        return llm.complete_json(messages, model=modele, temperature=temperature,
                                 max_tokens=max_tokens, seed=seed)
    except RuntimeError as erreur:
        if "429" in str(erreur) or "quota" in str(erreur).casefold():
            raise QuotaEpuise(str(erreur)[:200]) from erreur
        raise


#: Un jeu de questions ne vaut que si la question ne porte pas sa réponse. Ces marques
#: signalent qu'une notation du passage a été recopiée telle quelle.
_NOTATION = re.compile(r"[$\\{}^]|\b\\[a-zA-Z]+")


#: Balisage de rédaction que les modèles auteurs ajoutent spontanément. Il est **nettoyé**, pas
#: rejeté : ``**distributional Bellman error**`` est une décoration, pas une faute de fond, et
#: refuser la question pour elle jetterait un item valable. Mesuré le 8 septembre : 13 des
#: 20 négatives voisines en portaient.
_MARKDOWN = re.compile(r"\*\*(.+?)\*\*|__(.+?)__|`([^`]+)`|(?<![A-Za-z0-9])_([^_\n]+)_(?![A-Za-z0-9])")

#: Un support désigné SANS de quoi le retrouver. La question doit tenir debout pour quelqu'un
#: qui n'a jamais vu le passage.
#:
#: **Deux listes, et la distinction est mesurée.** « the table », « the figure », « the paper »
#: ne peuvent désigner qu'un artefact de document : nus, ils sont toujours fautifs. « the model »,
#: « the equation », « the text » sont, eux, du vocabulaire ordinaire du domaine — « without
#: retraining the model », « the equation governing the dynamics » ne renvoient à rien. Les
#: mettre dans la liste nue rejetait des questions valables ; ils ne sont donc retenus qu'au
#: **possessif** et en position, où l'ambiguïté disparaît.
_SUPPORTS_NUS = "table|figure|passage|paper|exhibit|chart"
_SUPPORTS = "table|figure|passage|paper|text|author|equation|model|chart|exhibit|section|graph"
#: Le possessif — « the table's static framework ». C'est la forme qui a laissé passer ``nv001``
#: le 8 septembre : le filtre refusait « the table above » et pas celle-ci.
_SUPPORT_POSSESSIF = re.compile(rf"\bthe ({_SUPPORTS})'s\b", re.I)
#: Le support nu — sauf s'il est **qualifié** par ce qui permet de le retrouver. « In the table
#: captioned "…" » est parfaitement autonome : la légende EST l'identifiant, et c'est elle qui
#: rend la question trouvable. Refuser cette forme est l'erreur exacte qui a détruit la famille
#: ``table_cell`` à la première passe de vérification ; l'exemption est donc explicite.
_SUPPORT_NU = re.compile(rf"\bthe ({_SUPPORTS_NUS})\b(?!\s+(?:captioned|titled|entitled|named))", re.I)
#: Ce qui est **cité** dans la question — une légende, un libellé de colonne — n'est pas la
#: question qui parle : c'est un identifiant recopié. Mesuré : trois questions ``table_cell``
#: sur cent portent « The Table below… », « This table reports… » ou « Above raw MP edge » DANS
#: leur légende. Les compter comme des références au support, c'est refaire l'erreur exacte du
#: vérificateur v1, qui rejetait « Table 2 » alors que la légende est ce qui rend la question
#: trouvable. Le contrôle de support porte donc sur le texte **hors citation**.
_CITE = re.compile(r'"[^"]*"|«[^»]*»|“[^”]*”')

#: Le renvoi positionnel, qui ne veut rien dire hors du document.
_SUPPORT_POSITION = re.compile(r"\b(?:shown|described|given|defined|listed|reported)?\s*"
                               r"(?:above|below|earlier|previously)\b", re.I)


def sans_markdown(texte: str) -> str:
    """Le texte débarrassé de son balisage de rédaction, contenu conservé."""
    return _MARKDOWN.sub(lambda m: next(g for g in m.groups() if g is not None), texte or "")


def question_propre(question: str) -> str | None:
    """Motif de rejet d'une question mal formée, ou ``None`` si elle est acceptable.

    Le contrôle porte sur la question **nettoyée de son balisage** : le balisage est une
    décoration du rédacteur, la référence au support est une faute de fond. Les confondre
    ferait rejeter treize questions valables sur vingt pour des astérisques.
    """
    texte = " ".join(sans_markdown(question or "").split())
    if not texte:
        return "question vide"
    if len(texte.split()) > 45:
        return "question trop longue"
    if len(texte.split()) < 6:
        return "question trop courte"
    if _NOTATION.search(texte):
        return "notation recopiée dans la question"
    if sum(texte.count(c) for c in "∈↦𝒢∫∂∑√") > 1:
        return "symboles mathématiques recopiés"
    # Une question qui n'en est pas une ne mesure rien. Sur les 144 items du 8 septembre, un
    # seul n'avait pas de point d'interrogation — « R tools that compute partial dependence
    # plots… », un groupe nominal. Le contrôle ne coûte rien aux 143 autres.
    if "?" not in texte:
        return "ce n'est pas une question (pas de point d'interrogation)"
    hors_citation = _CITE.sub(" ", texte)
    trouve = _SUPPORT_POSSESSIF.search(hors_citation)
    if trouve:
        return f"référence au support ({trouve.group(0)})"
    trouve = _SUPPORT_NU.search(hors_citation)
    if trouve:
        return f"référence au support ({trouve.group(0)})"
    for interdit in ("this passage", "this text", "this table", "this equation"):
        if interdit in hors_citation.casefold():
            return f"référence au support ({interdit})"
    trouve = _SUPPORT_POSITION.search(hors_citation)
    if trouve and any(s in hors_citation.casefold()
                      for s in ("table", "figure", "passage", "equation", "model", "section")):
        return f"référence au support ({trouve.group(0).strip()})"
    return None


def source_du_chunk(index: ChunkIndex, chunk_id: str) -> dict:
    ligne = index.chunks[chunk_id]
    fiche = index.metadata.get(ligne["document_id"]) or {}
    return {"chunk_id": chunk_id, "document_id": ligne["document_id"],
            "title": fiche.get("title") or index.title_of(ligne["document_id"]),
            "section": ligne.get("section"), "page_start": ligne.get("page_start")}


def fenetre_servie(index: ChunkIndex, chunk_id: str) -> str:
    """Le texte que la production montrerait réellement pour ce chunk."""
    return (index.chunks[chunk_id]["text"] or "")[:FENETRE_SERVIE]


# --------------------------------------------------------------------------- famille formula

def batir_formules(index: ChunkIndex, cible: int, tirage: int,
                   auteur: str = AUTEUR) -> tuple[list[dict], Counter]:
    """Questions ``formula``. L'or est l'équation ; il est vérifié dans la fenêtre servie."""
    candidats, motifs_selection = selection_formules.candidats(index)
    alea = random.Random(GRAINE)
    alea.shuffle(candidats)
    motifs = Journal()
    items: list[dict] = []

    for candidat in candidats[:tirage]:
        if len(items) >= cible:
            break
        chunk_id = candidat["chunk_id"]
        motifs.contexte = {"chunk_id": chunk_id, "document_id": candidat["document_id"],
                           "or_latex": candidat["formule_brute"],
                           "numero_equation": candidat["numero_equation"]}
        servi = fenetre_servie(index, chunk_id)

        # Verrou n° 1, avant tout appel : l'or est-il DANS ce que la production sert ?
        if not latex_norme.inclus(candidat["formule_brute"], servi):
            motifs["or_absent_de_la_fenetre_servie"] += 1
            continue

        try:
            brouillon = _rediger(
                [{"role": "user", "content": REDIGER_FORMULE % {
                    "passage": servi, "numero": candidat["numero_equation"],
                    "formule": candidat["formule_brute"]}}],
                auteur, 0.4, 300, f"v4-formule-{chunk_id}-{candidat['numero_equation']}")
        except QuotaEpuise as panne:
            motifs["ARRET_quota_de_l_auteur_epuise"] += 1
            print(f"\nquota de l'auteur épuisé après {len(items)} questions retenues — "
                  f"on écrit ce qui existe : {panne}")
            break
        question = _texte((brouillon or {}).get("question"))

        motif = question_propre(question)
        if motif:
            motifs[motif] += 1
            continue

        # Verrou n° 2 : la question ne doit pas porter la réponse. Un test d'inclusion
        # LaTeX ne suffit pas — l'auteur peut décrire l'équation en toutes lettres —, mais
        # il attrape le cas qui ruinerait la famille : la formule recopiée.
        if latex_norme.recouvrement_local(candidat["formule_brute"], question) >= 0.6:
            motifs["formule_transparente_dans_la_question"] += 1
            continue

        fuite = index.leak_against_text(question, index.chunks[chunk_id]["text"])
        if not (FUITE_PLANCHER <= fuite <= FUITE_PLAFOND):
            motifs[f"fuite hors bande ({'basse' if fuite < FUITE_PLANCHER else 'haute'})"] += 1
            continue

        items.append({
            "qid": f"f{len(items) + 1:03d}",
            "kind": "formula",
            "question": question,
            "gold_chunks": [chunk_id],
            "gold_documents": [candidat["document_id"]],
            "answer_facts": [f"The expression asked for is: {candidat['formule_brute']}"],
            "or_latex": candidat["formule_brute"],
            "or_canonique": candidat["formule_canonique"],
            "or_atomes": candidat["atomes"],
            "numero_equation": candidat["numero_equation"],
            "leurres": selection_formules.leurres(index, candidat),
            "or_hors_fenetre": False,
            "position_or": candidat["position_bloc"],
            "longueur_chunk": candidat["longueur_chunk"],
            "leak": round(fuite, 3),
            "auteur": auteur,
            "sources": [source_du_chunk(index, chunk_id)],
        })
        motifs["retenu"] += 1
    return items, motifs, motifs_selection


# --------------------------------------------------------------------------- famille table_cell

GABARIT = ('In the table captioned "%(legende)s", what value is reported '
           'for "%(ligne)s" in the "%(colonne)s" column?')

#: Longueur retenue de la légende dans la question. Au-delà, la question devient un paragraphe.
LEGENDE_MAX, LEGENDE_MIN_MOTS = 170, 4


def _legende_lisible(legende: str) -> str | None:
    """La légende débarrassée de sa notation, ou ``None`` s'il n'en reste rien d'identifiant.

    Les légendes du corpus portent des mathématiques inline — « … $( N = 8 , 0 2 5 )$ » —
    que MinerU a de surcroît éclatées. Les recopier dans la question la rendrait invalide et
    illisible ; les garder telles quelles ferait échouer ``question_propre``. On les retire,
    et si la légende n'a plus assez de mots pour désigner un tableau, la cellule est refusée
    plutôt que rattrapée.
    """
    texte = re.sub(r"\$[^$]*\$", " ", legende or "")
    texte = re.sub(r"\\[a-zA-Z]+|[{}^_\\]", " ", texte)
    texte = " ".join(texte.replace('"', "'").split())
    if len(re.findall(r"[A-Za-z]{2,}", texte)) < LEGENDE_MIN_MOTS:
        return None
    if len(texte) > LEGENDE_MAX:
        texte = texte[:LEGENDE_MAX].rsplit(" ", 1)[0]
    return texte or None


def batir_tableaux(index: ChunkIndex, cible: int, tirage: int,
                   auteur: str = AUTEUR, reformuler: bool = True) -> tuple[list[dict], Counter]:
    """Questions ``table_cell``. La question naît d'un gabarit, pas d'un modèle.

    **La fuite lexicale est mesurée mais ne barre pas cette famille**, et c'est une décision,
    pas un oubli. Désigner une cellule sans nommer sa légende, sa ligne et sa colonne est
    impossible : toute question de cette famille recopie donc du vocabulaire du passage, et
    sa fuite dépasse le plafond v3. La famille ne mesure pas la découverte — elle mesure la
    **lecture**, conditionnellement au passage servi, et le banc publie l'or servi à côté du
    score pour que les deux ne soient jamais confondus.
    """
    import selection_tableaux  # noqa: E402  (livré par le lot de scoreurs)

    motifs_selection = Counter()
    candidats = selection_tableaux.candidats(index, motifs=motifs_selection)
    alea = random.Random(GRAINE + 1)
    alea.shuffle(candidats)
    motifs = Journal()
    items: list[dict] = []

    for candidat in candidats[:tirage]:
        if len(items) >= cible:
            break
        chunk_id = candidat["chunk_id"]
        motifs.contexte = {"chunk_id": chunk_id, "document_id": candidat["document_id"],
                           "legende": (candidat["legende"] or "")[:160],
                           "libelle_ligne": candidat["libelle_ligne"],
                           "entete_colonne": candidat["entete_colonne"],
                           "or_valeur": candidat["valeur_brute"]}
        servi = fenetre_servie(index, chunk_id)
        valeur = norme_valeurs.lire(candidat["valeur_brute"])

        # Verrou : la cellule est-elle dans ce que la production sert ?
        if valeur is None or not norme_valeurs.contient(servi, valeur):
            motifs["or_absent_de_la_fenetre_servie"] += 1
            continue

        # Un libellé de ligne ou un en-tête de colonne qui porte de la notation — « PR{a> 0} »
        # existe dans le corpus — rendrait la question invalide au sens de ``question_propre``,
        # et surtout illisible : on ne peut pas demander une cellule en recopiant du LaTeX
        # cassé. Ces cellules sortent, comptées, plutôt que d'être nettoyées en douce.
        if any(marque in f"{candidat['libelle_ligne']}{candidat['entete_colonne']}"
               for marque in ("$", "\\", "{", "}", "^", "_")):
            motifs["notation_dans_le_libelle_ou_l_entete"] += 1
            continue
        legende = _legende_lisible(candidat["legende"])
        if legende is None:
            motifs["legende_illisible_apres_retrait_des_maths"] += 1
            continue

        question = GABARIT % {"legende": legende,
                              "ligne": candidat["libelle_ligne"],
                              "colonne": candidat["entete_colonne"]}
        if reformuler:
            try:
                brouillon = _rediger(
                    [{"role": "user", "content": REFORMULER_TABLEAU % {"question": question}}],
                    auteur, 0.3, 220, f"v4-tableau-{chunk_id}")
            except QuotaEpuise:
                # La reformulation est un confort ; le gabarit désigne déjà la cellule sans
                # ambiguïté. Une panne de quota dégrade la famille, elle ne l'annule pas.
                brouillon, reformuler = None, False
                motifs["reformulation_abandonnee_quota_epuise"] += 1
            reecrite = _texte((brouillon or {}).get("question"))
            # La reformulation n'est acceptée que si elle garde les trois identifiants.
            # Sinon on garde le gabarit : une question plus jolie qui ne désigne plus la
            # cellule est une question sans or.
            garde = all(marque.casefold() in reecrite.casefold()
                        for marque in (candidat["libelle_ligne"], candidat["entete_colonne"]))
            if reecrite and garde and not question_propre(reecrite):
                question = reecrite
            else:
                motifs["reformulation_refusee_gabarit_conserve"] += 1

        motif = question_propre(question)
        if motif:
            motifs[motif] += 1
            continue

        items.append({
            "qid": f"t{len(items) + 1:03d}",
            "kind": "table_cell",
            "question": question,
            "gold_chunks": [chunk_id],
            "gold_documents": [candidat["document_id"]],
            "answer_facts": [f"The value reported for {candidat['libelle_ligne']} in the "
                             f"{candidat['entete_colonne']} column is {candidat['valeur_brute']}"],
            "or_valeur": candidat["valeur_brute"],
            "or_canonique": str(valeur.decimal),
            "or_pourcentage": valeur.pourcentage,
            "autres_valeurs": candidat.get("autres_valeurs", []),
            "legende": candidat["legende"],
            "libelle_ligne": candidat["libelle_ligne"],
            "entete_colonne": candidat["entete_colonne"],
            "or_hors_fenetre": False,
            "longueur_chunk": candidat.get("longueur_chunk"),
            "leak": round(index.leak_against_text(question, index.chunks[chunk_id]["text"]), 3),
            "auteur": auteur if reformuler else "gabarit deterministe",
            "sources": [source_du_chunk(index, chunk_id)],
        })
        motifs["retenu"] += 1
    return items, motifs, motifs_selection


# --------------------------------------------------------------------------- famille négative voisine

def _passages_dense(question: str, profondeur: int = 50) -> list[dict]:
    import quant_rag  # noqa: E402  (import tardif : charge le modèle et ouvre Qdrant)
    return quant_rag.search(question, limit=profondeur, pool=profondeur, mode="dense",
                            rerank=False, auto_period=False, dedupe=False, per_document=0,
                            min_characters=0, log=False)


def batir_negatives(index: ChunkIndex, cible: int, tirage: int,
                    auteur: str = AUTEUR) -> tuple[list[dict], Counter]:
    """Questions ``negative_voisine``. L'absence est **prouvée**, jamais supposée.

    Trois preuves, et la troisième est celle qui distingue cette famille des négatives v3 :
    la grandeur demandée est absente du corpus entier (balayage littéral), absente du chunk
    d'or, absente des cinquante candidats denses — **et le chunk d'or reste dans ces
    cinquante**. C'est ce dernier point qui fait la « voisine » : le système voit bien le
    bon passage, sur le bon sujet, et doit constater lui-même que la grandeur n'y est pas.
    """
    positives = [q for q in lire_jsonl(HERE / "questions-v3.jsonl")
                 if q.get("kind") != "negative" and q.get("gold_chunks")]
    alea = random.Random(GRAINE + 2)
    alea.shuffle(positives)
    motifs = Journal()
    items: list[dict] = []
    grandeurs_prises: set[str] = set()

    for origine in positives[:tirage]:
        if len(items) >= cible:
            break
        chunk_id = origine["gold_chunks"][0]
        motifs.contexte = {"qid_origine": origine["qid"], "chunk_voisin": chunk_id,
                           "question_origine": origine["question"]}
        if chunk_id not in index.chunks:
            motifs["chunk_d_or_absent_du_corpus"] += 1
            continue
        servi = fenetre_servie(index, chunk_id)

        try:
            brouillon = _rediger(
                [{"role": "user", "content": REDIGER_NEGATIVE % {
                    "question": origine["question"], "passage": servi}}],
                auteur, 0.6, 420, f"v4-negative-{origine['qid']}")
        except QuotaEpuise as panne:
            motifs["ARRET_quota_de_l_auteur_epuise"] += 1
            print(f"\nquota de l'auteur épuisé après {len(items)} questions — {panne}")
            break
        question = _texte((brouillon or {}).get("question"))
        grandeur = _texte((brouillon or {}).get("quantite"))
        graphies = [g for g in (_texte(x) for x in (brouillon or {}).get("graphies") or []) if g]
        if grandeur and grandeur not in graphies:
            graphies.insert(0, grandeur)
        # Le brouillon n'existe qu'ici : sans cette mise à jour, une négative écartée pour
        # « grandeur présente dans le corpus » serait publiée sans dire QUELLE grandeur.
        motifs.contexte = {**(motifs.contexte or {}), "question": question,
                           "grandeur_proposee": grandeur, "graphies": graphies}

        motif = question_propre(question)
        if motif:
            motifs[motif] += 1
            continue
        if not grandeur or len(graphies) < 3:
            motifs["graphies_insuffisantes"] += 1
            continue
        cle = " ".join(grandeur.casefold().split())
        if cle in grandeurs_prises:
            motifs["grandeur_deja_utilisee"] += 1
            continue

        # Preuve 1 — absence du corpus entier, par balayage littéral de toutes les graphies.
        presentes = [g for g in graphies if not index.phrase_absent(g)]
        if presentes:
            motifs["grandeur_presente_dans_le_corpus"] += 1
            continue
        # Preuve 2 — absence du chunk d'or (redondante avec la 1, vérifiée quand même :
        # une preuve qui ne coûte rien et qu'on n'écrit pas est une preuve qu'on n'a pas).
        minuscule = servi.casefold()
        if any(g.casefold() in minuscule for g in graphies):
            motifs["grandeur_dans_le_chunk_d_or"] += 1
            continue
        # Preuve 3 — absence des cinquante candidats denses, ET or toujours présent parmi eux.
        candidats_denses = _passages_dense(question)
        textes = " \n ".join((r.get("text") or "") for r in candidats_denses).casefold()
        if any(g.casefold() in textes for g in graphies):
            motifs["grandeur_dans_le_top_50_dense"] += 1
            continue
        if chunk_id not in {r["chunk_id"] for r in candidats_denses}:
            motifs["or_d_origine_perdu_la_question_n_est_plus_voisine"] += 1
            continue

        grandeurs_prises.add(cle)
        items.append({
            "qid": f"nv{len(items) + 1:03d}",
            "kind": "negative",
            "sous_famille": "negative_voisine",
            "question": question,
            "gold_chunks": [],
            "gold_documents": [],
            "answer_facts": [],
            "absent_entities": graphies,
            "absent_source": grandeur,
            "honest_answer": (brouillon or {}).get("reponse_honnete", ""),
            "question_origine": origine["question"],
            "qid_origine": origine["qid"],
            "chunk_voisin": chunk_id,
            "leak": None,
            "auteur": auteur,
            "sources": [{**source_du_chunk(index, chunk_id), "role": "voisinage_seulement"}],
        })
        motifs["retenu"] += 1
    return items, motifs, Counter()


# --------------------------------------------------------------------------- vérification payante

#: Plafond d'appels de la phase de vérification, déclaré au §3.4 du pré-enregistrement.
PLAFOND_VERIFICATION = 100
#: Familles soumises à la vérification payante. **``table_cell`` n'y est plus** : la passe du
#: 8 septembre a écarté 0 question sur 45, et un contrôle qui n'écarte rien n'a aucun pouvoir
#: discriminant sur la population qu'il examine (amendement n° 5 du 9 septembre). Sa qualité
#: repose sur des garanties programmatiques testées, pas sur un avis de modèle.
FAMILLES_VERIFIEES = ("formula", "negative_voisine")


def verifier(index: ChunkIndex, plafond: int = PLAFOND_VERIFICATION,
             familles: tuple = FAMILLES_VERIFIEES) -> dict:
    """Le seul usage payant de la fabrique : un modèle **plus fort que l'auteur** relit.

    Il ne rédige rien et ne répare rien — il ne fait que refuser. Un « oui » ne prouve pas
    grand-chose (c'est le même genre de modèle qui a écrit la question) ; un « non » écarte,
    et c'est à cela qu'il sert. L'usage à sens unique d'un outil faillible reste légitime :
    c'est exactement la règle que ``generate_questions.make_negative`` applique déjà au
    retriever, et pour la même raison.

    Les négatives voisines sont soumises **à un autre prompt**, qui ne leur demande rien sur
    l'absence. Leur or EST une absence, établie mécaniquement par balayage littéral du corpus,
    et demander à un modèle de confirmer une absence, c'est lui demander d'attester de sa propre
    ignorance. Ce qu'un relecteur peut juger, en revanche : est-ce une vraie question, tient-elle
    debout sans le passage, demande-t-elle **une** grandeur nommée, et le piège est-il crédible —
    un quant s'attendrait-il à trouver cette grandeur là ? Un piège dont la grandeur n'a rien à
    voir avec le sujet ne teste rien. Le prompt le dit explicitement au modèle, pour qu'il ne
    soit pas tenté de trancher l'absence.

    Le prompt v1 était cassé, et il faut savoir comment
    ---------------------------------------------------
    La première passe a écarté **45 des 45** ``table_cell`` et **14 des 24** ``formula``,
    toutes pour le **même** contrôle, ``self_contained``, et aucune pour ``answerable``,
    ``answer_is_right`` ou ``unambiguous``. Quarante-cinq sur quarante-cinq sur un seul
    contrôle, ce n'est pas quarante-cinq mauvaises questions : c'est un instrument qui dit
    toujours la même chose.

    La cause était dans le prompt, hérité de ``generate_questions.VERIFY`` : « *Any reference
    to "the passage", "the author", "the table", "the above" makes it false* ». Or une
    question de la famille ``table_cell`` **doit** nommer son tableau — c'est la légende qui
    la rend trouvable. La règle de v3 marchait parce que v3 n'avait aucune famille obligée de
    nommer un tableau. Copier une validation d'une famille à l'autre a détruit celle-ci.

    Deux autres dérives sont apparues dans les motifs rendus : le vérificateur a cité « the
    passage » dans une question qui ne contient pas ces mots, et il a compté la
    **répondabilité** comme un défaut d'autonomie (« the passage explicitly provides the
    expression »), ce qui est le contraire de ce qu'un item de banc doit être.

    Le prompt v2 nomme ces trois pièges : il autorise explicitement à désigner un tableau par
    sa légende, exige de **citer** le pointeur fautif faute de quoi le contrôle passe, et
    rappelle que le passage est *censé* répondre. Les deux passes sont publiées.
    """
    rapport = {"modele": VERIFICATEUR, "version_du_prompt": VERSION_VERIFICATION,
               "plafond": plafond, "familles_soumises": list(familles), "familles": {}}
    appels = 0
    for famille in familles:
        chemin = HERE / FICHIERS[famille]
        items = lire_jsonl(chemin)
        if not items:
            continue
        gardes, ecartes, motifs = [], [], Counter()
        for item in items:
            if appels >= plafond:
                # Le plafond est une règle du pré-enregistrement, pas une suggestion : ce qui
                # n'a pas pu être relu est GARDÉ et **marqué**, jamais jeté en silence.
                gardes.append({**item, "verifie": None, "motif_verification": "plafond atteint"})
                motifs["non_relu_plafond_atteint"] += 1
                continue
            negative = famille == "negative_voisine"
            ancre = item["chunk_voisin"] if negative else item["gold_chunks"][0]
            servi = fenetre_servie(index, ancre)
            appels += 1
            if negative:
                contenu = VERIFIER_NEGATIVE_V4 % {
                    "question": item["question"], "passage": servi,
                    "grandeur": item.get("absent_source", "")}
                controles = CONTROLES_NEGATIVE
            else:
                contenu = VERIFIER_V4 % {
                    "question": item["question"], "passage": servi,
                    "or": item.get("or_latex") or item.get("or_valeur") or ""}
                controles = CONTROLES
            # La graine ne porte PLUS le qid. Elle le portait, et deux questions identiques
            # renumérotées d'un tirage à l'autre repayaient donc un verdict déjà rendu — un
            # appel sans bénéfice, ce que la philosophie du dépôt interdit. Le contenu de la
            # requête (question + passage + or) suffit à distinguer deux items ; la version du
            # prompt suffit à distinguer deux passes.
            verdict = llm.complete_json(
                [{"role": "user", "content": contenu}],
                model=VERIFICATEUR, temperature=0.0, max_tokens=320,
                seed=f"v4-verif-{VERSION_VERIFICATION}") or {}
            manques = [c for c in controles if not verdict.get(c)]
            if manques:
                motifs[f"refus/{'+'.join(manques)}"] += 1
                ecartes.append({**item, "verifie": False, "motif_verification": manques,
                                "raison_verification": str(verdict.get("reason", ""))[:200]})
                continue
            motifs["verifie"] += 1
            gardes.append({**item, "verifie": True})
        _ecrire_jsonl(chemin, gardes)
        # Fichier DISTINCT de celui de la fabrique : les deux passes écartent pour des
        # raisons différentes, et écraser l'une par l'autre ferait disparaître des pièces.
        _ecrire_jsonl(HERE / f"questions-v4-{famille.replace('_', '-')}-ecartees-verification.jsonl",
                      ecartes)
        rapport["familles"][famille] = {
            "avant": len(items), "apres": len(gardes), "ecartees": len(ecartes),
            "taux_de_refus": round(len(ecartes) / len(items), 3) if items else None,
            "motifs": dict(motifs), "sha256": empreinte(chemin),
        }
    rapport["appels_utilises"] = appels
    rapport["appels"] = llm.stats()
    _ecrire(CACHE / f"familles-v4-verification-{SIGNATURE}.json", rapport)
    return rapport


# --------------------------------------------------------------------------- exécution

FICHIERS = {
    "formula": "questions-v4-formula.jsonl",
    "table_cell": "questions-v4-table-cell.jsonl",
    "negative_voisine": "questions-v4-negative-voisine.jsonl",
}


def main() -> None:
    analyse = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    analyse.add_argument("--famille", required=True,
                         choices=("formula", "table_cell", "negative_voisine", "selection",
                                  "verification"))
    analyse.add_argument("--cible", type=int, default=45)
    analyse.add_argument("--tirage", type=int, default=90,
                         help="candidats soumis à l'auteur — borne le coût du run")
    analyse.add_argument("--auteur", default=AUTEUR,
                         help="modèle rédacteur. TOUT écart au défaut est un amendement au "
                              "pré-enregistrement : il est écrit dans chaque question et "
                              "entre dans la signature de comparabilité du banc.")
    analyse.add_argument("--verifier-familles", nargs="*", default=None,
                         choices=("formula", "table_cell", "negative_voisine"),
                         help="familles à soumettre à la vérification payante")
    analyse.add_argument("--sans-reformulation", action="store_true",
                         help="table_cell : garder le gabarit déterministe, aucun appel LLM")
    arguments = analyse.parse_args()

    index = ChunkIndex.load(verbose=False)
    debut = time.perf_counter()

    if arguments.famille == "selection":
        # Recensement pur, zéro appel : ce que le pré-enregistrement doit citer.
        formules, motifs_f = selection_formules.candidats(index)
        rapport = {"signature": SIGNATURE, "appels_llm": 0,
                   "formula": {"candidats": len(formules), "motifs": dict(motifs_f)}}
        try:
            import selection_tableaux
            motifs_t = Counter()
            tableaux = selection_tableaux.candidats(index, motifs=motifs_t)
            rapport["table_cell"] = {"candidats": len(tableaux), "motifs": dict(motifs_t),
                                     "documents": len({t["document_id"] for t in tableaux})}
        except ImportError:
            rapport["table_cell"] = {"candidats": None, "motifs": "selection_tableaux absent"}
        rapport["formula"]["documents"] = len({f["document_id"] for f in formules})
        _ecrire(HERE / f"selection-v4-{SIGNATURE}.json", rapport)
        print(json.dumps(rapport, ensure_ascii=False, indent=1)[:3000])
        return

    if arguments.famille == "verification":
        familles = tuple(arguments.verifier_familles or FAMILLES_VERIFIEES)
        rapport = verifier(index, familles=familles)
        print(json.dumps(rapport, ensure_ascii=False, indent=1))
        return

    if arguments.famille == "table_cell":
        items, motifs, motifs_selection = batir_tableaux(
            index, arguments.cible, arguments.tirage, auteur=arguments.auteur,
            reformuler=not arguments.sans_reformulation)
    else:
        batisseurs = {"formula": batir_formules, "negative_voisine": batir_negatives}
        items, motifs, motifs_selection = batisseurs[arguments.famille](
            index, arguments.cible, arguments.tirage, auteur=arguments.auteur)
    chemin = HERE / FICHIERS[arguments.famille]
    _ecrire_jsonl(chemin, items)
    # Les écartées de CE tirage, avec leur motif. Un taux de rejet sans ses pièces est une
    # affirmation ; le dépôt exige les pièces pour tout tirage.
    chemin_ecartes = HERE / f"questions-v4-{arguments.famille.replace('_', '-')}-ecartees.jsonl"
    _ecrire_jsonl(chemin_ecartes, motifs.ecartes)

    rejets = sum(v for k, v in motifs.items() if k != "retenu")
    soumis = rejets + len(items)
    resume = {
        "famille": arguments.famille, "signature": SIGNATURE,
        # L'auteur EFFECTIF, pas celui qu'on a demandé : « --sans-reformulation » sur
        # table_cell laisse le gabarit déterministe et n'appelle personne. Écrire « gemini »
        # dans ce cas ferait croire à une rédaction par modèle qui n'a pas eu lieu.
        "auteur": (auteur_effectif := (
            "gabarit deterministe"
            if arguments.famille == "table_cell" and arguments.sans_reformulation
            else arguments.auteur)),
        "auteur_declare_au_pre_enregistrement": AUTEUR,
        "amendement": (None if auteur_effectif == AUTEUR else
                       f"auteur : {AUTEUR} déclaré -> {auteur_effectif} employé"),
        "cible": arguments.cible, "obtenus": len(items),
        "soumis_a_l_auteur": soumis,
        "taux_de_rejet": round(rejets / soumis, 3) if soumis else None,
        "motifs": {k: v for k, v in sorted(motifs.items())},
        "fichier": chemin.name, "sha256": empreinte(chemin),
        "fichier_ecartees": chemin_ecartes.name, "ecartees": len(motifs.ecartes),
        "appels": llm.stats(), "secondes": round(time.perf_counter() - debut, 1),
    }
    _ecrire(CACHE / f"familles-v4-{arguments.famille}-{SIGNATURE}.json", resume)
    print(json.dumps({k: v for k, v in resume.items() if k != "motifs"},
                     ensure_ascii=False, indent=1))
    print("\nmotifs de rejet :")
    for nom, nombre in sorted(motifs.items(), key=lambda kv: -kv[1]):
        print(f"  {nom:52s} {nombre}")
    print(f"\nécartées versées dans {chemin_ecartes.name} : {len(motifs.ecartes)}")
    if resume["taux_de_rejet"] and resume["taux_de_rejet"] > 0.60:
        print(f"\nRÈGLE D'ARRÊT : taux de rejet {resume['taux_de_rejet']:.1%} > 60 % — "
              f"publier ce taux et s'arrêter, ne pas regénérer à l'aveugle (§5.2).")


if __name__ == "__main__":
    main()
