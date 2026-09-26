"""Fabrique de questions par LLM, conçue contre le biais known-item.

Le problème
-----------
Pour disposer d'une vérité terrain (« quel chunk répond ? »), il faut écrire la
question *à partir* du chunk. C'est mécanique, et ça fuit : celui qui rédige en
regardant la réponse en recopie les termes rares, et le retrieval n'a plus qu'à
faire du appariement de chaînes. Mesuré sur les 25 questions v1 rédigées à la
main : fuite lexicale médiane **0,533**, maximum **0,916** — sur q14, 92 % du
contenu informatif de la question est littéralement dans le passage cible.

La parade, en quatre temps
--------------------------
1. **Cadrage praticien.** Le rédacteur voit le passage mais doit poser la question
   telle qu'on la pose *avant* d'avoir trouvé la réponse : décrire la difficulté,
   pas nommer la solution.
2. **Blanchiment à l'aveugle.** Un second appel reçoit *uniquement la question*,
   jamais le passage. Ne disposant pas de la source, il lui est structurellement
   impossible d'y réintroduire du vocabulaire copié.
3. **Barrière objective.** ``corpus.leak_score`` mesure la fuite résiduelle. Au
   dessus du seuil, on renvoie la question au blanchiment en *nommant les termes
   fautifs*. Deux tours, puis abandon. La boucle est fermée par une métrique
   calculable, pas par un jugement.
4. **Vérification d'aboutissement.** Le blanchiment peut faire dériver la question
   au point qu'elle ne soit plus répondue par le passage. Un vérificateur relit
   question + passage et tranche : répondable, autoportante, et réellement
   dépendante de *ce* passage.

Ce que ce protocole ne prétend pas
----------------------------------
La fuite n'est pas ramenée à zéro et ne peut pas l'être : une question sur la
volatilité rugueuse contiendra « rough volatility ». L'objectif est de passer
sous le premier quartile de v1 (0,425), c'est-à-dire de rendre chaque question v2
plus difficile, lexicalement, que 75 % des questions v1. Le chiffre atteint est
consigné dans le fichier de sortie, question par question — on ne demande à
personne de croire sur parole.

Six familles
------------
    single    réponse dans un passage de texte
    table     le passage cible est un tableau (HTML brut en v2, Markdown depuis v3)
    multi     réponse répartie sur deux documents distincts
    negative  le corpus ne contient pas la réponse ; abstention attendue
    dated     (v3) même chose que single/multi, plus une contrainte de date de
              publication que l'appelant traduit en filtre ``year_min``/``year_max``
              (``retrieval_query`` + ``filters`` dans l'item) ; la borne est choisie
              pour exclure une part substantielle du corpus, sinon elle ne teste rien
    exact     (v3) question en langue naturelle qui contient **un seul** jeton exact
              (acronyme, nom propre) — la frontière du routeur « ≥ 2 jetons exacts »,
              où v1 q15 (« IBM ») échouait en dense. Est-ce isolé ou un motif ?

Usage :
    python rag/benchmark/generate_questions.py --single 22 --table 10 --multi 8 --negative 10
    # compléter v2 en v3 : les 50 questions v2 sont reprises telles quelles (--base),
    # leurs documents et sources absentes sont écartés des nouveaux tirages
    python rag/benchmark/generate_questions.py --base questions-v2.jsonl \\
        --single 30 --table 20 --multi 15 --dated 15 --exact 10 --negative 10 \\
        --output questions-v3.jsonl
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import pickle
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import corpus_overlay  # noqa: E402
import llm  # noqa: E402
from corpus import CACHE, ChunkIndex, STOPWORDS, is_table, sample_seed_chunks, tokenize  # noqa: E402
from quant_rag import exact_tokens  # noqa: E402  — le détecteur de production, pas une copie

OUTPUT = HERE / "questions-v2.jsonl"

#: Premier quartile de la distribution v1 (0,425), arrondi. Toute question qui
#: passe la barrière est donc plus difficile que 75 % du banc d'essai manuel.
LEAK_CEILING = 0.42

#: Plancher. Une question qui ne partage **aucun** mot de contenu avec le passage
#: censé y répondre n'est plus une reformulation, c'est une dérive : le
#: blanchiment a remplacé le besoin d'information au lieu de le reformuler.
#:
#: Mesuré sur la ligne de base v1 de ce banc : 6 questions à fuite exactement
#: 0,00, dont 4 sur des chunks de texte (pour un tableau en HTML brut, une fuite
#: nulle est structurelle et ne prouve rien). Le rappel s'en ressent — dense
#: recall@10 chunk 0,33 sous 0,20 de fuite, contre 0,64 entre 0,20 et 0,30.
#: Le plancher a été ajouté APRÈS la publication de results-e2e-v1.json ; la
#: prochaine génération l'appliquera, cette ligne de base ne l'a pas.
#:
#: Depuis v3 (tableaux en Markdown), le plancher s'applique aussi aux tableaux :
#: leurs en-têtes sont des mots, une fuite nulle y est désormais une dérive comme
#: ailleurs. Les dix questions-tableaux v2 (HTML) n'y étaient pas soumises.
LEAK_FLOOR = 0.10
LAUNDER_ROUNDS = 2
CHUNK_CHARACTERS = 4500

KINDS = ("single", "table", "multi", "dated", "exact", "negative")
PREFIX = {"single": "s", "table": "t", "multi": "m", "dated": "d", "exact": "x", "negative": "n"}

#: Une borne de date qui ne retire pas au moins ce tiers du corpus daté ne teste
#: rien : le filtre doit changer l'ensemble des candidats.
DATED_MAX_SHARE = 0.70
DATED_MIN_DOCUMENTS = 8

#: Ancre d'une question ``exact`` : un jeton présent dans 3 à 200 chunks (assez rare
#: pour que BM25 le distingue — 200 chunks, c'est 1 % du corpus, IDF ≈ 4,5 —, assez
#: répandu pour ne pas être une coquille), dans au moins deux documents, jamais écrit
#: en minuscules dans le passage. Première passe avec (3, 60) et deux occurrences
#: exigées : 59 graines sur 86 sans ancre, 1 question sur 10 — trop strict.
EXACT_DF = (3, 200)

#: Mots ordinaires qu'un rédacteur capitalise en milieu de phrase (« …, Specifically, … »)
#: et que le détecteur de production prend pour des noms propres : ramenés en
#: minuscules avant le comptage, comme un praticien les taperait.
_ORDINARY_DF = 300


# ---------------------------------------------------------------------- prompts

DRAFT = """You are building a retrieval benchmark over a quantitative-finance research corpus.

Below is one passage. Write ONE question that a working quant would genuinely ask, and whose answer is contained in that passage.

Hard constraints. A question breaking any of these is worthless for the benchmark:
1. SELF-CONTAINED. It must make sense to somebody who has never seen this passage. Never write "the passage", "the author", "this table", "the model above", "the paper", "the text".
2. ASKED FROM THE PROBLEM, NOT THE ANSWER. Phrase it the way somebody asks BEFORE they have found the answer: describe the situation, the difficulty, or the quantity they need. Do not name the solution.
3. DO NOT COPY DISTINCTIVE VOCABULARY. If the passage names a particular model, estimator, dataset, theorem or author, describe what it does instead of naming it. Ordinary domain words (volatility, portfolio, spread, hedge) are fine and expected.
4. STILL SPECIFIC. A generic textbook paragraph on the topic must NOT be an adequate answer. The distinguishing constraints of the passage must survive.
5. ONE question, no sub-questions, at most 30 words.

Constraints 3 and 4 pull against each other. That is deliberate: describe the specifics precisely, without using the passage's own labels for them.

Then list the atomic facts from the passage that any correct answer must state: 2 to 4 of them, each one short sentence, each checkable against the passage alone.

Return JSON:
{"question": "...", "answer_facts": ["...", "..."], "why_specific": "one sentence"}

PASSAGE
%s"""

DRAFT_EXACT = """You are building a retrieval benchmark over a quantitative-finance research corpus.

Below is one passage. Write ONE question that a working quant would genuinely ask, and whose answer is contained in that passage.

The asker remembers exactly ONE identifier from what they read: "%(anchor)s". The question MUST contain "%(anchor)s" verbatim, spelled exactly like that, once. Apart from that single identifier, the question must contain NO other proper noun, acronym, ticker, model name, dataset name, author name, or number: describe everything else in plain words.

Other hard constraints:
1. SELF-CONTAINED. Never write "the passage", "the author", "this table", "the paper".
2. ASKED FROM THE PROBLEM, NOT THE ANSWER: describe the situation or the quantity needed, do not name the solution.
3. Ordinary domain words (volatility, portfolio, spread, hedge) are fine; the passage's own distinctive labels are not, except the one identifier.
4. STILL SPECIFIC: a generic textbook paragraph on the topic must not be an adequate answer.
5. ONE question, at most 30 words.

Then list the atomic facts from the passage that any correct answer must state: 2 to 4, each one short sentence, each checkable against the passage alone.

Return JSON:
{"question": "...", "answer_facts": ["...", "..."], "why_specific": "one sentence"}

PASSAGE
%(passage)s"""

DRAFT_TABLE = """You are building a retrieval benchmark over a quantitative-finance research corpus.

Below is a TABLE taken from a research document, together with the document and section it sits in. Write ONE question that a working quant would genuinely ask, and that this table answers.

Hard constraints. A question breaking any of these is worthless for the benchmark:
1. SELF-CONTAINED AND SITUATED. The question must pin down the setting the table reports on — which model, experiment, market, asset class or estimator — precisely enough that, among %(documents)d quantitative-finance documents, no other table could answer it just as well. "What is the highest bid in the order book" is worthless: hundreds of documents print an example order book.
2. ABOUT WHAT THE TABLE ESTABLISHES, not about one arbitrary cell. Ask for a reported magnitude, a comparison between rows or columns, a ranking, a calibrated parameter, or the direction and size of an effect.
3. Never write "the table", "the passage", "the author", "the above", "this study".
4. DO NOT COPY DISTINCTIVE VOCABULARY where a description does the same work. Ordinary domain words (volatility, spread, factor, drawdown) are fine and expected.
5. ONE question, at most 30 words.

If the table is illustrative rather than a result — a worked example, a toy order book, a formatting demonstration — say so and return an empty question; a bad item silently corrupts every measurement taken with it.

Then list the atomic facts the table establishes that any correct answer must state: 2 to 4, each one short sentence, each checkable against the table alone.

Return JSON:
{"question": "...", "answer_facts": ["...", "..."], "why_specific": "one sentence"}

%(passage)s"""

DRAFT_MULTI = """You are building a retrieval benchmark over a quantitative-finance research corpus.

Below are TWO passages from TWO DIFFERENT documents. Write ONE question that can only be answered well by using BOTH — a comparison, a reconciliation, or a synthesis. Answering from a single passage must leave the answer clearly incomplete.

Hard constraints:
1. SELF-CONTAINED. Never refer to "the passages", "the first paper", "both texts", "the author".
2. DO NOT COPY DISTINCTIVE VOCABULARY from either passage: describe named models, estimators or datasets rather than naming them, where you can.
3. The question must be a real question a quant would ask, not an artificial pairing exercise.
4. ONE question, at most 35 words.

Then list 2 to 4 atomic facts a correct answer must state, drawing on both passages.

Return JSON:
{"question": "...", "answer_facts": ["...", "..."], "why_both": "one sentence on what each passage contributes"}

PASSAGE A
%s

PASSAGE B
%s"""

LAUNDER = """Rewrite this technical question the way a practitioner would actually type it — to a colleague, or into a search box.

You do NOT have the source material, and you must not invent any. Keep the information need EXACTLY as it is: do not narrow it, do not broaden it, do not add specifics that are not already present, do not drop constraints that are.

Change: phrasing, word order, register. Prefer plain descriptive wording wherever a plain phrase means the same thing as a technical label. Strip any bookish or exam-like framing.

Keep: every distinguishing constraint already there — numbers, asset classes, conditions, time horizons.

Return JSON: {"question": "..."}

QUESTION
%s"""

LAUNDER_AVOID = """

Additionally, these exact words must NOT appear in your rewrite. Express the same ideas differently — a description, a synonym, a plain-language paraphrase. Do not simply delete them, that would change the information need:
%s"""

LAUNDER_KEEP = """

One exception: the identifier "%s" must stay in the rewrite, spelled exactly like that. It is the one thing the asker remembers."""

VERIFY = """You are validating one item of a retrieval benchmark. Be strict: a bad item silently corrupts every measurement taken with it.

QUESTION
%(question)s

PASSAGE
%(passage)s

Three independent checks:
- answerable: does the PASSAGE actually contain the information needed to answer the QUESTION? "Related to" is not "answers".
- self_contained: does the QUESTION stand on its own, without the PASSAGE in front of you? Any reference to "the passage", "the author", "the table", "the above" makes it false.
- needs_this_passage: picture the whole corpus — %(documents)d quantitative-finance documents: textbooks, papers, preprints. Would many OTHER passages answer this question just as well, or would a competent generic textbook paragraph on the topic do? If either, answer false. True means the question singles out this particular source.

Return JSON: {"answerable": bool, "self_contained": bool, "needs_this_passage": bool, "reason": "one sentence"}"""

VERIFY_MULTI = """You are validating one item of a retrieval benchmark. Be strict.

QUESTION
%s

PASSAGE A
%s

PASSAGE B
%s

Checks:
- answerable: taken together, do the two passages contain what is needed to answer?
- self_contained: does the QUESTION stand on its own, with no reference to "the passages", "both texts", "the author"?
- needs_both: would an answer built from only ONE of the two passages be clearly incomplete?

Return JSON: {"answerable": bool, "self_contained": bool, "needs_both": bool, "reason": "one sentence"}"""

NEGATIVE = """You are building the UNANSWERABLE part of a retrieval benchmark over a quantitative-finance corpus of %(documents)d documents. These items check that the system replies "I do not have this" instead of inventing an answer.

A corpus passage is shown below ONLY to fix the topic. Do not quote it. Do not reuse its notation, its symbols, its variable names, its section numbers or its equation labels — whoever reads your question has never seen it, and a question full of borrowed symbols is unanswerable for the wrong reason.

Write ONE question that:
1. sits in the same topical neighbourhood, so that a retrieval system WILL return related-looking passages. An obviously off-topic question tests nothing: the whole point is that the near-misses are tempting.
2. asks for a SPECIFIC factual result — a reported number, a named finding, a calibrated parameter, a sample period, a dataset detail — attributed to ONE specific named source that is NOT this passage.
3. names a source that is REAL, never invented, and narrow enough that a %(documents)d-document corpus plausibly lacks it: an author-and-year attribution, a named proprietary dataset, a regional exchange, a specific institution's annual report.
4. reads like something a practitioner would ask in good faith. Plain language, at most 25 words.

The named source must carry the specificity, so that removing it would leave the question unanswerable.

Then list 3 to 5 SPELLINGS OF THAT SAME SOURCE — not alternative sources — as they might each appear in a document ("Dupuis and Ellis (1997)", "Dupuis & Ellis", "P. Dupuis"). Every spelling will be checked against the corpus.

Return JSON:
{"question": "...", "source": "the source as named in the question", "spellings": ["...", "..."], "honest_answer": "one sentence stating what a system without this source should say"}

PASSAGE
%(passage)s"""

NEGATIVE_CHECK = """You are validating an item that is supposed to be UNANSWERABLE from this corpus.

QUESTION
%s

The passages below are what the strongest retrieval configuration returned for it.

%s

Does ANY of these passages actually contain the specific fact the question asks for? Topical proximity is not an answer; a partial or approximate match is not an answer either. Only say true if a passage genuinely supplies the requested fact.

Return JSON: {"answered_by_corpus": bool, "which": "passage number or none", "reason": "one sentence"}"""


# ---------------------------------------------------------------------- outils

def shared_rare_terms(index: ChunkIndex, question: str, text: str, top: int = 6,
                      keep: str | None = None) -> list[str]:
    """Termes informatifs communs à la question et au passage, les plus rares d'abord."""
    target = set(tokenize(text))
    protected = set(tokenize(keep)) if keep else set()
    shared = {t for t in set(tokenize(question))
              if t in target and t not in STOPWORDS and len(t) > 2 and t not in protected}
    return sorted(shared, key=lambda t: -index.idf(t))[:top]


def launder(index: ChunkIndex, question: str, gold_text: str, seed: str,
            keep: str | None = None) -> tuple[str, float, int]:
    """Blanchiment à l'aveugle, relancé tant que la fuite dépasse le plafond.

    Le blanchisseur ne voit jamais ``gold_text`` ; celui-ci ne sert qu'à calculer
    la fuite et à désigner les termes à éviter au tour suivant. ``keep`` (famille
    ``exact``) : l'identifiant que la réécriture doit conserver tel quel.
    """
    def acceptable(value: float) -> bool:
        return LEAK_FLOOR <= value <= LEAK_CEILING

    def intact(candidate: str) -> bool:
        return keep is None or keep in candidate

    best, best_leak = question, index.leak_against_text(question, gold_text)
    if acceptable(best_leak):
        return best, best_leak, 0
    for round_index in range(LAUNDER_ROUNDS):
        prompt = LAUNDER % best
        if round_index:
            avoid = shared_rare_terms(index, best, gold_text, keep=keep)
            if not avoid:
                break
            prompt += LAUNDER_AVOID % ", ".join(avoid)
        if keep:
            prompt += LAUNDER_KEEP % keep
        parsed = llm.complete_json([{"role": "user", "content": prompt}], model=llm.AUTHOR,
                                   temperature=0.3, max_tokens=250, seed=f"{seed}-l{round_index}")
        candidate = (parsed or {}).get("question", "").strip()
        if not candidate or not intact(candidate):
            break
        leak = index.leak_against_text(candidate, gold_text)
        # Blanchir *juste assez*. Retenir systématiquement la fuite la plus basse
        # pousse la question hors de son passage : on préfère donc le candidat le
        # plus proche du plafond parmi ceux qui sont dans la bande, et on ne
        # descend en dessous du plancher que faute de mieux.
        if acceptable(leak) and (not acceptable(best_leak) or leak > best_leak):
            best, best_leak = candidate, leak
        elif not acceptable(best_leak) and best_leak > LEAK_CEILING and leak < best_leak:
            best, best_leak = candidate, leak
        if acceptable(best_leak):
            return best, best_leak, round_index + 1
    return best, best_leak, LAUNDER_ROUNDS


def rare_postings(index: ChunkIndex, low: int = 3, high: int = 60) -> dict:
    """Index inversé sur les termes rares — sert à apparier deux documents.

    Pourquoi pas le retriever pour trouver les paires ? Parce que fabriquer les
    questions multi-documents avec le système qu'on évalue avantagerait ce
    système. L'appariement se fait donc sur une statistique du corpus (le
    vocabulaire rare partagé), indépendante de toute configuration sous test.

    Biais résiduel assumé : cet appariement est lexical, il favorise donc
    légèrement BM25. C'est le sens conservateur pour la conclusion qu'on teste
    (l'hybride bat le dense), donc acceptable — mais c'est écrit noir sur blanc.
    """
    path = CACHE / f"rare-postings-{low}-{high}-{corpus_overlay.signature()}.pkl"
    if path.exists():
        with path.open("rb") as handle:
            return pickle.load(handle)
    postings: dict[str, list[str]] = {}
    for chunk_id, row in index.chunks.items():
        for term in set(tokenize(row["text"])):
            if low <= index.df.get(term, 0) <= high:
                postings.setdefault(term, []).append(chunk_id)
    CACHE.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(postings, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return postings


def find_partner(index: ChunkIndex, postings: dict, chunk: dict, allow_document=None) -> dict | None:
    """Meilleur chunk d'un *autre* document partageant le vocabulaire rare."""
    scores: dict[str, float] = {}
    for term in set(tokenize(chunk["text"])):
        holders = postings.get(term)
        if not holders:
            continue
        weight = index.idf(term)
        for other in holders:
            if index.chunks[other]["document_id"] != chunk["document_id"]:
                scores[other] = scores.get(other, 0.0) + weight
    for chunk_id, _ in sorted(scores.items(), key=lambda kv: -kv[1])[:20]:
        row = index.chunks[chunk_id]
        if allow_document is not None and not allow_document(row["document_id"]):
            continue
        if len(row["text"]) >= 900 and not is_table(row):
            return row
    return None


def source_of(row: dict, index: ChunkIndex) -> dict:
    return {"chunk_id": row["chunk_id"], "document_id": row["document_id"],
            "title": index.display_title(row["document_id"])[:120],
            "section": (row["section"] or "")[:120],
            "page_start": row["page_start"], "content_type": row["content_type"]}


# ---------------------------------------------------------------------- familles

def situate(chunk: dict, index: ChunkIndex, characters: int = CHUNK_CHARACTERS) -> str:
    """Passage précédé de sa provenance : sans elle, le rédacteur ne peut pas
    situer un tableau, et produit des questions vraies partout donc utiles nulle part.

    Le titre montré est le titre *propre* (métadonnées consolidées) : le titre
    d'export est parfois un nom de fichier de 200 caractères, et pour un document
    un tableau HTML entier."""
    title = index.display_title(chunk["document_id"])[:110]
    section = (chunk["section"] or "")[:110]
    head = f"DOCUMENT: {title}\nSECTION: {section}\n\n" if title or section else ""
    return f"{head}{chunk['text'][:characters]}"


def make_grounded(index: ChunkIndex, chunk: dict, kind: str, seed: str, log,
                  anchor: str | None = None) -> dict | None:
    """Question `single`, `table` ou `exact` : une source, un chunk d'or."""
    if kind == "exact":
        prompt = DRAFT_EXACT % {"anchor": anchor, "passage": situate(chunk, index)}
    elif kind == "table":
        prompt = DRAFT_TABLE % {"documents": len(index.documents), "passage": situate(chunk, index)}
    else:
        prompt = DRAFT % situate(chunk, index)
    draft = llm.complete_json([{"role": "user", "content": prompt}],
                              model=llm.AUTHOR, temperature=0.4, max_tokens=500, seed=seed)
    question = (draft or {}).get("question", "").strip()
    facts = [f.strip() for f in (draft or {}).get("answer_facts", []) if str(f).strip()]
    if not question or len(facts) < 2:
        return log("brouillon inexploitable")
    if anchor and anchor not in question:
        return log(f"ancre « {anchor} » absente du brouillon")

    leak_draft = index.leak_against_text(question, chunk["text"])
    question, leak, rounds = launder(index, question, chunk["text"], seed, keep=anchor)
    if leak > LEAK_CEILING:
        return log(f"fuite {leak:.2f} > {LEAK_CEILING}")
    if leak < LEAK_FLOOR:
        return log(f"fuite {leak:.2f} < {LEAK_FLOOR} (question dérivée de sa source)")
    if anchor:
        question = normalise_case(index, question, anchor)
        found = exact_tokens(question)
        if anchor not in question or found["n_exact"] != 1:
            return log(f"jetons exacts = {found['n_exact']} ({', '.join(found['acronyms'] + found['proper'] + found['numeric'])[:60]}), attendu 1")

    check = llm.complete_json(
        [{"role": "user", "content": VERIFY % {"question": question, "passage": chunk["text"][:CHUNK_CHARACTERS],
                                              "documents": len(index.documents)}}],
        model=llm.JUDGE, temperature=0.0, max_tokens=250, seed=seed) or {}
    if not (check.get("answerable") and check.get("self_contained") and check.get("needs_this_passage")):
        return log(f"vérif: {check.get('reason', 'rejetée')[:70]}")

    item = {"kind": kind, "question": question, "gold_chunks": [chunk["chunk_id"]],
            "gold_documents": [chunk["document_id"]], "answer_facts": facts,
            "leak": round(leak, 3), "leak_draft": round(leak_draft, 3), "laundering_rounds": rounds,
            "sources": [source_of(chunk, index)]}
    if anchor:
        item.update({"anchor": anchor, "anchor_df_chunks": index.df.get(anchor.casefold(), 0), "n_exact": 1})
    return item


def make_multi(index: ChunkIndex, chunk: dict, partner: dict, seed: str, log) -> dict | None:
    """Question `multi` : deux documents, deux chunks d'or."""
    draft = llm.complete_json(
        [{"role": "user", "content": DRAFT_MULTI % (chunk["text"][:3200], partner["text"][:3200])}],
        model=llm.AUTHOR, temperature=0.4, max_tokens=500, seed=seed)
    question = (draft or {}).get("question", "").strip()
    facts = [f.strip() for f in (draft or {}).get("answer_facts", []) if str(f).strip()]
    if not question or len(facts) < 2:
        return log("brouillon inexploitable")

    joined = chunk["text"] + "\n" + partner["text"]
    leak_draft = index.leak_against_text(question, joined)
    question, leak, rounds = launder(index, question, joined, seed)
    if leak > LEAK_CEILING:
        return log(f"fuite {leak:.2f} > {LEAK_CEILING}")
    if leak < LEAK_FLOOR:
        return log(f"fuite {leak:.2f} < {LEAK_FLOOR} (question dérivée de sa source)")

    check = llm.complete_json(
        [{"role": "user", "content": VERIFY_MULTI % (question, chunk["text"][:3200], partner["text"][:3200])}],
        model=llm.JUDGE, temperature=0.0, max_tokens=250, seed=seed) or {}
    if not (check.get("answerable") and check.get("self_contained") and check.get("needs_both")):
        return log(f"vérif: {check.get('reason', 'rejetée')[:70]}")

    return {"kind": "multi", "question": question,
            "gold_chunks": [chunk["chunk_id"], partner["chunk_id"]],
            "gold_documents": [chunk["document_id"], partner["document_id"]],
            "answer_facts": facts, "leak": round(leak, 3), "leak_draft": round(leak_draft, 3),
            "laundering_rounds": rounds,
            "sources": [source_of(chunk, index), source_of(partner, index)]}


def make_negative(index: ChunkIndex, chunk: dict, seed: str, log, verifier,
                  used_sources: set | None = None) -> dict | None:
    """Question `negative` : l'absence est prouvée lexicalement, pas devinée.

    Deux barrières, et l'asymétrie entre les deux compte :

    - **Barrière lexicale (preuve).** Le LLM fournit 3 à 5 *graphies de la même
      source* ; toutes doivent être introuvables dans le corpus normalisé, par
      balayage littéral de tous les chunks (``corpus.phrase_absent``). Le balayage
      porte sur le texte entier, pas sur un top-k de recherche : une absence
      constatée là est un fait, pas une présomption. Tester les tokens séparément
      ne suffirait pas — « Paul Dupuis » a ses deux tokens dans le corpus, dans
      deux documents sans rapport, alors que la locution n'y figure nulle part.
    - **Barrière par le retriever (réfutation seulement).** On interroge la
      configuration la plus forte et on demande à un LLM si l'un des passages
      rendus répond réellement. Un « oui » disqualifie la question. Un « non »
      ne prouve rien — ce serait demander au système sous test d'attester de sa
      propre ignorance. Le retriever ne sert donc qu'à *rejeter*, jamais à
      valider. L'usage à sens unique d'un outil circulaire reste légitime.
    """
    draft = llm.complete_json([{"role": "user", "content": NEGATIVE % {"documents": len(index.documents),
                                                                       "passage": chunk["text"][:3000]}}],
                              model=llm.AUTHOR, temperature=0.6, max_tokens=450, seed=seed)
    question = (draft or {}).get("question", "").strip()
    source = str((draft or {}).get("source", "")).strip()
    entities = [str(e).strip() for e in (draft or {}).get("spellings", []) if str(e).strip()]
    if source and source not in entities:
        entities.insert(0, source)
    if not question or not entities:
        return log("brouillon inexploitable")
    # Une question qui recopie la notation du passage-graine n'est pas une question.
    if sum(question.count(c) for c in "$∈↦𝒢∫∂") > 1 or re.search(r"\b\d+\.\d+\.\d+\b", question):
        return log("notation recopiée du passage-graine")

    # Toutes les graphies doivent être absentes : il suffit qu'une seule figure
    # dans le corpus pour que la question cesse d'être sans réponse.
    present = [e for e in entities if not index.phrase_absent(e)]
    if present:
        return log(f"source présente dans le corpus ({', '.join(present[:3])})")

    # Une même source absente réutilisée sur plusieurs questions corrèle les
    # échecs : le score de la famille bougerait en bloc au lieu de mesurer dix
    # situations indépendantes.
    key = " ".join(source.casefold().split())
    if used_sources is not None and key in used_sources:
        return log(f"source déjà utilisée ({source})")

    verdict = verifier(question, seed)
    if verdict.get("answered_by_corpus"):
        return log(f"le corpus répond ({verdict.get('which')})")

    if used_sources is not None:
        used_sources.add(key)
    return {"kind": "negative", "question": question, "gold_chunks": [], "gold_documents": [],
            "answer_facts": [], "absent_entities": entities, "absent_source": source,
            "honest_answer": (draft or {}).get("honest_answer", ""),
            "leak": None, "leak_draft": None, "laundering_rounds": 0,
            "sources": [{**source_of(chunk, index), "role": "topical_seed_only"}]}


# ---------------------------------------------------------------------- v3 : datées

def temporal_filter(gold_years: list[int], years: dict[str, int]) -> dict | None:
    """Borne(s) de date englobant les années d'or et retirant une part substantielle
    du corpus daté — sinon le filtre ne change rien et ne teste rien.

    Candidates : ``year_min`` seul, ``year_max`` seul, ou une fenêtre ; on retient
    celle dont la portée est la plus proche de 40 % des documents datés, à
    condition qu'elle en garde au moins ``DATED_MIN_DOCUMENTS`` et au plus
    ``DATED_MAX_SHARE``. Déterministe : mêmes années d'or, même filtre.
    """
    low, high = min(gold_years), max(gold_years)
    total = len(years)
    candidates = []
    for k in (0, 1, 2, 3):
        candidates.append({"year_min": low - k})
        candidates.append({"year_max": high + k})
        candidates.append({"year_min": low - k, "year_max": high + k})
    best = None
    for filters in candidates:
        scope = [d for d, y in years.items()
                 if (filters.get("year_min") is None or y >= filters["year_min"])
                 and (filters.get("year_max") is None or y <= filters["year_max"])]
        share = len(scope) / total
        if len(scope) < DATED_MIN_DOCUMENTS or share > DATED_MAX_SHARE:
            continue
        key = (abs(share - 0.40), len(filters), sorted(filters.items()))
        if best is None or key < best[0]:
            best = (key, filters, len(scope), round(share, 3))
    if best is None:
        return None
    _, filters, scope_size, share = best
    if "year_min" in filters and "year_max" in filters:
        clause = f"published between {filters['year_min']} and {filters['year_max']}"
    elif "year_min" in filters:
        clause = f"published in {filters['year_min']} or later"
    else:
        clause = f"published in {filters['year_max']} or earlier"
    return {"filters": filters, "clause": clause, "scope_documents": scope_size, "scope_share": share}


def date_question(item: dict, gold_years: list[int], years: dict[str, int], log) -> dict | None:
    """Ajoute la contrainte de date à une question déjà validée (single ou multi)."""
    chosen = temporal_filter(gold_years, years)
    if chosen is None:
        return log(f"aucune borne discriminante pour {gold_years}")
    base = item["question"].strip()
    stem = base[:-1].rstrip() if base.endswith("?") else base
    item.update({
        "kind": "dated",
        "retrieval_query": base,
        "question": f"{stem}, according to sources {chosen['clause']}?",
        "filters": chosen["filters"],
        "filter_scope_documents": chosen["scope_documents"],
        "filter_scope_share": chosen["scope_share"],
        "gold_years": gold_years,
        "dated_variant": "multi" if len(item["gold_chunks"]) > 1 else "single",
    })
    return item


# ---------------------------------------------------------------------- v3 : jeton exact

_CANDIDATE = re.compile(r"(?<![\w-])([A-Z][A-Za-z0-9]*(?:[-'][A-Za-z0-9]+)*)(?![\w-])")
_SENTENCE_START = re.compile(r"(?:^|[.!?:;]\s+|\n\s*)([A-Z][A-Za-z0-9'-]*)")
_ANCHOR_STOP = {"Table", "Figure", "Section", "Chapter", "Equation", "Appendix", "Theorem", "Lemma",
                "Proposition", "Corollary", "Definition", "Example", "Note", "Remark", "Proof", "Panel",
                "The", "This", "These", "That", "There", "Then", "Thus", "Hence", "However", "Therefore",
                "Since", "When", "Where", "While", "For", "In", "On", "If", "As", "At", "By", "To", "We", "It",
                "Let", "Note", "Consider", "Suppose", "Assume", "Given", "Using", "See", "Also", "Both",
                "First", "Second", "Third", "Finally", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
                "January", "February", "March", "April", "May", "June", "July", "August", "September",
                "October", "November", "December", "Chapter", "Part", "Vol", "No", "Eq", "Fig"}


def anchor_candidates(index: ChunkIndex, chunk: dict) -> list[dict]:
    """Jetons exacts qu'un lecteur retiendrait du passage : capitalisés, récurrents,
    jamais en minuscules dans le passage (donc pas un mot ordinaire), absents du
    titre du document (sinon le titre, embarqué dans chaque chunk, suffirait),
    présents dans 3 à 60 chunks du corpus et dans au moins deux documents."""
    text = chunk["text"]
    starts = set(_SENTENCE_START.findall(text))
    counts = collections.Counter(_CANDIDATE.findall(text))
    lowered = set(tokenize(text))
    title_tokens = set(tokenize(index.display_title(chunk["document_id"]))) | set(tokenize(index.title_of(chunk["document_id"])))
    out = []
    for token, n in counts.items():
        core = re.sub(r"[^A-Za-z0-9]", "", token)
        # trois lettres : seulement en capitales (IBM, GMV) ; « Cla », « Cop » sont des abréviations de colonne
        if len(core) < 3 or (len(core) == 3 and not core.isupper()) or token in _ANCHOR_STOP or core.isdigit():
            continue
        # un acronyme (capitales) est distinctif dès une occurrence ; un nom propre
        # doit revenir, ou ne pas être en début de phrase (où tout mot est capitalisé)
        acronym = core.isupper() and len(core) >= 3
        if not acronym and (n < 1 or (token in starts and n - 1 < 1)):
            continue
        folded = token.casefold()
        if folded in title_tokens or folded not in lowered:
            continue
        # jamais écrit en minuscules dans le passage : un nom, pas un mot
        if re.search(rf"(?<![\w-]){re.escape(folded)}(?![\w-])", text):
            continue
        df = index.df.get(folded, 0)
        if not (EXACT_DF[0] <= df <= EXACT_DF[1]):
            continue
        ex = exact_tokens(f"about {token} here")
        if ex["n_exact"] != 1:
            continue
        out.append({"anchor": token, "occurrences": n, "df": df})
    return sorted(out, key=lambda a: (-a["occurrences"], a["df"], a["anchor"]))


def anchor_documents(index: ChunkIndex, anchor: str) -> int:
    """Nombre de documents contenant le jeton (balayage direct : les postings rares
    s'arrêtent à 60 chunks, l'ancre peut aller jusqu'à 200)."""
    folded = anchor.casefold()
    pattern = re.compile(rf"(?<![\w-]){re.escape(folded)}(?![\w-])")
    return len({row["document_id"] for row in index.chunks.values() if pattern.search(row["text"].casefold())})


def normalise_case(index: ChunkIndex, question: str, anchor: str) -> str:
    """Ramène en minuscules les mots ordinaires capitalisés hors début de phrase.

    Le détecteur de jetons exacts de production compte « Specifically » comme un
    nom propre ; un praticien ne l'aurait pas capitalisé. Seuls les mots dont la
    forme minuscule est courante dans le corpus (df ≥ 300) sont touchés — jamais
    l'ancre, jamais un acronyme, jamais un mot rare.
    """
    words = question.split(" ")
    for i in range(1, len(words)):
        word = words[i]
        core = re.sub(r"[^A-Za-z]", "", word)
        if not core or word.startswith(anchor) or core.isupper() or not core[0].isupper():
            continue
        if words[i - 1].endswith((".", "?", "!")):
            continue
        if index.df.get(core.casefold(), 0) >= _ORDINARY_DF:
            words[i] = word.replace(core, core.lower(), 1)
    return " ".join(words)


# ---------------------------------------------------------------------- pilote

def negative_verifier(index: ChunkIndex):
    """Ferme la barrière de réfutation sur la configuration la plus forte."""
    import pipeline

    def verify(question: str, seed: str) -> dict:
        ranked = pipeline.retrieve(question, "rrf_rerank", index, limit=6)
        context = pipeline.build_context(ranked, passages=5)
        blob = pipeline.format_passages(context, characters=1200)
        return llm.complete_json(
            [{"role": "user", "content": NEGATIVE_CHECK % (question, blob)}],
            model=llm.JUDGE, temperature=0.0, max_tokens=250, seed=f"{seed}-neg") or {}

    return verify


def read_items(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def run(counts: dict, seed_value: int, oversample: float, exclude: Path | None = None,
        base_items: list[dict] | None = None, dated_multi: int = 5,
        allow_used_documents: bool = False) -> list[dict]:
    index = ChunkIndex.load()
    questions: list[dict] = []
    rejected: dict[str, list[str]] = {}
    used_documents: set[str] = set()
    used_chunks: set[str] = set()
    used_sources: set[str] = set()
    postings = None
    verifier = None
    years = index.years(reliable=True)
    next_number = collections.Counter()

    # Complément d'un banc existant : ne pas réutiliser ses documents ni ses
    # sources absentes, sans quoi le complément ferait doublon avec le socle.
    previous_items = list(base_items or [])
    if exclude and exclude.exists():
        previous_items.extend(read_items(exclude))
    for previous in previous_items:
        used_documents.update(previous.get("gold_documents", []))
        used_chunks.update(previous.get("gold_chunks", []))
        for chunk_source in previous.get("sources", []):
            used_documents.add(chunk_source.get("document_id"))
            used_chunks.add(chunk_source.get("chunk_id"))
        if previous.get("absent_source"):
            used_sources.add(" ".join(previous["absent_source"].casefold().split()))
        match = re.match(r"([a-z])(\d+)$", previous.get("qid", ""))
        if match:
            next_number[match.group(1)] = max(next_number[match.group(1)], int(match.group(2)))
    if previous_items:
        print(f"  exclusion : {len(used_documents)} documents et {len(used_sources)} sources "
              f"déjà pris par le banc de base")

    for kind, target in counts.items():
        if target <= 0:
            continue
        print(f"\n=== {kind} : objectif {target}")
        pool_kind = "table" if kind == "table" else "text"
        # hash() sur str est randomisé par processus : décalage stable à la place,
        # sans quoi deux exécutions ne tireraient pas les mêmes graines.
        offset = int(hashlib.sha256(kind.encode()).hexdigest()[:6], 16) % 100_000
        allow = (lambda doc: doc in years) if kind == "dated" else None
        # Complément d'une famille dont les documents vierges sont épuisés (tableaux :
        # 100 documents en portent, v2 en avait pris 10) : on autorise un document
        # déjà sollicité, jamais un chunk déjà cible.
        seeds = sample_seed_chunks(index, int(target * oversample) + 6, pool_kind,
                                   seed_value + offset,
                                   exclude_documents=set() if allow_used_documents else used_documents,
                                   exclude_chunks=used_chunks, allow_document=allow)
        if kind in ("multi", "dated") and postings is None:
            print("  appariement  index inversé sur les termes rares…")
            postings = rare_postings(index)
        if kind == "negative" and verifier is None:
            verifier = negative_verifier(index)

        made = 0
        for position, chunk in enumerate(seeds):
            if made >= target:
                break
            qid = f"{PREFIX[kind]}{next_number[PREFIX[kind]] + made + 1:02d}"
            reasons: list[str] = []

            def log(reason, _reasons=reasons):
                _reasons.append(reason)
                return None

            item = None
            try:
                if kind == "multi":
                    partner = find_partner(index, postings, chunk)
                    if partner is None:
                        log("aucun partenaire trouvé")
                    else:
                        item = make_multi(index, chunk, partner, f"{kind}-{position}", log)
                elif kind == "dated":
                    # les ``dated_multi`` dernières portent sur deux documents datés
                    want_multi = made >= target - dated_multi
                    if want_multi:
                        partner = find_partner(index, postings, chunk, allow_document=lambda d: d in years)
                        if partner is None:
                            log("aucun partenaire daté trouvé")
                        else:
                            item = make_multi(index, chunk, partner, f"{kind}-{position}", log)
                    else:
                        item = make_grounded(index, chunk, "single", f"{kind}-{position}", log)
                    if item is not None:
                        item = date_question(item, [years[d] for d in item["gold_documents"]], years, log)
                elif kind == "exact":
                    candidates = [a for a in anchor_candidates(index, chunk)
                                  if anchor_documents(index, a["anchor"]) >= 2]
                    if not candidates:
                        log("aucune ancre exploitable dans le passage")
                    else:
                        anchor = candidates[0]
                        item = make_grounded(index, chunk, "exact", f"{kind}-{position}", log, anchor=anchor["anchor"])
                        if item is not None:
                            item["anchor_df_documents"] = anchor_documents(index, anchor["anchor"])
                            item["anchor_occurrences_in_passage"] = anchor["occurrences"]
                elif kind == "negative":
                    item = make_negative(index, chunk, f"{kind}-{position}", log, verifier,
                                         used_sources)
                else:
                    item = make_grounded(index, chunk, kind, f"{kind}-{position}", log)
            except Exception as error:  # une graine fautive ne doit pas tuer la campagne
                log(f"{type(error).__name__}: {str(error)[:80]}")

            if item is None:
                rejected.setdefault(kind, []).extend(reasons)
                print(f"  ✗ {chunk['chunk_id'][:20]}  {reasons[-1] if reasons else 'rejetée'}")
                continue

            made += 1
            item = {"qid": qid, **item, "author_model": llm.AUTHOR, "verifier_model": llm.JUDGE,
                    "corpus_state": f"{corpus_overlay.LABEL}-{corpus_overlay.signature()}"}
            used_documents.update(item["gold_documents"])
            used_chunks.update(item["gold_chunks"])
            questions.append(item)
            leak = f"fuite {item['leak']:.2f}" if item["leak"] is not None else "—"
            extra = (f"  [{item['filters']}]" if kind == "dated" else f"  [{item['anchor']}]" if kind == "exact" else "")
            print(f"  ✓ {qid}  {leak}  {item['question'][:88]}{extra}")

        if made < target:
            print(f"  ! {kind} : {made}/{target} seulement "
                  f"({len(seeds)} graines épuisées, élargis --oversample)")

    print("\n--- rejets par motif")
    for kind, reasons in rejected.items():
        buckets: dict[str, int] = {}
        for reason in reasons:
            head = reason.split(":")[0].split("(")[0].strip()
            buckets[head] = buckets.get(head, 0) + 1
        summary = ", ".join(f"{k} ×{v}" for k, v in sorted(buckets.items(), key=lambda kv: -kv[1]))
        print(f"  {kind:9s} {len(reasons):>3} rejets — {summary}")
    return questions


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--single", type=int, default=22)
    parser.add_argument("--table", type=int, default=10)
    parser.add_argument("--multi", type=int, default=8)
    parser.add_argument("--dated", type=int, default=0)
    parser.add_argument("--dated-multi", type=int, default=5,
                        help="parmi les questions datées, combien portent sur deux documents")
    parser.add_argument("--exact", type=int, default=0)
    parser.add_argument("--negative", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--oversample", type=float, default=5.0,
                        help="graines tirées par question visée (les barrières en rejettent)")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--exclude", type=Path, default=None,
                        help="banc existant à compléter : ses documents et ses sources sont écartés")
    parser.add_argument("--base", type=Path, default=None,
                        help="banc existant repris tel quel en tête du fichier de sortie (implique --exclude)")
    parser.add_argument("--allow-used-documents", action="store_true",
                        help="complément : autorise un document déjà sollicité (jamais un chunk déjà cible)")
    args = parser.parse_args()

    started = time.perf_counter()
    counts = {"single": args.single, "table": args.table, "multi": args.multi,
              "dated": args.dated, "exact": args.exact, "negative": args.negative}
    base_items = read_items(args.base) if args.base else None
    questions = run(counts, args.seed, args.oversample, args.exclude, base_items, args.dated_multi,
                    args.allow_used_documents)

    lines = []
    if args.base:
        # octet pour octet : le socle n'est pas régénéré, il est recopié
        lines.extend(line for line in args.base.read_text(encoding="utf-8").splitlines() if line.strip())
    lines.extend(json.dumps(q, ensure_ascii=False) for q in questions)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    leaks = sorted(q["leak"] for q in questions if q["leak"] is not None)
    print(f"\n{len(questions)} questions nouvelles{f' + {len(base_items)} reprises' if base_items else ''} -> {args.output}")
    if leaks:
        print(f"fuite lexicale : médiane {leaks[len(leaks)//2]:.3f}  "
              f"max {leaks[-1]:.3f}  (v1 : médiane 0,533  max 0,916)")
    print(f"{time.perf_counter() - started:.0f} s, LLM {llm.stats()}")


if __name__ == "__main__":
    main()
