"""Le système sous test : configurations de retrieval + génération de réponse.

Ce module n'évalue rien. Il expose ce qu'on mesure, pour que ``run_benchmark.py``
puisse comparer des configurations sans dupliquer de logique.

Quatre configurations, reprises de ``eval_hybrid.py`` afin que les chiffres du
banc end-to-end soient directement comparables à ceux du banc known-item :

    dense          l'ancien chemin de production (dense seul)
    bm25           lexical seul
    rrf            fusion réciproque dense + BM25
    rrf_rerank     fusion puis cross-encoder bge-reranker-base
    router         le chemin de production actuel : ``quant_rag.search(mode="auto")``,
                   qui choisit dense ou rrf_rerank par requête (règle : ≥ 2 jetons exacts)

Le générateur de réponse est volontairement contraint : il ne dispose que des
passages, et il a une porte de sortie explicite — le sentinelle
``INSUFFICIENT_EVIDENCE``. Ce détail a une conséquence importante sur
l'évaluation : *l'abstention devient mesurable sans juge*. C'est une chaîne de
caractères, pas une appréciation.

Deux prompts de réponse coexistent (``ANSWER_PROMPTS``) : ``v1``, celui des lignes de
base jusqu'au 3 septembre 2026, et ``v2``, le défaut depuis — même contrat, mais
l'abstention n'est permise que lorsqu'aucun passage ne porte sur la question
(chantier E, ``abstention.py``). L'en-tête de chaque passage porte désormais la
référence courte (auteurs, année), comme la production (``PASSAGE_YEAR``).
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(ROOT / "src"))

import contrat  # noqa: E402  — la fenêtre servie, une seule fois, pour le banc et la production
import llm  # noqa: E402
import quant_rag  # noqa: E402
from retrieval.hybrid import reciprocal_rank_fusion  # noqa: E402

CONFIGS = ("dense", "bm25", "rrf", "rrf_rerank", "router")
POOL = 50

#: Nombre de passages remis au générateur. 5 est le réglage de production
#: (``quant_rag.search(limit=5)``) ; c'est donc ce qu'on évalue.
CONTEXT_PASSAGES = 5


def bm25():
    """L'index BM25 de la production, et lui seul (``quant_rag.bm25``).

    Le banc a eu son propre constructeur, qui relisait ``rows.jsonl`` — les
    19 443 chunks livrés, sans les overlays. Après la déduplication et la
    conversion des tableaux, ce repli aurait silencieusement rebâti un index
    sur un corpus que la production ne sert plus. Une seule source, nommée
    par la signature de l'état du corpus : voir ``quant_rag.BM25_PATH``.
    """
    return quant_rag.bm25()


def _with_text(rows: list[dict], index) -> list[dict]:
    """Complète des candidats BM25 avec leur texte, lu dans l'index corpus.

    Piège documenté au point 2 de la todolist : un candidat sans champ ``text``
    est silencieusement ignoré par le reranker, ce qui coûtait 8 points de R@1.

    Le titre vient de la **fiche consolidée**, comme en production. ``index.title_of``
    rend le titre de l'export, qui pour les 319 documents actifs est presque toujours
    dérivé du nom de fichier — « ssrn 4906546 », « 2026 08 03 Jacquier RoughBergomi
    turns grey ». La production, elle, sert le titre consolidé : ``_payload_row`` fait
    ``meta.get("title") or payload.get("title")``, et ``_fill_text`` passe par lui.
    Le banc montrait donc au générateur de moins bons titres que la production, et
    sous-estimait celle-ci sur les configurations qui passent par BM25.

    Aucune métrique de retrieval ne bouge : ``metrics`` ne lit que ``chunk_id`` et
    ``document_id``, et ``quant_rag._rerank`` ne lit que ``text``. Seul change ce que
    ``format_passages`` met sous les yeux du générateur.
    """
    out = []
    for row in rows:
        if row.get("text"):
            out.append(row)
            continue
        source = index.get(row["chunk_id"])
        if source:
            document = source["document_id"]
            fiche = index.metadata.get(document) or {}
            out.append({**row, "text": source["text"],
                        "title": fiche.get("title") or index.title_of(document),
                        "short_ref": fiche.get("short_ref"),
                        "section": source["section"], "page_start": source["page_start"]})
    return out


FILTER_KEYS = ("year_min", "year_max", "author")


def query_of(item: dict) -> str:
    """Texte soumis au retrieval. Une question datée porte sa contrainte en clair
    (« …, according to sources published in 2020 or later? ») pour le lecteur, le
    générateur et le juge ; l'appelant de production (Claude via MCP) la traduit en
    filtre et ne soumet que le besoin d'information : ``retrieval_query``."""
    return item.get("retrieval_query") or item["question"]


def filters_of(item: dict) -> dict:
    return {k: v for k, v in (item.get("filters") or {}).items() if k in FILTER_KEYS and v is not None}


def retrieve_item(item: dict, config: str, index, limit: int = POOL, filtered: bool = True) -> list[dict]:
    """``retrieve`` sur une question du banc, avec ses filtres bibliographiques
    (``filtered=False`` : même requête, sans filtre — le témoin des questions datées)."""
    return retrieve(query_of(item), config, index, limit=limit, filters=filters_of(item) if filtered else None)


def retrieve(question: str, config: str, index, limit: int = POOL, filters: dict | None = None) -> list[dict]:
    """Classement brut, sans déduplication ni plafonnement par document.

    Le filtrage cosmétique de ``quant_rag.search`` (plafond de 2 passages par
    document, suppression des quasi-doublons) est désactivé ici : il modifie le
    classement et rendrait les rangs incomparables entre configurations. Il est
    réappliqué au moment de construire le contexte, où il a du sens.

    ``filters`` (``year_min`` / ``year_max`` / ``author``) réduisent la portée avant
    la recherche, exactement comme en production : filtre Qdrant côté dense, même
    liste de documents côté BM25 (``quant_rag.document_scope``).

    ``limit > POOL``. La profondeur interrogée est ``max(limit, POOL)``, pas ``POOL``.
    Sans cela, ``limit`` n'était qu'une **troncature après** la requête : appelé avec
    ``limit=100``, ce code rendait 47 candidats en laissant croire qu'il en rendait 100
    (le filtre d'en-têtes de ``_select`` rogne le top-50 de Qdrant). Trouvé par la Phase 0
    du chantier « rappel du pool », qui avait précisément besoin d'une profondeur 100.

    Aucun appelant existant n'est déplacé : tous demandent ``limit=POOL`` ou moins, et
    ``max(limit, POOL) == POOL`` y est une identité. La ligne de base ne bouge pas — ce
    n'est pas une espérance, c'est le même appel avec les mêmes arguments.
    """
    if config not in CONFIGS:
        raise ValueError(f"configuration inconnue : {config}")
    filters = {k: v for k, v in (filters or {}).items() if k in FILTER_KEYS and v is not None}
    profondeur = max(limit, POOL)
    if config == "router":
        return quant_rag.search(question, limit=limit, pool=profondeur, mode="auto", auto_period=False,
                                dedupe=False, per_document=0, min_characters=0, log=False, **filters)

    dense = []
    if config != "bm25":
        # ``mode="dense"`` est obligatoire : depuis le routeur, le défaut de
        # ``quant_rag.search`` est ``"auto"``, et une ligne de base qui route n'est
        # plus une ligne de base (constaté : dense v2 0,476 -> 0,490, égal au routeur).
        # ``auto_period=False`` aussi : la requête du banc porte déjà ses filtres (``filters_of``),
        # la détection de clause est mesurée à part (eval_protocol.py), pas dans la référence.
        dense = quant_rag.search(question, limit=profondeur, pool=profondeur, mode="dense", rerank=False,
                                 auto_period=False, dedupe=False, per_document=0, min_characters=0,
                                 log=False, **filters)
    if config == "dense":
        return dense[:limit]

    scope = quant_rag.document_scope(None, **filters) if filters else None
    lexical = quant_rag._lexical(question, profondeur, scope)
    if config == "bm25":
        return _with_text(lexical, index)[:limit]

    fused = reciprocal_rank_fusion(dense, lexical)
    by_id = {row["chunk_id"]: row for row in dense}
    full = _with_text([by_id.get(row["chunk_id"]) or row for row in fused], index)
    if config == "rrf":
        return full[:limit]
    return quant_rag._rerank(question, full[:profondeur])[:limit]


def build_context(ranked: list[dict], passages: int = CONTEXT_PASSAGES,
                  per_document: int = 2) -> list[dict]:
    """Sélectionne les passages montrés au générateur — là, le plafond par
    document est légitime : on cherche de la couverture, pas un classement."""
    chosen: list[dict] = []
    seen: dict[str, int] = {}
    for row in ranked:
        if len(chosen) >= passages:
            break
        document = row.get("document_id")
        if per_document and seen.get(document, 0) >= per_document:
            continue
        seen[document] = seen.get(document, 0) + 1
        chosen.append(row)
    return chosen


ANSWER_SYSTEM = """You answer questions about quantitative finance using ONLY the numbered passages supplied to you. The passages were retrieved automatically and may be irrelevant, partially relevant, or wrong.

Rules, in order of priority:
1. Use only what the passages state. Never add facts from your own knowledge, however confident you are.
2. If the passages do not contain what is needed to answer, your entire reply must begin with the exact token INSUFFICIENT_EVIDENCE followed by one sentence naming what is missing. Do not then answer anyway.
3. Partial evidence still earns a partial answer. If the passages cover only part of what was asked, answer that part and say plainly which part they do not cover. Reserve INSUFFICIENT_EVIDENCE for when they support none of it.
4. The question will rarely use the passages' own vocabulary. Match on meaning, not on wording: a passage answers the question if it states the same fact under different terms.
5. Cite the passages you rely on inline as [1], [3].
6. At most 130 words. No preamble, no restating the question."""

#: Variante « v2 » du prompt de réponse — chantier E (abstention à tort). Mesuré sur la
#: ligne de base v3 (``abstention.py``) : 24 des 57 questions dont le chunk d'or était montré
#: se terminent par une abstention, et dans 6 cas la réponse *commence* par le jeton puis
#: donne quand même ce que les passages disent — la règle 2 (« entire reply must begin
#: with ») et la règle 3 (« partial evidence earns a partial answer ») se contredisent, et
#: le générateur les concilie en abstenant d'abord. v2 lève la contradiction : le jeton
#: n'est permis que lorsqu'il n'y a rien d'autre à écrire. Les règles 1, 4, 5, 6 sont
#: inchangées. Le prompt v1 reste le défaut : ``run_benchmark.py --answer-prompt v2``.
ANSWER_SYSTEM_V2 = """You answer questions about quantitative finance using ONLY the numbered passages supplied to you. The passages were retrieved automatically and may be irrelevant, partially relevant, or wrong.

Rules, in order of priority:
1. Use only what the passages state. Never add facts from your own knowledge, however confident you are.
2. The question will rarely use the passages' own vocabulary. Match on meaning, not on wording: a passage answers the question if it states the same fact under different terms, in a formula, or in a table row. Read every passage before deciding.
3. If at least one passage states something that bears on the question, answer with it, cite it, and then say plainly which part of the question the passages do not cover. Do not use the token INSUFFICIENT_EVIDENCE in that case.
4. Only if no passage states anything that bears on the question, reply with the exact token INSUFFICIENT_EVIDENCE followed by one sentence naming what is missing, and nothing else.
5. Cite the passages you rely on inline as [1], [3].
6. At most 130 words. No preamble, no restating the question."""

#: Le contrat de réponse « v3 » — chantier ``contrat-de-reponse``, 9 septembre 2026.
#:
#: **Il n'est pas écrit ici, et c'est le point.** v1 et v2 vivent dans ce module, donc dans le
#: banc seul, et le consommateur réel (``rag/agents/rag-scout.md``) en portait une paraphrase
#: française : deux textes, deux comportements, un seul chiffre publié. v3 est un fichier,
#: ``prompts/reponse-v3.txt``, lu par le banc **et** recopié mot pour mot dans l'agent, avec
#: ``tests/test_contrat_reponse.py`` qui refuse qu'ils divergent d'un caractère.
#:
#: Trois clauses le séparent de v2, chacune décidée par l'utilisateur le 9 septembre 2026 :
#:
#: - **la citation double** — marqueur ``[n]`` *et* extrait verbatim de 5 à 25 mots. Mesuré sur
#:   les 199 réponses v2 : le générateur produit déjà 16 extraits entre guillemets et **9 ne se
#:   retrouvent dans aucun passage servi** ; là où le passage écrit « Table 1: Various statistics
#:   on citation traversal », la réponse écrit « Table 1: Statistics of text units in the English
#:   Wikipedia ». Le plancher de 5 mots vient de ``score_citation.MOTS_MINIMUM`` : en deçà, une
#:   coïncidence de vocabulaire suffit à « retrouver » l'extrait dans n'importe quel passage ;
#: - **``NOT_IN_SOURCES: <grandeur>``** — le signal que v2 rendait impossible. Sa règle 3
#:   interdit le jeton d'abstention dès qu'un passage porte sur la question, donc toujours pour
#:   une négative voisine : le produit ne pouvait pas dire « sujet couvert, grandeur absente »,
#:   et 15 des 34 finissent en fabrication ;
#: - **LaTeX obligatoire**, et un plafond porté à 200 mots quand la réponse contient une formule
#:   ou une ligne de tableau — l'obligation de citer allonge mécaniquement les réponses, et sans
#:   marge le générateur arbitrerait entre citer et répondre.
#:
#: Ordre délibéré : la règle 3 (répondre) précède la 4 (citer). Un générateur qui doit trancher
#: répond. Et la clause 4 se referme sur « drop that claim and write what the passages do state
#: instead » — pas sur une invitation à s'abstenir.
CONTRAT_V3 = HERE / "prompts" / "reponse-v3.txt"
ANSWER_SYSTEM_V3 = CONTRAT_V3.read_text(encoding="utf-8").strip()

#: Le contrat « v4 » — **v2, plus les seules clauses de citation et de forme de v3**. Né du
#: verdict de v3, le 9 septembre 2026 au soir, et il vaut par ce qu'il isole.
#:
#: Mesuré sur les 199 questions, v3 contre v2 : les positives gagnent franchement
#: (`table_cell` si or servi 0,8387 → 0,9785, Δ apparié +0,130 IC95 [+0,050 ; +0,210] ;
#: affirmations sans citation 39,2 % → 12,3 % ; citations au mauvais passage 1 → 0) et les
#: **négatives régressent tout aussi franchement** (fabrications 15/34 → 26/34, Δ +0,294 IC95
#: [+0,118 ; +0,471] ; `negative_v3` 20/20 → 18/20 abstentions, le témoin calibré qui casse).
#:
#: Le mécanisme est mesuré, pas supposé : **l'abstention s'effondre** (17 → 7 sur les négatives
#: voisines), et le signal ``NOT_IN_SOURCES`` n'est pas protecteur mais **aggravant** — 9 des
#: 11 réponses qui l'émettent fabriquent (0,82 contre 0,70 sans lui), et **aucune ne s'abstient**.
#: Le générateur le lit comme une permission de répondre. Sommé de produire une preuve pour
#: chaque affirmation, il en trouve toujours une, quitte à répondre à une autre question :
#: ``"the optimal scaling constant is $a_{1,K} = 2$" [1]`` sur une question dont le sujet est
#: absent du corpus.
#:
#: v4 garde donc ce qui gagne et retire ce qui coûte. Ses clauses 1, 2 et 3 sont **celles de
#: v2, mot pour mot** ; ses clauses 4, 5 et 7 sont **celles de v3, mot pour mot** ; la clause 6
#: est la règle 4 de v2. Rien n'est réécrit — seule la renumérotation suit la suppression du
#: bloc d'abstention de v3. Deux contrastes en découlent, **à une variable chacun** :
#:
#:     v4 contre v2  →  l'effet des seules clauses de citation, de LaTeX et de plafond
#:     v4 contre v3  →  l'effet du seul bloc d'abstention (clause 3 réécrite + NOT_IN_SOURCES)
CONTRAT_V4 = HERE / "prompts" / "reponse-v4.txt"
ANSWER_SYSTEM_V4 = CONTRAT_V4.read_text(encoding="utf-8").strip()

#: Le jeton du signal « sujet couvert, grandeur absente ». Une **ligne**, pas un mot dans la
#: prose : lisible par regex, donc par le banc et par un client qui n'est pas un LLM. Il ne
#: contient pas ``INSUFFICIENT_EVIDENCE``, donc ``abstained`` ne le confond pas avec une
#: abstention — les deux signaux se mesurent séparément, ce qui est toute leur raison d'être.
NON_TROUVE = "NOT_IN_SOURCES"
_NON_TROUVE = re.compile(rf"^[^\S\n]*{NON_TROUVE}[^\S\n]*:[^\S\n]*(.+?)[^\S\n]*$", re.MULTILINE)

ANSWER_SYSTEM_V1 = ANSWER_SYSTEM
#: **Le placebo.** Le texte de v2, sous un autre nom — donc un bras qui ne diffère de la
#: référence par **rien**. Il existe parce que le chantier `representation` a mesuré, le
#: 9 septembre 2026, que ``mistral-small-latest`` n'est **pas reproductible** à température 0 :
#: re-générer 130 réponses sur des prompts identiques au bit près rend 51/130 identiques et
#: **9 bascules d'abstention**. Un Δ apparié entre une ligne de base servie par le cache et un
#: bras regénéré aujourd'hui contient donc le bruit du générateur, et rien ne le dit.
#:
#: Son empreinte est **celle de v2**, à dessein : ``divergences()`` ne voit aucun écart et la
#: comparaison est autorisée sans drapeau. C'est la définition même d'un placebo — si le banc
#: refusait de le comparer à la référence, il ne mesurerait pas ce qu'il prétend.
#:
#: Le tirage frais s'obtient en pointant ``llm.CACHE`` sur un répertoire neuf ; sans cela le
#: cache, indexé par le sha256 du payload, rendrait la réponse d'hier — et le placebo mesurerait
#: zéro par construction. Voir ``placebo_reponse.py``.
ANSWER_SYSTEM_V2_PLACEBO = ANSWER_SYSTEM_V2

ANSWER_PROMPTS = {"v1": ANSWER_SYSTEM_V1, "v2": ANSWER_SYSTEM_V2, "v3": ANSWER_SYSTEM_V3,
                  "v4": ANSWER_SYSTEM_V4, "v2placebo": ANSWER_SYSTEM_V2_PLACEBO}
#: Prompt de réponse par défaut du banc depuis le 3 septembre 2026 : v2. Mesuré sur les 150
#: questions v3 (results-e2e-v3-prompt-v2.json contre results-e2e-v3.json, même retrieval) :
#: abstention avec l'or montré 24/57 → 3/57 (−0,368 [−0,509 ; −0,228]), couverture 0,362 →
#: 0,569, négatives inchangées (abstention 20/20, fabrication 0/20) ; ancrage jugé 1,65 →
#: 1,56 — le prix d'une réponse partielle. ``run_benchmark.py --answer-prompt v1`` rejoue l'ancien.
#: **v4 depuis le 9 septembre 2026 au soir**, sur décision de l'utilisateur au vu de la mesure à
#: quatre bras (`docs/HANDOFF-REPONSE-FIN.md`). Ce qui a été démontré — IC disjoint de celui du
#: bras placebo — et qui justifie la bascule : les affirmations sans citation passent de 39,2 %
#: à 12,3 % sous les clauses de citation, et `table_cell` si or servi gagne +0,130.
#:
#: **Ce qui n'a PAS été démontré, et qu'il faut savoir avant de citer un chiffre de ce banc** :
#: tout le reste. Un bras placebo — le texte de v2 sous un autre nom, cache d'appels neuf — rend
#: sur `negative_voisine` un Δ de fabrications de **+0,1176, IC95 [+0,029 ; +0,235]**, à variable
#: nulle. 110 des 199 réponses seulement sont identiques d'un tirage à l'autre. La garde `a ≤ 2`
#: échoue sur ce placebo (`a = +3`). Aucune comparaison de prompt sur les familles négatives
#: n'est concluante sur ce banc tant qu'il n'est pas réparé.
#:
#: v4 plutôt que v3 : v3 avait de meilleurs chiffres sur les positives, mais il casse le **seul**
#: témoin insensible au bruit (`negative_v3`, 18/20 contre 20/20 pour v2, v4 et le placebo), et
#: v4 est le meilleur des quatre bras sur la seule métrique de citation que la mise en forme ne
#: peut pas fabriquer — les extraits verbatim **retrouvés** : 0,608, contre 0,528 (v3), 0,412
#: (v2) et 0,286 (placebo).
#:
#: Les lignes de base d'avant cette date portent v2 et ne se comparent aux suivantes qu'avec
#: `--drapeau` : `version_prompt` est un champ gouvernant de `banc_v4`.
#: ``run_benchmark.py --answer-prompt v2`` rejoue l'ancien.
DEFAULT_PROMPT = "v4"

#: En-tête de chaque passage montré au générateur et au juge. ``False`` = « titre — section »
#: (ligne de base). ``True`` ajoute la référence courte (auteurs, année) devant : sans elle,
#: une question datée (« … published in 2025 or later ») ne peut pas être honorée par le
#: générateur, qui s'abstient en le disant (d06, d14 sur la ligne de base v3). C'est ce que
#: la production affiche déjà (``source`` dans le MCP). Défaut depuis le 3 septembre 2026 ;
#: ``run_benchmark.py --no-passage-year`` rejoue l'en-tête d'origine.
PASSAGE_YEAR = True

SENTINEL = "INSUFFICIENT_EVIDENCE"
_SENTINEL = re.compile(rf"\b{SENTINEL}\b")


#: Fenêtre montrée par passage. **Ce n'est plus un nombre écrit ici** : c'est un alias de
#: ``contrat.PASSAGE_CHARACTERS``, la source unique de la fenêtre servie, et il n'y a plus
#: qu'un seul endroit à changer pour que le banc et la production restent d'accord.
#:
#: Deux ères précèdent celle-ci, et aucune des trois n'est comparable aux autres :
#:
#: - **1 600** jusqu'au 8 septembre 2026 — le banc sous-servait le générateur de 22,5 % par
#:   rapport à la production, sans que l'écart soit documenté nulle part (§21 duodecies :
#:   +0,123 de couverture, IC95 [+0,031 ; +0,215], rien qu'en réparant ce défaut) ;
#: - **2 500** le 8 septembre 2026 — la production tronquait à 2 500, le banc l'a reproduit ;
#: - **le passage entier**, plafonné à ``contrat.PASSAGE_CHARACTERS``, depuis le contrat de
#:   sortie. Le motif n'est **pas** la couverture — le fil ``characters`` a mesuré qu'élargir
#:   la fenêtre n'en fait pas gagner au-delà de la production (2 400 → 4 500 : +0,031, IC95
#:   [−0,061 ; +0,123], non significatif). Le motif est la **fidélité** : à 2 500, 178 des 760
#:   passages servis aux 155 questions étaient coupés en pleine formule, en pleine ligne de
#:   tableau ou en plein mot. Ils sont 0 désormais, pour +14,1 % de contexte.
#:
#: ``tests/test_characters.py`` refuse que le banc et la production divergent à nouveau.
CARACTERES_SERVIS = contrat.PASSAGE_CHARACTERS


def format_passages(context: list[dict], characters: int = CARACTERES_SERVIS,
                    year: bool | None = None) -> str:
    """La signature ne bouge pas : le chantier « instrument » l'appelle avec ses défauts.

    Seule la **coupe** change — ``contrat.couper_passage`` au lieu d'une troncature brute. Le
    générateur ne reçoit donc plus de formule ni de ligne de tableau à moitié.
    """
    year = PASSAGE_YEAR if year is None else year
    blocks = []
    for number, row in enumerate(context, 1):
        parts = [row.get("title"), row.get("section")]
        if year and row.get("short_ref"):
            parts = [row.get("short_ref")] + parts
        head = " — ".join(str(part) for part in parts if part)
        blocks.append(f"[{number}] {head[:220]}\n"
                      f"{contrat.couper_passage(row.get('text') or '', characters)}")
    return "\n\n".join(blocks)


def answer(question: str, context: list[dict], model: str = llm.GENERATOR, prompt: str = DEFAULT_PROMPT,
           characters: int | None = None) -> dict:
    """Réponse générée à partir des seuls passages fournis (``prompt`` : clé de ``ANSWER_PROMPTS``).

    ``characters`` est la fenêtre montrée par passage. ``None`` garde le défaut de
    ``format_passages`` — ``CARACTERES_SERVIS``, alias de ``contrat.PASSAGE_CHARACTERS``,
    donc **10 000** depuis l'intégration du 9 septembre 2026, ce que la production sert.

    Trois valeurs se sont succédé et l'écart n'a jamais été gratuit : **1 600** jusqu'au
    8 septembre, sans que la divergence soit documentée nulle part — le banc sous-servait le
    générateur de 22,5 % par rapport au produit (§21 duodecies) ; **2 500** après la réparation
    du banc ; **10 000** depuis que le contrat de sortie fait du plafond une garde et non une
    coupe. Le volume servi ne suit pas ce chiffre : mesuré sur le corpus, un passage fait
    1 734,7 c. en moyenne à un plafond de 6 000 et 1 734,9 c. sans plafond du tout. Ce que
    l'élargissement achète n'est pas du texte en plus partout, c'est de ne plus couper le seul
    tableau de 9 384 c. — le matériau que ce banc existe pour mesurer.
    """
    if not context:
        return {"answer": f"{SENTINEL} no passage was retrieved.", "abstained": True, "model": model}
    rendu = (format_passages(context) if characters is None
             else format_passages(context, characters=characters))
    text = llm.complete(
        [{"role": "system", "content": ANSWER_PROMPTS[prompt]},
         {"role": "user", "content": f"PASSAGES\n\n{rendu}\n\nQUESTION\n{question}"}],
        model=model, temperature=0.0, max_tokens=400,
    ).strip()
    return {"answer": text, "abstained": bool(_SENTINEL.search(text)), "model": model,
            "prompt": prompt, "characters": characters}


def abstained(text: str) -> bool:
    return bool(_SENTINEL.search(text or ""))


def non_trouve(text: str) -> list[str]:
    """Les grandeurs qu'une réponse déclare absentes des passages — ``NOT_IN_SOURCES: …``.

    Rend la liste des grandeurs nommées, vide si la réponse n'en signale aucune. C'est un
    **fait de texte**, pas une appréciation : le contrat v3 impose la ligne, et une expression
    régulière la lit, ici comme dans un client qui n'est pas un modèle de langue.

    La ligne est reconnue en tête de ligne, espaces et tabulations tolérées de part et d'autre —
    ``[^\\S\\n]`` et non ``\\s``, sans quoi ``re.MULTILINE`` laisserait un ``\\n`` être avalé et
    ferait passer pour une ligne ce qui n'en est pas une. Ce qui suit les deux-points est rendu
    tel quel, débarrassé de ses blancs de bord : le contrat ne dit rien de sa forme, seulement
    qu'il nomme la grandeur manquante.
    """
    return [correspondance.group(1) for correspondance in _NON_TROUVE.finditer(text or "")]


def empreinte_prompt(nom: str) -> str:
    """Empreinte courte du **texte effectivement envoyé** au générateur, pas du fichier.

    Le contrat v3 vit dans un fichier et le texte utilisé est ce fichier ``strip()``é : publier
    l'empreinte du fichier laisserait un saut de ligne final changer un chiffre gouvernant sans
    qu'aucun mot n'ait bougé. C'est le texte qui est la variable ; c'est lui qu'on hache.
    """
    return hashlib.sha256(ANSWER_PROMPTS[nom].encode("utf-8")).hexdigest()[:16]
