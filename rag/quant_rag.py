"""Retrieval layer of the Quant RAG corpus (Apple Silicon first, CPU fallback).

What it adds to a plain dense top-k:
  - no hardcoded root: every path is resolved from this file;
  - Qdrant runs **embedded by default** (``QdrantClient(path=...)``, exact brute-force
    search). ``QUANT_RAG_QDRANT_URL`` opens a server instead; the choice has a single
    home, ``rag/qdrant_backend.py``, and nothing else about retrieval changes;
  - device ``mps`` when available, ``cpu`` otherwise;
  - a **retrieval router** deciding, before any vector search, whether a query is
    served by the dense index alone or by the hybrid path (BM25 + RRF + rerank);
  - near-duplicate suppression and per-document capping on the result list;
  - **bibliographic metadata** (title, authors, publication year) consolidated by
    ``rag/metadata/extract_metadata.py`` and written into the payload by
    ``apply_metadata.py``: every result carries a citable ``source``
    ("López de Prado & Bailey (2014) — The Deflated Sharpe Ratio…"), and searches
    can be scoped with ``year_min`` / ``year_max`` / ``author``.

Routing (see ``rag/benchmark/calibrate_router.py``, ``results-router-calibration-v3.json``):

    mode="auto"    dense. Since the 150-question bench (v3, 3 September 2026) no rule
                   computable without an LLM beats dense on realistic questions: the
                   former rule "hybrid if >= 2 exact tokens" scored -0.019 nDCG@10
                   [-0.041, +0.002] on v3 (-0.072 on table questions, which are full of
                   tickers and years while the hybrid + rerank path is weak on tables),
                   for +0.032 [0.000, +0.088] on the 25 known-item questions of v1.
    mode="hybrid"  BM25 + RRF + rerank, on request — for a query made of remembered
                   exact identifiers ("Fukasawa SVI B2 B3"), the v1 situation.

AVERTISSEMENT (5 septembre 2026, corpus bb7bf33c37). Le chemin ``hybrid`` empile **deux
composants mesurés négatifs**, et il faut le savoir avant de le choisir :

  la fusion       RRF dense+BM25 sans reranker : −0,088 nDCG@10 pooled contre le dense seul
                  (IC95 [−0,124 ; −0,054]). Quatorze variantes de fusion mesurées, zéro en
                  GO ; la dégradation est monotone en le poids donné au lexical. Mécanisme :
                  sur 16 des 17 questions que la fusion sort du top 10, la cible est absente
                  du top-50 de BM25 — BM25 ne concurrence pas, il dilue.
                  Voir rag/benchmark/RAPPORT-FUSION-2026-09-05.md et docs/TODO.md §8.
  le reranker     ``bge-reranker-base`` sur un pool **dense** : −0,076 nDCG@10 pooled
                  (IC95 [−0,124 ; −0,027]), et −0,147 sur la famille ``single`` de v3.
                  Mesuré sur le corpus courant, 155 questions, docs/TODO.md §9.

Ce que cela ne dit pas : que le mode soit inutile. Sur v1 (identifiants mémorisés) il fait
passer les ratés de 1 à 0. Mais il n'a **aucune raison d'être choisi de bonne foi** en
dehors de ce cas précis, et les deux composants doivent être jugés séparément : un pool
dense reclassé par un cross-encoder plus récent (Qwen3-Reranker-0.6B) gagne +0,058 pooled
[+0,013 ; +0,103] et +0,103 de R@1, sans aucune fusion — mesuré sur ce corpus-ci. La bonne question n'est pas « hybride ou dense » mais
« qui de la fusion ou du reranker apportait quoi » — et la réponse mesurée est : la fusion
rien, le reranker tout, à condition de changer de reranker.

The exact-token detector (``exact_tokens``) is kept: every decision is logged with
its tokens, so the rule can be recalibrated on a larger bench without new code
(``EXACT_TOKENS_FOR_HYBRID = 2`` restores the former behaviour). A single exact
token never called for hybrid either: on the ten single-token questions of v3,
dense beats the hybrid path 4 times and loses once.

The reranker follows the mode: on in hybrid (rrf 0.457 -> rrf_rerank 0.547 on
routed queries of v1), off in dense (it regressed doc MRR 0.940 -> 0.896 on v1).
It can be forced either way with ``rerank=True/False``.
"""
from __future__ import annotations

import atexit
import hashlib
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STORAGE = ROOT / "qdrant_storage_local"
EXPORT = ROOT / "data" / "qdrant-export"
#: La collection servie. ``QUANT_RAG_COLLECTION`` la redirige — c'est ce qui permet de faire
#: passer le banc **entier** sur un corpus candidat sans faire traverser un nom de collection à
#: tout l'appareil de mesure : les deux bras empruntent alors exactement le même chemin de code,
#: ce qui est la condition du test apparié. Le nom effectif est publié par ``--status`` et par
#: l'en-tête de chaque fichier de résultat, donc un bras mesuré sur la mauvaise collection se
#: voit. À n'exporter que pour un processus de mesure, jamais dans un shell qui sert.
COLLECTION = os.environ.get("QUANT_RAG_COLLECTION", "quant_rag_ingested_all_qwen3_06b")
MODEL_ID = "Qwen/Qwen3-Embedding-0.6B"
#: Cross-encoder du chemin hybride. bge-reranker-base reste le défaut : sur le pool hybride
#: de v1 (identifiants mémorisés, le cas d'usage du mode hybride) il vaut 0,935 contre 0,917
#: pour bge-reranker-v2-m3, qui ne gagne que sur v3 (+0,110*), là où l'hybride n'est pas
#: recommandé — et coûte 5,4 s par requête au lieu de 1,6 (benchmark/README.md §12-D).
#: QUANT_RAG_RERANKER=BAAI/bge-reranker-v2-m3 l'essaie sans toucher au code.
RERANKER_ID = os.environ.get("QUANT_RAG_RERANKER", "BAAI/bge-reranker-base")
#: Le modèle du mode ``reclassement="selectif"`` — **distinct** de ``RERANKER_ID`` ci-dessus,
#: et il faut que les deux le restent. ``bge-reranker-base`` porte le chemin ``hybrid`` et a
#: été mesuré **nuisible** sur un pool dense (−0,076 nDCG@10, IC95 [−0,124 ; −0,027]) ;
#: ``Qwen3-Reranker-0.6B`` y gagne (+0,031 poolé à budget 10, corpus ``5530cba145``). Les
#: confondre reviendrait à servir le modèle qui perd. Voir ``rag/reranking.py``.
RECLASSEMENT_ID = os.environ.get("QUANT_RAG_RECLASSEMENT", "Qwen/Qwen3-Reranker-0.6B")
QUERY_INSTRUCTION = "Given a quantitative-finance research question, retrieve the passage that best answers it."
MAX_LENGTH = 1024

LEXICAL_DIR = ROOT / "data" / "lexical"
#: Quel titre le texte lexical de BM25 porte en tête (``retrieval.lexical.lexical_text``).
#:
#: ``"export"``    le titre de ``data/qdrant-export/payloads.jsonl``, dérivé du nom de
#:                 fichier — ``EngleGranger1987``, ``2026 08 03 Jacquier RoughBergomi turns
#:                 grey``. C'est ce que l'index d'origine portait, pour les 256 documents
#:                 venus de l'export ; les 63 importés portaient déjà le vrai titre.
#: ``"registry"``  le titre consolidé de ``metadata/documents-metadata-v1.json``. Défaut
#:                 depuis le 5 septembre 2026. Mesuré sur les deux bancs gelés
#:                 (``benchmark/eval_fusion.py``, index de l'ombre) : BM25 seul passe de
#:                 0,277 à 0,312 de nDCG@10 pooled, écart apparié +0,036 [+0,017 ; +0,058],
#:                 et de 0,711 à 0,772 sur v1 — le banc known-item, qui est le cas d'usage
#:                 du mode ``hybrid``. Le gain va surtout aux tableaux (+0,052) et aux
#:                 questions simples (+0,037) ; la famille ``exact`` ne bouge pas (+0,002),
#:                 ces questions citant le corps du texte et non le titre.
#:
#: Cette valeur **n'est pas dans la signature** — c'est une décision de code, pas un
#: fichier. Elle est donc inscrite au manifeste et vérifiée au chargement : sans quoi deux
#: index de contenus différents porteraient exactement le même nom.
BM25_TITLE_SOURCE = os.environ.get("QUANT_RAG_BM25_TITLES", "registry")
ROWS_PATH = ROOT / "data" / "embeddings" / "ingested-all-qwen3-06b" / "rows.jsonl"
DECISION_LOG = ROOT / "rag" / "logs" / "router-decisions.jsonl"
METADATA_PATH = ROOT / "rag" / "metadata" / "documents-metadata-v1.json"
#: Document-level fields copied into every chunk payload (see apply_metadata.py / build_index.py).
PAYLOAD_FIELDS = ("title", "authors", "publication_year", "first_year", "short_ref", "venue", "doc_type", "filename")

MODES = ("auto", "dense", "hybrid")
#: Number of exact tokens from which ``mode="auto"`` switches to the hybrid path;
#: ``None`` = never (auto is dense). Was 2 from 2 to 3 September 2026; set back to
#: 2 to restore the routing rule — the detector and the decision log are unchanged.
EXACT_TOKENS_FOR_HYBRID: int | None = None
#: Candidate pool before filtering/reranking. 50 is what the benchmarks used.
POOL = 50
#: Chunks that are only a heading, or too short to carry information, pollute the top-k.
MIN_CHARACTERS = 250

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
sys.path.insert(0, str(ROOT / "src"))  # reuse upstream BM25 and RRF, unchanged
sys.path.insert(0, str(ROOT / "rag"))
import contrat  # noqa: E402  — fenêtre servie, coupe sûre, drapeaux de qualité, traçabilité
import corpus_overlay  # noqa: E402  — documents retirés (duplicates-v1.json), tableaux convertis

#: BM25 index of the *current* corpus state. The overlay signature is part of the
#: name (``bm25-dedup-tables-v1-<sig>.json``): an index built on another state
#: is never loaded by mistake, it simply is not found and gets rebuilt. The
#: original 19 443-chunk reference keeps its own name, ``bm25-ingested-all-v1.json``.
#: Calculés à chaque accès, jamais figés à l'import : un processus qui modifie un overlay ou
#: le registre en cours d'exécution (``convert_tables.py``, ``apply_duplicates.py``, le futur
#: ``apply_delivery.py``) doit écrire sous le *nouveau* nom. Une constante de module aurait
#: gardé l'ancien — le contenu juste sous un nom qui ment, exactement ce que ce nommage doit
#: empêcher. ``quant_rag.BM25_PATH`` reste lisible de l'extérieur via ``__getattr__``.
def bm25_path() -> Path:
    return LEXICAL_DIR / f"{corpus_overlay.bm25_name()}.json"


def index_lexicaux() -> dict:
    """Inventaire de ``data/lexical/`` : qui est vivant, qui est mort, qui n'a pas de signature.

    Pourquoi cet inventaire existe
    -------------------------------
    ``rebuild_bm25()`` écrit un fichier nommé par la signature courante et **n'efface jamais
    le précédent**. Chaque état par lequel le corpus est passé a donc laissé son index —
    ≈ 50 Mo pièce — et rien ne les réclame. Relevé le 11 septembre 2026 : **170 fichiers pour
    un seul utile**.

    Trois catégories, et la distinction compte :

    - **vivant** : la signature du corpus servi aujourd'hui. On n'y touche jamais ;
    - **gelé** : la signature que ``rag/gel-corpus.json`` protège. Elle est en général égale
      à la vivante — mais quand elle ne l'est pas, c'est qu'une mesure est en cours sur elle,
      et supprimer son index la casserait. On n'y touche pas non plus ;
    - **mort** : une signature qui n'est ni l'une ni l'autre ;
    - **sans signature** : ``bm25_index.json``, ``hybrid-bm25-v1.json`` — des formats anciens
      dont le nom ne dit pas à quel corpus ils se rapportent. On ne peut donc pas prouver
      qu'ils sont morts : ils sont **listés à part et jamais supprimés automatiquement**.
    """
    import re

    vivante = corpus_overlay.signature()
    gelee = None
    declaration = ROOT / "rag" / "gel-corpus.json"
    if declaration.exists():
        gelee = json.loads(declaration.read_text(encoding="utf-8")).get("signature")
    proteges = {s for s in (vivante, gelee) if s}

    # ``bm25-<étiquette>-<signature de 10 hexa>`` puis ``.json`` ou ``.manifest.json``
    motif = re.compile(r"^bm25-.+-([0-9a-f]{10})(?:\.manifest)?\.json$")
    vivants, morts, sans_signature = [], [], []
    for chemin in sorted(LEXICAL_DIR.glob("*.json")) if LEXICAL_DIR.is_dir() else []:
        trouve = motif.match(chemin.name)
        fiche = {"nom": chemin.name, "octets": chemin.stat().st_size,
                 "signature": trouve.group(1) if trouve else None}
        if trouve is None:
            sans_signature.append(fiche)
        elif trouve.group(1) in proteges:
            vivants.append(fiche)
        else:
            morts.append(fiche)
    return {"signature_vivante": vivante, "signature_gelee": gelee,
            "vivants": vivants, "morts": morts, "sans_signature": sans_signature,
            "octets_morts": sum(f["octets"] for f in morts),
            "octets_sans_signature": sum(f["octets"] for f in sans_signature)}


def purger_index_morts(vraiment: bool = False) -> dict:
    """Supprime les index et manifestes des signatures mortes. **Rien sans ``vraiment``.**

    Ne touche jamais à la signature vivante, jamais à la signature gelée, et jamais aux
    formats sans signature — ceux-là sont listés, à charge d'un humain de décider.
    """
    inventaire = index_lexicaux()
    if vraiment:
        for fiche in inventaire["morts"]:
            (LEXICAL_DIR / fiche["nom"]).unlink(missing_ok=True)
    inventaire["supprimes"] = [f["nom"] for f in inventaire["morts"]] if vraiment else []
    inventaire["applique"] = vraiment
    return inventaire


def bm25_manifest_path() -> Path:
    return LEXICAL_DIR / f"{corpus_overlay.bm25_name()}.manifest.json"


def __getattr__(name: str):
    """``BM25_PATH`` / ``BM25_MANIFEST`` restent des attributs de module, mais recalculés."""
    if name == "BM25_PATH":
        return bm25_path()
    if name == "BM25_MANIFEST":
        return bm25_manifest_path()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# --------------------------------------------------------------------------- models & stores

def device() -> str:
    import torch

    return "mps" if torch.backends.mps.is_available() else "cpu"


@lru_cache(maxsize=1)
def embedder():
    from sentence_transformers import SentenceTransformer

    dev = device()
    model = SentenceTransformer(MODEL_ID, device=dev)
    if dev == "mps":
        model.half()
    model.max_seq_length = MAX_LENGTH
    return model


import qdrant_backend  # noqa: E402  — un seul domicile pour le choix du backend


@lru_cache(maxsize=1)
def client():
    #: Le choix du backend a UN SEUL domicile (rag/qdrant_backend.py). Sans
    #: QUANT_RAG_QDRANT_URL, ceci ouvre exactement ce qui était ouvert hier : path=STORAGE.
    #: ``collection`` est nommée pour armer la garde de dérive en mode serveur : c'est le
    #: chemin servi, donc celui qui doit refuser plutôt que répondre sur une copie périmée.
    handle = qdrant_backend.ouvrir(
        STORAGE, collection=COLLECTION,
        message_si_absent=f"Index absent. Lance d'abord: python {ROOT / 'rag' / 'build_index.py'}")
    atexit.register(handle.close)  # sinon __del__ se plaint pendant l'arrêt de l'interpréteur
    return handle


def rebuild_bm25():
    """(Re)build the BM25 file from the live collection — the single source of truth.

    Called after any change to the indexed text (documents removed, tables converted),
    so that the lexical path and the benchmark see the same corpus as the dense path.
    """
    from retrieval.lexical import BM25Index

    # Le titre entre dans le texte lexical (``lexical_text``). Voir ``BM25_TITLE_SOURCE`` :
    # l'index d'origine portait le titre *de l'export*, dérivé du nom de fichier ; le défaut
    # est désormais le titre consolidé, qui vaut +0,036 de nDCG@10 pooled à BM25 seul.
    original_title = {}
    if BM25_TITLE_SOURCE == "export":
        for line in (EXPORT / "payloads.jsonl").open(encoding="utf-8"):
            payload = json.loads(line)["payload"]
            original_title.setdefault(payload["document_id"], payload.get("title"))
    elif BM25_TITLE_SOURCE == "registry":
        original_title = {doc_id: record.get("title") for doc_id, record in document_metadata().items()}
    else:
        raise ValueError(f"BM25_TITLE_SOURCE inconnu : {BM25_TITLE_SOURCE!r} (attendu: export, registry)")
    rows, offset = [], None
    while True:
        points, offset = client().scroll(collection_name=COLLECTION, limit=4096, offset=offset,
                                         with_payload=["chunk_id", "document_id", "text", "title", "title_path", "content_type"])
        for point in points:
            p = point.payload
            rows.append(({"document_id": p["document_id"], "title": original_title.get(p["document_id"]) or p.get("title")},
                         {"chunk_id": p["chunk_id"], "document_id": p["document_id"], "text": p.get("text", ""),
                          "title_path": p.get("title_path"), "content_type": p.get("content_type")}))
        if offset is None:
            break
    index = BM25Index.from_corpus(rows)
    bm25_path().parent.mkdir(parents=True, exist_ok=True)
    index.dump(bm25_path())
    bm25_manifest_path().write_text(json.dumps({
        "index": bm25_path().name,
        # Empreinte du fichier tel qu'il vient d'être écrit. C'est elle que la garde
        # confronte au disque : un compte de points ne distingue pas deux index de
        # contenus différents bâtis sur la même collection.
        "index_sha256": hashlib.sha256(bm25_path().read_bytes()).hexdigest(),
        "built_from": {"collection": COLLECTION, "points": len(rows),
                       "documents": len({document["document_id"] for document, _ in rows})},
        "corpus": corpus_overlay.describe(),
        "corpus_signature": corpus_overlay.signature(),
        "titles_digest": corpus_overlay.titles_digest(),
        "records": index.doc_count, "terms": len(index.document_frequency),
        "k1": index.k1, "b": index.b,
        "title_source": BM25_TITLE_SOURCE,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }, indent=1, ensure_ascii=False), encoding="utf-8")
    bm25.cache_clear()
    return index


@lru_cache(maxsize=1)
def bm25():
    """BM25 over the production chunks (upstream ``src/retrieval/lexical.py``), built from the index.

    Le nom du fichier porte la signature de l'état du corpus, donc un index d'un autre
    état n'est pas trouvé. Ce n'est pas suffisant : le nom peut être juste et le contenu
    périmé (index rebâti à la main, import interrompu entre l'écriture des points et la
    reconstruction lexicale). On confronte donc le manifeste à la collection avant de
    servir l'index — le chemin lexical a déjà menti une fois, sous un nom correct.
    """
    from retrieval.lexical import BM25Index

    if bm25_path().exists():
        if _bm25_matches_collection():
            return BM25Index.load(bm25_path())
        print(f"ATTENTION : {bm25_path().name} ne correspond plus à la collection — reconstruction",
              file=sys.stderr)
    return rebuild_bm25()


def _bm25_matches_collection() -> bool:
    """L'index sur le disque est-il bien celui de cet état du corpus — **contenu compris** ?

    Le nom porte la signature, qui couvre depuis le 5 septembre 2026 les titres consolidés :
    un index bâti sur d'autres titres n'est plus trouvé. Restaient deux trous, tous deux
    ouverts par la même garde d'origine, qui ne comparait qu'un **compte de points** :

      - la signature dit *quels* titres, pas *quelle source* de titres. Un index bâti sur le
        titre d'export et un autre sur le titre consolidé portent le même nom, ont le même
        nombre de points, et le même contenu de collection derrière eux. Les deux passaient ;
      - un index réécrit à la main, ou à moitié écrit, garde le bon compte de points.

    On confronte donc le manifeste au disque et à l'état courant, du moins cher au plus cher.
    Le sha256 coûte ~0,6 s sur 51 Mo, une fois par processus (``bm25()`` est mémorisé) —
    le prix d'un démarrage contre la certitude de ne pas servir un index qui ment.
    """
    if not bm25_manifest_path().exists():
        return False
    try:
        manifest = json.loads(bm25_manifest_path().read_text(encoding="utf-8"))
        for champ, attendu in (("title_source", BM25_TITLE_SOURCE),
                               ("titles_digest", corpus_overlay.titles_digest()),
                               ("corpus_signature", corpus_overlay.signature())):
            if manifest.get(champ) != attendu:
                print(f"ATTENTION : {bm25_path().name} annonce {champ}={manifest.get(champ)!r}, "
                      f"attendu {attendu!r}", file=sys.stderr)
                return False
        empreinte = manifest.get("index_sha256")
        if not empreinte or empreinte != hashlib.sha256(bm25_path().read_bytes()).hexdigest():
            print(f"ATTENTION : {bm25_path().name} ne correspond pas à son manifeste "
                  f"(sha256)", file=sys.stderr)
            return False
        expected = manifest.get("built_from", {}).get("points")
        if expected is None:
            return False
        return int(expected) == client().count(COLLECTION, exact=True).count
    except Exception:
        return False


@lru_cache(maxsize=1)
def document_metadata() -> dict[str, dict]:
    """document_id -> consolidated bibliographic record (empty dict if the file is absent)."""
    if not METADATA_PATH.exists():
        return {}
    data = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    removed = corpus_overlay.removed_documents()
    return {row["document_id"]: row for row in data["documents"] if row["document_id"] not in removed}


def payload_fields(record: dict) -> dict:
    return {key: record.get(key) for key in PAYLOAD_FIELDS}


def _fold(text: str) -> str:
    import unicodedata

    return "".join(ch for ch in unicodedata.normalize("NFKD", text.lower()) if not unicodedata.combining(ch))


def document_scope(document_id: str | None = None, year_min: int | None = None, year_max: int | None = None,
                   author: str | None = None) -> list[str] | None:
    """Document ids satisfying the bibliographic constraints; ``None`` when unconstrained.

    A year bound excludes documents whose year is unknown (29 of 258 after v1): a
    user asking for "since 2020" does not want undated material mixed in silently.
    """
    if document_id is None and year_min is None and year_max is None and not author:
        return None
    needle = _fold(author) if author else None
    scope = []
    for doc_id, rec in document_metadata().items():
        if document_id and doc_id != document_id:
            continue
        year = rec.get("publication_year")
        if (year_min is not None or year_max is not None) and year is None:
            continue
        if year_min is not None and year < year_min:
            continue
        if year_max is not None and year > year_max:
            continue
        if needle and not any(needle in _fold(name) for name in rec.get("authors") or []):
            continue
        scope.append(doc_id)
    if document_id and not document_metadata():
        return [document_id]
    return scope


def encode_query(text: str):
    import numpy as np

    vector = embedder().encode(
        [text],
        prompt=f"Instruct: {QUERY_INSTRUCTION}\nQuery:",
        normalize_embeddings=True,
        show_progress_bar=False,
    )[0]
    return np.asarray(vector, dtype="float32").tolist()


# --------------------------------------------------------------------------- period clause

#: Words that make a year a *publication* constraint rather than a data period ("from
#: 1963 through 2012" is content; "sources published in 2022 or earlier" is scope).
_PUBLICATION = r"(?:publi(?:shed|cation|és?|ées?)|sources?|papers?|literature|research|works?|studies|preprints?|articles?|documents?|travaux|références?|littérature|études?)"
_YEAR = r"(?:19|20)\d\d"
_PERIOD_PATTERNS = [
    # "... according to sources published in 2022 or earlier" / "in 2025 or later"
    re.compile(rf"(?P<clause>,?\s*(?:according to|based on|using|from|in|d'après|selon)?\s*(?:the\s+|les\s+|des\s+)?"
               rf"{_PUBLICATION}\s+(?:{_PUBLICATION}\s+)?(?:in|de|en)\s+(?P<y>{_YEAR})\s+(?:or|ou)\s+(?P<dir>earlier|later|before|after|avant|après|plus tôt|plus tard))", re.I),
    # "... published before/after/since/until 2010", "sources from 2020 onwards", "papers prior to 2005"
    re.compile(rf"(?P<clause>,?\s*(?:according to|based on|using|from|in|d'après|selon)?\s*(?:the\s+|les\s+|des\s+)?"
               rf"{_PUBLICATION}\s+(?:{_PUBLICATION}\s+)?(?P<dir>before|after|since|until|up to|prior to|from|as of|avant|après|depuis|jusqu'à|jusqu'en|à partir de|d'avant|postérieur(?:e|s|es)? à|antérieur(?:e|s|es)? à)\s+(?P<y>{_YEAR})(?:\s+(?:onwards?|on|inclus))?)", re.I),
    # "... published between 2010 and 2020", "sources from 2015 to 2020", "littérature de 2010 à 2020"
    re.compile(rf"(?P<clause>,?\s*(?:according to|based on|using|from|in|d'après|selon)?\s*(?:the\s+|les\s+|des\s+)?"
               rf"{_PUBLICATION}\s+(?:{_PUBLICATION}\s+)?(?:between|from|entre|de)\s+(?P<y>{_YEAR})\s+(?:and|to|et|à)\s+(?P<y2>{_YEAR}))", re.I),
    # "pre-2010 literature", "post-2015 papers"
    re.compile(rf"(?P<clause>\b(?P<dir>pre|post)-(?P<y>{_YEAR})\s+{_PUBLICATION})", re.I),
]
_LOWER = {"earlier", "before", "until", "up to", "prior to", "avant", "plus tôt", "jusqu'à", "jusqu'en", "d'avant", "pre"}


def period_bounds(query: str) -> dict | None:
    """An explicit *publication* period in the question, as ``year_min`` / ``year_max``.

    Deliberately narrow: a year must be attached to a publication word ("sources
    published in 2022 or earlier", "papers before 2010", "literature between 2015 and
    2020", "pre-2010 research"). Years that are content — "from 1963 through 2012", "in
    the 2003 CBOE Annual Report", "Heston (2000)" — never match. Measured on the two
    benches (``benchmark/eval_protocol.py``): 15/15 dated questions of v3 recovered with
    the gold bounds, 0 false positive on the 135 other v3 questions and the 25 of v1.

    Returns ``{"year_min", "year_max", "clause", "query"}`` (``query`` = the question
    without the clause, what the embedding should see), or ``None``.
    """
    for pattern in _PERIOD_PATTERNS:
        match = pattern.search(query)
        if not match:
            continue
        year = int(match.group("y"))
        groups = match.groupdict()
        if groups.get("y2"):
            low, high = sorted((year, int(groups["y2"])))
            bounds = {"year_min": low, "year_max": high}
        else:
            direction = (groups.get("dir") or "").lower()
            if direction in {"since", "from", "as of", "depuis", "à partir de"} or direction.startswith(("post", "postérieur", "later", "after", "après", "plus tard")):
                bounds = {"year_min": year, "year_max": None}
            elif direction in _LOWER or direction.startswith(("antérieur",)):
                bounds = {"year_min": None, "year_max": year}
            else:
                continue
        clause = match.group("clause")
        stripped = (query[:match.start("clause")] + query[match.end("clause"):]).strip()
        stripped = re.sub(r"\s+([?.!,;:])", r"\1", re.sub(r"\s{2,}", " ", stripped)).strip(" ,")
        if stripped and not stripped.endswith(("?", ".", "!")) and query.rstrip().endswith("?"):
            stripped += "?"
        return {**bounds, "clause": clause.strip(" ,"), "query": stripped or query}
    return None


# --------------------------------------------------------------------------- router

#: Capitalised words that are not proper nouns in a question.
_CAP_STOP = {"I", "How", "What", "Why", "When", "Where", "Which", "Who", "Whose", "Under", "Does", "Do",
             "Is", "Are", "Can", "Could", "Should", "Would", "If", "In", "On", "For", "The", "A", "An",
             "Need", "Explain", "Prove", "Compare", "Describe", "Give", "Show", "Derive", "Using", "With"}
_WORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9\-'./]*")


def exact_tokens(question: str) -> dict:
    """Tokens the embedding dilutes and BM25 matches literally.

    - acronyms: >= 2 capitals, digits/hyphens allowed (SVI, HJB, IBM, MATLAB, B2)
    - proper nouns: capitalised, not sentence-initial, not in the stoplist (Fukasawa, Almgren-Chriss)
    - numeric: any token containing a digit (250-day, 2007, 0.0035)
    """
    acronyms, proper, numeric = [], [], []
    for position, word in enumerate(_WORD.findall(question)):
        core = re.sub(r"[^A-Za-z0-9]", "", word)
        if any(ch.isdigit() for ch in core):
            numeric.append(word)
        elif len(re.sub(r"[^A-Z]", "", core)) >= 2 and core.upper() == core:
            acronyms.append(word)
        elif position > 0 and core[:1].isupper() and word.rstrip("'s") not in _CAP_STOP and not core.isupper():
            proper.append(word)
    return {"acronyms": acronyms, "proper": proper, "numeric": numeric,
            "n_exact": len(acronyms) + len(proper) + len(numeric)}


def route(query: str, mode: str = "auto") -> dict:
    """Decide dense vs hybrid *before* touching the index. Pure function of the query text."""
    if mode not in MODES:
        raise ValueError(f"mode inconnu: {mode!r} (attendu: {', '.join(MODES)})")
    tokens = exact_tokens(query)
    found = tokens["acronyms"] + tokens["proper"] + tokens["numeric"]
    if mode != "auto":
        chosen, reason = mode, f"forcé par l'appelant (mode={mode})"
    elif EXACT_TOKENS_FOR_HYBRID is None:
        chosen = "dense"
        reason = ("dense par défaut (calibration v3 : aucune règle ne bat le dense)"
                  + (f" — {tokens['n_exact']} jeton(s) exact(s) : {', '.join(found[:5])}" if found else ""))
    elif tokens["n_exact"] >= EXACT_TOKENS_FOR_HYBRID:
        chosen = "hybrid"
        reason = f"{tokens['n_exact']} jetons exacts ≥ {EXACT_TOKENS_FOR_HYBRID} : {', '.join(found[:5])}"
    else:
        chosen = "dense"
        reason = (f"{tokens['n_exact']} jeton exact < {EXACT_TOKENS_FOR_HYBRID}"
                  + (f" ({', '.join(found)})" if found else "") + " — requête en langue naturelle")
    return {"mode": chosen, "requested": mode, "reason": reason, "exact_tokens": tokens}


def _log_decision(query: str, decision: dict, results: list[dict] | None = None,
                  limit: int | None = None, pool: list[str] | None = None,
                  tracabilite: dict | None = None) -> None:
    """One JSON line per query, for audit. Never allowed to break a search.

    Four fields exist for the *shadow* protocol and for nothing else — ``signature``,
    ``served``, ``limit`` and ``pool``. Without them a deferred replay cannot prove it is
    replaying the same corpus, returning the same passages, and asking for the same number of
    them; and a shadow that cannot prove that measures nothing.

    ``pool`` a été ajouté le 7 septembre 2026, et il manquait : le journal disait ce qui
    avait été **servi** sans dire parmi **quoi**. Un shadow de reclassement ne peut donc pas
    rejouer hors ligne ce qu'une autre règle d'ordonnancement aurait servi — il lui faudrait
    rouvrir Qdrant et refaire tourner l'embedder. Les dix premiers ``chunk_id`` suffisent :
    c'est le budget que le mode sélectif reclasse.

    They are written **into the log line only**: ``decision`` is returned to the caller as
    ``routing`` and is left untouched, so no caller sees a different value. Adding a call
    site was avoided on purpose — this function already exists, already swallows its errors,
    and is already on the path of every query.
    """
    try:
        DECISION_LOG.parent.mkdir(parents=True, exist_ok=True)
        row = {"timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               **(tracabilite or {}),
               "query": query[:300], **{k: v for k, v in decision.items() if k != "exact_tokens"},
               # ``signature`` garde son nom historique — le protocole shadow le lit — mais
               # elle est **reprise** de l'en-tête de traçabilité quand il existe, jamais
               # recalculée. ``corpus_overlay.signature()`` relit et rehache quatre artefacts à
               # chaque appel, par choix documenté : 21 ms. La payer deux fois par requête
               # ajoutait 21 ms à un chemin qui en met 74, pour écrire deux fois le même mot.
               "signature": (tracabilite or {}).get("corpus_signature") or corpus_overlay.signature(),
               "served": [r.get("chunk_id") for r in results] if results is not None else None,
               "pool": pool,
               "limit": limit,
               "exact_tokens": decision["exact_tokens"]}
        with DECISION_LOG.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    except OSError:
        pass


# --------------------------------------------------------------------------- retrieval stages

#: Les pages stockées dans le payload sont **0-basées** — elles reprennent le ``page_idx``
#: des blocs du parse, dont le bloc de titre porte 0. Vérifié le 7 septembre 2026 sur
#: ``doc-2b74a994fcfc1c9c`` : « 1 Introduction » est en ``page_idx`` 1, « 6 Conclusion » en
#: 34, et les chunks correspondants déclarent ``page_start`` 1 et 34. Une citation qui
#: reprend ce nombre envoie donc le lecteur **une page trop tôt**.
#:
#: On ne réécrit pas ``page_start`` : sa sémantique historique est « page de début du chunk,
#: telle que stockée », et des artefacts la lisent ainsi. On ajoute un champ distinct pour
#: l'affichage, et c'est lui — et lui seul — qui doit apparaître dans une citation.
DECALAGE_PAGE_UTILISATEUR = 1

#: Le séparateur entre deux passages joints, partout où le service en concatène plusieurs.
#: Nommé parce qu'un calcul de couverture de pages en dépend (``get_passage``).
SEPARATEUR_PASSAGES = "\n\n"


def pages_utilisateur(page_start, page_end) -> list[int] | None:
    """Les pages **couvertes**, telles qu'un lecteur les compte, ou ``None`` si inconnues.

    La liste est explicite, et non un couple de bornes : c'est ce qui permet à un appelant
    qui joint plusieurs passages d'en faire l'union sans affirmer les pages intermédiaires.
    """
    if page_start is None:
        return None
    debut = int(page_start) + DECALAGE_PAGE_UTILISATEUR
    fin = int(page_end) + DECALAGE_PAGE_UTILISATEUR if page_end is not None else debut
    return list(range(debut, max(debut, fin) + 1))


def citation_pages(pages) -> str:
    """« p. 35 » ou « p. 35–36 ». Une plage quand le passage en couvre plusieurs.

    **Le système n'a pas de pointeur de phrase au moment de la réponse.** Annoncer une page
    exacte pour un passage qui en couvre deux serait une précision qu'il ne possède pas ;
    la plage est la seule sortie honnête. Le pointeur de phrase est un chantier séparé, et
    tant qu'il n'existe pas cette fonction ne doit pas prétendre mieux.

    Symétriquement, une plage ne doit pas prétendre **plus** que la couverture réelle. La
    fratrie que ``get_passage`` joint est ordonnée par page et non par contiguïté : mesuré
    le 7 septembre 2026, ``chunk-5ea0681947ca79cb`` rend les pages 27, 28 et 31. Écrire
    « p. 27-31 » affirmerait deux pages où le lecteur ne trouvera rien. Les suites
    consécutives sont donc groupées, et les trous restent visibles — « p. 27-28, 31 ».
    """
    if not pages:
        return "page inconnue"
    suites: list[list[int]] = []
    for page in sorted(set(pages)):
        if suites and page == suites[-1][-1] + 1:
            suites[-1].append(page)
        else:
            suites.append([page])
    return "p. " + ", ".join(str(s[0]) if len(s) == 1 else f"{s[0]}\u2013{s[-1]}"
                             for s in suites)


def joindre_passages(payloads: list[dict], max_characters: int) -> tuple[str, list[dict]]:
    """Le texte joint, **et** les seuls passages qui y apparaissent encore après troncature.

    Séparé de ``get_passage`` pour être vérifiable sans Qdrant : c'est ce calcul qui décide
    quelles pages la citation doit couvrir, et une citation qui nomme la page d'un passage
    coupé par ``max_characters`` est aussi fausse qu'une citation qui en oublie un.

    **La cinquième troncature de la surface servie était ici.** Elle ne s'écrit pas
    ``row['text'][:6000]`` mais ``[:max_characters]``, et le balayage par expression régulière
    de ``tests/test_characters.py`` — qui garde les quatre autres — ne la voyait pas. Elle
    coupait 30,2 % des fenêtres de ``get_passage`` (médiane 4 520 c., max 13 982 pour un défaut
    à 6 000), en plein mot et en pleine formule comme les autres. Elle passe désormais par
    ``contrat.couper``.
    """
    textes = [p.get("text", "") for p in payloads]
    texte, _ = contrat.couper(SEPARATEUR_PASSAGES.join(textes), max_characters)
    montres, position = [], 0
    for payload, morceau in zip(payloads, textes):
        # Le passage *i* commence à ``position`` dans le texte joint : il est à l'écran si
        # et seulement si cette position y tombe encore.
        if position < len(texte):
            montres.append(payload)
        position += len(morceau) + len(SEPARATEUR_PASSAGES)
    return texte, montres


def _payload_row(payload: dict, score: float, kind: str) -> dict:
    meta = document_metadata().get(payload.get("document_id"), {})
    title = meta.get("title") or payload.get("title")
    short_ref = payload.get("short_ref") or meta.get("short_ref")
    return {
        "chunk_id": payload.get("chunk_id"),
        "document_id": payload.get("document_id"),
        "title": title,
        "authors": payload.get("authors") if payload.get("short_ref") else meta.get("authors", []),
        "publication_year": payload.get("publication_year") if payload.get("short_ref") else meta.get("publication_year"),
        "short_ref": short_ref,
        "source": f"{short_ref} — {title}" if short_ref else title,
        "section": payload.get("title_path") or payload.get("section"),
        # Conservés tels quels : « page de début / de fin du chunk », 0-basées, comme
        # stockées. Ce sont des clés internes, pas un affichage.
        "page_start": payload.get("page_start"),
        "page_end": payload.get("page_end"),
        # Le champ à citer. Distinct, nommé, et converti une seule fois.
        "pages_utilisateur": pages_utilisateur(payload.get("page_start"), payload.get("page_end")),
        "content_type": payload.get("content_type"),
        # L'ancre : où le passage vit dans le texte canonique de son document, et l'empreinte de
        # ce texte. C'est ce qui rend une citation **opposable** — la page, elle, est arrondie
        # au chunk et ne dit pas où chercher. Absente sur une collection bâtie avant le contrat
        # de sortie : le rendu se tait alors, il n'invente pas.
        "doc_text_sha256": payload.get("doc_text_sha256"),
        "ancrage_granularite": payload.get("ancrage_granularite"),
        "ancrage_intervalles": payload.get("ancrage_intervalles"),
        "score": float(score),
        "score_kind": kind,
        "text": payload.get("text", ""),
    }


def _scope_filter(scope: list[str] | None):
    from qdrant_client import models

    if scope is None:
        return None
    if len(scope) == 1:
        return models.Filter(must=[models.FieldCondition(key="document_id", match=models.MatchValue(value=scope[0]))])
    return models.Filter(must=[models.FieldCondition(key="document_id", match=models.MatchAny(any=scope))])


def _dense(query: str, pool: int, scope: list[str] | None) -> list[dict]:
    hits = client().query_points(collection_name=COLLECTION, query=encode_query(query), limit=pool,
                                 with_payload=True, query_filter=_scope_filter(scope)).points
    return [_payload_row(hit.payload or {}, hit.score, "cosine") for hit in hits]


def _lexical(query: str, pool: int, scope: list[str] | None) -> list[dict]:
    rows = [{"chunk_id": record["chunk_id"], "document_id": record["document_id"], "score": float(score)}
            for record, score in bm25().search(query, limit=pool * (5 if scope is not None else 1))]
    if scope is not None:
        allowed = set(scope)
        rows = [row for row in rows if row["document_id"] in allowed]
    return rows[:pool]


def _fill_text(rows: list[dict]) -> list[dict]:
    """Candidates that came from BM25 only have no text yet: fetch their payload in one scroll.

    Documented trap (todolist, point 2): a candidate without ``text`` is silently
    ignored by the reranker — it cost 8 points of R@1 the first time.
    """
    from qdrant_client import models

    missing = [row["chunk_id"] for row in rows if not row.get("text")]
    if not missing:
        return rows
    found, _ = client().scroll(
        collection_name=COLLECTION, limit=len(missing), with_payload=True,
        scroll_filter=models.Filter(must=[models.FieldCondition(key="chunk_id", match=models.MatchAny(any=missing))]),
    )
    payloads = {point.payload["chunk_id"]: point.payload for point in found}
    out = []
    for row in rows:
        if row.get("text"):
            out.append(row)
        elif row["chunk_id"] in payloads:
            out.append({**_payload_row(payloads[row["chunk_id"]], row.get("bm25_score", row.get("score", 0.0)), "bm25"),
                        **{k: row[k] for k in row if k.endswith("_rank") or k.endswith("_score")}})
    return out


def _hybrid(query: str, dense: list[dict], pool: int, scope: list[str] | None) -> list[dict]:
    from retrieval.hybrid import reciprocal_rank_fusion

    fused = reciprocal_rank_fusion(dense, _lexical(query, pool, scope))
    rows = _fill_text(fused)
    for row in rows:
        row["score"], row["score_kind"] = row["rrf_score"], "rrf"
    return rows[:pool]


@lru_cache(maxsize=1)
def _reranker():
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    dev = device()
    tok = AutoTokenizer.from_pretrained(RERANKER_ID)
    model = AutoModelForSequenceClassification.from_pretrained(
        RERANKER_ID, dtype=torch.float16 if dev == "mps" else torch.float32,
    ).to(dev).eval()
    return tok, model, dev


def _rerank(query: str, rows: list[dict]) -> list[dict]:
    import torch

    tok, model, dev = _reranker()
    rows = [row for row in rows if row.get("text")]
    scores: list[float] = []
    with torch.inference_mode():
        for start in range(0, len(rows), 8):
            batch = rows[start:start + 8]
            pairs = [[query, row["text"][:2000]] for row in batch]
            inputs = tok(pairs, padding=True, truncation=True, max_length=512, return_tensors="pt").to(dev)
            scores.extend(model(**inputs).logits.view(-1).float().cpu().tolist())
    for row, score in zip(rows, scores):
        row["rerank_score"] = score
        row["score"], row["score_kind"] = score, "rerank"
    return sorted(rows, key=lambda row: -row["rerank_score"])


# --------------------------------------------------------------------------- post-filtering

_WS = re.compile(r"\W+")


def _shingles(text: str, size: int = 8) -> set[str]:
    words = _WS.sub(" ", text.lower()).split()
    return {" ".join(words[i:i + size]) for i in range(0, max(len(words) - size, 0) + 1, 4)}


def _near_duplicate(a: str, b: str, threshold: float = 0.6) -> bool:
    sa, sb = _shingles(a), _shingles(b)
    if not sa or not sb:
        return False
    return len(sa & sb) / min(len(sa), len(sb)) >= threshold


def _select(rows: list[dict], limit: int, per_document: int, dedupe: bool, min_characters: int) -> list[dict]:
    selected: list[dict] = []
    per_doc: dict[str, int] = {}
    for row in rows:
        if len(selected) >= limit:
            break
        if row.get("content_type") == "heading" or len((row.get("text") or "").strip()) < min_characters:
            continue
        if per_document and per_doc.get(row["document_id"], 0) >= per_document:
            continue
        if dedupe and any(_near_duplicate(row["text"], kept["text"]) for kept in selected):
            continue
        per_doc[row["document_id"]] = per_doc.get(row["document_id"], 0) + 1
        selected.append(row)
    return selected


# --------------------------------------------------------------------------- public API

def search_explained(query: str, limit: int = 5, document_id: str | None = None, mode: str = "auto",
                     rerank: bool | None = None, per_document: int = 2, dedupe: bool = True,
                     pool: int | None = None, min_characters: int = MIN_CHARACTERS, log: bool = True,
                     year_min: int | None = None, year_max: int | None = None,
                     author: str | None = None, auto_period: bool = True,
                     reclassement: str | None = None) -> dict:
    """Routed search. Returns ``{"results": [...], "routing": {...}, "contrat": {...}}``.

    ``contrat`` est l'en-tête de traçabilité — ``request_id``, version du serveur, version du
    contrat de sortie, signature du corpus, hash de la configuration servie, appelant. Les
    mêmes champs ouvrent la ligne de ``rag/logs/router-decisions.jsonl``, et c'est ce qui
    permet de recoller une réponse à la décision qui l'a produite.

    **La signature ne gagne aucun paramètre** : l'appelant se déclare une fois à l'import
    (``contrat.enregistrer_serveur``) ou se déduit du point d'entrée. Ajouter un argument ici
    aurait cassé la garde qui protège les appels positionnels.

    ``mode``      "auto" lets the router decide; "dense"/"hybrid" force the path.
    ``rerank``    None follows the mode (on for hybrid, off for dense); True/False forces it.
    ``reclassement``  ``"selectif"`` reclasse les **dix premiers** candidats du pool par
                  ``Qwen3-Reranker-0.6B`` en **protégeant le rang 1 dense**, puis laisse la
                  sélection habituelle opérer. ``None`` (défaut) : rien ne change.
                  **Coût : ~3,2 s par requête contre 59 ms au chemin dense — un facteur 54**
                  (p50 3,21 s, p95 3,24 s, budget 10), et 2,4 Go résidents tant que le modèle
                  est chargé. À demander explicitement, jamais par défaut — **et depuis le
                  8 septembre 2026 c'est le coût, et lui seul, qui le dit** : voir ci-dessous.
                  Mesuré sur 155 questions (`5530cba145`) : l'or entre dans les cinq passages
                  servis pour **19** questions et en sort pour **5** ; les cinq pertes sont un
                  sous-ensemble strict des sept du reclassement plein.

                  **Ce que ces 19 et ces 5 valent au niveau RÉPONSE — mesuré le 8 septembre
                  2026** (`results-rotation-reponse-5530cba145.json`, 24 questions jugées dans
                  les deux bras) : sur les 5 « pertes », **une seule** perd vraiment sa réponse
                  — les trois autres mesurables avaient une couverture nulle **dans les deux
                  bras**, elles n'avaient rien à perdre. Sur les 19 « gains », **12** gagnent
                  vraiment une réponse. Le rapport réel est donc **12 pour 1**, non 19 pour 5,
                  et l'échange net vaut **+11 réponses** (+10 en comptant au pire les trois
                  questions v1 non notables). Trois du banc v1 ne sont pas mesurables faute
                  d'``answer_facts``.

                  **Conséquence, et elle est le seul changement de doctrine** : la raison de ne
                  pas l'activer systématiquement n'est plus le risque de régression — il est
                  cinq fois plus petit que ce qui était écrit ici — c'est **la mémoire et la
                  latence**. Sur une machine moins contrainte, le défaut mériterait d'être
                  rediscuté avec ces chiffres.
                  Un échec de reclassement **ne casse jamais la recherche** : l'ordre dense
                  est rendu et ``routing["reclassement"]["echec"]`` le dit.
    ``year_min`` / ``year_max`` / ``author``  restrict to documents whose consolidated
                  metadata match (undated documents are excluded by a year bound).
    ``auto_period``  when the caller gave no year bound and the query carries an explicit
                  publication period ("… according to sources published in 2022 or
                  earlier"), ``period_bounds`` turns it into the bound and removes the
                  clause from the text the embedding sees. Measured (benchmark/eval_protocol.py):
                  on the 15 dated questions of v3 it recovers the gold filter 15/15 and the
                  gain of the filter itself (+0.100 nDCG@10 in dense), with no false positive
                  on the 135 other v3 questions and the 25 of v1. The benchmark pins it off.
    """
    started = time.perf_counter()
    tracabilite = contrat.entete_tracabilite()
    pool = pool or POOL
    period = None
    if auto_period and year_min is None and year_max is None:
        period = period_bounds(query)
        if period:
            year_min, year_max, query = period["year_min"], period["year_max"], period["query"]
    decision = route(query, mode)
    if period:
        decision["period"] = {"clause": period["clause"], "year_min": year_min, "year_max": year_max, "query": query}
    scope = document_scope(document_id, year_min, year_max, author)
    if scope is not None:
        decision["filters"] = {k: v for k, v in (("document_id", document_id), ("year_min", year_min),
                                                 ("year_max", year_max), ("author", author)) if v is not None}
        decision["scope_documents"] = len(scope)
    if scope == []:
        decision.update(rerank=False, candidates=0, returned=0, latency_ms=0, dense_top1=None)
        if log:
            _log_decision(query, decision, tracabilite=tracabilite)
        return {"results": [], "routing": decision, "contrat": tracabilite}

    dense = _dense(query, pool, scope)
    decision["dense_top1"] = round(dense[0]["score"], 4) if dense else None
    if decision["mode"] == "hybrid":
        candidates = _hybrid(query, dense, pool, scope)
        do_rerank = True if rerank is None else rerank
    else:
        candidates = dense
        do_rerank = bool(rerank)
    # Le pool **tel que la récupération l'a rendu**, avant tout reclassement : c'est lui que
    # le journal doit porter. Journaliser l'ordre d'après reclassement rendrait le shadow
    # inutile — un rejeu doit pouvoir appliquer une *autre* règle au *même* point de départ.
    pool_initial = [row.get("chunk_id") for row in candidates[:10]]
    if do_rerank:
        candidates = _rerank(query, candidates)
    # Le reclassement sélectif s'insère au **même endroit** que le reranker historique :
    # après le pool, avant ``_select``. C'est ce qui permet au plafond par document de
    # s'appliquer à un meilleur ordre plutôt que de figer une éviction avant qu'on puisse
    # la défaire — les 14 questions ``EVICTION_DOCUMENT`` de l'audit des pertes en dépendent.
    if reclassement == "selectif":
        import reranking

        candidates, trace = reranking.reclasser(query, candidates, device(), RECLASSEMENT_ID)
        decision["reclassement"] = trace
    elif reclassement:
        raise ValueError(f"reclassement inconnu : {reclassement!r} (attendu : 'selectif' ou None)")

    results = _select(candidates, limit, per_document, dedupe, min_characters)
    decision.update(rerank=do_rerank, candidates=len(candidates), returned=len(results),
                    latency_ms=round((time.perf_counter() - started) * 1000))
    if log:
        _log_decision(query, decision, results, limit, pool_initial, tracabilite)
    return {"results": results, "routing": decision, "contrat": tracabilite}


def search(query: str, limit: int = 5, document_id: str | None = None, mode: str = "auto",
           rerank: bool | None = None, per_document: int = 2, dedupe: bool = True,
           pool: int | None = None, min_characters: int = MIN_CHARACTERS, log: bool = True,
           year_min: int | None = None, year_max: int | None = None, author: str | None = None,
           auto_period: bool = True, reclassement: str | None = None) -> list[dict]:
    """Routed search returning only the result rows (see ``search_explained``)."""
    return search_explained(query, limit, document_id, mode, rerank, per_document, dedupe,
                            pool, min_characters, log, year_min, year_max, author, auto_period,
                            reclassement)["results"]


def list_documents(author: str | None = None, year_min: int | None = None, year_max: int | None = None,
                   text: str | None = None, limit: int = 50) -> list[dict]:
    """Bibliographic listing (no embedding involved), newest first, then by title."""
    scope = document_scope(None, year_min, year_max, author)
    needle = _fold(text) if text else None
    rows = []
    for doc_id, rec in document_metadata().items():
        if scope is not None and doc_id not in scope:
            continue
        if needle and needle not in _fold(rec.get("title") or ""):
            continue
        rows.append({k: rec.get(k) for k in ("document_id", "short_ref", "title", "authors", "publication_year",
                                              "first_year", "venue", "doc_type", "confidence")})
    rows.sort(key=lambda r: (-(r["publication_year"] or 0), _fold(r["title"] or "")))
    return rows[:limit]


def timeline(topic: str, limit: int = 20, year_min: int | None = None, year_max: int | None = None,
             mode: str = "auto") -> dict:
    """Best passage per document for ``topic``, grouped by publication year (oldest first)."""
    out = search_explained(topic, limit=limit, mode=mode, per_document=1, pool=max(POOL, limit * 3),
                           year_min=year_min, year_max=year_max)
    groups: dict[str, list[dict]] = {}
    for row in out["results"]:
        groups.setdefault(str(row.get("publication_year") or "s.d."), []).append(row)
    ordered = sorted(groups.items(), key=lambda kv: (kv[0] == "s.d.", kv[0]))
    return {"topic": topic, "routing": out["routing"], "years": [{"year": y, "results": rs} for y, rs in ordered]}


def get_passage(chunk_id: str, max_characters: int | None = None, neighbours: int = 1) -> dict | None:
    """Return one chunk plus its immediate neighbours, for a longer citable passage.

    ``max_characters`` vaut ``contrat.PASSAGE_ENTIER_CHARACTERS`` (15 000) et non plus 6 000.
    Le défaut historique tronquait **30,2 %** des fenêtres du corpus — un outil dont le nom
    promet le passage entier en rendait moins une fois sur trois. Mesuré sur les 26 120
    fenêtres : médiane 4 520 c., p95 8 850, max 13 982.

    Les voisins ne sont **pas** filtrés comme ``_select`` filtre les résultats de recherche :
    un voisin en-tête est un intitulé de section, c'est-à-dire du contexte, et le seuil de
    250 caractères existe pour ne pas *classer* un fragment, pas pour ne pas le *montrer*.
    Ils sont en revanche **déclarés** : ``voisins`` dit ce qui a été joint et de quel type.

    La citation doit couvrir le texte **réellement rendu**, et non le seul chunk demandé.
    Mesuré le 7 septembre 2026 : ``get_passage("chunk-5ea0681947ca79cb")`` renvoie
    6 000 caractères pris aux pages 27, 28 et 31 — la fratrie est ordonnée par page, pas
    par contiguïté — sous la citation « p. 31 ». Un lecteur qui reprend cette page pour une
    phrase venue de la conclusion cite la mauvaise page, et le défaut est invisible : le
    texte est vrai, seule sa localisation est fausse.

    ``pages`` garde sa sémantique historique — les pages **du chunk demandé**, telles que
    stockées — et ``pages_utilisateur`` couvre la fenêtre après troncature.
    """
    from qdrant_client import models

    max_characters = contrat.PASSAGE_ENTIER_CHARACTERS if max_characters is None else max_characters
    found, _ = client().scroll(
        collection_name=COLLECTION, limit=1, with_payload=True,
        scroll_filter=models.Filter(must=[models.FieldCondition(key="chunk_id", match=models.MatchValue(value=chunk_id))]),
    )
    if not found:
        return None
    payload = found[0].payload
    window = [found[0]]
    if neighbours:
        siblings, _ = client().scroll(
            collection_name=COLLECTION, limit=64, with_payload=True,
            scroll_filter=models.Filter(must=[models.FieldCondition(key="parent_id", match=models.MatchValue(value=payload.get("parent_id")))]),
        )
        ordered = sorted(siblings, key=lambda point: (point.payload.get("page_start") or 0, point.payload.get("chunk_id") or ""))
        index = next((i for i, point in enumerate(ordered) if point.payload.get("chunk_id") == chunk_id), None)
        if index is not None:
            window = ordered[max(index - neighbours, 0): index + neighbours + 1]
    text, shown = joindre_passages([point.payload for point in window], max_characters)
    # **Union, pas enveloppe.** La fratrie est ordonnée par page, pas par contiguïté :
    # prendre min et max ferait affirmer les pages du trou.
    couvertes = sorted({page for p in shown
                        for page in (pages_utilisateur(p.get("page_start"), p.get("page_end")) or [])})
    head = _payload_row(payload, 0.0, "none")
    return {
        "chunk_id": chunk_id,
        "document_id": payload.get("document_id"),
        "title": head["title"],
        "short_ref": head["short_ref"],
        "source": head["source"],
        "section": payload.get("title_path") or payload.get("section"),
        "pages": [payload.get("page_start"), payload.get("page_end")],
        "pages_utilisateur": couvertes or None,
        "chunks_rendus": [p.get("chunk_id") for p in shown],
        "neighbours": neighbours,
        "voisins": [{"chunk_id": p.payload.get("chunk_id"),
                     "content_type": p.payload.get("content_type"),
                     "caracteres": len(p.payload.get("text") or "")}
                    for p in window if p.payload.get("chunk_id") != chunk_id],
        "content_type": payload.get("content_type"),
        "doc_text_sha256": payload.get("doc_text_sha256"),
        "ancrage_granularite": payload.get("ancrage_granularite"),
        "ancrage_intervalles": payload.get("ancrage_intervalles"),
        "qualite": contrat.drapeaux_qualite(
            text, tronque=len(SEPARATEUR_PASSAGES.join(p.payload.get("text") or "" for p in window)) > len(text)),
        "text": text,
    }


def _provenance() -> dict:
    """D'où viennent les documents du corpus, d'après le registre d'imports.

    Lit le fichier directement plutôt que d'importer ``rag/ingestion/registry.py`` :
    celui-ci importe ``quant_rag`` pour lire les plages d'identifiants de points, et
    l'inverse ferait un cycle.
    """
    if not corpus_overlay.REGISTRY.exists():
        return {"registry": None, "note": "registre absent : provenance inconnue"}
    from collections import Counter

    data = json.loads(corpus_overlay.REGISTRY.read_text(encoding="utf-8"))
    documents = data.get("documents", [])
    per_delivery = Counter(d.get("delivery", {}).get("id") for d in documents)
    return {
        "registry": str(corpus_overlay.REGISTRY.relative_to(ROOT)),
        "digest": corpus_overlay.registry_digest(),
        "documents": dict(Counter(d.get("status") for d in documents)),
        "parsers": dict(Counter(f"{d['parser']['backend']} {d['parser']['version']}"
                                for d in documents if d.get("parser"))),
        "deliveries": [{"id": entry.get("id"), "transport": entry.get("transport"),
                        "received_at": entry.get("received_at"),
                        "documents": per_delivery.get(entry.get("id"), 0)}
                       for entry in data.get("deliveries", [])],
    }


def corpus_status() -> dict:
    documents: dict[str, dict] = {}
    offset = None
    total = 0
    while True:
        points, offset = client().scroll(
            collection_name=COLLECTION, limit=4096, with_payload=["document_id", "title"], offset=offset,
        )
        for point in points:
            total += 1
            doc = documents.setdefault(point.payload["document_id"], {"title": point.payload.get("title"), "chunks": 0})
            doc["chunks"] += 1
        if offset is None:
            break
    meta = document_metadata()
    return {"collection": COLLECTION, "chunks": total, "documents": len(documents),
            # Ce qui est RÉELLEMENT interrogé, et non une constante de chemin : en mode
            # serveur, str(STORAGE) désignait un dossier embarqué inutilisé.
            "storage": qdrant_backend.description(STORAGE),
            "backend": qdrant_backend.mode(STORAGE), "device": device(),
            # Ce que le serveur sert, et son empreinte. Sans elles, un appelant ne peut pas dire
            # de quelle configuration vient un passage qu'il a reçu la semaine dernière.
            "contrat": {**contrat.configuration_servie(), "config_hash": contrat.config_hash()},
            "metadata": {"file": str(METADATA_PATH), "documents": len(meta),
                         "with_publication_year": sum(1 for r in meta.values() if r.get("publication_year")),
                         "with_authors": sum(1 for r in meta.values() if r.get("authors")),
                         "years": [min(r["publication_year"] for r in meta.values() if r.get("publication_year")),
                                   max(r["publication_year"] for r in meta.values() if r.get("publication_year"))]
                         if any(r.get("publication_year") for r in meta.values()) else None},
            "router": {"rule": (f"hybrid si n_exact ≥ {EXACT_TOKENS_FOR_HYBRID}" if EXACT_TOKENS_FOR_HYBRID is not None
                                else "auto = dense ; hybride sur demande (calibration v3)"),
                       "decision_log": str(DECISION_LOG)},
            "provenance": _provenance(),
            "bm25": {"index": str(bm25_path()), "present": bm25_path().exists(),
                     "matches_collection": _bm25_matches_collection() if bm25_path().exists() else None,
                     "manifest": json.loads(bm25_manifest_path().read_text(encoding="utf-8")) if bm25_manifest_path().exists() else None},
            "corpus_state": corpus_overlay.describe()}


def format_routing(decision: dict) -> str:
    period = decision.get("period")
    return ((f"période: « {period['clause']} » → year_min={period['year_min']} year_max={period['year_max']} · "
             f"requête: « {period['query'][:80]} »\n" if period else "")
            + f"routage: {decision['mode']} — {decision['reason']}"
            f" · rerank: {'oui' if decision.get('rerank') else 'non'}"
            + (f" · filtres: {decision['filters']} → {decision.get('scope_documents')} documents" if decision.get("filters") else "")
            + (f" · dense_top1={decision['dense_top1']:.3f}" if decision.get("dense_top1") is not None else "")
            + (f" · {decision['latency_ms']} ms" if decision.get("latency_ms") is not None else ""))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Recherche routée dans le corpus Quant RAG")
    parser.add_argument("query", nargs="?")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--mode", choices=MODES, default="auto")
    parser.add_argument("--rerank", dest="rerank", action="store_true", default=None, help="force le rerank")
    parser.add_argument("--no-rerank", dest="rerank", action="store_false", help="interdit le rerank")
    parser.add_argument("--year-min", type=int)
    parser.add_argument("--year-max", type=int)
    parser.add_argument("--author", help="filtre sur le nom d'un auteur (sous-chaîne, insensible aux accents)")
    parser.add_argument("--list", action="store_true", help="liste bibliographique (sans recherche)")
    parser.add_argument("--timeline", action="store_true", help="meilleur passage par document, groupé par année")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--rebuild-bm25", action="store_true",
                        help="rebâtit l'index BM25 de l'état courant du corpus depuis la collection")
    parser.add_argument("--purger-index-morts", action="store_true",
                        help="liste les index BM25 de data/lexical/ dont la signature n'est "
                             "ni la vivante ni la gelée, avec leur taille. LECTURE SEULE : "
                             "il faut --vraiment pour supprimer")
    parser.add_argument("--vraiment", action="store_true",
                        help="avec --purger-index-morts : supprime pour de bon")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    if args.purger_index_morts:
        rapport = purger_index_morts(vraiment=args.vraiment)
        if args.json:
            print(json.dumps(rapport, ensure_ascii=False, indent=1))
        else:
            mo = 1024 * 1024
            print(f"signature vivante : {rapport['signature_vivante']}   "
                  f"gelée : {rapport['signature_gelee']}")
            print(f"\n  VIVANTS ou GELÉS — jamais touchés ({len(rapport['vivants'])} fichier(s))")
            for f in rapport["vivants"]:
                print(f"    {f['octets'] / mo:8.1f} Mo  {f['nom']}")
            print(f"\n  MORTS — signature qui n'existe plus ({len(rapport['morts'])} fichier(s), "
                  f"{rapport['octets_morts'] / mo:.1f} Mo)")
            for f in rapport["morts"][:12]:
                print(f"    {f['octets'] / mo:8.1f} Mo  {f['nom']}")
            if len(rapport["morts"]) > 12:
                print(f"    … et {len(rapport['morts']) - 12} autre(s)")
            print(f"\n  SANS SIGNATURE — formats anciens, JAMAIS supprimés automatiquement "
                  f"({len(rapport['sans_signature'])} fichier(s), "
                  f"{rapport['octets_sans_signature'] / mo:.1f} Mo)")
            for f in rapport["sans_signature"]:
                print(f"    {f['octets'] / mo:8.1f} Mo  {f['nom']}")
            print("\n" + (f"  supprimé : {len(rapport['supprimes'])} fichier(s), "
                         f"{rapport['octets_morts'] / mo:.1f} Mo libérés"
                         if rapport["applique"] else
                         "  rien n'a été supprimé — ajouter --vraiment pour appliquer"))
    elif args.rebuild_bm25:
        started = time.perf_counter()
        index = rebuild_bm25()
        print(f"{bm25_path().name} : {index.doc_count} chunks, {len(index.document_frequency)} termes, "
              f"{time.perf_counter() - started:.0f} s\nmanifeste : {bm25_manifest_path()}")
    elif args.status:
        print(json.dumps(corpus_status(), indent=2, ensure_ascii=False))
    elif args.list:
        for row in list_documents(author=args.author, year_min=args.year_min, year_max=args.year_max,
                                  text=args.query, limit=args.limit if args.limit != 5 else 50):
            print(f"{str(row['publication_year'] or 's.d.'):5} {row['short_ref'][:34]:34} {row['title'][:80]}")
    elif args.timeline and args.query:
        out = timeline(args.query, limit=args.limit if args.limit != 5 else 20, year_min=args.year_min, year_max=args.year_max)
        print(format_routing(out["routing"]))
        for group in out["years"]:
            print(f"\n== {group['year']}")
            for row in group["results"]:
                print(f"  {row['source'][:100]}  ({citation_pages(row.get('pages_utilisateur'))}, "
                      f"{row['score_kind']}={row['score']:.3f})")
    elif args.query:
        out = search_explained(args.query, limit=args.limit, mode=args.mode, rerank=args.rerank,
                               year_min=args.year_min, year_max=args.year_max, author=args.author)
        if args.json:
            print(json.dumps(out, indent=2, ensure_ascii=False))
        else:
            print(format_routing(out["routing"]))
            for i, row in enumerate(out["results"], 1):
                head = (f"[{i}] {row['score_kind']}={row['score']:.3f}  {row['source']}  "
                        f"{citation_pages(row.get('pages_utilisateur'))}")
                print(f"\n{head}\n{'-' * len(head)}\n{row['section'] or ''}\n{row['text'][:900]}")
    else:
        parser.error("donne une question, ou --status / --list")
