"""Accès au corpus et outillage lexical — la moitié *objective* du banc d'essai.

Rien ici n'appelle un LLM. Tout ce qui est calculé dans ce module est
reproductible à l'identique : même corpus, même chiffre, pour toujours. C'est
volontaire — voir README.md, « Pourquoi deux familles de scores ».

Trois services :
  - ``ChunkIndex``  : chunk_id -> texte et métadonnées, chargé une fois, mis en cache ;
  - ``leak_score()``: fuite lexicale question -> passage cible, la mesure qui
                      quantifie le biais known-item ;
  - ``absent_from_corpus()`` : vérification d'absence d'un terme, par balayage
                      exhaustif du vocabulaire — jamais par le retriever
                      lui-même (ce serait circulaire).
"""
from __future__ import annotations

import json
import math
import pickle
import random
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CACHE = HERE / ".cache"
ROWS = ROOT / "data" / "embeddings" / "ingested-all-qwen3-06b" / "rows.jsonl"
#: ``rows.jsonl`` est l'artefact amont : il ne connaît que les chunks livrés. Sans ce
#: complément, la vue corpus du banc — mesure de fuite, vérification des ors, statistiques —
#: ignorerait les documents entrés par une livraison locale. Reconstruit par
#: ``rag/ingestion/registry.py``.
IMPORTED_ROWS = ROOT / "rag" / "ingestion" / "imported-rows.jsonl"
#: Métadonnées bibliographiques consolidées (rag/metadata/) : titre propre, année,
#: auteurs. Le titre d'export, dérivé du nom de fichier, peut être un nom de
#: fichier de 200 caractères — ou, pour un document, un tableau HTML entier.
METADATA = ROOT / "rag" / "metadata" / "documents-metadata-v1.json"

sys.path.insert(0, str(HERE.parent))
import corpus_overlay  # noqa: E402  — documents retirés, tableaux convertis (voir rag/corpus_overlay.py)

#: Même tokenisation que ``src/retrieval/lexical.py`` : elle garde les termes
#: techniques collés (« black-scholes », « garch(1,1) », « 5.4.2 »), ce qui compte
#: en finance quantitative où les noms propres et les numéros de section portent
#: le signal.
TOKEN_PATTERN = re.compile(r"[^\W_]+(?:[-./][^\W_]+)*|[^\s\w]", re.UNICODE)


def _corpus_lines():
    """Les lignes de ``rows.jsonl``, puis celles des documents importés."""
    with ROWS.open(encoding="utf-8") as handle:
        yield from handle
    if IMPORTED_ROWS.exists():
        with IMPORTED_ROWS.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield line


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(" ".join(str(text).casefold().split()))


#: Mots vides : ils ont un IDF faible de toute façon, mais les retirer rend la
#: mesure de fuite plus stable sur les questions courtes.
_NON_WORD = re.compile(r"[^0-9a-z\u00c0-\u024f]+")


def _normalise(text: str) -> str:
    """Casse pliée, ponctuation en espaces, espaces réduits."""
    return " ".join(_NON_WORD.sub(" ", str(text).casefold()).split())


STOPWORDS = frozenset("""a an and are as at be by can do does for from how in into is it its of on or
that the this to what when where which who why with you your does't don't not""".split())


class ChunkIndex:
    """Vue mémoire du corpus : le strict nécessaire, mis en cache sur disque.

    ``rows.jsonl`` fait 157 Mo ; le relire à chaque exécution coûte ~40 s. On en
    extrait une fois la projection utile (~45 Mo picklés) et on la recharge en 2 s.
    """

    #: bumper invalide le cache ; la signature de l'overlay entre aussi dans le nom.
    #: v5 (4 septembre 2026) : la vue inclut désormais les documents importés
    #: (``imported-rows.jsonl``). La signature ne change pas quand le *code* de la vue
    #: change — seul ce numéro le dit.
    VERSION = 5

    def __init__(self, chunks: dict, documents: dict, document_frequency: Counter, total: int):
        self.chunks = chunks
        self.documents = documents
        self.df = document_frequency
        self.total = total
        self._vocabulary = None
        self._blob = None
        self._metadata = None

    # ------------------------------------------------------------------ chargement

    @classmethod
    def load(cls, verbose: bool = True) -> "ChunkIndex":
        path = CACHE / f"chunk-index-v{cls.VERSION}-{corpus_overlay.signature()}.pkl"
        if path.exists():
            with path.open("rb") as handle:
                payload = pickle.load(handle)
            if verbose:
                print(f"  corpus  {len(payload['chunks'])} chunks depuis le cache ({path.name})")
            return cls(payload["chunks"], payload["documents"], payload["df"], payload["total"])

        if not ROWS.exists():
            sys.exit(f"corpus introuvable : {ROWS}")
        if verbose:
            print(f"  corpus  première lecture de {ROWS.name} (157 Mo, ~40 s)…")

        chunks, documents, df = {}, {}, Counter()
        for line in _corpus_lines():
            row = json.loads(line)
            document, chunk = row["document"], row["chunk"]
            text = corpus_overlay.apply(chunk["document_id"], chunk["chunk_id"], chunk.get("text") or "")
            if text is None:  # document retiré du corpus (duplicates-v1.json)
                continue
            chunks[chunk["chunk_id"]] = {
                "chunk_id": chunk["chunk_id"],
                "document_id": chunk["document_id"],
                "parent_id": chunk.get("parent_id"),
                "section": chunk.get("section") or chunk.get("title_path") or "",
                "page_start": chunk.get("page_start"),
                "content_type": chunk.get("content_type") or "",
                "token_count": chunk.get("token_count"),
                "text": text,
            }
            if chunk["document_id"] not in documents:
                documents[chunk["document_id"]] = {
                    "document_id": chunk["document_id"],
                    "title": document.get("title") or "",
                    "filename": document.get("filename") or "",
                    "chunks": 0,
                }
            documents[chunk["document_id"]]["chunks"] += 1
            df.update(set(tokenize(text)))

        CACHE.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            pickle.dump({"chunks": chunks, "documents": documents, "df": df, "total": len(chunks)},
                        handle, protocol=pickle.HIGHEST_PROTOCOL)
        if verbose:
            print(f"  corpus  {len(chunks)} chunks, {len(documents)} documents, "
                  f"{len(df)} termes distincts -> {path.name}")
        return cls(chunks, documents, df, len(chunks))

    # ------------------------------------------------------------------ accès

    def __getitem__(self, chunk_id: str) -> dict:
        return self.chunks[chunk_id]

    def get(self, chunk_id: str, default=None):
        return self.chunks.get(chunk_id, default)

    def document_of(self, chunk_id: str) -> str | None:
        row = self.chunks.get(chunk_id)
        return row and row["document_id"]

    def title_of(self, document_id: str) -> str:
        return self.documents.get(document_id, {}).get("title", "")

    # ------------------------------------------------------------------ métadonnées

    @property
    def metadata(self) -> dict:
        """document_id -> fiche consolidée (vide si le fichier est absent)."""
        if getattr(self, "_metadata", None) is None:
            self._metadata = {}
            if METADATA.exists():
                data = json.loads(METADATA.read_text(encoding="utf-8"))
                self._metadata = {row["document_id"]: row for row in data["documents"]
                                  if row["document_id"] in self.documents}
        return self._metadata

    def metadata_of(self, document_id: str) -> dict:
        return self.metadata.get(document_id, {})

    def display_title(self, document_id: str) -> str:
        """Titre propre s'il existe, titre d'export sinon."""
        return self.metadata_of(document_id).get("title") or self.title_of(document_id)

    def year_of(self, document_id: str, reliable: bool = True) -> int | None:
        """Année de publication (version présente dans le corpus).

        ``reliable`` n'accepte que les années dont la provenance est sûre ou moyenne et
        sans conflit entre sources : une question datée bâtie sur une année fausse
        mesurerait l'erreur des métadonnées, pas le retrieval.
        """
        rec = self.metadata_of(document_id)
        year = rec.get("publication_year")
        if year is None:
            return None
        if reliable:
            if rec.get("provenance", {}).get("year_confidence") not in ("high", "medium"):
                return None
            if any("année" in str(conflict) for conflict in rec.get("conflicts", [])):
                return None
        return int(year)

    def years(self, reliable: bool = True) -> dict[str, int]:
        return {doc: year for doc in self.documents if (year := self.year_of(doc, reliable)) is not None}

    def chunks_of(self, document_id: str) -> list[str]:
        return [cid for cid, row in self.chunks.items() if row["document_id"] == document_id]

    # ------------------------------------------------------------------ lexique

    def idf(self, term: str) -> float:
        """IDF lissé façon BM25. Un terme absent du corpus obtient l'IDF maximal."""
        n = self.df.get(term, 0)
        return math.log(1 + (self.total - n + 0.5) / (n + 0.5))

    @property
    def vocabulary(self) -> frozenset:
        if self._vocabulary is None:
            self._vocabulary = frozenset(self.df)
        return self._vocabulary

    def absent_from_corpus(self, term: str) -> bool:
        """Le terme n'apparaît nulle part dans les 19 443 chunks.

        Le balayage porte sur le vocabulaire complet, pas sur un top-k de
        recherche : c'est la seule façon d'établir une absence sans demander au
        système sous test de juger de sa propre ignorance.
        """
        parts = tokenize(term)
        return bool(parts) and all(part not in self.vocabulary for part in parts)

    def occurrences(self, term: str) -> int:
        """Borne supérieure du nombre de chunks pouvant contenir la locution.

        Zéro sur le token le plus rare suffit à prouver que la locution complète
        n'apparaît nulle part. Le test est *sain* mais grossier : « Paul Dupuis »
        est déclaré présent dès lors que « paul » et « dupuis » figurent dans le
        corpus, fût-ce dans deux documents différents. Pour trancher, voir
        ``phrase_absent``.
        """
        parts = tokenize(term)
        return min((self.df.get(part, 0) for part in parts), default=0)

    @property
    def blob(self) -> str:
        """Corpus entier, normalisé, en une chaîne — pour la recherche de locution.

        ~40 Mo en mémoire, reconstruits en quelques secondes et mis en cache.
        C'est ce qui permet de *prouver* qu'une locution est absente, plutôt que
        de le déduire de la présence séparée de ses tokens.
        """
        if self._blob is None:
            path = CACHE / f"corpus-blob-v{self.VERSION}-{corpus_overlay.signature()}.txt"
            if path.exists():
                self._blob = path.read_text(encoding="utf-8")
            else:
                self._blob = " \n ".join(_normalise(row["text"]) for row in self.chunks.values())
                CACHE.mkdir(parents=True, exist_ok=True)
                path.write_text(self._blob, encoding="utf-8")
        return self._blob

    def phrase_absent(self, term: str) -> bool:
        """La locution exacte n'apparaît dans aucun des 19 443 chunks.

        Balayage littéral du corpus normalisé (ponctuation neutralisée, casse
        pliée, espaces réduits) : « Friz and Hairer (2014) » et « friz & hairer,
        2014 » sont la même chose. Une absence ici est un fait vérifiable, pas
        une inférence — et surtout, elle ne demande rien au système sous test.
        """
        needle = _normalise(term)
        return bool(needle) and f" {needle} " not in f" {self.blob} "

    # ------------------------------------------------------------------ fuite

    def leak_score(self, question: str, chunk_id: str) -> float:
        """Part de l'information de la question littéralement présente dans le passage.

        C'est *la* mesure du biais known-item. Une question rédigée en regardant
        le passage en recopie les termes rares ; ces termes ont un IDF élevé, la
        part monte. Une question de praticien pose le problème avec son propre
        vocabulaire ; la part reste basse.

        Formellement, sur les tokens informatifs de la question :

            leak = sum_{t in Q inter C} idf(t) / sum_{t in Q} idf(t)

        Bornée [0, 1]. Voir README.md pour la distribution mesurée sur v1 et v2.
        """
        row = self.chunks.get(chunk_id)
        if row is None:
            return float("nan")
        return self.leak_against_text(question, row["text"])

    def leak_against_text(self, question: str, text: str) -> float:
        query = [t for t in set(tokenize(question)) if t not in STOPWORDS and len(t) > 1]
        if not query:
            return 0.0
        target = set(tokenize(text))
        total = sum(self.idf(t) for t in query)
        shared = sum(self.idf(t) for t in query if t in target)
        return shared / total if total else 0.0


# ---------------------------------------------------------------------- sélection

#: Un chunk « table des matières » (Tsay, par exemple) est du bruit : beaucoup de
#: texte, aucune affirmation. Les repérer évite de fabriquer des questions creuses.
_TOC = re.compile(r"(\.{4,}|\,\s*\d{1,4}\s*\n|\b\d+\.\d+(\.\d+)?\s+[A-Z][^\n]{0,60}\,\s*\d{1,4})")


def looks_like_toc(text: str) -> bool:
    hits = len(_TOC.findall(text[:4000]))
    return hits >= 6


def is_table(row: dict) -> bool:
    return row["content_type"] == "table" or "<td>" in row["text"] or "<tr>" in row["text"]


def sample_seed_chunks(index: ChunkIndex, count: int, kind: str, seed: int,
                       exclude_documents: set | None = None,
                       exclude_chunks: set | None = None,
                       allow_document=None) -> list[dict]:
    """Échantillon stratifié de chunks-sources, au plus un par document.

    Le plafond par document est important : les 12 plus gros documents pèsent 31 %
    de l'index (point 7 de la todolist). Un tirage uniforme sur les chunks
    produirait un banc d'essai qui parle surtout de Tsay.

    ``allow_document`` : prédicat optionnel sur le document (ex. : année connue).
    """
    rng = random.Random(seed)
    exclude_documents = set(exclude_documents or ())
    exclude_chunks = set(exclude_chunks or ())

    pool = []
    for row in index.chunks.values():
        if row["chunk_id"] in exclude_chunks or row["document_id"] in exclude_documents:
            continue
        if allow_document is not None and not allow_document(row["document_id"]):
            continue
        text = row["text"]
        table = is_table(row)
        if kind == "table" and not table:
            continue
        if kind != "table" and table:
            continue
        # Assez long pour porter une affirmation, pas une table des matières.
        if len(text) < (500 if kind == "table" else 900) or len(text) > 6000:
            continue
        if looks_like_toc(text):
            continue
        pool.append(row)

    rng.shuffle(pool)
    picked, seen_documents = [], set()
    for row in pool:
        if row["document_id"] in seen_documents:
            continue
        seen_documents.add(row["document_id"])
        picked.append(row)
        if len(picked) >= count:
            break
    return picked


if __name__ == "__main__":
    # Diagnostic : distribution de la fuite lexicale sur le banc known-item v1.
    index = ChunkIndex.load()
    path = HERE / "questions-v1.jsonl"
    scores = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        q = json.loads(line)
        scores.append((q["qid"], index.leak_score(q["question"], q["target_chunk"])))
    values = sorted(s for _, s in scores)
    n = len(values)
    print(f"\nfuite lexicale, {n} questions known-item v1")
    print(f"  min {values[0]:.3f}   p25 {values[n//4]:.3f}   médiane {values[n//2]:.3f}"
          f"   p75 {values[3*n//4]:.3f}   max {values[-1]:.3f}")
    print(f"  moyenne {sum(values)/n:.3f}")
    for qid, score in sorted(scores, key=lambda kv: -kv[1])[:5]:
        print(f"    {qid}  {score:.3f}")
