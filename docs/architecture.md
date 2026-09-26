# Architecture

The system has two halves that share one corpus state: an **offline** half that turns PDFs
into a searchable, versioned corpus, and a **serving** half that answers an agent through
the Model Context Protocol. A third part, the evaluation harness, is described in
[evaluation.md](evaluation.md).

All timings below were measured on an Apple M4 with 16 GB of memory, on the served corpus
of 418 documents and 26,120 passages.

## 1. Corpus state

A search result is only as reproducible as the corpus it came from, so the state of the
corpus is explicit and named.

| Piece | What it is | Where |
|---|---|---|
| Parsed documents | canonical blocks, passages (chunks) and parent sections per document | `data/processed/…` (private) |
| Vector export | the embedded passages, frozen | `data/qdrant-export/` (private) |
| Registry | which documents are active, with the SHA-256 of their four canonical files | `rag/ingestion/registry.py` |
| Overlays | decisions applied on top of the export, everywhere the corpus is read: removed duplicates, tables converted to Markdown, clean titles | `rag/corpus_overlay.py` |
| Signature | a hash of the document set, the text overlays and the consolidated titles | `corpus_overlay.signature()` |
| Freeze | the corpus is frozen between batches; unfreezing takes a written reason, kept in a history | `rag/gel_corpus.py` |

The signature is recomputed on every call (2.4 ms) rather than cached: a process that
edits the registry must see the new value at once. Every derived artefact carries it. The
BM25 index file is named `bm25-<label>-<signature>.json`, so an index built on another
corpus state can never be loaded by mistake, and `benchmark/check_bm25.py` proves that
production and benchmark serve the same index.

### Curation passes

Each pass was measured before it was kept.

- **Bibliographic metadata** (`rag/metadata/`). Titles, authors and years were missing or
  derived from file names. Four sources are consolidated with their provenance: the file
  name, the arXiv stamp printed on the first pages, the metadata embedded in the PDF, and an
  LLM reading of the first page that must quote its evidence for a year. A date prefix
  shared by five files or more is treated as a download date and never used. Every result
  now carries a citable source ("López de Prado & Bailey (2014) — The Deflated Sharpe
  Ratio …") and searches can be scoped by year and author.
- **Tables** (`rag/tables/`, `src/parsing/table_markdown.py`). About a fifth of the passages
  were raw HTML table fragments. They are converted to Markdown (standard library only,
  `colspan`/`rowspan` expanded, the header repeated on continuation fragments) and
  re-embedded; the original HTML is kept alongside.
- **Clean titles in the embedding** (`rag/titles/`). Each passage is embedded behind a
  `Document: <title>` line, and export titles were file names, 200-character strings or,
  in one case, a whole HTML table. Re-embedding with the consolidated title gained
  +0.031 nDCG@10 [+0.003, +0.059]; the embedding recipe was first reproduced on stored
  vectors (minimum cosine 0.99977) before anything was written.
- **Deduplication** (`rag/metadata/scan_duplicates.py`). Duplicate documents are removed
  through an overlay, never by editing the export.

### Anchors in the canonical text

`src/parsing/document_text.py` rebuilds the canonical text of each document from its
blocks, and `rag/ancrage.py` locates every passage in it: character offsets plus the
SHA-256 of the text. 100 % of the served passages are anchored, 86.9 % at the exact
granularity (the offsets delimit the passage word for word), the rest at block level
(reassembled tables, normalised spacing). Rebuilding the anchors takes 1.8 s.

## 2. Serving path

`rag/quant_rag.py` is the only retrieval path; the CLI, the MCP server and the benchmark
all call it. In order:

1. **Period clause.** A period written in the question ("sources published in 2022 or
   earlier", "papers before 2010", "depuis 2020") is parsed into `year_min` / `year_max`
   and removed from the text that is embedded. The response says so on a `période:` line.
   English and French patterns; the benchmark pins this off to measure raw rankings.
2. **Bibliographic scope.** Year and author bounds become a Qdrant filter on
   `document_id`, applied before the search; the same list filters BM25 candidates.
   Scores are not modified.
3. **Mode.** `auto` is dense. `hybrid` (BM25 over the current corpus state, reciprocal
   rank fusion with dense, then a cross-encoder) is available on request, for queries made
   of identifiers the user remembers. The exact-token detector that used to route
   automatically is still run and logged: a larger benchmark can recalibrate it without new
   code. Why `auto` is dense is the subject of [evaluation.md](evaluation.md#72-the-router).
4. **Dense search.** Qwen3-Embedding-0.6B with a retrieval instruction, top 50 from Qdrant
   (exact search in embedded mode).
5. **Selection.** Headings and passages under 250 characters are dropped; near-duplicates
   are removed (Jaccard on word 8-grams, threshold 0.6); at most two passages per document,
   so that one 800-passage textbook cannot fill the answer.
6. **Rendering.** The output contract (`rag/contrat.py`, see
   [output-contract.md](output-contract.md)) renders each passage whole, with its source,
   anchor, quality flags and the trace of the request.

Every decision is appended to `rag/logs/router-decisions.jsonl`: timestamp, query, mode
chosen and requested, reason, detected tokens, dense top-1 score, rerank, latency, and the
same request id as the served response.

### Storage and concurrency

Qdrant runs **embedded** by default (`QdrantClient(path=…)`, no daemon, no port). Embedded
storage takes an exclusive file lock: while the MCP server holds it, nothing else can open
the collection. `QUANT_RAG_QDRANT_URL` switches to a Qdrant server instead; the choice has a
single home, `rag/qdrant_backend.py`, and a parity check compares the two backends on the
same queries. Brute-force search in embedded mode costs about 0.7 µs per vector: the server
becomes the answer past roughly 100,000 passages.

Experiments do not need the lock: `benchmark/dense_matrix.py` rebuilds the vector matrix
from the export and the overlays (76 MB in memory, about 20 ms per query) and returns the
same top 50 as Qdrant, in the same order, on 155/155 benchmark questions.

### Measured latencies

| Step | Time |
|---|---:|
| Rebuild the embedded collection from the export and overlays | 15.6 s |
| Load the embedding model (first call of a process) | 10.5–12.4 s |
| Routing decision | < 1 ms |
| Dense search, warm (5 real queries on the running server) | p50 84.7 ms, p95 88.9 ms |
| Hybrid search, warm (BM25 + RRF + rerank of 50) | about 1.9 s |
| Rebuild the BM25 index (26,120 passages, 104,823 terms) | 8.6 s |
| Rebuild the entity graph from extraction results | 1.4 s |

## 3. The MCP server

`rag/mcp_server.py` exposes ten tools over stdio: `search_documents`, `get_passage`,
`list_documents`, `timeline`, `corpus_status`, `verify_citation`, `verify_citations`,
`search_graph`, `expand_entity` and `connect_entities`. Tool descriptions carry the real
corpus counts, computed at start-up: a count written by hand in a prompt had once been
wrong by 162 documents, and a model reasons on the number it is given.

`agents/rag-scout.md` is the minimal system prompt of an agent that consumes these tools.
It states only what the server does not do by itself: write the query in English and in
natural language, turn implicit periods ("recent", "before the crisis") into explicit
bounds, keep `hybrid` for remembered identifiers, cross-check with the graph, and verify
any quote that was not just read.

## 4. Entity graph

GLiNER2 extracts people, organisations, instruments, market concepts, measures, methods and
datasets, with ten typed relations (`uses_method`, `applies_to`, `is_a`, …). The graph was
re-extracted after the table conversion: on raw HTML, the most frequent "entity" had been
`rowspan` (5,303 mentions), and HTML tags accounted for 11,394 mentions.

`rag/graph_search.py` answers three joins in memory, without a graph database: passages
that mention an entity, entities related to one, and passages where two entities meet.
Fusing graph hits into the main ranking was measured and rejected (−0.105 nDCG@10); the
graph is served as separate exploration tools.

## 5. Ingestion

Two sources feed one chain: already-parsed deliveries (Source A) and PDFs parsed locally
with MinerU (Source B). From the delivery directory on, the path is the same:
`inspect_delivery.py` (reads, writes nothing) then `apply_delivery.py`, the only module that
writes to Qdrant, BM25 and the registry.

`batch_driver.py` runs a batch without an operator, one document at a time:

| Step | What it does | Measured on a real 6-page paper |
|---|---|---:|
| parse | MinerU: blocks, passages, parent sections | minutes for a new parse |
| survey | seal the list of the document's passages | < 1 s |
| diagnosis | addition, revision, duplicate or near-edition; gold passages at risk | 5 s |
| import | stage, metadata, embedding, upsert, promotion, registry, BM25 | 25.9 s |
| downstream | rebuild and verify the registry; check the BM25 index | 19.7 s |
| graph | entity extraction, then graph rebuild | 24.8 s |
| probes | is the document actually served? two real queries | 12.4 s |
| commit | version the new corpus state | about 5 s |

What makes it safe to leave running:

- it refuses to start on a doubtful environment (for instance a Qdrant lock held by another
  process, found with `lsof`, not assumed);
- an import holds an exclusive lock, released even when the import fails; a lock whose
  process is dead is taken over loudly, one whose process is alive never is;
- it resumes after a `kill -9` by observing the state of the corpus rather than trusting its
  own memory;
- an import can be rolled back symmetrically: the signature returns to its previous value
  and the points are removed;
- nothing it cannot decide is decided: every document that is not a clean addition is set
  aside with its reason.

With `--llm sans-llm`, titles and years come from the PDF metadata, the arXiv stamp or a
manual override; the example paper was ingested with zero LLM calls.

## 6. Checks that keep the state honest

- `verifier_installation.py`: every artefact the served path depends on, checked for
  coherence (index and collection counts against the registry, unresolved LFS pointers),
  with how to rebuild each; non-zero exit code if one is missing.
- `ingestion/registry.py --verify` and `benchmark/check_bm25.py`: no missing and no
  phantom passage between registry, collection and lexical index.
- Tests that forbid regressions of a decision rather than of a function: no literal
  truncation outside the contract, no hard-coded Qdrant client outside the backend module,
  one price sheet for cost accounting, no docstring that cites a line number.
