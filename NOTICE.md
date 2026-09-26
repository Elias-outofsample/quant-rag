# Notice — provenance

`rag/` — the retrieval layer, the MCP server, the output contract, citation verification,
the entity-graph queries, the ingestion chain and the whole evaluation harness — is
original work.

`src/` holds the small part of an earlier, shared parsing codebase that `rag/` imports,
kept so that the layer runs as published:

| File | Origin |
|---|---|
| `src/retrieval/lexical.py` | earlier shared codebase (BM25 index), unchanged |
| `src/retrieval/hybrid.py` | earlier shared codebase (reciprocal rank fusion), unchanged |
| `src/parsing/models.py` | earlier shared codebase (canonical document model), unchanged |
| `src/parsing/mineru_adapter.py` | earlier shared codebase (MinerU output to canonical blocks), unchanged |
| `src/parsing/canonical_chunker.py` | earlier shared codebase, **modified here**: section-path fix (label agreement 42.9 % → 89.1 %) |
| `src/parsing/document_text.py` | original: canonical text of a document and passage offsets |
| `src/parsing/table_markdown.py` | original: HTML tables to Markdown, standard library only |

The package `__init__.py` files were rewritten to drop imports of modules that are not
part of this repository.

## Models and tools used, not redistributed

- Embeddings: `Qwen/Qwen3-Embedding-0.6B`.
- Rerankers (experiments and hybrid mode): `BAAI/bge-reranker-base`, `BAAI/bge-reranker-v2-m3`,
  `Qwen/Qwen3-Reranker-0.6B`.
- Entity extraction: GLiNER2. PDF parsing: MinerU.
- Benchmark only (question drafting and judging): Mistral and Gemini APIs.

Model weights are downloaded from their publishers at first use, under their own licences.

## Corpus

The documents indexed by this system, and every artefact derived from them, are not
distributed. Test fixtures derived from the corpus were replaced by structure-preserving
scrambles (see `rag/benchmark/tests/test_contrat_coupe.py`).
