"""Deterministic local BM25 retrieval for the frozen corpus."""

import hashlib
import json
import math
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


TOKEN_PATTERN = re.compile(r"[^\W_]+(?:[-./][^\W_]+)*|[^\s\w]", re.UNICODE)


def tokenize(text):
    """Case-fold text while retaining technical terms, numbers, hyphens and symbols."""
    return TOKEN_PATTERN.findall(" ".join(str(text).casefold().split()))


def lexical_text(document, chunk):
    path = chunk.get("title_path") or document["title"]
    parts = [document["title"], path, chunk.get("text", "")]
    if chunk.get("content_type") == "table":
        metadata = chunk.get("metadata", {})
        for key in ("caption", "table_caption", "headers", "column_headers", "rows"):
            value = metadata.get(key)
            if value:
                parts.append(json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else str(value))
    return "\n".join(part for part in parts if part)


class BM25Index:
    backend = "local-bm25"
    backend_version = "1.0"

    def __init__(self, records, k1=1.5, b=0.75):
        self.records = records
        self.k1 = float(k1)
        self.b = float(b)
        self.doc_count = len(records)
        self.lengths = [sum(record["term_counts"].values()) for record in records]
        self.avgdl = sum(self.lengths) / self.doc_count if self.doc_count else 0.0
        self.document_frequency = Counter()
        self.postings = {}
        for index, record in enumerate(records):
            for term, count in record["term_counts"].items():
                self.document_frequency[term] += 1
                self.postings.setdefault(term, []).append((index, count))

    @classmethod
    def from_corpus(cls, rows, k1=1.5, b=0.75):
        records = []
        for document, chunk in rows:
            tokens = tokenize(lexical_text(document, chunk))
            records.append({
                "chunk_id": chunk["chunk_id"],
                "parent_id": chunk.get("parent_id"),
                "document_id": chunk["document_id"],
                "pages": [chunk.get("page_start"), chunk.get("page_end")],
                "title_path": chunk.get("title_path"),
                "term_counts": dict(Counter(tokens)),
            })
        return cls(records, k1=k1, b=b)

    def search(self, query, limit=10):
        if not str(query).strip() or not self.doc_count:
            return []
        scores = Counter()
        for term in set(tokenize(query)):
            frequency = self.document_frequency.get(term, 0)
            if not frequency:
                continue
            idf = math.log(1 + (self.doc_count - frequency + 0.5) / (frequency + 0.5))
            for index, count in self.postings[term]:
                denominator = count + self.k1 * (1 - self.b + self.b * self.lengths[index] / self.avgdl)
                scores[index] += idf * count * (self.k1 + 1) / denominator
        return [(self.records[index], score) for index, score in sorted(scores.items(), key=lambda item: (-item[1], self.records[item[0]]["chunk_id"]))[:limit]]

    def dump(self, path):
        Path(path).write_text(json.dumps({"k1": self.k1, "b": self.b, "records": self.records}, ensure_ascii=False), encoding="utf8")

    @classmethod
    def load(cls, path):
        payload = json.loads(Path(path).read_text(encoding="utf8"))
        return cls(payload["records"], k1=payload["k1"], b=payload["b"])


def config_hash(corpus_hash, k1, b):
    payload = json.dumps({"corpus_hash": corpus_hash, "tokenizer": TOKEN_PATTERN.pattern, "k1": k1, "b": b}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def manifest(corpus, corpus_hash, indexed_chunks, k1, b):
    return {
        "corpus_version": corpus["corpus_version"],
        "corpus_hash": corpus_hash,
        "indexed_chunks": indexed_chunks,
        "backend": BM25Index.backend,
        "backend_version": BM25Index.backend_version,
        "tokenizer": {"name": "unicode-technical-v1", "pattern": TOKEN_PATTERN.pattern, "casefold": True, "stopwords": "none"},
        "k1": k1,
        "b": b,
        "config_hash": config_hash(corpus_hash, k1, b),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
