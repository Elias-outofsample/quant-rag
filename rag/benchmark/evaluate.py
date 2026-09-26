"""Known-item retrieval benchmark: mesure recall@k et MRR, dense seul vs dense+rerank.

Protocole : chaque question a été écrite à partir d'un passage précis du corpus.
Le passage source est la cible ; on mesure à quel rang le système le retrouve.
Deux granularités :
  - strict   : le chunk exact doit remonter ;
  - document : n'importe quel passage du bon document compte (tolère le chunking).
"""
from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import quant_rag  # noqa: E402

POOL = 50
CUTOFFS = (1, 3, 5, 10)


def target_documents(questions):
    """chunk_id -> document_id, en une passe sur rows.jsonl."""
    wanted = {q["target_chunk"] for q in questions}
    mapping = {}
    path = quant_rag.ROOT / "data" / "embeddings" / "ingested-all-qwen3-06b" / "rows.jsonl"
    for line in path.open(encoding="utf-8"):
        row = json.loads(line)
        chunk_id = row["chunk"]["chunk_id"]
        if chunk_id in wanted:
            mapping[chunk_id] = row["chunk"]["document_id"]
    return mapping


def rank_of(rows, key, value):
    for index, row in enumerate(rows, 1):
        if row.get(key) == value:
            return index
    return None


def summarise(ranks):
    hits = [r for r in ranks if r is not None]
    out = {f"recall@{k}": sum(1 for r in hits if r <= k) / len(ranks) for k in CUTOFFS}
    out["MRR"] = sum(1 / r for r in hits) / len(ranks)
    out["median_rank"] = statistics.median(hits) if hits else None
    out["misses"] = len(ranks) - len(hits)
    return out


def main():
    questions = [json.loads(line) for line in (HERE / "questions-v1.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    doc_of = target_documents(questions)
    missing = [q["qid"] for q in questions if q["target_chunk"] not in doc_of]
    if missing:
        sys.exit(f"chunk cible introuvable dans le corpus: {missing}")

    results = {"dense": {"chunk": [], "doc": []}, "rerank": {"chunk": [], "doc": []}}
    per_question = []
    timings = {"dense": [], "rerank": []}

    for q in questions:
        target_chunk, target_doc = q["target_chunk"], doc_of[q["target_chunk"]]
        row = {"qid": q["qid"]}
        for mode in ("dense", "rerank"):
            started = time.perf_counter()
            hits = quant_rag.search(
                q["question"], limit=max(CUTOFFS), pool=POOL, mode="dense", rerank=(mode == "rerank"),
                dedupe=False, per_document=0, min_characters=0, log=False,
            )
            timings[mode].append(time.perf_counter() - started)
            chunk_rank = rank_of(hits, "chunk_id", target_chunk)
            doc_rank = rank_of(hits, "document_id", target_doc)
            results[mode]["chunk"].append(chunk_rank)
            results[mode]["doc"].append(doc_rank)
            row[mode] = {"chunk_rank": chunk_rank, "doc_rank": doc_rank}
        per_question.append(row)
        print(f"  {q['qid']}  dense chunk={row['dense']['chunk_rank'] or '-':>3} doc={row['dense']['doc_rank'] or '-':>3}"
              f"   |   rerank chunk={row['rerank']['chunk_rank'] or '-':>3} doc={row['rerank']['doc_rank'] or '-':>3}")

    print(f"\n{len(questions)} questions, pool={POOL}\n")
    header = f"{'':<12}" + "".join(f"{f'R@{k}':>9}" for k in CUTOFFS) + f"{'MRR':>9}{'rg. med.':>10}{'ratés':>7}"
    for level, label in (("chunk", "CHUNK EXACT"), ("doc", "DOCUMENT")):
        print(f"--- {label}")
        print(header)
        for mode in ("dense", "rerank"):
            s = summarise(results[mode][level])
            line = f"{mode:<12}" + "".join(f"{s[f'recall@{k}']:>9.2f}" for k in CUTOFFS)
            print(line + f"{s['MRR']:>9.3f}{str(s['median_rank'] or '-'):>10}{s['misses']:>7}")
        print()

    print(f"latence moyenne : dense {statistics.mean(timings['dense'])*1000:.0f} ms | rerank {statistics.mean(timings['rerank'])*1000:.0f} ms")
    (HERE / "results-v1.json").write_text(json.dumps({
        "pool": POOL, "questions": len(questions), "per_question": per_question,
        "summary": {mode: {level: summarise(results[mode][level]) for level in ("chunk", "doc")} for mode in ("dense", "rerank")},
    }, indent=2), encoding="utf-8")
    print(f"\nDétail écrit dans {HERE / 'results-v1.json'}")


if __name__ == "__main__":
    main()
