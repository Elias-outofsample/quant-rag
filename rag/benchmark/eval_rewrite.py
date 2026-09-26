"""Réécriture de requête par LLM et HyDE, contre le dense de production — chantier C.

Hypothèse testée : le routeur statistique ne sait pas reconnaître les six questions
« dense confiant et faux » ; un modèle de langage qui *comprend* la question peut-il
produire ce que l'embedding attend — une reformulation dans le vocabulaire du corpus,
ou un passage hypothétique qui lui ressemble ?

Configurations (toutes sur la matrice dense, aucune n'entre en production) :

    dense             la requête telle quelle — la ligne de base (identique au banc)
    dense_hyde        HyDE : un passage hypothétique de 80–150 mots écrit par Mistral,
                      embarqué comme un *document* (sans instruction de requête)
    dense_hyde_mix    moyenne des vecteurs requête + passage hypothétique
    dense_multi       RRF des classements de la requête et de 3 reformulations
                      (technique, mots-clés, autre angle)
    dense_multi_mix   moyenne des 4 vecteurs de requête (une seule recherche)
    dense_all         RRF requête + 3 reformulations + HyDE

Garde-fous : le LLM ne voit que la question — jamais le chunk d'or, ni le document, ni
les faits attendus ; pour une question datée, il voit le besoin d'information sans la
clause (ce que l'appelant soumet), et le filtre d'or est appliqué à toutes les
configurations. Chaque appel est en cache (``llm.py``) ; les réécritures sont écrites
dans le fichier de résultats pour inspection.

    .venv/bin/python rag/benchmark/eval_rewrite.py            # ~10 min la 1re fois (310 appels), puis secondes
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import corpus_overlay  # noqa: E402
import experiment  # noqa: E402
import llm  # noqa: E402
import pipeline  # noqa: E402
import quant_rag  # noqa: E402
from dense_matrix import Matrix  # noqa: E402

OUTPUT = HERE / "results-rewrite-v3.json"
CONFIGS = ("dense", "dense_hyde", "dense_hyde_mix", "dense_multi", "dense_multi_mix", "dense_all")
REWRITER = llm.GENERATOR  # mistral-small-latest

HYDE_SYSTEM = """You are a researcher in quantitative finance. Given a question, write the passage (80-150 words) that a research paper or textbook chapter answering it would contain. Use the field's standard terminology: name the models, estimators, quantities and results such a passage would name. Write only the passage — no preamble, no hedging, no mention that it is hypothetical."""

REWRITE_SYSTEM = """You turn a question into search queries for a corpus of quantitative-finance papers and textbooks (English). Return JSON: {"queries": [q1, q2, q3]} where
  q1 = a technical restatement of the question in the field's standard terminology;
  q2 = a compact keyword query (5-12 words) naming the specific concepts, models, estimators or quantities involved;
  q3 = a restatement from a different angle: the mechanism, result or method the question is really about.
Do not answer the question. Do not add facts that the question does not imply."""


def hyde(question: str) -> str:
    return llm.complete([{"role": "system", "content": HYDE_SYSTEM}, {"role": "user", "content": question}],
                        model=REWRITER, temperature=0.0, max_tokens=300).strip()


def rewrites(question: str) -> list[str]:
    parsed = llm.complete_json([{"role": "system", "content": REWRITE_SYSTEM}, {"role": "user", "content": question}],
                               model=REWRITER, temperature=0.0, max_tokens=300) or {}
    queries = [str(q).strip() for q in (parsed.get("queries") or []) if str(q).strip()]
    return queries[:3]


def document_vector(text: str) -> np.ndarray:
    """Le passage hypothétique est embarqué comme un document du corpus : sans instruction de requête."""
    return np.asarray(quant_rag.embedder().encode([text], normalize_embeddings=True, show_progress_bar=False)[0], dtype=np.float32)


def mean_vector(vectors: list[np.ndarray]) -> np.ndarray:
    v = np.mean(np.vstack(vectors), axis=0)
    return v / max(float(np.linalg.norm(v)), 1e-12)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--limit", type=int, help="rodage sur les n premières questions de chaque banc")
    args = parser.parse_args()

    started = time.perf_counter()
    items = experiment.load_items()
    if args.limit:
        items = [i for i in items if i["bench"] == "v1"][:args.limit] + [i for i in items if i["bench"] == "v3"][:args.limit]
    matrix = Matrix.load()
    print(f"{len(items)} questions · matrice {len(matrix.chunk_ids)} chunks · réécriture par {REWRITER}")

    per_question = []
    for position, item in enumerate(items, 1):
        query = pipeline.query_of(item)
        scope = experiment.scope_of(item)
        passage = hyde(query)
        alternatives = rewrites(query)
        q_vec = np.asarray(quant_rag.encode_query(query), dtype=np.float32)
        alt_vecs = [np.asarray(quant_rag.encode_query(q), dtype=np.float32) for q in alternatives]
        h_vec = document_vector(passage)

        rank_dense = matrix.search(q_vec, scope=scope)
        rank_alts = [matrix.search(v, scope=scope) for v in alt_vecs]
        rank_hyde = matrix.search(h_vec, scope=scope)
        rankings = {
            "dense": rank_dense,
            "dense_hyde": rank_hyde,
            "dense_hyde_mix": matrix.search(mean_vector([q_vec, h_vec]), scope=scope),
            "dense_multi": experiment.rrf([rank_dense] + rank_alts),
            "dense_multi_mix": matrix.search(mean_vector([q_vec] + alt_vecs), scope=scope),
            "dense_all": experiment.rrf([rank_dense] + rank_alts + [rank_hyde]),
        }
        row = {"bench": item["bench"], "key": item["key"], "qid": item["qid"], "kind": item.get("kind", "single"),
               "query": query, "rewrites": alternatives, "hyde": passage,
               **{config: experiment.measure(item, rows) for config, rows in rankings.items()}}
        per_question.append(row)
        print(f"  [{position}/{len(items)}] {item['key']:<8} dense@{str(row['dense']['first_rank'] or '-'):>3}"
              f"  hyde@{str(row['dense_hyde']['first_rank'] or '-'):>3}  multi@{str(row['dense_multi']['first_rank'] or '-'):>3}"
              f"  all@{str(row['dense_all']['first_rank'] or '-'):>3}   {llm.stats()['calls']} appels", flush=True)

    result = experiment.summarise(per_question, list(CONFIGS), reference="dense")
    result["wins_losses"] = {config: experiment.wins_losses(per_question, config, "dense") for config in CONFIGS if config != "dense"}
    experiment.print_summary(result, list(CONFIGS), "dense")
    for config in ("dense_hyde", "dense_multi", "dense_all"):
        wl = result["wins_losses"][config]
        print(f"\n=== {config} : {wl['n_wins']} gains nets, {wl['n_losses']} pertes nettes (|Δ nDCG| ≥ {wl['threshold']}) ===")
        for e in wl["wins"][:6]:
            print(f"  + {e['key']:<8} {e['kind']:<7} rang {e['rank_reference']} → {e['rank_config']}  Δ {e['delta_ndcg']:+.3f}")
        for e in wl["losses"][:6]:
            print(f"  - {e['key']:<8} {e['kind']:<7} rang {e['rank_reference']} → {e['rank_config']}  Δ {e['delta_ndcg']:+.3f}")

    payload = {"version": "rewrite-v1", "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "corpus_state": corpus_overlay.describe(), "rewriter": REWRITER,
               "prompts": {"hyde": HYDE_SYSTEM, "rewrite": REWRITE_SYSTEM}, "configs": list(CONFIGS), "reference": "dense",
               **result, "llm_usage": llm.stats(), "wall_clock_s": round(time.perf_counter() - started),
               "per_question": per_question}
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n-> {args.output}  ({payload['wall_clock_s']} s, LLM {llm.stats()})")


if __name__ == "__main__":
    main()
