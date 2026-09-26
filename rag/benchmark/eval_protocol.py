"""Le protocole d'usage du RAG, mesuré : ce que l'appelant fait avant d'appeler search_documents — chantier A.

Les règles d'usage vivent aujourd'hui dans un prompt (agent ``rag-scout``) et dans les
descriptions des outils MCP. Ce harnais mesure chacune d'elles sur les bancs, sans LLM,
pour savoir lesquelles méritent d'être déplacées dans le code — où elles s'appliquent
sans qu'on ait à s'en souvenir.

Règle 1 — traduire une période explicite en filtre ``year_min`` / ``year_max`` :

    naive           la question brute, clause de date comprise, sans filtre — l'appelant
                    qui ne fait rien
    auto_period     la question brute passée par ``quant_rag.period_bounds`` : clause retirée
                    du texte, bornes appliquées — la règle **dans le code**
    ideal           ``retrieval_query`` + filtre d'or — l'appelant idéal (ce que le banc mesure)

Règle 2 — hybride seulement sur des identifiants exacts : lue dans le cache de
calibration (dense contre rrf_rerank sur v1 et sur la famille ``exact``), pour mémoire.

Règle 3 — croiser la recherche dense avec le graphe d'entités sur les questions
multi-documents :

    graph_rrf       RRF du classement dense et des passages qui mentionnent les entités
                    du graphe reconnues dans la question (noms d'au moins 4 caractères,
                    présents dans ≤ 400 passages)
    graph_boost     le pool dense reclassé : les passages qui mentionnent une entité de la
                    question passent devant, à ordre dense conservé

Règle 4 — reformuler 2 ou 3 fois les questions ouvertes : c'est le chantier C
(``eval_rewrite.py``), qui demande un LLM ; ses chiffres sont repris ici s'ils existent.

    .venv/bin/python rag/benchmark/eval_protocol.py
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import corpus_overlay  # noqa: E402
import experiment  # noqa: E402
import graph_search  # noqa: E402
import metrics  # noqa: E402
import pipeline  # noqa: E402
import quant_rag  # noqa: E402
from dense_matrix import Matrix  # noqa: E402

OUTPUT = HERE / "results-protocol-v3.json"
REWRITE = HERE / "results-rewrite-v3.json"
CONFIGS = ("naive", "auto_period", "ideal", "graph_rrf", "graph_boost")
MAX_ENTITY_CHUNKS = 400   # une entité citée dans plus de passages ne discrimine rien (« volatility », « Delta »)
MIN_NAME = 4


def query_entities(g: graph_search.Graph, query: str, limit: int = 6) -> list[dict]:
    """Entités du graphe dont le nom apparaît, mot entier, dans la question."""
    folded = f" {graph_search.fold(query)} "
    found = []
    for name, eid in g.names:
        if len(name) < MIN_NAME or f" {name} " not in folded:
            continue
        node = g.entities[eid]
        chunks = node.get("source_chunks") or []
        if not chunks or len(chunks) > MAX_ENTITY_CHUNKS:
            continue
        found.append({"id": eid, "name": node["name"], "label": node["label"], "key": name, "chunks": len(chunks)})
    # un nom contenu dans un nom plus long retenu n'apporte rien (« volatility » vs « rough volatility »)
    found.sort(key=lambda e: (-len(e["key"]), e["chunks"]))
    kept: list[dict] = []
    for e in found:
        if any(e["key"] != k["key"] and e["key"] in k["key"] for k in kept):
            continue
        if not any(e["key"] == k["key"] for k in kept):
            kept.append(e)
    return kept[:limit]


def graph_ranking(g: graph_search.Graph, entities: list[dict], scope: set[str] | None, pool: int) -> list[dict]:
    """Passages qui mentionnent les entités de la question : rareté × entités distinctes, puis mentions."""
    import math

    total = max(len(g.chunk_document), 1)
    score: dict[str, float] = {}
    hits: dict[str, int] = {}
    for e in entities:
        idf = math.log(total / e["chunks"])
        for cid in g.entities[e["id"]].get("source_chunks") or ():
            doc = g.chunk_document.get(cid)
            if scope is not None and doc not in scope:
                continue
            score[cid] = score.get(cid, 0.0) + idf * (1 + 0.1 * g.chunk_mentions.get(cid, {}).get(e["id"], 1))
            hits[cid] = hits.get(cid, 0) + 1
    ranked = sorted(score, key=lambda c: (-hits[c], -score[c], c))[:pool]
    return [{"chunk_id": c, "document_id": g.chunk_document.get(c), "score": score[c], "score_kind": "graph", "entities": hits[c]} for c in ranked]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()

    started = time.perf_counter()
    items = experiment.load_items()
    matrix = Matrix.load()
    g = graph_search.graph()
    cache = experiment.cached_rankings()
    print(f"{len(items)} questions · matrice {len(matrix.chunk_ids)} chunks · graphe {len(g.entities)} entités")

    per_question = []
    period_hits = {"dated_recovered": 0, "dated_total": 0, "false_positives": []}
    for position, item in enumerate(items, 1):
        raw = item["question"]
        gold_scope = experiment.scope_of(item)
        # règle 1
        detected = quant_rag.period_bounds(raw)
        if item.get("kind") == "dated":
            period_hits["dated_total"] += 1
            period_hits["dated_recovered"] += bool(detected) and {k: v for k, v in detected.items() if k in ("year_min", "year_max") and v is not None} == pipeline.filters_of(item)
        elif detected:
            period_hits["false_positives"].append({"key": item["key"], "clause": detected["clause"]})
        auto_scope = quant_rag.document_scope(None, year_min=detected["year_min"], year_max=detected["year_max"]) if detected else None
        rank_naive = matrix.search_text(raw)
        rank_auto = matrix.search_text(detected["query"], scope=auto_scope) if detected else rank_naive
        rank_ideal = matrix.search_text(pipeline.query_of(item), scope=gold_scope)
        # règle 3 : le graphe, sur la requête du banc et dans la portée d'or
        entities = query_entities(g, pipeline.query_of(item))
        scope_set = set(gold_scope) if gold_scope is not None else None
        rank_graph = graph_ranking(g, entities, scope_set, pipeline.POOL) if entities else []
        rank_graph_rrf = experiment.rrf([rank_ideal, rank_graph]) if rank_graph else rank_ideal
        mentioned = {r["chunk_id"] for r in rank_graph}
        rank_boost = ([r for r in rank_ideal if r["chunk_id"] in mentioned] + [r for r in rank_ideal if r["chunk_id"] not in mentioned]) if mentioned else rank_ideal
        rankings = {"naive": rank_naive, "auto_period": rank_auto, "ideal": rank_ideal,
                    "graph_rrf": rank_graph_rrf, "graph_boost": rank_boost}
        row = {"bench": item["bench"], "key": item["key"], "qid": item["qid"], "kind": item.get("kind", "single"),
               "period_detected": ({k: detected[k] for k in ("year_min", "year_max", "clause")} if detected else None),
               "entities": [e["name"] for e in entities], "graph_candidates": len(rank_graph),
               **{config: experiment.measure(item, rows) for config, rows in rankings.items()}}
        per_question.append(row)
        print(f"  [{position}/{len(items)}] {item['key']:<8} naive@{str(row['naive']['first_rank'] or '-'):>3} auto@{str(row['auto_period']['first_rank'] or '-'):>3}"
              f" ideal@{str(row['ideal']['first_rank'] or '-'):>3} graph_rrf@{str(row['graph_rrf']['first_rank'] or '-'):>3}"
              f" boost@{str(row['graph_boost']['first_rank'] or '-'):>3}  entités={row['entities'][:4]}", flush=True)

    result = experiment.summarise(per_question, list(CONFIGS), reference="ideal")
    experiment.print_summary(result, list(CONFIGS), "ideal")

    # règle 1, vue sur les questions datées seulement
    dated = [r for r in per_question if r["kind"] == "dated"]
    rule1 = {"n": len(dated), **period_hits,
             **{config: round(statistics.mean(r[config]["ndcg"] for r in dated), 3) for config in ("naive", "auto_period", "ideal")},
             "recall@10": {config: round(sum(1 for r in dated if (r[config]["first_rank"] or 99) <= 10) / len(dated), 3) for config in ("naive", "auto_period", "ideal")},
             "auto_period_vs_naive": metrics.paired_delta([r["naive"]["ndcg"] for r in dated], [r["auto_period"]["ndcg"] for r in dated]),
             "auto_period_vs_ideal": metrics.paired_delta([r["ideal"]["ndcg"] for r in dated], [r["auto_period"]["ndcg"] for r in dated])}
    print(f"\n=== règle 1 — période explicite → filtre (questions datées, n={rule1['n']}) ===")
    print(f"  clause retrouvée avec les bornes d'or : {rule1['dated_recovered']}/{rule1['dated_total']} ; faux positifs ailleurs : {len(rule1['false_positives'])}")
    print(f"  nDCG@10  naive {rule1['naive']:.3f}  auto_period {rule1['auto_period']:.3f}  ideal {rule1['ideal']:.3f}"
          f"   auto − naive {rule1['auto_period_vs_naive']['delta']:+.3f} {rule1['auto_period_vs_naive']['ci95']}"
          f"   auto − ideal {rule1['auto_period_vs_ideal']['delta']:+.3f}")

    # règle 2, depuis le cache : où l'hybride bat le dense
    rule2 = {}
    for label, keys in (("v1 (identifiants mémorisés)", [i["key"] for i in items if i["bench"] == "v1"]),
                        ("v3 exact (un jeton exact)", [i["key"] for i in items if i.get("kind") == "exact"]),
                        ("v3 table", [i["key"] for i in items if i.get("kind") == "table"]),
                        ("v3 tout", [i["key"] for i in items if i["bench"] == "v3"])):
        by_key = {i["key"]: i for i in items}
        dense = [experiment.measure(by_key[k], experiment.rows_from_cache(cache[k]["dense"]))["ndcg"] for k in keys]
        hybrid = [experiment.measure(by_key[k], experiment.rows_from_cache(cache[k]["rrf_rerank"]))["ndcg"] for k in keys]
        rule2[label] = {"n": len(keys), "dense": round(statistics.mean(dense), 3), "rrf_rerank": round(statistics.mean(hybrid), 3),
                        "hybrid_vs_dense": metrics.paired_delta(dense, hybrid)}
    print("\n=== règle 2 — hybride sur identifiants exacts seulement (cache Qdrant) ===")
    for label, b in rule2.items():
        pd = b["hybrid_vs_dense"]
        print(f"  {label:<30} n={b['n']:>3}  dense {b['dense']:.3f}  rrf_rerank {b['rrf_rerank']:.3f}  Δ {pd['delta']:+.3f} [{pd['ci95'][0]:+.3f},{pd['ci95'][1]:+.3f}]{'*' if pd['significant'] else ''}")

    # règle 3, vue multi
    with_entities = [r for r in per_question if r["entities"]]
    rule3 = {"questions_with_entities": len(with_entities), "of": len(per_question),
             "multi": result["by_kind"].get("multi"),
             "with_entities": {config: round(statistics.mean(r[config]["ndcg"] for r in with_entities), 3) for config in ("ideal", "graph_rrf", "graph_boost")} if with_entities else None,
             "wins_losses": {config: experiment.wins_losses(per_question, config, "ideal") for config in ("graph_rrf", "graph_boost")}}
    print(f"\n=== règle 3 — graphe × dense : {len(with_entities)}/{len(per_question)} questions portent une entité reconnue ===")
    if with_entities:
        print("  sur celles-ci, nDCG@10 : " + "  ".join(f"{c} {v:.3f}" for c, v in rule3["with_entities"].items()))
    for config in ("graph_rrf", "graph_boost"):
        wl = rule3["wins_losses"][config]
        print(f"  {config}: {wl['n_wins']} gains nets / {wl['n_losses']} pertes nettes"
              + ("  ex. " + ", ".join(f"{e['key']}({e['delta_ndcg']:+.2f})" for e in (wl['wins'][:3] + wl['losses'][:3])) if wl['wins'] or wl['losses'] else ""))

    rule4 = None
    if REWRITE.exists():
        rw = json.loads(REWRITE.read_text(encoding="utf-8"))
        rule4 = {"file": REWRITE.name, "summary_v3": {c: rw["summary"]["v3"][c]["nDCG@10"] for c in rw["configs"]},
                 "paired_v3": rw["paired"]["v3"], "paired_v1": rw["paired"]["v1"]}
        print(f"\n=== règle 4 — reformulations (chantier C, {REWRITE.name}) : nDCG v3 " + ", ".join(f"{c} {v:.3f}" for c, v in rule4["summary_v3"].items()))

    payload = {"version": "protocol-v1", "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "corpus_state": corpus_overlay.describe(), "configs": list(CONFIGS), "reference": "ideal",
               "rule_1_period_filter": rule1, "rule_2_hybrid_on_identifiers": rule2, "rule_3_graph": rule3, "rule_4_rewrite": rule4,
               **result, "wall_clock_s": round(time.perf_counter() - started), "per_question": per_question}
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n-> {args.output}  ({payload['wall_clock_s']} s)")


if __name__ == "__main__":
    main()
