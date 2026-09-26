"""Qwen3-Reranker à budget réduit : combien de candidats faut-il vraiment reranker ?

Sur le pool dense (top-50 de Qdrant, en cache), Qwen3-Reranker-0.6B gagne +0,070 nDCG@10
sur v3 (0,549 → 0,620) et +0,061 en pooled, mais coûte **28,8 s par requête** (50 paires,
127 s au pire) : inutilisable en interactif.

    Correction du 5 septembre 2026 : ces 28,8 s étaient contaminées par le swap
    (``logits`` matérialisés en [lot, longueur, vocabulaire], 1,5 Go par lot). Sur machine
    saine le même code mesure 17,6 s p50, et 15,0 s avec ``logits_to_keep=1``. La question
    posée ci-dessous reste entière : 15 s restent inutilisables en interactif, et seul le
    budget 10 descend à 3,2 s. Voir ``eval_rerank_latency.py``. La question posée ici est étroite : le gain
survit-il si l'on ne reranke que les *B* premiers candidats du pool ?

Protocole. Même pool, mêmes 155 questions (v1 25 + v3 130), mêmes vecteurs (titres
propres), même modèle, même format de prompt. Pour un budget *B* : les *B* premiers
chunks du pool sont reclassés par le reranker, les suivants gardent leur ordre dense
**derrière** eux — le pool n'est jamais tronqué, seule sa tête est réordonnée. B = 50
est le rerank complet, déjà mesuré (`results-rerankers-qwen-dense-v1.json`, repris du
cache). La latence est chronométrée par requête, à chaud, sur la seule tête.

Une variante gratuite est mesurée dans la foulée, sans GPU, depuis les mêmes scores :
`bN_rrf`, la fusion réciproque (k = 60) du pool dense **et** du classement du reranker.
Elle répond à la question que la frontière fait apparaître : le budget qui rattrape les
chunks d'or profonds est aussi celui qui dérange les questions déjà bien classées.

Conséquence à garder en tête pour la lecture : à B = 10, R@10 est *par construction*
celui du dense — seul l'ordre des dix premiers change (nDCG@10, MRR, R@1). Il faut
B > 10 pour qu'un chunk d'or profond remonte dans le top-10.

    .venv/bin/python rag/benchmark/eval_rerank_budget.py                 # budgets 10, 20, 30
    .venv/bin/python rag/benchmark/eval_rerank_budget.py --budgets 5 10

Chaque question mesurée est écrite dans `.cache/rerank-budget-partial.json` (mesure,
chronomètre et scores par chunk) : une interruption ne coûte que la question en cours,
et les scores permettent de *dériver* n'importe quel budget plus petit sans GPU. Le
script vérifie d'ailleurs cette dérivation contre les mesures directes.
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
import eval_rerankers  # noqa: E402
import experiment  # noqa: E402
import metrics  # noqa: E402
import pipeline  # noqa: E402
import quant_rag  # noqa: E402
from corpus import ChunkIndex  # noqa: E402

OUTPUT = HERE / "results-rerank-budget-v1.json"
PARTIAL = HERE / ".cache" / "rerank-budget-partial.json"
FULL_PARTIAL = HERE / ".cache" / "rerankers-partial.json"   # mesures B = 50 du chantier D
BUDGETS = (10, 20, 30)
POOL = "dense"


def ranked_with_budget(rows: list[dict], scores: dict[str, float], budget: int) -> list[dict]:
    """Tête reclassée par le reranker, queue du pool inchangée derrière."""
    head, tail = rows[:budget], rows[budget:]
    scored = sorted((r for r in head if r["chunk_id"] in scores),
                    key=lambda r: -scores[r["chunk_id"]])
    rest = [r for r in head if r["chunk_id"] not in scores]
    return scored + rest + tail


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", default="qwen3-0.6b", choices=list(eval_rerankers.MODELS))
    parser.add_argument("--budgets", nargs="+", type=int, default=list(BUDGETS))
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--limit", type=int, default=None, help="sous-échantillon, pour un rodage")
    parser.add_argument("--time-sample", type=int, default=0, metavar="N",
                        help="rechronométrer les N premières questions de chaque budget, GPU libre, sans toucher aux mesures")
    args = parser.parse_args()

    started = time.perf_counter()
    index = ChunkIndex.load(verbose=False)
    items = experiment.load_items(index)
    if args.limit:
        items = items[:args.limit]
    cache = experiment.cached_rankings()
    budgets = sorted(args.budgets)
    print(f"{len(items)} questions · modèle {args.model} · budgets {budgets} · pool {POOL} · device {quant_rag.device()}")

    pool: dict[str, list[dict]] = {}
    for item in items:
        rows = experiment.rows_from_cache(cache[item["key"]][POOL])
        for r in rows:
            src = index.get(r["chunk_id"])
            r["text"] = src["text"] if src else ""
        pool[item["key"]] = rows

    per_question = [{"bench": i["bench"], "key": i["key"], "qid": i["qid"], "kind": i.get("kind", "single"),
                     "none": experiment.measure(i, pool[i["key"]])} for i in items]

    # B = 50 : le rerank complet du chantier D, repris tel quel
    configs = ["none"]
    full = json.loads(FULL_PARTIAL.read_text(encoding="utf-8")) if FULL_PARTIAL.exists() else {}
    full_key = f"{args.model}/{POOL}"
    if full_key in full:
        for position, item in enumerate(items):
            per_question[position]["b50"] = full[full_key]["measures"][item["key"]]
        print(f"=== b50 : repris de {FULL_PARTIAL.name} ({full[full_key]['latency']})")

    partial = json.loads(PARTIAL.read_text(encoding="utf-8")) if PARTIAL.exists() else {}
    scorer = None
    latency: dict[str, dict] = {}
    for budget in budgets:
        name = f"b{budget}"
        configs.append(name)
        block = partial.setdefault(f"{args.model}/{name}", {"measures": {}, "times": {}, "scores": {}})
        todo = [i for i in items if i["key"] not in block["measures"]]
        if todo and scorer is None:
            t0 = time.perf_counter()
            scorer = eval_rerankers.load(args.model)
            print(f"\n=== {args.model} ({eval_rerankers.MODELS[args.model]}) chargé en {time.perf_counter() - t0:.0f} s")
            eval_rerankers.rerank(scorer, "warm-up", pool[items[0]["key"]][:8])
        if todo:
            print(f"\n=== budget {budget} — {len(todo)} questions à mesurer")
        for position, item in enumerate(items, 1):
            key = item["key"]
            if key in block["measures"]:
                continue
            head = [r for r in pool[key][:budget] if r.get("text")]
            query = pipeline.query_of(item)
            t1 = time.perf_counter()
            scores = scorer.score(query, [r["text"][:eval_rerankers.TEXT_CHARACTERS] for r in head])
            elapsed = time.perf_counter() - t1
            by_chunk = {r["chunk_id"]: float(s) for r, s in zip(head, scores)}
            ranked = ranked_with_budget(pool[key], by_chunk, budget)
            block["measures"][key] = experiment.measure(item, ranked)
            block["times"][key] = round(elapsed, 3)
            block["scores"][key] = {c: round(s, 6) for c, s in by_chunk.items()}
            PARTIAL.parent.mkdir(parents=True, exist_ok=True)
            PARTIAL.write_text(json.dumps(partial), encoding="utf-8")
            print(f"  b{budget:<3} [{position}/{len(items)}] {key:<8} {elapsed:5.2f}s  "
                  f"pool@{str(per_question[position - 1]['none']['first_rank'] or '-'):>3} → "
                  f"@{str(block['measures'][key]['first_rank'] or '-'):>3}", end="\r", flush=True)
        if todo:
            print()
        for position, item in enumerate(items):
            per_question[position][name] = block["measures"][item["key"]]
        times = [block["times"][i["key"]] for i in items]
        latency[name] = {"mean_s": round(statistics.mean(times), 2), "max_s": round(max(times), 2),
                         "median_s": round(statistics.median(times), 2), "pairs_per_query": budget}
    if full_key in full:
        latency["b50"] = full[full_key]["latency"]
        configs.append("b50")

    # fusion : le classement du reranker et le pool dense pèsent chacun la moitié (RRF k=60).
    # Rien à recalculer — les scores sont dans le cache — et rien de nouveau n'est demandé au GPU.
    for budget in budgets:
        name = f"b{budget}_rrf"
        configs.append(name)
        scores_of = partial[f"{args.model}/b{budget}"]["scores"]
        for position, item in enumerate(items):
            rows = pool[item["key"]]
            reranked = ranked_with_budget(rows, scores_of[item["key"]], budget)
            per_question[position][name] = experiment.measure(item, experiment.rrf([rows, reranked]))

    # latence propre : les mesures ci-dessus ont pu tourner GPU partagé (le banc end-to-end
    # encodait ses requêtes en même temps). Un rechronométrage sur un échantillon donne des
    # temps comparables entre budgets, dans les mêmes conditions.
    clean: dict[str, dict] = {}
    if args.time_sample:
        step = max(1, len(items) // args.time_sample)   # échantillon étalé sur les deux bancs et les cinq familles
        sample = items[::step][:args.time_sample]
        if scorer is None:
            scorer = eval_rerankers.load(args.model)
        eval_rerankers.rerank(scorer, "warm-up", pool[items[0]["key"]][:8])
        for budget in sorted({*budgets, 50}):   # 50 = le rerank complet, pour comparer dans les mêmes conditions
            times = []
            for item in sample:
                head = [r for r in pool[item["key"]][:budget] if r.get("text")]
                t1 = time.perf_counter()
                scorer.score(pipeline.query_of(item), [r["text"][:eval_rerankers.TEXT_CHARACTERS] for r in head])
                times.append(time.perf_counter() - t1)
            clean[f"b{budget}"] = {"mean_s": round(statistics.mean(times), 2), "max_s": round(max(times), 2),
                                   "median_s": round(statistics.median(times), 2), "pairs_per_query": budget,
                                   "n": len(sample)}
            print(f"  latence propre b{budget} sur {len(sample)} questions étalées : "
                  f"{clean[f'b{budget}']['mean_s']:.2f} s en moyenne, {clean[f'b{budget}']['max_s']:.2f} s au pire")

    # dérivation : le score d'un couple (requête, passage) est ponctuel — un budget plus
    # petit se déduit des scores d'un budget plus grand. On le vérifie contre les mesures.
    derivation = {}
    largest = max(budgets)
    big = partial[f"{args.model}/b{largest}"]["scores"]
    for budget in budgets[:-1]:
        agree = 0
        for item in items:
            scores = {c: s for c, s in big[item["key"]].items()}
            derived = experiment.measure(item, ranked_with_budget(pool[item["key"]], scores, budget))
            direct = partial[f"{args.model}/b{budget}"]["measures"][item["key"]]
            agree += abs(derived["ndcg"] - direct["ndcg"]) < 1e-9
        derivation[f"b{budget}"] = {"identical": agree, "n": len(items), "from": f"b{largest}"}
        print(f"  dérivation b{budget} depuis b{largest} : {agree}/{len(items)} nDCG identiques")

    result = experiment.summarise(per_question, configs, reference="none")
    result["paired_vs_b50"] = {}
    if "b50" in configs:
        result["paired_vs_b50"] = {
            bench: {c: metrics.paired_delta([r["b50"]["ndcg"] for r in per_question if bench == "pooled" or r["bench"] == bench],
                                            [r[c]["ndcg"] for r in per_question if bench == "pooled" or r["bench"] == bench])
                    for c in configs if c not in ("b50",)}
            for bench in experiment.BENCHES if any(bench == "pooled" or r["bench"] == bench for r in per_question)}
    result["wins_losses_vs_pool"] = {c: experiment.wins_losses(per_question, c, "none") for c in configs if c != "none"}

    print(f"\n################ pool {POOL} — « none » = le pool dense tel quel, bN = les N premiers reclassés")
    experiment.print_summary(result, configs, "none")
    if result["paired_vs_b50"]:
        print("\n=== contre le rerank complet b50 (nDCG@10 chunk) ===")
        for bench, block2 in result["paired_vs_b50"].items():
            for c, pd in block2.items():
                print(f"  {bench:<8}{c:<8}{pd['delta']:+.3f} [{pd['ci95'][0]:+.3f},{pd['ci95'][1]:+.3f}]{' *' if pd['significant'] else ''}")
    print("\n=== frontière qualité / latence ===" + (" (latences du rechronométrage)" if clean else ""))
    print(f"{'budget':<8}{'nDCG v1':>9}{'nDCG v3':>9}{'pooled':>9}{'R@10 v3':>9}{'ratés':>7}{'moy s':>8}{'méd s':>8}{'max s':>8}")
    for c in configs:
        lat = clean.get(c) or latency.get(c, {})
        cells = "".join(f"{lat[k]:>8.2f}" if k in lat else f"{'—':>8}" for k in ("mean_s", "median_s", "max_s"))
        print(f"{c:<8}{result['summary']['v1'][c]['nDCG@10']:>9.3f}{result['summary']['v3'][c]['nDCG@10']:>9.3f}"
              f"{result['summary']['pooled'][c]['nDCG@10']:>9.3f}{result['summary']['v3'][c]['recall@10']:>9.2f}"
              f"{result['summary']['pooled'][c]['misses']:>7}{cells}")

    payload = {"version": "rerank-budget-v1", "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "corpus_state": corpus_overlay.describe(), "model": eval_rerankers.MODELS[args.model],
               "settings": {"pool": POOL, "budgets": budgets, "tail": "ordre dense conservé derrière la tête reclassée",
                            "max_length": eval_rerankers.MAX_LENGTH, "text_characters": eval_rerankers.TEXT_CHARACTERS,
                            "batch": eval_rerankers.BATCH, "device": quant_rag.device(),
                            "qwen_instruction": eval_rerankers.QWEN_INSTRUCTION, "pool_source": str(experiment.CACHE.name)},
               "latency": latency, "latency_clean": clean, "derivation_check": derivation,
               "results": {**result, "per_question": per_question}, "wall_clock_s": round(time.perf_counter() - started)}
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n-> {args.output}  ({payload['wall_clock_s']} s)")


if __name__ == "__main__":
    main()
