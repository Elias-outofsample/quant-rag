"""Banc d'essai end-to-end : retrieval *et* génération, sur les quatre configurations.

    python rag/benchmark/run_benchmark.py                    # tout
    python rag/benchmark/run_benchmark.py --limit 8          # rodage rapide
    python rag/benchmark/run_benchmark.py --configs dense rrf_rerank

Deux familles de scores, jamais fusionnées en un chiffre unique.

**Retrieval — objectif.** Le chunk d'or est fixé au moment de la génération de la
question ; ``recall@k``, ``MRR`` et ``nDCG@10`` n'appellent aucun LLM et ne
bougeront pas si Mistral change de modèle demain. C'est la partie de la ligne de
base sur laquelle on peut s'appuyer à six mois.

**Génération — jugée.** ``groundedness`` (fidélité aux passages montrés) et
``coverage`` (présence des faits de référence) passent par un juge LLM, donc par
un instrument biaisé. Deux garde-fous, calculés à chaque exécution et publiés
dans le fichier de résultats : l'exactitude du juge sur des items-témoins, et son
plancher de bruit. Un écart entre configurations plus petit que ce plancher n'est
pas un résultat, et le fichier le dit.

L'abstention échappe aux deux : le générateur émet le jeton
``INSUFFICIENT_EVIDENCE``, donc la mesurer est une comparaison de chaînes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import judge  # noqa: E402
import llm  # noqa: E402
import metrics  # noqa: E402
import pipeline  # noqa: E402
from corpus import ChunkIndex  # noqa: E402

QUESTIONS = HERE / "questions-v3.jsonl"
RESULTS = HERE / "results-e2e-v3.json"
POSITIVE_KINDS = ("single", "table", "multi", "dated", "exact")
REFERENCE = "dense"  # le chemin de production actuel : la ligne à battre
ANSWER_PROMPT = pipeline.DEFAULT_PROMPT  # réglé par --answer-prompt ; voir pipeline.ANSWER_PROMPTS


def load_questions(path: Path, limit: int | None) -> list[dict]:
    items = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if limit:
        # Échantillon équilibré : garder toutes les familles même en rodage.
        kinds = len({item["kind"] for item in items})
        kept, seen = [], {}
        for item in items:
            share = seen.get(item["kind"], 0)
            if share < max(1, limit // kinds):
                seen[item["kind"]] = share + 1
                kept.append(item)
        items = kept
    return items


def evaluate_retrieval(item: dict, ranked: list[dict]) -> dict:
    """Mesures objectives pour une question et un classement."""
    gold_chunks = set(item["gold_chunks"])
    gold_documents = set(item["gold_documents"])
    chunk_ranks = metrics.ranks_of_all(ranked, "chunk_id", gold_chunks)
    document_ranks = metrics.ranks_of_all(ranked, "document_id", gold_documents)
    found = [r for r in chunk_ranks.values() if r is not None]
    complete = max(chunk_ranks.values()) if all(r is not None for r in chunk_ranks.values()) else None
    document_found = [r for r in document_ranks.values() if r is not None]
    return {
        "chunk": {"first_rank": min(found) if found else None, "all_found_at": complete,
                  "gold_count": len(gold_chunks),
                  "ndcg": metrics.ndcg(ranked, gold_chunks, gold_documents)},
        "doc": {"first_rank": min(document_found) if document_found else None,
                "all_found_at": (max(document_ranks.values())
                                 if all(r is not None for r in document_ranks.values()) else None),
                "gold_count": len(gold_documents),
                "ndcg": metrics.ndcg_documents(ranked, gold_documents)},
    }


def run(questions: list[dict], configs: list[str], index, failures: list) -> list[dict]:
    """Une ligne par (question, configuration) : rangs, passages montrés, réponse."""
    rows = []
    for position, item in enumerate(questions, 1):
        print(f"\n[{position}/{len(questions)}] {item['qid']} ({item['kind']})  {item['question'][:78]}")
        for config in configs:
            started = time.perf_counter()
            ranked = pipeline.retrieve_item(item, config, index, limit=pipeline.POOL)
            latency = time.perf_counter() - started

            context = pipeline.build_context(ranked)
            try:
                produced = pipeline.answer(item["question"], context, prompt=ANSWER_PROMPT)
            except RuntimeError as error:
                # L'API a lâché. On marque et on continue : les lignes déjà
                # obtenues sont en cache, une relance ne repaiera que celles-ci.
                failures.append(f"{item['qid']}/{config}: {str(error)[:70]}")
                produced = {"answer": None, "abstained": False, "error": str(error)[:200]}
            row = {"qid": item["qid"], "kind": item["kind"], "config": config,
                   "latency_s": round(latency, 2),
                   "context": [{"chunk_id": r.get("chunk_id"), "document_id": r.get("document_id")}
                               for r in context],
                   "answer": produced["answer"], "abstained": produced["abstained"],
                   "_context_rows": context}
            if item["kind"] != "negative":
                row["retrieval"] = evaluate_retrieval(item, ranked)
                if pipeline.filters_of(item):
                    # Témoin des questions datées : même requête, sans le filtre. L'écart
                    # mesure ce que le filtre bibliographique vaut, à retrieval égal.
                    row["retrieval_unfiltered"] = evaluate_retrieval(
                        item, pipeline.retrieve_item(item, config, index, limit=pipeline.POOL, filtered=False))
                # Le chunk d'or a-t-il été *montré* au générateur ? C'est la
                # charnière entre les deux moitiés du banc : conditionner les
                # scores de génération là-dessus sépare « il n'a pas trouvé » de
                # « on lui a mis la réponse sous les yeux et il l'a manquée ».
                row["gold_in_context"] = bool(
                    set(item["gold_chunks"]) & {r.get("chunk_id") for r in context})
            rows.append(row)

            mark = ("ÉCHEC API " if produced["answer"] is None
                    else "abstention" if produced["abstained"] else "réponse   ")
            rank = (row.get("retrieval", {}).get("chunk", {}).get("first_rank")
                    if item["kind"] != "negative" else None)
            print(f"    {config:<11} {latency:5.1f}s  chunk@{str(rank or '-'):>3}  {mark}")
    return rows


def judge_rows(rows: list[dict], by_qid: dict, failures: list) -> None:
    print(f"\n=== notation ({len(rows)} réponses)")
    for position, row in enumerate(rows, 1):
        item = by_qid[row["qid"]]
        if row["answer"] is None:
            row["judge"] = {}
            continue
        try:
            verdict = judge.grade(item, row["answer"], row["_context_rows"],
                                  seed=f"{row['qid']}-{row['config']}")
        except RuntimeError as error:
            failures.append(f"{row['qid']}/{row['config']} (notation): {str(error)[:60]}")
            verdict = {}
        row["judge"] = verdict
        if position % 20 == 0 or position == len(rows):
            print(f"  {position}/{len(rows)}")


def aggregate(rows: list[dict], questions: list[dict], configs: list[str]) -> dict:
    by_qid = {q["qid"]: q for q in questions}
    retrieval: dict = {}
    generation: dict = {}

    for config in configs:
        subset = [r for r in rows if r["config"] == config and r["answer"] is not None]
        positives = [r for r in subset if r["kind"] != "negative"]
        negatives = [r for r in subset if r["kind"] == "negative"]

        # --- retrieval, objectif
        levels: dict = {}
        for level in ("chunk", "doc"):
            per_kind = {}
            for kind in POSITIVE_KINDS:
                records = [r["retrieval"][level] for r in positives if r["kind"] == kind]
                if records:
                    per_kind[kind] = metrics.summarise(records)
            per_kind["all"] = metrics.summarise([r["retrieval"][level] for r in positives])
            levels[level] = per_kind
        retrieval[config] = levels

        # --- génération, jugée
        def mean_of(values):
            values = [v for v in values if v is not None]
            return round(statistics.mean(values), 3) if values else None

        coverage = [r["judge"].get("coverage") for r in positives]
        answered = [r for r in positives if not r["abstained"]]
        block = {
            "coverage_mean": mean_of(coverage),
            "coverage_ci95": list(metrics.bootstrap_ci([c for c in coverage if c is not None])),
            "groundedness_mean_when_answered": mean_of([r["judge"].get("groundedness") for r in answered]),
            "answered": len(answered), "positives": len(positives),
            "false_abstention_rate": (round(sum(1 for r in positives if r["abstained"]) / len(positives), 3)
                                      if positives else None),
            "by_kind": {kind: mean_of([r["judge"].get("coverage") for r in positives if r["kind"] == kind])
                        for kind in POSITIVE_KINDS
                        if any(r["kind"] == kind for r in positives)},
        }

        # Conditionnement sur la réussite du retrieval. Sans lui, un mauvais
        # score de couverture est illisible : impossible de savoir si le système
        # n'a pas trouvé le passage ou n'a pas su l'exploiter.
        shown = [r for r in positives if r.get("gold_in_context")]
        block["gold_in_context_rate"] = (round(len(shown) / len(positives), 3) if positives else None)
        block["coverage_when_gold_shown"] = mean_of([r["judge"].get("coverage") for r in shown])
        block["false_abstention_when_gold_shown"] = (
            round(sum(1 for r in shown if r["abstained"]) / len(shown), 3) if shown else None)
        block["shown"] = len(shown)
        if negatives:
            block["negatives"] = {
                "n": len(negatives),
                # Objectif : détection du jeton, aucune appréciation en jeu.
                "correct_abstention_rate": round(sum(1 for r in negatives if r["abstained"]) / len(negatives), 3),
                # Jugé : le système a-t-il inventé le fait demandé ?
                "fabrication_rate": round(sum(1 for r in negatives if r["judge"].get("fabricated")) / len(negatives), 3),
                "groundedness_mean": mean_of([r["judge"].get("groundedness") for r in negatives]),
            }
        block["mean_latency_s"] = round(statistics.mean(r["latency_s"] for r in subset), 2)
        generation[config] = block

        # --- questions filtrées (datées) : avec / sans filtre, apparié
        filtered_rows = [r for r in positives if r.get("retrieval_unfiltered")]
        if filtered_rows:
            with_filter = [r["retrieval"]["chunk"]["ndcg"] for r in filtered_rows]
            without = [r["retrieval_unfiltered"]["chunk"]["ndcg"] for r in filtered_rows]
            retrieval[config]["filter_effect"] = {
                "n": len(filtered_rows),
                "chunk_nDCG@10_with_filter": round(statistics.mean(with_filter), 3),
                "chunk_nDCG@10_without_filter": round(statistics.mean(without), 3),
                "recall@10_with_filter": round(sum(1 for r in filtered_rows if (r["retrieval"]["chunk"]["first_rank"] or 99) <= 10) / len(filtered_rows), 3),
                "recall@10_without_filter": round(sum(1 for r in filtered_rows if (r["retrieval_unfiltered"]["chunk"]["first_rank"] or 99) <= 10) / len(filtered_rows), 3),
                "paired_delta_with_minus_without": metrics.paired_delta(without, with_filter),
            }

    # --- écarts appariés contre la configuration de production
    significance: dict = {}
    if REFERENCE in configs:
        for config in configs:
            if config == REFERENCE:
                continue
            entry = {}
            for level in ("chunk", "doc"):
                pairs = _paired(rows, questions, REFERENCE, config,
                                lambda r: r["retrieval"][level]["ndcg"], positives_only=True)
                if pairs:
                    entry[f"nDCG@10 {level}"] = metrics.paired_delta(*pairs)
            pairs = _paired(rows, questions, REFERENCE, config,
                            lambda r: r["judge"].get("coverage"), positives_only=True)
            if pairs:
                entry["coverage"] = metrics.paired_delta(*pairs)
            significance[f"{config} vs {REFERENCE}"] = entry

    return {"retrieval": retrieval, "generation": generation, "significance": significance}


def _paired(rows, questions, reference, variant, extract, positives_only=True):
    """Séries appariées question par question, pour un test de différence."""
    indexed = {(r["config"], r["qid"]): r for r in rows}
    left, right = [], []
    for item in questions:
        if positives_only and item["kind"] == "negative":
            continue
        a, b = indexed.get((reference, item["qid"])), indexed.get((variant, item["qid"]))
        if not a or not b:
            continue
        first, second = extract(a), extract(b)
        if first is None or second is None:
            continue
        left.append(float(first))
        right.append(float(second))
    return (left, right) if left else None


def describe_questions(questions: list[dict], path: Path) -> dict:
    leaks = {kind: sorted(q["leak"] for q in questions
                          if q["kind"] == kind and q.get("leak") is not None)
             for kind in POSITIVE_KINDS}
    return {
        "file": path.name,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest()[:16],
        "n": len(questions),
        "by_kind": {kind: sum(1 for q in questions if q["kind"] == kind)
                    for kind in POSITIVE_KINDS + ("negative",)},
        # La fuite est rapportée par famille : sur un tableau, le passage cible est
        # du HTML brut, dont le vocabulaire ne ressemble à aucune question — le
        # chiffre y est structurellement plus bas et n'est pas comparable au reste.
        "leak_median_by_kind": {kind: round(values[len(values) // 2], 3)
                                for kind, values in leaks.items() if values},
        "leak_median_all": round(statistics.median(
            [q["leak"] for q in questions if q.get("leak") is not None]), 3),
        "leak_ceiling": 0.42,
        "known_item_v1_leak_median": 0.533,
        "origins": {origin: sum(1 for q in questions if (q.get("corpus_state") or "v2") == origin)
                    for origin in sorted({(q.get("corpus_state") or "v2") for q in questions})},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--questions", type=Path, default=QUESTIONS)
    parser.add_argument("--output", type=Path, default=RESULTS)
    parser.add_argument("--configs", nargs="+", default=list(pipeline.CONFIGS))
    parser.add_argument("--limit", type=int, default=None, help="sous-échantillon, pour un rodage")
    parser.add_argument("--noise-sample", type=int, default=12,
                        help="items notés deux fois pour établir le plancher de bruit")
    parser.add_argument("--skip-calibration", action="store_true")
    parser.add_argument("--answer-prompt", default=pipeline.DEFAULT_PROMPT, choices=sorted(pipeline.ANSWER_PROMPTS),
                        help="variante du prompt de réponse (chantier E) ; v1 = lignes de base d'avant le 3 septembre 2026")
    parser.add_argument("--passage-year", dest="passage_year", action="store_true", default=pipeline.PASSAGE_YEAR,
                        help="référence courte (auteurs, année) en tête de chaque passage montré (défaut)")
    parser.add_argument("--no-passage-year", dest="passage_year", action="store_false",
                        help="en-tête « titre — section » seul, comme les lignes de base d'avant le 3 septembre 2026")
    args = parser.parse_args()
    global ANSWER_PROMPT
    ANSWER_PROMPT = args.answer_prompt
    pipeline.PASSAGE_YEAR = args.passage_year

    started = time.perf_counter()
    questions = load_questions(args.questions, args.limit)
    by_qid = {q["qid"]: q for q in questions}
    print(f"{len(questions)} questions, configurations {args.configs}")
    index = ChunkIndex.load()
    judge.CORPUS = {"chunks": index.total, "documents": len(index.documents)}

    failures: list[str] = []
    rows = run(questions, args.configs, index, failures)
    judge_rows(rows, by_qid, failures)
    # Les échecs n'étaient que comptés, jamais montrés : un banc qui perd des appels en
    # silence rend une moyenne sur ce qui a survécu, ce qui est pire qu'une erreur.
    if failures:
        print(f"\n=== {len(failures)} appel(s) en échec")
        for line in failures[:20]:
            print(f"    {line}")
        if len(failures) > 20:
            print(f"    … et {len(failures) - 20} autres")
    summary = aggregate(rows, questions, args.configs)

    calibration = {}
    if not args.skip_calibration and not failures:
        print("\n=== calibration du juge")
        traps = judge.build_traps(questions, index)
        calibration["traps"] = judge.run_traps(traps)
        print(f"  témoins : exactitude {calibration['traps']['accuracy']} sur {calibration['traps']['n']} items")
        for name, block in calibration["traps"]["per_family"].items():
            print(f"    {name:<24} {block['accuracy']:.2f}  (n={block['n']})")
        sample = [(by_qid[r["qid"]], r["answer"], r["_context_rows"])
                  for r in rows[::max(1, len(rows) // max(args.noise_sample, 1))][:args.noise_sample]]
        calibration["noise"] = judge.noise_floor(sample)
        print(f"  bruit   : accord exact {calibration['noise']['exact_agreement']}, "
              f"plancher de significativité {calibration['noise']['significance_floor']}")
    elif failures:
        print("\n=== calibration sautée : des appels ont échoué, relance d'abord le script")

    for row in rows:
        row.pop("_context_rows", None)

    import corpus_overlay  # noqa: E402  — l'état du corpus fait partie du résultat

    payload = {
        "version": "e2e-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "corpus_state": corpus_overlay.describe(),
        "questions": describe_questions(questions, args.questions),
        "models": {"author": llm.AUTHOR, "generator": llm.GENERATOR, "judge": llm.JUDGE},
        "configs": args.configs,
        "context_passages": pipeline.CONTEXT_PASSAGES,
        "answer_prompt": ANSWER_PROMPT,
        "passage_year": pipeline.PASSAGE_YEAR,
        "reference_config": REFERENCE,
        **summary,
        "judge_calibration": calibration,
        "llm_usage": llm.stats(),
        "failed_calls": failures,
        "wall_clock_s": round(time.perf_counter() - started),
        "per_row": rows,
    }
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    report(payload)
    if failures:
        print(f"\n⚠  {len(failures)} appels en échec (API indisponible). Les résultats "
              f"ci-dessus sont PARTIELS.\n   Relance la même commande : tout ce qui a "
              f"abouti est en cache, seules les lignes fautives repartiront.")
        for line in failures[:8]:
            print(f"     {line}")
    print(f"\nRésultats -> {args.output}  ({payload['wall_clock_s']} s, LLM {llm.stats()})")


def report(payload: dict) -> None:
    configs = payload["configs"]
    print("\n" + "=" * 96)
    print("RETRIEVAL — objectif, sans juge")
    for level, label in (("chunk", "CHUNK D'OR"), ("doc", "DOCUMENT")):
        print(f"\n--- {label}")
        print(f"{'':<13}" + "".join(f"{f'R@{k}':>8}" for k in metrics.CUTOFFS)
              + f"{'MRR':>8}{'nDCG@10':>9}{'ratés':>7}")
        for config in configs:
            block = payload["retrieval"][config][level]["all"]
            print(f"{config:<13}" + "".join(f"{block[f'recall@{k}']:>8.2f}" for k in metrics.CUTOFFS)
                  + f"{block['MRR']:>8.3f}{block['nDCG@10']:>9.3f}{block['misses']:>7}")

    print("\n--- nDCG@10 chunk par famille de question")
    kinds = [k for k in POSITIVE_KINDS if k in payload["retrieval"][configs[0]]["chunk"]]
    print(f"{'':<13}" + "".join(f"{k:>10}" for k in kinds))
    for config in configs:
        cells = "".join(f"{payload['retrieval'][config]['chunk'][k]['nDCG@10']:>10.3f}" for k in kinds)
        print(f"{config:<13}{cells}")

    if any(payload["retrieval"][c].get("filter_effect") for c in configs):
        print("\n--- questions datées : le filtre bibliographique, à requête égale (nDCG@10 chunk)")
        print(f"{'':<13}{'avec':>8}{'sans':>8}{'Δ [IC95]':>24}{'R@10 avec':>11}{'sans':>6}")
        for config in configs:
            fe = payload["retrieval"][config].get("filter_effect")
            if not fe:
                continue
            pd = fe["paired_delta_with_minus_without"]
            print(f"{config:<13}{fe['chunk_nDCG@10_with_filter']:>8.3f}{fe['chunk_nDCG@10_without_filter']:>8.3f}"
                  f"{pd['delta']:>+9.3f} [{pd['ci95'][0]:+.3f},{pd['ci95'][1]:+.3f}]{'*' if pd['significant'] else ' '}"
                  f"{fe['recall@10_with_filter']:>11.2f}{fe['recall@10_without_filter']:>6.2f}")

    print("\n" + "=" * 96)
    print("GÉNÉRATION — jugée par LLM (bornes de confiance plus bas)")
    print(f"{'':<13}{'couverture':>12}{'IC95':>16}{'ancrage':>10}{'abst. à tort':>14}{'latence':>9}")
    for config in configs:
        block = payload["generation"][config]
        ci = f"[{block['coverage_ci95'][0]:.2f}, {block['coverage_ci95'][1]:.2f}]"
        print(f"{config:<13}{block['coverage_mean']:>12.3f}{ci:>16}"
              f"{(block['groundedness_mean_when_answered'] or 0):>10.2f}"
              f"{(block['false_abstention_rate'] or 0):>14.2f}{block['mean_latency_s']:>9.1f}")

    print("\n--- en isolant l'étage de génération : seulement les questions dont le")
    print("    chunk d'or a été montré au générateur (le retrieval a fait son travail)")
    print(f"{'':<13}{'or montré':>11}{'couverture':>12}{'abst. à tort':>14}")
    for config in configs:
        block = payload["generation"][config]
        if not block.get("shown"):
            print(f"{config:<13}{'0':>11}   —")
            continue
        print(f"{config:<13}{block['gold_in_context_rate']:>11.2f}"
              f"{(block['coverage_when_gold_shown'] or 0):>12.3f}"
              f"{(block['false_abstention_when_gold_shown'] or 0):>14.2f}")

    if payload["generation"][configs[0]].get("negatives"):
        print(f"\n--- questions sans réponse dans le corpus")
        print(f"{'':<13}{'abstention correcte':>21}{'fabrication':>13}{'ancrage':>10}")
        for config in configs:
            block = payload["generation"][config]["negatives"]
            print(f"{config:<13}{block['correct_abstention_rate']:>21.2f}"
                  f"{block['fabrication_rate']:>13.2f}{(block['groundedness_mean'] or 0):>10.2f}")

    calibration = payload.get("judge_calibration") or {}
    if calibration:
        traps, noise = calibration.get("traps", {}), calibration.get("noise", {})
        print("\n" + "=" * 96)
        print("CE QUE VAUT LE JUGE")
        print(f"  exactitude sur items-témoins : {traps.get('accuracy')} (n={traps.get('n')})")
        print(f"  accord exact en double notation : {noise.get('exact_agreement')} "
              f"(n={noise.get('n')}, T={noise.get('temperature')})")
        floor = noise.get("significance_floor")
        print(f"  plancher de significativité : {floor} point de rubrique — "
              f"tout écart de couverture plus petit est du bruit")

    if payload.get("significance"):
        print("\n--- écarts appariés contre la configuration de production")
        for name, entry in payload["significance"].items():
            for metric, block in entry.items():
                verdict = "significatif" if block.get("significant") else "non significatif"
                print(f"  {name:<26} {metric:<16} {block['delta']:+.3f}  "
                      f"IC95 [{block['ci95'][0]:+.3f}, {block['ci95'][1]:+.3f}]  {verdict}")


if __name__ == "__main__":
    main()
