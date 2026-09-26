"""Comparer deux exécutions du banc end-to-end, question par question.

Deux fichiers `results-e2e-*.json` produits par `run_benchmark.py` sur les mêmes
questions : le script apparie les questions par `qid`, sépare ce qui relève du
**retrieval** (objectif : rang de l'or, nDCG@10, recall, MRR, ratés) de ce qui relève de
la **génération** (jugé, donc bruité : couverture, ancrage, abstention, négatives), et
donne l'écart apparié avec IC95 bootstrap sur le nDCG@10 chunk.

    .venv/bin/python rag/benchmark/compare_e2e.py référence.json nouveau.json

L'écart de génération n'est pas accompagné d'un IC apparié : le juge note deux
*réponses différentes*, et la calibration du juge (§6) donne un plancher de bruit de
l'ordre de 0,1 sur la couverture. À lire comme un ordre de grandeur, pas comme un test.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import metrics  # noqa: E402

KINDS = ("single", "table", "multi", "dated", "exact")


def rows_of(payload: dict, config: str) -> dict[str, dict]:
    return {r["qid"]: r for r in payload["per_row"] if r["config"] == config}


def state(payload: dict) -> str:
    corpus = payload.get("corpus_state", {})
    titles = corpus.get("embedding_titles")
    return (f"{corpus.get('signature')} · {corpus.get('text_overrides')} textes surchargés · "
            f"titres propres : {'oui' if titles and titles.get('vectors_present') else 'non'} · "
            f"prompt {payload.get('answer_prompt')} · année {'oui' if payload.get('passage_year') else 'non'} · "
            f"{payload.get('generated_at')}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--config", default="dense")
    args = parser.parse_args()

    a, b = (json.loads(p.read_text(encoding="utf-8")) for p in (args.reference, args.candidate))
    rows_a, rows_b = rows_of(a, args.config), rows_of(b, args.config)
    common = [q for q in rows_a if q in rows_b]
    print(f"référence  {args.reference.name} : {state(a)}")
    print(f"candidat   {args.candidate.name} : {state(b)}")
    print(f"{len(common)} questions communes, configuration « {args.config} »")

    positives = [q for q in common if rows_a[q].get("retrieval") and rows_b[q].get("retrieval")]
    print(f"\n=== retrieval (chunk, {len(positives)} questions positives) ===")
    print(f"{'famille':<9}{'n':>4}{'nDCG A':>9}{'nDCG B':>9}{'Δ apparié [IC95]':>28}{'R@10 A':>9}{'R@10 B':>9}{'ratés A':>9}{'ratés B':>9}")
    for kind in ("all", *KINDS):
        subset = [q for q in positives if kind == "all" or rows_a[q]["kind"] == kind]
        if not subset:
            continue
        va = [rows_a[q]["retrieval"]["chunk"] for q in subset]
        vb = [rows_b[q]["retrieval"]["chunk"] for q in subset]
        pd = metrics.paired_delta([r["ndcg"] for r in va], [r["ndcg"] for r in vb])
        cell = f"{pd['delta']:+.3f} [{pd['ci95'][0]:+.3f},{pd['ci95'][1]:+.3f}]{'*' if pd['significant'] else ' '}"
        rec = lambda v: sum(1 for r in v if r["first_rank"] and r["first_rank"] <= 10) / len(v)  # noqa: E731
        miss = lambda v: sum(1 for r in v if not r["first_rank"])  # noqa: E731
        print(f"{kind:<9}{len(subset):>4}{sum(r['ndcg'] for r in va) / len(va):>9.3f}{sum(r['ndcg'] for r in vb) / len(vb):>9.3f}"
              f"{cell:>28}{rec(va):>9.2f}{rec(vb):>9.2f}{miss(va):>9}{miss(vb):>9}")

    print(f"\n=== génération (jugé — bruité, pas de test apparié) ===")
    ga, gb = a["generation"][args.config], b["generation"][args.config]
    fields = [("coverage_mean", "couverture (positives)"), ("coverage_when_gold_shown", "couverture, or montré"),
              ("groundedness_mean_when_answered", "ancrage quand répondu"), ("answered", "réponses (sur positives)"),
              ("false_abstention_rate", "abstention à tort (juge)"), ("gold_in_context_rate", "or montré au générateur"),
              ("false_abstention_when_gold_shown", "abstention à tort, or montré")]
    print(f"{'':<32}{'A':>10}{'B':>10}{'Δ':>10}")
    for key, label in fields:
        if key not in ga or key not in gb:
            continue
        print(f"{label:<32}{ga[key]:>10.3f}{gb[key]:>10.3f}{gb[key] - ga[key]:>+10.3f}")
    for key, label in (("correct_abstention_rate", "négatives : abstention"), ("fabrication_rate", "négatives : fabrication")):
        print(f"{label:<32}{ga['negatives'][key]:>10.3f}{gb['negatives'][key]:>10.3f}{gb['negatives'][key] - ga['negatives'][key]:>+10.3f}")

    changed = [(q, rows_a[q], rows_b[q]) for q in positives
               if rows_a[q]["gold_in_context"] != rows_b[q]["gold_in_context"]]
    print(f"\n=== l'or entre ou sort du contexte : {len(changed)} questions ===")
    for q, ra, rb in changed:
        print(f"  {q:<5} {ra['kind']:<7} or montré {str(ra['gold_in_context']):<5} → {str(rb['gold_in_context']):<5} "
              f"rang {str(ra['retrieval']['chunk']['first_rank'] or '-'):>3} → {str(rb['retrieval']['chunk']['first_rank'] or '-'):>3}"
              f"  couverture {ra['judge'].get('coverage')} → {rb['judge'].get('coverage')}")


if __name__ == "__main__":
    main()
