"""Rejoue les deux bancs d'essai dans le *même* code, et compare leur difficulté.

Deux services en un, tous deux sans le moindre appel LLM :

1. **Contrôle de non-régression du pipeline.** Les chiffres v1 sont connus
   (`results-hybrid-v1.json`, produit par `eval_hybrid.py`). Les reproduire ici,
   avec le code de `pipeline.py`, atteste que les deux bancs mesurent bien la même
   chose et qu'un écart v1/v2 vient des *questions*, pas d'un bug de plomberie.
   C'est le contrôle qui a servi à valider l'effondrement de BM25 sur v2.

2. **Comparaison de difficulté.** Mêmes configurations, mêmes métriques, deux jeux
   de questions : l'écart chiffre ce que coûte le passage du known-item à des
   questions blanchies.

    .venv/bin/python rag/benchmark/compare_v1_v2.py
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import metrics  # noqa: E402
import pipeline  # noqa: E402
from corpus import ChunkIndex  # noqa: E402

#: Référence publiée dans results-hybrid-v1.json (eval_hybrid.py, pool 50).
V1_REFERENCE = {
    "dense": {"chunk_r1": 0.60, "doc_r1": 0.92},
    "bm25": {"chunk_r1": 0.44, "doc_r1": 0.76},
    "rrf": {"chunk_r1": 0.60, "doc_r1": 0.84},
    "rrf_rerank": {"chunk_r1": 0.80, "doc_r1": 0.96},
}


def load_v1(index: ChunkIndex, path: Path | None = None) -> list[dict]:
    """Met les questions v1 au format v2, pour les faire passer dans le même code.

    Une question dont le chunk cible a disparu du corpus est **conservée avec un or vide**
    — elle vaut alors 0 — au lieu d'être écartée. L'écarter changeait le dénominateur en
    silence : un re-découpage qui fait disparaître les ors des questions difficiles
    remontait la moyenne des survivantes sans qu'aucune qualité ait bougé. Mesuré le
    6 septembre 2026 sur la ligne de base réelle : retirer les 5 questions les plus dures
    des 155 rend un Δ apparent de **+0,0200**, le double du seuil tenu pour décisif.
    """
    items = []
    for line in (path or HERE / "questions-v1.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        cible = row.get("target_chunk")
        document = index.document_of(cible) if cible else None
        comparable = bool(row.get("comparable", True)) and document is not None
        if not comparable:
            print(f"  ! {row['qid']} : chunk cible absent du corpus — conservé à 0, "
                  f"le dénominateur ne bouge pas")
        items.append({"qid": row["qid"], "kind": "single", "question": row["question"],
                      "gold_chunks": [cible] if comparable else [],
                      "gold_documents": [document] if comparable else [],
                      "gold_poids": row.get("gold_poids") if comparable else None,
                      "comparable": comparable})
    return items


def load_bench(path: Path) -> list[dict]:
    """Les questions positives d'un banc au format v2/v3 (avec leurs filtres éventuels)."""
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and json.loads(line)["kind"] != "negative"]


def load_v2() -> list[dict]:
    return load_bench(HERE / "questions-v2.jsonl")


def load_v3() -> list[dict]:
    return load_bench(HERE / "questions-v3.jsonl")


def score(items: list[dict], index: ChunkIndex) -> dict:
    out: dict = {}
    for config in pipeline.CONFIGS:
        chunk_records, document_records = [], []
        for item in items:
            ranked = pipeline.retrieve_item(item, config, index, limit=pipeline.POOL)
            gold_chunks, gold_documents = set(item["gold_chunks"]), set(item["gold_documents"])
            chunk_records.append({
                "first_rank": metrics.rank_of(ranked, "chunk_id", gold_chunks),
                "all_found_at": None, "gold_count": len(gold_chunks),
                "ndcg": metrics.ndcg(ranked, gold_chunks, gold_documents)})
            document_records.append({
                "first_rank": metrics.rank_of(ranked, "document_id", gold_documents),
                "all_found_at": None, "gold_count": len(gold_documents),
                "ndcg": metrics.ndcg_documents(ranked, gold_documents)})
        out[config] = {"chunk": metrics.summarise(chunk_records),
                       "doc": metrics.summarise(document_records)}
        print(f"    {config} fait")
    return out


def main():
    index = ChunkIndex.load()
    v1, v2 = load_v1(index), load_v2()
    print(f"\nv1 : {len(v1)} questions known-item   |   v2 : {len(v2)} questions générées (hors négatives)")

    print("\n  v1…")
    v1_scores = score(v1, index)
    print("  v2…")
    v2_scores = score(v2, index)

    print("\n--- contrôle : les chiffres v1 sont-ils reproduits ?")
    print(f"{'':<13}{'R@1 chunk':>22}{'R@1 doc':>22}")
    print(f"{'':<13}{'obtenu':>10}{'attendu':>12}{'obtenu':>10}{'attendu':>12}")
    deviation = 0.0
    for config in pipeline.CONFIGS:
        if config not in V1_REFERENCE:
            continue  # configurations postérieures à la référence publiée (router)
        got_chunk = v1_scores[config]["chunk"]["recall@1"]
        got_doc = v1_scores[config]["doc"]["recall@1"]
        want = V1_REFERENCE[config]
        deviation = max(deviation, abs(got_chunk - want["chunk_r1"]), abs(got_doc - want["doc_r1"]))
        print(f"{config:<13}{got_chunk:>10.2f}{want['chunk_r1']:>12.2f}"
              f"{got_doc:>10.2f}{want['doc_r1']:>12.2f}")
    verdict = ("pipeline conforme" if deviation <= 0.05 else
               f"ÉCART DE {deviation:.2f} — le pipeline ne reproduit pas la référence")
    print(f"  écart maximal {deviation:.2f} -> {verdict}")

    print("\n--- difficulté : v1 (known-item) contre v2 (générées, blanchies)")
    print(f"{'':<13}{'nDCG@10 chunk':>26}{'nDCG@10 doc':>26}")
    print(f"{'':<13}{'v1':>8}{'v2':>9}{'écart':>9}{'v1':>8}{'v2':>9}{'écart':>9}")
    for config in pipeline.CONFIGS:
        row = ""
        for level in ("chunk", "doc"):
            a = v1_scores[config][level]["nDCG@10"]
            b = v2_scores[config][level]["nDCG@10"]
            row += f"{a:>8.3f}{b:>9.3f}{b - a:>+9.3f}"
        print(f"{config:<13}{row}")

    output = HERE / "results-difficulty-v1-vs-v2.json"
    output.write_text(json.dumps({
        "v1": {"n": len(v1), "scores": v1_scores},
        "v2": {"n": len(v2), "scores": v2_scores},
        "pipeline_check": {"max_deviation_vs_published_v1": round(deviation, 3),
                           "reference": V1_REFERENCE, "conforme": deviation <= 0.05},
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n-> {output}")


if __name__ == "__main__":
    main()
