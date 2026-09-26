"""Arbitrage de la fusion dense + BM25 sur les deux bancs gelés, sans aucun appel LLM.

Question posée : sur le corpus servi aujourd'hui, la fusion d'un classement dense et
d'un classement BM25 bat-elle le dense seul — et si oui, sous quelle fusion ?

Ce que ce script **n'est pas**. La ligne de base connue oppose ``dense`` à
``rrf_rerank`` : fusion *et* cross-encoder mélangés. Elle ne dit donc rien de la
fusion seule. Ici le reranker est absent de bout en bout : la seule chose qui varie
entre deux colonnes du tableau est la règle de fusion.

Provenance des candidats — identique pour toutes les configurations :

    dense     le cache de calibration ``.cache/router-retrievals-<signature>.json``,
              produit par ``pipeline.retrieve_item(item, "dense", …)``. Même texte de
              requête, même vecteur, mêmes paramètres Qdrant, mêmes filtres
              bibliographiques que la ligne de base enregistrée — et c'est *elle*,
              au sens strict : le contrôle A le vérifie chiffre par chiffre.
    lexical   ``quant_rag._lexical()`` recalculé. BM25 est un fichier déterministe,
              aucun modèle n'est chargé ; le contrôle B vérifie que ce recalcul
              redonne bien ce que le pipeline avait mis en cache.

Deux contrôles, qu'on fait échouer avant de les croire (``--sabotage``) :

    A  les chiffres ``dense`` reproduisent ``results-router-<banc>.json`` sur les
       quatre vues. Sinon : arrêt. Une grille de fusion posée sur une ligne de base
       qu'on n'arrive pas à reproduire ne mesure rien.
    B  le ``bm25`` du cache est une **sous-suite** de mon lexical recalculé — le
       pipeline n'en retire que les chunks absents de ``ChunkIndex`` (``_with_text``),
       jamais ne les réordonne. Et ma RRF k=60 retrouve le ``rrf`` du cache.

    .venv/bin/python rag/benchmark/eval_fusion.py
    .venv/bin/python rag/benchmark/eval_fusion.py --sabotage   # le contrôle A doit hurler
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))   # rag/ : corpus_overlay, quant_rag
sys.path.insert(0, str(ROOT / "src"))  # fusions amont, inchangées

import corpus_overlay  # noqa: E402
import metrics  # noqa: E402
import pipeline  # noqa: E402
import quant_rag  # noqa: E402
from compare_v1_v2 import load_bench, load_v1  # noqa: E402
from corpus import ChunkIndex  # noqa: E402
from retrieval.fusion import weighted_fusion  # noqa: E402
from retrieval.hybrid import reciprocal_rank_fusion  # noqa: E402

CACHE = HERE / ".cache" / f"router-retrievals-{corpus_overlay.signature()}.json"
OPEN_BENCH = "v3"

#: Profondeurs évaluées. 30 est ce que décrit ``config/hybrid-bm25-v1.json``
#: (dense_depth 30, bm25_depth 30) ; 50 est la convention des bancs du dépôt
#: (``pipeline.POOL``), et la profondeur à laquelle la ligne de base a été mesurée.
DEPTHS = (30, 50)

#: ``recall@30`` est demandé par le cahier des charges ; ``metrics.CUTOFFS`` s'arrête à 10.
#: On le calcule ici plutôt que d'élargir une constante dont dépendent six autres scripts.
EXTRA_CUTOFF = 30

#: Les cinq fusions à trancher, plus les deux références.
#: ``rrf60`` sans pondération est la fusion que le dépôt applique déjà (``pipeline`` config
#: « rrf », ``quant_rag._hybrid``) : elle est dans la grille pour être jugée comme les autres.
FUSIONS = {
    "rrf20":  lambda d, l: reciprocal_rank_fusion(d, l, rrf_k=20),
    "rrf60":  lambda d, l: reciprocal_rank_fusion(d, l, rrf_k=60),
    # La grille de ``config/hybrid-bm25-v1.json`` pondère les *scores* (``weighted``) mais
    # jamais les *rangs* : sa RRF donne au rang 1 de BM25 exactement le poids du rang 1 du
    # dense, quelle que soit la qualité de BM25 sur cette requête. ``reciprocal_rank_fusion``
    # accepte pourtant des poids. Sans ces deux lignes, un « vous n'avez pas essayé
    # l'évident » resterait légitime — et il séparerait deux causes que la grille confond :
    # la pondération, et le fait de fusionner sur des rangs plutôt que sur des scores.
    "rrf20p": lambda d, l: reciprocal_rank_fusion(d, l, rrf_k=20, dense_weight=0.75, lexical_weight=0.25),
    "rrf60p": lambda d, l: reciprocal_rank_fusion(d, l, rrf_k=60, dense_weight=0.75, lexical_weight=0.25),
    "w75_25": lambda d, l: weighted_fusion(d, l, 0.75, 0.25, limit=len(d) + len(l)),
    "w60_40": lambda d, l: weighted_fusion(d, l, 0.60, 0.40, limit=len(d) + len(l)),
    "w50_50": lambda d, l: weighted_fusion(d, l, 0.50, 0.50, limit=len(d) + len(l)),
}
BASELINE = "dense"
REFERENCES = ("dense", "bm25")
PRIMARY = ("recall@5", "recall@10", "MRR", "nDCG@10")


# --------------------------------------------------------------------------- données

def rows_of(triples: list) -> list[dict]:
    """``[chunk_id, document_id, score]`` du cache -> les dicts qu'attendent les fusions."""
    return [{"chunk_id": c, "document_id": d, "score": float(s)} for c, d, s in triples]


def lexical_of(item: dict, depth: int) -> list[dict]:
    """Le classement BM25 brut, tel que ``pipeline.retrieve`` le construit — sans ``_with_text``.

    Le pipeline, lui, retire ensuite les candidats absents de ``ChunkIndex``. On garde la
    liste entière : un chunk que la vue corpus ignore est quand même dans la collection,
    et il concourt. Le contrôle B mesure l'écart que cela fait.
    """
    filters = pipeline.filters_of(item)
    scope = quant_rag.document_scope(None, **filters) if filters else None
    return quant_rag._lexical(pipeline.query_of(item), depth, scope)


def measure(item: dict, ranked: list[dict]) -> dict:
    gold_chunks, gold_documents = set(item["gold_chunks"]), set(item["gold_documents"])
    return {
        "first_rank": metrics.rank_of(ranked, "chunk_id", gold_chunks),
        "all_found_at": None,
        "gold_count": len(gold_chunks),
        "ndcg": metrics.ndcg(ranked, gold_chunks, gold_documents),
        "doc_rank": metrics.rank_of(ranked, "document_id", gold_documents),
        "doc_ndcg": metrics.ndcg_documents(ranked, gold_documents),
    }


def summarise(records: list[dict]) -> dict:
    """``metrics.summarise`` plus les deux colonnes qui lui manquent ici."""
    out = metrics.summarise(records)
    total = len(records)
    hits = [r["first_rank"] for r in records if r["first_rank"] is not None]
    out[f"recall@{EXTRA_CUTOFF}"] = round(sum(1 for r in hits if r <= EXTRA_CUTOFF) / total, 3)
    out["doc_nDCG@10"] = round(statistics.mean(r["doc_ndcg"] for r in records), 3)
    out["doc_recall@10"] = round(sum(1 for r in records if r["doc_rank"] and r["doc_rank"] <= 10) / total, 3)
    return out


# --------------------------------------------------------------------------- contrôles

def controle_a(summary: dict, bench: str, views: list[str]) -> list[str]:
    """La ligne de base enregistrée, chiffre par chiffre. Le contrôle qui autorise le reste."""
    recorded = HERE / ("results-router-v1.json" if bench == "v2" else f"results-router-{bench}.json")
    if not recorded.exists():
        return [f"{recorded.name} absent : impossible de vérifier la ligne de base"]
    reference = json.loads(recorded.read_text(encoding="utf-8"))
    if reference["corpus"]["signature"] != corpus_overlay.signature():
        return [f"{recorded.name} mesuré sur la signature {reference['corpus']['signature']}, "
                f"le corpus est à {corpus_overlay.signature()}"]
    ecarts = []
    for view in views:
        attendu = reference["summary"].get(view, {}).get("dense")
        obtenu = summary[view][BASELINE]
        if attendu is None:
            ecarts.append(f"vue {view} absente de {recorded.name}")
            continue
        for key, value in attendu.items():
            if key in ("recall@30",):
                continue
            if key not in obtenu:
                ecarts.append(f"{view}/{key} : absent de la mesure")
            elif abs(obtenu[key] - value) > 1e-9:
                ecarts.append(f"{view}/{key} : {obtenu[key]} au lieu de {value}")
    return ecarts


def controle_b(cache: dict, lexical: dict, fused60: dict, index_alternatif: bool = False) -> dict:
    """Le recalcul BM25 et la fusion redonnent-ils ce que le pipeline avait mis en cache ?

    Avec un index de l'ombre (``--bm25``), la divergence est **attendue** : le cache a été
    produit sur l'index servi. Le contrôle change alors de sens — il ne prouve plus la
    fidélité du recalcul (elle l'a été sur l'index de production, une fois pour toutes),
    il mesure combien de questions le changement de titres déplace réellement.
    """
    def sous_suite(petit: list[str], grand: list[str]) -> bool:
        it = iter(grand)
        return all(any(x == y for y in it) for x in petit)

    bm25_ok = bm25_total = rrf_ok = rrf_total = 0
    retires = 0
    for key, entry in cache.items():
        if key not in lexical:
            continue
        cache_bm25 = [c for c, _, _ in entry["bm25"]]
        mine = [r["chunk_id"] for r in lexical[key]]
        bm25_total += 1
        bm25_ok += (mine[:10] == cache_bm25[:10]) if index_alternatif else sous_suite(cache_bm25, mine)
        retires += len(mine) - len(cache_bm25)
        cache_rrf = [c for c, _, _ in entry["rrf"]][:10]
        rrf_total += 1
        rrf_ok += [r["chunk_id"] for r in fused60[key]][:10] == cache_rrf
    if index_alternatif:
        return {"index": "de l'ombre — la divergence au cache est attendue et c'est la mesure",
                "questions_dont_le_top10_bm25_change": f"{bm25_total - bm25_ok}/{bm25_total}",
                "questions_dont_le_top10_rrf60_change": f"{rrf_total - rrf_ok}/{rrf_total}"}
    return {"bm25_sous_suite": f"{bm25_ok}/{bm25_total}",
            "bm25_candidats_retires_par_with_text": retires,
            "rrf60_top10_identique_au_cache": f"{rrf_ok}/{rrf_total}"}


# --------------------------------------------------------------------------- programme

def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--bench", default=OPEN_BENCH, choices=("v2", "v3"))
    parser.add_argument("--sabotage", action="store_true",
                        help="décale le classement dense d'un rang : le contrôle A doit refuser")
    parser.add_argument("--bm25", help="index BM25 alternatif (défaut : celui de la production)")
    parser.add_argument("--tag", default="v1", help="suffixe du fichier de résultats")
    args = parser.parse_args()
    open_bench = args.bench
    sortie = HERE / f"results-fusion-{args.tag}.json"

    if not CACHE.exists():
        sys.exit(f"cache de classements absent ({CACHE.name}) : lance calibrate_router.py")
    cache = json.loads(CACHE.read_text(encoding="utf-8"))
    index = ChunkIndex.load(verbose=True)
    if args.bm25:
        from retrieval.lexical import BM25Index

        alternatif = BM25Index.load(Path(args.bm25))
        # ``_lexical`` résout ``bm25()`` à l'appel : remplacer l'attribut de module suffit,
        # et rien hors de ce processus n'en sait quoi que ce soit.
        quant_rag.bm25 = lambda _index=alternatif: _index
        bm25 = alternatif
        print(f"  BM25    {Path(args.bm25).name} : {bm25.doc_count} chunks  (index de l'ombre)")
    else:
        bm25 = quant_rag.bm25()
        print(f"  BM25    {quant_rag.bm25_path().name} : {bm25.doc_count} chunks")
    v2_qids = {item["qid"] for item in load_bench(HERE / "questions-v2.jsonl")} if open_bench == "v3" else set()

    configs = list(REFERENCES) + [f"{name}@{d}" for d in DEPTHS for name in FUSIONS]
    per_question: list[dict] = []
    lexical_cache: dict[str, list[dict]] = {}
    fused60_cache: dict[str, list[dict]] = {}
    latence = {"bm25_ms": [], "fusion_ms": []}

    for bench, items in (("v1", load_v1(index)), (open_bench, load_bench(HERE / f"questions-{open_bench}.jsonl"))):
        print(f"{bench} : {len(items)} questions")
        for item in items:
            key = f"{bench}/{item['qid']}"
            if key not in cache:
                sys.exit(f"{key} absent du cache : lance calibrate_router.py --bench {open_bench}")
            dense = rows_of(cache[key]["dense"])
            if args.sabotage:
                dense = dense[1:] + dense[:1]

            started = time.perf_counter()
            lexical_full = lexical_of(item, max(DEPTHS))
            latence["bm25_ms"].append((time.perf_counter() - started) * 1000)
            lexical_cache[key] = lexical_full

            ranked = {"dense": dense, "bm25": lexical_full}
            started = time.perf_counter()
            for depth in DEPTHS:
                d, l = dense[:depth], lexical_full[:depth]
                for name, fuse in FUSIONS.items():
                    ranked[f"{name}@{depth}"] = fuse(d, l)
            latence["fusion_ms"].append((time.perf_counter() - started) * 1000 / (len(DEPTHS) * len(FUSIONS)))
            fused60_cache[key] = ranked[f"rrf60@{max(DEPTHS)}"]

            per_question.append({
                "bench": bench, "qid": item["qid"], "kind": item.get("kind", "single"),
                "in_v2": bench == "v1" or item["qid"] in v2_qids or open_bench == "v2",
                "filters": pipeline.filters_of(item) or None,
                "n_dense": len(dense), "n_lexical": len(lexical_full),
                **{config: measure(item, rows) for config, rows in ranked.items()},
            })
            print(f"  {item['qid']:<4} dense@{str(per_question[-1]['dense']['first_rank'] or '-'):>3}"
                  f"  bm25@{str(per_question[-1]['bm25']['first_rank'] or '-'):>3}"
                  f"  rrf60@{str(per_question[-1][f'rrf60@{max(DEPTHS)}']['first_rank'] or '-'):>3}", end="\r")
    print(" " * 78, end="\r")

    # --- synthèse par vue
    views = ["v1", open_bench, "pooled"] + (["v2-subset"] if open_bench == "v3" else [])
    summary, paired = {}, {}
    for view in views:
        if view == "v2-subset":
            subset = [r for r in per_question if r["bench"] == open_bench and r["in_v2"]]
        else:
            subset = [r for r in per_question if view == "pooled" or r["bench"] == view]
        summary[view] = {config: summarise([r[config] for r in subset]) for config in configs}
        base_ndcg = [r[BASELINE]["ndcg"] for r in subset]
        base_r10 = [1.0 if (r[BASELINE]["first_rank"] or 99) <= 10 else 0.0 for r in subset]
        paired[view] = {}
        for config in configs:
            if config == BASELINE:
                continue
            paired[view][config] = {
                "nDCG@10": metrics.paired_delta(base_ndcg, [r[config]["ndcg"] for r in subset]),
                "recall@10": metrics.paired_delta(
                    base_r10, [1.0 if (r[config]["first_rank"] or 99) <= 10 else 0.0 for r in subset]),
            }

    controles = {"A_ligne_de_base": controle_a(summary, open_bench, views),
                 "B_fidelite": controle_b(cache, lexical_cache, fused60_cache, bool(args.bm25))}

    print("\n=== contrôle A — reproduction de la ligne de base dense ===")
    if controles["A_ligne_de_base"]:
        for ligne in controles["A_ligne_de_base"]:
            print(f"  ÉCART  {ligne}")
    else:
        print("  les quatre vues reproduisent results-router-v3.json à l'identique")
    print("=== contrôle B — fidélité du recalcul lexical et de la fusion ===")
    for k, v in controles["B_fidelite"].items():
        print(f"  {k} : {v}")
    if args.sabotage:
        print("\n--sabotage : le contrôle A ci-dessus DOIT signaler des écarts.")
        return
    if controles["A_ligne_de_base"]:
        sys.exit("\nARRÊT : la ligne de base n'est pas reproduite, la grille de fusion ne mesure rien.")

    # --- tableaux
    ordre = ["dense", "bm25"] + [f"{n}@{d}" for d in DEPTHS for n in FUSIONS]
    for view in views:
        print(f"\n=== {view} (n={summary[view]['dense']['n']}) — niveau chunk ===")
        print(f"{'config':<12}{'R@1':>7}{'R@5':>7}{'R@10':>7}{'R@30':>7}{'MRR':>8}{'nDCG@10':>9}"
              f"{'docNDCG':>9}{'ratés':>7}   Δ nDCG@10 [IC95]")
        for config in ordre:
            s = summary[view][config]
            if config == BASELINE:
                ecart = "  (référence)"
            else:
                pd = paired[view][config]["nDCG@10"]
                ecart = f"  {pd['delta']:+.3f} [{pd['ci95'][0]:+.3f}, {pd['ci95'][1]:+.3f}]{' *' if pd['significant'] else ''}"
            print(f"{config:<12}{s['recall@1']:>7.3f}{s['recall@5']:>7.3f}{s['recall@10']:>7.3f}"
                  f"{s['recall@30']:>7.3f}{s['MRR']:>8.3f}{s['nDCG@10']:>9.3f}{s['doc_nDCG@10']:>9.3f}"
                  f"{s['misses']:>7}{ecart}")

    # --- règle de décision, appliquée sur pooled
    principal = "pooled"
    verdicts = {}
    for config in configs:
        if config == BASELINE:
            continue
        gain = summary[principal][config]["recall@10"] - summary[principal][BASELINE]["recall@10"]
        regressions = {view: {m: round(summary[view][config][m] - summary[view][BASELINE][m], 3)
                              for m in PRIMARY if summary[view][config][m] - summary[view][BASELINE][m] < -0.01}
                       for view in views}
        regressions = {v: r for v, r in regressions.items() if r}
        pd = paired[principal][config]["nDCG@10"]
        verdicts[config] = {
            "recall@10_delta_pooled": round(gain, 3),
            "regle_gain": bool(gain >= 0.01),
            "regressions_au_dela_de_-0.01": regressions,
            "regle_non_regression": not regressions,
            "ndcg_paired_pooled": pd,
            "verdict": ("GO" if gain >= 0.01 and not regressions and pd.get("significant")
                        else "HOLD" if gain >= 0.01 and not regressions
                        else "NO-GO"),
        }

    print("\n=== règle de décision (banc principal : pooled, 155 questions) ===")
    print(f"{'config':<12}{'ΔR@10':>8}{'gain≥.01':>10}{'sans régression':>17}{'IC95 exclut 0':>15}   verdict")
    for config in ordre:
        if config == BASELINE:
            continue
        v = verdicts[config]
        print(f"{config:<12}{v['recall@10_delta_pooled']:>+8.3f}{str(v['regle_gain']):>10}"
              f"{str(v['regle_non_regression']):>17}{str(bool(v['ndcg_paired_pooled'].get('significant'))):>15}"
              f"   {v['verdict']}")

    # --- sauvetage / détournement lexical, sur la meilleure fusion de la grille
    meilleure = max((c for c in configs if c not in REFERENCES),
                    key=lambda c: summary[principal][c]["recall@10"])
    sauvetage = detournement = 0
    for r in per_question:
        base, hyb = r[BASELINE]["first_rank"], r[meilleure]["first_rank"]
        sauvetage += bool((base is None or base > 10) and hyb is not None and hyb <= 10)
        detournement += bool(base is not None and base <= 10 and (hyb is None or hyb > 10))
    print(f"\n=== {meilleure} contre dense, sur les {len(per_question)} questions ===")
    print(f"  sauvetage lexical  : {sauvetage} question(s) hors du top 10 en dense y entrent")
    print(f"  détournement       : {detournement} question(s) dans le top 10 en dense en sortent")

    # --- par famille du banc ouvert
    by_kind = {}
    for kind in sorted({r["kind"] for r in per_question if r["bench"] == open_bench}):
        sub = [r for r in per_question if r["bench"] == open_bench and r["kind"] == kind]
        by_kind[kind] = {"n": len(sub),
                         **{c: round(statistics.mean(r[c]["ndcg"] for r in sub), 3) for c in configs},
                         "delta_meilleure": metrics.paired_delta([r[BASELINE]["ndcg"] for r in sub],
                                                                 [r[meilleure]["ndcg"] for r in sub])}
    print(f"\n=== {open_bench} par famille (nDCG@10 chunk) ===")
    print(f"{'famille':<10}{'n':>4}{'dense':>8}{'bm25':>8}{meilleure:>12}   Δ [IC95]")
    for kind, row in by_kind.items():
        pd = row["delta_meilleure"]
        print(f"{kind:<10}{row['n']:>4}{row['dense']:>8.3f}{row['bm25']:>8.3f}{row[meilleure]:>12.3f}"
              f"   {pd['delta']:+.3f} [{pd['ci95'][0]:+.3f}, {pd['ci95'][1]:+.3f}]{' *' if pd['significant'] else ''}")

    sortie.write_text(json.dumps({
        "corpus": {**corpus_overlay.describe(),
                   "chunks": quant_rag.client().count(quant_rag.COLLECTION, exact=True).count,
                   "collection": quant_rag.COLLECTION},
        "benches": {"known_item": "v1", "open": open_bench, "questions": len(per_question)},
        "protocole": {
            "candidats_dense": f"cache {CACHE.name} (pipeline.retrieve_item mode dense, pool {pipeline.POOL})",
            "candidats_lexicaux": "quant_rag._lexical recalculé, brut (sans _with_text)",
            "profondeurs": list(DEPTHS), "reranker": False, "external_llm_calls": 0,
            "bootstrap": {"draws": 4000, "seed": 20260901},
            "index_bm25": Path(args.bm25).name if args.bm25 else quant_rag.bm25_path().name,
            "title_source": "registry" if args.bm25 else "export (payloads.jsonl)",
        },
        "controles": controles,
        "summary": summary, "paired": paired, "verdicts": verdicts,
        "banc_principal": principal, "meilleure_fusion": meilleure,
        "sauvetage_detournement": {"config": meilleure, "sauvetage": sauvetage, "detournement": detournement},
        "by_kind": by_kind,
        "latence_ms": {k: {"p50": round(statistics.median(v), 2),
                           "p95": round(sorted(v)[max(0, int(len(v) * 0.95) - 1)], 2)}
                       for k, v in latence.items()},
        "per_question": per_question,
    }, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\n-> {sortie}")


if __name__ == "__main__":
    main()
