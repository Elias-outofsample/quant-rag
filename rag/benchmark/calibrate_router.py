"""Calibration du routeur de retrieval sur les deux bancs, sans aucun appel LLM.

Question posée : à partir de quoi décider, *avant* de répondre, qu'une requête
mérite le mode hybride (BM25 + RRF, ± rerank) plutôt que le dense seul ?

Deux familles de signaux sont mises en concurrence :

  pré-retrieval   statistiques IDF des termes de la requête sur le vocabulaire de
                  l'index BM25 (l'hypothèse du handoff : « un terme rare → hybride »)
  post-retrieval  confiance du dense (score du 1er résultat, marge avec les suivants)
                  et accord dense/BM25 — gratuits, puisque le dense tourne de toute façon

Pour chaque règle candidate et chaque seuil, on calcule le nDCG@10 chunk qu'aurait
obtenu un routeur appliquant cette règle, sur v1, sur le banc ouvert (v2, puis v3
qui le contient), et sur les deux réunis. La règle retenue doit satisfaire les
critères d'acceptation du point 1-bis : ne pas régresser sur v1, battre le dense
sur le banc ouvert.

    .venv/bin/python rag/benchmark/calibrate_router.py               # v1 + v3
    .venv/bin/python rag/benchmark/calibrate_router.py --bench v2    # v1 + v2 (historique)

Les classements sont mis en cache (.cache/router-retrievals.json) : la première
exécution coûte ~3 min (rerank), les suivantes quelques secondes.
"""
from __future__ import annotations

import json
import math
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))   # rag/ : corpus_overlay
sys.path.insert(0, str(ROOT / "src"))

import corpus_overlay  # noqa: E402
import argparse  # noqa: E402

import metrics  # noqa: E402
import pipeline  # noqa: E402
from compare_v1_v2 import load_bench, load_v1  # noqa: E402
from corpus import ChunkIndex  # noqa: E402
from retrieval.lexical import tokenize  # noqa: E402

#: Le nom porte la signature de l'état du corpus : un cache de classements calculé
#: sur un autre corpus n'est pas rechargé, il n'est pas trouvé. Il le fallait — un
#: import ajoute des chunks qui concourent contre l'or sans que l'or bouge.
CACHE = HERE / ".cache" / f"router-retrievals-{corpus_overlay.signature()}.json"
CONFIGS = ("dense", "bm25", "rrf", "rrf_rerank")
#: Le banc ouvert évalué avec v1. « v2 » reproduit le fichier historique
#: (results-router-calibration.json) ; « v3 » écrit results-router-calibration-v3.json.
OPEN_BENCH = "v3"


def output_path(bench: str) -> Path:
    return HERE / ("results-router-calibration.json" if bench == "v2" else f"results-router-calibration-{bench}.json")


# --------------------------------------------------------------------------- données

def retrievals(items: list[dict], bench: str, index: ChunkIndex) -> dict:
    """{qid: {config: [(chunk_id, document_id, score), ...]}} — mis en cache."""
    store = json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.exists() else {}
    changed = False
    for item in items:
        key = f"{bench}/{item['qid']}"
        entry = store.setdefault(key, {})
        for config in CONFIGS:
            if config in entry:
                continue
            ranked = pipeline.retrieve_item(item, config, index, limit=pipeline.POOL)
            entry[config] = [(r.get("chunk_id"), r.get("document_id"),
                              float(r.get("score") or r.get("rrf_score") or 0.0)) for r in ranked]
            changed = True
        print(f"  {key} ok", end="\r", flush=True)
    if changed:
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        CACHE.write_text(json.dumps(store), encoding="utf-8")
    print()
    return {item["qid"]: store[f"{bench}/{item['qid']}"] for item in items}


def scores(item: dict, ranked_by_config: dict) -> dict:
    gold_chunks, gold_documents = set(item["gold_chunks"]), set(item["gold_documents"])
    poids = item.get("gold_poids") or None      # voir eval_router.measure
    out = {}
    for config, ranked in ranked_by_config.items():
        rows = [{"chunk_id": c, "document_id": d} for c, d, _ in ranked]
        out[config] = {"chunk": metrics.ndcg(rows, gold_chunks, gold_documents, poids=poids),
                       "doc": metrics.ndcg_documents(rows, gold_documents),
                       "rank": metrics.rank_of(rows, "chunk_id", gold_chunks)}
    return out


# --------------------------------------------------------------------------- signaux

def idf_table(bm25):
    n = bm25.doc_count
    return {term: math.log(1 + (n - df + 0.5) / (df + 0.5)) for term, df in bm25.document_frequency.items()}


def signals(question: str, ranked_by_config: dict, bm25, idf: dict) -> dict:
    """Tout ce qu'un routeur pourrait regarder, sans LLM."""
    tokens = [t for t in tokenize(question) if (len(t) > 1 and t.isalnum()) or "-" in t or "." in t]
    present = [(t, idf[t]) for t in set(tokens) if t in idf]
    idf_sum = sum(v for _, v in present)
    absent = [t for t in set(tokens) if t not in idf]
    idfs = sorted((v for _, v in present), reverse=True)
    dense = ranked_by_config["dense"]
    lexical = ranked_by_config["bm25"]
    dense_scores = [s for _, _, s in dense]
    top_dense = {c for c, _, _ in dense[:10]}
    top_bm25 = {c for c, _, _ in lexical[:10]}
    return {
        "n_tokens": len(tokens),
        "n_absent": len(absent),
        "max_idf": idfs[0] if idfs else 0.0,
        "top2_idf_mean": statistics.mean(idfs[:2]) if idfs else 0.0,
        "mean_idf": statistics.mean(idfs) if idfs else 0.0,
        "n_numeric": sum(1 for t in tokens if any(ch.isdigit() for ch in t)),
        "dense_top1": dense_scores[0] if dense_scores else 0.0,
        "dense_margin5": (dense_scores[0] - statistics.mean(dense_scores[1:6])) if len(dense_scores) > 5 else 0.0,
        "dense_margin10": (dense_scores[0] - dense_scores[9]) if len(dense_scores) > 9 else 0.0,
        "bm25_top1": lexical[0][2] if lexical else 0.0,
        # marge : le 1er BM25 se détache-t-il du peloton ? (match distinctif vs. recouvrement générique)
        "bm25_margin": (lexical[0][2] - statistics.mean(s for _, _, s in lexical[1:6])) if len(lexical) > 5 else 0.0,
        # couverture : quelle part du « budget IDF » de la requête un seul chunk réunit-il ?
        "bm25_coverage": (lexical[0][2] / idf_sum) if (lexical and idf_sum) else 0.0,
        "overlap10": len(top_dense & top_bm25),
        "rare_terms": sorted(present, key=lambda p: (-p[1], p[0]))[:3],  # ex æquo par ordre alphabétique : sortie stable
    }


from quant_rag import exact_tokens  # noqa: E402  — la définition de production, pas une copie


def rare_count(sig: dict, question: str, idf: dict, threshold: float) -> int:
    tokens = set(tokenize(question))
    return sum(1 for t in tokens if t in idf and idf[t] >= threshold)


# --------------------------------------------------------------------------- règles

def evaluate_rule(rows: list[dict], decide, hybrid_config: str, open_bench: str = "v2") -> dict:
    """nDCG@10 chunk moyen obtenu en appliquant `decide(row) -> 'dense'|'hybrid'`."""
    out = {}
    for bench in ("v1", open_bench, "pooled"):
        subset = rows if bench == "pooled" else [r for r in rows if r["bench"] == bench]
        routed, n_hybrid = [], 0
        for r in subset:
            mode = decide(r)
            n_hybrid += mode == "hybrid"
            routed.append(r["scores"][hybrid_config if mode == "hybrid" else "dense"]["chunk"])
        out[bench] = {"ndcg": statistics.mean(routed), "hybrid_share": n_hybrid / len(subset)}
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--bench", default=OPEN_BENCH, choices=("v2", "v3"), help="banc ouvert évalué avec v1")
    parser.add_argument("--gold-signature", metavar="SIG",
                        help="calibre sur les questions ré-orées questions-<banc>-SIG.jsonl")
    args = parser.parse_args()
    open_bench = args.bench
    benches = ("v1", open_bench, "pooled")
    index = ChunkIndex.load(verbose=False)
    bm25 = pipeline.bm25()
    idf = idf_table(bm25)

    # distribution IDF du vocabulaire, pour situer les seuils
    values = sorted(idf.values())
    def pct(p): return values[min(int(p / 100 * len(values)), len(values) - 1)]
    print(f"vocabulaire BM25 : {len(values):,} termes sur {bm25.doc_count:,} chunks")
    print(f"  IDF p50={pct(50):.2f}  p80={pct(80):.2f}  p90={pct(90):.2f}  p95={pct(95):.2f}  max={values[-1]:.2f}")
    df1 = sum(1 for df in bm25.document_frequency.values() if df == 1)
    print(f"  termes présents dans un seul chunk : {df1:,} ({100*df1/len(values):.0f}% du vocabulaire)"
          f" -> « top 5 % de l'IDF » ne discrimine rien, il faut un seuil absolu")

    rows = []
    suffixe = f"-{getattr(args, 'gold_signature', None)}" if getattr(args, "gold_signature", None) else ""
    for bench, items in (("v1", load_v1(index, HERE / f"questions-v1{suffixe}.jsonl")),
                         (open_bench, load_bench(HERE / f"questions-{open_bench}{suffixe}.jsonl"))):
        print(f"\n{bench} : {len(items)} questions — classements…")
        ranked = retrievals(items, bench, index)
        for item in items:
            # Les signaux se calculent sur le texte soumis au retrieval : pour une
            # question datée, le besoin d'information sans la clause de date (la
            # clause est traduite en filtre par l'appelant, ses années ne sont pas
            # des jetons exacts de la requête). C'est aussi ce que route la production.
            query = pipeline.query_of(item)
            rows.append({"bench": bench, "qid": item["qid"], "question": query,
                         "kind": item.get("kind", "single"),
                         "_dense_top1_chunk": (ranked[item["qid"]]["dense"][0][0] if ranked[item["qid"]]["dense"] else None),
                         "scores": scores(item, ranked[item["qid"]]),
                         "sig": signals(query, ranked[item["qid"]], bm25, idf)})

    # --- référence et plafond
    print("\n=== référence par configuration (nDCG@10 chunk) ===")
    print(f"{'':<14}{'v1':>8}{open_bench:>8}{'pooled':>9}")
    ref = {}
    for config in CONFIGS:
        ref[config] = {b: statistics.mean(r["scores"][config]["chunk"] for r in rows if b == "pooled" or r["bench"] == b)
                       for b in benches}
        print(f"{config:<14}{ref[config]['v1']:>8.3f}{ref[config][open_bench]:>8.3f}{ref[config]['pooled']:>9.3f}")
    for hybrid in ("rrf", "rrf_rerank"):
        oracle = {b: statistics.mean(max(r["scores"]["dense"]["chunk"], r["scores"][hybrid]["chunk"])
                                     for r in rows if b == "pooled" or r["bench"] == b) for b in benches}
        print(f"{'oracle/'+hybrid:<14}{oracle['v1']:>8.3f}{oracle[open_bench]:>8.3f}{oracle['pooled']:>9.3f}   <- plafond d'un routeur parfait")

    # --- par famille, sur le banc ouvert : où l'hybride gagne-t-il, où perd-il ?
    kinds = sorted({r["kind"] for r in rows if r["bench"] == open_bench})
    print(f"\n=== {open_bench} par famille (nDCG@10 chunk) ===")
    print(f"{'':<14}" + "".join(f"{k:>9}" for k in kinds))
    per_kind = {}
    for config in ("dense", "rrf", "rrf_rerank"):
        per_kind[config] = {k: statistics.mean(r["scores"][config]["chunk"] for r in rows if r["bench"] == open_bench and r["kind"] == k)
                            for k in kinds}
        print(f"{config:<14}" + "".join(f"{per_kind[config][k]:>9.3f}" for k in kinds))
    print(f"{'n':<14}" + "".join(f"{sum(1 for r in rows if r['bench'] == open_bench and r['kind'] == k):>9}" for k in kinds))

    # --- balayage des règles : chaque règle = (feature -> valeur, seuils) ; hybride si valeur ≥ τ
    n_table_top1 = 0
    for r in rows:
        sg = r["sig"]
        ex = exact_tokens(r["question"])
        sg["exact"] = ex
        sg["n_exact"] = ex["n_exact"]
        top1_chunk = index.get(r["_dense_top1_chunk"]) if r.get("_dense_top1_chunk") else None
        sg["dense_top1_is_table"] = bool(top1_chunk and top1_chunk.get("content_type") == "table")
        n_table_top1 += sg["dense_top1_is_table"]
        sg["rare6"] = rare_count(sg, r["question"], idf, 6.0)
        sg["rare7"] = rare_count(sg, r["question"], idf, 7.0)
        sg["neg_dense_top1"] = -sg["dense_top1"]
        sg["neg_margin10"] = -sg["dense_margin10"]
        # règles combinées (OU) : 1.0 si l'une des conditions est vraie
        sg["top1le60_or_numeric"] = float(sg["dense_top1"] <= 0.60 or sg["n_numeric"] >= 1)
        sg["top1le60_or_cov"] = float(sg["dense_top1"] <= 0.60 or sg["bm25_coverage"] >= 1.0)
        sg["numeric_or_cov"] = float(sg["n_numeric"] >= 1 or sg["bm25_coverage"] >= 1.0)
        sg["top1le60_or_numeric_or_cov"] = float(sg["dense_top1"] <= 0.60 or sg["n_numeric"] >= 1 or sg["bm25_coverage"] >= 1.0)
        sg["top1le65_and_cov08"] = float(sg["dense_top1"] <= 0.65 and sg["bm25_coverage"] >= 0.8)
        # règles « jeton exact » et combinaisons, avec/sans garde tableau
        guard = sg["dense_top1_is_table"] and sg["n_exact"] == 0
        for tau in (0.55, 0.58, 0.60, 0.62, 0.65):
            key = f"exact1_or_top1le{int(tau*100)}"
            sg[key] = float(sg["n_exact"] >= 1 or sg["dense_top1"] <= tau)
            sg[key + "_guard"] = float((sg["n_exact"] >= 1 or sg["dense_top1"] <= tau) and not guard)
            sg[f"exact2_or_top1le{int(tau*100)}"] = float(sg["n_exact"] >= 2 or sg["dense_top1"] <= tau)
        sg["top1le60_guard"] = float(sg["dense_top1"] <= 0.60 and not guard)
        # garde tableau sur la règle de production : les questions-tableaux sont pleines
        # de jetons exacts (tickers, années, « S&P 500 ») et le chemin hybride + rerank y
        # est nettement moins bon que le dense depuis la conversion en Markdown
        sg["exact2_not_table"] = float(sg["n_exact"] >= 2 and not sg["dense_top1_is_table"])
        sg["exact3_not_table"] = float(sg["n_exact"] >= 3 and not sg["dense_top1_is_table"])
    print(f"\n  garde tableau : le top-1 dense est un chunk-tableau sur {n_table_top1}/{len(rows)} questions")

    sweeps = {
        "max_idf ≥ τ (absolu)":        ("max_idf",        [4.0, 5.0, 6.0, 7.0, 8.0, 9.0]),
        "n_rare(idf≥6) ≥ k":           ("rare6",          [1, 2, 3, 4]),
        "n_rare(idf≥7) ≥ k":           ("rare7",          [1, 2, 3]),
        "n_numeric ≥ k":               ("n_numeric",      [1, 2]),
        "dense_top1 ≤ τ":              ("neg_dense_top1", [-0.55, -0.58, -0.60, -0.62, -0.65, -0.70]),
        "dense_margin10 ≤ τ":          ("neg_margin10",   [-0.03, -0.05, -0.07, -0.10]),
        "bm25_top1 ≥ τ":               ("bm25_top1",      [15, 20, 25, 30, 35, 40]),
        "bm25_margin ≥ τ":             ("bm25_margin",    [2, 4, 6, 8, 10]),
        "bm25_coverage ≥ τ":           ("bm25_coverage",  [0.6, 0.8, 1.0, 1.2, 1.4, 1.6]),
        "overlap10 ≥ k":               ("overlap10",      [1, 2, 3, 4]),
        "top1≤.60 OU numeric":         ("top1le60_or_numeric", [1.0]),
        "top1≤.60 OU cov≥1":           ("top1le60_or_cov", [1.0]),
        "numeric OU cov≥1":            ("numeric_or_cov", [1.0]),
        "top1≤.60 OU numeric OU cov≥1": ("top1le60_or_numeric_or_cov", [1.0]),
        "top1≤.65 ET cov≥.8":          ("top1le65_and_cov08", [1.0]),
        "n_exact ≥ k":                 ("n_exact",        [1, 2, 3]),
        "exact≥2 ET top1 non-tableau":  ("exact2_not_table", [1.0]),
        "exact≥3 ET top1 non-tableau":  ("exact3_not_table", [1.0]),
        "exact≥1 OU top1≤τ":           ("__exact1_family", [0.55, 0.58, 0.60, 0.62, 0.65]),
        "exact≥1 OU top1≤τ, garde":    ("__exact1_guard_family", [0.55, 0.58, 0.60, 0.62, 0.65]),
        "exact≥2 OU top1≤τ":           ("__exact2_family", [0.55, 0.58, 0.60, 0.62, 0.65]),
        "top1≤.60, garde":             ("top1le60_guard", [1.0]),
    }
    # les familles paramétrées par τ pointent vers la clé calculée correspondante
    for r in rows:
        for tau in (0.55, 0.58, 0.60, 0.62, 0.65):
            r["sig"][f"__exact1_family@{tau}"] = r["sig"][f"exact1_or_top1le{int(tau*100)}"]
            r["sig"][f"__exact1_guard_family@{tau}"] = r["sig"][f"exact1_or_top1le{int(tau*100)}_guard"]
            r["sig"][f"__exact2_family@{tau}"] = r["sig"][f"exact2_or_top1le{int(tau*100)}"]
    candidates = []
    for label, (feature, thresholds) in sweeps.items():
        for threshold in thresholds:
            if feature.endswith("_family"):
                decide = lambda r, f=feature, t=threshold: "hybrid" if r["sig"][f"{f}@{t}"] >= 1.0 else "dense"
            else:
                decide = lambda r, f=feature, t=threshold: "hybrid" if r["sig"][f] >= t else "dense"
            for hybrid in ("rrf", "rrf_rerank"):
                res = evaluate_rule(rows, decide, hybrid, open_bench)
                ok = res["v1"]["ndcg"] >= ref["dense"]["v1"] - 0.005 and res[open_bench]["ndcg"] >= ref["dense"][open_bench] + 0.005
                candidates.append({"rule": label, "feature": feature, "threshold": round(threshold, 3), "hybrid": hybrid,
                                   "v1": res["v1"]["ndcg"], open_bench: res[open_bench]["ndcg"], "pooled": res["pooled"]["ndcg"],
                                   "share_v1": res["v1"]["hybrid_share"], f"share_{open_bench}": res[open_bench]["hybrid_share"],
                                   "accepted": ok})
    candidates.sort(key=lambda c: (-c["accepted"], -c["pooled"]))

    print(f"\n=== top 20 (nDCG@10 chunk routé ; ok = pas de régression v1 ET gain {open_bench}) ===")
    print(f"{'règle':<30}{'τ':>7}{'hybride':>12}{'v1':>8}{open_bench:>8}{'pooled':>9}{'part hyb v1/' + open_bench:>16}  ok")
    print(f"{'(dense seul)':<30}{'':>7}{'':>12}{ref['dense']['v1']:>8.3f}{ref['dense'][open_bench]:>8.3f}{ref['dense']['pooled']:>9.3f}")
    for c in candidates[:20]:
        print(f"{c['rule']:<30}{c['threshold']:>7.2f}{c['hybrid']:>12}{c['v1']:>8.3f}{c[open_bench]:>8.3f}{c['pooled']:>9.3f}"
              f"{c['share_v1']:>8.0%}{c[f'share_{open_bench}']:>8.0%}  {'✔' if c['accepted'] else ''}")

    print("\n=== courbes complètes (rrf_rerank) — un optimum plat vaut mieux qu'un pic ===")
    for label in ("dense_top1 ≤ τ", "bm25_coverage ≥ τ", "bm25_top1 ≥ τ", "max_idf ≥ τ (absolu)", "n_rare(idf≥6) ≥ k"):
        pts = [c for c in candidates if c["rule"] == label and c["hybrid"] == "rrf_rerank"]
        pts.sort(key=lambda c: c["threshold"])
        print(f"  {label:<24}" + "  ".join(f"τ={abs(c['threshold']):.2f}:{c['pooled']:.3f}{'✔' if c['accepted'] else ' '}" for c in pts))

    best = candidates[0]
    print(f"\n=== retenue : {best['rule']} τ={best['threshold']} → {best['hybrid']} ===")
    feature, threshold, hybrid = best["feature"], best["threshold"], best["hybrid"]
    def decide(r):
        if feature.endswith("_family"):
            return "hybrid" if r["sig"][f"{feature}@{threshold}"] >= 1.0 else "dense"
        return "hybrid" if r["sig"][feature] >= threshold else "dense"

    print("\n=== erreurs de routage (écart nDCG ≥ 0,15 en faveur de l'autre mode) ===")
    errors = []
    for r in rows:
        mode = decide(r)
        got = r["scores"][hybrid if mode == "hybrid" else "dense"]["chunk"]
        alt = r["scores"]["dense" if mode == "hybrid" else hybrid]["chunk"]
        if alt - got >= 0.15:
            sg = r["sig"]
            errors.append(f"  {r['bench']} {r['qid']:<4} routé {mode:<6} obtenu {got:.2f} / alt {alt:.2f}"
                          f"  top1={sg['dense_top1']:.2f} bm25={sg['bm25_top1']:.0f} cov={sg['bm25_coverage']:.2f}"
                          f" num={sg['n_numeric']} maxidf={sg['max_idf']:.1f}  {r['question'][:60]}")
    print("\n".join(errors)); print(f"  {len(errors)} erreurs sur {len(rows)} questions")

    # --- rerank sur le sous-ensemble routé hybride : justifié ?
    routed = [r for r in rows if decide(r) == "hybrid"]
    if routed:
        print(f"\n=== sur les {len(routed)} requêtes routées hybride : rrf vs rrf_rerank (nDCG@10 chunk) ===")
        for cfg in ("dense", "rrf", "rrf_rerank"):
            print(f"  {cfg:<12}{statistics.mean(r['scores'][cfg]['chunk'] for r in routed):.3f}")

    # --- stabilité : calibrer sur un banc, évaluer sur l'autre
    print("\n=== validation croisée entre bancs (règle retenue, seuil choisi sur l'autre banc) ===")
    fam = [c for c in candidates if c["rule"] == best["rule"] and c["hybrid"] == hybrid]
    for train, test in (("v1", open_bench), (open_bench, "v1")):
        chosen = max(fam, key=lambda c: c[train])
        print(f"  seuil optimal sur {train} : τ={chosen['threshold']}  ->  sur {test} : {chosen[test]:.3f}"
              f"  (dense {ref['dense'][test]:.3f}, seuil pooled {best['threshold']} donne {best[test]:.3f})")

    if open_bench == "v3":
        from compare_v1_v2 import load_v2
        v2_qids = {i["qid"] for i in load_v2()}
        print("\n=== v3 par moitiés : questions v2 (ont servi à calibrer la règle) vs nouvelles (jamais vues) — nDCG@10 chunk, Δ vs dense ===")
        print(f"{'règle':<30}{'τ':>6}{'hybride':>12}{'v2-subset Δ [IC95]':>26}{'nouvelles Δ [IC95]':>26}")
        watched = [c for c in candidates if c["rule"] in ("n_exact ≥ k", "exact≥2 ET top1 non-tableau", "exact≥3 ET top1 non-tableau")
                   and c["threshold"] in (1.0, 2.0, 3.0)]
        for c in watched:
            f, t = c["feature"], c["threshold"]
            cells = []
            for half in ("v2", "new"):
                sub = [r for r in rows if r["bench"] == open_bench and ((r["qid"] in v2_qids) == (half == "v2"))]
                ref_s = [r["scores"]["dense"]["chunk"] for r in sub]
                var_s = [r["scores"][c["hybrid"] if r["sig"][f] >= t else "dense"]["chunk"] for r in sub]
                pd = metrics.paired_delta(ref_s, var_s)
                cells.append(f"{pd['delta']:+.3f} [{pd['ci95'][0]:+.3f},{pd['ci95'][1]:+.3f}]{'*' if pd['significant'] else ' '} n={len(sub)}")
            c["halves"] = cells
            print(f"{c['rule']:<30}{c['threshold']:>6.2f}{c['hybrid']:>12}{cells[0]:>26}{cells[1]:>26}")

    print("\n=== écarts appariés vs dense (nDCG@10 chunk), IC95 bootstrap — règle de production, puis top 8 règles acceptées ===")
    print(f"{'règle':<30}{'τ':>6}{'v1 Δ [IC95]':>24}{open_bench + ' Δ [IC95]':>24}{'pooled Δ [IC95]':>26}")
    shown = 0
    production = [c for c in candidates if c["rule"] == "n_exact ≥ k" and c["threshold"] == 2 and c["hybrid"] == "rrf_rerank"]
    for c in production + [c for c in candidates if c not in production]:
        if c not in production and (not c["accepted"] or c["hybrid"] != "rrf_rerank"):
            continue
        f, t = c["feature"], c["threshold"]
        def dec(r, f=f, t=t):
            if f.endswith("_family"):
                return "hybrid" if r["sig"][f"{f}@{t}"] >= 1.0 else "dense"
            return "hybrid" if r["sig"][f] >= t else "dense"
        cells = []
        for b in benches:
            sub = rows if b == "pooled" else [r for r in rows if r["bench"] == b]
            ref_s = [r["scores"]["dense"]["chunk"] for r in sub]
            var_s = [r["scores"]["rrf_rerank" if dec(r) == "hybrid" else "dense"]["chunk"] for r in sub]
            pd = metrics.paired_delta(ref_s, var_s)
            cells.append(f"{pd['delta']:+.3f} [{pd['ci95'][0]:+.3f},{pd['ci95'][1]:+.3f}]{'*' if pd['significant'] else ' '}")
        print(f"{c['rule']:<30}{c['threshold']:>6.2f}{cells[0]:>24}{cells[1]:>24}{cells[2]:>26}")
        c["paired"] = cells
        shown += 1
        if shown >= 8:
            break

    print("\n=== jetons exacts détectés (contrôle du détecteur) ===")
    for r in rows:
        ex = r["sig"]["exact"]
        if ex["n_exact"]:
            print(f"  {r['bench']} {r['qid']:<4} acr={ex['acronyms']} propre={ex['proper']} num={ex['numeric']}")
    print(f"  requêtes sans aucun jeton exact : {sum(1 for r in rows if not r['sig']['n_exact'])}/{len(rows)}")

    # --- la règle de production, par famille du banc ouvert : où le routage aide, où il nuit
    def production_decide(r):
        return "hybrid" if r["sig"]["n_exact"] >= 2 else "dense"
    routed_by_kind = {}
    for k in kinds:
        sub = [r for r in rows if r["bench"] == open_bench and r["kind"] == k]
        routed = [r["scores"]["rrf_rerank" if production_decide(r) == "hybrid" else "dense"]["chunk"] for r in sub]
        dense_k = [r["scores"]["dense"]["chunk"] for r in sub]
        pd = metrics.paired_delta(dense_k, routed)
        routed_by_kind[k] = {"n": len(sub), "dense": round(statistics.mean(dense_k), 3), "router": round(statistics.mean(routed), 3),
                             "rrf_rerank": round(statistics.mean(r["scores"]["rrf_rerank"]["chunk"] for r in sub), 3),
                             "hybrid_share": round(sum(1 for r in sub if production_decide(r) == "hybrid") / len(sub), 3),
                             "router_vs_dense": pd}
    print(f"\n=== règle de production (n_exact ≥ 2 → rrf_rerank) par famille de {open_bench} ===")
    print(f"{'famille':<10}{'n':>4}{'dense':>8}{'rrf_rr':>8}{'router':>8}{'part hyb':>10}{'router−dense [IC95]':>26}")
    for k, v in routed_by_kind.items():
        pd = v["router_vs_dense"]
        print(f"{k:<10}{v['n']:>4}{v['dense']:>8.3f}{v['rrf_rerank']:>8.3f}{v['router']:>8.3f}{v['hybrid_share']:>10.0%}"
              f"{pd['delta']:>+10.3f} [{pd['ci95'][0]:+.3f},{pd['ci95'][1]:+.3f}]{'*' if pd['significant'] else ''}")

    OUTPUT = output_path(open_bench)
    OUTPUT.write_text(json.dumps({
        "benches": {"known_item": "v1", "open": open_bench},
        "reference": ref, "per_kind": per_kind, "production_rule_by_kind": routed_by_kind,
        "vocabulary": {"terms": len(values), "chunks": bm25.doc_count,
                       "idf_percentiles": {p: round(pct(p), 3) for p in (50, 80, 90, 95)}},
        "candidates": candidates, "selected": best,
        "per_question": [{"bench": r["bench"], "qid": r["qid"], "kind": r["kind"],
                          "scores": {c: r["scores"][c]["chunk"] for c in CONFIGS},
                          "signals": {k: v for k, v in r["sig"].items() if not k.startswith("__") and k not in ("rare_terms", "exact")},
                          "exact": r["sig"]["exact"],
                          "rare_terms": r["sig"]["rare_terms"]} for r in rows],
    }, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\n-> {OUTPUT}")


if __name__ == "__main__":
    main()
