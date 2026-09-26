"""Un reranker plus fort sur le même pool dense — chantier D.

``bge-reranker-base`` (278 M, 2023) dégrade le pool dense (v1 : MRR doc 0,940 → 0,896)
et n'aide que le pool hybride. La question posée ici est étroite : à pool identique
(le top-50 dense de Qdrant, en cache), un cross-encoder plus récent classe-t-il mieux ?
Rien d'autre ne change — ni la requête, ni le pool, ni l'index.

Candidats, choisis pour tourner sur un M4 16 Go en fp16 :

    bge-base      BAAI/bge-reranker-base          278 M   la production (référence)
    bge-v2-m3     BAAI/bge-reranker-v2-m3         568 M   XLM-R large, multilingue, 2024
    qwen3-0.6b    Qwen/Qwen3-Reranker-0.6B        0,6 G   même famille que l'embedder Qwen3,
                                                           instruction, score = P(« yes »)

Mesure : nDCG@10 / R@k / MRR chunk sur v1 (25) et v3 (130), écarts appariés contre le
dense *sans* rerank et contre bge-base, par famille ; latence par requête (50 paires,
à chaud). Secondairement, les mêmes rerankers sur le pool **rrf** du cache — le seul
endroit où la production reranke aujourd'hui (mode hybride).

    .venv/bin/python rag/benchmark/eval_rerankers.py                 # ~15–25 min (3 modèles × 2 pools)
    .venv/bin/python rag/benchmark/eval_rerankers.py --models bge-base qwen3-0.6b --pools dense

Chaque (modèle, pool) terminé est écrit dans ``.cache/rerankers-partial.json`` et repris
tel quel à la relance : une interruption ne coûte que la paire en cours. Qwen3-Reranker
tourne à 12–26 s par requête sur ce Mac (50 paires, modèle causal) : le mesurer sur le
pool ``rrf`` est optionnel.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import corpus_overlay  # noqa: E402
import experiment  # noqa: E402
import pipeline  # noqa: E402
import quant_rag  # noqa: E402
from corpus import ChunkIndex  # noqa: E402

OUTPUT = HERE / "results-rerankers-v1.json"
#: Mesures par (modèle, pool) déjà faites : reprise après interruption. **Le nom porte la
#: signature de l'état du corpus**, comme ``experiment.CACHE`` — il ne la portait pas, et
#: c'était un piège : les 63 documents entrés le 5 septembre déplacent 18 rangs d'or sur 155
#: dans le pool dense, si bien qu'une reprise silencieuse aurait rendu un fichier estampillé
#: du corpus courant avec les mesures d'un autre. Le cache d'un autre état n'est plus
#: rechargé par erreur : il n'est simplement pas trouvé.
PARTIAL = HERE / ".cache" / f"rerankers-partial-{corpus_overlay.signature()}.json"
MODELS = {
    "bge-base": "BAAI/bge-reranker-base",
    "bge-v2-m3": "BAAI/bge-reranker-v2-m3",
    "qwen3-0.6b": "Qwen/Qwen3-Reranker-0.6B",
}
POOLS = ("dense", "rrf")
MAX_LENGTH = 512      # comme la production (``quant_rag._rerank``)
TEXT_CHARACTERS = 2000
BATCH = 8
QWEN_INSTRUCTION = "Given a quantitative-finance research question, judge whether the passage answers it."


class CrossEncoder:
    """``AutoModelForSequenceClassification`` à un logit (bge-*), fp16 sur MPS."""

    def __init__(self, model_id: str):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.device = quant_rag.device()
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_id, dtype=torch.float16 if self.device == "mps" else torch.float32).to(self.device).eval()

    def score(self, query: str, texts: list[str]) -> list[float]:
        import torch

        out: list[float] = []
        with torch.inference_mode():
            for start in range(0, len(texts), BATCH):
                pairs = [[query, t] for t in texts[start:start + BATCH]]
                inputs = self.tok(pairs, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt").to(self.device)
                out.extend(self.model(**inputs).logits.view(-1).float().cpu().tolist())
        return out


class QwenReranker:
    """Qwen3-Reranker : modèle causal, score = P(yes) sur le dernier jeton (format de la fiche modèle)."""

    PREFIX = ("<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct "
              "provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n")
    SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

    def __init__(self, model_id: str):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = quant_rag.device()
        self.tok = AutoTokenizer.from_pretrained(model_id, padding_side="left")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=torch.float16 if self.device == "mps" else torch.float32).to(self.device).eval()
        self.yes = self.tok.convert_tokens_to_ids("yes")
        self.no = self.tok.convert_tokens_to_ids("no")
        self.prefix_ids = self.tok.encode(self.PREFIX, add_special_tokens=False)
        self.suffix_ids = self.tok.encode(self.SUFFIX, add_special_tokens=False)

    def score(self, query: str, texts: list[str]) -> list[float]:
        import torch

        out: list[float] = []
        budget = MAX_LENGTH + 128 - len(self.prefix_ids) - len(self.suffix_ids)
        with torch.inference_mode():
            for start in range(0, len(texts), BATCH):
                bodies = [f"<Instruct>: {QWEN_INSTRUCTION}\n<Query>: {query}\n<Document>: {t}" for t in texts[start:start + BATCH]]
                encoded = self.tok(bodies, padding=False, truncation=True, max_length=budget, add_special_tokens=False)["input_ids"]
                sequences = [self.prefix_ids + ids + self.suffix_ids for ids in encoded]
                batch = self.tok.pad({"input_ids": sequences}, padding=True, return_tensors="pt").to(self.device)
                # ``logits_to_keep=1`` : sans lui, ``forward`` matérialise [lot, longueur,
                # vocabulaire] — à lot 8, longueur ~640 et vocabulaire 151 669, **1,5 Go en
                # fp16 par lot** — et fait tourner la tête de langage sur 5 120 positions
                # alors qu'une seule sert. Le 5 septembre 2026 cela a mis la machine à
                # 11,1 Go de swap sur 12 et fait passer la latence de 17 s à 49 s par
                # question en cours de banc. Seule la dernière position est lue ici : le
                # paramètre ne change aucun score (prouvé par eval_rerank_latency.py, qui
                # exige des classements identiques avant et après).
                logits = self.model(**batch, logits_to_keep=1).logits[:, -1, :]
                pair = torch.stack([logits[:, self.no], logits[:, self.yes]], dim=1).float()
                out.extend(torch.log_softmax(pair, dim=1)[:, 1].exp().cpu().tolist())
        return out


def load(name: str):
    model_id = MODELS[name]
    return QwenReranker(model_id) if name.startswith("qwen") else CrossEncoder(model_id)


def rerank(scorer, query: str, rows: list[dict]) -> list[dict]:
    rows = [r for r in rows if r.get("text")]
    scores = scorer.score(query, [r["text"][:TEXT_CHARACTERS] for r in rows])
    ranked = [{**r, "rerank_score": s} for r, s in zip(rows, scores)]
    return sorted(ranked, key=lambda r: -r["rerank_score"])



def controle_ligne_de_base(result: dict, configs: list[str]) -> list[str]:
    """« none » sur le pool dense doit reproduire la ligne de base enregistrée, chiffre par chiffre.

    Même exigence que le contrôle A d'``eval_fusion.py`` : un banc de rerankers posé sur un
    pool qu'on n'arrive pas à reproduire ne mesure pas le reranker, il mesure la dérive du
    pool. ``--sabotage`` décale le pool d'un rang pour vérifier que ce contrôle sait échouer.
    """
    recorded = HERE / "results-router-v3.json"
    if not recorded.exists():
        return [f"{recorded.name} absent : impossible de vérifier la ligne de base"]
    reference = json.loads(recorded.read_text(encoding="utf-8"))
    if reference["corpus"]["signature"] != corpus_overlay.signature():
        return [f"{recorded.name} mesuré sur {reference['corpus']['signature']}, "
                f"le corpus est à {corpus_overlay.signature()}"]
    ecarts = []
    for bench in experiment.BENCHES:
        attendu = reference["summary"].get(bench, {}).get("dense")
        obtenu = result["summary"].get(bench, {}).get("none")
        if attendu is None or obtenu is None:
            ecarts.append(f"vue {bench} : comparaison impossible")
            continue
        for key, value in attendu.items():
            # ``all_gold@10`` n'est pas comparable entre les deux scripts : ``eval_router.measure``
            # fixe ``all_found_at`` à None par construction (il ne mesure que le premier rang),
            # là où ``experiment.measure`` le calcule vraiment. L'écart est dans le code du banc,
            # pas dans le pool — le laisser dans le contrôle en ferait une alarme qui crie toujours,
            # donc une alarme qu'on finit par ignorer.
            if key == "all_gold@10":
                continue
            if key in obtenu and abs(obtenu[key] - value) > 1e-9:
                ecarts.append(f"{bench}/{key} : {obtenu[key]} au lieu de {value}")
    return ecarts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    parser.add_argument("--pools", nargs="+", default=list(POOLS), choices=list(POOLS))
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--sabotage", action="store_true",
                        help="décale le pool d'un rang : le contrôle de ligne de base doit refuser")
    args = parser.parse_args()

    # Un fichier de résultats mesuré sur un autre état du corpus est une archive, pas une
    # cible : le 5 septembre 2026 une vérification écourtée a écrasé, sous le nom par défaut,
    # les mesures des deux rerankers sur les deux pools de la signature 6c21d82412 (restaurées
    # depuis git). Même famille de piège que le cache de reprise non signé — un nom qui ne dit
    # pas sur quoi il a été mesuré finit par être écrasé par autre chose.
    if args.output.exists():
        try:
            ancienne = json.loads(args.output.read_text(encoding="utf-8")).get("corpus_state", {}).get("signature")
        except Exception:  # noqa: BLE001
            ancienne = None
        if ancienne and ancienne != corpus_overlay.signature():
            sys.exit(f"{args.output.name} a été mesuré sur la signature {ancienne}, le corpus est à "
                     f"{corpus_overlay.signature()} : l'écraser perdrait une archive. "
                     f"Passe --output results-rerankers-<quelque chose>.json.")

    started = time.perf_counter()
    index = ChunkIndex.load(verbose=False)
    items = experiment.load_items(index)
    cache = experiment.cached_rankings()
    print(f"{len(items)} questions · pools {args.pools} · modèles {args.models} · device {quant_rag.device()}")

    # pools, avec le texte des chunks (overlays appliqués — le texte que la production sert)
    pools: dict[str, dict[str, list[dict]]] = {}
    for pool in args.pools:
        pools[pool] = {}
        for item in items:
            rows = experiment.rows_from_cache(cache[item["key"]][pool])
            if args.sabotage:
                rows = rows[1:] + rows[:1]
            for r in rows:
                src = index.get(r["chunk_id"])
                r["text"] = src["text"] if src else ""
            pools[pool][item["key"]] = rows

    per_question = {pool: [{"bench": i["bench"], "key": i["key"], "qid": i["qid"], "kind": i.get("kind", "single"),
                            "none": experiment.measure(i, pools[pool][i["key"]])} for i in items] for pool in args.pools}

    # Le contrôle passe **avant** de charger un modèle : 80 minutes de rerank sur un pool
    # qui a dérivé, ce sont 80 minutes perdues, et un fichier qu'on croirait.
    controle: list[str] = []
    if "dense" in args.pools:
        controle = controle_ligne_de_base(
            experiment.summarise(per_question["dense"], ["none"], reference="none"), ["none"])
        print("\n=== contrôle — le pool dense reproduit-il results-router-v3.json ? ===")
        if controle:
            for ligne in controle:
                print(f"  ÉCART  {ligne}")
        else:
            print("  les trois vues reproduisent la ligne de base à l'identique")
        if args.sabotage:
            print("\n--sabotage : le contrôle ci-dessus DOIT signaler des écarts.")
            return
        if controle:
            sys.exit("\nARRÊT : le pool a dérivé, le banc mesurerait la dérive et non le reranker.")
    latency: dict[str, dict] = {}
    partial = json.loads(PARTIAL.read_text(encoding="utf-8")) if PARTIAL.exists() else {}
    for name in args.models:
        todo = [pool for pool in args.pools if f"{name}/{pool}" not in partial]
        for pool in args.pools:
            if pool not in todo:
                done = partial[f"{name}/{pool}"]
                for position, item in enumerate(items):
                    per_question[pool][position][name] = done["measures"][item["key"]]
                latency.setdefault(name, {})[pool] = done["latency"]
                print(f"\n=== {name} / {pool} : repris de {PARTIAL.name}")
        if not todo:
            continue
        print(f"\n=== {name} ({MODELS[name]})")
        t0 = time.perf_counter()
        scorer = load(name)
        print(f"  chargé en {time.perf_counter() - t0:.0f} s")
        for pool in todo:
            # échauffement hors chrono (compilation des noyaux MPS)
            rerank(scorer, "warm-up", pools[pool][items[0]["key"]][:8])
            times = []
            for position, item in enumerate(items, 1):
                query = pipeline.query_of(item)
                t1 = time.perf_counter()
                ranked = rerank(scorer, query, pools[pool][item["key"]])
                times.append(time.perf_counter() - t1)
                per_question[pool][position - 1][name] = experiment.measure(item, ranked)
                print(f"  {pool:<6} [{position}/{len(items)}] {item['key']:<8} {times[-1]:5.2f}s  "
                      f"pool@{str(per_question[pool][position - 1]['none']['first_rank'] or '-'):>3} → "
                      f"@{str(per_question[pool][position - 1][name]['first_rank'] or '-'):>3}", end="\r", flush=True)
            print()
            latency.setdefault(name, {})[pool] = {"mean_s": round(sum(times) / len(times), 2), "max_s": round(max(times), 2),
                                                  "pairs_per_query": 50}
            partial[f"{name}/{pool}"] = {"latency": latency[name][pool],
                                         "measures": {item["key"]: per_question[pool][position - 1][name] for position, item in enumerate(items, 1)}}
            PARTIAL.parent.mkdir(parents=True, exist_ok=True)
            PARTIAL.write_text(json.dumps(partial), encoding="utf-8")
        del scorer
        gc.collect()
        try:
            import torch
            if quant_rag.device() == "mps":
                torch.mps.empty_cache()
        except Exception:  # noqa: BLE001
            pass

    results = {}
    for pool in args.pools:
        configs = ["none"] + list(args.models)
        result = experiment.summarise(per_question[pool], configs, reference="none")
        # écart contre la production (bge-base) aussi, quand elle est là
        if "bge-base" in args.models:
            result["paired_vs_bge_base"] = {
                bench: {name: __import__("metrics").paired_delta(
                    [r["bge-base"]["ndcg"] for r in per_question[pool] if bench == "pooled" or r["bench"] == bench],
                    [r[name]["ndcg"] for r in per_question[pool] if bench == "pooled" or r["bench"] == bench])
                    for name in args.models if name != "bge-base"}
                for bench in experiment.BENCHES}
        result["wins_losses_vs_pool"] = {name: experiment.wins_losses(per_question[pool], name, "none") for name in args.models}
        print(f"\n################ pool {pool} — « none » = le pool tel quel, sans rerank")
        experiment.print_summary(result, configs, "none")
        if "paired_vs_bge_base" in result:
            print("\n=== contre bge-base (nDCG@10 chunk) ===")
            for bench, block in result["paired_vs_bge_base"].items():
                for name, pd in block.items():
                    print(f"  {bench:<8}{name:<12}{pd['delta']:+.3f} [{pd['ci95'][0]:+.3f},{pd['ci95'][1]:+.3f}]{' *' if pd['significant'] else ''}")
        results[pool] = {**result, "per_question": per_question[pool]}

    payload = {"version": "rerankers-v1", "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "corpus_state": corpus_overlay.describe(), "models": {n: MODELS[n] for n in args.models},
               "settings": {"max_length": MAX_LENGTH, "text_characters": TEXT_CHARACTERS, "batch": BATCH,
                            "device": quant_rag.device(), "qwen_instruction": QWEN_INSTRUCTION, "pool_source": str(experiment.CACHE.name)},
               "controle_ligne_de_base": controle or "0 écart — le pool dense reproduit results-router-v3.json",
               "latency": latency, "pools": results, "wall_clock_s": round(time.perf_counter() - started)}
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nlatence : {json.dumps(latency)}")
    print(f"-> {args.output}  ({payload['wall_clock_s']} s)")


if __name__ == "__main__":
    main()
