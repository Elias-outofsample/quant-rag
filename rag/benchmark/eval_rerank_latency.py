"""Latence du reranker Qwen3, mesurée honnêtement — et l'optimisation prouvée sans effet sur le classement.

28,8 s par requête pour un modèle de 0,6 milliard de paramètres qui fait **une** passe avant
sur 50 paires : ce n'est pas le modèle, c'est l'implémentation. La cause est visible sans
rien lancer.

Ce que la mesure a ensuite corrigé, et qu'il faut lire avant les tableaux : les 28,8 s
enregistrées le 3 septembre étaient elles-mêmes contaminées par le swap. Sur machine
saine, la version d'origine mesure **17,6 s p50**, et le correctif ci-dessous en retire
**15 %** — non pas l'ordre de grandeur qu'un calcul de FLOPs laissait espérer. Ce qui est
réellement gagné est la **stabilité** : le rapport max/moyenne passe de 4,4 à 1,14, et le
banc complet cesse de dériver de 17 s à 49 s par question. ``Qwen3ForCausalLM.forward`` rend ``logits`` de forme
``[lot, longueur, vocabulaire]`` — soit, à lot 8, longueur ~640 et vocabulaire 151 669, un
tenseur de **1,5 Go en fp16 par lot**, calculé en entier alors que seule la dernière position
sert. La tête de langage tourne ainsi sur 5 120 positions au lieu de 8. ``forward`` accepte
pourtant ``logits_to_keep``.

Trois variantes, à profondeurs 50, 30 et 10 :

    A  telle quelle          ``model(**batch).logits[:, -1, :]`` — la ligne de base
    B  A + logits_to_keep=1  la tête de langage ne calcule que la position utile
    C  B + tri par longueur  les paires sont regroupées par taille : moins de remplissage

Le contrôle qui autorise le reste : **B et C doivent rendre exactement le classement de A**.
Une optimisation qui déplace un rang n'est pas une optimisation, c'est un autre système.

    .venv/bin/python rag/benchmark/eval_rerank_latency.py
    .venv/bin/python rag/benchmark/eval_rerank_latency.py --questions 15 --depths 50 30 10
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
import pipeline  # noqa: E402
import quant_rag  # noqa: E402
from corpus import ChunkIndex  # noqa: E402
from eval_rerankers import MAX_LENGTH, MODELS, QWEN_INSTRUCTION, TEXT_CHARACTERS  # noqa: E402

OUTPUT = HERE / "results-rerank-latency-v1.json"
VARIANTS = ("A-telle-quelle", "B-logits-to-keep", "C-tri-par-longueur")


class Qwen:
    """``eval_rerankers.QwenReranker``, avec les deux optimisations en option.

    Le corps est repris tel quel — même invite, même troncature, même arithmétique — pour
    que la seule chose qui varie soit ce qu'on mesure.
    """

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

    def _sequences(self, query: str, texts: list[str]) -> list[list[int]]:
        budget = MAX_LENGTH + 128 - len(self.prefix_ids) - len(self.suffix_ids)
        bodies = [f"<Instruct>: {QWEN_INSTRUCTION}\n<Query>: {query}\n<Document>: {t}" for t in texts]
        encoded = self.tok(bodies, padding=False, truncation=True, max_length=budget,
                           add_special_tokens=False)["input_ids"]
        return [self.prefix_ids + ids + self.suffix_ids for ids in encoded]

    def score(self, query: str, texts: list[str], batch: int = 8,
              keep_one: bool = False, sort_by_length: bool = False) -> list[float]:
        import torch

        sequences = self._sequences(query, texts)
        order = sorted(range(len(sequences)), key=lambda i: len(sequences[i])) if sort_by_length \
            else list(range(len(sequences)))
        out: list[float] = [0.0] * len(sequences)
        with torch.inference_mode():
            for start in range(0, len(order), batch):
                positions = order[start:start + batch]
                padded = self.tok.pad({"input_ids": [sequences[i] for i in positions]},
                                      padding=True, return_tensors="pt").to(self.device)
                logits = (self.model(**padded, logits_to_keep=1).logits[:, -1, :] if keep_one
                          else self.model(**padded).logits[:, -1, :])
                pair = torch.stack([logits[:, self.no], logits[:, self.yes]], dim=1).float()
                scores = torch.log_softmax(pair, dim=1)[:, 1].exp().cpu().tolist()
                for position, value in zip(positions, scores):
                    out[position] = value
        return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--questions", type=int, default=15)
    parser.add_argument("--depths", type=int, nargs="+", default=[50, 30, 10])
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()

    index = ChunkIndex.load(verbose=False)
    items = experiment.load_items(index)[:args.questions]
    cache = experiment.cached_rankings()
    pools = {}
    for item in items:
        rows = experiment.rows_from_cache(cache[item["key"]]["dense"])
        for r in rows:
            src = index.get(r["chunk_id"])
            r["text"] = src["text"] if src else ""
        pools[item["key"]] = [r for r in rows if r.get("text")]

    print(f"{len(items)} questions · profondeurs {args.depths} · lot {args.batch} · {quant_rag.device()}")
    scorer = Qwen(MODELS["qwen3-0.6b"])
    reglages = {"A-telle-quelle": {}, "B-logits-to-keep": {"keep_one": True},
                "C-tri-par-longueur": {"keep_one": True, "sort_by_length": True}}

    mesures: dict[str, dict] = {}
    classements: dict[str, dict] = {}
    for depth in args.depths:
        # échauffement hors chrono, à cette profondeur
        scorer.score("warm-up", [r["text"][:TEXT_CHARACTERS] for r in pools[items[0]["key"]][:args.batch]],
                     batch=args.batch)
        for variante, options in reglages.items():
            times, ranks = [], {}
            for item in items:
                rows = pools[item["key"]][:depth]
                textes = [r["text"][:TEXT_CHARACTERS] for r in rows]
                t0 = time.perf_counter()
                scores = scorer.score(pipeline.query_of(item), textes, batch=args.batch, **options)
                times.append(time.perf_counter() - t0)
                ordre = sorted(range(len(rows)), key=lambda i: -scores[i])
                ranks[item["key"]] = [rows[i]["chunk_id"] for i in ordre]
                print(f"  d{depth:<3} {variante:<20} [{len(times)}/{len(items)}] {times[-1]:6.2f}s", end="\r", flush=True)
            mesures[f"{variante}@{depth}"] = {
                "p50": round(statistics.median(times), 2),
                "p95": round(sorted(times)[max(0, int(len(times) * 0.95) - 1)], 2),
                "mean": round(statistics.mean(times), 2), "max": round(max(times), 2),
                "paires_par_requete": depth, "n": len(times)}
            classements[f"{variante}@{depth}"] = ranks
    print(" " * 78, end="\r")

    # --- contrôle : B et C rendent-elles le classement de A ?
    controle = {}
    for depth in args.depths:
        reference = classements[f"A-telle-quelle@{depth}"]
        for variante in VARIANTS[1:]:
            identiques = sum(1 for key in reference if classements[f"{variante}@{depth}"][key] == reference[key])
            controle[f"{variante}@{depth}"] = f"{identiques}/{len(reference)} classements identiques à A"

    print("\n=== contrôle — l'optimisation change-t-elle le classement ? ===")
    for nom, verdict in controle.items():
        print(f"  {nom:<28} {verdict}")

    print("\n=== latence par requête (s) ===")
    print(f"{'variante':<22}" + "".join(f"{f'd{d} p50':>10}{f'p95':>8}" for d in args.depths))
    for variante in VARIANTS:
        ligne = f"{variante:<22}"
        for depth in args.depths:
            m = mesures[f"{variante}@{depth}"]
            ligne += f"{m['p50']:>10.2f}{m['p95']:>8.2f}"
        print(ligne)

    reference = mesures[f"A-telle-quelle@{max(args.depths)}"]["p50"]
    print(f"\ncible « mode deep » : 2–3 s par requête. Départ : {reference:.1f} s")
    for variante in VARIANTS[1:]:
        for depth in args.depths:
            m = mesures[f"{variante}@{depth}"]
            if m["p50"] <= 3.0:
                print(f"  atteinte par {variante} à la profondeur {depth} : p50 {m['p50']:.2f} s "
                      f"(× {reference / m['p50']:.0f} plus rapide)")

    args.output.write_text(json.dumps({
        "version": "rerank-latency-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "corpus_state": corpus_overlay.describe(),
        "model": MODELS["qwen3-0.6b"], "device": quant_rag.device(),
        "settings": {"batch": args.batch, "max_length": MAX_LENGTH, "text_characters": TEXT_CHARACTERS,
                     "questions": len(items), "depths": args.depths},
        "cause": ("logits de forme [lot, longueur, vocabulaire] matérialisés en entier : à lot 8, "
                  "longueur ~640 et vocabulaire 151669, 1,5 Go en fp16 par lot, et la tête de langage "
                  "calculée sur 5120 positions au lieu de 8"),
        "controle_classement_inchange": controle,
        "latence": mesures,
    }, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\n-> {args.output}")


if __name__ == "__main__":
    main()
