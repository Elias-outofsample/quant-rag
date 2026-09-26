"""Avant / après : le graphe livré (tableaux HTML) contre le graphe rebâti (corpus assaini).

Ce que la reconstruction devait changer, et qu'on vérifie ici plutôt que de le supposer :
les balises HTML (« rowspan », « colspan », « td ») ne doivent plus être des entités ; les
entités réelles (Gatheral, rough volatility, SVI, López de Prado…) doivent rester, avec un
nombre de mentions du même ordre — moins les 2 documents retirés.

    .venv/bin/python rag/graph/compare_graphs.py --before <(git show 093d1e6:data/graph/graph-lite.json)
    .venv/bin/python rag/graph/compare_graphs.py --before /chemin/ancien.json --after data/graph/graph-lite.json

Écrit ``rag/graph/results-graph-v1.json``.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "rag"))
import graph_search  # noqa: E402  — fold(), is_noise()

HTML_TAGS = ("rowspan", "colspan", "td", "tr", "th", "table", "tbody", "thead")
PROBES = ("Gatheral", "rough volatility", "SVI", "López de Prado", "Fukasawa", "Heston", "Sharpe ratio",
          "Almgren", "Kelly criterion", "GARCH", "Black-Scholes", "S&P 500")
STRUCTURAL = {"mentions", "contains_chunk"}


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def summarise(payload: dict) -> dict:
    nodes, edges = payload["nodes"], payload["edges"]
    entities = [n for n in nodes if n["type"] == "entity"]
    by_fold: dict[str, list[dict]] = {}
    for node in entities:
        by_fold.setdefault(graph_search.fold(node["name"]), []).append(node)
    html = {tag: sum(n["mentions"] for n in by_fold.get(tag, [])) for tag in HTML_TAGS}
    noise = [n for n in entities if graph_search.is_noise(n["name"])]
    top = sorted(entities, key=lambda n: -n["mentions"])[:25]
    probes = {}
    for probe in PROBES:
        needle = graph_search.fold(probe)
        hits = [n for n in entities if needle in graph_search.fold(n["name"])]
        exact = [n for n in hits if graph_search.fold(n["name"]) == needle]
        probes[probe] = {"exact_mentions": sum(n["mentions"] for n in exact),
                         "exact_chunks": len({c for n in exact for c in n.get("source_chunks") or []}),
                         "variants": len(hits), "variant_mentions": sum(n["mentions"] for n in hits),
                         "labels": dict(Counter(n["label"] for n in exact))}
    return {
        "report": {k: payload.get("report", {}).get(k) for k in ("sample_chunks", "documents", "nodes", "entity_nodes",
                                                                 "edges", "mention_edges", "relation_edges", "corpus")},
        "entities": len(entities), "mentions": sum(n["mentions"] for n in entities),
        "labels": dict(Counter(n["label"] for n in entities).most_common()),
        "html_tag_mentions": html, "html_tag_mentions_total": sum(html.values()),
        "noise_entities": len(noise), "noise_mentions": sum(n["mentions"] for n in noise),
        "relation_types": dict(Counter(e["relation"] for e in edges if e["relation"] not in STRUCTURAL).most_common()),
        "top_entities": [{"name": n["name"], "label": n["label"], "mentions": n["mentions"]} for n in top],
        "probes": probes,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, default=ROOT / "data" / "graph" / "graph-lite.json")
    parser.add_argument("--output", type=Path, default=ROOT / "rag" / "graph" / "results-graph-v1.json")
    args = parser.parse_args()
    before, after = summarise(load(args.before)), summarise(load(args.after))
    delta = {
        "html_tag_mentions": {tag: [before["html_tag_mentions"][tag], after["html_tag_mentions"][tag]] for tag in HTML_TAGS},
        "noise_mentions": [before["noise_mentions"], after["noise_mentions"]],
        "entities": [before["entities"], after["entities"]],
        "mentions": [before["mentions"], after["mentions"]],
        "probes": {p: [before["probes"][p]["exact_mentions"], after["probes"][p]["exact_mentions"]] for p in PROBES},
    }
    out = {"before": before, "after": after, "delta": delta}
    args.output.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"{'':28} {'avant':>10} {'après':>10}")
    print(f"{'chunks':28} {before['report']['sample_chunks'] or 0:>10} {after['report']['sample_chunks'] or 0:>10}")
    print(f"{'entités':28} {before['entities']:>10} {after['entities']:>10}")
    print(f"{'mentions':28} {before['mentions']:>10} {after['mentions']:>10}")
    print(f"{'mentions balises HTML':28} {before['html_tag_mentions_total']:>10} {after['html_tag_mentions_total']:>10}")
    for tag in HTML_TAGS:
        print(f"  {tag:26} {before['html_tag_mentions'][tag]:>10} {after['html_tag_mentions'][tag]:>10}")
    print(f"{'mentions bruit (filtre)':28} {before['noise_mentions']:>10} {after['noise_mentions']:>10}")
    print(f"{'arêtes de relation':28} {sum(before['relation_types'].values()):>10} {sum(after['relation_types'].values()):>10}")
    print("\nsondes (mentions exactes, chunks)")
    for p in PROBES:
        b, a = before["probes"][p], after["probes"][p]
        print(f"  {p:26} {b['exact_mentions']:>6} ({b['exact_chunks']:>4}) {a['exact_mentions']:>6} ({a['exact_chunks']:>4})")
    print("\ntop 15 après :", ", ".join(f"{n['name']} ({n['mentions']})" for n in after["top_entities"][:15]))
    print(f"\n-> {args.output}")


if __name__ == "__main__":
    main()
