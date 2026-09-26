"""Requête du graphe d'entités (GLiNER2) — complète ``search_documents``, ne le remplace pas.

Le graphe (``data/graph/graph-lite.json``) est du JSON pur : nœuds ``document`` / ``chunk`` /
``entity``, arêtes ``contains_chunk`` / ``mentions`` / relations sémantiques (``uses_method``,
``applies_to``, ``is_a``…). Pas de LightRAG, pas d'Ollama : les trois usages réels tiennent
en trois jointures en mémoire, et le texte des passages vient de Qdrant comme pour la
recherche dense — même format de ligne : une ligne ``citer :`` (référence et page **d'un
lecteur**) et une ligne ``interne :`` pour les clés de travail. Les trois affichages ont
longtemps montré ``page_start``/``page_end`` bruts, qui sont **0-basés** : la correction du
7 septembre 2026 sur ``search_documents`` ne les avait pas touchés, et ils envoyaient donc
encore le lecteur une page trop tôt.

  - ``search_graph(query)``        les passages qui *mentionnent* une entité dont le nom
                                   contient ``query`` (« Gatheral », « SVI ») ;
  - ``expand_entity(name)``        les entités reliées à ``name`` par une relation
                                   (« quelles méthodes dépendent de la rough vol ? »), et
                                   celles citées dans les mêmes passages (co-mentions) ;
  - ``connect_entities(a, b)``     les passages et documents qui mentionnent *les deux*.

Décisions :
  - le graphe est chargé une fois (~1 s), indexé en mémoire ; pas d'index en amont ;
  - le bruit est filtré **à la lecture**, pas dans le fichier : entités d'un ou deux
    caractères (« x » : 3 102 mentions, « S », « i »), commandes LaTeX (« mathrm »),
    balises HTML d'anciens tableaux — 26 718 mentions sur 230 291 dans le graphe rebâti
    (40 043 sur 251 391 dans le graphe livré). ``status()`` dit combien ;
  - correspondance des noms par clé pliée (NFKC, casse, accents, accents espaçants
    « L´opez ») : « Lopez de Prado », « López de Prado » et « L´opez de Prado » sont la même
    entité de fait, sans correspondance floue au-delà ;
  - une entité exacte pèse 1, une entité dont le nom *contient* la requête pèse 0,5 :
    « SVI » remonte d'abord les passages sur SVI, puis ceux sur SSVI et eSSVI.
"""
from __future__ import annotations

import json
import re
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "rag"))
import contrat  # noqa: E402  — la fenêtre servie et la coupe sûre, partagées avec mcp_server
import corpus_overlay  # noqa: E402
import quant_rag  # noqa: E402

GRAPH_PATH = ROOT / "data" / "graph" / "graph-lite.json"
STRUCTURAL = {"mentions", "contains_chunk"}
ENTITY_LABELS = ("person", "organization", "financial_instrument", "market_concept", "measure", "method", "dataset")
#: Noms qui ne sont pas des entités : balises HTML des anciens tableaux, commandes LaTeX.
NOISE_NAMES = frozenset("""
td tr th table tbody thead rowspan colspan html div span br align center left right
mathrm mathbf mathcal mathbb mathit boldsymbol frac sqrt cdot cdots ldots dots left right
hat bar tilde vec overline underline text textbf textit label ref eqref begin end
""".split())
_SPACING_ACCENTS = re.compile("[´`ˊˋˈ’‘]")


def fold(text: str) -> str:
    """Clé de comparaison des noms : NFKC, casse pliée, accents (y compris espaçants) retirés."""
    text = unicodedata.normalize("NFKC", _SPACING_ACCENTS.sub("", str(text)))  # « ´ » avant NFKC, qui l'éclaterait en espace + accent
    text = "".join(ch for ch in unicodedata.normalize("NFKD", text.casefold()) if not unicodedata.combining(ch))
    return " ".join(text.split())


def is_noise(name: str) -> bool:
    core = name.strip()
    if len(core) <= 2:
        return True
    if not any(ch.isalpha() for ch in core):
        return True
    return fold(core) in NOISE_NAMES


class Graph:
    """Vue mémoire du graphe : entités, arêtes par extrémité, chunk -> document."""

    def __init__(self, payload: dict):
        self.report = payload.get("report", {})
        self.entities: dict[str, dict] = {}
        self.chunk_document: dict[str, str] = {}
        self.chunk_page: dict[str, int | None] = {}
        self.dropped_entities = self.dropped_mentions = 0
        for node in payload["nodes"]:
            kind = node["type"]
            if kind == "chunk":
                self.chunk_page[node["chunk_id"]] = node.get("page_start")
            elif kind == "entity":
                if is_noise(node.get("name") or ""):
                    self.dropped_entities += 1
                    self.dropped_mentions += node.get("mentions", 0)
                    continue
                self.entities[node["id"]] = node
        self.by_key: dict[str, list[str]] = defaultdict(list)
        for eid, node in self.entities.items():
            self.by_key[fold(node["name"])].append(eid)
        self.names: list[tuple[str, str]] = [(fold(node["name"]), eid) for eid, node in self.entities.items()]
        self.out_edges: dict[str, list[dict]] = defaultdict(list)
        self.in_edges: dict[str, list[dict]] = defaultdict(list)
        #: chunk_id -> Counter(entity_id -> mentions dans ce chunk)
        self.chunk_mentions: dict[str, Counter] = defaultdict(Counter)
        self.relation_edges = 0
        for edge in payload["edges"]:
            rel = edge["relation"]
            if rel == "contains_chunk":
                self.chunk_document[edge["target"].removeprefix("chunk:")] = edge["source"].removeprefix("document:")
            elif rel == "mentions":
                if edge["target"] in self.entities:
                    self.chunk_mentions[edge["source_chunk_id"]][edge["target"]] += 1
            else:
                if edge["source"] in self.entities and edge["target"] in self.entities:
                    self.out_edges[edge["source"]].append(edge)
                    self.in_edges[edge["target"]].append(edge)
                    self.relation_edges += 1
        self.relations = sorted({e["relation"] for edges in self.out_edges.values() for e in edges})

    # ------------------------------------------------------------------ entités

    def find(self, query: str, entity_type: str | None = None, limit: int = 50) -> list[dict]:
        """Entités dont le nom contient ``query`` ; les correspondances exactes d'abord, puis par mentions."""
        needle = fold(query)
        if not needle:
            return []
        found = []
        for name, eid in self.names:
            if needle in name:
                node = self.entities[eid]
                if entity_type and node["label"] != entity_type:
                    continue
                exact = name == needle
                found.append({"id": eid, "name": node["name"], "label": node["label"], "mentions": node["mentions"],
                              "chunks": len(node.get("source_chunks") or []), "exact": exact,
                              "weight": 1.0 if exact else 0.5})
        found.sort(key=lambda e: (not e["exact"], -e["mentions"], e["name"]))
        return found[:limit]

    # ------------------------------------------------------------------ voisinage

    def neighbours(self, entity_ids: list[str], relation: str | None = None) -> list[dict]:
        """Entités reliées, groupées par (relation, direction, voisin), avec les chunks qui les attestent."""
        groups: dict[tuple, dict] = {}
        origin = set(entity_ids)
        for eid in entity_ids:
            for direction, edges in (("out", self.out_edges.get(eid, ())), ("in", self.in_edges.get(eid, ()))):
                for edge in edges:
                    if relation and edge["relation"] != relation:
                        continue
                    other = edge["target"] if direction == "out" else edge["source"]
                    if other in origin:
                        continue
                    node = self.entities[other]
                    group = groups.setdefault((edge["relation"], direction, other), {
                        "relation": edge["relation"], "direction": direction, "entity_id": other,
                        "name": node["name"], "label": node["label"], "mentions": node["mentions"],
                        "support": 0, "chunk_ids": [], "via": []})
                    group["support"] += 1
                    if edge["source_chunk_id"] not in group["chunk_ids"]:
                        group["chunk_ids"].append(edge["source_chunk_id"])
                    if self.entities[eid]["name"] not in group["via"]:
                        group["via"].append(self.entities[eid]["name"])
        rows = list(groups.values())
        rows.sort(key=lambda g: (-g["support"], -g["mentions"], g["relation"], g["name"]))
        return rows

    def chunks_of(self, entity_ids: list[str]) -> set[str]:
        out: set[str] = set()
        for eid in entity_ids:
            out.update(self.entities[eid].get("source_chunks") or ())
        return out

    def co_mentioned(self, entity_ids: list[str], limit: int = 15) -> list[dict]:
        """Entités citées dans les mêmes passages, pondérées par leur rareté.

        Les relations typées sont clairsemées (17 881 arêtes pour 66 494 entités) ; la
        co-mention est la structure dense du graphe. Score = passages communs ×
        log(N / passages de l'entité voisine) : « volatility », citée partout, ne
        domine pas « rough Bergomi », citée dans les mêmes trente passages.
        """
        import math

        origin = set(entity_ids)
        chunks = self.chunks_of(entity_ids)
        counts: Counter = Counter()
        for cid in chunks:
            for eid in self.chunk_mentions.get(cid, ()):
                if eid not in origin:
                    counts[eid] += 1
        total = max(len(self.chunk_document), 1)
        rows = []
        for eid, shared in counts.items():
            node = self.entities[eid]
            df = max(len(node.get("source_chunks") or ()), 1)
            rows.append({"entity_id": eid, "name": node["name"], "label": node["label"], "mentions": node["mentions"],
                         "shared_chunks": shared, "score": round(shared * math.log(total / df), 2)})
        rows.sort(key=lambda r: (-r["score"], -r["shared_chunks"], r["name"]))
        return rows[:limit]


@lru_cache(maxsize=1)
def graph() -> Graph:
    if not GRAPH_PATH.exists():
        raise RuntimeError(f"Graphe absent : {GRAPH_PATH} — lance rag/graph/extract_entities.py puis build_graph.py")
    return Graph(json.loads(GRAPH_PATH.read_text(encoding="utf-8")))


# ---------------------------------------------------------------------- jointures

def _source_of(document_id: str | None) -> dict:
    meta = quant_rag.document_metadata().get(document_id or "", {})
    title, short_ref = meta.get("title"), meta.get("short_ref")
    return {"document_id": document_id, "title": title, "short_ref": short_ref,
            "authors": meta.get("authors", []), "publication_year": meta.get("publication_year"),
            "source": f"{short_ref} — {title}" if short_ref else (title or document_id or "?")}


def _fetch_rows(chunk_ids: list[str]) -> dict[str, dict]:
    """Texte et métadonnées des chunks, depuis Qdrant (même ligne que la recherche dense)."""
    from qdrant_client import models

    if not chunk_ids:
        return {}
    found, _ = quant_rag.client().scroll(
        collection_name=quant_rag.COLLECTION, limit=len(chunk_ids), with_payload=True,
        scroll_filter=models.Filter(must=[models.FieldCondition(key="chunk_id", match=models.MatchAny(any=chunk_ids))]),
    )
    return {point.payload["chunk_id"]: quant_rag._payload_row(point.payload, 0.0, "graph") for point in found}


def _rank_chunks(g: Graph, weights: dict[str, float], candidates: set[str]) -> list[tuple[str, float, list[str]]]:
    """Chunks classés par poids des entités qu'ils mentionnent (exact 1, inclusion 0,5), puis par mentions."""
    ranked = []
    for cid in candidates:
        counts = g.chunk_mentions.get(cid, Counter())
        hit = [eid for eid in weights if eid in counts]
        if not hit:
            continue
        score = sum(weights[eid] for eid in hit) + 0.01 * sum(min(counts[eid], 10) for eid in hit)
        ranked.append((cid, round(score, 3), hit))
    ranked.sort(key=lambda row: (-row[1], row[0]))
    return ranked


def _materialise(g: Graph, ranked: list[tuple[str, float, list[str]]], limit: int, per_document: int,
                 min_characters: int) -> list[dict]:
    """Applique le plafond par document sur le graphe (document connu sans Qdrant), puis charge les textes."""
    picked, per_doc = [], Counter()
    for cid, score, hit in ranked:
        doc = g.chunk_document.get(cid)
        if per_document and per_doc[doc] >= per_document:
            continue
        per_doc[doc] += 1
        picked.append((cid, score, hit))
        if len(picked) >= limit * 3:
            break
    payloads = _fetch_rows([cid for cid, _, _ in picked])
    rows = []
    for cid, score, hit in picked:
        row = payloads.get(cid)
        if row is None:
            continue
        row.update(score=score, score_kind="graph", entities=list(dict.fromkeys(g.entities[eid]["name"] for eid in hit)))
        rows.append(row)
    return quant_rag._select(rows, limit, per_document, dedupe=True, min_characters=min_characters)


# ---------------------------------------------------------------------- API publique

def search_graph(query: str, entity_type: str | None = None, relation: str | None = None,
                 top_k: int = 10, per_document: int = 2, min_characters: int = 150) -> dict:
    """Passages qui mentionnent une entité dont le nom contient ``query``.

    ``relation`` restreint aux passages où l'entité est *tête ou queue* d'une relation de ce
    type (« SVI » + ``uses_method`` : les passages où SVI utilise ou est utilisé par une méthode).
    """
    started = time.perf_counter()
    g = graph()
    matches = g.find(query, entity_type, limit=50)
    weights = {m["id"]: m["weight"] for m in matches}
    if relation:
        candidates = {edge["source_chunk_id"] for eid in weights
                      for edge in [*g.out_edges.get(eid, ()), *g.in_edges.get(eid, ())] if edge["relation"] == relation}
    else:
        candidates = g.chunks_of(list(weights))
    ranked = _rank_chunks(g, weights, candidates)
    results = _materialise(g, ranked, top_k, per_document, min_characters) if ranked else []
    return {"query": query, "entity_type": entity_type, "relation": relation,
            "entities": matches[:10], "entities_matched": len(matches),
            "candidates": len(candidates), "returned": len(results),
            "latency_ms": round((time.perf_counter() - started) * 1000), "results": results}


def expand_entity(entity_name: str, relation: str | None = None, entity_type: str | None = None,
                  limit: int = 20) -> dict:
    """Entités reliées à ``entity_name`` (toutes relations, ou une seule), avec les chunks sources.

    ``entity_name`` est résolu comme dans ``search_graph`` : l'entité exacte et ses variantes
    (« rough volatility », « rough volatility models »…) ; chaque voisin dit par quelle
    variante il est relié (``via``).
    """
    started = time.perf_counter()
    g = graph()
    origin = g.find(entity_name, entity_type, limit=50)
    ids = [m["id"] for m in origin]
    rows = g.neighbours(ids, relation)[:limit] if origin else []
    for row in rows:
        row["examples"] = [{"chunk_id": cid, **_source_of(g.chunk_document.get(cid)), "page_start": g.chunk_page.get(cid)}
                           for cid in row["chunk_ids"][:3]]
    co = g.co_mentioned(ids, limit=limit) if origin and not relation else []
    return {"entity": entity_name, "resolved": origin, "relation": relation,
            "relations_available": g.relations, "neighbours": rows, "co_mentioned": co,
            "latency_ms": round((time.perf_counter() - started) * 1000)}


def connect_entities(name_a: str, name_b: str, top_k: int = 10, per_document: int = 2) -> dict:
    """Passages qui mentionnent *les deux* entités, et les documents qui les relient."""
    started = time.perf_counter()
    g = graph()
    a, b = g.find(name_a, limit=50), g.find(name_b, limit=50)
    ids_a, ids_b = [m["id"] for m in a], [m["id"] for m in b]
    shared = g.chunks_of(ids_a) & g.chunks_of(ids_b)
    weights = {m["id"]: m["weight"] for m in a + b}
    ranked = _rank_chunks(g, weights, shared)
    results = _materialise(g, ranked, top_k, per_document, 150) if ranked else []
    documents = Counter(g.chunk_document.get(cid) for cid in shared)
    return {"a": a, "b": b, "shared_chunks": len(shared),
            "documents": [{**_source_of(doc), "chunks": n} for doc, n in documents.most_common(20)],
            "returned": len(results), "results": results,
            "latency_ms": round((time.perf_counter() - started) * 1000)}


def status() -> dict:
    g = graph()
    labels = Counter(node["label"] for node in g.entities.values())
    built_on = (g.report.get("corpus") or {}).get("signature")
    return {"graph": str(GRAPH_PATH), "corpus_signature_graph": built_on,
            "corpus_signature_current": corpus_overlay.signature(),
            "corpus_matches": built_on == corpus_overlay.signature() if built_on else None,
            "documents": len(set(g.chunk_document.values())), "chunks": len(g.chunk_document),
            "entities": len(g.entities), "entity_labels": dict(labels.most_common()),
            "noise_dropped": {"entities": g.dropped_entities, "mentions": g.dropped_mentions},
            "relation_edges": g.relation_edges, "relations": g.relations,
            "top_entities": [{"name": n["name"], "label": n["label"], "mentions": n["mentions"]}
                             for n in sorted(g.entities.values(), key=lambda n: -n["mentions"])[:15]]}


# ---------------------------------------------------------------------- affichage

def _qualite(row: dict) -> str:
    """La ligne « qualité : » du contrat de sortie, vide quand il n'y a rien à signaler.

    ``connect_entities`` servait **1 200** caractères là où ``search_graph`` en servait 2 500 —
    le même corpus, le même ``_payload_row``, le même bloc de rendu, deux fenêtres du simple au
    double. L'écart n'était documenté nulle part et rien ne le justifiait. Les deux passent
    désormais par ``contrat.PASSAGE_CHARACTERS``.
    """
    ligne = contrat.ligne_qualite(contrat.drapeaux_qualite(
        row.get("text") or "", tronque=len(row.get("text") or "") > contrat.PASSAGE_CHARACTERS))
    return f"    qualité : {ligne}\n" if ligne else ""


def format_search(out: dict) -> str:
    ents = ", ".join(f"{e['label']}:{e['name']} ({e['mentions']})" for e in out["entities"][:5])
    head = (f"graphe: {out['entities_matched']} entité(s) pour « {out['query']} »"
            + (f" [{out['entity_type']}]" if out.get("entity_type") else "")
            + (f" · relation {out['relation']}" if out.get("relation") else "")
            + f" — {ents}" + (" …" if out["entities_matched"] > 5 else "")
            + f" · {out['candidates']} passages candidats · {out['returned']} retournés · {out['latency_ms']} ms")
    if not out["results"]:
        return head + "\n\nAucun passage."
    blocks = [head]
    for i, row in enumerate(out["results"], 1):
        blocks.append(f"[{i}] graph={row['score']:.2f}\n"
                      f"    citer   : {row['source']}, "
                      f"{quant_rag.citation_pages(row.get('pages_utilisateur'))}\n"
                      f"    section : {row['section'] or '—'}\n"
                      f"    entités : {', '.join(row['entities'])}\n"
                      f"    interne : chunk_id={row['chunk_id']} · document_id={row['document_id']} · "
                      f"{row['content_type']} (clés de travail, ne pas citer)\n"
                      + _qualite(row)
                      + f"\n{contrat.couper_passage(row['text'], chunk_id=row['chunk_id'])}")
    return "\n\n---\n\n".join(blocks)


def format_expand(out: dict) -> str:
    if not out["resolved"]:
        return f"Aucune entité ne correspond à « {out['entity'] }»."
    origin = ", ".join(f"{m['label']}:{m['name']} ({m['mentions']})" for m in out["resolved"][:5])
    lines = [f"entité : {origin}" + (" …" if len(out["resolved"]) > 5 else "")
             + (f" · relation {out['relation']}" if out.get("relation") else "")
             + f" · {len(out['neighbours'])} voisin(s) · {out['latency_ms']} ms"]
    if not out["neighbours"]:
        lines.append(f"Aucune relation. Relations possibles : {', '.join(out['relations_available'])}.")
    for n in out["neighbours"]:
        arrow = f"—{n['relation']}→" if n["direction"] == "out" else f"←{n['relation']}—"
        ex = "; ".join(f"{e['short_ref'] or e['title'] or e['document_id']} "
                       f"{quant_rag.citation_pages(quant_rag.pages_utilisateur(e['page_start'], None))}"
                       f" [interne {e['chunk_id']}]"
                       for e in n["examples"])
        via = f" via {', '.join(n['via'])}" if len(out["resolved"]) > 1 else ""
        lines.append(f"- {arrow} {n['label']}:{n['name']} ({n['mentions']} mentions, {n['support']} passage(s){via})\n    {ex}")
    if out.get("co_mentioned"):
        lines.append("\nco-mentions (mêmes passages, pondérées par rareté) :")
        for c in out["co_mentioned"]:
            lines.append(f"- {c['label']}:{c['name']} — {c['shared_chunks']} passage(s) commun(s), {c['mentions']} mentions au total")
    return "\n".join(lines)


def format_connect(out: dict) -> str:
    if not out["a"] or not out["b"]:
        return "Une des deux entités est introuvable : " + json.dumps({"a": out["a"], "b": out["b"]}, ensure_ascii=False)
    lines = [f"{out['a'][0]['name']} × {out['b'][0]['name']} : {out['shared_chunks']} passage(s) commun(s), "
             f"{len(out['documents'])} document(s) · {out['latency_ms']} ms"]
    for d in out["documents"][:10]:
        lines.append(f"- {d['source']} · {d['chunks']} passage(s) · document_id={d['document_id']}")
    for i, row in enumerate(out["results"], 1):
        lines.append(f"\n[{i}] graph={row['score']:.2f}\n    citer   : {row['source']}, "
                     f"{quant_rag.citation_pages(row.get('pages_utilisateur'))}\n"
                     f"    section : {row['section'] or '—'}\n"
                     f"    interne : chunk_id={row['chunk_id']} (clé de travail, ne pas citer)\n"
                     + _qualite(row)
                     + f"\n{contrat.couper_passage(row['text'], chunk_id=row['chunk_id'])}")
    return "\n".join(lines)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Requête du graphe d'entités")
    parser.add_argument("query", nargs="*")
    parser.add_argument("--type", choices=ENTITY_LABELS)
    parser.add_argument("--relation")
    parser.add_argument("--expand", action="store_true", help="voisins de l'entité (query = nom)")
    parser.add_argument("--connect", action="store_true", help="passages communs à deux entités (query = A B)")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    if args.status:
        print(json.dumps(status(), indent=2, ensure_ascii=False))
    elif args.expand and args.query:
        out = expand_entity(" ".join(args.query), relation=args.relation, entity_type=args.type, limit=args.limit)
        print(json.dumps(out, indent=2, ensure_ascii=False) if args.json else format_expand(out))
    elif args.connect and len(args.query) == 2:
        out = connect_entities(args.query[0], args.query[1], top_k=args.limit)
        print(json.dumps(out, indent=2, ensure_ascii=False) if args.json else format_connect(out))
    elif args.query:
        out = search_graph(" ".join(args.query), entity_type=args.type, relation=args.relation, top_k=args.limit)
        print(json.dumps(out, indent=2, ensure_ascii=False) if args.json else format_search(out))
    else:
        parser.error("donne une entité, ou --status ; --connect attend deux noms entre guillemets")
