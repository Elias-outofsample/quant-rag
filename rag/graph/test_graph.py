"""Tests unitaires du graphe d'entités : construction (build_graph) et requête (graph_search).

    .venv/bin/python -m pytest rag/graph/test_graph.py -q

Aucun modèle, aucun Qdrant : un ``results.jsonl`` synthétique de trois chunks, et la
jointure Qdrant est remplacée par un dictionnaire.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import build_graph  # noqa: E402
import graph_search  # noqa: E402


def mention(text, start=0):
    return {"text": text, "start": start, "end": start + len(text)}


RESULTS = [
    {"chunk_id": "c1", "document_id": "d1", "page_start": 3, "page_end": 3, "text_chars": 500,
     "entities": {"person": [mention("Gatheral"), mention("Jim Gatheral", 20)],
                  "market_concept": [mention("rough volatility", 40), mention("SVI", 60)],
                  "measure": [mention("x", 70)]},
     "relations": {"uses_method": [{"head": mention("rough volatility", 40), "tail": mention("SVI", 60)}]},
     "relation_count": 1},
    {"chunk_id": "c2", "document_id": "d1", "page_start": 4, "page_end": 4, "text_chars": 500,
     "entities": {"person": [mention("L´opez de Prado")], "market_concept": [mention("rough volatility models", 30)]},
     "relations": {"applies_to": [{"head": mention("rough volatility models", 30), "tail": mention("option pricing", 60)}]},
     "relation_count": 1},
    {"chunk_id": "c3", "document_id": "d2", "page_start": 1, "page_end": 1, "text_chars": 500,
     "entities": {"person": [mention("López de Prado")], "market_concept": [mention("SVI")], "method": [mention("mathrm")]},
     "relations": {}, "relation_count": 0},
]


@pytest.fixture(scope="module")
def graph(monkeypatch_module):
    nodes, edges = build_graph.build(RESULTS)
    g = graph_search.Graph({"report": {"corpus": {"signature": "test"}}, "nodes": list(nodes.values()), "edges": edges})
    monkeypatch_module.setattr(graph_search, "graph", lambda: g)
    monkeypatch_module.setattr(graph_search, "_fetch_rows", lambda chunk_ids: {
        cid: {"chunk_id": cid, "document_id": g.chunk_document[cid], "title": "T", "short_ref": "Ref (2020)",
              "source": "Ref (2020) — T", "section": None, "page_start": 1, "page_end": 1, "content_type": "text",
              "score": 0.0, "score_kind": "graph", "text": "x" * 400 + cid} for cid in chunk_ids})
    monkeypatch_module.setattr(graph_search.quant_rag, "document_metadata", lambda: {
        "d1": {"title": "Doc 1", "short_ref": "A (2020)"}, "d2": {"title": "Doc 2", "short_ref": "B (2021)"}})
    return g


@pytest.fixture(scope="module")
def monkeypatch_module():
    from _pytest.monkeypatch import MonkeyPatch

    mp = MonkeyPatch()
    yield mp
    mp.undo()


def test_fold_unifies_accents_and_spacing_accents():
    assert graph_search.fold("L´opez de Prado") == graph_search.fold("López de Prado") == "lopez de prado"
    assert graph_search.fold("  Jim   GATHERAL ") == "jim gatheral"


def test_noise_filter():
    assert graph_search.is_noise("x") and graph_search.is_noise("S&") and not graph_search.is_noise("S&P")
    assert graph_search.is_noise("rowspan") and graph_search.is_noise("mathrm")
    assert graph_search.is_noise("123") and not graph_search.is_noise("SVI")


def test_build_keeps_upstream_format():
    nodes, edges = build_graph.build(RESULTS)
    assert nodes["entity:person:gatheral"]["mentions"] == 1
    assert nodes["entity:person:gatheral"]["source_chunks"] == ["c1"]
    assert nodes["document:d1"]["type"] == "document" and nodes["chunk:c1"]["page_start"] == 3
    relations = [e for e in edges if e["relation"] not in build_graph.STRUCTURAL]
    assert {(e["source"], e["relation"], e["target"]) for e in relations} == {
        ("entity:market_concept:rough volatility", "uses_method", "entity:market_concept:svi"),
        ("entity:market_concept:rough volatility models", "applies_to", "entity:untyped:option pricing"),
    }
    assert sum(e["relation"] == "contains_chunk" for e in edges) == 3


def test_graph_drops_noise_and_indexes(graph):
    assert "entity:measure:x" not in graph.entities and "entity:method:mathrm" not in graph.entities
    assert graph.dropped_entities == 2 and graph.dropped_mentions == 2
    assert graph.chunk_document == {"c1": "d1", "c2": "d1", "c3": "d2"}
    assert graph.relations == ["applies_to", "uses_method"]


def test_find_exact_first_then_variants(graph):
    found = graph.find("rough volatility")
    assert [f["name"] for f in found] == ["rough volatility", "rough volatility models"]
    assert found[0]["exact"] and found[0]["weight"] == 1.0 and not found[1]["exact"]
    assert [f["name"] for f in graph.find("gatheral", entity_type="person")] == ["Gatheral", "Jim Gatheral"]
    assert graph.find("prado") and {f["name"] for f in graph.find("lopez de prado")} == {"L´opez de Prado", "López de Prado"}
    assert graph.find("") == []


def test_search_graph_ranks_and_joins(graph):
    out = graph_search.search_graph("SVI", top_k=5, per_document=0)
    assert out["entities_matched"] == 1 and out["candidates"] == 2 and out["returned"] == 2
    assert {r["chunk_id"] for r in out["results"]} == {"c1", "c3"}
    assert all(r["score_kind"] == "graph" and r["entities"] == ["SVI"] for r in out["results"])
    out = graph_search.search_graph("SVI", relation="uses_method", top_k=5)
    assert [r["chunk_id"] for r in out["results"]] == ["c1"]
    assert graph_search.search_graph("nothing here")["results"] == []


def test_expand_entity_merges_variants(graph):
    out = graph_search.expand_entity("rough volatility")
    names = {(n["relation"], n["direction"], n["name"]) for n in out["neighbours"]}
    assert names == {("uses_method", "out", "SVI"), ("applies_to", "out", "option pricing")}
    by_name = {n["name"]: n for n in out["neighbours"]}
    assert by_name["option pricing"]["via"] == ["rough volatility models"]
    assert by_name["SVI"]["examples"][0]["chunk_id"] == "c1" and by_name["SVI"]["examples"][0]["short_ref"] == "A (2020)"
    assert graph_search.expand_entity("SVI", relation="uses_method")["neighbours"][0]["direction"] == "in"
    assert graph_search.expand_entity("unknown")["resolved"] == []


def test_co_mentions_weighted_by_rarity(graph):
    out = graph_search.expand_entity("Gatheral")
    co = {c["name"]: c for c in out["co_mentioned"]}
    assert set(co) == {"rough volatility", "SVI"}          # même passage c1, variantes de Gatheral exclues
    assert co["rough volatility"]["score"] > co["SVI"]["score"]  # SVI est dans 2 chunks sur 3, rough volatility dans 1
    assert graph_search.expand_entity("Gatheral", relation="is_a")["co_mentioned"] == []


def test_connect_entities(graph):
    out = graph_search.connect_entities("Gatheral", "SVI")
    assert out["shared_chunks"] == 1 and out["documents"][0]["document_id"] == "d1"
    assert [r["chunk_id"] for r in out["results"]] == ["c1"]
    assert graph_search.connect_entities("Gatheral", "prado")["shared_chunks"] == 0


def test_formatters_do_not_crash(graph):
    assert "graphe:" in graph_search.format_search(graph_search.search_graph("SVI"))
    assert "entité :" in graph_search.format_expand(graph_search.expand_entity("SVI"))
    assert "×" in graph_search.format_connect(graph_search.connect_entities("Gatheral", "SVI"))
    assert "Aucune" in graph_search.format_expand(graph_search.expand_entity("zzz"))
    assert json.dumps(graph_search.search_graph("SVI"))  # sérialisable (MCP)
