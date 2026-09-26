"""Rappel du pool — le plafond de chaque générateur de candidats, avant toute construction.

Question posée. Un reclassement réordonne un pool et n'y ajoute rien : les trois rerankers
mesurés le 5 septembre laissent le nombre de ratés à 27, 27, 27 sur les 155 questions des
deux bancs gelés. La contrainte liante n'est donc ni la fusion ni le reclassement, c'est la
**génération de candidats**. Ce script mesure, pour chaque générateur candidat, le majorant
de ce qu'il pourrait récupérer — sans rien construire, sans rien intégrer.

Un plafond est un majorant : il dit ce qu'un système parfait bâti sur ce générateur
atteindrait au mieux. S'il est nul ou négligeable, aucune ingénierie en aval ne le
rattrapera. C'est la seule mesure qui puisse fermer une avenue avant qu'on la creuse.

Protocole pré-enregistré, et commité avant la première mesure :
``rag/benchmark/RAPPORT-RAPPEL-DU-POOL-2026-09-05.md``, Partie I.

    pool de référence   dense limit=50 (pipeline.POOL), filtres de production
    seuil               N = 4 ratés récupérés (dérivé de metrics.paired_delta, cf. §3)
    générateurs         dense@100 · BM25@20/@50 · graphe 1-2 sauts · voisinage documentaire

Quatre étapes, une par processus. La règle « un processus lourd à la fois » n'est pas une
précaution : Qdrant embarqué est mono-processus, la machine a 16 Go, et trois incidents de
saturation mémoire sont documentés dans ce dépôt. Chaque étape écrit son cache et peut
être rejouée seule.

    .venv/bin/python rag/benchmark/eval_pool_recall.py --step offline    # BM25 + voisinage
    .venv/bin/python rag/benchmark/eval_pool_recall.py --step graph      # graph-lite.json, 69 Mo
    .venv/bin/python rag/benchmark/eval_pool_recall.py --step dense      # Qdrant + embedder
    .venv/bin/python rag/benchmark/eval_pool_recall.py --step report     # assemble et tranche

    .venv/bin/python rag/benchmark/eval_pool_recall.py --step offline --sabotage
    .venv/bin/python rag/benchmark/eval_pool_recall.py --step dense --sabotage
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(ROOT / "src"))

import corpus_overlay  # noqa: E402
import metrics  # noqa: E402
import pipeline  # noqa: E402
import quant_rag  # noqa: E402
from compare_v1_v2 import load_bench, load_v1  # noqa: E402
from corpus import ChunkIndex  # noqa: E402

CACHE = HERE / ".cache"
GRAPH = ROOT / "data" / "graph" / "graph-lite.json"
OPEN_BENCH = "v3"

#: Profondeur du pool de référence. C'est ``pipeline.POOL`` : la profondeur à laquelle la
#: ligne de base, la grille de fusion et les trois rerankers ont tous été mesurés.
REFERENCE_DEPTH = 50

#: Profondeur du générateur (a). Le pool est à 50 depuis les premiers bancs ; personne n'a
#: regardé au-delà, et c'est l'hypothèse la moins chère à réfuter.
DENSE_DEPTH = 100

#: Profondeurs déclarées du générateur (b), avant mesure.
BM25_DEPTHS = (20, 50)

#: Nombre de documents de tête étendus par le générateur (d), avant mesure.
NEIGHBOUR_DOCS = (10, 50)

#: Plafond de chunks par entité pour le générateur (c). Grille déclarée *a priori* : une
#: grille ne peut pas être réglée sur son résultat. ``None`` = aucun plafond, c'est le
#: plafond absolu du graphe indépendamment de toute politique de filtrage.
ENTITY_CAPS = (10, 25, 50, 100, None)

#: Seuil d'arrêt pré-engagé. Dérivé, pas choisi : avec k uns et 155-k zéros,
#: ``metrics.paired_delta`` donne [0,000 ; 0,045] à k=3 et [+0,006 ; +0,052] à k=4.
N_SEUIL = 4


# --------------------------------------------------------------------------- socle commun

def bench_items(index: ChunkIndex) -> list[tuple[str, dict]]:
    """Les 155 questions des deux bancs gelés, dans l'ordre où le banc les lit."""
    out = [("v1", item) for item in load_v1(index)]
    out += [(OPEN_BENCH, item) for item in load_bench(HERE / f"questions-{OPEN_BENCH}.jsonl")]
    return out


def reference_pools(signature: str) -> dict[str, list[tuple[str, str]]]:
    """Le pool dense de production, tel que la ligne de base l'a vu.

    Lu dans ``.cache/router-retrievals-<signature>.json`` — le cache produit par
    ``pipeline.retrieve_item(item, "dense", …)``, c'est-à-dire la ligne de base au sens
    strict, pas une reconstruction. Le contrôle C2 a vérifié que ces listes sont
    identiques entre ``bb7bf33c37`` et ``10390927db``, ordre compris, 155/155.
    """
    path = CACHE / f"router-retrievals-{signature}.json"
    if not path.exists():
        sys.exit(f"cache de classements absent ({path.name}) : lance calibrate_router.py")
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {key: [(c, d) for c, d, _ in entry["dense"]] for key, entry in raw.items()}


def build_frame(index: ChunkIndex, signature: str, sabotage: bool = False) -> dict:
    """Le cadre commun à toutes les étapes : pools, ors, et le jeu des ratés.

    ``sabotage`` retire le premier chunk d'or trouvé du pool de chaque question : le
    jeu des ratés enfle, et le contrôle C1 doit refuser de reconnaître les archives.
    """
    pools = reference_pools(signature)
    frame = {}
    for bench, item in bench_items(index):
        key = f"{bench}/{item['qid']}"
        if key not in pools:
            sys.exit(f"{key} absent du cache de classements")
        pool = pools[key]
        gold_chunks, gold_documents = set(item["gold_chunks"]), set(item["gold_documents"])
        if sabotage:
            pool = [(c, d) for c, d in pool if c not in gold_chunks]
        pool_chunks = [c for c, _ in pool]
        pool_documents = [d for _, d in pool]
        frame[key] = {
            "bench": bench, "qid": item["qid"], "kind": item.get("kind", "single"),
            "item": item,
            "filters": pipeline.filters_of(item) or None,
            "pool_chunks": pool_chunks,
            "pool_chunk_set": set(pool_chunks),
            "pool_documents": pool_documents,
            "gold_chunks": gold_chunks,
            "gold_documents": gold_documents,
            "n_pool": len(pool_chunks),
            "hit": bool(gold_chunks & set(pool_chunks)),
            "doc_hit": bool(gold_documents & set(pool_documents)),
        }
    return frame


def misses_of(frame: dict) -> list[str]:
    """Les questions dont le chunk d'or est absent du pool de référence, dans l'ordre du banc."""
    return [key for key, row in frame.items() if not row["hit"]]


def recovery_line(frame: dict, misses: list[str], recovered: set[str], budget: list[int]) -> dict:
    """Une ligne du tableau des plafonds, avec son intervalle apparié sur les 155.

    Le vecteur d'écarts vaut 1 pour une question récupérée et 0 partout ailleurs : l'union
    est monotone, un générateur ajouté ne peut pas retirer une cible du pool. C'est
    exactement la forme sur laquelle le seuil N a été calibré.
    """
    total = len(frame)
    base = [1.0 if frame[k]["hit"] else 0.0 for k in frame]
    variant = [1.0 if (frame[k]["hit"] or k in recovered) else 0.0 for k in frame]
    delta = metrics.paired_delta(base, variant)
    return {
        "recovered": len(recovered),
        "recovered_keys": sorted(recovered),
        "misses_before": len(misses),
        "misses_after": len(misses) - len(recovered),
        "pool_recall_before": round(sum(base) / total, 3),
        "pool_recall_after": round(sum(variant) / total, 3),
        "delta": delta.get("delta"),
        "ci95": delta.get("ci95"),
        "significant": delta.get("significant"),
        "budget_mean": round(statistics.mean(budget), 1) if budget else 0.0,
        "budget_median": round(statistics.median(budget), 1) if budget else 0.0,
        "budget_max": max(budget) if budget else 0,
        "passes": len(recovered) >= N_SEUIL,
    }


def _sha256(path: Path) -> str:
    """Empreinte d'une entrée. Un résultat sans l'empreinte de ce qu'il a lu n'est pas rejouable."""
    import hashlib

    if not path.exists():
        return "absent"
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for bloc in iter(lambda: handle.read(1 << 20), b""):
            digest.update(bloc)
    return digest.hexdigest()


def write_cache(name: str, signature: str, payload: dict) -> Path:
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"pool-{name}-{signature}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    print(f"  écrit  {path.name}")
    return path


def read_cache(name: str, signature: str) -> dict | None:
    path = CACHE / f"pool-{name}-{signature}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


# --------------------------------------------------------- étape 1 : hors ligne (b) et (d)

def controle_c1(frame: dict, misses: list[str], bm25_ranks: dict[str, int | None]) -> list[str]:
    """Les archives de la grille de fusion, recomptées. Le contrôle qui autorise le reste.

    ``results-fusion-titres-registre.json`` porte la signature ``bb7bf33c37`` et l'index
    BM25 servi aujourd'hui. Il doit donner exactement 27 ratés dense et 6 récupérations
    BM25@50 — sinon le pool que je manipule n'est pas celui qui a été mesuré, et tout ce
    qui suit compare des choses incomparables.
    """
    archive = HERE / "results-fusion-titres-registre.json"
    if not archive.exists():
        return [f"{archive.name} absent : impossible de vérifier le jeu des ratés"]
    data = json.loads(archive.read_text(encoding="utf-8"))
    per_question = {f"{r['bench']}/{r['qid']}": r for r in data["per_question"]}
    ecarts = []
    archived_misses = {k for k, r in per_question.items() if r["dense"]["first_rank"] is None}
    if archived_misses != set(misses):
        seuls_moi = sorted(set(misses) - archived_misses)
        seuls_archive = sorted(archived_misses - set(misses))
        ecarts.append(f"jeu des ratés : {len(misses)} chez moi, {len(archived_misses)} dans l'archive"
                      f" | chez moi seulement {seuls_moi[:6]} | archive seulement {seuls_archive[:6]}")
    archived_recovered = {k for k in archived_misses
                          if per_question[k]["bm25"]["first_rank"] is not None}
    mine = {k for k in misses if bm25_ranks.get(k) is not None}
    if archived_recovered != mine:
        ecarts.append(f"récupérations BM25@50 : {len(mine)} chez moi, {len(archived_recovered)} dans l'archive"
                      f" | écart {sorted(archived_recovered ^ mine)[:8]}")
    for key in sorted(set(misses) & archived_misses):
        attendu = per_question[key]["bm25"]["first_rank"]
        obtenu = bm25_ranks.get(key)
        if attendu != obtenu:
            ecarts.append(f"{key} : rang BM25 {obtenu} au lieu de {attendu}")
    return ecarts


def step_offline(index: ChunkIndex, signature: str, sabotage: bool) -> None:
    frame = build_frame(index, signature, sabotage=sabotage)
    misses = misses_of(frame)
    print(f"\n  pool de référence : dense limit={REFERENCE_DEPTH}, filtres de production")
    tailles = [frame[k]["n_pool"] for k in frame]
    print(f"  taille effective : min {min(tailles)} · médiane {int(statistics.median(tailles))} · max {max(tailles)}")
    print(f"  ratés : {len(misses)}/{len(frame)}  (rappel de pool {1 - len(misses)/len(frame):.3f})")

    # --- anatomie des ratés : le document d'or est-il déjà là ?
    doc_present = [k for k in misses if frame[k]["doc_hit"]]
    print(f"  dont le document d'or est DÉJÀ dans le pool : {len(doc_present)}")

    # --- (b) BM25, recalculé plutôt que relu : une reproduction indépendante de l'archive
    print(f"\n  (b) BM25 — index servi : {quant_rag.bm25_path().name}")
    # Les 155, pas seulement les 27 ratés. Ne calculer que les ratés suffirait au *plafond*
    # mais pas au *budget* : le budget d'un générateur est ce qu'il ajoute au pool sur toutes
    # les requêtes, pas sur celles où il sert. Un forfait pour les 128 autres questions —
    # 50 candidats neufs, ou zéro — se trompe dans les deux sens, et de beaucoup : BM25
    # recoupe largement le pool dense.
    started = time.perf_counter()
    bm25_lists: dict[str, list[tuple[str, str]]] = {}
    for key, row in frame.items():
        filters = row["filters"] or {}
        scope = quant_rag.document_scope(None, **filters) if filters else None
        lexical = quant_rag._lexical(pipeline.query_of(row["item"]), max(BM25_DEPTHS), scope)
        bm25_lists[key] = [(r["chunk_id"], r["document_id"]) for r in lexical]
    print(f"      {len(frame)} requêtes lexicales en {time.perf_counter() - started:.1f} s, aucun modèle chargé")

    bm25_ranks = {}
    for key in misses:
        gold = frame[key]["gold_chunks"]
        rank = next((i for i, (c, _) in enumerate(bm25_lists[key], 1) if c in gold), None)
        bm25_ranks[key] = rank

    ecarts = controle_c1(frame, misses, bm25_ranks)
    print("\n  === contrôle C1 — le jeu des ratés et les récupérations BM25 reproduisent l'archive ===")
    if ecarts:
        for ligne in ecarts[:12]:
            print(f"      ÉCART  {ligne}")
    else:
        print("      27 ratés et 6 récupérations BM25@50, identiques à results-fusion-titres-registre.json")
    if sabotage:
        print("\n  --sabotage : le contrôle C1 ci-dessus DOIT signaler des écarts.")
        return
    if ecarts:
        sys.exit("\nARRÊT : le jeu des ratés ne reproduit pas l'archive, rien de ce qui suit ne mesure quoi que ce soit.")

    lines, atomes = {}, {}
    for depth in BM25_DEPTHS:
        recovered = {k for k in misses if bm25_ranks[k] is not None and bm25_ranks[k] <= depth}
        budget = [len({c for c, _ in bm25_lists[key][:depth]} - frame[key]["pool_chunk_set"])
                  for key in frame]
        lines[f"bm25@{depth}"] = recovery_line(frame, misses, recovered, budget)
        atomes[f"bm25@{depth}"] = {k: sorted({c for c, _ in bm25_lists[k][:depth]}
                                             - frame[k]["pool_chunk_set"]) for k in frame}

    # --- (d) voisinage documentaire : les chunks des documents déjà présents
    print("\n  (d) voisinage documentaire — aucun modèle, aucun index")
    by_document: dict[str, list[str]] = defaultdict(list)
    for chunk_id, chunk in index.chunks.items():
        by_document[chunk["document_id"]].append(chunk_id)
    for depth in NEIGHBOUR_DOCS:
        recovered, budget = set(), []
        ajoutes_par_question = {}
        for key, row in frame.items():
            tete, vus = [], set()
            for document in row["pool_documents"]:
                if document not in vus:
                    vus.add(document)
                    tete.append(document)
                if len(tete) >= depth:
                    break
            voisins = {c for document in tete for c in by_document.get(document, [])}
            ajoutes_par_question[key] = sorted(voisins - row["pool_chunk_set"])
            budget.append(len(voisins - row["pool_chunk_set"]))
            if not row["hit"] and (voisins & row["gold_chunks"]):
                recovered.add(key)
        lines[f"voisinage@{depth}doc"] = recovery_line(frame, misses, recovered, budget)
        if depth == min(NEIGHBOUR_DOCS):
            atomes[f"voisinage@{depth}doc"] = ajoutes_par_question

    payload = {
        "signature": signature,
        "reference": {"depth": REFERENCE_DEPTH, "min": min(tailles),
                      "median": statistics.median(tailles), "max": max(tailles)},
        "misses": misses,
        "anatomie": {
            "n_misses": len(misses),
            "document_or_deja_dans_le_pool": sorted(doc_present),
            "par_banc": dict(Counter(frame[k]["bench"] for k in misses)),
            "par_famille": dict(Counter(frame[k]["kind"] for k in misses)),
            "par_famille_total": dict(Counter(frame[k]["kind"] for k in frame)),
            "avec_filtre": sum(1 for k in misses if frame[k]["filters"]),
            "detail": [{"key": k, "bench": frame[k]["bench"], "qid": frame[k]["qid"],
                        "kind": frame[k]["kind"], "n_pool": frame[k]["n_pool"],
                        "filters": frame[k]["filters"], "doc_hit": frame[k]["doc_hit"],
                        "n_gold_chunks": len(frame[k]["gold_chunks"]),
                        "bm25_rank": bm25_ranks[k]} for k in misses],
        },
        "controles": {"C1": ecarts or "reproduit"},
        "lines": lines,
        "bm25_index": quant_rag.bm25_path().name,
        "atomes": atomes,
    }
    write_cache("offline", signature, payload)
    print_lines(lines)


# ------------------------------------------------------------------ étape 2 : graphe (c)

def step_graph(index: ChunkIndex, signature: str) -> None:
    frame = build_frame(index, signature)
    misses = misses_of(frame)
    print(f"\n  (c) graphe — {GRAPH.name} ({GRAPH.stat().st_size / 1e6:.0f} Mo)")
    started = time.perf_counter()
    raw = json.loads(GRAPH.read_text(encoding="utf-8"))
    print(f"      chargé en {time.perf_counter() - started:.1f} s")

    report = raw["report"]
    entity_chunks: dict[str, list[str]] = {}
    for node in raw["nodes"]:
        if node["type"] == "entity":
            entity_chunks[node["id"]] = node.get("source_chunks") or []
    chunk_entities: dict[str, list[str]] = defaultdict(list)
    for eid, chunks in entity_chunks.items():
        for chunk_id in chunks:
            chunk_entities[chunk_id].append(eid)
    relations: dict[str, set[str]] = defaultdict(set)
    for edge in raw["edges"]:
        if edge["relation"] in ("mentions", "contains_chunk"):
            continue
        relations[edge["source"]].add(edge["target"])
        relations[edge["target"]].add(edge["source"])
    del raw

    # --- C4 : le graphe décrit-il le corpus courant ?
    graph_chunks = set(chunk_entities)
    corpus_chunks = set(index.chunks)
    hors_corpus = graph_chunks - corpus_chunks
    print("\n  === contrôle C4 — le graphe décrit-il le corpus servi ? ===")
    print(f"      graphe bâti à la signature {report['corpus']['signature']}, corpus à {signature}")
    print(f"      chunks du graphe : {len(graph_chunks)} · dont hors du corpus courant : {len(hors_corpus)}")
    print(f"      chunks du corpus sans entité : {len(corpus_chunks - graph_chunks)}")

    fan_out = sorted((len(v) for v in entity_chunks.values()), reverse=True)
    quantiles = {f"p{p}": fan_out[min(len(fan_out) - 1, int(len(fan_out) * (100 - p) / 100))]
                 for p in (50, 75, 90, 95, 99)}
    print(f"\n      fan-out par entité : max {fan_out[0]} · " +
          " · ".join(f"{k} {v}" for k, v in quantiles.items()))
    for cap in ENTITY_CAPS:
        if cap is not None:
            gardees = sum(1 for n in fan_out if n <= cap)
            print(f"      plafond {cap:>4} : {gardees}/{len(fan_out)} entités gardées "
                  f"({gardees / len(fan_out):.1%})")

    lines, atomes = {}, {}
    for cap in ENTITY_CAPS:
        etiquette = "∞" if cap is None else str(cap)

        def voisins_1(chunks: set[str]) -> tuple[set[str], set[str]]:
            entites = set()
            for chunk_id in chunks:
                for eid in chunk_entities.get(chunk_id, ()):
                    if cap is None or len(entity_chunks[eid]) <= cap:
                        entites.add(eid)
            atteints = {c for eid in entites for c in entity_chunks[eid]}
            return entites, atteints

        for sauts in (1, 2):
            recovered, budget = set(), []
            ajoutes_par_question = {}
            for key, row in frame.items():
                entites, atteints = voisins_1(row["pool_chunk_set"])
                if sauts == 2:
                    voisines = set()
                    for eid in entites:
                        for autre in relations.get(eid, ()):
                            if autre in entity_chunks and (cap is None or len(entity_chunks[autre]) <= cap):
                                voisines.add(autre)
                    atteints |= {c for eid in voisines for c in entity_chunks[eid]}
                atteints &= corpus_chunks
                if cap == 10 and sauts == 1:
                    ajoutes_par_question[key] = sorted(atteints - row["pool_chunk_set"])
                budget.append(len(atteints - row["pool_chunk_set"]))
                if not row["hit"] and (atteints & row["gold_chunks"]):
                    recovered.add(key)
            lines[f"graphe@{sauts}saut/cap{etiquette}"] = recovery_line(frame, misses, recovered, budget)
            if cap == 10 and sauts == 1:
                atomes[f"graphe@{sauts}saut/cap{etiquette}"] = ajoutes_par_question

    payload = {
        "signature": signature,
        "graphe": {"fichier": GRAPH.name, "signature_construction": report["corpus"]["signature"],
                   "noeuds": report["nodes"], "entites": report["entity_nodes"],
                   "aretes": report["edges"], "aretes_mention": report["mention_edges"],
                   "aretes_relation": report["relation_edges"]},
        "controles": {"C4": {"chunks_graphe": len(graph_chunks),
                             "hors_corpus_courant": len(hors_corpus),
                             "corpus_sans_entite": len(corpus_chunks - graph_chunks)}},
        "fan_out": {"max": fan_out[0], "moyenne": round(statistics.mean(fan_out), 2), **quantiles},
        "lines": lines,
        "atomes": atomes,
    }
    write_cache("graph", signature, payload)
    print_lines(lines)


# ------------------------------------------------------------- étape 3 : dense@100 (a)

def step_dense(index: ChunkIndex, signature: str, sabotage: bool) -> None:
    frame = build_frame(index, signature)
    misses = misses_of(frame)
    print(f"\n  (a) dense@{DENSE_DEPTH} — la seule étape qui charge un modèle")
    started = time.perf_counter()
    listes: dict[str, list[tuple[str, str]]] = {}
    latences = []
    for n, (key, row) in enumerate(frame.items(), 1):
        t0 = time.perf_counter()
        # `pipeline.retrieve` code en dur ``limit=POOL, pool=POOL`` et n'applique le ``limit``
        # demandé qu'en troncature *après* la requête : il ne peut pas produire une profondeur
        # supérieure à 50, et rendrait 47 candidats en croyant en rendre 100. On appelle donc
        # `quant_rag.search` directement, avec exactement les mêmes options que la branche
        # dense de `pipeline.retrieve` — seule la profondeur change.
        ranked = quant_rag.search(pipeline.query_of(row["item"]), limit=DENSE_DEPTH,
                                  pool=DENSE_DEPTH, mode="dense", rerank=False, auto_period=False,
                                  dedupe=False, per_document=0, min_characters=0, log=False,
                                  **(row["filters"] or {}))
        latences.append((time.perf_counter() - t0) * 1000)
        listes[key] = [(r["chunk_id"], r["document_id"]) for r in ranked]
        print(f"      {n}/{len(frame)}  {key:<10} {len(ranked):>3} candidats", end="\r")
    print(" " * 70, end="\r")
    print(f"      {len(frame)} requêtes en {time.perf_counter() - started:.0f} s "
          f"({statistics.median(latences):.0f} ms p50)")

    # --- C3 : les 50 premiers reproduisent le cache, question par question.
    # La troncature des en-têtes préserve l'ordre : le pool de référence doit donc être un
    # PRÉFIXE exact de la liste à profondeur 100. Si ce n'est pas le cas, la collection a
    # bougé sous la mesure et le plafond ne veut rien dire.
    ecarts = []
    for key, row in frame.items():
        prefixe = [c for c, _ in listes[key][:row["n_pool"]]]
        if sabotage:
            prefixe = prefixe[1:] + prefixe[:1]
        if prefixe != row["pool_chunks"]:
            ecarts.append(key)
    print("\n  === contrôle C3 — le pool de référence est un préfixe exact du dense@100 ===")
    if ecarts:
        print(f"      ÉCART sur {len(ecarts)}/{len(frame)} questions : {ecarts[:8]}")
    else:
        print(f"      {len(frame)}/{len(frame)} questions : le cache est reproduit à l'identique")
    if sabotage:
        print("\n  --sabotage : le contrôle C3 ci-dessus DOIT signaler des écarts.")
        return
    if ecarts:
        sys.exit("\nARRÊT : la collection ne reproduit pas le pool de référence.")

    recovered, budget = set(), []
    profondeurs = []
    for key, row in frame.items():
        chunks = {c for c, _ in listes[key]}
        profondeurs.append(len(chunks))
        budget.append(len(chunks - row["pool_chunk_set"]))
        if not row["hit"] and (chunks & row["gold_chunks"]):
            recovered.add(key)
    lines = {f"dense@{DENSE_DEPTH}": recovery_line(frame, misses, recovered, budget)}

    # Rang exact de la cible récupérée : dit si la profondeur 100 est généreuse ou juste.
    rangs = {}
    for key in sorted(recovered):
        gold = frame[key]["gold_chunks"]
        rangs[key] = next((i for i, (c, _) in enumerate(listes[key], 1) if c in gold), None)

    payload = {
        "signature": signature, "profondeur": DENSE_DEPTH,
        "taille_liste": {"min": min(profondeurs), "median": statistics.median(profondeurs),
                         "max": max(profondeurs)},
        "latence_ms": {"p50": round(statistics.median(latences)),
                       "p95": round(sorted(latences)[int(len(latences) * 0.95)])},
        "controles": {"C3": ecarts or "préfixe exact sur les 155"},
        "rangs_des_recuperees": rangs,
        "lines": lines,
        "atomes": {f"dense@{DENSE_DEPTH}": {k: sorted({c for c, _ in listes[k]} - frame[k]["pool_chunk_set"])
                                            for k in frame}},
    }
    write_cache("dense", signature, payload)
    print_lines(lines)


# ----------------------------------------- étape 3 bis : granularité du chunk (post-hoc)

def step_granularite(index: ChunkIndex, signature: str) -> None:
    """Mesure **post-hoc**, décidée après avoir vu le tableau des plafonds.

    Elle ne rejoue aucun seuil et ne peut pas rouvrir la règle d'arrêt : celle-ci est déjà
    tranchée, et par plusieurs générateurs. Ce qu'elle prépare est la Phase A.

    Le fait qui l'appelle : pour 18 des 27 ratés, le document d'or est **déjà** dans le pool.
    Le générateur (d) les récupère tous les 18, mais en ramenant *tous* les chunks des
    documents de tête — 1 194 candidats par question, pour des documents de 70 chunks en
    moyenne et jusqu'à 825. Si le chunk d'or est en réalité **voisin** d'un chunk déjà
    retenu, une fenêtre de quelques positions suffit, et le budget s'effondre.

    L'ordre de lecture est celui de ``ChunkIndex.chunks`` : les dicts Python conservent
    l'ordre d'insertion, et l'insertion suit ``rows.jsonl``. Vérifié plutôt que supposé —
    ``page_start`` est croissante dans cet ordre pour 317 des 319 documents.
    """
    frame = build_frame(index, signature)
    misses = misses_of(frame)

    ordre: dict[str, list[str]] = defaultdict(list)
    for chunk_id, chunk in index.chunks.items():
        ordre[chunk["document_id"]].append(chunk_id)
    position = {c: i for chunks in ordre.values() for i, c in enumerate(chunks)}
    parent = {c: index.chunks[c].get("parent_id") for c in index.chunks}

    # --- à quelle distance du pool le chunk d'or se trouve-t-il, quand son document est là ?
    distances = {}
    for key in misses:
        row = frame[key]
        best = None
        for gold in row["gold_chunks"]:
            if gold not in position:
                continue
            document = index.chunks[gold]["document_id"]
            voisins = [position[c] for c in row["pool_chunks"]
                       if c in position and index.chunks[c]["document_id"] == document]
            if voisins:
                ecart = min(abs(position[gold] - v) for v in voisins)
                best = ecart if best is None else min(best, ecart)
        distances[key] = best
    mesurables = sorted(v for v in distances.values() if v is not None)
    print(f"\n  distance positionnelle du chunk d'or au plus proche chunk du pool, "
          f"pour les {len(mesurables)} ratés dont le document est présent :")
    print(f"      {mesurables}")
    if mesurables:
        print(f"      médiane {statistics.median(mesurables)} · "
              f"≤1 : {sum(1 for d in mesurables if d <= 1)} · ≤2 : {sum(1 for d in mesurables if d <= 2)} · "
              f"≤3 : {sum(1 for d in mesurables if d <= 3)} · ≤5 : {sum(1 for d in mesurables if d <= 5)} · "
              f"≤10 : {sum(1 for d in mesurables if d <= 10)}")

    lines, atomes = {}, {}
    for fenetre in (1, 2, 3, 5, 10):
        recovered, budget = set(), []
        ajoutes_par_question = {}
        for key, row in frame.items():
            ajoutes = set()
            for chunk_id in row["pool_chunks"]:
                chunk = index.chunks.get(chunk_id)
                if chunk is None:
                    continue
                voisins = ordre[chunk["document_id"]]
                i = position[chunk_id]
                ajoutes.update(voisins[max(0, i - fenetre):i + fenetre + 1])
            ajoutes -= row["pool_chunk_set"]
            ajoutes_par_question[key] = sorted(ajoutes)
            budget.append(len(ajoutes))
            if not row["hit"] and (ajoutes & row["gold_chunks"]):
                recovered.add(key)
        lines[f"fenetre±{fenetre}"] = recovery_line(frame, misses, recovered, budget)
        atomes[f"fenetre±{fenetre}"] = ajoutes_par_question

    # --- variante par parent : les chunks issus du même bloc parent
    par_parent: dict[tuple, list[str]] = defaultdict(list)
    for chunk_id, chunk in index.chunks.items():
        if chunk.get("parent_id"):
            par_parent[(chunk["document_id"], chunk["parent_id"])].append(chunk_id)
    recovered, budget = set(), []
    for key, row in frame.items():
        ajoutes = set()
        for chunk_id in row["pool_chunks"]:
            chunk = index.chunks.get(chunk_id)
            if chunk and chunk.get("parent_id"):
                ajoutes.update(par_parent[(chunk["document_id"], chunk["parent_id"])])
        ajoutes -= row["pool_chunk_set"]
        budget.append(len(ajoutes))
        if not row["hit"] and (ajoutes & row["gold_chunks"]):
            recovered.add(key)
    lines["meme_parent"] = recovery_line(frame, misses, recovered, budget)

    # --- la combinaison qui compte : profondeur dense ET fenêtre, mesurées ensemble
    dense_cache = read_cache("dense", signature)
    if dense_cache:
        dense_recovered = set(dense_cache["lines"][f"dense@{DENSE_DEPTH}"]["recovered_keys"])
        dense_budget = dense_cache["lines"][f"dense@{DENSE_DEPTH}"]["budget_mean"]
        for fenetre in (2, 5):
            base = lines[f"fenetre±{fenetre}"]
            union = set(base["recovered_keys"]) | dense_recovered
            budget = [base["budget_mean"] + dense_budget] * len(frame)
            lines[f"dense@100+fenetre±{fenetre}"] = recovery_line(frame, misses, union, budget)

    payload = {"signature": signature, "post_hoc": True,
               "distances": distances, "lines": lines, "atomes": atomes}
    write_cache("granularite", signature, payload)
    print_lines(lines)


# --------------------------------------- étape 3 ter : la frontière coût/récupération

#: Combinaisons évaluées avec leur **budget exact** — l'union des ensembles de candidats,
#: pas la somme des budgets, qui compte deux fois ce que deux générateurs ramènent tous
#: les deux. Sur ces atomes le recouvrement est loin d'être négligeable.
COMBINAISONS = (
    ("dense@100",),
    ("bm25@20",), ("bm25@50",),
    ("fenetre±1",), ("fenetre±2",), ("fenetre±3",), ("fenetre±5",), ("fenetre±10",),
    ("graphe@1saut/cap10",),
    ("voisinage@10doc",),
    ("dense@100", "bm25@20"),
    ("dense@100", "bm25@50"),
    ("dense@100", "fenetre±2"),
    ("dense@100", "fenetre±5"),
    ("dense@100", "fenetre±10"),
    ("dense@100", "bm25@20", "fenetre±5"),
    ("dense@100", "bm25@50", "fenetre±5"),
    ("dense@100", "bm25@50", "fenetre±10"),
    ("dense@100", "bm25@50", "graphe@1saut/cap10"),
    ("dense@100", "bm25@50", "voisinage@10doc"),
)


def step_frontiere(index: ChunkIndex, signature: str) -> None:
    """La frontière de Pareto : combien de ratés récupérés, pour quel pool réellement servi.

    C'est la seule table qui puisse décider d'une Phase A. Un générateur ne se juge pas au
    nombre qu'il récupère — le graphe sans plafond en récupère 23 sur 27 — mais au couple
    (récupérés, budget), parce que le budget est ce que le reranker devra payer : 3,2 s à
    profondeur 10, 15,0 s à profondeur 50, et la latence croît avec le pool.
    """
    frame = build_frame(index, signature)
    misses = misses_of(frame)
    atomes: dict[str, dict[str, list[str]]] = {}
    for etape in ("offline", "graph", "dense", "granularite"):
        cache = read_cache(etape, signature)
        if cache is None:
            sys.exit(f"étape {etape} non exécutée : relance-la, elle persiste ses candidats")
        atomes.update(cache.get("atomes") or {})
    manquants = {n for combo in COMBINAISONS for n in combo} - set(atomes)
    if manquants:
        sys.exit(f"atomes absents des caches : {sorted(manquants)}")
    print(f"\n  {len(atomes)} atomes chargés : {', '.join(sorted(atomes))}")

    lignes = []
    for combo in COMBINAISONS:
        recovered, budget = set(), []
        for key, row in frame.items():
            ajoutes = set()
            for nom in combo:
                ajoutes.update(atomes[nom].get(key) or ())
            ajoutes -= row["pool_chunk_set"]
            budget.append(len(ajoutes))
            if not row["hit"] and (ajoutes & row["gold_chunks"]):
                recovered.add(key)
        ligne = recovery_line(frame, misses, recovered, budget)
        ligne["combo"] = list(combo)
        ligne["pool_servi_moyen"] = round(statistics.mean(
            [frame[k]["n_pool"] for k in frame]) + ligne["budget_mean"], 1)
        ligne["par_100_candidats"] = round(100 * ligne["recovered"] / ligne["budget_mean"], 2) \
            if ligne["budget_mean"] else 0.0
        lignes.append((" + ".join(combo), ligne))

    # --- frontière de Pareto : rien ne récupère plus pour moins cher
    pareto = []
    for nom, ligne in lignes:
        domine = any(autre["recovered"] >= ligne["recovered"]
                     and autre["budget_mean"] <= ligne["budget_mean"]
                     and (autre["recovered"], -autre["budget_mean"]) != (ligne["recovered"], -ligne["budget_mean"])
                     for _, autre in lignes)
        if not domine:
            pareto.append(nom)

    print(f"\n{'combinaison':<48}{'récup.':>8}{'budget':>9}{'pool servi':>12}"
          f"{'/100 cand.':>11}{'IC95':>18}   Pareto")
    for nom, ligne in sorted(lignes, key=lambda x: x[1]["budget_mean"]):
        ci = f"[{ligne['ci95'][0]:+.3f} ; {ligne['ci95'][1]:+.3f}]"
        marque = "  ◀" if nom in pareto else ""
        print(f"{nom:<48}{ligne['recovered']:>5}/27{ligne['budget_mean']:>9.1f}"
              f"{ligne['pool_servi_moyen']:>12.1f}{ligne['par_100_candidats']:>11.2f}{ci:>18}{marque}")

    payload = {"signature": signature, "post_hoc": True,
               "frontiere_pareto": pareto,
               "lignes": {nom: ligne for nom, ligne in lignes}}
    write_cache("frontiere", signature, payload)
    print(f"\n  frontière de Pareto : {' · '.join(pareto)}")


# ------------------------------------------------------------------ étape 4 : le tableau

def print_lines(lines: dict) -> None:
    print(f"\n{'générateur':<26}{'récupérés':>10}{'ratés':>7}{'rappel pool':>13}"
          f"{'Δ':>8}{'IC95':>18}{'budget/q':>10}{'':>4}")
    for name, line in lines.items():
        marque = "  ✱" if line["significant"] else ("  ·" if line["passes"] else "")
        ci = f"[{line['ci95'][0]:+.3f} ; {line['ci95'][1]:+.3f}]" if line["ci95"] else ""
        print(f"{name:<26}{line['recovered']:>7}/{line['misses_before']:<2}"
              f"{line['misses_after']:>7}{line['pool_recall_before']:>7.3f}→{line['pool_recall_after']:<6.3f}"
              f"{line['delta']:>+8.3f}{ci:>18}{line['budget_mean']:>10.1f}{marque}")


def step_report(index: ChunkIndex, signature: str) -> None:
    offline = read_cache("offline", signature)
    graph = read_cache("graph", signature)
    dense = read_cache("dense", signature)
    if offline is None:
        sys.exit("étape offline non exécutée")
    frame = build_frame(index, signature)
    misses = misses_of(frame)

    lines = {}
    if dense:
        lines.update(dense["lines"])
    lines.update(offline["lines"])
    if graph:
        lines.update(graph["lines"])

    print(f"\n{'=' * 100}")
    print(f"PLAFONDS — pool de référence dense@{REFERENCE_DEPTH}, "
          f"{len(misses)} ratés sur {len(frame)} questions, signature {signature}")
    print(f"seuil pré-enregistré : N = {N_SEUIL} ratés récupérés")
    print("=" * 100)
    print_lines(lines)

    # --- qui récupère quoi : deux générateurs à 6 ne valent pas la même chose
    print("\n=== recouvrement — les mêmes questions, ou des questions différentes ? ===")
    sets = {name: set(line["recovered_keys"]) for name, line in lines.items() if line["recovered"]}
    union = set().union(*sets.values()) if sets else set()
    print(f"  union de tous les générateurs : {len(union)}/{len(misses)} ratés récupérables")
    print(f"  jamais récupérés par personne  : {len(misses) - len(union)}")
    noms = list(sets)
    if noms:
        largeur = max(len(n) for n in noms)
        print(f"\n  {'':<{largeur}} " + " ".join(f"{n[:9]:>9}" for n in noms))
        for a in noms:
            cells = " ".join(f"{len(sets[a] & sets[b]):>9}" for b in noms)
            print(f"  {a:<{largeur}} {cells}")

    passants = [n for n, l in lines.items() if l["passes"]]
    verdict = {
        "seuil": N_SEUIL,
        "generateurs_au_dessus_du_seuil": passants,
        "union_recuperable": sorted(union),
        "irrecuperables": sorted(set(misses) - union),
        "regle_darret": ("Phase A ouverte sur : " + ", ".join(passants)) if passants else
                        "AUCUN générateur au-dessus du seuil — la frontière est la représentation, "
                        "Phase A ne s'ouvre pas, on s'arrête sans rien construire.",
    }
    print(f"\n=== règle d'arrêt pré-engagée ===\n  {verdict['regle_darret']}")

    payload = {
        # ``describe()`` entier, pas seulement son étiquette : un fichier de résultats qui ne
        # dit pas sur quel corpus il a été mesuré finit par être écrasé par autre chose.
        # C'est arrivé le 5 septembre.
        "corpus": {**corpus_overlay.describe(),
                   "chunks": len(index.chunks), "documents": len(index.documents),
                   "collection": quant_rag.COLLECTION},
        "entrees": {
            "questions-v1.jsonl": _sha256(HERE / "questions-v1.jsonl"),
            f"questions-{OPEN_BENCH}.jsonl": _sha256(HERE / f"questions-{OPEN_BENCH}.jsonl"),
            "graph-lite.json": _sha256(GRAPH),
            "index_bm25": offline["bm25_index"],
            "cache_dense": f"router-retrievals-{signature}.json",
        },
        "protocole": {
            "pool_de_reference": f"dense limit={REFERENCE_DEPTH}, filtres de production",
            "seuil_N": N_SEUIL,
            "seuil_derivation": "metrics.paired_delta sur k uns et 155-k zéros : k=3 [0,000;0,045], k=4 [+0,006;+0,052]",
            "bootstrap": {"draws": 4000, "seed": 20260901},
            "external_llm_calls": 0,
            "profondeurs": {"dense": DENSE_DEPTH, "bm25": list(BM25_DEPTHS),
                            "voisinage_documents": list(NEIGHBOUR_DOCS),
                            "graphe_plafonds": [c if c is not None else "inf" for c in ENTITY_CAPS]},
        },
        "controles": {
            "C1": offline["controles"]["C1"],
            "C2": "155/155 listes dense identiques entre bb7bf33c37 et 10390927db",
            "C3": dense["controles"]["C3"] if dense else "non exécuté",
            "C4": graph["controles"]["C4"] if graph else "non exécuté",
        },
        "reference": offline["reference"],
        "anatomie": offline["anatomie"],
        "fan_out": graph["fan_out"] if graph else None,
        "dense_rangs_des_recuperees": dense["rangs_des_recuperees"] if dense else None,
        "dense_latence_ms": dense["latence_ms"] if dense else None,
        "lines": lines,
        "verdict": verdict,
    }
    sortie = HERE / f"results-pool-recall-{signature}.json"
    sortie.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n  écrit  {sortie.name}")


# --------------------------------------------------------------------------- programme

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--step", required=True, choices=("offline", "graph", "dense", "granularite", "frontiere", "report"))
    parser.add_argument("--sabotage", action="store_true",
                        help="offline : retire l'or du pool (C1 doit hurler) ; "
                             "dense : décale le préfixe (C3 doit hurler)")
    args = parser.parse_args()

    signature = corpus_overlay.signature()
    index = ChunkIndex.load(verbose=True)
    print(f"  corpus  {len(index.chunks)} chunks · {len(index.documents)} documents · signature {signature}")

    {"offline": lambda: step_offline(index, signature, args.sabotage),
     "graph": lambda: step_graph(index, signature),
     "dense": lambda: step_dense(index, signature, args.sabotage),
     "granularite": lambda: step_granularite(index, signature),
     "frontiere": lambda: step_frontiere(index, signature),
     "report": lambda: step_report(index, signature)}[args.step]()


if __name__ == "__main__":
    main()
