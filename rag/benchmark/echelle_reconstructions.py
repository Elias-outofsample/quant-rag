"""Ce que les deux reconstructions coûteront à 1 000 puis 3 000 documents — mesuré, pas extrapolé.

Pourquoi cette mesure existe
-----------------------------
Le chantier `reprise-ingestion` a mesuré, sur le corpus du 8 septembre 2026 (26 120 points,
418 documents), que différer les reconstructions en fin de lot économise ≈ 16,7 s et
≈ 102 Mo écrits **par document** — soit ≈ 4 % d'un lot, quand le document de mission en
annonçait la moitié. À cette taille, le régime `fin-de-lot` ne se justifie donc pas par la
vitesse.

Il reste une justification, et une seule : **le coût des reconstructions croît avec le
corpus, celui de l'extraction non.** `rebuild_bm25` relit *toute* la collection et réécrit
*tout* l'index à chaque document ; GLiNER ne traite que les chunks neufs. Il existe donc une
taille de corpus au-delà de laquelle la reconstruction par document redevient le poste
dominant. Ce module la cherche — au lieu de l'affirmer.

Ce qui est mesuré, et pourquoi ce sont les bonnes grandeurs
------------------------------------------------------------
Pas la fonction `rebuild_bm25` entière : elle scrolle une collection Qdrant, donc mesurer à
100 000 points demanderait d'en construire une, et l'on mesurerait alors Qdrant autant que
l'index. On mesure les **deux étapes qui portent le coût**, séparément :

``BM25Index.from_corpus`` + ``dump``   la construction de l'index et son écriture disque
``build_graph.build``                  la construction des nœuds et des arêtes

et pour BM25 on mesure **aussi** un scroll réel à 26 120 points, pour connaître la part du
transport dans les 7,7 s mesurées de bout en bout.

La matière est le corpus réel, **répliqué** : les enregistrements sont dupliqués avec des
identifiants distincts jusqu'à atteindre le palier. Un corpus répliqué a le vocabulaire d'un
vrai corpus et une distribution de fréquences réaliste — ce qu'un texte aléatoire n'aurait
pas, et c'est cette distribution qui décide du coût d'un index lexical.

**Aucun appel d'API, aucune écriture dans le corpus servi.** Tout est écrit dans le dossier
de sortie, jetable.

    .venv/bin/python rag/benchmark/echelle_reconstructions.py --sortie /tmp/echelle
    .venv/bin/python rag/benchmark/echelle_reconstructions.py --paliers 26120 60000 100000 200000
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "rag"))
sys.path.insert(0, str(ROOT / "src"))

RAPPORT = pathlib.Path(__file__).resolve().parent / "echelle-reconstructions.json"
#: Les paliers de la trajectoire déclarée : aujourd'hui, ~1 000 documents, ~3 000 documents.
#: 26 120 points pour 418 documents donne ≈ 62,5 points par document, d'où 62 500 et 187 500.
PALIERS = (26120, 62500, 125000, 187500)


def lire_textes_servis() -> list[dict]:
    """Le TEXTE des chunks servis — et pas les enregistrements de l'index.

    Première version fausse, corrigée le 8 septembre 2026 : elle lisait
    ``data/lexical/bm25-….json``, dont les enregistrements portent ``term_counts`` et **pas**
    ``text``. La matière répliquée était donc vide, ``BM25Index.from_corpus`` ne faisait
    presque rien, et la mesure rendait 1,1 s pour 125 000 enregistrements là où le vrai
    ``rebuild_bm25`` en met 7,7 pour 26 120. Une mesure d'échelle qui mesure du vide est
    pire qu'une absence de mesure : elle rassure.

    On lit donc les deux sources dont le corpus est fait, exactement comme
    ``extract_entities.load_rows()`` : l'export figé de l'amont pour les documents
    d'origine, et le relais du registre pour ceux entrés depuis. Les overlays sont
    appliqués — un chunk-tableau doit être compté dans sa forme Markdown, celle qui est
    réellement indexée.
    """
    import corpus_overlay

    export = ROOT / "data" / "qdrant-export" / "payloads.jsonl"
    relais = ROOT / "rag" / "ingestion" / "imported-rows.jsonl"
    if not export.exists():
        sys.exit(f"absent : {export} — lance git lfs pull")
    retires = corpus_overlay.removed_documents()
    lignes, vus = [], set()
    for ligne in export.open(encoding="utf-8"):
        if not ligne.strip():
            continue
        charge = json.loads(ligne)["payload"]
        if charge["document_id"] in retires:
            continue
        texte = corpus_overlay.apply(charge["document_id"], charge["chunk_id"],
                                     charge.get("text") or "")
        if texte is None:
            continue
        vus.add(charge["chunk_id"])
        lignes.append({"chunk_id": charge["chunk_id"], "document_id": charge["document_id"],
                       "title": charge.get("title"), "title_path": charge.get("title_path"),
                       "content_type": charge.get("content_type"), "text": texte})
    if relais.exists():
        for ligne in relais.open(encoding="utf-8"):
            if not ligne.strip():
                continue
            chunk = json.loads(ligne)["chunk"]
            if chunk["chunk_id"] in vus or chunk["document_id"] in retires:
                continue
            texte = corpus_overlay.apply(chunk["document_id"], chunk["chunk_id"],
                                         chunk.get("text") or "")
            if texte is None:
                continue
            lignes.append({"chunk_id": chunk["chunk_id"], "document_id": chunk["document_id"],
                           "title": chunk.get("title"), "title_path": chunk.get("title_path"),
                           "content_type": chunk.get("content_type"), "text": texte})
    return lignes


def repliquer(source: list[dict], cible: int) -> list:
    """Porter la matière à ``cible`` enregistrements, identifiants distincts.

    On duplique plutôt qu'on ne tire au sort : le coût d'un index lexical dépend de la
    distribution des fréquences de termes, qu'un texte aléatoire n'a pas. Répliquer la
    conserve exactement, et c'est le pire cas honnête — un vrai corpus plus grand aurait un
    vocabulaire plus large, donc un index plus cher, pas moins.
    """
    lignes = []
    copie = 0
    while len(lignes) < cible:
        for enregistrement in source:
            if len(lignes) >= cible:
                break
            document_id = enregistrement.get("document_id") or "doc-inconnu"
            chunk_id = enregistrement.get("chunk_id") or f"chunk-{len(lignes)}"
            lignes.append((
                {"document_id": f"{document_id}#{copie}" if copie else document_id,
                 "title": enregistrement.get("title")},
                {"chunk_id": f"{chunk_id}#{copie}" if copie else chunk_id,
                 "document_id": f"{document_id}#{copie}" if copie else document_id,
                 "text": enregistrement.get("text", ""),
                 "title_path": enregistrement.get("title_path"),
                 "content_type": enregistrement.get("content_type")}))
        copie += 1
    return lignes


def mesurer_bm25(lignes: list, sortie: pathlib.Path) -> dict:
    from retrieval.lexical import BM25Index

    depart = time.perf_counter()
    index = BM25Index.from_corpus(lignes)
    construction = time.perf_counter() - depart
    depart = time.perf_counter()
    index.dump(sortie)
    ecriture = time.perf_counter() - depart
    return {"records": index.doc_count, "termes": len(index.document_frequency),
            "construction_s": round(construction, 2), "ecriture_s": round(ecriture, 2),
            "total_s": round(construction + ecriture, 2),
            "octets": sortie.stat().st_size,
            "mo": round(sortie.stat().st_size / 1e6, 1)}


def mesurer_graphe(lignes_extraction: list[dict], facteur: int) -> dict:
    """``build_graph.build`` sur des extractions répliquées.

    Les identifiants de chunk sont suffixés : sans cela, la réplication n'ajouterait aucun
    nœud (``nodes.setdefault``) et la mesure serait plate — elle mesurerait la lecture, pas
    la construction.
    """
    sys.path.insert(0, str(ROOT / "rag" / "graph"))
    import build_graph

    rows = []
    for copie in range(facteur):
        for ligne in lignes_extraction:
            if copie:
                rows.append({**ligne,
                             "chunk_id": f"{ligne['chunk_id']}#{copie}",
                             "document_id": f"{ligne['document_id']}#{copie}"})
            else:
                rows.append(ligne)
    depart = time.perf_counter()
    noeuds, aretes = build_graph.build(rows)
    return {"chunks": len(rows), "noeuds": len(noeuds), "aretes": len(aretes),
            "construction_s": round(time.perf_counter() - depart, 2)}


def main() -> None:
    parseur = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parseur.add_argument("--paliers", type=int, nargs="+", default=list(PALIERS))
    parseur.add_argument("--sortie", type=pathlib.Path, required=True,
                         help="dossier jetable où écrire les index d'essai")
    parseur.add_argument("--rapport", type=pathlib.Path, default=RAPPORT)
    parseur.add_argument("--sans-graphe", action="store_true")
    args = parseur.parse_args()
    args.sortie.mkdir(parents=True, exist_ok=True)

    source = lire_textes_servis()
    octets = sum(len(l["text"]) for l in source)
    print(f"matière : {len(source)} chunks servis, {octets / 1e6:.1f} Mo de texte")
    extraction = []
    if not args.sans_graphe:
        chemin = ROOT / "data" / "graph" / "gliner-results" / "results.jsonl"
        extraction = [json.loads(l) for l in chemin.read_text(encoding="utf-8").splitlines()
                      if l.strip()]
        print(f"          {len(extraction)} extractions GLiNER")

    rapport = {"source_records": len(source), "source_extractions": len(extraction),
               "paliers": []}
    for palier in sorted(args.paliers):
        print(f"\n--- {palier} enregistrements ---")
        entree = {"palier": palier}
        entree["bm25"] = mesurer_bm25(repliquer(source, palier),
                                      args.sortie / f"bm25-{palier}.json")
        print(f"  BM25   {entree['bm25']['total_s']} s "
              f"({entree['bm25']['construction_s']} construction + "
              f"{entree['bm25']['ecriture_s']} écriture), {entree['bm25']['mo']} Mo")
        if extraction:
            facteur = max(1, round(palier / max(len(extraction), 1)))
            entree["graphe"] = mesurer_graphe(extraction, facteur)
            print(f"  graphe {entree['graphe']['construction_s']} s pour "
                  f"{entree['graphe']['chunks']} chunks, "
                  f"{entree['graphe']['noeuds']} nœuds")
        rapport["paliers"].append(entree)

    base = rapport["paliers"][0]
    for entree in rapport["paliers"]:
        entree["bm25_x_base"] = round(entree["bm25"]["total_s"] / base["bm25"]["total_s"], 2)
        entree["points_x_base"] = round(entree["palier"] / base["palier"], 2)
        if "graphe" in entree and "graphe" in base:
            entree["graphe_x_base"] = round(
                entree["graphe"]["construction_s"] / base["graphe"]["construction_s"], 2)
    args.rapport.write_text(json.dumps(rapport, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nrapport : {args.rapport}")
    print(json.dumps([{k: e.get(k) for k in ("palier", "bm25_x_base", "points_x_base",
                                             "graphe_x_base")} for e in rapport["paliers"]],
                     ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
