"""Corriger le titre d'un document **déjà servi**, vecteurs et index compris.

Le titre n'est pas une étiquette posée à côté du texte : il entre dans ce que le corpus
indexe. L'embedding de chaque chunk commence par ``Document: <titre>`` (recette de
``reembed_titles.embedding_text``, appliquée à l'import par ``apply_delivery.embed_delivery``),
et le texte lexical de BM25 porte le titre du payload (``quant_rag.rebuild_bm25``). Corriger
un titre dans ``overrides.json`` sans rien d'autre ne corrige donc **rien** de ce qui est
interrogé : le corpus continue de répondre sur l'ancien.

``reembed_titles.py`` ne fait pas ce travail et ne prétend pas le faire — c'est une
expérience isolée qui écrit un overlay de vecteurs hors collection. Ce script-ci écrit dans
la collection, et seulement pour les documents nommés :

  1. l'override est reporté dans ``documents-metadata-v1.json`` (titre, provenance
     ``manual``, ``short_ref`` recalculé) — les trois mêmes gestes que la boucle
     d'``extract_metadata.py`` qui lit ``overrides.json`` ;
  2. les chunks du document sont ré-embarqués avec la recette du corpus, à l'identique,
     seul le titre ayant changé ;
  3. les points sont réécrits **à leurs identifiants existants** : aucune plage ne bouge,
     le registre reste vrai, `registry --verify` ne voit rien passer ;
  4. le payload porte le nouveau titre, et l'index BM25 est refait puisqu'il l'indexe ;
  5. le déplacement est mesuré : cosinus entre l'ancien vecteur et le nouveau, par chunk.

    .venv/bin/python rag/titles/retitle_documents.py doc-xxx doc-yyy
    .venv/bin/python rag/titles/retitle_documents.py doc-xxx --appliquer
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "rag"))
sys.path.insert(0, str(ROOT / "rag" / "metadata"))
sys.path.insert(0, str(HERE))
import corpus_overlay  # noqa: E402
import reembed_titles as rt  # noqa: E402

METADATA = ROOT / "rag" / "metadata" / "documents-metadata-v1.json"
OVERRIDES = ROOT / "rag" / "metadata" / "overrides.json"
RESULT = HERE / "results-retitle-v1.json"
BLOC = 16


def appliquer_override(fiche: dict, correction: dict) -> list[str]:
    """Les trois gestes de la boucle d'``extract_metadata.py`` : valeur, provenance, short_ref."""
    import extract_metadata as em

    changes = []
    for champ in ("title", "authors", "publication_year", "first_year", "venue", "doc_type"):
        if champ in correction and fiche.get(champ) != correction[champ]:
            changes.append(f"{champ}: {fiche.get(champ)!r} → {correction[champ]!r}")
            fiche[champ] = correction[champ]
            cle = "year" if champ == "publication_year" else champ
            if cle in fiche.get("provenance", {}):
                fiche["provenance"][cle] = "manual"
    fiche["sources"] = fiche.get("sources", {}) | {"manual": correction}
    fiche["short_ref"] = em.short_ref(fiche.get("authors"), fiche.get("publication_year"),
                                      fiche.get("title"))
    return changes


def main() -> int:
    analyseur = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    analyseur.add_argument("documents", nargs="+", metavar="DOCUMENT_ID")
    analyseur.add_argument("--appliquer", action="store_true",
                           help="sans ce drapeau, rien n'est écrit : on ne fait que le constat")
    arguments = analyseur.parse_args()

    donnees = json.loads(METADATA.read_text(encoding="utf-8"))
    par_id = {f["document_id"]: f for f in donnees["documents"]}
    overrides = json.loads(OVERRIDES.read_text(encoding="utf-8"))

    plan = []
    for document_id in arguments.documents:
        fiche = par_id.get(document_id)
        if fiche is None:
            sys.exit(f"REFUS : {document_id} n'a pas de fiche de métadonnées")
        correction = overrides.get(fiche.get("filename"))
        if not correction or "title" not in correction:
            sys.exit(f"REFUS : aucun titre corrigé dans overrides.json pour {fiche.get('filename')!r}. "
                     "Ce script applique une décision déjà écrite, il n'en prend aucune.")
        ancien = rt.clean_title(fiche.get("title") or "", fiche)
        nouveau = rt.clean_title(correction["title"], {**fiche, "title": correction["title"]})
        plan.append({"document_id": document_id, "filename": fiche.get("filename"),
                     "titre_avant": ancien, "titre_apres": nouveau, "fiche": fiche,
                     "correction": correction})

    import quant_rag
    from qdrant_client import models

    client = quant_rag.client()
    rapport = {"version": "retitle-v1", "documents": [],
               "signature_avant": corpus_overlay.signature()}
    modele = None

    for item in plan:
        filtre = models.Filter(must=[models.FieldCondition(
            key="document_id", match=models.MatchValue(value=item["document_id"]))])
        points, offset = [], None
        while True:
            lot, offset = client.scroll(collection_name=quant_rag.COLLECTION, limit=1024,
                                        offset=offset, scroll_filter=filtre,
                                        with_payload=True, with_vectors=True)
            points.extend(lot)
            if offset is None:
                break
        entree = {"document_id": item["document_id"], "filename": item["filename"],
                  "titre_avant": item["titre_avant"], "titre_apres": item["titre_apres"],
                  "points": len(points)}
        if not points:
            entree["note"] = "aucun point dans la collection — document retiré ou jamais servi"
            rapport["documents"].append(entree)
            continue
        if not arguments.appliquer:
            rapport["documents"].append(entree | {"status": "CONSTAT"})
            continue

        changes = appliquer_override(item["fiche"], item["correction"])
        entree["champs_modifies"] = changes

        if modele is None:
            modele = quant_rag.embedder()
        textes = [rt.embedding_text(item["titre_apres"], p.payload.get("title_path"),
                                    p.payload.get("text") or "") for p in points]
        vecteurs = []
        for debut in range(0, len(textes), BLOC):
            vecteurs.append(np.asarray(modele.encode(textes[debut:debut + BLOC],
                                                     normalize_embeddings=True,
                                                     show_progress_bar=False, batch_size=8),
                                       dtype=np.float32))
        vecteurs = np.vstack(vecteurs)
        anciens = np.vstack([np.asarray(p.vector, dtype=np.float32) for p in points])
        deplacement = (vecteurs * anciens).sum(axis=1)
        entree["cosinus_ancien_nouveau"] = {"min": round(float(deplacement.min()), 4),
                                            "moyen": round(float(deplacement.mean()), 4),
                                            "max": round(float(deplacement.max()), 4)}
        client.upsert(quant_rag.COLLECTION, wait=True, points=[
            models.PointStruct(id=p.id, vector=v.tolist(),
                               payload={**p.payload, "title": item["fiche"]["title"],
                                        "short_ref": item["fiche"]["short_ref"]})
            for p, v in zip(points, vecteurs)])
        entree["status"] = "COMPLETED"
        rapport["documents"].append(entree)

    if arguments.appliquer:
        donnees["generated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        METADATA.write_text(json.dumps(donnees, ensure_ascii=False, indent=1), encoding="utf-8")
        corpus_overlay.invalidate()
        index = quant_rag.rebuild_bm25()
        rapport["bm25"] = getattr(index, "doc_count", None)
        rapport["signature_apres"] = corpus_overlay.signature()
        RESULT.write_text(json.dumps(rapport, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(rapport, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
