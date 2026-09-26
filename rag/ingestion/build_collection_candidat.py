"""Collections Qdrant des bras du chantier — une par bras, nommée par sa signature.

Phase 4 du pré-enregistrement. **La collection servie
``quant_rag_ingested_all_qwen3_06b`` n'est jamais touchée** : ce module ne la lit ni ne
l'écrit, il en crée d'autres à côté. Annuler le chantier, c'est supprimer celles-ci.

    quant_rag_candidat_<signature>_<bras>      c1 · c2 · sabote

Ce que le payload porte, et pourquoi exactement celui-là
--------------------------------------------------------
Le même que la production (``apply_delivery.payload_for``) : les champs du chunk, le titre et
les auteurs du document, les métadonnées bibliographiques consolidées, et le **texte servi**
(overlay des tableaux appliqué). C'est ce que ``quant_rag._payload_row`` lit pour composer un
passage, et donc ce dont dépendent l'en-tête montré, les filtres d'année et le rang.

**Les trois bras partagent le même payload, à l'octet.** Seul le vecteur diffère — c'est
l'assertion I4 du contrôle d'identité, et ce module la rend vraie par construction en
écrivant le même payload dans les trois collections depuis la même source.

    .venv/bin/python rag/ingestion/build_collection_candidat.py --candidat 8d4ee77f1f --bras c1
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "rag"))

import quant_rag  # noqa: E402

CACHE = HERE / ".cache"
PROCESSED = ROOT / "data" / "processed"
METADATA = ROOT / "rag" / "metadata" / "documents-metadata-v1.json"
CHAMPS_DOCUMENT = ("title", "authors", "publication_year", "first_year", "short_ref",
                   "venue", "doc_type", "filename")
CHAMPS_CHUNK = ("chunk_id", "document_id", "page_start", "page_end", "part", "chapter",
                "section", "title_path", "parent_id", "content_type", "rag_eligible", "image_refs")


def nom_collection(signature: str, bras: str) -> str:
    return f"quant_rag_candidat_{signature}_{bras}"


def _metadonnees() -> dict[str, dict]:
    return {d["document_id"]: d
            for d in json.loads(METADATA.read_text(encoding="utf-8"))["documents"]}


def payloads(signature: str) -> list[dict]:
    """Le payload de chaque chunk servi du candidat — identique pour les trois bras."""
    racine = PROCESSED / f"candidat-{signature}"
    servis = {r["chunk_id"]: r["text"] for r in
              (json.loads(l) for l in (racine / "chunks-servis.jsonl").open(encoding="utf-8") if l.strip())}
    overlay = json.loads((racine / "tables-markdown-v1.json").read_text(encoding="utf-8"))["chunks"]
    meta = _metadonnees()
    out = []
    for row in (json.loads(l) for l in (racine / "rows.jsonl").open(encoding="utf-8") if l.strip()):
        chunk, document = row["chunk"], row["document"]
        value = {k: chunk.get(k) for k in CHAMPS_CHUNK}
        value.update(title=document.get("title"), authors=document.get("authors", []),
                     text=servis[chunk["chunk_id"]], corpus_version="ingested-all")
        if chunk["chunk_id"] in overlay:
            value["text_html"] = chunk.get("text")
        record = meta.get(chunk["document_id"]) or {}
        value.update({k: record.get(k) for k in CHAMPS_DOCUMENT if k in record})
        out.append(value)
    return out


def construire(signature: str, bras: str) -> dict:
    from qdrant_client import models

    vecteurs = CACHE / f"vectors-candidat-{signature}-{bras}.npz"
    if not vecteurs.exists():
        sys.exit(f"vecteurs absents : {vecteurs} — lance embed_candidat.py --bras {bras}")
    blob = np.load(vecteurs)
    par_id = dict(zip(blob["chunk_ids"].tolist(), blob["vectors"]))

    charges = payloads(signature)
    manquants = [p["chunk_id"] for p in charges if p["chunk_id"] not in par_id]
    if manquants:
        sys.exit(f"{len(manquants)} chunks sans vecteur — la collection serait incomplète en silence")

    nom = nom_collection(signature, bras)
    client = quant_rag.client()
    if nom == quant_rag.COLLECTION:
        sys.exit("refus : ce nom est celui de la collection SERVIE")
    if client.collection_exists(nom):
        client.delete_collection(nom)
    dimension = int(next(iter(par_id.values())).shape[0])
    client.create_collection(
        nom,
        vectors_config=models.VectorParams(size=dimension, distance=models.Distance.COSINE),
        # Même exactitude que la collection servie (build_index.py), et pour une raison plus
        # pressante ici : ces collections portent les MESURES qui décident d'un chantier. Sur
        # un serveur, une collection créée sans ces réglages rendrait un rappel@50 de 0,5868 —
        # 41 % du pool perdu — et un verdict de chantier calculé dessus serait faux sans que
        # rien ne le signale. Inerte en mode embarqué (le mode local n'a aucun index
        # approximatif) : l'ajouter ne coûte rien et ferme le cas serveur d'avance.
        hnsw_config=models.HnswConfigDiff(m=0),
        optimizers_config=models.OptimizersConfigDiff(indexing_threshold=0),
    )

    debut = time.perf_counter()
    for depart in range(0, len(charges), 512):
        lot = charges[depart:depart + 512]
        client.upsert(nom, wait=True, points=[
            models.PointStruct(id=depart + i, vector=par_id[p["chunk_id"]].tolist(), payload=p)
            for i, p in enumerate(lot)])
        print(f"    {min(depart + 512, len(charges))}/{len(charges)}", end="\r", flush=True)
    print()
    compte = client.count(nom).count
    rapport = {"collection": nom, "bras": bras, "signature": signature, "points": compte,
               "dimension": dimension, "secondes": round(time.perf_counter() - debut, 1),
               "collection_servie_touchee": False}
    print(json.dumps(rapport, ensure_ascii=False))
    if compte != len(charges):
        sys.exit(f"ÉCHEC : {compte} points écrits pour {len(charges)} attendus")
    return rapport


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--candidat", required=True, metavar="SIG")
    p.add_argument("--bras", required=True, choices=("c1", "c2", "sabote"))
    a = p.parse_args()
    construire(a.candidat, a.bras)


if __name__ == "__main__":
    main()
