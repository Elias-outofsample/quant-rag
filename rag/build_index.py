"""Reconstruit le Qdrant embarqué local depuis **deux** sources (aucun Docker requis).

Idempotent : relancer recrée la collection. Les métadonnées bibliographiques consolidées
(``rag/metadata/documents-metadata-v1.json``) sont injectées dans chaque payload au
chargement — un index rebâti ne les perd donc pas.

    1. ``data/qdrant-export/``  l'instantané daté des 258 documents livrés par l'amont ;
    2. le **registre d'imports**  les documents entrés depuis, dont les payloads viennent de
       ``data/processed/ingested/<dossier>/chunks.jsonl`` et les vecteurs de
       ``rag/ingestion/.cache/vectors-<livraison>.npz``.

Pourquoi deux sources plutôt qu'un export étendu : les vecteurs de l'export **ne servent
déjà plus à aucun chunk servi**. L'overlay des titres propres couvre les 18 636 chunks
conservés et prime ; ``vectors.npy`` (76 Mo) contribue 0 vecteur à la collection. Cette
reconstruction est donc, depuis le 3 septembre 2026, « payloads d'une source + vecteurs d'un
overlay » — les documents importés prolongent ce schéma au lieu d'en inventer un autre.
Écrire dans l'export ferait cohabiter deux recettes d'embedding dans un fichier dont
``reembed_titles.py`` et ``dense_matrix.py`` supposent qu'il porte celle de l'amont, et
créerait un objet LFS de 137 Mo par livraison.

Les identifiants de points des documents importés sont **repris du registre** : une
reconstruction les rend identiques, jamais réattribués.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from qdrant_client import QdrantClient, models

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ancrage  # noqa: E402  — offsets du passage dans le texte canonique de son document
import corpus_overlay  # noqa: E402
import qdrant_backend  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
VECTOR_OVERLAY = ROOT / "rag" / "tables" / ".cache" / "vectors-tables-markdown-v1.npz"
#: Vecteurs ré-embarqués avec le titre consolidé (rag/titles/, 3 septembre 2026) : tous les
#: chunks, calculés sur le texte des overlays ci-dessus — appliqués en dernier, ils priment.
TITLE_VECTORS = corpus_overlay.TITLE_VECTORS
EXPORT = ROOT / "data" / "qdrant-export"
STORAGE = ROOT / "qdrant_storage_local"
COLLECTION = "quant_rag_ingested_all_qwen3_06b"
METADATA = ROOT / "rag" / "metadata" / "documents-metadata-v1.json"
PAYLOAD_FIELDS = ("title", "authors", "publication_year", "first_year", "short_ref", "venue", "doc_type", "filename")
INGESTED = ROOT / "data" / "processed" / "ingested"
DELIVERY_VECTORS = ROOT / "rag" / "ingestion" / ".cache"
#: Livraison synthétique des documents antérieurs au registre : ils viennent de l'export.
BASELINE_DELIVERY = "corpus-initial"

#: Les trois clés d'ancrage ajoutées au payload par le contrat de sortie. Elles ne changent
#: **ni la signature du corpus ni les vecteurs** : ``corpus_overlay.signature()`` ne hache que
#: ``duplicates-v1.json``, ``tables-markdown-v1.json``, le registre et les titres, et les
#: vecteurs upsertés viennent de ``vectors.npy`` et des ``.npz``, jamais du payload.
CLES_ANCRAGE = ("doc_text_sha256", "ancrage_granularite", "ancrage_intervalles")


def ancrage_par_chunk() -> dict[str, dict]:
    """L'ancrage, construit à la demande s'il manque — il coûte 1,8 s et 0 appel.

    Il n'est pas versionné : 10,7 Mo de coordonnées régénérables à partir de fichiers qui, eux,
    le sont (``blocks.jsonl``, ``chunks.jsonl``). Le ``sha256`` du texte canonique voyage avec
    chaque offset, si bien qu'un ancrage reconstruit sur un texte différent **se voit** au lieu
    de mentir en silence.
    """
    index = ancrage.index_par_chunk()
    if index:
        return index
    print("  ancrage absent — construction (1,8 s, aucun appel)…", flush=True)
    resultat = ancrage.construire(verbeux=False)
    ancrage.ANCRAGE.write_text(json.dumps(resultat, ensure_ascii=False, separators=(",", ":")),
                               encoding="utf-8")
    ancrage.invalider()
    return ancrage.index_par_chunk(resultat)


def imported_points(metadata: dict, ancres: dict[str, dict] | None = None) -> tuple[list, list[str]]:
    """Points des documents entrés par une livraison : payloads canoniques, vecteurs du .npz.

    Retourne aussi les avertissements — un document au registre dont les vecteurs manquent
    n'est pas indexé, et il faut le dire fort : la collection serait incomplète en silence.
    """
    if not corpus_overlay.REGISTRY.exists():
        return [], []
    registry = json.loads(corpus_overlay.REGISTRY.read_text(encoding="utf-8"))
    imported = [d for d in registry.get("documents", [])
                if d.get("delivery", {}).get("id") not in (None, BASELINE_DELIVERY)
                and d.get("status") == "active"]
    if not imported:
        return [], []
    overrides = corpus_overlay.text_overrides()
    vectors_by_delivery, warnings, points = {}, [], []
    for entry in imported:
        delivery = entry["delivery"]["id"]
        if delivery not in vectors_by_delivery:
            path = DELIVERY_VECTORS / f"vectors-{delivery}.npz"
            if not path.exists():
                warnings.append(f"vecteurs absents pour la livraison {delivery} ({path.name}) — "
                                "ses documents ne sont PAS indexés ; relance l'embedding")
                vectors_by_delivery[delivery] = {}
            else:
                blob = np.load(path)
                vectors_by_delivery[delivery] = dict(zip(blob["chunk_ids"].tolist(), blob["vectors"]))
        vectors = vectors_by_delivery[delivery]
        folder = INGESTED / entry["folder"]
        chunks_path = folder / "chunks.jsonl"
        if not chunks_path.exists():
            warnings.append(f"{entry['document_id']} au registre mais absent du disque ({entry['folder']})")
            continue
        document = json.loads((folder / "document.json").read_text(encoding="utf-8"))
        eligible = [json.loads(line) for line in chunks_path.open(encoding="utf-8")
                    if line.strip() and json.loads(line).get("rag_eligible") is True]
        span = entry.get("point_ids") or {}
        if span.get("count") not in (None, len(eligible)):
            warnings.append(f"{entry['document_id']} : {len(eligible)} chunks éligibles mais "
                            f"{span['count']} points au registre — plage non réattribuée, à vérifier")
        first = span.get("first")
        if first is None:
            warnings.append(f"{entry['document_id']} : aucune plage d'identifiants au registre")
            continue
        record = metadata.get(entry["document_id"], {})
        for offset, chunk in enumerate(eligible):
            if chunk["chunk_id"] not in vectors:
                warnings.append(f"{chunk['chunk_id']} sans vecteur — non indexé")
                continue
            payload = {key: chunk.get(key) for key in
                       ("chunk_id", "document_id", "page_start", "page_end", "part", "chapter",
                        "section", "title_path", "parent_id", "content_type", "rag_eligible", "image_refs")}
            payload.update(title=document.get("title"), authors=document.get("authors", []),
                           text=chunk.get("text"), corpus_version="ingested-all")
            if chunk["chunk_id"] in overrides:
                payload["text_html"] = payload["text"]
                payload["text"] = overrides[chunk["chunk_id"]]
            payload.update(record)
            payload.update((ancres or {}).get(chunk["chunk_id"], {}))
            points.append(models.PointStruct(
                id=int(first + offset),
                vector=np.asarray(vectors[chunk["chunk_id"]], dtype=np.float32).tolist(),
                payload=payload))
    return points, warnings


def main(allow_missing: bool = False) -> None:
    missing = [p.name for p in (EXPORT / "vectors.npy", EXPORT / "ids.npy", EXPORT / "payloads.jsonl") if not p.exists()]
    if missing:
        sys.exit(f"Fichiers absents dans {EXPORT}: {missing}\nLance:  git lfs pull")

    vectors = np.load(EXPORT / "vectors.npy", mmap_mode="r")
    ids = np.load(EXPORT / "ids.npy")
    payloads = [json.loads(line)["payload"] for line in (EXPORT / "payloads.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    if not len(ids) == len(vectors) == len(payloads):
        sys.exit(f"Longueurs incohérentes: ids={len(ids)} vectors={len(vectors)} payloads={len(payloads)}")
    metadata = {}
    if METADATA.exists():
        records = json.loads(METADATA.read_text(encoding="utf-8"))["documents"]
        metadata = {r["document_id"]: {k: r.get(k) for k in PAYLOAD_FIELDS} for r in records}
        for payload in payloads:
            payload.update(metadata.get(payload.get("document_id"), {}))
    # L'ancrage, aux DEUX sites de payload — export ici, registre dans ``imported_points``.
    # Le schéma des deux sources doit rester identique : une divergence serait invisible et
    # rendrait ``verify_citation`` muet sur la moitié du corpus.
    ancres = ancrage_par_chunk()
    ancres_posees = 0
    for payload in payloads:
        ancre = ancres.get(payload.get("chunk_id"))
        if ancre:
            payload.update(ancre)
            ancres_posees += 1
    # overlays : documents retirés, tableaux convertis (texte) et leurs vecteurs ré-embarqués
    keep = [i for i, p in enumerate(payloads) if p.get("document_id") not in corpus_overlay.removed_documents()]
    overrides = corpus_overlay.text_overrides()
    replaced = {}
    if overrides:
        if VECTOR_OVERLAY.exists():
            blob = np.load(VECTOR_OVERLAY)
            replaced = dict(zip(blob["chunk_ids"].tolist(), blob["vectors"]))
        else:
            print(f"ATTENTION : {len(overrides)} textes convertis mais pas de vecteurs ré-embarqués "
                  f"({VECTOR_OVERLAY.name} absent) — lance rag/tables/convert_tables.py après cet index.")
        for i in keep:
            cid = payloads[i].get("chunk_id")
            if cid in overrides:
                payloads[i]["text_html"] = payloads[i]["text"]
                payloads[i]["text"] = overrides[cid]
    titles_applied = 0
    if corpus_overlay.TITLES.exists():
        if TITLE_VECTORS.exists():
            blob = np.load(TITLE_VECTORS)
            for cid, vec in zip(blob["chunk_ids"].tolist(), blob["vectors"]):
                replaced[cid] = vec
                titles_applied += 1
        else:
            print(f"ATTENTION : overlay des titres propres présent ({corpus_overlay.TITLES.name}) mais pas ses vecteurs "
                  f"({TITLE_VECTORS.name} absent) — lance rag/titles/reembed_titles.py, puis relance cet index.")

    # Les points importés sont composés **avant** de toucher la collection, et un vecteur
    # manquant arrête la reconstruction. Sans cette garde, la séquence était : supprimer la
    # collection, la rebâtir plus petite, imprimer un avertissement dans le défilement, puis
    # « status: COMPLETED ». Constaté le 7 septembre 2026 — deux livraisons du 6 septembre
    # (`2026-09-06-nuit-d01`, `-d02`) ont perdu leurs `.npz` : une reconstruction aurait rendu
    # 26 027 points au lieu de 26 120, soit deux documents entiers effacés en silence, et
    # leurs vecteurs n'existent plus nulle part ailleurs que dans la collection qu'on vient
    # de supprimer. Une perte irréversible ne doit pas tenir dans un avertissement.
    imported, warnings = imported_points(metadata, ancres)
    for warning in warnings:
        print(f"ATTENTION : {warning}")
    if warnings and not allow_missing:
        sys.exit(f"\n{len(warnings)} avertissement(s) : la collection reconstruite serait INCOMPLÈTE.\n"
                 "  · récupère les vecteurs perdus depuis la collection encore en place :\n"
                 "      .venv/bin/python rag/ingestion/recover_vectors.py --write\n"
                 "  · ou relance l'embedding de la livraison concernée ;\n"
                 "  · ou assume la perte explicitement : --allow-missing.")

    # ``garde=False`` et c'est le seul site du dépôt qui s'en dispense : la garde de dérive
    # refuse une collection dont le compte ne correspond pas au corpus courant, or c'est
    # exactement l'état que cette fonction est faite pour réparer — elle détruit la
    # collection à la ligne suivante et la reconstruit. L'exiger juste ici rendrait la
    # reconstruction impossible dès qu'elle serait nécessaire.
    client = qdrant_backend.ouvrir(STORAGE, garde=False)
    if client.collection_exists(COLLECTION):
        client.delete_collection(COLLECTION)
    client.create_collection(
        COLLECTION,
        vectors_config=models.VectorParams(size=int(vectors.shape[1]), distance=models.Distance.COSINE),
        # INERTE en mode embarqué — le mode local n'a AUCUN index approximatif : il fait une
        # recherche exacte par force brute et ignore ces réglages (``qdrant_local.QdrantLocal``, qui ignore ``search_params``).
        # PORTEUR dès qu'un serveur est en face : sans eux, un serveur construit un HNSW et
        # rend un rappel@50 de 0,5868 — 41 % du pool perdu — et pas même de façon
        # reproductible, deux constructions identiques ayant donné 26 puis 44 approximations
        # (RAPPORT-ETAPE-1-EXACTITUDE-2026-09-07.md). m=0 seul ne suffit pas : il laisse
        # indexed_vectors_count à 18 000, donc aucun témoin observable. Les deux ensemble
        # rendent l'exactitude VÉRIFIABLE par un seul appel : indexed_vectors_count == 0.
        hnsw_config=models.HnswConfigDiff(m=0),
        optimizers_config=models.OptimizersConfigDiff(indexing_threshold=0),
    )

    started = time.perf_counter()
    #: MESURÉ le 7 septembre 2026 : à 2 000 points de 1 024 dimensions, le corps JSON d'un
    #: upsert atteint 44 601 151 octets — au-dessus de la limite de requête par défaut d'un
    #: serveur Qdrant (33 554 432). Contre un stockage embarqué il n'y a aucune limite, donc
    #: le défaut est resté invisible ; contre un serveur, la reconstruction échoue en 400.
    #: 256 est la taille qu'``apply_delivery`` utilise déjà (``apply_delivery._apply``, boucle d'upsert) et
    #: qui passe partout — la chaîne d'ingestion a maintenant une seule taille de lot.
    #: AUCUN EFFET SÉMANTIQUE : mêmes points, mêmes identifiants, même ordre, plus de requêtes.
    batch = 256
    for start in range(0, len(keep), batch):
        block = keep[start:start + batch]
        client.upsert(COLLECTION, wait=True, points=[
            models.PointStruct(id=int(ids[i]),
                               vector=np.asarray(replaced.get(payloads[i].get("chunk_id"), vectors[i]), dtype=np.float32).tolist(),
                               payload=payloads[i])
            for i in block
        ])
        print(f"  {start + len(block)}/{len(keep)}", end="\r", flush=True)

    for start in range(0, len(imported), batch):
        client.upsert(COLLECTION, wait=True, points=imported[start:start + batch])
    if imported:
        print(f"  {len(imported)} points importés (registre) ajoutés")

    if metadata:
        for field, schema in (("publication_year", models.PayloadSchemaType.INTEGER),
                              ("authors", models.PayloadSchemaType.KEYWORD),
                              ("document_id", models.PayloadSchemaType.KEYWORD)):
            client.create_payload_index(COLLECTION, field_name=field, field_schema=schema, wait=True)
    count = client.count(COLLECTION, exact=True).count
    print(json.dumps({
        "status": "COMPLETED", "collection": COLLECTION, "points": count,
        "from_export": len(keep), "from_registry": len(imported),
        "registry_warnings": warnings,
        "corpus_signature": corpus_overlay.signature(),
        "metadata_documents": len(metadata),
        "removed_documents": len(ids) - len(keep) and len(corpus_overlay.removed_documents()),
        "points_skipped": len(ids) - len(keep), "table_texts_converted": len(overrides), "vectors_replaced": len(replaced),
        "clean_title_vectors": titles_applied,
        "ancrage": {"format": ancrage.FORMAT, "export": ancres_posees,
                    "registre": sum(1 for p in imported if p.payload.get("doc_text_sha256")),
                    "connus": len(ancres)},
        "dimension": int(vectors.shape[1]), "storage": str(STORAGE),
        "seconds": round(time.perf_counter() - started, 1),
    }, indent=2))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--allow-missing", action="store_true",
                        help="rebâtir malgré des vecteurs de livraison absents — la collection sera plus petite")
    main(allow_missing=parser.parse_args().allow_missing)
