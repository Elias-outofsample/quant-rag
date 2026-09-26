"""Diagnostic en lecture seule d'une livraison amont — ce qui entrerait, et ce que ça casserait.

Ce module n'écrit **rien** dans le corpus : ni Qdrant, ni BM25, ni le graphe, ni
``data/processed/``. Il lit une livraison, la confronte à l'état local, et rend un
rapport. C'est la première tranche du chantier d'ingestion : décider avant d'écrire.

Une livraison est un répertoire contenant les quatre fichiers canoniques par document,
soit directement, soit sous ``processed/`` :

    livraison/
      manifest.json                  (facultatif ; ``schema_version`` 5.1 attendu)
      processed/
        doc-<sha256 du PDF>[:16]/
          document.json  blocks.jsonl  chunks.jsonl  parents.jsonl

Trois faits mesurés dans ce dépôt commandent toute la logique de ce fichier :

  1. ``document_id = doc-sha1(sha256_pdf | backend | version | 'canonical-v1')[:16]``
     (258/258 vérifiés). L'identifiant **encode la version du parseur** : le même PDF
     relu par MinerU 3.5.0 porterait un autre ``document_id`` et passerait pour un
     document neuf. L'identité durable est donc le ``sha256`` du PDF, jamais l'identifiant.
  2. ``chunk_id`` est haché avant la passe de fusion du chunker : seuls 24,4 % des
     chunk_id du corpus (7 902/32 422) se recalculent depuis l'enregistrement livré.
     L'intégrité au niveau du chunk ne se vérifie pas par recalcul, seulement par
     somme de contrôle de fichier — ce que fait le manifeste amont.
  3. La signature de corpus (``corpus_overlay.signature()``) ne hache que les deux
     fichiers d'overlay : **ajouter des documents ne la change pas**. L'index BM25
     garderait son nom et serait rechargé, muet et incomplet.

    .venv/bin/python rag/ingestion/inspect_delivery.py <livraison>
    .venv/bin/python rag/ingestion/inspect_delivery.py --from-corpus      # auto-test
    .venv/bin/python rag/ingestion/inspect_delivery.py <livraison> --json rapport.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import fields as dataclass_fields
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "rag"))
sys.path.insert(0, str(ROOT / "src"))
import corpus_overlay  # noqa: E402

INGESTED = ROOT / "data" / "processed" / "ingested"
METADATA = ROOT / "rag" / "metadata" / "documents-metadata-v1.json"
QUESTIONS = [ROOT / "rag" / "benchmark" / "questions-v1.jsonl",
             ROOT / "rag" / "benchmark" / "questions-v3.jsonl"]
REQUIRED_FILES = ("document.json", "blocks.jsonl", "chunks.jsonl", "parents.jsonl")
#: Ce que la couche macOS sait relire aujourd'hui. Tout écart est un refus, pas une adaptation.
KNOWN_PARSERS = {("mineru", "3.4.5")}
KNOWN_SCHEMA_VERSIONS = {"5.1"}
#: Champs du payload Qdrant réellement lus par le retrieval (``scripts/append_ingested_incremental.py``).
PAYLOAD_CHUNK_FIELDS = ("chunk_id", "document_id", "page_start", "page_end", "part", "chapter",
                        "section", "title_path", "parent_id", "content_type", "rag_eligible", "image_refs", "text")
EMBEDDING_RATE = 2.6  # chunks/s mesurés sur ce M4 (model.md)
#: Seuils de ``rag/metadata/duplicates-v1.json``, la politique déjà appliquée au corpus.
#: Un même papier livré sous un autre PDF — réédition, miroir SSRN — a un sha256 différent
#: et passerait la garde d'identité sans un mot : c'est exactement le couple
#: deflated-sharpe.pdf / ssrn-2460551.pdf, retiré à la main le 2 septembre.
DUPLICATE_CONTAINMENT = 0.95   # doublon : refus
EDITION_CONTAINMENT = 0.50     # édition : décision explicite exigée

#: L'identité est indépendante de l'admissibilité : un document peut être une révision
#: *et* être refusé. Confondre les deux, c'est perdre l'information la plus utile du rapport.
IDENTITIES = ("sans-effet", "ajout", "révision", "re-parse", "doublon-retiré")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def derive_document_id(sha256: str, backend: str, version: str) -> str:
    """La règle du parseur amont (``src/parsing/mineru_adapter.py``), rejouée ici."""
    raw = "\x1f".join((sha256, backend, version, "canonical-v1"))
    return "doc-" + hashlib.sha1(raw.encode()).hexdigest()[:16]


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


# ----------------------------------------------------------------- état local (lecture seule)

def local_documents() -> dict[str, dict]:
    """sha256 du PDF -> ce que le corpus local sait déjà de ce document."""
    known = {}
    for folder in sorted(INGESTED.iterdir()):
        manifest = folder / "document.json"
        if not manifest.exists():
            continue
        document = json.loads(manifest.read_text(encoding="utf-8"))
        chunks_path = folder / "chunks.jsonl"
        known[document["sha256"]] = {
            "folder": folder.name,
            "document_id": document["document_id"],
            "parser": (document.get("parser_backend", ""), document.get("parser_version", "")),
            "title": document.get("title"),
            "chunks_jsonl_sha256": sha256_file(chunks_path) if chunks_path.exists() else None,
            "chunk_ids": {c["chunk_id"] for c in read_jsonl(chunks_path)} if chunks_path.exists() else set(),
        }
    return known


def corpus_shingles() -> dict[str, set[int]]:
    """Empreintes de contenu des documents **actifs**, par la méthode de ``scan_duplicates``.

    8-grammes de mots échantillonnés 1/4 par hachage, chunks éligibles hors tableaux.
    ``hash()`` étant salé par processus, ces empreintes ne sont comparables qu'à l'intérieur
    d'une même exécution : elles ne sont jamais mises en cache sur disque.
    """
    sys.path.insert(0, str(ROOT / "rag" / "metadata"))
    from scan_duplicates import shingles

    removed = corpus_overlay.removed_documents()
    sets = {}
    for folder in sorted(INGESTED.iterdir()):
        if not (folder / "chunks.jsonl").exists():
            continue
        document = json.loads((folder / "document.json").read_text(encoding="utf-8"))
        if document["document_id"] in removed:
            continue
        fingerprint: set[int] = set()
        for chunk in read_jsonl(folder / "chunks.jsonl"):
            if chunk.get("rag_eligible") is True and chunk.get("content_type") != "table":
                fingerprint |= shingles(chunk.get("text") or "")
        sets[document["document_id"]] = fingerprint
    return sets


def delivered_shingles(folder: Path) -> set[int]:
    from scan_duplicates import shingles

    fingerprint: set[int] = set()
    for chunk in read_jsonl(folder / "chunks.jsonl"):
        if chunk.get("rag_eligible") is True and chunk.get("content_type") != "table":
            fingerprint |= shingles(chunk.get("text") or "")
    return fingerprint


def near_duplicate_of(fingerprint: set[int], corpus: dict[str, set[int]],
                      exclude: str | None) -> list[dict]:
    """Documents actifs dont le contenu recouvre celui-ci. ``containment = |A∩B| / min``."""
    hits = []
    for document_id, other in corpus.items():
        if document_id == exclude or not fingerprint or not other:
            continue
        containment = len(fingerprint & other) / min(len(fingerprint), len(other))
        if containment >= EDITION_CONTAINMENT:
            hits.append({"document_id": document_id, "containment": round(containment, 3),
                         "verdict": "doublon" if containment >= DUPLICATE_CONTAINMENT else "édition"})
    return sorted(hits, key=lambda h: -h["containment"])


def gold_anchors() -> tuple[set[str], set[str]]:
    """Documents et chunks porteurs d'or dans les bancs — ce qu'une révision casserait."""
    documents, chunks = set(), set()
    for path in QUESTIONS:
        if not path.exists():
            continue
        for question in read_jsonl(path):
            documents.update(question.get("gold_documents") or [])
            chunks.update(question.get("gold_chunks") or [])
    return documents, chunks


def index_state() -> dict:
    """Compte et identifiant maximum de la collection. Dégrade proprement si Qdrant est pris."""
    try:
        import quant_rag

        client = quant_rag.client()
        count = client.count(quant_rag.COLLECTION, exact=True).count
        highest, offset = -1, None
        while True:
            points, offset = client.scroll(quant_rag.COLLECTION, limit=8192, offset=offset,
                                           with_payload=False, with_vectors=False)
            highest = max([highest] + [int(point.id) for point in points])
            if offset is None:
                break
        return {"available": True, "points": count, "highest_point_id": highest,
                "next_free_point_id": highest + 1, "gaps": highest + 1 - count}
    except Exception as error:  # Qdrant embarqué n'accepte qu'un processus (serveur MCP en cours)
        return {"available": False, "reason": f"{type(error).__name__}: {error}"[:200]}


# ----------------------------------------------------------------- lecture d'une livraison

def locate_documents(delivery: Path) -> list[Path]:
    base = delivery / "processed" if (delivery / "processed").is_dir() else delivery
    return sorted(p for p in base.iterdir() if p.is_dir() and (p / "document.json").exists())


def read_manifest(delivery: Path) -> dict:
    path = delivery / "manifest.json"
    if not path.exists():
        return {"present": False, "notes": ["aucun manifest.json : les sommes de contrôle amont "
                                            "ne peuvent pas être confrontées, seule la structure est vérifiée"]}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        return {"present": True, "readable": False, "error": str(error)}
    version = str(data.get("schema_version"))
    notes = []
    if version not in KNOWN_SCHEMA_VERSIONS:
        notes.append(f"schema_version {version!r} inconnue (connues : {sorted(KNOWN_SCHEMA_VERSIONS)}) — "
                     "refus : un adaptateur de version doit être écrit et testé avant tout import")
    return {"present": True, "readable": True, "schema_version": version,
            "config_hash": data.get("config_hash"), "producer_commit": data.get("producer_commit"),
            "documents_declared": len(data.get("documents") or []),
            "supported": version in KNOWN_SCHEMA_VERSIONS, "notes": notes,
            "by_document_id": {d["document_id"]: d for d in (data.get("documents") or []) if "document_id" in d}}


def schema_drift(documents: list[dict], chunks_sample: list[dict]) -> dict:
    """Champs inattendus ou absents par rapport aux modèles canoniques de ``src/parsing/models.py``."""
    from parsing.models import CanonicalChunk, CanonicalDocument

    expected_doc = {f.name for f in dataclass_fields(CanonicalDocument)}
    expected_chunk = {f.name for f in dataclass_fields(CanonicalChunk)}
    seen_doc = set().union(*(d.keys() for d in documents)) if documents else set()
    seen_chunk = set().union(*(c.keys() for c in chunks_sample)) if chunks_sample else set()
    return {
        "document_unknown_fields": sorted(seen_doc - expected_doc),
        "document_missing_fields": sorted(expected_doc - seen_doc),
        "chunk_unknown_fields": sorted(seen_chunk - expected_chunk),
        "chunk_missing_fields": sorted(expected_chunk - seen_chunk),
        "chunk_missing_payload_fields": sorted(set(PAYLOAD_CHUNK_FIELDS) - seen_chunk),
    }


#: Artefacts qu'un import périme sans que leur nom le dise. Chacun est annoncé **seulement
#: s'il existe** : une liste écrite en dur avertissait encore, le 4 septembre 2026, sur
#: ``router-retrievals.json`` et ``corpus-blob-v4.txt``, tous deux archivés depuis sous
#: ``-avant-import-2026-09-04`` et remplacés par des noms signés. Un avertissement qui
#: désigne un fichier absent n'est pas inoffensif : il apprend à survoler la liste, et c'est
#: la ligne qu'on survole qui coûte cher.
STALE_BY_HAND = (
    ("rag/benchmark/.cache/router-retrievals.json",
     "nom sans signature : un cache de classements calculé sur un autre corpus serait "
     "rechargé tel quel. Le renommer (patron de rag/titles/apply_titles.py) avant de "
     "rejouer un banc"),
    ("rag/benchmark/.cache/corpus-blob-v4.txt", "nom sans signature"),
    ("data/graph/graph-lite.json",
     "avec gliner-results/results.jsonl : extraction GLiNER à relancer sur les seuls "
     "nouveaux chunks, puis graphe rebâti"),
    ("rag/titles/.cache/vectors-clean-titles-v1.npz",
     "ne couvre pas les nouveaux chunks ; l'embedding incrémental doit l'étendre, jamais "
     "le recalculer"),
)


def to_invalidate_by_hand() -> list[str]:
    present = [f"{path} — {why}" for path, why in STALE_BY_HAND if (ROOT / path).exists()]
    return present or ["(aucun : tout artefact périssable porte la signature dans son nom)"]


def inspect_document(folder: Path, known: dict[str, dict], removed_by_id: dict,
                     manifest_entries: dict, metadata_ids: set[str],
                     corpus_fingerprints: dict[str, set[int]] | None = None) -> dict:
    """Verdict et anomalies pour un document livré. Ne lit que la livraison et l'état local."""
    problems, notes = [], []
    missing = [name for name in REQUIRED_FILES if not (folder / name).exists()]
    if missing:
        return {"folder": folder.name, "verdict": "rejet", "problems": [f"fichiers absents : {missing}"]}
    try:
        document = json.loads((folder / "document.json").read_text(encoding="utf-8"))
        chunks = read_jsonl(folder / "chunks.jsonl")
        parents = read_jsonl(folder / "parents.jsonl")
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        return {"folder": folder.name, "verdict": "rejet", "problems": [f"fichier illisible : {error}"]}

    sha = document.get("sha256")
    document_id = document.get("document_id")
    backend = document.get("parser_backend", "")
    version = document.get("parser_version", "")
    for field in ("document_id", "sha256", "parser_backend", "parser_version"):
        if not document.get(field):
            problems.append(f"champ obligatoire absent ou vide : {field}")

    # identité : le nom de dossier et l'identifiant se dérivent-ils comme dans ce corpus ?
    if sha and folder.name != f"doc-{sha[:16]}":
        notes.append(f"nom de dossier {folder.name!r} != doc-<sha256(pdf)[:16]> ({'doc-'+sha[:16]}) — "
                     "convention amont non respectée ; c'est le sha256 qui fait foi")
    derived = derive_document_id(sha, backend, version) if sha else None
    if derived and derived != document_id:
        problems.append(f"document_id {document_id!r} non dérivable de (sha256, {backend}, {version}) "
                        f"— attendu {derived!r} : la règle d'identité amont a changé")
    if (backend, version) not in KNOWN_PARSERS:
        problems.append(f"parseur ({backend} {version}) inconnu — connus : {sorted(KNOWN_PARSERS)}")

    # structure des chunks
    eligible = [c for c in chunks if c.get("rag_eligible")]
    non_boolean = sum(1 for c in chunks if not isinstance(c.get("rag_eligible"), bool))
    if non_boolean:
        problems.append(f"{non_boolean} chunks dont rag_eligible n'est pas un booléen")
    empty = sum(1 for c in eligible if not (c.get("text") or "").strip())
    if empty:
        problems.append(f"{empty} chunks éligibles au texte vide")
    foreign = sum(1 for c in chunks if c.get("document_id") != document_id)
    if foreign:
        problems.append(f"{foreign} chunks portant un autre document_id")
    delivered_ids = [c.get("chunk_id") for c in chunks]
    if len(set(delivered_ids)) != len(delivered_ids):
        problems.append(f"chunk_id dupliqués dans la livraison ({len(delivered_ids) - len(set(delivered_ids))})")
    parent_ids = {p.get("parent_id") for p in parents}
    orphans = {c.get("parent_id") for c in eligible} - parent_ids
    if orphans:
        notes.append(f"{len(orphans)} parent_id d'un chunk éligible absents de parents.jsonl")

    # somme de contrôle contre le manifeste amont
    chunks_sha = sha256_file(folder / "chunks.jsonl")
    entry = manifest_entries.get(document_id)
    if entry:
        if entry.get("chunks_jsonl_sha256") and entry["chunks_jsonl_sha256"] != chunks_sha:
            problems.append("chunks.jsonl ne correspond pas à sa somme de contrôle déclarée — transfert incomplet ?")
        if entry.get("eligible_chunks_count") not in (None, len(eligible)):
            problems.append(f"eligible_chunks_count déclaré {entry['eligible_chunks_count']}, compté {len(eligible)}")
    elif manifest_entries:
        notes.append("document absent du manifeste : livraison non déclarée")

    # identité : ce que ce document *est*, indépendamment de son admissibilité
    match = known.get(sha)
    removed_entry = removed_by_id.get(document_id) or (removed_by_id.get(match["document_id"]) if match else None)
    if removed_entry:
        identity = "doublon-retiré"
        notes.append("ce document a été retiré délibérément du corpus : "
                     + str(removed_entry.get("reason", ""))[:140])
    elif match is None:
        identity = "ajout"
    elif match["parser"] != (backend, version):
        identity = "re-parse"
        notes.append(f"même PDF (sha256 {sha[:16]}…), parseur {match['parser'][0]} {match['parser'][1]} "
                     f"-> {backend} {version} : document_id {match['document_id']} -> {document_id}. "
                     "Indexé sur l'identifiant, ce document passerait pour un ajout et le corpus "
                     "contiendrait deux fois le même texte.")
    elif match["chunks_jsonl_sha256"] == chunks_sha:
        identity = "sans-effet"
    else:
        identity = "révision"

    # quasi-doublon : le sha256 ne dit rien d'un même papier livré sous un autre PDF
    duplicates = []
    if corpus_fingerprints is not None and identity == "ajout":
        duplicates = near_duplicate_of(delivered_shingles(folder), corpus_fingerprints,
                                       exclude=match["document_id"] if match else None)
        for hit in duplicates:
            if hit["verdict"] == "doublon":
                problems.append(
                    f"contenu déjà dans le corpus : containment {hit['containment']} avec "
                    f"{hit['document_id']} (seuil doublon {DUPLICATE_CONTAINMENT}). Un même "
                    "papier sous un autre PDF a un autre sha256 et passerait l'identité")
            else:
                notes.append(
                    f"édition proche de {hit['document_id']} : containment {hit['containment']} "
                    f"(seuil {EDITION_CONTAINMENT}). Décision explicite exigée — l'importer "
                    "sans trancher mettrait deux éditions du même ouvrage en concurrence")

    admissible = not problems
    verdict = identity if admissible else "rejet"

    result = {
        "folder": folder.name, "verdict": verdict, "identity": identity, "admissible": admissible,
        "document_id": document_id, "sha256": sha,
        "title": document.get("title"), "parser": f"{backend} {version}",
        "chunks": len(chunks), "eligible_chunks": len(eligible), "parents": len(parents),
        "chunks_jsonl_sha256": chunks_sha,
        "known_locally": match["document_id"] if match else None,
        "has_local_metadata": document_id in metadata_ids,
        "near_duplicates": duplicates,
        "needs_edition_decision": [h["document_id"] for h in duplicates if h["verdict"] == "édition"],
        "problems": problems, "notes": notes,
    }
    if identity in ("révision", "re-parse") and match:
        delivered = {c.get("chunk_id") for c in chunks}
        result["chunk_id_delta"] = {
            "locally_present": len(match["chunk_ids"]),
            "delivered": len(delivered),
            "kept": len(match["chunk_ids"] & delivered),
            "disappearing": len(match["chunk_ids"] - delivered),
            "appearing": len(delivered - match["chunk_ids"]),
        }
    return result


# ----------------------------------------------------------------- rapport

def build_report(delivery: Path | None) -> dict:
    known = local_documents()
    removed = corpus_overlay.removed_documents()
    metadata_ids = ({d["document_id"] for d in json.loads(METADATA.read_text(encoding="utf-8"))["documents"]}
                    if METADATA.exists() else set())
    gold_documents, gold_chunks = gold_anchors()

    if delivery is None:                                    # auto-test : le corpus vu comme une livraison
        folders, manifest = sorted(p for p in INGESTED.iterdir() if p.is_dir()), {"present": False, "notes": [
            "auto-test : data/processed/ingested est relu comme s'il s'agissait d'une livraison ; "
            "tout doit ressortir « sans-effet »"]}
        source = str(INGESTED.relative_to(ROOT))
    else:
        folders, manifest = locate_documents(delivery), read_manifest(delivery)
        source = str(delivery)

    # Les empreintes du corpus coûtent ~1,8 s ; on ne les calcule que s'il y a un ajout
    # candidat, et jamais pour l'auto-test (où tout est déjà « sans-effet »).
    preliminary = [inspect_document(f, known, removed, manifest.get("by_document_id", {}), metadata_ids)
                   for f in folders]
    corpus_fingerprints = (corpus_shingles()
                           if any(d.get("identity") == "ajout" for d in preliminary) else None)
    documents = ([inspect_document(f, known, removed, manifest.get("by_document_id", {}),
                                   metadata_ids, corpus_fingerprints) for f in folders]
                 if corpus_fingerprints is not None else preliminary)
    counts = {i: sum(1 for d in documents if d.get("identity") == i) for i in IDENTITIES}
    counts = {k: v for k, v in counts.items() if v}
    refused = sum(1 for d in documents if not d.get("admissible", False))
    if refused:
        counts["refusés"] = refused

    accepted = [d for d in documents if d.get("admissible") and d.get("identity") in ("ajout", "révision", "re-parse")]
    new_chunks = sum(d["eligible_chunks"] for d in accepted if d["identity"] == "ajout")
    revised_chunks = sum(d["eligible_chunks"] for d in accepted if d["identity"] != "ajout")
    touched_gold_documents = sorted({d["known_locally"] for d in accepted
                                     if d["identity"] != "ajout" and d["known_locally"] in gold_documents})
    gold_at_risk = 0
    for d in accepted:
        if d["identity"] != "ajout" and d["known_locally"]:
            local = next((v for v in known.values() if v["document_id"] == d["known_locally"]), None)
            if local:
                gold_at_risk += len(local["chunk_ids"] & gold_chunks)

    index = index_state()
    to_embed = new_chunks + revised_chunks
    sample_chunks = []
    for folder in folders[:20]:
        path = folder / "chunks.jsonl"
        if path.exists():
            sample_chunks.extend(read_jsonl(path)[:20])
    sample_documents = [json.loads((f / "document.json").read_text(encoding="utf-8"))
                        for f in folders[:50] if (f / "document.json").exists()]

    blocking = [f"{d['folder']}: {p}" for d in documents for p in d["problems"]]
    blocking += manifest.get("notes", []) if not manifest.get("supported", True) else []

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "delivery": source,
        "read_only": True,
        "manifest": {k: v for k, v in manifest.items() if k != "by_document_id"},
        "local_corpus": {
            "canonical_documents": len(known),
            "indexed_points": index.get("points"),
            "removed_documents": sorted(removed),
            "corpus_signature": corpus_overlay.signature(),
            "bm25_index": f"{corpus_overlay.bm25_name()}.json",
        },
        "counts": counts,
        "schema": schema_drift(sample_documents, sample_chunks),
        "index": index,
        "impact": {
            "documents_to_add": sum(1 for d in accepted if d["identity"] == "ajout"),
            "documents_to_revise": sum(1 for d in accepted if d["identity"] in ("révision", "re-parse")),
            "chunks_to_embed": to_embed,
            "embedding_minutes_estimate": round(to_embed / EMBEDDING_RATE / 60, 1) if to_embed else 0,
            "documents_needing_metadata_extraction": sum(1 for d in accepted if not d["has_local_metadata"]),
            "point_id_allocation": (
                {"next_free_point_id": index["next_free_point_id"],
                 "upstream_rule_would_give": index["points"],
                 "collision": index["points"] <= index["highest_point_id"],
                 "points_overwritten_by_upstream_rule": max(0, index["highest_point_id"] - index["points"] + 1)}
                if index.get("available") else {"unavailable": index.get("reason")}),
            # Depuis le 4 septembre 2026, le registre entre dans corpus_overlay.signature() :
            # un import change la signature, donc le nom de l'index BM25 et des caches qui la
            # portent. Ceux-là s'invalident seuls. Les autres restent à la charge de l'opérateur.
            "corpus_signature_would_change": bool(accepted),
            "invalidated_automatically": [
                f"data/lexical/{corpus_overlay.bm25_name()}.json — renommé par la nouvelle "
                "signature, donc introuvable ; quant_rag.bm25() reconstruit (~6 s), et vérifie "
                "en plus que le compte du manifeste égale celui de la collection",
                f"rag/benchmark/.cache/chunk-index-v4-{corpus_overlay.signature()}.pkl",
                f"rag/benchmark/.cache/rare-postings-3-60-{corpus_overlay.signature()}.pkl",
            ],
            "to_invalidate_by_hand": to_invalidate_by_hand(),
            "benchmark": {
                "gold_documents_touched_by_revision": touched_gold_documents,
                "gold_chunks_at_risk": gold_at_risk,
                "note": ("Un ajout ne casse aucun or mais change les scores : les nouveaux chunks "
                         "concourent contre l'or sans que l'or bouge. Les chiffres d'avant et d'après "
                         "ne sont pas comparables. Une révision, elle, renumérote les chunk_id du "
                         "document et peut détruire l'or lui-même."),
            },
        },
        "decision": "REFUS" if blocking else ("RIEN À FAIRE" if not accepted else "IMPORT POSSIBLE"),
        "blocking": blocking,
        "documents": documents,
    }


def summarise(report: dict) -> str:
    lines = [f"livraison   {report['delivery']}",
             f"décision    {report['decision']}    (lecture seule : rien n'a été écrit)",
             ""]
    manifest = report["manifest"]
    lines.append(f"manifeste   {'présent, schema_version ' + str(manifest.get('schema_version')) if manifest.get('present') else 'absent'}"
                 + (f", {manifest['documents_declared']} documents déclarés" if manifest.get("documents_declared") else ""))
    for note in manifest.get("notes", []):
        lines.append(f"            ! {note}")
    lines.append("")
    lines.append("verdicts    " + ("  ".join(f"{v}={n}" for v, n in report["counts"].items()) or "aucun document"))
    drift = report["schema"]
    for key, label in (("document_unknown_fields", "champs inattendus dans document.json"),
                       ("chunk_unknown_fields", "champs inattendus dans chunks.jsonl"),
                       ("chunk_missing_payload_fields", "champs de payload absents des chunks")):
        if drift[key]:
            lines.append(f"schéma      ! {label} : {drift[key]}")
    impact = report["impact"]
    lines += ["",
              f"impact      {impact['documents_to_add']} ajouts, {impact['documents_to_revise']} révisions, "
              f"{impact['chunks_to_embed']} chunks à embarquer (~{impact['embedding_minutes_estimate']} min à 2,6 chunks/s)",
              f"            {impact['documents_needing_metadata_extraction']} documents sans métadonnées consolidées "
              "— sans elles, leur titre embarqué serait le nom de fichier"]
    allocation = impact["point_id_allocation"]
    if allocation.get("collision"):
        lines.append(f"            ! la règle amont (id = compte + i) donnerait {allocation['upstream_rule_would_give']} "
                     f"et écraserait {allocation['points_overwritten_by_upstream_rule']} points ; "
                     f"premier identifiant libre : {allocation['next_free_point_id']}")
    elif allocation.get("unavailable"):
        lines.append(f"            index non lu ({allocation['unavailable'][:60]})")
    if impact["chunks_to_embed"]:
        lines.append("            la signature de corpus changerait : ce qui la porte dans son nom "
                     "s'invalide seul (BM25, caches du banc signés)")
        lines.append("            ! à invalider à la main :")
        for artefact in impact["to_invalidate_by_hand"]:
            lines.append(f"              - {artefact}")
    bench = impact["benchmark"]
    pending = [(d["folder"], d["needs_edition_decision"]) for d in report["documents"]
               if d.get("needs_edition_decision")]
    if pending:
        lines.append("doublons    ! décision explicite exigée avant import :")
        for folder, others in pending:
            lines.append(f"              - {folder} : édition proche de {', '.join(others)}")
    if bench["gold_documents_touched_by_revision"]:
        lines.append(f"banc        ! {len(bench['gold_documents_touched_by_revision'])} documents porteurs d'or révisés, "
                     f"{bench['gold_chunks_at_risk']} chunks d'or menacés")
    if report["blocking"]:
        lines += ["", "bloquant"]
        lines += [f"            - {b}" for b in report["blocking"][:20]]
        if len(report["blocking"]) > 20:
            lines.append(f"            … et {len(report['blocking']) - 20} autres")
    interesting = [d for d in report["documents"] if d.get("identity") != "sans-effet" or not d.get("admissible")]
    if interesting:
        lines += ["", "documents"]
        for d in interesting[:30]:
            mark = "" if d.get("admissible") else "  [REFUSÉ]"
            lines.append(f"            {str(d.get('identity') or 'illisible'):<15} {d['folder']}"
                         f"  {str(d.get('title'))[:46]}{mark}")
            for note in d.get("notes", []):
                lines.append(f"                            {note}")
            if d.get("chunk_id_delta"):
                k = d["chunk_id_delta"]
                lines.append(f"                            chunk_id : {k['kept']} conservés, "
                             f"{k['disappearing']} disparaissent, {k['appearing']} apparaissent")
        if len(interesting) > 30:
            lines.append(f"            … et {len(interesting) - 30} autres")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("delivery", nargs="?", type=Path, help="répertoire de la livraison")
    parser.add_argument("--from-corpus", action="store_true",
                        help="auto-test : relit data/processed/ingested comme une livraison")
    parser.add_argument("--json", type=Path, help="écrit le rapport complet dans ce fichier")
    args = parser.parse_args()
    if not args.delivery and not args.from_corpus:
        parser.error("donner un répertoire de livraison, ou --from-corpus")
    if args.delivery and not args.delivery.is_dir():
        parser.error(f"répertoire introuvable : {args.delivery}")

    report = build_report(None if args.from_corpus else args.delivery)
    print(summarise(report))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
        print(f"\nrapport     {args.json}")
    sys.exit(1 if report["decision"] == "REFUS" else 0)


if __name__ == "__main__":
    main()
