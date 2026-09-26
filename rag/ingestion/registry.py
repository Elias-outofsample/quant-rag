"""Registre d'imports — d'où vient chaque document, et ce qu'il faut défaire pour l'enlever.

Le registre est le seul fichier qui répond à « quels documents composent ce corpus ». Il a
deux rôles, et c'est voulu qu'un seul fichier les tienne :

  traçabilité   par document : sa provenance, sa livraison, son parseur, les sommes de ses
                quatre fichiers canoniques, et la plage d'identifiants de points qu'il
                occupe dans Qdrant — donc ce qu'il faut supprimer pour l'annuler.
  invalidation  son empreinte entre dans ``corpus_overlay.signature()``. Ajouter ou retirer
                un document change la signature, donc le nom de l'index BM25 et des caches
                du banc : un index bâti sur un autre ensemble de documents n'est plus
                rechargé par erreur, il n'est simplement plus trouvé.

Sans ce second rôle, un import empoisonnerait le chemin lexical en silence : la signature
ne hachait que les deux fichiers d'overlay, et ``quant_rag.bm25()`` recharge l'index par son
nom sans contrôler son contenu.

**La clé est le ``sha256`` du PDF, jamais le ``document_id``.** Le ``document_id`` amont vaut
``doc-sha1(sha256 ⋮ backend ⋮ version ⋮ 'canonical-v1')[:16]`` : il encode la version du
parseur, donc le même PDF relu par MinerU 3.5.0 en porte un autre. Il reste la clé de
*jointure* avec tout le reste (payload Qdrant, métadonnées, overlays, or des bancs) ; il ne
peut pas être la clé d'*identité*.

L'empreinte ne porte que sur ce qui définit le corpus — ``sha256``, ``document_id``,
``status``, somme de ``chunks.jsonl`` — jamais sur les horodatages : reconstruire le
registre sans rien changer au corpus laisse la signature intacte.

    .venv/bin/python rag/ingestion/registry.py                 # résumé (lecture seule)
    .venv/bin/python rag/ingestion/registry.py --verify        # registre vs corpus réel
    .venv/bin/python rag/ingestion/registry.py --build         # (re)construit le registre
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "rag"))
import corpus_overlay  # noqa: E402
from inspect_delivery import REQUIRED_FILES, read_jsonl, sha256_file  # noqa: E402

PATH = corpus_overlay.REGISTRY
#: Chunks entrés par une livraison, au format de ``rows.jsonl`` ({document, chunk}).
#: ``rows.jsonl`` est l'artefact amont et ne connaît que les 19 443 chunks livrés ; tout ce
#: qui le lit — graphe d'entités, vue corpus du banc — serait aveugle aux imports. Ce
#: fichier est le complément, reconstruit depuis le registre et le canonique, jamais à la
#: main. Il est lisible sans Qdrant : ``.venv-gliner`` n'a pas ``qdrant_client``, et
#: l'isolement des environnements est délibéré.
IMPORTED_ROWS = HERE / "imported-rows.jsonl"
INGESTED = ROOT / "data" / "processed" / "ingested"
SCHEMA = "registry-v1"
#: Livraison synthétique des 258 documents présents avant l'existence du registre.
BASELINE = {"id": "corpus-initial", "transport": "dépôt (état livré)",
            "received_at": "2026-09-01T00:00:00+00:00", "schema_version": None,
            "producer": "windows-pipeline", "producer_commit": None,
            "note": "rétro-rempli le 4 septembre 2026 ; antérieur au registre"}
RECIPE = {"embedding": "clean-titles-v1", "metadata": "documents-metadata-v1",
          "text_overlays": ["tables-markdown-v1"]}


def point_ranges() -> dict[str, dict]:
    """document_id -> plage d'identifiants de points occupée. Vide si Qdrant est pris."""
    try:
        import quant_rag

        per: dict[str, list[int]] = collections.defaultdict(list)
        offset = None
        while True:
            points, offset = quant_rag.client().scroll(
                quant_rag.COLLECTION, limit=8192, offset=offset,
                with_payload=["document_id"], with_vectors=False)
            for point in points:
                per[point.payload["document_id"]].append(int(point.id))
            if offset is None:
                break
        return {doc: {"first": min(ids), "last": max(ids), "count": len(ids),
                      "contiguous": max(ids) - min(ids) + 1 == len(ids)}
                for doc, ids in per.items()}
    except Exception:
        return {}


def _source_classes() -> dict[str, str]:
    """La classe de provenance de chaque document, dérivée des métadonnées consolidées.

    Le tampon arXiv n'existe que là — il est lu dans le texte de la première page, pas dans
    le nom de fichier — et il décide à lui seul de 137 documents dont le nom ne dit rien
    d'arXiv. Sans les métadonnées, la règle reste applicable mais perd ce signal : voir
    ``_classer_sans_metadonnees``.
    """
    sys.path.insert(0, str(ROOT / "rag" / "metadata"))
    import source_class as sc

    chemin = ROOT / "rag" / "metadata" / "documents-metadata-v1.json"
    if not chemin.exists():
        return {}
    donnees = json.loads(chemin.read_text(encoding="utf-8"))
    return {row["document_id"]: sc.classer_ligne(row)
            for row in donnees.get("documents", [])}


def _classer_sans_metadonnees(document: dict) -> str:
    """Repli pour un document que les métadonnées consolidées ne connaissent pas encore.

    ``document.json`` porte ``filename`` et ``page_count`` ; il ne porte pas le tampon
    arXiv. Un préprint dont le nom ne dit rien d'arXiv sera donc classé ``autre`` jusqu'à ce
    que ``apply_metadata`` passe. Se taire vaudrait mieux que mentir, mais un champ absent
    rendrait le registre irrégulier ; le repli est donc explicite et documenté ici.
    """
    sys.path.insert(0, str(ROOT / "rag" / "metadata"))
    import source_class as sc

    return sc.classer(document.get("filename"), document.get("page_count"), None)


def build() -> dict:
    """Reconstruit le registre depuis le disque, **en gardant la provenance déjà connue**.

    La provenance est une information que le disque ne porte pas : elle vient de l'import.
    Une reconstruction qui la réécrirait à zéro effacerait la seule trace de l'origine des
    documents — et le fichier n'existe que pour ça.
    """
    removed = corpus_overlay.removed_documents()
    ranges = point_ranges()
    previous = load() or {}
    known = {d["sha256"]: d for d in previous.get("documents", [])}
    # ``source_class`` est DÉRIVÉ à chaque reconstruction, jamais recopié de l'entrée
    # précédente. La raison est mécanique : ``build()`` refait chaque entrée depuis un
    # littéral et ne reporte que ``delivery`` — tout champ simplement stocké serait effacé
    # au ``--build`` suivant, sans erreur et sans trace. Le même mécanisme fait déjà perdre
    # un identifiant à ``deliveries`` (2026-09-06-nuit-d02 manque aux 159 entrées alors que
    # 160 sont portés par les documents). Une règle ne peut pas s'effacer.
    classes = _source_classes()
    deliveries = {d["id"]: d for d in previous.get("deliveries", []) if d.get("id")}
    deliveries.setdefault(BASELINE["id"], dict(BASELINE))
    documents = []
    for folder in sorted(INGESTED.iterdir()):
        if not (folder / "document.json").exists():
            continue
        document = json.loads((folder / "document.json").read_text(encoding="utf-8"))
        chunks = read_jsonl(folder / "chunks.jsonl") if (folder / "chunks.jsonl").exists() else []
        document_id = document["document_id"]
        entry = removed.get(document_id)
        documents.append({
            "sha256": document["sha256"],                       # identité — la clé
            "document_id": document_id,                         # jointure — pas l'identité
            "folder": folder.name,
            "title": document.get("title"),
            "filename": document.get("filename"),
            "parser": {"backend": document.get("parser_backend"), "version": document.get("parser_version")},
            "delivery": dict((known.get(document["sha256"]) or {}).get("delivery") or BASELINE),
            "files": {name: (sha256_file(folder / name) if (folder / name).exists() else None)
                      for name in REQUIRED_FILES},
            "counts": {"chunks": len(chunks),
                       "eligible_chunks": sum(1 for c in chunks if c.get("rag_eligible") is True),
                       "parents": sum(1 for _ in (folder / "parents.jsonl").open(encoding="utf-8"))
                       if (folder / "parents.jsonl").exists() else 0},
            "point_ids": ranges.get(document_id),
            "status": "removed-duplicate" if entry else "active",
            "replaced_by": (entry or {}).get("kept", {}).get("document_id"),
            "reason": (entry or {}).get("reason"),
            "recipe": dict(RECIPE),
            # `arxiv` | `ssrn` | `livre` | `autre`, par rag/metadata/source_class.py.
            # N'entre PAS dans registry_digest (sha256, document_id, status, chunks.jsonl) :
            # l'ajouter ne renomme aucun index et ne lève aucun gel.
            "source_class": classes.get(document_id)
                            or _classer_sans_metadonnees(document),
        })
    documents.sort(key=lambda d: d["sha256"])
    by_status = collections.Counter(d["status"] for d in documents)
    seen = {d["delivery"]["id"] for d in documents}
    per_delivery = collections.Counter(d["delivery"]["id"] for d in documents)
    ordered = [dict(deliveries[i], documents=per_delivery[i]) for i in sorted(deliveries) if i in seen]
    return {
        "schema": SCHEMA,
        "key": "sha256 du PDF source — le document_id encode la version du parseur",
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "corpus_label": corpus_overlay.LABEL,
        "totals": {"documents": len(documents), "by_status": dict(by_status),
                   "active_chunks": sum(d["counts"]["eligible_chunks"] for d in documents
                                        if d["status"] == "active")},
        "deliveries": ordered or [dict(BASELINE)],
        "documents": documents,
    }


def write_imported_rows() -> int:
    """(Re)construit ``imported-rows.jsonl`` depuis le registre et les fichiers canoniques."""
    data = load()
    if not data:
        return 0
    rows = []
    for entry in sorted(data["documents"], key=lambda d: d["sha256"]):
        if entry.get("delivery", {}).get("id") in (None, BASELINE["id"]) or entry["status"] != "active":
            continue
        folder = INGESTED / entry["folder"]
        if not (folder / "chunks.jsonl").exists():
            continue
        document = json.loads((folder / "document.json").read_text(encoding="utf-8"))
        for chunk in read_jsonl(folder / "chunks.jsonl"):
            if chunk.get("rag_eligible") is True:
                rows.append({"document": document, "chunk": chunk})
    if rows:
        IMPORTED_ROWS.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                                 encoding="utf-8")
    else:
        IMPORTED_ROWS.unlink(missing_ok=True)
    return len(rows)


def load() -> dict | None:
    return json.loads(PATH.read_text(encoding="utf-8")) if PATH.exists() else None


def provenance() -> dict:
    """Résumé pour ``quant_rag.py --status``. Silencieux si le registre n'existe pas."""
    data = load()
    if not data:
        return {"registry": None, "note": "registre absent : provenance inconnue"}
    documents = data["documents"]
    deliveries = collections.Counter(d["delivery"]["id"] for d in documents)
    return {
        "registry": str(PATH.relative_to(ROOT)),
        "digest": corpus_overlay.registry_digest(),
        "documents": dict(collections.Counter(d["status"] for d in documents)),
        "deliveries": [{"id": entry["id"], "transport": entry["transport"],
                        "received_at": entry["received_at"], "documents": deliveries[entry["id"]]}
                       for entry in data.get("deliveries", [])],
        "parsers": dict(collections.Counter(f'{d["parser"]["backend"]} {d["parser"]["version"]}'
                                            for d in documents)),
    }


def verify() -> dict:
    """Le registre décrit-il le corpus réel ? Lecture seule, aucune écriture."""
    data = load()
    if not data:
        return {"status": "ABSENT", "problems": ["registre absent"]}
    problems = []
    registered = {d["sha256"]: d for d in data["documents"]}
    on_disk = {}
    for folder in sorted(INGESTED.iterdir()):
        if (folder / "document.json").exists():
            document = json.loads((folder / "document.json").read_text(encoding="utf-8"))
            on_disk[document["sha256"]] = (folder, document)
    for sha in registered.keys() - on_disk.keys():
        problems.append(f"au registre mais absent du disque : {registered[sha]['document_id']}")
    for sha in on_disk.keys() - registered.keys():
        problems.append(f"sur le disque mais absent du registre : {on_disk[sha][1]['document_id']}")
    changed = []
    for sha in registered.keys() & on_disk.keys():
        folder = on_disk[sha][0]
        if registered[sha]["files"]["chunks.jsonl"] != sha256_file(folder / "chunks.jsonl"):
            changed.append(registered[sha]["document_id"])
    if changed:
        problems.append(f"chunks.jsonl modifié depuis l'enregistrement : {changed[:5]}"
                        + (f" (+{len(changed) - 5})" if len(changed) > 5 else ""))
    ranges = point_ranges()
    if ranges:
        active = {d["document_id"] for d in data["documents"] if d["status"] == "active"}
        for missing in sorted(active - ranges.keys()):
            problems.append(f"actif au registre mais absent de la collection : {missing}")
        for extra in sorted(ranges.keys() - active):
            problems.append(f"dans la collection mais pas actif au registre : {extra}")
    return {"status": "OK" if not problems else "DIVERGENCE",
            "documents": len(registered), "index_read": bool(ranges),
            "digest": corpus_overlay.registry_digest(),
            "corpus_signature": corpus_overlay.signature(), "problems": problems}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--build", action="store_true", help="(re)construit le registre et l'écrit")
    parser.add_argument("--verify", action="store_true", help="compare le registre au corpus réel")
    args = parser.parse_args()

    if args.build:
        before = corpus_overlay.signature()
        data = build()
        PATH.parent.mkdir(parents=True, exist_ok=True)
        PATH.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
        imported = write_imported_rows()
        print(json.dumps({"status": "COMPLETED", "registry": str(PATH.relative_to(ROOT)),
                          **data["totals"], "imported_rows": imported,
                          "digest": corpus_overlay.registry_digest(),
                          "corpus_signature": {"before": before, "after": corpus_overlay.signature()}},
                         ensure_ascii=False, indent=1))
        return
    if args.verify:
        report = verify()
        print(json.dumps(report, ensure_ascii=False, indent=1))
        sys.exit(0 if report["status"] == "OK" else 1)
    print(json.dumps(provenance(), ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
