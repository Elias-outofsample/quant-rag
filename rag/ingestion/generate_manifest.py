"""Relevé d'une livraison — ce qui est arrivé, scellé à la réception.

Nous générons ce fichier nous-mêmes, sans rien demander à l'amont. Il faut être clair
sur ce que ça vaut :

  ce qu'il prouve       l'état exact des fichiers au moment de la réception, sur cette
                        machine. Toute modification ultérieure du répertoire de livraison
                        (édition accidentelle, import partiel, reprise ratée) devient
                        détectable en recalculant le relevé.
  ce qu'il ne prouve PAS que la livraison soit complète. Les sommes sont calculées sur ce
                        qui est arrivé, pas sur ce qui a été envoyé : un document oublié
                        en amont, ou un fichier tronqué avant l'archivage, produit un
                        relevé parfaitement cohérent. Seule la somme de l'archive
                        (``--archive``), communiquée par l'amont, ferme ce trou.

Ce n'est donc pas un manifeste au sens du ``corpus_manifest.json`` amont — celui-là est
une *attestation du producteur*. C'est un accusé de réception. Les deux ne se remplacent
pas, et confondre l'un avec l'autre, c'est croire vérifier un transfert qu'on ne vérifie pas.

Le relevé conserve les identités amont **telles quelles**. Il n'invente pas d'identifiant
local : ``document_id`` est la clé de jointure de tout ce qui existe déjà ici
(payload Qdrant, documents-metadata-v1, duplicates-v1, titles-clean-v1, or des bancs),
et les 32 422 ``chunk_id`` du corpus sont déjà uniques sans collision. L'identité *d'import*
— celle qui décide ajout/révision/re-parse — est le ``sha256`` du PDF, portée par le registre.

    .venv/bin/python rag/ingestion/generate_manifest.py <livraison> --id 2026-09-04-01
    .venv/bin/python rag/ingestion/generate_manifest.py <livraison> --archive livraison.tar.zst
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "rag"))
import corpus_overlay  # noqa: E402
from inspect_delivery import (REQUIRED_FILES, locate_documents, local_documents,  # noqa: E402
                              read_jsonl, sha256_file)

SCHEMA = "releve-livraison-v1"


def document_entry(folder: Path, known: dict[str, dict]) -> dict:
    document = json.loads((folder / "document.json").read_text(encoding="utf-8"))
    chunks = read_jsonl(folder / "chunks.jsonl") if (folder / "chunks.jsonl").exists() else []
    parents = read_jsonl(folder / "parents.jsonl") if (folder / "parents.jsonl").exists() else []
    blocks = sum(1 for line in (folder / "blocks.jsonl").open(encoding="utf-8")
                 if line.strip()) if (folder / "blocks.jsonl").exists() else 0
    sha = document.get("sha256")
    match = known.get(sha)
    files = {name: (sha256_file(folder / name) if (folder / name).exists() else None)
             for name in REQUIRED_FILES}
    # Aucun verdict ici : le relevé constate, il ne décide pas. La politique
    # (ajout / révision / re-parse / doublon-retiré / refus) a une seule implémentation,
    # dans inspect_delivery.py — deux en auraient divergé.
    return {
        # identités amont, conservées telles quelles — aucun identifiant local n'est inventé
        "document_id": document.get("document_id"),
        "folder": folder.name,
        "sha256": sha,                                   # identité d'import (le PDF)
        "title": document.get("title"),
        "filename": document.get("filename"),
        "parser": {"backend": document.get("parser_backend"), "version": document.get("parser_version")},
        "counts": {"blocks": blocks, "chunks": len(chunks), "parents": len(parents),
                   "eligible_chunks": sum(1 for c in chunks if c.get("rag_eligible") is True)},
        "files": files,
        "known_locally_as": match["document_id"] if match else None,
        "identical_to_local_chunks": bool(match and match["chunks_jsonl_sha256"] == files["chunks.jsonl"]),
        # ce que l'amont ne fournit pas : mesuré 1/258 sur le corpus actuel
        "upstream_bibliography": {"authors": bool(document.get("authors")),
                                  "publication_year": bool(document.get("publication_year"))},
    }


def build(delivery: Path, delivery_id: str | None, archive: Path | None) -> dict:
    known = local_documents()
    folders = locate_documents(delivery)
    documents = [document_entry(f, known) for f in folders]
    return {
        "schema": SCHEMA,
        "kind": "accusé de réception, calculé ici — pas une attestation du producteur",
        "attests": "l'état des fichiers au moment de la réception, sur cette machine",
        "does_not_attest": ("la complétude de la livraison : les sommes portent sur ce qui est "
                            "arrivé, pas sur ce qui a été envoyé"),
        "delivery_id": delivery_id or datetime.now(timezone.utc).strftime("%Y-%m-%d-%H%M"),
        "sealed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": {"path": str(delivery),
                   "archive": str(archive) if archive else None,
                   "archive_sha256": sha256_file(archive) if archive and archive.exists() else None},
        "corpus_at_reception": {"signature": corpus_overlay.signature(),
                                "label": corpus_overlay.LABEL,
                                "canonical_documents": len(known),
                                "removed_documents": len(corpus_overlay.removed_documents())},
        "totals": {"documents": len(documents),
                   "chunks": sum(d["counts"]["chunks"] for d in documents),
                   "eligible_chunks": sum(d["counts"]["eligible_chunks"] for d in documents),
                   "already_known_by_sha256": sum(1 for d in documents if d["known_locally_as"]),
                   "identical_to_local": sum(1 for d in documents if d["identical_to_local_chunks"])},
        "documents": sorted(documents, key=lambda d: d["folder"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("delivery", type=Path)
    parser.add_argument("--id", dest="delivery_id", help="identifiant de la livraison")
    parser.add_argument("--archive", type=Path,
                        help="archive reçue : sa somme est le seul contrôle de transfert bout en bout")
    parser.add_argument("--out", type=Path, help="par défaut : <livraison>/releve.json")
    args = parser.parse_args()
    if not args.delivery.is_dir():
        parser.error(f"répertoire introuvable : {args.delivery}")

    report = build(args.delivery, args.delivery_id, args.archive)
    out = args.out or args.delivery / "releve.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({"status": "COMPLETED", "releve": str(out), **report["totals"]},
                     ensure_ascii=False, indent=1))
    if not report["source"]["archive_sha256"]:
        print("\nATTENTION : sans --archive, ce relevé ne contrôle pas le transfert amont.\n"
              "            Demander la somme sha256 de l'archive et la comparer.")


if __name__ == "__main__":
    main()
