"""Source B — fabriquer les fichiers canoniques d'un PDF local, sans rien écrire dans le corpus.

Ce module est un **producteur**, pas un importeur. Il s'arrête là où
``inspect_delivery.py`` commence : il pose sur le disque une livraison au format du contrat
(``rag/ingestion/README.md``), et rien d'autre. Aucune écriture dans Qdrant, BM25, le
graphe, ``data/processed/ingested/`` ni les overlays — c'est ``apply_delivery.py`` qui les
fait, et lui seul.

    PDF local ─> MinerU 3.4.5 ─> parse_document + make_chunks + build_parents
                                   └─> livraisons/<id>/processed/doc-<sha256(pdf)[:16]>/
                                         document.json blocks.jsonl chunks.jsonl parents.jsonl

Il reproduit **délibérément** la pipeline amont de ``scripts/mineru_batch_runner.ps1`` et
``scripts/normalize_split_mineru.py`` : mêmes options MinerU, même découpage de repli, même
suffixage des ``block_id``. Ce n'est pas du mimétisme : un écart d'option change les blocs,
donc les chunks, donc la somme de ``chunks.jsonl`` — et un document produit ici cesserait
d'être comparable aux 258 que le corpus a reçus de l'amont.

Trois faits du dépôt commandent ce fichier :

  1. ``document_id = doc-sha1(sha256_pdf | backend | version | 'canonical-v1')[:16]``
     **encode la version du parseur**. On installe donc MinerU **3.4.5**, la version de
     l'amont — pas la plus récente disponible, celle qui préserve l'identité. Le script
     refuse de produire si la version installée n'est pas celle-là.
  2. La clé d'ingestion est le ``sha256`` du PDF. Un PDF déjà au registre est refusé, sauf
     ``--temoin`` : le re-parse d'un document connu n'a qu'un usage légitime, mesurer si ce
     Mac reproduit l'amont, et ce résultat-là ne s'importe jamais.
  3. ``extract_metadata`` lit le PDF dans ``data/papers/<filename>`` (source S3) et
     ``apply_delivery`` refuse un titre venu du nom de fichier. Le PDF est donc déposé là
     sous le nom que porte ``document.json``, sans quoi l'import échouerait plus tard, à
     l'étape la plus chère.

    .venv-mineru/bin/python rag/ingestion/parse_local.py <pdf>... --id 2026-09-04-source-b-01
    .venv-mineru/bin/python rag/ingestion/parse_local.py --entree --id ...
    .venv-mineru/bin/python rag/ingestion/parse_local.py <pdf> --temoin --out .../temoins
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))

SOURCE_B = HERE / "source-b"
ENTREE = SOURCE_B / "entree"
TRAVAIL = SOURCE_B / "travail"
LIVRAISONS = SOURCE_B / "livraisons"
PAPERS = ROOT / "data" / "papers"
REGISTRY = HERE / "registry-v1.json"

#: La version de l'amont. Elle n'est pas un défaut modifiable : elle est dans l'identité de
#: chaque document. Voir ``inspect_delivery.KNOWN_PARSERS``, qui refuse tout le reste.
EXPECTED_MINERU = "3.4.5"
#: Repli du ``.ps1`` amont quand MinerU échoue sur le document entier.
TRANCHE = 25


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def fail(message: str) -> None:
    print(f"\nREFUS : {message}", file=sys.stderr)
    raise SystemExit(1)


# ------------------------------------------------------------------ contrôles préalables

def check_mineru(python: Path) -> str:
    """La version installée doit être celle qui est encodée dans les identifiants."""
    import importlib.metadata as md

    try:
        version = md.version("mineru")
    except md.PackageNotFoundError:
        fail("MinerU n'est pas installé dans cet interpréteur. Ce script s'exécute avec "
             ".venv-mineru/bin/python, jamais avec .venv (transformers y est en 5.x, MinerU "
             "exige < 5).")
    if version != EXPECTED_MINERU:
        fail(f"MinerU {version} installé, {EXPECTED_MINERU} attendu. La version entre dans "
             f"document_id : produire avec {version} donnerait à chaque PDF une identité que "
             f"le corpus ne reconnaît pas, et inspect_delivery la refuserait "
             f"(KNOWN_PARSERS). Réinstalle : pip install 'mineru[pipeline]=={EXPECTED_MINERU}'")
    return version


def known_sha256() -> dict[str, dict]:
    """sha256 -> entrée du registre. La clé d'ingestion, jamais le document_id."""
    if not REGISTRY.exists():
        return {}
    data = json.loads(REGISTRY.read_text(encoding="utf-8"))
    return {d["sha256"]: d for d in data.get("documents", [])}


def page_count(pdf: Path) -> int:
    from pypdf import PdfReader

    return len(PdfReader(str(pdf)).pages)


# ------------------------------------------------------------------ MinerU

def run_mineru(pdf: Path, out: Path, start: int, end: int, device: str, log: Path) -> bool:
    """Un appel MinerU, aux options exactes de ``mineru_batch_runner.ps1``.

    ``-b pipeline -m auto -f true -t true`` : toute variation ici change les blocs, donc les
    chunks, donc la somme de contrôle qui décide ajout/révision. ``-l`` n'est pas passé —
    l'amont ne le passait pas non plus, et son défaut (``ch``) fait partie de la recette.
    """
    out.mkdir(parents=True, exist_ok=True)
    binary = Path(sys.executable).parent / "mineru"
    command = [str(binary), "-p", str(pdf), "-o", str(out),
               "-b", "pipeline", "-m", "auto", "-s", str(start), "-e", str(end),
               "-f", "true", "-t", "true"]
    environment = {**os.environ, "MINERU_DEVICE_MODE": device,
                   # miroir par défaut = ModelScope ; HF est plus rapide depuis l'Europe et
                   # ses délais d'attente ne se déguisent pas en échec d'installation.
                   "MINERU_MODEL_SOURCE": os.environ.get("MINERU_MODEL_SOURCE", "huggingface")}
    started = time.monotonic()
    with log.open("ab") as handle:
        handle.write(f"\n=== {now()} mineru -s {start} -e {end} -d {device}\n".encode())
        handle.flush()
        code = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT,
                              env=environment).returncode
    produced = list(out.rglob("*_content_list.json"))
    ok = code == 0 and bool(produced)
    print(f"    mineru -s {start} -e {end} : {'ok' if ok else 'ÉCHEC'} "
          f"({time.monotonic() - started:.0f} s, {len(produced)} content_list)")
    return ok


def mineru_document(pdf: Path, staging: Path, pages: int, device: str) -> Path:
    """Le document entier ; à défaut, les tranches du repli amont. Rend le dossier de travail."""
    log = staging / "mineru.log"
    staging.mkdir(parents=True, exist_ok=True)
    whole = staging / f"part-0-{pages}"
    if list(whole.rglob("*_content_list.json")):
        print(f"    (déjà parsé : {whole.name})")
        return staging
    if run_mineru(pdf, whole, 0, pages - 1, device, log):
        return staging
    print(f"    repli par tranches de {TRANCHE} pages")
    shutil.rmtree(whole, ignore_errors=True)
    for start in range(0, pages, TRANCHE):
        end = min(start + TRANCHE - 1, pages - 1)
        part = staging / f"part-{start}-{end}"
        if list(part.rglob("*_content_list.json")):
            continue
        if not run_mineru(pdf, part, start, end, device, log):
            fail(f"MinerU échoue sur {pdf.name}, pages {start}-{end}. Journal : {log}")
    return staging


# ------------------------------------------------------------------ normalisation

def normalise(pdf: Path, staging: Path, pages: int):
    """``scripts/normalize_split_mineru.py``, repris sans changement de sémantique.

    Un seul écart, et il est sûr : les tranches sont triées **numériquement** et non
    lexicographiquement. Le tri amont range ``part-100-124`` avant ``part-25-49`` ; au-delà
    de 99 pages, il recolle donc les blocs dans le désordre, et ``make_chunks`` suit cet
    ordre. Aucun document de moins de 100 pages n'est affecté — la reproduction du témoin
    reste exacte — et le repli ne sert de toute façon qu'en cas d'échec du document entier.
    """
    from parsing.base import parse_document
    from parsing.canonical_chunker import build_parents, make_chunks

    parts = sorted((p for p in staging.glob("part-*-*") if p.is_dir()),
                   key=lambda p: int(re.fullmatch(r"part-(\d+)-(\d+)", p.name).group(1))
                   if re.fullmatch(r"part-(\d+)-(\d+)", p.name) else 10 ** 9)
    document, blocks = None, []
    if not parts and list(staging.rglob("*_content_list.json")):
        document, blocks = parse_document(pdf, "mineru", output_dir=staging)
    for part in parts:
        match = re.fullmatch(r"part-(\d+)-(\d+)", part.name)
        if not match or not list(part.rglob("*_content_list.json")):
            continue
        start = int(match.group(1))
        parsed_document, parsed_blocks = parse_document(pdf, "mineru", output_dir=part)
        if document is None:
            document = parsed_document
        for index, block in enumerate(parsed_blocks):
            block.page_idx += start
            block.block_id = f"{block.block_id}-{start}-{index}"
            block.metadata["source_part"] = part.name
            blocks.append(block)
    if document is None:
        fail(f"aucune sortie MinerU exploitable pour {pdf.name} sous {staging}")
    document.page_count = pages
    document.title = document.title or pdf.stem
    document.source_path = str(pdf)
    chunks, _ = make_chunks(document, blocks)
    parents = build_parents(document, blocks, chunks)
    return document, blocks, chunks, parents


def write_canonical(folder: Path, document, blocks, chunks, parents) -> dict:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "document.json").write_text(
        json.dumps(document.as_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    for name, rows in (("blocks.jsonl", [b.as_dict() for b in blocks]),
                       ("chunks.jsonl", [c.as_dict() for c in chunks]),
                       ("parents.jsonl", parents)):
        (folder / name).write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    return {name: sha256_file(folder / name)
            for name in ("document.json", "blocks.jsonl", "chunks.jsonl", "parents.jsonl")}


# ------------------------------------------------------------------ dépôt du PDF

def place_pdf(pdf: Path, filename: str) -> str:
    """``data/papers/<filename>`` — sans quoi la source S3 d'extract_metadata est aveugle.

    Un PDF différent portant déjà ce nom est un refus : écraser ferait mentir la
    bibliographie de l'autre document, et le lien ``document.json::filename`` -> PDF est ce
    qui rattache un import à sa source.
    """
    target = PAPERS / filename
    digest = sha256_file(pdf)
    if target.exists():
        if sha256_file(target) == digest:
            return "déjà présent, identique"
        fail(f"data/papers/{filename} existe et diffère (sha256 {sha256_file(target)[:12]} "
             f"contre {digest[:12]}). Choisis un autre nom : ce fichier est la source "
             "bibliographique d'un autre document.")
    shutil.copy2(pdf, target)
    return "déposé"


# ------------------------------------------------------------------ un document

def produce(pdf: Path, delivery: Path, known: dict, temoin: bool, device: str,
            filename: str | None, source: str | None) -> dict:
    digest = sha256_file(pdf)
    folder_name = f"doc-{digest[:16]}"
    print(f"\n{pdf.name}  sha256 {digest[:16]}…  -> {folder_name}")

    entry = known.get(digest)
    if entry and not temoin:
        fail(f"{pdf.name} est déjà au registre sous {entry['document_id']} "
             f"({entry.get('status')}, livraison {entry.get('delivery', {}).get('id')}). "
             "La clé d'ingestion est le sha256 du PDF : ce document n'est pas un ajout. "
             "Pour le re-parser à seule fin de mesure, relance avec --temoin — son résultat "
             "ne s'importe pas.")
    if entry:
        print(f"    TÉMOIN : déjà au registre sous {entry['document_id']} — mesure seulement, "
              "aucun import possible")

    pages = page_count(pdf)
    staging = TRAVAIL / folder_name
    print(f"    {pages} pages, device {device}")
    mineru_document(pdf, staging, pages, device)

    document, blocks, chunks, parents = normalise(pdf, staging, pages)
    if filename:
        document.filename = filename
    eligible = sum(1 for c in chunks if c.rag_eligible)
    folder = delivery / "processed" / folder_name
    sums = write_canonical(folder, document, blocks, chunks, parents)

    placed = None if temoin else place_pdf(pdf, document.filename)
    print(f"    {len(blocks)} blocs · {len(chunks)} chunks dont {eligible} éligibles · "
          f"{len(parents)} parents")
    print(f"    document_id {document.document_id} · chunks.jsonl {sums['chunks.jsonl'][:16]}…")
    if placed:
        print(f"    data/papers/{document.filename} : {placed}")

    return {
        "folder": folder_name,
        "document_id": document.document_id,
        "sha256": digest,
        "filename": document.filename,
        "title_upstream": document.title,
        "pages": pages,
        "counts": {"blocks": len(blocks), "chunks": len(chunks), "eligible_chunks": eligible,
                   "parents": len(parents)},
        "files": sums,
        "parser": {"backend": document.parser_backend, "version": document.parser_version,
                   "device": device, "options": "-b pipeline -m auto -f true -t true"},
        # provenance du PDF lui-même : ce qui rendra crédible un futur verdict « édition »
        # si un miroir ressert le même papier sous un autre PDF, donc un autre sha256.
        "provenance": {"source": source, "original_name": pdf.name,
                       "bytes": pdf.stat().st_size, "collected_at": now()},
        "temoin": bool(entry),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("pdfs", nargs="*", type=Path)
    parser.add_argument("--entree", action="store_true",
                        help=f"prendre tous les PDF de {ENTREE.relative_to(ROOT)}")
    parser.add_argument("--id", dest="delivery_id",
                        help="identifiant de la livraison (défaut : horodatage)")
    parser.add_argument("--out", type=Path, help="répertoire de livraison (défaut : livraisons/<id>)")
    parser.add_argument("--temoin", action="store_true",
                        help="autorise un PDF déjà au registre — mesure seulement, jamais un import")
    parser.add_argument("--device", default="mps", choices=("mps", "cpu"),
                        help="MPS par défaut ; CPU est le repli quand MPS échoue (le coût est "
                             "ponctuel, et il préserve l'identité — contrairement à un autre backend)")
    parser.add_argument("--nom", action="append", default=[], metavar="ORIGINE=NOM",
                        help="renommer un PDF pour data/papers (répétable)")
    parser.add_argument("--source", action="append", default=[], metavar="ORIGINE=TEXTE",
                        help="provenance déclarée : arXiv avec sa version, SSRN avec la date "
                             "du snapshot (répétable)")
    args = parser.parse_args()

    version = check_mineru(Path(sys.executable))
    pdfs = list(args.pdfs)
    if args.entree:
        pdfs += sorted(ENTREE.glob("*.pdf"))
    if not pdfs:
        parser.error("aucun PDF : donne des chemins, ou --entree")
    for pdf in pdfs:
        if not pdf.is_file():
            parser.error(f"introuvable : {pdf}")

    renames = dict(item.split("=", 1) for item in args.nom)
    sources = dict(item.split("=", 1) for item in args.source)
    delivery_id = args.delivery_id or datetime.now(timezone.utc).strftime("%Y-%m-%d-%H%M")
    delivery = args.out or LIVRAISONS / delivery_id
    known = known_sha256()

    print(f"MinerU {version} · livraison {delivery_id} · {len(pdfs)} PDF · "
          f"{len(known)} documents au registre")
    documents = [produce(pdf, delivery, known, args.temoin, args.device,
                         renames.get(pdf.name), sources.get(pdf.name)) for pdf in pdfs]

    report = {
        "schema": "source-b-production-v1",
        "kind": "production locale de fichiers canoniques — aucune écriture dans le corpus",
        "delivery_id": delivery_id,
        "produced_at": now(),
        "producer": {"host": "rag", "mineru": version, "device": args.device,
                     "normalizer": "src/parsing (parse_document + make_chunks + build_parents)"},
        "totals": {"documents": len(documents),
                   "chunks": sum(d["counts"]["chunks"] for d in documents),
                   "eligible_chunks": sum(d["counts"]["eligible_chunks"] for d in documents)},
        "documents": documents,
    }
    (delivery / "production.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n{delivery}/production.json")
    print(f"\nSuite : inspect_delivery.py {delivery}  (il n'écrit rien)")


if __name__ == "__main__":
    main()
