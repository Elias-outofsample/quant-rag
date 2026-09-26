"""Importe une livraison amont dans le corpus — ajouts seuls, reprise, annulation.

Le seul module de ce dépôt qui écrive dans Qdrant, BM25, le graphe et le corpus canonique.
Tout ce qui le précède (``inspect_delivery``, ``generate_manifest``, ``registry``) est là
pour qu'il ne le fasse jamais à l'aveugle.

L'ordre des étapes n'est pas une commodité, chacune est là parce qu'elle a été payée :

    0. verrou Qdrant       l'embarqué n'accepte qu'un processus ; on le constate d'abord,
                           on n'échoue pas au milieu (un serveur MCP au repos ne tient rien
                           et prend le verrou à son premier appel d'outil)
    1. plan                ``inspect_delivery`` ; un seul refus suffit à tout arrêter
    2. stage               data/processed/incoming/<livraison>/ — hors du corpus servi,
                           réversible par un rm
    3. métadonnées         **refus si absentes** : ``clean_title()`` retomberait sur le titre
                           d'export, dérivé du nom de fichier, et l'embedding verrait ce
                           titre-là — l'inverse exact du chantier B (v1 0,853 → 0,897)
    4. tableaux            un tableau non converti s'indexe en HTML brut : dense sur les
                           tableaux 0,730 contre 0,773 après conversion
    5. embedding           recette locale, reprise sur interruption, .npz par livraison ;
                           jamais de ré-embedding du corpus pour ajouter des documents
    6. journal in-flight   **avant le premier upsert** : sans lui, un import mort en cours
                           laisse des points orphelins et rien ne dit où ils commencent
    7. Qdrant              à ``max(id) + 1``. La règle amont ``compte + i`` écraserait 807
                           points, les trous laissés par les documents retirés
    8. promotion           les dossiers entrent dans ingested/
    9. registre            il enregistre les plages de points — donc de quoi annuler
   10. BM25                en dernier : la signature n'est définitive qu'après 4 et 9

**Ajouts seuls.** Une révision ou un re-parse est un refus explicite : le ``chunk_id`` étant
haché sur ``(document_id, indice, texte)``, une révision renumérote les identifiants du
document et peut détruire l'or des bancs (157 chunks d'or dans 133 documents).

    .venv/bin/python rag/ingestion/apply_delivery.py <livraison> --plan
    .venv/bin/python rag/ingestion/apply_delivery.py <livraison> --apply
    .venv/bin/python rag/ingestion/apply_delivery.py --rollback <id-de-livraison>
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for extra in (str(HERE), str(ROOT / "rag"), str(ROOT / "rag" / "metadata"),
              str(ROOT / "rag" / "tables"), str(ROOT / "src")):
    if extra not in sys.path:
        sys.path.insert(0, extra)

import corpus_overlay  # noqa: E402
import gel_corpus  # noqa: E402  — consulté par apply(), jamais par rollback()
import inspect_delivery as diag  # noqa: E402
import registry as reg  # noqa: E402
import verrou_collection  # noqa: E402

INCOMING = ROOT / "data" / "processed" / "incoming"
INGESTED = ROOT / "data" / "processed" / "ingested"
JOURNAL_DIR = HERE / "journal"
VECTOR_DIR = HERE / ".cache"
METADATA = ROOT / "rag" / "metadata" / "documents-metadata-v1.json"
EMBED_BLOCK = 64          # chunks par sauvegarde partielle (reprise)


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def fail(message: str) -> None:
    sys.exit(f"REFUS : {message}")


# ------------------------------------------------------------------ 0. verrou

def take_lock():
    """Ouvre Qdrant avant tout travail, ou refuse avec un message utile."""
    import quant_rag

    try:
        client = quant_rag.client()
        client.count(quant_rag.COLLECTION, exact=True)
        return quant_rag, client
    except Exception as error:
        if "already accessed" in str(error):
            fail("Qdrant embarqué est tenu par un autre processus (serveur MCP ?). "
                 "Ferme-le et relance — un import ne peut pas partager le stockage.\n"
                 f"        {type(error).__name__}: {str(error)[:160]}")
        fail(f"Qdrant inaccessible : {type(error).__name__}: {str(error)[:160]}")


# ------------------------------------------------------------------ 1. plan

def plan(delivery: Path, only: str | None, accept_editions: tuple[str, ...] = ()) -> tuple[dict, list[dict]]:
    report = diag.build_report(delivery)
    if report["blocking"]:
        for line in report["blocking"][:10]:
            print(f"  bloquant : {line}")
        fail(f"{len(report['blocking'])} problème(s) — le diagnostic doit être vert avant d'écrire")
    accepted = [d for d in report["documents"] if d.get("admissible")]
    additions = [d for d in accepted if d["identity"] == "ajout"]
    revisions = [d for d in accepted if d["identity"] in ("révision", "re-parse")]
    if revisions:
        for d in revisions:
            print(f"  {d['identity']} : {d['folder']} ({d['document_id']})")
        fail("cette tranche n'accepte que les ajouts. Une révision renumérote les chunk_id du "
             "document et peut détruire l'or des bancs — décision explicite, pas un import")
    if only:
        additions = [d for d in additions if d["sha256"].startswith(only) or d["folder"] == only]
        if not additions:
            fail(f"aucun ajout ne correspond à {only!r}")
    # une édition proche d'un document actif n'entre pas sans décision nommée
    undecided = [d for d in additions if d.get("needs_edition_decision")
                 and not any(d["sha256"].startswith(a) or d["folder"] == a for a in accept_editions)]
    if undecided:
        for d in undecided:
            print(f"  édition : {d['folder']} proche de {', '.join(d['needs_edition_decision'])} "
                  f"(containment {max(h['containment'] for h in d['near_duplicates'])})")
        fail("décision explicite exigée. Importer une édition proche d'un document actif met "
             "deux éditions du même ouvrage en concurrence dans le même espace vectoriel — "
             "c'est ce que duplicates-v1.json a dû corriger à la main.\n"
             "        Relance avec --accept-edition <sha256|dossier> pour chacune, "
             "ou écarte-les de la livraison.")
    return report, additions


# ------------------------------------------------------------------ 3. métadonnées

def extract_metadata_for(folders: dict[str, Path], no_llm: bool) -> dict[str, dict]:
    """Métadonnées bibliographiques des documents importés, par les règles du corpus.

    Réutilise ``rag/metadata/extract_metadata.py`` document par document. Le PDF source
    n'est pas livré (source S3 indisponible) ; S1 (nom de fichier), S2 (tampon arXiv) et S4
    (LLM sur la première page de ``blocks.jsonl``) le sont, et S4 couvre les 258 documents
    du corpus actuel — c'est la source qui porte le titre lisible.
    """
    import extract_metadata as em

    rows = {}
    for document_id, folder in folders.items():
        document = json.loads((folder / "document.json").read_text(encoding="utf-8"))
        filename_fields = em.parse_filename(document["filename"], set())
        first, dated = em.first_page_text(folder / "blocks.jsonl")
        stamp = em.parse_arxiv_stamp(first + "\n" + dated)
        pdf = em.parse_pdf(em.PAPERS / document["filename"])
        llm = None if no_llm else em.llm_extract(first, dated)
        row = em.consolidate(document, filename_fields, stamp, pdf, llm)
        _apply_overrides(row)
        if not no_llm and not row.get("publication_year"):
            deep = em.llm_year_deep(folder / "blocks.jsonl")
            if deep and deep.get("publication_year"):
                row["publication_year"] = deep["publication_year"]
                row["provenance"].update(year="llm_deep_pages", year_confidence="medium")
        rows[document_id] = row
    return rows


#: Sources de titre acceptées. Le fichier de métadonnées du corpus porte
#: ``title_not_from_filename: 258`` : sur les 258 documents, aucun titre ne vient d'un nom
#: de fichier — 136 du LLM sur la première page, 122 des métadonnées du PDF. C'est la
#: politique du corpus, et un import ne doit pas y déroger.
TITLE_SOURCES = {"llm_first_page", "llm_deep_pages", "pdf_embedded", "arxiv_stamp", "manual"}


def _apply_overrides(row: dict) -> None:
    """Corrections manuelles de ``rag/metadata/overrides.json``, même logique que l'amont.

    ``extract_metadata.py`` les applique dans son ``main()``, hors des fonctions qu'on
    réutilise ici : sans ce relais, un document corrigé à la main serait importé sans sa
    correction, et la garde le refuserait alors qu'une réponse existe.
    """
    path = ROOT / "rag" / "metadata" / "overrides.json"
    if not path.exists():
        return
    fix = json.loads(path.read_text(encoding="utf-8")).get(row["filename"])
    if not fix:
        return
    for field in ("title", "authors", "publication_year", "first_year", "venue", "doc_type"):
        if field in fix:
            row[field] = fix[field]
            key = "year" if field == "publication_year" else field
            if key in row["provenance"]:
                row["provenance"][key] = "manual"
    row["sources"]["manual"] = fix
    row["short_ref"] = em_short_ref(row)


def em_short_ref(row: dict) -> str:
    import extract_metadata as em

    return em.short_ref(row.get("authors") or [], row.get("publication_year"), row.get("title"))


def metadata_gate(rows: dict[str, dict], documents: dict[str, dict]) -> None:
    """Refuse d'embarquer un titre dérivé du nom de fichier — jamais de repli silencieux.

    Le cas réel qui a montré le trou : ssrn-3725454.pdf. Son titre amont est une phrase du
    corps du texte, et sans appel LLM le titre consolidé retombe sur ``ssrn 3725454``
    (provenance ``filename_stem``). Ni vide, ni égal au titre amont : une garde qui ne
    testait que ces deux-là laissait passer exactement ce qu'elle devait arrêter.
    """
    for document_id, row in rows.items():
        title = (row.get("title") or "").strip()
        source = (row.get("provenance") or {}).get("title")
        if not title:
            fail(f"{document_id} : aucun titre consolidé. L'embedding verrait le titre "
                 "d'export — on n'embarque pas dans ces conditions")
        if source not in TITLE_SOURCES:
            fail(f"{document_id} : titre {title[:60]!r} de provenance {source!r}, hors des "
                 f"sources acceptées {sorted(TITLE_SOURCES)}. Le corpus porte "
                 "title_not_from_filename: 258 — aucun de ses titres ne vient d'un nom de "
                 "fichier, et l'embedding voit ce titre-là.\n"
                 "        Relance sans --no-llm, ou renseigne rag/metadata/overrides.json.")
        if title == (documents[document_id].get("title") or "").strip():
            print(f"  ATTENTION {document_id} : titre consolidé identique au titre d'export "
                  f"({title[:60]!r}) — vérifier avant de conclure sur ce document")


def merge_metadata(rows: dict[str, dict]) -> list[str]:
    data = json.loads(METADATA.read_text(encoding="utf-8"))
    known = {d["document_id"] for d in data["documents"]}
    added = [document_id for document_id in rows if document_id not in known]
    # ajout en queue, jamais de tri : réordonner un fichier suivi de 585 Ko produirait un
    # diff illisible et une annulation qui ne restaure pas l'ordre d'origine.
    data["documents"].extend(rows[document_id] for document_id in added)
    data["summary"]["documents"] = len(data["documents"])
    METADATA.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    return added


# ------------------------------------------------------------------ 4. tableaux

def convert_tables_for(chunks: list[dict], titles: dict[str, str]) -> tuple[dict[str, str], dict]:
    """Chunks-tableaux en Markdown, par le convertisseur du corpus.

    Deux défauts corrigés le 9 septembre 2026 (§11.1 du rapport ``reprise-ingestion``), et
    ils se tenaient l'un l'autre. Un tableau trop grand pour un chunk est découpé ; seul le
    premier fragment porte la rangée d'en-tête, et la passe de conversion du corpus la
    recopie sur les suivants. À l'import, elle ne le faisait pas :

    1. ``ct.part_chains()`` lisait ``data/processed/ingested``, or **l'étape 4 précède la
       promotion de l'étape 8** : les chunks de la livraison en cours n'y sont pas encore.
       La chaîne était vide, ``first_part`` rendait le fragment lui-même, et l'en-tête
       valait toujours ``None``. On lui passe donc la chaîne de la livraison ;
    2. le payload était construit **sans ``parent_id``**. Même avec la chaîne, la garde de
       ``first_part`` — « le fragment précédent appartient-il au même tableau ? » — aurait
       comparé ``None`` à ``None`` et accepté n'importe quel voisin.

    Coût du défaut, mesuré sur le corpus servi : **1 066 fragments** de 100 documents
    importés, **4,08 %** des 26 120 chunks, servis sans le nom de leurs colonnes
    (``rag/tables/recenser_en_tetes.py``). Le témoin était sous les yeux depuis le début :
    ``fragments_with_inherited_header`` valait 0 à chaque import.
    """
    import convert_tables as ct

    class Point:                                    # convert_all attend des points Qdrant
        def __init__(self, payload):
            self.payload = payload

    tables = [Point({k: c.get(k) for k in ("chunk_id", "document_id", "text", "title_path",
                                           "content_type", "part", "chapter", "section",
                                           "parent_id", "previous_chunk_id")})
              for c in chunks if c.get("content_type") == "table"]
    if not tables:
        return {}, {"tables": 0}
    converted, stats = ct.convert_all(tables, previous=ct.part_chains(chunks))
    return converted, stats


def ecrire_overlay_tables(data: dict) -> None:
    """L'overlay des tableaux, écrit dans la forme du fichier VERSIONNÉ — compacte.

    Le défaut que cette fonction ferme (mesuré le 10 septembre 2026, lot D)
    ---------------------------------------------------------------------
    Le fichier commité est **compact, sur une seule ligne** : 5 224 037 octets, **0 saut de
    ligne**. ``apply_delivery`` le réécrivait ``indent=1`` : 5 235 902 octets, **5 932 sauts
    de ligne**. Contenu rigoureusement identique — 5 914 entrées, 0 écart après ``json.loads``
    — pour **5 933 lignes de diff git à chaque import**, sur un fichier de 5 Mo. Et
    l'annulation ne le remettait pas dans sa forme : le lot D a dû finir à la main, par un
    ``git checkout --``.

    Ce n'est pas de la cosmétique. Un historique où chaque import ajoute 5 933 lignes de bruit
    est un historique que personne ne relit — donc une revue qui ne voit plus rien passer. Le
    lot C fera six cents imports.

    La forme de référence n'est pas un choix de ce lot : c'est celle qu'écrit déjà
    ``rag/tables/appliquer_en_tetes.py``, et c'est celle du fichier versionné.
    Vérifié à l'octet près : ``json.dumps(data, ensure_ascii=False)`` reproduit exactement les
    5 224 037 octets commités.

    Les **deux** sites d'écriture passent par ici — l'import et l'annulation — pour qu'ils ne
    puissent plus diverger l'un de l'autre.
    """
    corpus_overlay.TABLES.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def merge_table_overlay(converted: dict[str, str]) -> list[str]:
    data = json.loads(corpus_overlay.TABLES.read_text(encoding="utf-8"))
    added = [cid for cid in converted if cid not in data["chunks"]]
    data["chunks"].update({cid: converted[cid] for cid in added})
    ecrire_overlay_tables(data)
    corpus_overlay.invalidate()
    return added


# ------------------------------------------------------------------ 5. embedding

def clean_title_of(document_id: str, export_title: str, metadata: dict) -> str:
    """La recette du corpus (``rag/titles/reembed_titles.py::clean_title``), à l'identique."""
    sys.path.insert(0, str(ROOT / "rag" / "titles"))
    import reembed_titles as rt

    return rt.clean_title(export_title, metadata.get(document_id))


def embed_delivery(delivery_id: str, rows: list[dict], titles: dict[str, str]) -> dict[str, list]:
    """Vecteurs des chunks importés, recette locale, reprise sur interruption.

    ``rows`` : {chunk_id, document_id, title_path, text} — texte **après** conversion des
    tableaux, comme l'ont vu les 18 636 chunks du corpus.
    """
    import numpy as np
    import quant_rag
    import reembed_titles as rt

    VECTOR_DIR.mkdir(parents=True, exist_ok=True)
    final = VECTOR_DIR / f"vectors-{delivery_id}.npz"
    partial = VECTOR_DIR / f"vectors-{delivery_id}.partial.npz"
    done: dict[str, list] = {}
    if final.exists():
        blob = np.load(final)
        print(f"  vecteurs déjà calculés : {len(blob['chunk_ids'])}")
        return dict(zip(blob["chunk_ids"].tolist(), blob["vectors"]))
    if partial.exists():
        blob = np.load(partial)
        done = dict(zip(blob["chunk_ids"].tolist(), blob["vectors"]))
        print(f"  reprise : {len(done)} chunks déjà embarqués")

    todo = [r for r in rows if r["chunk_id"] not in done]
    todo.sort(key=lambda r: len(r["text"]))
    model = quant_rag.embedder()
    started = time.perf_counter()
    for start in range(0, len(todo), EMBED_BLOCK):
        block = todo[start:start + EMBED_BLOCK]
        texts = [rt.embedding_text(titles[r["document_id"]], r["title_path"], r["text"]) for r in block]
        vectors = np.asarray(model.encode(texts, normalize_embeddings=True, show_progress_bar=False,
                                          batch_size=8), dtype=np.float32)
        for r, v in zip(block, vectors):
            done[r["chunk_id"]] = v
        np.savez(partial, chunk_ids=np.asarray(list(done)), vectors=np.vstack(list(done.values())))
        print(f"    {len(done)}/{len(rows)}  {(start + len(block)) / max(time.perf_counter() - started, 1e-6):.1f} chunks/s",
              flush=True)
    ordered = [r["chunk_id"] for r in rows]
    np.savez_compressed(final, chunk_ids=np.asarray(ordered),
                        vectors=np.vstack([done[c] for c in ordered]))
    partial.unlink(missing_ok=True)
    return {c: done[c] for c in ordered}


# ------------------------------------------------------------------ 6-7. journal, Qdrant

def next_point_id(quant_rag, client) -> int:
    """``max(id) + 1``. Jamais ``compte + i`` : la collection a 807 trous."""
    highest, offset = -1, None
    while True:
        points, offset = client.scroll(quant_rag.COLLECTION, limit=8192, offset=offset,
                                       with_payload=False, with_vectors=False)
        highest = max([highest] + [int(p.id) for p in points])
        if offset is None:
            break
    return highest + 1


def payload_for(chunk: dict, document: dict, metadata: dict, text: str, html: str | None) -> dict:
    """Le payload de l'amont (``append_ingested_incremental.payload``) plus les métadonnées
    consolidées que ``build_index.py`` injecte à chaque reconstruction."""
    value = {key: chunk.get(key) for key in ("chunk_id", "document_id", "page_start", "page_end",
                                             "part", "chapter", "section", "title_path",
                                             "parent_id", "content_type", "rag_eligible", "image_refs")}
    value.update(title=document.get("title"), authors=document.get("authors", []),
                 text=text, corpus_version="ingested-all")
    if html is not None:
        value["text_html"] = html
    record = metadata.get(chunk["document_id"]) or {}
    value.update({k: record.get(k) for k in
                  ("title", "authors", "publication_year", "first_year", "short_ref",
                   "venue", "doc_type", "filename") if k in record})
    return value


def write_journal(entry: dict) -> Path:
    JOURNAL_DIR.mkdir(parents=True, exist_ok=True)
    path = JOURNAL_DIR / f"{entry['delivery_id']}.json"
    path.write_text(json.dumps(entry, ensure_ascii=False, indent=1), encoding="utf-8")
    return path


# ------------------------------------------------------------------ 9. provenance

def delivery_descriptor(delivery: Path, delivery_id: str, received_at: str,
                        manifest: dict | None = None) -> dict:
    """Qui a produit cette livraison — l'amont, ou ce Mac.

    Ce module écrivait ``producer: "windows-pipeline"`` et ``transport: "archive"`` pour
    **toute** livraison, y compris celles que ``parse_local.py`` produit ici : le registre a
    donc attribué à l'amont, le 4 septembre 2026, trois papiers parsés sur cette machine. Le
    registre n'existe que pour porter la provenance ; s'y tromper le vide de son seul usage.

    Le signe est le fichier que seul le producteur local écrit, ``production.json`` — il
    porte la version de MinerU et le device, c'est-à-dire ce qui a décidé des identifiants.
    """
    production = None
    for candidate in (delivery / "production.json", delivery.parent / "production.json"):
        if candidate.exists():
            production = json.loads(candidate.read_text(encoding="utf-8"))
            break
    manifest = manifest or {}
    if production:
        producer = production.get("producer", {})
        return {"id": delivery_id, "transport": "locale", "received_at": received_at,
                "schema_version": manifest.get("schema_version"),
                "producer": f"macos-source-b (mineru {producer.get('mineru')}, "
                            f"{producer.get('device')})",
                "producer_commit": None}
    return {"id": delivery_id, "transport": "archive", "received_at": received_at,
            "schema_version": manifest.get("schema_version"),
            "producer": "windows-pipeline",
            "producer_commit": manifest.get("producer_commit")}


# ------------------------------------------------------------------ orchestration

class Chrono:
    """Le coût d'un import, étape par étape — pour que le rapport de lot n'ait pas à le deviner.

    Le journal d'un lot disait jusqu'ici combien de temps chaque ÉTAPE DU DRIVER avait pris,
    en mesurant de l'extérieur : un seul chiffre pour tout ``apply_delivery``. Or l'import
    contient dix gestes de coûts très différents — l'embedding domine sur un gros document,
    la reconstruction BM25 sur un petit —, et sans les séparer on ne peut pas décider quoi
    optimiser. C'est ce qui a fait attribuer 61 % d'un lot à ``build_graph``, qui coûte 1,3 s.
    """

    def __init__(self) -> None:
        self.etapes: dict[str, float] = {}
        self._depart = time.perf_counter()

    def top(self, nom: str) -> None:
        maintenant = time.perf_counter()
        self.etapes[nom] = round(maintenant - self._depart, 2)
        self._depart = maintenant

    def total(self) -> float:
        return round(sum(self.etapes.values()), 2)


def stats_llm() -> dict:
    """Appels et jetons du titrage — ``{}`` si le module n'a jamais été chargé.

    L'import est tardif et gardé : ``llm.py`` vit dans ``rag/benchmark`` et lit une clé.
    Le charger pour rendre un compte nul sur un import ``--no-llm`` serait payer une
    dépendance pour rien.
    """
    if "llm" not in sys.modules:
        return {"calls": 0, "cached": 0, "retries": 0, "prompt_tokens": 0, "completion_tokens": 0}
    return dict(sys.modules["llm"].stats())


def nettoyer_staging(etat: dict) -> None:
    """Le staging ne survit pas à un import qui échoue — sauf s'il est encore utile.

    Le défaut (mesuré le 10 septembre 2026, lot D)
    ----------------------------------------------
    Le staging ``data/processed/incoming/<id>/`` est créé à l'étape 2 et n'était effacé qu'à
    la toute fin de l'étape 10. Entre les deux, **aucun ``finally``** : tout ``fail()`` précoce
    — la garde de titre, par exemple, qui refuse un titre de provenance ``filename_stem`` — le
    laissait sur disque. ``batch_driver`` avait son propre ménage et rattrapait ; mais
    ``apply_delivery`` invoqué à la main, lui, ne rattrapait rien.

    La seule exception, et pourquoi elle existe
    --------------------------------------------
    Si le journal est en ``in-flight``, des points sont déjà entrés dans Qdrant et le staging
    **est une pièce de l'annulation** : ``_rollback`` le lit pour savoir quels dossiers
    promouvoir ou défaire. L'effacer là transformerait une mort en vol, qui est réparable, en
    état qu'on ne sait plus défaire. On le garde, et c'est ``--rollback`` qui l'emporte.
    """
    staged = etat.get("staged")
    if staged is None or not Path(staged).exists():
        return
    if (etat.get("journal") or {}).get("state") == "in-flight":
        return
    shutil.rmtree(staged)


def apply(delivery: Path, only: str | None, no_llm: bool, delivery_id: str | None,
          accept_editions: tuple[str, ...] = (), sans_bm25: bool = False) -> dict:
    import numpy as np
    from qdrant_client import models

    # Le gel, consulté AVANT le verrou de collection : un import refusé ne doit pas avoir
    # pris un verrou pour rien, et le message du gel doit être la première chose que
    # l'appelant lise. `verifier()` quitte en code 1 sur un gel non accordé.
    #
    # Jusqu'au 10 septembre 2026, `batch_driver --go` était le SEUL appelant de la garde :
    # `apply_delivery` appelé directement écrivait sous gel sans un mot, alors qu'il réécrit
    # le registre, `duplicates-v1.json` et la collection servie. La docstring de
    # `rag/gel_corpus.py` le disait déjà ; le code ne le faisait pas.
    gel_corpus.verifier("l'import d'un document au corpus")

    chrono = Chrono()
    llm_avant = stats_llm()
    # Le verrou APPLICATIF, pris avant tout. Il ferme la fenêtre entre le
    # ``next_point_id()`` (une lecture) et l'upsert (une écriture) : deux importeurs
    # simultanés y liraient le même max(id) et le second effacerait les points du
    # premier, sans erreur des deux côtés. Le verrou exclusif du stockage embarqué
    # l'interdit aujourd'hui — mais il DISPARAÎT sur un serveur, et c'est la condition
    # bloquante de la bascule (§5 de docs/BASCULE-QDRANT-SERVEUR.md).
    # ``etat`` est rempli par ``_apply`` au fur et à mesure : le chemin du staging dès qu'il
    # existe, puis le dict du journal — le MÊME objet, si bien que son ``state`` est toujours
    # à jour ici. C'est ce qui permet au ``finally`` de savoir s'il a le droit de nettoyer.
    etat: dict = {}
    try:
        with verrou_collection.tenu(f"apply_delivery {delivery_id or 'sans-id'}"):
            return _apply(delivery, only, no_llm, delivery_id, accept_editions, sans_bm25,
                          chrono, llm_avant, etat)
    finally:
        nettoyer_staging(etat)


def _apply(delivery: Path, only: str | None, no_llm: bool, delivery_id: str | None,
           accept_editions: tuple[str, ...], sans_bm25: bool,
           chrono: "Chrono", llm_avant: dict, etat: dict | None = None) -> dict:
    import numpy as np
    from qdrant_client import models

    quant_rag, client = take_lock()
    print(f"0. verrou Qdrant obtenu ({client.count(quant_rag.COLLECTION, exact=True).count} points)")

    report, additions = plan(delivery, only, accept_editions)
    if not additions:
        print("1. plan : rien à ajouter")
        return {"status": "NOOP"}
    delivery_id = delivery_id or datetime.now(timezone.utc).strftime("%Y-%m-%d-%H%M%S")
    print(f"1. plan : {len(additions)} ajout(s) — livraison {delivery_id}")

    # 2. stage — hors du corpus servi
    staged = INCOMING / delivery_id
    if staged.exists():
        shutil.rmtree(staged)
    staged.mkdir(parents=True)
    # Déclaré à l'appelant dès qu'il existe : c'est son ``finally`` qui l'emportera si la
    # suite échoue avant que le journal ne passe en ``in-flight``. Voir ``nettoyer_staging``.
    if etat is not None:
        etat["staged"] = staged
    base = delivery / "processed" if (delivery / "processed").is_dir() else delivery
    folders, documents = {}, {}
    for d in additions:
        target = staged / d["folder"]
        shutil.copytree(base / d["folder"], target)
        document = json.loads((target / "document.json").read_text(encoding="utf-8"))
        folders[document["document_id"]] = target
        documents[document["document_id"]] = document
    print(f"2. stage : {staged.relative_to(ROOT)}")
    chrono.top("stage")

    # 2 bis. LE JOURNAL, ouvert ici et non à l'étape 6 — §11.2 du rapport reprise-ingestion.
    # Les étapes 3 (métadonnées) et 4 (overlay des tableaux) écrivent dans le corpus servi.
    # Tant que le journal n'était ouvert qu'après elles, une mort entre les deux laissait des
    # lignes orphelines qu'AUCUNE annulation ne couvrait — et une ligne de métadonnées
    # orpheline change titles_digest, donc la signature, donc le nom de l'index. Le corpus
    # n'était pas corrompu, mais il n'était plus à l'état d'avant, et apply_delivery
    # concluait « rien n'a été écrit dans Qdrant » : vrai, et incomplet.
    journal = {
        "delivery_id": delivery_id, "state": "pre-flight", "started_at": now(),
        "source": str(delivery), "staged": str(staged.relative_to(ROOT)),
        "documents": [{"sha256": documents[d]["sha256"], "document_id": d,
                       "folder": folders[d].name} for d in documents],
        "added_metadata": [], "added_table_chunks": [], "promoted": False,
        "note": ("ouvert AVANT la première écriture dans le corpus servi : à ce stade rien "
                 "n'est encore écrit hors du staging, et tout ce qui le sera est enregistré "
                 "ici au fur et à mesure pour que --rollback puisse le défaire"),
    }
    # Le MÊME objet que l'appelant regardera dans son ``finally`` : son ``state`` y est donc
    # toujours à jour, et c'est lui qui décide si le staging est encore une pièce utile.
    if etat is not None:
        etat["journal"] = journal
    write_journal(journal)

    # 3. métadonnées — refus si absentes
    rows = extract_metadata_for(folders, no_llm)
    metadata_gate(rows, documents)
    added_metadata = merge_metadata(rows)
    journal["added_metadata"] = added_metadata
    write_journal(journal)
    chrono.top("metadonnees")
    for document_id, row in rows.items():
        print(f"3. métadonnées : {document_id} → {row.get('short_ref')!r} "
              f"({row.get('publication_year')}, {row.get('provenance', {}).get('title', '?')})")

    # 4. tableaux — avant l'embedding, le texte indexé doit être le texte final
    all_chunks, chunk_document = [], {}
    for document_id, folder in folders.items():
        for chunk in diag.read_jsonl(folder / "chunks.jsonl"):
            all_chunks.append(chunk)
            chunk_document[chunk["chunk_id"]] = document_id
    metadata_now = {r["document_id"]: r for r in
                    json.loads(METADATA.read_text(encoding="utf-8"))["documents"]}
    export_titles = {d: documents[d].get("title") or "" for d in documents}
    converted, table_stats = convert_tables_for(
        all_chunks, {d: metadata_now[d].get("title") or export_titles[d] for d in documents})
    added_tables = merge_table_overlay(converted)
    journal["added_table_chunks"] = added_tables
    journal["tables"] = table_stats
    write_journal(journal)
    print(f"4. tableaux : {table_stats} — {len(added_tables)} ajoutés à l'overlay")
    # ``fragments_with_inherited_header`` remonte jusqu'au résultat et au journal : c'est le
    # témoin du défaut §11.1, et il valait 0 à chaque import depuis l'origine. Un compteur
    # qui n'apparaît que dans une ligne de sortie n'est pas un témoin, c'est une impression.

    # 5. embedding — sur le texte final, avec le titre consolidé
    titles = {d: clean_title_of(d, export_titles[d], metadata_now) for d in documents}
    for d, t in titles.items():
        print(f"5. titre embarqué : {d} → {t[:70]!r}")
    eligible = [c for c in all_chunks if c.get("rag_eligible") is True]
    rows_to_embed = [{"chunk_id": c["chunk_id"], "document_id": c["document_id"],
                      "title_path": c.get("title_path"),
                      "text": converted.get(c["chunk_id"], c.get("text") or "")}
                     for c in eligible]
    vectors = embed_delivery(delivery_id, rows_to_embed, titles)
    print(f"5. embedding : {len(vectors)} vecteurs")
    chrono.top("embedding")

    # 6. journal in-flight — AVANT le premier upsert
    first_id = next_point_id(quant_rag, client)
    ordered = [c for c in eligible]
    journal.update({
        "state": "in-flight",
        "watermark": first_id,
        "point_ids": {"first": first_id, "last": first_id + len(ordered) - 1, "count": len(ordered)},
        "chunk_ids": [c["chunk_id"] for c in ordered],
        "note": ("écrit avant le premier upsert : si l'import meurt ici, les points à "
                 "supprimer sont ceux dont l'identifiant est >= watermark. Le journal, lui, "
                 "existe depuis le staging — voir l'étape 2 bis"),
    })
    path = write_journal(journal)
    print(f"6. journal in-flight : {path.relative_to(ROOT)} (watermark {first_id})")

    # 7. Qdrant — à max(id) + 1
    started = time.perf_counter()
    points = []
    for offset, chunk in enumerate(ordered):
        document_id = chunk["document_id"]
        text = converted.get(chunk["chunk_id"], chunk.get("text") or "")
        html = chunk.get("text") if chunk["chunk_id"] in converted else None
        points.append(models.PointStruct(
            id=first_id + offset,
            vector=np.asarray(vectors[chunk["chunk_id"]], dtype=np.float32).tolist(),
            payload=payload_for(chunk, documents[document_id], metadata_now, text, html)))
    for start in range(0, len(points), 256):
        client.upsert(quant_rag.COLLECTION, wait=True, points=points[start:start + 256])
    after = client.count(quant_rag.COLLECTION, exact=True).count
    print(f"7. Qdrant : {len(points)} points écrits en {time.perf_counter() - started:.1f} s "
          f"({first_id}..{first_id + len(points) - 1}) — collection {after}")
    chrono.top("qdrant")

    # 8. promotion
    for document_id, folder in folders.items():
        destination = INGESTED / folder.name
        if destination.exists():
            fail(f"{destination} existe déjà — promotion refusée")
        shutil.copytree(folder, destination)
    journal.update(promoted=True, state="applied", applied_at=now(), points_after=after)
    write_journal(journal)
    print(f"8. promotion : {len(folders)} dossier(s) dans data/processed/ingested/")
    chrono.top("promotion")

    # 9. registre — il enregistre les plages de points, donc de quoi annuler
    registry_data = reg.build()
    descriptor = delivery_descriptor(delivery, delivery_id, journal["started_at"],
                                     report["manifest"])
    for entry in registry_data["documents"]:
        if entry["document_id"] in folders:
            entry["delivery"] = dict(descriptor)
    registry_data["deliveries"] = [d for d in registry_data["deliveries"] if d.get("id") != delivery_id]
    registry_data["deliveries"].append(dict(descriptor, documents=len(folders)))
    reg.PATH.write_text(json.dumps(registry_data, ensure_ascii=False, indent=1), encoding="utf-8")
    # Le relais que lit le graphe. ``registry.py --build`` le régénère ; cet import écrivait
    # le registre sans lui, et ``extract_entities.py`` aurait donc reconstruit le graphe sur
    # l'état d'avant — sans un mot, avec pour seule trace un compte de documents plus bas
    # que celui du corpus servi. C'est le défaut du 4 septembre (« le graphe était aveugle
    # aux imports ») revenu par une autre porte : le fichier existait, mais il était périmé.
    rows = reg.write_imported_rows()
    corpus_overlay.invalidate()
    print(f"9. registre : {registry_data['totals']['documents']} documents, "
          f"{rows} lignes dans imported-rows.jsonl, signature {corpus_overlay.signature()}")
    chrono.top("registre")

    # 10. BM25 — en dernier, la signature est définitive
    #
    # ``sans_bm25`` sert le mode « fin de lot » de ``batch_driver`` : reconstruire un index de
    # 50,8 Mo après CHAQUE document coûte 7,7 s et 50,8 Mo écrits, dont un seul exemplaire
    # survit au lot. La dette est **inscrite dans le journal de livraison**, et non seulement
    # tue : un import qui laisse l'index lexical en retard sans le dire est exactement la
    # famille de défaut que ce dépôt s'interdit. ``bm25_en_retard`` est ce que la reprise et
    # ``verifier_installation.py`` relisent pour savoir qu'il reste un geste à faire.
    # §11.5 — ``document_metadata()`` est mémoïsée par ``lru_cache`` et c'est d'ELLE que
    # ``rebuild_bm25`` tire les titres indexés (``quant_rag.rebuild_bm25``). Le module purgeait
    # ``corpus_overlay`` et ``bm25`` mais pas celle-là : un second import dans le MÊME
    # processus, ou le ``rollback`` qui suit un ``apply``, reconstruisait donc l'index
    # lexical sur une table de titres périmée. Sans effet tant que le driver lance
    # apply_delivery en sous-processus — mais « un index juste sous un nom qui ment »
    # par la porte d'à côté, et les tests hermétiques passent bien, eux, par un seul
    # processus.
    quant_rag.document_metadata.cache_clear()
    quant_rag.bm25.cache_clear()
    if sans_bm25:
        journal["bm25_en_retard"] = True
        write_journal(journal)
        index = None
        print("10. BM25 : DIFFÉRÉ (--sans-bm25) — l'index lexical est en retard sur la "
              "collection jusqu'à la reconstruction de fin de lot")
    else:
        journal.pop("bm25_en_retard", None)
        write_journal(journal)
        index = quant_rag.rebuild_bm25()
        print(f"10. BM25 : {quant_rag.bm25_path().name} — {index.doc_count} records")

    chrono.top("bm25")
    llm_apres = stats_llm()
    cout = {
        "secondes": chrono.etapes,
        "secondes_total": chrono.total(),
        "llm": {cle: llm_apres.get(cle, 0) - llm_avant.get(cle, 0)
                for cle in ("calls", "cached", "retries", "prompt_tokens", "completion_tokens")},
        "modele_titrage": None if no_llm else os.environ.get("QUANT_RAG_METADATA_LLM"),
        "octets_ecrits": {
            "index_bm25": (quant_rag.bm25_path().stat().st_size
                           if not sans_bm25 and quant_rag.bm25_path().exists() else 0),
            "vecteurs_npz": (VECTOR_DIR / f"vectors-{delivery_id}.npz").stat().st_size
                            if (VECTOR_DIR / f"vectors-{delivery_id}.npz").exists() else 0,
            "registre": reg.PATH.stat().st_size if reg.PATH.exists() else 0,
        },
        "points": len(points),
    }
    journal["cout"] = cout
    write_journal(journal)
    print(f"coût : {chrono.total()} s au total — " +
          " · ".join(f"{k} {v} s" for k, v in chrono.etapes.items()) +
          f" · {cout['llm']['calls']} appel(s) LLM, "
          f"{cout['llm']['prompt_tokens'] + cout['llm']['completion_tokens']} jetons")

    shutil.rmtree(staged)
    return {"status": "COMPLETED", "delivery_id": delivery_id,
            "documents": len(folders), "points": len(points),
            "point_ids": journal["point_ids"], "collection": after,
            "corpus_signature": corpus_overlay.signature(),
            "bm25": None if sans_bm25 else quant_rag.bm25_path().name,
            "bm25_en_retard": bool(sans_bm25),
            "cout": cout,
            "tables": table_stats,
            "graph": "à relancer : .venv-gliner/bin/python rag/graph/extract_entities.py "
                     "(reprise automatique, seuls les nouveaux chunks), puis build_graph"}


# ------------------------------------------------------------------ annulation

def rollback(delivery_id: str) -> dict:
    """Défait un import : points, dossiers, métadonnées, tableaux, registre, BM25.

    Sous le même verrou que l'import : une annulation retire des points et réécrit le
    registre, donc elle ne peut pas cohabiter avec un import qui calcule ``max(id)+1``.

    ⚠ **Le gel n'est PAS consulté ici, et ce n'est pas un oubli.** Un gel non accordé
    signifie que la signature vivante s'est écartée de la signature gelée — c'est-à-dire
    que quelque chose est entré au corpus qui n'aurait pas dû. Bloquer l'annulation
    derrière cette garde serait un piège parfait : elle refuserait précisément le geste
    qui remet la signature d'aplomb, et le seul moyen d'en sortir serait de lever le gel,
    donc de perdre la trace de ce qu'on protégeait. `apply()` consulte le gel ; `rollback()`
    ne le consultera jamais.
    """
    with verrou_collection.tenu(f"rollback {delivery_id}"):
        return _rollback(delivery_id)


def verifier_la_plage(quant_rag, client, ids: list[int], journal: dict) -> None:
    """Chaque point de la plage appartient-il bien à cette livraison ? Sinon, **refus**.

    §11.4 du rapport ``reprise-ingestion``. L'annulation supprimait une plage d'identifiants
    par ``PointIdsList`` sans jamais regarder ce qu'elle contenait. La seule protection était
    l'ordonnancement de ``batch_driver`` — c'est-à-dire une convention d'appelant, pas une
    garde. Une plage décalée, ou un journal rejoué après une reconstruction qui a
    réattribué les identifiants, aurait effacé les points d'un AUTRE document ;
    ``points_removed`` serait resté cohérent et le rapport aurait dit ``ROLLED_BACK``.

    Un identifiant **absent** n'est pas une anomalie : il a déjà été retiré, il n'y a rien à
    supprimer. Ce qui est refusé, c'est un point **présent** qui appartient à quelqu'un
    d'autre.
    """
    attendus = {document["document_id"] for document in journal.get("documents") or []}
    if not attendus:
        fail(f"le journal de {journal.get('delivery_id')!r} ne nomme aucun document : "
             "impossible de vérifier à qui appartiennent les points de la plage")
    intrus = []
    for depart in range(0, len(ids), 4096):
        tranche = ids[depart:depart + 4096]
        for point in client.retrieve(quant_rag.COLLECTION, ids=tranche,
                                     with_payload=["chunk_id", "document_id"]):
            document_id = (point.payload or {}).get("document_id")
            if document_id not in attendus:
                intrus.append((point.id, document_id, (point.payload or {}).get("chunk_id")))
    if intrus:
        apercu = ", ".join(f"{pid} → {doc}" for pid, doc, _ in intrus[:5])
        fail(f"la plage {ids[0]}..{ids[-1]} contient {len(intrus)} point(s) qui "
             f"n'appartiennent pas à cette livraison : {apercu}"
             f"{' …' if len(intrus) > 5 else ''}. Les supprimer effacerait un autre "
             f"document. Documents attendus : {sorted(attendus)}")


def _rollback(delivery_id: str) -> dict:
    from qdrant_client import models

    path = JOURNAL_DIR / f"{delivery_id}.json"
    if not path.exists():
        fail(f"aucun journal pour {delivery_id!r}")
    journal = json.loads(path.read_text(encoding="utf-8"))
    if journal["state"] == "rolled-back":
        print("déjà annulé"); return {"status": "NOOP"}
    quant_rag, client = take_lock()
    # Les chemins de l'index BM25 de la signature qu'on s'apprête à ABANDONNER, relevés ICI :
    # plus bas, l'overlay a déjà été invalidé et la signature est revenue à celle d'avant
    # l'import, si bien que ``bm25_path()`` désignerait l'index vivant. C'est le piège dans
    # lequel la première version de cette correction est tombée.
    abandonnes = (quant_rag.bm25_path(), quant_rag.bm25_manifest_path())
    before = client.count(quant_rag.COLLECTION, exact=True).count

    # Un journal ``pre-flight`` n'a pas de plage : l'import est mort avant l'étape 6, donc
    # avant tout upsert. Il reste à défaire les métadonnées, l'overlay des tableaux et le
    # staging — c'est exactement ce que le §11.2 laissait orphelin.
    plage = journal.get("point_ids")
    if plage:
        ids = list(range(plage["first"], plage["last"] + 1))
        verifier_la_plage(quant_rag, client, ids, journal)
        client.delete(quant_rag.COLLECTION, wait=True,
                      points_selector=models.PointIdsList(points=ids))
    else:
        print(f"  aucun point à retirer : le journal est en état {journal['state']!r}")
    after = client.count(quant_rag.COLLECTION, exact=True).count

    for document in journal["documents"]:
        target = INGESTED / document["folder"]
        if target.exists():
            shutil.rmtree(target)
    staged = ROOT / journal["staged"]
    if staged.exists():
        shutil.rmtree(staged)

    if journal["added_metadata"]:
        data = json.loads(METADATA.read_text(encoding="utf-8"))
        data["documents"] = [d for d in data["documents"]
                             if d["document_id"] not in set(journal["added_metadata"])]
        data["summary"]["documents"] = len(data["documents"])
        METADATA.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    if journal["added_table_chunks"]:
        data = json.loads(corpus_overlay.TABLES.read_text(encoding="utf-8"))
        for chunk_id in journal["added_table_chunks"]:
            data["chunks"].pop(chunk_id, None)
        ecrire_overlay_tables(data)
    corpus_overlay.invalidate()

    registry_data = reg.build()
    registry_data["deliveries"] = [d for d in registry_data["deliveries"] if d["id"] != delivery_id]
    reg.PATH.write_text(json.dumps(registry_data, ensure_ascii=False, indent=1), encoding="utf-8")
    # même raison qu'à l'étape 9 de l'import, dans l'autre sens : sans cette ligne, le relais
    # garderait les chunks qu'on vient d'annuler et le graphe les ressusciterait.
    reg.write_imported_rows()
    corpus_overlay.invalidate()

    vectors = VECTOR_DIR / f"vectors-{delivery_id}.npz"
    vectors.unlink(missing_ok=True)
    # Le point de reprise de l'embarquement. Il est écrit à chaque bloc (``:327``) et effacé
    # au succès (``:333``) : un import mort pendant l'embarquement le laisse donc sur disque,
    # et c'est voulu — c'est lui qui permet de reprendre là où on s'est arrêté.
    #
    # Après une ANNULATION, il n'y a plus rien à reprendre, et le garder est un piège : un
    # import ultérieur du même identifiant repartirait de vecteurs calculés pour un état du
    # corpus qui n'existe plus. Il part avec le reste.
    (VECTOR_DIR / f"vectors-{delivery_id}.partial.npz").unlink(missing_ok=True)
    # §11.5 — ``document_metadata()`` est mémoïsée par ``lru_cache`` et c'est d'ELLE que
    # ``rebuild_bm25`` tire les titres indexés (``quant_rag.rebuild_bm25``). Le module purgeait
    # ``corpus_overlay`` et ``bm25`` mais pas celle-là : un second import dans le MÊME
    # processus, ou le ``rollback`` qui suit un ``apply``, reconstruisait donc l'index
    # lexical sur une table de titres périmée. Sans effet tant que le driver lance
    # apply_delivery en sous-processus — mais « un index juste sous un nom qui ment »
    # par la porte d'à côté, et les tests hermétiques passent bien, eux, par un seul
    # processus.
    quant_rag.document_metadata.cache_clear()
    quant_rag.bm25.cache_clear()
    index = quant_rag.rebuild_bm25()
    # ``rebuild_bm25`` écrit un fichier nommé par la signature courante et n'efface jamais le
    # précédent (``quant_rag.rebuild_bm25``). Un import + son annulation laissaient donc DEUX
    # index morts, ≈ 50 Mo chacun, portant une signature qui n'existe plus. Le lot C fera six
    # cents imports : ≈ 30 Go sur un disque qui était à 90 % le 10 septembre 2026.
    #
    # La garde qui compte est la comparaison de chemins : si l'annulation n'a pas changé la
    # signature (rien n'était entré, ou l'import ne l'avait pas déplacée), ``abandonnes`` EST
    # l'index vivant, et on n'y touche pas.
    vivants = {quant_rag.bm25_path(), quant_rag.bm25_manifest_path()}
    retires = []
    for chemin in abandonnes:
        if chemin not in vivants and chemin.exists():
            chemin.unlink()
            retires.append(chemin.name)
    if retires:
        print(f"11. index abandonné supprimé : {', '.join(retires)}")
    journal.update(state="rolled-back", rolled_back_at=now(),
                   points_removed=before - after, points_after=after,
                   index_abandonnes=retires)
    write_journal(journal)
    return {"status": "ROLLED_BACK", "delivery_id": delivery_id,
            "points_removed": before - after, "collection": after,
            "corpus_signature": corpus_overlay.signature(), "bm25_records": index.doc_count}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("delivery", nargs="?", type=Path)
    parser.add_argument("--plan", action="store_true", help="diagnostic seul, aucune écriture")
    parser.add_argument("--apply", action="store_true", help="importe (ajouts seuls)")
    parser.add_argument("--rollback", metavar="ID", help="défait un import à partir de son journal")
    parser.add_argument("--document", help="n'importer qu'un document (sha256 ou nom de dossier)")
    parser.add_argument("--id", dest="delivery_id", help="identifiant de la livraison")
    parser.add_argument("--accept-edition", action="append", default=[], metavar="SHA256",
                        help="accepte explicitement une édition proche d'un document actif")
    parser.add_argument("--no-llm", action="store_true",
                        help="métadonnées par les règles seules (aucun appel réseau)")
    parser.add_argument("--sans-bm25", action="store_true",
                        help="ne pas reconstruire l'index BM25 : le lot le fera une fois à la "
                             "fin. La dette est inscrite dans le journal de livraison "
                             "(bm25_en_retard) — l'index lexical est en retard jusque-là")
    args = parser.parse_args()

    # ``VerrouTenu`` n'était rattrapée nulle part : ni ici, ni aux deux ``with
    # verrou_collection.tenu(...)``. Le refus le mieux écrit du dépôt — il nomme le tenant,
    # son pid, depuis quand, et le chemin du fichier à supprimer si le processus est mort —
    # arrivait donc à l'utilisateur sous une **trace Python**, là où tout le reste dit
    # ``REFUS :``. Le code de sortie était déjà 1 et rien n'était écrit : c'était la
    # lisibilité qui était en cause, et elle seule. Relevé par le lot D, corrigé ici.
    #
    # Le message n'est pas réécrit : il est bon. Il est seulement passé à ``fail()``, qui le
    # préfixe et quitte proprement.
    try:
        if args.rollback:
            print(json.dumps(rollback(args.rollback), ensure_ascii=False, indent=1))
            return
        if not args.delivery:
            parser.error("donner un répertoire de livraison, ou --rollback <id>")
        if args.plan or not args.apply:
            report, additions = plan(args.delivery, args.document, tuple(args.accept_edition))
            print(diag.summarise(report))
            print(f"\n{len(additions)} ajout(s) importable(s). Rien n'a été écrit "
                  f"(--apply pour importer).")
            return
        print(json.dumps(apply(args.delivery, args.document, args.no_llm, args.delivery_id,
                               tuple(args.accept_edition), args.sans_bm25),
                         ensure_ascii=False, indent=1))
    except verrou_collection.VerrouTenu as refus:
        fail(str(refus))


if __name__ == "__main__":
    main()
