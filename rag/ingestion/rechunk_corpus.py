"""Corpus candidat du chantier de représentation — écrit **à côté**, jamais dessus.

Phase 1 du plan (``docs/REPRESENTATION-CONCEPTION.md`` §6, pré-enregistrement §6 étape 1).
Ce module produit le corpus **C1** — la correction commune, telle que
``src/parsing/canonical_chunker`` la porte depuis le 6 septembre 2026 — dans un dossier
séparé, et **ne touche à rien de ce qui est servi** : ni la collection Qdrant, ni l'index
BM25, ni le registre, ni les overlays, ni les caches du banc. Qdrant n'est jamais ouvert.

    data/processed/candidat-<sig>/
        doc-…/document.json  chunks.jsonl  parents.jsonl  offsets.jsonl
        tables-markdown-v1.json      overlay des tableaux, TRANSPORTÉ (voir plus bas)
        rows.jsonl                   la vue corpus du banc, pour les 418 documents actifs
        retrieval-text.jsonl         ce qui sera plongé, en C1 et en C2, avec ses sha256
        manifeste.json               signature candidate, comptes, contrôles

La garde qui compte : la signature vivante
------------------------------------------
``corpus_overlay.signature()`` est relue **avant et après** chaque exécution et doit valoir
``5530cba145``, la signature gelée. Si elle bouge, quelque chose a écrit en production et le
module s'arrête, code 1. C'est la forme exécutable du **contrôle Z** pour cette phase : ici
l'aller-retour est trivial parce que rien n'est appliqué — et c'est exactement ce qu'il faut
vérifier, pas supposer.

L'overlay des tableaux est **transporté**, pas recalculé — et c'est mesuré
--------------------------------------------------------------------------
``tables-markdown-v1.json`` est indexé par ``chunk_id`` : au re-découpage il devient
intégralement pendant, et 5 742 chunks-tableaux repartiraient en HTML brut. Il faut donc le
reconstruire. Deux voies, et la première a été écartée **après mesure** :

**Recalculer** — rejouer ``convert_tables.convert_all`` sur les nouveaux chunks — ne reproduit
pas l'overlay actuel : sur les 6 182 chunks-tableaux des documents actifs, **1 066 rendent un
Markdown différent et 274 sont absents** de l'overlay versionné. La raison est dans le dépôt :
l'overlay a **deux producteurs**. ``convert_tables`` fait hériter l'en-tête d'un fragment de
tableau en remontant ``previous_chunk_id`` *tant que le parent est le même* ;
``apply_delivery.convert_tables_for`` construit ses points **sans ``parent_id``**, de sorte que
la comparaison ``payload.get("parent_id") != payload.get("parent_id")`` est toujours fausse et
que la chaîne remonte au-delà de la frontière de parent. Les documents de l'amont et ceux
entrés par livraison n'ont donc pas été convertis par la même règle. Choisir une règle unique
aujourd'hui changerait le texte servi de ~1 340 chunks **pour une raison étrangère au
chantier**, et contaminerait le Δ.

**Transporter** est exact, et la mesure le montre : la correction commune ne touche pas au
texte des tableaux — un tableau est toujours vidé seul (``flush(); current.append(b); flush()``)
— si bien que la **séquence ordonnée** des textes de tableaux est identique, document par
document, entre la référence et C1 : **6 182 = 6 182, 0 document divergent**. L'overlay est
donc reporté par ``(document_id, rang du tableau dans le document)``, ce qui est **non
ambigu** (un seul texte de tableau est dupliqué dans tout le corpus, et le rang le sépare).

Le texte servi du candidat est ainsi **identique à l'octet** à celui d'aujourd'hui partout où
le découpage n'a pas bougé. La dette des deux producteurs est **signalée, pas corrigée** :
la corriger est une seconde variable, et ce chantier n'en admet qu'une.

    .venv/bin/python rag/ingestion/rechunk_corpus.py --verifier
    .venv/bin/python rag/ingestion/rechunk_corpus.py --construire
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "rag"))
sys.path.insert(0, str(ROOT / "rag" / "titles"))

import corpus_overlay  # noqa: E402
import gel_corpus  # noqa: E402
from src.parsing.canonical_chunker import build_parents, make_chunks  # noqa: E402
from src.parsing.document_text import (  # noqa: E402
    provenance_du_chunk, sha256_texte, texte_canonique)
from src.parsing.models import CanonicalBlock, CanonicalDocument  # noqa: E402

INGESTED = ROOT / "data" / "processed" / "ingested"
PROCESSED = ROOT / "data" / "processed"
REGISTRY = HERE / "registry-v1.json"
#: La signature gelée, lue au gel lui-même. Elle était codée en dur à ``5530cba145`` ; le
#: corpus est passé à ``e1bdf36e2e`` le 9 septembre 2026 et le module refusait alors de
#: tourner en annonçant que le gel avait bougé. Une constante qui nomme un état vivant se
#: périme ; la lire là où elle est tenue est la seule forme qui ne se périme pas.
SIGNATURE_GELEE = gel_corpus.lire()["signature"]
#: Recette de plongement en vigueur (``rag/titles/reembed_titles.embedding_text``).
#: C2 insère une ligne ``Page: n`` entre ``Path:`` et le corps — c'est TOUTE la différence.
MAX_LENGTH = 1024


# ------------------------------------------------------------------ lecture


def _charger(dossier: Path):
    d = json.loads((dossier / "document.json").read_text(encoding="utf-8"))
    doc = CanonicalDocument(
        document_id=d["document_id"], source_path=d.get("source_path", ""),
        filename=d.get("filename", ""), sha256=d.get("sha256", ""), title=d.get("title"),
        subtitle=d.get("subtitle"), authors=d.get("authors") or [], editors=d.get("editors") or [],
        publication_year=d.get("publication_year"), edition=d.get("edition"),
        publisher=d.get("publisher"), page_count=d.get("page_count", 0),
        parser_backend=d.get("parser_backend", ""), parser_version=d.get("parser_version", ""),
        metadata=d.get("metadata") or {})
    blocs = [CanonicalBlock(
        block_id=b["block_id"], document_id=b["document_id"], page_idx=b["page_idx"],
        block_type=b["block_type"], text=b["text"], bbox=b.get("bbox"),
        heading_level=b.get("heading_level"), parent_heading=b.get("parent_heading"),
        content_type=b["content_type"], raw_text=b.get("raw_text"), part=b.get("part"),
        chapter=b.get("chapter"), section=b.get("section"), subsection=b.get("subsection"),
        image_refs=b.get("image_refs") or [], metadata=b.get("metadata") or {})
        for b in (json.loads(l) for l in (dossier / "blocks.jsonl").open(encoding="utf-8") if l.strip())]
    stockes = [json.loads(l) for l in (dossier / "chunks.jsonl").open(encoding="utf-8") if l.strip()]
    return doc, blocs, stockes, d


def _registre() -> list[dict]:
    return json.loads(REGISTRY.read_text(encoding="utf-8"))["documents"]


def titres_plonges() -> tuple[dict[str, str], dict]:
    """Le titre que chaque document porte **dans le vecteur servi aujourd'hui**.

    Il a deux sources, et il faut les deux — c'est une dette du dépôt, signalée ici et **non
    corrigée**, parce que la corriger serait une seconde variable :

    - ``titles/titles-clean-v1.json`` couvre **317** des 418 documents actifs. C'est le
      résultat figé de ``reembed_titles.clean_title`` au 3 septembre, et c'est avec lui que
      les vecteurs de ces documents ont été calculés ;
    - les **101** autres sont entrés par livraison, et ``apply_delivery`` (ligne 440) a
      recalculé leur titre à la volée par ``clean_title(titre d'export, métadonnées)``.

    Reproduire l'état servi impose donc de suivre la même bifurcation. Utiliser
    ``clean_title`` partout serait plus propre *et* changerait le titre plongé de tout
    document dont les métadonnées ont été corrigées depuis le 3 septembre — le rapport
    retourné compte ces cas, pour que la dette soit chiffrée et non seulement nommée.
    """
    import reembed_titles as rt

    fige = {d: e["clean_title"]
            for d, e in json.loads(corpus_overlay.TITLES.read_text(encoding="utf-8"))["documents"].items()}
    meta = {r["document_id"]: r
            for r in json.loads(corpus_overlay.METADATA.read_text(encoding="utf-8"))["documents"]}
    titres: dict[str, str] = {}
    stats = collections.Counter()
    derives: list[str] = []
    for entree in _registre():
        document_id = entree["document_id"]
        export = entree.get("title") or ""
        recalcule = rt.clean_title(export, meta.get(document_id))
        if document_id in fige:
            titres[document_id] = fige[document_id]
            stats["fige"] += 1
            if fige[document_id] != recalcule:
                stats["fige_mais_derive"] += 1
                derives.append(document_id)
        else:
            titres[document_id] = recalcule
            stats["recalcule"] += 1
    return titres, {"stats": dict(stats), "documents_derives": derives[:20]}


# ------------------------------------------------------------------ la signature, réimplémentée


def _digest_registre(entrees: list[tuple]) -> str:
    """Copie de ``corpus_overlay.registry_digest`` sur des entrées explicites."""
    rows = sorted([list(e) for e in entrees])
    return hashlib.sha256(json.dumps(rows, separators=(",", ":")).encode()).hexdigest()[:16]


def _digest_titres(paires: list[tuple[str, str]]) -> str:
    """Copie de ``corpus_overlay.titles_digest`` sur des paires explicites."""
    digest = hashlib.sha256()
    for document_id, titre in sorted(paires):
        digest.update(document_id.encode()); digest.update(b"\x00")
        digest.update((titre or "").encode()); digest.update(b"\x01")
    return digest.hexdigest()[:12]


def _signature(duplicates: Path, tables: Path, registre: str, titres: str) -> str:
    """Copie de ``corpus_overlay.signature`` sur des chemins explicites.

    Elle est **prouvée** équivalente à l'originale par ``--verifier``, qui l'applique aux
    chemins vivants et exige ``5530cba145``. Une copie non prouvée serait un second nom pour
    une chose qu'on croirait la même.
    """
    digest = hashlib.sha256()
    for path in (duplicates, tables):
        digest.update(path.name.encode())
        digest.update(corpus_overlay._content_digest(path).encode())
    digest.update(b"registry"); digest.update(registre.encode())
    digest.update(b"titles"); digest.update(titres.encode())
    return digest.hexdigest()[:10]


def _signature_vivante_reimplementee() -> str:
    entrees = [(d.get("sha256"), d.get("document_id"), d.get("status"),
                (d.get("files") or {}).get("chunks.jsonl")) for d in _registre()]
    retires = set(corpus_overlay.removed_documents())
    meta = json.loads(corpus_overlay.METADATA.read_text(encoding="utf-8"))["documents"]
    paires = [(r["document_id"], r.get("title") or "") for r in meta if r["document_id"] not in retires]
    return _signature(corpus_overlay.DUPLICATES, corpus_overlay.TABLES,
                      _digest_registre(entrees), _digest_titres(paires))


# ------------------------------------------------------------------ recette de plongement


def retrieval_text(titre: str, title_path: str | None, page: int | None, texte: str,
                   avec_page: bool) -> str:
    """La chaîne réellement plongée. ``avec_page`` distingue C2 de C1, et rien d'autre."""
    chemin = title_path or titre
    ligne_page = f"Page: {page}\n" if avec_page and page is not None else ""
    return f"Document: {titre}\nPath: {chemin}\n{ligne_page}\n{texte.strip()}"


# ------------------------------------------------------------------ transport de l'overlay


def transporter_overlay(actifs: list[dict]) -> tuple[dict[str, str], dict]:
    """Reporte ``tables-markdown-v1.json`` sur les nouveaux ``chunk_id``, par rang.

    Échoue bruyamment si la séquence ordonnée des textes de tableaux d'un document a changé :
    le transport ne serait plus exact, et recalculer changerait le texte servi pour une raison
    étrangère au chantier (voir la docstring du module).
    """
    ancien = corpus_overlay.text_overrides()
    nouveau: dict[str, str] = {}
    stats = collections.Counter()
    divergents: list[str] = []
    for entree in actifs:
        dossier = INGESTED / entree["folder"]
        doc, blocs, stockes, _ = _charger(dossier)
        candidats, _ = make_chunks(doc, blocs)
        ref = [c for c in stockes if c.get("content_type") == "table"]
        cand = [c for c in candidats if c.content_type == "table"]
        if [c["text"] for c in ref] != [c.text for c in cand]:
            divergents.append(entree["folder"])
            continue
        for ancien_chunk, nouveau_chunk in zip(ref, cand):
            stats["tableaux"] += 1
            valeur = ancien.get(ancien_chunk["chunk_id"])
            if valeur is None:
                stats["sans_conversion"] += 1
                continue
            nouveau[nouveau_chunk.chunk_id] = valeur
            stats["transportes"] += 1
    return nouveau, {"stats": dict(stats), "documents_divergents": divergents}


# ------------------------------------------------------------------ construction


def construire(sortie_racine: Path = PROCESSED) -> dict:
    avant = corpus_overlay.signature()
    if avant != SIGNATURE_GELEE:
        sys.exit(f"signature vivante {avant} != {SIGNATURE_GELEE} — le gel a bougé, on n'écrit rien")

    registre = _registre()
    actifs = [d for d in registre if d.get("status") == "active"]
    titres, rapport_titres = titres_plonges()

    print(f"transport de l'overlay des tableaux ({len(actifs)} documents actifs)…")
    overlay, rapport_overlay = transporter_overlay(actifs)
    if rapport_overlay["documents_divergents"]:
        sys.exit(f"séquence de tableaux divergente sur "
                 f"{len(rapport_overlay['documents_divergents'])} documents — transport impossible")
    print(f"  {rapport_overlay['stats']}")

    # Le dossier est nommé par la signature, donc calculé en deux temps : on écrit d'abord
    # dans un dossier provisoire, on calcule la signature, on renomme. Un dossier candidat
    # dont le nom mentirait sur son contenu serait exactement le défaut que le nommage par
    # signature existe pour empêcher.
    provisoire = sortie_racine / "candidat-en-cours"
    if provisoire.exists():
        shutil.rmtree(provisoire)
    provisoire.mkdir(parents=True)
    (provisoire / "tables-markdown-v1.json").write_text(
        json.dumps({"chunks": overlay}, ensure_ascii=False, indent=1), encoding="utf-8")

    entrees_registre: list[tuple] = []
    comptes = collections.Counter()
    rows: list[str] = []
    retrieval: list[str] = []
    #: Ce que `gold_ancrage --appliquer` attend : chunk_id, document_id, texte SERVI.
    servis: list[str] = []
    par_statut = {d["folder"]: d for d in registre}

    print(f"re-découpage des {len(registre)} documents…")
    for entree in registre:
        dossier = INGESTED / entree["folder"]
        doc, blocs, _stockes, document_json = _charger(dossier)
        chunks, _roles = make_chunks(doc, blocs)
        parents = build_parents(doc, blocs, chunks)
        texte, spans = texte_canonique(blocs)
        doc_sha = sha256_texte(texte)

        cible = provisoire / entree["folder"]
        cible.mkdir(parents=True, exist_ok=True)
        (cible / "document.json").write_text(
            json.dumps({**document_json, "doc_text_sha256": doc_sha}, ensure_ascii=False, indent=2),
            encoding="utf-8")
        (cible / "chunks.jsonl").write_text(
            "".join(json.dumps(c.as_dict(), ensure_ascii=False) + "\n" for c in chunks), encoding="utf-8")
        (cible / "parents.jsonl").write_text(
            "".join(json.dumps(p, ensure_ascii=False) + "\n" for p in parents), encoding="utf-8")
        (cible / "offsets.jsonl").write_text(
            "".join(json.dumps(provenance_du_chunk(c, spans, doc_sha).as_dict(), ensure_ascii=False) + "\n"
                    for c in chunks), encoding="utf-8")

        comptes["documents"] += 1
        comptes["chunks"] += len(chunks)
        entrees_registre.append((entree.get("sha256"), entree.get("document_id"),
                                 entree.get("status"),
                                 hashlib.sha256((cible / "chunks.jsonl").read_bytes()).hexdigest()))

        if par_statut[entree["folder"]].get("status") != "active":
            continue
        titre = titres[doc.document_id]
        for c in chunks:
            if not c.rag_eligible:
                continue
            comptes["chunks_actifs_eligibles"] += 1
            servi = overlay.get(c.chunk_id, c.text)
            rows.append(json.dumps({"document": document_json, "chunk": c.as_dict()}, ensure_ascii=False))
            servis.append(json.dumps({"chunk_id": c.chunk_id, "document_id": c.document_id,
                                      "text": servi}, ensure_ascii=False))
            c1 = retrieval_text(titre, c.title_path, c.page_start, servi, avec_page=False)
            c2 = retrieval_text(titre, c.title_path, c.page_start, servi, avec_page=True)
            retrieval.append(json.dumps({
                "chunk_id": c.chunk_id, "document_id": c.document_id, "page_start": c.page_start,
                "c1_sha256": hashlib.sha256(c1.encode()).hexdigest(),
                "c2_sha256": hashlib.sha256(c2.encode()).hexdigest(),
                "c1_caracteres": len(c1), "c2_caracteres": len(c2)}, ensure_ascii=False))

    (provisoire / "rows.jsonl").write_text("\n".join(rows) + "\n", encoding="utf-8")
    (provisoire / "retrieval-text.jsonl").write_text("\n".join(retrieval) + "\n", encoding="utf-8")
    (provisoire / "chunks-servis.jsonl").write_text("\n".join(servis) + "\n", encoding="utf-8")

    retires = set(corpus_overlay.removed_documents())
    meta = json.loads(corpus_overlay.METADATA.read_text(encoding="utf-8"))["documents"]
    paires = [(r["document_id"], r.get("title") or "") for r in meta if r["document_id"] not in retires]
    signature = _signature(corpus_overlay.DUPLICATES, provisoire / "tables-markdown-v1.json",
                           _digest_registre(entrees_registre), _digest_titres(paires))

    manifeste = {
        "signature_candidate": signature,
        "signature_reference": SIGNATURE_GELEE,
        "bras": "C1 — correction commune (flush avant état, chemin tronqué à la section)",
        "chunker": "src/parsing/canonical_chunker.make_chunks",
        "construit_le": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "comptes": dict(comptes),
        "overlay_tableaux": rapport_overlay,
        "titres_plonges": rapport_titres,
        "note": ("Corpus candidat, hors production. La collection servie, l'index BM25, le "
                 "registre et les overlays vivants ne sont pas touchés."),
    }
    (provisoire / "manifeste.json").write_text(
        json.dumps(manifeste, ensure_ascii=False, indent=1), encoding="utf-8")
    # Le corpus candidat est gitignoré — 362 Mo régénérables en 54 s. Son manifeste, lui, est
    # suivi : c'est lui qui dit quelle signature a été construite, avec quel chunker et quels
    # comptes. Même règle que l'index BM25, dont seul le manifeste est versionné.
    (HERE / f"manifeste-candidat-{signature}.json").write_text(
        json.dumps(manifeste, ensure_ascii=False, indent=1), encoding="utf-8")

    final = sortie_racine / f"candidat-{signature}"
    if final.exists():
        shutil.rmtree(final)
    provisoire.rename(final)

    apres = corpus_overlay.signature()
    if apres != SIGNATURE_GELEE:
        sys.exit(f"CONTRÔLE Z ÉCHOUÉ : signature vivante {avant} -> {apres}. "
                 f"Quelque chose a écrit en production.")
    manifeste["contro_z"] = {"avant": avant, "apres": apres, "inchangee": True}
    print(f"\ncorpus candidat écrit : {final.relative_to(ROOT)}")
    print(json.dumps(manifeste, ensure_ascii=False, indent=1))
    return manifeste


# ------------------------------------------------------------------ vérification


def verifier() -> int:
    """Les contrôles préalables. Rien n'est écrit ; sortie 1 au premier échec."""
    echecs = 0

    vivante = corpus_overlay.signature()
    ok = vivante == SIGNATURE_GELEE
    print(f"  signature vivante                      {vivante}  {'OK' if ok else 'ÉCHEC'}")
    echecs += not ok

    recalculee = _signature_vivante_reimplementee()
    ok = recalculee == SIGNATURE_GELEE
    print(f"  signature réimplémentée                {recalculee}  {'OK' if ok else 'ÉCHEC'}"
          f"   (la copie est équivalente à corpus_overlay.signature)")
    echecs += not ok

    titres, rapport_titres = titres_plonges()
    manquants = [d["document_id"] for d in _registre() if d["document_id"] not in titres]
    ok = not manquants
    print(f"  titre plongé connu pour chaque document  {rapport_titres['stats']}  {'OK' if ok else 'ÉCHEC'}")
    if rapport_titres["stats"].get("fige_mais_derive"):
        print(f"    DETTE signalée, non corrigée : {rapport_titres['stats']['fige_mais_derive']} documents "
              f"dont le titre figé diffère de clean_title(métadonnées d'aujourd'hui)")
    echecs += not ok

    actifs = [d for d in _registre() if d.get("status") == "active"]
    _, rapport = transporter_overlay(actifs)
    divergents = rapport["documents_divergents"]
    ok = not divergents
    print(f"  séquence des tableaux préservée        {rapport['stats']}  {'OK' if ok else 'ÉCHEC'}")
    if divergents:
        print(f"    documents divergents : {divergents[:5]}")
    echecs += not ok

    apres = corpus_overlay.signature()
    ok = apres == vivante
    print(f"  signature après vérification           {apres}  {'OK' if ok else 'ÉCHEC'}")
    echecs += not ok
    return 1 if echecs else 0


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--verifier", action="store_true", help="contrôles préalables, rien n'est écrit")
    p.add_argument("--construire", action="store_true", help="écrit data/processed/candidat-<sig>/")
    a = p.parse_args()
    if a.verifier:
        code = verifier()
        if code or not a.construire:
            sys.exit(code)
    if a.construire:
        construire()
        return
    if not a.verifier:
        p.print_help()


if __name__ == "__main__":
    main()
