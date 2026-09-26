"""Retirer du corpus servi un document déjà importé — doublon ou édition surclassée.

`duplicates-v1.json` décrivait des retraits décidés **avant** que le corpus ne soit servi :
les documents écartés le 2 septembre 2026 n'avaient jamais eu de point dans la collection.
Retirer un document déjà servi demande davantage, et dans cet ordre : ses points quittent la
collection, l'overlay le déclare retiré, le registre le passe en `removed-duplicate`, le
relais des imports cesse de le porter, l'index BM25 est refait, le graphe est rebâti. Sauter
une seule de ces étapes laisse un corpus qui se contredit — et `registry --verify` le dit aussitôt : « dans
la collection mais pas actif au registre ».

Le contrôle qui précède tout : **aucun passage-cible des bancs ne doit se trouver dans le
document retiré**. C'est la même question que `gold_chunks_at_risk` à l'import, posée dans
l'autre sens. Le script refuse de retirer un document qui porte de l'or, quel que soit
l'argument passé.

    .venv/bin/python rag/ingestion/retire_document.py doc-xxx --raison "…" --garde doc-yyy
    .venv/bin/python rag/ingestion/retire_document.py doc-xxx --raison "…" --garde doc-yyy --appliquer
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "rag"))
sys.path.insert(0, str(HERE))
import corpus_overlay  # noqa: E402
import gel_corpus  # noqa: E402  — consulté sur le seul chemin qui écrit (--appliquer)
import registry as reg  # noqa: E402

BANCS = ("questions-v1.jsonl", "questions-v2.jsonl", "questions-v3.jsonl")
INGESTED = ROOT / "data" / "processed" / "ingested"
METADATA = ROOT / "rag" / "metadata" / "documents-metadata-v1.json"


def dossier_de(document_id: str) -> Path | None:
    for dossier in sorted(INGESTED.iterdir()):
        fiche = dossier / "document.json"
        if fiche.exists() and json.loads(fiche.read_text(encoding="utf-8"))["document_id"] == document_id:
            return dossier
    return None


def livraison_de(document_id: str) -> str | None:
    """Le document est-il *produit* sans être encore importé ? Son identifiant est alors dans
    le ``production.json`` d'une livraison — c'est le cas d'un échange, où le remplaçant
    attend que le remplacé sorte."""
    for production in sorted((HERE / "source-b" / "livraisons").glob("*/production.json")):
        donnees = json.loads(production.read_text(encoding="utf-8"))
        if any(d["document_id"] == document_id for d in donnees.get("documents", [])):
            return production.parent.name
    return None


def chunks_de(dossier: Path) -> set[str]:
    return {json.loads(l)["chunk_id"] for l in (dossier / "chunks.jsonl").open(encoding="utf-8") if l.strip()}


def or_en_peril(document_id: str, chunks: set[str]) -> list[str]:
    """Questions des bancs dont la cible tombe dans ce document. Vide = retrait sans perte."""
    perdues = []
    for nom in BANCS:
        fichier = ROOT / "rag" / "benchmark" / nom
        if not fichier.exists():
            continue
        for ligne in fichier.read_text(encoding="utf-8").splitlines():
            if not ligne.strip():
                continue
            question = json.loads(ligne)
            vise = (question.get("target_chunk") in chunks
                    or document_id in (question.get("gold_documents") or [])
                    or bool(chunks & set(question.get("gold_chunks") or [])))
            if vise:
                perdues.append(f"{nom.split('-')[1].split('.')[0]}/{question['qid']}")
    return perdues


def main() -> int:
    analyseur = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    analyseur.add_argument("document_id")
    analyseur.add_argument("--raison", required=True, help="pourquoi, avec les mesures")
    analyseur.add_argument("--garde", required=True, metavar="DOCUMENT_ID",
                           help="le document conservé à sa place")
    analyseur.add_argument("--garde-en-attente", action="store_true",
                           help="le document conservé n'est pas encore importé : son identifiant "
                                "est lu dans une livraison produite mais pas encore appliquée. "
                                "Nécessaire pour un échange, où le remplaçant ne peut pas entrer "
                                "avant que le remplacé ne sorte — le diagnostic le refuserait "
                                "comme édition de celui qu'on s'apprête justement à retirer.")
    analyseur.add_argument("--appliquer", action="store_true",
                           help="sans ce drapeau, rien n'est écrit : on ne fait que le constat")
    arguments = analyseur.parse_args()

    dossier = dossier_de(arguments.document_id)
    if dossier is None:
        sys.exit(f"REFUS : {arguments.document_id} n'a pas de fichiers canoniques")
    dossier_garde = dossier_de(arguments.garde)
    if dossier_garde is None:
        livraison = livraison_de(arguments.garde)
        if livraison is None:
            sys.exit(f"REFUS : le document conservé {arguments.garde} n'est ni au corpus ni dans "
                     "une livraison produite. Un retrait doit nommer ce qui prend la place.")
        if not arguments.garde_en_attente:
            sys.exit(f"REFUS : {arguments.garde} est produit ({livraison}) mais pas encore importé. "
                     "Si c'est voulu — un échange — passe --garde-en-attente ; le retrait le "
                     "consignera comme en attente et il faudra l'importer aussitôt après.")
    if arguments.document_id in corpus_overlay.removed_documents():
        sys.exit(f"REFUS : {arguments.document_id} est déjà retiré")

    chunks = chunks_de(dossier)
    perdues = or_en_peril(arguments.document_id, chunks)
    fiche = json.loads((dossier / "document.json").read_text(encoding="utf-8"))
    constat = {"document_id": arguments.document_id, "filename": fiche.get("filename"),
               "chunks": len(chunks), "gold_chunks_at_risk": len(perdues), "questions": perdues[:10],
               "garde": arguments.garde}
    if perdues:
        print(json.dumps(constat, ensure_ascii=False, indent=1))
        sys.exit(f"REFUS : {len(perdues)} question(s) des bancs visent ce document. Retirer un "
                 "document qui porte de l'or déplace la mesure au lieu de corriger le corpus.")

    import quant_rag
    from qdrant_client import models

    client = quant_rag.client()
    filtre = models.Filter(must=[models.FieldCondition(
        key="document_id", match=models.MatchValue(value=arguments.document_id))])
    constat["points_avant"] = client.count(quant_rag.COLLECTION, exact=True, count_filter=filtre).count
    constat["collection_avant"] = client.count(quant_rag.COLLECTION, exact=True).count
    constat["signature_avant"] = corpus_overlay.signature()
    if not arguments.appliquer:
        constat["status"] = "CONSTAT"
        print(json.dumps(constat, ensure_ascii=False, indent=1))
        return 0

    # Le gel, consulté ici et pas plus haut : un constat n'écrit rien et doit rester
    # utilisable sous gel, exactement comme `batch_driver --dry-run`. Le retrait, lui,
    # change la collection, le registre et l'overlay : il change donc la signature, ce
    # qu'un chantier mesuré sous gel ne peut pas absorber. `verifier()` quitte en code 1.
    gel_corpus.verifier("le retrait d'un document du corpus")

    # 1. les points quittent la collection — par filtre sur document_id, pas par plage
    #    d'identifiants : un document réimporté a pu en recevoir de plusieurs tentatives.
    client.delete(quant_rag.COLLECTION, wait=True, points_selector=models.FilterSelector(filter=filtre))
    constat["points_apres"] = client.count(quant_rag.COLLECTION, exact=True, count_filter=filtre).count
    constat["collection_apres"] = client.count(quant_rag.COLLECTION, exact=True).count
    if constat["points_apres"]:
        sys.exit(f"ÉCHEC : {constat['points_apres']} point(s) subsistent après suppression")

    # 2. l'overlay le déclare retiré — c'est lui que lisent le registre, le relais et le graphe
    fichier = corpus_overlay.DUPLICATES
    donnees = json.loads(fichier.read_text(encoding="utf-8"))
    if dossier_garde is not None:
        garde = json.loads((dossier_garde / "document.json").read_text(encoding="utf-8"))
        conserve = {"document_id": arguments.garde, "filename": garde.get("filename")}
    else:
        conserve = {"document_id": arguments.garde, "filename": None,
                    "pending": f"produit dans {livraison_de(arguments.garde)}, à importer aussitôt"}
    donnees["remove"].append({
        "document_id": arguments.document_id, "filename": fiche.get("filename"),
        "short_ref": None, "chunks": len(chunks), "reason": arguments.raison,
        "kept": conserve,
        "removed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "points_removed": constat["points_avant"]})
    fichier.write_text(json.dumps(donnees, ensure_ascii=False, indent=1), encoding="utf-8")
    corpus_overlay.invalidate()

    # 3. registre, relais, BM25 — dans cet ordre, comme à l'étape 9 d'un import
    registre = reg.build()
    reg.PATH.write_text(json.dumps(registre, ensure_ascii=False, indent=1), encoding="utf-8")
    reg.write_imported_rows()
    corpus_overlay.invalidate()
    constat["registre"] = {"documents": registre["totals"]["documents"],
                           "par_statut": registre["totals"]["by_status"]}
    index = quant_rag.rebuild_bm25()
    constat["bm25"] = getattr(index, "doc_count", None) or len(getattr(index, "documents", []) or [])

    # 4. le graphe, sans quoi le retrait est incomplet et muet. ``results.jsonl`` s'écrit en
    #    ajout : les extractions du document retiré y restent, et ``build_graph.py`` les
    #    filtre depuis l'overlay — mais il faut le relancer pour que le fichier servi
    #    change. Sans cette étape, la collection, le registre, le relais et BM25 ont cessé
    #    de servir le document, et le graphe continue.
    import subprocess

    build = subprocess.run([str(ROOT / ".venv-gliner" / "bin" / "python"),
                            str(ROOT / "rag" / "graph" / "build_graph.py")],
                           capture_output=True, text=True)
    rapport_graphe = ROOT / "data" / "graph" / "graph-lite-report.json"
    if build.returncode != 0:
        constat["graphe"] = f"ÉCHEC : {build.stderr.strip()[-300:]}"
    elif rapport_graphe.exists():
        donnees_graphe = json.loads(rapport_graphe.read_text(encoding="utf-8"))
        constat["graphe"] = {"chunks": donnees_graphe.get("sample_chunks"),
                             "documents": donnees_graphe.get("documents"),
                             "retires_ecartes": donnees_graphe.get("removed_documents_skipped")}

    constat["signature_apres"] = corpus_overlay.signature()
    constat["status"] = "COMPLETED"
    print(json.dumps(constat, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
