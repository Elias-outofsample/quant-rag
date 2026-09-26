"""Les invariants du corpus que personne d'autre ne vérifie.

Deux, pour l'instant : les lecteurs voient-ils tout le corpus, et chaque document parsé
ici a-t-il gardé son PDF ?

--- 1. Les lecteurs voient-ils **tout** le corpus servi ?

`rows.jsonl` et `data/qdrant-export/payloads.jsonl` décrivent l'export initial : ils
ignorent tout document entré par une livraison. Trois outils lisaient encore l'un ou
l'autre et raisonnaient donc sur un corpus périmé sans le dire — un scan de doublons
aveugle à 62 documents ne rapporte pas « je n'ai pas regardé », il rapporte « rien à
signaler ».

Ce contrôle demande à chaque lecteur l'ensemble des ``document_id`` qu'il voit et le
compare aux documents **actifs** du registre. Il échoue si l'un d'eux en manque un.

--- 2. Chaque document parsé ici a-t-il son PDF durable ?

``data/papers/<filename>`` est la source S3 d'``extract_metadata`` et l'exemplaire suivi en
LFS. Les documents de l'amont n'en ont pas, et n'en ont jamais eu — ils sont arrivés parsés.
Mais un document produit par ``parse_local.py`` doit avoir le sien : sans lui, un
re-titrage ou une reprise de métadonnées travaillerait à l'aveugle. Le 5 septembre 2026,
deux documents réimportés après avoir été écartés se sont retrouvés sans PDF durable —
leur parse ayant été réutilisé, ``place_pdf`` n'a pas rejoué.

    .venv/bin/python rag/benchmark/check_readers.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
REGISTRY = ROOT / "rag" / "ingestion" / "registry-v1.json"


def actifs() -> set[str]:
    donnees = json.loads(REGISTRY.read_text(encoding="utf-8"))
    return {d["document_id"] for d in donnees["documents"] if d.get("status") == "active"}


def vus_par_reembed_titles() -> set[str]:
    sys.path.insert(0, str(ROOT / "rag" / "titles"))
    import reembed_titles

    lignes, _ = reembed_titles.load_rows()
    return {r["document_id"] for r in lignes}


def vus_par_scan_duplicates() -> set[str]:
    sys.path.insert(0, str(ROOT / "rag" / "metadata"))
    import scan_duplicates

    return set(scan_duplicates.corpus_chunks())


def vus_par_eval_hybrid() -> set[str]:
    sys.path.insert(0, str(HERE))
    import eval_hybrid

    return set(eval_hybrid.document_of_chunk().values())


LECTEURS = (
    ("rag/titles/reembed_titles.py", vus_par_reembed_titles),
    ("rag/metadata/scan_duplicates.py", vus_par_scan_duplicates),
    ("rag/benchmark/eval_hybrid.py", vus_par_eval_hybrid),
)


def papiers_manquants() -> list[tuple[str, str]]:
    """Documents parsés ici (transport « locale ») dont le PDF durable a disparu."""
    donnees = json.loads(REGISTRY.read_text(encoding="utf-8"))
    papiers = ROOT / "data" / "papers"
    manquants = []
    for document in donnees["documents"]:
        if document.get("status") != "active" or not document.get("filename"):
            continue
        if (document.get("delivery") or {}).get("transport") != "locale":
            continue
        if not (papiers / document["filename"]).exists():
            manquants.append((document["document_id"], document["filename"]))
    return manquants


def main() -> int:
    attendus = actifs()
    print(f"registre : {len(attendus)} documents actifs\n")
    print(f"{'lecteur':<38}{'documents vus':>15}{'actifs manquants':>19}  verdict")
    aveugles = []
    for nom, lire in LECTEURS:
        try:
            vus = lire()
        except Exception as erreur:                       # un lecteur qui ne se lit pas est un échec
            print(f"{nom:<38}{'—':>15}{'—':>19}  ERREUR : {type(erreur).__name__}: {erreur}")
            aveugles.append(nom)
            continue
        manquants = attendus - vus
        verdict = "OK" if not manquants else f"AVEUGLE à {len(manquants)} document(s)"
        print(f"{nom:<38}{len(vus):>15}{len(manquants):>19}  {verdict}")
        if manquants:
            aveugles.append(nom)
            exemples = sorted(manquants)[:3]
            print(f"{'':<38}exemples : {', '.join(exemples)}")
    print()
    manquants = papiers_manquants()
    print(f"PDF durables : {len(manquants)} document(s) parsé(s) ici sans fichier dans data/papers/")
    for document_id, nom in manquants[:5]:
        print(f"    {document_id}  {nom[:70]}")
    print()
    if aveugles or manquants:
        if aveugles:
            print(f"ÉCHEC : {len(aveugles)} lecteur(s) ne voient pas tout le corpus servi.")
        if manquants:
            print(f"ÉCHEC : {len(manquants)} document(s) parsé(s) ici ont perdu leur PDF durable.")
        return 1
    print("OK — les lecteurs voient les mêmes documents que le registre, et chaque document "
          "parsé ici a gardé son PDF.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
