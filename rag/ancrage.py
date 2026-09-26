"""L'ancrage des passages servis dans le texte canonique de leur document.

Un passage servi porte aujourd'hui trois choses qu'un lecteur peut vérifier — les auteurs,
l'année, une page — et **rien qui permette de retrouver le texte lui-même**. La page est celle
d'un lecteur, arrondie au chunk ; elle ne dit pas où, dans le document, la phrase citée se
trouve, et elle ne permet pas de prouver qu'elle s'y trouve.

Ce module calcule, hors ligne, ce qui manque : pour chacun des 26 120 passages servis, la
**liste d'intervalles** qu'il occupe dans le texte canonique de son document, la granularité
atteinte, et le ``sha256`` de ce texte canonique. Ces trois champs entrent dans le payload
Qdrant (``build_index.py``), sont servis par ``mcp_server`` et sont ce sur quoi
``verify_citation`` s'appuie.

**Le calcul lui-même n'est pas ici** : il est dans ``src/parsing/document_text.py``, qui
définit le texte canonique et l'ancrage d'un chunk, et que ce module se contente d'appliquer
au corpus servi. Ce fichier résout une seule difficulté propre à ``rag/`` : le dossier d'un
document sur le disque **n'est pas** son ``document_id``. Les 421 dossiers de
``data/processed/ingested/`` sont nommés ``doc-<sha256 du fichier source>[:16]`` tandis que le
``document_id`` est un autre identifiant ; seul ``rag/ingestion/registry-v1.json`` fait la
correspondance, par son champ ``folder``. Un code qui tenterait ``ingested/<document_id>/``
échouerait silencieusement sur les 421.

    .venv/bin/python rag/ancrage.py --construire
    .venv/bin/python rag/ancrage.py --controle
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "rag"))

from parsing import document_text  # noqa: E402

REGISTRY = ROOT / "rag" / "ingestion" / "registry-v1.json"
INGESTED = ROOT / "data" / "processed" / "ingested"
ANCRAGE = ROOT / "rag" / "metadata" / "ancrage-v1.json"

#: Version du format du fichier d'ancrage. Elle voyage avec lui : un offset dont on ne sait pas
#: de quel format il vient n'est pas opposable.
FORMAT = "ancrage-v1"


def dossiers_par_document() -> dict[str, str]:
    """``document_id`` → nom du dossier sur le disque, lu du registre et de lui seul."""
    registre = json.loads(REGISTRY.read_text(encoding="utf-8"))
    return {entree["document_id"]: entree["folder"]
            for entree in registre.get("documents", [])
            if entree.get("folder") and entree.get("status") == "active"}


def _lire_jsonl(chemin: Path) -> list[dict]:
    return [json.loads(ligne) for ligne in chemin.read_text(encoding="utf-8").splitlines() if ligne.strip()]


def ancrer_document(dossier: Path) -> tuple[str, dict[str, dict]]:
    """``(sha256 du texte canonique, {chunk_id: provenance})`` pour un document du disque."""
    blocs = _lire_jsonl(dossier / "blocks.jsonl")
    texte, spans = document_text.texte_canonique(blocs)
    sha = document_text.sha256_texte(texte)
    provenances = {}
    for chunk in _lire_jsonl(dossier / "chunks.jsonl"):
        provenance = document_text.provenance_du_chunk(chunk, spans, sha)
        provenances[chunk["chunk_id"]] = {
            "g": provenance.granularite,
            "i": [intervalle.as_list() for intervalle in provenance.intervalles],
            "inconnus": provenance.blocs_inconnus,
        }
    return sha, provenances


def construire(verbeux: bool = True) -> dict:
    """Ancre tous les documents actifs. Aucun appel réseau, aucun LLM, aucun accès à Qdrant."""
    dossiers = dossiers_par_document()
    documents: dict[str, dict] = {}
    manquants, echecs = [], []
    debut = time.perf_counter()
    for numero, (document_id, dossier) in enumerate(sorted(dossiers.items()), 1):
        chemin = INGESTED / dossier
        if not (chemin / "blocks.jsonl").exists() or not (chemin / "chunks.jsonl").exists():
            manquants.append(document_id)
            continue
        try:
            sha, provenances = ancrer_document(chemin)
        except (OSError, json.JSONDecodeError, KeyError) as erreur:
            echecs.append({"document_id": document_id, "erreur": f"{type(erreur).__name__}: {erreur}"})
            continue
        documents[document_id] = {"sha256": sha, "dossier": dossier, "chunks": provenances}
        if verbeux and numero % 50 == 0:
            print(f"    {numero}/{len(dossiers)} documents", flush=True)
    secondes = time.perf_counter() - debut

    total = sum(len(d["chunks"]) for d in documents.values())
    exacts = sum(1 for d in documents.values() for p in d["chunks"].values()
                 if p["g"] == document_text.GRANULARITE_EXACTE)
    inconnus = sum(1 for d in documents.values() for p in d["chunks"].values() if p["inconnus"])
    return {
        "format": FORMAT,
        "documents_ancres": len(documents),
        "documents_sans_fichiers": manquants,
        "documents_en_echec": echecs,
        "chunks_ancres": total,
        "chunks_exacts": exacts,
        "chunks_au_bloc": total - exacts,
        "chunks_avec_bloc_inconnu": inconnus,
        "secondes": round(secondes, 1),
        "documents": documents,
    }


_CHARGE: dict = {}


def invalider() -> None:
    """À appeler après avoir réécrit le fichier d'ancrage dans le même processus."""
    _CHARGE.clear()


def charger() -> dict:
    """Le fichier d'ancrage, ou ``{}`` s'il n'a pas été construit — jamais une exception.

    **Mémorisé**, et il faut l'être : 10,7 Mo de JSON relus et reparsés à chaque appel
    coûtaient 200 ms par vérification de citation, soit tout le budget de latence de l'outil
    pour lire un fichier qui ne change pas. Contrairement à ``corpus_overlay.signature()``, qui
    est délibérément relue à chaque fois parce qu'un écrivain peut la changer sous les pieds du
    lecteur, l'ancrage ne bouge qu'à une reconstruction — et ``invalider()`` est là pour ce
    cas-là.

    ``build_index.py`` doit pouvoir tourner sans lui : une collection sans ancrage est
    dégradée, pas cassée, et le refus de démarrer serait une régression pour les autres
    chantiers qui reconstruisent la collection sans se soucier du contrat de sortie.
    """
    if _CHARGE:
        return _CHARGE
    if not ANCRAGE.exists():
        return {}
    try:
        _CHARGE.update(json.loads(ANCRAGE.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError):
        return {}
    return _CHARGE


def fusionner(intervalles: list) -> list[list[int]]:
    """Les intervalles réduits à des couples ``[début, fin]``, les contigus réunis.

    Deux blocs voisins dans le texte canonique sont séparés par ``SEPARATEUR`` — deux
    caractères — et rien d'autre : ``[61365, 61379]`` suivi de ``[61381, 61955]`` désigne un
    texte continu. Les réunir fait passer la liste d'un chunk de dix couples à un seul dans
    l'immense majorité des cas, et rend la ligne « ancre : » lisible par un humain.

    Le ``block_id`` est **retiré** ici, et volontairement : le payload n'a besoin que des
    coordonnées dans le texte canonique. La provenance complète, blocs nommés, reste dans
    ``ancrage-v1.json``, qui est la source.
    """
    couples = sorted([int(i[1]), int(i[2])] for i in intervalles)
    if not couples:
        return []
    fusion = [couples[0]]
    ecart = len(document_text.SEPARATEUR)
    for debut, fin in couples[1:]:
        if debut <= fusion[-1][1] + ecart:
            fusion[-1][1] = max(fusion[-1][1], fin)
        else:
            fusion.append([debut, fin])
    return fusion


def index_par_chunk(ancrage: dict | None = None) -> dict[str, dict]:
    """``chunk_id`` → ``{sha256, granularité, intervalles fusionnés}`` — la forme du payload."""
    ancrage = charger() if ancrage is None else ancrage
    plat: dict[str, dict] = {}
    for entree in (ancrage.get("documents") or {}).values():
        for chunk_id, provenance in entree["chunks"].items():
            plat[chunk_id] = {"doc_text_sha256": entree["sha256"],
                              "ancrage_granularite": provenance["g"],
                              "ancrage_intervalles": fusionner(provenance["i"])}
    return plat


def texte_canonique_du_document(document_id: str, ancrage: dict | None = None) -> str | None:
    """Le texte canonique d'un document, relu du disque. ``None`` si le document est inconnu."""
    ancrage = charger() if ancrage is None else ancrage
    entree = (ancrage.get("documents") or {}).get(document_id)
    dossier = entree["dossier"] if entree else dossiers_par_document().get(document_id)
    if not dossier:
        return None
    chemin = INGESTED / dossier / "blocks.jsonl"
    if not chemin.exists():
        return None
    texte, _ = document_text.texte_canonique(_lire_jsonl(chemin))
    return texte


def controle() -> dict:
    """L'aller-retour, sur tout le corpus ancré : ce qui est promis exact l'est-il ?"""
    ancrage = charger()
    if not ancrage:
        return {"erreur": "aucun ancrage construit"}
    dossiers = {d: e["dossier"] for d, e in ancrage["documents"].items()}
    verifies = reussis = 0
    echecs = []
    for document_id, dossier in sorted(dossiers.items()):
        chemin = INGESTED / dossier
        blocs = _lire_jsonl(chemin / "blocks.jsonl")
        texte, spans = document_text.texte_canonique(blocs)
        if document_text.sha256_texte(texte) != ancrage["documents"][document_id]["sha256"]:
            echecs.append({"document_id": document_id, "cause": "sha256 du texte canonique changé"})
            continue
        for chunk in _lire_jsonl(chemin / "chunks.jsonl"):
            provenance = document_text.provenance_du_chunk(chunk, spans, "")
            verifies += 1
            if document_text.aller_retour(texte, chunk, provenance):
                reussis += 1
            elif len(echecs) < 20:
                echecs.append({"document_id": document_id, "chunk_id": chunk["chunk_id"],
                               "granularite": provenance.granularite})
    return {"chunks_verifies": verifies, "aller_retour_reussis": reussis,
            "echecs": echecs, "documents": len(dossiers)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--construire", action="store_true", help=f"écrit {ANCRAGE.name}")
    parser.add_argument("--controle", action="store_true", help="vérifie l'aller-retour sur tout le corpus")
    args = parser.parse_args()

    if args.construire:
        resultat = construire()
        documents = resultat.pop("documents")
        ANCRAGE.write_text(json.dumps({**resultat, "documents": documents},
                                      ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
        invalider()
        print(json.dumps(resultat, ensure_ascii=False, indent=1))
        print(f"écrit : {ANCRAGE} ({ANCRAGE.stat().st_size / 1e6:.1f} Mo)")
    if args.controle:
        print(json.dumps(controle(), ensure_ascii=False, indent=1)[:4000])
    if not args.construire and not args.controle:
        parser.print_help()


if __name__ == "__main__":
    main()
