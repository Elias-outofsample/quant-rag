"""Récupère, depuis la collection servie, les vecteurs de livraison perdus du cache.

Le problème, constaté le 7 septembre 2026. ``build_index.py`` compose les points d'un
document importé à partir de deux fichiers : ses chunks (``data/processed/ingested/…``,
versionnés) et ses vecteurs (``rag/ingestion/.cache/vectors-<livraison>.npz``,
**gitignorés**). Deux livraisons du 6 septembre — ``2026-09-06-nuit-d01`` et
``-d02`` — ont perdu les leurs. Les 93 passages concernés sont **encore servis** : ils
vivent dans ``qdrant_storage_local/``, et nulle part ailleurs.

Conséquence, et c'est elle qui justifie cet outil : la seule copie de ces vecteurs est
dans la collection que ``build_index.py`` **supprime avant de reconstruire**. Une
reconstruction les effaçait donc définitivement — 26 120 points devenaient 26 027, deux
documents disparaissaient, et le rapport final disait ``COMPLETED``. La garde ajoutée à
``build_index.py`` arrête maintenant ce scénario ; cet outil-ci le répare.

**Pourquoi extraire plutôt que ré-embarquer.** Un ré-embarquement rendrait des vecteurs
*presque* identiques : le plancher fp16 mesuré par le chantier représentation est de
2,5 × 10⁻⁴ en médiane sur des entrées rigoureusement égales. Assez pour retourner un
ex æquo de classement, donc assez pour qu'une reconstruction ne reproduise plus les
mesures publiées. L'extraction est exacte, au bit près, et coûte deux secondes.

Ce que l'outil ne fait pas : il **n'écrit ni dans la collection, ni dans le registre, ni
dans les chunks**. Il ne change pas la signature du corpus — celle-ci ne couvre pas les
caches de vecteurs. Il rétablit un fichier absent, rien d'autre, et le gel n'est pas
concerné : aucun document n'entre ni ne sort.

**Un second écart, de nature différente, et ``--realign`` le traite.** 127 passages de trois
documents ont dans le cache un vecteur qui n'est pas celui que la collection sert. Le texte
est identique — vérifié chunk par chunk — et un plongement frais avec la recette courante
reproduit le vecteur **servi** au bit près, tout en s'écartant du cache de 3 × 10⁻⁴ à
1,5 × 10⁻³ : c'est le plancher fp16 de composition de lots, que ``embed_delivery`` produit
en triant ses chunks par longueur (les lots ne se regroupent pas deux fois pareil). Aucun
des deux n'est « faux » ; un seul est celui que le système sert, et c'est le seul qui rende
une reconstruction fidèle. ``--realign`` aligne donc le cache sur la collection.

    .venv/bin/python rag/ingestion/recover_vectors.py            # constat seul
    .venv/bin/python rag/ingestion/recover_vectors.py --write    # écrit les .npz manquants
    .venv/bin/python rag/ingestion/recover_vectors.py --realign --write   # + réaligne les divergents
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
MACOS = HERE.parent
ROOT = MACOS.parent
sys.path.insert(0, str(MACOS))

import corpus_overlay
import qdrant_backend  # noqa: E402

CACHE = HERE / ".cache"
INGESTED = ROOT / "data" / "processed" / "ingested"
STORAGE = ROOT / "qdrant_storage_local"
COLLECTION = "quant_rag_ingested_all_qwen3_06b"
BASELINE_DELIVERY = "corpus-initial"


def documents_sans_vecteurs() -> list[dict]:
    """Documents actifs, entrés par une livraison, dont un chunk éligible n'a pas de vecteur.

    Le même critère d'éligibilité que ``build_index.imported_points`` — ``rag_eligible
    is True`` —, et la même source de vérité pour la plage d'identifiants : le registre.
    """
    registre = json.loads(corpus_overlay.REGISTRY.read_text(encoding="utf-8"))
    vecteurs: dict[str, set[str]] = {}
    manquants = []
    for entree in registre.get("documents", []):
        livraison = (entree.get("delivery") or {}).get("id")
        if entree.get("status") != "active" or livraison in (None, BASELINE_DELIVERY):
            continue
        if livraison not in vecteurs:
            chemin = CACHE / f"vectors-{livraison}.npz"
            vecteurs[livraison] = (set(np.load(chemin)["chunk_ids"].tolist())
                                   if chemin.exists() else set())
        chemin_chunks = INGESTED / entree["folder"] / "chunks.jsonl"
        if not chemin_chunks.exists():
            continue
        eligibles = [json.loads(ligne) for ligne in chemin_chunks.open(encoding="utf-8")
                     if ligne.strip() and json.loads(ligne).get("rag_eligible") is True]
        absents = [c["chunk_id"] for c in eligibles if c["chunk_id"] not in vecteurs[livraison]]
        if not absents:
            continue
        plage = entree.get("point_ids") or {}
        manquants.append({"document_id": entree["document_id"], "livraison": livraison,
                          "eligibles": [c["chunk_id"] for c in eligibles], "absents": absents,
                          "premier": plage.get("first"), "compte": plage.get("count")})
    return manquants


def extraire(manquants: list[dict]) -> tuple[dict[str, dict[str, np.ndarray]], list[str]]:
    """Lit les vecteurs dans la collection, en vérifiant l'alignement point ↔ chunk.

    La vérification n'est pas une politesse : ``build_index`` attribue les identifiants de
    points par **position** (``first + offset``), et l'appariement n'est prouvé que si le
    payload du point rendu porte bien le chunk attendu à cette position. Un décalage d'un
    rang produirait des vecteurs parfaitement plausibles et parfaitement faux.
    """
    from qdrant_client import QdrantClient

    client = qdrant_backend.ouvrir(STORAGE, collection=COLLECTION)
    recolte: dict[str, dict[str, np.ndarray]] = {}
    erreurs: list[str] = []
    try:
        for doc in manquants:
            premier, eligibles = doc["premier"], doc["eligibles"]
            if premier is None:
                erreurs.append(f"{doc['document_id']} : aucune plage d'identifiants au registre")
                continue
            if doc["compte"] not in (None, len(eligibles)):
                erreurs.append(f"{doc['document_id']} : {len(eligibles)} chunks éligibles pour "
                               f"{doc['compte']} points au registre — plage non fiable, ignoré")
                continue
            attendu = {int(premier) + rang: cid for rang, cid in enumerate(eligibles)}
            points = client.retrieve(COLLECTION, ids=sorted(attendu), with_payload=True, with_vectors=True)
            rendus = {int(p.id): p for p in points}
            if len(rendus) != len(attendu):
                erreurs.append(f"{doc['document_id']} : {len(rendus)}/{len(attendu)} points rendus "
                               "par la collection — récupération partielle refusée")
                continue
            desaccord = [pid for pid, cid in attendu.items()
                         if rendus[pid].payload.get("chunk_id") != cid]
            if desaccord:
                erreurs.append(f"{doc['document_id']} : {len(desaccord)} points dont le payload ne "
                               "porte pas le chunk attendu à cette position — appariement refusé")
                continue
            cible = recolte.setdefault(doc["livraison"], {})
            for pid, cid in attendu.items():
                if cid in doc["absents"]:
                    cible[cid] = np.asarray(rendus[pid].vector, dtype=np.float32)
    finally:
        client.close()
    return recolte, erreurs


def ecrire(recolte: dict[str, dict[str, np.ndarray]]) -> list[str]:
    """Fusionne dans le ``.npz`` de la livraison — jamais d'écrasement du contenu existant."""
    ecrits = []
    for livraison, vecteurs in recolte.items():
        chemin = CACHE / f"vectors-{livraison}.npz"
        garde: dict[str, np.ndarray] = {}
        if chemin.exists():
            blob = np.load(chemin)
            garde = dict(zip(blob["chunk_ids"].tolist(), blob["vectors"]))
        garde.update(vecteurs)                       # les récupérés complètent, ne remplacent pas
        identifiants = sorted(garde)
        np.savez(chemin, chunk_ids=np.asarray(identifiants),
                 vectors=np.asarray([garde[c] for c in identifiants], dtype=np.float32))
        ecrits.append(f"{chemin.name} : {len(vecteurs)} récupérés, {len(garde)} au total")
    return ecrits


def divergents() -> tuple[dict[str, dict[str, np.ndarray]], dict[str, int]]:
    """Chunks dont le vecteur en cache n'est **pas** celui que la collection sert.

    Comparaison d'égalité stricte sur les valeurs stockées, sans tolérance : ``build_index``
    téléverse le vecteur du ``.npz`` tel quel, donc toute différence signifie que la
    collection a été bâtie depuis autre chose que ce cache — et qu'une reconstruction ne la
    reproduirait pas. Une tolérance masquerait exactement ce qu'on cherche.
    """
    from qdrant_client import QdrantClient

    registre = json.loads(corpus_overlay.REGISTRY.read_text(encoding="utf-8"))
    client = qdrant_backend.ouvrir(STORAGE, collection=COLLECTION)
    recolte: dict[str, dict[str, np.ndarray]] = {}
    par_document: dict[str, int] = {}
    try:
        for entree in registre.get("documents", []):
            livraison = (entree.get("delivery") or {}).get("id")
            if entree.get("status") != "active" or livraison in (None, BASELINE_DELIVERY):
                continue
            chemin = CACHE / f"vectors-{livraison}.npz"
            chemin_chunks = INGESTED / entree["folder"] / "chunks.jsonl"
            plage = entree.get("point_ids") or {}
            if not chemin.exists() or not chemin_chunks.exists() or plage.get("first") is None:
                continue
            blob = np.load(chemin)
            cache = dict(zip(blob["chunk_ids"].tolist(), blob["vectors"]))
            eligibles = [json.loads(ligne) for ligne in chemin_chunks.open(encoding="utf-8")
                         if ligne.strip() and json.loads(ligne).get("rag_eligible") is True]
            premier = int(plage["first"])
            points = client.retrieve(COLLECTION, ids=list(range(premier, premier + len(eligibles))),
                                     with_payload=["chunk_id"], with_vectors=True)
            servi = {p.payload["chunk_id"]: np.asarray(p.vector, dtype=np.float32) for p in points}
            for chunk in eligibles:
                cid = chunk["chunk_id"]
                if cid not in cache or cid not in servi:
                    continue
                if np.array_equal(np.asarray(cache[cid], dtype=np.float32), servi[cid]):
                    continue
                recolte.setdefault(livraison, {})[cid] = servi[cid]
                par_document[entree["document_id"]] = par_document.get(entree["document_id"], 0) + 1
    finally:
        client.close()
    return recolte, par_document


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--write", action="store_true", help="écrire les .npz (sinon : constat seul)")
    parser.add_argument("--realign", action="store_true",
                        help="aligner aussi les vecteurs présents mais différents de ceux servis")
    args = parser.parse_args()

    if args.realign:
        recolte, par_document = divergents()
        total = sum(len(v) for v in recolte.values())
        if not total:
            print("Réalignement : aucun vecteur en cache ne diffère de celui qui est servi.")
        else:
            print(f"Réalignement : {total} passage(s) dans {len(par_document)} document(s) "
                  "dont le vecteur en cache n'est pas celui servi.")
            for doc, n in sorted(par_document.items(), key=lambda kv: -kv[1]):
                print(f"  {doc}  {n} passages")
            if args.write:
                for ligne in ecrire(recolte):
                    print(f"  écrit  {ligne}")
            else:
                print("Constat seul — ajoute --write pour aligner.")
        print()

    manquants = documents_sans_vecteurs()
    if not manquants:
        print("Aucun document actif sans vecteurs : les caches de livraison sont complets.")
        return
    total = sum(len(d["absents"]) for d in manquants)
    print(f"{len(manquants)} document(s), {total} passage(s) sans vecteur sur le disque :")
    for doc in manquants:
        print(f"  {doc['document_id']}  {len(doc['absents'])}/{len(doc['eligibles'])} chunks  "
              f"livraison {doc['livraison']}  points {doc['premier']}…")

    recolte, erreurs = extraire(manquants)
    for erreur in erreurs:
        print(f"ATTENTION : {erreur}")
    recuperes = sum(len(v) for v in recolte.values())
    print(f"\n{recuperes}/{total} vecteurs retrouvés dans la collection, alignement vérifié point à point.")
    if not args.write:
        print("Constat seul — relance avec --write pour écrire les .npz.")
        return
    for ligne in ecrire(recolte):
        print(f"  écrit  {ligne}")
    if erreurs:
        sys.exit(1)


if __name__ == "__main__":
    main()
