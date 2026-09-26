"""Copie contrôlée du stockage embarqué vers un serveur Qdrant — **la source n'est jamais écrite**.

Pourquoi une copie, et pas une restauration
--------------------------------------------
Les deux voies « standard » sont **fermées**, et l'étape 0 l'a établi dans le code installé :

- les **snapshots** lèvent ``NotImplementedError`` en mode local
  (``qdrant_local.QdrantLocal``, qui n'a pas de snapshot) — il n'y a rien à exporter ;
- le **dossier de stockage** est un SQLite de pickles Python (``persistence``, un SQLite de pickles)
  qu'un serveur ne sait pas monter.

Reste la ré-injection point par point : ``scroll(with_vectors=True)`` d'un côté, ``upsert`` de
l'autre. C'est un **transfert**, pas un recalcul — identifiants, vecteurs et payloads sont
recopiés tels quels, et rien n'est ré-embarqué.

La configuration de la cible porte l'exactitude
------------------------------------------------
``m=0`` **et** ``indexing_threshold=0``, mesurés à l'étape 1
(``RAPPORT-ETAPE-1-EXACTITUDE-2026-09-07.md``). Sans eux, un serveur rend un ``rappel@50`` de
**0,5868** — 41 % du pool perdu — et **pas même de façon reproductible** : deux constructions
de la même configuration ont rendu 26 puis 44 approximations. L'exactitude se demande donc à la
**création de la collection**, et non à la requête : c'est ce qui permet de ne pas toucher
``_dense`` pendant qu'un autre chantier le mesure.

L'empreinte, et ce qu'elle prouve d'un coup
--------------------------------------------
Plutôt que trois comparaisons séparées, une seule empreinte par côté :

    sha256 sur les points triés de   "<id>|<sha256 du vecteur>|<sha256 du payload canonique>"

Son égalité prouve **A2** (même ensemble d'identifiants), **A4** (payloads identiques champ à
champ) et **A5** (vecteurs identiques au bit) — et elle les prouve sur la **totalité**, pas sur
un échantillon. Une empreinte calculée en flux, jamais en mémoire.

Ce que ce module ne fait pas
-----------------------------
Il **n'écrit rien** dans la source : il ne fait que ``scroll``. Il ne déplace aucun pointeur de
configuration, ne définit ``QUANT_RAG_QDRANT_URL`` nulle part, et ne supprime pas le stockage
embarqué. La bascule et la suppression sont des gestes distincts, postérieurs, et soumis à
accord.

    .venv/bin/python rag/ingestion/migrer_vers_serveur.py \\
        --source ../autre-arbre/qdrant_storage_local --url http://localhost:6533   # un autre arbre
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
import time

import numpy as np
from qdrant_client import QdrantClient, models

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]

#: Le stockage de **l'arbre d'où ce fichier est lancé**, et non un chemin absolu.
#:
#: Corrigé le 8 septembre 2026 par le chantier ``reprise-ingestion``. La constante valait
#: ``<racine>/qdrant_storage_local`` en dur (chemin absolu de la machine), au motif qu'« un worktree n'a pas les
#: données gitignorées » : c'est vrai d'un worktree nu, faux dès qu'on lui en donne une copie,
#: et un worktree qui travaille sur Qdrant en a forcément une. L'effet était qu'un instrument
#: lancé depuis un autre worktree lisait le corpus **servi** sans le dire. Il n'a pas nui ici
#: parce qu'un verrou exclusif l'a arrêté — mais c'est le verrou qui a protégé, pas le code,
#: et sur un serveur ce verrou n'existe plus (§10.4 de FUSION-MIGRATION-QDRANT.md).
#:
#: ``--source`` reste disponible pour viser explicitement un autre arbre.
SOURCE_PAR_DEFAUT = ROOT / "qdrant_storage_local"
COLLECTION = "quant_rag_ingested_all_qwen3_06b"
LOT = 500

#: Les trois seuls index de payload que le dépôt pose (``build_index.main``).
INDEX_PAYLOAD = (
    ("publication_year", models.PayloadSchemaType.INTEGER),
    ("authors", models.PayloadSchemaType.KEYWORD),
    ("document_id", models.PayloadSchemaType.KEYWORD),
)


def _normaliser(vecteur) -> np.ndarray:
    """La distance Cosine fait normaliser le serveur À L'INSERTION ; l'embarqué normalise À LA
    RECHERCHE (``distances``) sans jamais le persister. Comparer les octets stockés
    reviendrait donc à exiger d'une migration correcte qu'elle échoue. On compare ce que les
    deux backends calculent réellement : le vecteur normalisé."""
    v = np.asarray(vecteur, dtype=np.float32)
    norme = np.linalg.norm(v)
    return v / norme if norme else v


def _payload_canonique(payload) -> str:
    """Sérialisation stable : l'ordre des clés ne doit pas décider d'une égalité."""
    return json.dumps(payload or {}, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def lire_source(source: pathlib.Path, collection: str):
    """Parcourir la collection embarquée. **Lecture seule** : uniquement des ``scroll``."""
    if not source.exists():
        sys.exit(f"refus : {source} est absent")
    client = QdrantClient(path=str(source))
    try:
        total = client.count(collection, exact=True).count
        suivant, vus = None, 0
        while True:
            points, suivant = client.scroll(collection, limit=LOT, offset=suivant,
                                            with_payload=True, with_vectors=True)
            if not points:
                break
            vus += len(points)
            print(f"  lu {vus}/{total}", end="\r", flush=True)
            yield total, points
            if suivant is None:
                break
    finally:
        client.close()


def creer_cible(client: QdrantClient, collection: str, dimension: int, force: bool) -> None:
    if client.collection_exists(collection):
        if not force and client.count(collection, exact=True).count:
            sys.exit(f"refus : {collection} existe déjà et n'est pas vide sur la cible. "
                     "Relance avec --force pour la remplacer.")
        client.delete_collection(collection)
    client.create_collection(
        collection,
        vectors_config=models.VectorParams(size=dimension, distance=models.Distance.COSINE),
        # L'exactitude, au niveau de la collection — étape 1. m=0 seul ne suffit pas :
        # il laisse indexed_vectors_count à 18 000, donc aucun témoin observable.
        hnsw_config=models.HnswConfigDiff(m=0),
        optimizers_config=models.OptimizersConfigDiff(indexing_threshold=0),
    )


def comparer(source_lots, cible: QdrantClient, collection: str) -> dict:
    """A2, A4 et A5 — **séparément**, parce qu'ils ne sont pas de même nature.

    Une empreinte unique par hachage était le premier dessin de ce module, et il était faux :
    elle collapse une vérification EXACTE (identifiants, payloads) et une vérification
    NUMÉRIQUE (vecteurs). Après normalisation, 10 394 points sur 26 120 diffèrent encore au
    dernier bit — un ``sha256`` n'absorbe aucune tolérance, et la porte échouait pour une
    raison qui n'était pas un défaut de migration.

    La comparaison se fait par lots : les identifiants d'un lot source sont relus sur la cible
    et confrontés sur place. Rien n'est accumulé en mémoire au-delà d'un lot.
    """
    ids_source, ids_absents, payloads_differents = set(), [], []
    ecart_max, points = 0.0, 0

    for _total, lot in source_lots:
        ids = [p.id for p in lot]
        ids_source.update(ids)
        rendus = {p.id: p for p in cible.retrieve(collection, ids=ids,
                                                  with_payload=True, with_vectors=True)}
        for p in lot:
            points += 1
            c = rendus.get(p.id)
            if c is None:
                ids_absents.append(p.id)
                continue
            if _payload_canonique(p.payload) != _payload_canonique(c.payload):
                payloads_differents.append(p.id)
            ecart = float(np.max(np.abs(_normaliser(p.vector) - _normaliser(c.vector))))
            ecart_max = max(ecart_max, ecart)

    return {"ids_source": ids_source, "ids_absents": ids_absents,
            "payloads_differents": payloads_differents,
            "ecart_vecteur_max": ecart_max, "points": points}


def portes_structurelles(cible: QdrantClient, collection: str, mesure: dict,
                         dimension: int, seuil_vecteur: float) -> dict:
    """Les assertions A1 à A7a, sur les artefacts et non sur l'intention."""
    info = cible.get_collection(collection)
    params = info.config.params.vectors
    schemas = {nom: str(getattr(champ, "data_type", champ))
               for nom, champ in (info.payload_schema or {}).items()}

    ids_cible = set()
    suivant = None
    while True:
        lot, suivant = cible.scroll(collection, limit=LOT, offset=suivant, with_payload=False)
        if not lot:
            break
        ids_cible.update(p.id for p in lot)
        if suivant is None:
            break

    en_trop = sorted(ids_cible - mesure["ids_source"])
    portes = {
        "A1_points": {"attendu": mesure["points"], "obtenu": info.points_count,
                      "ok": info.points_count == mesure["points"]},
        "A2_identifiants": {"absents_de_la_cible": len(mesure["ids_absents"]),
                            "en_trop_sur_la_cible": len(en_trop),
                            "exemples": (mesure["ids_absents"][:5] + en_trop[:5]),
                            "ok": not mesure["ids_absents"] and not en_trop},
        "A3_dimension": {"attendu": dimension, "obtenu": params.size, "ok": params.size == dimension},
        "A3_distance": {"attendu": "Cosine", "obtenu": str(params.distance),
                        "ok": str(params.distance).lower().endswith("cosine")},
        "A4_payloads": {"differents": len(mesure["payloads_differents"]),
                        "exemples": mesure["payloads_differents"][:5],
                        "ok": not mesure["payloads_differents"]},
        "A5_vecteurs_normalises": {
            "ecart_max": mesure["ecart_vecteur_max"], "seuil": seuil_vecteur,
            "ok": mesure["ecart_vecteur_max"] < seuil_vecteur,
            "note": ("comparaison APRÈS normalisation : la distance Cosine fait normaliser le "
                     "serveur à l'insertion et l'embarqué à la recherche. « Identiques au bit » "
                     "sur les octets stockés est insatisfiable par construction.")},
        "A6_index_payload": {"attendu": [n for n, _ in INDEX_PAYLOAD], "obtenu": sorted(schemas),
                             "ok": all(n in schemas for n, _ in INDEX_PAYLOAD)},
        "A7a_indexed_vectors_count": {"attendu": 0, "obtenu": info.indexed_vectors_count,
                                      "ok": info.indexed_vectors_count == 0},
    }
    portes["toutes_ouvertes"] = all(v["ok"] for v in portes.values() if isinstance(v, dict))
    return portes


SEUIL_VECTEUR = 1e-6


def main() -> None:
    parseur = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parseur.add_argument("--source", type=pathlib.Path, default=SOURCE_PAR_DEFAUT)
    parseur.add_argument("--url", required=True)
    parseur.add_argument("--collection", default=COLLECTION)
    parseur.add_argument("--force", action="store_true",
                         help="remplacer une collection cible non vide")
    parseur.add_argument("--verifier-seul", action="store_true",
                         help="ne rien copier : rejouer les portes sur une cible déjà peuplée")
    parseur.add_argument("--seuil-vecteur", type=float, default=SEUIL_VECTEUR)
    args = parseur.parse_args()

    if ":6333" in args.url:
        sys.exit("refus : 6333 est visé par ~63 fichiers hérités, dont certains appellent "
                 "delete_collection. Utilise un autre port tant qu'ils ne sont pas neutralisés.")

    cible = QdrantClient(url=args.url)
    depart = time.perf_counter()
    dimension, ecrits, total = None, 0, None
    normes_min, normes_max = None, None

    # ------------------------------------------------------------- passe 1 : la copie
    if not args.verifier_seul:
        print("passe 1 — copie", flush=True)
        for total, lot in lire_source(args.source, args.collection):
            if dimension is None:
                dimension = len(lot[0].vector)
                creer_cible(cible, args.collection, dimension, args.force)
            for point in lot:
                norme = float(np.linalg.norm(np.asarray(point.vector, dtype=np.float32)))
                normes_min = norme if normes_min is None else min(normes_min, norme)
                normes_max = norme if normes_max is None else max(normes_max, norme)
            cible.upsert(args.collection, wait=True, points=[
                models.PointStruct(id=p.id, vector=p.vector, payload=p.payload) for p in lot
            ])
            ecrits += len(lot)
        if not ecrits:
            sys.exit("refus : aucun point lu — rien à migrer")
        for nom, schema in INDEX_PAYLOAD:
            cible.create_payload_index(args.collection, field_name=nom,
                                       field_schema=schema, wait=True)

    # ------------------------------------------- passe 2 : la confrontation, source contre cible
    print("\npasse 2 — confrontation", flush=True)
    lots = lire_source(args.source, args.collection)
    premier = next(lots)
    if dimension is None:
        dimension = len(premier[1][0].vector)
    total = premier[0]

    def toutes_les_pages():
        yield premier
        yield from lots

    mesure = comparer(toutes_les_pages(), cible, args.collection)
    portes = portes_structurelles(cible, args.collection, mesure, dimension, args.seuil_vecteur)

    rapport = {
        "source": str(args.source), "url": args.url, "collection": args.collection,
        "points_lus": total, "points_ecrits": ecrits, "dimension": dimension,
        "verifier_seul": args.verifier_seul,
        "secondes": round(time.perf_counter() - depart, 1),
        "normes_source": {"min": normes_min, "max": normes_max,
                          "note": ("le stockage embarqué NE conserve PAS des vecteurs de norme 1 ; "
                                   "la distance Cosine les normalise — le serveur à l'insertion, "
                                   "l'embarqué à la recherche.")},
        "portes": portes,
        "la_source_na_pas_ete_ecrite": "seuls des scroll ont été émis vers le stockage embarqué",
    }
    (HERE.parent / "benchmark" / f"migration-serveur-{args.collection}.json").write_text(
        json.dumps(rapport, indent=1, ensure_ascii=False, default=str))
    print("\n" + json.dumps(rapport, indent=1, ensure_ascii=False, default=str))

    if not portes["toutes_ouvertes"]:
        sys.exit("\nARRÊT — une porte structurelle a échoué. La collection cible ne doit pas "
                 "servir, et le pré-enregistrement demande que le chantier s'arrête ici.")
    print("\nPortes A1 à A7a ouvertes. Restent B4, B5, B6 et B7 — étape 6.")


if __name__ == "__main__":
    main()
