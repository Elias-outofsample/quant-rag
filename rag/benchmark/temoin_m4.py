"""Témoin M4 — reconstruire depuis les artefacts, et confronter à ce que la collection sert.

L'étape 4 a **copié** ce que la collection servie contient (M3). Ce module **reconstruit** ce
que les artefacts versionnés disent qu'elle devrait contenir (M4), puis compare les deux point
par point. Deux chemins indépendants qui convergent valent bien mieux qu'un seul qui se
déclare correct.

Ce que M4 peut révéler, et qui n'est pas un défaut de migration
----------------------------------------------------------------
Un écart entre M3 et M4 ne dirait rien de la copie : il dirait que **l'index servi a dérivé de
ses sources**. C'est une famille de défaut que ce dépôt a déjà rencontrée — 93 passages sans
vecteur sur le disque et 410 dont le cache s'était écarté du servi, trouvés le 7 septembre par
``recover_vectors``. Si les deux convergent, la parité est prouvée deux fois. Si elles
divergent, le résultat est ailleurs, et il faut le dire comme tel.

Pourquoi ce module n'implémente PAS la composition
---------------------------------------------------
Elle vit dans ``build_index.main()`` et n'en est pas extractible sans refactorer un module de
production. La recopier ici serait exactement le défaut que le dépôt s'interdit — *« une copie
divergerait en silence »* (``dense_matrix.py``, à propos de ``imported_points``). Ce module
appelle donc **le vrai code de production**, redirigé par ``QUANT_RAG_QDRANT_URL`` (le point
d'injection de l'étape 2) et par une surcharge du **nom de collection**.

La garde qui compte
-------------------
``build_index`` fait ``delete_collection`` **avant** ``create_collection``. Lancé sur le nom
servi, il détruirait le résultat de l'étape 4. Ce module **refuse de démarrer** si le nom cible
est celui de la collection servie, et il **recompte la collection migrée avant et après** :
une reconstruction qui l'aurait touchée est détectée, pas supposée.

Note sur la configuration d'index : ``build_index`` crée la collection avec
``VectorParams(size, distance)`` et rien d'autre — donc **approximative** sur un serveur
(``rappel@50`` 0,5868, étape 1). Cela n'a **aucune incidence ici** : M4 est une comparaison de
**points stockés**, pas de résultats de recherche. L'index ne change pas ce qui est stocké.

    QUANT_RAG_QDRANT_URL=http://localhost:6533 \\
    .venv/bin/python rag/benchmark/temoin_m4.py --url http://localhost:6533
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time

from qdrant_client import QdrantClient

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "rag"))
sys.path.insert(0, str(ROOT / "rag" / "ingestion"))

import migrer_vers_serveur as mvs  # noqa: E402

SERVIE = "quant_rag_ingested_all_qwen3_06b"
TEMOIN = "temoin_m4_reconstruit"


def lots_depuis(client: QdrantClient, collection: str):
    """Adapter un client quelconque au format qu'attend ``mvs.comparer``."""
    total = client.count(collection, exact=True).count
    suivant = None
    while True:
        points, suivant = client.scroll(collection, limit=mvs.LOT, offset=suivant,
                                        with_payload=True, with_vectors=True)
        if not points:
            break
        yield total, points
        if suivant is None:
            break


def main() -> None:
    parseur = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parseur.add_argument("--url", default="http://localhost:6533")
    parseur.add_argument("--temoin", default=TEMOIN)
    parseur.add_argument("--migree", default=SERVIE)
    args = parseur.parse_args()

    if args.temoin == args.migree:
        sys.exit("refus : le témoin ne peut pas porter le nom de la collection migrée — "
                 "build_index supprime avant de créer.")
    if ":6333" in args.url:
        sys.exit("refus : 6333 est visé par ~63 fichiers hérités, dont certains suppriment.")

    cible = QdrantClient(url=args.url)
    avant = cible.count(args.migree, exact=True).count
    print(f"collection migrée avant reconstruction : {avant} points", flush=True)

    # Le vrai code de production, redirigé — jamais réimplémenté.
    os.environ[mvs.__dict__.get("VARIABLE", "QUANT_RAG_QDRANT_URL")] = args.url
    import build_index

    if build_index.COLLECTION == args.migree:
        build_index.COLLECTION = args.temoin
    assert build_index.COLLECTION == args.temoin, "la surcharge du nom de collection a échoué"

    depart = time.perf_counter()
    build_index.main()

    apres = cible.count(args.migree, exact=True).count
    if apres != avant:
        sys.exit(f"ARRÊT — la collection migrée est passée de {avant} à {apres} points pendant "
                 "la reconstruction. La garde de nom a échoué ; ne rien conclure.")

    print(f"\ncollection migrée après reconstruction : {apres} points — intacte", flush=True)
    print("confrontation M3 (copiée) contre M4 (reconstruite)…", flush=True)
    mesure = mvs.comparer(lots_depuis(cible, args.migree), cible, args.temoin)

    convergent = (not mesure["ids_absents"] and not mesure["payloads_differents"]
                  and mesure["ecart_vecteur_max"] < mvs.SEUIL_VECTEUR)
    rapport = {
        "url": args.url, "collection_migree": args.migree, "collection_temoin": args.temoin,
        "points_migree": apres, "points_temoin": cible.count(args.temoin, exact=True).count,
        "secondes": round(time.perf_counter() - depart, 1),
        "ids_absents_du_temoin": len(mesure["ids_absents"]),
        "exemples_absents": mesure["ids_absents"][:10],
        "payloads_differents": len(mesure["payloads_differents"]),
        "exemples_payloads": mesure["payloads_differents"][:10],
        "ecart_vecteur_max": mesure["ecart_vecteur_max"],
        "seuil": mvs.SEUIL_VECTEUR,
        "M3_ET_M4_CONVERGENT": convergent,
        "lecture": ("Une divergence ne serait PAS un défaut de migration : elle dirait que "
                    "l'index servi a dérivé de ses artefacts sources."),
    }
    (HERE / "temoin-m4.json").write_text(json.dumps(rapport, indent=1, ensure_ascii=False, default=str))
    print("\n" + json.dumps(rapport, indent=1, ensure_ascii=False, default=str))
    sys.exit(0 if convergent else 1)


if __name__ == "__main__":
    main()
