"""Latence de recherche : embarqué, serveur exact, serveur approximatif — sur les mêmes requêtes.

La question à laquelle ce module répond est **opérationnelle**, pas qualitative :
``verif_exactitude_serveur.py`` a déjà tranché la qualité (le bras HNSW par défaut perd du
pool ; ``m=0`` + ``indexing_threshold=0`` sont exacts). Reste à savoir ce que l'exactitude
coûte en temps, parce que c'est le seul argument qui pourrait la remettre en cause.

Trois cibles, une seule matière
--------------------------------
Les vecteurs de requête sont calculés **une fois** et rejoués tels quels sur les trois
cibles : le modèle d'embedding n'entre donc pas dans la mesure, et les trois latences sont
comparables au même bruit près. On mesure ``search`` seul — pas le reranking, pas le contrat
de sortie, pas le routeur : ce sont des coûts communs aux trois.

Ce qui est rapporté
--------------------
p50, p95 et p99 en millisecondes, plus le minimum et le maximum, par cible. La médiane
répond à « est-ce que ça change quelque chose au quotidien » ; le p95 à « est-ce que ça
s'effondre parfois ».

Un préchauffage est fait et **jeté** : la première requête d'un backend embarqué paie la
désérialisation des pickles (1 943 Mo mesurés au §11 de FUSION-MIGRATION-QDRANT.md), et la
compter reviendrait à mesurer un démarrage sous le nom d'une latence.

    .venv/bin/python rag/benchmark/latence_backends.py --url http://localhost:6533
    .venv/bin/python rag/benchmark/latence_backends.py --url … --repetitions 3
"""
from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "rag"))

SORTIE = pathlib.Path(__file__).resolve().parent / "latence-backends.json"
#: Les bras laissés par ``verif_exactitude_serveur.py --garder``. Ils portent les mêmes
#: 26 120 points que la collection servie, sous trois configurations d'index.
BRAS_EXACT = "verif_exactitude_seuil0"
BRAS_APPROXIMATIF = "verif_exactitude_defaut"


def quantiles(valeurs: list[float]) -> dict:
    ordonnees = sorted(valeurs)
    def q(p: float) -> float:
        if not ordonnees:
            return 0.0
        rang = min(len(ordonnees) - 1, int(round(p * (len(ordonnees) - 1))))
        return round(ordonnees[rang] * 1000, 2)
    return {"n": len(ordonnees), "p50_ms": q(0.50), "p95_ms": q(0.95), "p99_ms": q(0.99),
            "min_ms": round(ordonnees[0] * 1000, 2) if ordonnees else 0.0,
            "max_ms": round(ordonnees[-1] * 1000, 2) if ordonnees else 0.0,
            "moyenne_ms": round(statistics.fmean(ordonnees) * 1000, 2) if ordonnees else 0.0}


def mesurer(client, collection: str, vecteurs: list, limite: int, repetitions: int) -> dict:
    client.query_points(collection, query=vecteurs[0], limit=limite, with_payload=False)  # jeté
    durees = []
    for _ in range(repetitions):
        for vecteur in vecteurs:
            depart = time.perf_counter()
            client.query_points(collection, query=vecteur, limit=limite, with_payload=False)
            durees.append(time.perf_counter() - depart)
    return quantiles(durees)


def main() -> None:
    parseur = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parseur.add_argument("--url", required=True, help="serveur portant les bras de mesure")
    parseur.add_argument("--limite", type=int, default=50, help="profondeur du pool (défaut 50)")
    parseur.add_argument("--repetitions", type=int, default=3)
    parseur.add_argument("--sortie", type=pathlib.Path, default=SORTIE)
    args = parseur.parse_args()
    if ":6333" in args.url:
        sys.exit("refus : le port 6333 est une adresse piégée (≈63 fichiers hérités y visent "
                 "un serveur, certains font delete_collection). Utilise 6533.")

    from qdrant_client import QdrantClient
    import quant_rag

    # Les mêmes 155 questions que le banc et que les portes fonctionnelles : v1 (25) + v3 (130
    # retenues sur 150). On prend les fichiers tels quels, sans filtre — la latence ne dépend
    # pas de l'or, seulement du texte de la requête.
    ici = pathlib.Path(__file__).resolve().parent
    textes = []
    for nom in ("questions-v1.jsonl", "questions-v3.jsonl"):
        chemin = ici / nom
        if not chemin.exists():
            sys.exit(f"jeu de questions introuvable : {chemin}")
        textes += [json.loads(l)["question"] for l in chemin.read_text(encoding="utf-8").splitlines()
                   if l.strip()]
    print(f"{len(textes)} requêtes, {args.repetitions} répétition(s), limite {args.limite}")
    # ``encode_query`` du chemin servi, avec son instruction — pas une reconstitution.
    vecteurs = [quant_rag.encode_query(t) for t in textes]

    rapport = {"requetes": len(textes), "repetitions": args.repetitions, "limite": args.limite,
               "cibles": {}}

    embarque = quant_rag.client()
    rapport["cibles"]["embarque_exact"] = mesurer(embarque, quant_rag.COLLECTION, vecteurs,
                                                  args.limite, args.repetitions)
    print(f"  embarqué (exact)      p50 {rapport['cibles']['embarque_exact']['p50_ms']} ms")

    serveur = QdrantClient(url=args.url)
    try:
        for etiquette, collection in (("serveur_exact", BRAS_EXACT),
                                      ("serveur_hnsw", BRAS_APPROXIMATIF)):
            if not serveur.collection_exists(collection):
                rapport["cibles"][etiquette] = {"absent": collection,
                                                "note": "relance verif_exactitude_serveur.py --garder"}
                continue
            rapport["cibles"][etiquette] = mesurer(serveur, collection, vecteurs,
                                                   args.limite, args.repetitions)
            print(f"  {etiquette:20}  p50 {rapport['cibles'][etiquette]['p50_ms']} ms")
    finally:
        serveur.close()

    args.sortie.write_text(json.dumps(rapport, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(rapport, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
