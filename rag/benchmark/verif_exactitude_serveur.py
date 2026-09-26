"""Étape 1 — un serveur Qdrant rend-il le MÊME classement qu'une recherche exacte ?

Le mode embarqué fait une recherche **exacte** par force brute (``local_collection.LocalCollection``,
``distances``). Un serveur, lui, peut faire du HNSW **approximatif**. Migrer sans le
vérifier remplacerait en silence l'un par l'autre, et le classement dense n'est pas robuste :
à une perturbation de 10⁻⁵, 73 des 155 questions du banc peuvent se réordonner
(``parite-egalites-5530cba145.json``). Ce module tranche la question **avant** qu'une seule
donnée réelle ne soit migrée.

Trois bras, et le troisième est la garde
-----------------------------------------

===========  ==========================================================================
``defaut``   exactement ce que ``build_index.main`` écrit aujourd'hui —
             ``VectorParams(size, distance)`` et **rien d'autre**. C'est la vraie
             question : une migration naïve serait-elle déjà approximative ?
``m0``       ``HnswConfigDiff(m=0)`` — l'exactitude demandée au niveau de la
             **collection**, pour n'avoir pas à toucher ``_dense`` pendant que la
             session A y travaille
``m16``      ``m=16, ef_construct=100`` (le seul réglage HNSW du dépôt,
             ``scripts/reproduce_historical_qdrant_protocol_v1.py``) **et**
             ``indexing_threshold=1`` pour forcer la construction de l'index
===========  ==========================================================================

**``m16`` est un contrôle positif, et sans lui l'expérience ne vaut rien.** Une garde qu'on
n'a pas fait échouer n'est pas une garde : si ``m16`` rend lui aussi des résultats exacts,
c'est que le dispositif n'a **aucun pouvoir discriminant** — collection trop petite pour que
HNSW s'active — et le module le dit et sort en erreur, au lieu de conclure faussement que
``m0`` a prouvé quelque chose.

Deux pièges de dimensionnement, nommés parce qu'ils produisent tous deux un faux négatif
-----------------------------------------------------------------------------------------
1. ``optimizers_config.indexing_threshold`` vaut **20 000 vecteurs** par défaut : en dessous,
   Qdrant ne construit **aucun** index HNSW, quel que soit ``m``. D'où ``indexing_threshold=1``
   sur le bras ``m16``.
2. L'index se construit de façon **asynchrone**. Interroger trop tôt mesure un balayage
   complet sur un index absent. Le module attend donc que la collection soit ``green`` et
   publie ``indexed_vectors_count`` pour chaque bras — c'est la preuve directe qu'un index
   existe, ou n'existe pas.

Données : les **vrais** vecteurs de ``data/qdrant-export/vectors.npy`` (19 443 × 1024, lus en
``mmap``, jamais modifiés), et non un nuage synthétique — l'erreur d'approximation d'un HNSW
dépend de la distribution. Aucune donnée du corpus servi n'est écrite, lue ni déplacée : la
collection ``quant_rag_ingested_all_qwen3_06b`` n'est pas ouverte.

    .venv/bin/python rag/benchmark/verif_exactitude_serveur.py --url http://localhost:6533
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np
from qdrant_client import QdrantClient, models

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import parite_backend as pb  # noqa: E402  — un seul domicile pour la comparaison par groupes

# Les données vivantes ne sont pas versionnées : l'export est lu à la racine du dépôt.
VECTEURS = HERE.parents[1] / "data" / "qdrant-export" / "vectors.npy"

GRAINE = 20260907
REQUETES = 50
PROFONDEUR = 50
LOT = 1000
PREFIXE = "verif_exactitude"
# Le tau exigé pour la parité EST le résultat : un booléen ne dirait pas de combien.
TOLERANCES = (0.0, 1e-8, 1e-7, 1e-6, 1e-5)


def bras_definis() -> dict:
    """Les trois configurations de collection, écrites avant la mesure."""
    return {
        "defaut": {
            "hnsw_config": None,
            "optimizers_config": None,
            "attendu": "inconnu — c'est la question",
        },
        "m0": {
            "hnsw_config": models.HnswConfigDiff(m=0),
            "optimizers_config": None,
            "attendu": "exact — aucun index HNSW ne doit être construit",
        },
        "seuil0": {
            # indexed_vectors_count = 18 000 sur le bras m0 désigne le vrai levier : ce qui
            # déclenche HNSW n'est pas la connectivité m, c'est le SEUIL D'INDEXATION des
            # optimiseurs. À 0, Qdrant ne construit aucun index et balaye toujours.
            "hnsw_config": models.HnswConfigDiff(m=0),
            "optimizers_config": models.OptimizersConfigDiff(indexing_threshold=0),
            "attendu": "exact — ni index, ni seuil : le balayage complet est le seul chemin",
        },
        "m16": {
            "hnsw_config": models.HnswConfigDiff(m=16, ef_construct=100),
            "optimizers_config": models.OptimizersConfigDiff(indexing_threshold=1),
            "attendu": "APPROXIMATIF — contrôle positif, il DOIT diverger",
        },
    }


def charger_vecteurs(limite: int | None) -> np.ndarray:
    if not VECTEURS.exists():
        sys.exit(f"refus : {VECTEURS} est absent")
    brut = np.load(VECTEURS, mmap_mode="r")
    if limite:
        brut = brut[:limite]
    return brut


def normaliser(bloc: np.ndarray) -> np.ndarray:
    bloc = np.asarray(bloc, dtype=np.float32)
    normes = np.linalg.norm(bloc, axis=1, keepdims=True)
    return bloc / np.where(normes != 0.0, normes, 1e-12)


def reference_exacte(vecteurs: np.ndarray, requetes: np.ndarray) -> dict:
    """Le classement exact, calculé hors de tout serveur — par blocs, pour ne pas charger 80 Mo."""
    total = len(vecteurs)
    scores = np.empty((len(requetes), total), dtype=np.float32)
    for debut in range(0, total, 4096):
        bloc = normaliser(vecteurs[debut:debut + 4096])
        scores[:, debut:debut + len(bloc)] = requetes @ bloc.T

    classements = {}
    for i in range(len(requetes)):
        ordre = np.argsort(-scores[i], kind="stable")[:PROFONDEUR]
        classements[f"q{i:02d}"] = [[f"p{int(j)}", "doc", float(scores[i][j])] for j in ordre]
    return classements


def construire(client: QdrantClient, nom: str, config: dict, vecteurs: np.ndarray, dim: int) -> dict:
    if client.collection_exists(nom):
        client.delete_collection(nom)
    client.create_collection(
        nom,
        vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE),
        **{k: v for k, v in config.items() if k in ("hnsw_config", "optimizers_config") and v is not None},
    )

    for debut in range(0, len(vecteurs), LOT):
        bloc = normaliser(vecteurs[debut:debut + LOT])
        client.upsert(nom, wait=True, points=[
            models.PointStruct(id=debut + i, vector=bloc[i].tolist(), payload={"chunk_id": f"p{debut + i}"})
            for i in range(len(bloc))
        ])

    # L'index est asynchrone : interroger avant sa construction mesurerait un balayage complet.
    limite = time.time() + 300
    while time.time() < limite:
        info = client.get_collection(nom)
        if str(info.status) .endswith("green"):
            break
        time.sleep(2)
    info = client.get_collection(nom)
    return {
        "statut": str(info.status),
        "points": info.points_count,
        "indexed_vectors_count": info.indexed_vectors_count,
    }


def analyser_scores(exact: dict, rendu: dict, tau: float) -> dict:
    """Distinguer une APPROXIMATION d'une TRONCATURE D'EX ÆQUO — la tolérance ne peut pas le faire.

    Une tolérance regroupe des scores voisins, mais elle est impuissante à la frontière du
    top-k : si dix vecteurs sont à égalité stricte et que la fenêtre n'en garde que trois, les
    deux camps gardent trois membres différents à score IDENTIQUE, et aucun tau ne les
    réconcilie. Ce n'est pourtant pas une erreur de recherche — c'est une coupure arbitraire
    dans un groupe d'ex æquo, et le corpus en porte beaucoup (797 doublons sur 5 000).

    Le départage se lit sur les SCORES, pas sur les identifiants :
      · si la suite des scores rendus est la même que celle de l'exact, le serveur a trouvé
        d'aussi bons candidats — la divergence est une troncature d'ex æquo ;
      · si un score rendu est STRICTEMENT INFÉRIEUR à celui de l'exact au même rang, le
        serveur a manqué un meilleur candidat — c'est une approximation, et elle se voit.
    """
    troncature, approximation, rappels, deficits = [], [], [], []
    for qid, liste in exact.items():
        se = np.array([x[2] for x in liste], dtype=np.float64)
        sr = np.array([x[2] for x in rendu[qid]], dtype=np.float64)
        n = min(len(se), len(sr))
        communs = {x[0] for x in liste} & {x[0] for x in rendu[qid]}
        rappels.append(len(communs) / len(liste))
        if {x[0] for x in liste} == {x[0] for x in rendu[qid]}:
            continue
        deficit = float(np.max(se[:n] - sr[:n]))       # > 0 : le serveur a rendu moins bon
        deficits.append(deficit)
        (approximation if deficit > tau else troncature).append(qid)
    return {
        "rappel_at_50_moyen": round(float(np.mean(rappels)), 6),
        "troncature_dex_aequo": {"n": len(troncature), "qids": troncature[:10]},
        "approximation_reelle": {"n": len(approximation), "qids": approximation[:10]},
        "deficit_de_score_max": max(deficits) if deficits else 0.0,
    }


def interroger(client: QdrantClient, nom: str, requetes: np.ndarray) -> dict:
    classements = {}
    for i, requete in enumerate(requetes):
        points = client.query_points(collection_name=nom, query=requete.tolist(),
                                     limit=PROFONDEUR, with_payload=True).points
        classements[f"q{i:02d}"] = [[p.payload["chunk_id"], "doc", float(p.score)] for p in points]
    return classements


def main() -> None:
    parseur = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parseur.add_argument("--url", default="http://localhost:6533",
                         help="serveur de vérification — JAMAIS 6333, adresse piégée par ~63 sites hérités")
    parseur.add_argument("--vecteurs", type=int, default=None, help="limiter le nombre de vecteurs")
    parseur.add_argument("--garder", action="store_true", help="ne pas supprimer les collections de test")
    parseur.add_argument("--bras", nargs="*", default=None, help="n'exécuter que ces bras")
    args = parseur.parse_args()

    if ":6333" in args.url:
        sys.exit("refus : 6333 est l'adresse que visent ~63 fichiers hérités du dépôt, dont "
                 "certains font delete_collection. Utilise 6533.")

    vecteurs = charger_vecteurs(args.vecteurs)
    dim = int(vecteurs.shape[1])
    print(f"vecteurs : {len(vecteurs)} × {dim}   ({VECTEURS})", flush=True)

    echantillon = np.asarray(vecteurs[:min(len(vecteurs), 5000)], dtype=np.float32)
    doublons = {"examines": len(echantillon),
                "distincts": int(len(np.unique(echantillon, axis=0))),
                "note": "des vecteurs identiques rendent l'ordre arbitraire : ce sont des ex æquo, pas des erreurs"}
    doublons["doublons"] = doublons["examines"] - doublons["distincts"]
    print(f"doublons du jeu de test : {doublons['doublons']}/{doublons['examines']}", flush=True)
    del echantillon

    rng = np.random.default_rng(GRAINE)
    indices = rng.choice(len(vecteurs), size=REQUETES, replace=False)
    requetes = normaliser(vecteurs[np.sort(indices)])

    print("classement exact hors serveur…", flush=True)
    exact = reference_exacte(vecteurs, requetes)
    reference = {"signature_corpus": "verification-etape-1", "classements": exact}

    client = QdrantClient(url=args.url)
    resultats = {}
    for nom_bras, config in bras_definis().items():
        if args.bras and nom_bras not in args.bras:
            continue
        nom = f"{PREFIXE}_{nom_bras}"
        print(f"\n=== bras {nom_bras} — {config['attendu']}", flush=True)
        etat = construire(client, nom, config, vecteurs, dim)
        print(f"    {etat}", flush=True)
        rendu = interroger(client, nom, requetes)

        # Balayage de tolérance : c'est le tau exigé, et non un booléen, qui décide.
        # Le corpus porte des vecteurs DUPLIQUÉS (voir --doublons) : sans tolérance, des
        # ex æquo à 1,0 se répartissent arbitrairement et produisent un faux négatif.
        par_tolerance = {}
        for tau in TOLERANCES:
            c = pb.comparer(reference, {"classements": rendu}, tau)
            par_tolerance[f"{tau:g}"] = {
                "composition_top50_divergentes": len(
                    c["B1a_composition_PORTE_DURE"]["profondeurs"]["top50"]["divergentes"]),
                "composition_top5_divergentes": len(
                    c["B1a_composition_PORTE_DURE"]["profondeurs"]["top5"]["divergentes"]),
                "ordre_divergentes": c["B1b_ordre_interne_DIAGNOSTIC"]["divergentes"]["n"],
                "parite": bool(c["PARITE"]),
            }

        tau_suffisant = next((t for t in TOLERANCES if par_tolerance[f"{t:g}"]["parite"]), None)
        analyse = analyser_scores(exact, rendu, 1e-6)
        resultats[nom_bras] = {
            "collection": etat,
            "tau_suffisant": tau_suffisant,
            "exact_a_tau_0": par_tolerance["0"]["parite"],
            "par_tolerance": par_tolerance,
            "analyse_des_scores": analyse,
        }
        print(f"    rappel@50 {analyse['rappel_at_50_moyen']}  ·  approximations réelles "
              f"{analyse['approximation_reelle']['n']}  ·  troncatures d'ex æquo "
              f"{analyse['troncature_dex_aequo']['n']}  ·  déficit max {analyse['deficit_de_score_max']:.2e}",
              flush=True)
        print(f"    tau suffisant pour la parité : {tau_suffisant}"
              f"  ·  divergences top50 à tau=0 : {par_tolerance['0']['composition_top50_divergentes']}/{REQUETES}",
              flush=True)
        if not args.garder:
            client.delete_collection(nom)

    # ------------------------------------------------------------------ le verdict
    # Le contrôle positif doit diverger MÊME avec une tolérance large : une approximation
    # HNSW écarte de vrais candidats, elle ne se réduit pas à du bruit de dernier bit.
    pouvoir_discriminant = resultats["m16"]["analyse_des_scores"]["approximation_reelle"]["n"] > 0
    rapport = {
        "url": args.url,
        "serveur": "qdrant 1.19.0",
        "vecteurs": len(vecteurs),
        "dimension": dim,
        "requetes": REQUETES,
        "profondeur": PROFONDEUR,
        "graine": GRAINE,
        "bras": resultats,
        "POUVOIR_DISCRIMINANT": pouvoir_discriminant,
        "m0_tau_suffisant": resultats["m0"]["tau_suffisant"],
        "defaut_tau_suffisant": resultats["defaut"]["tau_suffisant"],
        "doublons_du_jeu_de_test": doublons,
    }
    (HERE / "verif-exactitude-serveur.json").write_text(json.dumps(rapport, indent=1, ensure_ascii=False))
    print("\n" + json.dumps(rapport, indent=1, ensure_ascii=False))

    if not pouvoir_discriminant:
        sys.exit("\nARRÊT — le contrôle positif a ÉCHOUÉ : le bras m16 rend lui aussi des "
                 "résultats exacts. Le dispositif n'a donc AUCUN pouvoir discriminant, et "
                 "l'exactitude de m0 ne prouve rien. Redimensionner (plus de vecteurs) avant "
                 "de conclure quoi que ce soit.")
    exacts = [b for b, r in resultats.items()
              if b != "m16" and r["analyse_des_scores"]["approximation_reelle"]["n"] == 0]
    rapport["bras_exacts"] = exacts
    (HERE / "verif-exactitude-serveur.json").write_text(json.dumps(rapport, indent=1, ensure_ascii=False))
    if not exacts:
        sys.exit("\nARRÊT — AUCUNE configuration de collection ne rend un serveur exact. "
                 "L'exactitude devra être demandée au niveau de la REQUÊTE, "
                 "SearchParams(exact=True) — donc dans _dense, à arbitrer avec la session A. "
                 "Note : le mode embarqué IGNORE search_params (``qdrant_local.QdrantLocal``), donc "
                 "un tel ajout serait inerte sur le backend actuel.")
    print(f"\nBras exacts (0 approximation réelle) : {exacts}. Le contrôle positif m16 a divergé. "
          f"L'exactitude est atteignable au niveau de la COLLECTION : _dense n'a pas à bouger.")


if __name__ == "__main__":
    main()
