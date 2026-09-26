"""Portes B1a, B1b, B4, B5, B7 — et les deux effets de bord du scroll.

L'étape 4 a prouvé que la copie est **fidèle** (A1–A7a). Elle n'a rien dit de ce que les deux
backends **servent**. Ce module produit, pour un backend donné, tout ce dont les portes
fonctionnelles ont besoin, puis compare deux productions.

Deux exécutions, une comparaison — et c'est voulu : ``quant_rag.client()`` est mémoïsé et lit
son environnement une seule fois. Interroger les deux backends dans un même processus
demanderait de casser ce cache, donc de mesurer autre chose que le chemin servi.

Ce qui est produit, et pourquoi
-------------------------------
==========  =================================================================================
``pool``    top-50 dense par question — la matière de B1a (composition) et B1b (ordre)
``servi``   les 5 passages réellement servis — **B7**, la porte produit
``filtres`` **B4**, et c'est la porte la plus exposée : sur les cinq champs filtrés, trois
            n'ont **aucun index de payload** (``chunk_id``, ``parent_id``, ``content_type``).
            En local le filtrage est un masque booléen Python, exact par construction ; sur un
            serveur c'est un autre chemin de code. ``get_passage`` filtre sur ``chunk_id`` et
            ``parent_id`` : c'est le chemin MCP servi à l'utilisateur
``scroll``  **B5** — l'ensemble des identifiants, **et leur ordre**. L'ensemble doit être
            identique ; l'ordre, lui, diffère légitimement (``_universal_id`` classe les
            chaînes avant les entiers). L'ordre est capturé parce que ``rebuild_bm25``
            accumule ses lignes **dans l'ordre du scroll, sans tri**
            (``quant_rag.rebuild_bm25``, boucle de ``scroll``) : si l'ordre change, l'``index_sha256`` du manifeste
            pourrait bouger sans qu'aucun texte n'ait changé, et ``check_bm25`` verrait une
            divergence fantôme
``comptes`` ``corpus_status`` — le dépôt a déjà annoncé deux fois des comptes faux au client
            MCP en 14 h ; ils doivent coïncider
==========  =================================================================================

``rebuild_bm25`` n'est **jamais appelé** : il écrit dans ``data/lexical/``, donc en production,
et le dossier interdit qu'un banc écrive. L'ordre du scroll est mesuré ; la conséquence sur
l'index est déduite de lui.

    QUANT_RAG_QDRANT_URL=http://localhost:6533 python portes_fonctionnelles.py --produire serveur.json
    python portes_fonctionnelles.py --produire embarque.json
    python portes_fonctionnelles.py --comparer embarque.json serveur.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for chemin in (ROOT, ROOT / "rag", ROOT / "src", HERE):
    sys.path.insert(0, str(chemin))

import pipeline  # noqa: E402
import quant_rag  # noqa: E402
from compare_v1_v2 import load_bench, load_v1  # noqa: E402
from corpus import ChunkIndex  # noqa: E402

POOL = 50
SERVI = 5
ECHANTILLON_FILTRES = 25


def _questions(index):
    for banc, items in (("v1", load_v1(index, HERE / "questions-v1.jsonl")),
                        ("v3", load_bench(HERE / "questions-v3.jsonl"))):
        for item in items:
            yield f"{banc}/{item['qid']}", item


def produire(sortie: pathlib.Path, storage: pathlib.Path | None = None) -> dict:
    if storage is not None:
        # Le worktree n'a pas les données gitignorées : le stockage embarqué vit dans l'arbre
        # principal. La surcharge est EXPLICITE et visible dans la ligne de commande — un lien
        # symbolique vers les données servies serait invisible et bien plus dangereux.
        quant_rag.STORAGE = storage
    index = ChunkIndex.load(verbose=False)   # le même que eval_dense_candidat:138
    pools, servis = {}, {}

    questions = list(_questions(index))
    for numero, (cle, item) in enumerate(questions, 1):
        print(f"  {numero}/{len(questions)}", end="\r", flush=True)
        classement = pipeline.retrieve_item(item, "dense", index, limit=POOL)
        pools[cle] = [[r["chunk_id"], r["document_id"], float(r["score"])] for r in classement]
        rendus = quant_rag.search(pipeline.query_of(item), limit=SERVI, mode="dense",
                                  rerank=False, auto_period=False, log=False)
        servis[cle] = [r["chunk_id"] for r in rendus]

    # --- B5 : l'ensemble ET l'ordre
    ordre, suivant = [], None
    while True:
        points, suivant = quant_rag.client().scroll(collection_name=quant_rag.COLLECTION,
                                                    limit=4096, offset=suivant,
                                                    with_payload=["chunk_id"])
        ordre.extend(p.payload["chunk_id"] for p in points)
        if suivant is None:
            break

    # --- B4 : les trois champs SANS index de payload, sur le chemin servi
    echantillon = [c for c in ordre[:ECHANTILLON_FILTRES]]
    passages = {}
    for chunk_id in echantillon:
        try:
            rendu = quant_rag.get_passage(chunk_id)
            passages[chunk_id] = {
                "chunk_id": rendu.get("chunk_id"),
                "pages": rendu.get("pages_utilisateur") or rendu.get("pages"),
                "empreinte_texte": hashlib.sha256((rendu.get("text") or "").encode()).hexdigest()[:16],
            }
        except Exception as erreur:                                  # noqa: BLE001
            passages[chunk_id] = {"erreur": f"{type(erreur).__name__}: {erreur}"}

    # --- B4 : le champ AVEC index de payload (document_id), sur le chemin dense
    doc = next(iter(quant_rag.document_metadata()))
    portee = quant_rag._dense("volatility", POOL, [doc])

    return {
        "collection": quant_rag.COLLECTION,
        "mode": "serveur" if quant_rag.os.environ.get("QUANT_RAG_QDRANT_URL") else "embarque",
        "pools": pools,
        "servis": servis,
        "scroll": {"n": len(ordre), "empreinte_ensemble": hashlib.sha256(
            "\n".join(sorted(ordre)).encode()).hexdigest(),
            "empreinte_ordre": hashlib.sha256("\n".join(ordre).encode()).hexdigest(),
            "dix_premiers": ordre[:10]},
        "filtres": {
            "get_passage": passages,
            "portee_document": {"document_id": doc, "n": len(portee),
                                "chunk_ids": [r["chunk_id"] for r in portee[:10]],
                                "tous_du_bon_document": all(r["document_id"] == doc for r in portee)},
        },
        "comptes": quant_rag.corpus_status(),
    }


def comparer(a: dict, b: dict) -> dict:
    import parite_backend as pb

    reference = {"signature_corpus": "portes", "classements": a["pools"]}
    b1 = pb.comparer(reference, {"classements": b["pools"]}, 0.0)

    servis_differents = sorted(q for q in a["servis"] if a["servis"][q] != b["servis"].get(q))
    passages_differents = sorted(c for c in a["filtres"]["get_passage"]
                                 if a["filtres"]["get_passage"][c] != b["filtres"]["get_passage"].get(c))

    portes = {
        "B1a_composition_pool": {"tenue": b1["B1a_composition_PORTE_DURE"]["tenue"],
                                 "profondeurs": {k: len(v["divergentes"]) for k, v in
                                                 b1["B1a_composition_PORTE_DURE"]["profondeurs"].items()}},
        "B1b_ordre_pool_DIAGNOSTIC": {
            "identiques": b1["B1b_ordre_interne_DIAGNOSTIC"]["identiques_position_a_position"],
            "permutations_dex_aequo": b1["B1b_ordre_interne_DIAGNOSTIC"]["equivalentes_par_permutation_dex_aequo"]["n"],
            "divergentes": b1["B1b_ordre_interne_DIAGNOSTIC"]["divergentes"]["n"]},
        "B4_filtres": {
            "get_passage_differents": len(passages_differents), "exemples": passages_differents[:5],
            "portee_document_identique": (a["filtres"]["portee_document"] == b["filtres"]["portee_document"]),
            "tenue": not passages_differents
                     and a["filtres"]["portee_document"] == b["filtres"]["portee_document"]},
        "B5_scroll": {
            "ensemble_identique": a["scroll"]["empreinte_ensemble"] == b["scroll"]["empreinte_ensemble"],
            "ordre_identique": a["scroll"]["empreinte_ordre"] == b["scroll"]["empreinte_ordre"],
            "n": [a["scroll"]["n"], b["scroll"]["n"]],
            "tenue": a["scroll"]["empreinte_ensemble"] == b["scroll"]["empreinte_ensemble"],
            "consequence_bm25": ("l'ordre diffère : rebuild_bm25 accumule ses lignes dans cet "
                                 "ordre sans tri, donc l'index_sha256 du manifeste pourrait "
                                 "bouger sans qu'aucun texte n'ait changé"
                                 if a["scroll"]["empreinte_ordre"] != b["scroll"]["empreinte_ordre"]
                                 else "l'ordre est identique : rebuild_bm25 rendrait le même index")},
        "B7_cinq_passages_servis": {"differents": len(servis_differents),
                                    "exemples": servis_differents[:10],
                                    "tenue": not servis_differents},
        # `storage` et `backend` DOIVENT différer entre les deux modes — c'est précisément
        # le correctif du 8 septembre : corpus_status annonce désormais ce qu'il interroge
        # vraiment. Les comparer ferait crier une divergence fantôme sur un champ dont la
        # différence est le comportement voulu. Ce sont les COMPTES qui doivent coïncider.
        "comptes_corpus_status": {
            "identiques_hors_backend": ({k: v for k, v in a["comptes"].items()
                                         if k not in ("storage", "backend")}
                                        == {k: v for k, v in b["comptes"].items()
                                            if k not in ("storage", "backend")}),
            "backend_a": a["comptes"].get("backend"), "storage_a": a["comptes"].get("storage"),
            "backend_b": b["comptes"].get("backend"), "storage_b": b["comptes"].get("storage")},
    }
    dures = ("B1a_composition_pool", "B4_filtres", "B5_scroll", "B7_cinq_passages_servis")
    portes["PORTES_DURES_OUVERTES"] = all(portes[nom]["tenue"] for nom in dures)
    return portes


def main() -> None:
    parseur = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parseur.add_argument("--produire", type=pathlib.Path)
    parseur.add_argument("--storage", type=pathlib.Path, default=None,
                         help="stockage embarqué à interroger (absolu)")
    parseur.add_argument("--comparer", nargs=2, type=pathlib.Path)
    args = parseur.parse_args()

    if args.produire:
        donnees = produire(args.produire, args.storage)
        args.produire.write_text(json.dumps(donnees, indent=1, ensure_ascii=False, default=str))
        print(f"\nproduit — mode {donnees['mode']}, {len(donnees['pools'])} questions, "
              f"{donnees['scroll']['n']} points au scroll")
        return

    if args.comparer:
        a = json.loads(args.comparer[0].read_text())
        b = json.loads(args.comparer[1].read_text())
        portes = comparer(a, b)
        rapport = {"a": {"fichier": str(args.comparer[0]), "mode": a["mode"]},
                   "b": {"fichier": str(args.comparer[1]), "mode": b["mode"]},
                   "portes": portes}
        (HERE / "portes-fonctionnelles.json").write_text(
            json.dumps(rapport, indent=1, ensure_ascii=False, default=str))
        print(json.dumps(rapport, indent=1, ensure_ascii=False, default=str))
        sys.exit(0 if portes["PORTES_DURES_OUVERTES"] else 1)

    parseur.error("choisis --produire ou --comparer")


if __name__ == "__main__":
    main()
