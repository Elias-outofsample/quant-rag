"""La concurrence — le seul bénéfice que ce chantier revendique, démontré au lieu d'être plaidé.

Le chantier de migration n'apporte **aucun gain de retrieval** : c'est établi et écrit. Le seul
bénéfice qu'il revendique est l'exploitation, et il tient en une phrase : *deux processus
peuvent lire en même temps*. Un argumentaire ne le prouve pas. Deux processus lancés
simultanément, si.

Le blocage n'est pas théorique, et le dossier en porte la trace
----------------------------------------------------------------
- ``REPRESENTATION-CONCEPTION.md`` F10 : *« `build_index.py` ne pourra pas s'exécuter tant
  qu'un serveur MCP est connecté, et une session parallèle ne pourra pas lire Qdrant pendant
  l'indexation. »* C'est écrit dans un **mode opératoire**, pas dans une note.
- ``rag/benchmark/tests/test_dense_matrix_couverture.py`` est resté **rouge** tant que trois
  ``mcp_server.py`` tenaient le verrou — il est repassé vert à la seconde où ils se sont
  arrêtés.
- Le stockage embarqué avertit lui-même : *« Local mode is not recommended for collections with
  more than 20 000 points »* — la collection en compte 26 120.

Le dispositif
-------------
``N`` processus **réellement distincts** (``subprocess``, pas des fils), lancés ensemble, qui
ouvrent chacun leur client et lisent. Chacun rend son verdict ; le parent compte. Sur le
backend embarqué, le verrou est **exclusif** : un seul doit passer. Sur le serveur, tous.

Le contraste est le résultat. Un seul des deux régimes prouve quelque chose sans l'autre.

    python demo_concurrence.py --mode embarque --storage qdrant_storage_local
    python demo_concurrence.py --mode serveur  --url http://localhost:6533
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
COLLECTION = "quant_rag_ingested_all_qwen3_06b"


def travail(cible: str, collection: str) -> dict:
    """Un lecteur : ouvrir, compter, parcourir. C'est ce que fait toute session du dossier."""
    from qdrant_client import QdrantClient

    debut = time.perf_counter()
    args = {"url": cible} if cible.startswith("http") else {"path": cible}
    client = QdrantClient(**args)
    n = client.count(collection, exact=True).count
    points, _ = client.scroll(collection, limit=200, with_payload=["chunk_id"])
    client.close()
    return {"ok": True, "points": n, "lus": len(points),
            "secondes": round(time.perf_counter() - debut, 2)}


def main() -> None:
    parseur = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parseur.add_argument("--mode", choices=("embarque", "serveur"), required=True)
    parseur.add_argument("--storage", default=str(HERE.parents[1] / "qdrant_storage_local"))
    parseur.add_argument("--url", default="http://localhost:6533")
    parseur.add_argument("--lecteurs", type=int, default=3)
    parseur.add_argument("--collection", default=COLLECTION)
    parseur.add_argument("--worker", action="store_true", help="usage interne")
    args = parseur.parse_args()

    cible = args.url if args.mode == "serveur" else args.storage

    if args.worker:
        try:
            print(json.dumps(travail(cible, args.collection)))
        except Exception as erreur:                                   # noqa: BLE001
            print(json.dumps({"ok": False, "erreur": f"{type(erreur).__name__}",
                              "message": str(erreur)[:160]}))
        return

    # Lancés ENSEMBLE : les processus sont créés avant qu'aucun n'ait fini d'ouvrir.
    processus = [subprocess.Popen(
        [sys.executable, __file__, "--worker", "--mode", args.mode,
         "--storage", args.storage, "--url", args.url, "--collection", args.collection],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        for _ in range(args.lecteurs)]

    resultats = []
    for p in processus:
        sortie, _ = p.communicate()
        try:
            resultats.append(json.loads(sortie.strip().splitlines()[-1]))
        except Exception:                                             # noqa: BLE001
            resultats.append({"ok": False, "erreur": "sortie illisible", "message": sortie[:160]})

    reussis = [r for r in resultats if r.get("ok")]
    echecs = [r for r in resultats if not r.get("ok")]
    rapport = {
        "mode": args.mode, "cible": cible, "lecteurs_simultanes": args.lecteurs,
        "reussis": len(reussis), "echecs": len(echecs),
        "detail": resultats,
        "verrou_exclusif": len(reussis) == 1 and len(echecs) == args.lecteurs - 1,
        "concurrence_reelle": len(reussis) == args.lecteurs,
    }
    print(json.dumps(rapport, indent=1, ensure_ascii=False))
    (HERE / f"concurrence-{args.mode}.json").write_text(
        json.dumps(rapport, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
