"""Contrôles opérationnels — chaque panne doit échouer **bruyamment**.

Le mode de défaillance que ce dossier redoute n'est pas la panne : c'est le **résultat partiel
silencieux**. Il l'a déjà payé deux fois — un index reconstruit plus petit qui imprimait un
avertissement puis ``"status": "COMPLETED"``, et une matrice dense qui mesurait sur 71 % du
corpus en rendant des résultats parfaitement plausibles.

Un backend en réseau ajoute des façons neuves de rendre un résultat partiel : conteneur arrêté,
URL fausse, port fermé, collection absente, version incompatible. Ce module les provoque toutes
et exige de chacune qu'elle **lève**, avec un message qui nomme la cause. Une panne qui rend
zéro résultat sans erreur serait le pire des cas — et c'est exactement ce qu'on vérifie ne pas
se produire.

    python controles_operationnels.py --url http://localhost:6533
    docker compose -f docker-compose.qdrant.yml stop && python controles_operationnels.py --arrete
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
COLLECTION = "quant_rag_ingested_all_qwen3_06b"


def _tenter(description: str, action) -> dict:
    """Exécuter, et exiger que ça lève. Un succès silencieux EST le défaut cherché."""
    try:
        resultat = action()
    except Exception as erreur:                                       # noqa: BLE001
        return {"scenario": description, "a_leve": True,
                "type": type(erreur).__name__, "message": str(erreur)[:180],
                "verdict": "ÉCHOUE BRUYAMMENT — attendu"}
    return {"scenario": description, "a_leve": False, "resultat": repr(resultat)[:180],
            "verdict": "N'A PAS LEVÉ — c'est le défaut que ce contrôle cherche"}


def scenarios(url: str, arrete: bool) -> list[dict]:
    from qdrant_client import QdrantClient

    resultats = []

    if arrete:
        resultats.append(_tenter(
            "conteneur arrêté — l'URL est bonne, rien n'écoute",
            lambda: QdrantClient(url=url, timeout=5).count(COLLECTION, exact=True)))
        return resultats

    resultats.append(_tenter(
        "port fermé — rien n'écoute sur 6599",
        lambda: QdrantClient(url="http://localhost:6599", timeout=5).count(COLLECTION, exact=True)))

    resultats.append(_tenter(
        "hôte inconnu — l'URL ne résout pas",
        lambda: QdrantClient(url="http://hote-qui-nexiste-pas.invalid:6533", timeout=5)
        .count(COLLECTION, exact=True)))

    resultats.append(_tenter(
        "collection absente — le serveur répond, la collection n'existe pas",
        lambda: QdrantClient(url=url, timeout=10).count("collection_qui_nexiste_pas", exact=True)))

    # MESURÉ le 7 septembre 2026, et c'est le seul scénario qui n'échoue PAS bruyamment :
    # `QdrantRemote._check_compatibility` (``QdrantRemote._check_compatibility``) appelle `show_warning`,
    # JAMAIS `raise` — et tout le bloc est enveloppé dans un `except Exception` qui retombe
    # lui aussi en avertissement. Le garde de version est donc CONSULTATIF : un serveur d'une
    # majeure différente serait utilisé quand même. La seule protection réelle est l'épingle
    # d'image du compose, ce qui la rend porteuse et non cosmétique.
    #
    # Le patch vise `qdrant_remote.get_server_version` et non `version_check.get_server_version` :
    # le module importe le symbole PAR SON NOM (import de ``show_warning`` en tête de ``qdrant_remote``), donc patcher l'origine
    # n'intercepte rien. Une première version de ce contrôle a fait cette erreur et a conclu
    # « n'a pas levé » pour la mauvaise raison.
    def version_incompatible():
        import warnings as _w
        from qdrant_client import qdrant_remote
        vrai = qdrant_remote.get_server_version
        try:
            qdrant_remote.get_server_version = lambda *a, **k: "3.0.0"
            with _w.catch_warnings(record=True) as captures:
                _w.simplefilter("always")
                QdrantClient(url=url, timeout=10, check_compatibility=True).get_collections()
            messages = [str(c.message) for c in captures if "incompatible" in str(c.message)]
            if not messages:
                return "AUCUN avertissement — le garde n'a rien dit du tout"
            raise RuntimeError("AVERTISSEMENT SEULEMENT (pas d'exception) : " + messages[0][:120])
        finally:
            qdrant_remote.get_server_version = vrai

    resultats.append(_tenter("version de serveur incompatible (3.0.0 contre un client 1.19.0)",
                             version_incompatible))

    resultats.append(_tenter(
        "URL malformée — ni schéma ni port",
        lambda: QdrantClient(url="pas-une-url", timeout=5).count(COLLECTION, exact=True)))

    return resultats


def main() -> None:
    parseur = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parseur.add_argument("--url", default="http://localhost:6533")
    parseur.add_argument("--arrete", action="store_true",
                         help="scénario « conteneur arrêté » — à lancer conteneur éteint")
    args = parseur.parse_args()

    resultats = scenarios(args.url, args.arrete)
    muets = [r for r in resultats if not r["a_leve"]]
    rapport = {"url": args.url, "scenarios": len(resultats),
               "echouent_bruyamment": len(resultats) - len(muets),
               "silencieux": len(muets), "detail": resultats,
               "TOUS_BRUYANTS": not muets}
    print(json.dumps(rapport, indent=1, ensure_ascii=False))
    nom = "controles-operationnels-arrete.json" if args.arrete else "controles-operationnels.json"
    (HERE / nom).write_text(json.dumps(rapport, indent=1, ensure_ascii=False))
    sys.exit(0 if not muets else 1)


if __name__ == "__main__":
    main()
