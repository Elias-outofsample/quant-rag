"""Le seul endroit qui décide **comment** Qdrant est ouvert — embarqué, ou serveur.

Quatre modules ouvrent aujourd'hui leur propre client, chacun avec la même ligne en dur
(``quant_rag.client``, ``build_index.main``, et deux sites de
``ingestion/recover_vectors``). Tant qu'il n'existe qu'un backend, quatre copies d'une même ligne ne coûtent rien.
Dès qu'il y en a deux, elles coûtent une divergence silencieuse : trois sites migrés et un
oublié servent deux corpus sous le même nom. Ce module existe pour qu'il n'y ait **qu'une
règle, à un seul endroit**.

Le contrat, et il est volontairement minuscule
-----------------------------------------------
``QUANT_RAG_QDRANT_URL`` **absente ou vide** ⇒ comportement d'aujourd'hui, **au bit près** :
``QdrantClient(path=<storage>)``, mode embarqué, recherche exacte par force brute.
``QUANT_RAG_QDRANT_URL`` **définie** ⇒ ``QdrantClient(url=<valeur>)``.

**Rien d'autre ne change.** Aucun ``search_params`` n'est ajouté, aucun paramètre de recherche,
aucun champ de payload, aucun contrat de sortie. L'exactitude d'un serveur ne se demande pas
ici : elle se fixe **à la création de la collection** (``m=0`` et ``indexing_threshold=0``,
mesuré par ``RAPPORT-ETAPE-1-EXACTITUDE-2026-09-07.md``), précisément pour que le chemin de
recherche n'ait pas à bouger.

Pourquoi une fonction pure séparée
-----------------------------------
``arguments()`` ne construit rien et n'importe rien : elle rend le dictionnaire d'arguments.
C'est ce qui la rend testable sans serveur, sans stockage et sans ``qdrant_client``, et c'est
ce qui permet de prouver la propriété qui compte — *variable absente, on ouvre exactement ce
qu'on ouvrait hier*.

Deux avertissements portés ici parce qu'ils n'ont pas d'autre domicile
----------------------------------------------------------------------
1. **Le port 6333 est une adresse piégée** tant que les ~63 fichiers hérités de l'amont
   (``scripts/``, ``src/``, ``tests/``) qui y visent un serveur n'ont pas été neutralisés :
   certains appellent ``delete_collection``. Ce module ne l'interdit pas — ce serait décider
   à la place d'une exploitation future — mais un serveur de travail devrait écouter ailleurs.
2. **Le mode embarqué ignore ``search_params``** (``qdrant_local.QdrantLocal``). Un paramètre de
   recherche ajouté « pour le serveur » serait donc inerte ici et actif là-bas : c'est
   exactement le genre d'asymétrie qui fait diverger deux backends sans qu'un test la voie.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

#: Le nom est stable et il est public : les rapports et les modes opératoires le citent.
VARIABLE = "QUANT_RAG_QDRANT_URL"

#: Où vit le manifeste de l'index BM25 — le seul témoin **hors ligne** de ce que la
#: collection servie contient. ``dense_matrix.served_counts()`` lit exactement le même.
LEXICAL = Path(__file__).resolve().parents[1] / "data" / "lexical"


def url_configuree() -> str | None:
    """L'URL du serveur, ou ``None``. Une valeur vide ou blanche vaut absence.

    Une variable exportée à vide est une erreur d'exploitation courante — et la traiter comme
    une URL produirait un client qui échoue à la connexion là où l'utilisateur croyait avoir
    désactivé le serveur.
    """
    valeur = (os.environ.get(VARIABLE) or "").strip()
    return valeur or None


def arguments(storage: str | Path) -> dict:
    """Les arguments d'ouverture du client. Fonction **pure** : rien n'est construit ici."""
    url = url_configuree()
    return {"url": url} if url else {"path": str(storage)}


def mode(storage: str | Path) -> str:
    """``"serveur"`` ou ``"embarque"`` — pour qu'un rapport puisse dire ce qu'il a interrogé."""
    return "serveur" if "url" in arguments(storage) else "embarque"


def description(storage: str | Path) -> str:
    """**Ce qui est réellement interrogé**, à afficher — jamais une constante de chemin.

    ``corpus_status`` annonçait ``str(STORAGE)`` en toutes circonstances. En mode serveur, cela
    désigne un dossier embarqué **qui n'est pas utilisé** : les comptes rendus à l'appelant MCP
    sont justes, mais leur provenance annoncée est fausse. C'est la famille de défaut que ce
    dépôt s'interdit — *« un index juste sous un nom qui ment »* — et elle a déjà coûté deux
    séries de comptes faux annoncés en 14 h.
    """
    args = arguments(storage)
    return args["url"] if "url" in args else args["path"]


def comptes_attendus() -> dict | None:
    """Ce que la collection servie doit contenir — ``None`` si aucun manifeste ne le dit.

    Le bloc ``built_from`` du manifeste BM25 est écrit **en lisant la collection elle-même**
    (``quant_rag.rebuild_bm25``) : c'est un témoin hors ligne, indépendant du backend qu'on
    s'apprête à ouvrir. C'est ce qui lui permet d'arbitrer entre deux backends qui prétendent
    tous deux servir la même chose.
    """
    import corpus_overlay

    path = LEXICAL / f"bm25-{corpus_overlay.LABEL}-{corpus_overlay.signature()}.manifest.json"
    if not path.exists():
        return None
    built = json.loads(path.read_text(encoding="utf-8")).get("built_from")
    return built if isinstance(built, dict) and built.get("points") else None


def verdict(observe: int | None, attendu: dict | None, collection: str, cible: str) -> str | None:
    """``None`` si l'ouverture est permise, sinon le **motif du refus**. Fonction pure.

    Elle est séparée d'``ouvrir`` pour la même raison qu'``arguments`` l'est : une décision
    qui se teste sans serveur, sans stockage et sans ``qdrant_client`` est une décision qu'on
    peut prouver. Les trois refus sont distincts parce qu'ils appellent trois gestes
    différents, et un message unique les confondrait.
    """
    if attendu is None:
        return (f"aucun manifeste d'index BM25 pour la signature courante : ce que {cible} "
                f"sert n'est pas prouvable hors ligne. Lance rag/benchmark/check_bm25.py, "
                f"ou passe garde=False en sachant que rien ne garantit alors que la "
                f"collection interrogée est celle du corpus courant.")
    if observe is None:
        return (f"la collection {collection!r} n'existe pas sur {cible} : il n'y a rien à "
                f"servir. Copie-la (rag/ingestion/migrer_vers_serveur.py) ou construis-la "
                f"(rag/build_index.py).")
    if observe != attendu.get("points"):
        return (f"{cible} sert {observe} points pour la collection {collection!r}, le corpus "
                f"courant en compte {attendu.get('points')} ({attendu.get('documents')} "
                f"documents). Ce serveur porte une copie **périmée** : la recherche "
                f"répondrait normalement sur un corpus qui n'est plus le vôtre. Recopie-la "
                f"(rag/ingestion/migrer_vers_serveur.py), ou reviens à l'embarqué en "
                f"retirant {VARIABLE} de l'environnement.")
    return None


def ouvrir(storage: str | Path, message_si_absent: str | None = None,
           collection: str | None = None, garde: bool = True):
    """Ouvrir un client — **neuf**, jamais mémoïsé : c'est l'appelant qui décide de son cycle.

    ``message_si_absent`` reproduit le refus historique de ``quant_rag.client()`` quand le
    stockage embarqué n'existe pas. Il ne s'applique **pas** en mode serveur : un serveur n'a
    aucun besoin d'un dossier local, et exiger sa présence rendrait la migration impossible
    sur une machine qui n'a jamais eu de collection embarquée.

    La garde de dérive — « parade B »
    ----------------------------------
    **En mode embarqué elle n'existe pas** : le chemin par défaut du dépôt est inchangé au
    bit près, et c'est la propriété que ce module a été écrit pour tenir.

    En mode serveur, elle compare le nombre de points servis au ``built_from.points`` du
    manifeste BM25 de la signature courante, et **refuse** en cas d'écart. Le défaut qu'elle
    interdit est nommé : un volume de test de 603 Mo porte aujourd'hui une copie complète de
    la collection ; ``export QUANT_RAG_QDRANT_URL=…`` suffirait à la servir. Elle est
    identique à celle du corpus **aujourd'hui**, et divergerait au premier document ingéré —
    sans que rien ne le dise. ``dense_matrix.Matrix.verify()`` applique déjà exactement cette
    comparaison au même artefact ; ce n'est pas une invention, c'est un précédent recopié.

    ``collection`` **doit** être nommée en mode serveur. Un défaut implicite rendrait la
    garde silencieusement inopérante chez tout appelant qui aurait oublié l'argument, et une
    garde qu'on peut désactiver par omission n'est pas une garde. ``garde=False`` reste
    possible — c'est le cas de ``build_index``, qui s'apprête à détruire puis reconstruire la
    collection et ne peut donc pas exiger qu'elle soit déjà juste — mais il faut l'écrire.

    Ce qu'elle **ne** détecte pas : deux collections de même cardinalité et de contenus
    différents. Le manifeste ne porte l'empreinte que de l'index BM25, pas des points.
    """
    from qdrant_client import QdrantClient

    args = arguments(storage)
    if "path" in args and message_si_absent and not Path(storage).exists():
        raise RuntimeError(message_si_absent)
    handle = QdrantClient(**args)
    if not garde or "url" not in args:
        return handle
    cible = args["url"]
    if collection is None:
        handle.close()
        raise RuntimeError(
            f"qdrant_backend : {VARIABLE} désigne {cible}, mais l'appelant n'a pas nommé la "
            f"collection qu'il attend — la garde de dérive ne peut donc rien vérifier. "
            f"Passe collection=… , ou garde=False si ce site a une raison de s'en dispenser.")
    try:
        observe = handle.count(collection).count if handle.collection_exists(collection) else None
    except Exception as erreur:  # une garde qui échoue doit se voir, pas s'effacer
        handle.close()
        raise RuntimeError(f"qdrant_backend : impossible de compter {collection!r} sur "
                           f"{cible} — {erreur}") from erreur
    motif = verdict(observe, comptes_attendus(), collection, cible)
    if motif:
        handle.close()
        raise RuntimeError(f"qdrant_backend : {motif}")
    return handle
