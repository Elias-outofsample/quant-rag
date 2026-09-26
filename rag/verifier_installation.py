"""Ce qui manque pour que ce dépôt serve — et qui n'est pas dans git.

Le problème
------------
Un `git clone` de ce dépôt ne sert rien. Le chemin servi dépend d'une dizaine d'artefacts
**hors git** : trois environnements Python, deux clés, un stockage Qdrant de 325 Mo, trois
overlays de vecteurs `.npz`, un index BM25 de 50,8 Mo, et des fichiers LFS qui peuvent être
présents sous la forme d'un pointeur de 130 octets. Aucun n'est signalé à l'absence : le
dépôt échoue plus tard, ailleurs, et pour une raison qui ne ressemble pas à la cause.

Ce module les nomme tous, dit comment chacun se régénère, et **rend un code de sortie non
nul** si l'un manque. Il ne répare rien : réparer demande des heures de calcul et parfois
les PDF d'origine, deux choses qu'un contrôle ne doit pas décider seul.

Présence n'est pas cohérence
-----------------------------
Un artefact présent peut être **le mauvais**. C'est la famille de défaut que tout ce dépôt
s'interdit — *« un index juste sous un nom qui ment »* — et elle a déjà coûté deux séries de
comptes faux annoncés en quatorze heures. Chaque contrôle porte donc, quand un témoin
existe, sur la **cohérence** et pas seulement sur l'existence :

- l'index BM25 : son ``built_from.points`` contre le ``totals.active_chunks`` du registre ;
- le stockage Qdrant : le nombre de points servis contre le même total ;
- les fichiers LFS : le contenu réel, pas la présence du fichier (un pointeur non résolu
  fait 130 octets et commence par ``version https://git-lfs``).

Ces confrontations ne sont pas inventées ici : ce sont celles de
``dense_matrix.Matrix.verify()``, ``quant_rag._bm25_matches_collection()`` et
``qdrant_backend.verdict()``. Un quatrième contrôle qui différerait des trois autres serait
un quatrième avis, pas une garantie.

    .venv/bin/python rag/verifier_installation.py            # code 0 si tout est là
    .venv/bin/python rag/verifier_installation.py --json     # pour un script
    .venv/bin/python rag/verifier_installation.py --avec-qdrant  # ouvre la collection (verrou)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[0]
sys.path.insert(0, str(HERE))

#: Un pointeur LFS non résolu : petit fichier texte qui commence par cette ligne. Le tester
#: sur la TAILLE seule attraperait aussi un vrai petit fichier ; on lit donc l'en-tête.
ENTETE_LFS = b"version https://git-lfs"

VERT, ROUGE, JAUNE = "OK    ", "MANQUE", "TIÈDE "


def est_un_pointeur_lfs(chemin: Path) -> bool:
    if not chemin.is_file() or chemin.stat().st_size > 1024:
        return False
    with chemin.open("rb") as fichier:
        return fichier.read(len(ENTETE_LFS)) == ENTETE_LFS


def controle(nom: str, present: bool, chemin: str, geste: str, sans_pdf: bool = True,
             detail: str = "", bloquant: bool = True) -> dict:
    return {"artefact": nom, "chemin": chemin, "present": present, "bloquant": bloquant,
            "regenerable_sans_les_pdf": sans_pdf, "comment": geste, "detail": detail}


def environnements() -> list[dict]:
    """Trois interpréteurs, et ils ne sont pas interchangeables.

    ``.venv-gliner`` existe parce que ``gliner2[local]`` impose ``transformers < 5`` et que
    rétrograder ``transformers`` dans ``.venv`` mettrait en danger la reproduction des
    vecteurs Qwen3 et du reranker. ``.venv-mineru`` isole MinerU. L'isolement est délibéré :
    un contrôle qui les confondrait laisserait passer la panne qu'il doit attraper.
    """
    attendus = (
        (".venv", "requirements.txt", "recherche, banc, ingestion"),
        (".venv-mineru", "requirements-mineru.txt", "parsing des PDF (MinerU 3.4.5)"),
        (".venv-gliner", None, "extraction d'entités (GLiNER2, transformers < 5)"),
    )
    sorties = []
    for nom, requirements, role in attendus:
        python = ROOT / nom / "bin" / "python"
        # Un worktree lie souvent ses venv à l'arbre principal. ``exists()`` suit le lien
        # sans le dire, et ``du -sh`` rend « 0B » sur une installation parfaitement saine :
        # on rapporte donc le chemin RÉSOLU, pas le chemin nominal.
        resolu = python.resolve() if python.exists() else python
        geste = (f"python3 -m venv {nom} && {nom}/bin/pip install -r {requirements}"
                 if requirements else
                 f"python3 -m venv {nom} && {nom}/bin/pip install 'gliner2[local]'")
        lien = f"  → {resolu}" if python.is_symlink() or (
            python.exists() and resolu != python) else ""
        sorties.append(controle(f"environnement {nom}", python.exists(),
                                f"{nom}/bin/python", geste, detail=role + lien))
    return sorties


def cles() -> list[dict]:
    """**Existence seulement.** Une clé ne se lit pas, ne s'affiche pas, ne se hache pas."""
    sorties = []
    for nom, role in ((".mistral-key", "titrage par LLM (compte à plafond nul en septembre)"),
                      (".google-key", "juge du banc et titrage de secours (500 appels/jour)")):
        chemin = HERE / "benchmark" / nom
        sorties.append(controle(f"clé {nom}", chemin.exists(),
                                f"rag/benchmark/{nom}",
                                "la déposer à la main — elle n'est ni dans git ni régénérable",
                                sans_pdf=True, detail=role, bloquant=False))
    return sorties


def fichiers_lfs() -> list[dict]:
    """L'export figé de l'amont et les vecteurs : présents en pointeur = absents en fait."""
    sorties = []
    for relatif, role in (
            ("data/qdrant-export/ids.npy", "identifiants de l'export figé"),
            ("data/qdrant-export/vectors.npy", "vecteurs de l'export figé"),
            ("data/qdrant-export/payloads.jsonl", "payloads de l'export figé"),
            ("data/embeddings/ingested-all-qwen3-06b/rows.jsonl",
             "chunks de l'amont — l'ordre d'extraction du graphe en dépend"),
            ("data/graph/graph-lite.json", "graphe servi (81 Mo)"),
    ):
        chemin = ROOT / relatif
        pointeur = est_un_pointeur_lfs(chemin)
        sorties.append(controle(
            f"LFS {Path(relatif).name}", chemin.exists() and not pointeur, relatif,
            f'git lfs pull -I "{relatif}"',
            detail=(f"{role} — POINTEUR LFS NON RÉSOLU ({chemin.stat().st_size} octets)"
                    if pointeur else role)))
    return sorties


def overlays() -> list[dict]:
    """Les trois `.npz` de vecteurs. **Deux d'entre eux ne se régénèrent pas sans les PDF.**"""
    import corpus_overlay

    sorties = [
        controle("overlay des tableaux (vecteurs)",
                 (HERE / "tables" / ".cache" / "vectors-tables-markdown-v1.npz").exists(),
                 "rag/tables/.cache/vectors-tables-markdown-v1.npz",
                 ".venv/bin/python rag/tables/convert_tables.py",
                 detail="vecteurs des chunks-tableaux convertis en Markdown"),
        controle("overlay des titres (vecteurs)", corpus_overlay.TITLE_VECTORS.exists(),
                 "rag/titles/.cache/vectors-clean-titles-v1.npz",
                 ".venv/bin/python rag/titles/reembed_titles.py  (≈ 2 h)",
                 detail="titres consolidés ré-embarqués — entrent dans la signature"),
    ]
    livraisons = sorted((HERE / "ingestion" / ".cache").glob("vectors-*.npz")) \
        if (HERE / "ingestion" / ".cache").exists() else []
    partiels = [c for c in livraisons if c.name.endswith(".partial.npz")]
    sorties.append(controle(
        "vecteurs des livraisons", bool(livraisons),
        f"rag/ingestion/.cache/vectors-*.npz ({len(livraisons)} fichier(s))",
        "IRRÉCUPÉRABLE : ni git, ni LFS, ni les PDF ne les rendent. Tant que la collection "
        "est en place : .venv/bin/python rag/ingestion/recover_vectors.py --write (~2 s, "
        "au bit près). Sinon build_index REFUSERA de reconstruire, et ce refus est juste — "
        "ne le désarme pas par --allow-missing. Copie ce répertoire hors bande avant de "
        "quitter la machine, au même titre que les clés",
        sans_pdf=False,
        detail=(f"{len(partiels)} embedding(s) interrompu(s) (.partial.npz) — "
                "une reprise les finira" if partiels else
                "les documents entrés depuis l'export figé tiennent leurs vecteurs d'ici")))
    return sorties


def index_et_collection(sans_qdrant: bool) -> list[dict]:
    """Les deux artefacts servis, et leur **cohérence** avec le registre.

    Le registre est le témoin hors ligne : ``totals.active_chunks`` dit ce que le corpus
    doit contenir. Un index ou une collection qui en diffère est **présent et faux**.
    """
    import corpus_overlay

    sorties = []
    registre = corpus_overlay.REGISTRY
    attendus = None
    if registre.exists():
        attendus = (json.loads(registre.read_text(encoding="utf-8")).get("totals") or {}) \
            .get("active_chunks")
    sorties.append(controle("registre d'imports", registre.exists(),
                            "rag/ingestion/registry-v1.json",
                            ".venv/bin/python rag/ingestion/registry.py --build",
                            detail=f"{attendus} chunks actifs déclarés" if attendus else ""))

    # La signature relit le disque à CHAQUE appel, volontairement et sans cache
    # (``corpus_overlay.signature()``). Pendant une écriture d'overlay elle est donc
    # transitoirement fausse — constaté le 8 septembre 2026 : un appel a rendu `7bbaf9fba8`
    # puis trois appels consécutifs `5530cba145`. Conclure sur la première valeur, c'est
    # annoncer « aucun index, installation cassée » sur une installation saine. On la lit
    # deux fois et on refuse de conclure si elle bouge.
    signature = corpus_overlay.signature()
    time.sleep(0.25)
    corpus_overlay.invalidate()
    seconde = corpus_overlay.signature()
    if seconde != signature:
        # On rend ce qui a déjà été constaté — le registre —, plus le refus de conclure sur
        # le reste : un contrôle qui jetterait ses observations pour ne garder que son
        # abstention rendrait moins d'information qu'il n'en a.
        sorties.append(controle(
            "signature de corpus stable", False, "rag/corpus_overlay.py",
            "une écriture est en cours (lot, convert_tables, reembed_titles) — "
            "attends qu'elle finisse et relance",
            detail=f"la signature a bougé pendant le contrôle ({signature} puis {seconde}) : "
                   "l'index BM25, la collection et le graphe ne sont PAS contrôlés"))
        return sorties
    index = ROOT / "data" / "lexical" / f"bm25-{corpus_overlay.LABEL}-{signature}.json"
    manifeste = index.with_suffix(".manifest.json")
    built = None
    if manifeste.exists():
        built = (json.loads(manifeste.read_text(encoding="utf-8")).get("built_from") or {})
    accord = built is not None and attendus is not None and built.get("points") == attendus
    sorties.append(controle(
        f"index BM25 de la signature {signature}", index.exists(),
        f"data/lexical/{index.name}",
        ".venv/bin/python rag/benchmark/check_bm25.py  (≈ 12 s, reconstruit et contrôle)",
        detail=(f"{built.get('points')} points indexés" if built else "aucun manifeste")))
    sorties.append(controle(
        "index BM25 accordé au registre", accord,
        f"data/lexical/{manifeste.name} :: built_from.points",
        ".venv/bin/python rag/benchmark/check_bm25.py",
        detail=(f"{built.get('points') if built else None} indexés contre {attendus} servis — "
                "l'index lexical est en RETARD sur la collection" if not accord else
                f"{attendus} des deux côtés")))

    stockage = ROOT / "qdrant_storage_local"
    sorties.append(controle(
        "stockage Qdrant embarqué", stockage.exists(), "qdrant_storage_local/",
        ".venv/bin/python rag/build_index.py  (≈ 14 s, DÉTRUIT et reconstruit la collection)",
        detail="reconstruit depuis l'export LFS + le registre + les overlays"))
    if sans_qdrant or not stockage.exists():
        return sorties
    try:
        import quant_rag
        points = quant_rag.client().count(quant_rag.COLLECTION, exact=True).count
        sorties.append(controle(
            "collection accordée au registre", attendus is not None and points == attendus,
            f"{quant_rag.COLLECTION}",
            ".venv/bin/python rag/build_index.py",
            detail=f"{points} points servis contre {attendus} au registre"))
    except Exception as erreur:                     # verrou tenu, collection absente…
        sorties.append(controle(
            "collection accordée au registre", False, "qdrant_storage_local/",
            "arrête le serveur MCP (pkill -f rag/mcp_server.py) puis relance ce contrôle",
            detail=f"{type(erreur).__name__}: {str(erreur)[:150]}", bloquant=False))
    return sorties


def graphe() -> list[dict]:
    resultats = ROOT / "data" / "graph" / "gliner-results" / "results.jsonl"
    return [controle(
        "extractions GLiNER", resultats.exists() and not est_un_pointeur_lfs(resultats),
        "data/graph/gliner-results/results.jsonl",
        ".venv-gliner/bin/python rag/graph/extract_entities.py  (≈ 2 h 30 pour 26 120 chunks "
        "à 0,714 chunk/s), puis rag/graph/build_graph.py  (1,3 s)",
        detail="une ligne par chunk — build_graph en est une fonction pure")]


def main() -> None:
    parseur = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parseur.add_argument("--json", action="store_true", help="rendre le rapport en JSON")
    parseur.add_argument("--avec-qdrant", action="store_true",
                         help="ouvrir la collection pour compter ses points. HORS DÉFAUT : "
                              "le stockage embarqué est mono-processus et l'ouvrir prend un "
                              "verrou exclusif qui ferait échouer tout ce qui tourne. "
                              "Arrête d'abord les serveurs MCP (pkill -f rag/mcp_server.py)")
    args = parseur.parse_args()

    controles = (environnements() + cles() + fichiers_lfs() + overlays()
                 + index_et_collection(not args.avec_qdrant) + graphe())
    manquants = [c for c in controles if not c["present"] and c["bloquant"]]
    tiedes = [c for c in controles if not c["present"] and not c["bloquant"]]

    if args.json:
        print(json.dumps({"racine": str(ROOT), "controles": controles,
                          "manquants": len(manquants), "tiedes": len(tiedes),
                          "complet": not manquants}, ensure_ascii=False, indent=1))
    else:
        print(f"installation — {ROOT}\n")
        for c in controles:
            etat = VERT if c["present"] else (ROUGE if c["bloquant"] else JAUNE)
            print(f"  {etat}  {c['artefact']:38} {c['chemin']}")
            if c["detail"]:
                print(f"          {c['detail']}")
            if not c["present"]:
                print(f"          → {c['comment']}")
                if not c["regenerable_sans_les_pdf"]:
                    print("          ⚠ ne se régénère PAS sans les PDF d'origine")
        print()
        if manquants:
            print(f"{len(manquants)} artefact(s) bloquant(s) manquant(s). "
                  "Lire docs/BOOTSTRAP-MACHINE-NEUVE.md pour l'ordre des gestes.")
        else:
            print("tout ce dont le chemin servi dépend est présent et accordé au registre.")
        if tiedes:
            print(f"{len(tiedes)} contrôle(s) non bloquant(s) en défaut "
                  f"({', '.join(c['artefact'] for c in tiedes)}).")
    sys.exit(1 if manquants else 0)


if __name__ == "__main__":
    main()
