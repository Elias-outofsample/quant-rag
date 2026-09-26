"""Les deux régimes de reconstruction, rejoués sur le même lot d'essai — et comparés par hash.

La question
------------
``batch_driver --reconstruction fin-de-lot`` diffère l'index BM25 et le graphe, et les
reconstruit **une fois** après le dernier document. Le régime ne vaut que s'il rend
**exactement** les mêmes artefacts que le régime historique. « Exactement » ne se déclare
pas : il se prouve par empreinte, sur un lot rejoué deux fois.

Ce qui est comparé, et ce qui ne peut pas l'être
-------------------------------------------------
=========================  ====================================================================
``{nodes, edges}``         sérialisé et haché. **Pas** ``graph-lite.json`` : ``build_graph``
                           encapsule son rapport dans le fichier servi (``build_graph.main``),
                           avec ``built_at`` et ``seconds``. Deux exécutions du *même* régime
                           en diffèrent déjà. L'exclusion est déclarée au §4 du
                           pré-enregistrement du 8 septembre, **avant** cette mesure
``index BM25``             haché tel quel : ``BM25Index.dump`` ne sérialise que
                           ``{k1, b, records}``, aucun horodatage (``retrieval.lexical.BM25Index.dump``)
``manifeste BM25``         comparé hors ``created_at``, le seul champ variable
``registre``               comparé hors ``built_at`` **et hors les ``received_at``**, qui
                           apparaissent deux fois par livraison (dans ``deliveries`` et dans
                           le ``delivery`` recopié sur chaque document). Ce sont des
                           horodatages, et le pré-enregistrement les exclut avant la mesure
``imported-rows.jsonl``    haché tel quel — c'est ce que lit l'extraction
``results.jsonl``          haché tel quel — son ORDRE décide du graphe, c'est donc la preuve
                           que reporter ``build_graph`` sans reporter l'extraction est sûr
=========================  ====================================================================

Les deux régimes emploient **le même identifiant de livraison** (``mesure-dNN``). Ce sont
les mêmes livraisons, et le descripteur entre dans le registre : les nommer différemment
faisait diverger le registre pour une raison qui venait du harnais, pas du régime. C'est
arrivé au premier passage, et c'est la seule divergence qu'il ait rendue.

Ce que le lot d'essai est
--------------------------
Trois documents fabriqués par ``make_fixture.py --single`` à partir de documents réels du
corpus : mêmes chunks, mêmes tableaux, même structure, identités refaites. Leur ``sha256``
est **déterministe** (``sha256("livraison-essai::<dossier>")``), donc les deux régimes
importent des documents identiques au bit — sans quoi la comparaison ne voudrait rien dire.
Aucun PDF n'est parsé : MinerU n'entre pas dans la mesure.

Ce que ce module écrit, et comment il le rend
----------------------------------------------
Il écrit **dans le corpus de l'arbre courant**, puis le restaure entre les deux régimes et
après le second, depuis un instantané pris au départ. Il **refuse de tourner dans un arbre
dont le nom est ``Rag``** : c'est l'arbre servi, et rien de ce module n'a à s'en approcher.

    .venv/bin/python rag/ingestion/mesure_regimes.py --instantane /tmp/pristine
    .venv/bin/python rag/ingestion/mesure_regimes.py --instantane /tmp/pristine --regime fin-de-lot
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "rag"))

import corpus_overlay  # noqa: E402

PY = ROOT / ".venv" / "bin" / "python"
PY_GLINER = ROOT / ".venv-gliner" / "bin" / "python"
SORTIE = ROOT / "rag" / "benchmark" / "mesure-regimes.json"

#: Trois documents réels, choisis parmi les plus petits du corpus : 12, 19 et 20 chunks.
#: Assez petits pour que l'extraction GLiNER tienne en une minute, assez nombreux pour que
#: « une fois à la fin » se distingue de « trois fois ».
SOURCES = ("doc-83046c141ab04bf3", "doc-e40ef0c462c52885", "doc-eba514fa27b4255f")

#: Ce qui est suivi par git et que le lot réécrit : ``git restore`` suffit à le rendre.
SUIVIS = (
    "rag/ingestion/registry-v1.json",
    "rag/ingestion/imported-rows.jsonl",
    "rag/metadata/documents-metadata-v1.json",
    "rag/tables/tables-markdown-v1.json",
    "data/graph/gliner-results/results.jsonl",
    "data/graph/gliner-results/report.json",
    "data/graph/graph-lite.json",
    "data/graph/graph-lite-report.json",
    "data/lexical/bm25-dedup-tables-registry-titles-v1-5530cba145.manifest.json",
    "rag/metadata/overrides.json",
    # Oublié au premier passage, et il l'a montré : ``check_bm25.py`` écrit son rapport ici
    # (nom dérivé de ``corpus_overlay.LABEL``), donc le lot d'essai y laissait la signature
    # du corpus d'essai. Un artefact du banc qui porte l'état d'un fixture est exactement le
    # genre de chiffre qu'on relit six mois plus tard en le croyant vrai.
    "rag/benchmark/results-bm25-dedup-tables-registry-titles-v1.json",
)


def semer_overrides(livraisons: list[tuple[str, Path]]) -> None:
    """Donner un titre de provenance ``manual`` aux documents d'essai — sans un appel d'API.

    ``apply_delivery`` refuse un titre de provenance ``filename_stem`` : *« le corpus porte
    title_not_from_filename: 258 — aucun de ses titres ne vient d'un nom de fichier, et
    l'embedding voit ce titre-là »*. La garde est juste et on ne la contourne pas : on lui
    donne ce qu'elle demande, une source acceptée. ``overrides.json`` rend ``manual``, qui en
    est une, et coûte **zéro appel** — la seule autre voie, ``--llm``, dépenserait du budget
    pour un texte qui n'est même pas de la prose.

    Les entrées sont retirées par ``restaurer`` (``overrides.json`` est dans ``SUIVIS``).
    """
    chemin = ROOT / "rag" / "metadata" / "overrides.json"
    donnees = json.loads(chemin.read_text(encoding="utf-8"))
    for _sha, dossier in livraisons:
        manifeste = json.loads((dossier / "manifest.json").read_text(encoding="utf-8"))
        entree = manifeste["documents"][0]
        processed = next((dossier / "processed").iterdir())
        document = json.loads((processed / "document.json").read_text(encoding="utf-8"))
        donnees[document["filename"]] = {
            "title": entree["title"],
            "note": "document d'essai de rag/ingestion/mesure_regimes.py — jamais dans le "
                    "corpus servi, retiré par la restauration.",
        }
    chemin.write_text(json.dumps(donnees, ensure_ascii=False, indent=1), encoding="utf-8")


def sha256_octets(donnees: bytes) -> str:
    return hashlib.sha256(donnees).hexdigest()


def sha256_fichier(chemin: Path) -> str | None:
    return sha256_octets(chemin.read_bytes()) if chemin.exists() else None


def courir(commande: list, env: dict | None = None) -> tuple[int, str, str, float]:
    """Rend ``(code, stdout, stderr, secondes)`` — **séparés**, et c'est la leçon d'un raté.

    Une première version concaténait les deux flux. Les modules du dépôt écrivent leur JSON
    de résultat sur ``stdout`` et leurs avertissements sur ``stderr`` ; la concaténation
    collait donc un ``UserWarning`` derrière le JSON, ``json.loads`` échouait, et
    l'instrument concluait « apply_delivery a échoué » sur un import parfaitement réussi.
    Une mesure qui se trompe de verdict est pire qu'une mesure absente.
    """
    depart = time.perf_counter()
    processus = subprocess.run([str(c) for c in commande], capture_output=True, text=True,
                               cwd=str(ROOT), env={**os.environ, **(env or {})})
    return (processus.returncode, processus.stdout or "", processus.stderr or "",
            round(time.perf_counter() - depart, 2))


def dernier_json(sortie: str) -> dict | None:
    """Le dernier objet JSON d'une sortie mêlée de texte.

    ``raw_decode`` plutôt que ``json.loads`` sur une tranche : il s'arrête à la fin de
    l'objet et ignore ce qui suit, donc un octet de trop derrière le JSON ne fait plus
    échouer la lecture.
    """
    decodeur = json.JSONDecoder()
    for position in range(len(sortie) - 1, -1, -1):
        if sortie[position] != "{":
            continue
        if position and sortie[position - 1] not in "\n\r":
            continue
        try:
            valeur, _ = decodeur.raw_decode(sortie[position:])
        except ValueError:
            continue
        if isinstance(valeur, dict):
            return valeur
    return None


# ------------------------------------------------------------------ instantané et restauration

def prendre_instantane(cible: Path) -> None:
    """Les artefacts **hors git** que le lot réécrit. Le reste se rend par ``git restore``."""
    cible.mkdir(parents=True, exist_ok=True)
    for nom, source in (("qdrant_storage_local", ROOT / "qdrant_storage_local"),
                        ("lexical", ROOT / "data" / "lexical"),
                        ("ingestion-cache", HERE / ".cache")):
        destination = cible / nom
        if destination.exists():
            shutil.rmtree(destination)
        if source.exists():
            shutil.copytree(source, destination)


def restaurer(instantane: Path, dossiers_neufs: list[str], livraisons: list[str]) -> None:
    """Rendre l'arbre à son état d'avant — hors git d'abord, git ensuite."""
    for nom, cible in (("qdrant_storage_local", ROOT / "qdrant_storage_local"),
                       ("lexical", ROOT / "data" / "lexical"),
                       ("ingestion-cache", HERE / ".cache")):
        source = instantane / nom
        if not source.exists():
            continue
        if cible.exists():
            shutil.rmtree(cible)
        shutil.copytree(source, cible)
    for dossier in dossiers_neufs:
        chemin = ROOT / "data" / "processed" / "ingested" / dossier
        if chemin.exists():
            shutil.rmtree(chemin)
    for livraison in livraisons:
        for chemin in (HERE / "journal" / f"{livraison}.json",
                       ROOT / "data" / "processed" / "incoming" / livraison):
            if chemin.exists():
                shutil.rmtree(chemin) if chemin.is_dir() else chemin.unlink()
    subprocess.run(["git", "restore", "--"] + list(SUIVIS), cwd=str(ROOT),
                   capture_output=True, text=True)


# ------------------------------------------------------------------ empreintes des artefacts

def empreintes() -> dict:
    """Ce que le lot a produit, réduit à des empreintes comparables entre deux régimes."""
    import quant_rag

    graphe = ROOT / "data" / "graph" / "graph-lite.json"
    resultat: dict = {}
    if graphe.exists():
        donnees = json.loads(graphe.read_text(encoding="utf-8"))
        charge = json.dumps({"nodes": donnees["nodes"], "edges": donnees["edges"]},
                            ensure_ascii=False).encode()
        resultat["graphe_nodes_edges_sha256"] = sha256_octets(charge)
        resultat["graphe_nodes"] = len(donnees["nodes"])
        resultat["graphe_edges"] = len(donnees["edges"])
        resultat["graphe_fichier_sha256"] = sha256_fichier(graphe)   # DOIT différer : report encapsulé

    index = quant_rag.bm25_path()
    resultat["bm25_index"] = index.name
    resultat["bm25_index_sha256"] = sha256_fichier(index)
    resultat["bm25_octets"] = index.stat().st_size if index.exists() else None

    manifeste = quant_rag.bm25_manifest_path()
    if manifeste.exists():
        contenu = json.loads(manifeste.read_text(encoding="utf-8"))
        resultat["bm25_manifeste_created_at"] = contenu.pop("created_at", None)
        resultat["bm25_manifeste_sha256_hors_created_at"] = sha256_octets(
            json.dumps(contenu, ensure_ascii=False, sort_keys=True).encode())
        resultat["bm25_built_from"] = contenu.get("built_from")

    registre = HERE / "registry-v1.json"
    if registre.exists():
        contenu = json.loads(registre.read_text(encoding="utf-8"))
        resultat["registre_built_at"] = contenu.pop("built_at", None)
        resultat["registre_totaux"] = contenu.get("totals")
        # Le registre porte un ``received_at`` par livraison — deux fois : dans l'entrée de
        # ``deliveries`` et dans le ``delivery`` recopié sur chaque document. Ce sont des
        # horodatages, et le pré-enregistrement les exclut de la comparaison **avant** la
        # mesure (§4 : « registre : identique hors horodatages »). Ce qui reste comparé, et
        # qui est ce qui compte : les documents, leurs sha256, leurs plages de points, leurs
        # statuts, et l'identité des livraisons.
        def _sans_horodatage(valeur):
            if isinstance(valeur, dict):
                return {c: _sans_horodatage(v) for c, v in valeur.items() if c != "received_at"}
            if isinstance(valeur, list):
                return [_sans_horodatage(v) for v in valeur]
            return valeur

        resultat["registre_sha256_hors_horodatages"] = sha256_octets(
            json.dumps(_sans_horodatage(contenu), ensure_ascii=False, sort_keys=True).encode())

    relais = HERE / "imported-rows.jsonl"
    resultat["imported_rows_sha256"] = sha256_fichier(relais)
    resultat["imported_rows_lignes"] = (
        sum(1 for l in relais.read_text(encoding="utf-8").splitlines() if l.strip())
        if relais.exists() else None)

    extraction = ROOT / "data" / "graph" / "gliner-results" / "results.jsonl"
    resultat["results_jsonl_sha256"] = sha256_fichier(extraction)
    resultat["results_jsonl_lignes"] = (
        sum(1 for l in extraction.read_text(encoding="utf-8").splitlines() if l.strip())
        if extraction.exists() else None)
    return resultat


# ------------------------------------------------------------------ un régime, de bout en bout

def fabriquer_livraisons(base: Path) -> list[tuple[str, Path]]:
    """Trois livraisons d'un document chacune. Déterministes : mêmes sha256 à chaque appel."""
    base.mkdir(parents=True, exist_ok=True)
    livraisons = []
    for source in SOURCES:
        sortie = base / f"essai-{source[4:12]}"
        code, _sortie, erreur, _ = courir([PY, HERE / "make_fixture.py", "--single", source,
                                           "--distinct", "--out", sortie])
        if code != 0:
            sys.exit(f"make_fixture --single {source} : {erreur.strip()[-400:]}")
        manifeste = json.loads((sortie / "manifest.json").read_text(encoding="utf-8"))
        livraisons.append((manifeste["documents"][0]["sha256"], sortie))
    return livraisons


def jouer(regime: str, livraisons: list[tuple[str, Path]], sans_extraction: bool) -> dict:
    """Un lot, dans un régime. Rend les durées par étape et les empreintes finales."""
    mesures: dict = {"regime": regime, "documents": [], "etapes": {}}
    semer_overrides(livraisons)
    identifiants = []
    for rang, (sha, dossier) in enumerate(livraisons, 1):
        # LE MÊME identifiant dans les deux régimes : ce sont les mêmes livraisons, et
        # le descripteur de livraison entre dans le registre. Les nommer différemment
        # aurait fait diverger le registre pour une raison qui vient du harnais et non
        # du régime mesuré — c'est exactement ce qui est arrivé au premier passage.
        identifiant = f"mesure-d{rang:02d}"
        identifiants.append(identifiant)
        commande = [PY, HERE / "apply_delivery.py", dossier, "--apply", "--id", identifiant, "--no-llm"]
        if regime == "fin-de-lot":
            commande.append("--sans-bm25")
        code, sortie, erreur, secondes = courir(commande)
        resultat = dernier_json(sortie) or {}
        if code != 0 or resultat.get("status") != "COMPLETED":
            return {**mesures, "echec": f"apply_delivery {identifiant} : "
                                       f"{(erreur or sortie).strip()[-600:]}"}
        entree = {"livraison": identifiant, "sha256": sha[:16],
                  "points": resultat.get("points"), "import_secondes": secondes,
                  "bm25_en_retard": resultat.get("bm25_en_retard")}

        if not sans_extraction:
            code, sortie, erreur, s = courir([PY_GLINER, ROOT / "rag" / "graph" / "extract_entities.py"])
            if code != 0:
                return {**mesures, "echec": f"extract_entities : {(erreur or sortie).strip()[-400:]}"}
            entree["extraction_secondes"] = s

        if regime == "par-document":
            code, sortie, erreur, s = courir([PY, ROOT / "rag" / "benchmark" / "check_bm25.py"])
            if code != 0:
                return {**mesures, "echec": f"check_bm25 : {(erreur or sortie).strip()[-400:]}"}
            entree["check_bm25_secondes"] = s
            code, sortie, erreur, s = courir([PY_GLINER, ROOT / "rag" / "graph" / "build_graph.py"])
            if code != 0:
                return {**mesures, "echec": f"build_graph : {(erreur or sortie).strip()[-400:]}"}
            entree["build_graph_secondes"] = s
        mesures["documents"].append(entree)

    if regime == "fin-de-lot":
        code, sortie, erreur, s = courir([PY_GLINER, ROOT / "rag" / "graph" / "build_graph.py"])
        if code != 0:
            return {**mesures, "echec": f"build_graph final : {(erreur or sortie).strip()[-400:]}"}
        mesures["etapes"]["build_graph_final_secondes"] = s
        code, sortie, erreur, s = courir([PY, ROOT / "rag" / "benchmark" / "check_bm25.py"])
        if code != 0:
            return {**mesures, "echec": f"check_bm25 final : {(erreur or sortie).strip()[-400:]}"}
        mesures["etapes"]["check_bm25_final_secondes"] = s

    mesures["livraisons"] = identifiants
    mesures["dossiers_neufs"] = [f"doc-{sha[:16]}" for sha, _ in livraisons]
    mesures["cumuls"] = cumuler(mesures)
    return mesures


def cumuler(mesures: dict) -> dict:
    """Ce que le régime a coûté, poste par poste — la seule comparaison qui décide."""
    somme = lambda cle: round(sum(d.get(cle) or 0 for d in mesures["documents"]), 2)
    etapes = mesures.get("etapes", {})
    return {
        "import": somme("import_secondes"),
        "extraction": somme("extraction_secondes"),
        "check_bm25": round(somme("check_bm25_secondes") +
                            (etapes.get("check_bm25_final_secondes") or 0), 2),
        "build_graph": round(somme("build_graph_secondes") +
                             (etapes.get("build_graph_final_secondes") or 0), 2),
        "appels_check_bm25": sum(1 for d in mesures["documents"] if d.get("check_bm25_secondes"))
                             + (1 if etapes.get("check_bm25_final_secondes") else 0),
        "appels_build_graph": sum(1 for d in mesures["documents"] if d.get("build_graph_secondes"))
                              + (1 if etapes.get("build_graph_final_secondes") else 0),
    }


def main() -> None:
    parseur = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parseur.add_argument("--instantane", type=Path, required=True,
                         help="dossier où prendre/relire l'instantané des artefacts hors git")
    parseur.add_argument("--livraisons", type=Path, default=Path("/tmp/mesure-regimes-livraisons"))
    parseur.add_argument("--sortie", type=Path, default=SORTIE)
    parseur.add_argument("--sans-extraction", action="store_true",
                         help="ne pas lancer GLiNER (le graphe restera celui d'avant)")
    parseur.add_argument("--garder", action="store_true",
                         help="ne pas restaurer après le second régime (pour inspecter)")
    args = parseur.parse_args()

    if ROOT.name == "Rag":
        sys.exit(f"REFUS : {ROOT} est l'arbre servi. Ce module écrit dans le corpus — "
                 "lance-le depuis un worktree jetable.")
    print(f"arbre : {ROOT}")

    if not (args.instantane / "qdrant_storage_local").exists():
        print(f"instantané absent, je le prends dans {args.instantane}")
        prendre_instantane(args.instantane)
    print("livraisons d'essai…")
    livraisons = fabriquer_livraisons(args.livraisons)
    print(f"  {len(livraisons)} livraisons : " + ", ".join(s[:12] for s, _ in livraisons))

    # Un artefact de mesure doit porter SON contexte : sans la date ni le corpus de départ,
    # on ne sait pas, six mois plus tard, à quel état du corpus ses empreintes se rapportent.
    # Manquait au premier jet ; constaté à la clôture du 9 septembre 2026.
    rapport: dict = {"date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                     "arbre": str(ROOT), "sources": list(SOURCES),
                     "signature_de_depart": corpus_overlay.signature(),
                     "sha256_livraisons": [s for s, _ in livraisons], "regimes": {}}
    for regime in ("par-document", "fin-de-lot"):
        print(f"\n=== régime {regime} ===")
        mesures = jouer(regime, livraisons, args.sans_extraction)
        if mesures.get("echec"):
            print(f"ÉCHEC : {mesures['echec']}")
            restaurer(args.instantane, mesures.get("dossiers_neufs")
                      or [f"doc-{s[:16]}" for s, _ in livraisons],
                      mesures.get("livraisons") or [])
            rapport["regimes"][regime] = mesures
            args.sortie.write_text(json.dumps(rapport, ensure_ascii=False, indent=1), encoding="utf-8")
            sys.exit(f"le régime {regime} a échoué — rapport dans {args.sortie}")
        mesures["empreintes"] = empreintes()
        rapport["regimes"][regime] = mesures
        print(f"  cumuls : {json.dumps(mesures['cumuls'], ensure_ascii=False)}")
        garder = args.garder and regime == "fin-de-lot"
        if not garder:
            restaurer(args.instantane, mesures["dossiers_neufs"], mesures["livraisons"])
            print("  arbre restauré")

    a = rapport["regimes"]["par-document"]["empreintes"]
    b = rapport["regimes"]["fin-de-lot"]["empreintes"]
    doivent_coincider = ("graphe_nodes_edges_sha256", "graphe_nodes", "graphe_edges",
                         "bm25_index_sha256", "bm25_index", "bm25_octets",
                         "bm25_manifeste_sha256_hors_created_at", "bm25_built_from",
                         "registre_sha256_hors_horodatages", "registre_totaux",
                         "imported_rows_sha256", "results_jsonl_sha256")
    verdict = {cle: {"par-document": a.get(cle), "fin-de-lot": b.get(cle),
                     "identique": a.get(cle) == b.get(cle)} for cle in doivent_coincider}
    rapport["verdict"] = verdict
    rapport["IDENTIQUES"] = all(v["identique"] for v in verdict.values())
    rapport["gain"] = {
        "check_bm25_appels": [rapport["regimes"]["par-document"]["cumuls"]["appels_check_bm25"],
                              rapport["regimes"]["fin-de-lot"]["cumuls"]["appels_check_bm25"]],
        "build_graph_appels": [rapport["regimes"]["par-document"]["cumuls"]["appels_build_graph"],
                               rapport["regimes"]["fin-de-lot"]["cumuls"]["appels_build_graph"]],
    }
    # L'économie porte sur TROIS postes, pas deux. `--sans-bm25` retire aussi une
    # reconstruction à l'intérieur d'`apply_delivery` (étape 10) : la compter dans
    # `import` et l'oublier ici sous-estimerait le gain d'un tiers. `extraction` est
    # délibérément exclue : elle fait le même travail dans les deux régimes, et l'écart
    # qu'on y observe est le bruit de la mesure — à publier comme tel, pas comme un gain.
    postes = ("import", "check_bm25", "build_graph")
    avant = rapport["regimes"]["par-document"]["cumuls"]
    apres = rapport["regimes"]["fin-de-lot"]["cumuls"]
    documents = len(rapport["regimes"]["par-document"]["documents"]) or 1
    rapport["gain"]["par_poste_s"] = {p: round(avant[p] - apres[p], 2) for p in postes}
    rapport["gain"]["secondes_economisees"] = round(
        sum(avant[p] - apres[p] for p in postes), 2)
    rapport["gain"]["secondes_par_document"] = round(
        rapport["gain"]["secondes_economisees"] / documents, 1)
    rapport["gain"]["bruit_extraction_s"] = round(
        avant["extraction"] - apres["extraction"], 2)
    args.sortie.write_text(json.dumps(rapport, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n" + json.dumps({"IDENTIQUES": rapport["IDENTIQUES"], "gain": rapport["gain"],
                             "divergences": [k for k, v in verdict.items() if not v["identique"]]},
                            ensure_ascii=False, indent=1))
    print(f"rapport : {args.sortie}")
    if not rapport["IDENTIQUES"]:
        sys.exit("LES DEUX RÉGIMES NE RENDENT PAS LES MÊMES ARTEFACTS")


if __name__ == "__main__":
    main()
