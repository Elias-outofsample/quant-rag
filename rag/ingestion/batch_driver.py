"""Driver de lot — faire entrer, sans supervision, tous les PDF de ``source-b/entree/``.

Ce module **n'ajoute aucune étape** à la chaîne d'ingestion : il l'ordonne. Chaque commande
qu'il lance est celle de ``docs/SOURCE-B.md``, dans le même ordre, avec les mêmes options.
Il n'écrit lui-même ni dans Qdrant, ni dans BM25, ni dans le graphe — il appelle ceux qui
le font, un à la fois, et constate le résultat.

    entree/*.pdf ──┬─> parse_local ─> generate_manifest ─> inspect_delivery
                   │       │                                     │
                   │       └── refus ──────────────────────> ecartes/ + raison
                   │                                             │
                   └─────────────> apply_delivery --apply <──────┘
                                        │
                            registry --build ─> --verify ─> check_bm25
                                        │
                            extract_entities ─> build_graph ─> sondes ─> commit

Quatre règles, chacune payée par une mesure du 4 septembre 2026 :

  1. **Un seul processus lourd à la fois.** GLiNER est tombé à 0,189 chunk/s et s'est fait
     évincer de la RAM (``STAT=U``, RSS en baisse) parce que ``calibrate_router`` chargeait
     Qwen3 sur le même MPS. Seul, il tient 0,424. Ces 16 Go ne se partagent pas : tout est
     séquentiel, et le driver refuse de démarrer si un autre travail lourd tourne déjà.
  2. **Arrêt et mise de côté.** Tout ce qui n'est pas un ``ajout`` net — édition, doublon,
     re-parse, refus, titrage impossible, erreur inattendue — sort du lot dans ``ecartes/``
     avec sa raison, et le document suivant commence. Jamais de ``--accept-edition``
     automatique, jamais de titre inventé : sans opérateur, un document douteux attend.
  3. **Reprise par constat, pas par mémoire.** L'état de chaque document est un fichier,
     mais l'autorité est le monde : journal d'import, registre, collection. Relancé après
     un ``kill -9``, le driver regarde ce qui existe *vraiment* avant de décider. C'est ce
     qui lui permet d'annuler une tentative laissée en vol plutôt que de la doubler.
  4. **Le banc une seule fois, à la fin.** Un banc par document donnerait N lignes de base
     dont aucune ne vaut : ``INGESTION-CONCEPTION`` §5 interdit de comparer deux corpus.

    .venv/bin/python rag/ingestion/batch_driver.py --dry-run     # tout sauf l'écriture
    .venv/bin/python rag/ingestion/batch_driver.py --go          # le lot complet
    .venv/bin/python rag/ingestion/batch_driver.py --etat        # où en est-on (lecture seule)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

SOURCE_B = HERE / "source-b"
ENTREE = SOURCE_B / "entree"
ECARTES = SOURCE_B / "ecartes"
IMPORTES = SOURCE_B / "importes"
ETAT = SOURCE_B / "etat"
JOURNAUX = SOURCE_B / "journaux"
LIVRAISONS = SOURCE_B / "livraisons"
SOURCES = SOURCE_B / "sources.json"
VERROU = ETAT / "verrou.json"

JOURNAL_IMPORT = HERE / "journal"
REGISTRY = HERE / "registry-v1.json"
INCOMING = ROOT / "data" / "processed" / "incoming"
GLINER_RESULTS = ROOT / "data" / "graph" / "gliner-results" / "results.jsonl"
#: Le driver s'appelle lui-même en sous-processus (sondes, réparation) : chemin absolu,
#: jamais ``__file__``, qui reste relatif au répertoire de lancement.
MOI = HERE / "batch_driver.py"

PY = ROOT / ".venv" / "bin" / "python"
PY_MINERU = ROOT / ".venv-mineru" / "bin" / "python"
PY_GLINER = ROOT / ".venv-gliner" / "bin" / "python"

#: Sous ce seuil, on ne commence pas un document de plus. Un MinerU qui manque de place
#: laisse un dossier de travail à moitié écrit, et c'est le genre de panne qui se découvre
#: le lundi. 20 Gio couvrent largement le pire cas mesuré (2,3 Gio pour trois papiers).
DISQUE_MINIMUM = 20 * 1024 ** 3
#: Le défaut du dépôt est Mistral ; il est à plafond nul (429, ``x-ratelimit-limit-req-minute: 0``).
#: L'écart est déclaré ici, tracé dans l'état de chaque document et dans le rapport final.
LLM_DEFAUT = "gemini-3.1-flash-lite"
#: Valeur de ``--llm`` qui désactive tout appel externe. Ajoutée la nuit du 6 septembre 2026 :
#: le plafond quotidien Gemini est tombé après une vingtaine de documents et le lot ne pouvait
#: plus avancer, alors que 36 des 79 PDF restants portaient un ``/Title`` exploitable et que
#: les 43 autres pouvaient être titrés hors ligne dans ``overrides.json``.
SANS_LLM = "sans-llm"

#: Ce qui ne doit pas tourner en même temps que le lot. On ne teste **jamais** la seule
#: présence du nom d'un script : le shell qui a lancé la commande le porte lui aussi dans
#: sa ligne de commande, et ``pgrep -f`` conclut alors qu'un travail continue dix minutes
#: après sa mort. Le filtre porte sur l'exécutable *et* sur le script.
SCRIPTS_LOURDS = ("run_benchmark.py", "calibrate_router.py", "eval_router.py", "eval_protocol.py",
                  "extract_entities.py", "build_graph.py", "parse_local.py", "apply_delivery.py",
                  "build_index.py", "reembed_titles.py", "check_bm25.py", "apply_metadata.py",
                  "scan_duplicates.py", "batch_driver.py", "mineru")
BINAIRES_PYTHON = ("python", "python3", "mineru")

ETAPES = ("parse", "relevé", "diagnostic", "import", "aval", "graphe", "sondes", "commit")
#: Passé l'import, le document **est dans le corpus** : l'écarter serait mentir. Il passe en
#: « à-vérifier », son PDF ne bouge pas, et le rapport dit ce qui a manqué.
ETAPES_APRES_IMPORT = ("aval", "graphe", "sondes", "commit")
#: Trois échecs d'import d'affilée ne sont plus des documents difficiles, c'est une panne
#: commune — un plafond LLM épuisé écarterait sinon tout le lot pour une cause transitoire,
#: et les PDF restants sont plus utiles intacts dans entree/ qu'écartés pour rien.
#: L'extraction GLiNER se fait par tranches bornées, chacune dans un processus neuf.
#: La nuit du 2026-09-05 a montré pourquoi : un manuel de 360 pages a présenté 433 chunks
#: d'un coup à ``extract_entities.py``, dont les 46 documents précédents n'avaient jamais
#: demandé plus d'une centaine. Au 216e chunk d'un même run, le processus réclamait 18 Gio
#: sur une machine qui en a 16 ; macOS l'a déclaré ``stuck``, RSS 17 Mo, 12 % de CPU et
#: 200 Mo/s de swap — zéro chunk en trois minutes. Le défaut est dans la durée d'un run,
#: pas dans les chunks : les 250 restants faisaient 2 154 caractères de médiane. Découper
#: ne change strictement rien à ce qui est extrait (aucun chunk n'en regarde un autre) ;
#: seule la mémoire, rendue à chaque sortie de processus, cesse de s'accumuler.
TRANCHE_EXTRACTION = 120
LOT_EXTRACTION = 4
TRANCHES_MAX = 80

#: Les deux régimes de reconstruction des artefacts dérivés du corpus.
#:
#: ``par-document`` est le régime historique et reste le défaut : après chaque document, l'index
#: BM25 est reconstruit **deux fois** (``apply_delivery._apply`` étape 10, puis ``check_bm25``), le
#: graphe une fois, et les sondes en rebâtissent une troisième si le manifeste ne correspond
#: plus (``quant_rag.bm25()`` se répare tout seul). Coût mesuré le 8 septembre 2026 sur le
#: corpus servi : ``rebuild_bm25()`` 7,7 s et 50,8 Mo écrits, ``build_graph.py`` 1,3 s.
#:
#: ``fin-de-lot`` diffère ces reconstructions et les fait **une fois**, après le dernier
#: document. Économie ≈ 16,7 s et ≈ 102 Mo écrits par document ; sur les 408 s/document
#: mesurées en septembre, cela vaut ≈ 4 % du temps de lot — et non la moitié.
#:
#: **Ce qui n'est PAS différé, et pourquoi.** Trois étapes restent par document :
#:
#: - le **registre** et ``imported-rows.jsonl`` : les différer ferait rendre ``False`` à
#:   ``registre_connait()`` pour chaque document déjà appliqué, et ``etape_reprise``
#:   lancerait un ``--reparer`` par document — lequel refait le registre **et** un
#:   ``rebuild_bm25`` complet. Le mode deviendrait plus cher que celui qu'il remplace ;
#: - l'**extraction GLiNER** : ``imported-rows.jsonl`` est trié par sha256
#:   (``registry.write_imported_rows``) et ``pending`` suit cet ordre. Par document, ``results.jsonl``
#:   s'écrit dans l'ordre chronologique des imports ; en fin de lot il s'écrirait dans
#:   l'ordre sha256 du lot entier. Or ``build_graph`` résout les extrémités de ses 25 327
#:   arêtes de relation par ``lookup.setdefault`` — **le premier label rencontré gagne**
#:   (``build_graph``, ``lookup.setdefault``) — et 9,3 % des clés d'entité portent au moins deux labels.
#:   Différer l'extraction changerait donc le graphe **sans changer ses compteurs** : la
#:   forme d'écart la plus silencieuse qui soit. C'est aussi ce qui rend l'identité
#:   bit-à-bit du graphe atteignable entre les deux régimes ;
#: - le **commit** : il porte le PDF et le registre, et un lot interrompu doit laisser une
#:   histoire lisible document par document.
RECONSTRUCTIONS = ("par-document", "fin-de-lot")

#: Le libellé du corpus, lu à l'import : ``CHEMINS_COMMIT`` en a besoin, et un nom d'artefact
#: recopié à la main finit toujours par mentir (voir le commentaire de cette constante).
sys.path.insert(0, str(ROOT / "rag"))
import corpus_overlay as _corpus_overlay  # noqa: E402

LABEL_CORPUS = _corpus_overlay.LABEL

ECHECS_IMPORT_AVANT_ARRET = 3
#: Seul l'aval arrête le lot : registre, relais et BM25 en échec, c'est le corpus servi qui
#: est incohérent, et le document suivant aggraverait. Un graphe ou un commit raté se
#: rattrape (l'extraction est incrémentale, le commit suivant reprend les deux).
ETAPES_QUI_ARRETENT = ("aval",)
#: Un import dont la réparation échoue arrête aussi le lot : le document est servi, mais le
#: registre — donc ``imported-rows.jsonl``, donc le graphe — ne le connaît pas. C'est l'état
#: qu'un ``kill -9`` entre les étapes 8 et 9 d'``apply_delivery`` produit, et il est muet.
ARRET_DEMANDE = False
#: Régime de reconstruction du lot en cours. Global comme ``ARRET_DEMANDE``, et pour la même
#: raison : les ``etape_*`` ont une signature fixe ``(doc, journal, sec)``, imposée par la
#: boucle de ``traiter``. Fixé une fois dans ``main()``, jamais ailleurs.
RECONSTRUCTION = "par-document"

#: L'étape du graphe est-elle sautée ? **Défaut : non.** Global pour la même raison que
#: ``RECONSTRUCTION``, fixé une fois dans ``main()``.
#:
#: Pourquoi le drapeau existe (10 septembre 2026, lot ``cloture-operationnelle``)
#: -----------------------------------------------------------------------------
#: L'extraction GLiNER coûte ~95 s par document — de loin le premier poste après le parse —
#: et le graphe est mesuré **sans apport**, ni au rang ni au niveau réponse (§12 de
#: ``docs/STRATEGIE.md``). Jusqu'ici aucun drapeau ne permettait de la sauter : un lot qui
#: n'avait pas besoin du graphe le payait quand même.
#:
#: Ce qu'on perd en le passant : ``search_graph``, ``expand_entity`` et ``connect_entities``
#: cessent de refléter les documents ajoutés par ce lot — le reste du chemin servi
#: (recherche dense, BM25, citations, ancrage) est intact, le graphe n'y entre pas.
#: Ce qu'on gagne : le temps de lot divisé par ~5.
#:
#: Ce que le drapeau ne fait PAS : il ne supprime rien, ne change pas le défaut, et ne
#: dispense pas de rattraper le graphe plus tard — l'extraction est incrémentale, un lot
#: suivant sans le drapeau reprend exactement les chunks laissés de côté.
SANS_GRAPHE = False


# ------------------------------------------------------------------ outils

def maintenant() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_fichier(chemin: Path) -> str:
    digest = hashlib.sha256()
    with chemin.open("rb") as handle:
        for bloc in iter(lambda: handle.read(1 << 20), b""):
            digest.update(bloc)
    return digest.hexdigest()


def ecrire_json(chemin: Path, valeur) -> None:
    """Écriture atomique : un ``kill -9`` ne peut pas laisser un état à moitié écrit."""
    chemin.parent.mkdir(parents=True, exist_ok=True)
    provisoire = chemin.with_suffix(chemin.suffix + ".tmp")
    provisoire.write_text(json.dumps(valeur, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(provisoire, chemin)


def lire_json(chemin: Path):
    return json.loads(chemin.read_text(encoding="utf-8")) if chemin.exists() else None


def dernier_json(sortie: str):
    """Le dernier objet JSON d'une sortie mêlée de traces — ce que rendent nos scripts.

    ``raw_decode`` s'arrête au premier objet complet et ignore ce qui le suit : il le faut,
    car ``generate_manifest`` imprime son JSON *puis* un avertissement en clair, et
    ``apply_delivery`` imprime dix lignes d'étape *avant* le sien. On ne considère que les
    accolades en début de ligne, pour ne pas ouvrir sur un objet imbriqué et indenté.
    """
    decodeur = json.JSONDecoder()
    trouve, depart = None, 0
    while True:
        indice = sortie.find("{", depart)
        if indice < 0:
            return trouve
        if indice == 0 or sortie[indice - 1] == "\n":
            try:
                trouve = decodeur.raw_decode(sortie[indice:])[0]
            except ValueError:
                pass
        depart = indice + 1


def courir(commande, journal: Path, env: dict | None = None) -> tuple[int, str, float]:
    """Un sous-processus, sa sortie dans le journal du lot **au fil de l'eau**, et son code.

    ``PYTHONUNBUFFERED`` n'est pas un détail : sans lui, les lignes d'étape d'apply_delivery
    restent dans le tampon de 8 Ko de l'enfant et le journal ne dit rien de ce qui se passe.
    Un lot qu'on ne peut pas suivre est un lot qu'on ne peut pas diagnostiquer le lundi.
    """
    debut = time.monotonic()
    lignes: list[str] = []
    environnement = {**os.environ, "PYTHONUNBUFFERED": "1", **(env or {})}
    journal.parent.mkdir(parents=True, exist_ok=True)
    with journal.open("a", encoding="utf-8") as sortie:
        sortie.write(f"\n=== {maintenant()}  {' '.join(str(c) for c in commande)}\n")
        sortie.flush()
        processus = subprocess.Popen([str(c) for c in commande], stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT, cwd=ROOT, env=environnement,
                                     text=True, bufsize=1)
        for ligne in processus.stdout:
            lignes.append(ligne)
            sortie.write(ligne)
            sortie.flush()
        code = processus.wait()
    return code, "".join(lignes), round(time.monotonic() - debut, 1)


def dire(message: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)


# ------------------------------------------------------------------ garde-fous

def processus_lourds() -> list[tuple[int, str]]:
    """Les travaux lourds qui tournent déjà, hors de notre propre groupe de processus.

    Deux pièges, tous deux payés le 4 septembre. ``pgrep -f <script>.py`` rend un faux
    positif : le shell lanceur porte le nom du script. ``ps aux | grep`` rend le faux
    négatif inverse : il tronque les lignes longues. On lit donc ``ps -Awwo`` (largeur
    illimitée), on exige que **l'exécutable** soit un Python ou MinerU, et on exclut notre
    propre groupe — nos enfants sont voulus, pas concurrents.
    """
    sortie = subprocess.run(["ps", "-Awwo", "pid=,pgid=,command="],
                            capture_output=True, text=True).stdout
    notre_groupe = os.getpgrp()
    trouves = []
    for ligne in sortie.splitlines():
        morceaux = ligne.split(None, 2)
        if len(morceaux) < 3:
            continue
        try:
            pid, pgid = int(morceaux[0]), int(morceaux[1])
        except ValueError:
            continue
        commande = morceaux[2]
        if pgid == notre_groupe or pid == os.getpid():
            continue
        binaire = Path(commande.split()[0]).name.lower()
        if not any(binaire.startswith(prefixe) for prefixe in BINAIRES_PYTHON):
            continue
        if any(script in commande for script in SCRIPTS_LOURDS):
            trouves.append((pid, commande[:160]))
    return trouves


def disque_libre() -> int:
    return shutil.disk_usage(ROOT).free


def gio(octets: int) -> str:
    return f"{octets / 1024 ** 3:.1f} Gio"


def verrou_qdrant_libre() -> tuple[bool, str]:
    """Constat, pas conjecture : ``lsof`` sur le stockage embarqué.

    Un serveur MCP qui tourne sans avoir servi ne tient rien — il prend le verrou à son
    **premier appel d'outil**. On ne tue donc personne : ``apply_delivery`` constate le
    verrou à son étape 0 et échoue proprement plutôt qu'à moitié.
    """
    stockage = ROOT / "qdrant_storage_local"
    if not stockage.exists():
        return True, "stockage absent"
    resultat = subprocess.run(["lsof", "+D", str(stockage)], capture_output=True, text=True)
    lignes = [l for l in resultat.stdout.splitlines()[1:] if l.strip()]
    return (not lignes), (f"{len(lignes)} handle(s) : " + "; ".join(l.split()[0] for l in lignes[:4])
                          if lignes else "libre")


def verifier_environnement() -> list[str]:
    """Ce qui doit être vrai avant de commencer. Chaque manque est un blocage, pas un avis."""
    problemes = []
    for nom, chemin in (("venv de production", PY), ("venv MinerU", PY_MINERU),
                        ("venv GLiNER", PY_GLINER)):
        if not chemin.exists():
            problemes.append(f"{nom} absent : {chemin.relative_to(ROOT)}")
    if PY_MINERU.exists():
        resultat = subprocess.run(
            [str(PY_MINERU), "-c", "import importlib.metadata as m; print(m.version('mineru'))"],
            capture_output=True, text=True)
        version = resultat.stdout.strip()
        if version != "3.4.5":
            problemes.append(f"MinerU {version or 'introuvable'} dans .venv-mineru, 3.4.5 attendu — "
                             "la version entre dans document_id")
    cles = [p for p in (ROOT / "rag" / "benchmark" / ".google-key",
                        ROOT / "rag" / "benchmark" / ".mistral-key") if p.exists()]
    if not cles:
        problemes.append("aucune clé LLM (.google-key / .mistral-key) : le titrage échouera et "
                         "tout document serait écarté")
    libre = disque_libre()
    if libre < DISQUE_MINIMUM:
        problemes.append(f"disque : {gio(libre)} libres, minimum {gio(DISQUE_MINIMUM)}")
    return problemes


def prendre_verrou(lot: str, sec: bool) -> None:
    """Un seul driver à la fois. Un verrou dont le processus est mort est repris, pas subi."""
    ancien = lire_json(VERROU)
    if ancien:
        try:
            os.kill(int(ancien["pid"]), 0)
            vivant = True
        except PermissionError:                 # il existe, il ne nous appartient pas
            vivant = True
        except (ProcessLookupError, ValueError, TypeError, KeyError):
            vivant = False
        if vivant:
            sys.exit(f"REFUS : un driver tourne déjà (pid {ancien['pid']}, lot {ancien['lot']}, "
                     f"depuis {ancien['depuis']}). Pour l'arrêter : kill -TERM -{ancien['pid']}")
        dire(f"verrou périmé repris (pid {ancien['pid']} mort, lot {ancien['lot']}) — "
             "reprise après arrêt brutal")
    if not sec:
        ecrire_json(VERROU, {"pid": os.getpid(), "pgid": os.getpgrp(), "lot": lot,
                             "depuis": maintenant()})


def rendre_verrou() -> None:
    VERROU.unlink(missing_ok=True)


# ------------------------------------------------------------------ état par document

def chemin_etat(sha: str) -> Path:
    return ETAT / f"{sha[:16]}.json"


def charger_etat(pdf: Path, lot: str) -> dict:
    sha = sha256_fichier(pdf)
    chemin = chemin_etat(sha)
    doc = lire_json(chemin)
    if doc is None:
        doc = {"pdf": pdf.name, "sha256": sha, "lot": lot, "etat": "en-attente", "etape": None,
               "raison": None, "document_id": None, "delivery_id": None, "tentatives": [],
               "octets": pdf.stat().st_size, "cree": maintenant(), "journal": [], "mesures": {}}
        ecrire_json(chemin, doc)
    return doc


def transition(doc: dict, etat: str, etape: str | None, resultat: str,
               secondes: float | None = None, raison: str | None = None) -> dict:
    doc["etat"] = etat
    doc["etape"] = etape
    if raison is not None:
        doc["raison"] = raison
    doc["journal"].append({"a": maintenant(), "etape": etape, "etat": etat,
                           "resultat": resultat[:400], "secondes": secondes})
    ecrire_json(chemin_etat(doc["sha256"]), doc)
    ecrire_todolist(doc["lot"])
    return doc


def ranger_importe(doc: dict) -> None:
    """Un PDF entré sort de la file. ``entree/`` doit rester ce qu'il dit être.

    Son exemplaire durable est ``data/papers/`` (suivi en LFS) ; celui-ci n'est plus qu'un
    reste. Le laisser en place ferait, au lot suivant, une liste de « déjà ok, ignoré » qui
    grandit sans fin — et on ne saurait plus, d'un coup d'œil, ce qui attend vraiment.
    """
    source = ENTREE / doc["pdf"]
    if not source.exists():
        return
    IMPORTES.mkdir(parents=True, exist_ok=True)
    cible = IMPORTES / doc["pdf"]
    if cible.exists():
        cible.unlink()
    shutil.move(str(source), str(cible))
    doc["range_vers"] = str(cible.relative_to(ROOT))
    ecrire_json(chemin_etat(doc["sha256"]), doc)


def ecarter(doc: dict, etape: str, raison: str, sec: bool) -> dict:
    """Le document sort du lot avec sa raison, et le suivant commence. Rien d'autre."""
    dire(f"  ÉCARTÉ ({etape}) : {raison[:150]}")
    if not sec:
        ECARTES.mkdir(parents=True, exist_ok=True)
        source = ENTREE / doc["pdf"]
        if source.exists():
            cible = ECARTES / doc["pdf"]
            if cible.exists():
                cible.unlink()
            shutil.move(str(source), str(cible))
            doc["ecarte_vers"] = str(cible.relative_to(ROOT))
    return transition(doc, "écarté", etape, raison, raison=raison)


# ------------------------------------------------------------------ constats sur le monde

def journal_import(delivery_id: str) -> dict | None:
    return lire_json(JOURNAL_IMPORT / f"{delivery_id}.json")


def registre_connait(delivery_id: str, document_ids: list[str]) -> bool:
    """Le registre porte-t-il déjà cet import ? C'est l'étape 9 d'``apply_delivery``.

    Un ``kill -9`` entre la promotion (étape 8) et le registre (étape 9) laisse un journal
    ``applied`` et un registre muet : le document serait attribué à ``corpus-initial``, donc
    absent d'``imported-rows.jsonl``, donc invisible au graphe. Sans un mot.
    """
    donnees = lire_json(REGISTRY)
    if not donnees:
        return False
    if not any(d.get("id") == delivery_id for d in donnees.get("deliveries", [])):
        return False
    par_id = {d["document_id"]: d for d in donnees.get("documents", [])}
    return all((par_id.get(i) or {}).get("delivery", {}).get("id") == delivery_id
               for i in document_ids)


# ------------------------------------------------------------------ étapes

def source_declaree(nom: str) -> str:
    declarees = lire_json(SOURCES) or {}
    return declarees.get(nom) or f"posé dans entree/, provenance non déclarée, vu {maintenant()}"


def etape_parse(doc: dict, journal: Path, sec: bool) -> tuple[bool, str]:
    livraison = LIVRAISONS / doc["delivery_id"]
    dossier = livraison / "processed" / f"doc-{doc['sha256'][:16]}"
    production = lire_json(livraison / "production.json")
    if production and all((dossier / n).exists() for n in
                          ("document.json", "blocks.jsonl", "chunks.jsonl", "parents.jsonl")):
        entree = next((d for d in production["documents"] if d["sha256"] == doc["sha256"]), None)
        if entree:
            doc["document_id"] = entree["document_id"]
            doc["mesures"]["parse"] = entree["counts"] | {"pages": entree["pages"]}
            doc["mesures"]["filename"] = entree["filename"]
            return True, "déjà produit (reprise)"
    code, sortie, secondes = courir(
        [PY_MINERU, HERE / "parse_local.py", ENTREE / doc["pdf"], "--id", doc["delivery_id"],
         "--out", livraison, "--source", f"{doc['pdf']}={source_declaree(doc['pdf'])}"], journal)
    if code != 0:
        refus = next((l for l in sortie.splitlines() if l.startswith("REFUS")), sortie.strip()[-300:])
        return False, refus
    production = lire_json(livraison / "production.json") or {"documents": []}
    entree = next((d for d in production["documents"] if d["sha256"] == doc["sha256"]), None)
    if not entree:
        return False, "production.json ne contient pas ce sha256 — production incohérente"
    doc["document_id"] = entree["document_id"]
    doc["mesures"]["parse"] = entree["counts"] | {"pages": entree["pages"], "secondes": secondes}
    doc["mesures"]["filename"] = entree["filename"]
    return True, (f"{entree['counts']['chunks']} chunks dont "
                  f"{entree['counts']['eligible_chunks']} éligibles, {secondes} s")


def etape_releve(doc: dict, journal: Path, sec: bool) -> tuple[bool, str]:
    livraison = LIVRAISONS / doc["delivery_id"]
    code, sortie, secondes = courir(
        [PY, HERE / "generate_manifest.py", livraison, "--id", doc["delivery_id"]], journal)
    if code != 0:
        return False, sortie.strip()[-300:]
    releve = dernier_json(sortie) or {}
    doc["mesures"]["releve"] = {k: releve.get(k) for k in ("documents", "chunks", "eligible_chunks")}
    return True, f"relevé scellé ({releve.get('chunks')} chunks)"


def etape_diagnostic(doc: dict, journal: Path, sec: bool) -> tuple[bool, str]:
    """Le seul juge. Tout ce qui n'est pas un ``ajout`` net et admissible sort du lot."""
    livraison = LIVRAISONS / doc["delivery_id"]
    rapport_json = livraison / "rapport.json"
    code, sortie, secondes = courir(
        [PY, HERE / "inspect_delivery.py", livraison, "--json", rapport_json], journal)
    rapport = lire_json(rapport_json)
    if rapport is None:
        return False, f"inspect_delivery n'a rien rendu (code {code})"
    entrees = [d for d in rapport["documents"] if d.get("sha256") == doc["sha256"]]
    doc["mesures"]["diagnostic"] = {"decision": rapport["decision"], "counts": rapport["counts"],
                                    "schema_drift": rapport["schema"],
                                    "gold_chunks_at_risk":
                                        rapport["impact"]["benchmark"]["gold_chunks_at_risk"]}
    if rapport["blocking"]:
        return False, "bloquant : " + " ; ".join(rapport["blocking"][:3])
    if len(entrees) != 1:
        return False, f"{len(entrees)} entrée(s) pour ce sha256 dans le rapport, 1 attendue"
    entree = entrees[0]
    doc["mesures"]["identite"] = entree["identity"]
    doc["mesures"]["near_duplicates"] = entree["near_duplicates"]
    if not entree.get("admissible"):
        return False, f"refusé : {' ; '.join(entree['problems'][:3])}"
    if entree["identity"] != "ajout":
        return False, (f"verdict « {entree['identity']} » — ce lot n'accepte que les ajouts. "
                       + " ".join(entree["notes"])[:200])
    if entree.get("needs_edition_decision"):
        proches = ", ".join(entree["needs_edition_decision"])
        containment = max((h["containment"] for h in entree["near_duplicates"]), default=None)
        return False, (f"édition proche de {proches} (containment {containment}) — décision "
                       "explicite exigée, jamais automatique")
    return True, f"ajout, {entree['eligible_chunks']} chunks éligibles, aucun quasi-doublon"


def prochaine_tentative(doc: dict) -> str:
    base = doc["delivery_id"]
    if not doc["tentatives"]:
        return base
    return f"{base}r{len(doc['tentatives']) + 1}"


def etape_reprise(doc: dict, journal: Path, llm: str) -> tuple:
    """Réconcilier le monde **avant** de rejuger quoi que ce soit.

    Le diagnostic ne vaut que sur un corpus que l'import n'a pas déjà touché. Un document
    promu dans ``ingested/`` puis abandonné par un ``kill -9`` ressort « sans-effet » :
    ``inspect_delivery`` compare le ``chunks.jsonl`` livré à celui du corpus, ils sont
    identiques, et le verdict est exact. Il est aussi hors sujet — et le lot écartait alors
    un document **déjà servi**, PDF déplacé dans ``ecartes/``, pendant que 43 points
    répondaient aux requêtes sans que le registre les connaisse. Mesuré le 4 septembre 2026
    au deuxième kill-test ; c'est la raison d'être de cette fonction.

    Rend ``("importé", …)`` si l'import est déjà appliqué (réparé au besoin), ``("neuf", …)``
    s'il n'y a rien à reprendre, ``(False, raison)`` si la réparation est impossible.
    """
    applique = None
    for tentative in doc["tentatives"]:
        precedent = journal_import(tentative)
        if precedent and precedent.get("state") == "applied":
            applique = (tentative, precedent)

    if applique:
        tentative, precedent = applique
        identifiants = [d["document_id"] for d in precedent["documents"]]
        # Le document est **dans la collection** : quoi qu'il arrive ensuite, l'écarter
        # serait mentir. Le drapeau le dit à ``traiter``.
        doc["mesures"]["deja_dans_le_corpus"] = True
        repare = "registre déjà à jour"
        if not registre_connait(tentative, identifiants):
            dire(f"  réparation : {tentative} promu mais absent du registre (mort entre les "
                 "étapes 8 et 9 d'apply_delivery) — registre, relais et BM25 refaits")
            code, sortie, _ = courir([PY, MOI, "--reparer", tentative], journal)
            if code != 0:
                return False, f"réparation impossible : {sortie.strip()[-300:]}"
            resume = dernier_json(sortie) or {}
            doc["mesures"]["reparation"] = resume
            repare = (f"registre et relais reconstruits ({resume.get('imported_rows')} lignes), "
                      f"BM25 {resume.get('bm25_records')} records")
        # **Pas** ``doc["delivery_id"] = tentative`` : ce champ nomme le *répertoire de
        # livraison* que parse_local a produit, et les tentatives d'import (…r2, …r3) n'en
        # ont pas. L'écraser ferait chercher une livraison qui n'existe pas à la relance
        # suivante — et le document repartirait d'un parse.
        doc["import_applique"] = tentative
        etat = etat_du_corpus()
        doc["mesures"]["import"] = {"point_ids": precedent["point_ids"],
                                    "collection": etat.get("chunks"),
                                    "signature": etat.get("signature"),
                                    "bm25": etat.get("bm25"), "reprise": True,
                                    "llm_titrage": llm}
        doc["mesures"]["metadonnees"] = metadonnees_du_document(doc["document_id"])
        # ``apply_delivery`` efface son staging à la toute fin ; mort avant, il le laisse.
        # 688 Ko par document abandonné, hors du corpus servi mais pas hors du disque.
        reste = INCOMING / tentative
        if reste.exists():
            shutil.rmtree(reste)
        return "importé", f"import {tentative} déjà appliqué — {repare}"

    annulations = []
    for tentative in doc["tentatives"]:
        precedent = journal_import(tentative)
        if precedent and precedent.get("state") == "in-flight":
            dire(f"  tentative {tentative} laissée en vol (watermark "
                 f"{precedent['watermark']}) — annulation avant toute réécriture")
            code, sortie, _ = courir([PY, HERE / "apply_delivery.py", "--rollback", tentative], journal)
            if code != 0:
                return False, f"annulation impossible pour {tentative} : {sortie.strip()[-300:]}"
            resume = dernier_json(sortie) or {}
            annulations.append({"delivery_id": tentative,
                                "points_removed": resume.get("points_removed"),
                                "collection": resume.get("collection")})
        # Un essai mort **avant** le journal n'a rien écrit dans Qdrant — le journal précède
        # le premier upsert, c'est l'invariant sur lequel tout ceci repose. Reste le staging.
        reste = INCOMING / tentative
        if reste.exists():
            shutil.rmtree(reste)
    if annulations:
        doc["mesures"]["annulations"] = annulations
        return "neuf", f"{len(annulations)} tentative(s) annulée(s) : " + json.dumps(
            annulations, ensure_ascii=False)
    return "neuf", "rien à reprendre"


def etape_import(doc: dict, journal: Path, llm: str, sec: bool) -> tuple[bool, str]:
    """L'unique écriture du lot. Ce qu'une mort brutale a laissé est déjà réconcilié
    par ``etape_reprise`` — ici, on n'ouvre qu'une tentative neuve."""
    livraison = LIVRAISONS / doc["delivery_id"]
    tentative = prochaine_tentative(doc)
    doc["tentatives"].append(tentative)
    ecrire_json(chemin_etat(doc["sha256"]), doc)
    commande = [PY, HERE / "apply_delivery.py", livraison, "--apply", "--id", tentative]
    # ``--llm sans-llm`` : titrage sans aucun appel externe. Le titre vient alors des
    # métadonnées embarquées du PDF (``pdf_embedded``) ou de ``overrides.json``
    # (``manual``) — deux provenances que ``TITLE_SOURCES`` accepte déjà. Un document
    # dont aucune des deux ne répond est écarté par la garde, comme il doit l'être :
    # ce mode ne relâche rien, il retire seulement le recours au LLM.
    if llm == SANS_LLM:
        commande.append("--no-llm")
    if RECONSTRUCTION == "fin-de-lot":
        commande.append("--sans-bm25")
    code, sortie, secondes = courir(commande, journal,
                                    env={"QUANT_RAG_METADATA_LLM": llm})
    if code != 0:
        reste = INCOMING / tentative
        if reste.exists():
            shutil.rmtree(reste)
        refus = next((l for l in sortie.splitlines() if l.startswith("REFUS")), sortie.strip()[-400:])
        # un échec après le journal a pu écrire des points : on annule avant de passer au suivant
        # ``pre-flight`` compte autant qu'``in-flight`` : depuis le 9 septembre 2026 le
        # journal s'ouvre AVANT les métadonnées et l'overlay des tableaux, donc un import
        # mort avant le premier upsert a déjà écrit dans le corpus servi. Ne rattraper que
        # `in-flight` laisserait ces lignes orphelines — c'est le défaut §11.2 lui-même.
        if (journal_import(tentative) or {}).get("state") in ("in-flight", "pre-flight"):
            courir([PY, HERE / "apply_delivery.py", "--rollback", tentative], journal)
            refus += "  [tentative annulée, corpus rendu à son état d'avant]"
        return False, refus
    resultat = dernier_json(sortie) or {}
    if resultat.get("status") != "COMPLETED":
        return False, f"apply_delivery rend {resultat.get('status')!r}"
    doc["import"] = resultat
    doc["mesures"]["import"] = {"point_ids": resultat["point_ids"], "collection": resultat["collection"],
                                "signature": resultat["corpus_signature"], "bm25": resultat["bm25"],
                                "secondes": secondes, "llm_titrage": llm,
                                # Le détail interne de l'import : sept étapes chronométrées,
                                # les appels et jetons du titrage, les octets écrits. Mesuré
                                # PAR apply_delivery, pas déduit de l'extérieur — un seul
                                # chiffre pour dix gestes de coûts très différents avait fait
                                # attribuer 61 % d'un lot à build_graph, qui coûte 1,3 s.
                                "cout": resultat.get("cout")}
    doc["mesures"]["metadonnees"] = metadonnees_du_document(doc["document_id"])
    return True, (f"{resultat['points']} points {resultat['point_ids']['first']}.."
                  f"{resultat['point_ids']['last']}, signature {resultat['corpus_signature']}")


def metadonnees_du_document(document_id: str) -> dict:
    """Le titre retenu et **d'où il vient** — la seule chose qu'un rapport de lot doit dire.

    On le lit dans le fichier de métadonnées plutôt que dans les traces d'``apply_delivery`` :
    une ligne de journal se reformate, un champ de provenance non. La politique du corpus
    (``TITLE_SOURCES``) n'accepte que ``llm_first_page``, ``llm_deep_pages``, ``pdf_embedded``,
    ``arxiv_stamp`` ou ``manual`` — un titre venu d'un nom de fichier fait échouer l'import.
    """
    donnees = lire_json(ROOT / "rag" / "metadata" / "documents-metadata-v1.json") or {}
    for record in donnees.get("documents", []):
        if record.get("document_id") == document_id:
            return {"title": record.get("title"), "short_ref": record.get("short_ref"),
                    "authors": record.get("authors"), "year": record.get("publication_year"),
                    "provenance": (record.get("provenance") or {}).get("title"),
                    "provenance_annee": (record.get("provenance") or {}).get("year")}
    return {}


def etape_aval(doc: dict, journal: Path, sec: bool) -> tuple[bool, str]:
    """Registre, relais, et les deux contrôles qui disent si le corpus servi est cohérent."""
    code, sortie, _ = courir([PY, HERE / "registry.py", "--build"], journal)
    if code != 0:
        return False, f"registry --build : {sortie.strip()[-300:]}"
    construit = dernier_json(sortie) or {}
    code, sortie, _ = courir([PY, HERE / "registry.py", "--verify"], journal)
    verification = dernier_json(sortie) or {}
    if code != 0 or verification.get("status") != "OK":
        return False, f"registry --verify : {json.dumps(verification, ensure_ascii=False)[:300]}"
    # ``check_bm25.py`` ne contrôle pas l'index, il le **reconstruit** avant de le comparer
    # (``check_bm25`` appelle ``quant_rag.rebuild_bm25()``). C'est la deuxième
    # reconstruction de 50,8 Mo du même document, après celle d'``apply_delivery._apply``.
    # En régime fin de lot, elle est reportée avec l'autre — et le message le dit, pour que
    # le commit n'affirme pas « check_bm25 OK » sur un contrôle qui n'a pas eu lieu.
    if RECONSTRUCTION == "fin-de-lot":
        doc["mesures"]["aval"] = {"registre": construit.get("totals") or construit.get("documents"),
                                  "imported_rows": construit.get("imported_rows"),
                                  "digest": construit.get("digest"),
                                  "verify": verification.get("status"),
                                  "check_bm25": "différé (fin-de-lot)", "secondes": 0.0}
        return True, (f"registre {construit.get('documents')} documents, "
                      f"{construit.get('imported_rows')} lignes de relais, "
                      f"check_bm25 différé (fin-de-lot)")
    code, sortie, secondes = courir([PY, ROOT / "rag" / "benchmark" / "check_bm25.py"], journal)
    if code != 0:
        return False, f"check_bm25 en échec : {sortie.strip()[-300:]}"
    doc["mesures"]["aval"] = {"registre": construit.get("totals") or construit.get("documents"),
                              "imported_rows": construit.get("imported_rows"),
                              "digest": construit.get("digest"),
                              "verify": verification.get("status"), "check_bm25": "OK",
                              "secondes": secondes}
    return True, (f"registre {construit.get('documents')} documents, "
                  f"{construit.get('imported_rows')} lignes de relais, check_bm25 OK")


def etape_graphe(doc: dict, journal: Path, sec: bool) -> tuple[bool, str]:
    """GLiNER reprend où il s'est arrêté : seuls les chunks nouveaux sont extraits.

    Par tranches de ``TRANCHE_EXTRACTION`` chunks, une par processus : voir le commentaire
    de la constante. La boucle s'arrête sur ``COMPLETED``, et refuse de tourner à vide —
    une tranche qui ne traite rien alors qu'il reste des chunks est une panne, pas une
    fin de travail.
    """
    if SANS_GRAPHE:
        # Sautée, et l'état du lot le dit : un « graphe » absent du rapport se lirait comme
        # un graphe à jour, ce qui est précisément le contraire de la vérité.
        doc["mesures"]["graphe"] = {"etape": "sautée", "motif": "--sans-graphe",
                                    "consequence": "search_graph, expand_entity et "
                                                   "connect_entities ne reflètent pas ce document",
                                    "secondes": 0.0}
        return True, ("sautée (--sans-graphe) — les trois outils MCP de graphe ne "
                      "refléteront pas ce document tant qu'un lot ne l'aura pas rattrapé")

    extraction: dict = {}
    secondes = 0.0
    for tranche in range(1, TRANCHES_MAX + 1):
        code, sortie, s = courir([PY_GLINER, ROOT / "rag" / "graph" / "extract_entities.py",
                                  "--limit", str(TRANCHE_EXTRACTION),
                                  "--batch-size", str(LOT_EXTRACTION)], journal)
        secondes += s
        if code != 0:
            return False, f"extract_entities : {sortie.strip()[-300:]}"
        extraction = dernier_json(sortie) or {}
        if extraction.get("status") == "COMPLETED":
            break
        if not extraction.get("processed_this_run"):
            return False, (f"extract_entities n'avance plus après {tranche} tranche(s) : "
                           f"{extraction.get('completed')}/{extraction.get('total')} chunks")
        dire(f"    extraction {extraction.get('completed')}/{extraction.get('total')} "
             f"(tranche {tranche}, {extraction.get('processed_this_run')} chunks)")
    else:
        return False, (f"{TRANCHES_MAX} tranches d'extraction sans arriver au bout : "
                       f"{extraction.get('completed')}/{extraction.get('total')} chunks")
    # L'extraction, elle, N'EST JAMAIS différée — voir le commentaire de ``RECONSTRUCTIONS`` :
    # l'ordre de ``results.jsonl`` décide vers quel nœud pointent les arêtes de relation.
    # Seule la reconstruction du graphe, qui est une fonction pure de ce fichier et coûte
    # 1,3 s, est reportée.
    if RECONSTRUCTION == "fin-de-lot":
        doc["mesures"]["graphe"] = {
            "extraction": {k: extraction.get(k) for k in
                           ("status", "completed", "total", "documents", "processed_this_run",
                            "errors_this_run", "chunks_per_second")},
            "graphe": "différé (fin-de-lot)",
            "secondes": round(secondes, 1)}
        return True, (f"extraction {extraction.get('completed')}/{extraction.get('total')}, "
                      f"graphe différé (fin-de-lot)")
    code, sortie, secondes_graphe = courir([PY_GLINER, ROOT / "rag" / "graph" / "build_graph.py"], journal)
    if code != 0:
        return False, f"build_graph : {sortie.strip()[-300:]}"
    graphe = dernier_json(sortie) or {}
    doc["mesures"]["graphe"] = {
        "extraction": {k: extraction.get(k) for k in
                       ("status", "completed", "total", "documents", "processed_this_run",
                        "errors_this_run", "chunks_per_second")},
        "graphe": {k: graphe.get(k) for k in ("nodes", "edges", "documents", "sample_chunks")},
        "signature": (graphe.get("corpus") or {}).get("signature"),
        "secondes": round(secondes + secondes_graphe, 1)}
    return True, (f"extraction {extraction.get('completed')}/{extraction.get('total')}, "
                  f"graphe reconstruit en {round(secondes + secondes_graphe)} s")


def phrase_de_contenu(doc: dict) -> str | None:
    """Une phrase prise **au milieu** du document, pour sonder son contenu et non son nom.

    Au milieu, délibérément : le premier chunk est la page de titre, qui reproduit ce que la
    sonde de titre demande déjà. On cherche une phrase d'au moins douze mots dans un chunk
    éligible hors tableau, tronquée à trente — assez longue pour être distinctive, assez
    courte pour rester une requête.
    """
    dossier = (LIVRAISONS / doc["delivery_id"] / "processed" / f"doc-{doc['sha256'][:16]}")
    fichier = dossier / "chunks.jsonl"
    if not fichier.exists():
        return None
    eligibles = []
    for ligne in fichier.open(encoding="utf-8"):
        if not ligne.strip():
            continue
        chunk = json.loads(ligne)
        if chunk.get("rag_eligible") is True and chunk.get("content_type") != "table":
            eligibles.append(chunk.get("text") or "")
    if not eligibles:
        return None
    # Une phrase de prose, pas une formule. Le preprint de Cont, Stoikov & Talreja a rendu
    # « $\tilde{X}_{A}(t), \tilde{X}_{B}(t), t \geq T$ follow independent birth-death » : la
    # sonde a tout de même trouvé le document, mais interroger le corpus en LaTeX ne prouve
    # pas grand-chose de son contenu. On exige donc une majorité de mots alphabétiques.
    def prose(mots: list[str]) -> bool:
        lisibles = sum(1 for m in mots if m.replace("-", "").isalpha())
        return len(mots) >= 12 and lisibles >= 0.7 * len(mots)

    for exigeante in (True, False):
        for texte in eligibles[len(eligibles) // 2:] + eligibles:
            for phrase in re.split(r"(?<=[.!?])\s+", texte):
                mots = phrase.split()
                if len(mots) >= 12 and (prose(mots) if exigeante else True):
                    return " ".join(mots[:30])
    return None


def etape_sondes(doc: dict, journal: Path, sec: bool) -> tuple[bool, str]:
    """Le document est-il vraiment servi ? On le demande au corpus, pas au journal.

    Deux sondes et une règle : le document doit être atteignable par **au moins l'une** des
    deux. Le titre échoue seul quand il est trop commun — c'est un défaut de la sonde, pas du
    corpus, et le message le dit. Les deux qui échouent, c'est un document écrit et
    introuvable : là, l'étape échoue.
    """
    # En régime fin de lot, les sondes sont reportées — et ce n'est pas un confort.
    # ``mode_interroger`` passe par ``quant_rag._lexical()``, donc par ``bm25()``, qui **se
    # répare tout seul** : si le manifeste ne correspond plus à la collection, il appelle
    # ``rebuild_bm25()``. Sonder après un import ``--sans-bm25``
    # reconstruirait donc les 50,8 Mo dans le sous-processus de sonde, et le régime
    # n'économiserait **rien** — il déplacerait la dépense en la rendant invisible.
    # Les sondes sont rejouées sur l'état de fin de lot, où elles ont plus de sens : elles
    # constatent alors ce que le corpus sert vraiment, index reconstruit compris.
    if RECONSTRUCTION == "fin-de-lot":
        doc["mesures"]["sondes"] = {"statut": "différées (fin-de-lot)"}
        return True, "sondes différées (fin-de-lot)"
    return sonder(doc, journal)


def sonder(doc: dict, journal: Path) -> tuple[bool, str]:
    """Le corps des sondes, sans la question du régime — pour que la reconstruction de fin
    de lot puisse les rejouer telles quelles, sans toucher au global ``RECONSTRUCTION``."""
    titre = (doc["mesures"].get("metadonnees", {}).get("title")
             or doc["mesures"].get("filename") or doc["pdf"])
    contenu = phrase_de_contenu(doc)
    commande = [PY, MOI, "--interroger", doc["document_id"], "--titre", titre]
    if contenu:
        commande += ["--contenu", contenu]
    code, sortie, secondes = courir(commande, journal)
    sondes = dernier_json(sortie)
    if code != 0 or not sondes:
        return False, f"sondes impossibles : {sortie.strip()[-300:]}"
    doc["mesures"]["sondes"] = sondes
    if not sondes["points"]:
        return False, "aucun point dans la collection pour ce document_id"
    par_titre = sondes["rang_dense"] is not None
    par_contenu = sondes.get("rang_dense_contenu") is not None
    if not par_titre and not par_contenu:
        return False, (f"le document ne sort ni sur son titre ({titre[:50]!r}) ni sur une phrase "
                       f"de son propre texte — il est écrit mais il n'est pas trouvé")
    if not par_titre:
        return True, (f"{sondes['points']} points · **titre introuvable** ({titre[:40]!r} est trop "
                      f"commun) mais contenu trouvé en dense rang {sondes['rang_dense_contenu']} "
                      f"({sondes['score_dense_contenu']}) · BM25 contenu rang "
                      f"{sondes.get('rang_lexical_contenu')} · graphe {sondes['chunks_graphe']} chunks")
    contenu_dit = (f" · contenu rang {sondes['rang_dense_contenu']}" if par_contenu
                   else " · contenu non sondé" if not contenu else " · contenu hors des 10")
    return True, (f"{sondes['points']} points · dense rang {sondes['rang_dense']} "
                  f"({sondes['score_dense']}) · BM25 rang {sondes['rang_lexical']}{contenu_dit} · "
                  f"graphe {sondes['chunks_graphe']} chunks")


#: Ce qu'un import réussi ajoute au dépôt. ``data/papers`` n'y est **pas** en bloc : un
#: dry-run y dépose déjà le PDF (``parse_local.place_pdf``), et un ``git add data/papers``
#: committerait des PDF de documents jamais importés. Seul le fichier du document en cours
#: est ajouté, nommément.
CHEMINS_COMMIT = (
    "data/processed/ingested", "data/lexical", "rag/ingestion/source-b/sources.json",
    "rag/ingestion/registry-v1.json", "rag/ingestion/imported-rows.jsonl",
    "rag/ingestion/journal", "rag/ingestion/source-b/etat",
    "rag/metadata/documents-metadata-v1.json", "rag/metadata/overrides.json",
    "rag/metadata/results-apply-metadata-v1.json", "rag/tables/tables-markdown-v1.json",
    # Le nom porte le LABEL du corpus, il ne se code pas en dur. Corrigé le 8 septembre 2026 :
    # la constante nommait `results-bm25-dedup-tables-registry-v1.json` alors que
    # `check_bm25.OUTPUT` vaut `results-bm25-{corpus_overlay.LABEL}.json`, et que LABEL vaut
    # `dedup-tables-registry-titles-v1` depuis l'entrée des titres consolidés. Le lot
    # committait donc, à chaque document, un rapport que plus rien ne mettait à jour — et
    # laissait le rapport réel hors du commit. C'est la classe de défaut que tout le nommage
    # par signature existe pour empêcher : un fichier juste sous un nom qui ment.
    f"rag/benchmark/results-bm25-{LABEL_CORPUS}.json",
)


def etape_commit(doc: dict, journal: Path, sec: bool) -> tuple[bool, str]:
    """Un commit par import réussi. Le graphe en est exclu, et c'est délibéré.

    ``data/graph/graph-lite.json`` fait 59 Mo sur une seule ligne et n'est pas suivi en LFS :
    un commit par document en ajouterait un exemplaire entier à chaque fois. Le graphe est
    donc reconstruit à chaque document — il est servi — mais committé une fois, à la fin du
    lot. Un crash ne coûte alors qu'un ``build_graph.py``, pas un import.
    """
    mesures = doc["mesures"]
    meta = mesures.get("metadonnees", {})
    identite = meta.get("short_ref") or mesures.get("filename") or doc["pdf"]
    imp = mesures.get("import", {})
    sondes = mesures.get("sondes", {})
    message = (
        f"feat(corpus): {identite[:60]} — {imp.get('point_ids', {}).get('count')} points entrés par le lot\n"
        f"\n"
        f"  {mesures.get('filename')} · {mesures.get('parse', {}).get('pages')} p. · "
        f"sha256 {doc['sha256'][:16]}\n"
        f"  titre retenu : {meta.get('title')!r}\n"
        f"  provenance du titre : {meta.get('provenance')} · année {meta.get('year')} "
        f"({meta.get('provenance_annee')})\n"
        f"  provenance déclarée : {source_declaree(doc['pdf'])}\n"
        f"\n"
        f"  points {imp.get('point_ids', {}).get('first')}..{imp.get('point_ids', {}).get('last')} · "
        f"collection {imp.get('collection')} · signature {imp.get('signature')}\n"
        f"  {mesures.get('parse', {}).get('chunks')} chunks dont "
        f"{mesures.get('parse', {}).get('eligible_chunks')} éligibles\n"
        f"\n"
        f"Diagnostic : ajout, aucun refus, aucun quasi-doublon, gold_chunks_at_risk "
        f"{mesures.get('diagnostic', {}).get('gold_chunks_at_risk')}.\n"
        f"Aval : registre --verify OK, check_bm25 "
        f"{mesures.get('aval', {}).get('check_bm25', 'OK')}, "
        f"{mesures.get('aval', {}).get('imported_rows')} lignes de relais.\n"
        # Un commit qui affirmerait des rangs que personne n'a mesurés serait un artefact
        # juste sous un nom qui ment — exactement ce que le nommage par signature interdit.
        + (f"Sondes : {sondes['statut']} — elles sont rejouées, avec l'index et le graphe, "
           f"à la reconstruction de fin de lot.\n" if sondes.get("statut") else
           f"Sondes : dense rang {sondes.get('rang_dense')} ({sondes.get('score_dense')}), "
           f"BM25 rang {sondes.get('rang_lexical')}, graphe {sondes.get('chunks_graphe')} chunks.\n")
        +
        f"\n"
        f"Lot {doc['lot']}, titrage {imp.get('llm_titrage')} (Mistral à plafond nul).\n"
        f"Produit sans supervision par rag/ingestion/batch_driver.py.\n"
        f"\n"
        f"Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>\n"
        f"Claude-Session: https://claude.ai/code/session_011u1XwL7fH8RFpFpNnrpSbs\n")
    chemins = [ROOT / c for c in CHEMINS_COMMIT]
    chemins.append(ROOT / "data" / "papers" / (mesures.get("filename") or doc["pdf"]))
    chemins.append(SOURCE_B / f"TODOLIST-{doc['lot']}.md")
    # ``-u`` et pas ``-A`` sur la file : il enregistre la **sortie** des PDF suivis (importés
    # dans importes/, écartés dans ecartes/) sans committer ceux qui attendent encore leur
    # tour. Un PDF téléchargé par le lot n'a rien à faire deux fois en LFS : son exemplaire
    # durable est data/papers/.
    courir(["git", "add", "-u", "--", str(ENTREE)], journal)
    courir(["git", "add", "-A", "--"] + [str(c) for c in chemins if c.exists()], journal)
    code, sortie, _ = courir(["git", "diff", "--cached", "--name-only"], journal)
    if not sortie.strip():
        return True, "rien à committer (état déjà enregistré)"
    fichiers = len(sortie.strip().splitlines())
    code, sortie, _ = courir(["git", "commit", "-m", message], journal)
    if code != 0:
        return False, f"git commit : {sortie.strip()[-300:]}"
    code, revision, _ = courir(["git", "rev-parse", "--short", "HEAD"], journal)
    # Le push suit chaque import : c'est le seul moyen, pour qui dort, de voir la nuit
    # avancer depuis un autre poste. Il ne commande rien — un dépôt injoignable est une
    # panne de réseau, pas une panne de corpus, et le lot continue avec la mention au
    # rapport.
    code_push, sortie_push, secondes_push = courir(["git", "push", "origin", "HEAD"], journal)
    push = "OK" if code_push == 0 else f"ÉCHEC : {sortie_push.strip()[-200:]}"
    doc["mesures"]["commit"] = {"revision": revision.strip(), "fichiers": fichiers,
                                "push": push, "secondes_push": secondes_push}
    return True, f"{revision.strip()} ({fichiers} fichiers) · push {push[:40]}"


# ------------------------------------------------------------------ modes internes

def mode_interroger(document_id: str, titre: str, contenu: str | None = None) -> int:
    """Sonde le corpus servi — dans un sous-processus, pour ne jamais garder le verrou Qdrant.

    Le driver n'ouvre jamais Qdrant lui-même : il vivrait alors le temps du lot et
    ``apply_delivery`` échouerait à son étape 0. Ici, le verrou meurt avec le processus.

    **Deux sondes, deux questions différentes.** La sonde de *titre* demande « ce document
    est-il reconnaissable par son nom ? » ; la sonde de *contenu*, tirée d'une phrase du
    document lui-même, demande « son texte est-il seulement atteignable ? ». Confondre les
    deux fait passer pour une panne ce qui n'est parfois qu'un titre trop commun : le NBER
    w19325 s'appelle « Carry », un seul mot, et aucune politique ne permet de l'allonger.
    Un document dont le contenu est introuvable est un vrai problème ; un titre d'un mot qui
    ne sort pas premier n'en est pas un — et seules deux sondes séparent ces deux cas.
    """
    sys.path.insert(0, str(ROOT / "rag"))
    import quant_rag
    from qdrant_client import models

    client = quant_rag.client()
    filtre = models.Filter(must=[models.FieldCondition(
        key="document_id", match=models.MatchValue(value=document_id))])
    points = client.count(quant_rag.COLLECTION, exact=True, count_filter=filtre).count
    total = client.count(quant_rag.COLLECTION, exact=True).count

    dense = quant_rag.search_explained(titre, limit=10, mode="dense", log=False)["results"]
    rang_dense = next((i + 1 for i, r in enumerate(dense) if r["document_id"] == document_id), None)
    score = next((round(r["score"], 4) for r in dense if r["document_id"] == document_id), None)
    lexical = quant_rag._lexical(titre, 10, None)     # BM25 seul : prouve l'index lexical
    rang_lexical = next((i + 1 for i, r in enumerate(lexical) if r["document_id"] == document_id), None)

    sonde_contenu = {}
    if contenu:
        dense_c = quant_rag.search_explained(contenu, limit=10, mode="dense", log=False)["results"]
        lexical_c = quant_rag._lexical(contenu, 10, None)
        sonde_contenu = {
            "sonde_contenu": contenu[:160],
            "rang_dense_contenu": next((i + 1 for i, r in enumerate(dense_c)
                                        if r["document_id"] == document_id), None),
            "score_dense_contenu": next((round(r["score"], 4) for r in dense_c
                                         if r["document_id"] == document_id), None),
            "rang_lexical_contenu": next((i + 1 for i, r in enumerate(lexical_c)
                                          if r["document_id"] == document_id), None)}

    chunks_graphe = 0
    if GLINER_RESULTS.exists():
        with GLINER_RESULTS.open(encoding="utf-8") as handle:
            for ligne in handle:
                if not ligne.strip():
                    continue
                try:
                    if json.loads(ligne).get("document_id") == document_id:
                        chunks_graphe += 1
                except json.JSONDecodeError:
                    continue
    print(json.dumps({"document_id": document_id, "sonde": titre[:120], "points": points,
                      "collection": total, "rang_dense": rang_dense, "score_dense": score,
                      "rang_lexical": rang_lexical, **sonde_contenu,
                      "chunks_graphe": chunks_graphe},
                     ensure_ascii=False))
    return 0


def mode_reparer(delivery_id: str) -> int:
    """Refait les étapes 9 et 10 d'``apply_delivery`` pour un import promu mais inachevé.

    Le journal passe à ``applied`` à la fin de l'étape 8 (promotion), avant le registre (9)
    et BM25 (10). Une mort brutale dans cet intervalle laisse des points servis, des dossiers
    promus, et un registre qui attribue le document à ``corpus-initial`` — donc un relais
    ``imported-rows.jsonl`` qui l'ignore, donc un graphe aveugle. Sans un mot.
    """
    for extra in (str(HERE), str(ROOT / "rag"), str(ROOT / "src")):
        if extra not in sys.path:
            sys.path.insert(0, extra)
    import corpus_overlay
    import quant_rag
    import registry as reg

    journal = journal_import(delivery_id)
    if not journal:
        sys.exit(f"REFUS : aucun journal pour {delivery_id!r}")
    if journal.get("state") != "applied":
        sys.exit(f"REFUS : journal {delivery_id} en état {journal.get('state')!r} — "
                 "la réparation ne vaut que pour un import promu")
    identifiants = {d["document_id"] for d in journal["documents"]}
    descripteur = descripteur_livraison(Path(journal["source"]), delivery_id, journal["started_at"])
    donnees = reg.build()
    for entree in donnees["documents"]:
        if entree["document_id"] in identifiants:
            entree["delivery"] = dict(descripteur)
    donnees["deliveries"] = [d for d in donnees["deliveries"] if d.get("id") != delivery_id]
    donnees["deliveries"].append(dict(descripteur, documents=len(identifiants)))
    reg.PATH.write_text(json.dumps(donnees, ensure_ascii=False, indent=1), encoding="utf-8")
    lignes = reg.write_imported_rows()
    corpus_overlay.invalidate()
    quant_rag.bm25.cache_clear()
    index = quant_rag.rebuild_bm25()
    print(json.dumps({"status": "REPARE", "delivery_id": delivery_id,
                      "documents": len(identifiants), "imported_rows": lignes,
                      "corpus_signature": corpus_overlay.signature(),
                      "bm25_records": index.doc_count}, ensure_ascii=False, indent=1))
    return 0


def descripteur_livraison(livraison: Path, delivery_id: str, recu: str) -> dict:
    """La provenance telle qu'``apply_delivery`` l'écrit — Source A ou Source B."""
    sys.path.insert(0, str(HERE))
    import apply_delivery as ad

    return ad.delivery_descriptor(livraison, delivery_id, recu)


# ------------------------------------------------------------------ banc, une seule fois

def dette_bm25() -> dict | None:
    """L'index lexical est-il en retard sur la collection ? ``None`` s'il est à jour.

    Hors ligne et **sans ouvrir Qdrant** — c'est la condition qui rend ce contrôle utilisable
    au démarrage d'un lot, quand le verrou exclusif peut être tenu par autre chose. On
    confronte le ``built_from.points`` du manifeste de la signature courante aux
    ``totals.active_chunks`` du registre. C'est la même confrontation que
    ``dense_matrix.Matrix.verify()`` et que la garde de dérive de ``qdrant_backend``, avec
    une troisième raison d'être : un lot ``fin-de-lot`` tué entre son dernier import et sa
    reconstruction laisse un corpus dont le chemin **dense** trouve les derniers documents
    et dont le chemin **lexical** les ignore. Ce retard doit être constatable, pas déductible.
    """
    sys.path.insert(0, str(ROOT / "rag"))
    import corpus_overlay

    manifeste = (ROOT / "data" / "lexical" /
                 f"bm25-{corpus_overlay.LABEL}-{corpus_overlay.signature()}.manifest.json")
    registre = json.loads(REGISTRY.read_text(encoding="utf-8")) if REGISTRY.exists() else {}
    attendus = (registre.get("totals") or {}).get("active_chunks")
    if not manifeste.exists():
        return {"motif": "aucun manifeste pour la signature courante",
                "manifeste": manifeste.name, "servis": attendus}
    built = json.loads(manifeste.read_text(encoding="utf-8")).get("built_from") or {}
    if attendus is not None and built.get("points") != attendus:
        return {"motif": "l'index lexical est en retard sur la collection",
                "manifeste": manifeste.name, "index": built.get("points"), "servis": attendus}
    return None


def etape_reconstruction_finale(lot: str, journal: Path, sec: bool) -> dict:
    """Ce que le régime ``fin-de-lot`` a différé, fait **une fois**, dans l'ordre obligatoire.

    L'ordre n'est pas cosmétique :

    1. ``build_graph`` — il lit ``results.jsonl``, que l'extraction par document a déjà
       complété. Le faire avant BM25 n'a pas d'importance pour lui, mais le faire **après**
       le registre en a une : c'est le défaut du 4 septembre (« le graphe était aveugle aux
       imports »). Le registre est écrit par document, donc la condition est tenue ;
    2. ``check_bm25`` — il **reconstruit** l'index puis le compare.
       Un seul appel suffit donc à faire les deux gestes que le régime par document faisait
       en deux reconstructions séparées ;
    3. les **sondes**, pour chaque document importé, sur l'état final : c'est le seul moment
       où elles constatent ce que le corpus sert vraiment, index et graphe compris.

    Elle s'exécute même après un SIGTERM. Un lot interrompu qui laisserait l'index lexical
    en retard serait pire que le lot lui-même : la recherche dense trouverait les documents,
    la recherche lexicale non, et rien ne le dirait à l'utilisateur.
    """
    resultat: dict = {"regime": "fin-de-lot", "etapes": {}}
    dire("reconstruction de fin de lot — ce que le régime a différé, fait une fois")

    code, sortie, secondes = courir([PY_GLINER, ROOT / "rag" / "graph" / "build_graph.py"], journal)
    graphe = dernier_json(sortie) or {}
    resultat["etapes"]["graphe"] = {"code": code, "secondes": secondes,
                                    "nodes": graphe.get("nodes"), "edges": graphe.get("edges"),
                                    "documents": graphe.get("documents"),
                                    "sample_chunks": graphe.get("sample_chunks")}
    if code != 0:
        resultat["echec"] = f"build_graph : {sortie.strip()[-300:]}"
        dire(f"  ÉCHEC build_graph : {sortie.strip()[-200:]}")
        return resultat
    dire(f"  graphe : {graphe.get('nodes')} nœuds, {graphe.get('edges')} arêtes en {secondes} s")

    code, sortie, secondes = courir([PY, ROOT / "rag" / "benchmark" / "check_bm25.py"], journal)
    controle = dernier_json(sortie) or {}
    resultat["etapes"]["bm25"] = {"code": code, "secondes": secondes,
                                  "records": (controle.get("rebuild") or {}).get("records")}
    if code != 0:
        resultat["echec"] = f"check_bm25 : {sortie.strip()[-300:]}"
        dire(f"  ÉCHEC check_bm25 : {sortie.strip()[-200:]}")
        return resultat
    dire(f"  BM25 reconstruit et contrôlé en {secondes} s "
         f"({(controle.get('rebuild') or {}).get('records')} records)")

    sondes = {}
    for etat in tous_les_etats(lot):
        if etat.get("etat") != "ok" or not etat.get("document_id"):
            continue
        if not (etat.get("mesures", {}).get("sondes", {}) or {}).get("statut"):
            continue                      # sondé par document : rien à rejouer
        ok, message = sonder(etat, journal)
        sondes[etat["pdf"]] = {"ok": ok, "message": message[:200],
                               "mesures": etat["mesures"].get("sondes")}
        ecrire_json(chemin_etat(etat["sha256"]), etat)
        dire(f"  sonde {etat['pdf']} : {'OK' if ok else 'ÉCHEC'} — {message[:90]}")
    resultat["etapes"]["sondes"] = {"documents": len(sondes),
                                    "echecs": sum(1 for s in sondes.values() if not s["ok"])}
    resultat["sondes"] = sondes
    dire(f"  sondes rejouées sur {len(sondes)} document(s), "
         f"{resultat['etapes']['sondes']['echecs']} échec(s)")
    resultat["dette_bm25_restante"] = dette_bm25()
    return resultat


def etape_bancs(lot: str, journal: Path) -> dict:
    """Calibration puis banc, à la fin du lot. Une nouvelle ligne de base, pas une comparaison."""
    mesures = {}
    for nom, commande in (("calibration", [PY, ROOT / "rag" / "benchmark" / "calibrate_router.py"]),
                          ("banc", [PY, ROOT / "rag" / "benchmark" / "eval_router.py"])):
        dire(f"banc : {nom}…")
        code, sortie, secondes = courir(commande, journal)
        mesures[nom] = {"code": code, "secondes": secondes,
                        "resultat": dernier_json(sortie) if code == 0 else sortie.strip()[-400:]}
        if code != 0:
            dire(f"  {nom} en échec (code {code}) — le lot reste importé, la ligne de base manque")
            return mesures
    # ``eval_router`` n'imprime pas ses chiffres, il les écrit. Le rapport les reprend, avec
    # la signature et les comptes qui disent **de quel corpus** ils parlent : deux fichiers
    # de résultats ne sont comparables que si ces trois valeurs-là sont identiques.
    resultats = ROOT / "rag" / "benchmark" / "results-router-v3.json"
    donnees = lire_json(resultats) or {}
    if donnees:
        mesures["ligne_de_base"] = {
            "fichier": resultats.name,
            "corpus": {k: (donnees.get("corpus") or {}).get(k)
                       for k in ("signature", "documents", "chunks")},
            "nDCG@10 (routeur)": {vue: (donnees["summary"][vue]["router"] or {}).get("nDCG@10")
                                  for vue in donnees.get("summary", {})},
            "recall@10 (routeur)": {vue: (donnees["summary"][vue]["router"] or {}).get("recall@10")
                                    for vue in donnees.get("summary", {})},
            "part hybride": {vue: donnees["summary"][vue].get("hybrid_share")
                             for vue in donnees.get("summary", {})},
        }
        dire("banc : " + " · ".join(f"{v} {n}" for v, n in
                                    mesures["ligne_de_base"]["nDCG@10 (routeur)"].items()))
    # Le banc ne tourne qu'une fois par lot ; une relance du driver (tout est déjà fait,
    # aucun import) ne doit pas effacer ses chiffres du rapport. On les garde sur le disque.
    ecrire_json(ETAT / f"banc-{lot}.json", mesures)
    return mesures


# ------------------------------------------------------------------ rapports

def tous_les_etats(lot: str) -> list[dict]:
    etats = [lire_json(p) for p in sorted(ETAT.glob("*.json")) if p.name != "verrou.json"]
    return [e for e in etats if e and e.get("lot") == lot]


def ecrire_todolist(lot: str) -> None:
    """Réécrite à chaque transition : c'est la vue que l'opérateur trouve le lundi matin."""
    etats = tous_les_etats(lot)
    symboles = {"ok": "✓", "écarté": "✗", "en-attente": "·", "en-cours": "→", "à-vérifier": "!"}
    lignes = [f"# Lot {lot} — état",
              "",
              f"*Réécrit à chaque transition. Dernière : {maintenant()}.*",
              "",
              "| | PDF | état | étape | ce qui est mesuré |",
              "|---|---|---|---|---|"]
    for etat in etats:
        mesures = etat.get("mesures", {})
        detail = ""
        if etat["etat"] == "ok":
            imp = mesures.get("import", {})
            sondes = mesures.get("sondes", {})
            detail = (f"{imp.get('point_ids', {}).get('count')} points, dense rang "
                      f"{sondes.get('rang_dense')}, {mesures.get('commit', {}).get('revision', '')}")
        elif etat["etat"] in ("écarté", "à-vérifier"):
            detail = (etat.get("raison") or "")[:110]
        elif mesures.get("parse"):
            detail = f"{mesures['parse'].get('chunks')} chunks produits"
        lignes.append(f"| {symboles.get(etat['etat'], '?')} | `{etat['pdf']}` | {etat['etat']} | "
                      f"{etat.get('etape') or '—'} | {detail} |")
    compte = {e: sum(1 for x in etats if x["etat"] == e)
              for e in ("ok", "écarté", "en-attente", "en-cours", "à-vérifier")}
    lignes += ["", f"**{compte['ok']} importés · {compte['écarté']} écartés · "
                   f"{compte['à-vérifier']} à vérifier · "
                   f"{compte['en-attente'] + compte['en-cours']} restants**", ""]
    (SOURCE_B / f"TODOLIST-{lot}.md").write_text("\n".join(lignes) + "\n", encoding="utf-8")


def temps_par_etape(etat: dict) -> str:
    """Le temps de chaque étape, **cumulé** sur toutes les tentatives.

    Un document repris trois fois a trois « relevé » et trois « diagnostic » dans son
    journal : les lister à la file ne dit rien. Ce qui se lit, c'est ce que l'étape a
    coûté au total — reprises comprises, puisque c'est ce que le lot a réellement payé.
    """
    total: dict[str, float] = {}
    for entree in etat.get("journal", []):
        if entree.get("secondes") and entree.get("etape"):
            total[entree["etape"]] = total.get(entree["etape"], 0.0) + entree["secondes"]
    ordre = {nom: rang for rang, nom in enumerate(("reprise",) + ETAPES)}
    return " · ".join(f"{nom} {round(valeur):d} s" for nom, valeur in
                      sorted(total.items(), key=lambda kv: ordre.get(kv[0], 99)))


def ecrire_rapport(lot: str, debut: str, bancs: dict, avant: dict, arret: str | None,
                   reconstruction: dict | None = None) -> Path:
    etats = tous_les_etats(lot)
    bancs = bancs or (lire_json(ETAT / f"banc-{lot}.json") or {})
    importes = [e for e in etats if e["etat"] == "ok"]
    ecartes = [e for e in etats if e["etat"] == "écarté"]
    apres = etat_du_corpus()
    lignes = [f"# Rapport de lot — {lot}",
              "",
              f"*Lot ouvert {debut}, clos {maintenant()}. Produit sans supervision par "
              f"`rag/ingestion/batch_driver.py`.*",
              "",
              f"*{len(importes)} importé(s), {len(ecartes)} écarté(s) sur "
              f"{len(etats)} PDF posés.*",
              ""]
    if arret:
        lignes += [f"> **Le lot s'est arrêté avant la fin** : {arret}", ""]
    lignes += ["## Ce qui est entré", ""]
    if not importes:
        lignes.append("Aucun document importé.")
    for etat in importes:
        m = etat["mesures"]
        imp, sondes = m.get("import", {}), m.get("sondes", {})
        meta = m.get("metadonnees", {})
        lignes += [
            f"### {meta.get('short_ref') or etat['pdf']} — {meta.get('title') or '?'}",
            "",
            f"| | |",
            f"|---|---|",
            f"| fichier | `{etat['pdf']}` → `data/papers/{m.get('filename')}` |",
            f"| sha256 | `{etat['sha256']}` |",
            f"| document_id | `{etat['document_id']}` |",
            f"| verdict | **ajout** — aucun quasi-doublon, `gold_chunks_at_risk` "
            f"{m.get('diagnostic', {}).get('gold_chunks_at_risk')} |",
            f"| titre retenu | {meta.get('title')!r} — provenance `{meta.get('provenance')}` |",
            f"| auteurs / année | {', '.join(meta.get('authors') or []) or '?'} · "
            f"{meta.get('year')} (`{meta.get('provenance_annee')}`) |",
            f"| provenance du PDF | {source_declaree(etat['pdf'])} |",
            f"| pages / chunks | {m.get('parse', {}).get('pages')} p. · "
            f"{m.get('parse', {}).get('chunks')} chunks dont "
            f"{m.get('parse', {}).get('eligible_chunks')} éligibles |",
            f"| points | {imp.get('point_ids', {}).get('first')}.."
            f"{imp.get('point_ids', {}).get('last')} ({imp.get('point_ids', {}).get('count')}) |",
            f"| signature après | `{imp.get('signature')}` |",
            f"| sondes | dense rang {sondes.get('rang_dense')} ({sondes.get('score_dense')}) · "
            f"BM25 rang {sondes.get('rang_lexical')} · graphe {sondes.get('chunks_graphe')} chunks |",
            f"| commit | `{m.get('commit', {}).get('revision')}` |",
            f"| temps | " + temps_par_etape(etat) + " |",
            ""]
        cout = imp.get("cout") or {}
        if cout:
            octets = cout.get("octets_ecrits") or {}
            llm_cout = cout.get("llm") or {}
            lignes += [
                f"*Import, dans le détail : {cout.get('secondes_total')} s — " +
                " · ".join(f"{nom} {duree} s" for nom, duree in (cout.get("secondes") or {}).items()) +
                f". Titrage : {llm_cout.get('calls', 0)} appel(s), "
                f"{llm_cout.get('cached', 0)} en cache, "
                f"{llm_cout.get('prompt_tokens', 0) + llm_cout.get('completion_tokens', 0)} jetons "
                f"({cout.get('modele_titrage') or 'aucun modèle'}). "
                f"Écrits : index BM25 {round((octets.get('index_bm25') or 0) / 1e6, 1)} Mo · "
                f"vecteurs {round((octets.get('vecteurs_npz') or 0) / 1e6, 2)} Mo · "
                f"registre {round((octets.get('registre') or 0) / 1e6, 2)} Mo.*", ""]
        if m.get("annulations"):
            lignes += [f"*Tentative(s) d'import annulée(s) avant celle-ci : "
                       f"{json.dumps(m['annulations'], ensure_ascii=False)}.*", ""]
        if m.get("reparation"):
            r = m["reparation"]
            lignes += [f"*Import repris après un arrêt brutal : promu mais absent du registre, "
                       f"réparé sans réécrire un point — registre et relais reconstruits "
                       f"({r.get('imported_rows')} lignes), BM25 {r.get('bm25_records')} records, "
                       f"signature `{r.get('corpus_signature')}`.*", ""]
    lignes += ["## Ce qui a été écarté, et pourquoi", ""]
    if not ecartes:
        lignes.append("Rien.")
    else:
        lignes += ["| PDF | étape | raison |", "|---|---|---|"]
        lignes += [f"| `{e['pdf']}` | {e.get('etape')} | {(e.get('raison') or '')[:200]} |"
                   for e in ecartes]
    a_verifier = [e for e in etats if e["etat"] == "à-vérifier"]
    if a_verifier:
        lignes += ["", "## Importés, mais une étape d'aval a manqué", "",
                   "Ces documents **sont dans le corpus servi** — leur PDF n'a pas été déplacé.",
                   "", "| PDF | étape | ce qui a manqué |", "|---|---|---|"]
        lignes += [f"| `{e['pdf']}` | {e.get('etape')} | {(e.get('raison') or '')[:200]} |"
                   for e in a_verifier]
    lignes += ["", "Les PDF importés sont sortis de la file dans "
                   "`rag/ingestion/source-b/importes/` — leur exemplaire durable est "
                   "`data/papers/`, suivi en LFS. Les PDF écartés sont dans "
                   "`rag/ingestion/source-b/ecartes/`. Rien n'a été "
                   "importé à moitié : le diagnostic précède toute écriture, et un import mort "
                   "en vol est annulé avant la tentative suivante.", ""]
    lignes += ["## Le corpus avant et après", "",
               "| | avant | après |", "|---|---|---|"]
    for cle, nom in (("chunks", "chunks"), ("documents", "documents"),
                     ("signature", "signature de corpus"), ("registre", "empreinte du registre"),
                     ("bm25", "index BM25"), ("bm25_accorde", "index accordé à la collection")):
        lignes.append(f"| {nom} | `{avant.get(cle)}` | `{apres.get(cle)}` |")
    if reconstruction:
        etapes = reconstruction.get("etapes", {})
        lignes += ["", "## Reconstruction de fin de lot", "",
                   f"Régime `{reconstruction.get('regime')}` : l'index BM25 et le graphe ont "
                   f"été reconstruits **une fois**, après le dernier document, et les sondes "
                   f"rejouées sur l'état final.", "",
                   "| étape | durée | résultat |", "|---|---|---|"]
        graphe = etapes.get("graphe") or {}
        bm25 = etapes.get("bm25") or {}
        sondes = etapes.get("sondes") or {}
        lignes += [f"| graphe | {graphe.get('secondes')} s | {graphe.get('nodes')} nœuds, "
                   f"{graphe.get('edges')} arêtes, {graphe.get('documents')} documents |",
                   f"| BM25 (reconstruit et contrôlé) | {bm25.get('secondes')} s | "
                   f"{bm25.get('records')} records |",
                   f"| sondes rejouées | — | {sondes.get('documents')} document(s), "
                   f"{sondes.get('echecs')} échec(s) |"]
        if reconstruction.get("echec"):
            lignes += ["", f"**ÉCHEC** : {reconstruction['echec']}"]
        if reconstruction.get("dette_bm25_restante"):
            lignes += ["", f"**DETTE RESTANTE** : "
                           f"`{json.dumps(reconstruction['dette_bm25_restante'], ensure_ascii=False)}`"]
        lignes.append("")
    lignes += ["", "## Banc — une seule ligne de base, à la fin", ""]
    if not bancs:
        lignes += ["Non rejoué (aucun import dans ce lot).", ""]
    else:
        base = bancs.get("ligne_de_base") or {}
        for nom, mesure in bancs.items():
            if nom == "ligne_de_base":
                continue
            lignes.append(f"- **{nom}** : {'OK' if mesure['code'] == 0 else 'ÉCHEC'} "
                          f"({mesure['secondes']} s)")
        if base:
            corpus = base.get("corpus", {})
            lignes += ["", f"`{base['fichier']}` — signature `{corpus.get('signature')}`, "
                           f"{corpus.get('documents')} documents, {corpus.get('chunks')} chunks.",
                       "", "| vue | nDCG@10 | recall@10 | part hybride |", "|---|---|---|---|"]
            for vue, valeur in base["nDCG@10 (routeur)"].items():
                lignes.append(f"| {vue} | {valeur} | {base['recall@10 (routeur)'].get(vue)} | "
                              f"{base['part hybride'].get(vue)} |")
        lignes += ["", "Les chiffres du banc **établissent une nouvelle ligne de base** ; ils ne se "
                       "comparent pas aux précédents. Les chunks entrés concourent contre l'or sans "
                       "que l'or bouge (`INGESTION-CONCEPTION` §5).", ""]
    lignes += ["## Traces", "",
               f"- journal du lot : `rag/ingestion/source-b/journaux/{lot}.log`",
               f"- état par document : `rag/ingestion/source-b/etat/*.json`",
               f"- suivi réécrit à chaque transition : `rag/ingestion/source-b/TODOLIST-{lot}.md`",
               ""]
    chemin = SOURCE_B / f"RAPPORT-BATCH-{lot}.md"
    chemin.write_text("\n".join(lignes) + "\n", encoding="utf-8")
    return chemin


def etat_du_corpus() -> dict:
    """L'état servi, lu dans un sous-processus : le verrou Qdrant ne survit pas à l'appel.

    Le driver n'ouvre jamais Qdrant lui-même. S'il le faisait, il tiendrait le verrou du
    stockage embarqué pendant tout le lot et ``apply_delivery`` échouerait à son étape 0 —
    contre son propre driver.
    """
    resultat = subprocess.run([str(PY), str(ROOT / "rag" / "quant_rag.py"), "--status"],
                              capture_output=True, text=True, cwd=ROOT)
    donnees = dernier_json(resultat.stdout) or {}
    manifeste = ((donnees.get("bm25") or {}).get("manifest") or {})
    # Le manifeste BM25 disparaît avec son index quand la signature bouge — c'est voulu.
    # La signature, elle, se calcule toujours : on ne la rapporte pas comme inconnue.
    signature = (manifeste.get("corpus") or {}).get("signature")
    if not signature:
        signature = subprocess.run(
            [str(PY), "-c", "import sys; sys.path.insert(0, 'rag'); import corpus_overlay;"
                            " print(corpus_overlay.signature())"],
            capture_output=True, text=True, cwd=ROOT).stdout.strip() or None
    return {"chunks": donnees.get("chunks"), "documents": donnees.get("documents"),
            "signature": signature,
            "registre": (donnees.get("provenance") or {}).get("digest"),
            "bm25": manifeste.get("index"),
            "bm25_accorde": (donnees.get("bm25") or {}).get("matches_collection"),
            "livraisons": [d.get("id") for d in (donnees.get("provenance") or {}).get("deliveries", [])]}


# ------------------------------------------------------------------ boucle

def traiter(doc: dict, journal: Path, llm: str, sec: bool) -> dict:
    """La chaîne, un document, dans l'ordre. Le premier échec écarte et rend la main.

    La **reprise passe avant tout**, y compris avant le diagnostic : celui-ci compare la
    livraison au corpus, et un import déjà promu fausse la comparaison au point d'inverser
    la conclusion. Réconcilier d'abord, juger ensuite.
    """
    dire(f"{doc['pdf']}  sha256 {doc['sha256'][:16]}…  ({doc['octets'] / 1e6:.1f} Mo)")
    etapes = [("parse", etape_parse), ("relevé", etape_releve), ("diagnostic", etape_diagnostic)]
    if not sec:
        etapes += [("import", etape_import), ("aval", etape_aval), ("graphe", etape_graphe),
                   ("sondes", etape_sondes), ("commit", etape_commit)]
        transition(doc, "en-cours", "reprise", "constat de l'état réel")
        try:
            statut, message = etape_reprise(doc, journal, llm)
        except Exception as erreur:
            statut, message = False, f"{type(erreur).__name__}: {erreur}"
        if statut is False:
            dire(f"  À VÉRIFIER (reprise) : {message[:150]}")
            return transition(doc, "à-vérifier", "reprise", message, raison=message)
        dire(f"  reprise : {message}")
        if statut == "importé":
            etapes = [e for e in etapes if e[0] in ("aval", "graphe", "sondes", "commit")]

    for nom, fonction in etapes:
        if ARRET_DEMANDE:
            return transition(doc, "en-cours", nom, "arrêt demandé avant cette étape")
        debut = time.monotonic()
        transition(doc, "en-cours", nom, "commencée")
        try:
            if nom == "import":
                ok, message = fonction(doc, journal, llm, sec)
            else:
                ok, message = fonction(doc, journal, sec)
        except Exception as erreur:                      # une panne inattendue écarte, jamais n'arrête
            ok, message = False, f"{type(erreur).__name__}: {erreur}"
        secondes = round(time.monotonic() - debut, 1)
        if not ok:
            if nom in ETAPES_APRES_IMPORT or doc["mesures"].get("deja_dans_le_corpus"):
                dire(f"  À VÉRIFIER ({nom}) : {message[:150]}")
                return transition(doc, "à-vérifier", nom, message, secondes, raison=message)
            return ecarter(doc, nom, message, sec)
        dire(f"  {nom} : {message}")
        transition(doc, "en-cours", nom, message, secondes)
    if not sec:
        ranger_importe(doc)
    return transition(doc, "en-attente" if sec else "ok", None,
                      "diagnostic vert, prêt à importer" if sec else "importé et servi")


def main() -> None:
    global ARRET_DEMANDE, RECONSTRUCTION, SANS_GRAPHE
    analyseur = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    analyseur.add_argument("--go", action="store_true", help="exécute le lot (écrit dans le corpus)")
    analyseur.add_argument("--dry-run", action="store_true",
                           help="jusqu'au diagnostic inclus : parse, relevé, inspection. "
                                "N'écrit rien dans le corpus servi")
    analyseur.add_argument("--etat", action="store_true", help="où en est le lot (lecture seule)")
    analyseur.add_argument("--id", dest="lot", help="identifiant du lot (défaut : la date)")
    analyseur.add_argument("--llm", default=LLM_DEFAUT,
                           help=f"modèle de titrage (défaut {LLM_DEFAUT}) ; "
                                f"{SANS_LLM!r} = aucun appel externe, titre pris dans le PDF "
                                "ou dans overrides.json")
    analyseur.add_argument("--reconstruction", choices=RECONSTRUCTIONS, default="par-document",
                           help="par-document (défaut, régime historique) : BM25 reconstruit "
                                "deux fois et le graphe une fois après chaque document. "
                                "fin-de-lot : les deux sont différés et faits une seule fois "
                                "à la fin, avec les sondes. Économie mesurée ≈ 16,7 s et "
                                "≈ 102 Mo écrits par document. L'extraction GLiNER et le "
                                "registre restent par document dans les deux régimes")
    analyseur.add_argument("--sans-banc", action="store_true", help="ne pas rejouer le banc à la fin")
    analyseur.add_argument("--sans-graphe", action="store_true",
                           help="sauter l'étape du graphe (extraction GLiNER + build_graph), "
                                "~95 s par document. Le défaut reste de la faire. Ce qu'on "
                                "perd : search_graph, expand_entity et connect_entities ne "
                                "reflètent pas les documents du lot, jusqu'à ce qu'un lot "
                                "sans ce drapeau les rattrape — l'extraction est incrémentale")
    analyseur.add_argument("--interroger", metavar="DOCUMENT_ID", help="usage interne : sonde le corpus")
    analyseur.add_argument("--titre", help="usage interne : requête de la sonde de titre")
    analyseur.add_argument("--contenu", help="usage interne : requête de la sonde de contenu")
    analyseur.add_argument("--reparer", metavar="ID", help="usage interne : finit un import promu")
    arguments = analyseur.parse_args()

    if arguments.interroger:
        raise SystemExit(mode_interroger(arguments.interroger, arguments.titre or "",
                                        arguments.contenu))
    if arguments.reparer:
        raise SystemExit(mode_reparer(arguments.reparer))

    lot = arguments.lot or datetime.now().strftime("%Y-%m-%d-lot")
    if arguments.etat:
        etats = tous_les_etats(lot)
        print(json.dumps({"lot": lot, "documents": [
            {k: e[k] for k in ("pdf", "etat", "etape", "raison", "document_id")} for e in etats]},
            ensure_ascii=False, indent=1))
        return
    if not (arguments.go or arguments.dry_run):
        analyseur.error("choisis --dry-run (rien n'est écrit dans le corpus) ou --go")

    # Un gel actif interdit de faire entrer un document : base et traitement d'un chantier
    # mesuré portent la même signature, sinon le Δ apparié confond le chantier avec les
    # documents ajoutés. --dry-run n'écrit rien, il passe.
    if arguments.go:
        sys.path.insert(0, str(ROOT / "rag"))
        import gel_corpus
        gel_corpus.verifier("l'import d'un document au corpus")

    RECONSTRUCTION = arguments.reconstruction
    SANS_GRAPHE = arguments.sans_graphe
    if SANS_GRAPHE:
        dire("ATTENTION --sans-graphe : l'étape du graphe est sautée pour tout le lot. "
             "search_graph, expand_entity et connect_entities ne refléteront pas ces "
             "documents tant qu'un lot sans ce drapeau ne les aura pas rattrapés")
    # Une dette d'index constatée AVANT de commencer : elle vient d'un lot fin-de-lot
    # interrompu, et le lot qui démarre la soldera à sa propre reconstruction finale.
    # On la dit, parce qu'un corpus dont le chemin lexical est en retard doit se voir.
    retard = dette_bm25()
    if retard:
        dire(f"ATTENTION dette d'index lexical : {retard['motif']} "
             f"({retard.get('index')} indexés contre {retard.get('servis')} servis) — "
             f"elle sera soldée par la reconstruction de fin de lot")

    sec = not arguments.go
    for dossier in (ETAT, JOURNAUX, ECARTES, IMPORTES, LIVRAISONS):
        dossier.mkdir(parents=True, exist_ok=True)
    journal = JOURNAUX / f"{lot}.log"

    os.setpgrp()                                   # nos enfants meurent avec nous : kill -TERM -<pid>

    def arreter(signal_recu, _cadre):
        global ARRET_DEMANDE
        ARRET_DEMANDE = True
        dire(f"signal {signal_recu} reçu — arrêt après l'étape en cours")
    for numero in (signal.SIGTERM, signal.SIGINT):
        signal.signal(numero, arreter)

    dire(f"lot {lot} · pid {os.getpid()} · {'DRY-RUN' if sec else 'IMPORT RÉEL'}")
    dire(f"pour arrêter proprement :  kill -TERM -{os.getpid()}   (le tiret devant le pid compte)")

    problemes = verifier_environnement()
    lourds = processus_lourds()
    libre, raison = verrou_qdrant_libre()
    if lourds:
        problemes.append("travail lourd déjà en cours : " +
                         " ; ".join(f"pid {p} {c[:80]}" for p, c in lourds))
    if not libre and not sec:
        problemes.append(f"verrou Qdrant tenu ({raison}) — apply_delivery ne pourrait pas écrire")
    if problemes:
        for probleme in problemes:
            print(f"  bloquant : {probleme}")
        sys.exit("REFUS : le lot ne démarre pas sur un environnement douteux")
    dire(f"environnement : MinerU 3.4.5 · disque {gio(disque_libre())} libres · "
         f"verrou Qdrant {raison} · titrage {arguments.llm}")

    prendre_verrou(lot, sec)
    # Un lot s'étale sur plusieurs invocations — un dry-run, un --go, une reprise après une
    # coupure. Le rapport doit décrire le **lot**, pas le dernier lancement : son ouverture
    # et l'état du corpus d'alors se figent au premier passage et ne bougent plus.
    ouverture = lire_json(ETAT / f"lot-{lot}.json")
    if ouverture is None:
        ouverture = {"ouvert": maintenant(), "corpus_avant": etat_du_corpus(),
                     "premier_mode": "dry-run" if sec else "import"}
        ecrire_json(ETAT / f"lot-{lot}.json", ouverture)
    debut, avant = ouverture["ouvert"], ouverture["corpus_avant"]
    dire(f"corpus avant : {avant.get('chunks')} chunks, {avant.get('documents')} documents, "
         f"signature {avant.get('signature')}")

    pdfs = sorted(ENTREE.glob("*.pdf"))
    dire(f"{len(pdfs)} PDF dans entree/")
    arret = None
    imports = 0
    echecs_import = 0
    bancs: dict = {}
    reconstruction: dict = {}
    try:
        for rang, pdf in enumerate(pdfs, 1):
            if ARRET_DEMANDE:
                arret = "arrêt demandé (SIGTERM/SIGINT)"
                break
            libre_maintenant = disque_libre()
            if libre_maintenant < DISQUE_MINIMUM:
                arret = (f"disque sous le seuil : {gio(libre_maintenant)} libres, minimum "
                         f"{gio(DISQUE_MINIMUM)}. Les documents restants n'ont pas été touchés")
                dire(f"ARRÊT : {arret}")
                break
            lourds = processus_lourds()
            if lourds:
                arret = ("un travail lourd a été lancé pendant le lot : " +
                         " ; ".join(f"pid {p}" for p, _ in lourds) +
                         ". Rien ne tourne à deux sur ces 16 Go")
                dire(f"ARRÊT : {arret}")
                break
            doc = charger_etat(pdf, lot)
            if doc["etat"] in ("ok", "écarté"):
                # le dry-run juge mais ne déplace rien : le premier --go finit le geste
                if doc["etat"] == "écarté" and not sec and not doc.get("ecarte_vers"):
                    ecarter(doc, doc.get("etape"), doc.get("raison") or "écarté au dry-run", sec)
                if doc["etat"] == "ok" and not sec:
                    ranger_importe(doc)
                dire(f"{pdf.name} : déjà {doc['etat']}, ignoré")
                continue
            doc["delivery_id"] = doc.get("delivery_id") or f"{lot}-d{rang:02d}"
            doc["lot"] = lot
            ecrire_json(chemin_etat(doc["sha256"]), doc)
            doc = traiter(doc, journal, arguments.llm, sec)
            if doc["etat"] == "ok":
                imports, echecs_import = imports + 1, 0
            elif doc["etat"] == "à-vérifier":
                imports += 1                    # il est dans le corpus, le banc doit en tenir compte
                if doc.get("etape") in ETAPES_QUI_ARRETENT or doc.get("etape") == "import":
                    arret = (f"{doc['pdf']} est importé mais son aval a échoué "
                             f"({(doc.get('raison') or '')[:160]}). Le corpus servi est "
                             "incohérent : le lot s'arrête plutôt que d'empiler un document de plus")
                    dire(f"ARRÊT : {arret}")
                    break
            elif doc["etat"] == "écarté" and doc.get("etape") == "import":
                echecs_import += 1
                if echecs_import >= ECHECS_IMPORT_AVANT_ARRET:
                    arret = (f"{echecs_import} imports échoués d'affilée — la cause est commune, "
                             "pas documentaire (plafond LLM épuisé ? verrou ? disque ?). Les PDF "
                             "restants n'ont pas été touchés ; ceux qui sont déjà dans ecartes/ "
                             "peuvent y retourner une fois la cause levée")
                    dire(f"ARRÊT : {arret}")
                    break
        # La reconstruction finale, elle, ne dépend PAS de ARRET_DEMANDE : un SIGTERM qui la
        # sauterait laisserait le corpus servi avec un index lexical en retard sur sa
        # collection — la recherche dense trouverait les derniers documents, la lexicale non.
        # Elle est aussi idempotente : la relancer sur un état déjà reconstruit ne coûte que
        # son temps. La solder ici est donc toujours le bon geste.
        if not sec and RECONSTRUCTION == "fin-de-lot" and (imports or retard):
            reconstruction = etape_reconstruction_finale(lot, journal, sec)
        if not sec and imports and not arguments.sans_banc and not ARRET_DEMANDE:
            bancs = etape_bancs(lot, journal)
    finally:
        rendre_verrou()
    ecrire_todolist(lot)
    rapport = ecrire_rapport(lot, debut, bancs if not sec else {}, avant, arret, reconstruction)
    dire(f"rapport : {rapport.relative_to(ROOT)}")
    etats = tous_les_etats(lot)
    dire(f"lot {lot} : {sum(1 for e in etats if e['etat'] == 'ok')} importés, "
         f"{sum(1 for e in etats if e['etat'] == 'écarté')} écartés")


if __name__ == "__main__":
    main()
