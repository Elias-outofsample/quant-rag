"""Le chemin d'écriture du corpus, exercé de bout en bout sur un arbre miniature.

``apply_delivery.py`` est le seul module du dépôt qui écrive dans Qdrant, BM25, les
overlays, les métadonnées et le registre. Jusqu'ici il n'était éprouvé que par des mesures
qui écrivent dans un vrai corpus (``mesure_regimes.py``) : coûteuses, non reproductibles,
et impossibles à lancer pendant qu'une autre mesure tourne. Ce module fait la même chose
sur un corpus jetable de deux documents, une collection Qdrant embarquée dans ``tmp_path``
et des vecteurs déposés à l'avance — **aucun modèle n'est chargé, aucun fichier du dépôt
n'est écrit**.

Ce qu'il garde, défaut par défaut
----------------------------------
=====================================  ========================================================
plage d'identifiants                   ``max(id) + 1``, jamais ``compte + i`` : la collection
                                       a des trous, et la règle amont écraserait des points
registre                               il porte les plages de points, donc de quoi annuler
sommes des quatre fichiers             sans elles, ``registry --verify`` ne peut plus dire
                                       qu'un document a été modifié sous le corpus
manifeste BM25                         ``built_from.points`` est le seul témoin hors ligne du
                                       contenu servi (``qdrant_backend.comptes_attendus``)
overlays                               l'import les *étend* ; il ne doit jamais les tronquer
``--rollback``                         la signature d'avant doit revenir **à l'identique**,
                                       sinon l'index BM25 d'avant n'est plus retrouvé
sha256 déjà connu                      un ajout seulement ; une révision renumérote les
                                       chunk_id et peut détruire l'or des bancs
=====================================  ========================================================

Le prix de l'herméticité — la dette de testabilité
---------------------------------------------------
Tous les chemins d'``apply_delivery`` et de ses dépendances sont des constantes de module
dérivées de ``__file__``. Aucune n'est injectable. Isoler l'import exige donc de remplacer
**vingt-six constantes réparties sur sept modules** (voir ``arbre``). C'est la mesure de
la dette : chaque constante remplacée ici est un endroit où la production ne sait pas dire
où elle écrit.

    .venv/bin/python -m pytest rag/ingestion/tests/test_apply_delivery.py -q
"""
from __future__ import annotations

import hashlib
import json
import time
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

RACINE = Path(__file__).resolve().parents[3]
for _relatif in ("rag", "src", "rag/ingestion", "rag/metadata", "rag/tables",
                 "rag/titles"):
    _chemin = str(RACINE / _relatif)
    if _chemin not in sys.path:
        sys.path.insert(0, _chemin)

import apply_delivery                                  # noqa: E402
import convert_tables                                  # noqa: E402
import corpus_overlay                                  # noqa: E402
import extract_metadata                                # noqa: E402
import gel_corpus                                      # noqa: E402
import inspect_delivery                                # noqa: E402
import quant_rag                                       # noqa: E402
import reembed_titles                                  # noqa: E402
import registry                                        # noqa: E402
import verrou_collection                               # noqa: E402
import scan_duplicates                                 # noqa: E402
from qdrant_client import QdrantClient, models         # noqa: E402

# ``inspect_delivery.corpus_shingles`` et ``apply_delivery.clean_title_of`` insèrent
# ``ROOT/...`` dans sys.path *au moment de l'appel* : avec un ROOT remplacé, ces chemins
# n'existent pas. Les modules sont donc importés ici, tant que ROOT est encore le vrai.
assert scan_duplicates.shingles and reembed_titles.clean_title

REEL_INGESTED = RACINE / "data" / "processed" / "ingested"
#: Deux documents réels parmi les plus petits du corpus : 12 chunks (2 éligibles) et
#: 19 chunks (11 éligibles, 2 tableaux). Assez petits pour qu'un import tienne en une
#: seconde, assez réels pour que le chemin traverse tout ce qu'il traverse en production.
CORPUS_MINIATURE = ("doc-83046c141ab04bf3", "doc-e40ef0c462c52885")
SOURCE_LIVRAISON = "doc-e40ef0c462c52885"
#: Dimension des vecteurs du corpus (Qwen3-Embedding-0.6B), relevée sur
#: ``data/qdrant-export/vectors.npy`` — 19 443 × 1024.
DIMENSION = 1024
#: Titre de provenance ``manual`` semé dans ``overrides.json``. ``apply_delivery`` refuse un
#: titre de provenance ``filename_stem`` ; on ne contourne pas la garde, on lui donne ce
#: qu'elle demande — sans dépenser un appel de LLM (même approche que
#: ``mesure_regimes.semer_overrides``).
TITRE_MANUEL = "Étude d'essai du chemin d'écriture du corpus"
LIVRAISON = "essai-01"


# --------------------------------------------------------------------- outils du banc d'essai

def _vecteurs(nombre: int, graine: int) -> np.ndarray:
    """Des vecteurs normalisés, déterministes, et **jamais calculés par un modèle**.

    Charger Qwen3 coûte 30 s et 2 Go pour une propriété que ce module ne mesure pas : ce
    qu'on éprouve ici est le chemin d'écriture, pas la qualité de l'embedding.
    """
    brut = np.random.default_rng(graine).normal(size=(nombre, DIMENSION)).astype(np.float32)
    return brut / np.linalg.norm(brut, axis=1, keepdims=True)


def _sha256(chemin: Path) -> str:
    return hashlib.sha256(chemin.read_bytes()).hexdigest()


def _lignes(chemin: Path) -> list[dict]:
    return [json.loads(l) for l in chemin.open(encoding="utf-8") if l.strip()]


def _points_par_document(client, collection: str) -> dict[str, list[int]]:
    par: dict[str, list[int]] = {}
    decalage = None
    while True:
        points, decalage = client.scroll(collection, limit=4096, offset=decalage,
                                         with_payload=["document_id"], with_vectors=False)
        for point in points:
            par.setdefault(point.payload["document_id"], []).append(int(point.id))
        if decalage is None:
            break
    return {document: sorted(identifiants) for document, identifiants in par.items()}


def _dossier_livre(livraison: Path) -> Path:
    return next((livraison / "processed").iterdir())


class Arbre:
    """L'état du banc : où sont les fichiers, ce que la collection contenait avant."""

    def __init__(self, racine: Path, client, points_avant: int, id_maximum: int,
                 signature_avant: str):
        self.racine = racine
        self.ingested = racine / "data" / "processed" / "ingested"
        self.lexical = racine / "data" / "lexical"
        self.journal = racine / "rag" / "ingestion" / "journal"
        self.cache_vecteurs = racine / "rag" / "ingestion" / ".cache"
        self.registre = racine / "rag" / "ingestion" / "registry-v1.json"
        self.imported_rows = racine / "rag" / "ingestion" / "imported-rows.jsonl"
        self.metadonnees = racine / "rag" / "metadata" / "documents-metadata-v1.json"
        self.overrides = racine / "rag" / "metadata" / "overrides.json"
        self.tables = racine / "rag" / "tables" / "tables-markdown-v1.json"
        self.vecteurs_tables = racine / "rag" / "tables" / ".cache" / "vectors-tables-markdown-v1.npz"
        self.titres = racine / "rag" / "titles" / "titles-clean-v1.json"
        self.vecteurs_titres = racine / "rag" / "titles" / ".cache" / "vectors-clean-titles-v1.npz"
        self.client = client
        self.points_avant = points_avant
        self.id_maximum = id_maximum
        self.signature_avant = signature_avant

    def compte(self) -> int:
        return self.client.count(quant_rag.COLLECTION, exact=True).count

    def journal_de(self, livraison: str = LIVRAISON) -> dict:
        return json.loads((self.journal / f"{livraison}.json").read_text(encoding="utf-8"))

    def registre_charge(self) -> dict:
        return json.loads(self.registre.read_text(encoding="utf-8"))


# --------------------------------------------------------------------- fabrication du banc

@pytest.fixture(scope="session")
def livraison_neuve(tmp_path_factory) -> Path:
    """Une livraison d'un document neuf, fabriquée une fois pour toute la session.

    ``--distinct`` casse les 8-grammes du document d'origine : sans lui la livraison est un
    doublon parfait du document dont elle dérive, et la garde de quasi-doublon la refuse —
    c'est la garde qui fait son travail, pas un défaut du banc.
    """
    sortie = tmp_path_factory.mktemp("livraisons") / "essai"
    fabrication = subprocess.run(
        [sys.executable, str(RACINE / "rag" / "ingestion" / "make_fixture.py"),
         "--single", SOURCE_LIVRAISON, "--distinct", "--out", str(sortie)],
        capture_output=True, text=True, cwd=str(RACINE))
    assert fabrication.returncode == 0, fabrication.stderr[-2000:]
    return sortie


@pytest.fixture
def arbre(tmp_path, monkeypatch, livraison_neuve) -> Arbre:
    """Un corpus jetable complet, et les vingt-six constantes qu'il faut détourner.

    La liste est le résultat de la mesure, pas une commodité : chaque ligne est un chemin
    qu'un module calcule depuis son propre ``__file__`` et qu'aucun appelant ne peut lui
    passer. Tant qu'elle est là, ce module est le seul moyen d'exercer l'import sans
    toucher au corpus servi.
    """
    racine = tmp_path / "arbre"
    ingested = racine / "data" / "processed" / "ingested"
    ingested.mkdir(parents=True)
    for nom in CORPUS_MINIATURE:
        shutil.copytree(REEL_INGESTED / nom, ingested / nom)
    for relatif in ("data/lexical", "rag/ingestion/journal", "rag/ingestion/.cache",
                    "rag/metadata", "rag/tables/.cache", "rag/titles/.cache"):
        (racine / relatif).mkdir(parents=True, exist_ok=True)

    documents = [json.loads((ingested / nom / "document.json").read_text(encoding="utf-8"))
                 for nom in CORPUS_MINIATURE]
    (racine / "rag" / "metadata" / "documents-metadata-v1.json").write_text(json.dumps({
        "summary": {"documents": len(documents)},
        "documents": [{"document_id": d["document_id"], "filename": d["filename"],
                       "title": f"Titre consolidé de {d['filename']}", "authors": ["Témoin, A."],
                       "publication_year": 2016, "first_year": None, "venue": None,
                       "doc_type": None, "short_ref": "Témoin (2016)",
                       "provenance": {"title": "manual", "authors": "manual", "year": "manual"}}
                      for d in documents]}, ensure_ascii=False, indent=1), encoding="utf-8")
    (racine / "rag" / "metadata" / "duplicates-v1.json").write_text(
        json.dumps({"version": "duplicates-essai", "remove": []}, ensure_ascii=False, indent=1),
        encoding="utf-8")
    (racine / "rag" / "metadata" / "overrides.json").write_text("{}\n", encoding="utf-8")
    # Trois témoins dans l'overlay de tableaux : l'import doit l'*étendre*, jamais le
    # réécrire. Un overlay tronqué rendrait du HTML brut au retrieval (0,730 contre 0,773).
    #
    # **Écrit COMPACT, sans indentation** — et c'est important. Le fichier réellement versionné
    # l'est : `git show HEAD:rag/tables/tables-markdown-v1.json | wc -l` rend **0**, et
    # `rag/tables/appliquer_en_tetes.py` l'écrit ainsi. Le semer indenté, comme ce banc
    # le faisait jusqu'au 11 septembre 2026, modélisait une forme que le dépôt n'emploie
    # nulle part — et rendait invisible ici le défaut que le lot D a payé en 5 933 lignes de
    # diff par import.
    (racine / "rag" / "tables" / "tables-markdown-v1.json").write_text(json.dumps({
        "version": "tables-markdown-v1",
        "chunks": {f"chunk-temoin-{i}": f"| entête {i} |\n| --- |\n| valeur {i} |" for i in (1, 2, 3)},
    }, ensure_ascii=False), encoding="utf-8")
    (racine / "rag" / "titles" / "titles-clean-v1.json").write_text(
        json.dumps({"version": "titles-clean-v1", "titles": {}}, ensure_ascii=False, indent=1),
        encoding="utf-8")
    for chemin, graine in ((racine / "rag" / "tables" / ".cache" / "vectors-tables-markdown-v1.npz", 11),
                           (racine / "rag" / "titles" / ".cache" / "vectors-clean-titles-v1.npz", 12)):
        np.savez_compressed(chemin, chunk_ids=np.asarray([f"chunk-temoin-{i}" for i in (1, 2, 3)]),
                            vectors=_vecteurs(3, graine))

    # --- la collection : identifiants NON contigus et ne partant pas de zéro, pour que
    # --- « max(id) + 1 » et « compte + i » ne puissent pas être confondus par accident.
    stockage = racine / "qdrant_storage_local"
    plages = {CORPUS_MINIATURE[0]: [0, 1], CORPUS_MINIATURE[1]: list(range(10, 21))}
    fabricant = QdrantClient(path=str(stockage))
    fabricant.create_collection(
        quant_rag.COLLECTION,
        vectors_config=models.VectorParams(size=DIMENSION, distance=models.Distance.COSINE))
    points, graine = [], 100
    for nom, identifiants in plages.items():
        document = json.loads((ingested / nom / "document.json").read_text(encoding="utf-8"))
        eligibles = [c for c in _lignes(ingested / nom / "chunks.jsonl")
                     if c.get("rag_eligible") is True]
        assert len(eligibles) == len(identifiants), f"{nom} : le banc a dérivé du corpus réel"
        vecteurs = _vecteurs(len(eligibles), graine)
        graine += 1
        for identifiant, chunk, vecteur in zip(identifiants, eligibles, vecteurs):
            points.append(models.PointStruct(id=identifiant, vector=vecteur.tolist(), payload={
                "chunk_id": chunk["chunk_id"], "document_id": document["document_id"],
                "text": chunk.get("text") or "", "title": f"Titre consolidé de {document['filename']}",
                "title_path": chunk.get("title_path"), "content_type": chunk.get("content_type"),
                "page_start": chunk.get("page_start"), "page_end": chunk.get("page_end"),
                "parent_id": chunk.get("parent_id"), "rag_eligible": True,
                "corpus_version": "ingested-all"}))
    fabricant.upsert(quant_rag.COLLECTION, wait=True, points=points)
    fabricant.close()          # l'embarqué n'accepte qu'un accès : on rend la main avant de patcher

    # --- les vingt-six constantes, module par module.
    for cible, nom, valeur in (
        (apply_delivery, "ROOT", racine),
        (apply_delivery, "INCOMING", racine / "data" / "processed" / "incoming"),
        (apply_delivery, "INGESTED", ingested),
        (apply_delivery, "JOURNAL_DIR", racine / "rag" / "ingestion" / "journal"),
        (apply_delivery, "VECTOR_DIR", racine / "rag" / "ingestion" / ".cache"),
        (apply_delivery, "METADATA", racine / "rag" / "metadata" / "documents-metadata-v1.json"),
        # ``describe()`` — appelée par ``rebuild_bm25`` — exprime chaque overlay en
        # relatif de ``HERE.parent`` : un overlay hors de l'arbre du module la fait
        # échouer par ValueError. HERE est donc à détourner comme le reste.
        (corpus_overlay, "HERE", racine / "rag"),
        (corpus_overlay, "DUPLICATES", racine / "rag" / "metadata" / "duplicates-v1.json"),
        (corpus_overlay, "TABLES", racine / "rag" / "tables" / "tables-markdown-v1.json"),
        (corpus_overlay, "TITLES", racine / "rag" / "titles" / "titles-clean-v1.json"),
        (corpus_overlay, "TITLE_VECTORS", racine / "rag" / "titles" / ".cache" / "vectors-clean-titles-v1.npz"),
        (corpus_overlay, "REGISTRY", racine / "rag" / "ingestion" / "registry-v1.json"),
        (corpus_overlay, "METADATA", racine / "rag" / "metadata" / "documents-metadata-v1.json"),
        (registry, "PATH", racine / "rag" / "ingestion" / "registry-v1.json"),
        (registry, "IMPORTED_ROWS", racine / "rag" / "ingestion" / "imported-rows.jsonl"),
        (registry, "INGESTED", ingested),
        (inspect_delivery, "ROOT", racine),
        (inspect_delivery, "INGESTED", ingested),
        (inspect_delivery, "METADATA", racine / "rag" / "metadata" / "documents-metadata-v1.json"),
        (inspect_delivery, "QUESTIONS", []),
        (quant_rag, "STORAGE", stockage),
        (quant_rag, "LEXICAL_DIR", racine / "data" / "lexical"),
        (quant_rag, "METADATA_PATH", racine / "rag" / "metadata" / "documents-metadata-v1.json"),
        (quant_rag, "BM25_TITLE_SOURCE", "registry"),
        (convert_tables, "PROCESSED", ingested),
        (extract_metadata, "PAPERS", racine / "data" / "papers"),
        # Vingt-septième constante, ajoutée le 9 septembre 2026 avec le verrou applicatif.
        # Un verrou est un état de machine : le laisser pointer sur l'arbre réel ferait
        # qu'un test poserait un fichier de verrou dans le dépôt.
        (verrou_collection, "VERROU", racine / "rag" / "ingestion" / "verrou-collection.json"),
        # Vingt-huitième, ajoutée le 10 septembre 2026 avec la garde du gel dans ``apply()``.
        # Sans elle, ces tests lisent le gel RÉEL du dépôt et le comparent à la signature de
        # l'arbre jetable — deux valeurs qui n'ont aucune raison de coïncider. La collection
        # de ce module vaut f66e3c3d76, la signature gelée du dépôt e1bdf36e2e : l'import
        # était refusé, à raison, pour un corpus qui n'est pas celui que le gel protège.
        # Le fichier détourné n'existe pas, donc ``lire()`` rend ``actif: False``, donc la
        # garde laisse passer — ce qui est bien la règle : on ne gèle pas un corpus jetable.
        (gel_corpus, "DECLARATION", racine / "rag" / "gel-corpus.json"),
    ):
        monkeypatch.setattr(cible, nom, valeur)
    # Une variable d'environnement suffirait à faire ouvrir un vrai serveur Qdrant.
    monkeypatch.delenv("QUANT_RAG_QDRANT_URL", raising=False)
    # Les mémoïsations survivent au remplacement des constantes : les purger fait partie
    # du détournement, et leur existence fait partie de la dette.
    corpus_overlay.invalidate()
    quant_rag.client.cache_clear()
    quant_rag.bm25.cache_clear()
    quant_rag.document_metadata.cache_clear()

    client = quant_rag.client()
    assert client.count(quant_rag.COLLECTION, exact=True).count == len(points)

    registry.PATH.write_text(json.dumps(registry.build(), ensure_ascii=False, indent=1),
                             encoding="utf-8")
    registry.write_imported_rows()
    corpus_overlay.invalidate()

    # Le titre `manual` que la garde de métadonnées exige, semé pour cette livraison.
    document_livre = json.loads((_dossier_livre(livraison_neuve) / "document.json")
                                .read_text(encoding="utf-8"))
    (racine / "rag" / "metadata" / "overrides.json").write_text(json.dumps(
        {document_livre["filename"]: {"title": TITRE_MANUEL, "note": "document d'essai"}},
        ensure_ascii=False, indent=1), encoding="utf-8")

    # Les vecteurs de la livraison, déposés là où `embed_delivery` les cherche : il rend la
    # main sans charger le modèle si `vectors-<id>.npz` existe déjà (reprise sur interruption).
    eligibles = [c["chunk_id"] for c in _lignes(_dossier_livre(livraison_neuve) / "chunks.jsonl")
                 if c.get("rag_eligible") is True]
    np.savez_compressed(racine / "rag" / "ingestion" / ".cache" / f"vectors-{LIVRAISON}.npz",
                        chunk_ids=np.asarray(eligibles), vectors=_vecteurs(len(eligibles), 7))

    etat = Arbre(racine, client, len(points), max(max(v) for v in plages.values()),
                 corpus_overlay.signature())
    yield etat
    # Purge à la SORTIE autant qu'à l'entrée. ``monkeypatch`` défait bien les constantes,
    # mais pas les mémoïsations qui ont été remplies pendant qu'elles pointaient ailleurs :
    # ``corpus_overlay`` garderait alors l'empreinte du registre miniature, et le test
    # voisin ``test_signature_freshness`` comparerait deux corpus différents en croyant
    # n'en voir qu'un. Constaté le 8 septembre 2026 : il rendait `a256e860a6` au lieu de
    # `5530cba145` dès qu'il tournait après ce module.
    client.close()
    quant_rag.client.cache_clear()
    quant_rag.bm25.cache_clear()
    quant_rag.document_metadata.cache_clear()
    corpus_overlay.invalidate()


@pytest.fixture
def import_reussi(arbre, livraison_neuve):
    """Un import mené à son terme — le point de départ des sept propriétés vérifiées ici."""
    resultat = apply_delivery.apply(livraison_neuve, None, True, LIVRAISON)
    assert resultat["status"] == "COMPLETED", resultat
    return resultat


# --------------------------------------------------------------------- 1. plage d'identifiants

def test_les_points_occupent_une_plage_contigue_a_partir_du_maximum(arbre, import_reussi):
    """``max(id) + 1``, jamais ``compte + i`` — la règle amont écraserait des points existants.

    La collection servie a 807 trous : des documents retirés à la main y ont laissé des
    identifiants libres au milieu. Écrire à ``compte + i`` viserait donc des identifiants
    déjà pris, et l'écrasement serait **silencieux** — un upsert ne refuse rien. Le banc
    reproduit exactement cette forme : 13 points, identifiants 0-1 et 10-20.
    """
    journal = arbre.journal_de()
    assert journal["point_ids"]["first"] == arbre.id_maximum + 1, (
        "le premier identifiant n'est pas max(id)+1 : des points existants seraient écrasés")
    assert journal["point_ids"]["first"] != arbre.points_avant, (
        "premier identifiant égal au compte : c'est la règle amont, celle qui écrase")

    par_document = _points_par_document(arbre.client, quant_rag.COLLECTION)
    document_id = journal["documents"][0]["document_id"]
    neufs = par_document[document_id]
    attendu = list(range(journal["point_ids"]["first"], journal["point_ids"]["last"] + 1))
    assert neufs == attendu, "la plage écrite n'est pas contiguë"
    assert len(neufs) == import_reussi["points"] == journal["point_ids"]["count"]

    anciens = sorted(i for document, ids in par_document.items()
                     if document != document_id for i in ids)
    assert anciens == [0, 1] + list(range(10, 21)), (
        "les points d'avant l'import ont bougé — un import n'écrase jamais l'existant")


# --------------------------------------------------------------------- 2. registre : point_ids

def test_le_registre_porte_les_point_ids_de_la_livraison(arbre, import_reussi):
    """Sans la plage au registre, ``--rollback`` n'a plus rien à supprimer.

    Le registre est le seul fichier qui dise quels points appartiennent à quel document :
    le payload Qdrant porte le ``document_id``, mais retrouver la plage par un balayage
    complet suppose que la collection soit encore lisible. C'est aussi ce que
    ``registry --verify`` confronte pour dire qu'un document actif est bien servi.
    """
    journal = arbre.journal_de()
    document_id = journal["documents"][0]["document_id"]
    entree = next(d for d in arbre.registre_charge()["documents"]
                  if d["document_id"] == document_id)

    assert entree["point_ids"] == {"first": journal["point_ids"]["first"],
                                   "last": journal["point_ids"]["last"],
                                   "count": journal["point_ids"]["count"],
                                   "contiguous": True}
    assert entree["delivery"]["id"] == LIVRAISON, "le document n'est pas rattaché à sa livraison"
    assert entree["status"] == "active"
    livraisons = {d["id"]: d for d in arbre.registre_charge()["deliveries"]}
    assert livraisons[LIVRAISON]["documents"] == 1
    # Le relais que lit le graphe : sans lui, l'extraction d'entités serait aveugle à l'import.
    assert len(_lignes(arbre.imported_rows)) == import_reussi["points"]


# --------------------------------------------------------------------- 3. sommes canoniques

def test_le_registre_enregistre_les_sha256_des_quatre_fichiers_canoniques(arbre, import_reussi,
                                                                          livraison_neuve):
    """Les sommes sont la seule preuve qu'un document promu est bien celui qui a été livré.

    ``registry --verify`` compare ``files["chunks.jsonl"]`` au disque pour dire « modifié
    depuis l'enregistrement ». Une somme absente ou calculée sur autre chose que le fichier
    promu vide cette vérification de son sens, sans qu'elle échoue jamais.
    """
    document_id = arbre.journal_de()["documents"][0]["document_id"]
    entree = next(d for d in arbre.registre_charge()["documents"]
                  if d["document_id"] == document_id)
    promu = arbre.ingested / entree["folder"]
    livre = _dossier_livre(livraison_neuve)

    assert set(entree["files"]) == set(inspect_delivery.REQUIRED_FILES)
    for nom in inspect_delivery.REQUIRED_FILES:
        assert entree["files"][nom] == _sha256(promu / nom), f"{nom} : somme du registre fausse"
        assert entree["files"][nom] == _sha256(livre / nom), (
            f"{nom} : le fichier promu diffère du fichier livré")


# --------------------------------------------------------------------- 4. manifeste BM25

def test_le_manifeste_bm25_compte_autant_de_points_que_la_collection(arbre, import_reussi):
    """``built_from.points`` est le seul témoin **hors ligne** de ce que la collection sert.

    ``quant_rag._bm25_matches_collection()`` s'en sert pour refuser un index périmé, et
    ``qdrant_backend.verdict()`` pour refuser un serveur qui porte une copie ancienne. Un
    manifeste qui compterait autre chose que la collection rendrait les deux gardes muettes
    — c'est-à-dire pires qu'absentes.
    """
    manifeste = json.loads(quant_rag.bm25_manifest_path().read_text(encoding="utf-8"))
    assert manifeste["built_from"]["points"] == arbre.compte()
    assert manifeste["built_from"]["points"] == arbre.points_avant + import_reussi["points"]
    assert manifeste["built_from"]["collection"] == quant_rag.COLLECTION
    assert manifeste["built_from"]["documents"] == len(CORPUS_MINIATURE) + 1
    assert manifeste["records"] == manifeste["built_from"]["points"]
    # Le nom porte la signature, le manifeste porte l'empreinte du fichier : les deux
    # gardes que ce dépôt a payées pour un index juste sous un nom qui ment.
    assert manifeste["corpus_signature"] == corpus_overlay.signature()
    assert manifeste["index_sha256"] == _sha256(quant_rag.bm25_path())
    assert manifeste["index"] == quant_rag.bm25_path().name


# --------------------------------------------------------------------- 5. overlays intacts

def test_les_overlays_ne_sont_ni_tronques_ni_corrompus_par_l_import(arbre, import_reussi):
    """L'import **étend** l'overlay de tableaux ; il ne le réécrit pas, et il n'y touche pas ailleurs.

    ``merge_table_overlay`` relit, fusionne et réécrit tout le fichier : une fusion qui
    perdrait les entrées d'avant rendrait du HTML brut au retrieval sur les chunks
    concernés — 0,730 contre 0,773 de nDCG, et sans le moindre message. Les vecteurs des
    overlays (tableaux, titres) n'ont aucune raison d'être touchés par un import ; le jour
    où ils le seraient, ce test le dirait.
    """
    overlay = json.loads(arbre.tables.read_text(encoding="utf-8"))
    assert overlay["version"] == "tables-markdown-v1", "l'import a perdu l'entête de l'overlay"
    for i in (1, 2, 3):
        assert overlay["chunks"][f"chunk-temoin-{i}"] == (
            f"| entête {i} |\n| --- |\n| valeur {i} |"), "témoin d'overlay altéré par l'import"
    ajoutes = arbre.journal_de()["added_table_chunks"]
    assert ajoutes, "la livraison porte deux tableaux : l'overlay aurait dû s'étendre"
    assert len(overlay["chunks"]) == 3 + len(ajoutes)
    assert set(ajoutes) <= set(overlay["chunks"])

    for chemin, graine in ((arbre.vecteurs_tables, 11), (arbre.vecteurs_titres, 12)):
        blob = np.load(chemin)
        assert blob["vectors"].shape == (3, DIMENSION), f"{chemin.name} tronqué"
        assert np.array_equal(blob["vectors"], _vecteurs(3, graine)), f"{chemin.name} corrompu"
        assert blob["chunk_ids"].tolist() == [f"chunk-temoin-{i}" for i in (1, 2, 3)]
    assert json.loads(arbre.titres.read_text(encoding="utf-8"))["version"] == "titles-clean-v1"


# --------------------------------------------------------------------- 6. annulation

def test_le_rollback_rend_la_signature_d_avant_et_retire_les_points(arbre, import_reussi):
    """La signature d'avant doit revenir **à l'identique**, sinon l'annulation n'annule rien.

    La signature nomme l'index BM25 et les caches du banc. Une annulation qui rendrait le
    contenu mais pas la signature laisserait le corpus d'avant sous un nom que plus rien ne
    cherche : tout serait reconstruit, et le lien avec les mesures déjà publiées serait
    rompu. La première annulation d'import a échoué exactement là — le contenu revenait, la
    sérialisation non — d'où l'empreinte sur le *contenu* JSON et non sur les octets.
    """
    signature_apres_import = corpus_overlay.signature()
    assert signature_apres_import != arbre.signature_avant, (
        "la signature n'a pas bougé après un import : l'index BM25 garderait son nom et "
        "serait rechargé incomplet")

    resultat = apply_delivery.rollback(LIVRAISON)

    assert resultat["status"] == "ROLLED_BACK"
    assert resultat["points_removed"] == import_reussi["points"]
    assert arbre.compte() == arbre.points_avant
    assert corpus_overlay.signature() == arbre.signature_avant, (
        "la signature d'avant n'est pas revenue : le corpus est restauré, son nom non")
    assert resultat["corpus_signature"] == arbre.signature_avant

    par_document = _points_par_document(arbre.client, quant_rag.COLLECTION)
    assert set(par_document) == {json.loads((arbre.ingested / nom / "document.json")
                                            .read_text(encoding="utf-8"))["document_id"]
                                 for nom in CORPUS_MINIATURE}
    assert sorted(i for ids in par_document.values() for i in ids) == [0, 1] + list(range(10, 21))
    assert arbre.journal_de()["state"] == "rolled-back"
    assert not (arbre.cache_vecteurs / f"vectors-{LIVRAISON}.npz").exists()
    assert len(arbre.registre_charge()["documents"]) == len(CORPUS_MINIATURE)


# --------------------------------------------------------------------- 7. sha256 déjà connu

def test_un_sha256_deja_connu_n_est_jamais_reimporte(arbre, import_reussi, livraison_neuve,
                                                     tmp_path):
    """Le ``sha256`` du PDF est la clé d'identité — le relivrer ne doit rien ajouter.

    Deux formes, et elles n'ont pas la même issue voulue :

      - livraison identique : le document est « sans-effet », il n'y a **rien** à ajouter.
        Un chemin qui le prendrait pour un ajout doublerait le texte dans l'espace
        vectoriel, avec deux jeux de ``chunk_id`` pour le même contenu ;
      - livraison au contenu modifié : c'est une **révision**, et elle est refusée. Le
        ``chunk_id`` est haché sur ``(document_id, indice, texte)`` : une révision
        renumérote les identifiants du document et peut détruire l'or des bancs
        (157 chunks d'or dans 133 documents).
    """
    compte_avant = arbre.compte()

    rejoue = apply_delivery.apply(livraison_neuve, None, True, "essai-02")
    assert rejoue == {"status": "NOOP"}, "un sha256 déjà connu a produit un nouvel import"
    assert arbre.compte() == compte_avant
    assert not (arbre.journal / "essai-02.json").exists()

    revision = tmp_path / "revision"
    shutil.copytree(livraison_neuve, revision)
    dossier = _dossier_livre(revision)
    chunks = _lignes(dossier / "chunks.jsonl")
    cible = next(c for c in chunks if c.get("rag_eligible") is True)
    cible["text"] = (cible.get("text") or "") + "\n\nParagraphe ajouté par une révision amont."
    (dossier / "chunks.jsonl").write_text(
        "".join(json.dumps(c, ensure_ascii=False) + "\n" for c in chunks), encoding="utf-8")
    # Le manifeste doit rester cohérent, sinon le refus viendrait de la somme de contrôle
    # et non de l'identité — ce n'est pas la garde que ce test éprouve.
    manifeste = json.loads((revision / "manifest.json").read_text(encoding="utf-8"))
    manifeste["documents"][0]["chunks_jsonl_sha256"] = _sha256(dossier / "chunks.jsonl")
    (revision / "manifest.json").write_text(json.dumps(manifeste, ensure_ascii=False, indent=1),
                                            encoding="utf-8")

    with pytest.raises(SystemExit) as refus:
        apply_delivery.apply(revision, None, True, "essai-03")
    assert "REFUS" in str(refus.value) and "ajouts" in str(refus.value)
    assert arbre.compte() == compte_avant, "une révision refusée a tout de même écrit"
    assert not (arbre.journal / "essai-03.json").exists()


# --------------------------------------------------------------------- 8. constat : --sans-bm25

def test_sans_bm25_inscrit_la_dette_au_journal_et_ne_reconstruit_pas(arbre, livraison_neuve):
    """CONSTAT — ce test décrit le comportement de ``--sans-bm25``, il ne le juge pas.

    Le mode « fin de lot » de ``batch_driver`` diffère la reconstruction de l'index lexical :
    la rebâtir après chaque document coûte 7,7 s et 50,8 Mo dont un seul exemplaire survit
    au lot. La contrepartie est que **l'index lexical est en retard sur la collection**
    jusqu'au geste de fin de lot. La dette est inscrite dans le journal de livraison
    (``bm25_en_retard``), et c'est ce que relisent la reprise et
    ``verifier_installation.py``. Ce que ce test fixe, c'est que la dette est bien
    **écrite** et que l'index n'est bien **pas** reconstruit — les deux moitiés du contrat.
    """
    resultat = apply_delivery.apply(livraison_neuve, None, True, LIVRAISON, (), True)

    assert resultat["status"] == "COMPLETED"
    assert resultat["bm25_en_retard"] is True
    assert resultat["bm25"] is None
    assert arbre.journal_de()["bm25_en_retard"] is True, (
        "la dette n'est pas inscrite au journal : rien ne dirait qu'un geste reste à faire")

    assert not quant_rag.bm25_path().exists(), "l'index a été reconstruit malgré --sans-bm25"
    assert not quant_rag.bm25_manifest_path().exists()
    assert list(arbre.lexical.iterdir()) == [], (
        "un artefact lexical a été écrit alors que la reconstruction était différée")
    # Et le corpus, lui, a bien changé : c'est précisément ce qui rend la dette réelle.
    assert arbre.compte() == arbre.points_avant + resultat["points"]
    assert corpus_overlay.signature() != arbre.signature_avant


# ------------------------------------------------- 9. l'en-tête des tableaux fragmentés (§11.1)

@pytest.fixture(scope="session")
def livraison_tableau_fragmente(tmp_path_factory) -> Path:
    """Une livraison qui porte un tableau coupé en trois fragments enchaînés.

    Seul le premier fragment porte la rangée d'en-tête ; c'est la forme que le chunker amont
    produit, et celle que l'étape 4 traitait mal.
    """
    sortie = tmp_path_factory.mktemp("livraisons") / "tableau"
    fabrication = subprocess.run(
        [sys.executable, str(RACINE / "rag" / "ingestion" / "make_fixture.py"),
         "--single", SOURCE_LIVRAISON, "--distinct", "--tableau-fragmente",
         "--out", str(sortie)],
        capture_output=True, text=True, cwd=str(RACINE))
    assert fabrication.returncode == 0, fabrication.stderr[-2000:]
    return sortie


def test_un_tableau_fragmente_herite_de_l_en_tete_de_sa_premiere_partie(
        arbre, livraison_tableau_fragmente):
    """§11.1 — le défaut qui touche la QUALITÉ DES RÉPONSES, et non la robustesse.

    Un tableau trop grand pour un chunk est découpé ; seul le premier fragment porte la
    rangée d'en-tête. La passe de conversion du corpus recopie cet en-tête sur les
    fragments suivants — sans quoi un passage servi est un tableau de chiffres dont les
    colonnes n'ont plus de nom, et « un tableau servi sans son en-tête est une défaillance
    fonctionnelle, pas un détail ».

    À l'import, cette recopie n'avait jamais lieu : ``part_chains()`` lisait
    ``data/processed/ingested``, où la livraison en vol n'est pas encore. Mesuré sur le
    corpus servi : **1 066 fragments** de 100 documents importés, soit 4,08 % des 26 120
    chunks, servis sans leur en-tête (``rag/tables/recenser_en_tetes.py``).

    Ce test échoue sur le code d'avant le 9 septembre 2026 : ``fragments_with_inherited_header``
    y vaut **0**, pour trois fragments dont deux devraient hériter.
    """
    # Des vecteurs déposés d'avance : ``embed_delivery`` saute alors le modèle. Sans cela ce
    # test chargerait Qwen3 pour quinze secondes, et mesurerait un embedding au lieu d'une
    # conversion de tableau.
    chunks = _lignes(_dossier_livre(livraison_tableau_fragmente) / "chunks.jsonl")
    eligibles = [c["chunk_id"] for c in chunks if c.get("rag_eligible") is True]
    np.savez_compressed(arbre.cache_vecteurs / "vectors-essai-tableau.npz",
                        chunk_ids=np.asarray(eligibles), vectors=_vecteurs(len(eligibles), 11))

    resultat = apply_delivery.apply(livraison_tableau_fragmente, None, True, "essai-tableau")
    assert resultat["status"] == "COMPLETED", resultat
    stats = resultat["tables"]
    assert stats.get("fragments_with_inherited_header", 0) >= 2, (
        "les fragments 2 et 3 n'ont pas hérité de l'en-tête du premier — l'étape 4 ne voit "
        f"pas la livraison en vol. Statistiques rendues : {stats}")

    # et l'en-tête est bien DANS le texte servi, pas seulement dans un compteur
    overlay = json.loads(arbre.tables.read_text(encoding="utf-8"))["chunks"]
    fragments = [c for c in _lignes(_dossier_livre(livraison_tableau_fragmente) / "chunks.jsonl")
                 if c.get("content_type") == "table" and c.get("previous_chunk_id")]
    herites = [overlay[c["chunk_id"]] for c in fragments if c["chunk_id"] in overlay]
    assert herites, "aucun fragment converti n'est entré dans l'overlay"
    assert any("Horizon" in texte and "Sharpe" in texte and "Turnover" in texte
               for texte in herites), (
        "le texte servi d'un fragment ne porte toujours pas les noms de colonnes du premier "
        f"fragment. Textes : {[t[:120] for t in herites]}")

    # ET la garde de parent : le SECOND tableau, voisin du premier dans la chaîne mais
    # d'un autre parent, ne doit PAS être coiffé de l'en-tête du premier. Sans ``parent_id``
    # dans le payload, la remontée franchit la frontière entre deux tableaux — et un
    # fixture à un seul tableau ne le verrait pas (None == None).
    second = [c for c in chunks if c.get("content_type") == "table"
              and "Volatilité" in (c.get("text") or "") or "RTY" in (c.get("text") or "")]
    textes_du_second = [overlay[c["chunk_id"]] for c in second if c["chunk_id"] in overlay]
    assert textes_du_second, "le second tableau n'est pas entré dans l'overlay"
    assert not any("Horizon" in texte or "Turnover" in texte for texte in textes_du_second), (
        "le second tableau a hérité de l'en-tête du PREMIER : la garde de parent_id ne "
        f"joue pas. Textes : {[t[:140] for t in textes_du_second]}")


# ------------------------------------------- 10. le verrou applicatif de la collection (§B.2)

def test_un_second_import_simultane_est_refuse_et_n_ecrit_rien(arbre, livraison_neuve):
    """Le prérequis nommé de toute bascule serveur.

    ``next_point_id()`` attribue par ``max(id) + 1``, calculé par un scroll : une **lecture**
    puis une **écriture**. Deux importeurs simultanés y liraient le même maximum, et le
    second effacerait les points du premier — sans erreur des deux côtés, avec un
    ``status: COMPLETED`` partout. Aujourd'hui le verrou exclusif du stockage embarqué
    l'interdit ; **sur un serveur, ce verrou disparaît** (§5 de BASCULE-QDRANT-SERVEUR.md).

    Ce test simule le premier importeur en tenant le verrou, puis lance le second.
    """
    points_avant = arbre.compte()
    registre_avant = arbre.registre.read_bytes()

    verrou_collection.acquerir("apply_delivery essai-concurrent (pid simulé)")
    try:
        with pytest.raises(verrou_collection.VerrouTenu) as leve:
            apply_delivery.apply(livraison_neuve, None, True, LIVRAISON)
    finally:
        verrou_collection.rendre()

    message = str(leve.value)
    assert "essai-concurrent" in message, "le refus ne nomme pas le tenant"
    assert "max(id)" in message, "le refus n'explique pas ce qu'il empêche"
    assert arbre.compte() == points_avant, "le second import a écrit des points"
    assert arbre.registre.read_bytes() == registre_avant, "le second import a touché le registre"
    assert not list(arbre.journal.glob("*.json")), "le second import a ouvert un journal"


def test_le_verrou_est_rendu_meme_quand_l_import_echoue(arbre, livraison_neuve, monkeypatch):
    """Une garde qui transforme une panne en blocage d'une heure coûte plus qu'elle ne protège."""
    monkeypatch.setattr(apply_delivery, "_apply",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("panne simulée")))
    with pytest.raises(RuntimeError, match="panne simulée"):
        apply_delivery.apply(livraison_neuve, None, True, LIVRAISON)
    assert verrou_collection.etat() is None, "le verrou est resté pris après une panne"


def test_un_verrou_dont_le_processus_est_mort_est_repris_bruyamment(arbre, capsys):
    """Un `kill -9` laisse son fichier derrière lui. Un verrou qu'on ne peut pas reprendre
    bloque le dépôt plus sûrement qu'il ne le protège — mais la reprise doit se voir."""
    verrou_collection.VERROU.write_text(json.dumps({
        "proprietaire": "apply_delivery mort-au-champ-d-honneur",
        "pid": 999_999,                       # aucun processus ne porte ce pid
        "horodatage": "2026-09-09T00:00:00+00:00", "horodatage_epoch": 0.0,
        "hote": "essai"}), encoding="utf-8")

    contenu = verrou_collection.acquerir("apply_delivery repreneur")
    try:
        assert contenu["proprietaire"] == "apply_delivery repreneur"
        assert "verrou périmé repris" in capsys.readouterr().out
    finally:
        verrou_collection.rendre()


def test_un_verrou_dont_le_processus_est_vivant_n_est_jamais_repris(arbre):
    """Le cas symétrique, et c'est lui qui compte : reprendre un verrou vivant ferait
    exactement le dégât que le verrou existe pour empêcher."""
    verrou_collection.VERROU.write_text(json.dumps({
        "proprietaire": "apply_delivery bien vivant",
        "pid": os.getpid(),                   # nous : donc vivant, par construction
        "horodatage": "2026-09-09T12:00:00+00:00",
        "horodatage_epoch": time.time(), "hote": "essai"}), encoding="utf-8")
    try:
        with pytest.raises(verrou_collection.VerrouTenu, match="bien vivant"):
            verrou_collection.acquerir("apply_delivery intrus")
    finally:
        verrou_collection.rendre()


def test_le_rollback_prend_le_meme_verrou_que_l_import(arbre, import_reussi):
    """Une annulation retire des points et réécrit le registre : elle ne peut pas cohabiter
    avec un import qui calcule ``max(id)+1``."""
    verrou_collection.acquerir("apply_delivery en cours")
    try:
        with pytest.raises(verrou_collection.VerrouTenu):
            apply_delivery.rollback(LIVRAISON)
    finally:
        verrou_collection.rendre()
    assert apply_delivery.rollback(LIVRAISON)["status"] == "ROLLED_BACK"


# --------------------------------- 11. le journal couvre les étapes 3 et 4 (§11.2)

def test_une_mort_avant_le_premier_upsert_est_entierement_annulable(
        arbre, livraison_neuve):
    """§11.2 — les métadonnées et l'overlay des tableaux sont écrits AVANT le premier point.

    Tant que le journal n'était ouvert qu'à l'étape 6, une mort entre l'étape 3 et elle
    laissait des lignes orphelines qu'aucune annulation ne couvrait. Et une ligne de
    métadonnées orpheline change ``titles_digest``, donc la **signature**, donc le nom de
    l'index : le corpus n'était pas corrompu, mais il n'était plus à l'état d'avant, et
    ``apply_delivery`` concluait « rien n'a été écrit dans Qdrant » — vrai, et incomplet.

    On tue l'import à l'étape 5, après que les métadonnées et l'overlay ont été écrits.
    """
    signature_avant = corpus_overlay.signature()
    metadonnees_avant = arbre.metadonnees.read_bytes()
    tables_avant = arbre.tables.read_bytes()

    # La mort est simulée dans SON PROPRE contexte : ``monkeypatch.undo()`` défait tous les
    # détournements en cours, y compris les vingt-sept de la fixture ``arbre`` — et le
    # ``rollback`` qui suit irait alors chercher le journal du dépôt réel.
    with pytest.MonkeyPatch.context() as mort:
        mort.setattr(apply_delivery, "embed_delivery",
                     lambda *a, **k: (_ for _ in ()).throw(RuntimeError("mort simulée")))
        with pytest.raises(RuntimeError, match="mort simulée"):
            apply_delivery.apply(livraison_neuve, None, True, LIVRAISON)

    # l'import a bien laissé des traces dans le corpus servi — sinon le test ne prouve rien
    corpus_overlay.invalidate()
    assert arbre.metadonnees.read_bytes() != metadonnees_avant, (
        "les métadonnées n'ont pas bougé : la mort a été simulée trop tôt, le test "
        "ne prouverait rien")
    assert corpus_overlay.signature() != signature_avant, (
        "la signature n'a pas bougé alors que les métadonnées ont changé")

    journal = arbre.journal_de()
    assert journal["state"] == "pre-flight", journal["state"]
    assert journal["added_metadata"], "le journal n'a pas enregistré les métadonnées ajoutées"

    assert apply_delivery.rollback(LIVRAISON)["status"] == "ROLLED_BACK"
    corpus_overlay.invalidate()
    assert corpus_overlay.signature() == signature_avant, (
        "la signature n'est pas revenue à son état d'avant après --rollback")
    assert arbre.metadonnees.read_bytes() == metadonnees_avant
    assert arbre.tables.read_bytes() == tables_avant


# ------------------------- 12. la table des titres mémoïsée, purgée à l'import (§11.5)

@pytest.fixture(scope="session")
def livraison_seconde(tmp_path_factory) -> Path:
    """Une seconde livraison, dérivée de l'AUTRE document du corpus miniature.

    Il en faut une : les deux autres fixtures dérivent du même document, et une fois la
    première importée la seconde en est un quasi-doublon que la garde de contenu refuse —
    à juste titre. Deux imports d'un même processus demandent donc deux documents distincts.
    """
    sortie = tmp_path_factory.mktemp("livraisons") / "seconde"
    fabrication = subprocess.run(
        [sys.executable, str(RACINE / "rag" / "ingestion" / "make_fixture.py"),
         "--single", CORPUS_MINIATURE[0], "--distinct", "--out", str(sortie)],
        capture_output=True, text=True, cwd=str(RACINE))
    assert fabrication.returncode == 0, fabrication.stderr[-2000:]
    return sortie


def test_deux_imports_dans_le_meme_processus_voient_la_table_des_titres_a_jour(
        arbre, livraison_neuve, livraison_seconde):
    """§11.5 — ``document_metadata()`` est mémoïsée, et ``rebuild_bm25`` en tire les titres.

    ``apply_delivery`` purgeait ``corpus_overlay`` et ``quant_rag.bm25``, mais pas celle-là.
    Un second import dans le même processus reconstruisait donc l'index lexical sur une
    table de titres **périmée** : le titre du document qui vient d'entrer manquait à l'index,
    sous un nom d'index pourtant correct.

    Le test lit la mémoïsation elle-même, pas un effet de bord : c'est elle qui est en cause.
    """
    quant_rag.document_metadata.cache_clear()
    titres_avant = set(quant_rag.document_metadata())

    apply_delivery.apply(livraison_neuve, None, True, LIVRAISON)
    apres_premier = set(quant_rag.document_metadata())
    assert apres_premier > titres_avant, (
        "la table des titres n'a pas vu le premier import : elle est restée mémoïsée")

    chunks = _lignes(_dossier_livre(livraison_seconde) / "chunks.jsonl")
    eligibles = [c["chunk_id"] for c in chunks if c.get("rag_eligible") is True]
    np.savez_compressed(arbre.cache_vecteurs / "vectors-essai-02.npz",
                        chunk_ids=np.asarray(eligibles), vectors=_vecteurs(len(eligibles), 13))
    # Le titre du second document doit être une source acceptée, comme pour le premier.
    overrides = json.loads(arbre.overrides.read_text(encoding="utf-8"))
    document = json.loads((_dossier_livre(livraison_seconde) / "document.json").read_text(encoding="utf-8"))
    overrides[document["filename"]] = {"title": TITRE_MANUEL + " (second)"}
    arbre.overrides.write_text(json.dumps(overrides, ensure_ascii=False, indent=1), encoding="utf-8")
    apply_delivery.apply(livraison_seconde, None, True, "essai-02")
    apres_second = set(quant_rag.document_metadata())
    assert apres_second > apres_premier, (
        "la table des titres n'a pas vu le SECOND import du même processus — c'est le "
        "défaut §11.5, et rebuild_bm25 a indexé un corpus dont il manque un titre")

    # et l'index reconstruit porte bien le titre du second document
    index = json.loads(quant_rag.bm25_path().read_text(encoding="utf-8"))
    documents_indexes = {r["document_id"] for r in index["records"]}
    nouveaux = apres_second - apres_premier
    assert nouveaux <= documents_indexes, (
        f"le document {nouveaux} est dans la collection mais pas dans l'index lexical")


# ----------------------- 13. l'annulation vérifie à qui appartiennent les points (§11.4)

def test_une_plage_qui_chevauche_un_autre_document_est_refusee(arbre, import_reussi):
    """§11.4 — ``rollback`` supprimait une plage d'identifiants sans regarder son contenu.

    La seule protection était l'ordonnancement de ``batch_driver`` : une convention
    d'appelant, pas une garde. Une plage décalée — un journal rejoué après une
    reconstruction qui a réattribué les identifiants — aurait effacé les points d'un AUTRE
    document, et le rapport aurait dit ``ROLLED_BACK`` avec un ``points_removed`` cohérent.

    On décale ici la plage du journal vers le bas, sur des points d'un document du corpus
    miniature. L'annulation doit refuser, et ne rien supprimer.
    """
    journal = arbre.journal_de()
    points_avant = arbre.compte()
    par_document = _points_par_document(arbre.client, quant_rag.COLLECTION)
    voisins = sorted(ids for doc, ids in par_document.items()
                     if doc not in {d["document_id"] for d in journal["documents"]})[0]

    journal["point_ids"] = {"first": min(voisins), "last": max(voisins),
                            "count": len(voisins)}
    (arbre.journal / f"{LIVRAISON}.json").write_text(
        json.dumps(journal, ensure_ascii=False, indent=1), encoding="utf-8")

    with pytest.raises(SystemExit) as refus:
        apply_delivery.rollback(LIVRAISON)
    assert "n'appartiennent pas à cette livraison" in str(refus.value)
    assert arbre.compte() == points_avant, "l'annulation refusée a quand même supprimé"


def test_un_identifiant_deja_retire_n_empeche_pas_l_annulation(arbre, import_reussi):
    """Un point absent n'est pas un intrus : il a déjà été retiré, il n'y a rien à faire.

    La garde du test précédent ne doit pas transformer une annulation légitime en refus —
    c'est le défaut symétrique, et il rendrait le rollback inutilisable après une reprise.
    """
    plage = arbre.journal_de()["point_ids"]
    arbre.client.delete(quant_rag.COLLECTION, wait=True,
                        points_selector=models.PointIdsList(points=[plage["first"]]))
    assert apply_delivery.rollback(LIVRAISON)["status"] == "ROLLED_BACK"
