"""Ce que la couche macOS a changé au corpus livré, sous forme de fichiers versionnés.

L'export Qdrant (`data/qdrant-export/`) et `rows.jsonl` restent intacts : ce sont les
sources de référence. Les modifications sont des *overlays* appliqués au chargement,
partout où le corpus est lu — index embarqué (`build_index.py`), BM25, vue corpus du
banc d'essai (`benchmark/corpus.py`) — pour qu'un index rebâti et le banc voient
exactement le même corpus que la production :

  - `metadata/duplicates-v1.json`  documents retirés (doublons, éditions remplacées) ;
  - `tables/tables-markdown-v1.json`  texte des chunks-tableaux converti en Markdown.

Un quatrième fichier n'est pas un overlay mais définit le corpus lui-même :

  - `ingestion/registry-v1.json`  quels documents le composent, et d'où ils viennent.

`signature()` change dès qu'un overlay **ou l'ensemble des documents** change : elle
entre dans le nom des caches et dans celui de l'index BM25 (`bm25_name()`), de sorte
qu'un index bâti sur un autre état du corpus n'est jamais rechargé par erreur — il n'a
simplement pas le bon nom.

Le registre y a été ajouté le 4 septembre 2026, en ouvrant le chantier d'ingestion : la
signature ne couvrait que les deux overlays, donc **ajouter des documents ne la changeait
pas**. L'index BM25 aurait gardé son nom, `quant_rag.bm25()` l'aurait rechargé — incomplet,
sans un bruit. Le mécanisme bâti pour empêcher exactement cela n'y protégeait pas.
La signature ne couvre que des *entrées* : ce qui en dérive (BM25, graphe, caches du banc)
est nommé par elle, jamais l'inverse — sans quoi rebâtir le graphe invaliderait l'index
qu'on vient de reconstruire.

Un troisième overlay ne touche que les *vecteurs*, pas le texte : `titles/titles-clean-v1.json`
(titre consolidé au lieu du titre d'export en tête de chaque texte embarqué, 3 septembre
2026 ; vecteurs dans `titles/.cache/vectors-clean-titles-v1.npz`, régénérés par
`titles/reembed_titles.py`). Il n'entre pas dans `signature()`, mais `describe()` le
rapporte, pour que chaque fichier de résultats dise sur quels vecteurs il a été mesuré.

Ce paragraphe disait, jusqu'au 5 septembre 2026 : « BM25, les caches du banc et le graphe
lisent le texte, qui n'a pas changé ». **C'était faux pour BM25**, qui lit le texte *et le
titre du document* (`retrieval.lexical.lexical_text` met le titre en tête du texte indexé).
Les *titres consolidés* (`metadata/documents-metadata-v1.json`) entrent donc désormais dans
`signature()` via `titles_digest()` : corriger un titre renomme l'index lexical, comme
ajouter un document le renomme depuis la veille.

Cela ne suffit pourtant pas à fermer le trou, et c'est la seconde leçon : la signature dit
*quels titres*, elle ne dit pas *quelle source de titres*. Choisir d'indexer le titre de
l'export plutôt que le titre consolidé est une décision de **code**, pas de fichier — deux
index de contenus différents porteraient le même nom. C'est le manifeste qui le dit
(`title_source`), et `quant_rag._bm25_matches_collection()` qui le vérifie.
"""
from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path

HERE = Path(__file__).resolve().parent
DUPLICATES = HERE / "metadata" / "duplicates-v1.json"
TABLES = HERE / "tables" / "tables-markdown-v1.json"
TITLES = HERE / "titles" / "titles-clean-v1.json"
TITLE_VECTORS = HERE / "titles" / ".cache" / "vectors-clean-titles-v1.npz"
#: Registre d'imports : quels documents composent le corpus (rag/ingestion/registry.py).
#: Il entre dans ``signature()`` — sans lui, ajouter un document ne changeait rien au nom
#: de l'index BM25, qui était donc rechargé incomplet et sans un bruit.
REGISTRY = HERE / "ingestion" / "registry-v1.json"
#: Métadonnées bibliographiques consolidées. Elles entrent dans ``signature()`` depuis le
#: 5 septembre 2026, pour la même raison que le registre y était entré la veille : le titre
#: consolidé est **indexé** par BM25 (``retrieval.lexical.lexical_text`` met le titre du
#: document en tête du texte lexical), donc le corriger change l'index sans changer son nom.
METADATA = HERE / "metadata" / "documents-metadata-v1.json"

#: Étiquette lisible de l'état du corpus. À faire évoluer quand un overlay est
#: ajouté ou remplacé (la signature, elle, change toute seule).
#:
#: ``dedup-tables-registry-v1`` (4 septembre 2026) : **le contenu du corpus n'a pas
#: changé** — toujours 256 documents, 18 636 chunks, 3 864 textes convertis. Ce qui a
#: changé, c'est ce que la signature couvre : elle inclut désormais l'ensemble des
#: documents, via le registre. L'index BM25 bâti sous l'ancien nom est identique record
#: par record à celui du nouveau ; seul le nom dit maintenant la vérité complète.
#: ``dedup-tables-registry-titles-v1`` (5 septembre 2026) : **le contenu du corpus n'a
#: toujours pas changé** — 319 documents actifs, 22 190 chunks. Ce qui a changé, c'est,
#: une seconde fois, ce que la signature couvre : les titres consolidés y entrent. Le
#: raisonnement qui les en excluait était écrit en tête de ce fichier et il était faux
#: sur un point — « BM25 lit le texte, qui n'a pas changé ». BM25 lit le texte **et le
#: titre du document**. Une correction de titre changeait donc l'index lexical sans
#: changer son nom : c'est exactement le défaut que ce nommage existe pour empêcher,
#: consigné au §5 sexies du TODO le 5 septembre et refermé ici.
LABEL = "dedup-tables-registry-titles-v1"


@lru_cache(maxsize=1)
def removed_documents() -> dict[str, dict]:
    """document_id -> entrée de suppression (raison, document conservé)."""
    if not DUPLICATES.exists():
        return {}
    data = json.loads(DUPLICATES.read_text(encoding="utf-8"))
    return {entry["document_id"]: entry for entry in data.get("remove", [])}


@lru_cache(maxsize=1)
def text_overrides() -> dict[str, str]:
    """chunk_id -> texte de remplacement (tableaux convertis)."""
    if not TABLES.exists():
        return {}
    return json.loads(TABLES.read_text(encoding="utf-8"))["chunks"]


#: Empreintes déjà calculées, indexées sur (chemin, mtime, taille). Ce n'est pas une
#: mémorisation au sens où l'était le ``lru_cache`` retiré : la clé change dès que le
#: fichier change, donc un écrivain voit toujours sa propre écriture.
_DIGESTS: dict[str, tuple[tuple[int, int], str]] = {}


def _content_digest(path: Path) -> str:
    """Empreinte du **contenu** JSON, pas des octets.

    Réécrire un overlay avec une autre indentation ne change pas l'état du corpus et ne
    doit donc pas invalider l'index BM25 ni les caches du banc. C'est ce qui a fait échouer
    la première annulation d'import : le contenu était restauré à l'identique, la
    sérialisation non, et la signature ne revenait pas à sa valeur d'avant.
    """
    if not path.exists():
        return "absent"
    stat = path.stat()
    fingerprint = (stat.st_mtime_ns, stat.st_size)
    cached = _DIGESTS.get(str(path))
    if cached and cached[0] == fingerprint:
        return cached[1]
    payload = json.dumps(json.loads(path.read_text(encoding="utf-8")),
                         sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    digest = hashlib.sha256(payload.encode()).hexdigest()
    _DIGESTS[str(path)] = (fingerprint, digest)          # une entrée par fichier
    return digest


def registry_digest() -> str:
    """Empreinte de l'ensemble des documents, insensible aux champs de traçabilité.

    Ne portent sur la signature que les quatre champs qui *définissent* le corpus :
    identité du PDF, identifiant de jointure, statut, et somme du ``chunks.jsonl``.
    Reconstruire le registre sans rien changer au corpus laisse donc la signature
    intacte — un horodatage ne doit jamais invalider un index.

    **Pas de mise en cache, volontairement** : un processus qui modifie le registre ou un
    overlay doit voir la nouvelle valeur immédiatement. 2,4 ms par appel, contre le risque
    de nommer un index avec une signature périmée — le dépôt a déjà payé deux fois pour un
    fichier juste sous un nom qui mentait.
    """
    if not REGISTRY.exists():
        return "absent"
    documents = json.loads(REGISTRY.read_text(encoding="utf-8")).get("documents", [])
    rows = sorted([entry.get("sha256"), entry.get("document_id"), entry.get("status"),
                   (entry.get("files") or {}).get("chunks.jsonl")] for entry in documents)
    return hashlib.sha256(json.dumps(rows, separators=(",", ":")).encode()).hexdigest()[:16]


def signature() -> str:
    """Signature de l'état du corpus : les overlays *et* l'ensemble des documents.

    Elle ne couvrait que les deux fichiers d'overlay : ajouter des documents ne la
    changeait pas, l'index BM25 gardait son nom, et ``quant_rag.bm25()`` le rechargeait
    incomplet — le mécanisme bâti pour empêcher exactement cela n'y protégeait pas.

    Relue à chaque appel : voir ``registry_digest``. Un écrivain qui modifie un overlay
    *et* lit ``text_overrides()`` ou ``removed_documents()`` dans la foulée doit appeler
    ``invalidate()`` — ces deux-là sont mémorisés, eux.
    """
    digest = hashlib.sha256()
    for path in (DUPLICATES, TABLES):
        digest.update(path.name.encode())
        digest.update(_content_digest(path).encode())
    digest.update(b"registry")
    digest.update(registry_digest().encode())
    digest.update(b"titles")
    digest.update(titles_digest().encode())
    return digest.hexdigest()[:10]


def titles_digest() -> str:
    """Empreinte des **titres consolidés** des documents actifs.

    Ils entrent dans la signature parce qu'ils entrent dans l'index : ``lexical_text`` met
    le titre du document en tête du texte indexé par BM25. Corriger un titre change donc
    l'index lexical — le 5 septembre 2026, trois titres corrigés l'ont changé sans changer
    son nom, et seul le fait qu'il ait été reconstruit dans la foulée a évité qu'un index
    périmé soit servi sous un nom correct.

    Ne haché que ``document_id`` et ``title``, triés, et seulement pour les documents
    **actifs** : ni la mise en forme du fichier, ni un champ sans effet sur l'index, ni un
    document retiré ne doivent renommer l'index. Relue à chaque appel, comme la signature.
    """
    if not METADATA.exists():
        return "absent"
    retires = removed_documents()
    data = json.loads(METADATA.read_text(encoding="utf-8"))
    paires = sorted((row["document_id"], row.get("title") or "")
                    for row in data.get("documents", []) if row["document_id"] not in retires)
    digest = hashlib.sha256()
    for document_id, title in paires:
        digest.update(document_id.encode())
        digest.update(b"\x00")
        digest.update(title.encode())
        digest.update(b"\x01")
    return digest.hexdigest()[:12]


def invalidate() -> None:
    """Oublie les fichiers d'overlay déjà lus.

    À appeler par tout processus qui **écrit** un overlay (``convert_tables.py``,
    ``apply_duplicates.py``, ``registry.py --build``, le futur ``apply_delivery.py``) avant
    de relire le corpus. ``signature()`` et ``registry_digest()`` n'en ont pas besoin — elles
    relisent toujours le disque — mais ``removed_documents()`` et ``text_overrides()`` sont
    mémorisés pour la durée du processus.
    """
    removed_documents.cache_clear()
    text_overrides.cache_clear()
    _DIGESTS.clear()


def bm25_name() -> str:
    """Nom de base de l'index BM25 de cet état du corpus : ``bm25-<étiquette>-<signature>``."""
    return f"bm25-{LABEL}-{signature()}"


def _nom_relatif(path: Path) -> str:
    """Le chemin d'un overlay, relatif à la racine du dépôt — ou absolu s'il est ailleurs.

    ``str(path.relative_to(HERE.parent))`` **lève** ``ValueError`` dès qu'un overlay sort de
    l'arbre du module. Ce n'est pas une hypothèse d'école : ``describe()`` est appelée par
    ``quant_rag.rebuild_bm25``, donc à l'**étape 10** d'un import — après l'upsert et après
    la promotion, c'est-à-dire pile dans la fenêtre que le journal existe pour décrire. Un
    overlay déplacé, ou un arbre de test, faisait donc échouer l'import au pire moment, et
    pour une raison qui n'avait rien à voir avec l'import (§11.3 du rapport
    ``reprise-ingestion``, 9 septembre 2026).

    Un nom de fichier n'est pas une garde. Il sert à relire un manifeste dans six mois : un
    chemin absolu y est moins joli qu'un chemin relatif, et infiniment plus utile qu'une
    exception.
    """
    try:
        return str(path.relative_to(HERE.parent))
    except ValueError:
        return str(path)


def describe() -> dict:
    """Ce qui définit l'état du corpus, pour les manifestes et les fichiers de résultats."""
    return {
        "label": LABEL,
        "signature": signature(),
        "overlays": [{"file": _nom_relatif(path), "present": path.exists(),
                      "sha256": hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None}
                     for path in (DUPLICATES, TABLES)],
        "registry": {"file": _nom_relatif(REGISTRY), "present": REGISTRY.exists(),
                     "digest": registry_digest()},
        "titles": {"file": _nom_relatif(METADATA), "present": METADATA.exists(),
                   "digest": titles_digest()},
        "removed_documents": len(removed_documents()),
        "text_overrides": len(text_overrides()),
        "embedding_titles": {"file": _nom_relatif(TITLES), "present": TITLES.exists(),
                             "sha256": hashlib.sha256(TITLES.read_bytes()).hexdigest() if TITLES.exists() else None,
                             "vectors_present": TITLE_VECTORS.exists()},
    }


def apply(document_id: str, chunk_id: str, text: str) -> str | None:
    """Texte à indexer pour ce chunk, ou ``None`` si son document est retiré."""
    if document_id in removed_documents():
        return None
    return text_overrides().get(chunk_id, text)
