"""Vérifier qu'une citation se trouve vraiment dans le document qu'elle nomme.

Le serveur sait dire d'où vient un passage. Il ne savait pas dire si une phrase **attribuée** à
un document s'y trouve — c'est-à-dire qu'il ne pouvait pas contredire une citation inventée,
déformée ou recollée à partir de deux endroits. Ce module le fait, et il le fait sans LLM :
la vérification d'une citation par un modèle de langue serait exactement le genre de preuve
que ce dépôt refuse.

La méthode, en trois temps
--------------------------
1. **Normalisation déterministe** du texte cité et du texte canonique du document : NFKC (qui
   rend les ligatures ``ﬁ``/``ﬀ`` à leurs lettres), retrait de ``<sup>``/``<sub>``, unification
   des tirets et des guillemets, réduction des blancs, casse repliée. Chaque caractère normalisé
   garde l'index de son origine, si bien qu'une correspondance trouvée sur le texte normalisé
   se rend en offsets du **texte canonique**, ceux du payload.
2. **Recherche exacte** de la citation normalisée. C'est elle, et elle seule, qui répond
   ``trouve``.
3. **Recherche tolérante par fenêtre de mots** quand l'exacte échoue. Elle ne rend jamais
   ``trouve`` : elle dit **à quelle distance** se trouve le passage le plus proche, et où. Une
   citation dont un mot a été changé n'est pas « presque vraie », elle est fausse — mais savoir
   *quel* mot diffère est ce qui rend le verdict utile.

Le cas des tableaux
-------------------
5 914 passages sont servis dans une version Markdown reconstruite par l'overlay des tableaux :
leur texte servi n'est **pas** une sous-chaîne du texte canonique. Une citation qui en vient
échouerait au temps 2 alors qu'elle est authentique. Le module cherche donc aussi dans le
**texte servi** du document, et le dit : ``source`` vaut ``"texte canonique"`` ou
``"texte servi"``, jamais rien d'implicite.
"""
from __future__ import annotations

import difflib
import re
import sys
import unicodedata
from functools import lru_cache
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import ancrage  # noqa: E402

#: Substitutions faites **avant** le repli de casse et la réduction des blancs. Chacune remplace
#: un caractère par un autre de même longueur, ou par une chaîne dont la longueur est suivie :
#: la table de correspondance vers les offsets d'origine en dépend.
_UN_POUR_UN = {
    "‐": "-", "‑": "-", "‒": "-", "–": "-", "—": "-", "―": "-",
    "‘": "'", "’": "'", "‚": "'", "‛": "'",
    "“": '"', "”": '"', "„": '"', "‟": '"',
    "«": '"', "»": '"', "′": "'", "″": '"',
    " ": " ", " ": " ", " ": " ", " ": " ", " ": " ",
}

_BALISE = re.compile(r"</?su[pb]>", re.IGNORECASE)

#: Les seuls segments qui demandent un traitement : un blanc à réduire, une balise à retirer,
#: une plage non-ASCII à normaliser. Tout le reste est de l'ASCII ordinaire, copié en bloc.
_SEGMENT = re.compile(r"\s+|</?su[pb]>|[^\x00-\x7F]+", re.IGNORECASE)

#: Longueur du n-gramme de mots qui sert d'amorce à la recherche tolérante.
AMORCE = 4

#: Au-dessus de ce rapport de similarité, la recherche tolérante rapporte un « plus proche ».
#: En dessous, elle se tait : proposer un voisinage à 40 % de ressemblance serait du bruit.
SEUIL_VOISINAGE = 0.60


def normaliser(texte: str) -> tuple[str, list[int]]:
    """``(texte normalisé, index d'origine de chaque caractère)``.

    La table de correspondance est ce qui distingue ce module d'un simple ``in`` : sans elle,
    on saurait qu'une citation existe sans pouvoir dire **où**, et un offset invérifiable ne
    vaut pas mieux qu'une page approximative.
    """
    sortie: list[str] = []
    origine: list[int] = []
    precedent_blanc = False
    position = 0

    def ascii_ordinaire(debut: int, fin: int) -> None:
        """Un segment d'ASCII sans blanc ni balise : ``lower()`` préserve la longueur.

        C'est ce qui rend la fonction utilisable : traiter le document caractère par caractère
        coûtait une centaine de millisecondes par vérification, soit la moitié du budget de
        l'outil pour recopier du texte. Ici la boucle Python ne tourne qu'une fois par segment
        remarquable — un blanc, une balise, une plage non-ASCII — et l'ASCII ordinaire, qui est
        l'essentiel du corpus, passe par une seule opération de chaîne.
        """
        nonlocal precedent_blanc
        if fin <= debut:
            return
        sortie.append(texte[debut:fin].lower())
        origine.extend(range(debut, fin))
        precedent_blanc = False

    def blanc(i: int) -> None:
        nonlocal precedent_blanc
        if not precedent_blanc and sortie:
            sortie.append(" ")
            origine.append(i)
            precedent_blanc = True

    for segment in _SEGMENT.finditer(texte):
        ascii_ordinaire(position, segment.start())
        position = segment.end()
        contenu = segment.group(0)
        if contenu[0] == "<":
            continue                                       # <sup>, </sub> : retirés, sans trace
        if contenu[0].isspace():
            blanc(segment.start())
            continue
        # Plage non-ASCII : rare, traitée caractère par caractère parce que NFKC peut changer
        # la longueur (« ﬁ » rend deux lettres) et que la table d'origine doit le suivre.
        for decalage, brut in enumerate(contenu):
            caractere = _UN_POUR_UN.get(brut, brut)
            if caractere.isspace():
                blanc(segment.start() + decalage)
                continue
            precedent_blanc = False
            for morceau in unicodedata.normalize("NFKC", caractere).casefold():
                sortie.append(morceau)
                origine.append(segment.start() + decalage)
    ascii_ordinaire(position, len(texte))
    while sortie and sortie[-1] == " ":
        sortie.pop()
        origine.pop()
    return "".join(sortie), origine


@lru_cache(maxsize=64)
def _document_brut(document_id: str) -> str | None:
    """Le texte canonique, tel quel. Lecture de ``blocks.jsonl`` et concaténation, rien de plus."""
    return ancrage.texte_canonique_du_document(document_id)


@lru_cache(maxsize=64)
def _document_normalise(document_id: str) -> tuple[str, tuple[int, ...], str] | None:
    """Le texte canonique normalisé, avec sa table d'origine. Mémorisé, et **payé tard**.

    Normaliser un document entier coûte une centaine de millisecondes. On ne le fait que si la
    recherche sur le texte brut a échoué : une citation recopiée d'un passage servi de
    granularité exacte est, mot pour mot, une sous-chaîne du texte canonique, et elle n'a
    besoin d'aucune normalisation pour être retrouvée.
    """
    brut = _document_brut(document_id)
    if brut is None:
        return None
    normalise, origine = normaliser(brut)
    return normalise, tuple(origine), brut


def _mots(normalise: str) -> list[tuple[str, int]]:
    return [(m.group(0), m.start()) for m in re.finditer(r"\S+", normalise)]


def _voisinage(normalise: str, citation: str) -> dict | None:
    """Le passage le plus proche, quand la recherche exacte a échoué. Ne conclut jamais.

    On amorce sur les quatre premiers mots de la citation — il y a peu de positions où ils
    apparaissent — puis on compare mot à mot sur une fenêtre de la longueur de la citation.
    Balayer tout le document par ``SequenceMatcher`` serait quadratique et inutile.
    """
    mots_citation = [m for m, _ in _mots(citation)]
    if len(mots_citation) < 2:
        return None
    mots_document = _mots(normalise)
    textes = [m for m, _ in mots_document]
    amorce = min(AMORCE, len(mots_citation))
    debut_cible = mots_citation[:amorce]
    candidats = [i for i in range(len(textes) - amorce + 1) if textes[i:i + amorce] == debut_cible]
    if not candidats:
        # L'amorce elle-même diffère : on retombe sur le premier mot, plus fréquent mais borné.
        candidats = [i for i, m in enumerate(textes) if m == mots_citation[0]][:200]
    meilleur = None
    for i in candidats:
        fenetre = textes[i:i + len(mots_citation)]
        rapport = difflib.SequenceMatcher(None, mots_citation, fenetre).ratio()
        if meilleur is None or rapport > meilleur[0]:
            meilleur = (rapport, i, fenetre)
    if meilleur is None or meilleur[0] < SEUIL_VOISINAGE:
        return None
    rapport, i, fenetre = meilleur
    differences = [mot for mot in difflib.ndiff(mots_citation, fenetre)
                   if mot.startswith(("+ ", "- "))]
    return {"ressemblance": round(rapport, 3),
            "mots_differents": len(differences),
            "offset_normalise": mots_document[i][1],
            "extrait_trouve": " ".join(fenetre)[:300]}


def _sha_du_document(document_id: str) -> str | None:
    return (ancrage.charger().get("documents") or {}).get(document_id, {}).get("sha256")


@lru_cache(maxsize=64)
def _couvertures(document_id: str) -> tuple[tuple[int, int, str, str], ...]:
    """``(début, fin, chunk_id, granularité)`` pour chaque intervalle fusionné d'un document.

    Calculé une fois par document : refaire la fusion des intervalles à chaque vérification
    coûtait plus cher que la recherche elle-même.
    """
    entree = (ancrage.charger().get("documents") or {}).get(document_id)
    if not entree:
        return ()
    couvertures = []
    for chunk_id, provenance in entree["chunks"].items():
        for debut, fin in ancrage.fusionner(provenance["i"]):
            couvertures.append((debut, fin, chunk_id, provenance["g"]))
    return tuple(sorted(couvertures))


def _pages_et_chunk(document_id: str, debut: int, fin: int) -> dict:
    """Le passage servi qui couvre ces offsets, et la granularité de son ancrage."""
    for couverture_debut, couverture_fin, chunk_id, granularite in _couvertures(document_id):
        if couverture_debut <= debut and fin <= couverture_fin:
            return {"chunk_id": chunk_id, "granularite": granularite}
    return {}


def verify_citation(document_id: str, quote: str, chunk_id: str | None = None) -> dict:
    """La citation ``quote`` se trouve-t-elle dans ``document_id`` ?

    ``trouve`` n'est vrai que pour une correspondance **exacte après normalisation**. Une
    citation dont un mot diffère n'est pas « presque trouvée » : elle est fausse, et le champ
    ``plus_proche`` dit alors de combien et où.
    """
    quote = (quote or "").strip()
    if not quote:
        return {"trouve": False, "raison": "citation vide"}
    brut = _document_brut(document_id)
    if brut is None:
        return {"trouve": False, "raison": f"document inconnu ou sans texte canonique : {document_id}"}

    # Chemin rapide : la citation est déjà, mot pour mot, dans le texte canonique. C'est le cas
    # des 86,9 % de passages servis en granularité exacte, et il évite de normaliser un document
    # entier pour rien.
    position = brut.find(quote)
    if position != -1:
        return {
            "trouve": True, "source": "texte canonique", "methode": "exacte",
            "offsets": [position, position + len(quote)],
            "occurrences": brut.count(quote),
            "doc_text_sha256": _sha_du_document(document_id),
            "distance": 0,
            **_pages_et_chunk(document_id, position, position + len(quote)),
        }

    prepare = _document_normalise(document_id)
    if prepare is None:                                    # pragma: no cover - déjà écarté plus haut
        return {"trouve": False, "raison": f"document inconnu : {document_id}"}
    normalise, origine, brut = prepare
    citation, _ = normaliser(quote)
    if not citation:
        return {"trouve": False, "raison": "citation vide après normalisation"}

    position = normalise.find(citation)
    if position != -1:
        debut = origine[position]
        fin = origine[min(position + len(citation), len(origine)) - 1] + 1
        occurrences = normalise.count(citation)
        return {
            "trouve": True,
            "source": "texte canonique",
            "methode": "exacte après normalisation",
            "offsets": [debut, fin],
            "occurrences": occurrences,
            "doc_text_sha256": _sha_du_document(document_id),
            "distance": 0,
            **_pages_et_chunk(document_id, debut, fin),
        }

    servi = _chercher_dans_le_servi(document_id, citation, chunk_id)
    if servi:
        return servi
    return {"trouve": False, "source": None, "methode": "exacte après normalisation",
            "distance": None, "plus_proche": _voisinage(normalise, citation),
            "raison": "la citation ne figure ni dans le texte canonique ni dans le texte servi"}


def _chercher_dans_le_servi(document_id: str, citation: str, chunk_id: str | None) -> dict | None:
    """Le repli des tableaux : 5 914 passages sont servis dans une réécriture Markdown.

    Leur texte n'est pas une sous-chaîne du texte canonique, et une citation qui en vient est
    pourtant authentique. On ouvre alors Qdrant — le seul endroit où le texte **servi** existe.
    """
    try:
        from qdrant_client import models

        import quant_rag
    except ImportError:                                    # pragma: no cover - dépendance absente
        return None
    try:
        conditions = [models.FieldCondition(key="document_id", match=models.MatchValue(value=document_id))]
        if chunk_id:
            conditions.append(models.FieldCondition(key="chunk_id", match=models.MatchValue(value=chunk_id)))
        points, _ = quant_rag.client().scroll(
            collection_name=quant_rag.COLLECTION, limit=4096, with_payload=True,
            scroll_filter=models.Filter(must=conditions))
    except Exception:                                      # pragma: no cover - Qdrant indisponible
        return None
    for point in points:
        texte_servi, origine_servi = normaliser(point.payload.get("text") or "")
        position = texte_servi.find(citation)
        if position == -1:
            continue
        return {
            "trouve": True,
            "source": "texte servi",
            "methode": "exacte après normalisation, sur le texte servi (overlay)",
            "offsets": point.payload.get("ancrage_intervalles"),
            "offsets_dans_le_passage": [origine_servi[position],
                                        origine_servi[min(position + len(citation),
                                                          len(origine_servi)) - 1] + 1],
            "chunk_id": point.payload.get("chunk_id"),
            "granularite": point.payload.get("ancrage_granularite"),
            "doc_text_sha256": point.payload.get("doc_text_sha256"),
            "pages": point.payload.get("page_start"),
            "distance": 0,
        }
    return None


def verify_citations(demandes: list[dict]) -> list[dict]:
    """Vérification en lot. ``demandes`` : ``[{"document_id": …, "quote": …, "chunk_id": …}]``."""
    return [{**demande, **verify_citation(demande.get("document_id", ""), demande.get("quote", ""),
                                          demande.get("chunk_id"))}
            for demande in demandes]
