"""Texte canonique d'un document, et provenance exacte d'un chunk — le contrat de données.

Pourquoi ce module existe
-------------------------
Aujourd'hui un chunk ne sait pas *où* il se trouve dans son document : il porte une page, un
chemin de section et une liste de ``block_ids``, mais aucun offset. Trois conséquences, toutes
mesurées :

1. le ré-ancrage du gold doit retrouver un passage par **recouvrement de 5-grammes**
   (``rag/benchmark/gold_ancrage.py``), ce qui se disperse quand les chunks rétrécissent —
   c'est la borne des 1 200 caractères de ``docs/STRATEGIE.md`` §2 ;
2. une citation ne peut désigner qu'un chunk entier, jamais le passage exact qui soutient une
   phrase (``docs/STRATEGIE.md`` §4.2) ;
3. ``human-v1`` ne peut être gelé que sur du texte, donc de façon approximative.

Un **offset** règle les trois d'un coup, à une condition : qu'il soit relatif à un texte
**stable d'un découpage à l'autre**. Le PDF ne convient pas (les positions de caractères y
sont perdues dès le parse). Les **blocs** conviennent : ``blocks.jsonl`` est versionné pour les
421 documents et son ``sha256`` est au registre.

Le texte canonique
------------------
``texte_canonique`` concatène le texte de **tous** les blocs, dans l'ordre du fichier, séparés
par ``\\n\\n``. Ce texte n'est jamais servi ni indexé : c'est un **système de coordonnées**.
Son ``sha256`` est ce qui rend un offset opposable — un offset dont le texte de référence a
changé ne désigne plus rien, et la somme le dit.

Deux granularités, et la seconde est déclarée
---------------------------------------------
``exacte``  le texte du chunk est la concaténation exacte de ses intervalles. **92,24 %**
            des 46 945 chunks (``audit_rechunk.py --offsets``).
``bloc``    le chunk vient de sous-blocs **synthétiques** que le chunker fabrique : un tableau
            découpé sur ``<tr>`` (le fragment est ``préfixe + lignes``, pas une sous-chaîne) ou
            un bloc de texte de plus de 1 200 jetons redécoupé par ``' '.join(words[...])``,
            qui **normalise les espaces**. L'offset désigne alors le **bloc parent**, plus
            grossier. **7,76 %** des chunks, et le parent est retrouvé dans **100 %** des cas.

Un intervalle unique ne suffit pas — 72,9 % seulement — parce qu'un chunk **saute** des blocs :
en-têtes, pieds de page et numéros de page sont écartés par le chunker. La provenance est donc
une **liste** d'intervalles, et c'est elle qui est exacte sans exception.

    .venv/bin/python src/parsing/document_text.py --controle
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

#: Séparateur entre deux blocs dans le texte canonique. Il ne change jamais : le changer
#: invaliderait tous les offsets déjà écrits, et c'est pour cela que le ``sha256`` du texte
#: canonique accompagne chaque provenance.
SEPARATEUR = "\n\n"

#: Suffixe des sous-blocs synthétiques fabriqués par ``canonical_chunker.make_chunks``.
_PART = re.compile(r"^(?P<parent>.+)-part\d+$")

GRANULARITE_EXACTE = "exacte"
GRANULARITE_BLOC = "bloc"


@dataclass(frozen=True)
class Intervalle:
    """Un segment du texte canonique : ``[debut, fin)``, demi-ouvert comme une tranche Python."""

    block_id: str
    debut: int
    fin: int

    def as_list(self) -> list:
        return [self.block_id, self.debut, self.fin]


@dataclass(frozen=True)
class Provenance:
    """Où vit un chunk dans le texte canonique de son document."""

    chunk_id: str
    document_id: str
    doc_text_sha256: str
    granularite: str
    intervalles: list[Intervalle] = field(default_factory=list)
    #: Blocs cités par le chunk et introuvables même par leur parent. Doit rester vide.
    blocs_inconnus: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"chunk_id": self.chunk_id, "document_id": self.document_id,
                "doc_text_sha256": self.doc_text_sha256, "granularite": self.granularite,
                "intervalles": [i.as_list() for i in self.intervalles],
                "blocs_inconnus": list(self.blocs_inconnus)}


def texte_canonique(blocs) -> tuple[str, dict[str, tuple[int, int]]]:
    """Le texte canonique d'un document et l'intervalle de chacun de ses blocs.

    ``blocs`` : itérable de ``CanonicalBlock`` **ou** de dictionnaires, dans l'ordre du
    fichier — c'est-à-dire l'ordre de lecture posé par ``mineru_adapter`` (page, puis ``bbox``,
    puis ``block_id``). Cet ordre est la seule chose qui définit les coordonnées ; le changer
    invalide les offsets, et le ``sha256`` retourné est là pour que cela se voie.
    """
    morceaux: list[str] = []
    spans: dict[str, tuple[int, int]] = {}
    position = 0
    for bloc in blocs:
        bid = bloc["block_id"] if isinstance(bloc, dict) else bloc.block_id
        texte = (bloc["text"] if isinstance(bloc, dict) else bloc.text) or ""
        spans[bid] = (position, position + len(texte))
        morceaux.append(texte)
        position += len(texte) + len(SEPARATEUR)
    return SEPARATEUR.join(morceaux), spans


def sha256_texte(texte: str) -> str:
    """Empreinte du texte canonique. C'est elle qui rend un offset opposable."""
    return hashlib.sha256(texte.encode("utf-8")).hexdigest()


def bloc_parent(block_id: str) -> str | None:
    """Le bloc dont un sous-bloc synthétique ``…-partN`` est issu, ou ``None``."""
    m = _PART.match(block_id)
    return m.group("parent") if m else None


def _block_ids(chunk) -> list[str]:
    metadata = chunk.get("metadata") if isinstance(chunk, dict) else chunk.metadata
    return list((metadata or {}).get("block_ids") or [])


def _champ(chunk, nom: str):
    return chunk.get(nom) if isinstance(chunk, dict) else getattr(chunk, nom)


def provenance_du_chunk(chunk, spans: dict[str, tuple[int, int]], doc_sha: str) -> Provenance:
    """La provenance d'un chunk : ses intervalles, et la granularité qu'ils atteignent.

    Un chunk dont **tous** les blocs sont connus reçoit la granularité ``exacte`` ; il suffit
    qu'un seul soit synthétique pour que l'ensemble retombe au **bloc parent** — mélanger les
    deux dans une même liste donnerait une provenance dont on ne saurait plus dire ce qu'elle
    promet.
    """
    ids = _block_ids(chunk)
    granularite = GRANULARITE_EXACTE if all(b in spans for b in ids) else GRANULARITE_BLOC
    intervalles: list[Intervalle] = []
    inconnus: list[str] = []
    vus: set[str] = set()
    for bid in ids:
        cible = bid if bid in spans else bloc_parent(bid)
        if cible is None or cible not in spans:
            inconnus.append(bid)
            continue
        if granularite == GRANULARITE_BLOC and cible in vus:
            continue          # deux fragments d'un même tableau ne citent leur parent qu'une fois
        vus.add(cible)
        debut, fin = spans[cible]
        intervalles.append(Intervalle(cible, debut, fin))
    return Provenance(chunk_id=_champ(chunk, "chunk_id"), document_id=_champ(chunk, "document_id"),
                      doc_text_sha256=doc_sha, granularite=granularite,
                      intervalles=intervalles, blocs_inconnus=inconnus)


def reconstruire(texte: str, provenance: Provenance) -> str:
    """Le texte désigné par une provenance, tel que le chunker l'aurait assemblé."""
    return SEPARATEUR.join(x for x in (texte[i.debut:i.fin] for i in provenance.intervalles) if x)


def aller_retour(texte: str, chunk, provenance: Provenance) -> bool:
    """L'aller-retour, et il n'est exigible qu'en granularité ``exacte``.

    En granularité ``bloc``, le texte du chunk n'est **pas** une sous-chaîne du texte canonique
    — le chunker a réassemblé un tableau ou normalisé des espaces. On vérifie alors seulement
    que le parent existe, ce que ``blocs_inconnus`` porte déjà.
    """
    if provenance.granularite != GRANULARITE_EXACTE:
        return not provenance.blocs_inconnus
    return reconstruire(texte, provenance).strip() == (_champ(chunk, "text") or "").strip()


# ------------------------------------------------------------------ contrôle


def _controle() -> int:
    """Aller-retour sur tout le corpus. Sortie non nulle si un seul chunk échoue."""
    import collections
    import json
    import pathlib
    import sys

    racine = pathlib.Path(__file__).resolve().parents[2]
    ingested = racine / "data" / "processed" / "ingested"
    compte = collections.Counter()
    echecs: list[tuple[str, str]] = []

    for dossier in sorted(p for p in ingested.iterdir() if p.is_dir()):
        blocs = [json.loads(l) for l in (dossier / "blocks.jsonl").open(encoding="utf-8") if l.strip()]
        chunks = [json.loads(l) for l in (dossier / "chunks.jsonl").open(encoding="utf-8") if l.strip()]
        texte, spans = texte_canonique(blocs)
        doc_sha = sha256_texte(texte)
        compte["documents"] += 1
        for chunk in chunks:
            compte["chunks"] += 1
            prov = provenance_du_chunk(chunk, spans, doc_sha)
            compte[prov.granularite] += 1
            if prov.blocs_inconnus:
                compte["blocs_inconnus"] += 1
            if aller_retour(texte, chunk, prov):
                compte["aller_retour_ok"] += 1
            else:
                compte["ÉCHEC"] += 1
                if len(echecs) < 10:
                    echecs.append((dossier.name, chunk["chunk_id"]))

    n = compte["chunks"]
    print(f"documents                         {compte['documents']:>6}")
    print(f"chunks                            {n:>6}")
    print(f"  granularité exacte              {compte[GRANULARITE_EXACTE]:>6}  "
          f"({100.0 * compte[GRANULARITE_EXACTE] / n:5.2f} %)")
    print(f"  granularité bloc (parent)       {compte[GRANULARITE_BLOC]:>6}  "
          f"({100.0 * compte[GRANULARITE_BLOC] / n:5.2f} %)")
    print(f"  blocs introuvables              {compte['blocs_inconnus']:>6}")
    print(f"  ALLER-RETOUR OK                 {compte['aller_retour_ok']:>6}  "
          f"({100.0 * compte['aller_retour_ok'] / n:5.2f} %)")
    print(f"  ÉCHECS                          {compte['ÉCHEC']:>6}")
    for nom, cid in echecs:
        print(f"    {nom}  {cid}")
    return 1 if compte["ÉCHEC"] or compte["blocs_inconnus"] else 0


if __name__ == "__main__":
    import argparse
    import sys

    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--controle", action="store_true",
                   help="aller-retour sur les 421 documents ; sortie 1 si un chunk échoue")
    a = p.parse_args()
    if not a.controle:
        p.print_help()
        sys.exit(0)
    sys.exit(_controle())
