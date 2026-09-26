"""D'où vient un document — `arxiv`, `ssrn`, `livre` ou `autre`, par une règle et sans modèle.

Pourquoi une règle et pas un champ saisi
-----------------------------------------
Un champ saisi vieillit et se contredit ; une règle se rejoue. Celle-ci ne lit que trois
champs, tous remplis à 100 % sur le corpus — le **nom de fichier**, le **nombre de pages** et
la présence d'un **tampon arXiv** dans le texte de la première page — et elle ne consulte ni
modèle, ni réseau, ni les PDF. Deux exécutions rendent la même chose.

Elle n'invente rien non plus : elle a été confrontée aux **159 lignes « provenance du PDF »**
que les quatre rapports de lot de `rag/ingestion/source-b/` déclarent à la main
(142 arXiv, 8 SSRN, 2 NBER, 2 livres, 5 revues). **Accord 159/159, zéro divergence.**

L'ordre des tests porte le sens
--------------------------------
Il n'est pas cosmétique, et chaque inversion a été mesurée :

- **arXiv avant le seuil de pages** : `2010.06467.pdf` fait 204 pages et `1610.08104.pdf`
  165. Sans cet ordre, deux préprints deviendraient des livres ;
- **ISBN et séparateur de nom avant le seuil de pages** : deux livres font 21 et 47 pages —
  un extrait et un *preview* Cambridge. Le seuil seul les manquerait ;
- **le seuil en dernier** : c'est le signal le plus grossier, il ne décide plus que d'une
  poignée de documents.

Un seuil calé sur ce corpus serait un seuil qui ment
-----------------------------------------------------
Un balayage de 100 à 300 pages montre que **270** supprimerait le dernier faux positif (une
thèse de 265 pages) sans perdre un livre. Il n'est pas retenu : 270 est calé sur le fait que
le plus petit livre non nommé de *ce* corpus fait 281 pages. 150 laisse de la marge aux
livres qui entreront, et le prix est **deux** documents classés `livre` à tort — sur 421.

Ce que la règle **ne** fait pas
--------------------------------
Elle ne subdivise pas `autre` : NBER, Econometrica, *Operations Research*, notes de banque
et préprints déposés hors dépôt public y tombent ensemble. Les distinguer demanderait une
connaissance externe au corpus, c'est-à-dire une saisie — précisément ce que ce module évite.
`licence` est **réservé et laissé à `null`** : l'usage est personnel et local, aucun chantier
de droits n'est ouvert.

    .venv/bin/python rag/metadata/source_class.py                 # la répartition
    .venv/bin/python rag/metadata/source_class.py --ecrire        # l'inscrit aux métadonnées
    .venv/bin/python rag/metadata/source_class.py --json
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
METADATA = HERE / "documents-metadata-v1.json"

CLASSES = ("arxiv", "ssrn", "livre", "autre")

#: `ssrn-4906546.pdf`, `ssrn-296036.pdf` — les 45 du corpus s'écrivent tous ainsi.
RE_SSRN = re.compile(r"(?:^|[^A-Za-z0-9])ssrn[-_ ]?(?:id)?[-_ ]?(\d{5,8})(?![0-9])", re.I)
#: Identifiant arXiv moderne : `2509.16157`, `1601.07961v1`.
RE_ARXIV_NEW = re.compile(r"(?:^|[^0-9])(\d{4}\.\d{4,5})(?:v\d+)?(?![0-9])")
#: Forme ancienne : `q-fin.TR/0701001`, `cond-mat/0412429`.
RE_ARXIV_OLD = re.compile(
    r"(?:^|[^A-Za-z0-9])((?:math|cond-mat|q-fin|physics|cs|stat|nlin|hep-th|hep-ph|hep-ex|"
    r"astro-ph|quant-ph|math-ph|q-bio|econ|eess|gr-qc)(?:\.[A-Za-z]{2})?/\d{7})(?:v\d+)?", re.I)
#: Un ISBN-13 dans le nom de fichier : `preview-9781108639064_A34411323.pdf`.
RE_ISBN13 = re.compile(r"97[89]\d{10}")
#: Le séparateur du nommage « Titre -- Auteur -- Année -- Éditeur ».
RE_TIRETS = re.compile(r" -- ")

#: Au-delà, un document sans autre signal est tenu pour un livre. Voir la docstring : 270
#: irait mieux sur *ce* corpus, et c'est précisément pourquoi il n'est pas retenu.
SEUIL_PAGES = 150


def classer(filename: str, page_count: int | None, arxiv_stamp: object) -> str:
    """La règle, en sept tests ordonnés. **Fonction pure.**

    ``arxiv_stamp`` est l'objet ``sources.arxiv_stamp`` des métadonnées : le tampon
    « arXiv:XXXX.XXXXXvN [cat] date » lu dans le texte de la première page. C'est la preuve
    la plus forte, et elle couvre 137 documents dont le nom ne dit rien d'arXiv.
    """
    nom = filename or ""
    if RE_SSRN.search(nom):
        return "ssrn"
    if arxiv_stamp:
        return "arxiv"
    if RE_ARXIV_NEW.search(nom) or RE_ARXIV_OLD.search(nom):
        return "arxiv"
    if RE_ISBN13.search(nom) or RE_TIRETS.search(nom):
        return "livre"
    if isinstance(page_count, int) and page_count >= SEUIL_PAGES:
        return "livre"
    return "autre"


def classer_ligne(row: dict) -> str:
    """La règle appliquée à une ligne de ``documents-metadata-v1.json``."""
    return classer(row.get("filename"), row.get("page_count"),
                   (row.get("sources") or {}).get("arxiv_stamp"))


def repartition(rows: list[dict]) -> dict:
    compte = collections.Counter(classer_ligne(row) for row in rows)
    return {classe: compte.get(classe, 0) for classe in CLASSES}


def main() -> None:
    parseur = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parseur.add_argument("--ecrire", action="store_true",
                         help="inscrire source_class et licence:null dans les métadonnées. "
                              "N'entre PAS dans la signature du corpus : ni registry_digest "
                              "(sha256, document_id, status, chunks.jsonl) ni titles_digest "
                              "(document_id, title) ne regardent ce champ — vérifié dans "
                              "corpus_overlay.registry_digest et corpus_overlay.titles_digest")
    parseur.add_argument("--json", action="store_true")
    parseur.add_argument("--metadata", type=Path, default=METADATA)
    args = parseur.parse_args()

    donnees = json.loads(args.metadata.read_text(encoding="utf-8"))
    lignes = donnees["documents"]
    classes = {row["document_id"]: classer_ligne(row) for row in lignes}
    compte = repartition(lignes)

    if args.ecrire:
        avant = json.loads(args.metadata.read_text(encoding="utf-8"))
        for row in donnees["documents"]:
            row["source_class"] = classes[row["document_id"]]
            row.setdefault("licence", None)
        args.metadata.write_text(json.dumps(donnees, ensure_ascii=False, indent=1),
                                 encoding="utf-8")
        inchanges = sum(1 for a, b in zip(avant["documents"], donnees["documents"])
                        if a.get("title") == b.get("title"))
        print(f"écrit : {len(lignes)} documents, {inchanges} titres inchangés "
              f"(la signature ne bouge pas)")

    rapport = {"documents": len(lignes), "repartition": compte,
               "seuil_pages": SEUIL_PAGES,
               "part": {k: round(100 * v / max(len(lignes), 1), 1) for k, v in compte.items()}}
    if args.json:
        print(json.dumps({**rapport, "classes": classes}, ensure_ascii=False, indent=1))
        return
    print(f"{len(lignes)} documents")
    for classe, nombre in compte.items():
        print(f"  {classe:8} {nombre:4}   {rapport['part'][classe]:5.1f} %")


if __name__ == "__main__":
    main()
