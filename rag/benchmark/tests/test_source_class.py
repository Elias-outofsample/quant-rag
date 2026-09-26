"""La règle de provenance — et surtout l'ORDRE de ses tests, qui porte le sens.

Une règle de classement se teste mal par ses cas nominaux : ils passent par construction.
Ce qui décide, ce sont les cas où deux signaux se contredisent, et c'est exactement là que
l'ordre des tests fait le travail. Chacun de ceux-ci correspond à un document **réel** du
corpus, nommé dans son docstring : un test taillé sur un exemple inventé prouverait la
règle contre elle-même.

Le corpus sert aussi de témoin : la règle est confrontée aux 150 lignes « provenance du
PDF » que les rapports de lot déclarent **à la main**, et l'accord doit rester total.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

MACOS = Path(__file__).resolve().parents[2]
ROOT = MACOS.parents[0]
sys.path.insert(0, str(MACOS / "metadata"))

import source_class as sc  # noqa: E402

METADATA = MACOS / "metadata" / "documents-metadata-v1.json"


# ------------------------------------------------------------------- l'ordre, cas par cas

def test_un_preprint_de_204_pages_reste_un_preprint():
    """`2010.06467.pdf`, 204 pages. Si le seuil de pages passait avant arXiv, ce serait un livre."""
    assert sc.classer("2010.06467.pdf", 204, None) == "arxiv"


def test_un_preprint_de_165_pages_aussi():
    """`1610.08104.pdf`, 165 pages — le second des deux que l'ordre sauve."""
    assert sc.classer("1610.08104.pdf", 165, None) == "arxiv"


def test_un_extrait_de_livre_de_47_pages_reste_un_livre():
    """`preview-9781108639064_A34411323.pdf` : un *preview* Cambridge de 47 pages.

    Loin sous le seuil : seul l'ISBN-13 le rattrape, et il doit donc être testé avant.
    """
    assert sc.classer("preview-9781108639064_A34411323.pdf", 47, None) == "livre"


def test_un_livre_nomme_titre_auteur_annee_editeur_est_un_livre():
    """Le séparateur « -- » du nommage Titre -- Auteur -- Année -- Éditeur, sur un extrait sous le seuil."""
    nom = "Analysis of Financial Time Series -- Ruey S_ Tsay -- 2005 -- Wiley.pdf"
    assert sc.classer(nom, 21, None) == "livre"


def test_le_tampon_arxiv_decide_meme_quand_le_nom_ne_dit_rien():
    """137 documents du corpus sont dans ce cas : leur nom n'évoque pas arXiv.

    C'est le signal le plus fort — il est lu dans le TEXTE de la première page — et sans
    lui la règle classerait ces documents `autre`.
    """
    nom = "2026-08-03_Zaugg_VolParametrizations_RandomCoeff.pdf"
    assert sc.classer(nom, 30, None) == "autre"
    assert sc.classer(nom, 30, {"arxiv_id": "2508.01234"}) == "arxiv"


def test_ssrn_passe_avant_tout():
    """Aucun `ssrn-*.pdf` du corpus ne porte de tampon arXiv — mais l'ordre doit tenir si
    l'un venait à en porter un : la source de dépôt prime sur la trace de préprint."""
    assert sc.classer("ssrn-4906546.pdf", 30, {"arxiv_id": "2508.01234"}) == "ssrn"


def test_l_ancienne_forme_arxiv_est_reconnue():
    """`q-fin.TR/0701001` — la forme d'avant avril 2007, encore présente au corpus."""
    assert sc.classer("cond-mat_0412429.pdf", 12, None) == "autre"
    assert sc.classer("q-fin.TR/0701001v1.pdf", 12, None) == "arxiv"


def test_ce_qui_ne_porte_aucun_signal_tombe_dans_autre():
    """NBER, Econometrica, notes de banque : la règle ne les subdivise pas, et le dit."""
    for nom in ("w19325.pdf", "EngleGranger1987.pdf", "static_options_replication.pdf"):
        assert sc.classer(nom, 30, None) == "autre"


def test_un_numero_a_quatre_chiffres_dans_un_nom_ne_suffit_pas():
    """La regex arXiv exige `NNNN.NNNNN` : une année seule ne doit pas déclencher."""
    assert sc.classer("Rapport-2024-final.pdf", 20, None) == "autre"
    assert sc.classer("EngleGranger1987.pdf", 20, None) == "autre"


# ------------------------------------------------------- la règle est PURE et rejouable

def test_deux_appels_rendent_la_meme_chose():
    lignes = json.loads(METADATA.read_text(encoding="utf-8"))["documents"]
    premier = {r["document_id"]: sc.classer_ligne(r) for r in lignes}
    second = {r["document_id"]: sc.classer_ligne(r) for r in reversed(lignes)}
    assert premier == second, "le classement dépend de l'ordre d'itération"


def test_tous_les_documents_du_corpus_sont_classes():
    lignes = json.loads(METADATA.read_text(encoding="utf-8"))["documents"]
    classes = {sc.classer_ligne(r) for r in lignes}
    assert classes <= set(sc.CLASSES)
    assert all(sc.classer_ligne(r) for r in lignes), "un document sans classe"


def test_le_champ_est_inscrit_pour_chaque_document():
    lignes = json.loads(METADATA.read_text(encoding="utf-8"))["documents"]
    manquants = [r["document_id"] for r in lignes if not r.get("source_class")]
    assert manquants == [], ("relance .venv/bin/python rag/metadata/source_class.py "
                             f"--ecrire — {len(manquants)} document(s) sans source_class")
    assert all("licence" in r for r in lignes), "le champ licence réservé n'est pas posé"


def test_le_champ_inscrit_est_bien_celui_que_la_regle_derive():
    """Le champ est **dérivé**, pas saisi : s'il diverge de la règle, c'est qu'on l'a édité."""
    lignes = json.loads(METADATA.read_text(encoding="utf-8"))["documents"]
    divergents = [(r["document_id"], r.get("source_class"), sc.classer_ligne(r))
                  for r in lignes if r.get("source_class") != sc.classer_ligne(r)]
    assert divergents == [], f"champ édité à la main : {divergents[:5]}"


def test_le_registre_porte_la_meme_classe_que_les_metadonnees():
    """Le registre la DÉRIVE lui aussi. Deux dérivations d'une même règle doivent coïncider ;
    si elles divergent, c'est que l'une des deux a été recopiée quelque part."""
    registre = json.loads((MACOS / "ingestion" / "registry-v1.json").read_text(encoding="utf-8"))
    lignes = {r["document_id"]: r.get("source_class")
              for r in json.loads(METADATA.read_text(encoding="utf-8"))["documents"]}
    divergents = [(d["document_id"], d.get("source_class"), lignes.get(d["document_id"]))
                  for d in registre["documents"]
                  if d["document_id"] in lignes and d.get("source_class") != lignes[d["document_id"]]]
    assert divergents == [], f"registre et métadonnées divergent : {divergents[:5]}"


# -------------------------------------------- le témoin : ce que les rapports déclarent

def test_la_regle_s_accorde_avec_les_provenances_declarees_a_la_main():
    """Les rapports de lot déclarent « provenance du PDF : arXiv:… » document par document.

    C'est une vérité terrain écrite par un humain, indépendante de la règle. L'accord doit
    être **total** : une seule divergence signifierait que la règle et l'auteur des rapports
    ne parlent pas du même corpus.
    """
    meta = {r["filename"]: r
            for r in json.loads(METADATA.read_text(encoding="utf-8"))["documents"]}
    couples, fichier = [], None
    for rapport in sorted((MACOS / "ingestion" / "source-b").glob("RAPPORT-BATCH-*.md")):
        for ligne in rapport.read_text(encoding="utf-8").splitlines():
            depart = re.match(r"\|\s*fichier\s*\|\s*`([^`]+\.pdf)`", ligne)
            if depart:
                fichier = depart.group(1)
                continue
            provenance = re.match(r"\|\s*provenance du PDF\s*\|\s*(.+)", ligne)
            if provenance and fichier:
                couples.append((fichier, provenance.group(1)))
                fichier = None
    assert len(couples) >= 100, f"trop peu de provenances déclarées lues ({len(couples)})"

    divergences = []
    verifies = 0
    for nom, provenance in couples:
        attendu = ("arxiv" if re.search(r"arxiv:", provenance, re.I) else
                   "ssrn" if re.search(r"ssrn", provenance, re.I) else None)
        if attendu is None or nom not in meta:
            continue
        verifies += 1
        obtenu = sc.classer_ligne(meta[nom])
        if obtenu != attendu:
            divergences.append((nom, attendu, obtenu))
    assert verifies >= 100, f"trop peu de documents confrontés ({verifies})"
    assert divergences == [], f"{len(divergences)} divergence(s) : {divergences[:5]}"


# ------------------------------------------------- le champ n'a pas le droit de renommer

def test_la_classe_n_entre_dans_aucune_empreinte_de_signature():
    """Sinon, l'ajouter aurait renommé l'index BM25 et levé le gel du corpus.

    ``registry_digest`` ne hache que sha256, document_id, status et le sha du
    ``chunks.jsonl`` ; ``titles_digest`` que document_id et title. Le test lit les deux
    sources plutôt que de faire confiance à un commentaire.
    """
    source = (MACOS / "corpus_overlay.py").read_text(encoding="utf-8")
    debut = source.index("def registry_digest")
    fin = source.index("def signature")
    assert "source_class" not in source[debut:fin]
    debut = source.index("def titles_digest")
    fin = source.index("def invalidate")
    assert "source_class" not in source[debut:fin]
