"""Le contrat de provenance : offsets exacts, repli au bloc parent, somme opposable."""
import hashlib
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.parsing.document_text import (  # noqa: E402
    GRANULARITE_BLOC, GRANULARITE_EXACTE, SEPARATEUR, aller_retour, bloc_parent,
    provenance_du_chunk, reconstruire, sha256_texte, texte_canonique)


def bloc(bid, texte):
    return {"block_id": bid, "text": texte}


def chunk(cid, block_ids, texte):
    return {"chunk_id": cid, "document_id": "doc-x", "text": texte,
            "metadata": {"block_ids": list(block_ids)}}


BLOCS = [bloc("b-0", "Titre"), bloc("b-1", "Prose une."), bloc("b-2", "12"),
         bloc("b-3", "Prose deux.")]


def test_texte_canonique_est_la_concatenation_ordonnee():
    texte, spans = texte_canonique(BLOCS)
    assert texte == SEPARATEUR.join(["Titre", "Prose une.", "12", "Prose deux."])
    for b in BLOCS:
        debut, fin = spans[b["block_id"]]
        assert texte[debut:fin] == b["text"]


def test_un_chunk_qui_saute_un_bloc_reste_exact():
    """Le chunker écarte les numéros de page : la provenance est une LISTE d'intervalles."""
    texte, spans = texte_canonique(BLOCS)
    sha = sha256_texte(texte)
    c = chunk("c-1", ["b-1", "b-3"], "Prose une.\n\nProse deux.")
    prov = provenance_du_chunk(c, spans, sha)
    assert prov.granularite == GRANULARITE_EXACTE
    assert [i.block_id for i in prov.intervalles] == ["b-1", "b-3"]
    assert reconstruire(texte, prov) == "Prose une.\n\nProse deux."
    assert aller_retour(texte, c, prov)


def test_un_intervalle_unique_ne_suffirait_pas():
    """La preuve du choix de conception : l'intervalle englobant contient le bloc sauté."""
    texte, spans = texte_canonique(BLOCS)
    debut = spans["b-1"][0]
    fin = spans["b-3"][1]
    assert "12" in texte[debut:fin]


def test_sous_bloc_synthetique_retombe_au_parent_et_le_declare():
    """Un tableau découpé sur <tr> n'est pas une sous-chaîne : granularité « bloc », déclarée."""
    blocs = BLOCS + [bloc("b-4", "<table><tr>a</tr><tr>b</tr></table>")]
    texte, spans = texte_canonique(blocs)
    sha = sha256_texte(texte)
    c = chunk("c-2", ["b-4-part0", "b-4-part1"], "Table: <tr>a</tr>")
    prov = provenance_du_chunk(c, spans, sha)
    assert prov.granularite == GRANULARITE_BLOC
    assert prov.blocs_inconnus == []
    # les deux fragments citent leur parent UNE fois, pas deux
    assert [i.block_id for i in prov.intervalles] == ["b-4"]
    # l'aller-retour exact n'est pas exigible, et la fonction le sait
    assert aller_retour(texte, c, prov)
    assert reconstruire(texte, prov) != c["text"]


def test_un_seul_bloc_synthetique_fait_retomber_tout_le_chunk():
    """Mélanger les granularités donnerait une provenance dont on ne sait plus ce qu'elle promet."""
    blocs = BLOCS + [bloc("b-4", "Table entière")]
    texte, spans = texte_canonique(blocs)
    prov = provenance_du_chunk(chunk("c-3", ["b-1", "b-4-part0"], "peu importe"),
                               spans, sha256_texte(texte))
    assert prov.granularite == GRANULARITE_BLOC


def test_bloc_introuvable_est_declare_jamais_ignore():
    texte, spans = texte_canonique(BLOCS)
    prov = provenance_du_chunk(chunk("c-4", ["b-1", "b-inexistant"], "Prose une."),
                               spans, sha256_texte(texte))
    assert prov.blocs_inconnus == ["b-inexistant"]
    assert not aller_retour(texte, chunk("c-4", ["b-1", "b-inexistant"], "x"), prov)


def test_la_somme_rend_les_offsets_opposables():
    """Changer l'ordre des blocs change le système de coordonnées, et la somme le dit."""
    texte_a, _ = texte_canonique(BLOCS)
    texte_b, _ = texte_canonique(list(reversed(BLOCS)))
    assert sha256_texte(texte_a) != sha256_texte(texte_b)
    assert sha256_texte(texte_a) == hashlib.sha256(texte_a.encode("utf-8")).hexdigest()


@pytest.mark.parametrize("bid,attendu", [
    ("block-abc-1-2-part0", "block-abc-1-2"),
    ("block-abc-1-2-part17", "block-abc-1-2"),
    ("block-abc-1-2", None),
    ("block-part-de-quelque-chose", None),
])
def test_bloc_parent(bid, attendu):
    assert bloc_parent(bid) == attendu
