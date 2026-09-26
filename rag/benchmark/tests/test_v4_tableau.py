"""La famille `table_cell` : ce que la sélection refuse, et ce que le scoreur appelle « juste ».

Ces tests portent sur des tableaux **réels** du corpus, désignés par leur ``chunk_id``. C'est
délibéré : une table Markdown écrite à la main pour un test est toujours propre, et la
sélection existe précisément pour survivre à celles qui ne le sont pas — en-têtes sur deux
niveaux, fragments sans lignes de données, valeurs répétées d'une colonne à l'autre.

Les trois cas réels qui structurent le fichier :

  ``chunk-0046a568f5c7bbda`` — TABLE 5.1, prix d'options SPX. Propre : une ligne d'en-tête,
      des libellés de ligne distincts (les strikes), des valeurs numériques. C'est le cas que
      la famille cherche.
  ``chunk-003deb25e50fc252`` — « Figure 203 », en-tête sur **deux niveaux** et colonnes
      dupliquées (le portefeuille de couverture répète le portefeuille court, valeur pour
      valeur). Doit être refusé : « la colonne Vega » y désigne deux colonnes.
  ``chunk-003f887b5001fe32`` — fragment d'un tableau de régression : une ligne d'en-tête,
      **aucune ligne de données**. Doit être refusé.
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import pytest

BENCH = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BENCH))
sys.path.insert(0, str(BENCH.parent))

import norme_valeurs as nv  # noqa: E402
import score_tableau as st  # noqa: E402
import selection_tableaux as sel  # noqa: E402
from corpus import ChunkIndex  # noqa: E402

PROPRE = "chunk-0046a568f5c7bbda"
ENTETE_MULTINIVEAU = "chunk-003deb25e50fc252"
SANS_DONNEES = "chunk-003f887b5001fe32"


def _index():
    if not hasattr(_index, "cache"):
        _index.cache = ChunkIndex.load(verbose=False)
    return _index.cache


def _candidats():
    if not hasattr(_candidats, "cache"):
        motifs = Counter()
        _candidats.cache = (sel.candidats(_index(), motifs=motifs), motifs)
    return _candidats.cache


# ------------------------------------------------------------------ analyse d'une table

def test_le_tableau_propre_du_corpus_est_analysable():
    """TABLE 5.1 : cinq colonnes, dix-huit strikes, tout numérique."""
    table = sel.analyser_tableau(_index().chunks[PROPRE]["text"])
    assert table is not None and table.motif is None
    assert "TABLE 5.1" in table.legende
    assert "Call Bid" in [entete.texte for entete in table.entetes]
    assert len(table.lignes) >= 15


def test_la_legende_est_lue_telle_que_l_overlay_l_ecrit():
    """L'overlay des tableaux préfixe la légende par « Table: » — c'est ce qui l'identifie."""
    texte = _index().chunks[PROPRE]["text"]
    assert texte.lstrip().startswith("Table:")
    table = sel.analyser_tableau(texte)
    assert not table.legende.startswith("Table:"), "le préfixe doit être retiré de la légende"


def test_un_fragment_sans_ligne_de_donnees_est_refuse():
    """Un tableau de régression fragmenté ne garde parfois que son en-tête. Rien à demander."""
    motifs = Counter()
    retenus = sel.candidats(_index(), motifs=motifs, limite=None)
    assert SANS_DONNEES not in {c["chunk_id"] for c in retenus}


def test_un_entete_multiniveau_est_refuse():
    """« | | Short Position | | Hedge Portfolio | | » : « la colonne Vega » y désigne deux colonnes."""
    retenus, _ = _candidats()
    assert ENTETE_MULTINIVEAU not in {c["chunk_id"] for c in retenus}


# ------------------------------------------------------------------ invariants de la sélection

def test_la_selection_rend_des_candidats_et_un_registre_de_rejets():
    retenus, motifs = _candidats()
    assert len(retenus) >= 40, "le pool doit couvrir largement la cible de 45 questions"
    assert motifs["retenu"] == len(retenus)
    assert sum(motifs.values()) > len(retenus), "les rejets doivent être comptés, pas tus"


def test_chaque_or_est_une_valeur_lisible_et_non_parenthesee():
    """Une parenthèse est ambiguë entre « négatif » et « écart-type » : elle n'est jamais de l'or."""
    for candidat in _candidats()[0]:
        valeur = nv.lire(candidat["valeur_brute"])
        assert valeur is not None, candidat["chunk_id"]
        assert not valeur.parenthesee, candidat["chunk_id"]


def test_chaque_or_est_unique_dans_son_tableau():
    """Si deux cellules portent la même valeur, une bonne réponse peut venir de la mauvaise."""
    for candidat in _candidats()[0]:
        or_valeur = nv.lire(candidat["valeur_brute"])
        jumelles = [autre for autre in candidat["autres_valeurs"]
                    if nv.egales(or_valeur, nv.lire(autre))]
        assert not jumelles, f"{candidat['chunk_id']} : {candidat['valeur_brute']} = {jumelles[:3]}"


def test_chaque_or_tient_dans_la_fenetre_servie():
    """Une cellule au-delà du caractère 2 500 n'est jamais montrée : elle mesurerait la troncature."""
    for candidat in _candidats()[0]:
        servi = _index().chunks[candidat["chunk_id"]]["text"][:2500]
        assert nv.contient(servi, nv.lire(candidat["valeur_brute"])), candidat["chunk_id"]


def test_les_entetes_et_les_libelles_sont_distincts():
    """Comparés sur leur **texte**, pas sur l'objet.

    Écrit d'abord en `set(table.entetes)` : `Cellule` n'a ni `__eq__` ni `__hash__`, donc
    l'ensemble comptait des identités d'objets et faisait passer le test à vide sur les
    216 candidats. Un test qui ne peut pas échouer ne garde rien.
    """
    for candidat in _candidats()[0]:
        table = sel.analyser_tableau(_index().chunks[candidat["chunk_id"]]["text"])
        entetes = [entete.texte for entete in table.entetes if entete.texte]
        assert len(entetes) == len(set(entetes)), f"{candidat['chunk_id']} : {entetes}"
        libelles = [ligne.libelle.texte for ligne in table.lignes if ligne.libelle.texte]
        assert len(libelles) == len(set(libelles)), f"{candidat['chunk_id']} : {libelles[:6]}"


def test_la_selection_est_reproductible():
    """Deux appels rendent la même liste, dans le même ordre — sinon rien n'est rejouable."""
    premier = [c["chunk_id"] for c in sel.candidats(_index())]
    second = [c["chunk_id"] for c in sel.candidats(_index())]
    assert premier == second


def test_le_quota_par_document_est_respecte():
    compte = Counter(c["document_id"] for c in _candidats()[0])
    assert max(compte.values()) <= sel.QUOTA_DOCUMENT


# ------------------------------------------------------------------ le scoreur

CELLULE = "0.250"


def test_la_bonne_valeur_est_juste():
    note = st.score("The table reports 0.250 for that maturity [1].", CELLULE)
    assert note["juste"] is True
    assert note["valeur_trouvee"] == "0.250"


def test_moins_de_decimales_reste_juste():
    """``0.250`` affirme trois décimales ; ``0.25`` dit la même chose."""
    assert st.score("It is 0.25 [1].", CELLULE)["juste"] is True


def test_une_valeur_differente_est_fausse():
    assert st.score("It is 0.26 [1].", CELLULE)["juste"] is False


def test_la_cellule_voisine_est_fausse_et_diagnostiquee():
    """L'erreur que la famille existe pour voir : le bon tableau, la mauvaise ligne."""
    note = st.score("The reported figure is 0.202 [1].", "0.282",
                    autres_valeurs=["0.202", "0.160", "1.059"])
    assert note["juste"] is False
    assert note["mauvaise_cellule"] is True
    assert note["autres_cellules"] == ["0.202"]


def test_une_valeur_absente_du_tableau_n_est_pas_une_mauvaise_cellule():
    note = st.score("The reported figure is 7.77 [1].", "0.282",
                    autres_valeurs=["0.202", "0.160"])
    assert note["juste"] is False
    assert note["mauvaise_cellule"] is False


def test_sans_autres_valeurs_le_diagnostic_est_absent_jamais_faux():
    note = st.score("The reported figure is 0.202 [1].", "0.282")
    assert note["mauvaise_cellule"] is False
    assert note["autres_cellules"] == []


def test_une_abstention_n_est_pas_juste_meme_si_le_chiffre_traine():
    """Le système a refusé de répondre : on enregistre qu'il a refusé."""
    note = st.score("INSUFFICIENT_EVIDENCE the passages only give 0.250 for another maturity.",
                    CELLULE)
    assert note["abstention"] is True
    assert note["juste"] is False


def test_une_abstention_en_gras_reste_une_abstention():
    assert st.abstention("**INSUFFICIENT_EVIDENCE** nothing covers it.") is True


def test_le_jeton_cite_au_milieu_n_est_pas_une_abstention():
    """« je n'ai pas répondu INSUFFICIENT_EVIDENCE parce que… » est un commentaire, pas un refus."""
    assert st.abstention("The value is 0.250; I did not need INSUFFICIENT_EVIDENCE here.") is False


def test_l_unite_divergente_est_separee_du_chiffre_faux():
    """« 84 » pour une cellule « 84% » : un défaut de rédaction, pas de lecture."""
    note = st.score("The probability is 84 [1].", "84%")
    assert note["juste"] is False
    assert note["unite_divergente"] is True


def test_le_pourcentage_correct_est_juste():
    assert st.score("The probability is 84% [1].", "84%")["juste"] is True


def test_le_negatif_unicode_est_lu():
    assert st.score("The coefficient is −0.229 [2].", "-0.229")["juste"] is True


def test_les_marqueurs_de_citation_ne_sont_pas_des_chiffres():
    """Sans cette précaution, « [1] » ferait passer pour juste toute cellule valant 1."""
    assert st.score("The answer is elsewhere [1].", "1")["juste"] is False


def test_un_or_illisible_leve_une_erreur_au_lieu_de_noter_faux():
    """Une question dont l'or est illisible doit sauter aux yeux, pas se noter « fausse »."""
    with pytest.raises(ValueError):
        st.score("anything", "n/a")


def test_un_vrai_candidat_du_corpus_se_note_juste_sur_sa_propre_cellule():
    """Le bout en bout : la valeur telle que le chunk l'écrit, restituée telle quelle."""
    candidat = next(c for c in _candidats()[0] if c["chunk_id"] == PROPRE)
    reponse = f"According to the table, the value is {candidat['valeur_brute']} [1]."
    note = st.score(reponse, candidat["valeur_brute"], candidat["autres_valeurs"])
    assert note["juste"] is True
    assert note["mauvaise_cellule"] is False
