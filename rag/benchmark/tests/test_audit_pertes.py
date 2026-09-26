"""L'arbre de la taxonomie doit être **total et sans recouvrement**, ou les comptes mentent.

Une taxonomie des pertes est utile exactement dans la mesure où ses classes se somment à la
population et ne se chevauchent pas. Une liste de critères testés indépendamment n'a aucune
de ces deux propriétés : une question dont l'or est au rang 3 *et* évincé par le plafond par
document appartiendrait à deux classes, et le total dépasserait 155 sans que rien ne le dise.

D'où un arbre **ordonné** : le premier test qui se déclenche donne la classe. Ces tests
vérifient l'ordre lui-même, pas seulement les cas faciles — et ils vérifient que la classe
``SERVI`` est décidée par ``pipeline.build_context``, la fonction de production, jamais par
une copie qui dériverait le jour où le plafond changerait.
"""
from __future__ import annotations

import sys
from pathlib import Path

BENCHMARK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCHMARK))
sys.path.insert(0, str(BENCHMARK.parent))

import audit_pertes  # noqa: E402
import pipeline  # noqa: E402


def _row(documents: list[str], ors: set[str], documents_or: set[str] | None = None) -> dict:
    """Une question dont on écrit le pool à la main : un chunk par rang, son document."""
    chunks = [f"c{i}" for i in range(1, len(documents) + 1)]
    return {"pool_chunks": chunks, "pool_documents": documents,
            "gold_chunks": ors, "n_pool": len(chunks),
            "gold_documents": documents_or if documents_or is not None else set(),
            "doc_hit": bool((documents_or or set()) & set(documents))}


def test_l_arbre_est_total_et_sans_recouvrement():
    """Chaque question reçoit exactement une classe, et c'est une classe connue."""
    cas = [
        _row(["dA"] * 8, {"c1"}),                                   # servi
        _row(["dA", "dA", "dA", "dB", "dC"], {"c3"}),               # évincé par le plafond
        _row(["dA", "dB", "dC", "dD", "dE", "dF", "dG"], {"c7"}),   # rang 6-10
        _row([f"d{i}" for i in range(20)], {"c15"}),                # rang 11-50
        _row(["dA", "dB"], {"cX"}, {"dA"}),                         # granularité
        _row(["dA", "dB"], {"cX"}, {"dZ"}),                         # découverte
        _row(["dA"], set()),                                        # or hors corpus
    ]
    vues = [audit_pertes.classer(row)[0] for row in cas]
    assert len(set(vues)) == len(vues), f"deux cas partagent une classe : {vues}"
    assert set(vues) <= set(audit_pertes.CLASSES)
    assert vues == ["SERVI", "EVICTION_DOCUMENT", "RANG_6_10", "RANG_11_50",
                    "GRANULARITE", "DECOUVERTE", "OR_HORS_CORPUS"]


def test_l_ordre_prime_sur_les_criteres_pris_isolement():
    """Un or au rang 3 servi est SERVI, pas EVICTION — l'ordre est le contrat."""
    servi = _row(["dA", "dB", "dC", "dD", "dE"], {"c3"})
    assert audit_pertes.classer(servi)[0] == "SERVI"
    # Mêmes rangs, mais les trois premiers viennent du même document : c3 est évincé.
    evince = _row(["dA", "dA", "dA", "dB", "dC"], {"c3"})
    classe, detail = audit_pertes.classer(evince)
    assert classe == "EVICTION_DOCUMENT" and detail["rang"] == 3


def test_un_or_hors_corpus_reste_compte_et_ne_sort_pas_du_denominateur():
    """La règle de population fixe : immesurable ≠ retiré. Retirer 5 questions rend +0,0200."""
    classe, detail = audit_pertes.classer(_row(["dA"], set()))
    assert classe == "OR_HORS_CORPUS"
    assert detail["n_or"] == 0


def test_granularite_et_decouverte_se_departagent_sur_le_document_seul():
    absent = _row(["dA", "dB"], {"cX"}, {"dZ"})
    present = _row(["dA", "dB"], {"cX"}, {"dB"})
    assert audit_pertes.classer(absent)[0] == "DECOUVERTE"
    assert audit_pertes.classer(present)[0] == "GRANULARITE"


def test_les_passages_servis_viennent_de_la_fonction_de_production():
    """Si ``build_context`` change de plafond, l'audit doit changer avec lui."""
    row = _row(["dA", "dA", "dA", "dA", "dB", "dC"], {"c9"})
    assert audit_pertes.passages_servis(row) == [
        p["chunk_id"] for p in pipeline.build_context(
            [{"chunk_id": c, "document_id": d}
             for c, d in zip(row["pool_chunks"], row["pool_documents"])])]
    # le plafond de production retient deux passages de dA, puis dB et dC
    assert audit_pertes.passages_servis(row) == ["c1", "c2", "c5", "c6"]


def test_le_rang_est_celui_du_premier_or_et_non_du_dernier():
    assert audit_pertes.rang_du_premier_or(["a", "b", "c"], {"c", "b"}) == 2
    assert audit_pertes.rang_du_premier_or(["a", "b"], {"z"}) is None


def test_la_grille_d_assemblage_commence_par_la_production():
    """Tous les écarts se lisent contre la production : elle doit être la première ligne."""
    assert audit_pertes.GRILLE_ASSEMBLAGE[0] == (pipeline.CONTEXT_PASSAGES, 2)
