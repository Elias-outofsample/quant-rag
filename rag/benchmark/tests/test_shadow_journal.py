"""L'observateur du shadow ne doit rien changer à ce que la production rend.

Une touche au chemin de production, si minime soit-elle, se paie d'une preuve. Ces tests
disent exactement deux choses : la valeur rendue par ``search_explained`` est identique au
bit près avec et sans les champs du shadow, et le journal les porte réellement — sans quoi
le rejeu différé ne pourrait pas prouver qu'il rejoue le même corpus.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HERE))

import quant_rag  # noqa: E402


@pytest.fixture
def journal(tmp_path, monkeypatch):
    chemin = tmp_path / "router-decisions.jsonl"
    monkeypatch.setattr(quant_rag, "DECISION_LOG", chemin)
    monkeypatch.setattr(quant_rag.corpus_overlay, "signature", lambda: "abcdef1234")
    return chemin


def _lignes(chemin):
    return [json.loads(l) for l in chemin.read_text(encoding="utf-8").splitlines() if l.strip()]


DECISION = {"mode": "dense", "reason": "test", "rerank": False, "exact_tokens": {"n_exact": 0}}


def test_le_journal_porte_la_signature_et_les_passages_servis(journal):
    quant_rag._log_decision("q", dict(DECISION), [{"chunk_id": "chunk-a"}, {"chunk_id": "chunk-b"}], limit=7)
    ligne, = _lignes(journal)
    assert ligne["signature"] == "abcdef1234"
    assert ligne["served"] == ["chunk-a", "chunk-b"]
    # `returned` vaut 2 mais l'appelant en demandait 7 : sans `limit`, le rejeu servirait
    # deux passages là où la production en cherchait sept, et le contrôle passerait à tort.
    assert ligne["limit"] == 7


def test_sans_resultats_le_champ_reste_nul_et_rien_ne_casse(journal):
    quant_rag._log_decision("q", dict(DECISION))
    ligne, = _lignes(journal)
    assert ligne["served"] is None and ligne["limit"] is None
    assert ligne["signature"] == "abcdef1234"


def test_la_decision_rendue_a_l_appelant_n_est_pas_modifiee(journal):
    decision = dict(DECISION)
    avant = json.dumps(decision, sort_keys=True)
    quant_rag._log_decision("q", decision, [{"chunk_id": "chunk-a"}])
    assert json.dumps(decision, sort_keys=True) == avant, (
        "les champs du shadow doivent vivre dans la ligne de journal, jamais dans le "
        "dictionnaire `routing` rendu à l'appelant")


def test_un_journal_illisible_ne_casse_jamais_une_recherche(tmp_path, monkeypatch):
    monkeypatch.setattr(quant_rag, "DECISION_LOG", tmp_path / "interdit" / "x.jsonl")
    monkeypatch.setattr(Path, "mkdir", lambda *a, **k: (_ for _ in ()).throw(OSError("nope")))
    quant_rag._log_decision("q", dict(DECISION), [{"chunk_id": "chunk-a"}])  # ne doit pas lever


def test_format_routing_ignore_les_champs_ajoutes():
    """`routing` est rendu à l'appelant : un champ inconnu ne doit pas apparaître à l'écran."""
    decision = {"mode": "dense", "reason": "test", "rerank": False,
                "signature": "abcdef1234", "served": ["chunk-a"], "limit": 7}
    rendu = quant_rag.format_routing(decision)
    assert "abcdef1234" not in rendu and "chunk-a" not in rendu
