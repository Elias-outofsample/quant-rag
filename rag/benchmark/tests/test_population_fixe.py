"""La population du banc est fixe : aucune dégradation ne doit rétrécir le dénominateur.

C'est la garantie centrale de la clôture du 6 septembre 2026, et elle vaut la peine d'être
énoncée en chiffres. Sur la ligne de base réelle — 155 questions, moyenne 0,5993 — retirer
du dénominateur les questions les plus dures rend, sans qu'aucune qualité ait bougé :

    5 questions retirées   -> +0,0200      (le double du seuil tenu pour décisif)
    10 questions retirées  -> +0,0413
    30 questions retirées  -> +0,1357

Deux chemins pouvaient produire cette attrition. Les deux sont fermés, et testés ici :

1. ``compare_v1_v2.load_v1`` écartait en silence toute question dont le chunk cible avait
   disparu du corpus — ce qui est exactement ce qu'un re-découpage provoque.
2. ``gold_ancrage.appliquer`` n'écrivait que les questions ré-ancrées (testé dans
   ``test_gold_ancrage.py``).

La convention retenue : une question non comparable **reste dans le fichier** et vaut 0.
La métrique de décision se lit sur la population d'origine ; celle du sous-ensemble
comparable est diagnostique et ne peut jamais valider un chantier.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

BANC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BANC))
sys.path.insert(0, str(BANC.parent))

import compare_v1_v2  # noqa: E402
import metrics  # noqa: E402


class _IndexFactice:
    """Un corpus où un seul chunk a disparu — la forme exacte d'un re-découpage."""

    def __init__(self, connus: dict):
        self._connus = connus

    def document_of(self, chunk_id):
        return self._connus.get(chunk_id)


def _ecrire(chemin: Path, questions: list[dict]) -> Path:
    chemin.write_text("\n".join(json.dumps(q, ensure_ascii=False) for q in questions) + "\n",
                      encoding="utf-8")
    return chemin


def test_load_v1_conserve_une_question_dont_le_chunk_a_disparu(tmp_path, capsys):
    chemin = _ecrire(tmp_path / "q1.jsonl", [
        {"qid": "q01", "question": "?", "target_chunk": "chunk-vivant"},
        {"qid": "q02", "question": "?", "target_chunk": "chunk-disparu"},
    ])
    items = compare_v1_v2.load_v1(_IndexFactice({"chunk-vivant": "doc-1"}), chemin)

    assert [i["qid"] for i in items] == ["q01", "q02"]        # le dénominateur ne bouge pas
    vivant, disparu = items
    assert vivant["comparable"] is True and vivant["gold_chunks"] == ["chunk-vivant"]
    assert disparu["comparable"] is False
    assert disparu["gold_chunks"] == [] and disparu["gold_documents"] == []
    assert "conservé à 0" in capsys.readouterr().out    # l'événement est dit, pas tu


def test_une_question_non_comparable_vaut_exactement_zero(tmp_path):
    """La convention doit être vérifiée sur le vrai calcul, pas supposée."""
    chemin = _ecrire(tmp_path / "q1.jsonl",
                     [{"qid": "q02", "question": "?", "target_chunk": "chunk-disparu"}])
    item = compare_v1_v2.load_v1(_IndexFactice({}), chemin)[0]
    rendus = [{"chunk_id": "peu importe", "document_id": "doc-1"},
              {"chunk_id": "autre", "document_id": "doc-1"}]
    assert metrics.ndcg(rendus, set(item["gold_chunks"]), set(item["gold_documents"])) == 0.0
    assert metrics.ndcg_documents(rendus, set(item["gold_documents"])) == 0.0


def test_l_attrition_aurait_gonfle_la_moyenne_et_ne_le_peut_plus(tmp_path):
    """La démonstration, en petit : la question dure disparaît du corpus.

    Ancien comportement — elle est écartée, la moyenne monte sur les survivantes.
    Nouveau — elle reste et vaut 0, la moyenne de la population baisse. Un chantier qui
    détruit de l'or est puni, jamais récompensé.
    """
    chemin = _ecrire(tmp_path / "q1.jsonl", [
        {"qid": "facile", "question": "?", "target_chunk": "chunk-facile"},
        {"qid": "dure", "question": "?", "target_chunk": "chunk-dur"},
    ])
    items = compare_v1_v2.load_v1(_IndexFactice({"chunk-facile": "doc-1"}), chemin)
    rendus = {"facile": [{"chunk_id": "chunk-facile", "document_id": "doc-1"}],
              "dure": [{"chunk_id": "n-importe-quoi", "document_id": "doc-9"}]}
    scores = {i["qid"]: metrics.ndcg(rendus[i["qid"]], set(i["gold_chunks"]),
                                     set(i["gold_documents"])) for i in items}
    assert scores == {"facile": 1.0, "dure": 0.0}

    population = sum(scores.values()) / len(scores)                       # 0,50
    survivantes = scores["facile"]                                        # 1,00
    assert survivantes > population        # le piège existe : il est nommé, pas subi
    assert population == 0.5               # et c'est cette valeur-là qui décide


def test_le_banc_lit_les_poids_quand_ils_existent():
    """Sans cette lecture, l'or pondéré serait décoratif et le piège de comptage de
    ``metrics.ndcg`` se rouvrirait sur tout banc ré-oré."""
    for module in ("eval_router.py", "calibrate_router.py"):
        source = (BANC / module).read_text(encoding="utf-8")
        assert 'item.get("gold_poids")' in source, module
        assert "poids=poids" in source, module


def test_le_banc_peut_etre_pointe_sur_un_banc_re_ore():
    """Une garde qu'on ne peut pas invoquer ne garde rien."""
    for module in ("eval_router.py", "calibrate_router.py"):
        source = (BANC / module).read_text(encoding="utf-8")
        assert "--gold-signature" in source, module
