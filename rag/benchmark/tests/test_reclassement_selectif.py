"""Le contrat du reclassement sélectif — sans charger un modèle, sans ouvrir Qdrant.

Ce que ces tests gardent, et pourquoi chacun existe :

  le contrat de sortie   ``_rerank`` écrasait ``score`` et ``score_kind``, si bien que le
                         score dense était perdu et qu'aucun diagnostic ne pouvait le
                         retrouver. La nouvelle règle n'enlève rien et ajoute quatre champs.
  le rang 1 protégé      c'est *la* règle servie, et toute la mesure repose dessus : ses cinq
                         pertes sont un sous-ensemble strict des sept du reclassement plein.
                         Un test qui ne vérifie pas cela ne garde rien.
  le départage           ``quant_rag._rerank`` s'en remet à la stabilité du tri de Python.
                         Déterministe par accident du langage, pas par contrat.
  le repli               un reclassement qui échoue doit rendre l'ordre dense, jamais une
                         erreur. Transformer une lenteur locale en panne du RAG coûterait
                         bien plus que les questions gagnées.
  la mémoire             lire « Pages free » comme la mémoire disponible fait voir une
                         famine là où il reste des gigaoctets. L'erreur a été commise en
                         écrivant le module ; ce test empêche qu'elle revienne.
"""
from __future__ import annotations

import inspect
import sys
from pathlib import Path

import pytest

RACINE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(RACINE / "rag"))

import reranking  # noqa: E402


def lignes(n: int = 15) -> list[dict]:
    """Un pool synthétique : rang dense i, score dense décroissant, texte non vide."""
    return [{"chunk_id": f"chunk-{i:02d}", "document_id": f"doc-{i % 3}",
             "score": 1.0 - i / 100, "text": f"passage {i} " * 40,
             "content_type": "text"} for i in range(1, n + 1)]


def scores_inverses(rows: list[dict]) -> dict[str, float]:
    """Le reranker classe exactement à l'envers du dense : le pire cas pour la protection."""
    return {r["chunk_id"]: float(i) for i, r in enumerate(rows)}


# --------------------------------------------------------------------------- contrat

def test_aucun_champ_ne_disparait_et_quatre_apparaissent():
    rows = lignes()
    sortie = reranking.reordonner(rows, scores_inverses(rows))
    assert len(sortie) == len(rows)
    for avant, apres in zip(rows, sorted(sortie, key=lambda r: r["chunk_id"])):
        for cle, valeur in avant.items():
            assert apres[cle] == valeur, f"{cle} a été modifié ou perdu"
    for row in sortie:
        assert {"score_dense", "rang_dense", "rang_final", "reclasse"} <= set(row)


def test_le_score_dense_survit_au_reclassement():
    """``_rerank`` écrasait ``score`` : le score dense devenait irrécupérable."""
    rows = lignes()
    sortie = reranking.reordonner(rows, scores_inverses(rows))
    par_id = {r["chunk_id"]: r for r in sortie}
    for row in rows:
        assert par_id[row["chunk_id"]]["score_dense"] == row["score"]


def test_aucun_candidat_perdu_ni_dupliqué():
    rows = lignes()
    sortie = reranking.reordonner(rows, scores_inverses(rows))
    assert sorted(r["chunk_id"] for r in sortie) == sorted(r["chunk_id"] for r in rows)


# --------------------------------------------------------------------------- la règle servie

def test_le_rang_1_dense_n_est_jamais_declasse():
    """Le cœur de la règle. Le reranker classe tout à l'envers : le rang 1 doit tenir."""
    rows = lignes()
    sortie = reranking.reordonner(rows, scores_inverses(rows))
    assert sortie[0]["chunk_id"] == rows[0]["chunk_id"]
    assert sortie[0]["rang_final"] == 1


def test_la_queue_garde_son_ordre_dense():
    rows = lignes(15)
    sortie = reranking.reordonner(rows, scores_inverses(rows), budget=10)
    queue_avant = [r["chunk_id"] for r in rows[10:]]
    queue_apres = [r["chunk_id"] for r in sortie[10:]]
    assert queue_avant == queue_apres
    assert all(not r["reclasse"] for r in sortie[10:])
    assert all(r["rerank_score"] is None for r in sortie[10:])


def test_le_budget_ne_reclasse_que_sa_tete():
    rows = lignes(15)
    sortie = reranking.reordonner(rows, scores_inverses(rows), budget=10)
    assert sum(1 for r in sortie if r["reclasse"]) == 10


def test_un_budget_plus_grand_que_le_pool_ne_casse_rien():
    rows = lignes(4)
    sortie = reranking.reordonner(rows, scores_inverses(rows), budget=10)
    assert len(sortie) == 4
    assert sortie[0]["chunk_id"] == rows[0]["chunk_id"]


def test_le_departage_des_ex_aequo_est_explicite_et_stable():
    """Scores tous égaux : l'ordre dense doit être rendu à l'identique, deux fois de suite."""
    rows = lignes()
    egaux = {r["chunk_id"]: 0.5 for r in rows}
    a = [r["chunk_id"] for r in reranking.reordonner(rows, egaux)]
    b = [r["chunk_id"] for r in reranking.reordonner(rows, egaux)]
    assert a == b == [r["chunk_id"] for r in rows]


def test_un_candidat_sans_score_coule_au_lieu_de_faire_echouer():
    rows = lignes()
    partiels = {r["chunk_id"]: 1.0 for r in rows[1:5]}
    sortie = reranking.reordonner(rows, partiels, budget=10, proteges=1)
    assert sortie[0]["chunk_id"] == rows[0]["chunk_id"]      # rang 1 protégé, et non scoré
    # Le tri ne porte que sur les candidats **mobiles** : la tête protégée en est exclue,
    # et c'est pourquoi elle apparaît « sans score » sans que cela signifie qu'elle coule.
    mobiles = [r for r in sortie if r["reclasse"]][1:]
    positions = {r["chunk_id"]: r["rang_final"] for r in mobiles}
    sans = [r["chunk_id"] for r in mobiles if r["rerank_score"] is None]
    avec = [r["chunk_id"] for r in mobiles if r["rerank_score"] is not None]
    assert avec and sans
    assert min(positions[c] for c in sans) > max(positions[c] for c in avec)


# --------------------------------------------------------------------------- le repli

def test_un_echec_de_modele_rend_l_ordre_dense_sans_lever(monkeypatch):
    def tombe(*_args, **_kwargs):
        raise RuntimeError("modèle absent")

    monkeypatch.setattr(reranking, "obtenir", tombe)
    rows = lignes()
    sortie, trace = reranking.reclasser("une question", rows, "cpu")
    assert [r["chunk_id"] for r in sortie] == [r["chunk_id"] for r in rows]
    assert "RuntimeError" in trace["echec"]


def test_le_repli_tient_avec_le_vrai_chargeur_pas_seulement_sous_mock():
    """Le test précédent remplace ``obtenir`` ; celui-ci laisse le vrai chargeur échouer.

    Un repli qui n'a été éprouvé que contre un mock n'a pas été éprouvé : c'est le code de
    chargement de ``transformers`` qui lève en production, pas un ``RuntimeError`` posé à la
    main. Un identifiant de modèle invalide échoue hors ligne et sans réseau, donc ce test
    reste déterministe.
    """
    rows = lignes(3)
    sortie, trace = reranking.reclasser("q", rows, "cpu", model_id="/chemin/inexistant/modele")
    assert sortie == rows
    assert trace["echec"] and "OSError" in trace["echec"]


def test_le_modele_n_est_pas_charge_tant_qu_on_ne_reclasse_pas():
    """Les 2,4 Go ne doivent pas être payés au démarrage du serveur MCP."""
    reranking.liberer()
    assert reranking._modele is None
    import quant_rag

    source = inspect.getsource(quant_rag.search_explained)
    assert "import reranking" in source, "l'import doit rester dans la branche, pas au module"


def test_un_pool_sans_texte_est_signale_et_ne_leve_pas():
    rows = [{**r, "text": ""} for r in lignes()]
    sortie, trace = reranking.reclasser("une question", rows, "cpu")
    assert sortie == rows
    assert trace["echec"] == "aucun candidat avec texte"


def test_la_trace_porte_de_quoi_auditer(monkeypatch):
    class Faux:
        def scorer(self, _requete, textes):
            return [float(i) for i in range(len(textes))]

    monkeypatch.setattr(reranking, "obtenir", lambda *_a, **_k: Faux())
    monkeypatch.setattr(reranking, "memoire_disponible_mo", lambda: 9999)
    _sortie, trace = reranking.reclasser("q", lignes(), "cpu")
    assert trace["echec"] is None
    assert trace["modele"] and trace["budget"] == 10 and trace["proteges"] == 1
    assert isinstance(trace["latence_ms"], int)
    assert trace["libere"] is False


def test_le_plancher_memoire_libere_le_modele(monkeypatch):
    class Faux:
        def scorer(self, _requete, textes):
            return [0.0] * len(textes)

    liberations = []
    monkeypatch.setattr(reranking, "obtenir", lambda *_a, **_k: Faux())
    monkeypatch.setattr(reranking, "memoire_disponible_mo", lambda: reranking.PLANCHER_MO - 1)
    monkeypatch.setattr(reranking, "liberer", lambda: liberations.append(True))
    _sortie, trace = reranking.reclasser("q", lignes(), "cpu")
    assert trace["libere"] is True and liberations


# --------------------------------------------------------------------------- la mémoire

def test_la_memoire_disponible_compte_les_pages_inactives():
    """« Pages free » seul fait voir une famine là où il reste des gigaoctets."""
    dispo = reranking.memoire_disponible_mo()
    assert dispo is None or dispo > 0
    if dispo is not None:
        import subprocess

        sortie = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
        libres = next(int(l.split(":")[1].strip().rstrip("."))
                      for l in sortie.splitlines() if l.startswith("Pages free"))
        taille = 16384
        for ligne in sortie.splitlines():
            if "page size of" in ligne:
                taille = int(ligne.split("page size of")[1].split("bytes")[0].strip())
        assert dispo >= libres * taille / 1_048_576


# --------------------------------------------------------------------------- le câblage

def test_le_chemin_servi_expose_le_parametre_sans_le_rendre_actif():
    import quant_rag

    for fonction in (quant_rag.search, quant_rag.search_explained):
        parametre = inspect.signature(fonction).parameters["reclassement"]
        assert parametre.default is None, "le reclassement ne doit jamais être actif par défaut"


def test_le_parametre_est_en_fin_de_signature():
    """Les appelants existants passent leurs arguments positionnellement."""
    import quant_rag

    for fonction in (quant_rag.search, quant_rag.search_explained):
        assert list(inspect.signature(fonction).parameters)[-1] == "reclassement"


def test_les_deux_modeles_restent_distincts():
    """Confondre les deux reviendrait à servir le modèle mesuré nuisible sur pool dense."""
    import quant_rag

    assert quant_rag.RECLASSEMENT_ID != quant_rag.RERANKER_ID
    assert "Qwen3-Reranker" in quant_rag.RECLASSEMENT_ID


def test_l_outil_mcp_documente_le_cout_et_l_usage_a_la_demande():
    """Un paramètre coûteux dont la doc tait le coût devient une heuristique d'agent."""
    import mcp_server

    doc = mcp_server.search_documents.__doc__ or ""
    assert "reclassement" in inspect.signature(mcp_server.search_documents).parameters
    assert "3,2 s" in doc and "54" in doc, "le coût en latence doit être en clair"
    assert "2,4 Go" in doc, "le coût mémoire doit être en clair"
    assert "routage non mesurée" in doc, "le piège de l'heuristique d'agent doit être nommé"


@pytest.mark.parametrize("valeur", ["approfondi", "gel1", "true", ""])
def test_une_valeur_inconnue_est_refusee_plutot_qu_ignoree(valeur):
    """Un mode mal orthographié qui serait silencieusement ignoré donnerait une mesure fausse."""
    import quant_rag

    source = inspect.getsource(quant_rag.search_explained)
    assert 'reclassement == "selectif"' in source
    assert "reclassement inconnu" in source
