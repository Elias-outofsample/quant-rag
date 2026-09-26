"""Le ré-ancrage du gold ne doit rien changer à ce qui est déjà mesuré.

Deux familles de preuves, et elles ne se remplacent pas :

1. ``metrics`` sans poids rend **exactement** ce qu'il rendait avant le 6 septembre 2026.
   La ligne de base commitée (v1 0,897 · v3 0,542 · pooled 0,599, signature 5530cba145)
   n'emprunte que ce chemin ; s'il bougeait d'un millième, tout le dossier bougerait.
   La preuve est un ré-implémentation figée de l'ancienne formule, comparée sur 500 cas
   tirés au sort — pas une relecture du code.

2. Le mappeur retrouve le bon texte, et **échoue quand il doit échouer**. Un mappeur qui
   ne peut pas échouer ne prouve rien : le test de sabotage est aussi obligatoire que
   celui d'identité.
"""
from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path

BANC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BANC))
sys.path.insert(0, str(BANC.parent))

import gold_ancrage  # noqa: E402
import metrics  # noqa: E402


# ------------------------------------------------------ 1. la ligne de base ne bouge pas


def _ndcg_avant(rows, gold_chunks, gold_documents, cutoff=10):
    """L'ancienne formule, figée ici. Toute dérive de ``metrics`` la fera diverger."""
    credited, observed = {}, []
    for row in rows:
        if row.get("chunk_id") in gold_chunks:
            observed.append(metrics.GAIN_GOLD)
            continue
        document = row.get("document_id")
        if document in gold_documents and credited.get(document, 0) < metrics.NEIGHBOUR_CREDITS:
            credited[document] = credited.get(document, 0) + 1
            observed.append(metrics.GAIN_SAME_DOCUMENT)
            continue
        observed.append(0.0)
    observed = observed[:cutoff]
    ideal = ([metrics.GAIN_GOLD] * len(gold_chunks))[:cutoff]
    dcg = sum(g / math.log2(i + 1) for i, g in enumerate(observed, 1))
    ref = sum(g / math.log2(i + 1) for i, g in enumerate(ideal, 1))
    return min(dcg / ref, 1.0) if ref else 0.0


def _tirage(graine: int):
    hasard = random.Random(graine)
    documents = [f"doc-{i}" for i in range(4)]
    chunks = [(f"chunk-{d}-{c}", d) for d in documents for c in range(6)]
    hasard.shuffle(chunks)
    rows = [{"chunk_id": c, "document_id": d} for c, d in chunks[:hasard.randint(1, 20)]]
    n_or = hasard.randint(1, 3)
    ors = hasard.sample([c for c, _ in chunks], n_or)
    docs_or = {d for c, d in chunks if c in ors}
    return rows, set(ors), docs_or


def test_sans_poids_identique_a_l_ancienne_formule():
    """500 cas tirés au sort : ``poids=None`` doit rendre l'ancien chiffre au bit près."""
    for graine in range(500):
        rows, ors, docs = _tirage(graine)
        assert metrics.ndcg(rows, ors, docs) == _ndcg_avant(rows, ors, docs), graine


def test_or_non_coupe_pondere_egale_or_non_pondere():
    """Un or resté d'un seul tenant (``{lui: 1.0}``) rend exactement le chiffre d'avant."""
    for graine in range(200):
        rows, ors, docs = _tirage(graine)
        if len(ors) != 1:
            continue
        unique = next(iter(ors))
        assert metrics.ndcg(rows, ors, docs, poids={unique: 1.0}) == \
            metrics.ndcg(rows, ors, docs), graine


def test_or_coupe_en_deux_ne_coute_rien_a_qui_rend_les_deux():
    """Le piège de comptage est bien neutralisé.

    Sans poids, un or coupé en deux porte l'idéal de 3,00 à 4,89 : le système qui rend
    les deux moitiés aux rangs 1 et 2 est correct et devrait valoir 1,00.
    """
    rows = [{"chunk_id": "a", "document_id": "d"}, {"chunk_id": "b", "document_id": "d"}]
    ors, docs = {"a", "b"}, {"d"}
    assert metrics.ndcg(rows, ors, docs, poids={"a": 0.5, "b": 0.5}) == 1.0
    # et le chemin non pondéré, lui, ne change pas : il vaut 1,0 aussi ici, mais l'idéal
    # n'est pas le même — c'est ce que la pondération corrige quand un seul morceau sort.
    seul = [{"chunk_id": "a", "document_id": "d"}, {"chunk_id": "z", "document_id": "z"}]
    partiel = metrics.ndcg(seul, ors, docs, poids={"a": 0.5, "b": 0.5})
    assert 0.0 < partiel < 1.0


# ------------------------------------------------------------------ 2. le mappeur


TEXTE_A = ("The deflated Sharpe ratio corrects the inflation induced by multiple testing. "
           "It requires the number of independent trials and the variance of the trial "
           "returns. Without those two quantities the correction cannot be applied at all.")
TEXTE_B = ("Market impact is concave in participation rate. The square root law is the "
           "usual parametrisation, and it holds across venues and asset classes.")


def _ancre(texte, document="doc-1", chunk="chunk-or"):
    return {"chunk_id": chunk, "document_id": document, "texte": texte,
            "qids": ["v1/q01"], "n_mots": len(texte.split())}


def test_identite_le_mappeur_retrouve_le_meme_chunk():
    ancres = {"chunk-or": _ancre(TEXTE_A)}
    nouveaux = [{"chunk_id": "chunk-or", "document_id": "doc-1", "text": TEXTE_A},
                {"chunk_id": "autre", "document_id": "doc-1", "text": TEXTE_B}]
    verdict = gold_ancrage.ancrer(ancres, nouveaux)["chunk-or"]
    assert verdict["statut"] == "ok"
    assert verdict["primaire"] == "chunk-or"
    assert verdict["couverture_union"] == 1.0
    assert verdict["poids"] == {"chunk-or": 1.0}


def test_contextualisation_un_prefixe_ne_derange_pas_l_ancre():
    ancres = {"chunk-or": _ancre(TEXTE_A)}
    nouveaux = [{"chunk_id": "ctx", "document_id": "doc-1",
                 "text": "This passage comes from a paper on backtest overfitting. " + TEXTE_A}]
    verdict = gold_ancrage.ancrer(ancres, nouveaux)["chunk-or"]
    assert verdict["statut"] == "ok"
    assert verdict["couverture_union"] == 1.0
    assert verdict["poids"] == {"ctx": 1.0}


def test_or_coupe_en_deux_donne_deux_poids_qui_somment_a_un():
    moitie = len(TEXTE_A) // 2
    nouveaux = [{"chunk_id": "g", "document_id": "doc-1", "text": TEXTE_A[:moitie]},
                {"chunk_id": "d", "document_id": "doc-1", "text": TEXTE_A[moitie:]}]
    verdict = gold_ancrage.ancrer({"chunk-or": _ancre(TEXTE_A)}, nouveaux)["chunk-or"]
    assert verdict["statut"] == "ok"
    assert set(verdict["poids"]) == {"g", "d"}
    assert abs(sum(verdict["poids"].values()) - 1.0) < 1e-6
    assert verdict["texte_disperse"] is True


def test_sabotage_le_mappeur_doit_echouer_quand_le_texte_a_disparu():
    """Sans ce test, l'identité ne prouverait rien : une garde qui ne peut pas échouer
    n'est pas une garde."""
    nouveaux = [{"chunk_id": "x", "document_id": "doc-1", "text": TEXTE_B}]
    verdict = gold_ancrage.ancrer({"chunk-or": _ancre(TEXTE_A)}, nouveaux)["chunk-or"]
    assert verdict["statut"] == "perdu"
    assert verdict["couverture_union"] < gold_ancrage.SEUIL_PERDU


def test_document_absent_du_nouveau_decoupage_est_perdu_pas_une_exception():
    verdict = gold_ancrage.ancrer({"chunk-or": _ancre(TEXTE_A)}, [])["chunk-or"]
    assert verdict["statut"] == "perdu"
    assert verdict["primaire"] is None


def test_les_seuils_pre_enregistres_n_ont_pas_bouge():
    """Les valeurs du commit ed7b7c0. Les déplacer serait du magasinage de seuil ;
    ce test rend le déplacement bruyant."""
    assert gold_ancrage.K_GRAMME == 5
    assert gold_ancrage.SEUIL_ANCRE == 0.80
    assert gold_ancrage.SEUIL_PERDU == 0.50
    assert gold_ancrage.MAX_UNION == 4
    # MIN_APPORT n'est pas pré-enregistré : il a été ajouté le 6 septembre après relecture
    # adverse, et déclaré comme tel. Il est figé ici pour la même raison — le déplacer doit
    # être un acte visible, pas un ajustement silencieux.
    assert gold_ancrage.MIN_APPORT == 0.05


# ------------------------------------- 3. les défauts trouvés par relecture adverse


def test_un_vrai_fragment_d_or_bat_toujours_un_voisin_quelconque():
    """Le défaut le plus grave trouvé le 6 septembre 2026.

    Avec ``gain = 3p``, un fragment portant moins d'un tiers de la réponse valait moins
    que le crédit de substitution de 1,0 accordé à n'importe quel chunk du bon document :
    rendre un vrai morceau de la réponse scorait SOUS rendre autre chose. Mesuré sur le
    contrôle II à 1 200 caractères : 303 des 517 membres d'union, 58,6 %, sous 1/3.
    """
    for part in (1.0, 0.5, 1 / 3, 0.25, 0.10, 0.01, 0.0):
        assert metrics.gain_pondere(part) >= metrics.GAIN_SAME_DOCUMENT, part
    assert metrics.gain_pondere(1.0) == metrics.GAIN_GOLD

    # et le classement, pas seulement le gain : un or coupé en quatre
    poids = {c: 0.25 for c in ("a", "b", "c", "d")}
    ors, docs = set(poids), {"d1"}
    fragment = [{"chunk_id": "a", "document_id": "d1"}]           # un vrai morceau
    voisin = [{"chunk_id": "zz", "document_id": "d1"}]            # n'importe quoi du doc
    assert metrics.ndcg(fragment, ors, docs, poids=poids) > \
        metrics.ndcg(voisin, ors, docs, poids=poids)


def test_une_ancre_perdue_n_inflate_pas_le_score_des_survivantes(tmp_path, monkeypatch):
    """Le second défaut : renormaliser sur les ancres survivantes gonflait le score.

    Une question à plusieurs ancres dont une disparaît doit SORTIR de la comparaison, pas
    voir son dénominateur rétrécir — sinon un chantier qui dégrade la couverture d'un fait
    affiche une amélioration.
    """
    monkeypatch.setattr(gold_ancrage, "HERE", tmp_path)
    for nom, chemin in (("QUESTIONS_V1", tmp_path / "q1.jsonl"),
                        ("QUESTIONS_V3", tmp_path / "q3.jsonl"),
                        ("ANCRES", tmp_path / "ancres.json")):
        monkeypatch.setattr(gold_ancrage, nom, chemin)

    (tmp_path / "q1.jsonl").write_text("", encoding="utf-8")
    (tmp_path / "q3.jsonl").write_text(json.dumps(
        {"qid": "m01", "question": "?", "gold_chunks": ["or-A", "or-B"],
         "gold_documents": ["doc-1"]}, ensure_ascii=False) + "\n", encoding="utf-8")
    (tmp_path / "ancres.json").write_text(json.dumps({
        "ancres": {"or-A": _ancre(TEXTE_A, chunk="or-A"),
                   "or-B": _ancre(TEXTE_B, chunk="or-B")}}, ensure_ascii=False),
        encoding="utf-8")
    # le nouveau découpage ne contient plus que le texte de l'ancre A
    (tmp_path / "neuf.jsonl").write_text(json.dumps(
        {"chunk_id": "n1", "document_id": "doc-1", "text": TEXTE_A}) + "\n", encoding="utf-8")

    rapport = gold_ancrage.appliquer(tmp_path / "neuf.jsonl", "sigcible99")
    assert rapport["comparables"]["v3"] == 0
    assert rapport["non_comparables"]["v3"] == ["m01"]   # déclarée, pas renormalisée
    # et surtout : elle reste dans la population, neutralisée
    assert rapport["population_origine"]["v3"] == 1
    sortie = [json.loads(l) for l in
              (tmp_path / "questions-v3-sigcible99.jsonl").read_text().splitlines() if l.strip()]
    assert len(sortie) == 1 and sortie[0]["qid"] == "m01"
    assert sortie[0]["comparable"] is False and sortie[0]["gold_chunks"] == []


def test_une_ancre_plus_courte_que_le_gramme_s_apparie_quand_meme():
    """Latent aujourd'hui — la plus courte des 182 ancres fait 58 mots — mais réel."""
    courte = "the risk premium is negative"
    long = f"Some preceding sentence. {courte}. And a following one, longer than five words."
    verdict = gold_ancrage.ancrer(
        {"c": {"chunk_id": "c", "document_id": "d", "texte": courte}},
        [{"chunk_id": "n", "document_id": "d", "text": long}])["c"]
    assert verdict["statut"] == "ok"
    assert verdict["couverture_union"] == 1.0


def test_le_plancher_d_apport_ecarte_un_candidat_de_gabarit():
    """Un chunk sans rapport ne doit pas entrer dans l'union pour trois mots de gabarit."""
    ancre = _ancre(TEXTE_A)
    gabarit = "Without those two quantities " + " ".join(f"unrelated{i}" for i in range(200))
    nouveaux = [{"chunk_id": "vrai", "document_id": "doc-1", "text": TEXTE_A},
                {"chunk_id": "gabarit", "document_id": "doc-1", "text": gabarit}]
    verdict = gold_ancrage.ancrer({"chunk-or": ancre}, nouveaux)["chunk-or"]
    assert verdict["poids"] == {"vrai": 1.0}
    assert "gabarit" not in verdict["union"]


def test_l_attrition_ne_peut_pas_gonfler_la_metrique_de_decision(tmp_path, monkeypatch):
    """Le test hostile de la clôture du 6 septembre 2026.

    Un chantier qui rend NON MESURABLES les questions difficiles doit être puni, pas
    récompensé. Mesuré sur la ligne de base réelle : retirer les 5 questions les plus
    dures des 155 rend un Δ apparent de +0,0200, le double du seuil tenu pour décisif ;
    en retirer 10 rend +0,0413. La garde tient si, et seulement si, la population reste
    fixe et les non comparables valent 0.
    """
    monkeypatch.setattr(gold_ancrage, "HERE", tmp_path)
    for nom, chemin in (("QUESTIONS_V1", tmp_path / "q1.jsonl"),
                        ("QUESTIONS_V3", tmp_path / "q3.jsonl"),
                        ("ANCRES", tmp_path / "ancres.json")):
        monkeypatch.setattr(gold_ancrage, nom, chemin)

    facile_a = "Value at risk at the ninety nine percent level is a quantile of the loss."
    facile_b = "Implied volatility surfaces are quoted in delta and tenor by most desks."
    (tmp_path / "q1.jsonl").write_text("", encoding="utf-8")
    (tmp_path / "q3.jsonl").write_text("\n".join(json.dumps(q, ensure_ascii=False) for q in [
        {"qid": "facile1", "question": "?", "gold_chunks": ["or-fa"], "gold_documents": ["doc-1"]},
        {"qid": "facile2", "question": "?", "gold_chunks": ["or-fb"], "gold_documents": ["doc-1"]},
        {"qid": "dure", "question": "?", "gold_chunks": ["or-A", "or-B"],
         "gold_documents": ["doc-1"]},
    ]) + "\n", encoding="utf-8")
    (tmp_path / "ancres.json").write_text(json.dumps({"ancres": {
        "or-fa": _ancre(facile_a, chunk="or-fa"), "or-fb": _ancre(facile_b, chunk="or-fb"),
        "or-A": _ancre(TEXTE_A, chunk="or-A"), "or-B": _ancre(TEXTE_B, chunk="or-B")}},
        ensure_ascii=False), encoding="utf-8")
    # le « chantier » conserve les faciles et DÉTRUIT une ancre de la question difficile
    (tmp_path / "neuf.jsonl").write_text("\n".join(json.dumps(r) for r in [
        {"chunk_id": "n-fa", "document_id": "doc-1", "text": facile_a},
        {"chunk_id": "n-fb", "document_id": "doc-1", "text": facile_b},
        {"chunk_id": "n-A", "document_id": "doc-1", "text": TEXTE_A},
    ]) + "\n", encoding="utf-8")

    rapport = gold_ancrage.appliquer(tmp_path / "neuf.jsonl", "hostile")

    # 1. la population ne rétrécit pas
    assert rapport["population_origine"]["v3"] == 3
    sortie = {q["qid"]: q for q in (json.loads(l) for l in
              (tmp_path / "questions-v3-hostile.jsonl").read_text().splitlines() if l.strip())}
    assert set(sortie) == {"facile1", "facile2", "dure"}

    # 2. la question dure est déclarée non comparable, avec son motif, et son or est vide
    assert sortie["dure"]["comparable"] is False
    assert sortie["dure"]["gold_chunks"] == [] and sortie["dure"]["gold_documents"] == []
    assert "or-B" in sortie["dure"]["motif_exclusion"]
    assert rapport["non_comparables"]["v3"] == ["dure"]
    assert rapport["multi_ancres"]["non_comparables"] == ["dure"]
    lignes = {c["qid"]: c for c in rapport["couverture"]}
    assert lignes["dure"]["comparable"] is False and lignes["dure"]["n_ancres"] == 2

    # 3. elle vaut exactement 0 — la métrique de décision ne peut pas monter grâce à elle
    rendus = [{"chunk_id": "n-A", "document_id": "doc-1"}]      # le système rend un morceau
    apres = {qid: metrics.ndcg(rendus, set(q["gold_chunks"]), set(q["gold_documents"]),
                               poids=q.get("gold_poids") or None)
             for qid, q in sortie.items()}
    assert apres["dure"] == 0.0

    # 4. et la démonstration du piège : la moyenne des SEULES comparables monte,
    #    celle de la population d'origine, non. C'est pourquoi la première ne décide pas.
    comparables = [v for qid, v in apres.items() if sortie[qid]["comparable"]]
    population = list(apres.values())
    assert sum(comparables) / len(comparables) > sum(population) / len(population)
