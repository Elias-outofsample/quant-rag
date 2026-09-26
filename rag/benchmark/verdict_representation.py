"""Verdict du chantier de représentation — la hiérarchie de `caa34b8`, appliquée telle qu'écrite.

Ce module **n'interprète pas**. Il applique le §0 bis du pré-enregistrement dans l'ordre,
premier cas satisfait l'emporte, et publie tout ce que le §5 de cette hiérarchie exige : le
sabotage et les sensibilités S1–S5, dans tous les cas et dans aucune branche.

    0. PORTES (un échec = arrêt)  contrôle Z · identité C1/C2 · exclusions hors négatives
                                  relues · borne >= 1 200 caractères
    1. BRAS RETENU                C2 si borne basse IC95(C2 − C1) > −0,005, sinon C1
    2. GO RETRIEVAL               bras retenu vs R : point >= +0,010 ET borne basse IC95 > 0
    3. NO-GO FRANC                IC95(C1 − R) entièrement < 0
    4. ACCEPTATION DE PROVENANCE  IC95(C1 − R) contient 0, garde réponse tenue, témoin >= −0,15
    5. sabotage et S1–S5          publiés dans tous les cas, dans aucune branche

Le test, verrouillé avant la mesure
------------------------------------
Bootstrap **apparié au niveau de la question** : unité = la question (155), tirage avec
remise, **B = 10 000**, graine **20260906**, IC95 par percentiles [2,5 ; 97,5]. Les questions
non comparables sont présentes et valent 0 **dans les deux bras** — leur Δ est exactement 0,
elles diluent l'écart sans le biaiser, et S4 publie de combien.

    .venv/bin/python rag/benchmark/verdict_representation.py --candidat 8d4ee77f1f
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(ROOT / "rag" / "ingestion"))

import corpus_overlay  # noqa: E402
import gel_corpus  # noqa: E402

B = 10_000
GRAINE = 20260906
SEUIL_GO = 0.010
SEUIL_NON_INFERIORITE = -0.005
BORNE_CARACTERES = 1200
MAX_LENGTH = 1024
REFERENCE = HERE / f"results-dense-controle-{corpus_overlay.signature()}.json"


# ------------------------------------------------------------------ bootstrap


def _percentile(valeurs: list[float], p: float) -> float:
    """Percentile par interpolation linéaire, sur une liste déjà triée."""
    if not valeurs:
        return float("nan")
    k = (len(valeurs) - 1) * p
    bas, haut = int(k), min(int(k) + 1, len(valeurs) - 1)
    return valeurs[bas] + (k - bas) * (valeurs[haut] - valeurs[bas])


def bootstrap_apparie(deltas: dict[str, float], b: int = B, graine: int = GRAINE) -> dict:
    """Δ moyen et IC95 par percentiles, rééchantillonnage **des questions** avec remise."""
    qids = sorted(deltas)
    valeurs = [deltas[q] for q in qids]
    n = len(qids)
    point = statistics.fmean(valeurs)
    rng = random.Random(graine)
    moyennes = []
    for _ in range(b):
        moyennes.append(statistics.fmean([valeurs[rng.randrange(n)] for _ in range(n)]))
    moyennes.sort()
    return {"n": n, "point": round(point, 6),
            "ic95_bas": round(_percentile(moyennes, 0.025), 6),
            "ic95_haut": round(_percentile(moyennes, 0.975), 6),
            "B": b, "graine": graine,
            "delta_nul_exact": sum(1 for v in valeurs if v == 0.0)}


# ------------------------------------------------------------------ chargement


def _ndcg_par_qid(chemin: Path) -> dict[str, float]:
    data = json.loads(chemin.read_text(encoding="utf-8"))
    return {f"{r['bench']}/{r['qid']}": r["ndcg"] for r in data["per_question"]}


def _resultats(chemin: Path) -> dict:
    return json.loads(chemin.read_text(encoding="utf-8"))


def _deltas(a: dict[str, float], b: dict[str, float]) -> dict[str, float]:
    if set(a) != set(b):
        manquants = (set(a) ^ set(b))
        sys.exit(f"appariement impossible : {len(manquants)} qid présents d'un seul côté "
                 f"({sorted(manquants)[:5]}) — c'est une erreur d'exécution, pas une question à écarter")
    return {q: a[q] - b[q] for q in a}


# ------------------------------------------------------------------ portes


def portes(signature: str) -> dict:
    """Cas 0 de la hiérarchie. Un échec, et rien d'autre n'est calculé."""
    resultat = {}

    vivante = corpus_overlay.signature()
    attendue = gel_corpus.lire()["signature"]
    resultat["controle_Z"] = {"signature_vivante": vivante, "attendue": attendue,
                              "ok": vivante == attendue}

    identite = HERE / f"identite-c1c2-{signature}.json"
    if identite.exists():
        v = json.loads(identite.read_text(encoding="utf-8"))
        detail = v.get("i4_i5") or {}
        complet = bool(detail.get("i4")) and bool(detail.get("i5"))
        resultat["identite_c1c2"] = {
            "toutes_assertions_passent": v["toutes_assertions_disponibles_passent"],
            "I4": detail.get("i4"), "I5": detail.get("i5"),
            "ok": v["toutes_assertions_disponibles_passent"] and complet}
    else:
        resultat["identite_c1c2"] = {"ok": False, "motif": "verdict d'identité absent"}

    remap = json.loads((HERE / f"gold-remap-{signature}.json").read_text(encoding="utf-8"))
    questions = {}
    for ligne in (HERE / "questions-v3.jsonl").read_text(encoding="utf-8").splitlines():
        if ligne.strip():
            q = json.loads(ligne)
            questions[q["qid"]] = q.get("kind")
    exclues = list(remap["non_comparables"]["v1"]) + list(remap["non_comparables"]["v3"])
    hors_negatives = [q for q in exclues if questions.get(q) != "negative"]
    resultat["exclusions"] = {"total": len(exclues), "negatives": len(exclues) - len(hors_negatives),
                              "hors_negatives": hors_negatives, "ok": not hors_negatives}

    racine = ROOT / "data" / "processed" / f"candidat-{signature}"
    tailles = sorted(len(r["text"]) for r in
                     (json.loads(l) for l in (racine / "chunks-servis.jsonl").open(encoding="utf-8") if l.strip()))
    mediane = tailles[len(tailles) // 2]
    resultat["borne_caracteres"] = {"mediane": mediane, "borne": BORNE_CARACTERES,
                                    "ok": mediane >= BORNE_CARACTERES}

    resultat["toutes_ouvertes"] = all(v.get("ok") for v in resultat.values() if isinstance(v, dict))
    return resultat


# ------------------------------------------------------------------ sensibilités


def sensibilites(signature: str, r_ndcg, c1_ndcg, c2_ndcg, retenu_ndcg, retenu_nom) -> dict:
    """S1–S5. Publiées dans tous les cas, dans **aucune** branche de décision."""
    out: dict = {}

    # S1 — or tronqué exclu. EXPLICATIVE seulement (amendement n°1, §0 bis (1)).
    tronques = _questions_a_or_tronque(signature)
    restants = {q: v for q, v in _deltas(c2_ndcg, c1_ndcg).items() if q not in tronques}
    out["S1_or_tronque_exclu"] = {
        "statut": "EXPLICATIVE — n'intervient dans aucune règle de décision",
        "questions_ecartees": len(tronques), "qids": sorted(tronques)[:10],
        "C2_moins_C1_sans_elles": bootstrap_apparie(restants) if restants else None}

    # S2 — sous-ensemble comparable (diagnostic historique).
    ref = _resultats(REFERENCE)
    comparables = {f"{r['bench']}/{r['qid']}" for r in
                   _resultats(HERE / f"results-dense-{signature}-c1.json")["per_question"] if r["comparable"]}
    out["S2_sous_ensemble_comparable"] = {
        "statut": "DIAGNOSTIC — ne valide jamais un chantier",
        "n": len(comparables),
        "C1_moins_R": bootstrap_apparie({q: v for q, v in _deltas(c1_ndcg, r_ndcg).items() if q in comparables})}

    # S3 — ventilation par banc et par famille : OÙ le gain se trouve, avant COMBIEN il vaut.
    par_kind = collections.defaultdict(dict)
    for r in _resultats(HERE / f"results-dense-{signature}-c1.json")["per_question"]:
        par_kind[r["kind"]][f"{r['bench']}/{r['qid']}"] = r["ndcg"]
    deltas_c1r = _deltas(c1_ndcg, r_ndcg)
    out["S3_ventilation"] = {
        "statut": "DIAGNOSTIC",
        "par_banc": {banc: round(statistics.fmean([v for q, v in deltas_c1r.items() if q.startswith(banc + "/")]), 4)
                     for banc in ("v1", "v3")},
        "par_famille": {kind: {"n": len(qids),
                               "delta_C1_moins_R": round(statistics.fmean([deltas_c1r[q] for q in qids]), 4)}
                        for kind, qids in sorted(par_kind.items())}}

    # S4 — part de la population à Δ = 0 exactement : de combien le test apparié est dilué.
    out["S4_delta_nul"] = {
        "statut": "DIAGNOSTIC",
        "C1_moins_R": sum(1 for v in deltas_c1r.values() if v == 0.0),
        "C2_moins_C1": sum(1 for v in _deltas(c2_ndcg, c1_ndcg).values() if v == 0.0),
        "population": len(deltas_c1r)}

    # S5 — effet sur les CINQ passages servis, pas seulement sur le rang.
    out["S5_cinq_passages"] = _cinq_passages(signature, retenu_nom)
    return out


def _questions_a_or_tronque(signature: str) -> set[str]:
    """Les questions dont un chunk d'or dépasse 1 024 jetons dans l'un des deux bras."""
    from transformers import AutoTokenizer

    from rechunk_corpus import retrieval_text, titres_plonges

    racine = ROOT / "data" / "processed" / f"candidat-{signature}"
    servis = {r["chunk_id"]: r["text"] for r in
              (json.loads(l) for l in (racine / "chunks-servis.jsonl").open(encoding="utf-8") if l.strip())}
    infos = {}
    for row in (json.loads(l) for l in (racine / "rows.jsonl").open(encoding="utf-8") if l.strip()):
        c = row["chunk"]
        infos[c["chunk_id"]] = (c["document_id"], c.get("title_path"), c.get("page_start"))
    titres, _ = titres_plonges()
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-Embedding-0.6B")

    ors: dict[str, list[str]] = {}
    for banc, fichier in (("v1", f"questions-v1-{signature}.jsonl"), ("v3", f"questions-v3-{signature}.jsonl")):
        for ligne in (HERE / fichier).read_text(encoding="utf-8").splitlines():
            if not ligne.strip():
                continue
            q = json.loads(ligne)
            cibles = q.get("gold_chunks") or ([q["target_chunk"]] if q.get("target_chunk") else [])
            ors[f"{banc}/{q['qid']}"] = cibles

    cache: dict[str, bool] = {}
    touchees = set()
    for qid, cibles in ors.items():
        for cid in cibles:
            if cid not in cache:
                document_id, chemin, page = infos.get(cid, (None, None, None))
                if document_id is None:
                    cache[cid] = False
                    continue
                texte = servis.get(cid, "")
                c1 = retrieval_text(titres[document_id], chemin, page, texte, avec_page=False)
                c2 = retrieval_text(titres[document_id], chemin, page, texte, avec_page=True)
                cache[cid] = (len(tok.encode(c1, add_special_tokens=False)) > MAX_LENGTH
                              or len(tok.encode(c2, add_special_tokens=False)) > MAX_LENGTH)
            if cache[cid]:
                touchees.add(qid)
    return touchees


def _cinq_passages(signature: str, retenu: str) -> dict:
    """L'or entre-t-il, ou sort-il, des cinq passages réellement servis ?

    Le §7.6 de ``STRATEGIE.md`` en fait le critère qui précède toute mesure de classement :
    un gain entre les rangs 6 et 10 est invisible à une production qui sert cinq passages.
    On rejoue donc la sélection de production — ``limit=5``, ``per_document=2``,
    ``MIN_CHARACTERS=250``, déduplication — sur la référence et sur le bras retenu.
    """
    import pipeline
    import quant_rag
    from build_collection_candidat import nom_collection
    from compare_v1_v2 import load_bench, load_v1
    from corpus import ChunkIndex
    from eval_dense_candidat import index_candidat

    def servis(collection, index, fichier_v1, fichier_v3):
        precedente, quant_rag.COLLECTION = quant_rag.COLLECTION, collection
        try:
            out = {}
            for banc, items in (("v1", load_v1(index, fichier_v1)), ("v3", load_bench(fichier_v3))):
                for item in items:
                    filtres = {k: v for k, v in (pipeline.filters_of(item) or {}).items()
                               if k in pipeline.FILTER_KEYS and v is not None}
                    rows = quant_rag.search(pipeline.query_of(item), limit=5, mode="dense", rerank=False,
                                            auto_period=False, log=False, **filtres)
                    out[f"{banc}/{item['qid']}"] = ({r["chunk_id"] for r in rows}
                                                    & set(item["gold_chunks"])) != set()
            return out
        finally:
            quant_rag.COLLECTION = precedente

    reference = servis(quant_rag.COLLECTION, ChunkIndex.load(verbose=False),
                       HERE / "questions-v1.jsonl", HERE / "questions-v3.jsonl")
    candidat = servis(nom_collection(signature, retenu), index_candidat(signature),
                      HERE / f"questions-v1-{signature}.jsonl", HERE / f"questions-v3-{signature}.jsonl")
    communs = set(reference) & set(candidat)
    entrees = sorted(q for q in communs if candidat[q] and not reference[q])
    sorties = sorted(q for q in communs if reference[q] and not candidat[q])
    return {"statut": "DIAGNOSTIC — où le gain se trouve, avant combien il vaut",
            "bras": retenu, "population": len(communs),
            "or_servi_reference": sum(reference[q] for q in communs),
            "or_servi_candidat": sum(candidat[q] for q in communs),
            "entrees": entrees, "sorties": sorties,
            "net": len(entrees) - len(sorties)}


# ------------------------------------------------------------------ la hiérarchie


def verdict(signature: str) -> dict:
    p = portes(signature)
    if not p["toutes_ouvertes"]:
        return {"cas": "0 — PORTE FERMÉE", "portes": p,
                "conclusion": "le chantier s'arrête ; aucune métrique n'est calculée"}

    r_ndcg = _ndcg_par_qid(REFERENCE)
    c1_ndcg = _ndcg_par_qid(HERE / f"results-dense-{signature}-c1.json")
    c2_ndcg = _ndcg_par_qid(HERE / f"results-dense-{signature}-c2.json")
    sab = HERE / f"results-dense-{signature}-sabote.json"
    sabote_ndcg = _ndcg_par_qid(sab) if sab.exists() else None

    garde_fichier = HERE / f"garde-{signature}.json"
    garde = json.loads(garde_fichier.read_text(encoding="utf-8")) if garde_fichier.exists() else None
    c2_c1 = bootstrap_apparie(_deltas(c2_ndcg, c1_ndcg))
    c1_r = bootstrap_apparie(_deltas(c1_ndcg, r_ndcg))

    # cas 1 — bras retenu. S1 n'y intervient pas.
    page_retenue = c2_c1["ic95_bas"] > SEUIL_NON_INFERIORITE
    retenu = "c2" if page_retenue else "c1"
    retenu_ndcg = c2_ndcg if page_retenue else c1_ndcg
    retenu_r = bootstrap_apparie(_deltas(retenu_ndcg, r_ndcg))

    # La garde du §11 est un VETO, et elle se décide hors ligne (garde_reponse.py).
    garde_tenue = bool(garde and garde["bras"][retenu]["tenue"])

    # cas 2, 3, 4 — premier satisfait l'emporte
    if retenu_r["point"] >= SEUIL_GO and retenu_r["ic95_bas"] > 0:
        cas = "2 — GO RETRIEVAL"
        conclusion = ("Le bras retenu bat la référence au-dessus du seuil pré-enregistré, "
                      "intervalle excluant zéro. Autorise un shadow. Seul verdict qui revendique un gain.")
    elif c1_r["ic95_haut"] < 0:
        cas = "3 — NO-GO FRANC"
        conclusion = ("L'IC95 de C1 − R est entièrement négatif : une étiquette fausse aidait le "
                      "classement. Aucune promotion, rapport dédié.")
    elif c1_r["ic95_bas"] <= 0 <= c1_r["ic95_haut"] and garde_tenue:
        cas = "4 — ACCEPTATION DE PROVENANCE (sous réserve du témoin >= −0,15)"
        conclusion = ("L'IC95 de C1 − R contient 0 et la garde de non-régression tient. Publiable "
                      "« neutre au classement, correct au contenu » si le témoin reste >= −0,15. "
                      "N'autorise aucun shadow et ne revendique aucun gain de retrieval.")
    elif c1_r["ic95_bas"] <= 0 <= c1_r["ic95_haut"]:
        cas = "AUCUN CAS SATISFAIT — la garde a fermé le cas 4"
        conclusion = ("L'IC95 de C1 − R contient 0, mais la garde de non-régression du §11 ÉCHOUE : "
                      "le cas 4 exige qu'elle tienne, et elle est un VETO. Les cas 2 et 3 ne sont pas "
                      "satisfaits non plus. Aucun cas de la hiérarchie ne l'est donc, et le candidat "
                      "n'est pas retenu. Ce n'est pas un NO-GO franc — l'IC de C1 − R contient 0 — "
                      "c'est une branche fermée par la seule garde, sans qu'un appel LLM ait été "
                      "nécessaire pour le savoir.")
    else:
        cas = "2bis — au-dessus de zéro mais sous le seuil"
        conclusion = ("L'IC95 exclut zéro par le haut mais le point est sous +0,010 : le cas 2 n'est "
                      "pas satisfait et le cas 4 non plus (l'IC ne contient pas 0). La hiérarchie ne "
                      "prévoit pas cet état ; il est publié comme LIMITE, pas comme amendement.")

    sortie = {
        "signature_candidate": signature,
        "hierarchie": "pré-enregistrement caa34b8, §0 bis — premier cas satisfait l'emporte",
        "portes": p,
        "cas_1_bras_retenu": {"retenu": retenu, "regle": "C2 si IC95_bas(C2 − C1) > −0,005",
                              "C2_moins_C1": c2_c1, "page_retenue": page_retenue},
        "cas_atteint": cas,
        "conclusion": conclusion,
        "garde_non_regression": ({"statut": "VETO — se décide hors ligne, aucun appel LLM",
                                  "population_n": garde["population"]["n"],
                                  "seuil_echec": garde["seuil_echec"],
                                  "bras": {k: {x: v[x] for x in ("b", "g", "part", "wilson_haut", "tenue")}
                                           for k, v in garde["bras"].items()},
                                  "tenue_sur_le_bras_retenu": garde_tenue}
                                 if garde else {"statut": "non calculée"}),
        "C1_moins_R": c1_r,
        "bras_retenu_moins_R": retenu_r,
        "sabotage": ({"statut": "CONTRÔLE NÉGATIF EXPLICATIF — dans aucune branche",
                      "sabote_moins_C1": bootstrap_apparie(_deltas(sabote_ndcg, c1_ndcg))}
                     if sabote_ndcg else {"statut": "non calculé"}),
        "sensibilites": sensibilites(signature, r_ndcg, c1_ndcg, c2_ndcg, retenu_ndcg, retenu),
    }
    return sortie


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--candidat", required=True, metavar="SIG")
    a = p.parse_args()
    v = verdict(a.candidat)
    chemin = HERE / f"verdict-representation-{a.candidat}.json"
    chemin.write_text(json.dumps(v, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(v, ensure_ascii=False, indent=1))
    print(f"\nécrit : {chemin.name}")


if __name__ == "__main__":
    main()
