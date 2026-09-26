"""`characters` comme paramètre de produit — quatre paliers, 130 questions, juge de référence.

Instrument du pré-enregistrement ``PRE-ENREGISTREMENT-CHARACTERS-2026-09-08.md``, Partie I
publiée en ``a91c14f`` **avant la première réponse générée**.

**Ce fil ne rouvre pas la question de mécanisme.** Le fil du matin a demandé si la coupe du
chunk d'or est la *cause* des échecs et a rendu NO-GO. Ici on demande autre chose : **quelle
fenêtre servir**. Il n'y a pas de bras saboté.

Le contexte est dérivé **une seule fois** par la sélection de production
(``eval_reranking.servis`` → ``quant_rag._select``) et il est **identique aux quatre paliers** —
``characters`` s'applique au rendu, pas au choix. La comparaison est donc parfaitement appariée.

Générateur ``mistral-small-latest``, juge ``mistral-medium-latest`` : la règle de la maison. **La
fenêtre du juge est fixe à 4 500** pour que l'instrument ne change pas avec ce qu'il mesure.

    .venv/bin/python rag/benchmark/eval_characters.py --etape temoins
    .venv/bin/python rag/benchmark/eval_characters.py --etape mesure
    .venv/bin/python rag/benchmark/eval_characters.py --etape verdict
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import corpus_overlay  # noqa: E402
import eval_reranking  # noqa: E402
import experiment  # noqa: E402
import judge  # noqa: E402
import llm  # noqa: E402
import pipeline  # noqa: E402
from corpus import ChunkIndex  # noqa: E402

SIGNATURE = corpus_overlay.signature()
GENERATEUR, JUGE = "mistral-small-latest", "mistral-medium-latest"
#: La grille, déclarée au §2. Le premier palier est la référence servie aujourd'hui.
GRILLE = (1600, 2400, 3200, 4500)
REFERENCE = GRILLE[0]
#: Fenêtre du juge, fixe au plus grand palier — l'instrument ne change pas avec le bras (§4).
FENETRE_JUGE = max(GRILLE)
#: Seuils calculés avant la mesure (§5), vérifiés sur ``garde_reponse.wilson_haut`` à n = 130.
GARDE_ABSTENTION, GARDE_DEGRADATION = 2, 6
#: Condition de lisibilité du §4 : la famille qui garde contre « plus fluide, moins fondé ».
FAMILLE_BLOQUANTE, SEUIL_FAMILLE = "unsupported_fluent", 0.75
GRAINE, TIRAGES = 20260908, 10000
CACHE = HERE / ".cache"


def _charger(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def _ecrire(p: Path, d: dict) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")


def population(index=None) -> tuple[dict, dict]:
    """Les 130 questions v3 positives et leur contexte servi — dérivé une seule fois.

    ``index`` injectable, et ce n'est pas une commodité : ``ChunkIndex.load()`` bâtit sa vue
    depuis ``rows.jsonl``, c'est-à-dire le corpus **servi**. Un chantier qui mesure un corpus
    candidat doit passer la sienne (``eval_dense_candidat.index_candidat``), faute de quoi le
    contexte serait composé des passages du servi pendant que le classement vient du candidat —
    deux corpus dans un même prompt, et rien pour le dire.
    """
    index = index if index is not None else ChunkIndex.load(verbose=False)
    cache = experiment.cached_rankings()
    items, contextes = {}, {}
    for ligne in (HERE / "questions-v3.jsonl").read_text(encoding="utf-8").splitlines():
        if not ligne.strip():
            continue
        item = json.loads(ligne)
        if item.get("kind") == "negative":
            continue
        cle = f"v3/{item['qid']}"
        rows = []
        for rang, (c, d, s) in enumerate(cache[cle]["dense"], 1):
            src = index.get(c) or {}
            fiche = index.metadata.get(d) or {}
            rows.append({"chunk_id": c, "document_id": d, "score": s, "rang_dense": rang,
                         "text": src.get("text", ""), "content_type": src.get("content_type"),
                         "section": src.get("section"), "page_start": src.get("page_start"),
                         "title": fiche.get("title") or index.title_of(d),
                         "short_ref": fiche.get("short_ref")})
        items[item["qid"]] = item
        contextes[item["qid"]] = eval_reranking.servis(rows)
    return items, contextes


def temoins(index=None) -> None:
    """Les 18 items-témoins du juge — **avant** toute lecture de verdict (§4).

    ``index`` injectable pour la même raison que ``population`` : les pièges sont bâtis sur des
    passages, et ceux d'un corpus candidat ne sont pas ceux du servi.
    """
    index = index if index is not None else ChunkIndex.load(verbose=False)
    tous = [json.loads(l) for l in (HERE / "questions-v3.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    avant = (llm.GENERATOR, llm.JUDGE)
    llm.GENERATOR, llm.JUDGE = GENERATEUR, JUGE
    try:
        resultat = judge.run_traps(judge.build_traps(tous, index))
    finally:
        llm.GENERATOR, llm.JUDGE = avant
    famille = (resultat["per_family"].get(FAMILLE_BLOQUANTE) or {}).get("accuracy")
    resultat["lisible"] = famille is not None and famille >= SEUIL_FAMILLE
    resultat["generateur"], resultat["juge"] = GENERATEUR, JUGE
    _ecrire(CACHE / f"characters-temoins-{SIGNATURE}.json", resultat)
    print(json.dumps({k: v for k, v in resultat.items() if k != "detail"}, ensure_ascii=False, indent=1))
    print(f"\n{FAMILLE_BLOQUANTE} = {famille}  seuil {SEUIL_FAMILLE}  -> "
          f"{'LISIBLE' if resultat['lisible'] else 'ILLISIBLE — le fil s’arrête'}")


def mesure() -> None:
    temoins_lus = _charger(CACHE / f"characters-temoins-{SIGNATURE}.json")
    if not temoins_lus:
        sys.exit("items-témoins non mesurés — lancer --etape temoins d'abord (§4)")
    if not temoins_lus.get("lisible"):
        sys.exit(f"{FAMILLE_BLOQUANTE} sous {SEUIL_FAMILLE} : aucun verdict ne serait lisible (§4)")
    items, contextes = population()
    reponses = _charger(CACHE / f"characters-reponses-{SIGNATURE}.json")
    verdicts = _charger(CACHE / f"characters-verdicts-{SIGNATURE}.json")
    plan = [(qid, c) for c in GRILLE for qid in sorted(items)]
    print(f"{len(plan)} couples (question, palier) — {GENERATEUR} / {JUGE}")
    for numero, (qid, fenetre) in enumerate(plan, 1):
        k = f"{qid}/{fenetre}"
        if k in verdicts:
            continue
        if k not in reponses:
            sortie = pipeline.answer(items[qid]["question"], contextes[qid],
                                     model=GENERATEUR, characters=fenetre)
            reponses[k] = {"palier": fenetre, "answer": sortie["answer"],
                           "abstained": sortie["abstained"]}
            _ecrire(CACHE / f"characters-reponses-{SIGNATURE}.json", reponses)
        v = judge.grade(items[qid], reponses[k]["answer"], contextes[qid],
                        seed=f"characters-{k}", model=JUGE, characters=FENETRE_JUGE)
        verdicts[k] = {"palier": fenetre, **v}
        _ecrire(CACHE / f"characters-verdicts-{SIGNATURE}.json", verdicts)
        print(f"  {numero:4d}/{len(plan)}  {k:16s} couverture {v.get('coverage')}", end="\r", flush=True)
    _ecrire(CACHE / f"characters-appels-{SIGNATURE}.json", llm.stats())
    print(f"\nfait : {len(verdicts)} verdicts, {llm.stats()}")


def _ic(deltas: list[float]) -> list[float] | None:
    if not deltas:
        return None
    alea = random.Random(GRAINE)
    t = sorted(statistics.fmean(alea.choice(deltas) for _ in deltas) for _ in range(TIRAGES))
    return [round(t[int(0.025 * TIRAGES)], 4), round(t[int(0.975 * TIRAGES)], 4)]


def verdict() -> dict:
    items, _ = population()
    verdicts = _charger(CACHE / f"characters-verdicts-{SIGNATURE}.json")
    reponses = _charger(CACHE / f"characters-reponses-{SIGNATURE}.json")
    temoins_lus = _charger(CACHE / f"characters-temoins-{SIGNATURE}.json")

    def couv(qid, f):
        v = verdicts.get(f"{qid}/{f}")
        return None if v is None else (v.get("coverage") or 0)

    def abst(qid, f):
        r = reponses.get(f"{qid}/{f}")
        return None if r is None else bool(r.get("abstained"))

    qids = [q for q in sorted(items) if couv(q, REFERENCE) is not None]
    paliers = {}
    for f in GRILLE:
        mesures = [q for q in qids if couv(q, f) is not None]
        deltas = [couv(q, f) - couv(q, REFERENCE) for q in mesures]
        nouvelles = [q for q in mesures if abst(q, f) and not abst(q, REFERENCE)]
        disparues = [q for q in mesures if abst(q, REFERENCE) and not abst(q, f)]
        a = len(nouvelles) - len(disparues)
        chutes = [q for q in mesures if couv(q, REFERENCE) == 2 and couv(q, f) == 0]
        paliers[f] = {
            "n": len(mesures),
            "couverture": round(statistics.fmean(couv(q, f) for q in mesures), 4) if mesures else None,
            "delta": round(statistics.fmean(deltas), 4) if deltas else None,
            "ic95": _ic(deltas) if f != REFERENCE else None,
            "abstentions": len([q for q in mesures if abst(q, f)]),
            "a": a, "nouvelles": nouvelles, "disparues": disparues,
            "garde_abstention": a <= GARDE_ABSTENTION,
            "b": len(chutes), "chutes": chutes,
            "garde_degradation": len(chutes) <= GARDE_DEGRADATION,
            "montent": sum(1 for d in deltas if d > 0), "descendent": sum(1 for d in deltas if d < 0),
        }
    eligibles = [f for f in GRILLE if f != REFERENCE
                 and paliers[f]["garde_abstention"] and paliers[f]["garde_degradation"]]
    gagnant = None
    if eligibles:
        gagnant = sorted(eligibles, key=lambda f: (-(paliers[f]["couverture"] or 0), f))[0]
    conclut = (gagnant is not None and paliers[gagnant]["ic95"] and paliers[gagnant]["ic95"][0] > 0)
    if not temoins_lus.get("lisible"):
        issue = "ILLISIBLE — les items-témoins du juge n'ont pas passé la condition du §4"
    elif conclut:
        issue = f"GO — characters = {gagnant}"
    elif gagnant is not None:
        issue = "NO-GO — aucun palier n'a de Δ dont la borne basse exclut zéro"
    else:
        meilleurs = [f for f in GRILLE if f != REFERENCE and paliers[f]["ic95"] and paliers[f]["ic95"][0] > 0]
        issue = ("HOLD — un palier gagne mais échoue une garde" if meilleurs
                 else "NO-GO — aucun palier ne tient les gardes")
    # point de bascule : le premier palier où l'abstention nette devient positive
    bascule = next((f for f in GRILLE if f != REFERENCE and paliers[f]["a"] > 0), None)
    out = {"signature": SIGNATURE, "generateur": GENERATEUR, "juge": JUGE,
           "fenetre_du_juge": FENETRE_JUGE, "population": len(qids),
           "pre_enregistrement": "PRE-ENREGISTREMENT-CHARACTERS-2026-09-08.md (a91c14f)",
           "temoins": {k: v for k, v in temoins_lus.items() if k != "detail"},
           "seuils": {"abstention": GARDE_ABSTENTION, "degradation": GARDE_DEGRADATION},
           "paliers": paliers, "eligibles": eligibles, "gagnant": gagnant,
           "point_de_bascule_abstention": bascule, "issue": issue,
           "appels": _charger(CACHE / f"characters-appels-{SIGNATURE}.json")}
    (HERE / f"results-characters-{SIGNATURE}.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    return out


def imprimer(r: dict) -> None:
    t = r["temoins"]
    print(f"\nsignature {r['signature']} · {r['generateur']} / {r['juge']} "
          f"(fenêtre du juge fixe à {r['fenetre_du_juge']}) · population {r['population']}")
    print(f"items-témoins du juge : exactitude {t.get('accuracy')} · "
          f"{FAMILLE_BLOQUANTE} {(t.get('per_family') or {}).get(FAMILLE_BLOQUANTE, {}).get('accuracy')} "
          f"-> {'LISIBLE' if t.get('lisible') else 'ILLISIBLE'}")
    print(f"\n{'palier':>7s} {'n':>4s} {'couv.':>7s} {'Δ':>8s} {'IC95':>20s} {'abst.':>6s} "
          f"{'a':>3s} {'garde':>6s} {'b':>3s} {'garde':>6s}  ↑/↓")
    for f in GRILLE:
        p = r["paliers"][f]
        ic = f"[{p['ic95'][0]:+.3f} ; {p['ic95'][1]:+.3f}]" if p["ic95"] else "(référence)"
        d = f"{p['delta']:+.4f}" if p["delta"] is not None else "  —"
        print(f"{f:7d} {p['n']:4d} {p['couverture']:7.4f} {d:>8s} {ic:>20s} {p['abstentions']:6d} "
              f"{p['a']:+3d} {'OK' if p['garde_abstention'] else 'ÉCHEC':>6s} "
              f"{p['b']:3d} {'OK' if p['garde_degradation'] else 'ÉCHEC':>6s}  "
              f"{p['montent']}/{p['descendent']}")
    print(f"\npoint de bascule de l'abstention : "
          f"{r['point_de_bascule_abstention'] or 'aucun palier ne fait monter l’abstention'}")
    print(f"paliers éligibles : {r['eligibles']}   gagnant : {r['gagnant']}")
    print(f"\nISSUE : {r['issue']}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--etape", choices=("temoins", "mesure", "verdict"), required=True)
    a = p.parse_args()
    if a.etape == "temoins":
        temoins()
    elif a.etape == "mesure":
        mesure()
    else:
        imprimer(verdict())


if __name__ == "__main__":
    main()
