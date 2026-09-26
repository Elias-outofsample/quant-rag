"""L'angle mort des gardes — les questions dont le contexte change sans que l'or bascule.

Instrument du pré-enregistrement ``PRE-ENREGISTREMENT-ANGLE-MORT-2026-09-08.md``, Partie I
publiée en ``3dfd93b`` **avant la première réponse générée**.

Sous ``gel1``, 114 des 155 questions voient leur contexte servi **changer** sans que
l'appartenance de l'or bouge. **Aucune garde du dépôt ne les compte.** Ce module en tire
30 au hasard, règle écrite d'avance, et mesure ce que le reclassement leur fait au niveau
**réponse**.

    .venv/bin/python rag/benchmark/eval_angle_mort.py --etape mesure
    .venv/bin/python rag/benchmark/eval_angle_mort.py --etape verdict
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
import eval_reclassement_selectif as ers  # noqa: E402
import eval_reranking  # noqa: E402
import experiment  # noqa: E402
import judge  # noqa: E402
import llm  # noqa: E402
import pipeline  # noqa: E402
from corpus import ChunkIndex  # noqa: E402

SIGNATURE = corpus_overlay.signature()
MODELE = "gemini-3.1-flash-lite"
VARIANTE = "gel1"
GRAINE, ECHANTILLON, TIRAGES = 20260908, 30, 10000
#: Les bornes du §4, déclarées avant la mesure.
BANDE, MOUVEMENTS_BENINS = 0.2, 6
CACHE = HERE / ".cache"


def _charger(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def _ecrire(p: Path, d: dict) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")


def contextes() -> tuple[dict, dict, list[str]]:
    """Les deux contextes par question, et la population de l'angle mort."""
    scores = json.loads(ers.SCORES.read_text(encoding="utf-8"))
    index = ChunkIndex.load(verbose=False)
    items = {i["key"]: i for i in experiment.load_items(index)}
    cache = experiment.cached_rankings()
    ctx, population, sans_faits = {}, [], []
    for cle, item in items.items():
        gold = set(item["gold_chunks"])
        rows = []
        for rang, (c, d, s) in enumerate(cache[cle]["dense"], 1):
            src = index.get(c) or {}
            fiche = index.metadata.get(d) or {}
            rows.append({"chunk_id": c, "document_id": d, "score": s, "rang_dense": rang,
                         "text": src.get("text", ""), "content_type": src.get("content_type"),
                         "section": src.get("section"), "page_start": src.get("page_start"),
                         "title": fiche.get("title") or index.title_of(d),
                         "short_ref": fiche.get("short_ref")})
        avant = eval_reranking.servis(rows)
        apres = eval_reranking.servis(ers.appliquer(rows, scores[cle], ers.VARIANTES[VARIANTE]))
        sa, sb = [r["chunk_id"] for r in avant], [r["chunk_id"] for r in apres]
        if bool(gold & set(sa)) != bool(gold & set(sb)) or sa == sb:
            continue                      # l'or bascule (les gardes le voient), ou rien ne change
        if not item.get("answer_facts"):
            sans_faits.append(cle)
            # Le juge note la couverture CONTRE ces faits : sans eux il ne peut pas noter. Les
            # questions du banc v1 n'en ont jamais eu. L'exclusion est instrumentale, ne dépend
            # d'aucun résultat, et elle est faite AVANT le tirage — sans quoi l'échantillon
            # serait amputé après coup. Le compte des exclues est publié.
            continue
        population.append(cle)
        ctx[cle] = {"dense": avant, VARIANTE: apres,
                    "meme_ensemble": set(sa) == set(sb)}
    globals()["_SANS_FAITS"] = sorted(sans_faits)
    return ctx, items, sorted(population)


def echantillon(population: list[str]) -> list[str]:
    tire = list(population)
    random.Random(GRAINE).shuffle(tire)
    return sorted(tire[:ECHANTILLON])


def mesure() -> None:
    ctx, items, population = contextes()
    choisies = echantillon(population)
    print(f"population de l'angle mort : {len(population)} · échantillon {len(choisies)}")
    _ecrire(CACHE / f"angle-mort-population-{SIGNATURE}.json",
            {"population": population, "echantillon": choisies,
             "exclues_sans_answer_facts": globals().get("_SANS_FAITS", [])})
    reponses = _charger(CACHE / f"angle-mort-reponses-{SIGNATURE}.json")
    verdicts = _charger(CACHE / f"angle-mort-verdicts-{SIGNATURE}.json")
    plan = [(c, b) for c in choisies for b in ("dense", VARIANTE)]
    for numero, (cle, bras) in enumerate(plan, 1):
        k = f"{cle}/{bras}"
        if k in verdicts:
            continue
        if k not in reponses:
            sortie = pipeline.answer(items[cle]["question"], ctx[cle][bras], model=MODELE)
            reponses[k] = {"bras": bras, "answer": sortie["answer"],
                           "abstained": sortie["abstained"],
                           "servis": [r["chunk_id"] for r in ctx[cle][bras]]}
            _ecrire(CACHE / f"angle-mort-reponses-{SIGNATURE}.json", reponses)
        v = judge.grade(items[cle], reponses[k]["answer"], ctx[cle][bras],
                        seed=f"angle-mort-{k}", model=MODELE)
        verdicts[k] = {"bras": bras, **v}
        _ecrire(CACHE / f"angle-mort-verdicts-{SIGNATURE}.json", verdicts)
        print(f"  {numero:3d}/{len(plan)}  {k:24s} couverture {v.get('coverage')}",
              end="\r", flush=True)
    _ecrire(CACHE / f"angle-mort-appels-{SIGNATURE}.json", llm.stats())
    print(f"\nfait : {len(verdicts)} verdicts, {llm.stats()}")


#: Juge de contrôle — la configuration de référence du dépôt.
JUGE_CONTROLE = "mistral-medium-latest"


def rejuger() -> None:
    """Re-note les **mêmes réponses** avec le juge de référence. Aucune génération.

    Même contrat que le contrôle du fil « fenêtre » : le verdict du §4 est rendu et publié, ce
    contrôle ne l'annule pas. Il dit seulement si « l'angle mort est bénin » résiste à un juge
    plus fort — et c'est la question qui compte, parce qu'un juge faible **sous-détecte** les
    différences, donc il est celui qui fait le plus facilement conclure « rien ne bouge ».
    """
    ctx, items, population = contextes()
    reponses = _charger(CACHE / f"angle-mort-reponses-{SIGNATURE}.json")
    verdicts = _charger(CACHE / f"angle-mort-verdicts-{JUGE_CONTROLE}-{SIGNATURE}.json")
    plan = sorted(reponses)
    for numero, k in enumerate(plan, 1):
        if k in verdicts:
            continue
        cle, bras = k.rsplit("/", 1)
        if cle not in ctx or not items[cle].get("answer_facts"):
            # Réponses écrites par le tout premier passage, avant que les questions du banc v1
            # ne soient écartées de la population : elles restent dans le cache et ne sont pas
            # notables — ni ici, ni dans le fil d'origine.
            continue
        v = judge.grade(items[cle], reponses[k]["answer"], ctx[cle][bras],
                        seed=f"angle-mort-{k}", model=JUGE_CONTROLE)
        verdicts[k] = {"bras": bras, **v}
        _ecrire(CACHE / f"angle-mort-verdicts-{JUGE_CONTROLE}-{SIGNATURE}.json", verdicts)
        print(f"  {numero:3d}/{len(plan)}  {k:24s} couverture {v.get('coverage')}", end="\r", flush=True)
    print(f"\nfait : {len(verdicts)} verdicts sous {JUGE_CONTROLE}, {llm.stats()}")


def verdict(juge: str | None = None) -> dict:
    pop = _charger(CACHE / f"angle-mort-population-{SIGNATURE}.json")
    suffixe = f"-{juge}" if juge else ""
    verdicts = _charger(CACHE / f"angle-mort-verdicts{suffixe}-{SIGNATURE}.json")
    reponses = _charger(CACHE / f"angle-mort-reponses-{SIGNATURE}.json")
    lignes = []
    for cle in pop.get("echantillon", []):
        a = verdicts.get(f"{cle}/dense"); b = verdicts.get(f"{cle}/{VARIANTE}")
        if a is None or b is None:
            continue
        ca, cb = a.get("coverage") or 0, b.get("coverage") or 0
        lignes.append({"cle": cle, "dense": ca, VARIANTE: cb, "delta": cb - ca,
                       "abstenue_dense": bool(reponses.get(f"{cle}/dense", {}).get("abstained")),
                       "abstenue_candidat": bool(reponses.get(f"{cle}/{VARIANTE}", {}).get("abstained"))})
    deltas = [l["delta"] for l in lignes]
    alea = random.Random(GRAINE)
    t = sorted(statistics.fmean(alea.choice(deltas) for _ in deltas) for _ in range(TIRAGES)) if deltas else []
    ic = [round(t[int(0.025 * TIRAGES)], 4), round(t[int(0.975 * TIRAGES)], 4)] if t else None
    moyen = round(statistics.fmean(deltas), 4) if deltas else None
    bougent = [l for l in lignes if l["delta"] != 0]
    montent = [l for l in bougent if l["delta"] > 0]
    descendent = [l for l in bougent if l["delta"] < 0]

    if moyen is None:
        issue = "INCOMPLET"
    elif ic and ic[1] < 0:
        issue = "LES GARDES SONT AVEUGLES À UN DOMMAGE RÉEL — le +11 de gel1 est surestimé"
    elif moyen < -BANDE:
        issue = "LES GARDES SONT AVEUGLES À UN DOMMAGE RÉEL — le +11 de gel1 est surestimé"
    elif moyen > BANDE and ic and ic[0] > 0:
        issue = "LES GARDES SONT AVEUGLES À UN GAIN RÉEL — gel1 vaut mieux que mesuré"
    elif abs(moyen) <= BANDE and len(bougent) <= MOUVEMENTS_BENINS:
        issue = "ANGLE MORT BÉNIN — le basculement de l'or résume bien l'effet"
    else:
        issue = "BRASSAGE SANS EFFET NET — l'angle mort est réel mais neutre en agrégat"

    out = {"signature": SIGNATURE, "variante": VARIANTE, "modele": MODELE,
           "pre_enregistrement": "PRE-ENREGISTREMENT-ANGLE-MORT-2026-09-08.md (3dfd93b)",
           "population": len(pop.get("population", [])), "echantillon": len(lignes),
           "delta_moyen": moyen, "ic95_diagnostic": ic,
           "bougent": len(bougent), "montent": len(montent), "descendent": len(descendent),
           "abstentions": {"dense": sum(1 for l in lignes if l["abstenue_dense"]),
                           "candidat": sum(1 for l in lignes if l["abstenue_candidat"])},
           "issue": issue, "detail": lignes,
           "appels_de_la_mesure": _charger(CACHE / f"angle-mort-appels-{SIGNATURE}.json")}
    out["juge"] = juge or MODELE
    (HERE / f"results-angle-mort{suffixe}-{SIGNATURE}.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    return out


def imprimer(r: dict) -> None:
    print(f"\nsignature {r['signature']} · {r['variante']} · population {r['population']} · "
          f"échantillon {r['echantillon']} · {r['pre_enregistrement']}")
    for l in r["detail"]:
        if l["delta"]:
            print(f"   {l['cle']:10s} couverture {l['dense']} → {l[r['variante']]}  ({l['delta']:+d})")
    print(f"\n   Δ moyen {r['delta_moyen']:+.4f}   IC95 diagnostic {r['ic95_diagnostic']}")
    print(f"   questions qui bougent : {r['bougent']}/{r['echantillon']} "
          f"({r['montent']} montent, {r['descendent']} descendent)")
    print(f"   abstentions : {r['abstentions']['dense']} → {r['abstentions']['candidat']}")
    print(f"\nISSUE : {r['issue']}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--etape", choices=("mesure", "rejuger", "verdict"), required=True)
    p.add_argument("--juge", default=None)
    a = p.parse_args()
    if a.etape == "mesure":
        mesure()
    elif a.etape == "rejuger":
        rejuger()
    else:
        imprimer(verdict(a.juge))


if __name__ == "__main__":
    main()
