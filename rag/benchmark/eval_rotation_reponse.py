"""La rotation de `gel1` au niveau **réponse** — recensement, pas test.

Instrument du pré-enregistrement ``PRE-ENREGISTREMENT-ROTATION-REPONSE-2026-09-08.md``,
Partie I publiée en ``cb1e159`` **avant la première réponse générée de ce fil**.

**24 questions nommées, closes, et antérieures à ce fil** : les 5 que `gel1` casse au classement
et les 19 qu'il répare, telles que le recensement nominatif les a fixées en ``2538a97``. Aucune
n'est ajoutée ni retirée.

Deux bras, une seule variable — le **classement** :

===========  ==========================================================
``dense``    l'ordre dense, ce que la production sert par défaut
``gel1``     ``eval_reclassement_selectif.appliquer(..., ordre_gel)``
===========  ==========================================================

La sélection est celle de **production** (``eval_reranking.servis`` → ``quant_rag._select`` :
5 passages, 2 par document, 250 caractères, filtre d'en-têtes, quasi-doublons). ``characters``
reste à **1 600**, le défaut : ce fil ne touche pas à la fenêtre du générateur, qui est l'objet
d'un autre fil et **ne doit pas être mélangé** avec celui-ci.

**Aucune borne de confiance ne décide ici**, et c'est le §5 du pré-enregistrement : le §21 ter
a établi qu'à n = 24 aucune ne peut discriminer, et en exiger une referait l'erreur qu'il vient
de nommer. Les intervalles sont publiés en diagnostic.

    .venv/bin/python rag/benchmark/eval_rotation_reponse.py --etape mesure
    .venv/bin/python rag/benchmark/eval_rotation_reponse.py --etape verdict
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
BRAS = ("dense", VARIANTE)
GRAINE, TIRAGES = 20260908, 10000
#: Le taux de conversion emprunté par la garde du §21 ter, et l'écart au-delà duquel son
#: plafond est à réviser — issue nommée d'avance au §5 du pré-enregistrement.
P_GAIN_EMPRUNTE, ECART_REVISION = 9 / 17, 0.15
CACHE = HERE / ".cache"


def _charger(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def _ecrire(p: Path, d: dict) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")


def population() -> dict:
    r = json.loads((HERE / f"recensement-rotation-{VARIANTE}-{SIGNATURE}.json").read_text(encoding="utf-8"))
    assert r["signature"] == SIGNATURE and r["variante"] == VARIANTE
    return {"cassees": [f["cle"] for f in r["perdues"]],
            "reparees": [f["cle"] for f in r["gagnees"]]}


def contextes() -> tuple[dict, dict]:
    """Les deux contextes servis par question — dérivés une fois, jamais recalculés."""
    scores = json.loads(ers.SCORES.read_text(encoding="utf-8"))
    index = ChunkIndex.load(verbose=False)
    items = {i["key"]: i for i in experiment.load_items(index)}
    cache = experiment.cached_rankings()
    pop = population()
    out, fiches = {}, {}
    for cle in pop["cassees"] + pop["reparees"]:
        item = items[cle]
        rows = []
        for rang, (chunk_id, document_id, score) in enumerate(cache[cle]["dense"], 1):
            src = index.get(chunk_id) or {}
            fiche = index.metadata.get(document_id) or {}
            rows.append({"chunk_id": chunk_id, "document_id": document_id, "score": score,
                         "rang_dense": rang, "text": src.get("text", ""),
                         "content_type": src.get("content_type"), "section": src.get("section"),
                         "title": fiche.get("title") or index.title_of(document_id),
                         "short_ref": fiche.get("short_ref"), "page_start": src.get("page_start")})
        out[cle] = {"dense": eval_reranking.servis(rows),
                    VARIANTE: eval_reranking.servis(ers.appliquer(rows, scores[cle],
                                                                  ers.VARIANTES[VARIANTE]))}
        fiches[cle] = item
    return out, fiches


#: Amendement du 8 septembre 2026 — les trois pertes restantes de l'union complète des huit
#: variantes publiées, avec la variante de plus petit budget qui les perd. Population close.
EXTENSION = {"v3/d13": "b10", "v3/d14": "b10", "v3/m19": "rrf"}


def extension() -> None:
    """Les trois pertes restantes, sous `dense` et sous la variante qui les perd. 12 appels."""
    scores = json.loads(ers.SCORES.read_text(encoding="utf-8"))
    index = ChunkIndex.load(verbose=False)
    items = {i["key"]: i for i in experiment.load_items(index)}
    cache = experiment.cached_rankings()
    reponses = _charger(CACHE / f"rotation-reponse-{SIGNATURE}.json")
    verdicts = _charger(CACHE / f"rotation-verdicts-{SIGNATURE}.json")
    regles = {"b10": ers.ordre_b10, **ers.VARIANTES}
    for cle, variante in EXTENSION.items():
        item = items[cle]
        rows = []
        for rang, (chunk_id, document_id, score) in enumerate(cache[cle]["dense"], 1):
            src = index.get(chunk_id) or {}
            fiche = index.metadata.get(document_id) or {}
            rows.append({"chunk_id": chunk_id, "document_id": document_id, "score": score,
                         "rang_dense": rang, "text": src.get("text", ""),
                         "content_type": src.get("content_type"), "section": src.get("section"),
                         "title": fiche.get("title") or index.title_of(document_id),
                         "short_ref": fiche.get("short_ref"), "page_start": src.get("page_start")})
        ctx = {"dense": eval_reranking.servis(rows),
               variante: eval_reranking.servis(ers.appliquer(rows, scores[cle], regles[variante]))}
        for bras in ("dense", variante):
            k = f"{cle}/{bras}"
            if k in verdicts:
                continue
            if k not in reponses:
                sortie = pipeline.answer(item["question"], ctx[bras], model=MODELE)
                reponses[k] = {"strate": "extension", "bras": bras, "answer": sortie["answer"],
                               "abstained": sortie["abstained"],
                               "servis": [r["chunk_id"] for r in ctx[bras]]}
                _ecrire(CACHE / f"rotation-reponse-{SIGNATURE}.json", reponses)
            v = judge.grade(item, reponses[k]["answer"], ctx[bras], seed=f"rotation-{k}", model=MODELE)
            verdicts[k] = {"strate": "extension", "bras": bras, **v}
            _ecrire(CACHE / f"rotation-verdicts-{SIGNATURE}.json", verdicts)
            print(f"  {k:20s} couverture {v.get('coverage')}")
    _ecrire(CACHE / f"rotation-appels-extension-{SIGNATURE}.json", llm.stats())
    print(f"fait — {llm.stats()}")


def mesure() -> None:
    pop = population()
    ctx, items = contextes()
    reponses = _charger(CACHE / f"rotation-reponse-{SIGNATURE}.json")
    verdicts = _charger(CACHE / f"rotation-verdicts-{SIGNATURE}.json")
    # Les questions du banc v1 n'ont pas d'``answer_facts`` : le juge note la couverture
    # CONTRE ces faits, il ne peut donc pas les noter. L'exclusion est purement instrumentale
    # — elle ne dépend d'aucun résultat, et ces questions n'ont jamais eu de faits — mais elle
    # rétrécit les strates, et le §5 du pré-enregistrement l'anticipe par son issue INCOMPLET.
    # Elle est déclarée, comptée, et une sensibilité la compense au §verdict.
    sans_faits = [c for c in pop["cassees"] + pop["reparees"] if not items[c].get("answer_facts")]
    plan = [(s, c, b) for s, cles in (("cassees", pop["cassees"]), ("reparees", pop["reparees"]))
            for c in cles if c not in sans_faits for b in BRAS]
    if sans_faits:
        print(f"non mesurables (pas d'answer_facts, banc v1) : {sans_faits}")
        _ecrire(CACHE / f"rotation-non-mesurables-{SIGNATURE}.json", {"cles": sans_faits})
    print(f"{len(plan)} couples (question, bras) — {MODELE}")
    for numero, (strate, cle, bras) in enumerate(plan, 1):
        k = f"{cle}/{bras}"
        if k in verdicts:
            continue
        if k not in reponses:
            sortie = pipeline.answer(items[cle]["question"], ctx[cle][bras], model=MODELE)
            reponses[k] = {"strate": strate, "bras": bras, "answer": sortie["answer"],
                           "abstained": sortie["abstained"],
                           "servis": [r["chunk_id"] for r in ctx[cle][bras]]}
            _ecrire(CACHE / f"rotation-reponse-{SIGNATURE}.json", reponses)
        v = judge.grade(items[cle], reponses[k]["answer"], ctx[cle][bras],
                        seed=f"rotation-{k}", model=MODELE)
        verdicts[k] = {"strate": strate, "bras": bras, **v}
        _ecrire(CACHE / f"rotation-verdicts-{SIGNATURE}.json", verdicts)
        print(f"  {numero:3d}/{len(plan)}  {k:20s} {strate:9s} couverture {v.get('coverage')}",
              end="\r", flush=True)
    # Même piège que le fil de la fenêtre : llm.stats() compte le processus courant, et
    # l'étape « verdict » en est un autre. On fige les compteurs de la mesure ici.
    _ecrire(CACHE / f"rotation-appels-{SIGNATURE}.json", llm.stats())
    print(f"\nfait : {len(verdicts)} verdicts, {llm.stats()}")


#: Juge de contrôle — la configuration de référence du dépôt, rendue possible le 8 septembre
#: 2026 par une clé Mistral au quota ouvert. Le fil d'origine a dû faire juger
#: ``gemini-3.1-flash-lite`` par lui-même, faute de second fournisseur.
JUGE_CONTROLE = "mistral-medium-latest"


def rejuger() -> None:
    """Re-note les **mêmes réponses** avec un juge plus fort. Aucune génération.

    **Ce que ce contrôle peut et ne peut pas faire.** Il ne mesure pas un nouveau bras : les
    réponses sont celles déjà générées, prises au cache, et seul le juge change. Il répond donc
    à une question et une seule : *la conclusion du §21 octies tient-elle si on la fait noter
    par le juge de référence du dépôt plutôt que par le générateur lui-même ?*

    **Déclaré avant de lancer** : la conclusion est révisée si ``p_perte`` dépasse **0,5** sous
    ce juge — c'est le seuil de l'amendement, et c'est le point où « perdre un or servi n'est
    majoritairement pas perdre une réponse » cesse d'être vrai. Les verdicts d'origine ne sont
    pas écrasés : ils vivent dans leur fichier, celui-ci est écrit à côté.
    """
    reponses = _charger(CACHE / f"rotation-reponse-{SIGNATURE}.json")
    index = ChunkIndex.load(verbose=False)
    items = {i["key"]: i for i in experiment.load_items(index)}
    ctx, _ = contextes()
    supplement, _ = {}, None
    for cle, variante in EXTENSION.items():
        pass
    verdicts = _charger(CACHE / f"rotation-verdicts-{JUGE_CONTROLE}-{SIGNATURE}.json")
    plan = sorted(reponses)
    for numero, k in enumerate(plan, 1):
        if k in verdicts:
            continue
        cle, bras = k.rsplit("/", 1)
        if not items[cle].get("answer_facts"):
            # v1/q13 : sa réponse a été écrite par le tout premier passage avant que le juge
            # ne lève sur l'absence d'``answer_facts``. Elle reste dans le cache et n'est pas
            # notable — ni ici, ni dans le fil d'origine.
            continue
        contexte = (ctx.get(cle) or {}).get(bras)
        if contexte is None:
            # bras de l'extension (b10 / rrf) : le contexte n'est pas dans `contextes()`
            servis = reponses[k].get("servis") or []
            contexte = [{"chunk_id": c, "document_id": (index.get(c) or {}).get("document_id"),
                         "text": (index.get(c) or {}).get("text", ""),
                         "section": (index.get(c) or {}).get("section"),
                         "title": index.title_of((index.get(c) or {}).get("document_id") or "")}
                        for c in servis if index.get(c)]
        v = judge.grade(items[cle], reponses[k]["answer"], contexte,
                        seed=f"rotation-{k}", model=JUGE_CONTROLE)
        verdicts[k] = {"strate": reponses[k].get("strate"), "bras": bras, **v}
        _ecrire(CACHE / f"rotation-verdicts-{JUGE_CONTROLE}-{SIGNATURE}.json", verdicts)
        print(f"  {numero:3d}/{len(plan)}  {k:20s} couverture {v.get('coverage')}", end="\r", flush=True)
    print(f"\nfait : {len(verdicts)} verdicts sous {JUGE_CONTROLE}, {llm.stats()}")
    _ecrire(CACHE / f"rotation-appels-{JUGE_CONTROLE}-{SIGNATURE}.json", llm.stats())


def _ic(valeurs: list[float]) -> list[float] | None:
    if not valeurs:
        return None
    alea = random.Random(GRAINE)
    t = sorted(statistics.fmean(alea.choice(valeurs) for _ in valeurs) for _ in range(TIRAGES))
    return [round(t[int(0.025 * TIRAGES)], 4), round(t[int(0.975 * TIRAGES)], 4)]


def verdict() -> dict:
    pop = population()
    verdicts = _charger(CACHE / f"rotation-verdicts-{SIGNATURE}.json")
    reponses = _charger(CACHE / f"rotation-reponse-{SIGNATURE}.json")

    def couv(cle, bras):
        v = verdicts.get(f"{cle}/{bras}")
        return None if v is None else (v.get("coverage") or 0)

    non_mesurables = _charger(CACHE / f"rotation-non-mesurables-{SIGNATURE}.json").get("cles", [])
    detail = {"cassees": [], "reparees": []}
    for strate, cles in (("cassees", pop["cassees"]), ("reparees", pop["reparees"])):
        for cle in cles:
            a, b = couv(cle, "dense"), couv(cle, VARIANTE)
            if a is None or b is None:
                continue
            detail[strate].append({"cle": cle, "dense": a, VARIANTE: b, "delta": b - a,
                                   "abstenue_dense": bool(reponses.get(f"{cle}/dense", {}).get("abstained")),
                                   "abstenue_candidat": bool(reponses.get(f"{cle}/{VARIANTE}", {}).get("abstained"))})

    cassees, reparees = detail["cassees"], detail["reparees"]
    perdues_reelles = [d for d in cassees if d["dense"] > 0 and d[VARIANTE] == 0]
    gagnees_reelles = [d for d in reparees if d["dense"] == 0 and d[VARIANTE] > 0]
    p_perte = len(perdues_reelles) / len(cassees) if cassees else None
    p_gain = len(gagnees_reelles) / len(reparees) if reparees else None
    mesurables = len(cassees) + len(reparees)

    echange = (p_gain * len(reparees) - p_perte * len(cassees)
               if p_gain is not None and p_perte is not None else None)
    tau_prime = p_gain / (1 + p_gain) if p_gain else None
    part_destructrice = len(pop["cassees"]) / (len(pop["cassees"]) + len(pop["reparees"]))

    if mesurables < 20:
        issue = "INCOMPLET — moins de 20 des 24 questions mesurables"
    elif echange is not None and echange <= 0:
        issue = f"L'ÉCHANGE DE {VARIANTE} EST NÉGATIF EN RÉPONSE — il est à retirer de la production"
    else:
        issue = f"L'ÉCHANGE DE {VARIANTE} EST FAVORABLE EN RÉPONSE"
    revision = (p_gain is not None and abs(p_gain - P_GAIN_EMPRUNTE) > ECART_REVISION)

    # Sensibilité déclarée : les non mesurables comptés au PIRE pour le candidat — chaque
    # cassée non mesurable comptée comme une perte réelle, chaque réparée comme un gain nul.
    n_cass_nm = sum(1 for c in non_mesurables if c in pop["cassees"])
    n_rep_nm = sum(1 for c in non_mesurables if c in pop["reparees"])
    p_perte_pire = ((len(perdues_reelles) + n_cass_nm) / (len(cassees) + n_cass_nm)
                    if cassees or n_cass_nm else None)
    p_gain_pire = (len(gagnees_reelles) / (len(reparees) + n_rep_nm)
                   if reparees or n_rep_nm else None)
    echange_pire = (p_gain_pire * (len(reparees) + n_rep_nm) - p_perte_pire * (len(cassees) + n_cass_nm)
                    if p_gain_pire is not None and p_perte_pire is not None else None)

    rapport = {
        "signature": SIGNATURE, "modele": MODELE, "variante": VARIANTE,
        "non_mesurables": {"cles": non_mesurables, "cause": "banc v1, pas d'answer_facts",
                           "cassees": n_cass_nm, "reparees": n_rep_nm},
        "sensibilite_au_pire": {"p_perte": round(p_perte_pire, 4) if p_perte_pire is not None else None,
                                "p_gain": round(p_gain_pire, 4) if p_gain_pire is not None else None,
                                "echange": round(echange_pire, 2) if echange_pire is not None else None},
        "pre_enregistrement": "PRE-ENREGISTREMENT-ROTATION-REPONSE-2026-09-08.md (cb1e159)",
        "n": {"cassees": len(cassees), "reparees": len(reparees), "mesurables": mesurables},
        "p_perte": round(p_perte, 4) if p_perte is not None else None,
        "p_gain": round(p_gain, 4) if p_gain is not None else None,
        "p_gain_emprunte_par_la_garde": round(P_GAIN_EMPRUNTE, 4),
        "revision_du_plafond_requise": revision,
        "tau_implique": round(tau_prime, 4) if tau_prime else None,
        "part_destructrice_observee": round(part_destructrice, 4),
        "echange_net_en_reponses": round(echange, 2) if echange is not None else None,
        "delta_moyen": {
            "cassees": round(statistics.fmean(d["delta"] for d in cassees), 4) if cassees else None,
            "reparees": round(statistics.fmean(d["delta"] for d in reparees), 4) if reparees else None,
        },
        "ic95_diagnostic": {
            "cassees": _ic([d["delta"] for d in cassees]),
            "reparees": _ic([d["delta"] for d in reparees]),
        },
        "issue": issue, "detail": detail,
        "appels_de_la_mesure": _charger(CACHE / f"rotation-appels-{SIGNATURE}.json"),
    }
    (HERE / f"results-rotation-reponse-{SIGNATURE}.json").write_text(
        json.dumps(rapport, ensure_ascii=False, indent=1), encoding="utf-8")
    return rapport


def imprimer(r: dict) -> None:
    print(f"\nsignature {r['signature']} · {r['variante']} · {r['modele']} · {r['pre_enregistrement']}")
    for strate, titre in (("cassees", "CASSÉES au classement"), ("reparees", "RÉPARÉES au classement")):
        print(f"\n{titre}  (n = {r['n'][strate]})")
        for d in r["detail"][strate]:
            print(f"   {d['cle']:10s} couverture {d['dense']} → {d[r['variante']]}  ({d['delta']:+d})"
                  + ("   abstention nouvelle" if d["abstenue_candidat"] and not d["abstenue_dense"] else ""))
        print(f"   Δ moyen {r['delta_moyen'][strate]:+.3f}   IC95 diagnostic {r['ic95_diagnostic'][strate]}")
    print(f"\n  p_perte = {r['p_perte']}   (part des cassées qui perdent VRAIMENT leur réponse)")
    print(f"  p_gain  = {r['p_gain']}   (part des réparées qui gagnent VRAIMENT une réponse)")
    print(f"  emprunté par la garde du §21 ter : {r['p_gain_emprunte_par_la_garde']}"
          f"   → révision du plafond requise : {r['revision_du_plafond_requise']}")
    print(f"  échange net en réponses : {r['echange_net_en_reponses']:+}")
    nm = r["non_mesurables"]
    if nm["cles"]:
        sp = r["sensibilite_au_pire"]
        print(f"\n  non mesurables ({nm['cause']}) : {nm['cles']}")
        print(f"  SENSIBILITÉ, comptés au pire pour le candidat : p_perte {sp['p_perte']}, "
              f"p_gain {sp['p_gain']}, échange {sp['echange']:+}")
    print(f"\nISSUE : {r['issue']}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--etape", choices=("mesure", "extension", "rejuger", "verdict"), required=True)
    a = p.parse_args()
    if a.etape == "mesure":
        mesure()
    elif a.etape == "extension":
        extension()
    elif a.etape == "rejuger":
        rejuger()
    else:
        imprimer(verdict())


if __name__ == "__main__":
    main()
