"""Le reranker change-t-il la *réponse*, ou seulement le classement ?

Six chantiers de retrieval ont rendu NO-GO, tous arbitrés en ``nDCG@10`` avec une barre
héritée de l'arbitrage de la fusion. Le seul chantier conduit au niveau **réponse** a trouvé
un effet de +0,462 là où ceux du classement se disputent le troisième chiffre. **Le taux de
change entre les deux monnaies n'a jamais été mesuré**, et c'est ce que ce script mesure.

``Qwen3-Reranker-0.6B`` est le seul objet qui le rende mesurable : gain de classement validé
(+0,058 pooled), seule décision de production encore ouverte, et un mécanisme d'action sur la
réponse qui est **exactement** celui que le chantier couverture a mesuré — faire entrer, ou
sortir, le chunk d'or des cinq passages servis.

    RÉFÉRENCE   build_context(dense@50)            — la production actuelle
    RECLASSÉ    build_context(dense@50 reclassé)   — même pool, même taille, autre ordre

Aucun reclassement neuf : les scores viennent du cache de 20 075 paires de la Phase A. La
seule dépense est en appels LLM, et elle est ordonnée en trois paliers pour que la décision
soit complète avant la garde.

Protocole pré-enregistré et commité avant le premier appel (``122e11f``) :
``rag/benchmark/RAPPORT-RERANKER-REPONSE-2026-09-05.md``, Partie I.

    .venv/bin/python rag/benchmark/eval_reranker_reponse.py --step gardes    # palier 1
    .venv/bin/python rag/benchmark/eval_reranker_reponse.py --step mesure    # paliers 2 puis 3
    .venv/bin/python rag/benchmark/eval_reranker_reponse.py --step rapport
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
import experiment  # noqa: E402
import judge  # noqa: E402
import llm  # noqa: E402
import metrics  # noqa: E402
import pipeline  # noqa: E402
from corpus import ChunkIndex  # noqa: E402
from eval_pool_rerank import build_pools  # noqa: E402

SIGNATURE = corpus_overlay.signature()
CACHE = HERE / ".cache"
PARTIAL = CACHE / f"pool-rerank-partial-{SIGNATURE}.json"
OUTPUT = HERE / f"results-reranker-reponse-{SIGNATURE}.json"

GENERATEUR = "gemini-3.1-flash-lite"
#: Un seul juge sur tout le chantier — mélanger deux juges dans un contraste apparié est un
#: confondant, et le chantier couverture l'a payé. Il n'est pas plus fort que le générateur :
#: contrainte subie (Mistral à quota nul, ``gemini-3.5-flash`` à 20 requêtes/jour), et c'est
#: exactement ce que les items-témoins mesurent — d'où le palier 1.
JUGE = "gemini-3.1-flash-lite"
GRAINE = 20260901

#: Le banc v1 est *known-item* : 0 de ses 25 questions ne porte d'``answer_facts``, vérifié.
#: Le juge de couverture n'a rien à quoi comparer. Exclusion structurelle, déclarée au §2.
BANC = "v3"
TEMOIN_N = 60

#: Bornes pré-enregistrées (§4). ``BORNE_INSTRUMENT`` est la perte acceptable que n=60 permet
#: de faire respecter ; la borne économique, elle, se dérive du Δ décisif *mesuré*, et l'écart
#: entre les deux est le risque résiduel que le shadow doit fermer — pas ce chantier.
BORNE_INSTRUMENT = -0.15
#: Ancre directionnelle, pas un seuil : (24 − 12) × 0,462 / 36. Elle ne sert qu'à définir
#: « plat » — un effet dont la borne haute de l'IC est sous l'ancre est mesuré petit, pas
#: manqué faute de puissance.
ANCRE = 0.154


# --------------------------------------------------------------------------------- socle

def socle(index: ChunkIndex) -> tuple[dict, dict]:
    """Pour chaque question v3 notable : ses deux contextes et sa strate.

    La strate se lit sur **ce qui arrive au chunk d'or dans les cinq passages servis**, pas
    sur le rang ni sur le nDCG : c'est le mécanisme que le chantier couverture a mesuré, et
    le seul par lequel un reclassement peut changer une réponse.
    """
    if not PARTIAL.exists():
        sys.exit(f"cache de paires absent : {PARTIAL}")
    scores = json.loads(PARTIAL.read_text(encoding="utf-8"))
    items = experiment.load_items(index)
    pools_dense = experiment.cached_rankings()
    pools = build_pools(index, items)
    cadre, strates = {}, {"entrent": [], "sortent": [], "neutres": []}
    for item in items:
        key = item["key"]
        if item["bench"] != BANC or not item.get("answer_facts") or key not in scores:
            continue
        note = scores[key]
        reference = pipeline.build_context(
            pipeline._with_text(experiment.rows_from_cache(pools_dense[key]["dense"]), index))
        membres = pools[key]["membres"]["rr/dense50"]
        sous = [r for r in pools[key]["union"]
                if r["chunk_id"] in membres and r["text"] and r["chunk_id"] in note]
        reclasse = pipeline.build_context(
            sorted(sous, key=lambda r: -note[r["chunk_id"]]))
        gold = set(item["gold_chunks"])
        avant = bool(gold & {r["chunk_id"] for r in reference})
        apres = bool(gold & {r["chunk_id"] for r in reclasse})
        strate = "entrent" if (apres and not avant) else "sortent" if (avant and not apres) else "neutres"
        strates[strate].append(key)
        cadre[key] = {"item": item, "kind": item.get("kind", "single"), "strate": strate,
                      "reference": reference, "reclasse": reclasse,
                      "rate": not avant and not apres and not bool(gold & {r["chunk_id"] for r in
                                                                          experiment.rows_from_cache(pools_dense[key]["dense"])})}
    return cadre, {k: sorted(v) for k, v in strates.items()}


def temoin_neutre(strates: dict, cadre: dict) -> list[str]:
    """60 neutres, tirage reproductible, **reflétant la composition** de la population neutre.

    Un témoin de non-régression tiré au hasard sans stratification serait dominé par les
    questions faciles ou par les ratés selon le tirage — or les ratés ont une couverture de
    0,115 et les autres de 0,550. La proportion réelle est ce qui doit être respectée.
    """
    neutres = strates["neutres"]
    rates = [k for k in neutres if cadre[k]["rate"]]
    autres = [k for k in neutres if not cadre[k]["rate"]]
    rng = random.Random(GRAINE)
    quota = round(TEMOIN_N * len(rates) / len(neutres))
    return sorted(rng.sample(sorted(rates), quota) + rng.sample(sorted(autres), TEMOIN_N - quota))


def charger(nom: str) -> dict:
    path = CACHE / f"reranker-reponse-{nom}-{SIGNATURE}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def ecrire(nom: str, payload: dict) -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    (CACHE / f"reranker-reponse-{nom}-{SIGNATURE}.json").write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8")


# ------------------------------------------------------------------------------ contrôles

def controle_R(cadre: dict, cles: list[str]) -> tuple[list[str], dict, dict]:
    """Le bras RÉFÉRENCE reconstruit ici est-il celui qu'a servi le chantier couverture ?

    Sans cette égalité au chunk près, réutiliser son cache mélangerait deux contextes
    différents dans un contraste apparié. Les questions qui échouent au contrôle sont
    **recalculées**, pas réutilisées.
    """
    vieilles_r = json.loads((CACHE / f"answer-gap-reponses-{SIGNATURE}.json").read_text(encoding="utf-8"))
    vieilles_v = json.loads((CACHE / f"answer-gap-verdicts-{SIGNATURE}.json").read_text(encoding="utf-8"))
    ecarts, reponses, verdicts = [], {}, {}
    for key in cles:
        vieux = vieilles_r.get(f"{key}/servi")
        if vieux is None:
            continue
        attendu = [r.get("chunk_id") for r in cadre[key]["reference"]]
        if list(vieux["context"]) != attendu:
            ecarts.append(f"{key} · contexte reconstruit ≠ contexte servi par le §11")
            continue
        reponses[f"{key}/reference"] = dict(vieux)
        verdicts[f"{key}/reference"] = dict(vieilles_v[f"{key}/servi"])
    return ecarts, reponses, verdicts


# --------------------------------------------------------------------- palier 1 : gardes

def step_gardes(index: ChunkIndex) -> None:
    llm.GENERATOR, llm.JUDGE = GENERATEUR, JUGE
    judge.CORPUS = {"chunks": len(index.chunks)}
    items = [json.loads(l) for l in (HERE / "questions-v3.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    print(f"\n  items-témoins — {JUGE} note des réponses dont la note correcte est connue")
    resultat = judge.run_traps(judge.build_traps(items, index, seed=GRAINE))
    print(f"  exactitude globale : {resultat['accuracy']} sur {resultat['n']} témoins")
    for famille, bloc in resultat["per_family"].items():
        print(f"      {famille:<24} {bloc['accuracy']:>5.3f}  ({bloc['n']} items)")

    cadre, _ = socle(index)
    reponses = charger("reponses")
    echantillon = []
    for marque, bloc in sorted(reponses.items())[:6]:
        key, bras = marque.rsplit("/", 1)
        if key in cadre and bloc.get("answer"):
            echantillon.append((cadre[key]["item"], bloc["answer"], cadre[key][
                "reclasse" if bras == "reclasse" else "reference"]))
    print(f"\n  plancher de bruit — double notation de {len(echantillon)} réponses à T=0,3")
    bruit = judge.noise_floor(echantillon) if echantillon else {}
    print(f"  {json.dumps(bruit, ensure_ascii=False)}")
    ecrire("gardes", {"temoins": resultat, "bruit": bruit,
                      "generateur": GENERATEUR, "juge": JUGE})


# ------------------------------------------------------- paliers 2 et 3 : les deux bras

def step_mesure(index: ChunkIndex, limit: int | None, palier: str) -> None:
    llm.GENERATOR, llm.JUDGE = GENERATEUR, JUGE
    cadre, strates = socle(index)
    decisives = sorted(strates["entrent"] + strates["sortent"])
    temoin = temoin_neutre(strates, cadre)
    print(f"\n  corpus {len(index.chunks)} chunks · signature {SIGNATURE}")
    print(f"  ENTRENT {len(strates['entrent'])} · SORTENT {len(strates['sortent'])} "
          f"· NEUTRES {len(strates['neutres'])}   (banc {BANC}, v1 exclu faute d'answer_facts)")
    print(f"  contraste décisif : {len(decisives)} · témoin de non-régression : {len(temoin)}")

    reponses, verdicts = charger("reponses"), charger("verdicts")
    ecarts, reprises_r, reprises_v = controle_R(cadre, decisives + temoin)
    print(f"\n  contrôle R — bras RÉFÉRENCE réutilisable depuis le §11 : "
          f"{len(reprises_r)} questions, {len(ecarts)} écart(s)")
    for ligne in ecarts[:4]:
        print(f"      {ligne}")
    for marque, bloc in reprises_r.items():
        reponses.setdefault(marque, bloc)
    for marque, bloc in reprises_v.items():
        verdicts.setdefault(marque, bloc)
    ecrire("reponses", reponses)
    ecrire("verdicts", verdicts)

    travaux = []
    if palier in ("decisif", "tout"):
        travaux += [(k, b) for k in decisives for b in ("reference", "reclasse")]
    if palier in ("temoin", "tout"):
        travaux += [(k, b) for k in temoin for b in ("reference", "reclasse")]
    if limit:
        travaux = travaux[:limit]
    reste = [t for t in travaux if f"{t[0]}/{t[1]}" not in verdicts]
    print(f"\n  {len(reste)}/{len(travaux)} (question, bras) à produire — "
          f"générateur {GENERATEUR}, juge {JUGE}")

    echecs = 0
    for n, (key, bras) in enumerate(reste, 1):
        entry = cadre[key]
        contexte = entry["reclasse" if bras == "reclasse" else "reference"]
        marque = f"{key}/{bras}"
        try:
            if marque not in reponses:
                produit = pipeline.answer(entry["item"]["question"], contexte, model=GENERATEUR)
                reponses[marque] = {
                    "answer": produit["answer"], "abstained": produit["abstained"],
                    "context": [r.get("chunk_id") for r in contexte],
                    "gold_in_context": bool(set(entry["item"]["gold_chunks"]) &
                                            {r.get("chunk_id") for r in contexte})}
                ecrire("reponses", reponses)
            note = judge.grade(entry["item"], reponses[marque]["answer"], contexte,
                               seed=f"rr-{marque}", model=JUGE)
            verdicts[marque] = note
            ecrire("verdicts", verdicts)
            etat = "abstention" if reponses[marque]["abstained"] else "réponse"
            print(f"      {n}/{len(reste)}  {marque:<24} [{entry['strate'][:4]}] "
                  f"cov={note.get('coverage')} gnd={note.get('groundedness')}  {etat}", flush=True)
        except RuntimeError as erreur:
            echecs += 1
            print(f"      {n}/{len(reste)}  {marque:<24} ÉCHEC API : {str(erreur)[:70]}", flush=True)
            if echecs >= 5:
                print("\n  cinq échecs consécutifs — arrêt. Le cache est intact, relancer reprend.")
                break
    print(f"\n  {len(verdicts)} verdicts en cache")


# ------------------------------------------------------------------------------- rapport

def precis(reference: list[float], variante: list[float], draws: int = 4000) -> dict:
    """``metrics.paired_delta`` sans son arrondi au millième — les bornes sont au centième."""
    deltas = [v - r for r, v in zip(reference, variante)]
    rng = random.Random(GRAINE)
    moyennes = sorted(statistics.mean(rng.choices(deltas, k=len(deltas))) for _ in range(draws))
    return {"n": len(deltas), "delta": statistics.mean(deltas),
            "ic95": [moyennes[int(0.025 * draws)], moyennes[int(0.975 * draws) - 1]],
            "deltas": deltas}


def permutation(cadre: dict, verdicts: dict, decisives: list[str], temoin: list[str],
                observe: float, tirages: int = 2000) -> dict:
    """La stratification désigne-t-elle vraiment où le reclassement change la réponse ?

    On permute l'appartenance à la strate décisive sur l'ensemble des questions mesurées et
    on regarde où tombe le Δ observé. C'est la permutation, pas la rotation : le §12 a montré
    qu'une rotation sur un bloc contigu est inerte, et qu'un contrôle qu'on ne sait pas faire
    échouer ne contrôle rien. Aucun appel LLM — on permute des étiquettes sur des Δ mesurés.
    """
    mesurees = [k for k in decisives + temoin
                if f"{k}/reference" in verdicts and f"{k}/reclasse" in verdicts]
    deltas = {k: verdicts[f"{k}/reclasse"]["coverage"] - verdicts[f"{k}/reference"]["coverage"]
              for k in mesurees}
    rng = random.Random(GRAINE)
    tires = []
    for _ in range(tirages):
        echantillon = rng.sample(mesurees, len(decisives))
        tires.append(statistics.mean(deltas[k] for k in echantillon))
    tries = sorted(tires)
    return {"tirages": tirages, "n_pool": len(mesurees), "moyenne": statistics.mean(tires),
            "p05": tries[int(0.05 * tirages)], "p95": tries[int(0.95 * tirages)],
            "max": tries[-1], "min": tries[0],
            "p_value": sum(1 for x in tires if x >= observe) / tirages}


def step_rapport(index: ChunkIndex) -> None:
    cadre, strates = socle(index)
    decisives = sorted(strates["entrent"] + strates["sortent"])
    temoin = temoin_neutre(strates, cadre)
    verdicts, reponses = charger("verdicts"), charger("reponses")
    gardes = charger("gardes")

    def complet(cles):
        return [k for k in cles if f"{k}/reference" in verdicts and f"{k}/reclasse" in verdicts]

    dec, neu = complet(decisives), complet(temoin)
    print(f"\n  contraste décisif : {len(dec)}/{len(decisives)} paires complètes")
    print(f"  témoin neutre     : {len(neu)}/{len(temoin)} paires complètes")
    if not dec:
        sys.exit("aucune paire décisive complète : lance --step mesure")

    def couverture(cles, bras):
        return [verdicts[f"{k}/{bras}"]["coverage"] for k in cles]

    d_dec = precis(couverture(dec, "reference"), couverture(dec, "reclasse"))
    d_neu = precis(couverture(neu, "reference"), couverture(neu, "reclasse")) if neu else None

    print(f"\n  === contraste décisif — {len(dec)} questions ===")
    print(f"    RÉFÉRENCE {statistics.mean(couverture(dec, 'reference')):.3f}  →  "
          f"RECLASSÉ {statistics.mean(couverture(dec, 'reclasse')):.3f}")
    print(f"    Δ = {d_dec['delta']:+.3f}  IC95 [{d_dec['ic95'][0]:+.3f} ; {d_dec['ic95'][1]:+.3f}]"
          f"   (ancre directionnelle {ANCRE:+.3f})")
    for strate in ("entrent", "sortent"):
        cles = [k for k in dec if cadre[k]["strate"] == strate]
        if cles:
            sous = precis(couverture(cles, "reference"), couverture(cles, "reclasse"))
            print(f"      {strate:<8} n={len(cles):<3} Δ = {sous['delta']:+.3f}  "
                  f"IC95 [{sous['ic95'][0]:+.3f} ; {sous['ic95'][1]:+.3f}]")

    borne_eco = -(len(dec) / max(len(strates["neutres"]), 1)) * d_dec["delta"]
    if d_neu:
        print(f"\n  === témoin de non-régression — {len(neu)} questions ===")
        print(f"    Δ = {d_neu['delta']:+.3f}  IC95 [{d_neu['ic95'][0]:+.3f} ; {d_neu['ic95'][1]:+.3f}]")
        print(f"    borne d'instrument {BORNE_INSTRUMENT:+.3f} · borne économique dérivée {borne_eco:+.3f}")

    # Effet net sur la population v3 entière, pondéré par les tailles de strate — statistique
    # dérivée de deux contrastes pré-enregistrés avec des poids pré-enregistrés (les strates
    # sont fixées au §2), mais elle-même absente du pré-enregistrement : déclarée comme telle.
    net = None
    if d_neu:
        poids = (len(strates["entrent"]) + len(strates["sortent"])), len(strates["neutres"])
        total = sum(poids)
        rng = random.Random(GRAINE)
        tires = sorted(
            poids[0] / total * statistics.mean(rng.choices(d_dec["deltas"], k=len(d_dec["deltas"])))
            + poids[1] / total * statistics.mean(rng.choices(d_neu["deltas"], k=len(d_neu["deltas"])))
            for _ in range(4000))
        net = {"delta": poids[0] / total * d_dec["delta"] + poids[1] / total * d_neu["delta"],
               "ic95": [tires[100], tires[3899]],
               "part_positive": sum(1 for x in tires if x > 0) / 4000,
               "poids": {"decisives": poids[0], "neutres": poids[1]}}
        print(f"\n  === effet net sur les {total} questions v3, pondéré {poids[0]}/{total} et {poids[1]}/{total} ===")
        print(f"    Δ = {net['delta']:+.3f}  IC95 [{net['ic95'][0]:+.3f} ; {net['ic95'][1]:+.3f}]"
              f"  ·  {net['part_positive']:.1%} des tirages positifs")

    perm = permutation(cadre, verdicts, dec, neu, d_dec["delta"])
    print(f"\n  === sabotage par permutation — {perm['tirages']} tirages sur {perm['n_pool']} questions ===")
    print(f"    {len(dec)} questions quelconques : moyenne {perm['moyenne']:+.3f} · "
          f"p95 {perm['p95']:+.3f} · max {perm['max']:+.3f}")
    print(f"    p = {perm['p_value']:.3f}")

    bas, haut = d_dec["ic95"]
    if haut < 0:
        issue = "RÉGRESSION"
    elif bas > 0:
        # Le pré-enregistrement exigeait les DEUX conditions pour un GO. Il n'avait pas
        # nommé le cas « le contraste passe, la garde échoue » : c'est un HOLD, et l'écrire
        # « GO retenu » serait habiller un HOLD en GO. Lacune du §4(c), déclarée au rapport.
        issue = "GO" if (not d_neu or d_neu["ic95"][0] > BORNE_INSTRUMENT) else "HOLD — contraste acquis, garde non franchie"
    elif haut < ANCRE:
        issue = "PLAT"
    else:
        issue = "NON CONCLUANT"
    print(f"\n  === ISSUE : {issue} ===")
    if d_neu and d_neu["ic95"][0] < borne_eco:
        print(f"    risque résiduel : la borne basse du témoin ({d_neu['ic95'][0]:+.3f}) est sous "
              f"la borne économique ({borne_eco:+.3f}) — le net pooled n'est pas garanti positif.")

    OUTPUT.write_text(json.dumps({
        "corpus": {"signature": SIGNATURE, "chunks": len(index.chunks)},
        "protocole": {"pre_enregistrement": "122e11f", "banc": BANC,
                      "generateur": GENERATEUR, "juge": JUGE, "graine": GRAINE,
                      "ancre_directionnelle": ANCRE, "borne_instrument": BORNE_INSTRUMENT,
                      "bootstrap": {"draws": 4000, "seed": GRAINE}},
        "strates": strates, "temoin_neutre": temoin,
        "decisif": {k: v for k, v in d_dec.items() if k != "deltas"},
        "decisif_par_strate": {s: {k: v for k, v in precis(
            couverture([x for x in dec if cadre[x]["strate"] == s], "reference"),
            couverture([x for x in dec if cadre[x]["strate"] == s], "reclasse")).items()
            if k != "deltas"} for s in ("entrent", "sortent")
            if [x for x in dec if cadre[x]["strate"] == s]},
        "temoin": {k: v for k, v in d_neu.items() if k != "deltas"} if d_neu else None,
        "borne_economique": borne_eco, "permutation": perm, "gardes": gardes,
        "net_pondere": net,
        "abstention": {bras: round(sum(1 for k in dec if reponses[f"{k}/{bras}"]["abstained"])
                                   / len(dec), 3) for bras in ("reference", "reclasse")},
        "issue": issue,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  écrit : {OUTPUT.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step", required=True, choices=("gardes", "mesure", "rapport"))
    parser.add_argument("--palier", default="tout", choices=("decisif", "temoin", "tout"))
    parser.add_argument("--limit", type=int, help="pilote : n premiers travaux")
    args = parser.parse_args()
    index = ChunkIndex.load()
    if args.step == "gardes":
        step_gardes(index)
    elif args.step == "mesure":
        step_mesure(index, args.limit, args.palier)
    else:
        step_rapport(index)


if __name__ == "__main__":
    main()
