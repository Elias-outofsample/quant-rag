"""Le chunk d'or manquant coûte-t-il une réponse ? — contraste apparié sur les 27 ratés.

Question posée. Le chantier « rappel du pool » a mesuré que le pool dense@50 rate
**27 questions sur 155**, et que l'élargir ne paie pas. Mais « raté » y est une métrique
d'**identité de chunk** : l'or est *un* chunk précis, fixé à la génération de la question.
Or pour 18 des 27 le bon *document* est dans le pool, et ``metrics.py`` crédite déjà le
chunk voisin à part. Personne n'a jamais mesuré si ces 27 produisent de mauvaises
**réponses** — et c'est sur elles que repose le chantier « découpage », le plus cher et le
seul irréversible.

Le dispositif. Sur **les mêmes 27 questions**, deux conditions qui ne diffèrent que par la
présence du chunk d'or :

    SERVI    build_context(pool dense@50) — exactement ce que la production affiche
    ORACLE   le ou les chunks d'or en tête, puis les mêmes passages, tronqué à 5

Comparer les 27 aux 128 autres confondrait « le retrieval a échoué » et « ces questions sont
difficiles » — c'est précisément parce qu'elles sont difficiles qu'elles sont ratées.
L'appariement supprime ce confondant, et il en supprime un second : le juge disponible est
Gemini, connu pour être **sévère au crédit** (12/12 pour condamner, 1/7 pour créditer).
Sa sévérité s'applique aux deux bras et s'annule largement dans la différence. **Toute
couverture absolue rapportée ici est un plancher, jamais une estimation.**

Protocole pré-enregistré et commité avant le premier appel LLM :
``rag/benchmark/RAPPORT-COUVERTURE-DES-RATES-2026-09-05.md``, Partie I.

    .venv/bin/python rag/benchmark/eval_answer_gap.py --step mesure --limit 3   # pilote
    .venv/bin/python rag/benchmark/eval_answer_gap.py --step mesure             # ~30 min
    .venv/bin/python rag/benchmark/eval_answer_gap.py --step gardes             # témoins + bruit
    .venv/bin/python rag/benchmark/eval_answer_gap.py --step rapport
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from collections import Counter, defaultdict
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

SIGNATURE = corpus_overlay.signature()
CACHE = HERE / ".cache"
OUTPUT = HERE / f"results-answer-gap-{SIGNATURE}.json"

#: Mistral est à quota de compte nul — pas une limite passagère. Générateur le moins coûteux
#: en jetons, juge plus fort que lui : la règle de la maison (``llm.JUDGE`` > ``llm.GENERATOR``).
GENERATEUR = "gemini-3.1-flash-lite"

#: Le juge devrait être plus fort que le générateur (règle de la maison). Il ne l'est pas ici,
#: et c'est une contrainte subie, pas un choix : ``gemini-3.5-flash`` porte un plafond gratuit
#: de **20 requêtes par jour** — épuisé après 16 notes, ce qui a fait échouer la première
#: course en HTTP 429. Mistral est à quota de compte nul. Le seul modèle dont le quota tienne
#: la distance est celui du générateur.
#:
#: Ce que cela coûte, et ce que cela ne coûte pas. Le contraste reste apparié : le **même**
#: juge note les deux bras de chaque paire, donc sa sévérité s'annule dans la différence.
#: Ce qui est perdu est la garantie « juge plus fort que générateur » — et c'est précisément
#: ce que les items-témoins mesurent. Ils sont donc exécutés **avant** la mesure, et si ce
#: juge les rate, la mesure ne vaut rien et le rapport doit le dire.
JUGE = "gemini-3.1-flash-lite"

#: Témoin : 40 questions parmi les 128 non ratées, stratifiées par famille. Il n'ancre qu'un
#: *niveau*, et ce niveau est un plancher. Il ne décide de rien.
TEMOIN_N = 40
GRAINE = 20260901

#: Seuils pré-enregistrés. La couverture est notée 0/1/2 : Δ = 0,25, c'est une question sur
#: quatre qui gagne un grade entier. En deçà, à n=27 et avec un juge dont le plancher de
#: bruit tourne autour de 0,1, l'effet n'est pas séparable de l'instrument.
SEUIL_DELTA = 0.25


# --------------------------------------------------------------------------- socle

def frame(index: ChunkIndex) -> dict:
    """Les 155 questions, leur pool dense@50, et le jeu des 27 ratés avec ses deux groupes."""
    pools = experiment.cached_rankings()
    out = {}
    for item in experiment.load_items(index):
        key = item["key"]
        rows = experiment.rows_from_cache(pools[key]["dense"])
        chunks = {r["chunk_id"] for r in rows}
        documents = {r["document_id"] for r in rows}
        gold_chunks, gold_documents = set(item["gold_chunks"]), set(item["gold_documents"])
        out[key] = {"item": item, "bench": item["bench"], "qid": item["qid"],
                    "kind": item.get("kind", "single"), "rows": rows,
                    "hit": bool(gold_chunks & chunks), "doc_hit": bool(gold_documents & documents)}
    return out


def contextes(entry: dict, index: ChunkIndex) -> tuple[list[dict], list[dict]]:
    """(SERVI, ORACLE). ORACLE place le ou les chunks d'or en tête, taille constante.

    ``build_context`` plafonne à 2 passages par document : mettre l'or en tête puis laisser
    la même règle s'appliquer garde un contexte **réaliste** — ce n'est pas « l'or plus cinq
    passages », c'est « l'or à la place du moins bien classé ».
    """
    servi = pipeline.build_context(pipeline._with_text(entry["rows"], index))
    or_rows = []
    for chunk_id in entry["item"]["gold_chunks"]:
        source = index.get(chunk_id)
        if source is None:
            continue
        fiche = index.metadata.get(source["document_id"]) or {}
        or_rows.append({"chunk_id": chunk_id, "document_id": source["document_id"],
                        "text": source["text"], "section": source["section"],
                        "page_start": source["page_start"],
                        "title": fiche.get("title") or index.title_of(source["document_id"]),
                        "short_ref": fiche.get("short_ref"), "score": 1.0})
    oracle = pipeline.build_context(or_rows + pipeline._with_text(entry["rows"], index))
    return servi, oracle


def charger(nom: str) -> dict:
    path = CACHE / f"answer-gap-{nom}-{SIGNATURE}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def ecrire(nom: str, payload: dict) -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    (CACHE / f"answer-gap-{nom}-{SIGNATURE}.json").write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8")


#: Le banc v1 est *known-item* : il ne porte que ``target_chunk``, aucun ``answer_facts``.
#: Le juge de couverture n'a donc rien à quoi comparer une réponse, et ``judge.grade`` le
#: refuse. C'est une propriété du banc, pas un choix : le contraste porte sur les **26 ratés
#: de v3**, et ``v1/q02`` en est exclu. Écart au pré-enregistrement, déclaré.
BANC = "v3"


def notable(entry: dict) -> bool:
    return entry["bench"] == BANC and bool(entry["item"].get("answer_facts"))


def temoins(cadre: dict) -> list[str]:
    """40 questions non ratées, stratifiées par famille, tirage reproductible."""
    disponibles = [k for k, e in cadre.items() if e["hit"] and notable(e)]
    par_famille = defaultdict(list)
    for k in disponibles:
        par_famille[cadre[k]["kind"]].append(k)
    rng = random.Random(GRAINE)
    quota = {f: max(1, round(TEMOIN_N * len(v) / len(disponibles))) for f, v in par_famille.items()}
    choisis = []
    for famille, cles in sorted(par_famille.items()):
        choisis += rng.sample(sorted(cles), min(quota[famille], len(cles)))
    return sorted(choisis)[:TEMOIN_N]


# ------------------------------------------------------------------- étape : mesure

def step_mesure(index: ChunkIndex, limit: int | None, sans_temoin: bool = False) -> None:
    cadre = frame(index)
    tous = [k for k, e in cadre.items() if not e["hit"]]
    rates = [k for k in tous if notable(cadre[k])]
    controle = [] if sans_temoin else temoins(cadre)
    exclus = sorted(set(tous) - set(rates))
    print(f"\n  ratés : {len(tous)}  ·  notables par le juge de couverture : {len(rates)}"
          f"  ·  exclus faute de faits de référence : {exclus}")
    print(f"  dont document d'or présent : {sum(1 for k in rates if cadre[k]['doc_hit'])}")
    print(f"  témoin : {len(controle)} questions" +
          (f", stratifiées ({dict(Counter(cadre[k]['kind'] for k in controle))})" if controle
           else " — écarté : il n'ancre qu'un niveau, et ce niveau est un plancher"))

    travaux = [(k, c) for k in rates for c in ("servi", "oracle")] + [(k, "servi") for k in controle]
    if limit:
        travaux = [t for t in travaux if t[0] in rates[:limit]]
    reponses, verdicts = charger("reponses"), charger("verdicts")
    reste = [t for t in travaux if f"{t[0]}/{t[1]}" not in verdicts]
    print(f"  {len(reste)}/{len(travaux)} (question, condition) à traiter — "
          f"générateur {GENERATEUR}, juge {JUGE}")

    for n, (key, condition) in enumerate(reste, 1):
        entry = cadre[key]
        servi, oracle = contextes(entry, index)
        contexte = oracle if condition == "oracle" else servi
        marque = f"{key}/{condition}"
        try:
            if marque not in reponses:
                produit = pipeline.answer(entry["item"]["question"], contexte, model=GENERATEUR)
                reponses[marque] = {"answer": produit["answer"], "abstained": produit["abstained"],
                                    "context": [r.get("chunk_id") for r in contexte],
                                    "gold_in_context": bool(set(entry["item"]["gold_chunks"]) &
                                                            {r.get("chunk_id") for r in contexte})}
                ecrire("reponses", reponses)
            note = judge.grade(entry["item"], reponses[marque]["answer"], contexte,
                               seed=f"gap-{marque}", model=JUGE)
            verdicts[marque] = note
            ecrire("verdicts", verdicts)
            etat = "abstention" if reponses[marque]["abstained"] else "réponse"
            print(f"      {n}/{len(reste)}  {marque:<22} cov={note.get('coverage')} "
                  f"gnd={note.get('groundedness')}  {etat}", flush=True)
        except RuntimeError as erreur:
            print(f"      {n}/{len(reste)}  {marque:<22} ÉCHEC API : {str(erreur)[:70]}", flush=True)
    print(f"\n  {len(verdicts)} verdicts en cache")


# ------------------------------------------------------------------- étape : gardes

def step_gardes(index: ChunkIndex) -> None:
    """Items-témoins et plancher de bruit — l'instrument avant les chiffres qu'il produit."""
    llm.GENERATOR, llm.JUDGE = GENERATEUR, JUGE
    items = [json.loads(l) for l in (HERE / "questions-v3.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    judge.CORPUS = {"chunks": len(index.chunks)}
    print(f"\n  items-témoins — le juge {JUGE} note des réponses dont la note est connue")
    pieges = judge.build_traps(items, index, seed=GRAINE)
    resultat = judge.run_traps(pieges)
    print(f"  exactitude globale : {resultat['accuracy']} sur {resultat['n']} témoins")
    for famille, bloc in resultat["per_family"].items():
        print(f"      {famille:<24} {bloc['accuracy']:>5.3f}  ({bloc['n']} items)")

    reponses = charger("reponses")
    cadre = frame(index)
    echantillon = []
    for marque, bloc in sorted(reponses.items())[:6]:
        key = marque.rsplit("/", 1)[0]
        if key not in cadre or not bloc.get("answer"):
            continue
        servi, oracle = contextes(cadre[key], index)
        echantillon.append((cadre[key]["item"], bloc["answer"],
                            oracle if marque.endswith("oracle") else servi))
    print(f"\n  plancher de bruit — double notation de {len(echantillon)} réponses à T=0,3")
    bruit = judge.noise_floor(echantillon) if echantillon else {}
    print(f"  {json.dumps(bruit, ensure_ascii=False)}")
    ecrire("gardes", {"temoins": resultat, "bruit": bruit,
                      "generateur": GENERATEUR, "juge": JUGE})


# ------------------------------------------------------------------ étape : rapport

def bloc(valeurs: list[float]) -> dict:
    if not valeurs:
        return {"n": 0}
    return {"n": len(valeurs), "moyenne": round(statistics.mean(valeurs), 3),
            "ic95": list(metrics.bootstrap_ci(valeurs))}


def step_rapport(index: ChunkIndex) -> None:
    cadre = frame(index)
    reponses, verdicts, gardes = charger("reponses"), charger("verdicts"), charger("gardes")
    tous = [k for k in cadre if not cadre[k]["hit"]]
    rates = [k for k in tous if notable(cadre[k])]
    apparies = [k for k in rates if f"{k}/servi" in verdicts and f"{k}/oracle" in verdicts]
    controle = [k for k in temoins(cadre) if f"{k}/servi" in verdicts]
    print(f"\n{'=' * 92}\nCOUVERTURE DES RATÉS — {len(apparies)}/{len(rates)} paires complètes "
          f"({len(tous)} ratés, {len(tous) - len(rates)} exclus : le banc v1 ne porte pas de "
          f"faits de référence), témoin {len(controle)}, signature {SIGNATURE}\n{'=' * 92}")

    if gardes:
        t = gardes["temoins"]
        print(f"\n=== garde-fous ===\n  juge {gardes['juge']} · items-témoins {t['accuracy']} sur {t['n']}")
        for famille, b in t["per_family"].items():
            print(f"      {famille:<24} {b['accuracy']:>5.3f}")
        b = gardes.get("bruit") or {}
        print(f"  plancher de bruit : {json.dumps({k: v for k, v in b.items() if k != 'detail'}, ensure_ascii=False)}")

    def cov(marque): return verdicts[marque].get("coverage")
    def gnd(marque): return verdicts[marque].get("groundedness")

    lignes = {}
    for nom, cles, cond in (("SERVI (les 27)", apparies, "servi"), ("ORACLE (les 27)", apparies, "oracle"),
                            ("TÉMOIN (non ratées)", controle, "servi")):
        c = [cov(f"{k}/{cond}") for k in cles if cov(f"{k}/{cond}") is not None]
        g = [gnd(f"{k}/{cond}") for k in cles if gnd(f"{k}/{cond}") is not None]
        a = [1.0 if reponses[f"{k}/{cond}"]["abstained"] else 0.0 for k in cles if f"{k}/{cond}" in reponses]
        lignes[nom] = {"coverage": bloc(c), "groundedness": bloc(g), "abstention": bloc(a)}
        print(f"\n  {nom:<22} couverture {lignes[nom]['coverage'].get('moyenne')} "
              f"{lignes[nom]['coverage'].get('ic95')}  ·  ancrage {lignes[nom]['groundedness'].get('moyenne')}"
              f"  ·  abstention {lignes[nom]['abstention'].get('moyenne')}  (n={len(cles)})")

    servi = [cov(f"{k}/servi") for k in apparies]
    oracle = [cov(f"{k}/oracle") for k in apparies]
    ecart = metrics.paired_delta([float(x or 0) for x in servi], [float(x or 0) for x in oracle])
    print(f"\n=== contraste apparié — Δ couverture (ORACLE − SERVI) sur les {len(apparies)} ratés ===")
    print(f"  Δ = {ecart['delta']:+.3f}  IC95 [{ecart['ci95'][0]:+.3f} ; {ecart['ci95'][1]:+.3f}]"
          f"  significatif={ecart['significant']}")

    groupes = {}
    for nom, filtre in (("A — document d'or DANS le pool", True), ("B — document d'or absent", False)):
        cles = [k for k in apparies if cadre[k]["doc_hit"] is filtre]
        if not cles:
            continue
        d = metrics.paired_delta([float(cov(f"{k}/servi") or 0) for k in cles],
                                 [float(cov(f"{k}/oracle") or 0) for k in cles])
        groupes[nom] = {"n": len(cles), **d}
        print(f"  {nom:<34} n={len(cles):>2}  Δ={d['delta']:+.3f}  "
              f"[{d['ci95'][0]:+.3f} ; {d['ci95'][1]:+.3f}]")

    plancher = ((gardes.get("bruit") or {}).get("coverage") or {}).get("mean_gap")
    passe = (ecart["delta"] >= SEUIL_DELTA and ecart["significant"]
             and (plancher is None or ecart["delta"] > plancher))
    verdict = ("VRAIS ÉCHECS — le chunk d'or manquant coûte une réponse" if passe else
               "PAS DES ÉCHECS DE RÉPONSE — l'IC95 contient zéro" if not ecart["significant"] else
               "NON CONCLUANT à n=27 — significatif mais sous le seuil de magnitude")
    print(f"\n=== verdict — seuils pré-enregistrés (Δ ≥ {SEUIL_DELTA}, IC95 hors zéro, Δ > plancher de bruit) ===")
    print(f"  >>> {verdict}")

    OUTPUT.write_text(json.dumps({
        "corpus": {**corpus_overlay.describe(), "chunks": len(index.chunks),
                   "documents": len(index.documents)},
        "protocole": {"generateur": GENERATEUR, "juge": JUGE, "seuil_delta": SEUIL_DELTA,
                      "temoin_n": TEMOIN_N, "graine": GRAINE,
                      "bootstrap": {"draws": 4000, "seed": GRAINE},
                      "dispositif": "contraste apparié SERVI vs ORACLE sur les mêmes questions",
                      "note_juge": "juge sévère au crédit ; toute couverture absolue est un plancher"},
        "perimetre": {"rates_total": len(tous), "rates_notables": len(rates),
                      "exclus": sorted(set(tous) - set(rates)),
                      "cause_exclusion": "le banc v1 est known-item et ne porte pas d'answer_facts ; "
                                         "le juge de couverture n'a rien à quoi comparer"},
        "gardes": gardes, "lignes": lignes,
        "contraste": ecart, "par_groupe": groupes, "plancher_de_bruit_couverture": plancher,
        "verdict": {"passe": passe, "texte": verdict},
        "par_question": [{"key": k, "kind": cadre[k]["kind"], "doc_hit": cadre[k]["doc_hit"],
                          "coverage_servi": cov(f"{k}/servi"), "coverage_oracle": cov(f"{k}/oracle"),
                          "groundedness_servi": gnd(f"{k}/servi"), "groundedness_oracle": gnd(f"{k}/oracle"),
                          "abstained_servi": reponses[f"{k}/servi"]["abstained"],
                          "abstained_oracle": reponses[f"{k}/oracle"]["abstained"]} for k in apparies],
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n  écrit  {OUTPUT.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--step", required=True, choices=("mesure", "gardes", "rapport"))
    parser.add_argument("--limit", type=int, help="pilote : n premiers ratés")
    parser.add_argument("--sans-temoin", action="store_true",
                        help="ne mesure que le contraste apparié — le témoin n'ancre qu'un niveau, "
                             "qui est de toute façon un plancher, et il ne décide de rien")
    args = parser.parse_args()
    llm.GENERATOR, llm.JUDGE = GENERATEUR, JUGE
    index = ChunkIndex.load(verbose=False)
    judge.CORPUS = {"chunks": len(index.chunks)}
    print(f"  corpus {len(index.chunks)} chunks · {len(index.documents)} documents · signature {SIGNATURE}")
    {"mesure": lambda: step_mesure(index, args.limit, args.sans_temoin),
     "gardes": lambda: step_gardes(index),
     "rapport": lambda: step_rapport(index)}[args.step]()


if __name__ == "__main__":
    main()
