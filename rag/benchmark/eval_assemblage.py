"""L'assemblage du contexte, au niveau réponse — une seule variable, et un témoin en recensement.

Le chantier : ``build_context(passages=5, per_document=2)`` → ``(10, 3)``. **Le pool dense et
son ordre sont bit-à-bit identiques entre les deux bras** ; seule change la règle qui en tire
les passages montrés. Ni le corpus, ni l'index, ni les vecteurs, ni le classement, ni les
questions, ni les prompts, ni le juge ne bougent.

Pré-enregistrement : ``PRE-ENREGISTREMENT-ASSEMBLAGE-2026-09-07.md``, commité **avant** le
premier appel de mesure, avec la liste gelée des questions par strate
(``strates-assemblage-<signature>.json``).

Les étapes, dans l'ordre, et les trois premières ne coûtent aucun appel :

    --step strates          les strates, gelées et publiées. Hors ligne.
    --step identite         le contrôle qui rend vraie la phrase « une seule variable ».
    --step gardes           items-témoins du juge et plancher de bruit.
    --step mesure           les deux bras, reprise par (question, bras).
    --step controle-juge    le juge note-t-il la couverture d'après les passages montrés ?
    --step rapport          le verdict, selon les règles pré-enregistrées.

**Pourquoi un recensement et non un échantillon.** Le §15 a tiré 60 témoins sur 94 pour
économiser 34 questions, et sa garde a échoué faute de puissance — borne basse −0,200 contre
une tolérance de −0,15. Ici la population de témoins tient en 56 questions ; les recenser
coûte ~50 appels de plus qu'un échantillon de 40 et supprime **toute** question de cadre de
tirage, de taille, de stratification et de graine. L'économie d'un échantillon serait une
fausse économie, et le dossier l'a déjà payée une fois.

**Pourquoi ``coverage`` décide et ``groundedness`` non.** ``judge.GRADE_POSITIVE`` note
l'étayage *contre les passages montrés*. Le bras candidat en montre deux fois plus : une même
réponse y est mécaniquement au moins aussi étayée. Rapporter un gain de ``groundedness``
mesurerait la taille du contexte, pas la qualité de la réponse. Il est publié comme
diagnostic, avec cette réserve, et il ne décide de rien.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import corpus_overlay  # noqa: E402
import experiment  # noqa: E402
import judge  # noqa: E402
import llm  # noqa: E402
import metrics  # noqa: E402
import pipeline  # noqa: E402
from corpus import ChunkIndex  # noqa: E402
from garde_reponse import wilson_haut  # noqa: E402

SIGNATURE = corpus_overlay.signature()
CACHE = HERE / ".cache"
BANC = "v3"                       # v1 n'a aucun ``answer_facts`` : 0 sur 25, vérifié
GENERATEUR = "gemini-3.1-flash-lite"
JUGE = "gemini-3.1-flash-lite"
GRAINE = 20260901                 # bootstrap, comme tout le dossier
GRAINE_CONTROLE_JUGE = 20260907   # tirage du contrôle J, déclaré au pré-enregistrement

#: Les deux bras. La production d'abord — c'est elle la référence de tous les écarts.
REFERENCE = {"passages": 5, "per_document": 2}
CANDIDAT = {"passages": 10, "per_document": 3}

#: Garde de dégradation stricte : plus petit ``b`` dont la borne haute de Wilson dépasse 10 %.
PLAFOND_DEGRADATION = 0.10
#: Borne d'instrument du témoin, reprise du §15 pour que les deux chantiers se comparent.
BORNE_INSTRUMENT = -0.15
#: Valeur marginale la **plus basse** jamais mesurée pour un or qui entre dans le contexte
#: (§11, population générale). Sert à définir « plat », et à rien d'autre.
ANCRE_BASSE = 0.462

STRATES = HERE / f"strates-assemblage-{SIGNATURE}.json"
RESULTATS = HERE / f"results-assemblage-{SIGNATURE}.json"


# ------------------------------------------------------------------------------- socle


def contexte(rows: list[dict], bras: dict) -> list[dict]:
    return pipeline.build_context(rows, passages=bras["passages"], per_document=bras["per_document"])


def socle(index: ChunkIndex) -> dict:
    """Pour chaque question v3 positive : le pool servi, les deux contextes, la strate.

    Le pool vient de ``.cache/router-retrievals-<signature>.json`` — les classements que
    Qdrant a réellement rendus, pas une reconstruction. Les deux bras partent de **la même
    liste, dans le même ordre** : c'est ce que ``--step identite`` vérifie.
    """
    pools = experiment.cached_rankings()
    cadre = {}
    for item in experiment.load_items(index):
        cle = item["key"]
        if item["bench"] != BANC or not item.get("answer_facts") or cle not in pools:
            continue
        rows = pipeline._with_text(experiment.rows_from_cache(pools[cle]["dense"]), index)
        reference, candidat = contexte(rows, REFERENCE), contexte(rows, CANDIDAT)
        ors = set(item["gold_chunks"])
        avant = bool(ors & {r["chunk_id"] for r in reference})
        apres = bool(ors & {r["chunk_id"] for r in candidat})
        strate = ("entrantes" if apres and not avant else "sortantes" if avant and not apres
                  else "temoins" if avant else "neutres")
        cadre[cle] = {"item": item, "kind": item.get("kind", "single"), "strate": strate,
                      "pool": [r["chunk_id"] for r in rows],
                      "reference": reference, "candidat": candidat}
    return cadre


def par_strate(cadre: dict) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {"entrantes": [], "temoins": [], "neutres": [], "sortantes": []}
    for cle, ligne in cadre.items():
        out[ligne["strate"]].append(cle)
    return {nom: sorted(cles) for nom, cles in out.items()}


def empreinte_des_pools(cadre: dict) -> str:
    """Empreinte de tous les pools servis : elle fige ce sur quoi les deux bras travaillent."""
    digest = hashlib.sha256()
    for cle in sorted(cadre):
        digest.update(cle.encode())
        digest.update("|".join(cadre[cle]["pool"]).encode())
    return digest.hexdigest()[:16]


# ------------------------------------------------------------------------------- caches


def charger(nom: str) -> dict:
    chemin = CACHE / f"assemblage-{nom}-{SIGNATURE}.json"
    return json.loads(chemin.read_text(encoding="utf-8")) if chemin.exists() else {}


def ecrire(nom: str, payload: dict) -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    (CACHE / f"assemblage-{nom}-{SIGNATURE}.json").write_text(
        json.dumps(payload, indent=1, ensure_ascii=False), encoding="utf-8")


# ------------------------------------------------------------------------------- étapes


def step_strates(index: ChunkIndex) -> None:
    cadre = socle(index)
    strates = par_strate(cadre)
    gele = {
        "signature": SIGNATURE, "banc": BANC, "population": len(cadre),
        "reference": REFERENCE, "candidat": CANDIDAT,
        "empreinte_des_pools": empreinte_des_pools(cadre),
        "strates": strates,
        "tailles": {nom: len(cles) for nom, cles in strates.items()},
        "familles": {nom: dict(sorted(
            (f, sum(1 for c in cles if cadre[c]["kind"] == f))
            for f in sorted({cadre[c]["kind"] for c in cles}))) for nom, cles in strates.items()},
        "seuil_degradation_stricte": next(
            b for b in range(len(strates["temoins"]) + 1)
            if wilson_haut(b, len(strates["temoins"])) > PLAFOND_DEGRADATION),
    }
    STRATES.write_text(json.dumps(gele, indent=1, ensure_ascii=False), encoding="utf-8")
    for nom, cles in strates.items():
        print(f"  {nom:<10} {len(cles):3}   {gele['familles'][nom]}")
    print(f"\n  empreinte des pools : {gele['empreinte_des_pools']}")
    print(f"  seuil de dégradation stricte : b >= {gele['seuil_degradation_stricte']} "
          f"sur {len(strates['temoins'])} témoins")
    print(f"\n  écrit  {STRATES.relative_to(ROOT)}")


def step_identite(index: ChunkIndex) -> None:
    """« Une seule variable » est une affirmation ; ceci en fait un contrôle.

    Trois choses doivent tenir, et la troisième est celle qu'on oublie : le contexte candidat
    doit être un **sur-ensemble ordonné** du contexte de référence chaque fois qu'aucun
    passage n'est évincé, sans quoi le changement ne serait pas seulement un élargissement.
    """
    cadre = socle(index)
    gele = json.loads(STRATES.read_text(encoding="utf-8"))
    echecs = []
    if gele["empreinte_des_pools"] != empreinte_des_pools(cadre):
        echecs.append("l'empreinte des pools a changé depuis le gel des strates")
    if gele["strates"] != par_strate(cadre):
        echecs.append("les strates recalculées diffèrent de celles qui ont été gelées")
    inclusions = 0
    for cle, ligne in cadre.items():
        ref = [r["chunk_id"] for r in ligne["reference"]]
        can = [r["chunk_id"] for r in ligne["candidat"]]
        if len(can) < len(ref):
            echecs.append(f"{cle}: le contexte candidat est plus court que la référence")
        inclusions += set(ref) <= set(can)
    print(f"  empreinte des pools     {gele['empreinte_des_pools']}  "
          f"{'identique' if not any('empreinte' in e for e in echecs) else 'CHANGÉE'}")
    print(f"  strates                 {'identiques au gel' if not any('strates' in e for e in echecs) else 'DIFFÉRENTES'}")
    print(f"  référence ⊆ candidat    {inclusions}/{len(cadre)} questions")
    print(f"  passages moyens         référence "
          f"{statistics.mean(len(l['reference']) for l in cadre.values()):.2f} · candidat "
          f"{statistics.mean(len(l['candidat']) for l in cadre.values()):.2f}")
    for ligne in echecs[:5]:
        print(f"  ÉCHEC  {ligne}")
    sys.exit(1 if echecs else 0)


def step_gardes(index: ChunkIndex) -> None:
    """Items-témoins et plancher de bruit — si le juge est cassé, on s'arrête là.

    Les items-témoins se construisent sur le banc v3 **brut** (négatives comprises) : c'est
    la population que `judge.build_traps` sait piéger, et v1 n'a aucun `answer_facts`.
    """
    llm.GENERATOR, llm.JUDGE = GENERATEUR, JUGE
    judge.CORPUS = {"chunks": len(index.chunks)}
    items = [json.loads(ligne) for ligne in
             (HERE / f"questions-{BANC}.jsonl").read_text(encoding="utf-8").splitlines()
             if ligne.strip()]
    print(f"  items-témoins — {JUGE} note des réponses dont la note correcte est connue")
    temoins = judge.build_traps(items, index, seed=GRAINE)   # construits une fois : ils appellent
    pieges = judge.run_traps(temoins)                        # le générateur pour deux familles
    print(f"  exactitude globale : {pieges['accuracy']} sur {pieges['n']} témoins")
    for famille, bloc in pieges["per_family"].items():
        print(f"      {famille:<24} {bloc['accuracy']:>5.3f}  ({bloc['n']} items)")

    cadre = socle(index)
    echantillon = []
    for cle in sorted(cadre)[:6]:
        ligne = cadre[cle]
        produit = pipeline.answer(ligne["item"]["question"], ligne["reference"], model=GENERATEUR)
        echantillon.append((ligne["item"], produit["answer"], ligne["reference"]))
    # L'exactitude globale mélange deux axes, et un seul décide. Un témoin peut échouer
    # uniquement sur `groundedness` — que le §4 du pré-enregistrement a exclu de la décision
    # parce qu'il se note *contre les passages montrés*. On publie donc l'exactitude
    # restreinte à `coverage`, sur les seuls témoins qui l'attendent.
    par_axe = {"coverage": [], "groundedness": []}
    attentes = {(t["trap"], t["qid"]): t["expect"] for t in temoins}
    for detail in pieges["detail"]:
        expect = attentes.get((detail["trap"], detail["qid"]), {})
        for axe in par_axe:
            if axe in expect:
                bas, haut = expect[axe]
                par_axe[axe].append(bas <= (detail["verdict"].get(axe) or 0) <= haut)
    exactitude_par_axe = {axe: {"n": len(v), "exactitude": round(sum(v) / len(v), 3)}
                          for axe, v in par_axe.items() if v}
    print("\n  exactitude par axe — seul `coverage` décide (§4 du pré-enregistrement) :")
    for axe, bloc in exactitude_par_axe.items():
        print(f"      {axe:<14} {bloc['exactitude']:>5.3f}  ({bloc['n']} items)")

    print(f"\n  plancher de bruit — double notation de {len(echantillon)} réponses à T=0,3")
    bruit = judge.noise_floor(echantillon) if echantillon else {}
    print(f"  {json.dumps(bruit, ensure_ascii=False)}")
    ecrire("gardes", {"pieges": pieges, "exactitude_par_axe": exactitude_par_axe,
                      "plancher_de_bruit": bruit,
                      "generateur": GENERATEUR, "juge": JUGE})


def step_mesure(index: ChunkIndex, limit: int | None, palier: str) -> None:
    llm.GENERATOR, llm.JUDGE = GENERATEUR, JUGE
    cadre = socle(index)
    gele = json.loads(STRATES.read_text(encoding="utf-8"))
    if gele["strates"] != par_strate(cadre):
        sys.exit("les strates ont changé depuis le gel — mesurer ici serait mesurer autre chose")

    ordre = ["entrantes", "temoins", "neutres"] if palier == "tout" else [palier]
    cles = [c for nom in ordre for c in gele["strates"][nom]]
    reponses, verdicts = charger("reponses"), charger("verdicts")
    travaux = [(c, b) for c in cles for b in ("reference", "candidat")]
    if limit:
        travaux = travaux[:limit]
    reste = [t for t in travaux if f"{t[0]}/{t[1]}" not in verdicts]
    print(f"  population {len(cadre)} · paliers {ordre} · {len(travaux)} travaux, "
          f"{len(reste)} à produire — générateur {GENERATEUR}, juge {JUGE}")

    echecs = 0
    for n, (cle, bras) in enumerate(reste, 1):
        ligne = cadre[cle]
        ctx = ligne[bras]
        marque = f"{cle}/{bras}"
        try:
            if marque not in reponses:
                produit = pipeline.answer(ligne["item"]["question"], ctx, model=GENERATEUR)
                reponses[marque] = {
                    "answer": produit["answer"], "abstained": produit["abstained"],
                    "context": [r["chunk_id"] for r in ctx],
                    "or_dans_le_contexte": bool(set(ligne["item"]["gold_chunks"]) &
                                                {r["chunk_id"] for r in ctx})}
                ecrire("reponses", reponses)
            note = judge.grade(ligne["item"], reponses[marque]["answer"], ctx,
                               seed=f"asm-{marque}", model=JUGE)
            verdicts[marque] = note
            ecrire("verdicts", verdicts)
            etat = "abstention" if reponses[marque]["abstained"] else "réponse"
            print(f"    {n}/{len(reste)}  {marque:<26} [{ligne['strate'][:4]}] "
                  f"cov={note.get('coverage')} gnd={note.get('groundedness')}  {etat}", flush=True)
            echecs = 0
        except RuntimeError as erreur:
            echecs += 1
            print(f"    {n}/{len(reste)}  {marque:<26} ÉCHEC API : {str(erreur)[:70]}", flush=True)
            if echecs >= 5:
                print("\n  cinq échecs consécutifs — arrêt. Le cache est intact, relancer reprend.")
                break
    print(f"\n  {len(verdicts)} verdicts en cache")


def step_controle_juge(index: ChunkIndex) -> None:
    """Le juge note-t-il la couverture d'après les passages montrés ?

    ``coverage`` est censé porter sur les **faits requis**, établis indépendamment de la
    réponse ; ``groundedness`` porte explicitement sur les passages. Si la couverture bougeait
    avec le contexte montré, le contraste décisif mesurerait la taille du contexte. On
    renote donc la réponse du bras candidat **en montrant le contexte de référence** : la
    couverture doit être inchangée.
    """
    llm.GENERATOR, llm.JUDGE = GENERATEUR, JUGE
    cadre = socle(index)
    gele = json.loads(STRATES.read_text(encoding="utf-8"))
    reponses, verdicts = charger("reponses"), charger("verdicts")
    tirage = random.Random(GRAINE_CONTROLE_JUGE)
    cibles = sorted(tirage.sample(sorted(gele["strates"]["temoins"]), 20))
    controle = charger("controle_juge")
    ecarts = []
    for n, cle in enumerate(cibles, 1):
        marque = f"{cle}/candidat"
        if marque not in reponses or marque not in verdicts:
            print(f"    {cle} : bras candidat non mesuré, ignoré")
            continue
        if marque not in controle:
            note = judge.grade(cadre[cle]["item"], reponses[marque]["answer"],
                               cadre[cle]["reference"], seed=f"asm-J-{marque}", model=JUGE)
            controle[marque] = note
            ecrire("controle_juge", controle)
        avant, apres = verdicts[marque].get("coverage"), controle[marque].get("coverage")
        if avant != apres:
            ecarts.append(f"{cle}: coverage {avant} → {apres} en changeant les passages montrés")
        print(f"    {n}/{len(cibles)}  {cle:<12} coverage {avant} → {apres}", flush=True)
    print(f"\n  {len(ecarts)}/{len(cibles)} questions dont la couverture dépend des passages montrés")
    for ligne in ecarts:
        print(f"    {ligne}")
    print("  seuil pré-enregistré : plus de 2 écarts sur 20 → le contraste décisif est nul")


# ------------------------------------------------------------------------------- rapport


def precis(reference: list[float], variante: list[float]) -> dict:
    return metrics.paired_delta(reference, variante, draws=4000)


def step_rapport(index: ChunkIndex) -> None:
    cadre = socle(index)
    gele = json.loads(STRATES.read_text(encoding="utf-8"))
    verdicts, reponses = charger("verdicts"), charger("reponses")

    def appariees(cles: list[str]) -> list[str]:
        """Les questions dont **les deux** bras sont mesurés. Un test apparié n'a pas d'autre
        population : filtrer chaque bras séparément produirait deux listes de même longueur
        portant sur des questions différentes, et le Δ serait un artefact."""
        return [c for c in cles if f"{c}/reference" in verdicts and f"{c}/candidat" in verdicts]

    def notes(cles, champ="coverage"):
        paires = appariees(cles)
        return ([verdicts[f"{c}/reference"][champ] for c in paires],
                [verdicts[f"{c}/candidat"][champ] for c in paires])

    rapport = {"signature": SIGNATURE, "banc": BANC, "generateur": GENERATEUR, "juge": JUGE,
               "graine": GRAINE, "reference": REFERENCE, "candidat": CANDIDAT,
               "tailles": gele["tailles"], "empreinte_des_pools": gele["empreinte_des_pools"],
               "gardes": charger("gardes"), "strates": {}}

    for nom in ("entrantes", "temoins", "neutres"):
        cles = gele["strates"][nom]
        ref, can = notes(cles)
        if not ref:
            continue
        delta = precis(ref, can)
        stricte = [c for c in cles
                   if verdicts.get(f"{c}/reference", {}).get("coverage") == 2
                   and verdicts.get(f"{c}/candidat", {}).get("coverage") == 0]
        rapport["strates"][nom] = {
            "n": len(ref), "taille_de_la_strate": len(cles),
            "complet": len(ref) == len(cles),
            "reference": round(statistics.mean(ref), 4),
            "candidat": round(statistics.mean(can), 4),
            "delta": delta.get("delta"), "ci95": delta.get("ci95"),
            "significant": delta.get("significant"),
            "degradation_stricte": len(stricte), "degradation_stricte_cles": stricte,
            "abstentions": {
                "reference": sum(1 for c in cles if reponses.get(f"{c}/reference", {}).get("abstained")),
                "candidat": sum(1 for c in cles if reponses.get(f"{c}/candidat", {}).get("abstained"))},
        }
        etayage_ref, etayage_can = notes(cles, "groundedness")
        if etayage_ref:
            rapport["strates"][nom]["groundedness"] = {
                "reference": round(statistics.mean(etayage_ref), 4),
                "candidat": round(statistics.mean(etayage_can), 4),
                "reserve": "noté contre les passages montrés — le bras candidat en montre "
                           "deux fois plus. Diagnostic, ne décide de rien."}

    # L'effet net se lit sur la population fixe **entière**, ou pas du tout. Un net calculé
    # sur les questions qui se trouvent avoir été mesurées serait un sous-ensemble choisi par
    # l'ordre d'exécution et par l'endroit où le quota est mort — c'est-à-dire par rien.
    toutes = sorted(cadre)
    ref, can = notes(toutes)
    rapport["population_fixe"] = (
        {"n": len(ref), **precis(ref, can), "reference": round(statistics.mean(ref), 4),
         "candidat": round(statistics.mean(can), 4)}
        if len(ref) == len(toutes) else
        {"rendu": False, "mesurees": len(ref), "population": len(toutes),
         "raison": "population incomplète — l'effet net n'est pas rendu"})
    rapport["verdict"] = verdict(rapport, gele)
    RESULTATS.write_text(json.dumps(rapport, indent=1, ensure_ascii=False), encoding="utf-8")
    imprimer(rapport)
    print(f"\n  écrit  {RESULTATS.relative_to(ROOT)}")


def verdict(rapport: dict, gele: dict) -> dict:
    """Les règles pré-enregistrées, appliquées par le code et non par la prose."""
    e = rapport["strates"].get("entrantes")
    t = rapport["strates"].get("temoins")
    if not e or not t:
        return {"issue": "INCOMPLET", "raison": "une strate décisive n'est pas mesurée"}
    bas_e, haut_e = e["ci95"]
    bas_t = t["ci95"][0]
    seuil_degradation = gele["seuil_degradation_stricte"]
    economique = -(e["n"] / t["n"]) * e["delta"]
    gardes = {
        "instrument": bas_t > BORNE_INSTRUMENT,
        "economique": bas_t > economique,
        "degradation_stricte": t["degradation_stricte"] < seuil_degradation,
        "borne_economique": round(economique, 4),
        "seuil_degradation": seuil_degradation,
    }
    if haut_e < 0:
        issue = "REGRESSION"
    elif haut_e < ANCRE_BASSE:
        issue = "PLAT"
    elif bas_e <= 0:
        issue = "NON_CONCLUANT"
    elif not gardes["degradation_stricte"] or not gardes["economique"]:
        issue = "NO_GO"
    elif not gardes["instrument"]:
        issue = "HOLD"
    else:
        issue = "GO"
    return {"issue": issue, "gardes": gardes,
            "delta_entrantes": e["delta"], "ci95_entrantes": e["ci95"],
            "delta_temoins": t["delta"], "ci95_temoins": t["ci95"]}


def imprimer(rapport: dict) -> None:
    print(f"\n  corpus {rapport['signature']} · banc {rapport['banc']} · "
          f"générateur {rapport['generateur']} · juge {rapport['juge']}")
    print(f"  pools {rapport['empreinte_des_pools']} — identiques entre les deux bras\n")
    print("  strate           n   référence  candidat      Δ [IC95]                   2→0  abst.")
    for nom, s in rapport["strates"].items():
        ic = f"[{s['ci95'][0]:+.3f} ; {s['ci95'][1]:+.3f}]" if s.get("ci95") else ""
        etat = "" if s["complet"] else f" PARTIELLE {s['n']}/{s['taille_de_la_strate']}"
        print(f"  {nom:<12}{etat:<4} {s['n']:3}   {s['reference']:8.3f}  {s['candidat']:8.3f}   "
              f"{s['delta']:+.3f} {ic:<24} {s['degradation_stricte']:3}  "
              f"{s['abstentions']['reference']}→{s['abstentions']['candidat']}")
    p = rapport["population_fixe"]
    if p.get("rendu") is False:
        print(f"\n  effet net sur la population fixe : NON RENDU — "
              f"{p['mesurees']}/{p['population']} questions mesurées dans les deux bras")
    else:
        ic = f"[{p['ci95'][0]:+.3f} ; {p['ci95'][1]:+.3f}]" if p.get("ci95") else ""
        print(f"\n  population fixe ({p['n']}) : {p['reference']:.3f} → {p['candidat']:.3f}   "
              f"{p['delta']:+.3f} {ic}")
    v = rapport["verdict"]
    print(f"\n  === {v['issue']} ===")
    if "gardes" in v:
        g = v["gardes"]
        print(f"    borne d'instrument (> {BORNE_INSTRUMENT})       {'tenue' if g['instrument'] else 'ÉCHOUÉE'}")
        print(f"    borne économique   (> {g['borne_economique']:+.4f})  {'tenue' if g['economique'] else 'ÉCHOUÉE'}")
        print(f"    dégradation stricte (< {g['seuil_degradation']})       "
              f"{'tenue' if g['degradation_stricte'] else 'ÉCHOUÉE'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--step", required=True,
                        choices=("strates", "identite", "gardes", "mesure", "controle-juge", "rapport"))
    parser.add_argument("--palier", default="tout",
                        choices=("entrantes", "temoins", "neutres", "tout"))
    parser.add_argument("--limit", type=int, help="pilote : n premiers travaux")
    args = parser.parse_args()
    index = ChunkIndex.load(verbose=False)
    if {"strates": step_strates, "identite": step_identite}.get(args.step):
        {"strates": step_strates, "identite": step_identite}[args.step](index)
    elif args.step == "gardes":
        step_gardes(index)
    elif args.step == "mesure":
        step_mesure(index, args.limit, args.palier)
    elif args.step == "controle-juge":
        step_controle_juge(index)
    else:
        step_rapport(index)


if __name__ == "__main__":
    main()
