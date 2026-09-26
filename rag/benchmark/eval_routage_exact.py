"""Le routage par famille — plafond d'un routeur *oracle*, calculé hors ligne.

La Phase A (``eval_pool_rerank.py``) a rendu un NO-GO sur l'élargissement du pool et laissé
un seul signal debout : la famille ``exact``, +0,117 [−0,001 ; +0,260] sur dix questions.
D'où l'idée que ce script met à l'épreuve : **n'élargir que là où l'élargissement paie**.

Rien n'est reclassé ici. Le routage ne crée aucun classement : il *choisit*, question par
question, entre deux classements déjà mesurés. Tout se calcule donc sur le cache de paires
de la Phase A, sans modèle, sans Qdrant, sans réseau.

Le routeur mesuré lit **l'étiquette de famille du banc**. C'est une annotation, pas une
prédiction : ce qui est calculé est un **majorant**, celui d'un routeur qui connaît la
réponse. Un détecteur de production perd du rappel et crée des faux positifs — et c'est là
que la « route sans regret » cesse de l'être.

Protocole pré-enregistré et commité avant la première métrique (``15a4850``) :
``rag/benchmark/RAPPORT-ROUTAGE-EXACT-2026-09-05.md``, Partie I.

    .venv/bin/python rag/benchmark/eval_routage_exact.py --step mesure
    .venv/bin/python rag/benchmark/eval_routage_exact.py --step sabotage   # S1 et S2 doivent hurler
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
import metrics  # noqa: E402
from corpus import ChunkIndex  # noqa: E402
from eval_pool_rerank import CONFIGS, LIBELLE, REFERENCE, build_pools  # noqa: E402

SIGNATURE = corpus_overlay.signature()
CACHE = HERE / ".cache"
PARTIAL = CACHE / f"pool-rerank-partial-{SIGNATURE}.json"
ARCHIVE = HERE / f"results-pool-rerank-{SIGNATURE}.json"
OUTPUT = HERE / f"results-routage-{SIGNATURE}.json"

#: Seuil de magnitude, hérité de l'arbitrage de la fusion, réutilisé en Phase A. Le plafond
#: étant un majorant, un plafond *égal* au seuil est un NO-GO, pas un GO.
SEUIL = 0.010
#: Rappel de détection supposé — délibérément généreux — pour la branche haute de la carte.
RAPPEL_DETECTEUR = 0.80
SEUIL_AVEC_MARGE = SEUIL / RAPPEL_DETECTEUR

CIBLE_PRIMAIRE = "rr/dense100+bm25"
CIBLE_SECONDAIRE = "rr/dense100"

#: La route primaire est la seule décisionnelle : ``exact`` a été *nommée* en Phase A §12,
#: dans un document antérieur, à propos d'une autre question. Les trois autres sont des
#: sélections post-hoc sur les mêmes 155 questions — descriptives, jamais décisionnelles.
ROUTE_PRIMAIRE = ("exact",)
ROUTES_DESCRIPTIVES = (
    ("exact", "table"),
    ("exact", "table", "dated"),
    ("exact", "table", "dated", "single"),
)


# ------------------------------------------------------------------ l'estimateur, précis

def delta_precis(reference: list[float], variante: list[float],
                 draws: int = 4000, seed: int = 20260901) -> dict:
    """``metrics.paired_delta`` sans son arrondi au millième.

    Le seuil de décision est à +0,010 et la marge de détecteur à +0,0125 : trois décimales
    ne suffisent pas à situer un plafond attendu autour de +0,008. L'algorithme est copié
    trait pour trait — même graine, même nombre de tirages, même ordre d'appel du RNG — et
    le contrôle P vérifie qu'arrondi au millième il rend exactement ``paired_delta``.
    """
    deltas = [v - r for r, v in zip(reference, variante)]
    rng = random.Random(seed)
    n = len(deltas)
    moyennes = sorted(statistics.mean(rng.choices(deltas, k=n)) for _ in range(draws))
    return {"delta": statistics.mean(deltas),
            "ci95": [moyennes[int(0.025 * draws)], moyennes[int(0.975 * draws) - 1]],
            "deltas": deltas}


# --------------------------------------------------------- recalcul intégral, depuis le cache

def recalcule(index: ChunkIndex, items: list[dict], scores: dict) -> list[dict]:
    """Reconstruit les pools, reclasse par score caché, mesure — la chaîne entière.

    C'est délibérément le chemin long. Lire ``per_question`` dans l'archive serait plus court
    et ne prouverait rien : le contrôle R existe pour vérifier que l'archive est *reproductible*
    depuis le cache de paires, pas pour la recopier.
    """
    pools = build_pools(index, items)
    sortie = []
    for item in items:
        key = item["key"]
        if key not in scores:
            continue
        note, union = scores[key], pools[key]["union"]
        kind = item.get("kind", "single")
        record = {"bench": item["bench"], "key": key, "kind": kind,
                  # L'étiquette de famille n'existe que sur v3. Sur v1 — banc *known-item*,
                  # sans annotation de famille — ``single`` est un défaut du chargeur, pas un
                  # label. Un routeur oracle ne peut être défini que là où les labels sont :
                  # les 25 questions v1 ne sont jamais routées, quelle que soit la route.
                  # C'est aussi la convention de la maison pour ``by_kind`` (v3 seulement,
                  # ``experiment.summarise``) — sans quoi la route « single » balaierait
                  # 77 questions au lieu des 52 que le tableau des familles décrit.
                  "famille": kind if item["bench"] == "v3" else None,
                  "none": experiment.measure(item, pools[key]["reference_order"])}
        for config in CONFIGS[1:]:
            membres = pools[key]["membres"][config]
            sous = [r for r in union if r["chunk_id"] in membres and r["text"] and r["chunk_id"] in note]
            record[config] = experiment.measure(item, sorted(sous, key=lambda r: -note[r["chunk_id"]]))
        sortie.append(record)
    return sortie


# ------------------------------------------------------------------------------ contrôles

def controle_R(recalculee: list[dict], archive: dict) -> list[str]:
    """La référence recalculée depuis le cache égale le ``per_question`` archivé. 155/155."""
    attendu = {r["key"]: r for r in archive["per_question"]}
    ecarts = []
    for row in recalculee:
        cible = attendu.get(row["key"])
        if cible is None:
            ecarts.append(f"{row['key']} · archive : absent")
            continue
        for config in CONFIGS:
            if abs(row[config]["ndcg"] - cible[config]["ndcg"]) > 1e-12:
                # Séparateur « · » et non « / » : la clé de question en contient déjà un
                # (``v1/q01``), et S2 découpait dessus — il accusait « v1 » au lieu de
                # « v1/q01 » et se déclarait en échec sur une question qu'il avait pourtant
                # correctement identifiée.
                ecarts.append(f"{row['key']} · {config} : nDCG {row[config]['ndcg']:.9f} "
                              f"au lieu de {cible[config]['ndcg']:.9f}")
    return ecarts


def controle_F(recalculee: list[dict], archive: dict) -> list[str]:
    """Effectifs et moyennes par famille — égalité au millième avec le ``by_kind`` archivé."""
    ecarts = []
    v1_mal_etiquetees = [r["key"] for r in recalculee if r["bench"] == "v1" and r["famille"] is not None]
    if v1_mal_etiquetees:
        ecarts.append(f"{len(v1_mal_etiquetees)} questions v1 portent une famille routable")
    for famille, attendu in archive["by_kind"].items():
        lignes = [r for r in recalculee if r["bench"] == "v3" and r["kind"] == famille]
        if len(lignes) != attendu["n"]:
            ecarts.append(f"{famille} : n = {len(lignes)} au lieu de {attendu['n']}")
            continue
        for config in CONFIGS:
            obtenu = statistics.mean(r[config]["ndcg"] for r in lignes)
            if abs(round(obtenu, 3) - attendu[config]) > 1e-9:
                ecarts.append(f"{famille}/{config} : {obtenu:.4f} au lieu de {attendu[config]:.4f}")
    return ecarts


def controle_Z(recalculee: list[dict]) -> list[str]:
    """Router l'ensemble vide doit rendre exactement la référence."""
    ref = [r[REFERENCE]["ndcg"] for r in recalculee]
    var = route_ndcg(recalculee, (), CIBLE_PRIMAIRE)
    if ref != var:
        return [f"{sum(1 for a, b in zip(ref, var) if a != b)} questions diffèrent"]
    mesure = delta_precis(ref, var)
    if mesure["delta"] != 0.0 or mesure["ci95"] != [0.0, 0.0]:
        return [f"Δ = {mesure['delta']} IC {mesure['ci95']} au lieu de 0 [0 ; 0]"]
    return []


def controle_P(recalculee: list[dict]) -> list[str]:
    """L'estimateur précis, arrondi au millième, doit rendre exactement ``paired_delta``."""
    ref = [r[REFERENCE]["ndcg"] for r in recalculee]
    ecarts = []
    for config in (CIBLE_PRIMAIRE, CIBLE_SECONDAIRE):
        var = [r[config]["ndcg"] for r in recalculee]
        maison = metrics.paired_delta(ref, var)
        precis = delta_precis(ref, var)
        attendu = {"delta": round(precis["delta"], 3),
                   "ci95": [round(precis["ci95"][0], 3), round(precis["ci95"][1], 3)]}
        if maison["delta"] != attendu["delta"] or maison["ci95"] != attendu["ci95"]:
            ecarts.append(f"{config} : maison {maison['delta']} {maison['ci95']} "
                          f"contre précis arrondi {attendu['delta']} {attendu['ci95']}")
    return ecarts


# --------------------------------------------------------------------------- les routes

def route_ndcg(recalculee: list[dict], route: tuple, cible: str) -> list[float]:
    """nDCG question par question sous une règle de routage : cible si routée, référence sinon."""
    return [r[cible]["ndcg"] if r["famille"] in route else r[REFERENCE]["ndcg"] for r in recalculee]


def plafond_de(recalculee: list[dict], route: tuple, cible: str) -> dict:
    ref = [r[REFERENCE]["ndcg"] for r in recalculee]
    var = route_ndcg(recalculee, route, cible)
    mesure = delta_precis(ref, var)
    routees = [r["key"] for r in recalculee if r["famille"] in route]
    bougees = [(r["key"], r[cible]["ndcg"] - r[REFERENCE]["ndcg"])
               for r in recalculee if r["famille"] in route
               and abs(r[cible]["ndcg"] - r[REFERENCE]["ndcg"]) > 1e-12]
    return {"route": list(route), "cible": cible, "n_routees": len(routees),
            "plafond": mesure["delta"], "ci95": mesure["ci95"],
            "gagnantes": sum(1 for _, d in bougees if d > 0),
            "perdantes": sum(1 for _, d in bougees if d < 0),
            "mouvements": [{"key": k, "delta": round(d, 4)} for k, d in sorted(bougees, key=lambda x: -x[1])]}


def clairvoyance(recalculee: list[dict], cible: str) -> dict:
    """Majorant de *toute* règle de routage : ne router que les questions qui y gagnent.

    Diagnostique, jamais décisionnel. Il borne l'avenue entière — si même lui est petit,
    aucun routeur d'aucune sorte ne vaut d'être construit.
    """
    gains = [max(0.0, r[cible]["ndcg"] - r[REFERENCE]["ndcg"]) for r in recalculee]
    positives = [r["key"] for r, g in zip(recalculee, gains) if g > 0]
    return {"plafond": sum(gains) / len(gains), "n_questions_gagnantes": len(positives),
            "questions": positives}


# ------------------------------------------------------------------------------- latence

def latences(archive: dict, route: tuple, cible: str, recalculee: list[dict]) -> dict:
    """Latence du système *routé*, dérivée comme en Phase A : taille de pool × cadence.

    Le calcul par question et non en bloc est ce qui rend le p95 lisible — et c'est le p95
    qui décide. L'intuition dit qu'une route qui n'envoie que 6 % du trafic sur la voie
    large ne coûte presque rien ; elle est vraie pour la moyenne et pour le p50, et fausse
    pour la queue, puisque les questions routées *sont* les plus lentes et occupent donc
    exactement le haut de la distribution.
    """
    chemin = CACHE / f"pool-rerank-latence-{SIGNATURE}.json"
    if not chemin.exists():
        return {}
    cadence = {r["key"]: r["seconds"] / r["pairs"] for r in json.loads(chemin.read_text(encoding="utf-8"))}
    tailles = archive["tailles_de_pool_par_question"]
    routees = {r["key"] for r in recalculee if r["famille"] in route}

    def profil(choix) -> dict:
        secondes = sorted(tailles[choix(r["key"])][r["key"]] * cadence[r["key"]] for r in recalculee)
        n = len(secondes)
        return {"p50": round(secondes[n // 2], 1), "p95": round(secondes[min(int(0.95 * n), n - 1)], 1),
                "max": round(secondes[-1], 1), "moyenne": round(statistics.mean(secondes), 1)}

    return {"reference": profil(lambda k: REFERENCE),
            "cible_partout": profil(lambda k: cible),
            "route": profil(lambda k: cible if k in routees else REFERENCE),
            "part_routee": round(len(routees) / len(recalculee), 4)}


# ------------------------------------------------------------------------------ sabotages

def sabotage_S1(index: ChunkIndex, items: list[dict], scores: dict, vrai: float,
                tirages: int = 2000) -> tuple[list[str], dict]:
    """Permutation des étiquettes de famille : le plafond doit s'effondrer.

    Le pré-enregistrement prévoyait une **rotation d'un cran**. Elle est inerte, et le
    constater est le seul intérêt qu'elle a eu : les questions ``exact`` du banc v3 sont
    **contiguës** dans l'ordre de chargement (``x01``…``x10``), donc décaler les étiquettes
    d'un rang laisse 8 des 10 questions routées en place — et les deux qui changent ont un δ
    nul, comme 141 questions sur 155. Le plafond survivait à l'identique. C'est la même
    famille de défaut que le sabotage de la Phase A, qui ne savait pas faire échouer son
    contrôle A : *un contrôle qu'on ne sait pas faire échouer ne contrôle rien.*

    Le remplaçant permute les étiquettes au hasard parmi les 130 questions v3. Il fait plus
    que saboter : sur 2 000 tirages il rend la distribution du plafond d'une route de dix
    questions **quelconques**, et situe le vrai plafond dedans. C'est un test de permutation
    de l'hypothèse « l'étiquette ``exact`` désigne les questions où l'élargissement paie ».
    """
    recalculee = recalcule(index, items, scores)
    v3 = [i for i, r in enumerate(recalculee) if r["bench"] == "v3"]
    labels = [recalculee[i]["famille"] for i in v3]
    rng = random.Random(20260901)
    plafonds = []
    for _ in range(tirages):
        melange = labels[:]
        rng.shuffle(melange)
        permutee = list(recalculee)
        for i, f in zip(v3, melange):
            permutee[i] = {**recalculee[i], "famille": f}
        plafonds.append(plafond_de(permutee, ROUTE_PRIMAIRE, CIBLE_PRIMAIRE)["plafond"])
    plafonds_tries = sorted(plafonds)
    distribution = {
        "tirages": tirages,
        "moyenne": statistics.mean(plafonds),
        "p50": plafonds_tries[tirages // 2],
        "p95": plafonds_tries[int(0.95 * tirages)],
        "p99": plafonds_tries[int(0.99 * tirages)],
        "max": plafonds_tries[-1],
        "p_value": sum(1 for x in plafonds if x >= vrai) / tirages,
    }
    print(f"      plafond réel {vrai:+.4f}")
    print(f"      dix questions quelconques : moyenne {distribution['moyenne']:+.4f} · "
          f"p50 {distribution['p50']:+.4f} · p95 {distribution['p95']:+.4f} · "
          f"p99 {distribution['p99']:+.4f} · max {distribution['max']:+.4f}")
    print(f"      part des permutations qui atteignent le vrai plafond : "
          f"{distribution['p_value']:.3f}")
    plaintes = []
    if distribution["moyenne"] >= vrai * 0.5:
        plaintes.append("le plafond survit à une permutation des étiquettes : "
                        "l'instrument ne lit pas les familles")
    return plaintes, distribution


def sabotage_S2(index: ChunkIndex, items: list[dict], scores: dict, archive: dict) -> list[str]:
    """Une question dont les scores sont inversés : le contrôle R doit hurler, sur elle seule."""
    attendu = {r["key"]: r for r in archive["per_question"]}
    cible = next(item["key"] for item in items
                 if item["key"] in scores and attendu.get(item["key"], {}).get(REFERENCE, {}).get("ndcg", 0) >= 0.9)
    truquee = dict(scores)
    truquee[cible] = {chunk: -note for chunk, note in scores[cible].items()}
    ecarts = controle_R(recalcule(index, items, truquee), archive)
    touchees = {e.split(" · ")[0] for e in ecarts}
    print(f"      question truquée : {cible} · {len(ecarts)} écarts sur {len(touchees)} question(s)")
    for ligne in ecarts[:4]:
        print(f"        {ligne}")
    plaintes = []
    if not ecarts:
        plaintes.append(f"le contrôle R n'a rien vu alors que {cible} est truquée")
    if touchees - {cible}:
        plaintes.append(f"le contrôle R accuse d'autres questions que {cible} : {sorted(touchees - {cible})}")
    return plaintes


# ---------------------------------------------------------------------------------- étapes

def charge():
    if not PARTIAL.exists():
        sys.exit(f"cache absent : {PARTIAL}")
    if not ARCHIVE.exists():
        sys.exit(f"archive absente : {ARCHIVE}")
    index = ChunkIndex.load()
    items = experiment.load_items(index)
    scores = json.loads(PARTIAL.read_text(encoding="utf-8"))
    archive = json.loads(ARCHIVE.read_text(encoding="utf-8"))
    return index, items, scores, archive


def step_mesure() -> None:
    index, items, scores, archive = charge()
    paires = sum(len(v) for v in scores.values())
    print(f"\n  corpus {archive['corpus']['chunks']} chunks · {archive['corpus']['documents']} documents "
          f"· signature {SIGNATURE}")
    print(f"  cache : {len(scores)} questions · {paires} paires scorées")

    recalculee = recalcule(index, items, scores)
    print(f"  {len(recalculee)} questions recalculées depuis le cache\n")

    print("  === contrôles, avant toute métrique ===")
    dur = False
    for nom, ecarts, exigence in (
            ("R — la référence recalculée reproduit l'archive", controle_R(recalculee, archive),
             f"{len(recalculee)}/{len(recalculee)} exact"),
            ("F — effectifs et moyennes par famille", controle_F(recalculee, archive), "au millième"),
            ("Z — router l'ensemble vide rend la référence", controle_Z(recalculee), "Δ ≡ 0"),
            ("P — l'estimateur précis est l'estimateur maison", controle_P(recalculee), "arrondi identique")):
        if ecarts:
            dur = True
            print(f"    ✗ {nom} — {len(ecarts)} écart(s)")
            for ligne in ecarts[:5]:
                print(f"        {ligne}")
        else:
            print(f"    ✓ {nom} ({exigence})")
    if dur:
        sys.exit("\nARRÊT : un contrôle a échoué. Aucune métrique n'est publiée.")

    primaire = plafond_de(recalculee, ROUTE_PRIMAIRE, CIBLE_PRIMAIRE)
    secondaire = plafond_de(recalculee, ROUTE_PRIMAIRE, CIBLE_SECONDAIRE)
    descriptives = [plafond_de(recalculee, route, CIBLE_PRIMAIRE) for route in ROUTES_DESCRIPTIVES]
    vue = clairvoyance(recalculee, CIBLE_PRIMAIRE)
    temps = latences(archive, ROUTE_PRIMAIRE, CIBLE_PRIMAIRE, recalculee)
    permutation_path = CACHE / f"routage-permutation-{SIGNATURE}.json"
    permutation = (json.loads(permutation_path.read_text(encoding="utf-8"))
                   if permutation_path.exists() else None)

    print(f"\n  === route primaire : exact → {LIBELLE[CIBLE_PRIMAIRE]} ===")
    print(f"    plafond {primaire['plafond']:+.4f}  IC95 [{primaire['ci95'][0]:+.4f} ; {primaire['ci95'][1]:+.4f}]"
          f"  ·  {primaire['n_routees']} questions routées")
    print(f"    mouvements : {primaire['gagnantes']} gagnantes, {primaire['perdantes']} perdantes")
    for m in primaire["mouvements"]:
        print(f"        {m['key']:<10} {m['delta']:+.4f}")

    print(f"\n  === secondaire descriptive : exact → {LIBELLE[CIBLE_SECONDAIRE]} ===")
    print(f"    plafond {secondaire['plafond']:+.4f}  IC95 [{secondaire['ci95'][0]:+.4f} ; {secondaire['ci95'][1]:+.4f}]"
          f"  ·  {secondaire['gagnantes']} gagnantes, {secondaire['perdantes']} perdantes")

    print("\n  === routes post-hoc, descriptives — aucune décision ne s'y appuie ===")
    for d in descriptives:
        print(f"    {'+'.join(d['route']):<32} {d['n_routees']:>3} questions   plafond {d['plafond']:+.4f}"
              f"  IC95 [{d['ci95'][0]:+.4f} ; {d['ci95'][1]:+.4f}]")

    if temps:
        print(f"\n  === latence — le routage protège la moyenne, pas la queue ===")
        for nom, cle in (("référence dense@50 + rr", "reference"),
                         ("candidat partout", "cible_partout"),
                         (f"ROUTÉ ({temps['part_routee']:.1%} du trafic)", "route")):
            t = temps[cle]
            print(f"    {nom:<28} p50 {t['p50']:>5.1f} s · p95 {t['p95']:>5.1f} s "
                  f"· max {t['max']:>5.1f} s · moyenne {t['moyenne']:>5.1f} s")

    print(f"\n  === plafond de clairvoyance (diagnostique) ===")
    print(f"    ne router que les questions qui y gagnent : {vue['plafond']:+.4f} "
          f"sur {vue['n_questions_gagnantes']} questions")
    if permutation:
        print(f"\n  === test de permutation (S1) — l'étiquette porte-t-elle l'information ? ===")
        print(f"    dix questions quelconques : moyenne {permutation['moyenne']:+.4f} · "
              f"p95 {permutation['p95']:+.4f} · max {permutation['max']:+.4f} "
              f"sur {permutation['tirages']} tirages")
        print(f"    p = {permutation['p_value']:.3f} — l'étiquette « exact » n'est pas dix "
              f"questions au hasard")

    if primaire["plafond"] <= SEUIL:
        verdict, motif = "NO-GO", ("le majorant lui-même est sous la barre : aucun détecteur "
                                   "ne peut faire mieux que le routeur oracle")
    elif primaire["plafond"] < SEUIL_AVEC_MARGE:
        verdict, motif = "NO-GO conditionnel", ("le majorant passe, mais sans la marge d'un "
                                                "détecteur imparfait")
    else:
        verdict, motif = "HOLD", "le plafond supporte un détecteur à 80 % de rappel"
    print(f"\n  === VERDICT : {verdict} ===")
    print(f"    plafond {primaire['plafond']:+.4f} contre seuil {SEUIL:+.4f} "
          f"(avec marge détecteur : {SEUIL_AVEC_MARGE:+.4f})")
    print(f"    {motif}")

    OUTPUT.write_text(json.dumps({
        "corpus": archive["corpus"],
        "protocole": {"pre_enregistrement": "15a4850", "reference": REFERENCE,
                      "cible_primaire": CIBLE_PRIMAIRE, "route_primaire": list(ROUTE_PRIMAIRE),
                      "seuil": SEUIL, "rappel_detecteur_suppose": RAPPEL_DETECTEUR,
                      "seuil_avec_marge": SEUIL_AVEC_MARGE,
                      "bootstrap": {"draws": 4000, "seed": 20260901},
                      "external_llm_calls": 0, "modeles_charges": [],
                      "paires_en_cache": paires, "questions": len(recalculee)},
        "controles": {"R": "155/155 exact", "F": "au millième", "Z": "Δ ≡ 0",
                      "P": "estimateur précis ≡ metrics.paired_delta après arrondi"},
        "primaire": primaire, "secondaire": secondaire, "latence": temps,
        "permutation": permutation,
        "descriptives": descriptives, "clairvoyance": vue,
        "verdict": {"verdict": verdict, "motif": motif, "plafond": primaire["plafond"]},
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  écrit : {OUTPUT.name}")


def step_sabotage() -> None:
    index, items, scores, archive = charge()
    recalculee = recalcule(index, items, scores)
    vrai = plafond_de(recalculee, ROUTE_PRIMAIRE, CIBLE_PRIMAIRE)["plafond"]
    print("\n  === S1 — rotation des étiquettes de famille ===")
    p1, distribution = sabotage_S1(index, items, scores, vrai)
    print("    " + ("✗ " + " · ".join(p1) if p1 else "✓ le plafond s'effondre quand les étiquettes sont fausses"))
    print("\n  === S2 — scores inversés sur une question ===")
    p2 = sabotage_S2(index, items, scores, archive)
    print("    " + ("✗ " + " · ".join(p2) if p2 else "✓ le contrôle R a hurlé, sur la bonne question"))
    if p1 or p2:
        sys.exit("\nARRÊT : un sabotage n'a pas fait échouer ce qu'il devait faire échouer.")
    print("\n  les deux contrôles savent échouer.")
    chemin = CACHE / f"routage-permutation-{SIGNATURE}.json"
    chemin.write_text(json.dumps(distribution, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  distribution de permutation écrite : {chemin.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step", required=True, choices=("mesure", "sabotage"))
    args = parser.parse_args()
    {"mesure": step_mesure, "sabotage": step_sabotage}[args.step]()


if __name__ == "__main__":
    main()
