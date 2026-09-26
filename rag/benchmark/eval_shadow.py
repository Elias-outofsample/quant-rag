"""Shadow — le reranker jugé sur des requêtes réelles, sans jamais être servi.

Le chantier ``RAPPORT-RERANKER-REPONSE-2026-09-05.md`` a rendu un **HOLD** : le contraste
décisif est acquis (Δ couverture +0,444), la garde de non-régression ne franchit pas sa
borne. Elle ne peut pas se fermer sur le banc — v3 n'a que 94 questions neutres. Le trafic
réel n'a pas cette limite, et c'est tout ce que ce script exploite.

**L'ombre observe, elle n'intercepte jamais.** La production sert le dense et continue de le
servir. ``quant_rag._log_decision`` journalisait déjà une ligne par requête ; trois champs y
ont été ajoutés (``signature``, ``served``, ``limit``) pour qu'un rejeu différé puisse
*prouver* qu'il rejoue le même corpus et reproduit les mêmes passages. Le reclassement, qui
coûte ~15 s à profondeur 50, ne tourne **jamais** en ligne : tout le bras reclassé est calculé
ici, après coup.

Fidélité : les deux contextes sont construits par ``quant_rag._select`` — la sélection de
**production** — et non par ``pipeline.build_context``, qui est celle du banc. Le shadow
mesure ce que la production sert, pas ce que le banc simule.

Protocole pré-enregistré et commité avant la première requête réelle observée (``4d5981c``) :
``rag/benchmark/RAPPORT-SHADOW-2026-09-05.md``. **Amendement n°1** (§8 à §12 du même
rapport), écrit compteur à zéro : la source du trafic et le juge sont Claude Code, en aveugle
et sous rubrique fixée avant le premier jugement ; deux conditions de comptabilisation en
sortent — la ligne de journal doit porter les trois champs (donc le serveur MCP doit être
postérieur à ``45cffdc``), et la requête doit avoir été servie en dense sans reranking.

    .venv/bin/python rag/benchmark/eval_shadow.py --step etat     # où en est le compteur
    .venv/bin/python rag/benchmark/eval_shadow.py --step ombre    # rejeu + les deux réponses
    .venv/bin/python rag/benchmark/eval_shadow.py --step kit      # le fichier à l'aveugle
    .venv/bin/python rag/benchmark/eval_shadow.py --step verdict  # après jugement
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
import llm  # noqa: E402
import pipeline  # noqa: E402
import quant_rag  # noqa: E402

SIGNATURE = corpus_overlay.signature()
JOURNAL = HERE.parent / "logs" / "router-decisions.jsonl"
OMBRE = HERE.parent / "logs" / f"shadow-{SIGNATURE}.json"
KIT = HERE.parent / "logs" / f"shadow-kit-{SIGNATURE}.md"
VERDICTS = HERE.parent / "logs" / f"shadow-verdicts-{SIGNATURE}.json"
RESULTAT = HERE / f"results-shadow-{SIGNATURE}.json"

GENERATEUR = "gemini-3.1-flash-lite"
GRAINE_ORDRE = 20260905

#: Palier 1 : valide le dispositif, mesure τ, porte la garde. Aucune décision primaire.
PALIER_1 = 50
#: Figée maintenant, sur la préférence *ancrée au banc* (58,4 %) — la recalculer sur les
#: données observées rendrait la règle adaptative. Seul τ est mesuré.
NON_EX_AEQUO_REQUIS = 131
#: Garde de dégradation stricte (profil v3/s10, v3/t11 : couverture 2 → 0). Banc : 1,54 %.
BORNE_DEGRADATION = 0.05
BORNE_DEGRADATION_IC = 0.10


# --------------------------------------------------------------------------------- journal

def entrees() -> tuple[list[dict], dict]:
    """Les lignes du journal exploitables par le shadow, et le compte des écarts.

    Une ligne d'avant le correctif n'a ni ``signature`` ni ``served`` : elle est écartée,
    pas devinée. Une ligne dont la signature diffère du corpus courant décrit un autre
    corpus — la rejouer mesurerait la dérive du corpus, pas celle du reclassement.

    Et une ligne servie autrement qu'en **dense sans reranking** est écartée aussi
    (amendement n°1, §8 du pré-enregistrement). Le contraste pré-enregistré oppose deux
    ordres du *même* pool dense ; une requête servie en ``hybrid`` opposerait BM25+fusion à
    dense+reclassement, soit deux variables à la fois. La règle d'usage du corpus prescrit
    l'hybride pour les requêtes d'identifiants exacts : ces requêtes-là existeront, elles
    sont légitimes, et elles ne sont simplement pas de ce chantier.
    """
    if not JOURNAL.exists():
        return [], {"journal_absent": True}
    ecarts = {"sans_champs_shadow": 0, "autre_signature": 0, "mode_non_dense": 0,
              "sans_resultat": 0, "doublon": 0}
    gardees, vues = [], set()
    for ligne in JOURNAL.read_text(encoding="utf-8").splitlines():
        if not ligne.strip():
            continue
        row = json.loads(ligne)
        if row.get("served") is None or row.get("signature") is None or row.get("limit") is None:
            ecarts["sans_champs_shadow"] += 1
            continue
        if row["signature"] != SIGNATURE:
            ecarts["autre_signature"] += 1
            continue
        if row.get("mode") != "dense" or row.get("rerank"):
            ecarts["mode_non_dense"] += 1
            continue
        if not row["served"]:
            ecarts["sans_resultat"] += 1
            continue
        cle = f"{row['timestamp']}|{row['query']}"
        if cle in vues:
            ecarts["doublon"] += 1
            continue
        vues.add(cle)
        row["cle"] = cle
        gardees.append(row)
    return gardees, ecarts


def filtres_de(row: dict) -> dict:
    f = dict(row.get("filters") or {})
    return {k: f.get(k) for k in ("document_id", "year_min", "year_max", "author")}


# ----------------------------------------------------------------------- palier 2 : l'ombre

def deux_bras(row: dict) -> dict | None:
    """Rejoue la requête, vérifie le rejeu, puis calcule le bras reclassé.

    Le rejeu part de la requête **telle que journalisée** — c'est-à-dire déjà dépouillée de
    sa clause de période si ``auto_period`` avait mordu — et des bornes d'années qui en sont
    sorties, avec ``auto_period=False``. Sans cela le rejeu ré-appliquerait la détection sur
    un texte dont la clause a déjà été retirée.
    """
    filtres = filtres_de(row)
    servi = quant_rag.search(row["query"], limit=row["limit"], mode=row.get("requested", "auto"),
                             log=False, auto_period=False, **filtres)
    if [r["chunk_id"] for r in servi] != row["served"]:
        return None
    pool = quant_rag.search(row["query"], limit=quant_rag.POOL, pool=quant_rag.POOL, mode="dense",
                            rerank=False, dedupe=False, per_document=0, min_characters=0,
                            log=False, auto_period=False, **filtres)
    reclasse = quant_rag._select(quant_rag._rerank(row["query"], pool), row["limit"],
                                 per_document=2, dedupe=True,
                                 min_characters=quant_rag.MIN_CHARACTERS)
    return {"reference": servi, "reclasse": reclasse}


def step_ombre(limite: int | None) -> None:
    gardees, ecarts = entrees()
    ombre = json.loads(OMBRE.read_text(encoding="utf-8")) if OMBRE.exists() else {}
    reste = [r for r in gardees if r["cle"] not in ombre]
    if limite:
        reste = reste[:limite]
    print(f"\n  journal : {len(gardees)} entrées exploitables · écarts {ecarts}")
    print(f"  {len(reste)} à traiter · {len(ombre)} déjà en ombre")
    if not reste:
        return
    rng_ordre = random.Random(GRAINE_ORDRE)
    echecs = 0
    for n, row in enumerate(reste, 1):
        bras = deux_bras(row)
        if bras is None:
            ombre[row["cle"]] = {"ecarte": "le rejeu ne reproduit pas les passages servis"}
            OMBRE.write_text(json.dumps(ombre, ensure_ascii=False), encoding="utf-8")
            print(f"      {n}/{len(reste)}  ÉCARTÉE — rejeu non conforme", flush=True)
            continue
        # Un 429 ne doit pas tuer la course : le quota journalier de flash-lite se ferme
        # autour de 320 appels, et le shadow en demande deux par requête. L'entrée fautive
        # est laissée hors de l'ombre — sans demi-écriture — et la relance la reprend.
        # Trouvé par la répétition à blanc, qui est faite pour ça.
        reponses = {}
        try:
            for nom, contexte in bras.items():
                produit = pipeline.answer(row["query"], contexte, model=GENERATEUR)
                reponses[nom] = {"answer": produit["answer"], "abstained": produit["abstained"],
                                 "context": [r["chunk_id"] for r in contexte]}
        except RuntimeError as erreur:
            echecs += 1
            print(f"      {n}/{len(reste)}  ÉCHEC API : {str(erreur)[:70]}", flush=True)
            if echecs >= 3:
                print("\n  trois échecs — arrêt. L'ombre est intacte, la relance reprend ici.")
                break
            continue
        # L'ordre A/B est tiré ici et consigné ici. Le kit ne le contient pas.
        inverse = rng_ordre.random() < 0.5
        ombre[row["cle"]] = {
            "timestamp": row["timestamp"], "query": row["query"], "signature": row["signature"],
            "limit": row["limit"], "filters": filtres_de(row), "reponses": reponses,
            "A": "reclasse" if inverse else "reference",
            "B": "reference" if inverse else "reclasse",
            "contextes_identiques": reponses["reference"]["context"] == reponses["reclasse"]["context"],
        }
        OMBRE.write_text(json.dumps(ombre, ensure_ascii=False), encoding="utf-8")
        print(f"      {n}/{len(reste)}  {row['timestamp']}  "
              f"{'contextes identiques' if ombre[row['cle']]['contextes_identiques'] else 'contextes différents'}"
              f"  · {row['query'][:60]}", flush=True)
    print(f"\n  {len(ombre)} entrées en ombre")


# ------------------------------------------------------------------------ palier 3 : le kit

def step_kit() -> None:
    ombre = json.loads(OMBRE.read_text(encoding="utf-8")) if OMBRE.exists() else {}
    jugeables = {k: v for k, v in ombre.items() if "reponses" in v}
    # L'en-tête dit ce qu'il faut faire, jamais pourquoi. Expliquer que « le reclassement
    # change les citations » orienterait le regard du juge vers les citations, ce qui est
    # exactement ce qu'un aveuglement doit empêcher. Le mécanisme vit dans le
    # pré-enregistrement, pas dans le kit.
    lignes = ["# Jugement à l'aveugle",
              "",
              f"*{len(jugeables)} requêtes, corpus `{SIGNATURE}`. Pour chacune : la question, "
              "puis deux réponses **A** et **B**, dans un ordre tiré au sort. Les passages "
              "qui les ont produites ne sont pas montrés.*",
              "",
              "**Pour chaque requête :** `A`, `B` ou `égal` — laquelle répond le mieux.",
              "**Si le verdict n'est pas `égal`** : la réponse *perdante* est-elle "
              "**inutilisable** — elle ne répond pas, ou répond faux, là où l'autre "
              "répond ? (`oui` / `non`)",
              "", "---", ""]
    for i, (cle, bloc) in enumerate(sorted(jugeables.items()), 1):
        lignes += [f"## {i}. `{cle.split('|')[0]}`", "",
                   f"**Question** — {bloc['query']}", ""]
        for etiquette in ("A", "B"):
            lignes += [f"### {etiquette}", "", bloc["reponses"][bloc[etiquette]]["answer"].strip(), ""]
        lignes += [f"**Verdict {i}** : `____`   ·   **perdante inutilisable** : `____`", "", "---", ""]
    KIT.write_text("\n".join(lignes), encoding="utf-8")
    print(f"\n  {len(jugeables)} requêtes · écrit : {KIT}")
    print(f"  (non commité : {KIT.parent.name}/ est gitignoré)")


# --------------------------------------------------------------------------------- verdict

def wilson(succes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Intervalle de Wilson — le bon estimateur pour une proportion à petit effectif.

    Le bootstrap de la maison sur une variable binaire à n = 13 rendrait des bornes qui
    collent aux observations ; Wilson ne dégénère pas quand le compte est nul ou plein,
    ce qui est exactement le cas de la garde de dégradation stricte.
    """
    if total == 0:
        return (0.0, 1.0)
    p = succes / total
    d = 1 + z ** 2 / total
    centre = (p + z ** 2 / (2 * total)) / d
    demi = z * ((p * (1 - p) / total + z ** 2 / (4 * total ** 2)) ** 0.5) / d
    return (max(0.0, centre - demi), min(1.0, centre + demi))


def step_verdict() -> None:
    ombre = json.loads(OMBRE.read_text(encoding="utf-8")) if OMBRE.exists() else {}
    if not VERDICTS.exists():
        sys.exit(f"aucun verdict : remplir le kit puis écrire {VERDICTS.name}\n"
                 '  format : {"<clé>": {"verdict": "A"|"B"|"egal", "inutilisable": true|false}}')
    rendus = json.loads(VERDICTS.read_text(encoding="utf-8"))
    apparies = [(cle, v, ombre[cle]) for cle, v in rendus.items() if cle in ombre and "reponses" in ombre[cle]]
    n = len(apparies)
    pour = {"reclasse": 0, "reference": 0}
    degradations = 0
    for cle, v, bloc in apparies:
        if v["verdict"] == "egal":
            continue
        gagnant = bloc[v["verdict"]]
        perdant = "reference" if gagnant == "reclasse" else "reclasse"
        pour[gagnant] += 1
        if v.get("inutilisable") and perdant == "reclasse":
            degradations += 1
    non_ex_aequo = pour["reclasse"] + pour["reference"]
    tau = 1 - non_ex_aequo / n if n else None
    print(f"\n  === shadow — {n} requêtes jugées ===")
    print(f"    ex æquo : {n - non_ex_aequo}/{n}   (τ = {tau:.3f})" if n else "")
    print(f"    préfèrent le reclassé : {pour['reclasse']} · le dense : {pour['reference']}")

    if non_ex_aequo:
        part = pour["reclasse"] / non_ex_aequo
        bas, haut = wilson(pour["reclasse"], non_ex_aequo)
        print(f"    part de préférence pour le reclassé : {part:.3f}  IC95 [{bas:.3f} ; {haut:.3f}]")
    dbas, dhaut = wilson(degradations, n)
    print(f"    dégradations strictes du reclassé : {degradations}/{n} = {degradations / max(n,1):.3f}"
          f"  IC95 [{dbas:.3f} ; {dhaut:.3f}]  (borne {BORNE_DEGRADATION:.2f}, IC < {BORNE_DEGRADATION_IC:.2f})")

    n2 = None
    if tau is not None and tau < 1:
        n2 = -(-NON_EX_AEQUO_REQUIS // max(1e-9, 1 - tau))
        print(f"\n    n₂ = ⌈{NON_EX_AEQUO_REQUIS} / (1 − τ)⌉ = {int(n2)} requêtes pour la lecture primaire")

    garde = dhaut < BORNE_DEGRADATION_IC and degradations / max(n, 1) < BORNE_DEGRADATION
    if n < PALIER_1:
        issue = f"PALIER 1 INCOMPLET — {n}/{PALIER_1}"
    elif n2 and n < n2:
        issue = ("PALIER 1 RENDU — garde franchie, décision primaire en attente de n₂"
                 if garde else "PALIER 1 RENDU — GARDE NON FRANCHIE")
    elif non_ex_aequo and wilson(pour["reclasse"], non_ex_aequo)[0] > 0.5 and garde:
        issue = "GO"
    else:
        issue = "NO-GO / non concluant à la lecture primaire unique"
    print(f"\n  === ISSUE : {issue} ===")

    RESULTAT.write_text(json.dumps({
        "corpus": {"signature": SIGNATURE},
        "protocole": {"pre_enregistrement": "4d5981c",
                      "amendement_1": "RAPPORT-SHADOW-2026-09-05.md §8–§12 — trafic et juge : "
                                      "Claude Code ; dense seul ; serveur postérieur à 45cffdc",
                      "palier_1": PALIER_1,
                      "non_ex_aequo_requis": NON_EX_AEQUO_REQUIS,
                      "borne_degradation": BORNE_DEGRADATION,
                      "borne_degradation_ic": BORNE_DEGRADATION_IC,
                      "graine_ordre": GRAINE_ORDRE, "generateur": GENERATEUR},
        "n": n, "tau": tau, "preferences": pour, "non_ex_aequo": non_ex_aequo,
        "degradations_strictes": degradations, "n2": int(n2) if n2 else None,
        "garde_franchie": garde, "issue": issue,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  écrit : {RESULTAT.name}")


def step_etat() -> None:
    gardees, ecarts = entrees()
    ombre = json.loads(OMBRE.read_text(encoding="utf-8")) if OMBRE.exists() else {}
    prets = sum(1 for v in ombre.values() if "reponses" in v)
    print(f"\n  corpus {SIGNATURE} · journal {JOURNAL}")
    print(f"    entrées exploitables par le shadow : {len(gardees)}")
    print(f"    écartées : {ecarts}")
    print(f"    en ombre (deux réponses produites) : {prets}")
    print(f"\n  palier 1 : {prets}/{PALIER_1} requêtes réelles")
    print(f"    il en manque {max(0, PALIER_1 - prets)}. Elles arrivent par l'usage, pas par un script.")
    if ecarts.get("sans_champs_shadow"):
        print(f"\n  note : {ecarts['sans_champs_shadow']} lignes antérieures au correctif du journal "
              f"n'ont ni signature ni passages servis. Elles sont écartées, pas devinées.")
    _diagnostic_serveur()


def _diagnostic_serveur() -> None:
    """Dit à voix haute quand le serveur qui écrit le journal est périmé.

    Le compte des écarts ne suffit pas : 276 lignes sans champs shadow se lisent comme un
    héritage inoffensif, alors que la *dernière* ligne sans champs dit tout autre chose —
    que le processus MCP en cours d'exécution sert le code d'avant le correctif, et
    qu'aucune requête posée maintenant ne comptera. Le journal est le seul endroit d'où ce
    fait est observable : le serveur, lui, ne dit pas son âge.
    """
    if not JOURNAL.exists():
        return
    lignes = [l for l in JOURNAL.read_text(encoding="utf-8").splitlines() if l.strip()]
    if not lignes:
        return
    derniere = json.loads(lignes[-1])
    horodatage = derniere.get("timestamp", "?")
    if derniere.get("served") is None:
        print(f"\n  ATTENTION — la dernière ligne du journal ({horodatage}) n'a pas les champs "
              f"du shadow.\n    Le serveur MCP en cours d'exécution est antérieur au correctif "
              f"(commit 45cffdc).\n    Toute requête posée avant son redémarrage est perdue pour "
              f"le palier 1.")
    else:
        print(f"\n  dernière ligne du journal : {horodatage} — champs du shadow présents.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step", required=True, choices=("etat", "ombre", "kit", "verdict"))
    parser.add_argument("--limit", type=int, help="pilote : n premières entrées")
    args = parser.parse_args()
    {"etat": lambda: step_etat(), "ombre": lambda: step_ombre(args.limit),
     "kit": lambda: step_kit(), "verdict": lambda: step_verdict()}[args.step]()


if __name__ == "__main__":
    main()
