"""Le fil « fenêtre générateur » — 1 600 contre 4 500 caractères par passage, apparié.

Instrument du pré-enregistrement ``PRE-ENREGISTREMENT-FENETRE-GENERATEUR-2026-09-08.md``,
Partie I publiée en ``f1dba73`` **avant la première réponse générée de ce fil**. Rien ici ne
décide en dehors de ce qui y est écrit : strates gelées, variable unique, quatre conditions
ensemble, issues nommées d'avance.

Trois bras, et le contexte est **identique** entre eux
------------------------------------------------------
Le pool vient une seule fois de ``.cache/router-retrievals-<signature>.json`` et le contexte
de ``pipeline.build_context`` — la fonction de production, 5 passages, 2 par document. **Seul
``characters`` change.**

===========  =================================================================
``ref``      ``characters = 1600`` — la ligne de base, ce que le banc sert aujourd'hui
``cand``     ``characters = 4500`` — la fenêtre depuis laquelle les questions ont été rédigées
``sabote``   ``characters = 4500``, mais le texte **au-delà du caractère 1 600** de chaque
             passage est remplacé par un texte de **même longueur** pris dans un autre chunk
             du même document. Le contexte grossit exactement autant, sans qu'aucun matériau
             qui répond soit restitué. C'est le contrôle négatif du mécanisme.
===========  =================================================================

Le juge voit **la fenêtre du bras qu'il note** (``judge.grade(characters=...)``) : à 1 400
caractères fixes, il noterait « non ancrée » une réponse correctement tirée du caractère 3 000,
et le biais irait contre le candidat.

Reprise et cache
----------------
Chaque réponse et chaque verdict sont écrits au fil de l'eau dans
``.cache/fenetre-{reponses,verdicts}-<signature>.json``. Une coupure ne coûte que l'appel en
cours ; ``llm`` met de son côté chaque appel en cache sur disque.

    .venv/bin/python rag/benchmark/eval_fenetre_generateur.py --etape mesure
    .venv/bin/python rag/benchmark/eval_fenetre_generateur.py --etape gardes
    .venv/bin/python rag/benchmark/eval_fenetre_generateur.py --etape verdict
"""
from __future__ import annotations

import argparse
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
import judge  # noqa: E402
import llm  # noqa: E402
import pipeline  # noqa: E402
from corpus import ChunkIndex  # noqa: E402

SIGNATURE = corpus_overlay.signature()

#: Un seul fournisseur est ouvert (Mistral à zéro sur le compte, mesuré le 8 septembre 2026).
#: L'écart à « le juge plus fort que le générateur » est assumé et déclaré au §9 du
#: pré-enregistrement ; sa contrepartie est la batterie d'items-témoins du juge.
MODELE = "gemini-3.1-flash-lite"

REF, CAND = 1600, 4500
BRAS = {"ref": REF, "cand": CAND, "sabote": CAND}
GRAINE = 20260908
#: Échantillon ABSENTS — règle écrite d'avance au §4 du pré-enregistrement.
ABSENTS = 20
#: Seuils du §6, dérivés avant la mesure. Voir le document pour leur dérivation.
GARDE_B_MAX, GARDE_A_MAX = 0, 2
#: Issue INDÉCIS du §8.
DELTA_PLANCHER = 0.25
TIRAGES = 10000

CACHE = HERE / ".cache"
STRATES = HERE / f"strates-fenetre-{SIGNATURE}.json"


def _charger(chemin: Path) -> dict:
    return json.loads(chemin.read_text(encoding="utf-8")) if chemin.exists() else {}


def _ecrire(chemin: Path, donnees: dict) -> None:
    chemin.parent.mkdir(parents=True, exist_ok=True)
    chemin.write_text(json.dumps(donnees, ensure_ascii=False), encoding="utf-8")


def population() -> tuple[dict, dict]:
    """Les strates gelées, et les questions v3 positives indexées par ``qid``."""
    strates = json.loads(STRATES.read_text(encoding="utf-8"))
    assert strates["signature"] == SIGNATURE, strates["signature"]
    items = {}
    for ligne in (HERE / "questions-v3.jsonl").read_text(encoding="utf-8").splitlines():
        if ligne.strip():
            item = json.loads(ligne)
            if item.get("kind") != "negative":
                items[item["qid"]] = item
    alea = random.Random(GRAINE)
    absents = sorted(strates["or_non_servi"])
    alea.shuffle(absents)
    strates["absents_echantillon"] = sorted(absents[:ABSENTS])
    return strates, items


def contexte_de(qid: str, cache: dict, index: ChunkIndex) -> list[dict]:
    """Le contexte servi, dérivé une seule fois — identique pour les trois bras."""
    dense = cache[f"v3/{qid}"]["dense"]
    classe = [{"chunk_id": c, "document_id": d} for c, d, _ in dense]
    choisis = pipeline.build_context(classe)
    rows = []
    for ligne in choisis:
        source = index.get(ligne["chunk_id"])
        if not source:
            continue
        fiche = index.metadata.get(source["document_id"]) or {}
        rows.append({"chunk_id": ligne["chunk_id"], "document_id": source["document_id"],
                     "text": source["text"], "section": source["section"],
                     "title": fiche.get("title") or index.title_of(source["document_id"]),
                     "short_ref": fiche.get("short_ref"), "page_start": source["page_start"]})
    return rows


def saboter(contexte: list[dict], index: ChunkIndex) -> list[dict]:
    """Remplace le texte au-delà de 1 600 c. par autant de caractères pris ailleurs.

    La source est un **autre chunk du même document**, choisi de façon déterministe (graine
    ``20260908``) ; si le document n'en offre pas d'assez long, le texte emprunté est répété
    jusqu'à la longueur voulue. La longueur du passage est conservée **exactement** : le
    contexte du bras saboté et celui du bras candidat ont la même taille, au caractère près.
    """
    alea = random.Random(GRAINE)
    out = []
    for ligne in contexte:
        texte = ligne["text"]
        manquant = len(texte) - REF
        if manquant <= 0:
            out.append(dict(ligne))
            continue
        voisins = [c for c in index.chunks_of(ligne["document_id"]) if c != ligne["chunk_id"]]
        emprunt = ""
        if voisins:
            source = index.get(sorted(voisins)[alea.randrange(len(voisins))])
            emprunt = (source or {}).get("text") or ""
        if not emprunt:
            emprunt = texte[:REF]                       # repli : jamais de passage raccourci
        while len(emprunt) < manquant:
            emprunt += emprunt
        out.append({**ligne, "text": texte[:REF] + emprunt[:manquant], "sabote": True})
    return out


def taches(strates: dict) -> list[tuple[str, str, str]]:
    """(strate, qid, bras) — l'ordre est déterministe, la reprise donc reproductible."""
    plan = []
    for nom, bras in (("tronquees", ("ref", "cand", "sabote")),
                      ("temoins", ("ref", "cand")),
                      ("milieu", ("ref", "cand")),
                      ("absents_echantillon", ("ref", "cand"))):
        for qid in strates[nom]:
            for b in bras:
                plan.append((nom, qid, b))
    return plan


def mesure() -> None:
    strates, items = population()
    index = ChunkIndex.load(verbose=False)
    cache_pool = json.loads((CACHE / f"router-retrievals-{SIGNATURE}.json").read_text(encoding="utf-8"))
    reponses = _charger(CACHE / f"fenetre-reponses-{SIGNATURE}.json")
    verdicts = _charger(CACHE / f"fenetre-verdicts-{SIGNATURE}.json")
    contextes: dict[str, list[dict]] = {}
    plan = taches(strates)
    print(f"{len(plan)} couples (question, bras) — {MODELE}, cache actif")
    for numero, (strate, qid, bras) in enumerate(plan, 1):
        cle = f"{qid}/{bras}"
        if cle in verdicts:
            continue
        if qid not in contextes:
            contextes[qid] = contexte_de(qid, cache_pool, index)
        contexte = contextes[qid] if bras != "sabote" else saboter(contextes[qid], index)
        fenetre = BRAS[bras]
        if cle not in reponses:
            sortie = pipeline.answer(items[qid]["question"], contexte, model=MODELE,
                                     characters=fenetre)
            reponses[cle] = {"strate": strate, "bras": bras, "fenetre": fenetre,
                             "answer": sortie["answer"], "abstained": sortie["abstained"]}
            _ecrire(CACHE / f"fenetre-reponses-{SIGNATURE}.json", reponses)
        verdict = judge.grade(items[qid], reponses[cle]["answer"], contexte,
                              seed=f"fenetre-{cle}", model=MODELE, characters=fenetre)
        verdicts[cle] = {"strate": strate, "bras": bras, **verdict}
        _ecrire(CACHE / f"fenetre-verdicts-{SIGNATURE}.json", verdicts)
        print(f"  {numero:4d}/{len(plan)}  {cle:16s} {strate:20s} "
              f"couverture {verdict.get('coverage')}", end="\r", flush=True)
    # Les compteurs sont ceux du PROCESSUS : l'étape « verdict » tourne dans un autre, et y
    # relire llm.stats() publiait « appels : 0 » pour un fil qui en a coûté deux cent cinquante.
    _ecrire(CACHE / f"fenetre-appels-{SIGNATURE}.json", llm.stats())
    print(f"\nfait : {len(verdicts)} verdicts, {llm.stats()}")


def gardes() -> None:
    """Les items-témoins du juge, exigés au §9 — sans eux aucun verdict n'est lisible.

    ``judge.build_traps(items, index)`` ne prend **pas** de modèle : il lit ``llm.GENERATOR``
    pour fabriquer les deux familles qui demandent une génération (``off_topic``,
    ``unsupported_fluent``), et ``judge.run_traps`` lit ``llm.JUDGE`` **à l'appel**. Les deux
    sont donc réglés ici, et remis en place après — c'est le piège que la docstring de
    ``judge.grade`` documente déjà, et le traverser sans le dire aurait fait tourner les
    témoins sur un modèle mort. Coût : 8 appels générateur + 18 appels juge = **26**.
    """
    index = ChunkIndex.load(verbose=False)
    tous = []
    for ligne in (HERE / "questions-v3.jsonl").read_text(encoding="utf-8").splitlines():
        if ligne.strip():
            tous.append(json.loads(ligne))
    avant = (llm.GENERATOR, llm.JUDGE)
    llm.GENERATOR = llm.JUDGE = MODELE
    try:
        pieges = judge.build_traps(tous, index)
        resultat = judge.run_traps(pieges)
    finally:
        llm.GENERATOR, llm.JUDGE = avant
    _ecrire(CACHE / f"fenetre-gardes-{SIGNATURE}.json",
            {"pieges": resultat, "modele": MODELE, "appels": llm.stats()})
    print(json.dumps({k: v for k, v in resultat.items() if k != "detail"},
                     ensure_ascii=False, indent=1))


#: Juge de contrôle — la configuration de référence du dépôt, redevenue possible le
#: 8 septembre 2026 avec une clé Mistral au quota ouvert.
JUGE_CONTROLE = "mistral-medium-latest"


def rejuger() -> None:
    """Re-note les **mêmes réponses** avec le juge de référence. Aucune génération.

    **Ce que ce contrôle NE PEUT PAS faire, et c'est déclaré avant de le lancer.** Le verdict
    du §8 a été rendu, il est publié, et il **reste le verdict de ce fil** : NO-GO. Re-noter
    avec un autre juge après avoir vu le résultat ne peut pas l'annuler — ce serait choisir son
    juge après coup, et le fil entier perdrait sa valeur.

    **Ce qu'il peut faire** : dire si ce NO-GO était un **artefact du juge**. Il se jouait sur
    deux marges d'épaisseur exactement nulle — une borne basse à +0,000 et un bras saboté à
    exactement la moitié —, donc il est, par construction, le verdict le plus sensible du
    dossier au bruit de notation. Si le juge de référence rend un contraste franc, cela ne
    change pas le NO-GO : cela dit que **le fil mérite d'être refait proprement**, avec le bon
    juge dès le départ, et c'est une information de décision.

    Le juge voit la **fenêtre du bras qu'il note**, comme au fil d'origine.
    """
    strates, items = population()
    reponses = _charger(CACHE / f"fenetre-reponses-{SIGNATURE}.json")
    verdicts = _charger(CACHE / f"fenetre-verdicts-{JUGE_CONTROLE}-{SIGNATURE}.json")
    index = ChunkIndex.load(verbose=False)
    cache_pool = json.loads((CACHE / f"router-retrievals-{SIGNATURE}.json").read_text(encoding="utf-8"))
    contextes: dict[str, list[dict]] = {}
    plan = sorted(reponses)
    for numero, k in enumerate(plan, 1):
        if k in verdicts:
            continue
        qid, bras = k.rsplit("/", 1)
        if qid not in contextes:
            contextes[qid] = contexte_de(qid, cache_pool, index)
        contexte = contextes[qid] if bras != "sabote" else saboter(contextes[qid], index)
        v = judge.grade(items[qid], reponses[k]["answer"], contexte,
                        seed=f"fenetre-{k}", model=JUGE_CONTROLE, characters=BRAS[bras])
        verdicts[k] = {"strate": reponses[k].get("strate"), "bras": bras, **v}
        _ecrire(CACHE / f"fenetre-verdicts-{JUGE_CONTROLE}-{SIGNATURE}.json", verdicts)
        print(f"  {numero:3d}/{len(plan)}  {k:18s} couverture {v.get('coverage')}", end="\r", flush=True)
    print(f"\nfait : {len(verdicts)} verdicts sous {JUGE_CONTROLE}, {llm.stats()}")
    _ecrire(CACHE / f"fenetre-appels-{JUGE_CONTROLE}-{SIGNATURE}.json", llm.stats())


def _ic_apparie(paires: list[tuple[float, float]]) -> dict:
    ecarts = [b - a for a, b in paires]
    if not ecarts:
        return {"n": 0}
    alea = random.Random(GRAINE)
    tirages = sorted(statistics.fmean(alea.choice(ecarts) for _ in ecarts) for _ in range(TIRAGES))
    return {"n": len(ecarts), "reference": round(statistics.fmean(a for a, _ in paires), 4),
            "candidat": round(statistics.fmean(b for _, b in paires), 4),
            "delta": round(statistics.fmean(ecarts), 4),
            "ic95": [round(tirages[int(0.025 * TIRAGES)], 4),
                     round(tirages[int(0.975 * TIRAGES)], 4)]}


def verdict(juge: str | None = None) -> dict:
    strates, items = population()
    suffixe = f"-{juge}" if juge else ""
    verdicts = _charger(CACHE / f"fenetre-verdicts{suffixe}-{SIGNATURE}.json")
    reponses = _charger(CACHE / f"fenetre-reponses-{SIGNATURE}.json")

    def couv(qid, bras):
        v = verdicts.get(f"{qid}/{bras}")
        return None if v is None else (v.get("coverage") or 0)

    def abst(qid, bras):
        r = reponses.get(f"{qid}/{bras}")
        return None if r is None else bool(r.get("abstained"))

    def paires(qids, bras="cand"):
        out = []
        for q in qids:
            a, b = couv(q, "ref"), couv(q, bras)
            if a is not None and b is not None:
                out.append((a, b))
        return out

    tronquees = strates["tronquees"]
    contraste = _ic_apparie(paires(tronquees))
    saboté = _ic_apparie(paires(tronquees, "sabote"))
    temoins = strates["temoins"]
    b = sum(1 for q in temoins if couv(q, "ref") == 2 and couv(q, "cand") == 0)
    pertes_qids = [q for q in temoins if couv(q, "ref") == 2 and couv(q, "cand") == 0]
    nouvelles = sum(1 for q in temoins if abst(q, "cand") and not abst(q, "ref"))
    disparues = sum(1 for q in temoins if abst(q, "ref") and not abst(q, "cand"))
    a = nouvelles - disparues

    conditions = {
        "contraste": bool(contraste.get("ic95") and contraste["ic95"][0] > 0),
        "garde_degradation": b <= GARDE_B_MAX,
        "garde_abstention": a <= GARDE_A_MAX,
        "sabotage": bool(contraste.get("delta") is not None and saboté.get("delta") is not None
                         and saboté["delta"] < contraste["delta"] / 2),
    }
    delta = contraste.get("delta")
    # L'ordre d'évaluation des issues du §8, rendu non ambigu. Le §8 les nomme sans les
    # ordonner ; à contraste nul, la condition de sabotage (« gagner nettement moins », soit
    # moins de la moitié du Δ candidat, opérationnalisée dans l'instrument publié en 1b939ca)
    # deviendrait ininterprétable et ferait lire « l'effet est la taille du contexte » là où
    # il n'y a aucun effet. Un Δ insuffisant prime donc sur tout le reste. Aucune règle du
    # §8 n'est modifiée : seule leur précédence est fixée, et elle l'est avant la mesure.
    if delta is None or contraste.get("n", 0) == 0:
        issue = "INCOMPLET"
    elif delta < DELTA_PLANCHER:
        issue = "NO-GO — la troncature n'est pas le mécanisme"
    elif not conditions["sabotage"]:
        issue = "NO-GO — l'effet est la taille du contexte, pas la restitution"
    elif all(conditions.values()):
        issue = "GO EXPÉRIMENTAL"
    elif conditions["contraste"]:
        issue = "HOLD — contraste acquis, garde échouée"
    else:
        issue = "INDÉCIS — sous-dimensionné, pas réfuté"

    rapport = {
        "signature": SIGNATURE, "modele": MODELE, "bras": BRAS,
        "pre_enregistrement": "PRE-ENREGISTREMENT-FENETRE-GENERATEUR-2026-09-08.md (f1dba73)",
        "contraste_tronquees": contraste,
        "bras_sabote_tronquees": saboté,
        "garde_degradation": {"b": b, "seuil": GARDE_B_MAX, "n": len(temoins),
                              "tenue": b <= GARDE_B_MAX, "pertes": pertes_qids},
        "garde_abstention": {"a": a, "nouvelles": nouvelles, "disparues": disparues,
                             "seuil": GARDE_A_MAX, "n": len(temoins), "tenue": a <= GARDE_A_MAX},
        "secondaires": {
            "milieu": _ic_apparie(paires(strates["milieu"])),
            "absents": _ic_apparie(paires(strates["absents_echantillon"])),
            "temoins": _ic_apparie(paires(temoins)),
        },
        "conditions": conditions,
        "issue": issue,
        "appels_de_la_mesure": _charger(CACHE / f"fenetre-appels-{SIGNATURE}.json"),
    }
    rapport["juge"] = juge or MODELE
    (HERE / f"results-fenetre-generateur-ab{suffixe}-{SIGNATURE}.json").write_text(
        json.dumps(rapport, ensure_ascii=False, indent=1), encoding="utf-8")
    return rapport


def imprimer(r: dict) -> None:
    c, s = r["contraste_tronquees"], r["bras_sabote_tronquees"]
    print(f"\nsignature {r['signature']}   modèle {r['modele']}   {r['pre_enregistrement']}")
    print(f"\nCONTRASTE — TRONQUÉES (n={c.get('n')})")
    if c.get("ic95"):
        print(f"   référence {c['reference']:.3f}   candidat {c['candidat']:.3f}   "
              f"Δ {c['delta']:+.3f}  IC95 [{c['ic95'][0]:+.3f} ; {c['ic95'][1]:+.3f}]")
    if s.get("ic95"):
        print(f"   bras saboté  {s['candidat']:.3f}   Δ {s['delta']:+.3f}  "
              f"IC95 [{s['ic95'][0]:+.3f} ; {s['ic95'][1]:+.3f}]")
    g, ga = r["garde_degradation"], r["garde_abstention"]
    print(f"\nGARDES — TÉMOINS (n={g['n']})")
    print(f"   dégradation stricte 2→0 : b = {g['b']}  seuil {g['seuil']}  "
          f"{'TENUE' if g['tenue'] else 'ÉCHOUÉE'}  {g['pertes'] or ''}")
    print(f"   abstention nette        : a = {ga['a']} ({ga['nouvelles']} nouvelles, "
          f"{ga['disparues']} disparues)  seuil {ga['seuil']}  "
          f"{'TENUE' if ga['tenue'] else 'ÉCHOUÉE'}")
    print("\nSECONDAIRES")
    for nom, v in r["secondaires"].items():
        if v.get("ic95"):
            print(f"   {nom:10s} n={v['n']:3d}  {v['reference']:.3f} → {v['candidat']:.3f}  "
                  f"Δ {v['delta']:+.3f}  IC95 [{v['ic95'][0]:+.3f} ; {v['ic95'][1]:+.3f}]")
    print(f"\nconditions : {r['conditions']}")
    print(f"ISSUE : {r['issue']}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--etape", choices=("mesure", "gardes", "rejuger", "verdict"), required=True)
    p.add_argument("--juge", default=None)
    a = p.parse_args()
    if a.etape == "mesure":
        mesure()
    elif a.etape == "gardes":
        gardes()
    elif a.etape == "rejuger":
        rejuger()
    else:
        imprimer(verdict(a.juge))


if __name__ == "__main__":
    main()
