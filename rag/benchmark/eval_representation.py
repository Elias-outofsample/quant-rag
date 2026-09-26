"""Le niveau réponse sur les 130 positives v3, à **corpus variable** et fenêtre fixe.

Pourquoi un module et pas ``eval_characters``
----------------------------------------------
``eval_characters`` pose une autre question — *quelle fenêtre servir* — et toute sa logique de
verdict en découle : ``REFERENCE = GRILLE[0]``, un gagnant parmi les paliers, un point de
bascule. Sa grille s'arrête d'ailleurs à 4 500, quand la production sert 10 000 depuis le lot 4.
La tordre pour lui faire comparer deux corpus laisserait un instrument dont le nom ment sur ce
qu'il mesure.

Ce module lui **emprunte** ce qui est validé et n'en réinvente rien : la population des 130
(``eval_characters.population``), les items-témoins, les deux gardes et leurs seuils calibrés à
n = 130, et l'intervalle par bootstrap. Il ne change que le contraste : **deux corpus, une
fenêtre**, appariés par question.

Le piège que ce module ferme d'avance
--------------------------------------
Le lot 4 a trouvé que le cache des verdicts du banc n'était **pas indexé par la fenêtre du
juge** : re-noter à 10 000 réutilisait des verdicts rendus à 2 500 et écrivait un fichier dont
l'en-tête mentait sur sa propre fenêtre. Ici la fenêtre entre dans la **clé** du cache et dans
chaque enregistrement ; un verdict rendu à une autre fenêtre est ignoré, jamais réutilisé.

Les caches sont nommés par la signature du corpus. Les deux bras ne peuvent donc pas se
mélanger : le monde servi écrit sous ``e1bdf36e2e``, le monde candidat sous le sien.

    QUANT_RAG_BUDGET_MAX_EUR=1.5 .venv/bin/python rag/benchmark/eval_representation.py --etape temoins
    QUANT_RAG_BUDGET_MAX_EUR=1.5 .venv/bin/python rag/benchmark/eval_representation.py --etape mesure
    .venv/bin/python rag/benchmark/eval_representation.py --etape verdict --reference e1bdf36e2e
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

sys.path.insert(0, str(HERE.parent / "ingestion"))

import contrat  # noqa: E402
import corpus_overlay  # noqa: E402
import eval_characters  # noqa: E402
import experiment  # noqa: E402
import judge  # noqa: E402
import llm  # noqa: E402
import pipeline  # noqa: E402

CACHE = HERE / ".cache"
SIGNATURE = corpus_overlay.signature()
GENERATEUR, JUGE = eval_characters.GENERATEUR, eval_characters.JUGE
#: La fenêtre servie du contrat, des deux côtés — celle du générateur et celle du juge. Elle
#: n'est pas une constante de ce module : la lire au contrat garantit qu'un changement de
#: produit ne laisse pas l'instrument mesurer une fenêtre que plus personne ne sert.
FENETRE = contrat.PASSAGE_CHARACTERS
#: Gardes et bootstrap : ceux d'``eval_characters``, calibrés à n = 130. Les redéfinir ici
#: serait ouvrir la porte à deux seuils divergents sous le même nom.
GARDE_ABSTENTION = eval_characters.GARDE_ABSTENTION
GARDE_DEGRADATION = eval_characters.GARDE_DEGRADATION
GRAINE, TIRAGES = 20260909, 10_000
#: Le bras du candidat. Un seul : la page a été retenue par non-infériorité au classement le
#: 7 septembre, et refaire le bras « sans page » coûterait un jeu de vecteurs pour une question
#: déjà tranchée.
BRAS = "c2"


def _charger(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def _ecrire(p: Path, d: dict) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")


def _monde(candidat: str | None):
    """Nomme le corpus mesuré, injecte sa vue, et refuse si la collection ne suit pas.

    **Le corpus mesuré se nomme, il ne se déguise pas.** On a d'abord essayé de « basculer le
    monde » — échanger l'overlay des tableaux pour que ``corpus_overlay.signature()`` rende la
    signature du candidat, et laisser tous les noms suivre. C'est **structurellement
    impossible** : la signature d'un candidat incorpore le ``sha256`` du ``chunks.jsonl``
    **re-découpé** de chaque document (``rechunk_corpus.construire``), là où
    ``corpus_overlay.signature`` lit les empreintes enregistrées au registre, qui sont celles
    du corpus servi. Aucun échange d'overlay ne peut les rendre égales.

    Trois choses doivent malgré tout désigner le même corpus, sinon la mesure est fausse en
    silence : le **nom** porté par les caches et les résultats, la **collection** interrogée,
    et la **vue corpus** qui compose le contexte. Les trois sont réglées ici, ensemble.
    """
    global SIGNATURE
    if not candidat:
        return None
    import quant_rag
    from build_collection_candidat import nom_collection
    from eval_dense_candidat import index_candidat

    attendue = nom_collection(candidat, BRAS)
    if quant_rag.COLLECTION != attendue:
        sys.exit(f"la collection interrogée est {quant_rag.COLLECTION}, pas {attendue} — "
                 f"exporter QUANT_RAG_COLLECTION={attendue}, sinon le classement viendrait du "
                 f"corpus servi et le contexte du candidat")
    SIGNATURE = candidat
    # ``eval_characters`` nomme SES caches — dont les items-témoins — par sa propre signature,
    # lue à l'import. Sans cette ligne, les témoins du candidat s'écriraient sous le nom du
    # corpus servi et **écraseraient ceux de la ligne de base**.
    eval_characters.SIGNATURE = candidat
    experiment.CACHE = experiment.CACHE.with_name(f"router-retrievals-{candidat}.json")
    return index_candidat(candidat)


def classements(candidat: str) -> None:
    """Le cache de classements du candidat, au format exact de ``calibrate_router``.

    Zéro appel LLM. Il est bâti ici plutôt que par ``calibrate_router`` parce que celui-ci
    compose sa vue avec ``ChunkIndex.load()`` — le corpus servi — et calibrerait donc un
    routeur sur un mélange des deux corpus.
    """
    import pipeline
    from eval_dense_candidat import index_candidat

    index = index_candidat(candidat)
    store = json.loads(experiment.CACHE.read_text(encoding="utf-8")) if experiment.CACHE.exists() else {}
    items = [json.loads(l) for l in (HERE / "questions-v3.jsonl").read_text(encoding="utf-8").splitlines()
             if l.strip()]
    positives = [i for i in items if i.get("kind") != "negative"]
    manquants = [i for i in positives if "dense" not in store.get(f"v3/{i['qid']}", {})]
    if not manquants:
        print(f"classements déjà en cache : {experiment.CACHE.name}")
        return
    print(f"{len(manquants)} classements à calculer sur {experiment.CACHE.name} (0 appel LLM)…")
    for n, item in enumerate(manquants, 1):
        ranked = pipeline.retrieve_item(item, "dense", index, limit=pipeline.POOL)
        store.setdefault(f"v3/{item['qid']}", {})["dense"] = [
            (r.get("chunk_id"), r.get("document_id"), float(r.get("score") or 0.0)) for r in ranked]
        print(f"  {n}/{len(manquants)}", end="\r", flush=True)
    experiment.CACHE.parent.mkdir(parents=True, exist_ok=True)
    experiment.CACHE.write_text(json.dumps(store), encoding="utf-8")
    print(f"\nécrit : {experiment.CACHE.name}")


def mesure(candidat: str | None = None, placebo: bool = False) -> None:
    """``placebo`` : le MÊME corpus, mais les réponses regénérées.

    Mesuré le 9 septembre 2026 et c'est la raison d'être de ce bras : re-générer les 130
    réponses de la ligne de base sur des prompts **identiques au bit près**, à température 0,
    en rend **51 sur 130 identiques** — 79 diffèrent, et **9 basculent en abstention**. Le
    générateur n'est pas reproductible, et la garde ``a ≤ 2`` est donc calibrée *sous* le
    plancher de bruit de l'instrument qu'elle surveille.

    Sans ce bras, un Δ de l'ordre de quelques centièmes se lit contre zéro et on lui attribue
    une cause. Avec lui, il se lit contre ce que le bruit seul produit. C'est la leçon du
    chantier « apport du graphe » (+0,025 en contraste naïf, **−0,075** contre placebo),
    appliquée là où elle manquait encore.
    """
    index = _monde(candidat)
    if placebo:
        if candidat:
            sys.exit("le placebo est un bras de la RÉFÉRENCE : il regénère sur le corpus servi, "
                     "pas sur un candidat")
        # Cache d'appels séparé : sans cela, la génération serait servie par le cache disque et
        # rendrait mot pour mot les réponses qu'on cherche précisément à ne pas réutiliser.
        llm.CACHE = CACHE / f"llm-placebo-{SIGNATURE}"
        llm.CACHE.mkdir(parents=True, exist_ok=True)
    temoins = _charger(CACHE / f"characters-temoins-{SIGNATURE}.json")
    if not temoins:
        sys.exit("items-témoins non mesurés — lancer --etape temoins d'abord")
    if not temoins.get("lisible"):
        sys.exit(f"{eval_characters.FAMILLE_BLOQUANTE} sous {eval_characters.SEUIL_FAMILLE} : "
                 f"aucun verdict ne serait lisible")

    nom = f"{SIGNATURE}-placebo" if placebo else SIGNATURE
    items, contextes = eval_characters.population(index)
    reponses = _charger(CACHE / f"representation-reponses-{nom}.json")
    verdicts = _charger(CACHE / f"representation-verdicts-{nom}.json")
    print(f"{len(items)} questions · fenêtre {FENETRE} des deux côtés · {GENERATEUR} / {JUGE}"
          f"{'  · BRAS PLACEBO (même corpus, réponses regénérées)' if placebo else ''}")

    for numero, qid in enumerate(sorted(items), 1):
        cle = f"{qid}/{FENETRE}"
        if cle in verdicts and verdicts[cle].get("fenetre_du_juge") == FENETRE:
            continue
        if cle not in reponses:
            sortie = pipeline.answer(items[qid]["question"], contextes[qid],
                                     model=GENERATEUR, characters=FENETRE)
            reponses[cle] = {"fenetre_servie": FENETRE, "answer": sortie["answer"],
                             "abstained": sortie["abstained"]}
            _ecrire(CACHE / f"representation-reponses-{nom}.json", reponses)
        v = judge.grade(items[qid], reponses[cle]["answer"], contextes[qid],
                        seed=f"representation-{cle}", model=JUGE, characters=FENETRE)
        verdicts[cle] = {"fenetre_du_juge": FENETRE, **v}
        _ecrire(CACHE / f"representation-verdicts-{nom}.json", verdicts)
        print(f"  {numero:4d}/{len(items)}  {qid:10s} couverture {v.get('coverage')}",
              end="\r", flush=True)
    _ecrire(CACHE / f"representation-appels-{nom}.json", llm.stats())
    print(f"\nfait : {len(verdicts)} verdicts, {llm.stats()}")


def _ic(deltas: list[float]) -> list[float] | None:
    if not deltas:
        return None
    alea = random.Random(GRAINE)
    t = sorted(statistics.fmean(alea.choice(deltas) for _ in deltas) for _ in range(TIRAGES))
    return [round(t[int(0.025 * TIRAGES)], 4), round(t[int(0.975 * TIRAGES)], 4)]


def _bras(signature: str) -> tuple[dict, dict]:
    verdicts = _charger(CACHE / f"representation-verdicts-{signature}.json")
    reponses = _charger(CACHE / f"representation-reponses-{signature}.json")
    if not verdicts:
        sys.exit(f"aucun verdict pour {signature} — le bras n'a pas été mesuré")
    etrangers = [c for c, v in verdicts.items() if v.get("fenetre_du_juge") != FENETRE]
    if etrangers:
        sys.exit(f"{len(etrangers)} verdicts de {signature} rendus à une autre fenêtre que "
                 f"{FENETRE} — les réutiliser ferait mentir l'en-tête (leçon du lot 4)")
    return verdicts, reponses


def verdict(reference: str) -> dict:
    v_cand, r_cand = _bras(SIGNATURE)
    v_ref, r_ref = _bras(reference)

    communs = sorted(set(v_cand) & set(v_ref))
    if not communs:
        sys.exit("aucune question commune aux deux bras")

    def couv(v, c):
        return v[c].get("coverage") or 0

    def abst(r, c):
        return bool((r.get(c) or {}).get("abstained"))

    deltas = [couv(v_cand, c) - couv(v_ref, c) for c in communs]
    nouvelles = [c for c in communs if abst(r_cand, c) and not abst(r_ref, c)]
    disparues = [c for c in communs if abst(r_ref, c) and not abst(r_cand, c)]
    chutes = [c for c in communs if couv(v_ref, c) == 2 and couv(v_cand, c) == 0]
    a, b = len(nouvelles) - len(disparues), len(chutes)
    ic = _ic(deltas)
    point = round(statistics.fmean(deltas), 4)

    gardes_ok = a <= GARDE_ABSTENTION and b <= GARDE_DEGRADATION
    if not gardes_ok:
        issue = "NO-GO — une garde échoue"
    elif ic and ic[0] < -0.10:
        issue = "NO-GO — la borne basse de l'IC95 passe sous −0,10"
    elif point >= -0.02:
        issue = "GO — non-infériorité tenue, aucune garde en défaut"
    else:
        issue = "DÉCISION UTILISATEUR — entre les deux règles pré-enregistrées"

    out = {
        "candidat": SIGNATURE, "reference": reference,
        "fenetre_servie": FENETRE, "fenetre_du_juge": FENETRE,
        "generateur": GENERATEUR, "juge": JUGE,
        "pre_enregistrement": "PRE-ENREGISTREMENT-REPRESENTATION-2026-09-09.md",
        "n": len(communs),
        "couverture_candidat": round(statistics.fmean(couv(v_cand, c) for c in communs), 4),
        "couverture_reference": round(statistics.fmean(couv(v_ref, c) for c in communs), 4),
        "delta": point, "ic95": ic,
        "montent": sum(1 for d in deltas if d > 0),
        "descendent": sum(1 for d in deltas if d < 0),
        "a": a, "nouvelles": nouvelles, "disparues": disparues,
        "garde_abstention": a <= GARDE_ABSTENTION,
        "b": b, "chutes": chutes,
        "garde_degradation": b <= GARDE_DEGRADATION,
        "seuils": {"abstention": GARDE_ABSTENTION, "degradation": GARDE_DEGRADATION},
        "graine": GRAINE, "tirages": TIRAGES,
        "issue": issue,
        "appels": _charger(CACHE / f"representation-appels-{SIGNATURE}.json"),
    }
    (HERE / f"results-representation-{SIGNATURE}-contre-{reference}.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return out


def imprimer(r: dict) -> None:
    ic = f"[{r['ic95'][0]:+.4f} ; {r['ic95'][1]:+.4f}]" if r["ic95"] else "—"
    print(f"\ncandidat {r['candidat']} contre référence {r['reference']}")
    print(f"fenêtre {r['fenetre_servie']} des deux côtés · {r['generateur']} / {r['juge']} · "
          f"n = {r['n']}")
    print(f"\ncouverture : référence {r['couverture_reference']:.4f} -> candidat "
          f"{r['couverture_candidat']:.4f}")
    print(f"Δ = {r['delta']:+.4f}   IC95 {ic}   ↑{r['montent']} / ↓{r['descendent']}")
    print(f"\ngarde abstention  a = {r['a']:+d}  (seuil {r['seuils']['abstention']})  "
          f"{'OK' if r['garde_abstention'] else 'ÉCHEC'}")
    print(f"garde dégradation b = {r['b']}   (seuil {r['seuils']['degradation']})  "
          f"{'OK' if r['garde_degradation'] else 'ÉCHEC'}")
    if r["chutes"]:
        print(f"  chutes 2->0 : {', '.join(r['chutes'])}")
    print(f"\nISSUE : {r['issue']}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--etape", choices=("temoins", "classements", "mesure", "verdict"), required=True)
    p.add_argument("--candidat", metavar="SIG",
                   help="mesure un corpus candidat : sa vue corpus est injectée, et le monde "
                        "doit avoir été basculé (monde_candidat.py --activer SIG)")
    p.add_argument("--reference", help="signature du bras de référence (étape verdict)")
    p.add_argument("--placebo", action="store_true",
                   help="bras placebo : MÊME corpus, réponses regénérées. Il donne le plancher "
                        "de bruit du générateur, sans lequel un Δ de quelques centièmes se lit "
                        "contre zéro au lieu de se lire contre le bruit.")
    a = p.parse_args()
    if a.etape == "temoins":
        eval_characters.temoins(_monde(a.candidat))
    elif a.etape == "classements":
        if not a.candidat:
            sys.exit("--candidat <signature> est requis pour construire un cache de classements")
        _monde(a.candidat)
        classements(a.candidat)
    elif a.etape == "mesure":
        mesure(a.candidat, a.placebo)
    else:
        if not a.reference:
            sys.exit("--reference <signature> est requis pour le verdict")
        _monde(a.candidat)
        if a.placebo:
            global SIGNATURE
            SIGNATURE = f"{SIGNATURE}-placebo"
        imprimer(verdict(a.reference))


if __name__ == "__main__":
    main()
