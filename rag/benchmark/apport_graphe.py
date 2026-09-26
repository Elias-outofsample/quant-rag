"""L'apport du graphe d'entités **aux réponses** — la seule question qu'on n'ait jamais posée.

L'antériorité, donnée avant le protocole
-----------------------------------------
Le §10 du ``docs/TODO.md`` a mesuré le graphe **comme générateur de candidats** et rendu
**NO-GO** : « graphe 1 saut, cap 10 » récupère 13/27 pour **258 candidats par question** ;
« graphe 2 sauts, sans plafond » 23/27 pour **13 255 candidats, soit 60 % du corpus** ; à
plafond serré, le second saut ne rapporte **rien** (13 → 13). Le §14 range l'élargissement de
pool parmi les sous-espaces fermés.

**Mesurer le rang de l'or ramené par le graphe, c'est du rappel — donc ce sous-espace, déjà
fermé. Ce module ne le fait pas.** Le refaire à l'identique redonnerait le même NO-GO.

Ce qui n'a jamais été mesuré, et pourquoi c'est admissible
-----------------------------------------------------------
Le §14 nomme quatre choses qui comptent comme information nouvelle. La troisième est **« une
mesure au niveau réponse qui contredit une mesure de classement »**. C'est exactement la
catégorie de ce qui suit. La question « le graphe aide-t-il les **réponses** ? » n'a jamais
été posée, alors que le graphe coûte **61 % du temps d'ingestion** et que la décision de le
garder en dépend.

**Ce que ce résultat peut changer, borné d'avance : rien n'est branché en production.** La
mesure alimente la décision « garder ou jeter le graphe », qui appartient à l'utilisateur.
Aucun GO ne sort d'ici vers le chemin servi ; le §14 reste intact.

Le bras qui rend la mesure lisible
-----------------------------------
Trois bras, pas deux. Sans le **placebo** — trois passages de plus, pris au pool dense aux
rangs 6 à 8 —, un gain du bras « avec graphe » se confondrait avec « on a donné plus de
contexte », et on aurait attribué au graphe ce que n'importe quel passage supplémentaire
aurait produit. Le fil `characters` a déjà montré que plus de contexte *par passage* vaut
+0,0308 [−0,061 ; +0,123], c'est-à-dire rien de démontré ; le placebo mesure ici plus de
contexte *en passages*. Le contraste décisif est donc **avec_graphe − placebo**, et le
contraste naïf **avec_graphe − sans_graphe** est publié à côté, pour qu'on voie la différence
que le placebo fait.

L'amorçage, déclaré d'avance
-----------------------------
Correspondance **déterministe** question ↔ entités : les n-grammes de 1 à 4 mots de la
question sont pliés (``graph_search.fold``) et cherchés par **égalité exacte** dans l'index
des noms d'entités (``Graph.by_key``). Aucun LLM, aucun plongement, aucune sous-chaîne — le
§10 amorçait autrement (``Graph.find`` fait de la sous-chaîne, ce qui ramenait « volatility »
sur toute question qui contient ce mot). C'est l'information neuve nommée.
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import banc_v4  # noqa: E402
import corpus_overlay  # noqa: E402
import familles_v4  # noqa: E402
import graph_search  # noqa: E402
import judge  # noqa: E402
import llm  # noqa: E402
import pipeline  # noqa: E402

SIGNATURE = corpus_overlay.signature()
GENERATEUR, JUGE = banc_v4.GENERATEUR, banc_v4.JUGE
FENETRE_SERVIE = banc_v4.FENETRE_SERVIE
FENETRE_JUGE = banc_v4.FENETRE_JUGE

#: Population déclarée au pré-enregistrement : 40 questions, stratifiées en deux moitiés.
CIBLE_TOTALE, CIBLE_PAR_STRATE = 40, 20
#: Passages ajoutés par le bras graphe et par le placebo. Le même nombre des deux côtés —
#: sinon les deux bras ne diffèreraient pas que par la provenance.
AJOUTS = 3
#: Plafond de chunks par entité, repris du §10 (« graphe 1 saut, cap 10 ») pour que le
#: budget du bras soit celui que le §10 a déjà jugé abordable.
CAP_PAR_ENTITE = 10
#: Longueur maximale d'un n-gramme d'amorçage. Au-delà, aucun nom d'entité ne correspond.
NGRAMME_MAX = 4
GRAINE, TIRAGES = banc_v4.GRAINE, banc_v4.TIRAGES
PASSAGES = banc_v4.PASSAGES
CACHE = HERE / ".cache"

BRAS = ("sans_graphe", "avec_graphe", "placebo")


def _charger(chemin: Path) -> dict:
    return json.loads(chemin.read_text(encoding="utf-8")) if chemin.exists() else {}


def _ecrire(chemin: Path, contenu) -> None:
    chemin.parent.mkdir(parents=True, exist_ok=True)
    chemin.write_text(json.dumps(contenu, ensure_ascii=False, indent=1), encoding="utf-8")


# --------------------------------------------------------------------------- amorçage

def _ngrammes(question: str, maximum: int = NGRAMME_MAX) -> list[str]:
    mots = graph_search.fold(question).split()
    sortie = []
    for taille in range(min(maximum, len(mots)), 0, -1):
        for depart in range(len(mots) - taille + 1):
            sortie.append(" ".join(mots[depart:depart + taille]))
    return sortie


def amorcer(question: str, g) -> list[dict]:
    """Entités du graphe nommées **littéralement** par la question.

    Égalité exacte sur la forme pliée, jamais sous-chaîne : ``Graph.find`` — ce qu'utilise
    ``search_graph``, et ce que le §10 amorçait — ramène « stochastic volatility » sur toute
    question contenant « volatility », et c'est cette dilution qui produisait les 258
    candidats par question. Ici une entité n'est amorcée que si son nom **est** un n-gramme
    de la question.
    """
    vus, sortie = set(), []
    for ngramme in _ngrammes(question):
        for eid in g.by_key.get(ngramme, ()):
            if eid in vus:
                continue
            vus.add(eid)
            noeud = g.entities[eid]
            sortie.append({"id": eid, "name": noeud["name"], "label": noeud["label"],
                           "mentions": noeud["mentions"], "via": ngramme,
                           "chunks": len(noeud.get("source_chunks") or ())})
    sortie.sort(key=lambda e: (-len(e["via"].split()), -e["mentions"], e["name"]))
    return sortie


def passages_du_graphe(question: str, deja_servis: set[str], index, combien: int = AJOUTS) -> dict:
    """Jusqu'à ``combien`` passages ramenés par le graphe, jamais un déjà servi."""
    depart = time.perf_counter()
    g = graph_search.graph()
    entites = amorcer(question, g)
    poids, candidats = {}, set()
    for entite in entites:
        poids[entite["id"]] = 1.0
        chunks = list((g.entities[entite["id"]].get("source_chunks") or ()))[:CAP_PAR_ENTITE]
        candidats.update(chunks)
    candidats -= deja_servis
    classes = graph_search._rank_chunks(g, poids, candidats) if candidats else []
    retenus = []
    for chunk_id, score, _ in classes:
        ligne = index.get(chunk_id)
        if ligne is None or len((ligne.get("text") or "").strip()) < 250:
            continue
        fiche = index.metadata.get(ligne["document_id"]) or {}
        retenus.append({"chunk_id": chunk_id, "document_id": ligne["document_id"],
                        "title": fiche.get("title") or index.title_of(ligne["document_id"]),
                        "short_ref": fiche.get("short_ref"), "section": ligne.get("section"),
                        "page_start": ligne.get("page_start"), "text": ligne["text"],
                        "content_type": ligne.get("content_type"), "score": score})
        if len(retenus) >= combien:
            break
    return {"passages": retenus, "entites_amorcees": len(entites),
            "entites": entites[:6], "candidats": len(candidats),
            "secondes": round(time.perf_counter() - depart, 3)}


# --------------------------------------------------------------------------- population

def _servis(question: str, limite: int):
    import quant_rag  # noqa: E402
    return quant_rag.search(question, limit=limite, mode="dense", rerank=False,
                            auto_period=False, log=False)


def population() -> dict:
    """40 questions v3 positives, stratifiées **avant** la première réponse. Zéro appel LLM.

    La stratification est le contrôle exigé au §5.7 du handoff : sur les 20 questions dont
    l'or est déjà servi, le graphe ne peut rien *trouver* — ce qu'on y mesure est donc ce que
    son contexte **coûte**. Sur les 20 autres, il peut apporter. Mélanger les deux moitiés
    aurait rendu un chiffre moyen dont personne n'aurait su lire le signe.
    """
    from corpus import ChunkIndex  # noqa: E402

    index = ChunkIndex.load(verbose=False)
    positives = [q for q in familles_v4.lire_jsonl(HERE / "questions-v3.jsonl")
                 if q.get("kind") != "negative" and q.get("gold_chunks")]
    alea = random.Random(GRAINE)
    alea.shuffle(positives)

    avec_or, sans_or, contextes = [], [], {}
    for item in positives:
        if len(avec_or) >= CIBLE_PAR_STRATE and len(sans_or) >= CIBLE_PAR_STRATE:
            break
        rangs = _servis(item["question"], PASSAGES + AJOUTS)
        cinq = rangs[:PASSAGES]
        or_servi = bool({r["chunk_id"] for r in cinq} & set(item["gold_chunks"]))
        strate = avec_or if or_servi else sans_or
        if len(strate) >= CIBLE_PAR_STRATE:
            continue
        graphe = passages_du_graphe(item["question"], {r["chunk_id"] for r in cinq}, index)
        champs = ("chunk_id", "document_id", "title", "short_ref", "section",
                  "page_start", "text", "content_type", "score")
        contextes[item["qid"]] = {
            "question": item["question"],
            "or_servi": or_servi,
            "sans_graphe": [{k: r.get(k) for k in champs} for r in cinq],
            "placebo": ([{k: r.get(k) for k in champs} for r in cinq]
                        + [{k: r.get(k) for k in champs} for r in rangs[PASSAGES:PASSAGES + AJOUTS]]),
            "avec_graphe": [{k: r.get(k) for k in champs} for r in cinq] + graphe["passages"],
            "graphe": {k: v for k, v in graphe.items() if k != "passages"},
            "ajouts_graphe": len(graphe["passages"]),
            "ajouts_placebo": len(rangs[PASSAGES:PASSAGES + AJOUTS]),
        }
        strate.append(item["qid"])

    sortie = {"signature": SIGNATURE, "graine": GRAINE, "appels_llm": 0,
              "strates": {"or_servi": avec_or, "or_non_servi": sans_or},
              "n": len(avec_or) + len(sans_or),
              "contextes": contextes,
              "items": {q["qid"]: q for q in positives if q["qid"] in contextes}}
    _ecrire(CACHE / f"graphe-population-{SIGNATURE}.json", sortie)
    sans_apport = [q for q, c in contextes.items() if c["ajouts_graphe"] == 0]
    print(f"population {sortie['n']} — or servi {len(avec_or)}, or non servi {len(sans_or)}")
    print(f"questions pour lesquelles le graphe ne ramène AUCUN passage : {len(sans_apport)} "
          f"({100 * len(sans_apport) / max(sortie['n'], 1):.0f} %)  -> elles restent dans la "
          f"population, notées : c'est un zéro honnête")
    moyenne = statistics.fmean(c["graphe"]["entites_amorcees"] for c in contextes.values())
    print(f"entités amorcées par question : {moyenne:.1f} en moyenne")
    print(f"candidats du graphe par question : "
          f"{statistics.fmean(c['graphe']['candidats'] for c in contextes.values()):.1f} "
          f"(le §10 en comptait 258 à un saut, cap 10)")
    return sortie


# --------------------------------------------------------------------------- mesure

def mesure() -> None:
    temoins = _charger(CACHE / f"v4-temoins-{SIGNATURE}.json")
    if not temoins:
        sys.exit("items-témoins non mesurés — banc_v4.py --etape temoins d'abord")
    if not temoins.get("lisible"):
        sys.exit(f"{banc_v4.FAMILLE_BLOQUANTE} sous {banc_v4.SEUIL_FAMILLE} : rien ne serait lisible")

    pop = _charger(CACHE / f"graphe-population-{SIGNATURE}.json")
    if not pop:
        sys.exit("population non tirée — --etape population d'abord (elle ne coûte aucun appel)")

    reponses = _charger(CACHE / f"graphe-reponses-{SIGNATURE}.json")
    verdicts = _charger(CACHE / f"graphe-verdicts-{SIGNATURE}.json")
    registre: dict = {}
    # Question par question, et non bras par bras. L'ordre inverse — celui qui vient
    # naturellement — rend la mesure INUTILISABLE si elle s'interrompt : on se retrouve avec
    # quarante « sans_graphe » et zéro « placebo », donc aucun triplet, donc aucun contraste.
    # Trié par question, tout arrêt laisse N triplets complets et le verdict se calcule dessus.
    # Le total d'appels est le même ; c'est la valeur de ce qui est déjà payé qui change.
    plan = [(qid, bras) for qid in sorted(pop["contextes"]) for bras in BRAS]
    print(f"{len(plan)} couples (question, bras) — {GENERATEUR} / {JUGE} · "
          f"fenêtre servie {FENETRE_SERVIE} · fenêtre du juge {FENETRE_JUGE}")

    for numero, (qid, bras) in enumerate(plan, 1):
        cle = f"{qid}/{bras}"
        if cle in verdicts:
            continue
        contexte = pop["contextes"][qid][bras]
        if cle not in reponses:
            sortie, _ = banc_v4._facturer(registre, GENERATEUR, lambda: pipeline.answer(
                pop["contextes"][qid]["question"], contexte, model=GENERATEUR,
                characters=FENETRE_SERVIE))
            reponses[cle] = {"bras": bras, "answer": sortie["answer"],
                             "abstained": sortie["abstained"]}
            _ecrire(CACHE / f"graphe-reponses-{SIGNATURE}.json", reponses)
        v, _ = banc_v4._facturer(registre, JUGE, lambda: judge.grade(
            pop["items"][qid], reponses[cle]["answer"], contexte, seed=f"graphe-{cle}",
            model=JUGE, characters=FENETRE_JUGE))
        verdicts[cle] = {"bras": bras, **v}
        _ecrire(CACHE / f"graphe-verdicts-{SIGNATURE}.json", verdicts)
        print(f"  {numero:4d}/{len(plan)}  {cle:28s} couverture {v.get('coverage')}",
              end="\r", flush=True)

    _ecrire(CACHE / f"graphe-facturation-{SIGNATURE}.json",
            {"registre": registre, "chiffrage": banc_v4.chiffrer(registre)})
    print(f"\nfait — {llm.stats()}")
    print(json.dumps(banc_v4.chiffrer(registre), ensure_ascii=False, indent=1))


# --------------------------------------------------------------------------- verdict

def _ic(deltas: list[float]) -> list[float] | None:
    if not deltas:
        return None
    alea = random.Random(GRAINE)
    t = sorted(statistics.fmean(alea.choice(deltas) for _ in deltas) for _ in range(TIRAGES))
    return [round(t[int(0.025 * TIRAGES)], 4), round(t[int(0.975 * TIRAGES)], 4)]


#: Au-delà de ce nombre de mentions, une entité ne distingue plus rien dans ce corpus.
#: 200 mentions, c'est l'ordre de « price » ou « volatility » — des mots, pas des repères.
MENTIONS_GENERIQUE = 200


def _generiques(pop: dict, qids: list) -> dict:
    """Combien l'amorçage attrape-t-il de mots ordinaires plutôt que de vraies entités ?"""
    tetes, generiques, total = [], 0, 0
    for qid in qids:
        entites = pop["contextes"][qid]["graphe"].get("entites") or []
        if not entites:
            continue
        tetes.append(entites[0]["mentions"])
        generiques += sum(1 for e in entites if e["mentions"] >= MENTIONS_GENERIQUE)
        total += len(entites)
    return {
        "seuil_mentions": MENTIONS_GENERIQUE,
        "part_des_entites_de_tete_generiques": round(
            sum(1 for m in tetes if m >= MENTIONS_GENERIQUE) / len(tetes), 3) if tetes else None,
        "part_des_entites_generiques": round(generiques / total, 3) if total else None,
        "mentions_medianes_de_l_entite_de_tete": (
            round(statistics.median(tetes), 1) if tetes else None),
        "lecture": ("une entité à plus de 200 mentions n'est pas un repère, c'est un mot ; "
                    "si cette part est haute, le graphe est amorcé sur du vocabulaire et non "
                    "sur des objets, ce qui borne d'avance ce qu'il peut apporter"),
    }


def verdict() -> dict:
    pop = _charger(CACHE / f"graphe-population-{SIGNATURE}.json")
    verdicts = _charger(CACHE / f"graphe-verdicts-{SIGNATURE}.json")
    reponses = _charger(CACHE / f"graphe-reponses-{SIGNATURE}.json")
    temoins = _charger(CACHE / f"v4-temoins-{SIGNATURE}.json")
    facturation = _charger(CACHE / f"graphe-facturation-{SIGNATURE}.json")

    def couverture(qid, bras):
        v = verdicts.get(f"{qid}/{bras}")
        return None if v is None else (v.get("coverage") or 0)

    def abstenue(qid, bras):
        r = reponses.get(f"{qid}/{bras}")
        return None if r is None else bool(r.get("abstained"))

    qids = [q for q in sorted(pop.get("contextes", {})) if couverture(q, "sans_graphe") is not None]
    strates = {"tout": qids,
               "or_servi": [q for q in qids if pop["contextes"][q]["or_servi"]],
               "or_non_servi": [q for q in qids if not pop["contextes"][q]["or_servi"]]}

    par_bras = {}
    for bras in BRAS:
        mesures = [q for q in qids if couverture(q, bras) is not None]
        par_bras[bras] = {
            "n": len(mesures),
            "couverture": round(statistics.fmean(couverture(q, bras) for q in mesures), 4) if mesures else None,
            "abstentions": sum(1 for q in mesures if abstenue(q, bras)),
            "passages_moyens": round(statistics.fmean(
                len(pop["contextes"][q][bras]) for q in mesures), 2) if mesures else None,
        }

    contrastes = {}
    for nom, (a, b) in {"decisif_graphe_moins_placebo": ("avec_graphe", "placebo"),
                        "naif_graphe_moins_sans": ("avec_graphe", "sans_graphe"),
                        "temoin_placebo_moins_sans": ("placebo", "sans_graphe")}.items():
        bloc = {}
        for strate, membres in strates.items():
            couples = [q for q in membres
                       if couverture(q, a) is not None and couverture(q, b) is not None]
            deltas = [couverture(q, a) - couverture(q, b) for q in couples]
            nouvelles = [q for q in couples if abstenue(q, a) and not abstenue(q, b)]
            disparues = [q for q in couples if abstenue(q, b) and not abstenue(q, a)]
            ecart = len(nouvelles) - len(disparues)
            chutes = [q for q in couples if couverture(q, b) == 2 and couverture(q, a) == 0]
            bloc[strate] = {
                "n": len(deltas),
                "delta": round(statistics.fmean(deltas), 4) if deltas else None,
                "ic95": _ic(deltas),
                "montent": sum(1 for d in deltas if d > 0),
                "descendent": sum(1 for d in deltas if d < 0),
                "a": ecart, "garde_abstention": ecart <= banc_v4.GARDE_ABSTENTION,
                "b": len(chutes), "garde_degradation": len(chutes) <= banc_v4.GARDE_DEGRADATION,
            }
        contrastes[nom] = bloc

    decisif = contrastes["decisif_graphe_moins_placebo"]["tout"]
    if not temoins.get("lisible"):
        issue = "ILLISIBLE — les items-témoins du juge n'ont pas passé la condition"
    elif decisif["ic95"] and decisif["ic95"][0] > 0:
        issue = ("APPORT DÉMONTRÉ — le graphe fait mieux que trois passages denses de plus, "
                 "borne basse de l'IC95 au-dessus de zéro")
    elif decisif["ic95"] and decisif["ic95"][1] < 0:
        issue = "NUISIBLE — le graphe fait moins bien que le placebo, IC95 entièrement négatif"
    else:
        issue = ("AUCUN APPORT DÉMONTRÉ — l'IC95 du contraste décisif contient zéro ; "
                 "le graphe ne fait pas mieux que trois passages denses de plus")

    sans_apport = [q for q in qids if pop["contextes"][q]["ajouts_graphe"] == 0]
    sortie = {
        "signature": SIGNATURE, "generateur": GENERATEUR, "juge": JUGE,
        "fenetre_servie": FENETRE_SERVIE, "fenetre_du_juge": FENETRE_JUGE,
        "pre_enregistrement": "PRE-ENREGISTREMENT-INSTRUMENT-V4-2026-09-08.md §7",
        "anteriorite": {
            "section": "docs/TODO.md §10 et §14",
            "verdict_au_rang": "NO-GO — 13/27 pour 258 candidats/question à un saut cap 10 ; "
                               "23/27 pour 13 255 candidats (60 % du corpus) à deux sauts",
            "ce_qui_est_neuf": "mesure au niveau RÉPONSE (troisième des quatre informations "
                               "nouvelles admises par le §14), et amorçage par égalité exacte "
                               "de n-gramme, là où le §10 amorçait par sous-chaîne",
            "borne": "rien n'est branché en production ; le §14 reste intact",
        },
        "population": len(qids), "strates": {k: len(v) for k, v in strates.items()},
        # Le nombre de TRIPLETS complets, et non la population : les contrastes ne se
        # calculent que sur les questions dont les trois bras ont un verdict. Afficher la
        # population seule ferait lire « 40 » un résultat porté par cinq questions.
        "triplets_complets": sum(1 for q in qids
                                 if all(couverture(q, b) is not None for b in BRAS)),
        "mesure_complete": all(couverture(q, b) is not None for q in qids for b in BRAS),
        "questions_sans_apport_du_graphe": len(sans_apport),
        "temoins": {k: v for k, v in temoins.items() if k != "detail"},
        "par_bras": par_bras, "contrastes": contrastes, "issue": issue,
        "cout": facturation.get("chiffrage"),
        "amorcage": {
            "regle": "n-grammes de 1 à 4 mots, forme pliée, ÉGALITÉ exacte sur Graph.by_key",
            "cap_par_entite": CAP_PAR_ENTITE, "ajouts": AJOUTS,
            # Faiblesse **constatée après le tirage et publiée telle quelle**, sans toucher au
            # protocole : l'égalité exacte de n-gramme apparie surtout des mots ordinaires —
            # « price » (858 mentions, 496 chunks), « prices », « company ». Le graphe est alors
            # amorcé sur des entités qui ne distinguent rien, et ce qu'il ramène en dépend.
            # C'est une explication candidate d'un éventuel « aucun apport », et elle vaut
            # d'être chiffrée plutôt que corrigée après coup : un amorçage réglé une fois le
            # résultat vu ne serait plus une mesure.
            "entites_generiques": _generiques(pop, qids),
            "entites_moyennes": round(statistics.fmean(
                pop["contextes"][q]["graphe"]["entites_amorcees"] for q in qids), 2) if qids else None,
            "candidats_moyens": round(statistics.fmean(
                pop["contextes"][q]["graphe"]["candidats"] for q in qids), 1) if qids else None,
            "latence_p50_secondes": round(statistics.median(
                pop["contextes"][q]["graphe"]["secondes"] for q in qids), 3) if qids else None,
        },
    }
    (HERE / f"results-graphe-{SIGNATURE}.json").write_text(
        json.dumps(sortie, ensure_ascii=False, indent=1), encoding="utf-8")
    return sortie


def imprimer(r: dict) -> None:
    print(f"\nsignature {r['signature']} · {r['generateur']} / {r['juge']} · "
          f"fenêtre servie {r['fenetre_servie']} · fenêtre du juge {r['fenetre_du_juge']}")
    print(f"antériorité — {r['anteriorite']['section']} : {r['anteriorite']['verdict_au_rang']}")
    print(f"ce qui est neuf — {r['anteriorite']['ce_qui_est_neuf']}")
    print(f"borne — {r['anteriorite']['borne']}")
    t = r["temoins"]
    print(f"\nitems-témoins : exactitude {t.get('accuracy')} -> "
          f"{'LISIBLE' if t.get('lisible') else 'ILLISIBLE'}")
    print(f"population {r['population']} (or servi {r['strates']['or_servi']}, "
          f"or non servi {r['strates']['or_non_servi']}) · "
          f"{r['questions_sans_apport_du_graphe']} questions sans aucun apport du graphe")
    print(f"TRIPLETS COMPLETS : {r['triplets_complets']} / {r['population']}"
          + ("" if r["mesure_complete"] else "   ← mesure INCOMPLÈTE, les contrastes ne portent "
                                             "que sur ces triplets"))
    a = r["amorcage"]
    print(f"amorçage : {a['entites_moyennes']} entités/question, {a['candidats_moyens']} "
          f"candidats/question (le §10 : 258), p50 {a['latence_p50_secondes']} s")
    print(f"\n{'bras':14s} {'n':>4s} {'couverture':>11s} {'abst.':>6s} {'passages':>9s}")
    for bras, bloc in r["par_bras"].items():
        print(f"{bras:14s} {bloc['n']:4d} {bloc['couverture']:11.4f} {bloc['abstentions']:6d} "
              f"{bloc['passages_moyens']:9.2f}")
    for nom, bloc in r["contrastes"].items():
        print(f"\n{nom}")
        for strate, c in bloc.items():
            ic = f"[{c['ic95'][0]:+.3f} ; {c['ic95'][1]:+.3f}]" if c["ic95"] else "—"
            d = f"{c['delta']:+.4f}" if c["delta"] is not None else "—"
            print(f"  {strate:14s} n={c['n']:3d}  Δ {d:>8s}  {ic:>20s}  ↑{c['montent']}/↓{c['descendent']}"
                  f"  a={c['a']:+d} {'OK' if c['garde_abstention'] else 'ÉCHEC'}"
                  f"  b={c['b']} {'OK' if c['garde_degradation'] else 'ÉCHEC'}")
    cout = r.get("cout") or {}
    print(f"\ncoût : {cout.get('total_usd')} USD")
    print(f"\nISSUE : {r['issue']}")


def main() -> None:
    analyse = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    analyse.add_argument("--etape", choices=("population", "mesure", "verdict"), required=True)
    arguments = analyse.parse_args()
    if arguments.etape == "population":
        population()
    elif arguments.etape == "mesure":
        mesure()
    else:
        imprimer(verdict())


if __name__ == "__main__":
    main()
