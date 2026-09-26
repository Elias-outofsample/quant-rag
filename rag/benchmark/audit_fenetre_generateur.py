"""La fenêtre de 1 600 caractères entre le passage servi et le générateur — hors ligne.

**Le fait qui ouvre ce fil.** Le générateur ne voit pas le passage servi : il en voit les
**1 600 premiers caractères** (``pipeline.format_passages``, l. 241, appelée sans surcharge
par ``pipeline.answer`` l. 259). Le juge en voit **1 400** (``judge.py`` l. 146). Les questions
du banc v3, elles, ont été rédigées à partir des **4 500 premiers** caractères du chunk d'or
(``generate_questions.CHUNK_CHARACTERS``). Et le plongement tronque encore ailleurs, à
**1 024 tokens** (``quant_rag.MAX_LENGTH``).

Il existe donc une fenêtre structurelle de 2 900 caractères dans laquelle le fait qui répond
peut se trouver **sans jamais atteindre le générateur**. Le §20 octies avait trouvé un
gradient de longueur du côté du *rappel* ; celui-ci est du côté de la *génération*, et il est
d'un autre mécanisme.

**Ce que ce module est.** Un diagnostic hors ligne, zéro appel LLM, aucune écriture en
production. Il relit les verdicts déjà payés (``results-e2e-*.json``) et mesure si la
couverture d'une réponse dépend de la position du matériau qui répond à l'intérieur du chunk
d'or servi.

**Ce qu'il n'est pas.** Un pré-enregistrement. Il ne décide aucune promotion. Il dit si un
pré-enregistrement d'A/B sur ``characters`` est justifié, et rien d'autre.

Aveu de reconnaissance, à porter avec le résultat
-------------------------------------------------
Une reconnaissance a été faite **avant** l'écriture de ce module, avec une méthode
*différente* (fenêtre glissante de 400 c., pas de 100, franchissement binaire de la coupe) sur
trois runs. Elle a rendu 73,3 % contre 41,7 %, 49,1 % contre 33,3 %, et 68,1 % contre 66,7 % —
deux runs concordants, un plat. L'auteur de ce module connaît donc l'ordre de grandeur et le
sens attendus. Les protections déclarées ci-dessous existent pour ça, et la méthode de la
reconnaissance est rejouée en sensibilité (``fenetre_glissante``) pour que l'écart entre les
deux lectures soit visible et non caché.

La méthode, déclarée avant la première mesure
---------------------------------------------
1. **Mesure continue, sans seuil à choisir.** Pour chaque chunk d'or *effectivement servi*,
   on tokenise son texte en gardant les positions, on retient les tokens qui appartiennent au
   vocabulaire des ``answer_facts`` de la question, et on leur donne le poids ``1/(1+df)`` de
   ``ChunkIndex.df`` — un mot rare pèse, un mot banal non. ``part_dehors`` est la part de cette
   masse située **au-delà du caractère 1 600**. Aucun seuil n'entre dans sa définition.
2. **Le contrôle qui décide, et c'est le seul qui vaille.** ``part_dehors > 0`` exige
   ``len(texte) > 1 600`` : comparer « tronqué » à « non tronqué » compare aussi *long* à
   *court*, et la longueur est déjà connue pour prédire l'échec (§20 octies). L'analyse
   **primaire est donc restreinte aux chunks d'or servis de plus de 1 600 caractères**, tous
   longs, et regarde la couverture **en fonction de ``part_dehors`` seul**.
3. **Bornes déclarées** : ``[0 ; 0,001[``, ``[0,001 ; 0,25[``, ``[0,25 ; 0,5[``,
   ``[0,5 ; 0,75[``, ``[0,75 ; 1]``. Ce qui est lu est la **monotonie** de la couverture le
   long de ces bornes, pas un contraste entre deux cases choisies après coup.
4. **Agrégation d'une question à plusieurs ors** : ``part_dehors`` de la question est le
   **maximum** sur ses chunks d'or servis — il suffit qu'une part du matériau soit hors de vue
   pour que la réponse soit amputée. Le **minimum** est publié en sensibilité.
5. **Strates** : chaque couple (fichier de résultats, configuration) est une strate. Une même
   question apparaît dans plusieurs strates : l'intervalle poolé est obtenu par **bootstrap
   groupé par ``qid``** (2 000 tirages, graine 20260908), jamais par une binomiale qui
   supposerait l'indépendance.

Les contrôles, tous obligatoires
--------------------------------
- **négatif** : les questions dont l'or **n'est pas** servi. La couverture y est basse par
  construction et doit être **plate** le long de ``part_dehors`` ; sinon la variable capte
  autre chose que la troncature ;
- **confusion de type** : ventilation par ``kind`` (``single``/``table``/``multi``/``dated``/
  ``exact``) et par présence d'un overlay de tableau — la longueur du chunk corrèle avec le
  type de document ;
- **confusion de longueur** : ventilation par longueur du chunk d'or à l'intérieur même de la
  strate « > 1 600 » ;
- **confusion de rang** : position du passage d'or dans le prompt (1 à 5) — un or au rang 5 est
  lu après 4 autres passages, et c'est un mécanisme concurrent ;
- **sensibilité** : la méthode de la reconnaissance (fenêtre glissante) et l'agrégation par
  minimum.

Les issues, nommées avant de regarder
-------------------------------------
- **pas de tendance monotone, ou tendance absorbée par un contrôle** → la troncature n'est pas
  le mécanisme. Le fil se ferme, et c'est un chantier économisé ;
- **tendance monotone, contrôle négatif plat** → un pré-enregistrement d'A/B sur
  ``characters`` est justifié. Il est écrit, commité, et **la mesure n'est pas lancée sans
  quota** ;
- **tendance présente mais n insuffisant** → chiffrer n et l'écart, dire ce qu'il faudrait,
  s'arrêter là.

Correction apportée après le premier passage, et déclarée ici
------------------------------------------------------------
Le contrôle négatif du premier passage était **vide** : la part n'était calculée que sur les
ors *servis*, or le contrôle porte précisément sur les questions dont l'or **n'est pas** servi.
C'était un défaut de l'instrument, pas un résultat. ``part_max_or`` la calcule désormais sur
tous les chunks d'or — c'est une propriété du chunk, pas de son service. Aucune borne, aucune
population primaire, aucune issue n'a bougé. Les ventilations, elles, passent des cinq bornes
déclarées à **deux cases** (``< 0,25`` / ``>= 0,25``) : leurs effectifs sont trop creux pour
cinq, et un ``n/a`` cacherait l'information. **L'analyse primaire garde ses cinq bornes.**

Invariant : ``characters`` est un paramètre du **banc**. Le chemin servi n'est pas touché.

    .venv/bin/python rag/benchmark/audit_fenetre_generateur.py
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import math
import os
import random
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import corpus_overlay  # noqa: E402
import pipeline  # noqa: E402
from corpus import ChunkIndex  # noqa: E402

#: La coupe réellement appliquée au générateur (``pipeline.format_passages``, défaut).
COUPE = 1600
#: Bornes déclarées avant la première mesure.
BORNES = (0.0, 0.001, 0.25, 0.5, 0.75, 1.0001)
#: Bootstrap groupé par question.
TIRAGES, GRAINE = 2000, 20260908
#: Sensibilité — la méthode de la reconnaissance.
FENETRE, PAS = 400, 100

_MOT = re.compile(r"[A-Za-z][A-Za-z0-9\-]+")


def mots_positionnes(texte: str) -> list[tuple[str, int]]:
    return [(m.group(0).casefold(), m.start()) for m in _MOT.finditer(texte or "")]


def vocabulaire(faits: list[str]) -> set[str]:
    out: set[str] = set()
    for f in faits or ():
        out.update(m.group(0).casefold() for m in _MOT.finditer(f))
    return out


def part_dehors(texte: str, faits: list[str], df: dict, coupe: int = COUPE) -> float | None:
    """Part de la masse IDF du matériau qui répond située **au-delà de la coupe**.

    ``None`` si aucun mot des faits n'est retrouvé dans le chunk — la mesure n'a alors pas
    d'objet, et la question est comptée à part plutôt que rangée dans une case par défaut.
    """
    cible = vocabulaire(faits)
    if not cible:
        return None
    dedans = dehors = 0.0
    for mot, pos in mots_positionnes(texte):
        if mot not in cible:
            continue
        poids = 1.0 / (1.0 + df.get(mot, 0))
        if pos < coupe:
            dedans += poids
        else:
            dehors += poids
    total = dedans + dehors
    return None if total <= 0 else dehors / total


def fenetre_glissante(texte: str, faits: list[str], df: dict, coupe: int = COUPE) -> bool | None:
    """Sensibilité — la méthode de la reconnaissance, rejouée telle quelle."""
    cible = collections.Counter()
    for f in faits or ():
        cible.update(m.group(0).casefold() for m in _MOT.finditer(f))
    if not cible:
        return None
    poids = {m: 1.0 / (1.0 + df.get(m, 0)) for m in cible}
    meilleur = (-1.0, 0)
    for depart in range(0, max(1, len(texte) - FENETRE + 1), PAS):
        presents = {m for m, _ in mots_positionnes(texte[depart:depart + FENETRE])}
        score = sum(poids[m] for m in cible if m in presents)
        if score > meilleur[0]:
            meilleur = (score, depart)
    return meilleur[1] + FENETRE > coupe


def borne_de(valeur: float) -> int:
    for i in range(len(BORNES) - 1):
        if BORNES[i] <= valeur < BORNES[i + 1]:
            return i
    return len(BORNES) - 2


def wilson(k: int, n: int, z: float = 1.959963985) -> tuple[float, float]:
    if not n:
        return (0.0, 1.0)
    p, d = k / n, 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    demi = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - demi), min(1.0, centre + demi))


def questions() -> dict[str, dict]:
    """Les deux bancs, indexés par ``qid`` — v1 n'a pas d'``answer_facts``, et c'est déclaré."""
    out = {}
    for chemin in (HERE / "questions-v1.jsonl", HERE / "questions-v3.jsonl"):
        for ligne in chemin.read_text(encoding="utf-8").splitlines():
            if not ligne.strip():
                continue
            item = json.loads(ligne)
            ors = item.get("gold_chunks")
            if ors is None and item.get("target_chunk"):
                ors = [item["target_chunk"]]
            out[item["qid"]] = {**item, "gold_chunks": ors or [],
                                "kind": item.get("kind", "single"),
                                "banc": "v1" if chemin.name.endswith("v1.jsonl") else "v3"}
    return out


def sources() -> list[tuple[str, dict]]:
    """Tous les ``results-e2e-*.json`` porteurs de verdicts appariés au contexte servi."""
    out = []
    for chemin in sorted(glob.glob(str(HERE / "results-e2e-*.json"))):
        try:
            donnees = json.loads(Path(chemin).read_text(encoding="utf-8"))
        except Exception:
            continue
        lignes = donnees.get("per_row")
        if not isinstance(lignes, list) or not lignes:
            continue
        premiere = lignes[0]
        if not ({"judge", "context", "gold_in_context"} <= set(premiere)):
            continue
        out.append((os.path.basename(chemin), donnees))
    return out


def observations(index: ChunkIndex, bancs: dict) -> list[dict]:
    """Une observation par (fichier, config, question) — l'unité de toutes les tables."""
    tableaux = set(corpus_overlay.text_overrides() or {})
    obs = []
    for nom, donnees in sources():
        for ligne in donnees["per_row"]:
            item = bancs.get(ligne["qid"])
            if item is None or item["kind"] == "negative" or not item["gold_chunks"]:
                continue
            verdict = ligne.get("judge") or {}
            couverture = verdict.get("coverage")
            if couverture is None:
                continue
            servis = [c["chunk_id"] for c in ligne.get("context") or ()]
            rang = {c: i + 1 for i, c in enumerate(servis)}
            tous_les_ors = [c for c in item["gold_chunks"] if index.get(c)]
            ors_servis = [c for c in tous_les_ors if c in set(servis)]
            faits = item.get("answer_facts") or []
            parts, glissantes, longueurs, rangs = [], [], [], []
            for chunk in ors_servis:
                texte = index.get(chunk)["text"]
                p = part_dehors(texte, faits, index.df)
                if p is not None:
                    parts.append(p)
                    g = fenetre_glissante(texte, faits, index.df)
                    if g is not None:
                        glissantes.append(g)
                longueurs.append(len(texte))
                rangs.append(rang[chunk])
            # la part est une propriété du chunk d'or, pas du fait qu'il soit servi : la
            # calculer sur TOUS les ors est ce qui rend le contrôle négatif calculable.
            parts_or, longueurs_or = [], []
            for chunk in tous_les_ors:
                texte = index.get(chunk)["text"]
                p = part_dehors(texte, faits, index.df)
                if p is not None:
                    parts_or.append(p)
                longueurs_or.append(len(texte))
            obs.append({
                "fichier": nom, "config": ligne.get("config"), "qid": ligne["qid"],
                "banc": item["banc"], "kind": item["kind"],
                "or_servi": bool(ors_servis),
                "couverture": float(couverture), "reussie": float(couverture) > 0,
                "abstenue": bool(ligne.get("abstained")),
                "part_max": max(parts) if parts else None,
                "part_min": min(parts) if parts else None,
                "part_max_or": max(parts_or) if parts_or else None,
                "longueur_max_or": max(longueurs_or) if longueurs_or else None,
                "glissante": any(glissantes) if glissantes else None,
                "longueur_max": max(longueurs) if longueurs else None,
                "rang_or": min(rangs) if rangs else None,
                "tableau": any(c in tableaux for c in item["gold_chunks"]),
            })
    return obs


def table_par_borne(obs: list[dict], cle: str = "part_max") -> list[dict]:
    seaux = collections.defaultdict(lambda: [0, 0])
    for o in obs:
        if o.get(cle) is None:
            continue
        seau = seaux[borne_de(o[cle])]
        seau[1] += 1
        seau[0] += 1 if o["reussie"] else 0
    out = []
    for i in range(len(BORNES) - 1):
        k, n = seaux.get(i, [0, 0])
        bas, haut = wilson(k, n)
        out.append({"borne": [BORNES[i], min(1.0, BORNES[i + 1])], "n": n, "reussies": k,
                    "taux": round(k / n, 4) if n else None,
                    "ic95": [round(bas, 4), round(haut, 4)] if n else None})
    return out


def pente_bootstrap(obs: list[dict], cle: str = "part_max") -> dict:
    """Écart de taux entre la borne haute et la borne basse, groupé par question."""
    utiles = [o for o in obs if o.get(cle) is not None]
    par_qid = collections.defaultdict(list)
    for o in utiles:
        par_qid[o["qid"]].append(o)
    qids = sorted(par_qid)

    def ecart(echantillon):
        bas_k = bas_n = haut_k = haut_n = 0
        for o in echantillon:
            b = borne_de(o[cle])
            if b == 0:
                bas_n += 1
                bas_k += 1 if o["reussie"] else 0
            elif b >= len(BORNES) - 3:
                haut_n += 1
                haut_k += 1 if o["reussie"] else 0
        if not bas_n or not haut_n:
            return None
        return haut_k / haut_n - bas_k / bas_n

    point = ecart(utiles)
    alea = random.Random(GRAINE)
    tirages = []
    for _ in range(TIRAGES):
        choisis = [alea.choice(qids) for _ in qids]
        echantillon = [o for q in choisis for o in par_qid[q]]
        valeur = ecart(echantillon)
        if valeur is not None:
            tirages.append(valeur)
    tirages.sort()
    if not tirages:
        return {"point": point, "ic95": None, "n_questions": len(qids)}
    bas = tirages[int(0.025 * len(tirages))]
    haut = tirages[min(len(tirages) - 1, int(0.975 * len(tirages)))]
    return {"point": round(point, 4) if point is not None else None,
            "ic95": [round(bas, 4), round(haut, 4)], "n_questions": len(qids),
            "n_observations": len(utiles), "tirages": len(tirages)}


#: Coupe grossière des ventilations. Les cinq bornes déclarées sont trop creuses à
#: l'intérieur d'une strate de contrôle : un `n/a` y cacherait l'information au lieu de la
#: montrer. Deux cases suffisent à lire un contrôle. **L'analyse primaire garde ses cinq
#: bornes déclarées** ; ce dispositif ne la touche pas.
COUPE_VENTILATION = 0.25


def _deux_cases(obs: list[dict], cle: str = "part_max") -> dict:
    seaux = {False: [0, 0], True: [0, 0]}
    for o in obs:
        if o.get(cle) is None:
            continue
        seau = seaux[o[cle] >= COUPE_VENTILATION]
        seau[1] += 1
        seau[0] += 1 if o["reussie"] else 0
    out = {}
    for cle_case, (k, n) in seaux.items():
        nom = "au_dela" if cle_case else "en_deca"
        out[nom] = {"n": n, "reussies": k, "taux": round(k / n, 4) if n else None,
                    "ic95": [round(v, 4) for v in wilson(k, n)] if n else None}
    a, b = seaux[True], seaux[False]
    out["ecart"] = (round(a[0] / a[1] - b[0] / b[1], 4) if a[1] and b[1] else None)
    return out


def ventile(obs: list[dict], cle_strate) -> dict:
    out = {}
    for nom in sorted({cle_strate(o) for o in obs}, key=str):
        sous = [o for o in obs if cle_strate(o) == nom]
        out[str(nom)] = {"n": len(sous), "cases": _deux_cases(sous),
                         "table": table_par_borne(sous)}
    return out


def audit() -> dict:
    index = ChunkIndex.load(verbose=False)
    bancs = questions()
    obs = observations(index, bancs)

    servis = [o for o in obs if o["or_servi"]]
    longs = [o for o in servis if (o["longueur_max"] or 0) > COUPE]
    non_servis = [o for o in obs if not o["or_servi"]]
    non_servis_longs = [o for o in non_servis if (o["longueur_max_or"] or 0) > COUPE]

    #: la population de contrôle de longueur : or servi mais court, jamais tronqué
    courts = [o for o in servis if (o["longueur_max"] or 0) <= COUPE]

    # inventaire des longueurs d'or servi, indépendant des verdicts
    par_question = {}
    for o in servis:
        par_question.setdefault(o["qid"], o["longueur_max"])
    longueurs = [v for v in par_question.values() if v]
    perdu = [max(0, v - COUPE) for v in longueurs]

    return {
        "signature": corpus_overlay.signature(),
        "coupe_generateur": COUPE,
        "coupe_juge": 1400,
        "coupe_redaction_des_questions": 4500,
        "coupe_plongement_tokens": 1024,
        "appels_llm": 0,
        "sources": [nom for nom, _ in sources()],
        "observations": {"total": len(obs), "or_servi": len(servis),
                         "or_servi_et_long": len(longs), "or_servi_et_court": len(courts),
                         "or_non_servi": len(non_servis),
                         "or_non_servi_et_long": len(non_servis_longs),
                         "questions_distinctes": len({o["qid"] for o in obs})},
        "longueur_or_servi": {
            "questions": len(longueurs),
            "mediane": sorted(longueurs)[len(longueurs) // 2] if longueurs else None,
            "part_au_dessus_de_la_coupe": round(sum(1 for v in longueurs if v > COUPE) / len(longueurs), 4) if longueurs else None,
            "caracteres_perdus_medians": sorted(perdu)[len(perdu) // 2] if perdu else None,
            "caracteres_perdus_total": sum(perdu),
        },
        "primaire_or_servi_long": {"table": table_par_borne(longs), "pente": pente_bootstrap(longs)},
        "controle_negatif_or_non_servi": {
            "definition": ("or NON servi, chunk d'or de plus de 1 600 caractères ; la part est "
                           "calculée sur le chunk d'or lui-même, qui n'a pas été montré — la "
                           "couverture ne peut donc pas en dépendre, et cette table doit être plate"),
            "table": table_par_borne(non_servis_longs, "part_max_or"),
            "pente": pente_bootstrap(non_servis_longs, "part_max_or")},
        "controle_longueur_or_court": {"table": table_par_borne(courts),
                                       "pente": pente_bootstrap(courts)},
        "sensibilite_agregation_min": {"table": table_par_borne(longs, "part_min"),
                                       "pente": pente_bootstrap(longs, "part_min")},
        "sensibilite_fenetre_glissante": _glissante(longs),
        "ventilation_kind": ventile(longs, lambda o: o["kind"]),
        "ventilation_tableau": ventile(longs, lambda o: o["tableau"]),
        "ventilation_rang": ventile(longs, lambda o: o["rang_or"]),
        "ventilation_longueur": ventile(longs, lambda o: _seau_longueur(o["longueur_max"])),
        "ventilation_strate": ventile(longs, lambda o: f"{o['fichier']}::{o['config']}"),
        "abstention_par_borne": _abstention(longs),
    }


def _seau_longueur(valeur: int | None) -> str:
    if not valeur:
        return "inconnue"
    for haut in (2000, 2500, 3000, 4000):
        if valeur < haut:
            return f"<{haut}"
    return ">=4000"


def _glissante(obs: list[dict]) -> dict:
    seaux = collections.defaultdict(lambda: [0, 0])
    for o in obs:
        if o.get("glissante") is None:
            continue
        seau = seaux[bool(o["glissante"])]
        seau[1] += 1
        seau[0] += 1 if o["reussie"] else 0
    out = {}
    for cle, (k, n) in seaux.items():
        bas, haut = wilson(k, n)
        out[str(cle)] = {"n": n, "reussies": k, "taux": round(k / n, 4) if n else None,
                         "ic95": [round(bas, 4), round(haut, 4)]}
    return out


def _abstention(obs: list[dict]) -> list[dict]:
    seaux = collections.defaultdict(lambda: [0, 0])
    for o in obs:
        if o.get("part_max") is None:
            continue
        seau = seaux[borne_de(o["part_max"])]
        seau[1] += 1
        seau[0] += 1 if o["abstenue"] else 0
    out = []
    for i in range(len(BORNES) - 1):
        k, n = seaux.get(i, [0, 0])
        out.append({"borne": [BORNES[i], min(1.0, BORNES[i + 1])], "n": n,
                    "abstentions": k, "taux": round(k / n, 4) if n else None})
    return out


def imprimer(rapport: dict) -> None:
    o = rapport["observations"]
    print(f"signature {rapport['signature']}   coupe générateur {rapport['coupe_generateur']} c."
          f"   juge {rapport['coupe_juge']}   rédaction des questions {rapport['coupe_redaction_des_questions']}")
    print(f"observations {o['total']} sur {o['questions_distinctes']} questions   "
          f"or servi {o['or_servi']} (long {o['or_servi_et_long']}, court {o['or_servi_et_court']})   "
          f"or non servi {o['or_non_servi']}   appels LLM {rapport['appels_llm']}")
    L = rapport["longueur_or_servi"]
    print(f"or servi : médiane {L['mediane']} c., {100 * (L['part_au_dessus_de_la_coupe'] or 0):.1f} % "
          f"au-dessus de la coupe, {L['caracteres_perdus_total']} caractères d'or jamais montrés")

    def bloc(titre, entree):
        print(f"\n{titre}")
        for l in entree["table"]:
            if not l["n"]:
                continue
            print(f"   part hors vue [{l['borne'][0]:.3f} ; {l['borne'][1]:.3f}[  n={l['n']:4d}  "
                  f"couverture>0 {l['reussies']:4d}  {100 * l['taux']:5.1f} %  "
                  f"IC95 [{100 * l['ic95'][0]:.1f} ; {100 * l['ic95'][1]:.1f}]")
        p = entree["pente"]
        if p.get("ic95"):
            print(f"   écart haut−bas {100 * p['point']:+.1f} pts  IC95 groupé par question "
                  f"[{100 * p['ic95'][0]:+.1f} ; {100 * p['ic95'][1]:+.1f}]  "
                  f"({p['n_questions']} questions, {p['n_observations']} observations)")

    bloc("PRIMAIRE — or servi de plus de 1 600 caractères (longueur contrôlée)",
         rapport["primaire_or_servi_long"])
    bloc("CONTRÔLE NÉGATIF — or non servi (doit être plat)",
         rapport["controle_negatif_or_non_servi"])
    bloc("CONTRÔLE DE LONGUEUR — or servi de 1 600 caractères ou moins",
         rapport["controle_longueur_or_court"])
    bloc("SENSIBILITÉ — agrégation par le minimum", rapport["sensibilite_agregation_min"])
    print("\nSENSIBILITÉ — méthode de la reconnaissance (fenêtre glissante)")
    for cle, v in sorted(rapport["sensibilite_fenetre_glissante"].items()):
        print(f"   faits franchissant la coupe = {cle:5s}  n={v['n']:4d}  {100 * v['taux']:5.1f} %  "
              f"IC95 [{100 * v['ic95'][0]:.1f} ; {100 * v['ic95'][1]:.1f}]")
    for titre, cle in (("kind", "ventilation_kind"), ("overlay tableau", "ventilation_tableau"),
                       ("rang de l'or dans le prompt", "ventilation_rang"),
                       ("longueur du chunk d'or", "ventilation_longueur")):
        print(f"\nVENTILATION — {titre}")
        for nom, v in rapport[cle].items():
            c = v["cases"]
            ec = f"{100 * c['ecart']:+.1f}" if c.get("ecart") is not None else "  n/a"
            dec, au = c["en_deca"], c["au_dela"]
            t1 = f"{100 * dec['taux']:5.1f} %" if dec["taux"] is not None else "    — "
            t2 = f"{100 * au['taux']:5.1f} %" if au["taux"] is not None else "    — "
            print(f"   {nom:24s} n={v['n']:4d}   < 0,25 : {t1} (n={dec['n']:3d})   "
                  f">= 0,25 : {t2} (n={au['n']:3d})   écart {ec} pts")
    print("\nABSTENTION par part hors vue")
    for l in rapport["abstention_par_borne"]:
        if l["n"]:
            print(f"   [{l['borne'][0]:.3f} ; {l['borne'][1]:.3f}[  n={l['n']:4d}  "
                  f"abstentions {l['abstentions']:3d}  {100 * l['taux']:5.1f} %")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--sortie", default=None)
    a = p.parse_args()
    rapport = audit()
    chemin = Path(a.sortie) if a.sortie else HERE / f"results-fenetre-generateur-{rapport['signature']}.json"
    chemin.write_text(json.dumps(rapport, ensure_ascii=False, indent=1), encoding="utf-8")
    imprimer(rapport)
    print(f"\nécrit : {chemin.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
