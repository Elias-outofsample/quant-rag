"""Le banc v4 : une commande, un JSON, et le refus de comparer ce qui n'est pas comparable.

Ce que ce banc fait que le précédent ne faisait pas
---------------------------------------------------
**Il note sans juge ce qui peut l'être.** Une formule restituée, une cellule lue, une
abstention : trois verdicts déterministes, reproductibles à l'octet, gratuits. Le juge de
référence ne sert plus qu'au diagnostic et aux familles qui en ont besoin — ce qui divise le
coût d'un run et retire l'instrument le plus biaisé du chemin critique.

**Il refuse de partir sans ses témoins.** Repris tel quel d'``eval_characters.py`` (l. 112-116,
commit ``164a74c``) : la mesure s'arrête si les items-témoins du juge n'ont pas tourné, et
elle s'arrête si ``unsupported_fluent`` tombe sous 0,75 — la famille qui garde contre « plus
fluide, moins fondé ». Un verdict rendu par un juge illisible n'est pas un verdict prudent,
c'est un chiffre sans référent.

**Il refuse de comparer deux runs qui ne portent pas le même instrument.** Cinq champs
d'en-tête gouvernent : la signature du corpus, la **fenêtre servie**, la **fenêtre du juge**,
les modèles, la version des questions. Il y en a bien **deux, pas une** — le fil `characters`
a jugé à fenêtre fixe 4 500 alors que ``judge.grade`` a désormais pour défaut 2 500, et un
drapeau qui ne surveillerait que la fenêtre servie ne verrait pas l'écart. Comparer sans
``--drapeau`` est refusé, et le drapeau écrit dans le JSON *ce qui* diffère.

**Il met le « où » avant le « combien ».** L'or servi est établi sur les classements, avant
tout appel LLM — l'ordre exigé au §6 du pré-enregistrement `characters`, et la leçon du
chantier routage. Un score de famille qui bouge sans que l'or servi bouge est un mouvement de
lecture ; l'inverse est un mouvement de récupération. Les confondre a déjà coûté un chantier.

Ce que ce banc ne fait pas
---------------------------
Il ne modifie ni le chemin servi, ni ``pipeline.py``, ni ``llm.py``, ni le défaut de fenêtre
de ``judge.py``. Il ne mesure aucune variante de retrieval (§14 du ``docs/TODO.md``). Il ne
répare aucune question : ce qui ne se vérifie pas par programme n'entre pas dans la population.

    .venv/bin/python rag/benchmark/banc_v4.py --etape temoins
    .venv/bin/python rag/benchmark/banc_v4.py --etape mesure
    .venv/bin/python rag/benchmark/banc_v4.py --etape verdict

``--gold-signature SIG`` rejoue formula/table_cell/negative_voisine sur un gold ré-ancré
(``gold_ancrage.py --appliquer-v4``, pour un corpus candidat) ; sans l'option, inchangé.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import random
import shutil
import statistics
import sys
import tempfile
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import corpus_overlay  # noqa: E402
import familles_v4  # noqa: E402
import garde_reponse  # noqa: E402
import judge  # noqa: E402
import latex_norme  # noqa: E402
import llm  # noqa: E402
import pipeline  # noqa: E402
import score_citation  # noqa: E402
import score_formule  # noqa: E402
import score_tableau  # noqa: E402
from corpus import ChunkIndex  # noqa: E402

#: Le corpus mesuré. ``--candidat SIG`` la remplace, et c'est la seule façon correcte de
#: mesurer un corpus candidat : sa signature **ne peut pas** s'obtenir en échangeant des
#: fichiers d'overlay, parce qu'elle incorpore le ``sha256`` du ``chunks.jsonl`` re-découpé de
#: chaque document (``rechunk_corpus.construire``) là où ``corpus_overlay.signature`` lit les
#: empreintes enregistrées au registre. On nomme donc le corpus mesuré, on ne le déguise pas.
SIGNATURE = corpus_overlay.signature()
#: Vue corpus injectée par ``--candidat``. ``ChunkIndex.load()`` bâtit la sienne depuis
#: ``rows.jsonl`` — le corpus servi — et composerait le contexte avec ses passages pendant que
#: le classement viendrait du candidat : deux corpus dans un même prompt.
INDEX: ChunkIndex | None = None


def _index() -> ChunkIndex:
    return INDEX if INDEX is not None else ChunkIndex.load(verbose=False)


GENERATEUR, JUGE = "mistral-small-latest", "mistral-medium-latest"
#: Fenêtre servie au générateur. Défaut de la production depuis ``4eee997``.
FENETRE_SERVIE = pipeline.CARACTERES_SERVIS
#: Fenêtre montrée au JUGE. Distincte de la précédente — c'est le piège que ce banc traque.
#: Défaut : la même que la fenêtre servie, pour que le juge voie ce que le générateur a vu.
#: Le fil `characters` avait figé la sienne à 4 500 ; un run v4 qui s'y compare doit le déclarer.
FENETRE_JUGE = FENETRE_SERVIE

#: Repris d'``eval_characters.py`` l. 50 — seuils calculés AVANT la mesure, à n ≈ 130.
#: ``a`` : abstention nette nouvelle. ``b`` : dégradation stricte d'une couverture 2 vers 0.
GARDE_ABSTENTION, GARDE_DEGRADATION = 2, 6
#: Repris d'``eval_characters.py`` l. 52 — la condition de lisibilité du juge.
FAMILLE_BLOQUANTE, SEUIL_FAMILLE = "unsupported_fluent", 0.75
GRAINE, TIRAGES = 20260908, 10000

PASSAGES = 5
CACHE = HERE / ".cache"


def _cache_reponses(bras: str) -> Path:
    """Le fichier d'état des réponses, **nommé par le bras**.

    Il ne l'était pas, et il ne pouvait pas l'être avant qu'il existe deux contrats. Le laisser
    tel quel aurait détruit la ligne de base à la première mesure : ``mesure`` compare chaque
    réponse gardée à la clé de cache que le prompt courant produirait (étage 0) ; sous ``v3``,
    les 199 réponses ``v2`` auraient été déclarées périmées, **et leurs contextes jetés avec
    elles**. Le bras de référence aurait disparu du cache d'état, et les 199 récupérations
    Qdrant auraient été refaites pour rien.

    Le suffixe portait d'abord le nom du **contrat** ; il porte depuis le 10 septembre 2026
    celui du **bras**, et c'est ce qui permet à un placebo d'exister : un placebo a exactement
    le contrat et la fenêtre de sa référence — s'il partageait aussi son fichier d'état, il
    l'écraserait. Pour un bras dont le nom est celui d'un contrat (``v2``, ``v3``, ``v4``), le
    chemin est **inchangé à l'octet près** : les artefacts publiés le 9 septembre restent lus.

    Les **contextes** ne portent pas ce suffixe, et c'est délibéré : ni le contrat de réponse ni
    la fenêtre servie n'entrent dans la récupération. Tous les bras voient donc, question par
    question, les mêmes cinq passages — ce qui est la condition d'un appariement, pas une
    économie.
    """
    return CACHE / f"v4-reponses-{SIGNATURE}-{bras}.json"


def _cache_verdicts(bras: str) -> Path:
    """Idem pour les verdicts : le juge note une réponse, donc il note un bras."""
    return CACHE / f"v4-verdicts-{SIGNATURE}-{bras}.json"


def _cache_tirages(bras: str) -> Path:
    """Les tirages 2..k d'un bras — voir ``--tirages``. Le tirage 1 vit dans le fichier ci-dessus."""
    return CACHE / f"v4-tirages-{SIGNATURE}-{bras}.json"


def _cache_facturation(bras: str) -> Path:
    return CACHE / f"v4-facturation-{SIGNATURE}-{bras}.json"


def _fichier_run() -> Path:
    """Le manifeste du run : quels bras, quel rôle, quel identifiant. Il fait exister « le même
    run » comme un fait vérifiable, et non comme une intention de l'opérateur."""
    return CACHE / f"v4-run-{SIGNATURE}.json"


# --------------------------------------------------------------------------- les bras

@dataclass(frozen=True)
class Bras:
    """Un bras de mesure : un nom, un contrat, une fenêtre servie, un rôle.

    Ce que cette classe rend possible et qui ne l'était pas
    --------------------------------------------------------
    Jusqu'au 9 septembre 2026, un run du banc mesurait **un** état du système, et les Δ se
    faisaient entre deux runs — donc entre deux moments. Trois chantiers de suite ont buté sur
    la même conséquence : ``mistral-small-latest`` n'est pas reproductible à température 0
    (51/130, 54/130, 110/199 réponses identiques à prompt identique au bit près), si bien qu'un
    Δ entre deux moments contient un bruit dont **aucun chiffre ne disait l'ampleur**.

    Les bras tirés dans le **même run** et entrelacés **par question** retirent tout ce qui
    sépare deux moments : même corpus chargé, même processus, mêmes secondes, mêmes passages.
    Ce qui reste dans le Δ, c'est la variable — plus le bruit du générateur, que le bras
    ``placebo`` mesure enfin dans les mêmes conditions.
    """

    nom: str
    prompt: str
    fenetre_servie: int
    role: str = "variante"          #: reference | variante | placebo
    jetable: bool = False           #: cache d'appels neuf, effacé en fin de run

    def resume(self) -> dict:
        return {"nom": self.nom, "prompt": self.prompt,
                "empreinte_prompt": pipeline.empreinte_prompt(self.prompt),
                "fenetre_servie": self.fenetre_servie, "role": self.role,
                "jetable": self.jetable}


def analyser_bras(specification: str, fenetre_par_defaut: int) -> list[Bras]:
    """``a1:fenetre=2500,a2:fenetre=10000,plancher:placebo`` → trois bras.

    Grammaire, volontairement minuscule : ``nom`` puis des jetons séparés par ``:``, chacun
    ``clé=valeur`` (``prompt``, ``fenetre``) ou le drapeau nu ``placebo``.

    Les deux règles qui ne sont pas des commodités :

    - **le premier bras est la référence.** Tous les Δ sont appariés contre lui ;
    - **un placebo hérite du contrat ET de la fenêtre de la référence, et ne peut pas en
      différer.** Un « placebo » qui porterait une autre valeur de la variable ne serait pas un
      placebo mais un second candidat, et le plancher de bruit qu'il publierait serait faux —
      d'une fausseté rassurante, puisqu'elle élargirait le plancher et rendrait tout non
      concluant. Une erreur de frappe suffirait ; on la refuse ici plutôt qu'après la dépense.
    """
    bras: list[Bras] = []
    for rang, morceau in enumerate(m.strip() for m in specification.split(",") if m.strip()):
        jetons = morceau.split(":")
        nom, options = jetons[0].strip(), jetons[1:]
        if not nom:
            sys.exit(f"--bras : nom de bras vide dans « {morceau} »")
        prompt = nom if nom in pipeline.ANSWER_PROMPTS else None
        fenetre, placebo = None, False
        for jeton in (j.strip() for j in options):
            if jeton == "placebo":
                placebo = True
            elif jeton.startswith("prompt="):
                prompt = jeton.removeprefix("prompt=")
            elif jeton.startswith("fenetre="):
                fenetre = int(jeton.removeprefix("fenetre="))
            else:
                sys.exit(f"--bras : jeton « {jeton} » inconnu (attendu prompt=, fenetre=, placebo)")
        if placebo and bras:
            reference = bras[0]
            if prompt not in (None, reference.prompt) or fenetre not in (None, reference.fenetre_servie):
                sys.exit(f"--bras : le placebo « {nom} » porte une variable de son propre "
                         f"(prompt={prompt}, fenetre={fenetre}) alors que la référence "
                         f"« {reference.nom} » porte prompt={reference.prompt}, "
                         f"fenetre={reference.fenetre_servie}. Un placebo ne diffère PAR RIEN "
                         f"de sa référence : c'est ce qui fait de son Δ un plancher de bruit.")
            prompt, fenetre = reference.prompt, reference.fenetre_servie
        if prompt is None:
            prompt = pipeline.DEFAULT_PROMPT
        if prompt not in pipeline.ANSWER_PROMPTS:
            sys.exit(f"--bras : contrat « {prompt} » inconnu (connus : "
                     f"{', '.join(sorted(pipeline.ANSWER_PROMPTS))})")
        bras.append(Bras(nom=nom, prompt=prompt,
                         fenetre_servie=fenetre if fenetre is not None else fenetre_par_defaut,
                         role="placebo" if placebo else "reference" if rang == 0 else "variante",
                         jetable=placebo))
    if not bras:
        sys.exit("--bras : aucun bras")
    noms = [b.nom for b in bras]
    if len(set(noms)) != len(noms):
        sys.exit(f"--bras : deux bras portent le même nom ({noms}) — ils partageraient leurs "
                 f"fichiers d'état et le second écraserait le premier.")
    placebos = [b for b in bras if b.role == "placebo"]
    if len(placebos) > 1:
        sys.exit(f"--bras : {len(placebos)} placebos. Un seul plancher par run.")
    if len(bras) > 1 and bras[0].role == "placebo":
        sys.exit("--bras : dans un run à plusieurs bras, le placebo ne peut pas être le "
                 "premier — il est défini par rapport à la référence, qui doit le précéder.")
    return bras


@contextlib.contextmanager
def cache_d_appels(chemin: Path | None):
    """Détourne ``llm.CACHE`` le temps d'un bras jetable, puis le remet.

    ``llm._cache_path`` lit ``llm.CACHE`` **à l'appel** : une affectation suffit, et elle suffit
    aussi à garantir la propriété qui compte — les appels d'un bras jetable ne sont ni servis
    par le cache du dépôt (donc ils sont bien un tirage neuf) ni écrits dedans (donc le prochain
    run de la référence reste reproductible). Le lot 6 a établi la leçon dans l'autre sens :
    un placebo qui pollue le cache détruit ce qu'il mesure.
    """
    if chemin is None:
        yield
        return
    avant = llm.CACHE
    llm.CACHE = chemin
    try:
        yield
    finally:
        llm.CACHE = avant

#: Les familles v4 et le fichier qui les porte. Les 20 négatives v3 entrent aussi dans la
#: population : elles sont le témoin d'abstention déjà calibré du banc.
FAMILLES = {
    "formula": "questions-v4-formula.jsonl",
    "table_cell": "questions-v4-table-cell.jsonl",
    "negative_voisine": "questions-v4-negative-voisine.jsonl",
    "negative_v3": "questions-v3.jsonl",
}
#: Familles soumises au juge. Les deux familles déterministes ne le sont qu'en DIAGNOSTIC,
#: et les négatives pour la seule fabrication. Ce choix est ce qui rend le run abordable.
JUGEES = ("formula", "negative_voisine", "negative_v3")

#: Familles dont l'or a été ré-ancré sur un corpus candidat (gold_ancrage.py
#: --construire-v4 / --appliquer-v4). ``negative_v3`` n'en fait pas partie : son fichier
#: (``questions-v3.jsonl``) appartient au banc v1/v3, hors du périmètre de cette option.
FAMILLES_V4 = {"formula", "table_cell", "negative_voisine"}

#: Signature du corpus candidat visée par ``--gold-signature``. ``None`` par défaut : le
#: banc lit alors ``FAMILLES`` sans suffixe, à l'octet près du comportement d'avant cette
#: option. Positionnée une fois, dans ``main()``, avant tout appel de ``population``/
#: ``en_tete``.
GOLD_SIGNATURE: str | None = None


def _fichier_famille(nom: str, fichier: str) -> Path:
    """Chemin du fichier de questions d'une famille — suffixé par ``GOLD_SIGNATURE`` si
    elle fait partie de ``FAMILLES_V4`` et que l'option est active, inchangé sinon."""
    if GOLD_SIGNATURE and nom in FAMILLES_V4:
        return HERE / fichier.replace(".jsonl", f"-{GOLD_SIGNATURE}.jsonl")
    return HERE / fichier


# --------------------------------------------------------------------------- entrées/sorties

def _charger(chemin: Path) -> dict:
    return json.loads(chemin.read_text(encoding="utf-8")) if chemin.exists() else {}


def _ecrire(chemin: Path, contenu) -> None:
    chemin.parent.mkdir(parents=True, exist_ok=True)
    chemin.write_text(json.dumps(contenu, ensure_ascii=False, indent=1), encoding="utf-8")


# --------------------------------------------------------------------------- facturation

def _facturer(registre: dict, modele: str, appel):
    """Exécute ``appel`` et impute les jetons consommés **au modèle demandé**.

    ``llm._stats`` est global : il compte des jetons, pas des jetons *par modèle*. Multiplier
    ce total par un seul prix donnerait un coût faux d'un facteur dix entre ``small`` et
    ``medium``. On encadre donc chaque appel et on impute le delta. Le surcoût est nul ;
    l'alternative — modifier ``llm.py`` — est interdite au §4 du handoff, et de toute façon
    ce chantier n'a pas à toucher l'instrument commun pour se facturer lui-même.
    """
    avant = llm.stats()
    depart = time.perf_counter()
    resultat = appel()
    duree = time.perf_counter() - depart
    apres = llm.stats()
    ligne = registre.setdefault(modele, {"appels": 0, "cache": 0, "entree": 0, "sortie": 0})
    ligne["appels"] += apres["calls"] - avant["calls"]
    ligne["cache"] += apres["cached"] - avant["cached"]
    ligne["entree"] += apres["prompt_tokens"] - avant["prompt_tokens"]
    ligne["sortie"] += apres["completion_tokens"] - avant["completion_tokens"]
    return resultat, duree


def chiffrer(registre: dict) -> dict:
    """Coût du run, en dollars, depuis ``prix-modeles.json`` — via l'unique analyseur du dépôt.

    Le barème a existé en deux schémas jusqu'au 9 septembre 2026 ; il n'en reste qu'un, et
    ``llm.cout_usd`` est le seul à le lire. Un modèle dont le coût est inconnu est **signalé et
    exclu du total**, jamais compté zéro : un total muet sur ce qu'il ignore a l'air d'un total.
    """
    detail, total, inconnus = {}, 0.0, []
    for modele, ligne in registre.items():
        cout = llm.cout_usd(modele, ligne["entree"], ligne["sortie"])
        detail[modele] = {**ligne, "cout_usd": None if cout is None else round(cout, 4),
                          "tarif_connu": cout is not None}
        if cout is None:
            inconnus.append(modele)
        else:
            total += cout
    taux = (llm.prix().get("taux_usd_par_eur") or {}).get("valeur")
    return {"par_modele": detail, "total_usd": round(total, 4),
            "total_eur_majore": round(total, 4),
            "total_eur": round(total / taux, 4) if taux else None,
            "modeles_sans_tarif": inconnus,
            "note": "total_eur_majore : 1 USD compté pour 1 EUR — majoration prudente. "
                    "total_eur : taux daté de prix-modeles.json. Les deux ne s'additionnent pas."}


# --------------------------------------------------------------------------- facturation depuis le cache

def _cle_generation(question: str, contexte: list[dict], fenetre: int, modele: str,
                    prompt: str = pipeline.DEFAULT_PROMPT):
    """Clé de cache de l'appel de génération — reproduit exactement ``pipeline.answer``.

    ``prompt`` a un défaut, mais **aucun appelant de ce module ne l'omet**, et c'est ce qui
    évite un piège qui aurait coûté le budget entier : ce module lisait
    ``pipeline.DEFAULT_PROMPT`` en dur. Un run mené sous ``--prompt v3`` aurait reconstruit la
    clé du prompt **par défaut**, déclaré périmée chacune des 199 réponses qu'il venait
    d'écrire, jeté leurs contextes avec elles (``mesure``, étage 0), et repayé le run entier à
    chaque relance — sans qu'aucun message ne dise autre chose que « 199 réponses invalidées ».
    """
    rendu = pipeline.format_passages(contexte, characters=fenetre)
    messages = [{"role": "system", "content": pipeline.ANSWER_PROMPTS[prompt]},
                {"role": "user", "content": f"PASSAGES\n\n{rendu}\n\nQUESTION\n{question}"}]
    return llm._cache_path({"model": modele, "messages": messages, "temperature": 0.0,
                            "max_tokens": 400, "_seed": None})


def _cle_juge(item: dict, reponse: str, contexte: list[dict], graine: str, fenetre: int,
              modele: str):
    """Clé de cache de l'appel de notation — reproduit exactement ``judge.grade``."""
    passages = pipeline.format_passages(contexte, characters=fenetre) or "(no passage was retrieved)"
    if item["kind"] == "negative":
        prompt = judge.GRADE_NEGATIVE % {
            "entities": ", ".join(item.get("absent_entities", [])) or "the entity named in the question",
            "chunks": str(judge.CORPUS.get("chunks") or "its"),
            "question": item["question"], "passages": passages, "answer": reponse}
    else:
        facts = "\n".join(f"- {fait}" for fait in item["answer_facts"])
        prompt = judge.GRADE_POSITIVE % {"question": item["question"], "passages": passages,
                                         "facts": facts, "answer": reponse}
    return llm._cache_path({"model": modele, "messages": [{"role": "user", "content": prompt}],
                            "temperature": 0.0, "max_tokens": 450,
                            "response_format": {"type": "json_object"}, "_seed": graine})


def _usage(chemin) -> dict | None:
    if not chemin.exists():
        return None
    try:
        return json.loads(chemin.read_text(encoding="utf-8")).get("usage") or {}
    except (json.JSONDecodeError, OSError):
        return None


def facturer_depuis_le_cache(appels) -> dict:
    """Registre reconstruit depuis le **cache disque**, seule source de vérité du coût.

    Pourquoi le registre en mémoire ne suffit pas, et ce que ça a coûté de le croire
    ---------------------------------------------------------------------------------
    ``_facturer`` tient ses comptes en mémoire et ne les écrit qu'à la fin de la mesure. Une
    campagne interrompue les emporte : celle du graphe, relancée à mi-course, a rapporté
    **0,6319 USD** pour une dépense réelle de **0,9094** — les appels du premier run n'étaient
    dans aucun fichier. Un compteur en mémoire ne survit pas au processus, et une campagne
    longue finit toujours par être interrompue.

    Le cache, lui, garde tout : chaque réponse écrite porte son ``usage``. Reconstruire les
    clés d'un run et sommer ces ``usage`` donne le coût **exact**, reprises comprises, et le
    donne encore des jours plus tard.

    ``appels`` est une suite de ``(modele, chemin_de_cache)``. Le champ ``manquants`` compte
    ce qui n'a pas été retrouvé : il doit valoir zéro. Non nul, il signale que la
    reconstruction a dérivé de ce que le code appelle réellement — un silence serait pire
    qu'une erreur, puisqu'il rendrait un coût sous-estimé qui a l'air d'un coût.
    """
    registre, manquants = {}, 0
    for modele, chemin in appels:
        usage = _usage(chemin)
        if usage is None:
            manquants += 1
            continue
        ligne = registre.setdefault(modele, {"appels": 0, "cache": 0, "entree": 0, "sortie": 0})
        ligne["appels"] += 1
        ligne["entree"] += usage.get("prompt_tokens", 0)
        ligne["sortie"] += usage.get("completion_tokens", 0)
    return {"registre": registre, "manquants": manquants}


def _reponse_appartient(item: dict, contexte: list[dict], reponse: dict,
                        fenetre_servie: int, prompt: str) -> bool:
    """La réponse gardée est-elle bien celle de cette question ?

    On reconstruit la clé de cache que ``pipeline.answer`` aurait produite pour la question
    courante et son contexte. Si le fichier existe et porte le même texte, la réponse lui
    appartient : les deux ont été produites par le même prompt. Sinon, le numéro a changé de
    question et la réponse doit être refaite.
    """
    chemin = _cle_generation(item["question"], contexte, fenetre_servie, GENERATEUR, prompt)
    if not chemin.exists():
        return False
    try:
        garde = json.loads(chemin.read_text(encoding="utf-8")).get("content")
    except (json.JSONDecodeError, OSError):
        return False
    return isinstance(garde, str) and garde.strip() == (reponse.get("answer") or "").strip()


def _verdict_appartient(item: dict, contexte: list[dict], reponse: str, graine: str,
                        fenetre_juge: int) -> bool:
    """Le verdict gardé a-t-il été rendu à CETTE fenêtre du juge, sur CETTE réponse ?

    Symétrique de ``_reponse_appartient``, et découvert pour la même raison : le cache des
    verdicts était indexé par ``famille/qid`` seul. Or ``judge.grade`` reçoit ``characters`` —
    re-noter la même population à une autre fenêtre du juge trouvait donc l'ancien verdict,
    sautait l'appel, et écrivait un fichier annonçant ``fenetre_du_juge: 10000`` dont les
    verdicts avaient été rendus à 2 500. ``divergences()`` refuse de comparer deux runs dont la
    fenêtre du juge diffère ; elle ne pouvait rien contre un run dont l'en-tête ment sur la
    sienne, parce que la corruption est à l'intérieur du fichier.

    Le contrôle est le même que pour la génération : on reconstruit la clé de cache que
    ``judge.grade`` aurait produite ici et maintenant. Si elle existe, le verdict gardé vient du
    même prompt — même fenêtre, même réponse, même contexte. Sinon il doit être refait, et
    l'appel sera de toute façon servi par le cache disque s'il a déjà été payé un jour.
    """
    return _cle_juge(item, reponse, contexte, graine, fenetre_juge, JUGE).exists()


def appels_du_run(items: dict, contextes: dict, reponses: dict, fenetre_servie: int,
                  fenetre_juge: int, prompt: str):
    """Les ``(modele, clé de cache)`` de tous les appels qu'un run du banc a dû faire."""
    for cle, item in items.items():
        if cle not in reponses or cle not in contextes:
            continue
        contexte = contextes[cle]
        yield GENERATEUR, _cle_generation(item["question"], contexte, fenetre_servie,
                                          GENERATEUR, prompt)
        if item["famille"] in JUGEES:
            yield JUGE, _cle_juge(item, reponses[cle]["answer"], contexte, f"v4-{cle}",
                                  fenetre_juge, JUGE)


# --------------------------------------------------------------------------- en-tête

def en_tete(fenetre_servie: int, fenetre_juge: int, prompt: str) -> dict:
    """Les **six** champs qui gouvernent la comparabilité de deux runs.

    Le sixième — ``version_prompt`` — est né du chantier ``contrat-de-reponse``. Sans lui, un
    run v3 et un run v2 se comparaient sans un mot : les cinq champs d'origine sont identiques
    des deux côtés, puisque le corpus, les fenêtres, les modèles et les questions ne bougent
    pas. Le texte du contrat est pourtant **la** variable, et un banc qui refuse de comparer
    deux fenêtres du juge doit a fortiori refuser deux contrats de réponse.

    L'empreinte porte sur le **texte envoyé**, pas sur le nom de la version : renommer ``v3``
    ne tromperait personne, en réécrire une phrase, si.
    """
    return {
        "signature": SIGNATURE,
        "fenetre_servie": fenetre_servie,
        "fenetre_du_juge": fenetre_juge,
        "modeles": {"generateur": GENERATEUR, "juge": JUGE, "auteur": familles_v4.AUTEUR},
        "version_questions": {nom: familles_v4.empreinte(_fichier_famille(nom, fichier))
                              for nom, fichier in FAMILLES.items()},
        "version_prompt": {"nom": prompt, "sha256_16": pipeline.empreinte_prompt(prompt)},
    }


CHAMPS_GOUVERNANTS = ("signature", "fenetre_servie", "fenetre_du_juge", "modeles",
                      "version_questions", "version_prompt")


def _valeur_gouvernante(resultat: dict, champ: str):
    """La valeur qui gouverne la comparabilité — pour le contrat, son **texte**, pas son nom.

    Le nom d'une version est une étiquette ; le texte est la variable. Comparer le dict entier
    faisait du nom un champ gouvernant, et **le bras placebo l'a révélé** : `v2placebo` porte le
    texte de `v2`, octet pour octet, et le banc refusait de les comparer. Or un placebo dont on
    ne peut pas mesurer le contraste avec sa référence ne mesure rien — c'est le seul bras dont
    la comparaison doit être autorisée sans discussion, puisqu'il ne diffère par rien.

    L'inverse reste vrai et c'est ce qui compte : deux textes différents ont deux empreintes
    différentes, et aucun renommage ne peut les faire passer l'un pour l'autre. Le nom demeure
    dans le JSON, en information ; il ne décide plus.
    """
    valeur = resultat.get(champ)
    if champ == "version_prompt" and isinstance(valeur, dict):
        return valeur.get("sha256_16")
    return valeur


def divergences(a: dict, b: dict) -> list[str]:
    """Champs d'en-tête par lesquels deux runs diffèrent — la liste qui autorise ou refuse."""
    return [champ for champ in CHAMPS_GOUVERNANTS
            if _valeur_gouvernante(a, champ) != _valeur_gouvernante(b, champ)]


# --------------------------------------------------------------------------- population

def population(index: ChunkIndex | None = None) -> list[dict]:
    """Les questions v4, plus les 20 négatives v3. Population **fixe** : rien n'est écarté après coup.

    ``index`` n'est **pas lu** — il ne l'a jamais été. Le paramètre est gardé parce que des
    appelants le passent (``format_reponse_v3.py``), mais il est désormais facultatif, et les
    trois étapes qui n'en ont pas d'autre usage ne chargent plus rien : ``ChunkIndex.load()``
    coûte **184 Mo** de mémoire résidente, mesurés. Ce n'est pas une optimisation gratuite —
    c'est ce chargement inutile, additionné à celui d'une seconde session, qui a fait tuer par
    le système un run de trois bras à 76/199 le 10 septembre 2026.

    ``temoins()`` garde le sien : ``judge.build_traps`` en a réellement besoin.
    """
    items = []
    for famille, fichier in FAMILLES.items():
        for item in familles_v4.lire_jsonl(_fichier_famille(famille, fichier)):
            if famille == "negative_v3":
                if item.get("kind") != "negative":
                    continue
                item = {**item, "famille": "negative_v3"}
            else:
                item = {**item, "famille": famille}
            item["qid_banc"] = f"{famille}/{item['qid']}"
            items.append(item)
    return items


def servis(question: str) -> list[dict]:
    """Les cinq passages que la production servirait — ``_select`` compris."""
    import quant_rag  # noqa: E402  (import tardif : charge le modèle, ouvre Qdrant)
    return quant_rag.search(question, limit=PASSAGES, mode="dense", rerank=False,
                            auto_period=False, log=False)


# --------------------------------------------------------------------------- témoins

def temoins() -> dict:
    """Les 18 items-témoins du juge, **avant** toute lecture de verdict.

    Repris d'``eval_characters.temoins`` : mêmes témoins, même seuil, même famille bloquante.
    Le réinventer aurait produit un second seuil, et deux seuils qui divergent valent moins
    qu'un seul qu'on relit.
    """
    index = _index()
    tous = familles_v4.lire_jsonl(HERE / "questions-v3.jsonl")
    avant = (llm.GENERATOR, llm.JUDGE)
    llm.GENERATOR, llm.JUDGE = GENERATEUR, JUGE
    registre: dict = {}
    try:
        resultat, duree = _facturer(registre, JUGE,
                                    lambda: judge.run_traps(judge.build_traps(tous, index)))
    finally:
        llm.GENERATOR, llm.JUDGE = avant
    famille = (resultat["per_family"].get(FAMILLE_BLOQUANTE) or {}).get("accuracy")
    resultat["lisible"] = famille is not None and famille >= SEUIL_FAMILLE
    resultat["generateur"], resultat["juge"] = GENERATEUR, JUGE
    resultat["secondes"] = round(duree, 1)
    resultat["facturation"] = registre
    _ecrire(CACHE / f"v4-temoins-{SIGNATURE}.json", resultat)
    print(json.dumps({k: v for k, v in resultat.items() if k != "detail"},
                     ensure_ascii=False, indent=1))
    print(f"\n{FAMILLE_BLOQUANTE} = {famille}  seuil {SEUIL_FAMILLE}  -> "
          f"{'LISIBLE' if resultat['lisible'] else 'ILLISIBLE — le banc refuse de partir'}")
    return resultat


# --------------------------------------------------------------------------- le « où », seul

def contextes_seuls(bras: Bras, limite: int | None = None) -> dict:
    """L'étage 1 pour toute la population — récupération seule, **zéro appel LLM**.

    Le banc met déjà le « où » avant le « combien » question par question. En faire une étape
    qu'on puisse jouer seule a trois vertus, et aucune n'est cosmétique :

    - **l'or servi est connu avant d'avoir dépensé un centime.** Si la récupération s'est
      effondrée, on le sait avant de payer 200 générations pour le mesurer ;
    - **un budget qui s'arrête ne laisse pas les mains vides** : le « où » est acquis, publiable,
      et c'est la moitié du diagnostic ;
    - le run payant qui suit ne fait plus que des appels, sans attendre Qdrant à chaque tour.

    Les contextes périmés — ceux dont le numéro a changé de question — sont invalidés ici,
    au même titre que dans ``mesure``. L'étape se juge du point de vue d'**un** bras, celui de
    la référence : le « où » ne dépend ni du contrat ni de la fenêtre servie, seule la lecture
    de péremption en dépend.
    """
    fenetre_servie, prompt = bras.fenetre_servie, bras.prompt
    items = population()
    if limite:
        items = items[:limite]
    contextes = _charger(CACHE / f"v4-contextes-{SIGNATURE}.json")
    reponses = _charger(_cache_reponses(bras.nom))
    verdicts = _charger(_cache_verdicts(bras.nom))
    perimes, neufs, latences = 0, 0, []

    for numero, item in enumerate(items, 1):
        cle = item["qid_banc"]
        if (cle in reponses and cle in contextes
                and not _reponse_appartient(item, contextes[cle], reponses[cle],
                                            fenetre_servie, prompt)):
            perimes += 1
            for cache in (contextes, reponses, verdicts):
                cache.pop(cle, None)
        if cle not in contextes:
            depart = time.perf_counter()
            rangs = servis(item["question"])
            latences.append(time.perf_counter() - depart)
            contextes[cle] = [{k: r.get(k) for k in
                               ("chunk_id", "document_id", "title", "short_ref", "section",
                                "page_start", "text", "content_type", "score")}
                              for r in rangs]
            neufs += 1
            _ecrire(CACHE / f"v4-contextes-{SIGNATURE}.json", contextes)
        print(f"  {numero:4d}/{len(items)}  {cle:34s}", end="\r", flush=True)

    _ecrire(_cache_reponses(bras.nom), reponses)
    _ecrire(_cache_verdicts(bras.nom), verdicts)

    par_famille = {}
    for famille in FAMILLES:
        membres = [i for i in items if i["famille"] == famille and i.get("gold_chunks")]
        if not membres:
            continue
        servi = sum(1 for i in membres
                    if {r["chunk_id"] for r in contextes[i["qid_banc"]]} & set(i["gold_chunks"]))
        par_famille[famille] = {"n": len(membres), "or_servi": servi,
                                "part": round(servi / len(membres), 4)}
    sortie = {"signature": SIGNATURE, "fenetre_servie": fenetre_servie, "appels_llm": 0,
              "population": len(items), "contextes_neufs": neufs, "perimes_invalides": perimes,
              "latence_recuperation": _percentiles(latences), "or_servi": par_famille}
    _ecrire(CACHE / f"v4-ou-{SIGNATURE}.json", sortie)
    print(f"\n{json.dumps(sortie, ensure_ascii=False, indent=1)}")
    return sortie


# --------------------------------------------------------------------------- mesure

def _identifiant_de_run(bras: list[Bras], tirages: int) -> str:
    """Un identifiant qui change dès que le run change — c'est tout ce qu'on lui demande.

    Il sert à une seule chose, et elle est vitale : prouver au verdict que le plancher qu'on lui
    présente vient du **même** run que le bras qu'il juge. Un placebo tiré hier ne mesure pas le
    bruit d'aujourd'hui ; c'est même précisément l'erreur que ce banc existe pour interdire.
    """
    empreinte = hashlib.sha256(
        json.dumps([b.resume() for b in bras] + [tirages], sort_keys=True).encode()
    ).hexdigest()[:8]
    return f"{time.strftime('%Y%m%dT%H%M%S')}-{empreinte}"


def mesure(bras: list[Bras], fenetre_juge: int, limite: int | None = None,
           tirages: int = 1, cache_jetable_repris: Path | None = None) -> dict:
    """Un passage complet sur TOUS les bras — entrelacés **par question**.

    Ce que l'entrelacement retire, et pourquoi il fallait le retirer
    -----------------------------------------------------------------
    Mesurer le bras A en entier puis le bras B en entier, c'est comparer deux moments : deux
    états du service distant, deux files d'attente, deux versions possibles d'un alias
    ``-latest``, et deux tirages d'un générateur dont on a mesuré qu'il n'est pas reproductible.
    Entrelacer par question borne tout cela à l'intervalle entre deux appels consécutifs. La
    récupération, elle, n'est faite **qu'une fois par question** et partagée par tous les bras :
    c'est ce qui fait de la comparaison un appariement — mêmes cinq passages, à la ligne près.

    Les bras jetables (``placebo``) écrivent dans un cache d'appels temporaire, effacé à la fin
    après que leur coût en a été extrait. Leur état est **reparti de zéro à chaque run** : un
    placebo repris sur cache ne serait pas un tirage neuf, donc pas un plancher.

    ``cache_jetable_repris`` est la seule porte de sortie de cette règle, et elle est **explicite
    par construction**. Le 10 septembre 2026, un run de trois bras a été tué à 76/199 par une
    pression mémoire venue d'une autre session : les deux bras ordinaires ont repris sur leur
    état écrit au fil de l'eau, mais **le placebo avait tout perdu** — son état n'était écrit
    qu'à la fin — alors que ses 121 appels étaient payés et intacts dans son répertoire
    temporaire, que le ``rmtree`` de fin n'avait jamais eu l'occasion d'effacer.

    Deux corrections en sont sorties, et la première est la vraie :

    - **l'état d'un bras jetable s'écrit maintenant au fil de l'eau**, comme celui des autres.
      Un run conçu pour être repris ne peut pas avoir un bras qui ne l'est pas ;
    - ``--plancher-cache DIR`` rend ce répertoire au run suivant. Le placebo reprend alors sur
      ses propres appels — donc sur le **même** tirage, ce qui est la seule reprise correcte —
      et son état n'est pas remis à zéro. Sans l'option, rien ne change : cache neuf, état
      neuf. Il faut le demander, on ne peut pas y tomber.
    """
    temoins_lus = _charger(CACHE / f"v4-temoins-{SIGNATURE}.json")
    if not temoins_lus:
        sys.exit("items-témoins non mesurés — lancer --etape temoins d'abord (verrou d'eval_characters l. 112-116)")
    if not temoins_lus.get("lisible"):
        sys.exit(f"{FAMILLE_BLOQUANTE} sous {SEUIL_FAMILLE} : aucun verdict ne serait lisible")

    items = population()
    if limite:
        items = items[:limite]
    contextes = _charger(CACHE / f"v4-contextes-{SIGNATURE}.json")

    # Un bras jetable repart de zéro : son état d'hier a été produit par un cache d'appels qui
    # n'existe plus, et son intérêt tient tout entier dans le fait d'être tiré aujourd'hui.
    # L'exception, explicite, est la reprise d'un run interrompu — voir la docstring.
    reprise = cache_jetable_repris is not None
    etats = {b.nom: {"reponses": {} if (b.jetable and not reprise) else _charger(_cache_reponses(b.nom)),
                     "verdicts": {} if (b.jetable and not reprise) else _charger(_cache_verdicts(b.nom)),
                     "tirages": {}}
             for b in bras}
    registres: dict[str, dict] = {b.nom: {} for b in bras}
    perimes = {b.nom: 0 for b in bras}
    perimes_juge = {b.nom: 0 for b in bras}
    latences = {"recuperation": [], "generation": [], "notation": []}

    racine_temporaire = (cache_jetable_repris if reprise
                         else Path(tempfile.mkdtemp(prefix="banc-v4-jetable-")))
    caches_jetables = {b.nom: racine_temporaire / b.nom for b in bras if b.jetable}
    # Les tirages 2..k passent par un cache neuf QUEL QUE SOIT le bras : à température 0 le
    # cache est indexé par l'empreinte du payload, donc un second tirage servi par le cache
    # rendrait la première réponse et la variance intra-question publiée vaudrait zéro par
    # construction. Ce serait le pire des chiffres : faux, et rassurant.
    #
    # Un cache par (bras, tirage), et non par tirage seul : un placebo porte le MÊME payload
    # que sa référence — c'est sa définition. Un cache partagé au rang t servirait donc au
    # placebo la réponse que la référence vient d'y écrire, et le plancher de bruit tomberait à
    # zéro par construction, au rang où on le mesure.
    caches_tirages = {(b.nom, t): racine_temporaire / f"{b.nom}-tirage-{t + 1}"
                      for b in bras for t in range(1, tirages)}
    for chemin in (*caches_jetables.values(), *caches_tirages.values()):
        chemin.mkdir(parents=True, exist_ok=True)

    run_id = _identifiant_de_run(bras, tirages)
    print(f"{len(items)} questions · {len(bras)} bras entrelacés par question · "
          f"{tirages} tirage(s) · run {run_id}")
    print(f"générateur {GENERATEUR} · juge {JUGE} · fenêtre du juge {fenetre_juge}")
    for b in bras:
        marque = {"reference": "référence", "placebo": "PLACEBO (plancher de bruit)",
                  "variante": "variante"}[b.role]
        print(f"  {b.nom:16s} contrat {b.prompt} ({pipeline.empreinte_prompt(b.prompt)}) · "
              f"fenêtre servie {b.fenetre_servie:6d} · {marque}"
              + ("  · cache d'appels NEUF, effacé en fin de run" if b.jetable else ""))
    print()

    for numero, item in enumerate(items, 1):
        cle = item["qid_banc"]

        # --- étage 0 : ce qui est en cache appartient-il à CETTE question ?
        #
        # Le cache est indexé par ``famille/qid``, et un qid n'est PAS une identité : quand une
        # famille grandit, la question n° 7 d'hier peut être la n° 9 d'aujourd'hui. Réutiliser
        # au seul vu du numéro, ce serait noter une question avec la réponse d'une autre — une
        # faute de mesure, pas une économie ratée. On reconstruit la clé de cache LLM que
        # ``pipeline.answer`` aurait produite pour la question courante et son contexte gardé :
        # si elle existe et porte le même texte, les deux viennent du même prompt.
        #
        # **Le contexte ne tombe que si AUCUN bras ne le reconnaît.** À un seul bras, c'est
        # exactement l'ancien comportement (20 entrées périmées sur la population du 9
        # septembre). À plusieurs, laisser un bras périmé jeter le contexte ferait repayer la
        # récupération et la génération de tous les autres — et surtout casserait
        # l'appariement, qui repose sur des passages communs.
        reconnu, inconnu = False, True
        for b in bras:
            etat = etats[b.nom]
            if cle in etat["reponses"] and cle in contextes:
                inconnu = False
                # Sous le cache du BRAS, et pas sous celui du dépôt. ``_reponse_appartient``
                # reconstruit une clé de cache d'appels ; les réponses d'un bras jetable vivent
                # dans son répertoire temporaire. Chercher ailleurs les déclarerait toutes
                # périmées, ce qui ferait repayer le run entier au moment précis où l'on croit
                # le reprendre.
                with cache_d_appels(caches_jetables.get(b.nom)):
                    appartient = _reponse_appartient(item, contextes[cle], etat["reponses"][cle],
                                                     b.fenetre_servie, b.prompt)
                if appartient:
                    reconnu = True
                else:
                    perimes[b.nom] += 1
                    etat["reponses"].pop(cle, None)
                    etat["verdicts"].pop(cle, None)
        if not inconnu and not reconnu:
            contextes.pop(cle, None)

        # --- étage 1 : le « où ». Aucun appel LLM, il vient EN PREMIER, et il est COMMUN.
        if cle not in contextes:
            depart = time.perf_counter()
            rangs = servis(item["question"])
            latences["recuperation"].append(time.perf_counter() - depart)
            contextes[cle] = [{k: r.get(k) for k in
                               ("chunk_id", "document_id", "title", "short_ref", "section",
                                "page_start", "text", "content_type", "score")}
                              for r in rangs]
            _ecrire(CACHE / f"v4-contextes-{SIGNATURE}.json", contextes)
        contexte = contextes[cle]

        for b in bras:
            etat, registre = etats[b.nom], registres[b.nom]
            with cache_d_appels(caches_jetables.get(b.nom)):
                # --- étage 2 : la réponse, telle que la production la produirait.
                if cle not in etat["reponses"]:
                    sortie, duree = _facturer(registre, GENERATEUR, lambda b=b: pipeline.answer(
                        item["question"], contexte, model=GENERATEUR, prompt=b.prompt,
                        characters=b.fenetre_servie))
                    latences["generation"].append(duree)
                    etat["reponses"][cle] = {"answer": sortie["answer"],
                                             "abstained": sortie["abstained"]}
                    # Au fil de l'eau, **y compris pour un bras jetable**. « Jetable » qualifie
                    # son cache d'appels, pas son travail : le 10 septembre, un run tué à 76/199
                    # a perdu 121 appels de placebo déjà payés parce que son état n'était écrit
                    # qu'à la fin. Un run conçu pour être repris ne peut pas avoir un bras qui
                    # ne l'est pas.
                    _ecrire(_cache_reponses(b.nom), etat["reponses"])

                # --- étage 2 bis : les tirages 2..k. Générateur seul, jamais le juge (§4c).
                if tirages > 1:
                    autres = []
                    for t in range(1, tirages):
                        with cache_d_appels(caches_tirages[(b.nom, t)]):
                            sortie, duree = _facturer(
                                registre, GENERATEUR, lambda b=b: pipeline.answer(
                                    item["question"], contexte, model=GENERATEUR,
                                    prompt=b.prompt, characters=b.fenetre_servie))
                            latences["generation"].append(duree)
                            autres.append({"answer": sortie["answer"],
                                           "abstained": sortie["abstained"]})
                    etat["tirages"][cle] = autres

                # --- étage 3 : le juge, et seulement pour les familles qui en ont besoin.
                #
                # Même garde qu'à l'étage 0, sur l'autre fenêtre : un verdict rendu à 2 500 ne
                # vaut pas pour un run qui juge à 10 000, et rien dans son numéro ne le dit.
                if (item["famille"] in JUGEES and cle in etat["verdicts"]
                        and not _verdict_appartient(item, contexte,
                                                    etat["reponses"][cle]["answer"],
                                                    f"v4-{cle}", fenetre_juge)):
                    etat["verdicts"].pop(cle)
                    perimes_juge[b.nom] += 1

                if item["famille"] in JUGEES and cle not in etat["verdicts"]:
                    v, duree = _facturer(registre, JUGE, lambda: judge.grade(
                        item, etat["reponses"][cle]["answer"], contexte, seed=f"v4-{cle}",
                        model=JUGE, characters=fenetre_juge))
                    latences["notation"].append(duree)
                    etat["verdicts"][cle] = v
                    _ecrire(_cache_verdicts(b.nom), etat["verdicts"])

        print(f"  {numero:4d}/{len(items)}  {cle:34s}", end="\r", flush=True)

    # Les bras jetables : leur coût est extrait de leur cache temporaire AVANT sa destruction.
    # Le faire après rendrait `appels_manquants` égal à la population et le coût publié serait
    # celui du registre en mémoire — juste ici, mais faux dès qu'un run est repris. On préfère
    # le chiffre exact tant qu'il est encore lisible.
    couts = {}
    for b in bras:
        etat = etats[b.nom]
        if etat["tirages"]:
            _ecrire(_cache_tirages(b.nom), etat["tirages"])
        with cache_d_appels(caches_jetables.get(b.nom)):
            depuis_cache = facturer_depuis_le_cache(
                appels_du_run({i["qid_banc"]: i for i in items}, contextes, etat["reponses"],
                              b.fenetre_servie, fenetre_juge, b.prompt))
        couts[b.nom] = {**chiffrer(depuis_cache["registre"]),
                        "appels_manquants": depuis_cache["manquants"]}
        _ecrire(_cache_facturation(b.nom),
                {"run_id": run_id, "bras": b.resume(), "registre": registres[b.nom],
                 "chiffrage": chiffrer(registres[b.nom]),
                 "chiffrage_depuis_le_cache": couts[b.nom],
                 "latences_secondes": {etage: sorted(v) for etage, v in latences.items()},
                 "stats_globales": llm.stats()})

    # Le répertoire d'une reprise appartient à l'opérateur : il l'a nommé, il le supprimera.
    # L'effacer d'office ferait perdre les appels d'un run qui vient peut-être d'être interrompu
    # une seconde fois — exactement la panne qu'on est en train de fermer.
    if not reprise:
        shutil.rmtree(racine_temporaire, ignore_errors=True)
    elif caches_jetables:
        print(f"cache jetable CONSERVÉ (repris à votre demande) : {racine_temporaire}")

    manifeste = {
        "run_id": run_id, "signature": SIGNATURE, "date": time.strftime("%Y-%m-%d"),
        "fenetre_du_juge": fenetre_juge, "tirages": tirages, "population": len(items),
        "bras": [b.resume() for b in bras],
        "reference": bras[0].nom,
        "placebo": next((b.nom for b in bras if b.role == "placebo"), None),
        "lecture": ("le verdict n'accepte un plancher que s'il porte CE run_id — un placebo "
                    "tiré un autre jour ne mesure pas le bruit de celui-ci"),
    }
    _ecrire(_fichier_run(), manifeste)

    print(f"\nfait — run {run_id} — {llm.stats()}")
    for b in bras:
        if perimes[b.nom]:
            print(f"{b.nom}: {perimes[b.nom]} réponse(s) invalidée(s) — elles ne "
                  f"correspondaient plus à la question portant ce numéro.")
        if perimes_juge[b.nom]:
            print(f"{b.nom}: {perimes_juge[b.nom]} verdict(s) invalidé(s) — rendus à une autre "
                  f"fenêtre du juge que {fenetre_juge}, ou sur une autre réponse.")
        print(f"{b.nom}: {json.dumps(couts[b.nom]['total_usd'])} USD "
              f"(run entier, depuis le cache ; manquants {couts[b.nom]['appels_manquants']})")
    total = sum(c["total_usd"] for c in couts.values())
    print(f"total du run : {round(total, 4)} USD")
    return manifeste


# --------------------------------------------------------------------------- notation

def _percentiles(valeurs: list[float]) -> dict | None:
    if not valeurs:
        return None
    ordre = sorted(valeurs)
    return {"n": len(ordre),
            "p50": round(ordre[int(0.50 * (len(ordre) - 1))], 3),
            "p95": round(ordre[int(0.95 * (len(ordre) - 1))], 3)}


def noter(item: dict, reponse: dict, contexte: list[dict], verdict: dict,
          fenetre_servie: int) -> dict:
    """Le verdict d'une question — déterministe d'abord, jugé seulement si nécessaire."""
    texte = reponse.get("answer") or ""
    or_servi = bool({r["chunk_id"] for r in contexte} & set(item.get("gold_chunks") or []))
    grandeurs_absentes = pipeline.non_trouve(texte)
    ligne = {
        "famille": item["famille"], "qid": item["qid"],
        "or_servi": or_servi,
        "abstenue": bool(reponse.get("abstained")),
        # Le second signal du contrat v3, lu sans juge. Il est **distinct** de l'abstention et
        # doit le rester : « rien ne porte sur la question » et « le sujet est couvert, la
        # grandeur non » sont deux états du produit, et les confondre est exactement ce qui
        # rendait `negative_voisine` illisible. Vide sous v1 et v2, qui ne le connaissent pas.
        "non_trouve": grandeurs_absentes,
        "signale_absence": bool(grandeurs_absentes),
        "mots": len(texte.split()),
        "documents_servis": sorted({r["document_id"] for r in contexte}),
    }
    if item["famille"] in ("formula", "table_cell"):
        note = (score_formule.score(texte, item["or_latex"], item.get("leurres"))
                if item["famille"] == "formula"
                else score_tableau.score(texte, item["or_valeur"], item.get("autres_valeurs")))
        # Les deux scoreurs rendent leur propre lecture de l'abstention, **plus stricte** que
        # celle du pipeline : ils exigent le jeton au DÉBUT, `pipeline.abstained` le cherche
        # n'importe où. L'écart n'est pas du bruit — le chantier E a mesuré 6 réponses v3 qui
        # commencent par le jeton puis répondent quand même. On garde les deux, nommées, plutôt
        # que d'en écraser une avec l'autre au détour d'un `**note`.
        note = dict(note)
        ligne["abstenue_en_tete"] = bool(note.pop("abstenue", note.pop("abstention", False)))
        ligne.update({"score": int(bool(note["juste"])), **note})
    else:
        # Familles négatives : le score EST l'abstention, mesurée sur une chaîne, sans juge.
        ligne["score"] = int(bool(reponse.get("abstained")))
        ligne["fabriquee"] = bool((verdict or {}).get("fabricated"))
    if verdict:
        ligne["couverture_jugee"] = verdict.get("coverage")
        ligne["ancrage_juge"] = verdict.get("groundedness")
    # **Les citations sont désormais vérifiées sur les quatre familles.** Elles ne l'étaient que
    # sur les deux positives, et l'agrégat publié — « 39,2 % d'affirmations sans citation » —
    # ne portait donc pas sur les 199 questions mais sur 145, c'est-à-dire **pas** là où sont
    # les fabrications. Une négative voisine qui affirme un chiffre absent produit une
    # affirmation vérifiable comme une autre ; l'exclure du compte revenait à ne pas regarder
    # l'endroit qu'on cherche. Les deux populations sont publiées séparément par
    # ``_agreger_citations`` : mélanger 153 affirmations sur 145 questions avec un total sur
    # 199 ferait bouger un taux sans qu'aucune réponse ait changé.
    ligne["citation"] = score_citation.verifier(texte, contexte, fenetre=fenetre_servie)
    return ligne


def _ic(deltas: list[float]) -> list[float] | None:
    """Bootstrap **apparié** — on rééchantillonne les différences, pas les deux bras séparément."""
    if not deltas:
        return None
    alea = random.Random(GRAINE)
    tirages = sorted(statistics.fmean(alea.choice(deltas) for _ in deltas) for _ in range(TIRAGES))
    return [round(tirages[int(0.025 * TIRAGES)], 4), round(tirages[int(0.975 * TIRAGES)], 4)]


def _test_exact(deltas: list[float]) -> dict | None:
    """McNemar exact sur les paires discordantes — le compagnon obligatoire du bootstrap à n petit.

    Pourquoi il fallait l'ajouter
    ------------------------------
    Le bootstrap rééchantillonne la population observée. Quand cette population est petite et
    que les écarts y sont rares, il rend des intervalles **dégénérés** : le lot 6 a publié un
    ``[0 ; 0]`` sur une famille de 20 items, c'est-à-dire un intervalle qui affirme une
    précision infinie là où il n'y a presque pas d'information. Un lecteur pressé y lit « Δ nul,
    et très bien mesuré » ; la vérité est « aucune paire discordante, donc rien à dire ».

    Le test exact ne peut pas mentir de cette façon : sur ``b + c`` paires discordantes il rend
    la probabilité binomiale exacte, et à ``b + c = 0`` il rend ``p = 1`` en disant qu'il n'a vu
    aucune discordance. Il ne remplace pas le bootstrap — il ne donne pas d'intervalle sur
    l'effet — il **l'accompagne**, et il est publié pour toutes les familles parce qu'il ne
    coûte rien ; il est *exigé* pour n <= 20.

    Ne s'applique qu'à des scores binaires appariés, ce que sont tous les scores de famille du
    banc (``int(bool(...))``). Rend ``None`` sinon plutôt qu'un chiffre qui n'aurait pas de sens.
    """
    if not deltas or any(d not in (-1.0, 0.0, 1.0) for d in map(float, deltas)):
        return None
    from scipy.stats import binomtest  # noqa: PLC0415  — import local : scipy coûte ~0,3 s

    montent = sum(1 for d in deltas if d > 0)
    descendent = sum(1 for d in deltas if d < 0)
    discordantes = montent + descendent
    p = (1.0 if discordantes == 0
         else binomtest(montent, discordantes, 0.5, alternative="two-sided").pvalue)
    return {"nom": "McNemar exact (binomial two-sided sur les paires discordantes)",
            "paires_discordantes": discordantes, "montent": montent, "descendent": descendent,
            "p": round(float(p), 6), "significatif_5pct": bool(p < 0.05),
            "lecture": ("p = 1 sans aucune paire discordante signifie « aucune information », "
                        "pas « aucun effet »")}


def _recouvrent(a: list[float] | None, b: list[float] | None) -> bool | None:
    """Deux intervalles se recouvrent-ils ? ``None`` si l'un des deux manque."""
    if not a or not b:
        return None
    return a[0] <= b[1] and b[0] <= a[1]


def _lignes_du_bras(bras: Bras, items: dict, contextes: dict) -> dict:
    """Les lignes notées d'un bras, lues depuis ses seuls fichiers d'état. Zéro appel."""
    reponses = _charger(_cache_reponses(bras.nom))
    verdicts = _charger(_cache_verdicts(bras.nom))
    return {cle: noter(items[cle], reponses[cle], contextes[cle], verdicts.get(cle),
                       bras.fenetre_servie)
            for cle in items
            if cle in reponses and cle in contextes}


def _variance_intra_question(bras: Bras, items: dict, contextes: dict, lignes: dict) -> dict | None:
    """La dispersion du score d'une MÊME question sur k tirages — le plancher, par question.

    C'est la grandeur qui manquait pour qu'un seuil soit un seuil. Un Δ entre deux bras se lit
    contre le bruit **entre** bras (le placebo) ; la stabilité d'une question donnée se lit
    contre le bruit **à l'intérieur** d'une question, et les deux ne se déduisent pas l'un de
    l'autre. Une famille dont chaque question bascule d'un tirage à l'autre ne peut porter aucun
    seuil, même si sa moyenne est stable.

    Le **juge n'est pas rejoué** sur les tirages 2..k : ce serait multiplier par k le modèle le
    plus cher pour la grandeur la moins reproductible. La variance est donc publiée sur ce qui
    se note sans juge — score déterministe des positives, abstention des négatives — et le champ
    le dit.
    """
    autres = _charger(_cache_tirages(bras.nom))
    if not autres:
        return None
    par_famille: dict[str, list[float]] = {}
    retenues = instables = 0
    for cle, tirages_de_la_question in autres.items():
        if cle not in lignes or cle not in items:
            continue
        scores = [lignes[cle]["score"]]
        for tirage in tirages_de_la_question:
            scores.append(noter(items[cle], tirage, contextes[cle], None,
                                bras.fenetre_servie)["score"])
        instable = 1.0 if len(set(scores)) > 1 else 0.0
        par_famille.setdefault(lignes[cle]["famille"], []).append(instable)
        retenues += 1
        instables += int(instable)
    if not retenues:
        return None
    n_tirages = 1 + max((len(v) for v in autres.values()), default=0)
    return {
        "tirages": n_tirages,
        "questions": retenues,
        "questions_instables": instables,
        "part_instable": round(instables / retenues, 4),
        "par_famille": {f: {"n": len(v), "instables": int(sum(v)),
                            "part_instable": round(statistics.fmean(v), 4)}
                        for f, v in par_famille.items()},
        "note": ("« instable » = le score de cette question n'est pas le même sur les k "
                 "tirages. Le juge n'est pas rejoué : la mesure porte sur les scores "
                 "déterministes et sur l'abstention, jamais sur `fabriquee` ni sur la "
                 "couverture jugée."),
    }


def verdict(bras: Bras, fenetre_juge: int, reference: Path | None, drapeau: bool,
            plancher: Path | None = None, sans_plancher: bool = False) -> dict:
    # Le bras vient du contrat de réponse : il nomme les fichiers d'état et entre dans les
    # champs gouvernants. La vue corpus injectable de la représentation (``_index()``) ne sert
    # PAS ici — ``population`` ne lit pas d'index —, et l'y charger coûtait 184 Mo pour rien.
    fenetre_servie, prompt = bras.fenetre_servie, bras.prompt
    items = {i["qid_banc"]: i for i in population()}
    contextes = _charger(CACHE / f"v4-contextes-{SIGNATURE}.json")
    reponses = _charger(_cache_reponses(bras.nom))
    verdicts = _charger(_cache_verdicts(bras.nom))
    temoins_lus = _charger(CACHE / f"v4-temoins-{SIGNATURE}.json")
    facturation = _charger(_cache_facturation(bras.nom))
    manifeste = _charger(_fichier_run())

    lignes = {}
    for cle, item in items.items():
        if cle not in reponses or cle not in contextes:
            continue
        lignes[cle] = noter(item, reponses[cle], contextes[cle], verdicts.get(cle),
                            fenetre_servie)

    # Un verdict à population nulle n'est pas un verdict vide : c'est un fichier de résultat
    # d'apparence normale, avec ses en-têtes, ses coûts à zéro et ses familles absentes, écrit
    # PAR-DESSUS celui du même nom. Le cas s'est produit le 10 septembre 2026 — un bras dont
    # l'état vivait dans un autre worktree —, et il a écrasé un artefact publié en une seconde,
    # sans un mot. Le banc refuse désormais d'écrire ce qu'il n'a pas mesuré.
    if not lignes:
        sys.exit(f"REFUS d'écrire un verdict à population NULLE pour le bras « {bras.nom} ».\n"
                 f"  Aucune question n'a à la fois un contexte et une réponse en cache d'état.\n"
                 f"  Attendus :  {_cache_reponses(bras.nom)}\n"
                 f"              {CACHE / f'v4-contextes-{SIGNATURE}.json'}\n"
                 f"  Un fichier results-v4-{SIGNATURE}-{bras.nom}.json existe peut-être déjà et "
                 f"aurait été écrasé par un résultat vide.")

    par_famille = {}
    for famille in FAMILLES:
        membres = [l for l in lignes.values() if l["famille"] == famille]
        if not membres:
            continue
        scores = [l["score"] for l in membres]
        servis_ = [l for l in membres if l["or_servi"]]
        par_famille[famille] = {
            "n": len(membres),
            # Population FIXE : les zéros comptent. Un score moyen calculé sur les seules
            # questions « réussies » monterait mécaniquement à chaque question perdue.
            "score": round(statistics.fmean(scores), 4),
            "zeros": sum(1 for s in scores if s == 0),
            "abstentions": sum(1 for l in membres if l["abstenue"]),
            "or_servi": sum(1 for l in membres if l["or_servi"]),
            "score_si_or_servi": (round(statistics.fmean(l["score"] for l in servis_), 4)
                                  if servis_ else None),
            "couverture_jugee": _moyenne([l.get("couverture_jugee") for l in membres]),
        }
        if famille == "formula":
            par_famille[famille].update({
                "juste_strict": sum(1 for l in membres if l.get("juste_strict")),
                "recouvrement_moyen": _moyenne([l.get("recouvrement") for l in membres]),
                "balisee": sum(1 for l in membres if l.get("balisee")),
                "leurre_inclus": sum(1 for l in membres if l.get("leurre_inclus")),
            })
        if famille == "table_cell":
            # « mauvaise cellule » : pas la bonne valeur, mais une valeur du même tableau.
            # C'est la seule erreur que cette famille existe pour voir — le système avait le
            # passage sous les yeux et a lu la ligne d'à côté.
            par_famille[famille]["mauvaise_cellule"] = sum(
                1 for l in membres if l.get("mauvaise_cellule"))
            par_famille[famille]["unite_divergente"] = sum(
                1 for l in membres if l.get("unite_divergente"))
        if famille.startswith("negative"):
            par_famille[famille]["fabrications"] = sum(1 for l in membres if l.get("fabriquee"))

    citations = _agreger_citations(lignes)
    signal = _agreger_signal(lignes)
    garde = _garde_or_servi(lignes)

    # Le coût publié vient du CACHE. Le registre en mémoire meurt avec le processus et
    # sous-estime toute campagne reprise (0,6319 contre 0,9094 sur le graphe).
    #
    # Les deux chiffres sont vrais et ne mesurent PAS la même chose — les confondre serait
    # le piège que cette correction est censée fermer :
    #   - depuis le cache : ce que coûte de produire ce run ENTIER à froid. C'est le chiffre
    #     à citer pour « combien coûtera de rejouer le banc », et il est stable aux reprises ;
    #   - le registre : ce que CE processus a payé. Le reste était déjà en cache — soit d'un
    #     run antérieur, soit d'une passe interrompue dont le compteur est mort avec elle.
    depuis_cache = facturer_depuis_le_cache(
        appels_du_run(items, contextes, reponses, fenetre_servie, fenetre_juge, prompt))
    cout = {**chiffrer(depuis_cache["registre"]),
            "source": "cache disque — reconstitution du run entier",
            "appels_manquants": depuis_cache["manquants"]}
    registre_memoire = facturation.get("chiffrage")
    if registre_memoire:
        cout["marginal_de_ce_processus_usd"] = registre_memoire.get("total_usd")
        cout["deja_en_cache_usd"] = round(
            cout["total_usd"] - (registre_memoire.get("total_usd") or 0.0), 4)

    sortie = {
        **en_tete(fenetre_servie, fenetre_juge, prompt),
        "bras": bras.resume(),
        "run_id": facturation.get("run_id"),
        "date": "2026-09-08",
        "pre_enregistrement": "PRE-ENREGISTREMENT-INSTRUMENT-V4-2026-09-08.md",
        "population": len(lignes),
        "temoins": {k: v for k, v in temoins_lus.items() if k != "detail"},
        "par_famille": par_famille,
        "citations": citations,
        "signal_absence": signal,
        "garde_or_servi": garde,
        "variance_intra_question": _variance_intra_question(bras, items, contextes, lignes),
        "seuils": {"abstention": GARDE_ABSTENTION, "degradation": GARDE_DEGRADATION,
                   "famille_bloquante": FAMILLE_BLOQUANTE, "seuil_famille": SEUIL_FAMILLE,
                   "avertissement": ("`abstention` et `degradation` viennent d'eval_characters, "
                                     "calibrés sur un générateur supposé déterministe. Le "
                                     "placebo du 9 septembre 2026 rend a = +3 à +10 à variable "
                                     "NULLE : ils ne concluent qu'accompagnés du plancher du "
                                     "même run.")},
        "latence": {etage: _percentiles(v) for etage, v in
                    (facturation.get("latences_secondes") or {}).items()},
        "cout": cout,
        "detail": lignes,
    }

    if reference is not None:
        # --- le plancher est OBLIGATOIRE, et c'est le cœur de la réparation.
        #
        # Trois chantiers de suite ont publié des Δ dont un bras placebo, tiré après coup, a
        # reproduit le seul résultat significatif à variable nulle (+0,1176 [+0,029 ; +0,235]
        # sur les fabrications, exactement). Publier un Δ sans son plancher n'est pas une
        # imprudence de lecture : c'est produire un chiffre dont on sait qu'il peut être
        # entièrement du bruit, et le publier quand même. Le banc refuse désormais.
        #
        # Un bras dont le rôle EST « placebo » est exempté : son Δ contre la référence est le
        # plancher — lui demander le sien serait une régression à l'infini.
        role_place = bras.role == "placebo" or manifeste.get("placebo") == bras.nom
        plancher_lu = None
        if plancher is not None:
            plancher_lu = _charger(plancher)
            if not plancher_lu:
                sys.exit(f"--plancher {plancher} : fichier introuvable ou vide.")
            comparaison_plancher = plancher_lu.get("comparaison")
            if not comparaison_plancher:
                sys.exit(f"--plancher {plancher} : ce fichier ne porte pas de `comparaison`. "
                         f"Un plancher est le Δ du placebo CONTRE LA MÊME RÉFÉRENCE — produire "
                         f"d'abord `--etape verdict --bras <placebo> --reference {reference}`.")
            mien, sien = sortie.get("run_id"), plancher_lu.get("run_id")
            if mien and sien and mien != sien and not sans_plancher:
                sys.exit(f"REFUS : le plancher vient du run {sien}, ce bras du run {mien}. Un "
                         f"placebo tiré un autre jour ne mesure pas le bruit de celui-ci. "
                         f"Rejouer les deux bras dans le même run, ou assumer avec "
                         f"--sans-plancher (l'écart sera écrit dans le JSON).")
            comparaison_plancher = {**comparaison_plancher,
                                    "bras": (plancher_lu.get("bras") or {}).get("nom"),
                                    "run_id": sien}
        elif plancher_requis(bras, manifeste, plancher, sans_plancher):
            sys.exit(
                "REFUS de publier un Δ sans plancher de bruit.\n"
                "  Le générateur n'est pas reproductible à température 0 (51/130, 54/130 et "
                "110/199 réponses identiques à prompt identique au bit près), et un bras "
                "placebo a déjà reproduit à variable NULLE le seul Δ significatif d'un "
                "chantier entier.\n"
                "  Tirer le plancher :  --etape mesure --bras <ref>,<candidat>,<nom>:placebo\n"
                "  puis                 --etape verdict --bras <nom> --reference <ref.json>\n"
                "  puis                 --etape verdict --bras <candidat> --reference <ref.json> "
                "--plancher <nom.json>\n"
                "  Pour passer outre en l'assumant :  --sans-plancher  (écrit dans le JSON).")

        comparaison = comparer(sortie, _charger(reference), drapeau)
        if plancher is None and sans_plancher and not role_place:
            comparaison = _conclure_contre_le_plancher(comparaison, None)
            comparaison["plancher"]["sans_plancher_assume"] = True
            comparaison["plancher"]["consequence"] = (
                "aucun Δ de ce fichier n'est concluant : il n'a pas de plancher de bruit, et "
                "l'opérateur a explicitement demandé à s'en passer")
        elif role_place:
            comparaison["plancher"] = {"present": False, "regle": REGLE_DU_PLANCHER,
                                       "role": "ce bras EST le placebo — son Δ est le plancher"}
            for bloc in comparaison["par_famille"].values():
                bloc["conclusion"] = "PLANCHER"
        else:
            comparaison = _conclure_contre_le_plancher(comparaison, comparaison_plancher)
        sortie["comparaison"] = comparaison
    # Le nom porte le bras. Sans lui, le run v3 écraserait le fichier du run v2 à l'endroit même
    # où l'on vient de faire en sorte que les deux coexistent — et un placebo, qui porte le même
    # contrat que sa référence, écraserait la référence elle-même.
    _ecrire(HERE / f"results-v4-{SIGNATURE}-{bras.nom}.json", sortie)
    return sortie


def _moyenne(valeurs) -> float | None:
    propres = [v for v in valeurs if v is not None]
    return round(statistics.fmean(propres), 4) if propres else None


def _compter_citations(lignes: list[dict]) -> dict:
    """L'agrégat de citation d'un sous-ensemble de lignes — et le verbatim, qui est neuf.

    ``extraits_retrouves`` est la seule métrique de citation que le contrat v3 ne peut pas
    satisfaire par sa seule mise en forme, et c'est pour cela qu'elle existe. Exiger un extrait
    verbatim fait mécaniquement monter « appuyée » — le chiffre cité est dans la citation, donc
    dans le passage. Un générateur qui cite **beaucoup** et cite **faux** fait, lui, monter le
    nombre d'extraits et baisser ce taux. Ligne de base v2 : 16 extraits, 7 retrouvés (0,4375).
    """
    total = Counter()
    extraits = retrouves = 0
    for ligne in lignes:
        bloc = ligne.get("citation") or {}
        for categorie in ("appuyee", "mauvais_passage", "absente", "sans_citation", "hors_bornes"):
            total[categorie] += (bloc.get("comptes") or {}).get(categorie, 0)
        for extrait in bloc.get("extraits") or []:
            extraits += 1
            retrouves += bool(extrait.get("retrouve"))
    citees = total["appuyee"] + total["mauvais_passage"] + total["absente"]
    affirmations = citees + total["sans_citation"]
    return {
        "questions": len(lignes),
        "affirmations": affirmations,
        "comptes": dict(total),
        "precision_des_citations": round(total["appuyee"] / citees, 4) if citees else None,
        "part_sans_citation": round(total["sans_citation"] / affirmations, 4) if affirmations else None,
        "part_mauvais_passage": round(total["mauvais_passage"] / citees, 4) if citees else None,
        "extraits_verbatim": extraits,
        "extraits_retrouves": retrouves,
        "part_extraits_retrouves": round(retrouves / extraits, 4) if extraits else None,
    }


def _agreger_citations(lignes: dict) -> dict:
    """Deux populations, jamais mélangées : les 145 positives, et les 199 questions.

    La ligne de base du 9 septembre ne mesurait les citations que sur les positives. Publier
    aujourd'hui un seul taux sur 199 le ferait bouger sans qu'aucune réponse ait changé — le
    piège exact que le §10.7 a documenté pour les familles. On publie donc les deux, nommées,
    et ``positives`` reste la grandeur comparable à l'avant.
    """
    toutes = list(lignes.values())
    positives = [l for l in toutes if not l["famille"].startswith("negative")]
    agregat = _compter_citations(positives)
    agregat["positives"] = _compter_citations(positives)
    agregat["toutes_familles"] = _compter_citations(toutes)
    agregat["negatives"] = _compter_citations([l for l in toutes
                                               if l["famille"].startswith("negative")])
    agregat["lecture"] = ("les champs de tête portent sur les questions POSITIVES seules — "
                          "c'est la population de la ligne de base du 9 septembre ; "
                          "toutes_familles ajoute les 54 négatives, mesurées pour la "
                          "première fois par ce chantier")
    return agregat


def _agreger_signal(lignes: dict) -> dict:
    """Le signal ``NOT_IN_SOURCES``, par famille, et ce qu'il vaut là où on l'attend.

    Deux lectures opposées, et il faut les deux :

    - sur ``negative_voisine``, le signal est **ce qu'on veut voir** : le sujet est couvert, la
      grandeur est absente, et c'est exactement le cas que v2 rendait indicible ;
    - sur une positive dont l'or est servi, le même signal est une **sur-abstention déguisée** —
      la réponse était là et le système a déclaré une absence. C'est le risque nommé au §19 bis,
      et sans ce compte il passerait pour un progrès.
    """
    par_famille = {}
    for famille in FAMILLES:
        membres = [l for l in lignes.values() if l["famille"] == famille]
        if not membres:
            continue
        servis_ = [l for l in membres if l["or_servi"]]
        par_famille[famille] = {
            "n": len(membres),
            "signale_absence": sum(1 for l in membres if l["signale_absence"]),
            "signale_absence_si_or_servi": sum(1 for l in servis_ if l["signale_absence"]),
            "n_or_servi": len(servis_),
            "et_abstenue": sum(1 for l in membres if l["signale_absence"] and l["abstenue"]),
        }
    return par_famille


def _garde_or_servi(lignes: dict) -> dict:
    """La garde de perte d'or **servi**, reprise de ``garde_reponse.py`` (§11, PLAFOND 0,15).

    Elle ne dit rien ici d'une perte — il n'y a pas de bras candidat —, mais elle publie le
    seuil et la population qui la rendraient lisible au prochain run, celui du chantier 1.
    """
    positives = [l for l in lignes.values() if not l["famille"].startswith("negative")]
    n = sum(1 for l in positives if l["or_servi"])
    return {
        "regle": garde_reponse.garde.__doc__ or "",
        "plafond": garde_reponse.PLAFOND,
        "population_or_servi": n,
        "seuil_echec_b": garde_reponse.seuil_echec(n) if n else None,
        "lecture": ("population des questions v4 positives dont l'or est dans les cinq "
                    "passages servis ; au prochain run, b pertes >= ce seuil font échouer la garde"),
    }


def comparer(courant: dict, reference: dict, drapeau: bool) -> dict:
    """Comparaison appariée, et refus si l'instrument n'est pas le même."""
    ecarts = divergences(courant, reference)
    if ecarts and not drapeau:
        sys.exit(f"REFUS de comparer : les runs diffèrent par {ecarts}. "
                 f"Relancer avec --drapeau pour l'assumer, et l'écart sera écrit dans le JSON.")
    apparie = {}
    detail_ref = reference.get("detail") or {}
    for famille in FAMILLES:
        deltas = [courant["detail"][c]["score"] - detail_ref[c]["score"]
                  for c in courant["detail"]
                  if c in detail_ref and courant["detail"][c]["famille"] == famille]
        if not deltas:
            continue
        nouvelles = [c for c in courant["detail"] if c in detail_ref
                     and courant["detail"][c]["famille"] == famille
                     and courant["detail"][c]["abstenue"] and not detail_ref[c]["abstenue"]]
        disparues = [c for c in courant["detail"] if c in detail_ref
                     and courant["detail"][c]["famille"] == famille
                     and detail_ref[c]["abstenue"] and not courant["detail"][c]["abstenue"]]
        chutes = [c for c in courant["detail"] if c in detail_ref
                  and courant["detail"][c]["famille"] == famille
                  and detail_ref[c].get("couverture_jugee") == 2
                  and courant["detail"][c].get("couverture_jugee") == 0]
        # --- la garde ``a``, ORIENTÉE PAR FAMILLE. Elle ne l'était pas, et elle comptait donc
        # à l'envers sur la moitié de la population.
        #
        # ``a = nouvelles − disparues`` a été repris d'``eval_characters``, qui ne mesurait que
        # des questions **positives**. Sur une positive, s'abstenir d'une question répondable
        # est une dégradation : le sens est bon. Sur une **négative**, l'abstention EST la bonne
        # réponse — c'est même le score de la famille. Une abstention nouvelle y est un gain, et
        # la garde d'origine la comptait comme une faute ; symétriquement, une abstention qui
        # disparaît sur une négative est le mode d'échec le plus grave du produit (le système se
        # met à répondre là où le corpus ne le permet pas) et la garde le comptait comme un
        # progrès. Elle ne se trompait pas d'un peu : elle se trompait de signe.
        #
        # ``a_degradation`` est donc orienté pour que **positif = le produit s'est dégradé**,
        # dans les deux cas. ``a_brut`` reste publié à côté : c'est le chiffre des runs
        # antérieurs, et un chiffre corrigé se publie à côté de l'ancien.
        negative = famille.startswith("negative")
        a_brut = len(nouvelles) - len(disparues)
        a_degradation = -a_brut if negative else a_brut
        apparie[famille] = {
            "n": len(deltas), "delta": round(statistics.fmean(deltas), 4), "ic95": _ic(deltas),
            "montent": sum(1 for d in deltas if d > 0), "descendent": sum(1 for d in deltas if d < 0),
            "test_exact": _test_exact(deltas),
            "bootstrap_degenere": _ic(deltas) == [0.0, 0.0] or len(deltas) <= 20,
            "abstentions_nouvelles": len(nouvelles), "abstentions_disparues": len(disparues),
            "a_brut": a_brut,
            "a": a_degradation,
            "orientation_de_a": ("négative : une abstention NOUVELLE est un gain, "
                                 "a = disparues − nouvelles"
                                 if negative else
                                 "positive : une abstention nouvelle est une dégradation, "
                                 "a = nouvelles − disparues"),
            "b": len(chutes), "garde_degradation": len(chutes) <= GARDE_DEGRADATION,
            # Les deux gardes restent CALCULÉES mais ne concluent plus seules : leurs seuils
            # (a <= 2, b <= 6) ont été fixés sur un générateur supposé déterministe, et le bras
            # placebo du lot 6 a rendu a = +3 à +10 **à variable nulle**. Un seuil sous le
            # plancher de bruit de l'instrument qu'il surveille n'est pas un seuil. La
            # conclusion est donc rendue plus bas, contre le plancher du run, par
            # ``_conclure_contre_le_plancher``.
            "garde_abstention_seuil_pre_enregistre": a_degradation <= GARDE_ABSTENTION,
            "garde_abstention": None,
            "note_sur_les_gardes": ("seuils pré-enregistrés conservés pour mémoire ; sans "
                                    "plancher de bruit du même run ils ne concluent pas"),
        }
        # Le contraste **des fabrications**, apparié, et il ne se déduit pas du précédent.
        # Sur une famille négative, ``score`` est l'abstention par jeton — pas la fabrication.
        # Une réponse peut cesser d'abstenir ET cesser de fabriquer (elle répond sur le sujet et
        # signale la grandeur absente : le comportement que le contrat v3 demande), auquel cas
        # le Δ des scores est négatif et le Δ des fabrications aussi. Lire le premier pour le
        # second aurait conclu à une régression là où le produit fait exactement ce qu'on lui
        # demande. C'est le critère principal du chantier `contrat-de-reponse` ; il est calculé
        # ici pour toutes les familles qui le portent, et il n'existait pas.
        fabrications = [int(bool(courant["detail"][c].get("fabriquee")))
                        - int(bool(detail_ref[c].get("fabriquee")))
                        for c in courant["detail"]
                        if c in detail_ref and courant["detail"][c]["famille"] == famille
                        and ("fabriquee" in courant["detail"][c] or "fabriquee" in detail_ref[c])]
        if fabrications:
            apparie[famille]["fabrications"] = {
                "n": len(fabrications),
                "delta": round(statistics.fmean(fabrications), 4),
                "ic95": _ic(fabrications),
                "test_exact": _test_exact(fabrications),
                "apparues": sum(1 for d in fabrications if d > 0),
                "disparues": sum(1 for d in fabrications if d < 0),
            }
    return {"reference": reference.get("date"), "divergences_d_en_tete": ecarts,
            "drapeau_leve": bool(ecarts and drapeau), "par_famille": apparie}


# --------------------------------------------------------------------------- le plancher

#: La règle du lot 6, §1 quater, devenue du code. Elle vaut dans les DEUX sens, et c'est tout
#: son intérêt : un gain dont l'IC recouvre celui du placebo n'est pas un gain, et une perte
#: dont l'IC recouvre celui du placebo n'est pas une perte. Publier la seconde moitié seulement
#: serait un instrument qui ne sait détecter que les mauvaises nouvelles.
REGLE_DU_PLANCHER = ("tout Δ dont l'IC 95 % recouvre celui du placebo du même run est NON "
                     "CONCLUANT — gains comme pertes")


def plancher_requis(bras: Bras, manifeste: dict, plancher: Path | None,
                    sans_plancher: bool) -> bool:
    """Ce bras doit-il présenter un plancher de bruit pour que son Δ soit publiable ?

    Oui, sauf dans deux cas, et ils sont les seuls :

    - l'opérateur a **fourni** un plancher, ou a **explicitement** demandé à s'en passer
      (``--sans-plancher``, dont l'aveu est écrit dans le JSON) ;
    - ce bras **est** le placebo du run. Son Δ contre la référence *est* le plancher ; lui
      demander le sien serait une régression à l'infini.
    """
    if plancher is not None or sans_plancher:
        return False
    return not (bras.role == "placebo" or manifeste.get("placebo") == bras.nom)


def _conclure_contre_le_plancher(comparaison: dict, plancher: dict | None) -> dict:
    """Applique la règle du plancher, famille par famille, et rend la comparaison enrichie.

    ``plancher`` est la comparaison **du bras placebo contre la même référence**. Son Δ ne peut
    contenir que du bruit : même corpus, mêmes passages, même contrat, même fenêtre, mêmes
    modèles, même run — la variable y est nulle par construction, et ``analyser_bras`` refuse
    qu'elle ne le soit pas.

    Les gardes ``a`` et ``b`` sont conclues de la même façon, et pour la même raison : leur
    seuil pré-enregistré (``a <= 2``) est passé sous le plancher de bruit le 9 septembre 2026
    (placebo à ``a = +3`` à ``+10`` à variable nulle). Le seuil retenu est donc le **plus
    exigeant des deux qui soient tenables** : le pré-enregistré, ou celui que le bruit impose.
    """
    if plancher is None:
        for bloc in comparaison["par_famille"].values():
            bloc["conclusion"] = "SANS_PLANCHER"
            bloc["plancher"] = None
        comparaison["plancher"] = {"present": False, "regle": REGLE_DU_PLANCHER,
                                   "consequence": "aucun Δ de ce fichier n'est concluant"}
        return comparaison

    par_famille_plancher = plancher.get("par_famille") or {}
    for famille, bloc in comparaison["par_famille"].items():
        sol = par_famille_plancher.get(famille)
        if not sol:
            bloc["conclusion"] = "SANS_PLANCHER"
            bloc["plancher"] = None
            continue
        recouvre = _recouvrent(bloc["ic95"], sol["ic95"])
        bloc["plancher"] = {"delta": sol["delta"], "ic95": sol["ic95"], "a": sol["a"],
                            "b": sol["b"]}
        bloc["conclusion"] = "NON_CONCLUANT" if recouvre else "CONCLUANT"
        bloc["ic_recouvre_le_plancher"] = recouvre
        # Les gardes, conclues contre le bruit et non contre un seuil d'hier.
        seuil_a = max(GARDE_ABSTENTION, sol["a"])
        seuil_b = max(GARDE_DEGRADATION, sol["b"])
        bloc["garde_abstention"] = bloc["a"] <= seuil_a
        bloc["garde_abstention_seuil_retenu"] = seuil_a
        bloc["garde_degradation"] = bloc["b"] <= seuil_b
        bloc["garde_degradation_seuil_retenu"] = seuil_b
        bloc["seuil_pre_enregistre_sous_le_bruit"] = bool(sol["a"] > GARDE_ABSTENTION
                                                          or sol["b"] > GARDE_DEGRADATION)
        fab, fab_sol = bloc.get("fabrications"), sol.get("fabrications")
        if fab and fab_sol:
            fab["plancher"] = {"delta": fab_sol["delta"], "ic95": fab_sol["ic95"]}
            fab_recouvre = _recouvrent(fab["ic95"], fab_sol["ic95"])
            fab["ic_recouvre_le_plancher"] = fab_recouvre
            fab["conclusion"] = "NON_CONCLUANT" if fab_recouvre else "CONCLUANT"
    comparaison["plancher"] = {"present": True, "regle": REGLE_DU_PLANCHER,
                               "bras": plancher.get("bras"), "run_id": plancher.get("run_id")}
    return comparaison


# --------------------------------------------------------------------------- affichage

def imprimer(resultat: dict) -> None:
    temoins_lus = resultat["temoins"]
    print(f"\nsignature {resultat['signature']} · {resultat['modeles']['generateur']} / "
          f"{resultat['modeles']['juge']} · auteur {resultat['modeles']['auteur']}")
    print(f"fenêtre servie {resultat['fenetre_servie']} · fenêtre du JUGE "
          f"{resultat['fenetre_du_juge']}"
          + ("  (identiques)" if resultat['fenetre_servie'] == resultat['fenetre_du_juge']
             else "  ← DEUX FENÊTRES DIFFÉRENTES, à déclarer dans toute comparaison"))
    print(f"items-témoins : exactitude {temoins_lus.get('accuracy')} · {FAMILLE_BLOQUANTE} "
          f"{(temoins_lus.get('per_family') or {}).get(FAMILLE_BLOQUANTE, {}).get('accuracy')}"
          f" -> {'LISIBLE' if temoins_lus.get('lisible') else 'ILLISIBLE'}")
    print(f"\n{'famille':18s} {'n':>4s} {'score':>7s} {'zéros':>6s} {'or servi':>9s} "
          f"{'si or servi':>12s} {'abst.':>6s} {'couv. jugée':>12s}")
    for famille, bloc in resultat["par_famille"].items():
        si_servi = ("—" if bloc["score_si_or_servi"] is None
                    else f"{bloc['score_si_or_servi']:.4f}")
        couverture = "—" if bloc["couverture_jugee"] is None else f"{bloc['couverture_jugee']:.4f}"
        print(f"{famille:18s} {bloc['n']:4d} {bloc['score']:7.4f} {bloc['zeros']:6d} "
              f"{bloc['or_servi']:9d} {si_servi:>12s} {bloc['abstentions']:6d} {couverture:>12s}")
    formule = resultat["par_famille"].get("formula")
    if formule:
        print(f"\nformula — strict {formule['juste_strict']}/{formule['n']} · recouvrement moyen "
              f"{formule['recouvrement_moyen']} · balisée {formule['balisee']}/{formule['n']} · "
              f"leurre inclus {formule['leurre_inclus']}")
    citations = resultat["citations"]
    for nom in ("positives", "toutes_familles"):
        bloc = citations.get(nom) or {}
        if not bloc.get("affirmations"):
            continue
        print(f"\ncitations [{nom}, {bloc['questions']} questions] — {bloc['affirmations']} "
              f"affirmations vérifiables · précision {bloc['precision_des_citations']} · "
              f"sans citation {bloc['part_sans_citation']} · "
              f"mauvais passage {bloc['part_mauvais_passage']}")
        print(f"  comptes : {bloc['comptes']}")
        print(f"  extraits verbatim : {bloc['extraits_verbatim']} · retrouvés "
              f"{bloc['extraits_retrouves']} · part {bloc['part_extraits_retrouves']}")
    signal = resultat.get("signal_absence") or {}
    if any(bloc["signale_absence"] for bloc in signal.values()):
        print(f"\nsignal NOT_IN_SOURCES — {'famille':18s} {'n':>4s} {'signalé':>8s} "
              f"{'dont or servi':>14s}")
        for famille, bloc in signal.items():
            print(f"{'':22s} {famille:18s} {bloc['n']:4d} {bloc['signale_absence']:8d} "
                  f"{bloc['signale_absence_si_or_servi']:14d}")
    variance = resultat.get("variance_intra_question")
    if variance:
        print(f"\nvariance intra-question — {variance['tirages']} tirages sur "
              f"{variance['questions']} questions · instables "
              f"{variance['questions_instables']} ({variance['part_instable']:.1%})")
        for famille, bloc in variance["par_famille"].items():
            print(f"  {famille:18s} n={bloc['n']:4d}  instables {bloc['instables']:4d}  "
                  f"{bloc['part_instable']:.1%}")
        print("  (juge non rejoué : score déterministe et abstention seulement)")
    garde = resultat["garde_or_servi"]
    print(f"\ngarde d'or servi — population {garde['population_or_servi']}, "
          f"seuil d'échec b >= {garde['seuil_echec_b']} (plafond {garde['plafond']})")
    latence = resultat.get("latence") or {}
    for etage, bloc in latence.items():
        if bloc:
            print(f"latence {etage:14s} n={bloc['n']:4d}  p50 {bloc['p50']:6.2f} s  p95 {bloc['p95']:6.2f} s")
    cout = resultat.get("cout") or {}
    print(f"\ncoût du run ENTIER : {cout.get('total_usd')} USD "
          f"(compté {cout.get('total_eur_majore')} EUR) — {cout.get('source', 'registre')}")
    for modele, ligne in (cout.get("par_modele") or {}).items():
        print(f"  {modele:24s} {ligne['appels']:4d} appels · "
              f"{ligne['entree']:>9,d} jetons in · {ligne['sortie']:>7,d} out · "
              f"{ligne['cout_usd']:.4f} USD")
    if "appels_manquants" in cout:
        manquants = cout["appels_manquants"]
        print(f"  appels non retrouvés dans le cache : {manquants}"
              f"{'' if manquants == 0 else '  <- le coût est SOUS-ESTIMÉ'}")
        if "marginal_de_ce_processus_usd" in cout:
            print(f"  dont payé par CE processus : "
                  f"{cout['marginal_de_ce_processus_usd']} USD · "
                  f"déjà en cache : {cout['deja_en_cache_usd']} USD")
    comparaison = resultat.get("comparaison")
    if comparaison:
        print(f"\ncomparaison contre {comparaison['reference']} — divergences d'en-tête : "
              f"{comparaison['divergences_d_en_tete'] or 'aucune'}")
        sol = comparaison.get("plancher") or {}
        if sol.get("present"):
            print(f"plancher de bruit : bras « {sol.get('bras')} », run {sol.get('run_id')}")
        else:
            print(f"plancher de bruit : ABSENT — {sol.get('role') or sol.get('consequence', '')}")
        print(f"règle : {sol.get('regle', REGLE_DU_PLANCHER)}")

        def _borne(intervalle):
            return (f"[{intervalle[0]:+.3f} ; {intervalle[1]:+.3f}]" if intervalle else "—")

        for famille, bloc in comparaison["par_famille"].items():
            marque = {"CONCLUANT": "CONCLUANT", "NON_CONCLUANT": "non concluant (dans le bruit)",
                      "SANS_PLANCHER": "non concluant (sans plancher)",
                      "PLANCHER": "= le plancher"}.get(bloc.get("conclusion"), "")
            print(f"\n  {famille:18s} n={bloc['n']:3d} Δ {bloc['delta']:+.4f} "
                  f"{_borne(bloc['ic95']):>20s}   {marque}")
            if bloc.get("plancher"):
                print(f"  {'':18s} plancher   Δ {bloc['plancher']['delta']:+.4f} "
                      f"{_borne(bloc['plancher']['ic95']):>20s}")
            exact = bloc.get("test_exact")
            if exact:
                print(f"  {'':18s} exact      p={exact['p']:.4f} sur "
                      f"{exact['paires_discordantes']} paires discordantes"
                      f"{'   <- bootstrap dégénéré, lire l exact' if bloc.get('bootstrap_degenere') else ''}")
            garde_a = bloc.get("garde_abstention")
            seuil_a = bloc.get("garde_abstention_seuil_retenu")
            etat_a = "—" if garde_a is None else ("OK" if garde_a else "ÉCHEC")
            print(f"  {'':18s} a={bloc['a']:+d} (brut {bloc['a_brut']:+d}) {etat_a}"
                  f"{f' seuil retenu {seuil_a}' if seuil_a is not None else ''} · "
                  f"b={bloc['b']} {'OK' if bloc['garde_degradation'] else 'ÉCHEC'}")
            print(f"  {'':18s} {bloc['orientation_de_a']}")
            fab = bloc.get("fabrications")
            if fab:
                exclut = fab["ic95"] and (fab["ic95"][1] < 0 or fab["ic95"][0] > 0)
                marque_f = {"CONCLUANT": "CONCLUANT",
                            "NON_CONCLUANT": "non concluant (dans le bruit)"}.get(
                    fab.get("conclusion"), "sans plancher")
                print(f"  {'':18s} fabrications Δ {fab['delta']:+.4f} "
                      f"{_borne(fab['ic95']):>20s} (−{fab['disparues']} / +{fab['apparues']})"
                      f"{'  IC exclut zéro' if exclut else ''}   {marque_f}")
                if fab.get("plancher"):
                    print(f"  {'':18s} fab. plancher Δ {fab['plancher']['delta']:+.4f} "
                          f"{_borne(fab['plancher']['ic95']):>20s}")


def main() -> None:
    analyse = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    analyse.add_argument("--etape", choices=("temoins", "contextes", "mesure", "verdict"),
                         required=True)
    analyse.add_argument("--fenetre-servie", type=int, default=FENETRE_SERVIE)
    analyse.add_argument("--fenetre-juge", type=int, default=FENETRE_JUGE)
    analyse.add_argument("--reference", type=Path, default=None,
                         help="fichier results-v4-*.json auquel comparer (bootstrap apparié)")
    analyse.add_argument("--drapeau", action="store_true",
                         help="assumer une comparaison entre runs d'instruments différents")
    analyse.add_argument("--limite", type=int, default=None, help="sous-échantillon, pour un rodage")
    analyse.add_argument("--prompt", default=pipeline.DEFAULT_PROMPT,
                         choices=sorted(pipeline.ANSWER_PROMPTS),
                         help="version du contrat de réponse — la variable du chantier "
                              "contrat-de-reponse. Elle nomme les fichiers d'état ET entre "
                              "dans les champs gouvernants : deux contrats ne se comparent "
                              "pas sans --drapeau. Raccourci pour --bras <PROMPT>.")
    analyse.add_argument("--bras", metavar="SPEC",
                         help="les bras du run, tirés DANS LE MÊME RUN et entrelacés par "
                              "question : `nom[:prompt=P][:fenetre=N][:placebo],...`. Le "
                              "premier est la référence ; le bras `:placebo` en hérite le "
                              "contrat et la fenêtre, tire dans un cache d'appels neuf et "
                              "publie le plancher de bruit. Exemple : "
                              "`a1:fenetre=2500,a2:fenetre=10000,plancher:placebo`.")
    analyse.add_argument("--tirages", type=int, default=1, metavar="K",
                         help="K générations par question et par bras (défaut 1). Les tirages "
                              "2..K passent par un cache neuf — sinon ils rendraient la même "
                              "réponse — et publient la variance intra-question. Le juge n'est "
                              "PAS rejoué. Coût du générateur multiplié par K.")
    analyse.add_argument("--plancher", type=Path, default=None, metavar="FICHIER",
                         help="results-v4-*.json du bras PLACEBO, comparé à la MÊME référence. "
                              "Obligatoire avec --reference : sans lui aucun Δ n'est publiable.")
    analyse.add_argument("--plancher-cache", type=Path, default=None, metavar="DIR",
                         help="REPRENDRE un run interrompu : rend au placebo le répertoire de "
                              "cache d'appels d'un run tué, pour qu'il reprenne sur SES propres "
                              "tirages (la seule reprise correcte) au lieu d'en payer de "
                              "nouveaux. Son état n'est alors pas remis à zéro, et le "
                              "répertoire n'est pas effacé à la fin. Sans l'option : cache "
                              "neuf, état neuf — on ne peut pas y tomber par accident.")
    analyse.add_argument("--sans-plancher", action="store_true",
                         help="publier un Δ SANS plancher de bruit, en l'assumant. L'aveu est "
                              "écrit dans le JSON et toutes les familles sont marquées "
                              "SANS_PLANCHER, donc non concluantes.")
    analyse.add_argument("--gold-signature", metavar="SIG",
                         help="joue formula/table_cell/negative_voisine sur les questions "
                              "ré-orées questions-v4-<famille>-SIG.jsonl (gold_ancrage.py "
                              "--appliquer-v4). negative_v3 n'est pas concernée. Sans "
                              "l'option, comportement inchangé.")
    analyse.add_argument("--candidat", metavar="SIG",
                         help="mesure le corpus candidat SIG : nomme le run par sa signature, "
                              "injecte sa vue corpus, et implique --gold-signature SIG. "
                              "QUANT_RAG_COLLECTION doit désigner sa collection.")
    arguments = analyse.parse_args()

    global GOLD_SIGNATURE, SIGNATURE, INDEX
    GOLD_SIGNATURE = arguments.gold_signature or arguments.candidat
    if arguments.candidat:
        sys.path.insert(0, str(HERE.parent / "ingestion"))
        import quant_rag
        from build_collection_candidat import nom_collection
        from eval_dense_candidat import index_candidat

        attendue = nom_collection(arguments.candidat, "c2")
        if quant_rag.COLLECTION != attendue:
            sys.exit(f"--candidat {arguments.candidat} : la collection interrogée est "
                     f"{quant_rag.COLLECTION}, pas {attendue}. Exporter "
                     f"QUANT_RAG_COLLECTION={attendue}, sinon le classement viendrait du "
                     f"corpus servi et le contexte du candidat.")
        SIGNATURE = arguments.candidat
        INDEX = index_candidat(arguments.candidat)
    if GOLD_SIGNATURE:
        manquants = [str(_fichier_famille(nom, FAMILLES[nom])) for nom in FAMILLES_V4
                    if not _fichier_famille(nom, FAMILLES[nom]).exists()]
        if manquants:
            sys.exit(f"--gold-signature {GOLD_SIGNATURE} : fichier(s) introuvable(s) — lancer "
                     f"gold_ancrage.py --appliquer-v4 d'abord : {', '.join(manquants)}")

    # ``--bras`` et ``--prompt`` disent la même chose à deux échelles. Sans ``--bras``, un seul
    # bras portant le nom du contrat : les chemins d'état et les noms de fichiers sont alors
    # ceux d'avant cette option, à l'octet près, et les artefacts du 9 septembre restent lus.
    bras = (analyser_bras(arguments.bras, arguments.fenetre_servie) if arguments.bras
            else [Bras(nom=arguments.prompt, prompt=arguments.prompt,
                       fenetre_servie=arguments.fenetre_servie, role="reference")])
    if arguments.tirages < 1:
        sys.exit("--tirages doit valoir au moins 1")
    if arguments.etape == "verdict" and len(bras) > 1:
        sys.exit(f"--etape verdict porte sur UN bras à la fois (reçu {len(bras)}). "
                 f"Le rendre pour chacun, puis passer le placebo en --plancher.")

    if arguments.etape == "temoins":
        temoins()
    elif arguments.etape == "contextes":
        contextes_seuls(bras[0], arguments.limite)
    elif arguments.etape == "mesure":
        if arguments.plancher_cache and not arguments.plancher_cache.is_dir():
            sys.exit(f"--plancher-cache {arguments.plancher_cache} : répertoire introuvable. "
                     f"Sans lui le placebo repaierait ses appels ; mieux vaut s'arrêter que "
                     f"dépenser en croyant reprendre.")
        mesure(bras, arguments.fenetre_juge, arguments.limite, arguments.tirages,
               arguments.plancher_cache)
    else:
        imprimer(verdict(bras[0], arguments.fenetre_juge, arguments.reference,
                         arguments.drapeau, arguments.plancher, arguments.sans_plancher))


if __name__ == "__main__":
    main()
