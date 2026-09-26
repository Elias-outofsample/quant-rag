"""Validation mécanique d'un lot ``human-v1`` — le schéma, et surtout les ancres.

Pourquoi ce module existe
-------------------------
``human-v1`` coûte du temps humain, la ressource la plus chère du dossier. Une erreur
d'annotation découverte après coup se paie deux fois : une fois pour l'annotation
perdue, une fois pour l'annotation refaite. Ce validateur attrape avant le gel tout ce
qui est attrapable **par la machine**, pour que la relecture humaine se concentre sur ce
qui ne l'est pas — la pertinence de la question, la justesse du fait.

Ce qu'il vérifie, et pourquoi chaque contrôle existe
-----------------------------------------------------

1. **schéma** — champs obligatoires, valeurs dans les domaines autorisés, cohérence des
   statuts. Un item ``repondable: non`` qui porte des ancres est une contradiction, pas
   une variante.

2. **ancre retrouvée** — le texte de l'ancre doit exister dans le document annoncé, à une
   couverture ``>= 0,80`` (le seuil pré-enregistré de ``gold_ancrage``). Une ancre
   recopiée à la main peut avoir perdu un mot ou gagné une correction de coquille : le
   corpus porte « diferent », « afects », « suficient », et un annotateur consciencieux
   les corrige sans y penser. C'est le défaut le plus probable, et il est silencieux.

3. **unicité** — la marge entre le meilleur candidat et le second. Mesuré (sonde du
   6 septembre 2026, 60 documents, 1 255 ancres de 15 à 50 mots) : marge médiane 0,96,
   mais **5,9 % des ancres ont une marge < 0,20**, et la queue est concentrée sur les
   tableaux (18,7 %) et les formules. Dans ces cas le texte est présent *deux fois* dans
   le document : l'ancrage retrouve bien le passage, mais pas **lequel**. L'alerte n'est
   donc pas un rejet — c'est une demande de désambiguïsant.

4. **blocs** — ``chunk.metadata.block_ids`` vient du parse (``mineru_adapter._id`` sur le
   document, l'index, la page, le type et le texte du bloc), **pas du découpage**. Mesuré :
   100 % de couverture, 156 602 identifiants, aucun partagé entre deux chunks ; et sur les
   ancres réellement ambiguës par le texte, les blocs en tranchent **27 sur 37 (73 %)**,
   dont 24 des 34 cas de tableau. Le validateur les renseigne automatiquement : c'est
   gratuit à la capture et irrattrapable après.

5. **longueur** — mesuré : une ancre de 20 mots ou plus survit à 100 % au re-découpage
   simulé (cibles 900 à 2 400 caractères), contre 99,8 % à 10 mots. Sous 15 mots, refus ;
   entre 15 et 20, alerte.

Ce qu'il ne vérifie pas, et ne peut pas vérifier
-------------------------------------------------
Que la question soit intéressante. Que le fait attendu soit **vrai**. Que l'ancre
*soutienne* réellement le fait — un texte peut être présent, unique, bien ancré, et ne
pas dire ce que l'annotateur lui fait dire. Ces trois-là sont le travail humain, et
aucun compte de n-grammes ne les remplacera.

Usage
-----
    .venv/bin/python rag/benchmark/human-v1/valider.py items-pilote-v0.jsonl
    .venv/bin/python rag/benchmark/human-v1/valider.py items-pilote-v0.jsonl --enrichir

``--enrichir`` réécrit le fichier en y ajoutant, pour chaque ancre, les ``blocs``, la
``page`` et la ``marge`` mesurés — les champs que l'annotateur n'a pas à saisir.

Lecture seule sur le corpus. Aucun appel LLM. Aucun accès à Qdrant (le serveur MCP tient
le verrou : tout passe par ``rows.jsonl``).
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

# ---------------------------------------------------------------- seuils, pré-enregistrés

#: Couverture minimale pour qu'une ancre soit considérée retrouvée. Même valeur que
#: ``gold_ancrage.SEUIL_ANCRE`` — délibérément : deux instruments qui décident sur la
#: même grandeur doivent décider au même endroit.
SEUIL_COUVERTURE = 0.80

#: Sous cette marge, l'ancre est ambiguë : un autre passage du même document la couvre
#: presque aussi bien. Alerte, pas rejet — le désambiguïsant par blocs existe.
SEUIL_MARGE = 0.20

#: Plancher dur de longueur d'ancre. Sous 15 mots la survie au re-découpage décroche
#: (99,8 % contre 100 % au-dessus de 20) et la marge d'unicité se dégrade.
MOTS_MINIMUM = 15

#: En dessous, l'ancre passe mais l'annotateur est prévenu.
MOTS_CONFORTABLES = 20

FAMILLES = {
    "factuelle_localisee", "controverse_chiffree", "condition_de_validite",
    "protocole_chiffre", "formule_et_mesure", "comparaison_multi_documents",
    "partiellement_couvert", "hors_corpus", "tableau", "definition_contrastee",
}
DOMAINES = {
    "microstructure", "execution", "market_making", "volatilite", "options", "risque",
    "backtest", "crypto_defi", "arbitrage", "calibration", "methodologie",
}
REPONDABLE = {"oui", "partiel", "non"}
STATUTS = {"idee", "candidat", "pilote", "ai_reviewed", "valide"}

#: Mode de revue d'une campagne. La frontière que porte ce champ est la seule qui compte :
#: un lot validé par des modèles ne doit jamais pouvoir se faire passer pour un lot validé
#: par une personne. Panne évitée : une campagne AI-reviewed citée comme « évaluée par un
#: expert », ce qui transformerait une mesure utile en revendication fausse.
REVIEW_MODES = {"humain", "multi_agent_ai"}
ROLES = {"porte_le_fait", "pose_la_condition", "conteste", "corrobore", "contexte"}
EXIGENCES = {"obligatoire", "souhaitable"}
CONSEQUENCES = {"reponse_fausse", "reponse_incomplete"}

#: Statut d'un **fait attendu**, distinct du statut de l'item.
#:
#: ``gold``            compte dans le score principal.
#: ``gold_candidate``  ne compte dans **aucun** score. En attente de revue source.
#: ``gold_retire``     a compté, puis a été jugé faux. Conservé avec son motif — un fait
#:                     ne disparaît jamais du fichier, sinon l'historique des scores
#:                     devient irreproductible.
#: ``gold_ai_reviewed``  validé par revue contradictoire multi-agent. Compte **uniquement**
#:                       dans une campagne ``multi_agent_ai``, jamais dans une campagne
#:                       humaine — et réciproquement. La frontière est symétrique : une
#:                       campagne IA ne s'attribue pas non plus le travail humain.
STATUTS_FAIT = {"gold", "gold_ai_reviewed", "gold_candidate", "gold_retire"}

#: Statut de fait comptant, par mode de revue. C'est cette table, et non un booléen, qui
#: rend la frontière lisible et testable dans les deux sens.
COMPTE_PAR_MODE = {"humain": "gold", "multi_agent_ai": "gold_ai_reviewed"}

#: Nature d'un fait attendu. Panne évitée : une inférence promue comme un énoncé
#: textuellement écrit dans la source — le lecteur croit lire le papier, il lit un
#: raisonnement. Une inférence doit montrer ses prémisses et ses hypothèses.
NATURES = {"explicite", "inference"}

#: Origine d'un fait candidat. ``revue_corpus`` = trouvé par lecture humaine, sans passer
#: par une sortie du système. Tout le reste nomme la configuration qui l'a fait apparaître,
#: et c'est cette traçabilité qui permet d'écarter le fait d'une comparaison où cette
#: configuration est juge et partie (voir ``gold_a_la_version``).
ORIGINES = {"revue_corpus", "execution"}

#: Marqueurs de nom de modèle. Trois décisions du contrat — promouvoir un candidat, retirer
#: un fait, déclarer un item ``valide`` — ne peuvent pas être signées par un modèle.
#:
#: **Ce que ce filtre est, et ce qu'il n'est pas.** Aucun champ ne peut prouver qu'un humain
#: a écrit une ligne : un modèle sait taper un prénom. Ce filtre attrape le **glissement
#: mécanique**, celui où la signature est littéralement le modèle qui a produit le lot — et
#: ce cas s'est produit : `H004` du lot pilote portait un fait apparu dans une sortie du
#: système, promu d'emblée en gold. La preuve de la relecture humaine n'est pas ici, elle
#: est dans la fiche signée du paquet de revue. Ce filtre est un garde-fou, pas une preuve,
#: et son mode d'échec est **permissif** : un modèle au nom absent de cette liste passerait.
MODELES = re.compile(r"claude|gpt|gemini|mistral|llama|qwen|opus|sonnet|haiku|deepseek|grok",
                     re.IGNORECASE)

#: Date ISO. Une promotion sans date ne peut pas être située par rapport à une campagne.
_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

_BALISE = re.compile(r"</?[a-zA-Z][^>]{0,40}>")
K_GRAMME = 5


def signature_humaine(nom) -> bool:
    """Le nom donné peut-il être celui d'une personne ? Voir ``MODELES``."""
    return bool(nom) and isinstance(nom, str) and not MODELES.search(nom)


# ---------------------------------------------------------------- corpus


def racine_corpus(explicite: str | None) -> Path:
    """Racine du dépôt qui porte ``data/``.

    Le worktree de conception est en *sparse checkout* sans ``data/`` (1,9 Go de LFS pour
    des documents Markdown : la duplication ne se justifiait pas). On retombe donc sur le
    dépôt principal, que git sait localiser.
    """
    if explicite:
        return Path(explicite)
    ici = Path.cwd()
    for base in (ici, HERE.parents[2]):
        if (base / "data" / "embeddings").exists():
            return base
    try:
        commun = subprocess.run(["git", "rev-parse", "--path-format=absolute",
                                 "--git-common-dir"], capture_output=True, text=True,
                                check=True, cwd=HERE).stdout.strip()
        principal = Path(commun).parent
        if (principal / "data" / "embeddings").exists():
            return principal
    except (subprocess.CalledProcessError, OSError):
        pass
    sys.exit("corpus introuvable : passer --racine-corpus <chemin du dépôt portant data/>")


def charger_corpus(racine: Path) -> dict[str, list[dict]]:
    chemins = [racine / "data" / "embeddings" / "ingested-all-qwen3-06b" / "rows.jsonl",
               racine / "rag" / "ingestion" / "imported-rows.jsonl"]
    par_document: dict[str, list[dict]] = {}
    for chemin in chemins:
        if not chemin.exists():
            continue
        for ligne in chemin.open(encoding="utf-8"):
            if not ligne.strip():
                continue
            c = json.loads(ligne)["chunk"]
            if not c.get("rag_eligible", True):
                continue
            par_document.setdefault(c["document_id"], []).append({
                "chunk_id": c["chunk_id"],
                "text": c.get("text") or "",
                "page_start": c.get("page_start"),
                "page_end": c.get("page_end"),
                "content_type": c.get("content_type") or "",
                "blocs": list((c.get("metadata") or {}).get("block_ids") or ()),
            })
    if not par_document:
        sys.exit(f"aucun chunk lu sous {racine} — vérifier --racine-corpus")
    return par_document


# ---------------------------------------------------------------- ancrage
# Même normalisation et même représentation que ``gold_ancrage`` : deux instruments qui
# décident sur la même grandeur doivent la calculer de la même façon.


def normaliser(texte: str) -> str:
    sans_balise = _BALISE.sub(" ", texte or "")
    plie = re.sub(r"[^\w\s]", " ", sans_balise.casefold(), flags=re.UNICODE)
    return re.sub(r"\s+", " ", plie).strip()


def grammes(texte: str, k: int = K_GRAMME) -> set[str]:
    mots = normaliser(texte).split()
    if not mots:
        return set()
    if len(mots) <= k:
        return {" ".join(mots)}
    return {" ".join(mots[i:i + k]) for i in range(len(mots) - k + 1)}


def couverture(ancre: set[str], candidat: set[str]) -> float:
    return len(ancre & candidat) / len(ancre) if ancre else 0.0


def situer(texte: str, candidats: list[dict]) -> dict:
    """Où ce texte se trouve-t-il dans le document, et à quel point est-ce univoque ?"""
    k = min(K_GRAMME, len(normaliser(texte).split()) or K_GRAMME)
    cible = grammes(texte, k)
    if not cible or not candidats:
        return {"couverture": 0.0, "marge": 0.0, "chunk": None, "blocs": [],
                "page": None, "content_type": None, "ex_aequo": 0}
    scores = sorted(((couverture(cible, grammes(c["text"], k)), c["chunk_id"], c)
                     for c in candidats), key=lambda t: (-t[0], t[1]))
    meilleure, _, chunk = scores[0]
    seconde = scores[1][0] if len(scores) > 1 else 0.0
    ex_aequo = sum(1 for s, _, _ in scores if s >= meilleure - 1e-9)
    return {"couverture": round(meilleure, 4), "marge": round(meilleure - seconde, 4),
            "chunk": chunk["chunk_id"], "blocs": chunk["blocs"],
            "page": chunk["page_start"], "content_type": chunk["content_type"],
            "ex_aequo": ex_aequo}


# ---------------------------------------------------------------- gold versionné


def valider_fait_versionne(ident: str, numero: int, fait: dict, version_lot: int | None) -> list[str]:
    """Contrôles de statut et de version d'un fait attendu.

    Le principe : un score n'est interprétable que si l'on sait **contre quel gold** il a
    été calculé, et si ce gold est reconstructible à n'importe quelle version antérieure.
    D'où une version portée **par fait** (``introduit_en``) et non seulement par lot :
    ``gold_a_la_version`` peut alors rejouer « le gold tel qu'il était en v1 » depuis le
    fichier courant, sans conserver de copies.
    """
    erreurs = []
    prefixe = f"{ident}: fait {numero}"
    statut = fait.get("statut_fait")
    if statut not in STATUTS_FAIT:
        return [f"{prefixe} — statut_fait invalide ({statut!r}) ; attendu {sorted(STATUTS_FAIT)}"]

    introduit, propose, retire = fait.get("introduit_en"), fait.get("propose_en"), fait.get("retire_en")

    if statut == "gold":
        if not isinstance(introduit, int) or introduit < 1:
            erreurs.append(f"{prefixe} — un fait « gold » doit porter introduit_en")
        if propose is not None and isinstance(introduit, int) and propose > introduit:
            erreurs.append(f"{prefixe} — proposé en v{propose} mais introduit en v{introduit}")
        # Un fait qui a été candidat puis promu doit porter la trace de sa promotion : une
        # version **nouvelle**, un auteur, un motif. Sans cela, la promotion serait un
        # changement de statut indiscernable d'un fait d'origine — c'est-à-dire exactement
        # la voie par laquelle une sortie du système modifie le contrat d'évaluation.
        if isinstance(propose, int):
            if isinstance(introduit, int) and introduit <= propose:
                erreurs.append(f"{prefixe} — promu en v{introduit} alors qu'il était proposé "
                               f"en v{propose} : une promotion crée une version nouvelle")
            if not fait.get("promu_par"):
                erreurs.append(f"{prefixe} — un candidat promu doit nommer l'auteur de la "
                               "promotion (promu_par)")
            elif not signature_humaine(fait["promu_par"]):
                erreurs.append(f"{prefixe} — promu_par={fait['promu_par']!r} est un modèle. "
                               "Une promotion est une décision humaine : sans cela le contrat "
                               "d'évaluation se met à suivre les sorties qu'il doit juger")
            if not _DATE.match(str(fait.get("promu_le") or "")):
                erreurs.append(f"{prefixe} — un candidat promu doit porter promu_le (AAAA-MM-JJ) : "
                               "sans date, la promotion ne peut pas être située par rapport "
                               "à une campagne")
            if not fait.get("motif_promotion"):
                erreurs.append(f"{prefixe} — un candidat promu doit porter motif_promotion — "
                               "*pourquoi* il entre au gold")
            if not fait.get("revue_source"):
                erreurs.append(f"{prefixe} — un candidat promu doit porter revue_source — "
                               "*où c'est écrit*. Le motif et la source ne se remplacent pas")
    elif statut == "gold_candidate":
        if not isinstance(propose, int) or propose < 1:
            erreurs.append(f"{prefixe} — un candidat doit porter propose_en")
        if introduit is not None:
            erreurs.append(f"{prefixe} — un candidat ne peut pas porter introduit_en : "
                           "il ne compte dans aucun score tant qu'il n'est pas promu")
        if fait.get("origine") not in ORIGINES:
            erreurs.append(f"{prefixe} — origine invalide ({fait.get('origine')!r}) ; "
                           f"attendu {sorted(ORIGINES)}")
        if fait.get("origine") == "execution" and not fait.get("decouvert_par"):
            erreurs.append(f"{prefixe} — un candidat issu d'une exécution doit nommer la "
                           "configuration qui l'a fait apparaître (decouvert_par) : sans elle, "
                           "on ne peut pas l'écarter d'une comparaison où elle est juge et partie")
    elif statut == "gold_ai_reviewed":
        # La promotion par revue multi-agent ne peut pas s'appuyer sur une signature humaine
        # — il n'y en a pas. Ce qui la remplace n'est pas un nom, c'est une **trace** : qui a
        # revu, quelle décision a été rendue, et ce qui restait en désaccord. Panne évitée :
        # « validé par revue contradictoire » affirmé sans qu'aucune contradiction n'existe.
        if not isinstance(introduit, int) or introduit < 1:
            erreurs.append(f"{prefixe} — un fait AI-reviewed doit porter introduit_en")
        if isinstance(propose, int) and isinstance(introduit, int) and introduit <= propose:
            erreurs.append(f"{prefixe} — promu en v{introduit} alors qu'il était proposé en "
                           f"v{propose} : une promotion crée une version nouvelle")
        if not _DATE.match(str(fait.get("promu_le") or "")):
            erreurs.append(f"{prefixe} — un fait AI-reviewed doit porter promu_le (AAAA-MM-JJ)")
        for champ, quoi in (("promu_par", "l'adjudicateur"),
                            ("motif_promotion", "*pourquoi* il entre au gold"),
                            ("revue_source", "*où c'est écrit*"),
                            ("adjudication", "le raisonnement de la décision")):
            if not fait.get(champ):
                erreurs.append(f"{prefixe} — un fait AI-reviewed doit porter {champ} — {quoi}")
        agents = fait.get("agents")
        if not isinstance(agents, list) or len(agents) < 2:
            erreurs.append(f"{prefixe} — agents doit lister au moins deux revues identifiables : "
                           "une « revue contradictoire » à un seul agent n'est pas contradictoire")
        if "desaccords" not in fait:
            erreurs.append(f"{prefixe} — desaccords est obligatoire, fût-il vide : une liste "
                           "absente ne se distingue pas d'un désaccord effacé")
        elif not isinstance(fait["desaccords"], list):
            erreurs.append(f"{prefixe} — desaccords doit être une liste")
        if fait.get("nature") not in NATURES:
            erreurs.append(f"{prefixe} — nature invalide ({fait.get('nature')!r}) ; "
                           f"attendu {sorted(NATURES)}")
    elif statut == "gold_retire":
        if not isinstance(introduit, int) or not isinstance(retire, int):
            erreurs.append(f"{prefixe} — un fait retiré doit porter introduit_en et retire_en")
        elif retire <= introduit:
            erreurs.append(f"{prefixe} — retire_en ({retire}) doit suivre introduit_en ({introduit})")
        if not fait.get("motif_retrait"):
            erreurs.append(f"{prefixe} — un fait retiré doit porter son motif. "
                           "Retirer un fait parce qu'il fait baisser un score est interdit")
        # Le motif dit *pourquoi*, `retire_par` dit *qui*. Les deux sont nécessaires et ils
        # ne se remplacent pas : un motif sans auteur est une décision que personne n'a
        # prise, et c'est précisément la forme qu'aurait un retrait de confort.
        if not fait.get("retire_par"):
            erreurs.append(f"{prefixe} — un fait retiré doit nommer l'auteur de la décision "
                           "(retire_par)")
        elif not fait.get("agents") and not signature_humaine(fait["retire_par"]):
            erreurs.append(f"{prefixe} — retire_par={fait['retire_par']!r} est un modèle et le "
                           "retrait ne porte aucune revue (agents) : un retrait est soit signé "
                           "par un humain, soit adjudiqué par une revue contradictoire tracée. "
                           "Sinon c'est la porte par laquelle un fait gênant sortirait du gold")
        # Un retrait change ce qui compte. Le motif dit pourquoi, mais seul une **source**
        # rend le motif vérifiable, et seule l'**adjudication** dit comment la décision a été
        # rendue. Sans les deux, « ce fait est faux » est une assertion.
        for champ, quoi in (("source_correction", "la source qui établit la correction"),
                            ("adjudication", "la décision de retrait et son raisonnement")):
            if not fait.get(champ):
                erreurs.append(f"{prefixe} — un fait retiré doit porter {champ} — {quoi}")

    if version_lot is not None:
        for champ, valeur in (("introduit_en", introduit), ("propose_en", propose), ("retire_en", retire)):
            if isinstance(valeur, int) and valeur > version_lot:
                erreurs.append(f"{prefixe} — {champ}=v{valeur} dépasse la gold_version du lot "
                               f"(v{version_lot})")
    return erreurs


def aptitude_a_noter(item: dict) -> list[str]:
    """Un item peut-il servir de barème sans rendre le score arbitraire ?

    Ce n'est **pas** une validation de schéma : le fichier peut être parfaitement formé et
    l'item rester inutilisable. Deux contradictions internes rendent une note ininterprétable,
    et les deux existaient dans `ai-reviewed-v1` sans qu'aucun contrôle ne les voie :

    **Un.** Un fait marqué ``exigence: obligatoire`` dont la revue porte
    ``verdict_revue: refuse``. L'item exige alors de la réponse un fait que sa propre revue
    déclare faux. Quoi que réponde le système, la note est arbitraire : l'énoncer coûte
    d'être faux, l'omettre coûte la couverture. Trouvé sur huit faits par le rôle D.

    **Deux.** Un item dont **aucun** fait obligatoire n'a survécu à la revue. Sa couverture
    ne mesure plus rien — sauf pour un item d'abstention, qui n'en a jamais eu.
    """
    ident = item.get("id", "?")
    obligatoires = [(n, f) for n, f in enumerate(item.get("faits_attendus") or [], 1)
                    if f.get("exigence") == "obligatoire"]
    empeche = []
    for numero, fait in obligatoires:
        verdict = (fait.get("revue_ai") or {}).get("verdict_revue")
        if verdict == "refuse":
            empeche.append(f"{ident}: fait {numero} est exigé de la réponse alors que sa revue le "
                           "refuse — quoi que le système réponde, la note est arbitraire")
    if obligatoires and not any(f.get("statut_fait") in COMPTE_PAR_MODE.values()
                                for _, f in obligatoires):
        empeche.append(f"{ident}: aucun fait obligatoire n'a été promu — la couverture de cet item "
                       "ne mesure rien")
    return empeche


def gold_a_la_version(item: dict, version: int, exclure_decouvert_par: set[str] | None = None,
                      review_mode: str | None = None) -> list[dict]:
    """Le gold de cet item **tel qu'il était** à la version donnée.

    C'est la fonction qui rend les scores interprétables après une évolution du gold : on
    peut recalculer un score à n'importe quelle version antérieure depuis le fichier
    courant, donc comparer deux exécutions à gold fixé même si le gold a bougé entre elles.

    ``exclure_decouvert_par`` retire les faits qu'une configuration donnée a fait
    apparaître. À utiliser dans toute comparaison A/B : un gold enrichi par les sorties de
    A favorise A, et cette contamination-là ne se voit pas dans le score.
    """
    # `review_mode=None` vaut « humain » : c'est le mode historique, et le défaut doit être
    # le plus restrictif. Une campagne IA doit se déclarer pour que ses faits comptent —
    # jamais l'inverse. Panne évitée : un score annoncé sans mode, calculé avec des faits
    # validés par des modèles.
    attendu = COMPTE_PAR_MODE.get(review_mode or "humain")
    exclure = exclure_decouvert_par or set()
    retenus = []
    for fait in item.get("faits_attendus") or []:
        introduit = fait.get("introduit_en")
        statut = fait.get("statut_fait")
        if statut == "gold_retire":
            # Un fait retiré comptait avant son retrait — mais dans **sa** campagne. Le mode
            # d'origine se lit sur `agents` : seul un fait passé par une revue multi-agent en
            # porte. Sans cela, un fait IA retiré compterait dans la reconstruction d'une
            # campagne humaine, et la frontière fuirait par le passé.
            if (COMPTE_PAR_MODE["multi_agent_ai"] if fait.get("agents")
                    else COMPTE_PAR_MODE["humain"]) != attendu:
                continue
        elif statut != attendu:
            continue
        if not isinstance(introduit, int):
            continue
        if introduit > version:
            continue
        retire = fait.get("retire_en")
        if isinstance(retire, int) and retire <= version:
            continue
        if fait.get("decouvert_par") in exclure:
            continue
        retenus.append(fait)
    return retenus


def valider_lot(items: list[dict]) -> list[str]:
    """Contrôles qui portent sur le lot entier, pas sur un item isolé."""
    erreurs = []
    versions = {i.get("gold_version") for i in items if i.get("gold_version") is not None}
    if len(versions) > 1:
        erreurs.append(f"gold_version incohérente dans le lot : {sorted(versions)} — "
                       "un lot porte un seul contrat d'évaluation")
    identifiants = [i.get("id") for i in items]
    doublons = {i for i in identifiants if identifiants.count(i) > 1}
    if doublons:
        erreurs.append(f"identifiants en double : {sorted(doublons)}")

    # L'invariant temporel, rendu vérifiable : **le gold n'évolue pas pendant une campagne.**
    # Une promotion au milieu de l'annotation biaiserait tout ce qui a été annoté avant elle
    # — les premiers items seraient notés contre un gold plus pauvre que les derniers, et le
    # score du lot mélangerait deux contrats. La forme que cela prend dans le fichier est
    # exactement celle-ci : deux gold_version sous une même version_campagne.
    par_campagne: dict[int, set[int]] = {}
    for item in items:
        campagne = item.get("version_campagne")
        if isinstance(campagne, int):
            par_campagne.setdefault(campagne, set()).add(item.get("gold_version"))
    for campagne, versions_vues in sorted(par_campagne.items()):
        if len(versions_vues) > 1:
            erreurs.append(f"campagne {campagne} : le gold a changé pendant l'annotation "
                           f"(gold_version {sorted(v for v in versions_vues if v is not None)}) — "
                           "une promotion se fait entre deux campagnes, jamais pendant")
    # Et le sens inverse, qui manquait. Panne évitée : rejouer une campagne contre un gold
    # **antérieur** choisi après avoir vu le résultat. Le gold n'a le droit que d'avancer.
    suite = [(c, min(v for v in versions if v is not None))
             for c, versions in sorted(par_campagne.items())
             if any(v is not None for v in versions)]
    for (campagne, version), (suivante, apres) in zip(suite, suite[1:]):
        if apres < version:
            erreurs.append(f"campagne {suivante} notée contre la gold v{apres} alors que la "
                           f"campagne {campagne} l'était contre la v{version} : le gold recule. "
                           "Une version ne peut qu'avancer, sinon on choisit celle qui flatte")
    return erreurs


def resume_gold(items: list[dict]) -> dict:
    """De quoi lire, d'un coup d'œil, ce qui compte et ce qui ne compte pas."""
    compte = {statut: 0 for statut in STATUTS_FAIT}
    origines: dict[str, int] = {}
    for item in items:
        for fait in item.get("faits_attendus") or []:
            statut = fait.get("statut_fait")
            if statut in compte:
                compte[statut] += 1
            if statut == "gold_candidate":
                cle = fait.get("decouvert_par") or fait.get("origine") or "?"
                origines[cle] = origines.get(cle, 0) + 1
    return {"par_statut": compte, "candidats_par_origine": origines}


# ---------------------------------------------------------------- schéma


#: Un chiffre nu dans un fait attendu. Sert à repérer les faits dont la lecture dépend d'un
#: en-tête de colonne, d'une unité ou d'une ligne de légende.
_CHIFFRE = re.compile(r"\d")

#: Ce qui, dans un extrait ancré, porte l'appariement colonne ↔ grandeur. Le texte canonique
#: du corpus est du HTML ; la forme servie est sa conversion markdown. Les deux sont acceptées.
_ENTETE_TABLEAU = (re.compile(r"<table[\s>]"), re.compile(r"^\s*\|[^\n]*\|\s*\n\s*\|\s*-{2,}", re.M))


def porte_l_entete_de_tableau(texte: str) -> bool:
    """L'extrait contient-il la première ligne du tableau, celle qui nomme les colonnes ?"""
    return any(motif.search(texte or "") for motif in _ENTETE_TABLEAU)


def valider_lecture_des_tableaux(item: dict) -> list[str]:
    """Un chiffre ne devient pas gold si l'ancre ne dit pas de quelle colonne il vient.

    Défaut générique, mesuré le 7 septembre 2026 sur `C05/A1` : l'ancre couvrait les trois
    lignes de données du tableau de Fama-French et **s'arrêtait juste après la ligne
    d'en-tête** — 74898 au lieu de 74760. Les six moyennes et les six t-statistiques étaient
    donc citées sans que rien, dans l'intervalle cité, ne dise laquelle appartient à SMB et
    laquelle à HML. Un relecteur qui ouvre l'extrait ne peut pas vérifier le fait ; il peut
    seulement constater que les nombres y figurent.

    La règle ne s'applique qu'aux faits **chiffrés** appuyés sur une ancre de tableau, et
    elle est levée dès qu'une des ancres du fait porte l'en-tête. Elle ne bloque pas un fait
    candidat : un candidat ne compte nulle part. Elle bloque la **promotion**.
    """
    erreurs = []
    ident = item.get("id", "?")
    ancres = {a.get("ancre_id"): a for a in item.get("appuis") or []}
    for numero, fait in enumerate(item.get("faits_attendus") or [], 1):
        if fait.get("statut_fait") not in {"gold", "gold_ai_reviewed"}:
            continue
        if not _CHIFFRE.search(fait.get("texte") or ""):
            continue
        utilisees = [ancres[a] for a in fait.get("ancres") or [] if a in ancres]
        tableaux = [a for a in utilisees if a.get("content_type") == "table"]
        if not tableaux:
            continue
        if not any(porte_l_entete_de_tableau(a.get("texte") or "") for a in utilisees):
            erreurs.append(
                f"{ident}: fait {numero} est chiffré et s'appuie sur un tableau "
                f"({', '.join(a.get('ancre_id', '?') for a in tableaux)}) dont aucune ancre "
                "ne porte la ligne d'en-tête : l'appariement colonne ↔ grandeur n'est pas "
                "dans ce qui est cité, et le fait n'est donc pas vérifiable par un lecteur")
    return erreurs


def valider_schema(item: dict) -> list[str]:
    erreurs = []
    ident = item.get("id", "?")

    for champ in ("id", "question", "repondable", "famille", "domaine", "statut"):
        if not item.get(champ):
            erreurs.append(f"{ident}: champ obligatoire manquant — {champ}")
    if item.get("repondable") not in REPONDABLE:
        erreurs.append(f"{ident}: repondable invalide ({item.get('repondable')!r})")
    if item.get("famille") not in FAMILLES:
        erreurs.append(f"{ident}: famille inconnue ({item.get('famille')!r})")
    if item.get("domaine") not in DOMAINES:
        erreurs.append(f"{ident}: domaine inconnu ({item.get('domaine')!r})")
    if item.get("statut") not in STATUTS:
        erreurs.append(f"{ident}: statut inconnu ({item.get('statut')!r})")
    # `valide` est la seule affirmation que le fichier fait au monde extérieur : « cet item
    # peut entrer dans une moyenne publiée ». Elle exige donc une signature humaine et une
    # date. Panne évitée : un lot écrit par un modèle publié comme banc humain — il ne lève
    # aucun des biais qu'on lui demande de lever, et il a l'air plus légitime.
    # `ai_reviewed` est l'affirmation « ce lot a été validé par une revue contradictoire de
    # modèles ». Elle est utile et elle est vraie — à condition de déclarer son mode, sans
    # quoi rien ne la distingue d'une validation humaine dans un tableau de résultats.
    if item.get("statut") == "ai_reviewed":
        if item.get("review_mode") != "multi_agent_ai":
            erreurs.append(f"{ident}: statut « ai_reviewed » sans review_mode="
                           "'multi_agent_ai' — un score doit toujours dire par quoi il a "
                           "été validé")
        if not item.get("valide_par"):
            erreurs.append(f"{ident}: statut « ai_reviewed » sans valide_par (l'adjudicateur)")
        if not _DATE.match(str(item.get("valide_le") or "")):
            erreurs.append(f"{ident}: statut « ai_reviewed » sans valide_le (AAAA-MM-JJ)")
    if item.get("review_mode") is not None and item.get("review_mode") not in REVIEW_MODES:
        erreurs.append(f"{ident}: review_mode invalide ({item.get('review_mode')!r}) ; "
                       f"attendu {sorted(REVIEW_MODES)}")

    if item.get("statut") == "valide":
        if not item.get("valide_par"):
            erreurs.append(f"{ident}: statut « valide » sans valide_par — un item n'est validé "
                           "que par une personne, et son nom fait partie du résultat")
        elif not signature_humaine(item["valide_par"]):
            erreurs.append(f"{ident}: valide_par={item['valide_par']!r} est un modèle ; "
                           "« valide » exige une relecture humaine")
        if not _DATE.match(str(item.get("valide_le") or "")):
            erreurs.append(f"{ident}: statut « valide » sans valide_le (AAAA-MM-JJ)")

    appuis = item.get("appuis") or []
    faits = item.get("faits_attendus") or []
    identifiants = {a.get("ancre_id") for a in appuis}

    if item.get("repondable") == "non":
        if appuis:
            erreurs.append(f"{ident}: un item non répondable ne peut pas porter d'ancres")
        if not item.get("abstention_attendue"):
            erreurs.append(f"{ident}: un item non répondable doit porter abstention_attendue")
        absence = item.get("absence") or {}
        for champ in ("porte", "preuve", "force_de_la_preuve"):
            if not absence.get(champ):
                erreurs.append(f"{ident}: absence.{champ} est obligatoire pour un item non répondable")
        # **Une absence lexicale n'est pas une absence.** Constaté le 7 septembre 2026 :
        # « SEC Rule 605 » ne figure pas une seule fois dans les 26 120 passages, et pourtant
        # la question était répondable — le corpus porte les règles 11Ac1-5 et 11Ac1-6, que
        # la SEC a renumérotées depuis. Une abstention promue sur un `grep` aurait été fausse.
        recherche = absence.get("recherche_semantique")
        if not isinstance(recherche, list) or not recherche:
            erreurs.append(f"{ident}: absence.recherche_semantique est obligatoire — variantes, "
                           "synonymes, anciennes appellations et termes voisins effectivement "
                           "cherchés. Une absence ne se prouve pas par une chaîne de caractères")
        else:
            for n, essai in enumerate(recherche, 1):
                if not isinstance(essai, dict) or not essai.get("variante") \
                        or not isinstance(essai.get("occurrences"), int):
                    erreurs.append(f"{ident}: absence.recherche_semantique[{n}] doit porter "
                                   "variante et occurrences")
            if absence.get("force_de_la_preuve") == "forte" and len(recherche) < 2:
                erreurs.append(f"{ident}: une preuve d'absence « forte » sur une seule variante "
                               "n'est pas forte — c'est le cas SEC 605 exactement")
    else:
        if not appuis:
            erreurs.append(f"{ident}: un item répondable doit porter au moins une ancre")
        if not faits:
            erreurs.append(f"{ident}: un item répondable doit porter au moins un fait attendu")

    # Une ancre est **citée** dès qu'un fait ou une condition la nomme : c'est elle que le
    # relecteur, puis l'utilisateur, verront comme référence. Le critère est comportemental
    # et non déclaratif — le `role` dit une intention, la liste `ancres` dit ce qui sera
    # montré, et c'est la seconde qui engage.
    citees = {a for f in faits for a in (f.get("ancres") or [])}
    citees |= {a for c in (item.get("conditions") or []) for a in (c.get("ancres") or [])}

    for appui in appuis:
        aid = appui.get("ancre_id", "?")
        cite = aid in citees
        if not appui.get("document_id"):
            erreurs.append(f"{ident}/{aid}: document_id manquant")
        if not appui.get("texte"):
            erreurs.append(f"{ident}/{aid}: texte d'ancre manquant")
        if appui.get("role") not in ROLES:
            erreurs.append(f"{ident}/{aid}: role inconnu ({appui.get('role')!r})")
        # Une citation utilisateur est *document + page + offsets + extrait + empreinte*. Les
        # `block_ids` sont une clé **interne** : ils désignent un bloc du parse, que
        # l'utilisateur ne voit pas et qui ne survit pas à un changement de parseur. Une
        # ancre citée sans offsets retomberait sur `chunk_id`, qui ne survit pas non plus à
        # un re-découpage — c'est le trou de produit du §4.2 de `STRATEGIE.md`.
        if cite and appui.get("page") is None:
            erreurs.append(f"{ident}/{aid}: page manquante alors que l'ancre est citée")
        # Un extrait qui court sur deux pages n'a pas *une* page. Annoncer « p. 30 » quand il
        # couvre 30-31 est une précision que le système ne possède pas : il n'a pas de
        # pointeur de phrase. La plage honnête est la seule sortie disponible.
        couvertes = appui.get("pages_couvertes")
        if isinstance(couvertes, list) and len(couvertes) > 1:
            if appui.get("page_fin") != max(couvertes):
                erreurs.append(f"{ident}/{aid}: l'ancre couvre les pages {couvertes} — elle doit "
                               "déclarer sa plage (page_fin), pas une page unique")
        offsets = appui.get("offsets")
        if offsets is None:
            if cite:
                erreurs.append(f"{ident}/{aid}: ancre citée par un fait ou une condition "
                               "(porte_le_fait au sens du contrat) sans offsets — une citation "
                               "sans offsets retombe sur un chunk_id, qui ne survit pas à un "
                               "re-découpage. Retirer l'ancre des faits, ou lui donner ses offsets")
            elif "offsets" in appui and not appui.get("offsets_motif"):
                erreurs.append(f"{ident}/{aid}: offsets absents sans motif déclaré")
        else:
            manquants = [c for c in ("debut", "fin", "doc_text_sha256")
                         if offsets.get(c) in (None, "")]
            if manquants:
                erreurs.append(f"{ident}/{aid}: offsets incomplets — {manquants} ; "
                               "sans doc_text_sha256 un offset n'est pas opposable")
            elif offsets["fin"] <= offsets["debut"]:
                erreurs.append(f"{ident}/{aid}: intervalle d'offsets vide ou inversé")

    # Une réponse partielle ne devient pas complète par défaut. L'item doit dire ce qui est
    # couvert, ce qui ne l'est pas, et porter la condition qui rend **fausse** l'invention de
    # la part manquante — sinon un générateur qui comble le trou serait noté comme s'il avait
    # répondu.
    if item.get("repondable") == "partiel":
        couverture = item.get("couverture_partielle") or {}
        manquants = [c for c in ("couvert", "non_couvert", "preuve") if not couverture.get(c)]
        if manquants:
            erreurs.append(f"{ident}: item partiel — couverture_partielle incomplète {manquants}")
        elif not any((c.get("consequence_si_omis") == "reponse_fausse")
                     for c in (item.get("conditions") or [])):
            erreurs.append(f"{ident}: item partiel sans condition à consequence_si_omis="
                           "'reponse_fausse' — rien n'empêche alors d'inventer la part "
                           "non couverte")

    version_lot = item.get("gold_version")
    if not isinstance(version_lot, int) or version_lot < 1:
        erreurs.append(f"{ident}: gold_version manquante ou invalide ({version_lot!r}) — "
                       "un score sans version de gold n'est pas interprétable")
        version_lot = None

    # Deux versions distinctes, et les confondre serait un piège. `gold_version` dit
    # *contre quoi* on note ; `version_campagne` dit *quand* on a noté. C'est le second qui
    # rend l'invariant temporel vérifiable : un candidat promu au milieu d'une campagne
    # produirait deux gold_version sous une même version_campagne, et cela se voit.
    campagne = item.get("version_campagne")
    if campagne is not None and (not isinstance(campagne, int) or campagne < 1):
        erreurs.append(f"{ident}: version_campagne invalide ({campagne!r})")

    for numero, fait in enumerate(faits, 1):
        if not fait.get("texte"):
            erreurs.append(f"{ident}: fait {numero} sans texte")
        # Une inférence est acceptable ; la confondre avec un énoncé écrit ne l'est pas. Elle
        # doit donc montrer d'où elle part (prémisses citées) et ce qu'elle suppose.
        nature = fait.get("nature")
        if nature is not None and nature not in NATURES:
            erreurs.append(f"{ident}: fait {numero} — nature invalide ({nature!r}) ; "
                           f"attendu {sorted(NATURES)}")
        elif nature == "inference":
            premisses = fait.get("premisses")
            if not isinstance(premisses, list) or not premisses:
                erreurs.append(f"{ident}: fait {numero} est une inférence sans premisses — "
                               "une inférence dont on ne voit pas le point de départ est un "
                               "fait déguisé")
            elif set(premisses) - identifiants:
                erreurs.append(f"{ident}: fait {numero} — premisses inconnues "
                               f"{sorted(set(premisses) - identifiants)}")
            if not isinstance(fait.get("hypotheses"), list) or not fait.get("hypotheses"):
                erreurs.append(f"{ident}: fait {numero} est une inférence sans hypotheses "
                               "visibles")
        if fait.get("exigence") not in EXIGENCES:
            erreurs.append(f"{ident}: fait {numero} — exigence invalide ({fait.get('exigence')!r})")
        inconnues = set(fait.get("ancres") or []) - identifiants
        if inconnues:
            erreurs.append(f"{ident}: fait {numero} renvoie à des ancres inexistantes {sorted(inconnues)}")
        if not fait.get("ancres"):
            erreurs.append(f"{ident}: fait {numero} n'est rattaché à aucune ancre — "
                           "un fait attendu sans source est une opinion")
        erreurs += valider_fait_versionne(ident, numero, fait, version_lot)

    erreurs += valider_lecture_des_tableaux(item)

    for numero, condition in enumerate(item.get("conditions") or [], 1):
        if condition.get("consequence_si_omis") not in CONSEQUENCES:
            erreurs.append(f"{ident}: condition {numero} — consequence_si_omis invalide "
                           f"({condition.get('consequence_si_omis')!r})")
        inconnues = set(condition.get("ancres") or []) - identifiants
        if inconnues:
            erreurs.append(f"{ident}: condition {numero} renvoie à des ancres inexistantes {sorted(inconnues)}")

    citation = item.get("citation_minimale") or {}
    if "documents_distincts" not in citation:
        erreurs.append(f"{ident}: citation_minimale.documents_distincts est obligatoire")
    else:
        disponibles = len({a.get("document_id") for a in appuis})
        if citation["documents_distincts"] > disponibles:
            erreurs.append(f"{ident}: citation_minimale exige {citation['documents_distincts']} "
                           f"documents mais l'item n'en ancre que {disponibles}")
    return erreurs


# ---------------------------------------------------------------- ancres


def valider_ancres(item: dict, corpus: dict[str, list[dict]]) -> tuple[list[str], list[str], list[dict]]:
    erreurs, alertes, mesures = [], [], []
    ident = item.get("id", "?")
    for appui in item.get("appuis") or []:
        aid = appui.get("ancre_id", "?")
        texte = appui.get("texte") or ""
        document = appui.get("document_id")
        mots = len(normaliser(texte).split())
        candidats = corpus.get(document)
        if candidats is None:
            erreurs.append(f"{ident}/{aid}: document inconnu du corpus ({document})")
            continue
        if mots < MOTS_MINIMUM:
            erreurs.append(f"{ident}/{aid}: ancre trop courte ({mots} mots, minimum {MOTS_MINIMUM})")
            continue
        if mots < MOTS_CONFORTABLES:
            alertes.append(f"{ident}/{aid}: ancre courte ({mots} mots) — 20 mots survivent à 100 %")

        situation = situer(texte, candidats)
        mesures.append({"item": ident, "ancre": aid, **situation, "mots": mots})

        if situation["couverture"] < SEUIL_COUVERTURE:
            erreurs.append(f"{ident}/{aid}: ancre introuvable dans le document "
                           f"(couverture {situation['couverture']:.4f} < {SEUIL_COUVERTURE}) — "
                           "texte recopié inexactement, ou coquille du parse corrigée par mégarde")
            continue
        if situation["marge"] < SEUIL_MARGE:
            detail = f"{situation['ex_aequo']} candidats ex aequo" if situation["ex_aequo"] > 1 else "second candidat proche"
            alertes.append(f"{ident}/{aid}: ancre ambiguë (marge {situation['marge']:.4f}, {detail}, "
                           f"{situation['content_type']}) — les blocs devront trancher")
        annonce = appui.get("chunk_id_origine")
        if annonce and annonce != situation["chunk"]:
            alertes.append(f"{ident}/{aid}: chunk_id_origine annoncé {annonce}, "
                           f"meilleur candidat {situation['chunk']}")
    return erreurs, alertes, mesures


# ---------------------------------------------------------------- survie au re-découpage

_FIN_PHRASE = re.compile(r"(?<=[.!?])\s+")


def redecouper(chunks: list[dict], cible: int) -> list[dict]:
    """Re-découpage simulé, aux frontières de phrase, en propageant les blocs.

    Même forme que ``gold_ancrage.redecouper`` — le flux de texte est conservé, seules
    les coupures changent — plus la propagation des blocs, que le chantier de
    représentation devra assurer.
    """
    phrases = []
    for chunk in sorted(chunks, key=lambda c: c["chunk_id"]):
        for p in _FIN_PHRASE.split(chunk["text"]):
            if p.strip():
                phrases.append((p.strip(), chunk["blocs"]))
    neufs, courant, blocs = [], "", set()
    for texte, bl in phrases:
        if courant and len(courant) + len(texte) + 1 > cible:
            neufs.append({"text": courant, "blocs": sorted(blocs)})
            courant, blocs = texte, set(bl)
        else:
            courant = f"{courant} {texte}".strip()
            blocs |= set(bl)
    if courant:
        neufs.append({"text": courant, "blocs": sorted(blocs)})
    return [{**n, "chunk_id": f"neo-{i:04d}", "page_start": None, "page_end": None,
             "content_type": ""} for i, n in enumerate(neufs)]


def charger_gold_ancrage():
    """Le module de ré-ancrage du banc, s'il est atteignable.

    On l'**importe** au lieu de réimplémenter : la survie d'une ancre doit se décider sur
    la même grandeur, avec le même algorithme et le même seuil que pour le gold du banc.
    Une seconde implémentation, c'est une seconde occasion de diverger — et la première
    version de ce contrôle en a fait la démonstration : elle ne regardait que le meilleur
    chunk seul, et déclarait « perdues » des ancres dont le texte était intégralement
    présent, réparti sur deux chunks voisins. C'est exactement le piège que le contrôle II
    de ``gold_ancrage`` avait déjà réfuté, et que l'or pondéré corrige.
    """
    sys.path.insert(0, str(HERE.parent))
    try:
        import gold_ancrage  # noqa: PLC0415
        return gold_ancrage
    except Exception:  # noqa: BLE001 — dépendances du banc absentes d'un worktree sparse
        return None


def controle_redecoupage(items: list[dict], corpus: dict[str, list[dict]],
                         cibles=(900, 1200, 1733, 2400)) -> int:
    """Les ancres du lot survivent-elles à un re-découpage du corpus ?

    C'est le contrôle qui décide si ``human-v1`` peut être annoté **avant** le chantier de
    représentation. Il est exécutable aujourd'hui et ne le sera plus après : une fois le
    corpus re-découpé, il n'existe plus de vérité contre laquelle valider le mappeur.

    La grandeur qui décide est la **couverture de l'union**, pas celle du meilleur chunk :
    un re-découpage qui coupe une ancre en deux ne l'a pas détruite, il l'a répartie.
    """
    ga = charger_gold_ancrage()
    if ga is None:
        print("\n⚠ gold_ancrage introuvable — contrôle de re-découpage sauté")
        return 0
    print(f"\n{'='*94}\nCONTRÔLE — survie des ancres à un re-découpage simulé "
          f"(union, seuils {ga.SEUIL_ANCRE}/{ga.SEUIL_PERDU} de gold_ancrage)\n{'='*94}")
    print(f"{'cible':>7} {'ancres':>7} {'ok':>6} {'partiel':>8} {'perdu':>6} "
          f"{'union méd.':>11} {'union min':>10} {'dispersées':>11} {'blocs tranchent':>16}")
    total_perdues = 0
    for cible in cibles:
        redecoupes = {d: redecouper(c, cible) for d, c in corpus.items()}
        unions, statuts, disperses, ambigues, tranchees, n = [], {"ok": 0, "partiel": 0, "perdu": 0}, 0, 0, 0, 0
        for item in items:
            for appui in item.get("appuis") or []:
                candidats = redecoupes.get(appui["document_id"])
                if not candidats:
                    continue
                n += 1
                verdict = ga.ancrer_un({"texte": appui["texte"],
                                        "document_id": appui["document_id"]}, candidats)
                unions.append(verdict["couverture_union"])
                statuts[verdict["statut"]] += 1
                disperses += bool(verdict["texte_disperse"])
                if verdict["statut"] == "perdu":
                    total_perdues += 1
                # les blocs tranchent-ils quand plusieurs chunks portent l'ancre ?
                if len(verdict["poids"]) > 1:
                    ambigues += 1
                    attendus = set(appui.get("blocs") or [])
                    porteurs = [c for c in candidats if c["chunk_id"] in verdict["poids"]]
                    if attendus and any(attendus & set(c["blocs"]) for c in porteurs):
                        tranchees += 1
        unions.sort()
        milieu = len(unions) // 2
        detail = f"{tranchees}/{ambigues}" if ambigues else "—"
        print(f"{cible:>7} {n:>7} {statuts['ok']:>6} {statuts['partiel']:>8} "
              f"{statuts['perdu']:>6} {unions[milieu]:>11.4f} {unions[0]:>10.4f} "
              f"{disperses:>11} {detail:>16}")
    return total_perdues


# ---------------------------------------------------------------- rapport


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("fichier", nargs="?", default=str(HERE / "items-pilote-v0.jsonl"))
    parser.add_argument("--racine-corpus", default=None)
    parser.add_argument("--enrichir", action="store_true",
                        help="réécrit le fichier avec blocs, page et marge mesurés")
    parser.add_argument("--redecoupage", action="store_true",
                        help="contrôle de survie des ancres à un re-découpage simulé")
    args = parser.parse_args()

    chemin = Path(args.fichier)
    if not chemin.exists():
        sys.exit(f"fichier introuvable : {chemin}")
    items = [json.loads(l) for l in chemin.read_text(encoding="utf-8").splitlines() if l.strip()]

    racine = racine_corpus(args.racine_corpus)
    corpus = charger_corpus(racine)
    print(f"corpus   {sum(len(v) for v in corpus.values())} chunks, {len(corpus)} documents "
          f"({racine})")
    print(f"lot      {len(items)} items — {chemin.name}\n")

    erreurs, alertes, mesures = valider_lot(items), [], []
    for item in items:
        erreurs += valider_schema(item)
        e, a, m = valider_ancres(item, corpus)
        erreurs += e
        alertes += a
        mesures += m

    print(f"{'item':<6} {'ancre':<6} {'mots':>5} {'couv.':>7} {'marge':>7} {'blocs':>6} "
          f"{'page':>5}  {'type':<9} chunk")
    for m in mesures:
        print(f"{m['item']:<6} {m['ancre']:<6} {m['mots']:>5} {m['couverture']:>7.4f} "
              f"{m['marge']:>7.4f} {len(m['blocs']):>6} {str(m['page']):>5}  "
              f"{m['content_type']:<9} {m['chunk']}")

    par_statut: dict[str, int] = {}
    par_repondable: dict[str, int] = {}
    for item in items:
        par_statut[item.get("statut", "?")] = par_statut.get(item.get("statut", "?"), 0) + 1
        par_repondable[item.get("repondable", "?")] = par_repondable.get(item.get("repondable", "?"), 0) + 1
    print(f"\nstatuts     : {par_statut}")
    print(f"répondable  : {par_repondable}")
    documents = {a["document_id"] for i in items for a in (i.get("appuis") or [])}
    print(f"documents   : {len(documents)} distincts ancrés")
    ancres = [a for i in items for a in (i.get("appuis") or [])]
    avec_offsets = sum(1 for a in ancres if a.get("offsets"))
    exactes = sum(1 for a in ancres if (a.get("offsets") or {}).get("exact"))
    print(f"offsets     : {avec_offsets}/{len(ancres)} ancres, dont {exactes} au caractère près "
          "— document + page + offsets + extrait, c'est ce que porte une citation")

    # Le nom du lot n'est pas décoratif. Un jeu écrit par un modèle et non relu par un humain
    # n'est pas un « banc humain indépendant » : il ne lève aucun des biais qu'on lui demande
    # de lever, et le présenter ainsi serait la seule erreur irrattrapable de cette couche.
    # La condition est lue dans le fichier, jamais déclarée dans un document à côté.
    auteurs = {i.get("auteur", "?") for i in items}
    non_relus = all(i.get("statut") in {"idee", "candidat", "pilote"} for i in items)
    if non_relus and not any("humain" in (a or "").lower() for a in auteurs):
        print(f"\nNOM DU LOT  : « pilote de schéma human-v1, à validation humaine ».")
        print(f"              Écrit par {sorted(auteurs)}, relu par personne. Ce n'est pas un")
        print("              banc humain indépendant, et aucun de ses chiffres n'est publiable.")

    # ---- le contrat d'évaluation : ce qui compte, ce qui ne compte pas
    versions = sorted({i.get("gold_version") for i in items if i.get("gold_version")})
    gold = resume_gold(items)
    print(f"\ngold_version: {versions if versions else '— ABSENTE'}")
    print(f"faits       : {gold['par_statut']}")

    # ---- l'aptitude à noter, distincte de la validité du schéma
    empechements = [e for item in items for e in aptitude_a_noter(item)]
    aptes = [i["id"] for i in items if not aptitude_a_noter(i)]
    print(f"aptitude    : {len(aptes)}/{len(items)} items peuvent servir de barème sans rendre "
          f"la note arbitraire — {aptes}")
    for e in empechements:
        print(f"  ⚠ {e}")
    if gold["candidats_par_origine"]:
        print(f"candidats   : par origine {gold['candidats_par_origine']} "
              "— ne comptent dans aucun score")
    if versions:
        courante = versions[-1]
        print(f"\ngold reconstructible par version (faits comptant dans le score) :")
        for v in range(1, courante + 1):
            total = sum(len(gold_a_la_version(i, v)) for i in items)
            print(f"  v{v} : {total} faits")
        contaminants = {f.get("decouvert_par") for i in items
                        for f in (i.get("faits_attendus") or []) if f.get("decouvert_par")}
        for source in sorted(c for c in contaminants if c):
            total = sum(len(gold_a_la_version(i, courante, {source})) for i in items)
            plein = sum(len(gold_a_la_version(i, courante)) for i in items)
            print(f"  v{courante} hors faits découverts par « {source} » : {total} faits "
                  f"({plein - total} écarté(s)) — à utiliser dès que cette configuration est comparée")

    if alertes:
        print(f"\n{len(alertes)} alerte(s) — à relire, pas bloquantes :")
        for a in alertes:
            print(f"  ⚠ {a}")
    if erreurs:
        print(f"\n{len(erreurs)} erreur(s) :")
        for e in erreurs:
            print(f"  ✗ {e}")
    else:
        print("\n✓ schéma et ancres valides")

    if args.enrichir and not erreurs:
        index = {(m["item"], m["ancre"]): m for m in mesures}
        for item in items:
            for appui in item.get("appuis") or []:
                m = index.get((item["id"], appui.get("ancre_id")))
                if not m:
                    continue
                appui["blocs"] = m["blocs"]
                appui["page"] = m["page"]
                appui["content_type"] = m["content_type"]
                appui["ancrage"] = {"couverture": m["couverture"], "marge": m["marge"],
                                    "mots": m["mots"], "ex_aequo": m["ex_aequo"],
                                    "chunk_id_origine": m["chunk"]}
        chemin.write_text("\n".join(json.dumps(i, ensure_ascii=False) for i in items) + "\n",
                          encoding="utf-8")
        print(f"\nenrichi -> {chemin}")

    if args.redecoupage and not erreurs:
        echecs = controle_redecoupage(items, corpus)
        print("\n✓ toutes les ancres survivent" if not echecs
              else f"\n✗ {echecs} ancre(s) perdue(s) au re-découpage")

    raise SystemExit(1 if erreurs else 0)


if __name__ == "__main__":
    main()
