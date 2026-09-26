"""Ré-ancrage du gold du banc sur le **texte**, et non sur l'identifiant de chunk.

Pourquoi ce module existe
-------------------------
``rag/ingestion/make_fixture.py`` fabrique l'identifiant d'un chunk ainsi ::

    "chunk-" + sha1(document_id ‖ str(index) ‖ text)[:16]

L'identifiant dépend donc **du texte et du rang**. Tout chantier qui re-découpe change
l'un et l'autre : la réparation des frontières de phrase change les deux, la
contextualisation avant plongement change le texte. Dans les deux cas, les
**182 identifiants d'or** du banc deviennent pendants d'un coup.

Ce qui survivrait sans ce module, et ce qui ne survivrait pas :

===================  ======================================  ==================================
repli                v1 (25 questions)                       v3 (150 questions)
===================  ======================================  ==================================
``gold_documents``   **absent du schéma**                    présent mais **saturé** —
                     (``qid, question, target_chunk, note``) ``doc_recall@10`` 1,000
``answer_facts``     **0 / 25**                              130 / 150
===================  ======================================  ==================================

Autrement dit, sans ré-ancrage, un re-découpage rend **v1 intégralement immesurable** —
et v1 est le banc *known-item*, celui à 0,897.

Le contrôle qui donne sa valeur au module
-----------------------------------------
Ré-ancrer sur le découpage **actuel** doit rendre les mêmes 182 identifiants, 182/182.
Ce contrôle d'identité est **impossible à exécuter après le re-découpage** : après, il
n'existe plus de vérité contre laquelle valider le mappeur. C'est la raison pour laquelle
ce module est écrit avant le chantier, et non pendant.

Trois autres contrôles l'accompagnent, dont un qui doit **échouer** (``--controle``).

Le piège de comptage — et pourquoi la première règle ne suffisait pas
--------------------------------------------------------------------
``metrics.ndcg`` calcule ``ideal = [GAIN_GOLD] * len(gold_chunks)``. Faire passer un or
de 1 à 3 chunks porterait l'idéal de 3,00 à 4,89 : un système rendant le bon passage au
rang 1 tomberait de 1,00 à 0,61 **sans qu'aucune qualité ait bougé**. La première version
de ce module (commit ``ed7b7c0``) en tirait la règle « un ancre -> exactement un chunk
d'or primaire ».

**Le contrôle II l'a réfutée**, et les chiffres de cet échec sont conservés ici :

    re-découpage simulé à 1 200 caractères, seuils pré-enregistrés inchangés
    n=182   ok=35   partiel=53   perdu=94   dispersés=147
    couverture du primaire : médiane 0,4928

Lecture : les 182 ancres avaient une **union** couvrant >= 0,80 — le texte d'or n'était
jamais détruit, seulement redistribué entre deux chunks voisins. La règle du primaire
unique déclarait « perdues » 94 questions dont la réponse était intégralement présente.

Ce que le contrôle imposait n'était pas un seuil plus bas — cela aurait été du
magasinage de seuil — mais un **or pondéré**, qui est la généralisation correcte :
``metrics.ndcg(..., poids=...)`` donne à chaque chunk d'or un gain ``3 × poids`` et
calcule l'idéal sur les mêmes poids. Un or non coupé (``{lui: 1.0}``) rend exactement
les chiffres d'avant, au bit près ; un or coupé en deux ne coûte plus rien à qui rend
les deux morceaux.

**Les seuils pré-enregistrés n'ont pas bougé** (0,80 / 0,50). Ce qui a changé, c'est la
grandeur sur laquelle ils se lisent : la couverture de l'**union** effectivement versée
dans le gold, et non celle d'un primaire que le piège de comptage imposait artificiellement.

Règle de sélection
------------------
1. Les candidats sont les chunks du **même** ``document_id``. Les identifiants de document
   dérivent du sha256 du PDF : ils ne bougent pas au re-découpage.
2. Normalisation : balises indice/exposant retirées, casse pliée, ponctuation en espaces,
   espaces réduits (``corpus._normalise`` précédé du retrait des balises).
3. Représentation : **5-grammes de mots**. Sous 5 mots, l'ancre entière fait un gramme.
4. ``couverture(ancre, candidat) = |G(ancre) ∩ G(candidat)| / |G(ancre)|``.
5. **Union** = recouvrement glouton d'au plus ``MAX_UNION`` candidats. Chaque chunk retenu
   porte les grammes qu'il est le premier à couvrir : les poids somment à 1 et sont
   déterministes.
6. **Primaire** = couverture maximale d'un chunk seul ; égalité tranchée par l'identifiant
   le plus petit. Diagnostic — il dit si l'or tient encore dans un seul chunk.

Statuts, et ce qu'ils déclenchent
---------------------------------
``ok``       couverture de l'union >= 0,80 — la question reste mesurable.
``partiel``  0,50 <= couverture < 0,80 — question **déclarée**, sortie de la comparaison
             appariée primaire, rapportée à part.
``perdu``    couverture < 0,50 — le passage d'or n'existe plus sous une forme retrouvable ;
             c'est un résultat du chantier, pas un défaut du mappeur.

Un ancre dont l'union couvre >= 0,80 alors que le primaire est sous 0,80 porte
``texte_disperse`` : le chantier n'a pas détruit la réponse, il l'a coupée en morceaux.
La distinction sépare une régression d'un effet de découpe.

La règle de décision — population fixe, non comparables à zéro
--------------------------------------------------------------
Corriger la renormalisation multi-ancres ne suffisait pas. Il restait une porte : si le
score qui décide se lit sur les seules questions *conservées*, un chantier qui rend non
mesurables les questions difficiles remonte la moyenne sans rien améliorer.

Ce n'est pas une hypothèse. Mesuré le 6 septembre 2026 sur la ligne de base réelle
(155 questions, moyenne 0,5993), en retirant simplement du dénominateur les questions les
plus dures ::

    5 retirées   -> +0,0200      10 retirées  -> +0,0413      30 retirées  -> +0,1357

Le seuil que le dossier tient pour décisif est de l'ordre de **+0,010**. Cinq questions
suffisaient donc à le franchir sans qu'aucune qualité ait bougé.

D'où la convention, et elle est non négociable :

``nDCG population d'origine``  155 questions, **dénominateur fixe**, non comparables à 0.
                               **C'est la métrique qui décide.** Conservatrice par
                               construction : un chantier qui perd de l'or le paie.
``nDCG comparable``            sous-ensemble intégralement ré-ancré. **Diagnostic seul**,
                               ne valide jamais un chantier.

``appliquer`` écrit donc **toutes** les questions d'entrée : celles dont une ancre n'a pas
survécu sont neutralisées (or vide des deux côtés, ``comparable: false``,
``motif_exclusion`` renseigné) et valent 0. ``compare_v1_v2.load_v1`` a été corrigé de la
même façon : il conservait autrefois le silence sur une question dont le chunk cible avait
disparu, ce qui rétrécissait le dénominateur exactement de la même manière.

Contrepartie assumée : le biais est **contre** le chantier. Une exclusion due à une limite
du mappeur, et non à une destruction réelle, le pénalise à tort. C'est pourquoi toute
exclusion doit être relue à la main avant d'accepter un verdict — et pourquoi, sous la
limite d'exploitation déclarée (cible >= 1 200 caractères), le contrôle II n'en produit
aucune.

Limites connues, mesurées, non supprimées
-----------------------------------------
Relecture adverse du 6 septembre 2026. Deux défauts trouvés ont été corrigés dans le code
(le croisement du gain pondéré, dans ``metrics`` ; la renormalisation multi-ancres, dans
``appliquer``). Ces deux-ci demeurent, et il vaut mieux les écrire que les taire :

1. **Un candidat sans rapport peut entrer dans l'union par du gabarit partagé.** Le
   recouvrement glouton ne juge que le chevauchement de n-grammes. Sur un corpus à 32 % de
   manuels, un en-tête de tableau ou une formule bibliographique se répète d'un tableau à
   l'autre du même document. ``MIN_APPORT`` retire la poussière (20 membres sur 517 au
   contrôle II) mais ne règle pas un gabarit massivement partagé.
   **Le préjudice est borné** : depuis la correction du gain, un membre de poids w vaut
   ``1 + 2w`` là où un voisin quelconque vaut déjà 1. Le crédit indu est donc au plus
   ``2w``, et il faudrait un distracteur à fort recouvrement pour qu'il pèse. Les
   contrôles I–IV ne le verraient pas : le sabotage permute des documents entiers, pas des
   coïncidences de gabarit intra-document.

2. **Le départage des poids entre deux candidats symétriques est arbitraire.** Deux chunks
   couvrant l'ancre à parts égales reçoivent 2/3 et 1/3 selon l'ordre d'un identifiant, qui
   est un sha1 — sans rapport avec le contenu. L'erreur est bornée (somme à 1) et symétrique
   en espérance ; elle ajoute du bruit, pas un biais de signe.

Usage
-----
    .venv/bin/python rag/benchmark/gold_ancrage.py --construire
    .venv/bin/python rag/benchmark/gold_ancrage.py --controle
    .venv/bin/python rag/benchmark/gold_ancrage.py --sensibilite
    .venv/bin/python rag/benchmark/gold_ancrage.py --appliquer nouveaux-chunks.jsonl \
        --signature-cible <sig>
    .venv/bin/python rag/benchmark/gold_ancrage.py --construire-v4
    .venv/bin/python rag/benchmark/gold_ancrage.py --appliquer-v4 nouveaux-chunks.jsonl \
        --signature-cible <sig>

``--appliquer`` attend un JSONL portant au minimum ``chunk_id``, ``document_id``, ``text``.
Les commandes ``-v4`` couvrent formula/table_cell/negative_voisine sur un fichier d'ancres
distinct (``gold-anchors-v4.json``) et écrivent ``questions-v4-<famille>-<sig>.jsonl`` —
jamais les fichiers d'origine, qui restent la seule source de vérité du banc.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys

HERE = pathlib.Path(__file__).resolve().parent
RACINE = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import corpus  # noqa: E402  — ChunkIndex : le corpus tel qu'il est servi
import corpus_overlay  # noqa: E402

ANCRES = HERE / "gold-anchors-v1.json"
QUESTIONS_V1 = HERE / "questions-v1.jsonl"
QUESTIONS_V3 = HERE / "questions-v3.jsonl"

#: Ancres v4 : fichier séparé, délibérément — gold-anchors-v1.json est l'artefact validé
#: par les quatre contrôles, rien de ce qui suit ne doit pouvoir le faire bouger.
ANCRES_V4 = HERE / "gold-anchors-v4.json"
QUESTIONS_V4 = {
    "formula": HERE / "questions-v4-formula.jsonl",
    "table_cell": HERE / "questions-v4-table-cell.jsonl",
    "negative_voisine": HERE / "questions-v4-negative-voisine.jsonl",
}

#: Taille du gramme. 5 mots : assez long pour ne pas s'apparier par hasard entre deux
#: passages du même document (qui partagent le vocabulaire), assez court pour survivre à
#: un déplacement de frontière. Fixé avant toute mesure.
K_GRAMME = 5

#: Couverture de l'union au-delà de laquelle la question reste mesurable. Pré-enregistré.
SEUIL_ANCRE = 0.80

#: En dessous, l'ancre est déclarée perdue. Entre les deux, « partiel ».
SEUIL_PERDU = 0.50

#: Nombre maximal de candidats dans le recouvrement glouton. Au-delà, un « passage d'or »
#: n'en est plus un : c'est une section entière.
MAX_UNION = 4

#: Part minimale de l'ancre qu'un candidat doit apporter pour entrer dans l'union.
#:
#: Ajouté le 6 septembre 2026 après relecture adverse, et déclaré comme tel : le glouton
#: n'avait aucun plancher, donc un chunk sans rapport avec la réponse pouvait entrer par
#: quelques n-grammes de gabarit partagé — en-tête de tableau, formule bibliographique —
#: sur un corpus qui est à 32 % des manuels. Mesuré sur le contrôle II : 20 membres sur
#: 517 (3,9 %) sont sous ce plancher, et AUCUN statut d'ancre n'en change.
#:
#: Ce plancher retire la poussière, il ne règle pas le cas d'un gabarit massivement
#: partagé — voir la limite connue en fin de docstring.
MIN_APPORT = 0.05

_BALISE = re.compile(r"</?[a-zA-Z][^>]{0,40}>")


# ------------------------------------------------------------------ normalisation


def normaliser(texte: str) -> str:
    """Balises retirées, puis la normalisation du banc (casse, ponctuation, espaces)."""
    return corpus._normalise(_BALISE.sub(" ", str(texte or "")))


def grammes(texte: str, k: int = K_GRAMME) -> set[str]:
    """n-grammes de mots, n = k. Sous k mots, un seul gramme : le texte entier.

    Le repli « texte entier » ne vaut que si le CANDIDAT est découpé au même k — sinon
    une ancre de 4 mots ne pouvait s'apparier à aucun candidat plus long, même contenant
    l'ancre mot pour mot : ses fenêtres de 5 mots ne peuvent jamais égaler une chaîne de
    4. ``ancrer_un`` fixe donc un k effectif commun aux deux côtés. Défaut latent
    aujourd'hui — la plus courte des 182 ancres réelles fait 58 mots — mais réel, et
    trouvé par relecture adverse le 6 septembre 2026.
    """
    mots = normaliser(texte).split()
    if not mots:
        return set()
    if len(mots) < k:
        return {" ".join(mots)}
    return {" ".join(mots[i:i + k]) for i in range(len(mots) - k + 1)}


def couverture(ancre: set[str], candidat: set[str]) -> float:
    return len(ancre & candidat) / len(ancre) if ancre else 0.0


# ------------------------------------------------------------------ construction


def gold_du_banc() -> dict[str, list[str]]:
    """Identifiant d'or -> qids qui s'en servent, sur les deux bancs."""
    usage: dict[str, list[str]] = {}
    for ligne in QUESTIONS_V1.read_text(encoding="utf-8").splitlines():
        if not ligne.strip():
            continue
        q = json.loads(ligne)
        usage.setdefault(q["target_chunk"], []).append(f"v1/{q['qid']}")
    for ligne in QUESTIONS_V3.read_text(encoding="utf-8").splitlines():
        if not ligne.strip():
            continue
        q = json.loads(ligne)
        for chunk in q.get("gold_chunks") or []:
            usage.setdefault(chunk, []).append(f"v3/{q['qid']}")
    return usage


def construire(index=None) -> dict:
    """Écrit ``gold-anchors-v1.json`` : le texte servi de chaque chunk d'or."""
    index = index or corpus.ChunkIndex.load()
    usage = gold_du_banc()
    ancres, absents = {}, []
    for chunk_id, qids in sorted(usage.items()):
        row = index.chunks.get(chunk_id)
        if row is None:
            absents.append(chunk_id)
            continue
        texte = row["text"]
        ancres[chunk_id] = {
            "chunk_id": chunk_id,
            "document_id": row["document_id"],
            "qids": sorted(qids),
            "texte": texte,
            "n_mots": len(normaliser(texte).split()),
            "n_grammes": len(grammes(texte)),
            "section": row.get("section") or "",
            "page_start": row.get("page_start"),
            "content_type": row.get("content_type") or "",
        }
    charge = {
        "version": 1,
        "signature_construction": corpus_overlay.signature(),
        "documents_corpus": len(index.documents),
        "chunks_corpus": len(index.chunks),
        "k_gramme": K_GRAMME,
        "seuil_ancre": SEUIL_ANCRE,
        "seuil_perdu": SEUIL_PERDU,
        "max_union": MAX_UNION,
        "ancres": ancres,
        "absents": absents,
    }
    ANCRES.write_text(json.dumps(charge, ensure_ascii=False, indent=1), encoding="utf-8")
    return charge


def charger_ancres() -> dict:
    if not ANCRES.exists():
        sys.exit(f"ancres introuvables : {ANCRES} — lancer --construire")
    return json.loads(ANCRES.read_text(encoding="utf-8"))


# ------------------------------------------------------------- v4 : formula, table_cell, négative


def _lire_jsonl(chemin: pathlib.Path) -> list[dict]:
    return [json.loads(l) for l in chemin.read_text(encoding="utf-8").splitlines() if l.strip()]


def _ancre_question_v4(famille: str, q: dict) -> str:
    """Chunk d'or d'une question v4.

    ``negative_voisine`` n'a pas d'or positif — ``gold_chunks`` y est vide par
    construction, la question porte sur une absence prouvée, pas sur une réponse. Son
    ancre est ``chunk_voisin`` : c'est déjà la convention de ``familles_v4.verifier``
    (même champ, même rôle, ligne ~773). Sans ce repli, la famille resterait hors du
    ré-ancrage sans qu'aucune erreur ne le signale, puisque son schéma contient bien un
    champ ``gold_chunks`` — juste toujours vide.

    Les deux autres familles portent aujourd'hui exactement un ``gold_chunks`` chacune ;
    on prend le premier, comme ``familles_v4`` le fait déjà pour construire ses propres
    prompts de vérification.
    """
    return q["chunk_voisin"] if famille == "negative_voisine" else q["gold_chunks"][0]


def gold_du_banc_v4() -> dict[str, list[str]]:
    """Identifiant d'or -> qids qui s'en servent, sur les trois familles v4."""
    usage: dict[str, list[str]] = {}
    for famille, chemin in QUESTIONS_V4.items():
        for q in _lire_jsonl(chemin):
            chunk_id = _ancre_question_v4(famille, q)
            usage.setdefault(chunk_id, []).append(f"v4-{famille}/{q['qid']}")
    return usage


def construire_v4(index=None) -> dict:
    """Écrit ``gold-anchors-v4.json`` : même méthode que ``construire()``, gold distinct.

    Corps volontairement dupliqué plutôt que factorisé avec ``construire()`` : cette
    dernière est l'artefact validé par les quatre contrôles, et n'est pas modifiée ici.
    """
    index = index or corpus.ChunkIndex.load()
    usage = gold_du_banc_v4()
    ancres, absents = {}, []
    for chunk_id, qids in sorted(usage.items()):
        row = index.chunks.get(chunk_id)
        if row is None:
            absents.append(chunk_id)
            continue
        texte = row["text"]
        ancres[chunk_id] = {
            "chunk_id": chunk_id,
            "document_id": row["document_id"],
            "qids": sorted(qids),
            "texte": texte,
            "n_mots": len(normaliser(texte).split()),
            "n_grammes": len(grammes(texte)),
            "section": row.get("section") or "",
            "page_start": row.get("page_start"),
            "content_type": row.get("content_type") or "",
        }
    charge = {
        "version": 1,
        "signature_construction": corpus_overlay.signature(),
        "documents_corpus": len(index.documents),
        "chunks_corpus": len(index.chunks),
        "k_gramme": K_GRAMME,
        "seuil_ancre": SEUIL_ANCRE,
        "seuil_perdu": SEUIL_PERDU,
        "max_union": MAX_UNION,
        "ancres": ancres,
        "absents": absents,
    }
    ANCRES_V4.write_text(json.dumps(charge, ensure_ascii=False, indent=1), encoding="utf-8")
    return charge


def charger_ancres_v4() -> dict:
    if not ANCRES_V4.exists():
        sys.exit(f"ancres v4 introuvables : {ANCRES_V4} — lancer --construire-v4")
    return json.loads(ANCRES_V4.read_text(encoding="utf-8"))


# ------------------------------------------------------------------ mappage


def ancrer_un(ancre: dict, candidats: list[dict]) -> dict:
    """Ré-ancre un chunk d'or sur un nouveau découpage du même document.

    Rend un or **pondéré** : ``poids[chunk] = part de l'ancre que ce chunk porte``,
    somme 1 sur la partie retrouvée. Un or non coupé donne ``{lui: 1.0}`` et le nDCG
    est alors identique au chiffre d'avant, au bit près.
    """
    # k effectif commun aux deux côtés : sans cela, une ancre plus courte que K_GRAMME
    # ne pourrait s'apparier à aucun candidat plus long (voir grammes()).
    k = min(K_GRAMME, len(normaliser(ancre["texte"]).split()) or K_GRAMME)
    cible = grammes(ancre["texte"], k)
    vide = {"primaire": None, "couverture_primaire": 0.0, "union": [], "poids": {},
            "couverture_union": 0.0, "statut": "perdu", "texte_disperse": False,
            "n_candidats": len(candidats)}
    if not cible:
        return vide

    scores = []
    for candidat in candidats:
        g = grammes(candidat["text"], k)
        scores.append((couverture(cible, g), candidat["chunk_id"], g))
    if not scores:
        return vide
    # Primaire : couverture maximale, égalité tranchée par l'identifiant le plus petit.
    # Diagnostic — il dit si l'or tient encore dans un seul chunk.
    scores.sort(key=lambda triplet: (-triplet[0], triplet[1]))
    couv_primaire, primaire, _ = scores[0]

    # Union : recouvrement glouton. Chaque chunk retenu porte les grammes qu'il est le
    # premier à couvrir — d'où des poids déterministes qui somment à 1.
    par_id = {chunk_id: g for _c, chunk_id, g in scores}
    restant, union, apport = set(cible), [], {}
    for _ in range(MAX_UNION):
        meilleur, gain_max = None, 0
        for _c, chunk_id, g in scores:
            if chunk_id in union:
                continue
            gain = len(restant & g)
            if gain > gain_max:
                meilleur, gain_max = chunk_id, gain
        if meilleur is None or gain_max < MIN_APPORT * len(cible):
            break
        union.append(meilleur)
        apport[meilleur] = gain_max
        restant -= par_id[meilleur]
    couvert = sum(apport.values())
    couv_union = couvert / len(cible)
    poids = {chunk_id: round(part / couvert, 6) for chunk_id, part in apport.items()} \
        if couvert else {}

    if couv_union >= SEUIL_ANCRE:
        statut = "ok"
    elif couv_union >= SEUIL_PERDU:
        statut = "partiel"
    else:
        statut = "perdu"
    return {"primaire": primaire, "couverture_primaire": round(couv_primaire, 4),
            "union": union, "poids": poids, "couverture_union": round(couv_union, 4),
            "statut": statut, "texte_disperse": couv_union >= SEUIL_ANCRE > couv_primaire,
            "n_candidats": len(candidats)}


def ancrer(ancres: dict, nouveaux: list[dict]) -> dict:
    """Ré-ancre tous les ors sur un nouveau découpage.

    ``nouveaux`` : itérable de dicts portant ``chunk_id``, ``document_id``, ``text``.
    """
    par_document: dict[str, list[dict]] = {}
    for row in nouveaux:
        par_document.setdefault(row["document_id"], []).append(row)
    return {chunk_id: ancrer_un(ancre, par_document.get(ancre["document_id"], []))
            for chunk_id, ancre in ancres.items()}


def resume(mappage: dict) -> dict:
    compte = {"ok": 0, "partiel": 0, "perdu": 0}
    for verdict in mappage.values():
        compte[verdict["statut"]] += 1
    prim = sorted(v["couverture_primaire"] for v in mappage.values())
    uni = sorted(v["couverture_union"] for v in mappage.values())
    tailles = sorted(len(v["poids"]) for v in mappage.values())
    milieu = len(prim) // 2
    return {"n": len(mappage), **compte,
            "disperses": sum(1 for v in mappage.values() if v["texte_disperse"]),
            "couverture_primaire_min": prim[0] if prim else None,
            "couverture_primaire_mediane": prim[milieu] if prim else None,
            "couverture_union_min": uni[0] if uni else None,
            "couverture_union_mediane": uni[milieu] if uni else None,
            "chunks_par_or_mediane": tailles[milieu] if tailles else None,
            "chunks_par_or_max": tailles[-1] if tailles else None,
            "identite": sum(1 for cid, v in mappage.items() if v["primaire"] == cid)}


# ------------------------------------------------------------------ contrôles


def _chunks_des_documents(index, documents: set[str]) -> dict[str, list[dict]]:
    par_document: dict[str, list[dict]] = {}
    for row in index.chunks.values():
        if row["document_id"] in documents:
            par_document.setdefault(row["document_id"], []).append(row)
    return par_document


_FIN_PHRASE = re.compile(r"(?<=[.!?])\s+")


def redecouper(textes: list[str], cible: int = 1200) -> list[str]:
    """Re-découpage simulé : même texte, frontières de phrase, autre taille visée.

    C'est exactement la forme du chantier « réparer les frontières » : le flux de texte
    est conservé, seules les coupures changent.
    """
    phrases = [p for bloc in textes for p in _FIN_PHRASE.split(bloc) if p.strip()]
    morceaux, courant = [], ""
    for phrase in phrases:
        if courant and len(courant) + len(phrase) + 1 > cible:
            morceaux.append(courant)
            courant = phrase
        else:
            courant = f"{courant} {phrase}".strip()
    if courant:
        morceaux.append(courant)
    return morceaux


def controle(index=None) -> dict:
    """Quatre contrôles, dont un qui doit échouer."""
    index = index or corpus.ChunkIndex.load()
    charge = charger_ancres()
    ancres = charge["ancres"]
    documents = {a["document_id"] for a in ancres.values()}
    par_document = _chunks_des_documents(index, documents)
    rapports = {}

    # I — identité : ré-ancrer sur le découpage actuel doit rendre les mêmes identifiants.
    actuels = [row for rows in par_document.values() for row in rows]
    m1 = ancrer(ancres, actuels)
    r1 = resume(m1)
    r1["verdict"] = "OK" if r1["identite"] == r1["n"] and r1["ok"] == r1["n"] else "ÉCHEC"
    rapports["I_identite"] = r1

    # II — re-découpage simulé aux frontières de phrase : la forme du chantier réel.
    simules = []
    for document_id, rows in par_document.items():
        for i, morceau in enumerate(redecouper([r["text"] for r in rows])):
            simules.append({"chunk_id": f"sim-{document_id}-{i:04d}",
                            "document_id": document_id, "text": morceau})
    m2 = ancrer(ancres, simules)
    r2 = resume(m2)
    r2["n_chunks_simules"] = len(simules)
    r2["verdict"] = "OK" if r2["perdu"] == 0 else "ÉCHEC"
    rapports["II_redecoupage"] = r2

    # III — contextualisation simulée : un préfixe généré devant chaque chunk.
    contextualises = []
    for document_id, rows in par_document.items():
        titre = index.documents.get(document_id, {}).get("title", "")
        for row in rows:
            prefixe = (f"Ce passage provient de {titre}, section "
                       f"{row.get('section') or 'sans titre'}. ")
            contextualises.append({"chunk_id": f"ctx-{row['chunk_id']}",
                                   "document_id": document_id,
                                   "text": prefixe + row["text"]})
    m3 = ancrer(ancres, contextualises)
    r3 = resume(m3)
    r3["verdict"] = "OK" if r3["ok"] == r3["n"] else "ÉCHEC"
    rapports["III_contextualisation"] = r3

    # IV — sabotage : le texte de chaque document d'or remplacé par celui d'un autre.
    # La garde doit **échouer**. Un contrôle qui ne peut pas échouer ne contrôle rien.
    ordre = sorted(par_document)
    melange = ordre[1:] + ordre[:1]
    sabotes = []
    for cible_doc, source_doc in zip(ordre, melange):
        for row in par_document[source_doc]:
            sabotes.append({"chunk_id": f"sab-{row['chunk_id']}",
                            "document_id": cible_doc, "text": row["text"]})
    m4 = ancrer(ancres, sabotes)
    r4 = resume(m4)
    # On exige que le sabotage soit vu : au plus 5 % d'ancres survivent par hasard.
    r4["verdict"] = "OK (sabotage détecté)" if r4["ok"] <= 0.05 * r4["n"] else "ÉCHEC"
    rapports["IV_sabotage"] = r4

    rapports["signature"] = corpus_overlay.signature()
    rapports["signature_construction"] = charge["signature_construction"]
    return rapports


# ------------------------------------------------------------------ application


def _neutraliser(question: dict, motif: str) -> dict:
    """Rend une question **non comparable** sans la retirer de la population.

    Or vide des deux côtés : ``metrics.ndcg`` rend alors 0,0 (vérifié — la référence est
    nulle, le score aussi). La question reste donc au dénominateur et pèse 0. C'est la
    convention de la métrique de décision.
    """
    question.update({"target_chunk": None, "gold_chunks": [], "gold_documents": [],
                     "gold_poids": {}, "comparable": False, "motif_exclusion": motif})
    return question


def appliquer(chemin: pathlib.Path, signature_cible: str) -> dict:
    """Ré-ancre sur un nouveau découpage et écrit les questions ré-orées.

    **La population de sortie est celle d'entrée, toujours.** Une question dont une ancre
    n'a pas survécu n'est pas retirée du fichier : elle est neutralisée et vaut 0. Retirer
    des questions du dénominateur laisserait un chantier améliorer son score en dégradant
    la couverture — mesuré sur la ligne de base réelle du 6 septembre 2026 : retirer les
    5 questions les plus dures des 155 rend **+0,0200** de nDCG apparent, le double du
    seuil que le dossier tient pour décisif ; en retirer 10, **+0,0413**.

    Deux populations sortent du même fichier, et la règle de décision les distingue :

    ``nDCG population d'origine``  — 155 questions, dénominateur fixe, exclues à 0.
                                     **C'est la métrique qui décide.** Elle est
                                     conservatrice par construction : un chantier qui perd
                                     de l'or le paie, jamais l'inverse.
    ``nDCG comparable``            — sous-ensemble intégralement ré-ancré. **Diagnostic
                                     seul.** Elle ne peut jamais valider un chantier.

    Une exclusion doit être relue à la main avant d'accepter un verdict : elle peut dire
    « le chantier a détruit ce passage » (résultat réel) comme « le mappeur n'a pas su le
    retrouver » (limite de l'instrument). Sous la limite d'exploitation déclarée — cible
    >= 1 200 caractères — le contrôle II ne produit aucune exclusion.
    """
    charge = charger_ancres()
    nouveaux = [json.loads(l) for l in chemin.read_text(encoding="utf-8").splitlines() if l.strip()]
    mappage = ancrer(charge["ancres"], nouveaux)
    retenus = {cid: v for cid, v in mappage.items() if v["statut"] == "ok"}
    document_de = {r["chunk_id"]: r["document_id"] for r in nouveaux}

    def _motif(origines: list[str]) -> str:
        morceaux = []
        for c in origines:
            v = mappage.get(c)
            morceaux.append(f"{c}: absent des ancres" if v is None else
                            f"{c}: {v['statut']} (union {v['couverture_union']:.4f})")
        return " ; ".join(morceaux) or "aucune ancre"

    couverture: list[dict] = []
    sortie_v1, exclues_v1 = [], []
    for ligne in QUESTIONS_V1.read_text(encoding="utf-8").splitlines():
        if not ligne.strip():
            continue
        q = json.loads(ligne)
        origine = q["target_chunk"]
        verdict = retenus.get(origine)
        if verdict is None:
            exclues_v1.append(q["qid"])
            _neutraliser(q, _motif([origine]))
        else:
            q["target_chunk"] = verdict["primaire"]
            q["gold_chunks"] = list(verdict["union"])
            q["gold_poids"] = verdict["poids"]
            q["gold_documents"] = sorted({document_de[c] for c in verdict["union"]})
            q["comparable"] = True
            q["ancre_origine"] = {"chunk_id": origine,
                                  "couverture_union": verdict["couverture_union"]}
        couverture.append({"banc": "v1", "qid": q["qid"], "n_ancres": 1,
                           "comparable": q.get("comparable", True),
                           "motif": q.get("motif_exclusion")})
        sortie_v1.append(q)          # TOUJOURS : la population ne rétrécit pas

    sortie_v3, exclues_v3 = [], []
    for ligne in QUESTIONS_V3.read_text(encoding="utf-8").splitlines():
        if not ligne.strip():
            continue
        q = json.loads(ligne)
        origines = list(q.get("gold_chunks") or [])
        # Une question n'est comparable que si TOUTES ses ancres sont « ok ». Renormaliser
        # sur les survivantes gonflerait le score sans que le système ait changé : une
        # question à trois ancres dont une disparaît passait de 0,765 à 1,000 alors qu'elle
        # ratait toujours le même fait. C'est le piège de comptage du cas mono-ancre, resté
        # ouvert au niveau multi-ancres — trouvé par relecture adverse le 6 septembre 2026.
        # 28 des 150 questions v3 portent deux ancres et y étaient exposées.
        if not origines or any(c not in retenus for c in origines):
            exclues_v3.append(q["qid"])
            _neutraliser(q, _motif(origines))
        else:
            verdicts = [retenus[c] for c in origines]
            # Les poids se partagent à parts égales entre les ancres, puis se répartissent
            # à l'intérieur de chaque ancre.
            poids: dict[str, float] = {}
            for verdict in verdicts:
                for chunk_id, part in verdict["poids"].items():
                    poids[chunk_id] = poids.get(chunk_id, 0.0) + part / len(verdicts)
            q["gold_chunks"] = sorted(poids)
            q["gold_poids"] = {c: round(p, 6) for c, p in sorted(poids.items())}
            q["gold_documents"] = sorted({document_de[c] for c in poids})
            q["comparable"] = True
        couverture.append({"banc": "v3", "qid": q["qid"], "n_ancres": len(origines),
                           "comparable": q.get("comparable", True),
                           "motif": q.get("motif_exclusion")})
        sortie_v3.append(q)          # TOUJOURS

    (HERE / f"questions-v1-{signature_cible}.jsonl").write_text(
        "\n".join(json.dumps(q, ensure_ascii=False) for q in sortie_v1) + "\n", encoding="utf-8")
    (HERE / f"questions-v3-{signature_cible}.jsonl").write_text(
        "\n".join(json.dumps(q, ensure_ascii=False) for q in sortie_v3) + "\n", encoding="utf-8")

    multi = [c for c in couverture if c["n_ancres"] > 1]
    rapport = {
        "signature_cible": signature_cible,
        "regle_de_decision": (
            "métrique de décision = nDCG sur la population d'origine, dénominateur fixe, "
            "questions non comparables à 0. La métrique sur le sous-ensemble comparable "
            "est DIAGNOSTIQUE et ne peut jamais valider un chantier."),
        "population_origine": {"v1": len(sortie_v1), "v3": len(sortie_v3),
                               "total": len(sortie_v1) + len(sortie_v3)},
        "comparables": {"v1": sum(1 for c in couverture if c["banc"] == "v1" and c["comparable"]),
                        "v3": sum(1 for c in couverture if c["banc"] == "v3" and c["comparable"]),
                        "total": sum(1 for c in couverture if c["comparable"])},
        "non_comparables": {"v1": exclues_v1, "v3": exclues_v3,
                            "total": len(exclues_v1) + len(exclues_v3)},
        "multi_ancres": {"n": len(multi),
                         "non_comparables": [c["qid"] for c in multi if not c["comparable"]]},
        "resume_ancres": resume(mappage),
        "couverture": couverture,
        "detail": mappage,
    }
    (HERE / f"gold-remap-{signature_cible}.json").write_text(
        json.dumps(rapport, ensure_ascii=False, indent=1), encoding="utf-8")
    return rapport


def appliquer_v4(chemin: pathlib.Path, signature_cible: str) -> dict:
    """Ré-ancre les trois familles v4 sur un nouveau découpage et écrit les questions ré-orées.

    Même geste que ``appliquer()`` pour v1/v3 (``questions-v1-<sig>.jsonl``), pour la même
    raison : ``banc_v4.population`` lit ``questions-v4-*.jsonl`` en dur, et son diagnostic
    « or servi » (``noter()``) compare les ``chunk_id`` servis à ``gold_chunks`` — des
    identifiants d'un autre découpage sans ce fichier. Seul le champ porteur d'or de chaque
    question change (``gold_chunks``, ou ``chunk_voisin`` pour ``negative_voisine`` — elle
    n'a pas de ``gold_chunks``, voir ``_ancre_question_v4``) ; le reste est copié tel quel.
    Une ancre qui ne survit pas (non « ok ») laisse ce champ vide plutôt que de pointer sur
    un chunk douteux : ``retenus`` reprend exactement le filtre de ``appliquer()``.
    """
    charge = charger_ancres_v4()
    mappage = ancrer(charge["ancres"], _lire_jsonl(chemin))
    retenus = {cid: v for cid, v in mappage.items() if v["statut"] == "ok"}

    par_famille, detail, non_servi = {}, {}, []
    for famille, fichier in QUESTIONS_V4.items():
        compte = {"n": 0, "ok": 0, "partiel": 0, "perdu": 0}
        lignes, sortie = [], []
        for q in _lire_jsonl(fichier):
            origine = _ancre_question_v4(famille, q)
            verdict = mappage.get(origine)
            if verdict is None:
                # L'ancre elle-même est absente de gold-anchors-v4.json : un défaut de
                # construction (chunk introuvable dans le corpus SERVI), pas un verdict
                # du mappeur. Compté à part pour ne pas le confondre avec un « perdu ».
                statut = "absent_du_servi"
                primaire, union, couv = None, [], 0.0
            else:
                statut = verdict["statut"]
                primaire, union, couv = verdict["primaire"], verdict["union"], verdict["couverture_union"]
            compte["n"] += 1
            compte[statut] = compte.get(statut, 0) + 1
            ligne = {"qid": q["qid"], "famille": famille, "chunk_origine": origine,
                     "statut": statut, "primaire": primaire, "union": union,
                     "couverture_union": couv}
            lignes.append(ligne)
            if statut != "ok":
                non_servi.append(ligne)

            q = dict(q)  # copie : ne jamais muter l'objet lu depuis le fichier d'origine
            if famille == "negative_voisine":
                q["chunk_voisin"] = primaire if origine in retenus else None
            else:
                q["gold_chunks"] = list(union) if origine in retenus else []
            sortie.append(q)

        nom_sortie = fichier.name.replace(".jsonl", f"-{signature_cible}.jsonl")
        (HERE / nom_sortie).write_text(
            "\n".join(json.dumps(q, ensure_ascii=False) for q in sortie) + "\n", encoding="utf-8")
        par_famille[famille] = compte
        detail[famille] = lignes

    rapport = {"signature_cible": signature_cible, "n_ancres_v4": len(charge["ancres"]),
               "resume_ancres": resume(mappage), "par_famille": par_famille,
               "or_non_servi": non_servi, "detail": detail}
    (HERE / f"gold-remap-v4-{signature_cible}.json").write_text(
        json.dumps(rapport, ensure_ascii=False, indent=1), encoding="utf-8")
    return rapport


# ------------------------------------------------------------------ CLI


def _afficher(titre: str, r: dict) -> None:
    print(f"\n  {titre}")
    print(f"    n={r['n']}  ok={r['ok']}  partiel={r['partiel']}  perdu={r['perdu']}"
          f"  dispersés={r['disperses']}")
    print(f"    couverture de l'union  : min {r['couverture_union_min']} · "
          f"médiane {r['couverture_union_mediane']}")
    print(f"    couverture du primaire : min {r['couverture_primaire_min']} · "
          f"médiane {r['couverture_primaire_mediane']}")
    print(f"    chunks par or : médiane {r['chunks_par_or_mediane']} · "
          f"max {r['chunks_par_or_max']}")
    if "n_chunks_simules" in r:
        print(f"    chunks du découpage simulé : {r['n_chunks_simules']}")
    print(f"    identité (primaire == ancre) : {r['identite']}/{r['n']}")
    print(f"    -> {r['verdict']}")


def sensibilite(index=None, cibles=(600, 900, 1200, 1733, 2400, 3200)) -> list[dict]:
    """Comment l'instrument se comporte selon l'agressivité du re-découpage.

    Ce n'est pas un choix de seuil : c'est la caractérisation de l'instrument sur toute
    la plage, publiée en entier. La médiane servie est 1 733 caractères et la médiane
    d'un chunk d'or 2 281 — en dessous, un or est coupé par construction.
    """
    index = index or corpus.ChunkIndex.load()
    charge = charger_ancres()
    ancres = charge["ancres"]
    par_document = _chunks_des_documents(index, {a["document_id"] for a in ancres.values()})
    lignes = []
    for cible in cibles:
        simules = []
        for document_id, rows in par_document.items():
            for i, morceau in enumerate(redecouper([r["text"] for r in rows], cible)):
                simules.append({"chunk_id": f"sim{cible}-{document_id}-{i:04d}",
                                "document_id": document_id, "text": morceau})
        r = resume(ancrer(ancres, simules))
        r["cible"] = cible
        r["n_chunks_simules"] = len(simules)
        lignes.append(r)
    return lignes


def main() -> None:
    a = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    a.add_argument("--construire", action="store_true", help="écrit gold-anchors-v1.json")
    a.add_argument("--controle", action="store_true", help="les quatre contrôles")
    a.add_argument("--sensibilite", action="store_true",
                   help="caractérise l'instrument sur toute la plage de re-découpage")
    a.add_argument("--appliquer", metavar="JSONL", help="nouveau découpage à ré-ancrer")
    a.add_argument("--signature-cible", help="signature du nouveau découpage")
    a.add_argument("--construire-v4", action="store_true",
                   help="écrit gold-anchors-v4.json (formula, table_cell, negative_voisine)")
    a.add_argument("--appliquer-v4", metavar="JSONL",
                   help="nouveau découpage à ré-ancrer sur gold-anchors-v4.json")
    args = a.parse_args()

    if args.construire:
        charge = construire()
        print(f"  ancres écrites : {len(charge['ancres'])} chunks d'or, "
              f"{len(charge['absents'])} absent(s) du corpus")
        print(f"  signature de construction : {charge['signature_construction']}")
        print(f"  -> {ANCRES.relative_to(RACINE)}")
    elif args.controle:
        r = controle()
        print(f"\n  signature du corpus : {r['signature']} "
              f"(ancres construites à {r['signature_construction']})")
        _afficher("I — identité sur le découpage actuel", r["I_identite"])
        _afficher("II — re-découpage simulé aux frontières de phrase", r["II_redecoupage"])
        _afficher("III — contextualisation simulée", r["III_contextualisation"])
        _afficher("IV — sabotage (doit être détecté)", r["IV_sabotage"])
        echecs = [k for k, v in r.items() if isinstance(v, dict) and "ÉCHEC" in v["verdict"]]
        print(f"\n  {'TOUS LES CONTRÔLES PASSENT' if not echecs else 'ÉCHECS : ' + ', '.join(echecs)}")
        sys.exit(1 if echecs else 0)
    elif args.sensibilite:
        print(f"\n  {'cible':>6}  {'chunks':>7}  {'ok':>4} {'part.':>5} {'perdu':>5}"
              f"  {'union méd.':>10}  {'prim. méd.':>10}  {'chunks/or méd.':>14}")
        for r in sensibilite():
            print(f"  {r['cible']:>6}  {r['n_chunks_simules']:>7}  {r['ok']:>4} "
                  f"{r['partiel']:>5} {r['perdu']:>5}  {r['couverture_union_mediane']:>10.4f}"
                  f"  {r['couverture_primaire_mediane']:>10.4f}  {r['chunks_par_or_mediane']:>14}")
        print("\n  médiane servie 1 733 car. · médiane d'un chunk d'or 2 281 car.")
    elif args.appliquer:
        if not args.signature_cible:
            sys.exit("--appliquer exige --signature-cible")
        r = appliquer(pathlib.Path(args.appliquer), args.signature_cible)
        print(json.dumps(r["resume_ancres"], ensure_ascii=False, indent=1))
        pop, comp, non = r["population_origine"], r["comparables"], r["non_comparables"]
        print(f"\n  population d'origine : {pop['total']} questions "
              f"(v1 {pop['v1']} · v3 {pop['v3']}) — DÉNOMINATEUR DE LA DÉCISION")
        print(f"  comparables          : {comp['total']} (v1 {comp['v1']} · v3 {comp['v3']})")
        print(f"  non comparables      : {non['total']} — comptées 0, jamais retirées")
        for banc in ("v1", "v3"):
            if non[banc]:
                print(f"    {banc} : {', '.join(non[banc])}")
        if r["multi_ancres"]["non_comparables"]:
            print(f"  dont multi-ancres    : "
                  f"{', '.join(r['multi_ancres']['non_comparables'])}")
        print("\n  Relire chaque exclusion à la main : elle peut dire « le chantier a détruit\n"
              "  ce passage » comme « le mappeur n'a pas su le retrouver ».")
    elif args.construire_v4:
        charge = construire_v4()
        print(f"  ancres v4 écrites : {len(charge['ancres'])} chunks d'or, "
              f"{len(charge['absents'])} absent(s) du corpus")
        print(f"  signature de construction : {charge['signature_construction']}")
        print(f"  -> {ANCRES_V4.relative_to(RACINE)}")
    elif args.appliquer_v4:
        if not args.signature_cible:
            sys.exit("--appliquer-v4 exige --signature-cible")
        r = appliquer_v4(pathlib.Path(args.appliquer_v4), args.signature_cible)
        print(f"\n  ancres v4 : {r['n_ancres_v4']} — signature cible {r['signature_cible']}")
        for famille in QUESTIONS_V4:
            print(f"    -> questions-v4-{famille.replace('_', '-')}-{args.signature_cible}.jsonl")
        for famille, c in r["par_famille"].items():
            print(f"    {famille:18s} n={c['n']:3d}  ok={c.get('ok', 0):3d}  "
                  f"partiel={c.get('partiel', 0):3d}  perdu={c.get('perdu', 0):3d}"
                  + (f"  absent_du_servi={c['absent_du_servi']}" if c.get("absent_du_servi") else ""))
        if r["or_non_servi"]:
            print("\n  or non intégralement servi :")
            for q in r["or_non_servi"]:
                print(f"    {q['famille']:18s} {q['qid']:8s} {q['statut']:16s} "
                      f"couverture={q['couverture_union']:.4f}")
        else:
            print("\n  toutes les ancres v4 survivent (ok).")
    else:
        a.print_help()


if __name__ == "__main__":
    main()
