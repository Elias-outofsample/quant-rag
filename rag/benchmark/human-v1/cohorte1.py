"""Construit la cohorte 1 de `human-v1` — dix propositions **à validation humaine**.

**Ce que ce script produit n'est pas un banc.** C'est un lot de dix propositions écrites par
un modèle, dont aucune n'a été relue par une personne. Chaque item sort en `statut:
"candidat"`, chaque fait en `statut_fait: "gold_candidate"`, et le validateur refuse de les
appeler autrement tant qu'aucun `valide_par` humain n'apparaît. La cohorte est une
**calibration** : elle sert à mesurer le coût réel de l'annotation et à faire remonter les
ambiguïtés du barème, pas à produire un chiffre.

**Les ancres sont extraites, jamais retranscrites.** Chaque ancre est désignée par
`(chunk_id, phrase de début, phrase de fin)` et le texte est découpé dans le corpus lui-même.
La transcription à la main a déjà coûté une ancre au lot pilote — une espace insécable
aplatie — et une ancre qui ne se retrouve pas dans le texte canonique n'a pas d'offsets, donc
pas de citation. Ici l'égalité est vraie par construction, et `offsets.py --verifier` le
recontrôle.

**Comment les questions ont été choisies.** Pas au hasard, et pas parmi les plus faciles du
banc : chaque item vise une région où une perte a été **mesurée** — les 30 questions dont
l'or est hors du pool dense (classes `GRANULARITE` et `DECOUVERTE` du
`RAPPORT-PERTES-2026-09-07.md`), ou les questions `multi` que le contexte doublé fait
abstenir. Le champ `provenance` de chaque item nomme la sienne.

**Les questions ne réutilisent aucune question du banc v3.** Les documents sont repris, les
questions sont écrites à neuf, et l'or est un **ensemble de faits**, pas le chunk d'où la
question serait sortie. C'est ce qui sépare ce lot du biais mesuré de v3 ; le contrôle de
fuite lexicale est au §fuite de `verifier.py`.

    .venv/bin/python rag/benchmark/human-v1/cohorte1.py            # constat
    .venv/bin/python rag/benchmark/human-v1/cohorte1.py --ecrire   # écrit le JSONL

**À exécuter une fois.** Après la revue humaine, le JSONL fait foi et ce script ne doit plus
l'écraser : il refuse si le fichier existe, sauf `--forcer`.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BENCHMARK = HERE.parent
ROOT = BENCHMARK.parents[1]
sys.path.insert(0, str(BENCHMARK))
sys.path.insert(0, str(BENCHMARK.parent))
sys.path.insert(0, str(HERE))

import offsets as mod_offsets  # noqa: E402
from corpus import ChunkIndex  # noqa: E402

LOT = HERE / "items-cohorte1-v0.jsonl"
AUTEUR = "claude-opus-5 (proposition, non relue)"
CAMPAGNE = 1
DATE = "2026-09-07"
SIGNATURE = "5530cba145"


def _span(index: ChunkIndex, chunk_id: str, debut: str, fin: str) -> tuple[str, dict]:
    """Le texte exact entre deux phrases repères, bornes comprises, dans le chunk donné."""
    chunk = index.chunks.get(chunk_id)
    if chunk is None:
        sys.exit(f"chunk introuvable : {chunk_id}")
    texte = chunk["text"] or ""
    i = texte.find(debut)
    if i < 0:
        sys.exit(f"{chunk_id} : phrase de début introuvable — {debut[:60]!r}")
    j = texte.find(fin, i)
    if j < 0:
        sys.exit(f"{chunk_id} : phrase de fin introuvable — {fin[:60]!r}")
    return texte[i:j + len(fin)], chunk


def ancre(index: ChunkIndex, aid: str, role: str, chunk_id: str, debut: str, fin: str) -> dict:
    extrait, chunk = _span(index, chunk_id, debut, fin)
    return {"ancre_id": aid, "role": role, "document_id": chunk["document_id"],
            "page": chunk.get("page_start"), "texte": extrait,
            "content_type": chunk.get("content_type") or "",
            # Clé interne, jamais une citation utilisateur : elle sert au ré-ancrage et au
            # départage d'un texte présent deux fois dans le même document.
            "chunk_id_origine": chunk_id}


def ancre_de_tableau(index: ChunkIndex, documents: "mod_offsets.Documents", aid: str, role: str,
                     chunk_id: str, debut: str, fin: str) -> dict:
    """Ancre d'un tableau : le texte visé est le **HTML canonique**, pas le markdown servi.

    Défaut trouvé en construisant ce lot, et il est structurel. Les 5 914 chunks couverts par
    ``tables-markdown-v1`` sont servis dans une **conversion** : le corpus montre
    ``| Mean | 0.51 | …`` quand le texte canonique — celui où vivent les offsets — porte
    ``<table><tr><td>Mean</td><td>0.51</td>…``. Une ancre prise sur le markdown n'a donc
    aucun offset, et sans offset elle n'est pas citable (P6 du contrat).

    La sortie est double, et il le faut : ``texte`` vise la **source** et porte les offsets
    opposables ; ``rendu_servi`` porte la forme **lisible**, celle que le relecteur humain et
    l'utilisateur voient. Confondre les deux serait soit citer du HTML à un lecteur, soit
    pointer des offsets vers un texte qui n'existe pas.
    """
    _, chunk = _span(index, chunk_id, "|", "|")          # existence et métadonnées
    trouve = documents.texte(chunk["document_id"])
    if trouve is None:
        sys.exit(f"{chunk_id} : blocs du document absents, offsets impossibles")
    canonique = trouve[0]
    i = canonique.find(debut)
    j = canonique.find(fin, i) if i >= 0 else -1
    if i < 0 or j < 0:
        sys.exit(f"{chunk_id} : repères introuvables dans le texte canonique")
    return {"ancre_id": aid, "role": role, "document_id": chunk["document_id"],
            "page": chunk.get("page_start"), "texte": canonique[i:j + len(fin)],
            "content_type": chunk.get("content_type") or "",
            "chunk_id_origine": chunk_id,
            "rendu_servi": chunk["text"],
            "note_de_forme": "le texte ancré est le HTML canonique du tableau ; « rendu_servi » "
                             "est la conversion markdown que le corpus sert et que la fiche "
                             "de revue affiche"}


def fait(texte: str, ancres: list[str], exigence: str = "obligatoire") -> dict:
    """Un fait attendu **candidat**. Rien n'entre au gold sans revue humaine."""
    return {"texte": texte, "ancres": ancres, "exigence": exigence,
            "statut_fait": "gold_candidate", "propose_en": 1, "origine": "revue_corpus"}


# --------------------------------------------------------------------------- les dix

def construire(index: ChunkIndex) -> list[dict]:
    documents = mod_offsets.Documents()
    A = lambda *a: ancre(index, *a)  # noqa: E731
    T = lambda *a: ancre_de_tableau(index, documents, *a)  # noqa: E731
    items: list[dict] = []

    # ---------------------------------------------------------------- C01
    items.append({
        "id": "C01",
        "famille": "controverse_chiffree",
        "domaine": "methodologie",
        "difficulte": "moyenne",
        "situations": ["limite_methodologique", "resultat_chiffre", "deux_passages_meme_document"],
        "question": "A colleague shows me a new cross-sectional return factor with a t-statistic "
                    "of 2.8 and says it is significant. What significance threshold does the "
                    "multiple-testing literature recommend instead, and how solid is that number?",
        "intention": "Le chiffre seul (3,0) est une mauvaise réponse : il faut le cadre qui le "
                     "produit, et la réserve que les auteurs posent eux-mêmes.",
        "repondable": "oui",
        "provenance": {"type": "erreur_observee",
                       "detail": "doc-2b74a994fcfc1c9c porte l'or de v3/t03, classée DECOUVERTE "
                                 "— ni le passage ni le document ne sont dans le pool dense@50."},
        "appuis": [
            A("A1", "porte_le_fait", "chunk-c805a232549b6731",
              "We argue that a newly discovered factor today",
              "fail to exceed our recommended cutofs."),
            A("A2", "pose_la_condition", "chunk-c805a232549b6731",
              "While a t-statistic of 3.0",
              "3.0 is too low."),
            A("A3", "pose_la_condition", "chunk-d1680a7916349845",
              "There are limitations to our framework.",
              "a t-statistic of 2.0 is too low."),
        ],
        "faits_attendus": [
            fait("Le seuil recommandé est un t supérieur à 3,0, et non le seuil usuel de 2,0.", ["A1"]),
            fait("Ce seuil sort d'un cadre de tests multiples appliqué au grand nombre de facteurs "
                 "déjà testés ; il n'est pas une convention.", ["A1"]),
            fait("Les auteurs jugent que 3,0 est probablement trop bas, parce que le décompte des "
                 "facteurs testés sous-estime les essais jamais publiés.", ["A2"]),
            fait("Le seuil ne s'applique pas uniformément : un facteur issu d'une théorie mérite "
                 "une barre plus basse qu'un facteur purement empirique.", ["A3"], "souhaitable"),
        ],
        "conditions": [
            {"texte": "La réponse ne doit pas présenter 3,0 comme un seuil universel et définitif : "
                      "les auteurs disent dans la même page qu'il est probablement trop bas et "
                      "qu'il ne convient pas à tous les facteurs.",
             "ancres": ["A2", "A3"], "consequence_si_omis": "reponse_incomplete"},
        ],
        "citation_minimale": {"documents_distincts": 1,
                              "justification": "un seul papier porte le seuil ; c'est sa réserve "
                                               "qui doit être citée, pas seulement son chiffre"},
        "pieges": ["Donner 3,0 sans la réserve, ou l'attribuer à une convention générale.",
                   "Confondre le seuil recommandé avec le t observé de 2,57 de Fama-MacBeth cité "
                   "en introduction."],
        "risques_ambiguite": ["« Solide » est vague : le relecteur doit décider si une réponse qui "
                              "donne 3,0 et la seule réserve « trop bas » suffit, ou s'il faut "
                              "aussi la nuance théorie/empirique."],
    })

    # ---------------------------------------------------------------- C02
    items.append({
        "id": "C02",
        "famille": "condition_de_validite",
        "domaine": "volatilite",
        "difficulte": "difficile",
        "situations": ["restriction_de_modele"],
        "question": "In the short-time implied-volatility model built on indexes, why do the authors "
                    "work with a market index rather than a single stock? What breaks if the same "
                    "construction is applied to one stock?",
        "intention": "La restriction est explicite et motivée dans le texte ; une réponse qui la "
                     "présente comme un choix de commodité rate le point.",
        "repondable": "oui",
        "provenance": {"type": "erreur_observee",
                       "detail": "doc-a9780e92754b1b75 porte l'or de v3/s05, classée GRANULARITE "
                                 "— le document est dans le pool, le passage non."},
        "appuis": [
            A("A1", "porte_le_fait", "chunk-ce305c961c431b21",
              "To produce the stable power-like term structure",
              "instead of an individual stock."),
            A("A2", "corrobore", "chunk-ce305c961c431b21",
              "Note that in reality, European options",
              "focus on index options."),
        ],
        "faits_attendus": [
            fait("Pour produire une structure par termes en loi de puissance stable, il faudrait "
                 "que le sous-jacent reste proche du niveau R en permanence — ce qui est irréaliste "
                 "pour une action.", ["A1"]),
            fait("Les auteurs choisissent l'autre option : introduire davantage de discontinuités "
                 "dans la fonction de volatilité locale, en travaillant sur des indices.", ["A1"]),
            fait("La justification empirique est que les options européennes portent le plus souvent "
                 "sur des indices, et que les études du blow-up du skew portent sur des options "
                 "d'indice.", ["A2"], "souhaitable"),
        ],
        "conditions": [],
        "citation_minimale": {"documents_distincts": 1,
                              "justification": "la restriction est propre à ce papier"},
        "pieges": ["Répondre « parce que les indices sont plus liquides » — le texte ne dit pas cela.",
                   "Présenter le choix comme une simplification technique plutôt que comme la "
                   "condition sous laquelle le phénomène visé peut exister."],
        "risques_ambiguite": ["Le texte est mathématique et cite Pigato (2019) et Fukasawa (2021) ; "
                              "le relecteur doit décider si citer ces travaux est exigible."],
    })

    # ---------------------------------------------------------------- C03
    items.append({
        "id": "C03",
        "famille": "factuelle_localisee",
        "domaine": "backtest",
        "difficulte": "moyenne",
        "situations": ["calibration_estimation_fenetre", "deux_passages_meme_document"],
        "question": "I have a backtested Sharpe ratio computed on a short, skewed return series, and "
                    "I tried many strategy variants before keeping this one. What corrections does "
                    "the literature offer, and what does each one correct for?",
        "intention": "Deux instruments distincts, deux corrections distinctes. Une réponse qui les "
                     "confond, ou qui n'en donne qu'un, est incomplète.",
        "repondable": "oui",
        "provenance": {"type": "besoin_metier",
                       "detail": "question de pratique courante ; le document est celui de la "
                                 "classe DECOUVERTE de v3/m16."},
        "appuis": [
            A("A1", "porte_le_fait", "chunk-ddc18dc202962588",
              "The probabilistic Sharpe ratio (PSR) provides",
              "greater than a hypothetical $S R ^ { * }$ ."),
            A("A2", "porte_le_fait", "chunk-4b4555bae7fffcc8",
              "The deflated Sharpe ratio (DSR) is a PSR",
              "is no longer user-defined."),
        ],
        "faits_attendus": [
            fait("Le Sharpe probabiliste (PSR) corrige l'effet inflationniste d'une série courte "
                 "et de rendements asymétriques ou à queues épaisses.", ["A1"]),
            fait("Le PSR estime la probabilité que le Sharpe observé dépasse un Sharpe de "
                 "référence choisi par l'utilisateur.", ["A1"]),
            fait("Le Sharpe dégonflé (DSR) est un PSR dont le seuil de rejet est ajusté pour tenir "
                 "compte du nombre d'essais, et dont le Sharpe de référence n'est plus choisi par "
                 "l'utilisateur mais estimé.", ["A2"]),
        ],
        "conditions": [
            {"texte": "La réponse doit distinguer les deux corrections : la longueur et la forme de "
                      "la série d'un côté, la multiplicité des essais de l'autre.",
             "ancres": ["A1", "A2"], "consequence_si_omis": "reponse_incomplete"},
        ],
        "citation_minimale": {"documents_distincts": 1,
                              "justification": "les deux instruments viennent du même ouvrage, "
                                               "à 2 pages d'écart"},
        "pieges": ["Donner le DSR seul en laissant croire qu'il corrige aussi la forme de la série.",
                   "Réciter la formule sans dire ce que chaque correction corrige."],
        "risques_ambiguite": ["Le relecteur doit décider si la formule de SR* est exigible ou si la "
                              "description qualitative suffit."],
    })

    # ---------------------------------------------------------------- C04
    items.append({
        "id": "C04",
        "famille": "factuelle_localisee",
        "domaine": "backtest",
        "difficulte": "moyenne",
        "situations": ["biais_de_backtest_leakage", "deux_passages_meme_document"],
        "question": "My k-fold cross-validation on a trading model gives excellent results. What "
                    "specific mechanism makes that number untrustworthy on financial data, and what "
                    "two operations are prescribed to fix it? What does each one remove?",
        "intention": "Purge et embargo sont deux opérations distinctes qui traitent deux fuites "
                     "distinctes. Les fusionner est l'erreur exacte que la question vise.",
        "repondable": "oui",
        "provenance": {"type": "erreur_observee",
                       "detail": "doc-417886529d967e20 porte l'un des ors de v3/m16, classée "
                                 "DECOUVERTE. Question multi-passages, la classe que le contexte "
                                 "doublé fait abstenir."},
        "appuis": [
            A("A1", "porte_le_fait", "chunk-e656c65b19842505",
              "Leakage takes place when the training set contains information",
              "By placing t and $t + 1$ in different sets, information is leaked."),
            A("A2", "porte_le_fait", "chunk-d8deca68d3f98bcf",
              "One way to reduce leakage is to purge",
              "I call this process “embargo.”"),
            A("A3", "porte_le_fait", "chunk-8643b8923cac5ee2",
              "For those cases where purging is not able to prevent all leakage",
              "an embargo on training observations after every test set."),
        ],
        "faits_attendus": [
            fait("La fuite vient du fait que les observations financières ne sont pas i.i.d. : une "
                 "caractéristique auto-corrélée et des étiquettes construites sur des fenêtres qui "
                 "se chevauchent font que placer t et t+1 dans deux ensembles différents transmet "
                 "de l'information.", ["A1"]),
            fait("La purge retire de l'ensemble d'entraînement toute observation dont l'étiquette "
                 "recouvre dans le temps une étiquette de l'ensemble de test.", ["A2"]),
            fait("L'embargo retire en plus les observations qui suivent immédiatement un ensemble "
                 "de test, pour la part de fuite que la purge ne supprime pas.", ["A2", "A3"]),
        ],
        "conditions": [
            {"texte": "La réponse doit dire que l'embargo ne porte que sur ce qui suit le test, "
                      "pas sur ce qui le précède.",
             "ancres": ["A3"], "consequence_si_omis": "reponse_incomplete"},
        ],
        "citation_minimale": {"documents_distincts": 1,
                              "justification": "trois passages du même chapitre ; la citation doit "
                                               "distinguer la purge de l'embargo"},
        "pieges": ["Traiter purge et embargo comme un seul remède.",
                   "Attribuer la fuite à la seule sélection de modèle (le second échec du k-fold), "
                   "qui n'est pas celui que purge et embargo traitent."],
        "risques_ambiguite": ["Le texte donne aussi une seconde cause d'échec du k-fold — le "
                              "réemploi du jeu de test. Le relecteur doit décider si l'omettre "
                              "rend la réponse incomplète."],
    })

    # ---------------------------------------------------------------- C05
    items.append({
        "id": "C05",
        "famille": "tableau",
        "domaine": "risque",
        "difficulte": "facile",
        "situations": ["resultat_chiffre_tableau_legende", "deux_passages_meme_document"],
        "question": "For the factors of the Fama-French five-factor model plus momentum, what are the "
                    "average monthly returns and t-statistics, over which sample period, and which "
                    "factor is the least significant?",
        "intention": "Les nombres sont dans un tableau, la période est dans sa légende. Une réponse "
                     "qui donne les chiffres sans la période n'est pas utilisable.",
        "repondable": "oui",
        "provenance": {"type": "erreur_observee",
                       "detail": "doc-8ea1f86a8ee1c84b porte l'or de v3/t07, classée GRANULARITE : "
                                 "le document est dans le pool, le passage du tableau non."},
        "appuis": [
            T("A1", "porte_le_fait", "chunk-5ea0681947ca79cb",
              "<tr><td>Mean</td><td>0.51</td>",
              "<td>2.83</td><td>2.20</td><td>3.15</td><td>2.88</td><td>4.04</td><td>4.05</td></tr>"),
            A("A2", "porte_le_fait", "chunk-8b0d656af519df96",
              "Averages, standard deviations, and t-statistics for monthly factor returns",
              "minus the one-month Treasury bill rate."),
        ],
        "faits_attendus": [
            fait("Les rendements mensuels moyens sont : marché 0,51 ; SMB 0,27 ; HML 0,36 ; "
                 "RMW 0,25 ; CMA 0,32 ; MOM 0,69.", ["A1"]),
            fait("Les t-statistiques sont : marché 2,83 ; SMB 2,20 ; HML 3,15 ; RMW 2,88 ; "
                 "CMA 4,04 ; MOM 4,05.", ["A1"]),
            fait("SMB est le moins significatif des six, avec t = 2,20.", ["A1"]),
            fait("La période est juillet 1963 à décembre 2014, soit 618 mois.", ["A2"]),
        ],
        "conditions": [
            {"texte": "La réponse doit donner la période. Des rendements de facteurs sans fenêtre "
                      "d'estimation ne peuvent pas être comparés à quoi que ce soit.",
             "ancres": ["A2"], "consequence_si_omis": "reponse_incomplete"},
        ],
        "citation_minimale": {"documents_distincts": 1,
                              "justification": "le tableau et sa légende sont deux passages "
                                               "distincts, et la citation doit porter les deux"},
        "pieges": ["Donner les chiffres sans la période, ou inventer une période plausible.",
                   "Confondre l'écart-type (ligne du milieu) avec la t-statistique."],
        "risques_ambiguite": ["Les valeurs sont en pourcentage mensuel ; le tableau ne le dit pas "
                              "dans la ligne extraite. Le relecteur doit décider si une réponse "
                              "sans unité est acceptable."],
    })

    # ---------------------------------------------------------------- C06
    items.append({
        "id": "C06",
        "famille": "condition_de_validite",
        "domaine": "options",
        "difficulte": "moyenne",
        "situations": ["deux_passages_meme_document", "robustesse"],
        "question": "Can the Kelly criterion be applied as-is to size a systematic put-writing "
                    "strategy? What assumption of the criterion is violated, and what does this "
                    "paper propose instead?",
        "intention": "L'hypothèse de Kelly — perte totale de la mise — ne décrit pas une vente de "
                     "put. Le papier propose autre chose, et les deux passages sont éloignés.",
        "repondable": "oui",
        "provenance": {"type": "erreur_observee",
                       "detail": "doc-81aad1db72edf724 porte l'or de v3/s52, classée GRANULARITE."},
        "appuis": [
            A("A1", "porte_le_fait", "chunk-376cab89375cb667",
              "The Kelly criterion (Kelly Jr., 1956) provides",
              "the wagered amount is entirely forfeited."),
            A("A2", "porte_le_fait", "chunk-e686abd8ca797d9d",
              "First, it introduces a novel hybrid sizing methodology",
              "while maintaining attractive growth prospects."),
        ],
        "faits_attendus": [
            fait("Le critère de Kelly maximise le taux de croissance du capital en maximisant "
                 "l'espérance du logarithme de la richesse finale.", ["A1"]),
            fait("Sa formulation suppose que, en cas de perte, la mise est intégralement perdue — "
                 "ce qui ne décrit pas le profil de perte d'une vente de put.", ["A1"]),
            fait("Le papier propose à la place une méthode hybride combinant des estimations "
                 "prospectives par simulation de Monte-Carlo et des signaux de régime de "
                 "volatilité en temps réel.", ["A2"]),
        ],
        "conditions": [
            {"texte": "La réponse doit nommer l'hypothèse violée, pas seulement dire que Kelly est "
                      "« trop agressif ».",
             "ancres": ["A1"], "consequence_si_omis": "reponse_incomplete"},
        ],
        "citation_minimale": {"documents_distincts": 1,
                              "justification": "l'hypothèse est p. 4, la proposition p. 1 ; les "
                                               "deux doivent être citées"},
        "pieges": ["Répondre par la règle empirique du demi-Kelly, qui n'est pas ce que dit le texte.",
                   "Décrire la méthode hybride sans dire ce qu'elle remplace."],
        "risques_ambiguite": ["Le texte ne dit pas explicitement « donc Kelly ne s'applique pas » ; "
                              "le lien entre l'hypothèse et la vente de put est une inférence. Le "
                              "relecteur doit trancher si elle est exigible ou seulement acceptable."],
    })

    # ---------------------------------------------------------------- C07
    items.append({
        "id": "C07",
        "famille": "comparaison_multi_documents",
        "domaine": "microstructure",
        "difficulte": "difficile",
        "situations": ["plusieurs_documents"],
        "question": "Dealer gamma hedging is said to set a volatility regime. How does the sign of "
                    "net dealer gamma translate into market behaviour, and what does the standard "
                    "way of modelling this feedback leave out?",
        "intention": "Le mécanisme est dans un document, la limite du modèle usuel dans un autre. "
                     "Une réponse mono-source ne peut pas porter les deux.",
        "repondable": "oui",
        "provenance": {"type": "erreur_observee",
                       "detail": "doc-337ca248f5cdb0bc porte l'un des ors de v3/m20, classée "
                                 "GRANULARITE ; doc-b7363e426f9ed640 porte celui de v3/s44, "
                                 "classée DECOUVERTE."},
        "appuis": [
            A("A1", "porte_le_fait", "chunk-311dad0d9d6e6739",
              "A dealer who is long gamma sees her book",
              "It only says whether moves get absorbed or extended."),
            A("A2", "porte_le_fait", "chunk-41c6fefe20b0dc15",
              "Prior research has established that dealer hedging",
              "yet most formulations treat feedback as approximately linear and assume constant sensitivity."),
        ],
        "faits_attendus": [
            fait("Un dealer long gamma achète les baisses et vend les hausses : le flux est "
                 "stabilisant. Court gamma, il fait l'inverse et amplifie le mouvement en cours.", ["A1"]),
            fait("Le signe du gamma agrégé définit donc un régime de volatilité — amortissement "
                 "ou amplification — et ne dit rien de la direction du marché.", ["A1"]),
            fait("La plupart des formulations traitent cette rétroaction comme approximativement "
                 "linéaire et supposent une sensibilité constante.", ["A2"]),
        ],
        "conditions": [
            {"texte": "La réponse ne doit pas présenter un gamma positif comme haussier ni un gamma "
                      "négatif comme baissier : le texte dit explicitement que le régime ne dit "
                      "rien de la direction.",
             "ancres": ["A1"], "consequence_si_omis": "reponse_fausse"},
        ],
        "citation_minimale": {"documents_distincts": 2,
                              "justification": "le mécanisme et la limite du modèle usuel viennent "
                                               "de deux papiers différents"},
        "pieges": ["Traiter le signe du gamma comme un signal directionnel.",
                   "Donner la limite du modèle linéaire sans source, ou l'attribuer au papier GEX."],
        "risques_ambiguite": ["« Ce que le modèle usuel laisse de côté » peut se lire comme la "
                              "non-linéarité, ou comme la dépendance au bêta annoncée par le titre "
                              "du second papier. Le relecteur doit fixer laquelle est exigible."],
    })

    # ---------------------------------------------------------------- C08
    items.append({
        "id": "C08",
        "famille": "comparaison_multi_documents",
        "domaine": "methodologie",
        "difficulte": "difficile",
        "situations": ["divergence_entre_sources", "plusieurs_documents"],
        "question": "Two papers in this corpus both argue that published anomaly returns are "
                    "overstated. Do they say the same thing? Where do they differ, and what does "
                    "that difference change for someone deciding whether to trade an anomaly?",
        "intention": "Les deux convergent sur le constat et divergent sur le mécanisme et le "
                     "remède. Une réponse qui les fond en un seul argument rate l'item.",
        "repondable": "oui",
        "provenance": {"type": "hypothese_de_test",
                       "detail": "construit pour tester la divergence entre sources ; les deux "
                                 "documents sont respectivement DECOUVERTE (v3/t03) et "
                                 "DECOUVERTE (v3/d01)."},
        "appuis": [
            A("A1", "porte_le_fait", "chunk-c805a232549b6731",
              "We argue that a newly discovered factor today",
              "Many published factors fail to exceed our recommended cutofs."),
            A("A2", "conteste", "chunk-d6f0ad74cb4104a4",
              "The literature largely ignores trading costs",
              "the data from earlier decades are not representative of the future."),
            A("A3", "conteste", "chunk-9804efd2694e1d13",
              "We find that the distribution oft-stats",
              "consistent with our out-of-sample tests."),
        ],
        "faits_attendus": [
            fait("Le premier papier attribue la surestimation aux tests multiples et y répond en "
                 "relevant la barre statistique : un t supérieur à 3,0.", ["A1"]),
            fait("Le second attribue la surestimation aux coûts de transaction ignorés et à "
                 "l'ancienneté des données, et non d'abord à un problème de seuil.", ["A2"]),
            fait("Le second mesure que, sur les rendements nets post-publication et post-2005, "
                 "seuls 7 % des t dépassent 2,0 en valeur absolue, et que l'anomalie au 90ᵉ "
                 "centile a une espérance de rendement d'environ 10 points de base par mois.", ["A3"]),
            fait("Pour un praticien, la différence est de nature : relever le seuil filtre les "
                 "découvertes futures, tandis que les coûts et l'ancienneté des données réduisent "
                 "l'espérance de gain des anomalies déjà publiées.", ["A1", "A2", "A3"], "souhaitable"),
        ],
        "conditions": [
            {"texte": "La réponse ne doit pas présenter les deux papiers comme disant la même chose "
                      "sous deux formes : leurs mécanismes et leurs remèdes diffèrent.",
             "ancres": ["A1", "A2"], "consequence_si_omis": "reponse_fausse"},
        ],
        "citation_minimale": {"documents_distincts": 2,
                              "justification": "une divergence ne peut pas être attestée par une "
                                               "seule source"},
        "pieges": ["Fondre les deux en « les anomalies ne survivent pas hors échantillon ».",
                   "Attribuer le chiffre de 7 % ou les 10 bps au papier sur les tests multiples."],
        "risques_ambiguite": ["Les deux papiers ne sont pas strictement contradictoires — ils sont "
                              "compatibles. Le relecteur doit décider si « divergence » exige de "
                              "dire qu'ils ne s'opposent pas."],
    })

    # ---------------------------------------------------------------- C09
    items.append({
        "id": "C09",
        "famille": "partiellement_couvert",
        "domaine": "microstructure",
        "difficulte": "moyenne",
        "situations": ["reponse_partielle_attendue"],
        "question": "What is the zero-gamma flip level and how should a crossing of it be read? "
                    "Also give me the exact formula this engine uses to compute per-strike dealer "
                    "gamma, and the input data it requires.",
        "intention": "La première moitié est couverte, la seconde ne l'est pas. La bonne réponse "
                     "traite l'une et décline explicitement l'autre — c'est la règle 3 du prompt "
                     "de réponse v2, et elle n'a jamais été éprouvée sur un cas construit pour ça.",
        "repondable": "partiel",
        "provenance": {"type": "hypothese_de_test",
                       "detail": "doc-b7363e426f9ed640 n'est présent dans le corpus qu'à hauteur "
                                 "de 4 chunks / 2 pages : les sections conceptuelles y sont, "
                                 "l'implémentation non. Couverture partielle vérifiée, non supposée."},
        "appuis": [
            A("A1", "porte_le_fait", "chunk-288e60ef258dfcea",
              "Because per-strike gamma depends on where spot sits",
              "rather than an ordinary support/resistance touch."),
            A("A2", "corrobore", "chunk-311dad0d9d6e6739",
              "Aggregated across all dealers and all strikes",
              "It only says whether moves get absorbed or extended."),
        ],
        "faits_attendus": [
            fait("Le niveau de bascule (zero-gamma ou flip) est le prix auquel le gamma net des "
                 "dealers change de signe ; il sépare le régime d'amortissement du régime "
                 "d'amplification.", ["A1"]),
            fait("Un franchissement décisif de ce niveau est un changement de caractère du marché, "
                 "et non un simple contact de support ou de résistance.", ["A1"]),
            fait("Au-dessus du niveau, le régime amortit ; en dessous, il amplifie.", ["A1", "A2"]),
        ],
        "conditions": [
            {"texte": "La réponse doit dire explicitement que les passages ne contiennent ni la "
                      "formule de calcul du gamma par strike, ni la liste des données d'entrée. "
                      "Une réponse qui invente une formule, même correcte par ailleurs, est fausse.",
             "ancres": ["A1"], "consequence_si_omis": "reponse_fausse"},
        ],
        "couverture_partielle": {
            "couvert": "le niveau de bascule, sa lecture, les deux régimes",
            "non_couvert": "la formule de gamma par strike et les données d'entrée du moteur",
            # Page corrigée après la revue : les 4 chunks couvrent les pages 2-3 pour un lecteur
            # (la page 1 est du frontmatter, non ingérée). L'erreur venait du `page_start`
            # 0-basé, celle-là même que le correctif de citation traite.
            "preuve": "le document ne compte que 4 chunks dans le corpus, couvrant les "
                      "pages 2-3 pour un lecteur : introduction, économie du hedging, "
                      "niveau de bascule, flux de second ordre. La section de méthode "
                      "n'est pas ingérée, et aucune formule de gamma par strike n'existe "
                      "ailleurs dans le corpus.",
        },
        "citation_minimale": {"documents_distincts": 1,
                              "justification": "un seul document couvre la partie répondable"},
        "pieges": ["Produire une formule de gamma plausible tirée de connaissances générales.",
                   "S'abstenir sur l'ensemble alors que la première moitié est couverte."],
        "risques_ambiguite": ["Le barème doit dire comment noter une réponse partielle : le §4.6 du "
                              "README compte les faits, mais rien ne dit encore si l'aveu explicite "
                              "de la partie manquante vaut un point ou seulement évite une pénalité."],
    })

    # ---------------------------------------------------------------- C10
    items.append({
        "id": "C10",
        "famille": "hors_corpus",
        "domaine": "methodologie",
        "difficulte": "moyenne",
        "situations": ["abstention_attendue"],
        "question": "What net-of-fees annualised return did Renaissance Technologies' Medallion fund "
                    "achieve, and over what period?",
        "intention": "Question très plausible pour un quant, et le corpus n'en porte rien. Le "
                     "voisinage est tentant : le corpus est plein de ratios de Sharpe et de "
                     "rendements de stratégies.",
        "repondable": "non",
        "provenance": {"type": "hypothese_de_test",
                       "detail": "construit comme cas d'abstention ; absence vérifiée par balayage "
                                 "exhaustif, pas supposée."},
        "appuis": [],
        "faits_attendus": [],
        "conditions": [],
        "abstention_attendue": "La réponse doit refuser, en nommant ce qui manque : le corpus ne "
                               "porte aucune donnée de performance de fonds. Une réponse qui cite "
                               "la seule mention de Renaissance Technologies présente dans le "
                               "corpus — sa fondation en 1982 par James Simons — pour dire qu'elle "
                               "ne contient pas de rendement est **meilleure** qu'un refus nu.",
        "absence": {
            "porte": "le rendement net annualisé du fonds Medallion, sur toute période",
            "preuve": "balayage sémantique des 26 120 passages du corpus (signature 5530cba145), "
                      "variantes ci-dessous. La seule mention de la firme donne sa fondation en "
                      "1982 et aucun chiffre de performance.",
            # Une absence ne se prouve pas par une chaîne de caractères : « SEC Rule 605 » est
            # absent à 0 occurrence et pourtant répondable, le corpus portant les règles
            # 11Ac1-5 et 11Ac1-6 renumérotées depuis. D'où les variantes, et leur lecture.
            "recherche_semantique": [
                {"variante": "medallion", "occurrences": 0},
                {"variante": "jim simons", "occurrences": 0},
                {"variante": "james simons", "occurrences": 1,
                 "lecture": "« Finding Alphas » p. 26 — fondation de Renaissance Technologies "
                            "en 1982, aucun chiffre de performance"},
                {"variante": "renaissance technolog", "occurrences": 1,
                 "lecture": "même passage"},
                {"variante": "hedge fund returns net of fees", "occurrences": 0},
                {"variante": "fund-level performance", "occurrences": 0},
            ],
            "force_de_la_preuve": "forte",
            "voisinage_tentant": "le corpus contient de nombreux ratios de Sharpe, rendements de "
                                 "stratégies et discussions de performance : le retrieval rendra "
                                 "des passages qui ressemblent à une réponse.",
        },
        "citation_minimale": {"documents_distincts": 0,
                              "justification": "aucune citation n'est attendue pour un refus ; en "
                                               "citer une serait un défaut"},
        "pieges": ["Répondre de mémoire — le chiffre est célèbre et le modèle le connaît "
                   "probablement. C'est exactement ce que la règle 1 du prompt interdit.",
                   "Citer un ratio de Sharpe d'un autre papier comme s'il concernait Medallion."],
        "risques_ambiguite": ["Le relecteur doit décider si mentionner la fondation de 1982 est "
                              "un bonus ou un hors-sujet."],
    })

    for item in items:
        item.update({"statut": "candidat", "auteur": AUTEUR, "date": DATE,
                     "methode": "document_dabord_question_ecrite_a_neuf",
                     "corpus_signature": SIGNATURE,
                     "gold_version": 1, "version_campagne": CAMPAGNE,
                     "relecture_humaine": None})
    return items


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--ecrire", action="store_true")
    parser.add_argument("--forcer", action="store_true", help="écraser un lot existant")
    args = parser.parse_args()

    index = ChunkIndex.load(verbose=False)
    items = construire(index)
    items, rapport = mod_offsets.enrichir(items)
    compte = rapport["compte"]
    print(f"{len(items)} items · {compte['ancres']} ancres · {compte['avec_offsets']} avec offsets "
          f"· {compte['exactes']} exactes au caractère · {compte['ambigues']} ambiguës "
          f"· {compte['refusees']} refusées")
    for ligne in rapport["refus"]:
        print(f"  REFUS  {ligne}")
    for item in items:
        print(f"  {item['id']}  {item['famille']:<26} {item['repondable']:<8} "
              f"{len(item['appuis'])} ancres  {len(item['faits_attendus'])} faits  "
              f"{','.join(item['situations'])}")
    if not args.ecrire:
        print("\nConstat seul — relance avec --ecrire.")
        return
    if LOT.exists() and not args.forcer:
        sys.exit(f"\n{LOT.name} existe déjà. Après la revue humaine il fait foi et ce script ne "
                 "doit plus l'écraser : --forcer pour passer outre.")
    LOT.write_text("\n".join(json.dumps(i, ensure_ascii=False) for i in items) + "\n",
                   encoding="utf-8")
    print(f"\n  écrit  {LOT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
