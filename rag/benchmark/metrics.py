"""Métriques de retrieval — la moitié du banc d'essai qui n'appelle aucun LLM.

Séparation délibérée. Si le score de retrieval dépendait d'un juge LLM, une mise
à jour silencieuse de ``mistral-medium`` invaliderait rétroactivement toute la
ligne de base, sans que rien ne change dans le dépôt. Ici, ``recall@k``, ``MRR``
et ``nDCG@10`` ne dépendent que du corpus et du chunk d'or fixé à la génération :
même index, même chiffre, dans six mois comme aujourd'hui.
"""
from __future__ import annotations

import math
import random
import statistics

CUTOFFS = (1, 3, 5, 10)

#: Gains de pertinence pour le nDCG.
#: 3 — un chunk d'or, celui dont la question a été tirée ;
#: 1 — un autre chunk du bon document : souvent acceptable, une réponse pouvant
#:     chevaucher une frontière de découpage ;
#: 0 — le reste.
GAIN_GOLD, GAIN_SAME_DOCUMENT = 3.0, 1.0

#: Chunks à gain 1 crédités par document d'or. Le gain 1 est un crédit de
#: *substitution*, pas d'*addition* : il récompense le système qui a rendu le
#: passage voisin au lieu du bon — une réponse peut chevaucher une frontière de
#: découpage — et non celui qui empile dix passages du même document.
#:
#: D'où deux conséquences dans le calcul, l'une et l'autre délibérées :
#:   - le plafond est à 1, donc empiler ne rapporte rien de plus qu'approcher ;
#:   - le classement idéal ne contient **que** les chunks d'or. Réserver dans
#:     l'idéal des places pour des voisins rendrait le plafond de 1,0 inatteignable
#:     par un système qui fait exactement ce qu'on lui demande : rendre le bon
#:     passage, et rien de redondant. Un score que le comportement correct ne peut
#:     pas atteindre n'est pas une métrique, c'est un piège.
NEIGHBOUR_CREDITS = 1


def rank_of(rows: list[dict], key: str, wanted: set) -> int | None:
    """Rang (1-indexé) du premier élément appartenant à l'ensemble visé."""
    return next((i for i, row in enumerate(rows, 1) if row.get(key) in wanted), None)


def ranks_of_all(rows: list[dict], key: str, wanted: set) -> dict:
    """Rang de chaque cible ; ``None`` si absente du classement."""
    found = {}
    for position, row in enumerate(rows, 1):
        value = row.get(key)
        if value in wanted and value not in found:
            found[value] = position
    return {value: found.get(value) for value in wanted}


#: Gold **pondéré** — ajouté le 6 septembre 2026, strictement optionnel.
#:
#: Il n'existe que pour une raison : un re-découpage peut couper un passage d'or en
#: plusieurs morceaux. Sans poids, ``ideal = [GAIN_GOLD] * len(gold_chunks)`` ferait
#: passer l'idéal de 3,00 à 4,89 pour un or coupé en deux, et un système rendant le bon
#: passage au rang 1 tomberait de 1,00 à 0,61 **sans qu'aucune qualité ait bougé**.
#:
#: Le gain d'un fragment d'or **interpole entre le crédit de substitution et le gain d'or
#: plein** :
#:
#:     gain(p) = GAIN_SAME_DOCUMENT + (GAIN_GOLD - GAIN_SAME_DOCUMENT) * p
#:
#: Ce n'est pas ``GAIN_GOLD * p``, et la différence est un défaut réel corrigé le
#: 6 septembre 2026 après relecture adverse. Avec ``3p``, un fragment portant moins d'un
#: tiers de la réponse valait MOINS que le crédit de substitution de 1,0 accordé à un
#: chunk quelconque du bon document : un système qui rend un vrai morceau de la réponse
#: scorait sous un système qui rend n'importe quoi d'autre du même document. Mesuré sur
#: le contrôle II à 1 200 caractères : **303 des 517 membres d'union (58,6 %) sous 1/3** —
#: le régime dominant, pas un cas limite.
#:
#: L'interpolation supprime le croisement : elle est monotone en p, vaut exactement
#: ``GAIN_GOLD`` en p=1, et tend vers ``GAIN_SAME_DOCUMENT`` quand p tend vers 0 — ce qui
#: est la bonne limite : un fragment qui ne porte rien vaut ce que vaut un voisin.
#: Un or non coupé (``{chunk: 1.0}``) rend donc exactement les chiffres d'avant.
#:
#: ``poids=None`` — le défaut, et le seul chemin qu'emprunte la ligne de base commitée —
#: laisse le calcul historique inchangé, au bit près (``test_metrics_poids.py``).


def gain_pondere(part: float) -> float:
    """Gain d'un fragment d'or portant ``part`` de la réponse. Voir la note ci-dessus."""
    return GAIN_SAME_DOCUMENT + (GAIN_GOLD - GAIN_SAME_DOCUMENT) * part


def gains(rows: list[dict], gold_chunks: set, gold_documents: set,
          poids: dict | None = None) -> list[float]:
    credited: dict[str, int] = {}
    out = []
    for row in rows:
        if row.get("chunk_id") in gold_chunks:
            out.append(gain_pondere(poids[row["chunk_id"]]) if poids else GAIN_GOLD)
            continue
        document = row.get("document_id")
        if document in gold_documents and credited.get(document, 0) < NEIGHBOUR_CREDITS:
            credited[document] = credited.get(document, 0) + 1
            out.append(GAIN_SAME_DOCUMENT)
            continue
        out.append(0.0)
    return out


def _dcg(values: list[float]) -> float:
    return sum(gain / math.log2(position + 1) for position, gain in enumerate(values, 1))


def ndcg(rows: list[dict], gold_chunks: set, gold_documents: set, cutoff: int = 10,
         poids: dict | None = None) -> float:
    observed = gains(rows, gold_chunks, gold_documents, poids)[:cutoff]
    if poids:
        ideal = sorted((gain_pondere(poids[c]) for c in gold_chunks),
                       reverse=True)[:cutoff]
    else:
        ideal = ([GAIN_GOLD] * len(gold_chunks))[:cutoff]
    reference = _dcg(ideal)
    if not reference:
        return 0.0
    # Écrêtage : les crédits de substitution peuvent, en théorie, faire dépasser
    # l'idéal (voisins bien classés *et* chunk d'or trouvé). Le score reste borné.
    return min(_dcg(observed) / reference, 1.0)


def ndcg_documents(rows: list[dict], gold_documents: set, cutoff: int = 10) -> float:
    """nDCG au niveau document.

    Il lui faut sa propre fonction : au niveau chunk le gain 3 récompense un
    identifiant précis, ici il récompense la *première* apparition de chaque bon
    document — les suivantes valent 1, et sont plafonnées comme au niveau chunk.
    Réutiliser ``ndcg`` en lui passant des identifiants de document là où il
    attend des chunks donnerait un score qui ne peut jamais atteindre 1.
    """
    seen: dict[str, int] = {}
    observed = []
    for row in rows[:cutoff]:
        document = row.get("document_id")
        if document not in gold_documents:
            observed.append(0.0)
            continue
        rank = seen.get(document, 0)
        seen[document] = rank + 1
        observed.append(GAIN_GOLD if rank == 0 else 0.0)
    ideal = ([GAIN_GOLD] * len(gold_documents))[:cutoff]
    reference = _dcg(ideal)
    return min(_dcg(observed) / reference, 1.0) if reference else 0.0


def summarise(records: list[dict]) -> dict:
    """Agrège les mesures par question en un tableau de configuration.

    ``records`` : un dict par question, avec ``first_rank`` (rang de la première
    cible), ``all_found_at`` (rang auquel *toutes* les cibles sont réunies, pour
    les questions multi-documents) et ``ndcg``.
    """
    if not records:
        return {}
    total = len(records)
    firsts = [r["first_rank"] for r in records]
    hits = [r for r in firsts if r is not None]
    out = {f"recall@{k}": round(sum(1 for r in hits if r <= k) / total, 3) for k in CUTOFFS}
    out["MRR"] = round(sum(1 / r for r in hits) / total, 3)
    out["nDCG@10"] = round(statistics.mean(r["ndcg"] for r in records), 3)
    out["misses"] = total - len(hits)
    out["n"] = total
    completes = [r.get("all_found_at") for r in records]
    if any(r.get("gold_count", 1) > 1 for r in records):
        out["all_gold@10"] = round(sum(1 for r in completes if r is not None and r <= 10) / total, 3)
    return out


def bootstrap_ci(values: list[float], draws: int = 4000, level: float = 0.95,
                 seed: int = 20260901) -> tuple[float, float]:
    """Intervalle de confiance par bootstrap sur la moyenne.

    Avec 50 questions, un écart de 0,04 entre deux configurations n'a aucune
    raison d'être un résultat. Publier une moyenne nue à trois décimales sur un
    échantillon de cette taille, c'est inviter à sur-interpréter du bruit
    d'échantillonnage — l'intervalle est là pour l'interdire.
    """
    if not values:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    n = len(values)
    means = sorted(statistics.mean(rng.choices(values, k=n)) for _ in range(draws))
    low = means[int((1 - level) / 2 * draws)]
    high = means[int((1 + level) / 2 * draws) - 1]
    return (round(low, 3), round(high, 3))


def paired_delta(reference: list[float], variant: list[float], draws: int = 4000,
                 seed: int = 20260901) -> dict:
    """Écart apparié entre deux configurations, avec son intervalle.

    Apparié : les deux configurations voient les mêmes questions, donc c'est la
    différence question par question qu'il faut rééchantillonner, pas les deux
    moyennes séparément. L'intervalle est bien plus serré, et c'est le bon test.
    """
    if len(reference) != len(variant) or not reference:
        return {}
    deltas = [v - r for r, v in zip(reference, variant)]
    low, high = bootstrap_ci(deltas, draws=draws, seed=seed)
    return {"delta": round(statistics.mean(deltas), 3), "ci95": [low, high],
            "significant": bool(low > 0 or high < 0)}
