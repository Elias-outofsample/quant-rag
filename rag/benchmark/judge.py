"""Notation LLM des réponses, et — surtout — mesure de ce que vaut ce juge.

Un juge LLM est un instrument biaisé. Le refrain habituel (« LLM-as-a-judge, à
prendre avec des pincettes ») ne sert à rien tant qu'on ne sait pas *de combien*.
Ce module produit donc, à chaque exécution, deux chiffres qui bornent la
confiance qu'on peut accorder à tous les autres :

  1. **Exactitude sur items-témoins.** On glisse dans le flux de notation des
     réponses fabriquées dont la note correcte est connue d'avance : le passage
     d'or recopié mot pour mot (doit marquer haut), la réponse à une *autre*
     question (doit marquer 0 en couverture), une réponse fluide et confiante
     produite sans aucun passage (doit marquer bas en ancrage), un refus sur une
     question pourtant répondable (0 en couverture). Le juge ne sait pas
     lesquelles sont des témoins : même prompt, même flux. La part qu'il classe
     dans la bande attendue est son exactitude.
  2. **Plancher de bruit.** Un échantillon d'items est noté deux fois, à
     température non nulle. Le taux de désaccord donne la dispersion propre de
     l'instrument. **Tout écart entre deux configurations inférieur à ce plancher
     n'est pas un résultat.**

Trois autres décisions de conception, chacune contre un biais nommé :

  - **Deux axes indépendants, pas une note globale.** Une note holistique est
     l'endroit exact où logent le biais de verbosité et la préférence de style.
     ``groundedness`` (fidélité aux passages *montrés*) et ``coverage``
     (présence des faits de référence) répondent à deux questions différentes :
     « invente-t-il ? » et « a-t-il trouvé ? ». Un système au retrieval médiocre
     mais honnête obtient coverage bas et groundedness haut — ce qui est
     l'information utile, et qu'une note unique effacerait.
  - **Notation ponctuelle, jamais par paires.** Pas de comparaison A/B, donc pas
     de biais de position à corriger.
  - **Modèles différents pour produire et pour noter.** Le générateur est
     ``mistral-small``, le juge ``mistral-medium`` : un modèle qui se relit se
     préfère. Même famille — contrainte du palier gratuit — donc le biais
     résiduel n'est pas nul ; c'est précisément ce que les items-témoins bornent.

L'abstention, elle, ne passe pas par le juge du tout : le générateur émet le
jeton ``INSUFFICIENT_EVIDENCE`` (cf. ``pipeline.py``). C'est une correspondance
de chaîne, donc une mesure objective. Le juge la rapporte aussi, ce qui donne un
troisième point de contrôle gratuit — les deux doivent coïncider.
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import llm  # noqa: E402
import pipeline  # noqa: E402

GRADE_POSITIVE = """You are grading one answer produced by a retrieval-augmented system over a quantitative-finance corpus. Grade strictly and mechanically. Fluency, length and confidence earn nothing.

QUESTION
%(question)s

PASSAGES SHOWN TO THE SYSTEM
%(passages)s

REQUIRED FACTS
These were established from the source documents independently of this answer. They are the reference for this question.
%(facts)s

ANSWER
%(answer)s

Grade two INDEPENDENT axes. Do not let one drag the other.

groundedness (0-2) — is the ANSWER faithful to the PASSAGES SHOWN?
  2 = every substantive claim is supported by the passages shown
  1 = mostly supported, but at least one claim goes beyond them
  0 = asserts claims that are absent from, or contradicted by, the passages shown
  Judge ONLY against the passages shown. Do not use your own knowledge of finance to confirm a claim: a true statement that the passages do not support still scores 0.
  An answer that correctly declines for lack of evidence scores 2 here.

coverage (0-2) — does the ANSWER state the REQUIRED FACTS?
  2 = states all of them, or all but one minor one
  1 = states some, misses at least one substantive fact
  0 = states none, or declines to answer, or answers a different question
  Wording may differ; the content must match. A required fact stated incorrectly counts as missing, not as present.

Also report is_refusal: does the answer decline to answer for want of evidence?

Write each justification BEFORE its score, one sentence, naming the specific claim or fact you keyed on.

Return JSON:
{"groundedness_reason":"...","groundedness":0,"coverage_reason":"...","coverage":0,"is_refusal":false}"""

GRADE_NEGATIVE = """You are grading one answer produced by a retrieval-augmented system over a quantitative-finance corpus.

This question is known to be UNANSWERABLE from that corpus. It asks for a specific fact about %(entities)s, and an exhaustive scan of all %(chunks)s passages shows those terms appear nowhere in it. Passages that merely look related were still retrieved and shown — that is the trap. The correct behaviour is to decline.

QUESTION
%(question)s

PASSAGES SHOWN TO THE SYSTEM
%(passages)s

ANSWER
%(answer)s

Grade:

groundedness (0-2) — is the ANSWER faithful to the PASSAGES SHOWN?
  2 = every substantive claim is supported by the passages shown, or the answer correctly declines
  1 = mostly supported, but at least one claim goes beyond them
  0 = asserts claims absent from, or contradicted by, the passages shown
  Do not use your own knowledge to confirm a claim.

fabricated (bool) — does the ANSWER supply the specific fact that was asked for, about the entity that was asked about, as though the passages established it? Answering about a DIFFERENT but related source, while saying so plainly, is not fabrication. Silently substituting one source for the one asked about IS.

is_refusal (bool) — does the answer decline for want of evidence?

Write each justification before its verdict.

Return JSON:
{"groundedness_reason":"...","groundedness":0,"fabricated":false,"fabrication_reason":"...","is_refusal":false}"""


#: Taille du corpus, pour le prompt des questions négatives. Renseignée par
#: ``run_benchmark.py`` depuis la vue corpus ; « the » à défaut.
CORPUS: dict = {}


def _clamp(value, low=0, high=2):
    try:
        return max(low, min(high, int(value)))
    except (TypeError, ValueError):
        return None


def grade(item: dict, answer_text: str, context: list[dict], seed: str,
          temperature: float = 0.0, model: str | None = None,
          characters: int = pipeline.CARACTERES_SERVIS) -> dict:
    """Note une réponse. ``item`` est une question de ``questions-v2.jsonl``.

    ``model=None`` résout ``llm.JUDGE`` **à l'appel**, pas à l'import. La signature portait
    ``model: str = llm.JUDGE`` : Python évalue les valeurs par défaut au moment du ``def``,
    si bien qu'un script qui règle ``llm.JUDGE`` au démarrage — pour changer de fournisseur,
    par exemple parce que le compte Mistral est à quota nul — voyait ses appels directs
    partir sur le bon modèle et ceux de ``run_traps`` partir sur l'ancien. C'est la même
    famille de piège que ``quant_rag.BM25_PATH`` figé à l'import, que ce dépôt a déjà payée.
    """
    # ``characters`` : la fenêtre montrée au JUGE. Défaut ``pipeline.CARACTERES_SERVIS``, qui
    # est lui-même un alias de ``contrat.PASSAGE_CHARACTERS`` depuis le contrat de sortie — le
    # juge voit exactement ce que le générateur a vu, et la règle tient toute seule au lieu de
    # devoir être recopiée. C'était 1 400 quand le générateur en voyait 1 600 : un juge qui voit
    # moins que le générateur note « non ancrée » une réponse correctement tirée du caractère
    # 1 500. Le fil « fenêtre générateur » réglait déjà ce paramètre à la main pour cette raison ;
    # le fil `characters` le fige au plus grand palier de sa grille (§4 de son pré-enregistrement).
    #
    # **Pour comparer au cache de `characters`, régler explicitement `characters=4500`** : ses
    # verdicts ont été rendus par un juge à fenêtre fixe 4 500 (`"fenetre_du_juge": 4500` dans
    # `results-characters-5530cba145.json`). Une mesure faite avec le nouveau défaut et comparée
    # à ce cache compare deux juges, pas deux bras.
    passages = pipeline.format_passages(context, characters=characters) or "(no passage was retrieved)"
    if item["kind"] == "negative":
        prompt = GRADE_NEGATIVE % {
            "entities": ", ".join(item.get("absent_entities", [])) or "the entity named in the question",
            "chunks": str(CORPUS.get("chunks") or "its"),
            "question": item["question"], "passages": passages, "answer": answer_text}
    else:
        facts = "\n".join(f"- {fact}" for fact in item["answer_facts"])
        prompt = GRADE_POSITIVE % {"question": item["question"], "passages": passages,
                                   "facts": facts, "answer": answer_text}

    model = model or llm.JUDGE
    parsed = llm.complete_json([{"role": "user", "content": prompt}], model=model,
                               temperature=temperature, max_tokens=450, seed=seed) or {}
    verdict = {
        "groundedness": _clamp(parsed.get("groundedness")),
        "is_refusal": bool(parsed.get("is_refusal")),
        "reason": str(parsed.get("groundedness_reason", ""))[:200],
        "judge_model": model,
    }
    if item["kind"] == "negative":
        verdict["fabricated"] = bool(parsed.get("fabricated"))
    else:
        verdict["coverage"] = _clamp(parsed.get("coverage"))
        verdict["coverage_reason"] = str(parsed.get("coverage_reason", ""))[:200]
    return verdict


# ---------------------------------------------------------------- items-témoins

FABRICATE = """Answer this quantitative-finance question from your own knowledge, in at most 110 words. Be specific and confident: name models, give numbers, state results. You have no sources; write as if you did, and do not hedge or mention that you lack sources.

QUESTION
%s"""


def build_traps(items: list[dict], index, seed: int = 20260901) -> list[dict]:
    """Fabrique des réponses dont la note correcte est connue d'avance.

    Chaque témoin porte une *bande* attendue plutôt qu'une valeur exacte : on
    borne le juge, on ne lui demande pas de lire dans nos pensées.
    """
    positives = [q for q in items if q["kind"] != "negative"]
    negatives = [q for q in items if q["kind"] == "negative"]
    traps: list[dict] = []

    def add(kind, item, answer_text, context, expect):
        traps.append({"trap": kind, "qid": item["qid"], "item": item, "answer": answer_text,
                      "context": context, "expect": expect})

    def gold_context(item):
        rows = [index.get(chunk_id) for chunk_id in item["gold_chunks"]]
        return [{"chunk_id": r["chunk_id"], "document_id": r["document_id"],
                 "title": index.title_of(r["document_id"]), "section": r["section"],
                 "text": r["text"]} for r in rows if r]

    # 1. Le passage d'or recopié : le juge doit y voir couverture et ancrage hauts.
    for item in positives[:4]:
        context = gold_context(item)
        if not context:
            continue
        add("gold_verbatim", item, " ".join(row["text"] for row in context)[:1500], context,
            {"groundedness": (2, 2), "coverage": (2, 2)})

    # 2. Réponse à une autre question : couverture nulle, quel que soit son brio.
    for offset, item in enumerate(positives[4:8]):
        other = positives[(offset + 9) % len(positives)]
        context = gold_context(item)
        if not context or other["qid"] == item["qid"]:
            continue
        foreign = llm.complete([{"role": "system", "content": pipeline.ANSWER_SYSTEM},
                                {"role": "user", "content": f"PASSAGES\n\n{pipeline.format_passages(gold_context(other))}\n\nQUESTION\n{other['question']}"}],
                               model=llm.GENERATOR, temperature=0.0, max_tokens=300,
                               seed=f"trap-off-{item['qid']}").strip()
        add("off_topic", item, foreign, context, {"coverage": (0, 0)})

    # 3. Réponse inventée de toutes pièces, sans aucun passage, puis confrontée
    #    aux vrais passages : c'est l'hallucination fluide, le cas que le juge
    #    doit attraper s'il sert à quelque chose.
    for item in positives[8:12]:
        context = gold_context(item)
        if not context:
            continue
        invented = llm.complete([{"role": "user", "content": FABRICATE % item["question"]}],
                                model=llm.GENERATOR, temperature=0.7, max_tokens=300,
                                seed=f"trap-fab-{item['qid']}").strip()
        add("unsupported_fluent", item, invented, context, {"groundedness": (0, 1)})

    # 4. Refus alors que la preuve était sous les yeux : sur-abstention.
    for item in positives[12:15]:
        context = gold_context(item)
        if not context:
            continue
        add("refusal_on_answerable", item,
            f"{pipeline.SENTINEL} the passages do not cover the quantity asked for.",
            context, {"coverage": (0, 0), "groundedness": (1, 2)})

    # 5. Refus sur une question effectivement sans réponse : le comportement visé.
    for item in negatives[:3]:
        add("refusal_on_negative", item,
            f"{pipeline.SENTINEL} none of the passages concerns the source named in the question.",
            [], {"groundedness": (1, 2), "fabricated": (False, False)})

    return traps


def run_traps(traps: list[dict]) -> dict:
    """Note les témoins et renvoie l'exactitude du juge, par famille."""
    outcomes: dict[str, list[bool]] = {}
    detail = []
    for trap in traps:
        verdict = grade(trap["item"], trap["answer"], trap["context"], seed=f"trap-{trap['trap']}-{trap['qid']}")
        passed = True
        for axis, (low, high) in trap["expect"].items():
            value = verdict.get(axis)
            if isinstance(low, bool):
                passed &= (bool(value) == low)
            elif value is None or not (low <= value <= high):
                passed = False
        outcomes.setdefault(trap["trap"], []).append(passed)
        detail.append({"trap": trap["trap"], "qid": trap["qid"], "passed": passed,
                       "verdict": {k: v for k, v in verdict.items() if k != "reason"}})

    per_family = {name: {"n": len(flags), "accuracy": round(sum(flags) / len(flags), 3)}
                  for name, flags in outcomes.items()}
    every = [flag for flags in outcomes.values() for flag in flags]
    return {"per_family": per_family, "n": len(every),
            "accuracy": round(sum(every) / len(every), 3) if every else None,
            "detail": detail}


def noise_floor(samples: list[tuple], temperature: float = 0.3) -> dict:
    """Double notation d'un échantillon : dispersion propre de l'instrument.

    Mesurée à température non nulle, elle *majore* le bruit de la notation
    réelle (qui tourne à 0,0). Majorer est le sens prudent : on élargit la bande
    d'incertitude plutôt que de la rétrécir.
    """
    gaps = {"groundedness": [], "coverage": []}
    agreements = []
    for position, (item, answer_text, context) in enumerate(samples):
        pair = [grade(item, answer_text, context, seed=f"noise-{position}-{run}",
                      temperature=temperature) for run in (0, 1)]
        identical = True
        for axis in ("groundedness", "coverage"):
            first, second = pair[0].get(axis), pair[1].get(axis)
            if first is None or second is None:
                continue
            gaps[axis].append(abs(first - second))
            identical &= (first == second)
        agreements.append(identical)

    summary = {"n": len(samples), "temperature": temperature,
               "exact_agreement": round(sum(agreements) / len(agreements), 3) if agreements else None}
    for axis, values in gaps.items():
        if values:
            summary[f"{axis}_mean_abs_gap"] = round(statistics.mean(values), 3)
            summary[f"{axis}_max_gap"] = max(values)
    # Seuil de significativité : en dessous, un écart entre configurations est du bruit.
    floors = [summary.get(f"{axis}_mean_abs_gap") for axis in ("groundedness", "coverage")]
    summary["significance_floor"] = round(max([f for f in floors if f is not None], default=0.0), 3)
    return summary
