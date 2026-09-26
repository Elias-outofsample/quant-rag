"""Le contrôle à livre fermé — un item qu'un générateur seul réussit ne mesure pas le RAG.

**Pourquoi ce contrôle prime sur une baseline.** Un banc RAG mesure la chaîne
retrieval → contexte → réponse. Si le générateur produit les faits attendus **sans aucun
passage**, l'item note la mémoire paramétrique du modèle et rien d'autre : il donnera le même
score à un système dont l'index est vide.

Deux conditions, parce qu'elles répondent à deux questions différentes :

    NU     la question seule, sans prompt de contrainte  -> le modèle SAIT-il ?
    PROD   le prompt de production, zéro passage         -> le système servi s'abstient-il ?

**L'instrument réellement utilisé le 7 septembre 2026 n'est pas celui-ci.** Les deux
fournisseurs du banc étaient fermés — Google en HTTP 429, Mistral avec un plafond de compte à
zéro — et la campagne a donc été passée avec `claude-opus-5`, en substitution **déclarée**.
Le résultat est asymétrique et doit être lu comme tel :

  * une couverture **0** se transfère *a fortiori* — un modèle plus faible ne fera pas mieux,
    donc l'item exige réellement la récupération ;
  * une couverture **2** ne se transfère **pas** — elle ne prouve pas que le générateur du banc
    connaîtrait la réponse, seulement que l'item n'est plus *sûr*. C'est une borne supérieure
    de fuite, pas la fuite du banc.

Ce script reste l'instrument de référence : il faut le rejouer dès que le quota revient, et
il écrit alors son propre bloc `instrument`, à côté de celui du substitut, sans l'écraser.
La comparaison des deux blocs est le seul moyen de savoir de combien la substitution a menti.

    .venv/bin/python rag/benchmark/human-v1/ai-reviewed-v1/controle_livre_ferme.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
HUMAN = HERE.parent
BENCHMARK = HUMAN.parent
ROOT = BENCHMARK.parents[1]
sys.path.insert(0, str(BENCHMARK))
sys.path.insert(0, str(BENCHMARK.parent))

import judge  # noqa: E402
import llm  # noqa: E402
import pipeline  # noqa: E402

LOT = HERE / "items-ai-reviewed-v1.jsonl"
SORTIE = HERE / "controle-livre-ferme.json"
GENERATEUR = "gemini-3.1-flash-lite"
JUGE = "gemini-3.1-flash-lite"

#: Le verdict se lit sur la condition NU, et seulement sur elle : PROD mesure l'abstention,
#: pas la connaissance.
VERDICTS = {2: "ne_discrimine_pas", 1: "discrimine_mal", 0: "discrimine"}


def charger() -> list[dict]:
    return [json.loads(l) for l in LOT.read_text(encoding="utf-8").splitlines() if l.strip()]


def faits_obligatoires(item: dict) -> list[str]:
    """Le barème du contrôle, et il est volontairement le même que celui d'une campagne.

    Y compris quand ces faits sont eux-mêmes `gold_candidate` : le contrôle demande « cet item
    discriminerait-il *s'il servait* », pas « ce gold est-il bon ». Mélanger les deux
    questions rendrait le résultat ininterprétable.
    """
    return [f["texte"] for f in item["faits_attendus"] if f["exigence"] == "obligatoire"]


def sans_passage(question: str, modele: str) -> tuple[str, str]:
    """Les deux conditions, dans l'ordre où elles se lisent."""
    nu = llm.complete(
        [{"role": "user",
          "content": "Answer from your own knowledge. At most 130 words, no preamble, "
                     f"no restating the question. If you do not know, say so.\n\n{question}"}],
        model=modele, temperature=0.0, max_tokens=400).strip()
    prod = llm.complete(
        [{"role": "system", "content": pipeline.ANSWER_PROMPTS[pipeline.DEFAULT_PROMPT]},
         {"role": "user", "content": "PASSAGES\n\n(no passage was retrieved)\n\n"
                                     f"QUESTION\n{question}"}],
        model=modele, temperature=0.0, max_tokens=400).strip()
    return nu, prod


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--items", nargs="*", help="limiter à ces identifiants")
    parser.add_argument("--cle", default="banc", help="nom du bloc instrument à écrire")
    args = parser.parse_args()
    llm.GENERATOR, llm.JUDGE = GENERATEUR, JUGE

    artefact = json.loads(SORTIE.read_text(encoding="utf-8")) if SORTIE.exists() else {}
    # Panne évitée : écraser la campagne substitut effacerait la seule mesure existante et,
    # avec elle, la possibilité de chiffrer ce que la substitution a coûté.
    runs = artefact.setdefault("runs", {})
    resultats = runs.setdefault(args.cle, {"generateur": GENERATEUR, "juge": JUGE, "items": {}})["items"]

    for item in charger():
        if args.items and item["id"] not in args.items:
            continue
        if item["id"] in resultats:
            continue
        obligatoires = faits_obligatoires(item)
        if not obligatoires:
            resultats[item["id"]] = {"sans_objet": "aucun fait obligatoire (item d'abstention)"}
            continue
        try:
            nu, prod = sans_passage(item["question"], GENERATEUR)
            notes = {}
            for nom, texte in (("nu", nu), ("prod", prod)):
                note = judge.grade({"kind": "single", "question": item["question"],
                                    "answer_facts": obligatoires},
                                   texte, [], seed=f"lf-{nom}-{item['id']}", model=JUGE)
                notes[nom] = {"reponse": texte, "coverage": note.get("coverage"),
                              "coverage_reason": note.get("coverage_reason"),
                              "is_refusal": note.get("is_refusal"),
                              "abstention": pipeline.abstained(texte)}
            resultats[item["id"]] = {
                "faits_obligatoires": len(obligatoires), **notes,
                "verdict_discrimination": VERDICTS.get(notes["nu"]["coverage"], "indetermine"),
            }
            print(f"  {item['id']}  NU={notes['nu']['coverage']} PROD={notes['prod']['coverage']}",
                  flush=True)
        except RuntimeError as erreur:
            print(f"  {item['id']}  ÉCHEC API : {str(erreur)[:70]}", flush=True)
            runs[args.cle]["echec"] = str(erreur)[:400]
            break
        SORTIE.write_text(json.dumps(artefact, indent=1, ensure_ascii=False), encoding="utf-8")

    mesures = [v for v in resultats.values() if "verdict_discrimination" in v]
    if mesures:
        moyenne = sum(m["nu"]["coverage"] for m in mesures) / len(mesures)
        print(f"\n  {len(mesures)} items mesurés · couverture à livre fermé {moyenne:.2f} / 2")
        for cle, v in sorted(resultats.items()):
            if "verdict_discrimination" in v:
                print(f"    {cle}  NU {v['nu']['coverage']}/2  {v['verdict_discrimination']}")


if __name__ == "__main__":
    main()
