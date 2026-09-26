"""Le générateur rend-il deux fois la même réponse au même prompt ? — non, et ça change tout.

Ce que ce contrôle a trouvé, le 9 septembre 2026
------------------------------------------------
Re-générer les 130 réponses d'une ligne de base sur des prompts **identiques au bit près**
(le cache d'appels est indexé par le sha256 du payload : un hit *est* la preuve de l'identité),
à ``temperature=0.0``, sur ``mistral-small-latest`` :

    130 réponses · 51 identiques (39,2 %) · 79 différentes · 9 BASCULES D'ABSTENTION

**Le générateur n'est pas reproductible.** Neuf abstentions basculent sans qu'aucune entrée
n'ait changé. Or la garde d'abstention du banc échoue à ``a > 2`` : elle est calibrée **sous le
plancher de bruit de l'instrument qu'elle surveille**, et son taux de fausse alarme publié
supposait un générateur déterministe.

Pourquoi ce contrôle doit précéder tout Δ apparié
--------------------------------------------------
Une ligne de base servie par le cache disque et un bras regénéré aujourd'hui ne viennent pas du
même tirage. Le Δ mesuré contient alors le bruit du générateur, et rien ne le dit. Ce contrôle
coûte **~0,06 €** — une génération par question, aucun appel au juge — là où le bras placebo qui
le quantifie proprement en coûte dix fois plus. À passer *avant*, donc.

Il ne remplace pas le bras placebo : il dit **que** le bruit existe et de quelle ampleur en
nombre de réponses, pas ce qu'il vaut **sur la métrique**. C'est
``eval_representation.py --etape mesure --placebo`` qui répond à la seconde question.

    .venv/bin/python rag/benchmark/controle_derive_generateur.py --bras e1bdf36e2e
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import llm  # noqa: E402

CACHE = HERE / ".cache"


def controle(bras: str, fenetre: int) -> dict:
    # Cache neuf : rien ne peut être servi par un appel d'hier, et rien n'est écrit dans le
    # cache du dépôt. C'est toute l'astuce du contrôle.
    llm.CACHE = Path(tempfile.mkdtemp(prefix="derive-generateur-"))

    import eval_characters
    import pipeline

    anciennes = json.loads((CACHE / f"representation-reponses-{bras}.json").read_text(encoding="utf-8"))
    items, contextes = eval_characters.population()

    identiques, differentes, details = 0, 0, []
    for n, qid in enumerate(sorted(items), 1):
        cle = f"{qid}/{fenetre}"
        if cle not in anciennes:
            continue
        sortie = pipeline.answer(items[qid]["question"], contextes[qid],
                                 model=eval_characters.GENERATEUR, characters=fenetre)
        if sortie["answer"] == anciennes[cle]["answer"]:
            identiques += 1
        else:
            differentes += 1
            details.append({"qid": qid, "abstenue_avant": anciennes[cle]["abstained"],
                            "abstenue_apres": sortie["abstained"]})
        print(f"  {n}/{len(items)}  identiques {identiques}  différentes {differentes}",
              end="\r", flush=True)
    print()
    n = identiques + differentes
    return {"bras": bras, "fenetre": fenetre, "generateur": eval_characters.GENERATEUR,
            "temperature": 0.0, "n": n, "identiques": identiques, "differentes": differentes,
            "part_identiques": round(identiques / max(n, 1), 4),
            "bascules_d_abstention": sum(1 for d in details
                                         if d["abstenue_avant"] != d["abstenue_apres"]),
            "detail": details, "appels": llm.stats()}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--bras", required=True, help="signature du bras dont on regénère les réponses")
    p.add_argument("--fenetre", type=int, default=10_000)
    a = p.parse_args()
    resultat = controle(a.bras, a.fenetre)
    (HERE / f"results-derive-generateur-{a.bras}.json").write_text(
        json.dumps(resultat, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in resultat.items() if k != "detail"},
                     ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
