"""Le contrat v3 est-il *suivi* ? — dix questions, avant de payer le run entier.

Pourquoi une étape séparée, et pourquoi avant
----------------------------------------------
Un contrat de réponse ne vaut que si le générateur le suit. L'hypothèse (c) du handoff —
« ``mistral-small-latest`` suit un format ``NOT_IN_SOURCES`` de façon fiable » — est une
hypothèse sur un modèle, pas sur le produit, et elle se réfute pour 0,005 USD au lieu de 0,63.
Le §5.3 du handoff en fait une obligation : **corriger le texte avant le run, pas après.**

Ce module ne note rien et ne compare rien. Il compte des faits de forme :

    - la ligne ``NOT_IN_SOURCES: <grandeur>`` est-elle produite, et **seule sur sa ligne** ?
    - les extraits entre guillemets tiennent-ils dans la fenêtre 5–25 mots du contrat ?
    - les affirmations chiffrées portent-elles un marqueur ``[n]`` ?
    - les formules sont-elles en ``$…$`` plutôt qu'en Unicode ?
    - la réponse tient-elle dans son plafond de mots ?

Les dix questions sont **fixées d'avance et déclarées** : les cinq premières négatives
voisines — c'est là que le signal doit apparaître —, les trois premières ``table_cell`` et les
deux premières ``formula``, où il ne doit **pas** apparaître et où les citations sont attendues.
Un tirage aléatoire aurait rendu ce contrôle irreproductible ; un tirage choisi après coup
l'aurait rendu complaisant.

    .venv/bin/python rag/benchmark/format_reponse_v3.py --prompt v3
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import banc_v4  # noqa: E402
import latex_norme  # noqa: E402
import llm  # noqa: E402
import pipeline  # noqa: E402
import score_citation  # noqa: E402

#: L'échantillon, écrit avant le premier appel. ``(famille, rang dans la famille)``.
ECHANTILLON = ([("negative_voisine", n) for n in range(5)]
               + [("table_cell", n) for n in range(3)]
               + [("formula", n) for n in range(2)])

#: Symboles mathématiques Unicode que la clause 5 interdit au profit du LaTeX. La liste est
#: courte et vise ce qu'un générateur écrit spontanément — lettres grecques, opérateurs,
#: indices et exposants — et non tout le plan mathématique d'Unicode : il s'agit de voir si la
#: clause est suivie, pas de dresser un inventaire.
UNICODE_MATH = "αβγδεζηθικλμνξπρστυφχψωΑΒΓΔΘΛΞΠΣΦΨΩ√∑∏∫≤≥≠≈±∞∂∇⁰¹²³⁴⁵⁶⁷⁸⁹₀₁₂₃₄₅₆₇₈₉"

_LIGNE_NON_TROUVE = re.compile(rf"{pipeline.NON_TROUVE}\s*:", re.IGNORECASE)


def _fiche(texte: str, contexte: list[dict]) -> dict:
    """Ce que la forme d'une réponse dit, sans juger son contenu."""
    grandeurs = pipeline.non_trouve(texte)
    mentions = len(_LIGNE_NON_TROUVE.findall(texte))
    extraits = score_citation.extraits(texte)
    verifies = score_citation.verifier_extraits(extraits, "\n\n".join(
        (p.get("text") or "")[:pipeline.CARACTERES_SERVIS] for p in contexte))
    # Toutes les paires de guillemets, y compris celles que ``extraits`` écarte parce qu'elles
    # font moins de 5 mots : c'est précisément ce que le contrat interdit, et l'ignorer ici
    # ferait passer une citation trop courte pour une absence de citation.
    toutes = re.findall(r'"([^"\n]{1,600}?)"', texte or "")
    # La fenêtre 5–25 **mots** ne s'applique qu'à la prose : le contrat exempte les formules, et
    # une formule éclatée par MinerU — ``\mathbb { E } \left. p _ { n } ^ { 2 } \right.`` — compte
    # une quarantaine de « mots » qu'aucun humain n'y lit. Les compter aurait déclaré « trop
    # longue » chaque citation de formule exacte, et fait corriger le contrat contre un défaut
    # de l'instrument. Mesuré au premier passage du test : 5 extraits « trop longs » sur 5.
    longueurs = [len(bloc.split()) for bloc in toutes if not latex_norme.expressions(bloc)]
    affirmations = score_citation.affirmations(texte)
    return {
        "mots": len(texte.split()),
        "non_trouve": grandeurs,
        # Une mention non lue par le lecteur de ligne est une ligne mal formée : le jeton est
        # là, la regex ne le voit pas. C'est la panne que ce test existe pour attraper.
        "mentions_mal_formees": mentions - len(grandeurs),
        "abstenue": pipeline.abstained(texte),
        "affirmations": len(affirmations),
        "affirmations_sans_marqueur": sum(1 for a in affirmations if not a["marqueurs"]),
        "guillemets": len(toutes),
        "extraits_5_a_25_mots": sum(1 for n in longueurs if 5 <= n <= 25),
        "extraits_trop_courts": sum(1 for n in longueurs if n < 5),
        "extraits_trop_longs": sum(1 for n in longueurs if n > 25),
        "extraits_retrouves": sum(1 for e in verifies if e["retrouve"]),
        "extraits_verifiables": len(verifies),
        "latex": len(re.findall(r"\$[^$]+\$", texte or "")),
        "unicode_math": sum(1 for c in (texte or "") if c in UNICODE_MATH),
    }


def controler(prompt: str, limite: int | None = None) -> dict:
    from corpus import ChunkIndex  # noqa: PLC0415  (charge le corpus : import tardif)

    items = banc_v4.population(ChunkIndex.load(verbose=False))
    par_famille: dict[str, list] = {}
    for item in items:
        par_famille.setdefault(item["famille"], []).append(item)
    contextes = banc_v4._charger(banc_v4.CACHE / f"v4-contextes-{banc_v4.SIGNATURE}.json")

    choisis = [par_famille[famille][rang] for famille, rang in (ECHANTILLON[:limite] if limite
                                                               else ECHANTILLON)]
    lignes = []
    for item in choisis:
        cle = item["qid_banc"]
        contexte = contextes.get(cle)
        if contexte is None:
            contexte = banc_v4.servis(item["question"])
        sortie = pipeline.answer(item["question"], contexte, model=banc_v4.GENERATEUR,
                                 prompt=prompt, characters=pipeline.CARACTERES_SERVIS)
        lignes.append({"qid_banc": cle, "famille": item["famille"],
                       "or_servi": bool({r["chunk_id"] for r in contexte}
                                        & set(item.get("gold_chunks") or [])),
                       "reponse": sortie["answer"], **_fiche(sortie["answer"], contexte)})
        print(f"  {cle:28s} {lignes[-1]['mots']:4d} mots · "
              f"NOT_IN_SOURCES {len(lignes[-1]['non_trouve'])} · "
              f"guillemets {lignes[-1]['guillemets']} · latex {lignes[-1]['latex']}")

    negatives = [l for l in lignes if l["famille"].startswith("negative")]
    positives = [l for l in lignes if not l["famille"].startswith("negative")]
    resume = {
        "prompt": prompt,
        "empreinte_prompt": pipeline.empreinte_prompt(prompt),
        "signature": banc_v4.SIGNATURE,
        "generateur": banc_v4.GENERATEUR,
        "n": len(lignes),
        "lignes_mal_formees": sum(l["mentions_mal_formees"] for l in lignes),
        "negatives_avec_signal": sum(1 for l in negatives if l["non_trouve"]),
        "negatives": len(negatives),
        "positives_avec_signal": sum(1 for l in positives if l["non_trouve"]),
        "positives": len(positives),
        "positives_citant": sum(1 for l in positives if l["extraits_5_a_25_mots"]),
        "extraits_trop_courts": sum(l["extraits_trop_courts"] for l in lignes),
        "extraits_trop_longs": sum(l["extraits_trop_longs"] for l in lignes),
        "extraits_retrouves": sum(l["extraits_retrouves"] for l in lignes),
        "extraits_verifiables": sum(l["extraits_verifiables"] for l in lignes),
        "unicode_math": sum(l["unicode_math"] for l in lignes),
        "au_dela_du_plafond": sum(1 for l in lignes if l["mots"] > 200),
        "cout": llm.stats(),
        "detail": lignes,
    }
    chemin = HERE / f"format-reponse-{prompt}-{banc_v4.SIGNATURE}.json"
    chemin.write_text(json.dumps(resume, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n{json.dumps({k: v for k, v in resume.items() if k != 'detail'}, ensure_ascii=False, indent=1)}")
    print(f"\nécrit : {chemin}")
    return resume


def main() -> None:
    analyse = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    analyse.add_argument("--prompt", default="v3", choices=sorted(pipeline.ANSWER_PROMPTS))
    analyse.add_argument("--limite", type=int, default=None)
    arguments = analyse.parse_args()
    controler(arguments.prompt, arguments.limite)


if __name__ == "__main__":
    main()
