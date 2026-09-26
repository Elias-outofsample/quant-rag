"""Pourquoi une perturbation aléatoire **améliore** « or servi » : l'asymétrie du seuil.

Le balayage d'ampleur de `bancs_latex.py --balayage` montre un biais qui n'était pas cherché :
une perturbation isotrope **sans aucun contenu** rend un net *positif* tant que le cosinus
reste au-dessus de 0,98 (jusqu'à +1,65 en moyenne vers 0,995), et ne devient délétère qu'en
dessous. Autrement dit : **toucher les vecteurs au hasard fait monter la métrique.**

Ce script en cherche le mécanisme, qui est une propriété du seuil et non du modèle. « Or
servi » est un indicateur binaire « l'or est-il dans les 5 passages servis ? ». Autour de ce
seuil, deux populations s'opposent :

- les questions dont l'or est **juste dehors** (rangs 6 à 10) : une secousse aléatoire leur
  donne une chance de rentrer, donc un gain possible ;
- les questions dont l'or est **juste dedans** (rangs 4 et 5) : la même secousse peut les
  faire sortir, donc une perte possible.

Si la première population est plus nombreuse que la seconde, l'espérance du net sous
perturbation nulle-en-contenu est **positive**, et toute modification des vecteurs paraîtra
légèrement bénéfique. C'est exactement le piège qu'un bras placebo est censé révéler — et que
ce lot n'a vu qu'en balayant l'ampleur.

    .venv/bin/python rag/benchmark/asymetrie_seuil.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "titles"))

import bancs_latex as B  # noqa: E402
import pipeline  # noqa: E402
import quant_rag  # noqa: E402
from dense_matrix import Matrix  # noqa: E402

RESULT = HERE / "asymetrie-seuil.json"


def main() -> None:
    _, textes, _ = B.charger_corpus()
    matrix = Matrix.load()
    items = B.charger_questions()
    gold = {f"{b}/{it['qid']}": set(it.get("gold_chunks") or []) for b, it in items}

    rangs: dict[str, int | None] = {}
    servis: dict[str, bool] = {}
    for b, it in items:
        cle = f"{b}/{it['qid']}"
        filtres = {k: v for k, v in (pipeline.filters_of(it) or {}).items() if v is not None}
        scope = quant_rag.document_scope(None, **filtres) if filtres else None
        pool = B.chercher(matrix, np.asarray(quant_rag.encode_query(pipeline.query_of(it)),
                                            dtype=np.float32), scope, textes)
        rangs[cle] = B.rang_de_l_or(pool, gold[cle])
        servis[cle] = bool(B.servi(pool) & gold[cle])

    bandes = Counter(B.bande(rangs[c]) for c in rangs)
    print(f"rang du premier or, référence, {len(rangs)} questions")
    for b in ("1", "2-3", "4-5", "6-10", "11-20", "21-50", "hors pool"):
        print(f"  {b:>10} : {bandes[b]:4d}")
    print(f"  or servi : {sum(servis.values())}/{len(servis)}")

    # Les deux populations qui décident du signe du biais.
    juste_dedans = [c for c in rangs if servis[c] and rangs[c] is not None and rangs[c] in (4, 5)]
    juste_dehors = [c for c in rangs if not servis[c] and rangs[c] is not None and 6 <= rangs[c] <= 10]
    # Plus largement : tout or non servi mais présent dans le pool peut rentrer ; tout or servi
    # peut sortir, mais d'autant moins facilement que son rang est petit.
    non_servis_dans_le_pool = [c for c in rangs if not servis[c] and rangs[c] is not None]
    servis_au_rang_1 = [c for c in rangs if servis[c] and rangs[c] == 1]

    print(f"\nla population qui peut GAGNER (or non servi, rangs 6-10)   : {len(juste_dehors)}")
    print(f"la population qui peut PERDRE (or servi, rangs 4-5)        : {len(juste_dedans)}")
    print(f"  rapport : {len(juste_dehors) / max(len(juste_dedans), 1):.1f} contre 1")
    print(f"\nplus largement :")
    print(f"  or non servi mais dans le pool (peut rentrer)            : {len(non_servis_dans_le_pool)}")
    print(f"  or servi au rang 1 (quasi insensible à une secousse)     : {len(servis_au_rang_1)}")
    print(f"  or hors pool (inatteignable par une secousse de ce genre): {bandes['hors pool']}")

    lecture = (
        f"{len(juste_dehors)} questions ont leur or juste en dehors des 5 servis (rangs 6-10) "
        f"contre {len(juste_dedans)} juste en dedans (rangs 4-5), soit "
        f"{len(juste_dehors) / max(len(juste_dedans), 1):.1f} contre 1. Une secousse aléatoire a "
        "donc structurellement plus d'occasions de faire entrer un or que de l'en faire sortir, "
        "et l'espérance du net sous variable nulle est positive. « Or servi » est un indicateur "
        "binaire de franchissement de seuil : il n'est pas neutre vis-à-vis du bruit, il est "
        "biaisé vers le haut tant que la perturbation reste petite."
    )
    print(f"\n-> {lecture}")

    RESULT.write_text(json.dumps({
        "questions": len(rangs), "or_servi": sum(servis.values()),
        "bandes_de_rang": dict(bandes),
        "peut_gagner_rangs_6_10": len(juste_dehors),
        "peut_perdre_rangs_4_5": len(juste_dedans),
        "or_non_servi_dans_le_pool": len(non_servis_dans_le_pool),
        "or_servi_au_rang_1": len(servis_au_rang_1),
        "or_hors_pool": bandes["hors pool"],
        "lecture": lecture,
        "appels_llm": 0,
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"-> {RESULT}")


if __name__ == "__main__":
    main()
