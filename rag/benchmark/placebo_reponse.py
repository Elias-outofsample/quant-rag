"""Le bras placebo du chantier `contrat-de-reponse` : le même contrat, un tirage neuf.

Pourquoi ce bras existe, et pourquoi sans lui rien n'est lisible
-----------------------------------------------------------------
Le chantier `representation` a mesuré le 9 septembre 2026 que **`mistral-small-latest` n'est pas
reproductible à température 0** : re-générer 130 réponses sur des prompts **identiques au bit
près** rend 51/130 identiques, 79 différentes, et **9 bascules d'abstention**. Il en tire une
règle que ce module applique :

> tout Δ apparié qui compare un bras servi par le **cache** à un bras **regénéré** est confondu.

C'est exactement la configuration de ce chantier. Le bras v2 vient intégralement du cache du
8-9 septembre ; les bras v3 et v4 ont été tirés aujourd'hui. Une part de chaque Δ publié est
donc du bruit de génération, et **aucun chiffre ne dit laquelle** — jusqu'à ce bras.

Ce qu'il fait
--------------
Il rejoue les **199 questions** sous le contrat **v2**, celui de la référence, avec un cache
d'appels **neuf**. Le contraste (placebo − référence) ne peut alors contenir que le bruit du
générateur : même corpus, mêmes passages, même contrat, mêmes modèles, mêmes questions. C'est
le **plancher** sous lequel aucun Δ de ce chantier ne veut rien dire.

    .venv/bin/python rag/benchmark/placebo_reponse.py

Ce qu'il ne fait pas
---------------------
Il ne corrige rien. Un placebo ne répare pas un instrument bruité : il en publie le bruit, pour
qu'on sache lire les autres bras. La réparation — tirer les deux bras dans le même run — est une
décision qui appartient au propriétaire du banc, et elle est proposée au rapport, pas prise ici.

Le cache neuf vit dans un répertoire temporaire et **rien n'est écrit dans le cache du dépôt** :
un placebo qui polluerait le cache rendrait le prochain run de la référence non reproductible,
c'est-à-dire qu'il détruirait ce qu'il mesure.

SUPERSÉDÉ LE 10 SEPTEMBRE 2026 — et c'est ce module qui l'avait demandé
------------------------------------------------------------------------
Sa dernière phrase disait : *« La réparation — tirer les deux bras dans le même run — est une
décision qui appartient au propriétaire du banc, et elle est proposée au rapport, pas prise
ici. »* La décision a été prise, et le mécanisme est **dans le banc** :

    banc_v4.py --etape mesure --bras "<ref>,<candidat>,<nom>:placebo"

C'est strictement mieux, pour une raison qui n'est pas de commodité : ce module compare un bras
tiré **aujourd'hui** à une référence tirée **hier**, si bien que son Δ mêle le bruit du
générateur à tout ce qui sépare deux moments. Le placebo intégré est entrelacé question par
question avec sa référence — le bruit qu'il publie est celui du run qu'on lit.

Ce module reste exécutable pour rejouer l'artefact ``results-v4-e1bdf36e2e-v2placebo.json`` du
9 septembre, et il délègue désormais au banc plutôt que d'en dupliquer la mécanique.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

#: Le nom du bras. Son **texte** est celui de v2 : ``pipeline.empreinte_prompt`` rend la même
#: empreinte, donc ``banc_v4.divergences`` ne voit aucun écart et la comparaison est autorisée
#: sans drapeau. C'est voulu — un placebo dont le banc refuserait la comparaison ne mesurerait
#: pas ce qu'il prétend mesurer.
BRAS = "v2placebo"


def main() -> None:
    analyse = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    analyse.add_argument("--limite", type=int, default=None,
                         help="sous-échantillon, pour un rodage")
    analyse.add_argument("--garder-le-cache", action="store_true",
                         help="ne pas effacer le cache temporaire à la fin (diagnostic)")
    arguments = analyse.parse_args()

    import banc_v4  # noqa: PLC0415
    import pipeline  # noqa: PLC0415

    assert pipeline.ANSWER_PROMPTS[BRAS] == pipeline.ANSWER_SYSTEM_V2, \
        "le placebo doit porter le texte de la référence, sinon ce n'est pas un placebo"
    assert pipeline.empreinte_prompt(BRAS) == pipeline.empreinte_prompt("v2"), \
        "même texte, donc même empreinte : le banc doit accepter la comparaison sans drapeau"

    # ``jetable=True`` fait tout ce que ce module faisait à la main : cache d'appels neuf, état
    # reparti de zéro, coût extrait du cache temporaire AVANT sa destruction, rien d'écrit dans
    # le cache du dépôt. Le rôle est déclaré ``placebo`` pour que le verdict n'exige pas de ce
    # bras un plancher — il EST le plancher.
    #
    # Le bras est construit ici plutôt que par ``analyser_bras`` parce qu'un placebo y est
    # défini par rapport à une référence *du même run*, et que ce module, par construction
    # historique, n'en a pas : sa référence est le run du 9 septembre, déjà en cache. C'est
    # exactement la limite qui a fait remplacer ce module.
    bras = banc_v4.Bras(nom=BRAS, prompt=BRAS, fenetre_servie=banc_v4.FENETRE_SERVIE,
                        role="placebo", jetable=True)
    print("cache d'appels NEUF, géré par le banc : aucun appel ne peut être servi par le cache "
          "du dépôt, et rien n'y est écrit.\n")
    banc_v4.mesure([bras], banc_v4.FENETRE_JUGE, arguments.limite)
    if arguments.garder_le_cache:
        print("\n--garder-le-cache n'a plus d'effet : le banc extrait le coût du cache "
              "temporaire avant de l'effacer, si bien que `appels_manquants` vaut désormais 0 "
              "au lieu de 199 et qu'il n'y a plus rien à conserver pour diagnostic.")


if __name__ == "__main__":
    main()
