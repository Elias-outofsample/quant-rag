"""Quel effet ce banc peut-il **voir** sur « ors servis nets » ? — et le seuil du plan y est-il ?

Le plan a pré-enregistré un seuil absolu de `+3` ors servis nets sur 300 questions. Ce
script demande ce qu'aucune des mesures du lot ne demande directement : **à partir de quelle
taille d'effet ce banc distingue-t-il un vrai gain d'un tirage ?**

La métrique est un total de ±1 par question. Seules les questions *discordantes* — celles qui
basculent dans un sens ou dans l'autre — portent de l'information ; les autres n'apportent
rien, et c'est pourquoi le test approprié est celui de McNemar. Le lot a observé **6 paires
discordantes sur 300** pour le candidat et **12** pour le placebo : le banc est donc, sur
cette métrique, un instrument à très peu de degrés de liberté.

La simulation ci-dessous n'est pas un modèle du système : c'est un modèle de **l'échantillon**.
Elle prend le taux de discordance observé pour acquis et fait varier la seule chose que le
protocole contrôle — le nombre de questions et la vraie taille d'effet — pour dire à quel
moment un `+3` cesse d'être indiscernable de zéro.

    .venv/bin/python rag/benchmark/resolution_banc.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RESULT = HERE / "resolution-banc.json"
GRAINE = 20260910
TIRAGES = 20000


def nets_simules(n_questions: int, p_discordance: float, delta: float, tirages: int,
                 rng: np.random.Generator) -> np.ndarray | None:
    """Tirages du net pour ``n_questions``, un taux de discordance et un effet vrai.

    ``delta`` est l'effet vrai exprimé **en ors servis nets pour 300 questions**, pour rester
    dans l'unité du plan : il fixe une probabilité par question (``delta/300``), et l'espérance
    du net à ``n_questions`` vaut donc ``n_questions × delta / 300``. C'est la bonne façon de
    changer d'échelle — mettre *à la fois* l'effet et le seuil à l'échelle compterait deux fois.
    """
    par_question = delta / 300.0
    if p_discordance <= 0:
        return None
    # p_haut − p_bas = par_question, p_haut + p_bas = p_discordance
    p_haut = (p_discordance + par_question) / 2
    p_bas = (p_discordance - par_question) / 2
    if not (0 <= p_bas <= p_haut <= 1):
        return None
    tirage = rng.random((tirages, n_questions))
    return ((tirage < p_haut).sum(axis=1)
            - ((tirage >= p_haut) & (tirage < p_haut + p_bas)).sum(axis=1))


def resume(nets: np.ndarray, seuil: float | None = None) -> dict:
    r = {"net_moyen": round(float(nets.mean()), 2), "ecart_type": round(float(nets.std()), 2),
         "ic95": [int(np.quantile(nets, .025)), int(np.quantile(nets, .975))],
         "part_au_moins_3": round(float((nets >= 3).mean()), 4)}
    if seuil is not None:
        r["puissance_au_seuil_calibre"] = round(float((nets >= seuil).mean()), 4)
    return r


def simuler(n_questions: int, p_discordance: float, delta: float, tirages: int,
            rng: np.random.Generator) -> dict:
    nets = nets_simules(n_questions, p_discordance, delta, tirages, rng)
    return {"impossible": True} if nets is None else resume(nets)


def main() -> None:
    rng = np.random.default_rng(GRAINE)
    mesure = json.loads((HERE / "resultats-latex-e1bdf36e2e.json").read_text(encoding="utf-8"))
    c = mesure["bras"]["c"]["tous"]
    p = mesure["bras"]["p"]["tous"]
    n = c["n"]
    disc_c = (c["n_gagnes"] + c["n_perdus"]) / n
    disc_p = (p["n_gagnes"] + p["n_perdus"]) / n
    print(f"observé sur {n} questions : candidat {c['n_gagnes']}+/{c['n_perdus']}− "
          f"({c['n_gagnes'] + c['n_perdus']} discordantes, {100 * disc_c:.1f} %) · "
          f"placebo {p['n_gagnes']}+/{p['n_perdus']}− "
          f"({p['n_gagnes'] + p['n_perdus']} discordantes, {100 * disc_p:.1f} %)")

    # ---- 1. sous variable nulle, à quelle fréquence le seuil +3 se déclenche-t-il ? ----
    print("\n1. VARIABLE NULLE — un bras sans aucun effet, au taux de discordance observé")
    nulles = {}
    for nom, disc in (("comme le candidat", disc_c), ("comme le placebo", disc_p)):
        r = simuler(n, disc, 0.0, TIRAGES, rng)
        nulles[nom] = r
        print(f"   {nom:20s} : net {r['net_moyen']:+.2f} ± {r['ecart_type']:.2f} · "
              f"IC95 {r['ic95']} · **atteint +3 dans {100 * r['part_au_moins_3']:.1f} % des cas**")

    # ---- 2. quelle taille d'effet ce banc voit-il ? ----
    print(f"\n2. RÉSOLUTION — puissance à 80 %, {n} questions, discordance du candidat")
    resolution = {}
    for delta in (2, 3, 5, 8, 10, 15, 20, 30):
        r = simuler(n, disc_c, float(delta), TIRAGES, rng)
        if r.get("impossible"):
            continue
        resolution[delta] = r
        marque = "  <- le seuil du plan" if delta == 3 else ""
        print(f"   effet vrai {delta:+3d} : net {r['net_moyen']:+6.2f} ± {r['ecart_type']:.2f} · "
              f"IC95 {r['ic95']} · atteint +3 dans {100 * r['part_au_moins_3']:.1f} % des cas{marque}")

    # ---- 3. combien de questions faudrait-il pour voir un +3 ? ----
    # Le critère n'est plus « net >= 3 » — un seuil fixe perd son sens quand le banc grandit —
    # mais un **test calibré sur la nulle** : à chaque taille, le seuil est le 95ᵉ centile du
    # net sous variable nulle (test unilatéral à 5 %), et la puissance est la part des tirages
    # à effet vrai qui le dépassent.
    print("\n3. TAILLE DE BANC — effet vrai de même ampleur que « +3 pour 300 », test calibré à 5 %")
    print(f"   {'questions':>9} {'seuil nul':>10} {'net moyen':>10} {'écart-type':>11} {'puissance':>10}")
    taille = {}
    for facteur in (1, 2, 3, 4, 6, 8, 12, 16):
        n_q = n * facteur
        sous_nulle = nets_simules(n_q, disc_c, 0.0, TIRAGES, rng)
        seuil = float(np.quantile(sous_nulle, .95))
        nets = nets_simules(n_q, disc_c, 3.0, TIRAGES, rng)
        r = resume(nets, seuil) | {"seuil_calibre": seuil,
                                   "ecart_type_sous_nulle": round(float(sous_nulle.std()), 2)}
        taille[n_q] = r
        print(f"   {n_q:9d} {seuil:10.1f} {r['net_moyen']:10.2f} {r['ecart_type']:11.2f} "
              f"{100 * r['puissance_au_seuil_calibre']:9.1f} %")

    suffisantes = [q for q, r in taille.items() if r["puissance_au_seuil_calibre"] >= 0.80]
    conclusion = (f"il faut de l'ordre de {min(suffisantes)} questions pour qu'un effet vrai de "
                  f"l'ampleur de « +3 pour 300 » soit détecté 8 fois sur 10 par un test à 5 %"
                  if suffisantes else
                  f"aucune des tailles essayées (jusqu'à {max(taille)}) ne suffit")
    print(f"\n-> {conclusion}")

    RESULT.write_text(json.dumps({
        "observe": {"questions": n, "discordantes_candidat": c["n_gagnes"] + c["n_perdus"],
                    "discordantes_placebo": p["n_gagnes"] + p["n_perdus"],
                    "net_candidat": c["net"], "net_placebo": p["net"]},
        "hypothese": ("le taux de discordance observé est pris pour acquis ; on ne fait varier "
                      "que le nombre de questions et la vraie taille d'effet. C'est un modèle de "
                      "l'échantillon, pas du système."),
        "variable_nulle": nulles, "resolution_a_300_questions": resolution,
        "taille_de_banc_necessaire": taille, "conclusion": conclusion,
        "graine": GRAINE, "tirages": TIRAGES,
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"-> {RESULT}")


if __name__ == "__main__":
    main()
