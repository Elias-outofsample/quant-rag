"""Lecture secondaire du fil `characters` — ré-ancrée sur la production, pas sur le banc.

**Cette lecture a été déclarée dans `PRE-ENREGISTREMENT-CHARACTERS-2026-09-08.md` §2 bis
(commit `a8e9cda`), pendant que la mesure tournait et avant qu'aucun verdict n'ait été
agrégé.** Elle ne change ni la grille, ni les gardes, ni la règle de sélection : le verdict
pré-enregistré reste celui d'`eval_characters.py --etape verdict`. Elle est écrite dans un
module séparé exprès, pour qu'aucun octet de l'instrument qui décide n'ait été touché après
avoir vu un résultat.

Le banc rend 1 600 caractères par passage, **la production en rend 2 500**
(`rag/mcp_server.py` l. 223). Le palier 2 400 est à 1,9 % du contexte de production. Donc :

- Δ(2 400 − 1 600) ne mesure **aucun gain** : il mesure de combien le banc **sous-estime** le
  produit tel qu'il tourne aujourd'hui ;
- Δ(3 200 − 2 400) et Δ(4 500 − 2 400) mesurent ce qu'**élever** la production achèterait.
  C'est celui-là, et lui seul, qui décide d'une bascule de produit.

    .venv/bin/python rag/benchmark/reancrage_characters.py
"""
from __future__ import annotations

import json
import math
import random
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import eval_characters as ec  # noqa: E402

ANCRE_PRODUIT = 2400  #: le palier qui reproduit la production (2 500 c.), à 1,9 % près


def _ic(deltas: list[float]) -> list[float]:
    alea = random.Random(ec.GRAINE)
    t = sorted(statistics.fmean(alea.choice(deltas) for _ in deltas) for _ in range(ec.TIRAGES))
    return [round(t[int(0.025 * ec.TIRAGES)], 4), round(t[int(0.975 * ec.TIRAGES)], 4)]


def _signe(a: int, b: int) -> float:
    """p bilatéral du test des signes sur les paires discordantes (McNemar exact)."""
    n = a + b
    if n == 0:
        return 1.0
    k = max(a, b)
    queue = sum(math.comb(n, i) for i in range(k, n + 1)) / 2 ** n
    return round(min(1.0, 2 * queue), 4)


def analyse() -> dict:
    items, _ = ec.population()
    verdicts = ec._charger(ec.CACHE / f"characters-verdicts-{ec.SIGNATURE}.json")
    reponses = ec._charger(ec.CACHE / f"characters-reponses-{ec.SIGNATURE}.json")
    qids = sorted(items)

    def couv(q, f):
        return (verdicts[f"{q}/{f}"].get("coverage") or 0)

    def abst(q, f):
        return bool(reponses[f"{q}/{f}"].get("abstained"))

    def contraste(bas: int, haut: int) -> dict:
        deltas = [couv(q, haut) - couv(q, bas) for q in qids]
        na = [q for q in qids if abst(q, haut) and not abst(q, bas)]
        nd = [q for q in qids if abst(q, bas) and not abst(q, haut)]
        return {
            "de": bas, "vers": haut,
            "delta": round(statistics.fmean(deltas), 4), "ic95": _ic(deltas),
            "montent": sum(1 for d in deltas if d > 0),
            "descendent": sum(1 for d in deltas if d < 0),
            "abst_nouvelles": len(na), "abst_disparues": len(nd),
            "abst_net": len(na) - len(nd), "abst_p_signes": _signe(len(na), len(nd)),
            "chutes_2_0": sum(1 for q in qids if couv(q, bas) == 2 and couv(q, haut) == 0),
        }

    paliers = {f: {
        "couverture": round(statistics.fmean(couv(q, f) for q in qids), 4),
        "pleine": sum(1 for q in qids if couv(q, f) == 2),
        "nulle": sum(1 for q in qids if couv(q, f) == 0),
        "abstentions": sum(1 for q in qids if abst(q, f)),
    } for f in ec.GRILLE}

    contrastes = {
        "banc_vers_production": contraste(1600, ANCRE_PRODUIT),
        "production_vers_3200": contraste(ANCRE_PRODUIT, 3200),
        "production_vers_4500": contraste(ANCRE_PRODUIT, 4500),
        "3200_vers_4500": contraste(3200, 4500),
    }
    out = {"signature": ec.SIGNATURE, "ancre_produit": ANCRE_PRODUIT, "n": len(qids),
           "declaree_dans": "PRE-ENREGISTREMENT-CHARACTERS-2026-09-08.md §2 bis (a8e9cda)",
           "paliers": paliers, "contrastes": contrastes}
    (HERE / f"results-characters-reancrage-{ec.SIGNATURE}.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    return out


def imprimer(r: dict) -> None:
    print(f"\nré-ancrage sur la production ({r['ancre_produit']} ≈ 2 500 c. servis) · n = {r['n']}\n")
    print(f"{'palier':>7s} {'couv.':>7s} {'pleine':>7s} {'nulle':>6s} {'abst.':>6s}")
    for f, p in r["paliers"].items():
        print(f"{int(f):7d} {p['couverture']:7.4f} {p['pleine']:7d} {p['nulle']:6d} {p['abstentions']:6d}")
    print(f"\n{'contraste':>26s} {'Δ':>8s} {'IC95':>20s} {'↑/↓':>7s} "
          f"{'abst +/−':>9s} {'net':>4s} {'p':>6s} {'2→0':>4s}")
    for nom, c in r["contrastes"].items():
        ic = f"[{c['ic95'][0]:+.3f} ; {c['ic95'][1]:+.3f}]"
        print(f"{nom:>26s} {c['delta']:+8.4f} {ic:>20s} "
              f"{c['montent']:3d}/{c['descendent']:<3d} "
              f"{c['abst_nouvelles']:4d}/{c['abst_disparues']:<4d} {c['abst_net']:+4d} "
              f"{c['abst_p_signes']:6.3f} {c['chutes_2_0']:4d}")


if __name__ == "__main__":
    imprimer(analyse())
