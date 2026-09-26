"""La garde conditionnée à la rotation — règle du pré-enregistrement du 8 septembre 2026.

Partie I publiée en ``dcb892c`` **avant l'ouverture du moindre fichier de résultats concerné**.
Ce module ne fait que l'**appliquer**, sans l'ajuster : ``b = 0`` tient ; sinon la borne haute
de l'IC95 de Wilson de ``b / (b + g)`` doit être ``<= TAU``.

``TAU = p / (1 + p)`` avec ``p = 9/17``, la part des entrantes qui convertissent en réponse
meilleure (§19 bis, publiée le 7 septembre, la plus conservatrice des deux mesures publiées).
Un or perdu détruit la réponse 12 fois sur 12 (§15). Aucun nombre appartenant à un candidat
n'entre dans ce calcul.

    .venv/bin/python rag/benchmark/garde_rotation.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from garde_reponse import wilson_haut  # noqa: E402 — le même estimateur, sans variante

#: Taux de conversion d'un or entrant, §19 bis (9/17). Le §15 donnait 14/24 ; le minimum des
#: deux est retenu, et ce choix est déclaré au §3 du pré-enregistrement.
P_GAIN = 9 / 17
TAU = P_GAIN / (1 + P_GAIN)


def tenue(b: int, g: int) -> dict:
    """La règle, telle qu'écrite au §4 du pré-enregistrement."""
    rotation = b + g
    if b == 0:
        return {"b": b, "g": g, "rotation": rotation, "part": 0.0, "wilson_haut": 0.0,
                "tenue": True, "motif": "b = 0 — aucune destruction à borner"}
    haut = wilson_haut(b, rotation)
    return {"b": b, "g": g, "rotation": rotation, "part": round(b / rotation, 4),
            "wilson_haut": round(haut, 4), "tenue": haut <= TAU,
            "motif": f"borne haute {haut:.4f} {'<=' if haut <= TAU else '>'} τ = {TAU:.4f}"}


def rotation_necessaire(part: float, maximum: int = 4000) -> int | None:
    """Rotation minimale pour qu'une part donnée franchisse la garde. ``None`` si jamais."""
    if part <= 0:
        return 0
    for n in range(2, maximum + 1):
        b = round(part * n)
        if b and wilson_haut(b, n) <= TAU:
            return n
    return None


def cas() -> list[dict]:
    """Tous les cas passés disponibles, avec la garde qui leur est applicable."""
    out = []
    fichier = HERE / "results-reclassement-selectif-5530cba145.json"
    if fichier.exists():
        d = json.loads(fichier.read_text(encoding="utf-8"))
        for nom, v in d["verdicts"].items():
            g = v.get("garde", v)
            out.append({"chantier": "reclassement sélectif", "variante": nom,
                        "surface": "demandée", "b": g["b"], "g": g["gagnees"],
                        "n_servi": g["n"], "seuil_ancien": g["seuil"],
                        "ancienne_tenue": g["b"] <= g["seuil"]})
    fichier = HERE / "results-recherche-approfondie-5530cba145.json"
    if fichier.exists():
        d = json.loads(fichier.read_text(encoding="utf-8"))
        for nom, v in d["verdicts"].items():
            out.append({"chantier": "recherche approfondie", "variante": nom,
                        "surface": "demandée", "b": v["perdues"], "g": v["gagnees"],
                        "n_servi": 76, "seuil_ancien": 5,
                        "ancienne_tenue": v["conditions"]["garde"],
                        "decisive": v.get("decisive")})
    return out


def rapport() -> dict:
    lignes = []
    for c in cas():
        verdict = tenue(c["b"], c["g"])
        requise = rotation_necessaire(verdict["part"])
        # Le banc que cette rotation supposerait, à taux de rotation constant : la rotation
        # observée rapportée à la population fixe de 155 donne le facteur d'échelle.
        banc = round(155 * requise / verdict["rotation"]) if requise else None
        lignes.append({**c, **verdict, "rotation_necessaire_a_part_egale": requise,
                       "banc_necessaire": banc})
    return {"tau": round(TAU, 4), "p_gain": round(P_GAIN, 4),
            "pre_enregistrement": "PRE-ENREGISTREMENT-GARDE-ROTATION-2026-09-08.md (dcb892c)",
            "cas": lignes, "appels_llm": 0}


def main() -> None:
    r = rapport()
    (HERE / "results-garde-rotation-5530cba145.json").write_text(
        json.dumps(r, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"τ = {r['tau']}   (p_gain = {r['p_gain']}, §19 bis)   estimateur : borne haute de Wilson")
    print(f"\n{'chantier':22s} {'variante':9s} {'b':>3s} {'g':>3s} {'rot':>4s} {'part':>6s} "
          f"{'Wilson↑':>8s} {'nouvelle':>9s} {'ancienne':>9s} {'rot. requise':>13s} {'banc':>7s}")
    for l in r["cas"]:
        req = l["rotation_necessaire_a_part_egale"]
        print(f"{l['chantier']:22s} {l['variante']:9s} {l['b']:3d} {l['g']:3d} {l['rotation']:4d} "
              f"{l['part']:6.3f} {l['wilson_haut']:8.4f} "
              f"{'TENUE' if l['tenue'] else 'ÉCHOUÉE':>9s} "
              f"{'TENUE' if l['ancienne_tenue'] else 'ÉCHOUÉE':>9s} "
              f"{(str(req) if req else '—'):>13s} "
              f"{(str(l['banc_necessaire']) if l['banc_necessaire'] else '—'):>7s}")
    n_new = sum(1 for l in r["cas"] if l["tenue"])
    n_old = sum(1 for l in r["cas"] if l["ancienne_tenue"])
    n_pt = sum(1 for l in r["cas"] if l["part"] <= TAU)
    print(f"\n  garde de rotation tenue : {n_new} / {len(r['cas'])}")
    print(f"  ancienne garde tenue    : {n_old} / {len(r['cas'])}")
    print(f"  DIAGNOSTIC — part ponctuelle <= τ, sans exigence de confiance : "
          f"{n_pt} / {len(r['cas'])}")
    print(f"\nécrit : rag/benchmark/results-garde-rotation-5530cba145.json")


if __name__ == "__main__":
    main()
