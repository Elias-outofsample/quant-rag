"""Le verdict du chantier `contrat-de-reponse` : les critères pré-enregistrés, cochés un par un.

Pourquoi un module et pas un tableau écrit à la main
-----------------------------------------------------
Les critères ont été fixés le 9 septembre 2026 avant le premier appel
(``PRE-ENREGISTREMENT-REPONSE-2026-09-09.md``). Les recopier à la main dans un rapport, c'est
se donner la possibilité de les arrondir. Ils sont donc **écrits ici, en dur**, avec leur seuil
et leur taux de fausse alarme, et le module dit lui-même lesquels sont tenus.

Il n'appelle rien : il lit deux ``results-v4-*.json`` produits par ``banc_v4`` et les compare.

    .venv/bin/python rag/benchmark/verdict_reponse.py \
        --candidat rag/benchmark/results-v4-e1bdf36e2e-v3.json \
        --reference rag/benchmark/results-v4-e1bdf36e2e-v2.json

Ce que ce module refuse de faire
---------------------------------
**Il ne conclut pas GO à partir des seules métriques de citation.** Le contrat v3 optimise
directement « part sans citation » et « précision » : exiger un extrait verbatim met le chiffre
cité dans la citation, donc dans le passage. La règle de lecture du §7 du pré-enregistrement est
appliquée telle quelle — C1 (fabrications) tranche, C3 (non-régression) peut annuler, et un gain
de C2 seul se conclut par un rapport de forme, pas par un GO.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from math import comb
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from audit_fenetre_generateur import wilson as _wilson  # noqa: E402  — l'unique IC du dépôt


def wilson(succes: int, n: int) -> str:
    """L'IC de Wilson du dépôt, rendu lisible. Écrire un second calcul aurait produit deux
    chiffres qui divergent ; le formater ici n'en produit aucun."""
    if not n:
        return "—"
    bas, haut = _wilson(succes, n)
    return f"[{bas:.4f} ; {haut:.4f}]"

#: Ligne de base [lot 4], relevée avant la mesure. Chaque entrée : (valeur, n).
BASE = {
    "fabrications_nv": (15, 34),
    "sans_citation": (60, 153),
    "precision": (82, 93),
    "mauvais_passage": (1, 93),
    "table_cell_si_or": (78, 93),
    "formula_si_or": (5, 15),
    "formula_balisee_si_or": (5, 15),
    "extraits_retrouves": (7, 16),
}


def _binom_le(k: int, n: int, p: float) -> float:
    return sum(comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(0, min(k, n) + 1))


def _famille(resultat: dict, nom: str) -> dict:
    return resultat["par_famille"].get(nom) or {}


def _lignes(resultat: dict, famille: str) -> list[dict]:
    return [l for l in resultat["detail"].values() if l["famille"] == famille]


def _si_or_servi(resultat: dict, famille: str, champ: str = "score") -> tuple[int, int]:
    servis = [l for l in _lignes(resultat, famille) if l["or_servi"]]
    return sum(1 for l in servis if l.get(champ)), len(servis)


def _citations(resultat: dict, population: str = "positives") -> dict:
    bloc = resultat["citations"]
    return bloc.get(population) or bloc


def critere(nom: str, tenu: bool | None, avant: str, apres: str, seuil: str,
            fausse_alarme: str = "—", note: str = "") -> dict:
    return {"critere": nom, "tenu": tenu, "avant": avant, "apres": apres, "seuil": seuil,
            "fausse_alarme": fausse_alarme, "note": note}


def evaluer(candidat: dict, reference: dict) -> dict:
    """Les critères du pré-enregistrement, chacun avec ce qui le tient ou le manque."""
    sortie = []

    # --- C1 — le critère principal, et le seul que la forme du contrat ne peut pas satisfaire.
    contraste = ((candidat.get("comparaison") or {}).get("par_famille") or {})
    fab = (contraste.get("negative_voisine") or {}).get("fabrications") or {}
    ic = fab.get("ic95")
    fab_c = _famille(candidat, "negative_voisine").get("fabrications")
    fab_r = _famille(reference, "negative_voisine").get("fabrications")
    n_nv = _famille(candidat, "negative_voisine").get("n")
    sortie.append(critere(
        "C1 — fabrications negative_voisine (IC apparié exclut zéro)",
        None if ic is None else ic[1] < 0,
        f"{fab_r}/{n_nv}", f"{fab_c}/{n_nv}",
        "Δ < 0, IC95 apparié excluant zéro",
        note=(f"Δ {fab['delta']:+.4f} IC95 [{ic[0]:+.4f} ; {ic[1]:+.4f}] "
              f"(−{fab['disparues']} / +{fab['apparues']})" if ic else "contraste absent")))

    # --- C2 — la traçabilité. Trois seuils du handoff, dont un remplacé avant la mesure.
    for population in ("positives", "toutes_familles"):
        av, ap = _citations(reference, population), _citations(candidat, population)
        suffixe = "" if population == "positives" else " [199 questions, population neuve]"
        sortie.append(critere(
            f"C2a — affirmations sans citation{suffixe}",
            ap["part_sans_citation"] <= 0.15 if population == "positives" else None,
            f"{av['part_sans_citation']:.4f}", f"{ap['part_sans_citation']:.4f}",
            "≤ 0,15" if population == "positives" else "—",
            "1,5e-11" if population == "positives" else "—",
            f"{av['comptes'].get('sans_citation')}/{av['affirmations']} → "
            f"{ap['comptes'].get('sans_citation')}/{ap['affirmations']}"))
        if population == "positives":
            sortie.append(critere(
                "C2b — précision des citations", ap["precision_des_citations"] >= 0.90,
                f"{av['precision_des_citations']:.4f}", f"{ap['precision_des_citations']:.4f}",
                "≥ 0,90", "atteint par hasard 32,7 %",
                f"IC95 Wilson {wilson(ap['comptes']['appuyee'], ap['comptes']['appuyee'] + ap['comptes']['mauvais_passage'] + ap['comptes']['absente'])}"))
            sortie.append(critere(
                "C2c — citations au mauvais passage",
                ap["comptes"].get("mauvais_passage", 0) <= 2,
                str(av["comptes"].get("mauvais_passage", 0)),
                str(ap["comptes"].get("mauvais_passage", 0)),
                "≤ 2 (le « = 0 » du handoff a 63,4 % de fausse alarme)", "7,9 %"))
            sortie.append(critere(
                "C2d — extraits verbatim retrouvés", None,
                f"{av['extraits_retrouves']}/{av['extraits_verbatim']}",
                f"{ap['extraits_retrouves']}/{ap['extraits_verbatim']}",
                "diagnostic — la seule métrique de citation non circulaire", "—",
                "un générateur qui cite beaucoup et cite faux fait baisser ce taux"))

    # --- C3 — la non-régression de la réponse. Elle peut annuler C1.
    tc_c, tc_n = _si_or_servi(candidat, "table_cell")
    tc_r, tc_rn = _si_or_servi(reference, "table_cell")
    sortie.append(critere(
        "C3a — table_cell, score si or servi", tc_c / tc_n >= 0.80 if tc_n else None,
        f"{tc_r}/{tc_rn} = {tc_r / tc_rn:.4f}", f"{tc_c}/{tc_n} = {tc_c / tc_n:.4f}",
        "≥ 0,80", "16,1 %", f"IC95 Wilson {wilson(tc_c, tc_n)}"))
    f_c, f_n = _si_or_servi(candidat, "formula")
    f_r, f_rn = _si_or_servi(reference, "formula")
    sortie.append(critere(
        "C3b — formula, score si or servi", None,
        f"{f_r}/{f_rn} = {f_r / f_rn:.4f}", f"{f_c}/{f_n} = {f_c / f_n:.4f}",
        "diagnostic — aucun seuil utile à n = 15", "40,4 %",
        f"IC95 Wilson {wilson(f_c, f_n)}"))
    for famille in ("formula", "table_cell"):
        bloc = contraste.get(famille) or {}
        if not bloc:
            continue
        sortie.append(critere(
            f"C3c — {famille}, Δ apparié du score et garde d'abstention",
            bloc["delta"] >= -0.02 and bloc["garde_abstention"] and bloc["garde_degradation"],
            "—", f"Δ {bloc['delta']:+.4f}",
            "Δ ≥ −0,02 et a ≤ 2 et b ≤ 6", "—",
            f"IC95 [{bloc['ic95'][0]:+.4f} ; {bloc['ic95'][1]:+.4f}] · a={bloc['a']:+d} "
            f"b={bloc['b']} · {bloc['montent']}↑ {bloc['descendent']}↓"))
    nv3 = _famille(candidat, "negative_v3")
    sortie.append(critere(
        "C3d — negative_v3, abstentions", nv3.get("abstentions") == nv3.get("n"),
        f"{_famille(reference, 'negative_v3').get('abstentions')}/20",
        f"{nv3.get('abstentions')}/{nv3.get('n')}", "= 20/20", "—",
        "le témoin d'abstention déjà calibré : toute perte est une régression"))

    # --- C4 — la forme, et la sur-abstention déguisée.
    b_c, b_n = _si_or_servi(candidat, "formula", "balisee")
    b_r, b_rn = _si_or_servi(reference, "formula", "balisee")
    sortie.append(critere(
        "C4a — formula balisée si or servi", b_c / b_n >= 0.50 if b_n else None,
        f"{b_r}/{b_rn} = {b_r / b_rn:.4f}", f"{b_c}/{b_n} = {b_c / b_n:.4f}",
        "≥ 0,50", "atteint par hasard 8,8 %", f"IC95 Wilson {wilson(b_c, b_n)}"))
    signal = candidat.get("signal_absence") or {}
    nv = signal.get("negative_voisine") or {}
    positives_signal = sum((signal.get(f) or {}).get("signale_absence_si_or_servi", 0)
                           for f in ("formula", "table_cell"))
    sortie.append(critere(
        "C4b — NOT_IN_SOURCES là où il est attendu", None,
        "0 (v2 ne peut pas l'émettre)",
        f"{nv.get('signale_absence', 0)}/{nv.get('n', 0)} négatives voisines",
        "diagnostic", "—", "le signal que le contrat v2 rendait indicible"))
    sortie.append(critere(
        "C4c — NOT_IN_SOURCES sur une positive dont l'or est servi",
        positives_signal == 0, "0", str(positives_signal),
        "= 0 — c'est une sur-abstention déguisée", "—",
        "la réponse était là et le système a déclaré une absence (risque du §19 bis)"))

    return {"criteres": sortie,
            "principal_tenu": sortie[0]["tenu"],
            "regressions": [c["critere"] for c in sortie if c["tenu"] is False]}


def imprimer(bilan: dict, candidat: dict, reference: dict) -> None:
    print(f"\ncontrat candidat : {candidat['version_prompt']['nom']} "
          f"({candidat['version_prompt']['sha256_16']})   contre   "
          f"{reference['version_prompt']['nom']} ({reference['version_prompt']['sha256_16']})")
    print(f"corpus {candidat['signature']} · fenêtre servie {candidat['fenetre_servie']} · "
          f"fenêtre du juge {candidat['fenetre_du_juge']} · population {candidat['population']}\n")
    marque = {True: "TENU  ", False: "MANQUÉ", None: "  —   "}
    for c in bilan["criteres"]:
        print(f"[{marque[c['tenu']]}] {c['critere']}")
        print(f"           {c['avant']:>22s}  →  {c['apres']:<22s} seuil {c['seuil']}")
        if c["fausse_alarme"] != "—":
            print(f"           fausse alarme : {c['fausse_alarme']}")
        if c["note"]:
            print(f"           {c['note']}")
    print(f"\ncritère principal (C1) : "
          f"{'TENU' if bilan['principal_tenu'] else 'NON TENU'}")
    if bilan["regressions"]:
        print("critères manqués : " + " · ".join(bilan["regressions"]))
    cout_c = (candidat.get("cout") or {}).get("total_eur_majore")
    print(f"\ncoût du bras candidat, run entier : {cout_c} € (majoré, 1 USD = 1 EUR)")


def main() -> None:
    analyse = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    analyse.add_argument("--candidat", type=Path, required=True)
    analyse.add_argument("--reference", type=Path, required=True)
    analyse.add_argument("--sortie", type=Path,
                         default=HERE / "verdict-contrat-reponse-v3.json")
    arguments = analyse.parse_args()
    candidat = json.loads(arguments.candidat.read_text(encoding="utf-8"))
    reference = json.loads(arguments.reference.read_text(encoding="utf-8"))
    bilan = evaluer(candidat, reference)
    imprimer(bilan, candidat, reference)
    arguments.sortie.write_text(json.dumps(
        {"candidat": str(arguments.candidat), "reference": str(arguments.reference),
         "pre_enregistrement": "PRE-ENREGISTREMENT-REPONSE-2026-09-09.md", **bilan},
        ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"écrit : {arguments.sortie}")


if __name__ == "__main__":
    main()
