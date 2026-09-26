"""Le contrôle de ``verify_citation`` : cent citations, dont cinquante fausses d'un seul mot.

Un vérificateur qui dit « oui » à tout est inutile, et un vérificateur qui dit « oui » à une
citation dont un mot a changé est **pire qu'inutile** : il donne une garantie à une déformation.
Le contrôle mesure donc les deux sens, sur du matériau réel — des extraits tirés des passages
que le serveur a effectivement servis aux 155 questions du banc.

L'altération est d'**un seul mot**, choisi au milieu de l'extrait et remplacé par un autre mot
du même corpus. C'est le cas le plus difficile : une différence d'un mot sur cinquante laisse
une ressemblance de 98 %, et c'est exactement le seuil où un vérificateur trop tolérant se
trompe.

    .venv/bin/python rag/benchmark/controle_citation.py

Aucun appel LLM. Aucune écriture hors du fichier de résultats demandé.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import citation  # noqa: E402
import corpus_overlay  # noqa: E402
import llm  # noqa: E402  — pour le bloc de dépense, jamais pour un appel
import quant_rag  # noqa: E402

#: Graine fixe : le contrôle doit rendre le même verdict à chaque exécution, sans quoi un échec
#: ne serait pas reproductible et ne prouverait rien.
GRAINE = 20260908

REMPLACANTS = ["volatility", "estimator", "portfolio", "liquidity", "drawdown", "kurtosis",
               "hedging", "arbitrage", "momentum", "covariance"]


def passages_servis(nombre: int, graine: int = GRAINE) -> list[dict]:
    """Des passages réellement servis, tirés au sort dans la collection servie."""
    points, _ = quant_rag.client().scroll(collection_name=quant_rag.COLLECTION,
                                          limit=4096, with_payload=True)
    candidats = [p.payload for p in points
                 if len(p.payload.get("text") or "") > 900
                 and p.payload.get("doc_text_sha256")
                 and p.payload.get("ancrage_granularite") == "exacte"]
    return random.Random(graine).sample(candidats, min(nombre, len(candidats)))


def extraire(texte: str, mots: int = 30, graine: int = 0) -> str:
    """Un extrait de ``mots`` mots pris au milieu du passage, comme un lecteur le copierait."""
    decoupe = texte.split()
    if len(decoupe) <= mots:
        return texte
    debut = random.Random(graine).randrange(0, len(decoupe) - mots)
    return " ".join(decoupe[debut:debut + mots])


def alterer(extrait: str, graine: int = 0) -> tuple[str, str, str]:
    """Change **un** mot. Rend ``(extrait altéré, mot d'origine, mot substitué)``."""
    hasard = random.Random(graine)
    mots = extrait.split()
    positions = [i for i, m in enumerate(mots) if len(m) > 4 and m.isalpha()]
    if not positions:
        positions = list(range(1, max(len(mots) - 1, 2)))
    i = hasard.choice(positions)
    origine = mots[i]
    remplacant = hasard.choice([m for m in REMPLACANTS if m.lower() != origine.lower()])
    mots[i] = remplacant
    return " ".join(mots), origine, remplacant


def controler(nombre: int = 50) -> dict:
    echantillon = passages_servis(nombre)
    vrais, faux, latences = [], [], []

    for rang, payload in enumerate(echantillon):
        extrait = extraire(payload["text"], graine=rang)
        debut = time.perf_counter()
        verdict = citation.verify_citation(payload["document_id"], extrait, payload["chunk_id"])
        latences.append((time.perf_counter() - debut) * 1000)
        vrais.append({"document_id": payload["document_id"], "chunk_id": payload["chunk_id"],
                      "trouve": verdict.get("trouve"), "source": verdict.get("source"),
                      "offsets": verdict.get("offsets"),
                      "extrait": extrait[:120]})

        altere, origine, remplacant = alterer(extrait, graine=rang)
        debut = time.perf_counter()
        verdict = citation.verify_citation(payload["document_id"], altere, payload["chunk_id"])
        latences.append((time.perf_counter() - debut) * 1000)
        faux.append({"document_id": payload["document_id"], "chunk_id": payload["chunk_id"],
                     "trouve": verdict.get("trouve"),
                     "mot_remplace": f"{origine} -> {remplacant}",
                     "ressemblance": (verdict.get("plus_proche") or {}).get("ressemblance"),
                     "mots_differents": (verdict.get("plus_proche") or {}).get("mots_differents")})

    latences.sort()
    retrouves = sum(1 for v in vrais if v["trouve"])
    acceptes = sum(1 for f in faux if f["trouve"])
    voisinage = sum(1 for f in faux if f["ressemblance"] is not None)
    return {
        "corpus_signature": corpus_overlay.signature(),
        "depense": llm.tracabilite_run(),
        "extraits_reels": {"soumis": len(vrais), "retrouves": retrouves,
                           "attendu": f"{len(vrais)}/{len(vrais)}"},
        "extraits_alteres_d_un_mot": {"soumis": len(faux), "acceptes": acceptes,
                                      "attendu": f"0/{len(faux)}",
                                      "voisinage_signale": voisinage},
        "latence_ms": {"p50": round(latences[len(latences) // 2], 1),
                       "p95": round(latences[int(0.95 * len(latences))], 1),
                       "max": round(latences[-1], 1)},
        "verdict": ("CONFORME" if retrouves == len(vrais) and acceptes == 0 else "ÉCHEC"),
        "detail_reels": vrais,
        "detail_alteres": faux,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--nombre", type=int, default=50)
    parser.add_argument("--sortie", type=Path, default=None)
    args = parser.parse_args()

    resultat = controler(args.nombre)
    print(f"verdict : {resultat['verdict']}")
    print(f"  extraits réels           : {resultat['extraits_reels']['retrouves']}"
          f"/{resultat['extraits_reels']['soumis']} retrouvés")
    print(f"  extraits altérés d'un mot: {resultat['extraits_alteres_d_un_mot']['acceptes']}"
          f"/{resultat['extraits_alteres_d_un_mot']['soumis']} acceptés"
          f" ({resultat['extraits_alteres_d_un_mot']['voisinage_signale']} voisinages signalés)")
    print(f"  latence p50/p95/max      : {resultat['latence_ms']['p50']}"
          f" / {resultat['latence_ms']['p95']} / {resultat['latence_ms']['max']} ms")
    if args.sortie:
        args.sortie.write_text(json.dumps(resultat, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"  écrit : {args.sortie}")


if __name__ == "__main__":
    main()
