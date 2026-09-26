"""La fenêtre positionnelle, appliquée à l'**assemblage du contexte** — plafond hors ligne.

Deux formes de la même idée, et une seule est ouverte.

*En pool* : les voisins deviennent des candidats que le reranker doit scorer. Arbitrée par la
Phase 0 §12 et l'arithmétique de dilution de la Phase A §16 — 15/27 pour un pool de 274, gain
+0,006 contre −0,014 à −0,029 de dilution, ~87 s par question.

*En contexte* : les voisins accompagnent le passage qui a **déjà** gagné sa place. Aucun
candidat de plus, aucun reranking, aucune latence de scoring. C'est la forme mesurée ici.

Ce script ne mesure pas l'effet : il mesure son **plafond**, et il le fait sans un seul appel
LLM. La question est « pour combien des 26 ratés appariés le chunk d'or entre-t-il dans le
contexte, et à quel prix en taille de contexte ». Si le plafond est sous le seuil hérité du
chantier couverture, il n'y a pas de mesure à faire.

Protocole : ``rag/benchmark/RAPPORT-FENETRE-CONTEXTE-2026-09-05.md``.

    .venv/bin/python rag/benchmark/eval_fenetre_contexte.py --step plafond
    .venv/bin/python rag/benchmark/eval_fenetre_contexte.py --step plafond --sabotage
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import corpus_overlay  # noqa: E402
import experiment  # noqa: E402
import pipeline  # noqa: E402
from corpus import ChunkIndex  # noqa: E402

SIGNATURE = corpus_overlay.signature()
CACHE = HERE / ".cache"
VERDICTS = CACHE / f"answer-gap-verdicts-{SIGNATURE}.json"
REPONSES = CACHE / f"answer-gap-reponses-{SIGNATURE}.json"
OUTPUT = HERE / f"results-fenetre-contexte-{SIGNATURE}.json"

#: Hérités du chantier couverture (``998c84e``), sur la même population et le même juge.
#: Rien n'est choisi ici : c'est ce qui rend le plafond lisible sans pré-enregistrement neuf.
SEUIL_DELTA = 0.25
PLANCHER_DE_BRUIT = 0.167

FENETRES = (1, 2, 3, 5, 10)
ANCRES = (("1er passage servi", 1), ("2 premiers servis", 2),
          ("3 premiers servis", 3), ("les 5 servis", 5), ("pool dense@50 entier", None))


# ------------------------------------------------------------------------------- le socle

def ordre_documentaire(index: ChunkIndex) -> tuple[dict, dict]:
    """Position de chaque chunk dans son document, dans l'ordre de ``ChunkIndex.chunks``.

    Les dicts Python conservent l'ordre d'insertion et l'insertion suit ``rows.jsonl``. Le
    contrôle O le vérifie plutôt que de le supposer, comme la Phase 0 l'avait fait.
    """
    ordre: dict[str, list[str]] = defaultdict(list)
    for chunk_id, chunk in index.chunks.items():
        ordre[chunk["document_id"]].append(chunk_id)
    return ordre, {c: i for chunks in ordre.values() for i, c in enumerate(chunks)}


def socle(index: ChunkIndex) -> dict:
    """Les 26 ratés appariés du chantier couverture, leur contexte servi et leur pool."""
    verdicts = json.loads(VERDICTS.read_text(encoding="utf-8"))
    cles = sorted({k.rsplit("/", 1)[0] for k in verdicts if k.endswith("/oracle")})
    pools = experiment.cached_rankings()
    items = {item["key"]: item for item in experiment.load_items(index)}
    out = {}
    for key in cles:
        rows = experiment.rows_from_cache(pools[key]["dense"])
        servi = [r["chunk_id"] for r in pipeline.build_context(pipeline._with_text(rows, index))]
        # δ par question du chantier couverture : ORACLE − SERVI, sur la couverture notée 0-2.
        delta = verdicts[f"{key}/oracle"]["coverage"] - verdicts[f"{key}/servi"]["coverage"]
        out[key] = {"servi": servi, "pool": [r["chunk_id"] for r in rows],
                    "or": set(items[key]["gold_chunks"]), "delta_oracle": delta}
    return out


# ------------------------------------------------------------------------------ contrôles

def controle_S(cadre: dict) -> list[str]:
    """Le contexte servi reconstruit ici est **exactement** celui qu'a vu le chantier couverture."""
    if not REPONSES.exists():
        return [f"{REPONSES.name} absent"]
    servis = json.loads(REPONSES.read_text(encoding="utf-8"))
    ecarts = []
    for key, entree in cadre.items():
        attendu = servis.get(f"{key}/servi", {}).get("context")
        if attendu is None:
            ecarts.append(f"{key} · absent du cache de réponses")
        elif list(attendu) != entree["servi"]:
            ecarts.append(f"{key} · contexte reconstruit ≠ contexte servi")
    return ecarts


def controle_O(index: ChunkIndex, ordre: dict) -> list[str]:
    """``page_start`` croissante dans l'ordre de lecture — 317 des 319 documents en Phase 0."""
    conformes = 0
    for chunks in ordre.values():
        pages = [index.chunks[c].get("page_start") for c in chunks]
        pages = [p for p in pages if p is not None]
        conformes += pages == sorted(pages)
    if conformes < len(ordre) - 2:
        return [f"{conformes}/{len(ordre)} documents en ordre de page croissant "
                f"(la Phase 0 en mesurait {len(ordre) - 2})"]
    return []


# -------------------------------------------------------------------------------- mesures

def distances(cadre: dict, index: ChunkIndex, position: dict) -> list[int]:
    """Distance positionnelle de l'or au plus proche chunk du contexte servi."""
    mesurees = []
    for entree in cadre.values():
        meilleure = None
        for gold in entree["or"]:
            if gold not in position:
                continue
            document = index.chunks[gold]["document_id"]
            ancres = [position[c] for c in entree["servi"]
                      if c in position and index.chunks[c]["document_id"] == document]
            if ancres:
                ecart = min(abs(position[gold] - a) for a in ancres)
                meilleure = ecart if meilleure is None else min(meilleure, ecart)
        if meilleure is not None:
            mesurees.append(meilleure)
    return sorted(mesurees)


def configuration(cadre: dict, index: ChunkIndex, ordre: dict, position: dict,
                  n_ancres: int | None, fenetre: int) -> dict:
    """Une case de l'espace : combien d'atteintes, à quel prix en contexte, pour quel plafond."""
    atteintes, ajoutes, caracteres, base = [], [], [], []
    for key, entree in cadre.items():
        ancres = entree["pool"] if n_ancres is None else entree["servi"][:n_ancres]
        elargi = set()
        for chunk_id in ancres:
            chunk = index.chunks.get(chunk_id)
            if chunk is None:
                continue
            i = position[chunk_id]
            elargi.update(ordre[chunk["document_id"]][max(0, i - fenetre):i + fenetre + 1])
        elargi -= set(entree["servi"])
        ajoutes.append(len(elargi))
        caracteres.append(sum(len(index.chunks[c]["text"]) for c in elargi if c in index.chunks))
        base.append(sum(len(index.chunks[c]["text"]) for c in entree["servi"] if c in index.chunks))
        if elargi & entree["or"]:
            atteintes.append(key)
    n = len(cadre)
    socle_car = statistics.mean(base)
    return {
        "ancres": "pool" if n_ancres is None else n_ancres, "fenetre": fenetre,
        "atteintes": atteintes, "n_atteintes": len(atteintes),
        "chunks_ajoutes": round(statistics.mean(ajoutes), 1),
        "caracteres": round(socle_car + statistics.mean(caracteres)),
        "grossissement": round((socle_car + statistics.mean(caracteres)) / socle_car, 1),
        # Sans hypothèse : chaque atteinte gagne le grade maximum.
        "borne_arithmetique": round(len(atteintes) * 2 / n, 3),
        # Avec hypothèse : ce que ces questions ont gagné quand l'or leur était donné *en tête*.
        # La fenêtre le livre enfoui dans un contexte bien plus grand, donc ORACLE majore.
        "plafond_oracle": round(sum(max(0, cadre[k]["delta_oracle"]) for k in atteintes) / n, 3),
    }


def step_plafond(sabotage: bool) -> None:
    index = ChunkIndex.load()
    ordre, position = ordre_documentaire(index)
    if sabotage:
        # Mélanger l'ordre de lecture des documents : la voisinage positionnel devient
        # arbitraire. Les distances doivent exploser et les atteintes s'effondrer. Un
        # sabotage qui ne casse pas la structure qu'on prétend exploiter ne prouve rien.
        rng = __import__("random").Random(20260901)
        for document in ordre:
            rng.shuffle(ordre[document])
        position = {c: i for chunks in ordre.values() for i, c in enumerate(chunks)}
    cadre = socle(index)
    print(f"\n  corpus {len(index.chunks)} chunks · signature {SIGNATURE}")
    print(f"  {len(cadre)} ratés appariés (chantier couverture)")

    print("\n  === contrôles ===")
    dur = False
    for nom, ecarts in (("S — le contexte reconstruit est celui qui a été servi", controle_S(cadre)),
                        ("O — ordre de lecture ≡ ordre de pagination", controle_O(index, ordre))):
        if ecarts and not sabotage:
            dur = True
            print(f"    ✗ {nom} — {len(ecarts)} écart(s)")
            for ligne in ecarts[:4]:
                print(f"        {ligne}")
        elif ecarts:
            print(f"    ✓ {nom} — {len(ecarts)} écart(s), attendu sous sabotage")
        else:
            print(f"    ✓ {nom}")
    if dur:
        sys.exit("\nARRÊT : un contrôle a échoué.")

    docs_pool = sum(1 for e in cadre.values()
                    if {index.chunks[g]["document_id"] for g in e["or"] if g in index.chunks}
                    & {index.chunks[c]["document_id"] for c in e["pool"] if c in index.chunks})
    docs_servi = sum(1 for e in cadre.values()
                     if {index.chunks[g]["document_id"] for g in e["or"] if g in index.chunks}
                     & {index.chunks[c]["document_id"] for c in e["servi"] if c in index.chunks})
    print(f"\n  === le goulot ===")
    print(f"    document d'or présent dans le pool dense@50 (46 chunks) : {docs_pool}/{len(cadre)}")
    print(f"    document d'or présent dans le contexte servi (5 passages) : {docs_servi}/{len(cadre)}")

    d = distances(cadre, index, position)
    print(f"\n  distance positionnelle or ↔ contexte servi, sur les {len(d)} où elle est définie :")
    print(f"      {d}")
    if d:
        print(f"      médiane {statistics.median(d)}  (la Phase 0, ancrée sur le pool, mesurait 5)")

    grille = []
    print(f"\n  === l'espace de conception ===")
    print(f"  {'ancre':<24}{'W':>3}{'atteint':>10}{'contexte':>10}{'car.':>8}"
          f"{'arith.':>9}{'ORACLE':>9}")
    for libelle, n_ancres in ANCRES:
        for fenetre in FENETRES:
            case = configuration(cadre, index, ordre, position, n_ancres, fenetre)
            case["libelle"] = libelle
            grille.append(case)
            print(f"  {libelle:<24}{fenetre:>3}{case['n_atteintes']:>7}/{len(cadre):<2}"
                  f"{case['grossissement']:>9.1f}×{case['caracteres'] / 1000:>7.0f}k"
                  f"{case['borne_arithmetique']:>+9.3f}{case['plafond_oracle']:>+9.3f}")

    demandee = next(c for c in grille if c["ancres"] == 5 and c["fenetre"] == 5)
    passe = demandee["plafond_oracle"] >= SEUIL_DELTA
    servables = [c for c in grille if c["grossissement"] <= 10 and c["plafond_oracle"] >= SEUIL_DELTA]
    verdict = "GO — mesure à lancer" if passe or servables else "NO-GO"
    print(f"\n  === VERDICT : {verdict} ===")
    print(f"    configuration demandée (±5 sur les 5 servis) : {demandee['n_atteintes']}/{len(cadre)} "
          f"atteintes, contexte ×{demandee['grossissement']}, plafond {demandee['plafond_oracle']:+.3f}")
    print(f"    seuil hérité {SEUIL_DELTA:.3f} · plancher de bruit {PLANCHER_DE_BRUIT:.3f}")
    print(f"    configurations servables (contexte ≤ ×10) au-dessus du seuil : {len(servables)}")

    if sabotage:
        print("\n  (sabotage : ordre documentaire mélangé — les atteintes doivent s'effondrer)")
        return

    OUTPUT.write_text(json.dumps({
        "corpus": {"signature": SIGNATURE, "chunks": len(index.chunks)},
        "protocole": {"seuil_herite": SEUIL_DELTA, "plancher_de_bruit": PLANCHER_DE_BRUIT,
                      "herite_de": "998c84e (chantier couverture)",
                      "external_llm_calls": 0, "modeles_charges": [],
                      "n_rates_apparies": len(cadre)},
        "controles": {"S": "26/26 contexte identique au servi", "O": "ordre ≡ pagination"},
        "goulot": {"document_or_dans_le_pool": docs_pool,
                   "document_or_dans_le_contexte_servi": docs_servi},
        "distances_au_contexte_servi": d,
        "mediane": statistics.median(d) if d else None,
        "grille": grille,
        "verdict": {"verdict": verdict, "demandee": demandee,
                    "servables_au_dessus_du_seuil": len(servables)},
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  écrit : {OUTPUT.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step", required=True, choices=("plafond",))
    parser.add_argument("--sabotage", action="store_true",
                        help="mélange l'ordre documentaire : les atteintes doivent s'effondrer")
    args = parser.parse_args()
    step_plafond(args.sabotage)


if __name__ == "__main__":
    main()
