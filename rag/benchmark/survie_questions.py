"""Les questions v4 survivent-elles à un changement de corpus ?

Pourquoi ce module existe
-------------------------
Une famille de questions v4 n'est pas un texte : c'est un **ancrage** dans un corpus donné.
`gold_chunks` désigne des identifiants, l'or est vérifié *dans le texte* de ces chunks, et une
négative voisine repose sur une **absence** du corpus entier. Changez le corpus, et ces trois
appuis peuvent céder — sans que rien dans le fichier de questions ne bouge.

Le banc, lui, refusera bien de comparer deux runs de signatures différentes : `signature` est
l'un des cinq champs gouvernants. Mais refuser la comparaison ne dit pas **combien** de
questions sont mortes, ni lesquelles. C'est ce que ce module mesure, pour **zéro appel**.

Ce qu'il a trouvé la première fois qu'il a tourné
--------------------------------------------------
Au passage de ``5530cba145`` à ``e1bdf36e2e`` (correction des en-têtes de tableaux par le
chantier `reprise-ingestion`, 9 septembre 2026) : les 26 120 `chunk_id` sont **stables**,
`formula` survit 45/45 et `negative_voisine` 34/34 — mais **5 questions `table_cell` sur 100**
tombent, parce que leur tableau, lisible avant, ne l'est plus après.

Ce n'est pas un verdict sur la correction : sur les 989 chunks touchés, elle fait passer les
tableaux utilisables de **37 à 95**. Elle en casse **29** au passage, ce que les contrôles du
chantier voisin — cosinus texte/vecteur, BM25, ligne de base dense — ne pouvaient pas voir,
puisqu'aucun ne lit un tableau **en tant que tableau**.

    .venv/bin/python rag/benchmark/survie_questions.py --contre e1bdf36e2e
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import corpus_overlay  # noqa: E402
import familles_v4  # noqa: E402
import latex_norme  # noqa: E402
import norme_valeurs  # noqa: E402
import selection_tableaux  # noqa: E402

CACHE = HERE / ".cache"
SIGNATURE = corpus_overlay.signature()
FENETRE = familles_v4.FENETRE_SERVIE

#: Où chercher un index déjà construit pour une autre signature. Les worktrees frères en
#: portent un ; le reconstruire coûterait 40 s et 157 Mo de lecture pour rien.
WORKTREES = (HERE, Path.home() / "Rag-reprise" / "rag" / "benchmark",
             Path.home() / "Rag" / "rag" / "benchmark")


def charger_chunks(signature: str) -> dict:
    """Les chunks d'une signature, depuis le premier cache qui la porte."""
    for base in WORKTREES:
        chemin = base / ".cache" / f"chunk-index-v5-{signature}.pkl"
        if chemin.exists():
            with chemin.open("rb") as handle:
                return pickle.load(handle)["chunks"]
    sys.exit(f"aucun index en cache pour {signature} — cherché dans {[str(w) for w in WORKTREES]}")


def _tableau(texte: str):
    """(utilisable, motif) pour le tableau porté par ce texte."""
    analyse = selection_tableaux.analyser_tableau(texte)
    if not analyse:
        return False, "pas_de_tableau"
    return bool(analyse.utilisable), (analyse.motif or "utilisable")


def survie(apres: dict, avant: dict) -> dict:
    """Question par question : l'ancrage tient-il encore ?"""
    rapport: dict = {"signature_avant": None, "signature_apres": None, "appels_llm": 0}
    morts: dict[str, list] = {}

    # formula — l'or est une expression, vérifiée DANS la fenêtre servie.
    tombees = []
    items = familles_v4.lire_jsonl(HERE / familles_v4.FICHIERS["formula"])
    for item in items:
        chunk = apres.get(item["gold_chunks"][0])
        if chunk is None:
            tombees.append({"qid": item["qid"], "motif": "chunk d'or disparu"})
        elif not latex_norme.inclus(item["or_latex"], (chunk["text"] or "")[:FENETRE]):
            tombees.append({"qid": item["qid"], "motif": "or absent de la fenêtre servie"})
    morts["formula"] = tombees
    rapport["formula"] = {"n": len(items), "survivantes": len(items) - len(tombees)}

    # table_cell — l'or est une CELLULE : il ne suffit pas que la valeur soit là, il faut que
    # le tableau reste lisible et que la colonne demandée existe encore. C'est ce contrôle-là,
    # et lui seul, qui a vu la régression des en-têtes.
    tombees = []
    items = familles_v4.lire_jsonl(HERE / familles_v4.FICHIERS["table_cell"])
    for item in items:
        chunk = apres.get(item["gold_chunks"][0])
        if chunk is None:
            tombees.append({"qid": item["qid"], "motif": "chunk d'or disparu"})
            continue
        lisible, motif = _tableau((chunk["text"] or "")[:FENETRE])
        if not lisible:
            tombees.append({"qid": item["qid"], "motif": f"tableau devenu illisible ({motif})"})
    morts["table_cell"] = tombees
    rapport["table_cell"] = {"n": len(items), "survivantes": len(items) - len(tombees)}

    # negative_voisine — le risque est INVERSE : la grandeur absente peut être APPARUE.
    blob = " \n ".join((r["text"] or "").lower() for r in apres.values())
    tombees = []
    items = familles_v4.lire_jsonl(HERE / familles_v4.FICHIERS["negative_voisine"])
    for item in items:
        for entite in item["absent_entities"]:
            aiguille = " ".join(entite.strip().lower().split())
            if len(aiguille) >= 8 and aiguille in blob:
                tombees.append({"qid": item["qid"],
                                "motif": f"« {entite} » est apparue dans le corpus"})
                break
    morts["negative_voisine"] = tombees
    rapport["negative_voisine"] = {"n": len(items), "survivantes": len(items) - len(tombees)}

    rapport["mortes"] = morts
    rapport["total_mortes"] = sum(len(v) for v in morts.values())

    # Le corpus lui-même, pour situer ce que les questions subissent.
    ids_a, ids_b = set(avant), set(apres)
    touches = [c for c in ids_a & ids_b if avant[c]["text"] != apres[c]["text"]]
    perdus, gagnes = Counter(), 0
    for chunk_id in touches:
        ok_a, _ = _tableau(avant[chunk_id]["text"])
        ok_b, motif_b = _tableau(apres[chunk_id]["text"])
        if ok_a and not ok_b:
            perdus[motif_b] += 1
        elif ok_b and not ok_a:
            gagnes += 1
    rapport["corpus"] = {
        "chunks_avant": len(ids_a), "chunks_apres": len(ids_b),
        "identifiants_disparus": len(ids_a - ids_b), "identifiants_apparus": len(ids_b - ids_a),
        "textes_modifies": len(touches),
        "tableaux_perdus": sum(perdus.values()), "motifs_de_perte": dict(perdus),
        "tableaux_gagnes": gagnes,
    }
    return rapport


def main() -> None:
    analyse = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    analyse.add_argument("--contre", required=True,
                         help="signature du corpus à confronter aux questions courantes")
    analyse.add_argument("--depuis", default=SIGNATURE,
                         help="signature de référence (défaut : celle du worktree)")
    arguments = analyse.parse_args()

    avant = charger_chunks(arguments.depuis)
    apres = charger_chunks(arguments.contre)
    rapport = survie(apres, avant)
    rapport["signature_avant"] = arguments.depuis
    rapport["signature_apres"] = arguments.contre

    chemin = HERE / f"survie-questions-{arguments.depuis}-vers-{arguments.contre}.json"
    chemin.write_text(json.dumps(rapport, ensure_ascii=False, indent=1), encoding="utf-8")

    corpus = rapport["corpus"]
    print(f"\n{arguments.depuis} -> {arguments.contre} · appels LLM : 0")
    print(f"  identifiants : {corpus['identifiants_disparus']} disparus, "
          f"{corpus['identifiants_apparus']} apparus, {corpus['textes_modifies']} textes modifiés")
    print(f"  tableaux, sur les chunks touchés : +{corpus['tableaux_gagnes']} gagnés, "
          f"-{corpus['tableaux_perdus']} perdus {corpus['motifs_de_perte'] or ''}")
    print()
    for famille in ("formula", "table_cell", "negative_voisine"):
        bloc = rapport[famille]
        print(f"  {famille:18s} {bloc['survivantes']:3d}/{bloc['n']:3d} survivent")
        for mort in rapport["mortes"][famille][:8]:
            print(f"      {mort['qid']} — {mort['motif']}")
    print(f"\n{rapport['total_mortes']} question(s) à refaire avant de mesurer sur "
          f"{arguments.contre}. Écrit dans {chemin.name}.")


if __name__ == "__main__":
    main()
