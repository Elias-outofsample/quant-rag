"""Ré-extraire les chunks dont le TEXTE a changé — sans toucher à l'ordre de ``results.jsonl``.

Le problème
------------
``results.jsonl`` s'écrit en **ajout** et ``extract_entities.py`` ne repasse jamais sur un
``chunk_id`` déjà vu, quel que soit le changement de son texte. Un overlay qui corrige un
passage laisse donc le graphe décrire un texte que le corpus ne sert plus — la divergence
silencieuse que ce dossier s'interdit.

Pourquoi l'ordre compte, et pourquoi il faut le rendre
--------------------------------------------------------
``build_graph.build`` résout les extrémités de ses arêtes de relation par
``lookup.setdefault`` : **le premier label rencontré gagne** (``build_graph.build``, ``lookup.setdefault``), et
9,3 % des clés d'entité portent au moins deux labels. L'ordre des lignes de ``results.jsonl``
décide donc vers quel nœud pointe une relation.

Or retirer des lignes puis relancer l'extraction les remet **en queue** de fichier. Le graphe
changerait alors pour deux raisons mêlées : le texte corrigé, et le déplacement. On ne
saurait plus laquelle. Ce module **rend l'ordre d'origine** après la ré-extraction : le seul
écart qui subsiste est le contenu des lignes visées.

    .venv/bin/python rag/graph/reextraire_chunks.py --chunks liste.json          # à blanc
    .venv/bin/python rag/graph/reextraire_chunks.py --chunks liste.json --appliquer
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

RESULTATS = ROOT / "data" / "graph" / "gliner-results" / "results.jsonl"
PY_GLINER = ROOT / ".venv-gliner" / "bin" / "python"
#: Mêmes tranches que ``batch_driver`` : la fuite mémoire de GLiNER est dans la DURÉE d'un
#: run, pas dans les chunks (18 Gio réclamés au 216e chunk d'un même processus, le
#: 5 septembre 2026). Chaque tranche est un processus neuf, qui rend sa mémoire en sortant.
TRANCHE = 120
LOT = 4
TRANCHES_MAX = 80


def lignes(chemin: Path) -> list[tuple[str, str]]:
    """``(chunk_id, ligne brute)`` dans l'ordre du fichier. La ligne brute est conservée
    telle quelle : la réécrire par ``json.dumps`` changerait des octets sans changer le sens,
    et le contrôle d'identité ne saurait plus distinguer les deux."""
    sortie = []
    for ligne in chemin.read_text(encoding="utf-8").splitlines():
        if not ligne.strip():
            continue
        sortie.append((json.loads(ligne)["chunk_id"], ligne))
    return sortie


def main() -> None:
    parseur = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parseur.add_argument("--chunks", type=Path, required=True,
                         help="JSON : une liste de chunk_id, ou un objet portant 'chunks'")
    parseur.add_argument("--appliquer", action="store_true")
    parseur.add_argument("--resultats", type=Path, default=RESULTATS)
    arguments = parseur.parse_args()

    charge = json.loads(arguments.chunks.read_text(encoding="utf-8"))
    cibles = set(charge if isinstance(charge, list) else charge["chunks"])
    avant = lignes(arguments.resultats)
    ordre = [chunk_id for chunk_id, _ in avant]
    presents = cibles & set(ordre)
    print(f"{len(avant)} lignes dans {arguments.resultats.name}")
    print(f"{len(cibles)} chunks visés, dont {len(presents)} présents dans le fichier")
    if len(presents) != len(cibles):
        print(f"  ATTENTION {len(cibles) - len(presents)} chunk(s) visé(s) sont ABSENTS : "
              "ils n'ont jamais été extraits, l'extraction les prendra normalement")
    if not arguments.appliquer:
        print("\n--appliquer absent : rien n'a été écrit.")
        return

    depart = time.perf_counter()
    conserve = [(chunk_id, ligne) for chunk_id, ligne in avant if chunk_id not in cibles]
    arguments.resultats.write_text("".join(ligne + "\n" for _c, ligne in conserve),
                                   encoding="utf-8")
    print(f"{len(avant) - len(conserve)} ligne(s) retirée(s) — {len(conserve)} restantes")

    for tranche in range(1, TRANCHES_MAX + 1):
        processus = subprocess.run(
            [str(PY_GLINER), str(HERE / "extract_entities.py"),
             "--limit", str(TRANCHE), "--batch-size", str(LOT)],
            capture_output=True, text=True, cwd=str(ROOT))
        if processus.returncode != 0:
            sys.exit(f"extract_entities (tranche {tranche}) : {processus.stderr.strip()[-500:]}")
        etat = {}
        for ligne in reversed(processus.stdout.splitlines()):
            if ligne.startswith("{"):
                try:
                    etat = json.loads(processus.stdout[processus.stdout.rindex("\n{") + 1:])
                except Exception:
                    etat = {}
                break
        fait = etat.get("completed")
        print(f"  tranche {tranche} : {fait}/{etat.get('total')} "
              f"({etat.get('processed_this_run')} chunks)")
        if etat.get("status") == "COMPLETED":
            break
        if not etat.get("processed_this_run"):
            sys.exit(f"extract_entities n'avance plus après {tranche} tranche(s)")
    else:
        sys.exit(f"{TRANCHES_MAX} tranches sans arriver au bout")

    # RENDRE L'ORDRE. Sans cela, le graphe changerait pour deux raisons mêlées.
    apres = dict(lignes(arguments.resultats))
    manquants = [c for c in ordre if c not in apres]
    if manquants:
        sys.exit(f"{len(manquants)} chunk(s) n'ont pas été ré-extraits : {manquants[:5]}")
    nouveaux = [c for c in apres if c not in set(ordre)]
    arguments.resultats.write_text(
        "".join(apres[c] + "\n" for c in ordre) + "".join(apres[c] + "\n" for c in nouveaux),
        encoding="utf-8")

    final = lignes(arguments.resultats)
    identiques = sum(1 for (a, la), (b, lb) in zip(avant, final) if a == b and la == lb)
    print(f"ordre rendu : {len(final)} lignes, {identiques} identiques à l'octet près, "
          f"{len(final) - identiques} changées")
    print(json.dumps({"lignes_avant": len(avant), "lignes_apres": len(final),
                      "cibles": len(cibles), "reextraits": len(presents),
                      "lignes_identiques": identiques,
                      "ordre_conserve": [c for c, _ in avant] == [c for c, _ in final][:len(avant)],
                      "secondes": round(time.perf_counter() - depart, 1)},
                     ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
