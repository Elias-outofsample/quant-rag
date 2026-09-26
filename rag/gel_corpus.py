"""Le gel du corpus — la garde qui manquait, et le défaut qu'elle empêche.

Le défaut, mesuré le 6 septembre 2026
-------------------------------------
Le corpus est passé de 262 à 418 documents en 72 heures, sept index BM25 construits pour
la seule matinée du 6. Deux conséquences, toutes deux mécaniques, toutes deux constatées :

1. **Un Δ apparié entre deux signatures ne mesure pas ce qu'il prétend.** Comparer une
   base à 322 documents à un traitement à 418 confond le chantier avec 96 concurrents
   ajoutés. Le biais a un signe — plus de concurrents, known-item plus bas — donc un bon
   découpage peut être rejeté pour une raison qui n'est pas la sienne.
2. **Le shadow ne peut pas se remplir.** Il écarte en ``autre_signature`` ; pendant le lot,
   la signature vivait environ cinq minutes. Le compteur serait resté à 0/50 même serveur
   allumé.

La règle
--------
    **Base et traitement d'un chantier mesuré portent la même signature de corpus.**

Ce n'est pas une politesse : c'est la condition d'existence du test apparié, qui est
l'instrument de décision de tout le dossier.

⚠ Ce que la garde fait — et ce qu'elle ne fait PAS
---------------------------------------------------
**Ce module est un détecteur de dérive, pas un verrou.** La distinction a été relevée le
8 septembre 2026 par le chantier ``reprise-ingestion``, et deux phrases de cette docstring
étaient fausses. Elles sont corrigées ici ; **le comportement, lui, n'a pas été touché**.

``etat()`` calcule ``accorde = (not actif) or signature_gelée == signature_vivante``, et
``verifier()`` ne quitte que si ``accorde`` est faux. Autrement dit :

- gel actif, signatures **égales** : ``accorde`` vaut **True**, et ``batch_driver --go``
  **passe**. C'est l'état du dépôt au 8 septembre 2026 ;
- gel actif, signatures **différentes** : elle échoue bruyamment, en nommant le chantier
  protégé et en disant comment lever le gel. Mais à ce moment-là **le corpus a déjà bougé** :
  la garde constate la dérive, elle ne l'a pas empêchée ;
- gel inactif : elle ne dit rien.

« Tant qu'un gel est actif, rien n'entre au corpus » est donc **faux**, et l'était depuis
l'origine. Le corriger en verrou serait un changement de comportement : il interdirait de
rejouer un lot interrompu, qui est précisément ce qu'un gel actif doit permettre. La
décision est laissée à l'utilisateur (§5 de
``rag/benchmark/PRE-ENREGISTREMENT-REPRISE-2026-09-08.md``).

Qui consulte la garde — relevé le 10 septembre 2026
---------------------------------------------------
Vérifié par ``grep -rn "gel_corpus.verifier(" rag``, **trois appelants**, tous sur un
chemin qui écrit dans le corpus servi :

- ``ingestion/batch_driver.py`` — sous ``if arguments.go:`` ; ``--dry-run`` passe ;
- ``ingestion/apply_delivery.py`` — au début de ``apply()``, **avant** le verrou de
  collection ; ``--plan`` n'écrit rien et ne passe pas par là ;
- ``ingestion/retire_document.py`` — après le constat, sur le seul chemin ``--appliquer``.

``rollback()`` de ``apply_delivery.py`` **ne consulte pas le gel, et ne le consultera
jamais** : un gel non accordé veut dire que la signature a déjà dérivé, et l'annulation est
justement le geste qui la remet d'aplomb. La bloquer derrière la garde serait un piège.

Jusqu'au 10 septembre 2026, ``batch_driver.py`` était le seul appelant : ``apply_delivery``
et ``retire_document`` appelés directement écrivaient sous gel sans un mot, alors qu'ils
réécrivent le registre, ``duplicates-v1.json`` et la collection. Le lot ``cloture-
operationnelle`` a ajouté les deux appels manquants — **sans toucher au comportement de la
garde elle-même**, qui reste un détecteur de dérive.

Les deux entrées du banc (``eval_router.main``, ``eval_dense_candidat.bras``) n'appellent
que ``etat()``, pour l'inscrire dans leurs résultats ; ``rechunk_corpus.py`` et
``verdict_representation.py`` n'appellent que ``lire()``.

Lever ou poser le gel
---------------------
    .venv/bin/python rag/gel_corpus.py --etat
    .venv/bin/python rag/gel_corpus.py --geler "chantier découpage — contextualisation"
    .venv/bin/python rag/gel_corpus.py --degeler "lot arXiv de septembre"

Le fichier est versionné : poser et lever un gel laisse une trace datée dans l'historique,
comme un pré-enregistrement.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
from datetime import datetime, timezone

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import corpus_overlay  # noqa: E402

DECLARATION = HERE / "gel-corpus.json"


def _maintenant() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def lire() -> dict:
    if not DECLARATION.exists():
        return {"actif": False, "signature": None, "chantier": None, "historique": []}
    return json.loads(DECLARATION.read_text(encoding="utf-8"))


def ecrire(etat: dict) -> None:
    DECLARATION.write_text(json.dumps(etat, ensure_ascii=False, indent=1) + "\n",
                           encoding="utf-8")


def etat() -> dict:
    """Où en est le gel, sans rien exiger."""
    declaration = lire()
    vivante = corpus_overlay.signature()
    return {"actif": declaration["actif"], "signature_gelee": declaration["signature"],
            "signature_vivante": vivante, "chantier": declaration.get("chantier"),
            "depuis": declaration.get("depuis"),
            "accorde": (not declaration["actif"]) or declaration["signature"] == vivante}


def verifier(action: str = "cette action", sortir: bool = True) -> bool:
    """Vraie si l'action est permise. Échoue bruyamment sinon.

    ``sortir=False`` rend un booléen au lieu de quitter : c'est ce dont les tests ont
    besoin, et une garde qu'on ne peut pas faire échouer à volonté ne prouve rien.
    """
    e = etat()
    if e["accorde"]:
        return True
    message = (
        f"\n  GEL DU CORPUS ACTIF — {action} est refusée.\n"
        f"    chantier protégé : {e['chantier']}\n"
        f"    gelé depuis      : {e['depuis']}\n"
        f"    signature gelée  : {e['signature_gelee']}\n"
        f"    signature vivante: {e['signature_vivante']}\n"
        f"\n  Base et traitement d'un chantier mesuré portent la même signature. Faire\n"
        f"  entrer un document maintenant rendrait le Δ apparié ininterprétable : il\n"
        f"  confondrait le chantier avec les documents ajoutés.\n"
        f"\n  Lever le gel, si c'est bien ce que vous voulez :\n"
        f"    .venv/bin/python rag/gel_corpus.py --degeler \"raison\"\n")
    if sortir:
        sys.exit(message)
    print(message, file=sys.stderr)
    return False


def annoncer() -> None:
    """Une ligne, pour que le gel ne soit jamais une surprise dans une sortie de banc."""
    e = etat()
    if e["actif"]:
        accord = "accordée" if e["accorde"] else "DÉSACCORDÉE"
        print(f"  gel     actif sur {e['signature_gelee']} ({e['chantier']}) — "
              f"signature vivante {accord}")


def geler(chantier: str, forcer: bool = False) -> dict:
    """Pose le gel sur la signature vivante.

    Refuse d'écraser un gel actif : un second gel posé pendant qu'une mesure court
    remplacerait silencieusement le chantier protégé, et la trace mentirait sur ce qui
    était gelé au moment de la mesure. ``forcer=True`` le permet, explicitement.
    """
    declaration = lire()
    if declaration.get("actif") and not forcer:
        sys.exit(
            f"\n  UN GEL EST DÉJÀ ACTIF — refus d'en poser un second en silence.\n"
            f"    chantier protégé : {declaration.get('chantier')}\n"
            f"    depuis           : {declaration.get('depuis')}\n"
            f"    signature gelée  : {declaration.get('signature')}\n"
            f"\n  Lever d'abord le gel en cours, ou forcer si c'est bien l'intention :\n"
            f"    .venv/bin/python rag/gel_corpus.py --geler \"…\" --forcer\n")
    signature = corpus_overlay.signature()
    declaration.setdefault("historique", []).append(
        {"a": _maintenant(), "acte": "gel", "signature": signature, "chantier": chantier})
    declaration.update({"actif": True, "signature": signature, "chantier": chantier,
                        "depuis": _maintenant()})
    ecrire(declaration)
    return declaration


def degeler(raison: str) -> dict:
    declaration = lire()
    declaration.setdefault("historique", []).append(
        {"a": _maintenant(), "acte": "dégel", "signature": declaration.get("signature"),
         "chantier": declaration.get("chantier"), "raison": raison})
    declaration.update({"actif": False, "chantier": None, "depuis": None})
    ecrire(declaration)
    return declaration


def main() -> None:
    a = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    a.add_argument("--etat", action="store_true", help="où en est le gel")
    a.add_argument("--geler", metavar="CHANTIER", help="gèle sur la signature vivante")
    a.add_argument("--degeler", metavar="RAISON", help="lève le gel")
    a.add_argument("--forcer", action="store_true",
                   help="autorise --geler à écraser un gel déjà actif")
    args = a.parse_args()
    if args.geler:
        d = geler(args.geler, forcer=args.forcer)
        print(f"  gel posé sur {d['signature']} — {d['chantier']}")
    elif args.degeler:
        d = degeler(args.degeler)
        print(f"  gel levé — {args.degeler}")
    else:
        print(json.dumps(etat(), ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
