"""Parité de backend — figer la référence dense, recenser les égalités, comparer un candidat.

Le chantier de migration Qdrant (embarqué → serveur) ne doit rien changer au classement.
Ce module est l'instrument qui le prouve ou le réfute, et il est écrit **avant** qu'un
serveur existe : il ne peut donc pas être taillé sur son résultat.

Trois raisons d'exister, dans l'ordre où elles se présentent
------------------------------------------------------------

1. **La référence n'est pas opposable aujourd'hui.** Les classements réellement rendus par
   Qdrant vivent dans ``rag/benchmark/.cache/router-retrievals-<sig>.json``, que
   ``.gitignore:33`` exclut du dépôt. Un cache n'est pas une mesure : il est régénérable,
   écrasable, et rien ne dit qu'il n'a pas bougé. ``--geler`` en extrait la **seule branche
   dense** — celle dont la migration dépend, et la seule que le chantier reranking ne
   touchera pas — et l'écrit en artefact **versionné**, avec l'empreinte de sa source et la
   sienne.

2. **Les égalités de score doivent être nommées avant la mesure, pas découvertes après.**
   Le mode local trie par ``np.argsort(scores)[::-1]`` (``local_collection.LocalCollection``) : aucune
   clé de départage secondaire, tri non stable, et l'inversion renverse en plus l'ordre
   relatif des ex æquo. Un serveur applique ses propres règles. Toute paire à score égal peut
   donc permuter **sans qu'aucune erreur ait eu lieu**. ``--egalites`` les recense toutes.

3. **Un écart de dernier bit n'est pas un écart de classement — sauf là où l'écart de score
   est plus petit que lui.** Les vecteurs sont copiés à l'identique par la migration, mais la
   somme de 1 024 produits en float32 ne s'effectue pas dans le même ordre d'un noyau à
   l'autre. ``--egalites`` publie donc la distribution des écarts entre rangs adjacents :
   c'est elle, et rien d'autre, qui fixe la tolérance numérique que le pré-enregistrement
   pourra opposer.

Ce que la comparaison exige
---------------------------
``--comparer`` ne compare pas deux listes position à position : ce serait exiger d'un serveur
qu'il départage les ex æquo comme numpy, ce qu'aucune spécification ne promet. Elle compare
des **suites de groupes de score** : deux classements sont équivalents si, à chaque frontière
où les scores diffèrent réellement, l'ensemble des ``chunk_id`` au-dessus est le même. Une
permutation à l'intérieur d'un groupe d'ex æquo est acceptée et **comptée** ; toute autre
différence est un échec, et le module sort en code 1.

Aucun appel LLM, aucune ouverture de Qdrant, aucune écriture hors de ses propres artefacts.

    .venv/bin/python rag/benchmark/parite_backend.py --geler
    .venv/bin/python rag/benchmark/parite_backend.py --egalites
    .venv/bin/python rag/benchmark/parite_backend.py --comparer <classements-candidat.json>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]

CACHE = HERE / ".cache"
GEL = ROOT / "rag" / "gel-corpus.json"

# Tolérances publiées, dans l'ordre de grandeur du bruit attendu d'un noyau à l'autre.
PALIERS = (0.0, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3)


def signature_gelee() -> str:
    """La signature de corpus active, lue au gel — jamais recalculée, jamais devinée."""
    gel = json.loads(GEL.read_text())
    if not gel.get("actif"):
        sys.exit("refus : le gel n'est pas actif — la référence de parité n'aurait pas de domicile")
    return gel["signature"]


def _sha256(chemin: pathlib.Path) -> str:
    return hashlib.sha256(chemin.read_bytes()).hexdigest()


def _empreinte_contenu(objet) -> str:
    """Empreinte du CONTENU, pas des octets — la sérialisation ne doit pas la faire bouger."""
    return hashlib.sha256(json.dumps(objet, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def chemin_reference(signature: str) -> pathlib.Path:
    return HERE / f"parite-reference-dense-{signature}.json"


# ----------------------------------------------------------------------------- geler


def geler(signature: str) -> dict:
    source = CACHE / f"router-retrievals-{signature}.json"
    if not source.exists():
        sys.exit(f"refus : {source} est absent — rien à figer")

    brut = json.loads(source.read_text())
    classements = {}
    for qid, configs in brut.items():
        if "dense" not in configs:
            sys.exit(f"refus : {qid} n'a pas de branche dense")
        # [chunk_id, document_id, score] — recopié tel quel, sans arrondi ni tri
        classements[qid] = [[c, d, float(s)] for c, d, s in configs["dense"]]

    reference = {
        "signature_corpus": signature,
        "branche": "dense",
        "questions": len(classements),
        "source": {
            "fichier": str(source.relative_to(ROOT)),
            "sha256": _sha256(source),
            "versionne": False,
            "note": "cache gitignoré (.gitignore:33) — c'est la raison d'être de ce gel",
        },
        "classements": classements,
    }
    reference["empreinte_contenu"] = _empreinte_contenu(classements)
    return reference


# -------------------------------------------------------------------------- égalités


def _groupes(liste: list, tolerance: float) -> list:
    """Découpe un classement en groupes de score indiscernables à ``tolerance`` près.

    Un groupe est ouvert tant que le score court reste à ``tolerance`` du score qui a ouvert
    le groupe — jamais du précédent, sinon une longue chaîne d'écarts minuscules fusionnerait
    des scores réellement distincts.
    """
    groupes = []
    for chunk_id, _document_id, score in liste:
        if groupes and abs(groupes[-1][0] - score) <= tolerance:
            groupes[-1][1].add(chunk_id)
        else:
            groupes.append([score, {chunk_id}])
    return [(score, frozenset(membres)) for score, membres in groupes]


def egalites(reference: dict) -> dict:
    classements = reference["classements"]

    ex_aequo = {}          # qid -> liste des groupes d'ex æquo stricts
    ecarts_min = {}        # qid -> plus petit écart entre rangs adjacents
    total_paires = 0
    total_ex_aequo = 0

    for qid, liste in classements.items():
        scores = [s for _c, _d, s in liste]
        paires = [(scores[i], scores[i + 1]) for i in range(len(scores) - 1)]
        total_paires += len(paires)

        groupes_stricts = [(s, sorted(m)) for s, m in _groupes(liste, 0.0) if len(m) > 1]
        if groupes_stricts:
            ex_aequo[qid] = [{"score": s, "chunk_ids": m, "rangs": len(m)} for s, m in groupes_stricts]
            total_ex_aequo += sum(len(m) - 1 for _s, m in groupes_stricts)

        if paires:
            ecarts_min[qid] = min(a - b for a, b in paires)

    # Combien de questions deviennent fragiles à chaque palier de perturbation ?
    # Une perturbation d'amplitude d peut permuter toute paire adjacente d'écart < 2d.
    fragilite = {}
    for palier in PALIERS:
        seuil = 2 * palier
        fragilite[f"{palier:g}"] = sum(1 for e in ecarts_min.values() if e <= seuil)

    return {
        "signature_corpus": reference["signature_corpus"],
        "questions": len(classements),
        "paires_adjacentes": total_paires,
        "ex_aequo_stricts": {
            "questions_touchees": len(ex_aequo),
            "rangs_permutables": total_ex_aequo,
            "detail": ex_aequo,
        },
        "ecart_adjacent_minimal": {
            "min": min(ecarts_min.values()) if ecarts_min else None,
            "mediane": sorted(ecarts_min.values())[len(ecarts_min) // 2] if ecarts_min else None,
        },
        "questions_fragiles_par_amplitude_de_perturbation": fragilite,
        "lecture": ("Une perturbation de score d'amplitude d peut réordonner toute paire "
                    "adjacente dont l'écart est inférieur à 2d. La ligne '0' compte les "
                    "questions portant au moins un ex æquo strict : celles-là peuvent "
                    "permuter sans qu'aucune erreur n'ait eu lieu."),
    }


# -------------------------------------------------------------------------- comparer


PROFONDEURS = (5, 10, 50)


def comparer(reference: dict, candidat: dict, tolerance: float) -> dict:
    """B1a (porte dure) et B1b (diagnostic), §4 bis du pré-enregistrement.

    **B1a — composition.** Pour chaque *k*, l'``ensemble`` des ``chunk_id`` du top-*k* doit
    être identique. C'est la porte : un candidat qui entre ou sort change ce que le produit
    sert et ce qu'un reranker reçoit. Trois profondeurs et non la seule plus grande, parce
    qu'une permutation entre les rangs 5 et 6 ne touche pas l'ensemble du top-50 mais bien
    celui du top-5.

    **B1b — ordre interne.** La séquence à l'intérieur d'un même top-*k*. Diagnostic : un
    reranker reclasse ce qu'il reçoit, et un déplacement de rang qui ne change pas les cinq
    passages servis est invisible à l'utilisateur.

    Ce module ne juge que B1a et B1b. **B6 (métrique) et B7 (cinq passages servis) sont des
    portes dures qu'il ne voit pas** : le verdict rendu ici est partiel, et il le dit.
    """
    ref = reference["classements"]
    cand = candidat.get("classements", candidat)

    manquantes = sorted(set(ref) - set(cand))
    en_trop = sorted(set(cand) - set(ref))
    communes = sorted(set(ref) & set(cand))

    composition = {}
    for k in PROFONDEURS:
        divergentes_k = [qid for qid in communes
                         if {c for c, _d, _s in ref[qid][:k]} != {c for c, _d, _s in cand[qid][:k]}]
        composition[f"top{k}"] = {
            "identiques": len(communes) - len(divergentes_k),
            "divergentes": divergentes_k,
        }

    identiques, permutations, divergences = [], [], []

    for qid in communes:
        gr = _groupes(ref[qid], tolerance)
        gc = _groupes([[c, d, float(s)] for c, d, s in cand[qid]], tolerance)

        suite_ref = [membres for _s, membres in gr]
        suite_cand = [membres for _s, membres in gc]

        if suite_ref != suite_cand:
            premiere = next((i for i in range(min(len(suite_ref), len(suite_cand)))
                             if suite_ref[i] != suite_cand[i]), min(len(suite_ref), len(suite_cand)))
            divergences.append({
                "qid": qid,
                "premier_groupe_divergent": premiere,
                "reference": sorted(suite_ref[premiere]) if premiere < len(suite_ref) else None,
                "candidat": sorted(suite_cand[premiere]) if premiere < len(suite_cand) else None,
            })
            continue

        sequence_ref = [c for c, _d, _s in ref[qid]]
        sequence_cand = [c for c, _d, _s in cand[qid]]
        (identiques if sequence_ref == sequence_cand else permutations).append(qid)

    population_intacte = not manquantes and not en_trop
    composition_intacte = all(not v["divergentes"] for v in composition.values())
    b1a = population_intacte and composition_intacte

    if not b1a:
        verdict = "REFUS"
    elif not divergences:
        verdict = "PARITE STRICTE" if tolerance == 0.0 else f"PARITE NUMERIQUE (tau={tolerance:g})"
    else:
        verdict = "REFUS"

    return {
        "signature_corpus": reference["signature_corpus"],
        "tolerance": tolerance,
        "questions_comparees": len(communes),
        "B1a_composition_PORTE_DURE": {
            "tenue": b1a,
            "profondeurs": composition,
        },
        "B1b_ordre_interne_DIAGNOSTIC": {
            "identiques_position_a_position": len(identiques),
            "equivalentes_par_permutation_dex_aequo": {"n": len(permutations), "qids": permutations},
            "divergentes": {"n": len(divergences), "detail": divergences},
        },
        "questions_manquantes": manquantes,
        "questions_en_trop": en_trop,
        "verdict_partiel": verdict,
        "portee": ("B1a et B1b seulement. B6 (metrique, eval_dense_candidat --controle) et "
                   "B7 (les cinq passages servis) sont des portes dures que ce module ne voit "
                   "pas : aucun verdict de parite n'est complet sans elles."),
        "PARITE": b1a and not divergences,
    }


# ------------------------------------------------------------------------------ CLI


def main() -> None:
    parseur = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parseur.add_argument("--geler", action="store_true",
                         help="extraire la branche dense du cache et l'écrire en artefact versionné")
    parseur.add_argument("--egalites", action="store_true",
                         help="recenser les ex æquo et la fragilité numérique du classement")
    parseur.add_argument("--comparer", metavar="FICHIER",
                         help="comparer un classement candidat à la référence gelée")
    parseur.add_argument("--tolerance", type=float, default=0.0,
                         help="écart de score en deçà duquel deux rangs sont indiscernables (défaut 0)")
    args = parseur.parse_args()

    signature = signature_gelee()
    cible = chemin_reference(signature)

    if args.geler:
        reference = geler(signature)
        if cible.exists():
            ancienne = json.loads(cible.read_text())
            if ancienne.get("empreinte_contenu") != reference["empreinte_contenu"]:
                sys.exit(f"refus : {cible.name} existe déjà avec un contenu DIFFÉRENT.\n"
                         f"  gelée   {ancienne.get('empreinte_contenu')}\n"
                         f"  calculée {reference['empreinte_contenu']}\n"
                         "  Une référence de parité ne se réécrit pas : elle se rejette ou se renomme.")
            print(f"déjà gelée, contenu identique — {cible.name}")
        else:
            cible.write_text(json.dumps(reference, indent=1, ensure_ascii=False))
            print(f"gelée — {cible.name}")
        print(json.dumps({k: v for k, v in reference.items() if k != "classements"},
                         indent=1, ensure_ascii=False))
        return

    if not cible.exists():
        sys.exit(f"refus : {cible.name} est absent — commence par --geler")
    reference = json.loads(cible.read_text())

    if args.egalites:
        rapport = egalites(reference)
        sortie = HERE / f"parite-egalites-{signature}.json"
        sortie.write_text(json.dumps(rapport, indent=1, ensure_ascii=False))
        resume = {k: v for k, v in rapport.items() if k != "ex_aequo_stricts"}
        resume["ex_aequo_stricts"] = {k: v for k, v in rapport["ex_aequo_stricts"].items() if k != "detail"}
        print(json.dumps(resume, indent=1, ensure_ascii=False))
        return

    if args.comparer:
        candidat = json.loads(pathlib.Path(args.comparer).read_text())
        rapport = comparer(reference, candidat, args.tolerance)
        print(json.dumps(rapport, indent=1, ensure_ascii=False))
        sys.exit(0 if rapport["PARITE"] else 1)

    parseur.error("choisis --geler, --egalites ou --comparer")


if __name__ == "__main__":
    main()
