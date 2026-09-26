"""Les quatre bras du lot `latex-recolle`, et le verdict pré-enregistré.

Protocole : `PRE-ENREGISTREMENT-LATEX-2026-09-10.md`. Zéro appel LLM, zéro écriture Qdrant,
zéro écriture d'overlay servi : tout passe par `dense_matrix`, qui reconstruit hors ligne
exactement ce que la collection contient.

    .venv/bin/python rag/benchmark/bancs_latex.py --bras z     # ~110 min de MPS
    .venv/bin/python rag/benchmark/bancs_latex.py --bras c
    .venv/bin/python rag/benchmark/bancs_latex.py --bras p
    .venv/bin/python rag/benchmark/bancs_latex.py --mesure     # quelques minutes, sans modèle

Chaque bras est reprenable : son checkpoint `.cache/latex-<bras>-<signature>.npz` est
réécrit après chaque lot, et une exécution coupée repart où elle s'est arrêtée. Le
checkpoint porte une clé de configuration ; changer la transformation, la population ou le
batch invalide le cache au lieu de mélanger deux mondes.
"""
from __future__ import annotations

import argparse
import gc
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "titles"))

import familles_v4  # noqa: E402
import pipeline  # noqa: E402
import quant_rag  # noqa: E402
import recollage  # noqa: E402
import reembed_titles as rt  # noqa: E402
from compare_v1_v2 import load_bench, load_v1  # noqa: E402
from corpus import ChunkIndex  # noqa: E402
from dense_matrix import Matrix  # noqa: E402

CACHE = HERE / ".cache"
SEED = 20260910
BATCH = 8
BRAS = ("z", "c", "p")
ETIQUETTES = {"z": "identité (texte inchangé)", "c": "candidat (LaTeX recollé)",
              "p": "placebo (blancs retirés au hasard, longueur appariée à c)"}
FAMILLES = ("v1", "v3", "v4_formula", "v4_table_cell")


# --------------------------------------------------------------------------- population et bras

def charger_corpus() -> tuple[list[dict], dict[str, str], dict[str, dict]]:
    rows, export_title = rt.load_rows()
    return rows, {r["chunk_id"]: r["text"] for r in rows}, rt.titles_table(export_title)


def construire_bras(textes: dict[str, str], voulus: tuple[str, ...] = BRAS
                    ) -> tuple[list[str], dict[str, dict[str, str]]]:
    """Population triée par `chunk_id`, et texte des bras `voulus`.

    Le placebo est **toujours engendré pour toute la population, dans l'ordre trié**, même
    quand on ne le demande pas : sa graine est consommée passage par passage, donc sauter un
    passage décalerait toute la suite et le placebo du bras P ne serait pas celui que le
    pré-enregistrement décrit. Ce qui n'est pas demandé est engendré puis jeté — c'est le
    prix de la reproductibilité, et il est en temps de calcul, pas en mémoire.

    Ne matérialiser que les bras demandés est une économie de mémoire qui compte : la machine
    a tué un premier encodage par pression mémoire alors que les trois jeux de textes étaient
    tenus en même temps sans qu'aucun des deux autres serve.
    """
    population: list[str] = []
    candidat: dict[str, str] = {}
    longueur_candidat: dict[str, int] = {}
    for chunk_id in sorted(textes):
        recolle = recollage.recoller_passage(textes[chunk_id])
        if recolle != textes[chunk_id]:
            population.append(chunk_id)
            longueur_candidat[chunk_id] = len(recolle)
            if "c" in voulus:
                candidat[chunk_id] = recolle
    sortie: dict[str, dict[str, str]] = {}
    rng = random.Random(SEED)
    placebo: dict[str, str] = {}
    for chunk_id in population:
        tire = recollage.placebo_passage(textes[chunk_id], longueur_candidat[chunk_id], rng)
        if "p" in voulus:
            placebo[chunk_id] = tire
    if "z" in voulus:
        sortie["z"] = {c: textes[c] for c in population}
    if "c" in voulus:
        sortie["c"] = candidat
    if "p" in voulus:
        sortie["p"] = placebo
    return population, sortie


def cle_config(population: list[str], bras: str) -> str:
    """Empreinte de ce que le checkpoint contient : population, bras, modèle, batch."""
    import hashlib
    h = hashlib.sha256("\n".join(population).encode()).hexdigest()[:16]
    return f"{bras}|{len(population)}|{h}|{quant_rag.MODEL_ID}|{BATCH}"


#: Nombre de passages ré-encodés par bras pour prouver que son checkpoint correspond bien à
#: la transformation **du code courant**. La clé de configuration ne hache que la population :
#: une transformation modifiée qui laisserait l'ensemble des `chunk_id` inchangé — c'est le
#: cas de tout changement du placebo seul — rendrait un cache périmé sans le dire, et on
#: mesurerait un verdict sur les vecteurs d'un autre monde. 24 passages suffisent : il n'y a
#: pas de changement de transformation qui n'en touche aucun.
CONTROLE_FRAICHEUR = 24


def controler_fraicheur(bras: str, population: list[str], textes_bras: dict[str, str],
                        texte_plonge, vecteurs: dict[str, np.ndarray], graine: int = SEED) -> dict:
    """Ré-encode quelques passages et compare au checkpoint. Refuse si ça ne colle pas."""
    # `hash()` d'une chaîne n'est pas stable d'un processus à l'autre (PYTHONHASHSEED) : la
    # graine viendrait du rang du bras, pas d'une empreinte aléatoire, sans quoi deux
    # exécutions ne contrôleraient pas les mêmes passages.
    rng = random.Random(graine + BRAS.index(bras))
    echantillon = rng.sample(sorted(population), min(CONTROLE_FRAICHEUR, len(population)))
    model = quant_rag.embedder()
    attendus = model.encode([texte_plonge(c, textes_bras[c]) for c in echantillon],
                            normalize_embeddings=True, show_progress_bar=False, batch_size=BATCH)
    cos = []
    for chunk_id, attendu in zip(echantillon, attendus):
        stocke = np.asarray(vecteurs[chunk_id], dtype=np.float32)
        stocke = stocke / max(float(np.linalg.norm(stocke)), 1e-12)
        cos.append(float(np.dot(stocke, np.asarray(attendu, dtype=np.float32))))
    arr = np.asarray(cos)
    rapport = {"n": len(cos), "min": round(float(arr.min()), 6), "mediane": round(float(np.median(arr)), 6)}
    if arr.min() < 0.999:
        sys.exit(f"bras {bras} : checkpoint PÉRIMÉ — le cosinus minimal du contrôle de fraîcheur "
                 f"est {arr.min():.6f} sur {len(cos)} passages ré-encodés avec la transformation "
                 "du code courant. Le checkpoint vient d'une autre transformation ; supprime-le "
                 f"(.cache/latex-{bras}-*.npz) et ré-encode, ou reviens au code qui l'a produit.")
    return rapport


# --------------------------------------------------------------------------- encodage

def vider_le_cache_du_peripherique() -> None:
    """Rend au système les blocs que l'allocateur de PyTorch garde en réserve.

    Sans appel, deux encodages de ce lot ont été tués par la machine. `torch` n'est importé
    qu'ici : le reste du module n'en a pas besoin, et la mesure tourne sans modèle.
    """
    try:
        import torch  # noqa: PLC0415
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as erreur:                       # noqa: BLE001
        print(f"  (cache du périphérique non purgeable : {erreur})", flush=True)


def encoder(bras: str, population: list[str], textes_bras: dict[str, str],
            texte_plonge, signature: str) -> dict[str, np.ndarray]:
    """`model.encode` lot par lot, reprise sur checkpoint après chaque lot.

    L'ordre d'encodage est trié par longueur — jamais la population elle-même, dont
    l'ordre trié par `chunk_id` fait la reproductibilité du placebo. Sans ce tri local, un
    lot MPS mélange courts et longs, rembourre tout au plus long et recompile un noyau par
    forme de lot rencontrée (cf. `reembed_titles.reembed`).
    """
    chemin = CACHE / f"latex-{bras}-{signature}.npz"
    cle = cle_config(population, bras)
    fait: dict[str, np.ndarray] = {}
    if chemin.exists():
        blob = np.load(chemin, allow_pickle=True)
        if str(blob["cle_config"]) == cle:
            fait = dict(zip(blob["ids"].tolist(), blob["vecteurs"]))
            print(f"  reprise : {len(fait)}/{len(population)} déjà calculés", flush=True)
        else:
            print(f"  checkpoint d'une autre configuration — ignoré, recalcul complet", flush=True)
    # Les textes plongés ne sont tenus que pour ce qui reste à faire : à la reprise d'un bras
    # déjà aux deux tiers, c'est trois fois moins de chaînes en mémoire. Le tri par longueur
    # se fait sur la longueur seule, sans garder les chaînes de ce qui est déjà calculé.
    longueurs = {c: len(texte_plonge(c, textes_bras[c])) for c in population}
    ordre = [c for c in sorted(population, key=lambda c: longueurs[c]) if c not in fait]  # noqa: F821 — lu par sorted() avant le del
    del longueurs
    if not ordre:
        print(f"  bras {bras} déjà complet", flush=True)
        return fait
    plonges = {c: texte_plonge(c, textes_bras[c]) for c in ordre}
    textes_bras.clear()
    gc.collect()
    model = quant_rag.embedder()
    debut = time.perf_counter()
    #: Le checkpoint est réécrit en entier à chaque sauvegarde, donc son coût croît avec ce
    #: qu'il contient. À 14 597 vecteurs, sauver après *chaque* lot de 8 recopierait 60 Mo
    #: toutes les deux secondes en fin de bras. Sauver tous les 32 lots met au plus 256
    #: vecteurs en jeu (~2 min de recalcul) pour un coût d'écriture 32 fois moindre.
    LOTS_PAR_SAUVEGARDE = 32
    for n, i in enumerate(range(0, len(ordre), BATCH), 1):
        lot = ordre[i:i + BATCH]
        vecteurs = model.encode([plonges.pop(c) for c in lot], normalize_embeddings=True,
                                show_progress_bar=False, batch_size=BATCH)
        for c, v in zip(lot, vecteurs):
            fait[c] = np.asarray(v, dtype=np.float32)
        if n % LOTS_PAR_SAUVEGARDE == 0 or i + BATCH >= len(ordre):
            CACHE.mkdir(parents=True, exist_ok=True)
            np.savez(chemin, ids=np.asarray(list(fait)), vecteurs=np.vstack(list(fait.values())),
                     cle_config=cle)
            # **La purge du cache MPS est la correction qui fait tenir le bras.** L'allocateur
            # de PyTorch garde les blocs qu'il a servis ; avec des lots de longueurs toutes
            # différentes — notre ordre est trié par longueur, donc chaque lot a une forme
            # nouvelle — il se fragmente et ne rend rien au système. La machine a tué deux
            # encodages de ce bras pour cette raison (swap 7,6 Go sur 8 ; retombé à 4,7 Go dès
            # la mort du processus, ce qui désigne le coupable). Purger au même rythme que la
            # sauvegarde coûte quelques dizaines de millisecondes.
            vider_le_cache_du_peripherique()
        ecoule = time.perf_counter() - debut
        reste = (len(population) - len(fait)) * ecoule / max(len(fait) - (len(population) - len(ordre)), 1)
        print(f"  {bras} {len(fait)}/{len(population)}  {ecoule / 60:.0f} min écoulées, "
              f"reste ~{reste / 60:.0f} min", flush=True)
    return fait


# --------------------------------------------------------------------------- questions

def charger_questions() -> list[tuple[str, dict]]:
    index = ChunkIndex.load(verbose=False)
    return ([("v1", it) for it in load_v1(index)]
            + [("v3", it) for it in load_bench(HERE / "questions-v3.jsonl")]
            + [("v4_formula", it) for it in familles_v4.lire_jsonl(HERE / "questions-v4-formula.jsonl")]
            + [("v4_table_cell", it) for it in familles_v4.lire_jsonl(HERE / "questions-v4-table-cell.jsonl")])


def chercher(matrix: Matrix, vecteur, scope, textes: dict[str, str]) -> list[dict]:
    pool = matrix.search(vecteur, pool=quant_rag.POOL, scope=scope, skip_headings=True)
    for row in pool:
        row["text"] = textes.get(row["chunk_id"], "")
    return pool


def servi(pool: list[dict]) -> set[str]:
    return {r["chunk_id"] for r in quant_rag._select(pool, limit=5, per_document=2, dedupe=True,
                                                     min_characters=quant_rag.MIN_CHARACTERS)}


def rang_de_l_or(pool: list[dict], ors: set[str]) -> int | None:
    for i, row in enumerate(pool, 1):
        if row["chunk_id"] in ors:
            return i
    return None


def patcher(reference: Matrix, vecteurs: dict[str, np.ndarray], etiquette: str) -> tuple[Matrix, list[str], np.ndarray]:
    matrice = reference.vectors.copy()
    absents, cosinus = [], []
    for chunk_id, vecteur in vecteurs.items():
        i = reference.position_of(chunk_id)
        if i is None:
            absents.append(chunk_id)
            continue
        v = np.asarray(vecteur, dtype=np.float32)
        v = v / max(float(np.linalg.norm(v)), 1e-12)
        cosinus.append(float(np.dot(matrice[i], v)))
        matrice[i] = v
    patchee = Matrix(reference.chunk_ids, reference.document_ids, matrice, etiquette,
                     reference.content_types)
    patchee.signature = reference.signature
    return patchee, absents, np.asarray(cosinus)


def gagnes_perdus(reference: dict[str, bool], bras: dict[str, bool], prefixe: str | None = None) -> dict:
    cles = [k for k in reference if prefixe is None or k.startswith(prefixe + "/")]
    gagnes = sorted(k for k in cles if bras.get(k, False) and not reference[k])
    perdus = sorted(k for k in cles if reference[k] and not bras.get(k, False))
    return {"n": len(cles), "or_servi_reference": sum(reference[k] for k in cles),
            "or_servi_bras": sum(bras.get(k, False) for k in cles),
            "gagnes": gagnes, "n_gagnes": len(gagnes), "perdus": perdus, "n_perdus": len(perdus),
            "net": len(gagnes) - len(perdus)}


# --------------------------------------------------------------------------- grandeurs secondaires

TIRAGES = 10000


def mcnemar_exact(gagnes: int, perdus: int) -> float:
    """p bilatéral du test exact de McNemar sur les paires discordantes.

    Sous l'hypothèse nulle « le changement est symétrique », les ``g + p`` questions qui ont
    basculé se répartissent comme une binomiale de paramètre 1/2. C'est le test approprié
    pour un appariement question par question, et il ne suppose rien sur les questions qui
    n'ont pas bougé — celles-là n'apportent aucune information sur le sens du changement.
    """
    n = gagnes + perdus
    if n == 0:
        return 1.0
    from math import comb
    k = min(gagnes, perdus)
    queue = sum(comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return min(1.0, 2 * queue)


def bootstrap_net(deltas: dict[str, int], graine: int = SEED, tirages: int = TIRAGES) -> dict:
    """IC à 95 % du net par ré-échantillonnage **des questions**, avec remise.

    ``deltas[q]`` vaut +1 si le bras gagne la question, −1 s'il la perd, 0 sinon. L'unité de
    ré-échantillonnage est la question, parce que c'est l'unité d'échantillonnage du banc :
    ré-échantillonner les ors ou les passages donnerait un IC qui répond à une autre
    question. Le net est rendu à l'échelle du banc (somme sur `n` questions tirées).

    **Cette grandeur ne décide de rien** : la règle du §8 du pré-enregistrement porte sur
    les points, pas sur les intervalles. Elle dit seulement à quelle précision le banc
    connaît ce qu'il mesure.
    """
    cles = sorted(deltas)
    if not cles:
        return {"net": 0, "ic95": [0, 0], "tirages": tirages}
    valeurs = np.asarray([deltas[k] for k in cles], dtype=np.int16)
    rng = np.random.default_rng(graine)
    tirage = rng.integers(0, len(valeurs), size=(tirages, len(valeurs)))
    nets = valeurs[tirage].sum(axis=1)
    return {"net": int(valeurs.sum()), "ic95": [int(np.quantile(nets, .025)), int(np.quantile(nets, .975))],
            "part_des_tirages_au_moins_3": round(float((nets >= 3).mean()), 4),
            "part_des_tirages_positifs": round(float((nets > 0).mean()), 4), "tirages": tirages}


def deltas_de(reference: dict[str, bool], bras: dict[str, bool]) -> dict[str, int]:
    return {k: int(bras.get(k, False)) - int(v) for k, v in reference.items()}


# --------------------------------------------------------------------------- plancher de bruit

def bras_bruit(matrix_ref: Matrix, population: list[str], cosinus: np.ndarray, tirages: int,
               mesurer, bavard: bool = True) -> dict:
    """Le **taux de fausse alarme** du seuil absolu, par perturbation isotrope répétée.

    Le bras identité (Z) est *une* réalisation du bruit de la procédure : il dit de combien
    le ré-embarquement déplace les ors servis cette fois-ci, pas de combien il pourrait les
    déplacer. Or « un seuil sans son taux de fausse alarme n'est pas un seuil » — c'est la
    leçon que le chantier `instrument-v4` a payée en voyant 3 de ses 5 seuils de
    non-régression violés à système inchangé.

    Ce bras remplace donc le ré-embarquement par une perturbation **de même ampleur
    angulaire** : chaque vecteur de la population est tourné d'un angle tiré dans la
    distribution des cosinus réellement observés (celle de Z s'il existe, sinon celle de
    l'échantillon de reproduction), dans une direction aléatoire orthogonale. Répété
    ``tirages`` fois, il rend la **distribution** du net sous variable nulle.

    Ce n'est pas le vrai bruit : la perturbation d'un ré-embarquement n'est pas isotrope.
    C'en est un modèle, et il encadre Z au lieu de le remplacer — les deux sont publiés.
    """
    index = np.asarray([matrix_ref.position_of(c) for c in population
                        if matrix_ref.position_of(c) is not None])
    base = matrix_ref.vectors[index]
    cibles = np.clip(np.asarray(cosinus, dtype=np.float32), -1.0, 1.0)
    nets, nets_formula = [], []
    for t in range(tirages):
        rng = np.random.default_rng(SEED + 1000 + t)
        direction = rng.standard_normal(base.shape).astype(np.float32)
        direction -= np.sum(direction * base, axis=1, keepdims=True) * base   # composante orthogonale
        direction /= np.maximum(np.linalg.norm(direction, axis=1, keepdims=True), 1e-12)
        cos = cibles[rng.integers(0, len(cibles), size=len(index))][:, None]
        perturbe = cos * base + np.sqrt(np.maximum(1 - cos ** 2, 0)) * direction
        perturbe /= np.maximum(np.linalg.norm(perturbe, axis=1, keepdims=True), 1e-12)
        matrice = matrix_ref.vectors.copy()
        matrice[index] = perturbe
        bruitee = Matrix(matrix_ref.chunk_ids, matrix_ref.document_ids, matrice,
                         f"bruit-{t}", matrix_ref.content_types)
        bruitee.signature = matrix_ref.signature
        net, net_formula = mesurer(bruitee)
        nets.append(net)
        nets_formula.append(net_formula)
        if bavard:
            print(f"  bruit {t + 1}/{tirages} : net {net:+d} (formula {net_formula:+d})", flush=True)
    nets = np.asarray(nets)
    return {
        "tirages": tirages,
        "cosinus_imite": {"mediane": round(float(np.median(cibles)), 6),
                          "min": round(float(cibles.min()), 6)},
        "net": {"min": int(nets.min()), "p05": int(np.quantile(nets, .05)),
                "mediane": int(np.median(nets)), "p95": int(np.quantile(nets, .95)),
                "max": int(nets.max()), "moyenne": round(float(nets.mean()), 2),
                "ecart_type": round(float(nets.std()), 2)},
        "net_formula": {"min": int(min(nets_formula)), "max": int(max(nets_formula)),
                        "mediane": int(np.median(nets_formula)),
                        "moyenne": round(float(np.mean(nets_formula)), 2)},
        "net_formula_tous": list(nets_formula),
        "taux_de_fausse_alarme_du_seuil_3": round(float((nets >= 3).mean()), 4),
        "taux_de_fausse_alarme_bilateral_3": round(float((np.abs(nets) >= 3).mean()), 4),
        "lecture": ("part des tirages à variable nulle qui atteindraient le seuil absolu de "
                    "+3 ors servis nets ; un seuil dont ce taux n'est pas petit ne départage "
                    "rien"),
        "tous_les_nets": nets.tolist(),
    }


def bootstrap_contraste(deltas_c: dict[str, int], deltas_p: dict[str, int],
                        graine: int = SEED, tirages: int = TIRAGES) -> dict:
    """IC du contraste C − P, **apparié question par question** : les deux bras voient
    exactement le même tirage de questions, donc l'IC porte sur la différence et non sur la
    somme de deux incertitudes indépendantes."""
    cles = sorted(set(deltas_c) & set(deltas_p))
    diff = np.asarray([deltas_c[k] - deltas_p[k] for k in cles], dtype=np.int16)
    rng = np.random.default_rng(graine + 1)
    tirage = rng.integers(0, len(diff), size=(tirages, len(diff)))
    nets = diff[tirage].sum(axis=1)
    return {"contraste": int(diff.sum()),
            "ic95": [int(np.quantile(nets, .025)), int(np.quantile(nets, .975))],
            "part_des_tirages_positifs": round(float((nets > 0).mean()), 4), "tirages": tirages}


# --------------------------------------------------------------------------- bande de rang

def bande(rang: int | None) -> str:
    if rang is None:
        return "hors pool"
    for borne, nom in ((1, "1"), (3, "2-3"), (5, "4-5"), (10, "6-10"), (20, "11-20"), (50, "21-50")):
        if rang <= borne:
            return nom
    return "hors pool"


# --------------------------------------------------------------------------- programme

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--bras", choices=BRAS, help="encode un bras (long : ~110 min de MPS)")
    parser.add_argument("--mesure", action="store_true", help="mesure et verdict, sans modèle")
    parser.add_argument("--population", action="store_true", help="recense seulement, sans modèle")
    parser.add_argument("--bruit", type=int, default=0, metavar="N",
                       help="N tirages de perturbation isotrope : le taux de fausse alarme du seuil")
    parser.add_argument("--balayage", type=int, default=0, metavar="N",
                       help="N tirages par niveau de cosinus : le net répond-il à l'AMPLEUR "
                            "de la perturbation plutôt qu'à son contenu ?")
    args = parser.parse_args()

    debut = time.perf_counter()
    rows, textes, titles = charger_corpus()
    n_passages = len(rows)
    par_id = {r["chunk_id"]: r for r in rows}
    # Un seul bras est matérialisé quand un seul est demandé : tenir les trois jeux de textes
    # en même temps a coûté un premier encodage, tué par pression mémoire à 6 144/14 597.
    voulus = (args.bras,) if args.bras else BRAS
    population, bras_textes = construire_bras(textes, voulus)
    signature = __import__("corpus_overlay").signature()
    print(f"corpus {signature} · {n_passages} passages · population {len(population)}")

    # `par_id` n'est plus utile que pour la population — le reste du corpus ne sera pas plongé.
    titre_chemin = {c: (titles[par_id[c]["document_id"]]["clean_title"], par_id[c]["title_path"])
                    for c in population}
    del rows, par_id
    if not (args.mesure or args.population):
        textes = {}          # la mesure en a besoin (min_characters, dédoublonnage), pas l'encodage
    gc.collect()

    def texte_plonge(chunk_id: str, texte: str) -> str:
        titre, chemin = titre_chemin[chunk_id]
        return rt.embedding_text(titre, chemin, texte)

    if args.population:
        for bras in BRAS:
            longueurs = np.asarray([len(bras_textes[bras][c]) for c in population])
            print(f"  {bras} : longueur médiane {int(np.median(longueurs))} "
                  f"· total {int(longueurs.sum())}")
        return

    if args.bras:
        print(f"bras {args.bras} — {ETIQUETTES[args.bras]}")
        encoder(args.bras, population, bras_textes[args.bras], texte_plonge, signature)
        print(f"terminé en {(time.perf_counter() - debut) / 60:.0f} min")
        return

    if not args.mesure:
        parser.error("choisis --bras, --mesure ou --population")

    # ------------------------------------------------------------------ mesure
    matrix_ref = Matrix.load()
    print(f"matrice de référence : {len(matrix_ref.chunk_ids)} chunks, signature {matrix_ref.signature}")

    items = charger_questions()
    gold = {f"{b}/{it['qid']}": set(it.get("gold_chunks") or []) for b, it in items}
    sans_or = sorted(k for k, v in gold.items() if not v)
    vecteurs_q = {f"{b}/{it['qid']}": np.asarray(quant_rag.encode_query(pipeline.query_of(it)),
                                                dtype=np.float32) for b, it in items}
    scopes = {}
    for b, it in items:
        filtres = {k: v for k, v in (pipeline.filters_of(it) or {}).items() if v is not None}
        scopes[f"{b}/{it['qid']}"] = quant_rag.document_scope(None, **filtres) if filtres else None
    print(f"questions : {len(items)} · ors distincts {len(set().union(*gold.values()))} "
          f"· sans or {len(sans_or)}")

    pools_ref = {cle: chercher(matrix_ref, vecteurs_q[cle], scopes[cle], textes) for cle in gold}
    reference = {cle: bool(servi(pool) & gold[cle]) for cle, pool in pools_ref.items()}
    rangs_ref = {cle: rang_de_l_or(pools_ref[cle], gold[cle]) for cle in gold}
    print(f"or servi, référence : {sum(reference.values())}/{len(items)}")

    resultats: dict[str, dict] = {}
    rangs_bras: dict[str, dict[str, int | None]] = {}
    deltas: dict[str, dict[str, int]] = {}
    for bras in BRAS:
        chemin = CACHE / f"latex-{bras}-{signature}.npz"
        if not chemin.exists():
            print(f"  bras {bras} : ABSENT ({chemin.name}) — ignoré")
            continue
        blob = np.load(chemin, allow_pickle=True)
        if str(blob["cle_config"]) != cle_config(population, bras):
            sys.exit(f"checkpoint {chemin.name} d'une autre configuration — refus de mesurer dessus")
        vecteurs = dict(zip(blob["ids"].tolist(), blob["vecteurs"]))
        if len(vecteurs) != len(population):
            print(f"  bras {bras} : INCOMPLET {len(vecteurs)}/{len(population)} — ignoré")
            continue
        fraicheur = controler_fraicheur(bras, population, bras_textes[bras], texte_plonge, vecteurs)
        print(f"  bras {bras} : fraîcheur OK ({fraicheur['n']} passages ré-encodés, "
              f"cosinus min {fraicheur['min']})")
        # Les textes du bras ne servent plus qu'à ce contrôle : la mesure travaille sur les
        # vecteurs et sur le texte **servi**, qui ne change pas. La mesure est l'invocation la
        # plus lourde du lot (modèle + matrice + trois jeux de textes) et la machine a déjà tué
        # deux processus de ce lot pour pression mémoire.
        bras_textes[bras].clear()
        gc.collect()
        matrice, absents, cos = patcher(matrix_ref, vecteurs, f"latex-{bras}")
        pools = {cle: chercher(matrice, vecteurs_q[cle], scopes[cle], textes) for cle in gold}
        or_bras = {cle: bool(servi(pool) & gold[cle]) for cle, pool in pools.items()}
        rangs_bras[bras] = {cle: rang_de_l_or(pools[cle], gold[cle]) for cle in gold}
        deltas[bras] = deltas_de(reference, or_bras)
        tous = gagnes_perdus(reference, or_bras)
        # Entrants nouveaux : passages patchés qui entrent dans un pool top-50 où ils
        # n'étaient pas à la référence. C'est exactement l'angle mort de la sonde, qui ne
        # patchait que les chunks déjà dans un pool et sous-estimait donc ses pertes.
        entrants = sum(len({r["chunk_id"] for r in pools[cle]}
                           - {r["chunk_id"] for r in pools_ref[cle]}) for cle in gold)
        resultats[bras] = {
            "etiquette": ETIQUETTES[bras],
            "chunks_patches": len(vecteurs), "absents_de_la_matrice": len(absents),
            "cosinus_contre_reference": {
                "min": round(float(cos.min()), 6), "p05": round(float(np.quantile(cos, .05)), 6),
                "mediane": round(float(np.median(cos)), 6), "max": round(float(cos.max()), 6)},
            "controle_de_fraicheur": fraicheur,
            "entrants_nouveaux_dans_les_pools": entrants,
            "tous": tous,
            "bootstrap_net": bootstrap_net(deltas[bras]),
            "mcnemar_p": round(mcnemar_exact(tous["n_gagnes"], tous["n_perdus"]), 5),
            "familles": {f: gagnes_perdus(reference, or_bras, prefixe=f) for f in FAMILLES},
        }
        t = resultats[bras]["tous"]
        print(f"  bras {bras} : or servi {t['or_servi_bras']}/{t['n']} "
              f"(réf {t['or_servi_reference']}) · +{t['n_gagnes']}/−{t['n_perdus']} = net {t['net']:+d} "
              f"· cosinus médian {resultats[bras]['cosinus_contre_reference']['mediane']}")

    # ------------------------------------------------------------------ verdict
    verdict: dict[str, object] = {"regle": ("GO si (C_net − P_net) > 0 ET C_net ≥ +3 ET aucune "
                                           "famille ne perd plus qu'elle ne gagne ; sinon NO-GO")}
    if "c" in resultats and "p" in resultats:
        c_net = resultats["c"]["tous"]["net"]
        p_net = resultats["p"]["tous"]["net"]
        familles_negatives = {f: resultats["c"]["familles"][f]["net"]
                              for f in FAMILLES if resultats["c"]["familles"][f]["net"] < 0}
        conditions = {"contraste_C_moins_P_strictement_positif": (c_net - p_net) > 0,
                      "C_net_au_moins_3": c_net >= 3,
                      "aucune_famille_negative": not familles_negatives}
        go = all(conditions.values())
        # Recoupement candidat/placebo, question par question. C'est la lecture la plus serrée
        # que ce banc permette : une question que le placebo gagne aussi n'est pas gagnée *par
        # le recollage*, elle est gagnée par le fait d'avoir touché la chaîne. Ce qui reste
        # après retrait du recoupement est le seul effet imputable à la variable.
        gains_c = set(resultats["c"]["tous"]["gagnes"])
        gains_p = set(resultats["p"]["tous"]["gagnes"])
        pertes_c = set(resultats["c"]["tous"]["perdus"])
        pertes_p = set(resultats["p"]["tous"]["perdus"])
        verdict |= {"C_net": c_net, "P_net": p_net, "contraste": c_net - p_net,
                    "familles_negatives": familles_negatives, "conditions": conditions,
                    "verdict": "GO" if go else "NO-GO",
                    "bootstrap_contraste": bootstrap_contraste(deltas["c"], deltas["p"]),
                    "recoupement_candidat_placebo": {
                        "gains_communs": sorted(gains_c & gains_p),
                        "gains_propres_au_candidat": sorted(gains_c - gains_p),
                        "pertes_communes": sorted(pertes_c & pertes_p),
                        "pertes_propres_au_candidat": sorted(pertes_c - pertes_p),
                        "net_propre_au_candidat": len(gains_c - gains_p) - len(pertes_c - pertes_p),
                        "lecture": ("une question que le placebo gagne aussi n'est pas gagnée par le "
                                    "recollage mais par le fait d'avoir touché la chaîne ; le net "
                                    "propre est ce qui reste quand on retire le recoupement")}}
        if "z" in resultats:
            z_net = resultats["z"]["tous"]["net"]
            verdict["Z_net"] = z_net
            verdict["seuil_sous_le_plancher_de_bruit"] = abs(z_net) >= 3
            verdict["lecture_du_plancher"] = (
                f"le bras identité déplace {z_net:+d} or servi net à texte inchangé ; "
                + ("le seuil absolu de +3 est DANS le bruit de la procédure — un GO obtenu à "
                   "+3 ne serait pas un résultat" if abs(z_net) >= 3 else
                   "le seuil absolu de +3 est au-dessus du bruit de la procédure"))
        print(f"\nVERDICT {verdict['verdict']} · C {c_net:+d} · P {p_net:+d} · "
              f"contraste {c_net - p_net:+d} · Z {verdict.get('Z_net', 'absent')}")
        for nom, ok in conditions.items():
            print(f"  {'OK  ' if ok else 'NON '} {nom}")
        if familles_negatives:
            print(f"  familles négatives : {familles_negatives}")
    else:
        verdict["verdict"] = "INCOMPLET — bras manquants"
        print("\nmesure incomplète : il manque au moins un des bras c / p")

    # ------------------------------------------------------------------ plancher de bruit
    bruit = None
    if args.bruit:
        def mesurer(matrice: Matrix) -> tuple[int, int]:
            or_b = {cle: bool(servi(chercher(matrice, vecteurs_q[cle], scopes[cle], textes)) & gold[cle])
                    for cle in gold}
            return (gagnes_perdus(reference, or_b)["net"],
                    gagnes_perdus(reference, or_b, prefixe="v4_formula")["net"])

        # La distribution imitée est celle du bras identité s'il existe — c'est le vrai bruit
        # du ré-embarquement, mesuré ; sinon celle de l'échantillon de reproduction.
        if "z" in resultats:
            blob = np.load(CACHE / f"latex-z-{signature}.npz", allow_pickle=True)
            vecteurs_z = dict(zip(blob["ids"].tolist(), blob["vecteurs"]))
            cosinus = np.asarray([float(np.dot(matrix_ref.vectors[matrix_ref.position_of(c)],
                                              np.asarray(v) / max(float(np.linalg.norm(v)), 1e-12)))
                                  for c, v in vecteurs_z.items()
                                  if matrix_ref.position_of(c) is not None], dtype=np.float32)
            source = "bras identité (le vrai bruit du ré-embarquement, mesuré sur la population)"
        else:
            cosinus = np.asarray([0.999761, 0.999784, 0.99985, 0.99992, 1.0, 1.000062], dtype=np.float32)
            source = "échantillon de reproduction (le bras identité n'était pas disponible)"
        print(f"\nplancher de bruit : {args.bruit} tirages, cosinus imités de {source}")
        bruit = bras_bruit(matrix_ref, population, cosinus, args.bruit, mesurer) | {"source_des_cosinus": source}
        print(f"  net sous variable nulle : médiane {bruit['net']['mediane']:+d} · "
              f"[{bruit['net']['min']:+d} ; {bruit['net']['max']:+d}] · "
              f"écart-type {bruit['net']['ecart_type']}")
        print(f"  TAUX DE FAUSSE ALARME du seuil +3 : {100 * bruit['taux_de_fausse_alarme_du_seuil_3']:.1f} % "
              f"(bilatéral {100 * bruit['taux_de_fausse_alarme_bilateral_3']:.1f} %)")

    # ------------------------------------------------------------------ balayage d'ampleur
    balayage = None
    if args.balayage:
        def mesurer_b(matrice: Matrix) -> tuple[int, int]:
            or_b = {cle: bool(servi(chercher(matrice, vecteurs_q[cle], scopes[cle], textes)) & gold[cle])
                    for cle in gold}
            return (gagnes_perdus(reference, or_b)["net"],
                    gagnes_perdus(reference, or_b, prefixe="v4_formula")["net"])

        # La question que le bras identité rend décisive : il rend net 0 à cosinus 1,0, donc le
        # bruit de la procédure est nul. Reste à savoir si les +2 du candidat et les +4 du
        # placebo sont autre chose que l'effet de **l'ampleur** de leur perturbation. On
        # perturbe donc la même population, isotropiquement et sans aucun contenu, à des
        # cosinus choisis pour encadrer ceux des deux bras (0,9972 pour C, 0,9874 pour P).
        niveaux = (1.0, 0.999, 0.9972, 0.995, 0.99, 0.9874, 0.98, 0.96, 0.93, 0.90)
        print(f"\nbalayage d'ampleur : {args.balayage} tirages par niveau de cosinus, "
              f"perturbation isotrope sans contenu")
        print(f"  {'cosinus':>8} {'net moyen':>10} {'écart-type':>11} {'min':>5} {'max':>5} "
              f"{'net formula moyen':>18}")
        balayage = {}
        for niveau in niveaux:
            r = bras_bruit(matrix_ref, population, np.asarray([niveau], dtype=np.float32),
                           args.balayage, mesurer_b, bavard=False)
            balayage[f"{niveau}"] = r
            repere = ""
            if abs(niveau - 0.9972) < 1e-6:
                repere = "  <- ampleur du CANDIDAT (net réel +2)"
            elif abs(niveau - 0.9874) < 1e-6:
                repere = "  <- ampleur du PLACEBO (net réel +4)"
            print(f"  {niveau:8.4f} {r['net']['moyenne']:10.2f} {r['net']['ecart_type']:11.2f} "
                  f"{r['net']['min']:5d} {r['net']['max']:5d} "
                  f"{np.mean(r['net_formula_tous']):18.2f}{repere}")

    # ------------------------------------------------------------------ rangs formula
    tableau_rangs = {}
    for bras in ("z", "c", "p"):
        if bras not in rangs_bras:
            continue
        avant = Counter(bande(rangs_ref[k]) for k in gold if k.startswith("v4_formula/"))
        apres = Counter(bande(rangs_bras[bras][k]) for k in gold if k.startswith("v4_formula/"))
        tableau_rangs[bras] = {"avant": dict(avant), "apres": dict(apres)}
    if "c" in tableau_rangs:
        print("\nrangs de l'or, famille formula (45 questions)")
        bandes = ("1", "2-3", "4-5", "6-10", "11-20", "21-50", "hors pool")
        print(f"  {'bande':>10} {'référence':>10} {'candidat':>9} {'placebo':>8} {'identité':>9}")
        for b in bandes:
            print(f"  {b:>10} {tableau_rangs['c']['avant'].get(b, 0):>10} "
                  f"{tableau_rangs['c']['apres'].get(b, 0):>9} "
                  f"{tableau_rangs.get('p', {}).get('apres', {}).get(b, 0):>8} "
                  f"{tableau_rangs.get('z', {}).get('apres', {}).get(b, 0):>9}")

    # sort des ors hors pool à la référence
    hors_pool_ref = sorted(k for k in gold if k.startswith("v4_formula/") and rangs_ref[k] is None)
    sort_hors_pool = {k: {bras: rangs_bras[bras].get(k) for bras in rangs_bras} for k in hors_pool_ref}

    sortie = {
        "lot": "latex-recolle",
        "pre_enregistrement": "PRE-ENREGISTREMENT-LATEX-2026-09-10.md",
        "signature_corpus": signature,
        "modele": quant_rag.MODEL_ID, "device": quant_rag.device(), "batch": BATCH, "graine": SEED,
        "population": {"n": len(population), "part_du_corpus": round(len(population) / n_passages, 4)},
        "banc": {"questions": len(items), "par_famille": dict(Counter(b for b, _ in items)),
                 "ors_distincts": len(set().union(*gold.values())), "questions_sans_or": sans_or,
                 "or_servi_reference": sum(reference.values())},
        "bras": resultats,
        "verdict": verdict,
        "plancher_de_bruit": bruit,
        "balayage_d_ampleur": balayage,
        "rangs_formula": tableau_rangs,
        "ors_formula_hors_pool_a_la_reference": sort_hors_pool,
        "appels_llm": 0, "ecritures_qdrant": 0, "ecritures_overlay_servi": 0,
        "secondes": round(time.perf_counter() - debut, 1),
    }
    chemin = HERE / f"resultats-latex-{signature}.json"
    chemin.write_text(json.dumps(sortie, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n-> {chemin}")


if __name__ == "__main__":
    main()
