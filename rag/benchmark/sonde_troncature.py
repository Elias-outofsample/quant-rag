"""Sonde (c) — la fenêtre de 1 024 jetons coupe-t-elle le classement des passages longs ?

Contexte : `PRE-ENREGISTREMENT-REPRESENTATION-2026-09-09.md` §5. Cette sonde ne décide
d'aucun changement de production — elle dit seulement si l'avenue « relever `MAX_LENGTH` »
mérite un lot à elle. §8.7 et §10 de `REPRESENTATION-CONCEPTION.md` l'interdisent d'entrer
dans le lot en cours même si elle gagne : une sonde gagnante ouvre son propre lot, elle ne
s'ajoute pas à un Δ qui porte déjà une autre variable.

Le protocole, et pourquoi il a deux bras
-----------------------------------------
Ce dépôt a payé deux fois pour avoir mesuré un bras sans placebo : le graphe (+0,025 naïf,
−0,075 contre placebo) et le débalisage (placebo identique question par question). Donc :

  - **bras**    : les passages servis de plus de 4 000 caractères, ré-embarqués avec
                  `model.max_seq_length = 2048` (Qwen3-Embedding-0.6B, la recette exacte de
                  `src/embeddings/base.py` / `titles/reembed_titles.py` : ``Document:
                  <titre propre>\\nPath: <title_path>\\n\\n<texte>``) ;
  - **placebo**  : les MÊMES passages, ré-embarqués à `max_seq_length` inchangé (1 024) —
                  un pur recalcul. Il capture le bruit de recomposition des lots en fp16
                  (le dernier bit bouge), pas la fenêtre. Sans lui, ce bruit serait imputé
                  à `max_length`.

La grandeur lue est l'« or servi » net (gagnés/perdus séparément, jamais seulement le
solde) sur les 200 questions à or connu (v1 + v3 positives + v4 `formula`), calculé en
reproduisant la sélection de production — `quant_rag._select`, limit=5, per_document=2,
min_characters=250, dedupe — **sur la matrice dense, jamais sur Qdrant** : `dense_matrix.py`
existe précisément pour ça, et un autre chantier de ce worktree tient Qdrant/le corpus
candidat ouverts en ce moment.

Zéro appel LLM. Zéro écriture Qdrant, overlay, table ou titre — seul un fichier JSON est
écrit, à côté de ce script.

    .venv/bin/python rag/benchmark/sonde_troncature.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np


CHECKPOINT_DIR = Path(__file__).resolve().parent / ".cache"


def encoder_avec_reprise(model, ids: list[str], textes_par_id: dict[str, str], batch_size: int,
                         cle_config: str, etiquette: str) -> dict[str, np.ndarray]:
    """`model.encode`, lot par lot, avec reprise sur un `.cache/sonde_troncature-<étiquette>.npz`.

    Sur cette machine, un lot MPS peut prendre de quelques secondes à plusieurs minutes de
    façon imprévisible (contention observée pendant le développement de cette sonde), et un
    seul budget d'appel d'outil ne suffit pas toujours à traverser 684 passages. Chaque lot
    terminé est donc sauvegardé aussitôt : une exécution coupée reprend exactement où elle
    s'est arrêtée au lieu de recommencer — le même mécanisme que `reembed_titles.PARTIAL`.
    `cle_config` (ex. ``"bras-2048"``) protège contre une reprise sur la mauvaise variante
    si le script est relancé après avoir changé `max_length` entre-temps.
    """
    chemin = CHECKPOINT_DIR / f"sonde_troncature-{etiquette}.npz"
    fait: dict[str, np.ndarray] = {}
    if chemin.exists():
        blob = np.load(chemin, allow_pickle=True)
        if str(blob["cle_config"]) == cle_config:
            fait = dict(zip(blob["ids"].tolist(), blob["vecteurs"]))
            print(f"    {etiquette} : reprise, {len(fait)}/{len(ids)} déjà calculés", flush=True)
        else:
            print(f"    {etiquette} : checkpoint d'une autre configuration ({blob['cle_config']} != "
                 f"{cle_config}) — ignoré, recalcul complet", flush=True)
    restant = [c for c in ids if c not in fait]
    debut = time.perf_counter()
    for i in range(0, len(restant), batch_size):
        lot_ids = restant[i:i + batch_size]
        lot_textes = [textes_par_id[c] for c in lot_ids]
        t0 = time.perf_counter()
        vecteurs = model.encode(lot_textes, normalize_embeddings=True, show_progress_bar=False,
                                batch_size=batch_size)
        dt = time.perf_counter() - t0
        for c, v in zip(lot_ids, vecteurs):
            fait[c] = v
        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        np.savez(chemin, ids=np.asarray(list(fait)), vecteurs=np.vstack(list(fait.values())), cle_config=cle_config)
        ecoule = time.perf_counter() - debut
        print(f"    {etiquette} {len(fait)}/{len(ids)}  (lot {dt:.1f}s, {ecoule:.0f}s écoulées depuis reprise)",
             flush=True)
    return fait


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "titles"))

import familles_v4  # noqa: E402  — lire_jsonl, même lecture que le reste du banc v4
import pipeline  # noqa: E402  — query_of / filters_of : le même texte et les mêmes filtres qu'en production
import quant_rag  # noqa: E402  — _select, MIN_CHARACTERS, POOL, document_scope, encode_query, embedder
import reembed_titles as rt  # noqa: E402  — la recette d'embedding et le texte servi : pas réimplémentés
from compare_v1_v2 import load_bench, load_v1  # noqa: E402
from corpus import ChunkIndex  # noqa: E402
from dense_matrix import Matrix  # noqa: E402

RESULT = HERE / "sonde_troncature.json"
SEUIL_CARACTERES = 4000
MAX_LENGTH_BRAS = 2048
BATCH = 8  # même lot que reembed_titles.BATCH — comparable au recalcul de production


# --------------------------------------------------------------------------- questions

def charger_questions() -> list[tuple[str, dict]]:
    """v1 (25) + v3 positives (130, les 20 négatives n'ont pas d'or : « or servi » n'a pas
    de sens pour elles — choix explicite, elles sont donc absentes de la population) +
    v4 `formula` (45). 200 questions à or connu."""
    index = ChunkIndex.load(verbose=False)
    v1 = [("v1", it) for it in load_v1(index)]
    v3 = [("v3", it) for it in load_bench(HERE / "questions-v3.jsonl")]
    v4 = [("v4_formula", it) for it in familles_v4.lire_jsonl(HERE / "questions-v4-formula.jsonl")]
    return v1 + v3 + v4


def encoder_questions(items: list[tuple[str, dict]]) -> dict[str, np.ndarray]:
    """Un vecteur par question, calculé UNE fois et réutilisé pour référence/bras/placebo :
    ré-encoder la question à chaque bras mélangerait le bruit fp16 de la question à celui,
    seul intéressant ici, des vecteurs de passage qu'on manipule."""
    return {f"{banc}/{it['qid']}": np.asarray(quant_rag.encode_query(pipeline.query_of(it)), dtype=np.float32)
            for banc, it in items}


def scopes_des_questions(items: list[tuple[str, dict]]) -> dict[str, list[str] | None]:
    out = {}
    for banc, it in items:
        filtres = {k: v for k, v in (pipeline.filters_of(it) or {}).items() if v is not None}
        out[f"{banc}/{it['qid']}"] = quant_rag.document_scope(None, **filtres) if filtres else None
    return out


# --------------------------------------------------------------------------- sélection servie

def or_servi(matrix: Matrix, textes: dict[str, str], items: list[tuple[str, dict]],
            vecteurs: dict[str, np.ndarray], scopes: dict) -> dict[str, bool]:
    """Reproduit `quant_rag._select` (limit=5, per_document=2, min_characters=250, dedupe=True)
    à partir du pool dense de la matrice fournie — jamais Qdrant, c'est tout l'intérêt de
    patcher `dense_matrix.Matrix` en mémoire plutôt que d'appeler `quant_rag.search`."""
    out = {}
    for banc, it in items:
        cle = f"{banc}/{it['qid']}"
        pool = matrix.search(vecteurs[cle], pool=quant_rag.POOL, scope=scopes[cle], skip_headings=True)
        for row in pool:
            row["text"] = textes.get(row["chunk_id"], "")
        selection = quant_rag._select(pool, limit=5, per_document=2, dedupe=True,
                                      min_characters=quant_rag.MIN_CHARACTERS)
        servi = {r["chunk_id"] for r in selection}
        out[cle] = bool(servi & set(it.get("gold_chunks") or []))
    return out


def gagnes_perdus(reference: dict[str, bool], variant: dict[str, bool]) -> dict:
    gagnes = sorted(k for k in variant if variant[k] and not reference.get(k, False))
    perdus = sorted(k for k in reference if reference[k] and not variant.get(k, False))
    return {"gagnes": gagnes, "n_gagnes": len(gagnes), "perdus": perdus, "n_perdus": len(perdus),
            "net": len(gagnes) - len(perdus)}


def patcher(reference: Matrix, vecteurs: dict[str, np.ndarray]) -> tuple[Matrix, list[str]]:
    """Une copie de `reference` où les vecteurs listés sont remplacés — en mémoire seulement.
    Chaque vecteur patché est renormalisé individuellement, comme le fait `Matrix.load` pour
    tout overlay avant la normalisation globale ; ici la normalisation globale a déjà eu lieu,
    donc chaque remplacement se renormalise lui-même."""
    matrice = reference.vectors.copy()
    absents = []
    for chunk_id, vecteur in vecteurs.items():
        i = reference.position_of(chunk_id)
        if i is None:
            absents.append(chunk_id)
            continue
        v = np.asarray(vecteur, dtype=np.float32)
        matrice[i] = v / max(float(np.linalg.norm(v)), 1e-12)
    patchee = Matrix(reference.chunk_ids, reference.document_ids, matrice,
                     reference.label + " (patchée)", reference.content_types)
    return patchee, absents


# --------------------------------------------------------------------------- programme

def main() -> None:
    started = time.perf_counter()
    print("chargement du corpus servi (dense_matrix, sans Qdrant)…")
    matrix_ref = Matrix.load()
    print(f"  signature {matrix_ref.signature} · {len(matrix_ref.chunk_ids)} passages")

    rows, export_title = rt.load_rows()
    titles = rt.titles_table(export_title)
    textes = {r["chunk_id"]: r["text"] for r in rows}

    longs = [r for r in rows if len(r["text"]) > SEUIL_CARACTERES]
    print(f"population : {len(longs)} passages servis > {SEUIL_CARACTERES} caractères")

    # Corroboration, pas une deuxième population : le seuil en caractères n'est qu'un proxy
    # de la vraie coupe, qui se fait en JETONS sur la chaîne plongée complète
    # (Document:/Path:/texte). Elle dit ce que la population > 4 000 caractères vaut
    # réellement — la sonde garde la population du pré-enregistrement, en caractères.
    model = quant_rag.embedder()
    textes_complets = {r["chunk_id"]: rt.embedding_text(titles[r["document_id"]]["clean_title"],
                                                        r["title_path"], r["text"]) for r in longs}
    print(f"  tokenisation de contrôle sur {len(textes_complets)} chaînes…", flush=True)
    t0 = time.perf_counter()
    depasse_1024 = 0
    for j, t in enumerate(textes_complets.values(), 1):
        if len(model.tokenizer(t, truncation=False)["input_ids"]) > quant_rag.MAX_LENGTH:
            depasse_1024 += 1
        if j % 100 == 0:
            print(f"    tokenisation {j}/{len(textes_complets)}  ({time.perf_counter() - t0:.0f}s écoulées)",
                 flush=True)
    print(f"  dont réellement tronqués à {quant_rag.MAX_LENGTH} jetons (chaîne plongée complète) : "
          f"{depasse_1024}/{len(longs)}  ({time.perf_counter() - t0:.0f}s)")

    items = charger_questions()
    print(f"{len(items)} questions à or connu (v1+v3 positives+v4_formula)")
    vecteurs = encoder_questions(items)
    scopes = scopes_des_questions(items)

    reference = or_servi(matrix_ref, textes, items, vecteurs, scopes)
    print(f"or servi, référence : {sum(reference.values())}/{len(items)}  "
          f"({time.perf_counter() - started:.0f} s écoulées)")

    # Triés par longueur avant le lot, comme `reembed_titles.reembed` : sans ça, un lot MPS
    # mélange des séquences courtes et longues, rembourre tout à la plus longue et force le
    # backend à recompiler un noyau par forme de lot rencontrée — le vrai coût, pas le calcul.
    ids = sorted((r["chunk_id"] for r in longs), key=lambda c: len(textes_complets[c]))
    a_embarquer = [textes_complets[c] for c in ids]

    print(f"bras : ré-embarquement de {len(ids)} passages à max_length={MAX_LENGTH_BRAS}…", flush=True)
    t0 = time.perf_counter()
    model.max_seq_length = MAX_LENGTH_BRAS
    vecteurs_bras = encoder_avec_reprise(model, ids, textes_complets, BATCH, f"bras-{MAX_LENGTH_BRAS}", "bras")
    dt_bras = time.perf_counter() - t0
    print(f"  {dt_bras:.0f} s")

    print(f"placebo : ré-embarquement des mêmes passages à max_length={quant_rag.MAX_LENGTH} (recalcul, rien ne change)…",
         flush=True)
    t0 = time.perf_counter()
    model.max_seq_length = quant_rag.MAX_LENGTH  # restauré AVANT le placebo — c'est sa définition
    vecteurs_placebo = encoder_avec_reprise(model, ids, textes_complets, BATCH,
                                            f"placebo-{quant_rag.MAX_LENGTH}", "placebo")
    dt_placebo = time.perf_counter() - t0
    print(f"  {dt_placebo:.0f} s")

    matrix_bras, absents_bras = patcher(matrix_ref, vecteurs_bras)
    matrix_placebo, absents_placebo = patcher(matrix_ref, vecteurs_placebo)
    if absents_bras or absents_placebo:
        print(f"  ATTENTION : chunk_ids absents de la matrice — bras={len(absents_bras)} placebo={len(absents_placebo)}")

    or_bras = or_servi(matrix_bras, textes, items, vecteurs, scopes)
    or_placebo = or_servi(matrix_placebo, textes, items, vecteurs, scopes)

    diff_bras = gagnes_perdus(reference, or_bras)
    diff_placebo = gagnes_perdus(reference, or_placebo)
    contraste_net = diff_bras["net"] - diff_placebo["net"]
    bat_le_placebo = contraste_net > 0
    verdict = ("le bras bat son placebo : l'avenue « relever max_length » mérite un lot à elle "
               "(hors de ce lot — §8.7/§10 de REPRESENTATION-CONCEPTION.md)" if bat_le_placebo else
               "le bras ne bat pas son placebo : l'avenue se ferme, avec ses chiffres")

    resultat = {
        "sonde": "c_troncature",
        "regle_de_lecture": ("un bras qui ne bat pas son placebo en or servi net ferme son avenue ; "
                             "un bras qui le bat ouvre un lot à part, il n'entre pas dans celui-ci "
                             "(PRE-ENREGISTREMENT-REPRESENTATION-2026-09-09.md §5)"),
        "signature_corpus": matrix_ref.signature,
        "modele": quant_rag.MODEL_ID, "device": quant_rag.device(), "batch_size": BATCH,
        "seuil_caracteres": SEUIL_CARACTERES,
        "population": {
            "n": len(longs),
            "note_692_vs_684": ("le §5 du pré-enregistrement du 9 septembre cite 692 passages > 4 000 "
                                "caractères ; cette sonde en mesure 684 sur la signature servie e1bdf36e2e. "
                                "L'écart n'est pas ré-audité ici ; l'hypothèse la plus probable est la "
                                "réparation des 989 en-têtes de tableaux (c41b49c, §0 du même document), "
                                "postérieure au comptage de 692, qui a pu déplacer une poignée de chunks "
                                "de part et d'autre du seuil."),
            "reellement_tronques_a_1024_jetons": depasse_1024,
            "surprise": (f"sur les {len(longs)} passages sélectionnés pour dépasser 4 000 caractères, "
                        f"seuls {depasse_1024} dépassent RÉELLEMENT 1 024 jetons une fois plongés. "
                        "Le seuil en caractères est un proxy bruité de la vraie coupe : la quasi-totalité "
                        "du bras n'a rien à gagner de max_length=2048 sinon du bruit fp16 — exactement ce "
                        "que le placebo mesure. Un signal, s'il existe, ne peut venir statistiquement que "
                        f"de ces {depasse_1024} passages-là."),
        },
        "questions": {"n": len(items), "cles": [f"{b}/{it['qid']}" for b, it in items]},
        "or_servi_reference": sum(reference.values()),
        "bras": {"max_length": MAX_LENGTH_BRAS, "secondes": round(dt_bras, 1), **diff_bras},
        "placebo": {
            "max_length": quant_rag.MAX_LENGTH, "secondes": round(dt_placebo, 1),
            "definition": ("les mêmes passages, recalculés à max_length inchangé (1 024) — capture le "
                          "bruit de recalcul fp16 (composition des lots, dernier bit), pas la fenêtre"),
            **diff_placebo,
        },
        "contraste_net": contraste_net,
        "bat_le_placebo": bat_le_placebo,
        "verdict": verdict,
        "appels_llm": 0,
        "ecritures_qdrant": 0,
        "ecritures_overlay": 0,
        "secondes_totales": round(time.perf_counter() - started, 1),
    }
    RESULT.write_text(json.dumps(resultat, ensure_ascii=False, indent=1), encoding="utf-8")
    # Nettoyage : les checkpoints de reprise n'ont plus d'usage une fois le JSON final écrit.
    for etiquette in ("bras", "placebo"):
        (CHECKPOINT_DIR / f"sonde_troncature-{etiquette}.npz").unlink(missing_ok=True)

    print(f"\nbras    gagnés={diff_bras['n_gagnes']:3d}  perdus={diff_bras['n_perdus']:3d}  net={diff_bras['net']:+d}")
    print(f"placebo gagnés={diff_placebo['n_gagnes']:3d}  perdus={diff_placebo['n_perdus']:3d}  net={diff_placebo['net']:+d}")
    print(f"contraste net (bras - placebo) : {contraste_net:+d}")
    print(verdict)
    print(f"\n-> {RESULT}  ({time.perf_counter() - started:.0f} s au total)")


if __name__ == "__main__":
    main()
