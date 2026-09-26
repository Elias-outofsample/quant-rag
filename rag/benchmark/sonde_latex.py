"""Sonde (d) — le LaTeX éclaté par MinerU dégrade-t-il le rang de l'or ?

Contexte : `PRE-ENREGISTREMENT-REPRESENTATION-2026-09-09.md` §5, et le diagnostic
`diagnostic_formules.py` qui a posé la question sans la trancher. Cette sonde ne décide
d'aucun changement de production : elle dit seulement si l'avenue mérite un lot à elle.
§8.7/§10 de `REPRESENTATION-CONCEPTION.md` l'excluent du lot en cours même si elle gagne.

MinerU rend les formules caractère par caractère : ``V _ { i , t } ^ { \\mathrm { b i d } }``
au lieu de ``V_{i,t}^{\\mathrm{bid}}``. Les deux bras :

  - **bras**    : dans la chaîne plongée des passages porteurs de ``$$``, chaque région
                  mathématique (``$$…$$`` ou ``$…$``) est recollée — cf. `recoller()`
                  ci-dessous. La formule n'est PAS renormalisée (pas de synonymes, pas de
                  polices unifiées comme dans `latex_norme.py`, qui sert un autre usage) :
                  seuls les espaces que MinerU a insérés entre caractères disparaissent.
  - **placebo**  : dans le MÊME passage, le MÊME nombre de caractères espace est retiré
                  puis réinséré à des positions aléatoires (graine publiée, §-graine plus
                  bas) — la chaîne est perturbée d'autant de caractères, sans recoller une
                  seule paire LaTeX. Il capture « on a touché la chaîne », pas « on l'a
                  rendue plus juste ».

La grandeur lue est l'« or servi » net (gagnés/perdus séparément) — d'abord sur la famille
`formula` (45 questions, la cible de la sonde), puis sur les 200 questions à or connu pour
voir si l'effet dépasse cette famille. Sélection de production reproduite sur la matrice
dense (`quant_rag._select`, limit=5, per_document=2, min_characters=250, dedupe) — jamais
Qdrant : un autre chantier de ce worktree le tient ouvert (corpus candidat, réembarquement).

Réduction de la population, et pourquoi elle est exacte et non approximative
-----------------------------------------------------------------------------
9 228 des 26 120 passages servis (35 %) portent un ``$$`` : les ré-embarquer tous, deux
fois (bras + placebo), déborderait largement l'heure de MPS. Mais un passage qu'on NE
PATCHE PAS garde un vecteur, donc un score, strictement identique pour toute question :
il ne peut donc changer aucun rang. Le seul sous-ensemble qui PEUT changer un « or servi »
sur nos 200 questions est : les chunks d'or ``$$`` de ces questions, union les chunks
``$$`` déjà présents dans le pool dense (top-50) d'au moins une d'entre elles à la
référence. Patcher hors de cet ensemble ne changerait la mesure d'aucune façon décelable ;
s'en tenir à cet ensemble n'est donc pas un raccourci qui perd du signal, c'est un
sous-ensemble suffisant *pour ce banc précis*. Un garde-fou temporel s'ajoute par
prudence (§ plus bas) : si même cet ensemble menace l'heure de MPS, il est sous-échantillonné
au hasard, avec sa graine, et le journal le dit — jamais en silence.

    .venv/bin/python rag/benchmark/sonde_latex.py
"""
from __future__ import annotations

import json
import random
import re
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CHECKPOINT_DIR = HERE / ".cache"


def encoder_avec_reprise(model, ids: list[str], textes_par_id: dict[str, str], batch_size: int,
                         cle_config: str, etiquette: str) -> dict[str, np.ndarray]:
    """`model.encode`, lot par lot, avec reprise sur `.cache/sonde_latex-<étiquette>.npz`.

    Même mécanisme que `sonde_troncature.encoder_avec_reprise` (dupliqué plutôt que partagé
    entre les deux sondes, chacune restant un script autonome) : un lot MPS a pris, pendant
    le développement de cette sonde, de quelques secondes à plusieurs minutes de façon
    imprévisible. Un checkpoint après chaque lot rend une exécution coupée reprenable.
    """
    chemin = CHECKPOINT_DIR / f"sonde_latex-{etiquette}.npz"
    fait: dict[str, np.ndarray] = {}
    if chemin.exists():
        blob = np.load(chemin, allow_pickle=True)
        if str(blob["cle_config"]) == cle_config:
            fait = dict(zip(blob["ids"].tolist(), blob["vecteurs"]))
            print(f"    {etiquette} : reprise, {len(fait)}/{len(ids)} déjà calculés", flush=True)
        else:
            print(f"    {etiquette} : checkpoint d'une autre configuration — ignoré, recalcul complet", flush=True)
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


ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "titles"))

import familles_v4  # noqa: E402
import pipeline  # noqa: E402
import quant_rag  # noqa: E402
import reembed_titles as rt  # noqa: E402
from compare_v1_v2 import load_bench, load_v1  # noqa: E402
from corpus import ChunkIndex  # noqa: E402
from dense_matrix import Matrix  # noqa: E402

RESULT = HERE / "sonde_latex.json"
BATCH = 8
SEED = 20260909  # graine publiée — celle du bootstrap du même lot (§7.2 du pré-enregistrement)
#: Budget dur, avec marge sous l'heure demandée : le calibrage se fait sur un petit lot,
#: puis la population est sous-échantillonnée si elle menace de le dépasser.
BUDGET_SECONDES = 40 * 60

# --------------------------------------------------------------------------- recollage LaTeX

#: Atome : une commande LaTeX nommée (``\mathrm``, ``\sum``…) ou tout autre caractère non
#: blanc, un par un — exactement l'éclatement que rend MinerU, donc l'inverse exact à défaire.
_ATOME = re.compile(r"\\[A-Za-z]+|\S")
#: Régions mathématiques d'un passage — même regex que `diagnostic_formules._MATHS` (``$$``
#: d'abord, sans quoi ``$…$`` couperait un bloc affiché en deux).
_MATHS = re.compile(r"\$\$.+?\$\$|\$[^$\n]+\$", re.DOTALL)


def recoller(expression: str) -> str:
    """Recolle un LaTeX éclaté caractère par caractère, sans changer la formule.

    Aucun séparateur entre deux atomes, SAUF quand l'atome qui précède est une commande
    nommée (``\\in``, ``\\mathrm``…) et que l'atome qui suit commence par une lettre : sans
    cette exception, ``\\in`` suivi de ``t`` redeviendrait ``\\int`` — une commande
    différente. Toute autre jointure est sans risque : un caractère nu (lettre, chiffre,
    ``_``, ``{``…) ne peut pas prolonger le nom d'une commande, et un ``\\`` démarre
    toujours sans ambiguïté un nouveau token, glue ou pas.
    Vérifié sur l'exemple même de `latex_norme.py` : ``V _ { i , t } ^ { \\mathrm { b i d
    } } = \\frac { 1 } { N } \\sum _ { s \\in t } V _ { i , s }`` redevient
    ``V_{i,t}^{\\mathrm{bid}}=\\frac{1}{N}\\sum_{s\\in t}V_{i,s}``.
    """
    atomes = _ATOME.findall(expression)
    sortie: list[str] = []
    for atome in atomes:
        if sortie:
            precedent = sortie[-1]
            risque = precedent.startswith("\\") and precedent[1:].isalpha() and atome[:1].isalpha()
            sortie.append((" " if risque else "") + atome)
        else:
            sortie.append(atome)
    return "".join(sortie)


def recoller_passage(texte: str) -> str:
    """Recolle chaque région ``$$…$$`` / ``$…$`` du passage ; la prose autour n'est pas touchée."""
    return _MATHS.sub(lambda m: recoller(m.group()), texte)


def melanger_espaces(texte: str, n: int, rng: random.Random) -> str:
    """Placebo : retire ``n`` espaces choisis au hasard, les réinsère à des positions
    aléatoires. Même longueur et même nombre d'espaces que le texte ORIGINAL (pas que le
    texte recollé, plus court) — le placebo perturbe la référence, il n'imite pas la taille
    du bras : c'est la sonde (c), pas celle-ci, qui porte la question de la longueur."""
    if n <= 0:
        return texte
    positions = [i for i, c in enumerate(texte) if c == " "]
    n = min(n, len(positions))
    retirer = set(rng.sample(positions, n))
    reste = [c for i, c in enumerate(texte) if i not in retirer]
    for _ in range(n):
        reste.insert(rng.randint(0, len(reste)), " ")
    return "".join(reste)


# --------------------------------------------------------------------------- questions (identique à sonde_troncature.py)

def charger_questions() -> list[tuple[str, dict]]:
    index = ChunkIndex.load(verbose=False)
    v1 = [("v1", it) for it in load_v1(index)]
    v3 = [("v3", it) for it in load_bench(HERE / "questions-v3.jsonl")]
    v4 = [("v4_formula", it) for it in familles_v4.lire_jsonl(HERE / "questions-v4-formula.jsonl")]
    return v1 + v3 + v4


def encoder_questions(items: list[tuple[str, dict]]) -> dict[str, np.ndarray]:
    return {f"{banc}/{it['qid']}": np.asarray(quant_rag.encode_query(pipeline.query_of(it)), dtype=np.float32)
            for banc, it in items}


def scopes_des_questions(items: list[tuple[str, dict]]) -> dict[str, list[str] | None]:
    out = {}
    for banc, it in items:
        filtres = {k: v for k, v in (pipeline.filters_of(it) or {}).items() if v is not None}
        out[f"{banc}/{it['qid']}"] = quant_rag.document_scope(None, **filtres) if filtres else None
    return out


def chercher(matrix: Matrix, vecteur: np.ndarray, scope, textes: dict[str, str]) -> list[dict]:
    pool = matrix.search(vecteur, pool=quant_rag.POOL, scope=scope, skip_headings=True)
    for row in pool:
        row["text"] = textes.get(row["chunk_id"], "")
    return pool


def or_servi_de(pool: list[dict], gold_chunks) -> bool:
    selection = quant_rag._select(pool, limit=5, per_document=2, dedupe=True,
                                  min_characters=quant_rag.MIN_CHARACTERS)
    servi = {r["chunk_id"] for r in selection}
    return bool(servi & set(gold_chunks or []))


def gagnes_perdus(reference: dict[str, bool], variant: dict[str, bool], prefixe: str | None = None) -> dict:
    cles_ref = [k for k in reference if prefixe is None or k.startswith(prefixe)]
    gagnes = sorted(k for k in cles_ref if variant.get(k, False) and not reference[k])
    perdus = sorted(k for k in cles_ref if reference[k] and not variant.get(k, False))
    return {"n": len(cles_ref), "gagnes": gagnes, "n_gagnes": len(gagnes), "perdus": perdus,
            "n_perdus": len(perdus), "net": len(gagnes) - len(perdus)}


def patcher(reference: Matrix, vecteurs: dict[str, np.ndarray]) -> tuple[Matrix, list[str]]:
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
    dollars = {r["chunk_id"] for r in rows if "$$" in r["text"]}
    print(f"passages porteurs de $$ (population totale) : {len(dollars)}/{len(rows)} "
          f"({100 * len(dollars) / len(rows):.1f} %)")

    items = charger_questions()
    gold_by_key = {f"{b}/{it['qid']}": it.get("gold_chunks") or [] for b, it in items}
    vecteurs = encoder_questions(items)
    scopes = scopes_des_questions(items)

    # surprise notée avant la sonde elle-même : la moitié de TOUT l'or du banc porte du $$,
    # pas seulement celui de la famille formula.
    tous_ors = set().union(*gold_by_key.values()) if gold_by_key else set()
    ors_dollar = {c for c in tous_ors if c in dollars}
    print(f"chunks d'or (v1+v3+v4_formula) porteurs de $$ : {len(ors_dollar)}/{len(tous_ors)}")

    # ----- référence, et construction simultanée de la population réduite -----
    pools_reference: dict[str, list[dict]] = {}
    for banc, it in items:
        cle = f"{banc}/{it['qid']}"
        pools_reference[cle] = chercher(matrix_ref, vecteurs[cle], scopes[cle], textes)
    reference = {cle: or_servi_de(pool, gold_by_key[cle]) for cle, pool in pools_reference.items()}
    print(f"or servi, référence : {sum(reference.values())}/{len(items)}  "
          f"({time.perf_counter() - started:.0f} s écoulées)")

    pool_chunks = {row["chunk_id"] for pool in pools_reference.values() for row in pool}
    population = sorted((pool_chunks | ors_dollar) & dollars)
    print(f"population réduite (or $$ des 200 questions ∪ $$ déjà dans un pool à la référence) : "
          f"{len(population)}  — un passage $$ hors de cet ensemble ne peut changer aucun or "
          f"servi de ce banc, puisque son vecteur n'est pas patché et son score ne bouge donc pas")

    # ----- garde-fou temporel : calibrage sur un petit lot avant de s'engager -----
    model = quant_rag.embedder()
    index_lignes = {r["chunk_id"]: r for r in rows}
    texte_de = lambda c, txt: rt.embedding_text(titles[index_lignes[c]["document_id"]]["clean_title"],
                                                index_lignes[c]["title_path"], txt)

    calibrage = sorted(population, key=lambda c: len(textes[c]))[: min(8, len(population))]
    t0 = time.perf_counter()
    _ = model.encode([texte_de(c, textes[c]) for c in calibrage], normalize_embeddings=True,
                     show_progress_bar=False, batch_size=BATCH)
    dt_calibrage = max(time.perf_counter() - t0, 1e-6)
    debit = len(calibrage) / dt_calibrage  # passages/s, un seul bras — le double pour bras+placebo
    temps_estime = 2 * len(population) / debit
    print(f"débit mesuré : {debit:.2f} passages/s -> estimation bras+placebo sur {len(population)} : "
          f"{temps_estime / 60:.1f} min (budget {BUDGET_SECONDES / 60:.0f} min)")

    rng_reduction = random.Random(SEED)
    if temps_estime > BUDGET_SECONDES:
        maximum = max(2, int(BUDGET_SECONDES * debit / 2))
        # les chunks d'or $$ sont gardés en priorité absolue : perdre un des leurs rendrait la
        # sonde aveugle exactement là où elle doit voir.
        prioritaires = [c for c in population if c in ors_dollar]
        reste = [c for c in population if c not in ors_dollar]
        rng_reduction.shuffle(reste)
        avant = len(population)
        population = sorted(set(prioritaires + reste[: max(0, maximum - len(prioritaires))]))
        print(f"  RÉDUCTION EXPLICITE : {avant} -> {len(population)} passages (budget {BUDGET_SECONDES}s, "
              f"débit {debit:.2f} p/s, graine {SEED}) — les {len(prioritaires)} chunks d'or $$ sont "
              f"conservés en priorité, le reste est un tirage aléatoire sans remise")
        reduction_notes = {"appliquee": True, "avant": avant, "apres": len(population),
                           "graine": SEED, "debit_mesure_passages_par_s": round(debit, 3)}
    else:
        reduction_notes = {"appliquee": False, "avant": len(population), "apres": len(population),
                           "graine": SEED, "debit_mesure_passages_par_s": round(debit, 3)}

    # ----- construction du bras et du placebo -----
    rng = random.Random(SEED)  # une seule graine, un seul ordre déterministe (chunk_id trié)
    textes_bras: dict[str, str] = {}
    textes_placebo: dict[str, str] = {}
    n_retires_total = 0
    for c in population:  # `population` est déjà triée : ordre déterministe pour la graine
        original = textes[c]
        colle = recoller_passage(original)
        n = len(original) - len(colle)
        n_retires_total += n
        textes_bras[c] = texte_de(c, colle)
        textes_placebo[c] = texte_de(c, melanger_espaces(original, n, rng))
    print(f"caractères espace retirés par le recollage, au total sur la population : {n_retires_total} "
          f"(moyenne {n_retires_total / max(len(population), 1):.0f}/passage)")

    # Triés par longueur au moment de l'encodage seulement — jamais `population` elle-même,
    # dont l'ordre trié par chunk_id fait la reproductibilité du placebo. Sans ce tri local,
    # un lot MPS mélange courts et longs, rembourre tout au plus long et force une
    # recompilation de noyau par forme de lot rencontrée (cf. `reembed_titles.reembed`).
    ordre_bras = sorted(population, key=lambda c: len(textes_bras[c]))
    ordre_placebo = sorted(population, key=lambda c: len(textes_placebo[c]))

    print(f"bras : ré-embarquement de {len(population)} passages (LaTeX recollé)…", flush=True)
    t0 = time.perf_counter()
    vecteurs_bras = encoder_avec_reprise(model, ordre_bras, textes_bras, BATCH, "bras", "bras")
    dt_bras = time.perf_counter() - t0
    print(f"  {dt_bras:.0f} s")

    print(f"placebo : ré-embarquement de {len(population)} passages (espaces déplacés au hasard)…", flush=True)
    t0 = time.perf_counter()
    vecteurs_placebo = encoder_avec_reprise(model, ordre_placebo, textes_placebo, BATCH, "placebo", "placebo")
    dt_placebo = time.perf_counter() - t0
    print(f"  {dt_placebo:.0f} s")

    matrix_bras, absents_bras = patcher(matrix_ref, vecteurs_bras)
    matrix_placebo, absents_placebo = patcher(matrix_ref, vecteurs_placebo)
    if absents_bras or absents_placebo:
        print(f"  ATTENTION : chunk_ids absents de la matrice — bras={len(absents_bras)} placebo={len(absents_placebo)}")

    or_bras = {cle: or_servi_de(chercher(matrix_bras, vecteurs[cle], scopes[cle], textes), gold_by_key[cle])
              for cle in gold_by_key}
    or_placebo = {cle: or_servi_de(chercher(matrix_placebo, vecteurs[cle], scopes[cle], textes), gold_by_key[cle])
                 for cle in gold_by_key}

    diff_bras_formula = gagnes_perdus(reference, or_bras, prefixe="v4_formula/")
    diff_placebo_formula = gagnes_perdus(reference, or_placebo, prefixe="v4_formula/")
    diff_bras_tous = gagnes_perdus(reference, or_bras)
    diff_placebo_tous = gagnes_perdus(reference, or_placebo)
    contraste_formula = diff_bras_formula["net"] - diff_placebo_formula["net"]
    contraste_tous = diff_bras_tous["net"] - diff_placebo_tous["net"]
    bat_le_placebo = contraste_formula > 0
    verdict = ("le bras bat son placebo sur la famille formula : l'avenue « recoller le LaTeX » "
               "mérite un lot à elle (hors de ce lot — §8.7/§10 de REPRESENTATION-CONCEPTION.md)"
               if bat_le_placebo else
               "le bras ne bat pas son placebo sur la famille formula : l'avenue se ferme, avec ses chiffres")

    resultat = {
        "sonde": "d_latex_eclate",
        "regle_de_lecture": ("un bras qui ne bat pas son placebo en or servi net ferme son avenue ; "
                             "un bras qui le bat ouvre un lot à part, il n'entre pas dans celui-ci "
                             "(PRE-ENREGISTREMENT-REPRESENTATION-2026-09-09.md §5)"),
        "signature_corpus": matrix_ref.signature,
        "modele": quant_rag.MODEL_ID, "device": quant_rag.device(), "batch_size": BATCH, "graine": SEED,
        "population_dollar_totale": len(dollars),
        "population_dollar_part_du_corpus": round(len(dollars) / len(rows), 4),
        "surprise_or_dollar": (f"{len(ors_dollar)}/{len(tous_ors)} chunks d'or de TOUT le banc "
                              "(v1+v3+v4_formula, pas seulement formula) portent du $$ — l'éclatement "
                              "LaTeX n'est pas un artefact propre à la famille formula"),
        "reduction_population": {
            **reduction_notes,
            "justification": ("population = chunks d'or $$ des 200 questions ∪ chunks $$ déjà présents "
                              "dans le pool dense (top-50) d'au moins une question à la référence — un "
                              "chunk $$ hors de cet ensemble garde un vecteur non patché, donc un score "
                              "inchangé pour toute question, donc ne peut changer aucun or servi de ce "
                              "banc ; le sous-échantillonnage éventuel au-delà ne touche jamais les ors."),
        },
        "n_population_mesuree": len(population),
        "caracteres_espace_retires_total": n_retires_total,
        "questions": {"n": len(items), "n_formula": sum(1 for b, _ in items if b == "v4_formula")},
        "or_servi_reference": sum(reference.values()),
        "bras": {"secondes": round(dt_bras, 1), "formula": diff_bras_formula, "tous": diff_bras_tous},
        "placebo": {
            "secondes": round(dt_placebo, 1),
            "definition": ("mêmes passages, même nombre de caractères espace déplacés au hasard (graine "
                          f"{SEED}), même longueur que le texte ORIGINAL — capture la perturbation de la "
                          "chaîne, pas le recollage"),
            "formula": diff_placebo_formula, "tous": diff_placebo_tous,
        },
        "contraste_net_formula": contraste_formula,
        "contraste_net_tous": contraste_tous,
        "bat_le_placebo_formula": bat_le_placebo,
        "verdict": verdict,
        "appels_llm": 0,
        "ecritures_qdrant": 0,
        "ecritures_overlay": 0,
        "secondes_totales": round(time.perf_counter() - started, 1),
    }
    RESULT.write_text(json.dumps(resultat, ensure_ascii=False, indent=1), encoding="utf-8")
    for etiquette in ("bras", "placebo"):
        (CHECKPOINT_DIR / f"sonde_latex-{etiquette}.npz").unlink(missing_ok=True)

    print(f"\n[formula] bras    gagnés={diff_bras_formula['n_gagnes']:3d}  perdus={diff_bras_formula['n_perdus']:3d}  "
          f"net={diff_bras_formula['net']:+d}")
    print(f"[formula] placebo gagnés={diff_placebo_formula['n_gagnes']:3d}  perdus={diff_placebo_formula['n_perdus']:3d}  "
          f"net={diff_placebo_formula['net']:+d}")
    print(f"[formula] contraste net (bras - placebo) : {contraste_formula:+d}")
    print(f"[tous]    contraste net (bras - placebo) : {contraste_tous:+d}")
    print(verdict)
    print(f"\n-> {RESULT}  ({time.perf_counter() - started:.0f} s au total)")


if __name__ == "__main__":
    main()
