"""Le retrait des balises `<sub>` / `<sup>` — trois bras, hors ligne, rien en production.

Instrument du pré-enregistrement ``PRE-ENREGISTREMENT-DEBALISAGE-2026-09-08.md``, Partie I
publiée en ``8ca3a2d`` **avant le premier vecteur candidat**.

Les vecteurs des 3 505 chunks porteurs de balises sont recalculés avec la recette **exacte** de
la production (``rechunk_corpus.retrieval_text`` + ``quant_rag.embedder``) et **remplacés en
mémoire** dans ``dense_matrix.Matrix``. Le top-50 des 155 questions est recalculé, puis la
sélection.

**Aucune écriture dans Qdrant, aucun changement de signature, gel non levé, zéro appel LLM.**

Trois bras
----------
=============  ==========================================================
``reference``  les vecteurs servis aujourd'hui, tels quels
``candidat``   ``re.sub(r"</?(?:sub|sup)>", "", texte)`` — retrait pur
``placebo``    balises remplacées par un marqueur neutre **de même longueur**
               (``<sub>``→``<zzz>``, ``</sub>``→``</zzz>``, idem ``sup``)
=============  ==========================================================

Le placebo déplace exactement autant de vecteurs, avec la même perturbation de longueur, sans
retirer le bruit. **S'il gagne autant, le gain n'est pas la réparation.**

Reprise
-------
Les vecteurs de chaque bras sont écrits dans ``.cache/debalisage-<bras>-<signature>.npz`` par
blocs de 512. Une coupure coûte au plus un bloc, jamais les 24 minutes du bras.

    .venv/bin/python rag/benchmark/eval_debalisage.py --etape vecteurs
    .venv/bin/python rag/benchmark/eval_debalisage.py --etape verdict
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(ROOT / "rag" / "ingestion"))

import numpy as np  # noqa: E402

import corpus_overlay  # noqa: E402
import eval_reranking  # noqa: E402
import pipeline  # noqa: E402
import quant_rag  # noqa: E402
from corpus import ChunkIndex  # noqa: E402
from rechunk_corpus import retrieval_text, titres_plonges  # noqa: E402

SIGNATURE = corpus_overlay.signature()
BALISE = re.compile(r"</?(?:sub|sup)>")
BATCH, BLOC = 8, 512
CACHE = HERE / ".cache"
BRAS = ("candidat", "placebo")


def transformer(texte: str, bras: str) -> str:
    if bras == "candidat":
        return BALISE.sub("", texte)
    if bras == "placebo":
        # même longueur, même structure, jeton différent : la perturbation sans la réparation
        return (texte.replace("<sub>", "<zzz>").replace("</sub>", "</zzz>")
                     .replace("<sup>", "<zzz>").replace("</sup>", "</zzz>"))
    raise ValueError(bras)


def cibles(index: ChunkIndex) -> list[str]:
    return sorted(c for c, r in index.chunks.items() if BALISE.search(r.get("text") or ""))


def vecteurs(bras: str) -> None:
    index = ChunkIndex.load(verbose=False)
    titres, _ = titres_plonges()
    cles = cibles(index)
    chemin = CACHE / f"debalisage-{bras}-{SIGNATURE}.npz"
    faits: dict[str, np.ndarray] = {}
    if chemin.exists():
        charge = np.load(chemin, allow_pickle=False)
        faits = {c: charge["v"][i] for i, c in enumerate(charge["c"])}
        print(f"reprise : {len(faits)} vecteurs déjà calculés")
    restants = [c for c in cles if c not in faits]
    modele = quant_rag.embedder()
    depart = time.perf_counter()
    for i in range(0, len(restants), BLOC):
        lot = restants[i:i + BLOC]
        textes = []
        for c in lot:
            r = index.get(c)
            textes.append(retrieval_text(titres.get(r["document_id"], ""), r.get("title_path"),
                                         r.get("page_start"), transformer(r["text"], bras),
                                         avec_page=False))
        # tri par longueur, comme apply_delivery : la composition des lots entre dans les
        # vecteurs en fp16, et la reproduire est la condition de comparabilité
        ordre = sorted(range(len(lot)), key=lambda k: len(textes[k]))
        v = modele.encode([textes[k] for k in ordre], normalize_embeddings=True,
                          show_progress_bar=False, batch_size=BATCH)
        for rang, k in enumerate(ordre):
            faits[lot[k]] = np.asarray(v[rang], dtype=np.float32)
        cles_faites = sorted(faits)
        np.savez(chemin, c=np.array(cles_faites), v=np.stack([faits[c] for c in cles_faites]))
        ecoule = time.perf_counter() - depart
        reste = (len(restants) - i - len(lot)) * ecoule / max(1, i + len(lot))
        print(f"  {len(faits)}/{len(cles)}  reste ~{reste/60:.1f} min", end="\r", flush=True)
    print(f"\n{bras} : {len(faits)} vecteurs écrits dans {chemin.name}")


def _ecrire(chemin: Path, donnees: dict) -> None:
    chemin.parent.mkdir(parents=True, exist_ok=True)
    chemin.write_text(json.dumps(donnees, ensure_ascii=False), encoding="utf-8")


def _matrice_patchee(bras: str | None):
    from dense_matrix import Matrix

    m = Matrix.load()
    if bras is None:
        return m, 0
    charge = np.load(CACHE / f"debalisage-{bras}-{SIGNATURE}.npz", allow_pickle=False)
    par_cle = {c: charge["v"][i] for i, c in enumerate(charge["c"])}
    position = {str(c): i for i, c in enumerate(m.chunk_ids)}
    # Empreinte par projection aléatoire : un vecteur de 26 120 flottants (100 ko) au lieu
    # d'une copie de la matrice (107 Mo). Trois bras × copie = la mémoire qui a tué le premier
    # passage sur une machine de 16 Go déjà en swap.
    graine = np.random.default_rng(20260908)
    sonde = graine.standard_normal(m.vectors.shape[1]).astype(np.float32)
    avant = m.vectors @ sonde
    touches = 0
    for c, v in par_cle.items():
        i = position.get(c)
        if i is not None:
            m.vectors[i] = v
            touches += 1
    # contrôle d'intégrité du §3 : rien d'autre n'a bougé
    apres = m.vectors @ sonde
    change = set(np.where(avant != apres)[0].tolist())
    attendus = {position[c] for c in par_cle if c in position}
    if change - attendus:
        raise SystemExit(f"correctif de matrice faux : {len(change - attendus)} "
                         "lignes hors cible ont bougé")
    return m, touches


def or_servi(m, items, index) -> tuple[dict, dict]:
    """{cle: bool} par ``build_context`` (primaire) et par ``_select`` (diagnostic)."""
    primaire, production = {}, {}
    for cle, item in items.items():
        gold = set(item["gold_chunks"])
        filtres = {k: v for k, v in (pipeline.filters_of(item) or {}).items()
                   if k in pipeline.FILTER_KEYS and v is not None}
        scope = quant_rag.document_scope(None, **filtres) if filtres else None
        vecteur = np.asarray(quant_rag.encode_query(pipeline.query_of(item)), dtype=np.float32)
        rangs = m.search(vecteur, pool=quant_rag.POOL, scope=scope)
        lignes = []
        for r in rangs:
            src = index.get(r["chunk_id"]) or {}
            lignes.append({**r, "text": src.get("text", ""), "section": src.get("section")})
        primaire[cle] = bool(gold & {p["chunk_id"] for p in pipeline.build_context(lignes)})
        production[cle] = eval_reranking.or_servi(lignes, gold)
    return primaire, production


def bras_seul(nom: str) -> None:
    """Un bras, un processus, un fichier — la matrice et le modèle tiennent mal à trois.

    Le premier passage chargeait les trois bras dans le même processus et se faisait tuer sans
    trace sur une machine de 16 Go déjà en swap. Chaque bras écrit désormais son résultat dans
    ``.cache/debalisage-servi-<bras>-<signature>.json`` et ``verdict`` les recolle.
    """
    import experiment

    index = ChunkIndex.load(verbose=False)
    items = {i["key"]: i for i in experiment.load_items(index) if i["gold_chunks"]}
    m, touches = _matrice_patchee(None if nom == "reference" else nom)
    prim, prod = or_servi(m, items, index)
    _ecrire(CACHE / f"debalisage-servi-{nom}-{SIGNATURE}.json",
            {"vecteurs_remplaces": touches, "primaire": prim, "production": prod})
    print(f"  {nom:10s} vecteurs remplacés {touches:5d}  "
          f"or servi build_context {sum(prim.values()):3d}  _select {sum(prod.values()):3d}")


def verdict() -> dict:
    import experiment

    index = ChunkIndex.load(verbose=False)
    items = {i["key"]: i for i in experiment.load_items(index) if i["gold_chunks"]}
    cles_balisees = set(cibles(index))
    #: questions dont AUCUN or ne porte de balise — la population de garde du §5.2
    intactes = [c for c, i in items.items() if not (set(i["gold_chunks"]) & cles_balisees)]

    resultats = {}
    for nom in ("reference", "candidat", "placebo"):
        chemin = CACHE / f"debalisage-servi-{nom}-{SIGNATURE}.json"
        if not chemin.exists():
            raise SystemExit(f"{chemin.name} absent — lancer --etape bras --bras {nom}")
        resultats[nom] = json.loads(chemin.read_text(encoding="utf-8"))

    ref = resultats["reference"]
    out = {"signature": SIGNATURE, "appels_llm": 0, "qdrant": "jamais écrit",
           "pre_enregistrement": "PRE-ENREGISTREMENT-DEBALISAGE-2026-09-08.md (8ca3a2d)",
           "population": len(items), "chunks_balises": len(cles_balisees),
           "questions_a_or_intact": len(intactes), "bras": {}}
    for nom in ("reference", "candidat", "placebo"):
        r = resultats[nom]
        bloc = {"vecteurs_remplaces": r["vecteurs_remplaces"]}
        for champ in ("primaire", "production"):
            servi = r[champ]
            gagnees = sorted(c for c in items if servi[c] and not ref[champ][c])
            perdues = sorted(c for c in items if ref[champ][c] and not servi[c])
            gi = [c for c in gagnees if c in intactes]
            pi = [c for c in perdues if c in intactes]
            bloc[champ] = {"or_servi": sum(servi.values()),
                           "net": sum(servi.values()) - sum(ref[champ].values()),
                           "gagnees": gagnees, "perdues": perdues,
                           "garde_collaterale": {"gagnees": gi, "perdues": pi,
                                                 "net": len(gi) - len(pi)}}
        out["bras"][nom] = bloc

    c = out["bras"]["candidat"]["primaire"]
    p = out["bras"]["placebo"]["primaire"]
    conditions = {
        "gain": c["net"] > 0,
        "garde_collaterale": c["garde_collaterale"]["net"] >= 0,
        "placebo": p["net"] < max(c["net"], 0) / 2,
    }
    if not conditions["gain"]:
        issue = "NO-GO — le balisage ne nuit pas au rappel"
    elif not conditions["placebo"]:
        issue = "NO-GO — le gain est le déplacement des vecteurs, pas la réparation"
    elif not conditions["garde_collaterale"]:
        issue = "HOLD — le gain se paie ailleurs"
    else:
        issue = "GO — le débalisage est proposé en production"
    out["conditions"], out["issue"] = conditions, issue
    (HERE / f"results-debalisage-{SIGNATURE}.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    return out


def imprimer(r: dict) -> None:
    print(f"\nsignature {r['signature']} · {r['chunks_balises']} chunks balisés · "
          f"population {r['population']} · appels LLM {r['appels_llm']}")
    print(f"questions dont aucun or ne porte de balise : {r['questions_a_or_intact']}")
    for champ, titre in (("primaire", "PRIMAIRE — build_context"), ("production", "DIAGNOSTIC — _select complet")):
        print(f"\n{titre}")
        for nom in ("reference", "candidat", "placebo"):
            b = r["bras"][nom][champ]
            g = b["garde_collaterale"]
            print(f"   {nom:10s} or servi {b['or_servi']:3d}  net {b['net']:+3d}  "
                  f"(+{len(b['gagnees'])} / −{len(b['perdues'])})   "
                  f"garde collatérale net {g['net']:+3d} (+{len(g['gagnees'])} / −{len(g['perdues'])})")
    c = r["bras"]["candidat"]["primaire"]
    print(f"\nrecensement nominal du candidat :")
    print(f"   gagnées : {c['gagnees']}")
    print(f"   perdues : {c['perdues']}")
    print(f"\nconditions : {r['conditions']}")
    print(f"ISSUE : {r['issue']}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--etape", choices=("vecteurs", "bras", "verdict"), required=True)
    p.add_argument("--bras", choices=("reference",) + BRAS)
    a = p.parse_args()
    if a.etape == "vecteurs":
        for bras in ([a.bras] if a.bras else list(BRAS)):
            vecteurs(bras)
    elif a.etape == "bras":
        bras_seul(a.bras or "reference")
    else:
        imprimer(verdict())


if __name__ == "__main__":
    main()
