"""Réemployer les vecteurs d'un candidat précédent — mais seulement ce qui est prouvé.

Le problème que ce module résout
---------------------------------
Le chantier de représentation a construit le candidat ``8d4ee77f1f`` le 6 septembre 2026 et
gardé ses trois jeux de vecteurs (183 Mo, **8 h 31 de calcul**). Le corpus servi est passé
depuis à ``e1bdf36e2e`` : ``c41b49c`` a rendu leur en-tête à 989 passages de tableaux. Le
candidat reconstruit aujourd'hui porte donc une autre signature — mais **les mêmes 26 529
chunk_id, dans le même ordre**, produits par le même chunker sur les mêmes blocs.

Ré-embarquer les 26 529 coûterait 2 h 47 de MPS pour recalculer 25 540 vecteurs identiques.
Les réemployer sans preuve serait exactement le défaut que ``F7`` nomme : **asseoir un
déclencheur de ré-embarquement sur autre chose que la chaîne réellement plongée**.

Ce que le module exige avant de réemployer un seul vecteur
-----------------------------------------------------------
1. **Le diff est exhaustif, pas échantillonné.** Les deux overlays de tableaux — celui
   d'avant la réparation, passé en argument, et celui d'aujourd'hui — sont transportés sur
   les identifiants du candidat et comparés **entrée par entrée**. Ce qui diffère est
   ré-embarqué ; le reste est réemployé. Rien n'est deviné.
2. **La recette est prouvée sur l'ancien jeu.** Un échantillon de chunks **inchangés** est
   ré-embarqué aujourd'hui et comparé aux vecteurs de l'ancien ``.npz``. Sous le cosinus de
   refus, **rien n'est écrit** — c'est la garde de ``convert_tables`` et de
   ``embed_candidat.check``, appliquée ici à la réutilisation elle-même.
3. **Le ``sha256`` de la chaîne plongée est écrit à côté de chaque vecteur** (invariant F7),
   pour que le prochain chantier n'ait pas à reconstruire ce diff depuis git.

Ce que le réemploi ne prétend pas
----------------------------------
Les 989 vecteurs recalculés le sont dans une composition de lots différente de celle du
6 septembre. En fp16 cela déplace les derniers bits — le corpus servi est déjà dans cet état
depuis ``appliquer_en_tetes.py``, qui a ré-embarqué 1 066 chunks de la même façon. Le cosinus
mesuré de l'échantillon dit ce que cet écart vaut ; il est publié, jamais supposé.

    .venv/bin/python rag/ingestion/reemploi_candidat.py \\
        --candidat 5bc1fb3f53 --ancien 8d4ee77f1f --bras c2 \\
        --overlay-avant /chemin/tables-markdown-v1.json --echantillon 200
    # ajouter --ecrire pour produire le .npz ; sans lui, il ne fait que dire et prouver
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "rag"))
sys.path.insert(0, str(ROOT / "rag" / "titles"))

import corpus_overlay  # noqa: E402
import quant_rag  # noqa: E402

CACHE = HERE / ".cache"
PROCESSED = ROOT / "data" / "processed"
#: Sous ce cosinus entre un vecteur réemployé et le même texte ré-embarqué aujourd'hui, la
#: recette n'est plus celle qui a produit l'ancien jeu : le réemploi est refusé. Même seuil
#: que ``convert_tables.main`` et ``embed_candidat.check``.
COSINUS_REFUS = 0.99
BATCH = 8


def cosinus(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Le cosinus ligne à ligne — renormalisé, accumulé en float64.

    Pourquoi ce n'est pas un simple produit scalaire (mesuré le 10 septembre 2026, lot B,
    §14.3 de ``docs/STRATEGIE.md``)
    -----------------------------------------------------------------------------------
    ``normalize_embeddings=True`` **ne rend pas des vecteurs unitaires**. Sur les vecteurs
    du corpus servi, les normes vont de ``0,99975586`` à ``1,00047958`` : le modèle
    normalise en float32, et l'arrondi laisse quelques 1e-4 de jeu. Un produit scalaire
    brut entre deux vecteurs *identiques* rend donc ``‖v‖²``, c'est-à-dire une valeur qui
    s'écarte de 1 de deux fois l'erreur de norme.

    Le chiffre : sur 14 597 vecteurs **identiques au bit près**, le produit scalaire brut
    en met **3 625 sous 0,9999**. Il ne mesurait pas un écart angulaire, il mesurait
    l'erreur de norme du modèle.

    La garde tenait quand même, parce que ``COSINUS_REFUS`` vaut 0,99 — vingt fois plus
    large que l'artefact. Mais une garde qui ne sait pas ce qu'elle mesure est une garde
    qu'on ne peut pas resserrer : au premier seuil un peu serré, elle refuserait des
    vecteurs identiques. D'où cette fonction, et le test qui la tient.

    **Le seuil de 0,99 n'a pas bougé** : le resserrer est une décision de mesure, pas une
    décision de clôture.
    """
    a64 = np.asarray(a, dtype=np.float64)
    b64 = np.asarray(b, dtype=np.float64)
    na = np.linalg.norm(a64, axis=1, keepdims=True)
    nb = np.linalg.norm(b64, axis=1, keepdims=True)
    # Un vecteur nul n'a pas de direction : son cosinus n'existe pas. On le laisse à 0
    # plutôt que de rendre un NaN qui traverserait silencieusement `min()`.
    na[na == 0.0] = 1.0
    nb[nb == 0.0] = 1.0
    return ((a64 / na) * (b64 / nb)).sum(axis=1)


def _transport_avant(chemin_overlay: Path) -> dict[str, str]:
    """L'overlay des tableaux d'AVANT, transporté sur les identifiants du candidat.

    ``corpus_overlay.TABLES`` est substitué le temps du transport : ``text_overrides()`` est
    mémorisé, d'où l'``invalidate()`` des deux côtés — un oubli ferait lire deux fois le même
    overlay et rendrait un diff vide, c'est-à-dire un réemploi total non fondé.
    """
    import rechunk_corpus

    vivant = corpus_overlay.TABLES
    try:
        corpus_overlay.TABLES = chemin_overlay
        corpus_overlay.invalidate()
        actifs = [d for d in rechunk_corpus._registre() if d.get("status") == "active"]
        transporte, _ = rechunk_corpus.transporter_overlay(actifs)
    finally:
        corpus_overlay.TABLES = vivant
        corpus_overlay.invalidate()
    return transporte


def diff_exhaustif(racine: Path, chemin_overlay_avant: Path) -> tuple[set[str], dict]:
    """Les chunk_id du candidat dont le **texte servi** diffère de celui de l'ancien jeu."""
    avant = _transport_avant(chemin_overlay_avant)
    apres = json.loads((racine / "tables-markdown-v1.json").read_text(encoding="utf-8"))["chunks"]
    servis = {r["chunk_id"] for r in
              (json.loads(l) for l in (racine / "chunks-servis.jsonl").open(encoding="utf-8") if l.strip())}
    changes = {cid for cid in set(avant) | set(apres) if avant.get(cid) != apres.get(cid)}
    return changes & servis, {"entrees_avant": len(avant), "entrees_apres": len(apres),
                              "changes_total": len(changes),
                              "changes_servis": len(changes & servis)}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--candidat", required=True, help="signature du candidat d'aujourd'hui")
    p.add_argument("--ancien", required=True, help="signature du candidat dont on réemploie les vecteurs")
    p.add_argument("--bras", required=True, choices=("c1", "c2", "sabote"))
    p.add_argument("--overlay-avant", required=True, type=Path,
                   help="tables-markdown-v1.json tel qu'il était quand l'ancien jeu a été calculé")
    p.add_argument("--echantillon", type=int, default=200, help="chunks inchangés ré-embarqués pour la preuve")
    p.add_argument("--ecrire", action="store_true", help="écrit le .npz ; sans lui, rien n'est écrit")
    a = p.parse_args()

    import embed_candidat
    from rechunk_corpus import titres_plonges

    racine = PROCESSED / f"candidat-{a.candidat}"
    if not racine.is_dir():
        sys.exit(f"corpus candidat introuvable : {racine}")
    ancien_npz = CACHE / f"vectors-candidat-{a.ancien}-{a.bras}.npz"
    if not ancien_npz.exists():
        sys.exit(f"ancien jeu introuvable : {ancien_npz}")
    final = CACHE / f"vectors-candidat-{a.candidat}-{a.bras}.npz"
    if final.exists():
        sys.exit(f"{final.name} existe déjà — le supprimer d'abord si l'on veut le refaire")

    blob = np.load(ancien_npz, allow_pickle=True)
    anciens_ids = [str(x) for x in blob["chunk_ids"]]
    anciens_vecteurs = blob["vectors"]

    chunks = embed_candidat._chunks(racine)
    titres, _ = titres_plonges()
    textes, _ = embed_candidat.textes_du_bras(chunks, a.bras, titres)
    ids = [c["chunk_id"] for c in chunks]
    if ids != anciens_ids:
        sys.exit(f"les identifiants diffèrent — réemploi impossible "
                 f"({len(set(ids) ^ set(anciens_ids))} en écart de symétrie, ordre "
                 f"{'identique' if set(ids) == set(anciens_ids) else 'et ensemble différents'})")

    changes, stats = diff_exhaustif(racine, a.overlay_avant)
    print(f"diff exhaustif des textes servis : {stats}")
    print(f"  à ré-embarquer : {len(changes)}    réemployés : {len(ids) - len(changes)}")

    par_id = dict(zip(ids, textes))
    index = {cid: i for i, cid in enumerate(ids)}
    inchanges = [cid for cid in ids if cid not in changes]

    pas = max(len(inchanges) // a.echantillon, 1)
    echantillon = inchanges[::pas][:a.echantillon]
    modele = quant_rag.embedder()
    print(f"\npreuve du réemploi : {len(echantillon)} chunks INCHANGÉS ré-embarqués…")
    calcules = np.asarray(modele.encode([par_id[c] for c in echantillon], normalize_embeddings=True,
                                        show_progress_bar=False, batch_size=BATCH), dtype=np.float32)
    stockes = np.asarray([anciens_vecteurs[index[c]] for c in echantillon], dtype=np.float32)
    cos = cosinus(calcules, stockes)
    preuve = {"chunks": len(echantillon), "cosinus_min": float(cos.min()),
              "cosinus_median": float(np.median(cos)), "cosinus_moyen": float(cos.mean())}
    print(f"  cosinus min {preuve['cosinus_min']:.6f} · médian {preuve['cosinus_median']:.6f}")
    if preuve["cosinus_min"] < COSINUS_REFUS:
        sys.exit(f"REFUS : cosinus min {preuve['cosinus_min']:.6f} < {COSINUS_REFUS}. "
                 f"L'ancien jeu n'a pas été produit par la recette d'aujourd'hui — "
                 f"ré-embarquer le candidat entier.")

    vecteurs = np.array(anciens_vecteurs, dtype=np.float32, copy=True)
    if changes:
        cibles = [cid for cid in ids if cid in changes]
        print(f"\nré-embarquement des {len(cibles)} chunks dont le texte a changé…")
        t0 = time.time()
        neufs = np.asarray(modele.encode([par_id[c] for c in cibles], normalize_embeddings=True,
                                         show_progress_bar=False, batch_size=BATCH), dtype=np.float32)
        secondes = time.time() - t0
        for cid, v in zip(cibles, neufs):
            vecteurs[index[cid]] = v
        print(f"  {secondes:.1f} s ({len(cibles) / max(secondes, 1e-9):.2f} chunks/s)")
    else:
        secondes = 0.0

    empreintes = [hashlib.sha256(par_id[c].encode("utf-8")).hexdigest() for c in ids]
    rapport = {
        "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "candidat": a.candidat, "ancien": a.ancien, "bras": a.bras,
        "overlay_avant": str(a.overlay_avant),
        "diff": stats, "reembarques": len(changes), "reemployes": len(ids) - len(changes),
        "preuve_reemploi": preuve, "secondes_embedding": round(secondes, 1),
        "cosinus_refus": COSINUS_REFUS,
    }
    if not a.ecrire:
        print("\n--ecrire absent : rien n'est écrit.")
        print(json.dumps(rapport, ensure_ascii=False, indent=1))
        return

    # ``np.asarray`` et non ``dtype=object`` : ``embed_candidat`` écrit des chaînes (`<U22`), et
    # un tableau d'objets ne se relit qu'avec ``allow_pickle=True``, que les lecteurs du dépôt
    # ne passent pas. Un fichier illisible par `build_collection_candidat` n'est pas un fichier.
    np.savez_compressed(final, chunk_ids=np.asarray(ids), vectors=vecteurs,
                        embedded_sha256=np.asarray(empreintes))
    rapport["fichier"] = final.name
    (HERE / f"resultats-reemploi-{a.candidat}-{a.bras}.json").write_text(
        json.dumps(rapport, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"\nécrit : {final.name} ({len(ids)} vecteurs, dont {len(changes)} recalculés)")


if __name__ == "__main__":
    main()
