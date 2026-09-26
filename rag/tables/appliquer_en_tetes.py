"""Rendre leur en-tête aux fragments de tableaux des documents importés — texte ET vecteur.

Ce que cet instrument répare
-----------------------------
Un tableau trop grand pour un chunk est découpé ; seul le premier fragment porte la rangée
d'en-tête, et la passe de conversion du corpus la recopie sur les suivants. L'**import** ne
le faisait pas : l'étape 4 d'``apply_delivery`` s'exécute avant la promotion de l'étape 8 et
ne voyait donc pas la livraison en vol. Corrigé pour les imports futurs le 9 septembre 2026.
Le corpus servi, lui, portait encore le défaut sur **1 066 chunks d'overlay** de 100
documents, dont **989 sont des points réellement servis** — **3,786 % des 26 120**. Les 77
autres sont des chunks-tableaux ``rag_eligible: false`` : présents dans l'overlay, jamais
indexés. La distinction a été trouvée en appliquant la correction, et le chiffre de 4,081 %
qu'annonçait le recensement est corrigé ici à côté de lui.

Pourquoi ce n'est pas un simple remplacement de texte
------------------------------------------------------
Vérifié dans ``build_index.py`` avant d'écrire une ligne : pour un point **importé**, le
texte vient de l'overlay des tableaux (``build_index.imported_points``) mais le **vecteur** vient
du ``.npz`` de sa livraison (``:113``). L'overlay de vecteurs ``vectors-tables-markdown-v1``
n'est appliqué qu'au **bloc de l'export** (``:139-148``). Changer le texte sans le vecteur
les rendrait donc incohérents à la première reconstruction — le corpus servirait un tableau
dont le vecteur décrit un autre texte.

Il faut donc, pour chaque chunk corrigé : ré-embarquer, écrire vecteur **et** texte dans la
collection, **et** réécrire le vecteur dans le ``.npz`` de sa livraison, qui en est la seule
copie hors collection.

Ce que l'instrument refuse de faire
------------------------------------
- il **refuse d'écrire** si l'embedding local ne reproduit pas les vecteurs stockés à
  cosinus ≥ 0,99, mesuré sur des chunks-tableaux importés **dont le texte ne change pas** —
  même garde que ``convert_tables.main()``, et pour la même raison ;
- il **refuse d'écrire** si un chunk d'un document de l'**export** figure parmi les
  changements : la passe du corpus les a déjà traités, un écart là signalerait que la règle
  a bougé et non que l'import était en retard ;
- sans ``--appliquer``, il ne fait que **dire** ce qu'il changerait.

Chaque ``.npz`` de livraison touché est copié en
``avant-en-tetes-2026-09-09--<livraison>.npz`` **avant** toute écriture : ces fichiers sont
hors git et sont la seule copie des vecteurs des documents importés. Le préfixe les met hors
du motif ``vectors-*.npz``, qui désigne les vecteurs d'une livraison.

    .venv/bin/python rag/tables/appliquer_en_tetes.py                 # ce qui changerait
    .venv/bin/python rag/tables/appliquer_en_tetes.py --appliquer     # l'écriture
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for extra in ("rag", "rag/tables", "rag/titles", "rag/ingestion", "src"):
    chemin = str(ROOT / extra)
    if chemin not in sys.path:
        sys.path.insert(0, chemin)

import convert_tables as ct        # noqa: E402
import corpus_overlay             # noqa: E402
import quant_rag                  # noqa: E402
import reembed_titles as rt       # noqa: E402
import registry as reg            # noqa: E402
import recenser_en_tetes as rec    # noqa: E402

INGESTED = ROOT / "data" / "processed" / "ingested"
VECTEURS_LIVRAISONS = ROOT / "rag" / "ingestion" / ".cache"
#: Les sauvegardes ne doivent PAS répondre au motif ``vectors-*.npz`` : ce motif désigne les
#: vecteurs d'une livraison, et ``verifier_installation`` comme un futur lecteur les
#: compteraient comme telles. Le préfixe les met hors du motif tout en les gardant à côté
#: des fichiers qu'elles sauvegardent.
PREFIXE_SAUVEGARDE = "avant-en-tetes-2026-09-09--"
RAPPORT = HERE / "resultats-en-tetes-2026-09-09.json"
COSINUS_MINIMUM = 0.99
TEMOINS = 40


def titre_propre(entree: dict, document: dict, metadonnees: dict) -> str:
    """Le titre vu par l'embedding des vecteurs stockés — la recette des points importés.

    ``apply_delivery.embed_delivery`` embarque avec ``clean_title(titre d'export, métadonnées)``.
    Reproduire avec un autre titre rendrait un cosinus de 0,466, et la garde arrêterait tout —
    c'est exactement ce qui est arrivé à ``convert_tables`` le 3 septembre 2026.
    """
    return rt.clean_title(document.get("title") or "", metadonnees.get(entree["document_id"]))


def etat_du_corpus() -> tuple[dict, dict, dict]:
    """Les documents importés, ceux de l'export, et les métadonnées consolidées."""
    donnees = json.loads(corpus_overlay.REGISTRY.read_text(encoding="utf-8"))
    base = reg.BASELINE["id"]
    importes, origine = {}, {}
    for entree in donnees["documents"]:
        if entree.get("status") != "active":
            continue
        cible = origine if (entree.get("delivery") or {}).get("id") in (None, base) else importes
        cible[entree["document_id"]] = entree
    metadonnees = {r["document_id"]: r for r in
                   json.loads(corpus_overlay.METADATA.read_text(encoding="utf-8"))["documents"]}
    return importes, origine, metadonnees


def candidats(entrees: dict) -> dict[str, str]:
    """Ce que l'étape 4 aurait rendu — par le convertisseur de production, chaîne fournie."""
    sortie: dict[str, str] = {}
    for entree in entrees.values():
        chunks = rec.chunks_du_document(entree)
        tableaux = [c for c in chunks if c.get("content_type") == "table"]
        if not tableaux:
            continue
        points = [rec.Point(rec.payload_de(c)) for c in tableaux]
        converti, _ = ct.convert_all(points, previous=ct.part_chains(chunks))
        sortie.update(converti)
    return sortie


def point_de(entree: dict, chunk_id: str) -> int | None:
    """L'identifiant du point : première place de la livraison + rang du chunk **éligible**.

    C'est la règle d'``apply_delivery`` (``first_id + offset`` sur les chunks éligibles dans
    l'ordre du fichier) et de ``build_index.imported_points``. On la relit ici plutôt que de
    scroller la collection : un appariement par position se vérifie ensuite sur le payload,
    et c'est ce que fait l'appelant.
    """
    eligibles = [c["chunk_id"] for c in rec.chunks_du_document(entree)
                 if c.get("rag_eligible") is True]
    if chunk_id not in eligibles:
        return None
    premier = (entree.get("point_ids") or {}).get("first")
    return None if premier is None else premier + eligibles.index(chunk_id)


def controle_de_reproduction(client, modele, temoins: list[tuple]) -> dict:
    """Ré-embarquer des chunks **inchangés** et comparer aux vecteurs stockés.

    Si l'embedding local ne reproduit pas ce que la collection porte, tout ce qui suit
    écrirait des vecteurs qui ne sont pas comparables aux autres. On ne mesure donc pas la
    correction, on mesure d'abord l'instrument.
    """
    if not temoins:
        return {"n": 0, "cosinus_min": None, "note": "aucun témoin disponible"}
    textes = [rt.embedding_text(titre, chunk.get("title_path"), texte)
              for _pid, chunk, titre, texte in temoins]
    vecteurs = np.asarray(modele.encode(textes, normalize_embeddings=True,
                                        show_progress_bar=False, batch_size=8), dtype=np.float32)
    cosinus = []
    for (pid, chunk, _t, _x), vecteur in zip(temoins, vecteurs):
        points = client.retrieve(quant_rag.COLLECTION, ids=[pid], with_vectors=True,
                                 with_payload=["chunk_id"])
        if not points or points[0].payload.get("chunk_id") != chunk["chunk_id"]:
            continue
        cosinus.append(float(np.dot(vecteur, np.asarray(points[0].vector, dtype=np.float32))))
    tableau = np.asarray(cosinus)
    return {"n": int(tableau.size),
            "cosinus_min": round(float(tableau.min()), 6) if tableau.size else None,
            "cosinus_median": round(float(np.median(tableau)), 6) if tableau.size else None}


def main() -> None:
    parseur = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parseur.add_argument("--appliquer", action="store_true",
                         help="écrire réellement — sans ce drapeau, rien n'est modifié")
    parseur.add_argument("--rapport", type=Path, default=RAPPORT)
    arguments = parseur.parse_args()
    depart = time.perf_counter()
    print(f"arbre : {ROOT}")
    print(f"signature avant : {corpus_overlay.signature()}")

    importes, origine, metadonnees = etat_du_corpus()
    servi = json.loads(corpus_overlay.TABLES.read_text(encoding="utf-8"))
    overlay = servi["chunks"]

    # 1. ce qui changerait, chez les importés ET chez ceux de l'export
    neufs_importes = candidats(importes)
    changes = {cid: texte for cid, texte in neufs_importes.items()
               if cid in overlay and overlay[cid] != texte}
    neufs_origine = candidats(origine)
    changes_origine = [cid for cid, texte in neufs_origine.items()
                       if cid in overlay and overlay[cid] != texte]
    print(f"documents importés {len(importes)} · export {len(origine)}")
    print(f"chunks qui changeraient : {len(changes)} (importés) · "
          f"{len(changes_origine)} (export)")
    if changes_origine:
        sys.exit(f"REFUS : {len(changes_origine)} chunk(s) de documents de l'EXPORT "
                 f"changeraient — la passe du corpus les a déjà traités. Un écart ici "
                 f"signifie que la règle a bougé, pas que l'import était en retard. "
                 f"Exemples : {changes_origine[:5]}")
    if not changes:
        print("rien à corriger."); return

    # 2. apparier chaque chunk à son point, et préparer les témoins
    client = quant_rag.client()
    par_livraison: dict[str, list] = {}
    a_ecrire, temoins = [], []
    for document_id, entree in sorted(importes.items()):
        chunks = {c["chunk_id"]: c for c in rec.chunks_du_document(entree)}
        document = json.loads((INGESTED / entree["folder"] / "document.json").read_text(encoding="utf-8"))
        titre = titre_propre(entree, document, metadonnees)
        livraison = (entree.get("delivery") or {}).get("id")
        for chunk_id, chunk in chunks.items():
            if chunk.get("content_type") != "table" or chunk_id not in overlay:
                continue
            pid = point_de(entree, chunk_id)
            if pid is None:
                continue
            if chunk_id in changes:
                a_ecrire.append((pid, chunk, titre, changes[chunk_id], livraison))
                par_livraison.setdefault(livraison, []).append(chunk_id)
            elif len(temoins) < TEMOINS:
                temoins.append((pid, chunk, titre, overlay[chunk_id]))
    print(f"{len(a_ecrire)} points à réécrire, dans {len(par_livraison)} livraison(s)")

    # 3. LA GARDE : l'instrument reproduit-il ce que la collection porte ?
    modele = quant_rag.embedder()
    reproduction = controle_de_reproduction(client, modele, temoins)
    print(f"reproduction de la recette : {json.dumps(reproduction, ensure_ascii=False)}")
    if reproduction["cosinus_min"] is None or reproduction["cosinus_min"] < COSINUS_MINIMUM:
        sys.exit(f"ÉCHEC : l'embedding local ne reproduit pas les vecteurs stockés "
                 f"(cosinus min {reproduction['cosinus_min']}) — ne rien écrire.")

    if not arguments.appliquer:
        print("\n--appliquer absent : rien n'a été écrit.")
        print(json.dumps({"changes": len(changes), "livraisons": sorted(par_livraison),
                          "reproduction": reproduction}, ensure_ascii=False, indent=1))
        return

    # 4. ré-embarquer les seuls chunks corrigés
    debut = time.perf_counter()
    textes = [rt.embedding_text(titre, chunk.get("title_path"), texte)
              for _pid, chunk, titre, texte, _liv in a_ecrire]
    vecteurs = np.asarray(modele.encode(textes, normalize_embeddings=True,
                                        show_progress_bar=False, batch_size=8), dtype=np.float32)
    secondes_embedding = time.perf_counter() - debut
    print(f"embedding : {len(vecteurs)} vecteurs en {secondes_embedding:.0f} s "
          f"({len(vecteurs) / max(secondes_embedding, 1e-6):.2f} chunks/s)")

    # 5. sauvegarder les .npz AVANT toute écriture — ils sont hors git
    sauvegardes = []
    for livraison in sorted(par_livraison):
        source = VECTEURS_LIVRAISONS / f"vectors-{livraison}.npz"
        copie = VECTEURS_LIVRAISONS / f"{PREFIXE_SAUVEGARDE}{livraison}.npz"
        if source.exists() and not copie.exists():
            shutil.copy2(source, copie)
            sauvegardes.append(copie.name)
    print(f"{len(sauvegardes)} sauvegarde(s) de .npz déposée(s)")

    # 6. la collection : vecteur ET texte, avec l'original conservé en text_html
    from qdrant_client import models

    ecrits = 0
    for depart_bloc in range(0, len(a_ecrire), 256):
        bloc = a_ecrire[depart_bloc:depart_bloc + 256]
        indices = range(depart_bloc, depart_bloc + len(bloc))
        client.update_vectors(quant_rag.COLLECTION, wait=True, points=[
            models.PointVectors(id=pid, vector=vecteurs[i].tolist())
            for i, (pid, *_r) in zip(indices, bloc)])
        for pid, chunk, _titre, texte, _liv in bloc:
            client.set_payload(quant_rag.COLLECTION, wait=True, points=[pid],
                               payload={"text": texte, "text_html": chunk.get("text")})
        ecrits += len(bloc)
        print(f"  collection {ecrits}/{len(a_ecrire)}", end="\r", flush=True)
    print()

    # 7. les .npz des livraisons — la seule copie hors collection.
    #
    # On y écrit ce que la collection SERT, relu point par point, et non ce qu'on vient de
    # lui donner. Qdrant renormalise le vecteur à l'insertion sous la distance Cosine :
    # l'écart est de l'ordre de 1e-5 par composante — invisible au classement, mais
    # ``recover_vectors.divergents()`` compare par ÉGALITÉ STRICTE, sans tolérance, et à
    # raison : ``build_index`` téléverse le vecteur du .npz tel quel, donc un cache qui
    # diffère signifie qu'une reconstruction ne reproduirait pas ce qui est servi. Mesuré le
    # 9 septembre 2026 : 989 passages divergents sur les 989 écrits, écart max 1,05e-5,
    # cosinus 1,0001. Relire coûte un aller-retour et ferme la question.
    par_chunk = {}
    for depart_bloc in range(0, len(a_ecrire), 256):
        bloc = a_ecrire[depart_bloc:depart_bloc + 256]
        rendus = client.retrieve(quant_rag.COLLECTION, ids=[pid for pid, *_r in bloc],
                                 with_vectors=True, with_payload=["chunk_id"])
        for point in rendus:
            par_chunk[point.payload["chunk_id"]] = np.asarray(point.vector, dtype=np.float32)
    absents = [chunk["chunk_id"] for _pid, chunk, *_r in a_ecrire
               if chunk["chunk_id"] not in par_chunk]
    if absents:
        sys.exit(f"{len(absents)} vecteur(s) écrits mais non relus : {absents[:5]}")
    npz_touches = []
    for livraison, chunk_ids in sorted(par_livraison.items()):
        chemin = VECTEURS_LIVRAISONS / f"vectors-{livraison}.npz"
        if not chemin.exists():
            print(f"  ATTENTION {chemin.name} absent — vecteurs non mis à jour pour {livraison}")
            continue
        blob = np.load(chemin)
        identifiants = blob["chunk_ids"].tolist()
        table = np.asarray(blob["vectors"], dtype=np.float32).copy()
        remplaces = 0
        for chunk_id in chunk_ids:
            if chunk_id in identifiants:
                table[identifiants.index(chunk_id)] = par_chunk[chunk_id]
                remplaces += 1
        np.savez_compressed(chemin, chunk_ids=np.asarray(identifiants), vectors=table)
        npz_touches.append({"livraison": livraison, "remplaces": remplaces,
                            "attendus": len(chunk_ids)})
    print(f"{len(npz_touches)} .npz de livraison mis à jour")

    # 8. l'overlay de texte — seules les clés changées bougent
    overlay.update(changes)
    servi["chunks"] = overlay
    servi["corrections"] = {
        **(servi.get("corrections") or {}),
        "en-tetes-fragments-importes-2026-09-09": {
            "chunks": len(changes),
            "note": "en-tête de la première partie recopié sur les fragments des documents "
                    "importés — la même règle que celle appliquée aux documents de l'export "
                    "et que celle qu'appliquent désormais les imports (§11.1)",
        },
    }
    corpus_overlay.TABLES.write_text(json.dumps(servi, ensure_ascii=False), encoding="utf-8")
    corpus_overlay.invalidate()
    quant_rag.document_metadata.cache_clear()
    quant_rag.bm25.cache_clear()
    signature = corpus_overlay.signature()
    print(f"overlay écrit — signature {signature}")

    # 9. l'index lexical
    index = quant_rag.rebuild_bm25()
    print(f"BM25 : {quant_rag.bm25_path().name} — {index.doc_count} records")

    rapport = {
        "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "signature_apres": signature,
        "chunks_corriges": len(changes),
        "documents_concernes": len(par_livraison),
        "chunks_export_changes": len(changes_origine),
        "reproduction": reproduction,
        "embedding": {"secondes": round(secondes_embedding, 1),
                      "chunks_par_seconde": round(len(vecteurs) / max(secondes_embedding, 1e-6), 2)},
        "npz_livraisons": npz_touches,
        "sauvegardes": sauvegardes,
        "bm25_records": index.doc_count,
        "collection": client.count(quant_rag.COLLECTION, exact=True).count,
        "secondes_total": round(time.perf_counter() - depart, 1),
    }
    arguments.rapport.write_text(json.dumps(rapport, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(rapport, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
