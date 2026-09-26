"""Vecteurs des trois bras du chantier de représentation — phase 4, avec reprise.

Trois bras, et l'ordre compte : **C1**, puis **C2**, puis le **saboté**. Les deux premiers
portent les deux règles de décision du pré-enregistrement ; le troisième n'est qu'explicatif
et peut attendre si quelque chose casse.

| bras     | ``retrieval_text``                                        |
|----------|-----------------------------------------------------------|
| ``c1``   | ``Document:`` + ``Path:`` (réparé)                        |
| ``c2``   | c1 + ``Page: n``                                          |
| ``sabote`` | c1, mais ``Path:`` emprunté à une **autre section du même document** |

La recette est celle de la production, à l'identique
----------------------------------------------------
``rag/titles/reembed_titles.embedding_text`` et ``apply_delivery.embed_delivery`` :
Qwen3-Embedding-0.6B, ``max_seq_length`` 1 024, fp16 sur MPS, ``normalize_embeddings=True``,
``batch_size`` 8, et le **tri par longueur de texte** avant le découpage en blocs — il réduit
le remplissage, et le retirer changerait la composition des lots donc, en fp16, les vecteurs
au dernier bit. ``--check`` le vérifie avant d'écrire quoi que ce soit : 40 chunks de la
collection **servie** sont ré-embarqués et comparés à leurs vecteurs stockés ; sous un cosinus
de 0,99, rien n'est écrit.

La reprise
----------
Un bloc de 64 chunks est sauvegardé dans ``…partial.npz`` après chaque calcul. Une coupure
coûte au plus un bloc, jamais les 2 h 47 du bras. Relancer reprend là où l'on s'était arrêté ;
si le ``.npz`` final existe, le bras est déjà fait et le module le dit sans rien recalculer.

    .venv/bin/python rag/ingestion/embed_candidat.py --check
    .venv/bin/python rag/ingestion/embed_candidat.py --candidat 8d4ee77f1f --bras c1
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "rag"))
sys.path.insert(0, str(ROOT / "rag" / "titles"))

import quant_rag  # noqa: E402
from rechunk_corpus import retrieval_text, titres_plonges  # noqa: E402  — un seul domicile

CACHE = HERE / ".cache"
PROCESSED = ROOT / "data" / "processed"
#: Chunks par lot d'encodage. 64, comme ``apply_delivery.EMBED_BLOCK``.
BLOC = 64
#: Lots entre deux sauvegardes partielles. Chaque sauvegarde empile les vecteurs déjà
#: calculés — 108 Mo à la fin d'un bras — et la machine a été vue à 75 Mo de pages libres le
#: 6 septembre 2026, le harnais tuant des tâches pour cette raison. Sauvegarder tous les huit
#: lots divise cette allocation par huit ; une coupure coûte alors au plus **512 chunks**,
#: soit ~2 min au débit observé, contre les 2 h 47 d'un bras entier. Les vecteurs sont
#: inchangés : seule la cadence du point de reprise bouge.
SAUVEGARDES_TOUS_LES = 8
BATCH = 8
BRAS = ("c1", "c2", "sabote")
GRAINE = 20260906


def _candidat(signature: str) -> Path:
    chemin = PROCESSED / f"candidat-{signature}"
    if not chemin.is_dir():
        sys.exit(f"corpus candidat introuvable : {chemin}")
    return chemin


def _chunks(racine: Path) -> list[dict]:
    servis = {r["chunk_id"]: r["text"] for r in
              (json.loads(l) for l in (racine / "chunks-servis.jsonl").open(encoding="utf-8") if l.strip())}
    out = []
    for row in (json.loads(l) for l in (racine / "rows.jsonl").open(encoding="utf-8") if l.strip()):
        c = row["chunk"]
        out.append({"chunk_id": c["chunk_id"], "document_id": c["document_id"],
                    "title_path": c.get("title_path"), "page_start": c.get("page_start"),
                    "text": servis[c["chunk_id"]]})
    return out


def chemins_sabotes(chunks: list[dict]) -> tuple[dict[str, str], dict]:
    """Le ``Path:`` du bras saboté : celui d'une **autre section du même document**.

    Permutation des étiquettes entre sections **voisines** — les chemins distincts d'un
    document sont pris dans l'ordre de première apparition et échangés deux à deux
    (1↔2, 3↔4, …). Déterministe, graine ``20260906`` inutile ici puisque l'ordre du document
    la remplace, et **intra-document** : un sabotage inter-documents serait détecté par le
    seul ``Document:``, et ne dirait rien sur l'étiquette de section.

    Un document à moins de deux chemins distincts ne peut pas être saboté ; ses chunks gardent
    le leur, et le rapport les compte — un sabotage partiel qu'on tairait affaiblirait la
    lecture du contrôle négatif.
    """
    par_document: dict[str, list[str]] = collections.defaultdict(list)
    for c in chunks:
        chemin = c.get("title_path") or ""
        if chemin and chemin not in par_document[c["document_id"]]:
            par_document[c["document_id"]].append(chemin)
    echange: dict[tuple[str, str], str] = {}
    stats = collections.Counter()
    for document_id, chemins in par_document.items():
        if len(chemins) < 2:
            stats["documents_non_sabotables"] += 1
            continue
        stats["documents_sabotes"] += 1
        for i in range(0, len(chemins) - 1, 2):
            echange[(document_id, chemins[i])] = chemins[i + 1]
            echange[(document_id, chemins[i + 1])] = chemins[i]
        if len(chemins) % 2:                       # le dernier, impair, reste seul
            stats["chemins_non_permutes"] += 1
    sabote: dict[str, str] = {}
    for c in chunks:
        chemin = c.get("title_path") or ""
        nouveau = echange.get((c["document_id"], chemin))
        if nouveau is None:
            stats["chunks_inchanges"] += 1
            sabote[c["chunk_id"]] = chemin
        else:
            stats["chunks_sabotes"] += 1
            sabote[c["chunk_id"]] = nouveau
    return sabote, dict(stats)


def textes_du_bras(chunks: list[dict], bras: str, titres: dict[str, str]) -> tuple[list[str], dict]:
    if bras == "sabote":
        chemins, stats = chemins_sabotes(chunks)
    else:
        chemins, stats = {c["chunk_id"]: c.get("title_path") for c in chunks}, {}
    avec_page = bras == "c2"
    textes = [retrieval_text(titres[c["document_id"]], chemins[c["chunk_id"]],
                             c["page_start"], c["text"], avec_page=avec_page) for c in chunks]
    return textes, stats


# ------------------------------------------------------------------ la garde de reproduction


def check(n: int = 40) -> dict:
    """40 chunks de la collection **servie**, ré-embarqués et comparés à leurs vecteurs.

    C'est la garde de ``convert_tables`` et de ``reembed_titles``, reprise telle quelle : si
    la recette locale ne reproduit pas les vecteurs du corpus, tout ce qui suit mesurerait
    autre chose que ce qu'on croit.
    """
    from qdrant_client import models

    titres, _ = titres_plonges()
    client = quant_rag.client()
    flt = models.Filter(must_not=[models.FieldCondition(
        key="content_type", match=models.MatchValue(value="table"))])
    points, offset = [], None
    while True:
        lot, offset = client.scroll(quant_rag.COLLECTION, limit=2048, offset=offset,
                                    scroll_filter=flt, with_payload=True, with_vectors=True)
        points.extend(lot)
        if offset is None:
            break
    echantillon = points[:: max(len(points) // n, 1)][:n]
    textes = [retrieval_text(titres[p.payload["document_id"]], p.payload.get("title_path"),
                             p.payload.get("page_start"), p.payload["text"], avec_page=False)
              for p in echantillon]
    modele = quant_rag.embedder()
    calcules = np.asarray(modele.encode(textes, normalize_embeddings=True,
                                        show_progress_bar=False, batch_size=BATCH), dtype=np.float32)
    stockes = np.asarray([p.vector for p in echantillon], dtype=np.float32)
    cos = (calcules * stockes).sum(axis=1)
    return {"chunks": len(echantillon), "cosinus_min": float(cos.min()),
            "cosinus_median": float(np.median(cos)), "cosinus_moyen": float(cos.mean())}


# ------------------------------------------------------------------ embedding avec reprise


def embarquer(signature: str, bras: str) -> dict:
    if bras not in BRAS:
        sys.exit(f"bras inconnu : {bras} (attendu {BRAS})")
    racine = _candidat(signature)
    CACHE.mkdir(parents=True, exist_ok=True)
    final = CACHE / f"vectors-candidat-{signature}-{bras}.npz"
    partiel = CACHE / f"vectors-candidat-{signature}-{bras}.partial.npz"
    if final.exists():
        blob = np.load(final)
        print(f"bras {bras} déjà calculé : {len(blob['chunk_ids'])} vecteurs ({final.name})")
        return {"bras": bras, "vecteurs": int(len(blob["chunk_ids"])), "deja_fait": True}

    chunks = _chunks(racine)
    titres, _ = titres_plonges()
    textes, stats = textes_du_bras(chunks, bras, titres)
    par_id = {c["chunk_id"]: t for c, t in zip(chunks, textes)}
    if stats:
        print(f"  sabotage : {stats}")

    fait: dict[str, np.ndarray] = {}
    if partiel.exists():
        blob = np.load(partiel)
        fait = dict(zip(blob["chunk_ids"].tolist(), blob["vectors"]))
        print(f"  reprise : {len(fait)} chunks déjà embarqués")

    # Tri par longueur, comme apply_delivery : moins de remplissage, donc plus de débit —
    # et surtout la MÊME composition de lots que la production.
    todo = sorted((c["chunk_id"] for c in chunks if c["chunk_id"] not in fait),
                  key=lambda cid: len(par_id[cid]))
    modele = quant_rag.embedder()
    debut = time.perf_counter()
    for numero, depart in enumerate(range(0, len(todo), BLOC), 1):
        lot = todo[depart:depart + BLOC]
        vecteurs = np.asarray(modele.encode([par_id[cid] for cid in lot], normalize_embeddings=True,
                                            show_progress_bar=False, batch_size=BATCH), dtype=np.float32)
        for cid, v in zip(lot, vecteurs):
            fait[cid] = v
        dernier = depart + BLOC >= len(todo)
        if numero % SAUVEGARDES_TOUS_LES == 0 or dernier:
            np.savez(partiel, chunk_ids=np.asarray(list(fait)), vectors=np.vstack(list(fait.values())))
        ecoule = max(time.perf_counter() - debut, 1e-6)
        debit = (depart + len(lot)) / ecoule
        reste = (len(todo) - depart - len(lot)) / max(debit, 1e-6)
        print(f"    {len(fait)}/{len(chunks)}  {debit:.2f} chunks/s  reste ~{reste / 60:.0f} min",
              flush=True)

    ordre = [c["chunk_id"] for c in chunks]
    np.savez_compressed(final, chunk_ids=np.asarray(ordre),
                        vectors=np.vstack([fait[c] for c in ordre]))
    partiel.unlink(missing_ok=True)
    rapport = {"bras": bras, "signature": signature, "vecteurs": len(ordre),
               "secondes": round(time.perf_counter() - debut, 1), "sabotage": stats}
    (CACHE / f"vectors-candidat-{signature}-{bras}.json").write_text(
        json.dumps(rapport, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"  écrit {final.name} — {len(ordre)} vecteurs")
    return rapport


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--check", action="store_true", help="reproduction de la recette, rien n'est écrit")
    p.add_argument("--candidat", metavar="SIG")
    p.add_argument("--bras", choices=BRAS)
    a = p.parse_args()
    if a.check:
        r = check()
        print(json.dumps(r, ensure_ascii=False))
        if r["cosinus_min"] < 0.99:
            sys.exit("ÉCHEC : la recette locale ne reproduit pas les vecteurs servis — ne rien écrire")
        print("recette reproduite (cosinus min >= 0,99)")
        if not a.candidat:
            return
    if not (a.candidat and a.bras):
        p.print_help()
        return
    print(json.dumps(embarquer(a.candidat, a.bras), ensure_ascii=False))


if __name__ == "__main__":
    main()
