"""Applique ``documents-metadata-v1.json`` au payload Qdrant — sans ré-embedding — et le prouve.

Qdrant met à jour le payload d'un point sans toucher à son vecteur (``set_payload``).
Ce script ne le suppose pas, il le vérifie : avant la mise à jour, il fige les vecteurs
de 40 points et le top-50 dense de huit requêtes ; après, il compare — vecteurs égaux à
l'epsilon float32 près (1e-6, et écart maximal rapporté), vecteurs égaux à l'export
``data/qdrant-export/vectors.npy`` (la vérité de référence), identité des identifiants
et des scores du classement.

    python rag/metadata/apply_metadata.py            # applique + vérifie
    python rag/metadata/apply_metadata.py --dry-run  # compte, n'écrit rien

Un seul processus à la fois sur ``qdrant_storage_local/`` (verrou du Qdrant embarqué).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import quant_rag  # noqa: E402

PROBES = (
    "rough volatility and the short-time ATM skew",
    "deflated Sharpe ratio backtest overfitting",
    "Almgren Chriss optimal execution 2001",
    "SVI parametrisation butterfly arbitrage",
    "Kelly criterion fractional sizing",
    "Hamilton regime switching Markov model business cycle",
    "vanna volga FX options pricing",
    "cointegration Engle Granger two step",
)
RESULT = Path(__file__).resolve().parent / "results-apply-metadata-v1.json"
#: Pourquoi la comparaison à ``data/qdrant-export/vectors.npy`` ne peut plus être une égalité.
#: Elle en était une le 2 septembre 2026 (``max_abs_delta_vs_export: 1.5e-08``), parce que la
#: collection *était* l'export : 19 443 points, aucun overlay. La collection servie ne l'est
#: plus — les titres consolidés remplacent le titre d'export dans ``embedding_text``, et
#: ``INGESTION-CONCEPTION`` §2 chiffre l'écart à **0,896 de cosinus moyen**. Mesuré à nouveau
#: le 4 septembre sur 40 points de sonde : 40/40 divergents, cosinus moyen 0,9048. Faire
#: échouer là-dessus revenait à accuser l'écriture du payload d'un écart qu'elle n'a pas
#: produit — la mesure du même passage disait `delta 0.0` et `rankings_identical: true`.
EXPORT_NOTE = ("la collection servie applique l'overlay des titres propres, que l'export ne "
               "connaît pas (INGESTION-CONCEPTION §2 : 0,896 de cosinus moyen). L'égalité "
               "n'était vraie que sur la collection amont brute, avant overlays.")


def snapshot(client, point_ids: list[int]) -> dict:
    from qdrant_client import models  # noqa: F401

    vectors = {int(p.id): p.vector for p in client.retrieve(quant_rag.COLLECTION, ids=point_ids, with_vectors=True)}
    rankings = {}
    for probe in PROBES:
        hits = client.query_points(quant_rag.COLLECTION, query=quant_rag.encode_query(probe), limit=50).points
        rankings[probe] = [(int(h.id), round(float(h.score), 6)) for h in hits]
    return {"vectors": vectors, "rankings": rankings}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--metadata", type=Path, default=quant_rag.METADATA_PATH)
    args = parser.parse_args()

    from qdrant_client import models

    records = json.loads(args.metadata.read_text(encoding="utf-8"))["documents"]
    client = quant_rag.client()
    total = client.count(quant_rag.COLLECTION, exact=True).count
    print(f"{len(records)} documents dans {args.metadata.name} ; {total} points dans la collection")
    if args.dry_run:
        return

    # Les points de sonde se tirent parmi les identifiants **réellement présents**, jamais
    # dans range(0, total, pas) : la collection compte 807 trous laissés par les documents
    # retirés, et un identifiant supposé dense finit par tomber dedans — ce qui est arrivé le
    # 4 septembre 2026, au premier import qui a porté le compte à 18 919 : le pas est passé à
    # 472, les points 472 et 944 n'existaient pas, et la vérification est morte sur un
    # KeyError **après** avoir écrit les payloads. C'est le même piège que la règle amont
    # « id = compte + i » que ce dépôt a déjà écartée pour l'écriture ; il restait dans la
    # relecture.
    existing = sorted(p.id for p in client.scroll(quant_rag.COLLECTION, limit=total,
                                                  with_payload=False, with_vectors=False)[0])
    point_ids = existing[::max(len(existing) // 40, 1)][:40]
    before = snapshot(client, point_ids)

    started = time.perf_counter()
    updated_docs = 0
    for record in records:
        payload = quant_rag.payload_fields(record)
        client.set_payload(
            quant_rag.COLLECTION, payload=payload, wait=True,
            points=models.Filter(must=[models.FieldCondition(key="document_id",
                                                             match=models.MatchValue(value=record["document_id"]))]),
        )
        updated_docs += 1
        print(f"  {updated_docs}/{len(records)}  {record['short_ref'][:48]}", end="\r", flush=True)
    print()
    for field, schema in (("publication_year", models.PayloadSchemaType.INTEGER),
                          ("authors", models.PayloadSchemaType.KEYWORD),
                          ("document_id", models.PayloadSchemaType.KEYWORD)):
        client.create_payload_index(quant_rag.COLLECTION, field_name=field, field_schema=schema, wait=True)
    seconds = round(time.perf_counter() - started, 1)

    after = snapshot(client, point_ids)
    import numpy as np

    max_delta = max(float(np.abs(np.asarray(before["vectors"][i], dtype=np.float32)
                                 - np.asarray(after["vectors"][i], dtype=np.float32)).max()) for i in point_ids)
    vectors_identical = max_delta <= 1e-6
    export = np.load(quant_rag.EXPORT / "vectors.npy", mmap_mode="r")
    position = {int(pid): k for k, pid in enumerate(np.load(quant_rag.EXPORT / "ids.npy"))}
    max_delta_export = max(float(np.abs(np.asarray(after["vectors"][i], dtype=np.float32)
                                        - np.asarray(export[position[i]], dtype=np.float32)).max()) for i in point_ids)
    rankings_identical = all(before["rankings"][p] == after["rankings"][p] for p in PROBES)

    with_year = client.count(quant_rag.COLLECTION, exact=True, count_filter=models.Filter(
        must=[models.FieldCondition(key="publication_year", range=models.Range(gte=1900))])).count
    with_authors = client.count(quant_rag.COLLECTION, exact=True, count_filter=models.Filter(
        must_not=[models.IsEmptyCondition(is_empty=models.PayloadField(key="authors"))])).count
    summary = {
        "documents_updated": updated_docs, "points": total, "seconds": seconds,
        "points_with_publication_year": with_year, "points_with_authors": with_authors,
        "verification": {"probe_points": len(point_ids), "vectors_identical": vectors_identical,
                         "max_abs_delta_before_after": max_delta, "max_abs_delta_vs_export": max_delta_export,
                         "vectors_match_export": max_delta_export <= 1e-6,
                         "export_comparison": EXPORT_NOTE,
                         "probe_queries": len(PROBES), "rankings_identical": rankings_identical},
        "metadata_file": args.metadata.name,
        "created_at": json.loads(args.metadata.read_text(encoding="utf-8")).get("created_at"),
    }
    RESULT.write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    # La garde ne porte que sur ce qu'elle prétend prouver : que **l'écriture du payload**
    # n'a touché ni les vecteurs ni le classement. La comparaison à l'export reste mesurée et
    # publiée, mais ne fait plus échouer — voir EXPORT_NOTE.
    if not (vectors_identical and rankings_identical):
        sys.exit("ÉCHEC : la mise à jour du payload a modifié les vecteurs ou le classement — "
                 "rebâtir avec build_index.py")
    if max_delta_export > 1e-6:
        print(f"\nNote : écart à l'export {max_delta_export:.4f} — attendu, {EXPORT_NOTE}")


if __name__ == "__main__":
    main()
