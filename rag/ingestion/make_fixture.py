"""Fabrique une livraison d'essai à partir du corpus local — sept cas, aucun effet de bord.

Le fixture est dérivé de vrais documents du corpus (tronqués à quelques chunks pour
rester petit) et couvre les sept situations que ``inspect_delivery.py`` doit distinguer :

    sans-effet      document déjà présent, octet pour octet
    ajout           document neuf (sha256 inédit, document_id dérivé de la même règle)
    révision        document connu dont un chunk a changé de texte
    re-parse        même PDF, parser_version 3.5.0 : nouveau document_id pour le même contenu
    doublon-retiré  document écarté délibérément par rag/metadata/duplicates-v1.json
    rejet           parents.jsonl absent
    somme fausse    document neuf dont la somme de contrôle déclarée ne correspond pas
    dérive          champ inconnu ajouté, champ de payload retiré, rag_eligible non booléen

Le manifeste produit est celui de l'amont (``schema_version`` 5.1) et déclare les sommes
de contrôle réelles, sauf pour un document dont la somme est volontairement fausse.

    .venv/bin/python rag/ingestion/make_fixture.py                      # -> tests/fixtures/livraison-v1
    .venv/bin/python rag/ingestion/make_fixture.py --out /tmp/livraison --schema-version 6.0
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
INGESTED = ROOT / "data" / "processed" / "ingested"
DEFAULT_OUT = Path(__file__).resolve().parent / "tests" / "fixtures" / "livraison-v1"

#: Documents réels utilisés comme matière première (les plus petits du corpus).
SOURCE = "doc-83046c141ab04bf3"          # 12 chunks
SOURCE_B = "doc-501c39dabd3fa516"    # 27 chunks, porteur d'or au banc v3
REMOVED = "doc-ca4a1e834a2954b8"         # retiré par duplicates-v1.json
LIMIT = 6                                 # chunks conservés par document du fixture


def derive_document_id(sha256: str, backend: str, version: str) -> str:
    return "doc-" + hashlib.sha1("\x1f".join((sha256, backend, version, "canonical-v1")).encode()).hexdigest()[:16]


def derive_chunk_id(document_id: str, index: int, text: str) -> str:
    """La règle de ``src/parsing/canonical_chunker.py`` : elle dépend du document_id *et* du texte."""
    return "chunk-" + hashlib.sha1("\x1f".join((document_id, str(index), text)).encode()).hexdigest()[:16]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(folder: str) -> tuple[dict, list[dict], list[dict], list[dict]]:
    base = INGESTED / folder
    read = lambda name: [json.loads(l) for l in (base / name).open(encoding="utf-8") if l.strip()]
    return (json.loads((base / "document.json").read_text(encoding="utf-8")),
            read("blocks.jsonl"), read("chunks.jsonl"), read("parents.jsonl"))


def write(out: Path, folder: str, document: dict, blocks: list[dict], chunks: list[dict],
          parents: list[dict], skip: tuple[str, ...] = ()) -> Path:
    target = out / "processed" / folder
    target.mkdir(parents=True, exist_ok=True)
    if "document.json" not in skip:
        target.joinpath("document.json").write_text(json.dumps(document, ensure_ascii=False, indent=1), encoding="utf-8")
    for name, rows in (("blocks.jsonl", blocks), ("chunks.jsonl", chunks), ("parents.jsonl", parents)):
        if name in skip:
            continue
        target.joinpath(name).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    return target


def retarget(document: dict, chunks: list[dict], parents: list[dict], blocks: list[dict],
             document_id: str) -> tuple[dict, list[dict], list[dict], list[dict]]:
    """Réattribue toutes les identités comme le ferait le chunker amont.

    Le ``chunk_id`` est haché sur ``(document_id, indice, texte)`` : changer de
    ``document_id`` — donc changer de version de parseur — renouvelle *tous* les
    chunk_id du document. Un fixture qui les conserverait mentirait sur le coût réel
    d'un re-parse.
    """
    old = document["document_id"]
    remap = {c["chunk_id"]: derive_chunk_id(document_id, i, c["text"]) for i, c in enumerate(chunks)}
    swap_parent = lambda value: (value.replace(old[4:], document_id[4:])
                                 if isinstance(value, str) else value)
    document = {**document, "document_id": document_id}
    chunks = [{**c, "document_id": document_id, "chunk_id": remap[c["chunk_id"]],
               "parent_id": swap_parent(c.get("parent_id")),
               "previous_chunk_id": remap.get(c.get("previous_chunk_id")),
               "next_chunk_id": remap.get(c.get("next_chunk_id"))} for c in chunks]
    parents = [{**p, "document_id": document_id, "parent_id": swap_parent(p["parent_id"]),
                "child_chunk_ids": [remap[x] for x in p.get("child_chunk_ids", []) if x in remap]}
               for p in parents]
    blocks = [{**b, "document_id": document_id} for b in blocks]
    return document, blocks, chunks, parents


#: Jeton inséré par ``--distinct``. Un mot inventé, absent du corpus, et **stable** : la
#: livraison doit être reproductible au bit près d'un appel à l'autre.
JETON_DISTINCT = "zzqvx"


def rendre_distinct(chunks: list[dict], toutes_les: int = 4) -> list[dict]:
    """Rend le contenu **réellement neuf** au regard de la garde de quasi-doublon.

    ``--single`` recopie un document réel mot pour mot : sa livraison a donc un
    ``containment`` de 1,00 avec son original, et ``inspect_delivery`` la refuse comme
    doublon — c'est la garde qui fait son travail, pas un défaut du fixture. Or une mesure
    du **chemin d'écriture** a besoin d'un document que ce chemin accepte.

    La garde compare des **8-grammes de mots** échantillonnés
    (``scan_duplicates.shingles``, ``k=8``). Insérer un jeton tous les quatre mots casse donc
    chaque 8-gramme du document d'origine, sans changer ni le nombre de chunks, ni leur
    ordre, ni leurs longueurs à quelques pour cent près — c'est-à-dire sans changer ce que la
    mesure observe : le coût d'embedding, d'upsert, d'extraction et de reconstruction.

    **Le texte produit n'est pas de la prose**, et n'a pas à l'être : aucune mesure de
    qualité de réponse ne s'appuie sur ce fixture, et il n'entre jamais dans le corpus servi.
    """
    sortis = []
    for chunk in chunks:
        texte = chunk.get("text") or ""
        if chunk.get("rag_eligible") is not True or chunk.get("content_type") == "table" or not texte:
            sortis.append(dict(chunk))
            continue
        mots, melange = texte.split(), []
        for indice, mot in enumerate(mots):
            melange.append(mot)
            if indice % toutes_les == toutes_les - 1:
                melange.append(JETON_DISTINCT)
        sortis.append({**chunk, "text": " ".join(melange)})
    return sortis


#: Un tableau réel coupé en trois, tel que le chunker amont le produit : seul le premier
#: fragment porte la rangée d'en-tête, les deux suivants ne sont que des lignes de données.
#: C'est la forme exacte que l'étape 4 d'``apply_delivery`` traitait mal — voir
#: ``rag/tables/recenser_en_tetes.py``.
FRAGMENTS_TABLEAU = (
    "<table><tr><td>Horizon</td><td>Sharpe</td><td>Turnover</td></tr>"
    "<tr><td>1 jour</td><td>0.41</td><td>2.10</td></tr></table>",
    "<table><tr><td>5 jours</td><td>0.77</td><td>0.94</td></tr>"
    "<tr><td>10 jours</td><td>0.83</td><td>0.61</td></tr></table>",
    "<table><tr><td>21 jours</td><td>0.68</td><td>0.33</td></tr>"
    "<tr><td>63 jours</td><td>0.52</td><td>0.18</td></tr></table>",
)


#: Un SECOND tableau, immédiatement voisin du premier dans la chaîne du document mais
#: d'un autre ``parent_id``. C'est le cas que la garde de ``first_part`` existe pour
#: attraper : sans ``parent_id``, la remontée franchirait la frontière entre deux tableaux
#: et coifferait celui-ci de l'en-tête de l'autre.
SECOND_TABLEAU = (
    "<table><tr><td>Actif</td><td>Volatilité</td></tr>"
    "<tr><td>SPX</td><td>0.18</td></tr></table>",
    "<table><tr><td>NDX</td><td>0.24</td></tr>"
    "<tr><td>RTY</td><td>0.29</td></tr></table>",
)


def ajouter_tableau_fragmente(chunks: list[dict], document_id: str) -> list[dict]:
    """Ajoute en queue **deux** tableaux fragmentés, voisins et de parents différents.

    Le premier a trois fragments, le second deux. Ils se suivent dans la chaîne
    ``previous_chunk_id`` du document — comme deux tableaux d'une même section — mais ne
    partagent pas leur ``parent_id``. Il en faut deux, et pas un : avec un seul tableau,
    supprimer ``parent_id`` du payload ne change rien (``None == None``), et un test bâti
    dessus ne prouverait que la moitié du correctif. Mesuré : la mutation qui retire
    ``parent_id`` ne tuait pas le test tant que le fixture n'avait qu'un tableau.

    Les identifiants sont dérivés par la règle du chunker amont (``derive_chunk_id``), pour
    que le manifeste et les sommes de contrôle restent cohérents.
    """
    debut = len(chunks)
    ajoutes: list[dict] = []
    precedent = None
    rang = 0
    for suffixe, textes in (("tableau", FRAGMENTS_TABLEAU), ("tableau-2", SECOND_TABLEAU)):
        parent = "parent-" + hashlib.sha1(f"{document_id}::{suffixe}".encode()).hexdigest()[:16]
        for texte in textes:
            chunk_id = derive_chunk_id(document_id, debut + rang, texte)
            ajoutes.append({
                "chunk_id": chunk_id, "document_id": document_id, "parent_id": parent,
                "previous_chunk_id": precedent, "next_chunk_id": None,
                "content_type": "table", "rag_eligible": True,
                "title_path": "3. RÉSULTATS", "section": "3. RÉSULTATS",
                "part": None, "chapter": None,
                "page_start": 4, "page_end": 4, "text": texte,
            })
            if precedent is not None:
                ajoutes[-2]["next_chunk_id"] = chunk_id
            precedent = chunk_id
            rang += 1
    return chunks + ajoutes


def single(out: Path, source: str, schema_version: str, distinct: bool = False,
           tableau_fragmente: bool = False) -> None:
    """Livraison d'un seul document neuf, dérivé d'un document réel du corpus.

    Tous ses chunks, ses tableaux, sa structure : seules les identités sont refaites
    (sha256 inédit, document_id dérivé par la règle amont, chunk_id re-hachés). C'est la
    livraison la plus proche du réel qu'on puisse fabriquer sans MinerU.

    ``distinct`` casse en plus les 8-grammes du document d'origine — sans quoi la livraison
    est un doublon parfait et le chemin d'écriture la refuse (voir ``rendre_distinct``).
    """
    document, blocks, chunks, parents = load(source)
    if distinct:
        chunks = rendre_distinct(chunks)
    new_sha = hashlib.sha256(
        (f"livraison-essai{'-distincte' if distinct else ''}"
         f"{'-tableau' if tableau_fragmente else ''}::{source}").encode()).hexdigest()
    new_id = derive_document_id(new_sha, document["parser_backend"], document["parser_version"])
    d, b, c, p = retarget({**document, "sha256": new_sha,
                           "filename": f"2026-09-04_Essai_{source[4:12]}.pdf",
                           "source_path": rf"C:\corpus\data\papers\2026-09-04_Essai_{source[4:12]}.pdf",
                           "title": f"2026 09 04 Essai {source[4:12]}"},
                          chunks, parents, blocks, new_id)
    # Après ``retarget`` : les fragments sont ajoutés avec leur propre chaîne, qui ne doit
    # pas être réécrite par le remappage des identifiants du document d'origine.
    if tableau_fragmente:
        c = ajouter_tableau_fragmente(c, new_id)
    folder = f"doc-{new_sha[:16]}"
    write(out, folder, d, b, c, p)
    entry = {"document_id": new_id, "title": d["title"], "sha256": new_sha,
             "parser_backend": d["parser_backend"], "parser_version": d["parser_version"],
             "chunks_count": len(c),
             "eligible_chunks_count": sum(1 for x in c if x.get("rag_eligible") is True),
             "chunks_jsonl_sha256": sha256_file(out / "processed" / folder / "chunks.jsonl"),
             "parents_jsonl_sha256": sha256_file(out / "processed" / folder / "parents.jsonl")}
    (out / "manifest.json").write_text(json.dumps({
        "corpus_version": "quant-rag-corpus-v1", "schema_version": schema_version,
        "parser_backend": d["parser_backend"], "parser_version": d["parser_version"],
        "producer": "windows-pipeline", "producer_commit": "0000000",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "added": [new_id], "revised": [], "removed": [],
        "documents": [entry], "config_hash": hashlib.sha256(b"essai").hexdigest(),
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({"status": "COMPLETED", "out": str(out), "derived_from": source,
                      "folder": folder, "document_id": new_id,
                      "chunks": entry["chunks_count"], "eligible": entry["eligible_chunks_count"],
                      "tables": sum(1 for x in c if x.get("content_type") == "table")},
                     ensure_ascii=False, indent=1))


#: Les deux couples réels de ``rag/metadata/duplicates-v1.json``, relivrés sous un PDF
#: différent. C'est le scénario que la garde d'identité ne peut pas voir : même papier,
#: autre sha256, donc « ajout » — seule la comparaison de contenu l'attrape.
DUPLICATE_CASES = [
    ("doc-ca4a1e834a2954b8", None, "doublon",
     "deflated-sharpe.pdf relivré : containment 1,00 avec le miroir SSRN conservé"),
    ("doc-0571f22324646fe1", 120, "édition",
     "Tsay 2e édition relivrée : contenue à 81 % dans la 3e, conservée"),
]


def duplicates(out: Path, schema_version: str) -> None:
    """Livraison de deux documents que seul le contenu dénonce."""
    entries, cases = [], {}
    for source, limit, kind, why in DUPLICATE_CASES:
        document, blocks, chunks, parents = load(source)
        if limit:
            chunks, blocks = chunks[:limit], blocks[:limit * 2]
        new_sha = hashlib.sha256(f"relivraison::{source}".encode()).hexdigest()
        new_id = derive_document_id(new_sha, document["parser_backend"], document["parser_version"])
        d, b, c, p = retarget({**document, "sha256": new_sha,
                               "filename": f"2026-09-04_Relivraison_{kind}.pdf",
                               "title": f"2026 09 04 Relivraison {kind}"},
                              chunks, parents, blocks, new_id)
        folder = f"doc-{new_sha[:16]}"
        write(out, folder, d, b, c, p)
        cases[folder] = f"{kind} — {why}"
        entries.append({"document_id": new_id, "title": d["title"], "sha256": new_sha,
                        "parser_backend": d["parser_backend"], "parser_version": d["parser_version"],
                        "chunks_count": len(c),
                        "eligible_chunks_count": sum(1 for x in c if x.get("rag_eligible") is True),
                        "chunks_jsonl_sha256": sha256_file(out / "processed" / folder / "chunks.jsonl"),
                        "parents_jsonl_sha256": sha256_file(out / "processed" / folder / "parents.jsonl")})
    (out / "manifest.json").write_text(json.dumps({
        "corpus_version": "quant-rag-corpus-v1", "schema_version": schema_version,
        "parser_backend": "mineru", "parser_version": "3.4.5", "producer": "windows-pipeline",
        "producer_commit": "0000000", "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "added": [e["document_id"] for e in entries], "revised": [], "removed": [],
        "documents": entries, "config_hash": hashlib.sha256(b"doublons").hexdigest(),
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "CAS.md").write_text("# Quasi-doublons\n\n" + "".join(
        f"- `{k}` — {v}\n" for k, v in sorted(cases.items())), encoding="utf-8")
    print(json.dumps({"status": "COMPLETED", "out": str(out), "cases": cases},
                     ensure_ascii=False, indent=1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--schema-version", default="5.1", help="5.1 = connue ; autre = doit être refusée")
    parser.add_argument("--single", metavar="DOSSIER",
                        help="livraison d'un seul document neuf, dérivé de ce dossier du corpus")
    parser.add_argument("--distinct", action="store_true",
                        help="avec --single : casse les 8-grammes du document d'origine pour que "
                             "la livraison soit un vrai ajout et non un doublon refusé")
    parser.add_argument("--tableau-fragmente", action="store_true",
                        help="avec --single : ajoute un tableau coupé en trois fragments "
                             "enchaînés, pour éprouver l'héritage d'en-tête à l'import")
    parser.add_argument("--duplicates", action="store_true",
                        help="livraison de deux quasi-doublons réels (doublon et édition)")
    args = parser.parse_args()
    out = args.out
    if args.single or args.duplicates:
        if out.exists():
            shutil.rmtree(out)
        out.mkdir(parents=True)
        if args.single:
            single(out, args.single, args.schema_version, args.distinct,
                   args.tableau_fragmente)
        else:
            duplicates(out, args.schema_version)
        return
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    entries, cases = [], {}
    document, blocks, chunks, parents = load(SOURCE)
    eligible = [c for c in chunks if c.get("rag_eligible")]
    chunks, blocks = (eligible or chunks)[:LIMIT], blocks[:LIMIT * 3]

    # 1. sans-effet — le document tel quel : mêmes octets que le corpus local
    full_doc, full_blocks, full_chunks, full_parents = load(SOURCE)
    folder = f"doc-{full_doc['sha256'][:16]}"
    write(out, folder, full_doc, full_blocks, full_chunks, full_parents)
    cases[folder] = "sans-effet"

    # 2. ajout — un PDF inédit : sha256 neuf, document_id dérivé par la même règle
    new_sha = hashlib.sha256(b"fixture-nouveau-papier-2026").hexdigest()
    new_id = derive_document_id(new_sha, "mineru", "3.4.5")
    d, b, c, p = retarget({**document, "sha256": new_sha, "filename": "2026-09-04_Nouveau_Papier.pdf",
                           "source_path": r"C:\corpus\data\papers\2026-09-04_Nouveau_Papier.pdf",
                           "title": "2026 09 04 Nouveau Papier"}, chunks, parents, blocks, new_id)
    folder = f"doc-{new_sha[:16]}"
    write(out, folder, d, b, c, p)
    cases[folder] = "ajout"

    # 3. révision — document connu, un chunk réécrit en amont
    d, b, c, p = load(SOURCE_B)
    c = [dict(x) for x in c]
    target = next(i for i, x in enumerate(c) if x.get("rag_eligible"))
    text = c[target]["text"] + "\n\nParagraphe ajouté par une révision amont."
    c[target] = {**c[target], "text": text, "chunk_id": derive_chunk_id(d["document_id"], target, text)}
    folder = f"doc-{d['sha256'][:16]}"
    write(out, folder, d, b, c, p)
    cases[folder] = "révision"

    # 4. re-parse — même PDF, MinerU 3.5.0 : le document_id change, le document est le même
    d0, b0, c0, p0 = load(SOURCE)
    reparse_id = derive_document_id(d0["sha256"], "mineru", "3.5.0")
    d, b, c, p = retarget({**d0, "parser_version": "3.5.0"}, c0[:LIMIT], p0, b0[:LIMIT * 3], reparse_id)
    folder = f"doc-{d0['sha256'][:16]}-reparse"     # même sha256, dossier distinct : le cas piégeux
    write(out, folder, d, b, c, p)
    cases[folder] = "re-parse"

    # 5. doublon-retiré — un document écarté délibérément, relivré
    d, b, c, p = load(REMOVED)
    folder = f"doc-{d['sha256'][:16]}"
    write(out, folder, d, b[:LIMIT * 3], c[:LIMIT], p)
    cases[folder] = "doublon-retiré"

    # 6. rejet — parents.jsonl absent
    new_sha = hashlib.sha256(b"fixture-livraison-incomplete").hexdigest()
    d, b, c, p = retarget({**document, "sha256": new_sha, "filename": "incomplet.pdf"},
                          chunks, parents, blocks, derive_document_id(new_sha, "mineru", "3.4.5"))
    folder = f"doc-{new_sha[:16]}"
    write(out, folder, d, b, c, p, skip=("parents.jsonl",))
    cases[folder] = "rejet (parents.jsonl absent)"

    # 7. somme de contrôle fausse — transfert incomplet, sur un document par ailleurs sain
    new_sha = hashlib.sha256(b"fixture-somme-fausse").hexdigest()
    d, b, c, p = retarget({**document, "sha256": new_sha, "filename": "transfert-tronque.pdf",
                           "title": "Transfert tronque"},
                          chunks, parents, blocks, derive_document_id(new_sha, "mineru", "3.4.5"))
    folder = f"doc-{new_sha[:16]}"
    write(out, folder, d, b, c, p)
    cases[folder] = "somme de contrôle fausse"

    # 8. dérive de schéma — champ inconnu, champ de payload retiré, rag_eligible non booléen
    new_sha = hashlib.sha256(b"fixture-schema-derive").hexdigest()
    d, b, c, p = retarget({**document, "sha256": new_sha, "filename": "derive.pdf",
                           "chunking_profile": "semantic-v2"},           # champ inconnu du modèle
                          chunks, parents, blocks, derive_document_id(new_sha, "mineru", "3.4.5"))
    c = [{k: v for k, v in x.items() if k != "title_path"} | {"rag_eligible": 1, "embedding_hint": "dense"}
         for x in c]
    folder = f"doc-{new_sha[:16]}"
    write(out, folder, d, b, c, p)
    cases[folder] = "dérive de schéma"

    # manifeste : sommes réelles, sauf une volontairement fausse (transfert incomplet simulé)
    for folder_path in sorted((out / "processed").iterdir()):
        doc_path = folder_path / "document.json"
        chunks_path = folder_path / "chunks.jsonl"
        if not doc_path.exists() or not chunks_path.exists():
            continue
        doc = json.loads(doc_path.read_text(encoding="utf-8"))
        rows = [json.loads(l) for l in chunks_path.open(encoding="utf-8") if l.strip()]
        digest = sha256_file(chunks_path)
        if cases.get(folder_path.name) == "somme de contrôle fausse":
            digest = "0" * 64                      # somme fausse : la livraison doit être refusée
        entries.append({"document_id": doc["document_id"], "title": doc.get("title"),
                        "sha256": doc["sha256"], "chunks_count": len(rows),
                        "eligible_chunks_count": sum(1 for r in rows if r.get("rag_eligible") is True),
                        "chunks_jsonl_sha256": digest,
                        "parents_jsonl_sha256": sha256_file(folder_path / "parents.jsonl")
                        if (folder_path / "parents.jsonl").exists() else None})

    (out / "manifest.json").write_text(json.dumps({
        "corpus_version": "quant-rag-corpus-v1", "schema_version": args.schema_version,
        "parser_backend": "mineru", "parser_version": "3.4.5",
        "producer": "windows-pipeline", "producer_commit": "0000000",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "documents": entries,
        "config_hash": hashlib.sha256(b"fixture").hexdigest(),
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "CAS.md").write_text("# Cas couverts\n\n" + "".join(
        f"- `{k}` — {v}\n" for k, v in sorted(cases.items())), encoding="utf-8")
    print(json.dumps({"status": "COMPLETED", "out": str(out), "documents": len(entries),
                      "cases": cases, "schema_version": args.schema_version}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
