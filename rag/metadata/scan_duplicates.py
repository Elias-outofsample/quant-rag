"""Détection systématique des documents en double — sur le contenu, pas sur les noms.

Pour chacune des 33 153 paires de documents : recouvrement de contenu (8-grammes de
mots échantillonnés 1/4 par hachage, hors tableaux ; *containment* = |A∩B| / min(|A|,|B|)),
similarité des titres consolidés (`documents-metadata-v1.json`) et auteurs communs.
Une paire est rapportée si containment ≥ 0,15, ou titre ≥ 0,70, ou (auteur commun et
titre ≥ 0,45). Les questions des deux bancs qui visent un document de la paire sont listées.

    python rag/metadata/scan_duplicates.py   → results-duplicates-scan-v1.json
"""
from __future__ import annotations

import collections
import difflib
import itertools
import json
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
#: Les fichiers canoniques, et non ``data/qdrant-export/payloads.jsonl``. L'export décrit le
#: corpus initial et ne bouge plus : il ignorait 62 des 318 documents actifs, et un scan de
#: doublons aveugle à 62 documents ne rapporte pas « je n'ai pas regardé », il rapporte
#: « rien à signaler ». ``inspect_delivery.corpus_shingles()`` lit déjà d'ici, et compare
#: chaque entrant à ce même ensemble — les deux gardes voient enfin la même chose.
INGESTED = ROOT / "data" / "processed" / "ingested"
BENCH = ROOT / "rag" / "benchmark"
OUT = HERE / "results-duplicates-scan-v1.json"
_WORD = re.compile(r"\w+")


def fold(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", (text or "").lower()) if not unicodedata.combining(c))


def shingles(text: str, k: int = 8, keep: int = 4) -> set[int]:
    words = _WORD.findall(text.lower())
    out = set()
    for i in range(max(len(words) - k, 0) + 1):
        h = hash(" ".join(words[i:i + k]))
        if h % keep == 0:
            out.add(h)
    return out


def corpus_chunks() -> dict[str, list[dict]]:
    """Chunks **éligibles** des documents actifs, lus dans les fichiers canoniques.

    Même source et même filtre que ``inspect_delivery.corpus_shingles()`` : un document
    retiré par l'overlay des doublons n'est pas là, et les tableaux sont exclus par
    ``shingles`` en aval, pas ici — ils comptent dans le nombre de chunks affiché.
    """
    import sys

    sys.path.insert(0, str(ROOT / "rag"))
    import corpus_overlay

    retires = corpus_overlay.removed_documents()
    docs: dict[str, list[dict]] = collections.defaultdict(list)
    for dossier in sorted(INGESTED.iterdir()):
        if not (dossier / "chunks.jsonl").exists() or not (dossier / "document.json").exists():
            continue
        document = json.loads((dossier / "document.json").read_text(encoding="utf-8"))
        identifiant = document["document_id"]
        if identifiant in retires:
            continue
        for ligne in (dossier / "chunks.jsonl").open(encoding="utf-8"):
            if not ligne.strip():
                continue
            chunk = json.loads(ligne)
            if chunk.get("rag_eligible") is not True:
                continue
            docs[identifiant].append({"chunk_id": chunk["chunk_id"], "document_id": identifiant,
                                      "content_type": chunk.get("content_type"),
                                      "text": chunk.get("text") or ""})
    return dict(docs)


def main() -> None:
    meta = {r["document_id"]: r for r in json.loads((HERE / "documents-metadata-v1.json").read_text(encoding="utf-8"))["documents"]}
    docs = corpus_chunks()
    sets = {d: set().union(*[shingles(p["text"]) for p in ps if p["content_type"] != "table"] or [set()]) for d, ps in docs.items()}

    chunk_doc = {p["chunk_id"]: p["document_id"] for ps in docs.values() for p in ps}
    touched: dict[str, list[str]] = collections.defaultdict(list)
    for line in (BENCH / "questions-v1.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            q = json.loads(line)
            touched[chunk_doc.get(q["target_chunk"], "?")].append(f"v1/{q['qid']}")
    for line in (BENCH / "questions-v2.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            q = json.loads(line)
            for d in q.get("gold_documents") or []:
                touched[d].append(f"v2/{q['qid']}")

    def surnames(rec: dict) -> set[str]:
        return {fold(a.split()[-1]) for a in rec.get("authors") or [] if a.split()}

    pairs = []
    for a, b in itertools.combinations(sorted(docs), 2):
        A, B = sets[a], sets[b]
        containment = len(A & B) / min(len(A), len(B)) if A and B else 0.0
        ra, rb = meta[a], meta[b]
        title_sim = difflib.SequenceMatcher(None, fold(ra["title"]), fold(rb["title"])).ratio()
        shared = len(surnames(ra) & surnames(rb))
        if containment >= 0.15 or title_sim >= 0.70 or (shared >= 1 and title_sim >= 0.45):
            pairs.append({
                "containment": round(containment, 3), "title_similarity": round(title_sim, 3), "shared_authors": shared,
                "documents": [{"document_id": d, "short_ref": meta[d]["short_ref"], "title": meta[d]["title"],
                               "filename": meta[d]["filename"], "chunks": len(docs[d]), "benchmark": touched.get(d, [])}
                              for d in (a, b)],
                "verdict": ("duplicate" if containment >= 0.95 else "edition" if containment >= 0.5
                            else "related-distinct"),
            })
    pairs.sort(key=lambda p: (-p["containment"], -p["title_similarity"]))
    summary = {"documents": len(docs), "source": "data/processed/ingested (fichiers canoniques, chunks éligibles)",
               "pairs_compared": len(docs) * (len(docs) - 1) // 2, "pairs_reported": len(pairs),
               "verdicts": dict(collections.Counter(p["verdict"] for p in pairs))}
    OUT.write_text(json.dumps({"version": "duplicates-scan-v1", "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                               "method": {"shingles": "8 mots, échantillon 1/4 par hachage, tableaux exclus",
                                          "containment": "|A∩B| / min(|A|,|B|)",
                                          "report_if": "containment ≥ 0,15 ou titre ≥ 0,70 ou (auteur commun et titre ≥ 0,45)",
                                          "verdict": {"duplicate": "≥ 0,95", "edition": "≥ 0,50", "related-distinct": "sinon"}},
                               "summary": summary, "pairs": pairs}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    for p in pairs[:6]:
        print(f"  {p['containment']:.2f} {p['verdict']:16} {p['documents'][0]['short_ref'][:28]:28} / {p['documents'][1]['short_ref'][:28]}")
    print("écrit :", OUT)


if __name__ == "__main__":
    main()
