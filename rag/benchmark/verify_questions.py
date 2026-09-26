"""Re-vérifie des questions existantes contre leurs chunks d'or *actuels*.

Cas d'usage : une question dont le chunk d'or a été re-ciblé (``gold_remapped``,
document retiré du corpus → jumeau dans l'édition conservée) a été validée sur un
passage qui n'existe plus. Le jumeau contient 76 à 97 % du texte d'origine, ce qui
est une présomption, pas une preuve. On rejoue donc le vérificateur de la
génération (répondable, autoportante, dépendante de ce passage) sur le nouveau
chunk, et on recalcule la fuite lexicale.

    .venv/bin/python rag/benchmark/verify_questions.py                         # les questions re-ciblées de v3
    .venv/bin/python rag/benchmark/verify_questions.py --qid s08 m02 --write   # écrit ``gold_reverified`` dans le fichier

Aucun retrieval n'intervient : c'est le passage désigné qui est jugé, pas le
système sous test.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import generate_questions as gen  # noqa: E402
import llm  # noqa: E402
from corpus import ChunkIndex  # noqa: E402


def verify(item: dict, index: ChunkIndex) -> dict:
    chunks = [index.get(c) for c in item["gold_chunks"]]
    if any(c is None for c in chunks):
        return {"error": "chunk d'or absent du corpus"}
    if len(chunks) == 1:
        check = llm.complete_json(
            [{"role": "user", "content": gen.VERIFY % {"question": item["question"],
                                                          "passage": chunks[0]["text"][:gen.CHUNK_CHARACTERS],
                                                          "documents": len(index.documents)}}],
            model=llm.JUDGE, temperature=0.0, max_tokens=250, seed=f"reverify-{item['qid']}") or {}
        keys = ("answerable", "self_contained", "needs_this_passage")
        text = chunks[0]["text"]
    else:
        check = llm.complete_json(
            [{"role": "user", "content": gen.VERIFY_MULTI % (item["question"], chunks[0]["text"][:3200], chunks[1]["text"][:3200])}],
            model=llm.JUDGE, temperature=0.0, max_tokens=250, seed=f"reverify-{item['qid']}") or {}
        keys = ("answerable", "self_contained", "needs_both")
        text = "\n".join(c["text"] for c in chunks)
    verdict = {k: bool(check.get(k)) for k in keys}
    verdict["reason"] = str(check.get("reason", ""))[:200]
    verdict["passed"] = all(verdict[k] for k in keys)
    verdict["leak"] = round(index.leak_against_text(item["question"], text), 3)
    verdict["model"] = llm.JUDGE
    verdict["checked_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return verdict


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--questions", type=Path, default=HERE / "questions-v3.jsonl")
    parser.add_argument("--qid", nargs="*", default=None, help="par défaut : les questions portant gold_remapped")
    parser.add_argument("--write", action="store_true", help="écrit le verdict dans le fichier (champ gold_reverified)")
    args = parser.parse_args()

    lines = args.questions.read_text(encoding="utf-8").splitlines()
    items = [json.loads(line) for line in lines if line.strip()]
    wanted = set(args.qid) if args.qid else {i["qid"] for i in items if i.get("gold_remapped")}
    index = ChunkIndex.load(verbose=False)

    out_lines, failed = [], []
    for line in lines:
        if not line.strip():
            continue
        item = json.loads(line)
        if item["qid"] not in wanted:
            out_lines.append(line)
            continue
        verdict = verify(item, index)
        flag = "✓" if verdict.get("passed") else "✗"
        print(f"{flag} {item['qid']:<4} fuite {verdict.get('leak')}  {verdict}")
        if not verdict.get("passed"):
            failed.append(item["qid"])
        if args.write:
            item["gold_reverified"] = verdict
            out_lines.append(json.dumps(item, ensure_ascii=False))
        else:
            out_lines.append(line)
    if args.write:
        args.questions.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
        print(f"-> {args.questions}")
    print(f"{len(wanted) - len(failed)}/{len(wanted)} validées ; en échec : {failed or 'aucune'}")


if __name__ == "__main__":
    main()
