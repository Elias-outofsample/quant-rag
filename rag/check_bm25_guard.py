"""Prouve que la garde de l'index BM25 sait refuser — avant de lui faire confiance.

``quant_rag._bm25_matches_collection()`` ne comparait qu'un compte de points. Deux index
bâtis sur la même collection avec des titres différents le passaient tous les deux : c'est
le « contenu périmé sous un nom correct » que le nommage par signature existe pour empêcher,
et il n'y protégeait pas. La garde vérifie désormais quatre choses. Ce script les fait
échouer une par une.

Rien n'est écrit hors d'un répertoire temporaire : le manifeste et l'index servis sont
copiés, et ce sont les copies que la garde examine (``bm25_path`` et ``bm25_manifest_path``
sont remplacés le temps du contrôle).

    .venv/bin/python rag/check_bm25_guard.py
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[0] / "src"))

import corpus_overlay  # noqa: E402
import quant_rag  # noqa: E402


def main() -> int:
    vrai_index, vrai_manifeste = quant_rag.bm25_path(), quant_rag.bm25_manifest_path()
    if not vrai_manifeste.exists():
        sys.exit(f"manifeste absent : {vrai_manifeste}")

    with tempfile.TemporaryDirectory() as tmp:
        boite = Path(tmp)
        index, manifeste = boite / vrai_index.name, boite / vrai_manifeste.name
        shutil.copy2(vrai_index, index)
        shutil.copy2(vrai_manifeste, manifeste)
        quant_rag.bm25_path = lambda: index
        quant_rag.bm25_manifest_path = lambda: manifeste
        origine = json.loads(manifeste.read_text(encoding="utf-8"))

        def essai(nom: str, transforme, attendu: bool) -> bool:
            manifeste.write_text(json.dumps(transforme(json.loads(json.dumps(origine))),
                                            indent=1, ensure_ascii=False), encoding="utf-8")
            obtenu = quant_rag._bm25_matches_collection()
            ok = obtenu is attendu
            print(f"  {'OK  ' if ok else 'RATÉ'}  {nom:<52} garde -> {obtenu}")
            return ok

        def altere_index(m):
            index.write_bytes(index.read_bytes() + b" ")
            return m

        print("=== la garde doit ACCEPTER l'index servi, tel quel ===")
        resultats = [essai("index intact", lambda m: m, True)]

        print("\n=== la garde doit REFUSER, quatre fois ===")
        resultats.append(essai("source des titres différente (export vs registry)",
                               lambda m: {**m, "title_source": "export"}, False))
        resultats.append(essai("titres corrigés depuis la construction",
                               lambda m: {**m, "titles_digest": "0" * 12}, False))
        resultats.append(essai("index d'un autre état du corpus",
                               lambda m: {**m, "corpus_signature": "0" * 10}, False))
        resultats.append(essai("fichier modifié après écriture du manifeste",
                               altere_index, False))
        shutil.copy2(vrai_index, index)   # remettre l'index intact pour le dernier essai
        resultats.append(essai("manifeste sans empreinte de contenu (format d'avant)",
                               lambda m: {k: v for k, v in m.items() if k != "index_sha256"}, False))

    total, reussis = len(resultats), sum(resultats)
    print(f"\n{reussis}/{total} contrôles conformes")
    if reussis != total:
        print("La garde ne fait pas ce qu'elle annonce : ne pas basculer la production.")
        return 1
    print("La garde refuse chacun des cas qu'elle prétend couvrir.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
