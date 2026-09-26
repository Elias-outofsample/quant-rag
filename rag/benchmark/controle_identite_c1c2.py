"""Contrôle d'identité C1/C2 — §4 du pré-enregistrement. **Il échoue, le chantier s'arrête.**

Les deux candidates ne doivent différer que par une ligne ``Page: n`` dans ``retrieval_text``.
Ce module transforme cette affirmation en vérification, sur les **artefacts** et non sur
l'intention. Lecture seule : aucun appel LLM, aucune écriture hors de son propre verdict
``identite-c1c2-<sig>.json``. Qdrant n'est **lu** que pour I4 et I5, jamais écrit.

Les neuf assertions, et ce que chacune peut réellement prouver aujourd'hui
--------------------------------------------------------------------------
Cinq d'entre elles sont **structurelles** : C1 et C2 partagent physiquement le même fichier,
et le contrôle vérifie qu'il n'en existe qu'un. C'est une garantie plus forte qu'une
comparaison, parce qu'elle ne peut pas être contournée par une régénération.

===== =========================================== ===========================================
I1    même ensemble de ``chunk_id``               structurel — un seul ``chunks.jsonl``/doc
I2    mêmes champs de chunk                       structurel — idem
I3    offsets identiques                          structurel — un seul ``offsets.jsonl``/doc
I4    payloads Qdrant identiques                  **calculé** dès que les deux collections
                                                  existent — scroll et comparaison champ à champ
I5    texte lexical identique (BM25)              **calculé** — ``lexical_text`` ne porte pas la
                                                  page, les deux empreintes doivent coïncider
I6    or identique                                structurel — un seul ``gold-remap-<sig>``
I7    overlay des tableaux identique              structurel — un seul fichier
I8    la SEULE différence de ``retrieval_text``   **calculé, chunk par chunk**
      est une ligne ``^Page: \\d+$`` insérée
I9    ``retrieval_text_sha256`` diffère partout   **calculé**, et recoupé avec le manifeste
===== =========================================== ===========================================

I8 est le contrôle qui porte tous les autres : il ne compare pas des résumés, il reconstruit
les deux chaînes et exige que leur différence soit **exactement** une ligne, à la bonne
position, le reste égal caractère à caractère. Une différence d'espace le fait échouer.

    .venv/bin/python rag/benchmark/controle_identite_c1c2.py --candidat 8d4ee77f1f
    .venv/bin/python rag/benchmark/controle_identite_c1c2.py --candidat 8d4ee77f1f --troncature
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import pathlib
import re
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "rag"))
sys.path.insert(0, str(ROOT / "rag" / "ingestion"))

import corpus_overlay  # noqa: E402
from rechunk_corpus import retrieval_text, titres_plonges  # noqa: E402  — un seul domicile

PROCESSED = ROOT / "data" / "processed"
LIGNE_PAGE = re.compile(r"^Page: \d+$")
MAX_LENGTH = 1024


def _candidat(signature: str) -> pathlib.Path:
    chemin = PROCESSED / f"candidat-{signature}"
    if not chemin.is_dir():
        sys.exit(f"corpus candidat introuvable : {chemin}")
    return chemin


def _titres() -> dict[str, str]:
    """Le titre plongé, par la MÊME fonction que la construction — c'est le seul moyen pour
    que I9 vérifie quelque chose. Le premier jet de ce contrôle avait sa propre recette et
    a signalé 4 169 désaccords qui étaient les siens, pas ceux du corpus."""
    titres, _ = titres_plonges()
    return titres


def _chunks_servis(racine: pathlib.Path) -> dict[str, str]:
    return {r["chunk_id"]: r["text"]
            for r in (json.loads(l) for l in (racine / "chunks-servis.jsonl").open(encoding="utf-8") if l.strip())}


def _eligibles(racine: pathlib.Path):
    """Les chunks actifs éligibles, dans l'ordre du corpus candidat."""
    servis = _chunks_servis(racine)
    for row in (json.loads(l) for l in (racine / "rows.jsonl").open(encoding="utf-8") if l.strip()):
        chunk = row["chunk"]
        yield chunk, servis[chunk["chunk_id"]]


def _difference_est_une_ligne_page(c1: str, c2: str) -> str | None:
    """``None`` si la différence est exactement une ligne ``Page: n``, sinon le motif d'échec."""
    l1, l2 = c1.split("\n"), c2.split("\n")
    if len(l2) != len(l1) + 1:
        return f"{len(l1)} lignes -> {len(l2)}"
    # la ligne insérée est la première position où les deux listes divergent
    i = next((k for k, (a, b) in enumerate(zip(l1, l2)) if a != b), len(l1))
    if not LIGNE_PAGE.match(l2[i]):
        return f"ligne insérée non conforme : {l2[i]!r}"
    if l1[:i] != l2[:i] or l1[i:] != l2[i + 1:]:
        return "le reste des deux chaînes n'est pas identique"
    if i != 2:
        return f"ligne insérée en position {i}, attendue en 2 (après Document: et Path:)"
    return None


def _payloads_collection(client, nom: str) -> dict[str, dict]:
    out, offset = {}, None
    while True:
        lot, offset = client.scroll(nom, limit=4096, offset=offset, with_payload=True, with_vectors=False)
        for point in lot:
            out[point.payload["chunk_id"]] = point.payload
        if offset is None:
            break
    return out


def _bm25_sha(payloads: dict[str, dict]) -> str:
    """L'empreinte de ce que BM25 indexerait — sans écrire d'index.

    ``retrieval.lexical.lexical_text`` compose ``titre ‖ title_path ‖ texte`` (plus la légende
    et les en-têtes pour un tableau). **La page n'y entre pas** : C1 et C2 doivent donc rendre
    la même empreinte, et I5 est exactement cette égalité. La calculer sans construire
    l'index évite d'écrire un fichier sous un nom qui pourrait mentir.
    """
    sys.path.insert(0, str(ROOT / "src"))
    from retrieval.lexical import lexical_text, tokenize

    digest = hashlib.sha256()
    for chunk_id in sorted(payloads):
        p = payloads[chunk_id]
        texte = lexical_text({"title": p.get("title")}, p)
        digest.update(chunk_id.encode()); digest.update(b"\x00")
        digest.update(" ".join(tokenize(texte)).encode()); digest.update(b"\x01")
    return digest.hexdigest()[:16]


def _i4_i5(signature: str) -> dict:
    """I4 (payloads) et I5 (texte lexical) — dès que les deux collections existent."""
    sys.path.insert(0, str(ROOT / "rag" / "ingestion"))
    from build_collection_candidat import nom_collection

    import quant_rag

    noms = {bras: nom_collection(signature, bras) for bras in ("c1", "c2")}
    client = quant_rag.client()
    absentes = [b for b, n in noms.items() if not client.collection_exists(n)]
    if absentes:
        return {"lignes": [("I4     payloads Qdrant identiques", f"EN ATTENTE (collections {absentes})"),
                           ("I5     texte lexical identique (BM25)", "EN ATTENTE")],
                "detail": {"collections_absentes": absentes}}

    charges = {bras: _payloads_collection(client, nom) for bras, nom in noms.items()}
    c1, c2 = charges["c1"], charges["c2"]
    memes_cles = set(c1) == set(c2)
    differents = [cid for cid in c1 if memes_cles and c1[cid] != c2[cid]]
    ok4 = memes_cles and not differents
    sha1, sha2 = _bm25_sha(c1), _bm25_sha(c2)
    ok5 = sha1 == sha2
    return {
        "lignes": [
            ("I4     payloads Qdrant identiques",
             f"{len(c1)} points  {'OK' if ok4 else 'ÉCHEC — ' + str(len(differents)) + ' payloads différents'}"),
            ("I5     texte lexical identique (BM25)",
             f"{sha1}  {'OK' if ok5 else 'ÉCHEC — ' + sha2}")],
        "detail": {"points": len(c1), "payloads_differents": differents[:10],
                   "lexical_sha_c1": sha1, "lexical_sha_c2": sha2, "i4": ok4, "i5": ok5},
    }


def controle(signature: str, troncature: bool = False) -> int:
    racine = _candidat(signature)
    manifeste = json.loads((racine / "manifeste.json").read_text(encoding="utf-8"))
    titres = _titres()
    echecs = 0

    print(f"corpus candidat  {racine.name}   bras déclaré : {manifeste['bras']}\n")

    # ---- structurel : un seul artefact partagé par les deux bras
    docs = sorted(p for p in racine.iterdir() if p.is_dir())
    structurels = [
        ("I1/I2  un seul chunks.jsonl par document", all((d / "chunks.jsonl").is_file() for d in docs)),
        ("I3     un seul offsets.jsonl par document", all((d / "offsets.jsonl").is_file() for d in docs)),
        ("I7     un seul overlay de tableaux", (racine / "tables-markdown-v1.json").is_file()),
        ("I6     un seul gold-remap", (HERE / f"gold-remap-{signature}.json").is_file()),
        ("       aucun corpus par bras (candidat-*-c1 / -c2)",
         not list(PROCESSED.glob(f"candidat-{signature}-*"))),
    ]
    for libelle, ok in structurels:
        print(f"  {libelle:<52} {'OK' if ok else 'ÉCHEC'}")
        echecs += not ok

    resultats_i45 = _i4_i5(signature)
    for libelle, etat in resultats_i45["lignes"]:
        print(f"  {libelle:<52} {etat}")
        echecs += etat.startswith("ÉCHEC")
    print()

    # ---- calculé : I8 et I9
    declares = {r["chunk_id"]: r for r in
                (json.loads(l) for l in (racine / "retrieval-text.jsonl").open(encoding="utf-8") if l.strip())}
    compte = collections.Counter()
    motifs = collections.Counter()
    exemples: list[tuple[str, str]] = []
    longueurs_c1: list[int] = []
    longueurs_c2: list[int] = []

    for chunk, servi in _eligibles(racine):
        compte["chunks"] += 1
        titre = titres[chunk["document_id"]]
        c1 = retrieval_text(titre, chunk.get("title_path"), chunk.get("page_start"), servi, avec_page=False)
        c2 = retrieval_text(titre, chunk.get("title_path"), chunk.get("page_start"), servi, avec_page=True)
        motif = _difference_est_une_ligne_page(c1, c2)
        if motif:
            compte["I8_echec"] += 1
            motifs[motif.split(":")[0]] += 1
            if len(exemples) < 5:
                exemples.append((chunk["chunk_id"], motif))
        else:
            compte["I8_ok"] += 1
        d = declares.get(chunk["chunk_id"])
        if d is None:
            compte["I9_absent"] += 1
        elif (d["c1_sha256"] != hashlib.sha256(c1.encode()).hexdigest()
              or d["c2_sha256"] != hashlib.sha256(c2.encode()).hexdigest()):
            compte["I9_desaccord"] += 1
        elif d["c1_sha256"] == d["c2_sha256"]:
            compte["I9_identiques"] += 1
        else:
            compte["I9_ok"] += 1
        if troncature:
            longueurs_c1.append(len(c1))
            longueurs_c2.append(len(c2))

    n = compte["chunks"]
    ok8 = compte["I8_echec"] == 0
    ok9 = compte["I9_ok"] == n
    print(f"  {'I8     différence = une ligne Page: n':<52} "
          f"{compte['I8_ok']}/{n}  {'OK' if ok8 else 'ÉCHEC'}")
    print(f"  {'I9     sha256 déclarés justes et distincts':<52} "
          f"{compte['I9_ok']}/{n}  {'OK' if ok9 else 'ÉCHEC'}")
    echecs += (not ok8) + (not ok9)
    for cle in ("I9_absent", "I9_desaccord", "I9_identiques"):
        if compte[cle]:
            print(f"      {cle} : {compte[cle]}")
    for cid, motif in exemples:
        print(f"      {cid}  {motif}")

    if troncature:
        _troncature(racine, titres)

    verdict = {
        "signature_candidate": signature,
        "structurels": {libelle.strip(): bool(ok) for libelle, ok in structurels},
        "i4_i5": resultats_i45["detail"],
        "I8_ok": compte["I8_ok"], "I8_echec": compte["I8_echec"],
        "I9_ok": compte["I9_ok"], "chunks": n,
        "toutes_assertions_disponibles_passent": not echecs,
        "echecs": echecs,
    }
    (HERE / f"identite-c1c2-{signature}.json").write_text(
        json.dumps(verdict, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n  {'VERDICT':<52} {'TOUTES LES ASSERTIONS DISPONIBLES PASSENT' if not echecs else str(echecs) + ' ÉCHEC(S) — LE CHANTIER S ARRÊTE'}")
    return 1 if echecs else 0


def _troncature(racine: pathlib.Path, titres: dict[str, str]) -> None:
    """La nuisance du §10, **recalculée sur le candidat** comme le pré-enregistrement l'exige."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-Embedding-0.6B")
    c1_jetons, c2_jetons = [], []
    for chunk, servi in _eligibles(racine):
        titre = titres[chunk["document_id"]]
        corps = retrieval_text(titre, chunk.get("title_path"), chunk.get("page_start"), servi, avec_page=False)
        avec = retrieval_text(titre, chunk.get("title_path"), chunk.get("page_start"), servi, avec_page=True)
        c1_jetons.append(len(tok.encode(corps, add_special_tokens=False)))
        c2_jetons.append(len(tok.encode(avec, add_special_tokens=False)))
    n = len(c1_jetons)
    t1 = sum(1 for x in c1_jetons if x > MAX_LENGTH)
    t2 = sum(1 for x in c2_jetons if x > MAX_LENGTH)
    bascule = sum(1 for a, b in zip(c1_jetons, c2_jetons) if a <= MAX_LENGTH < b)
    surcout = sorted(b - a for a, b in zip(c1_jetons, c2_jetons))
    print(f"\n  troncature recalculée sur le candidat ({n} chunks, MAX_LENGTH {MAX_LENGTH})")
    print(f"    C1 tronqués                                  {t1:>6}  ({100.0 * t1 / n:5.2f} %)")
    print(f"    C2 tronqués                                  {t2:>6}  ({100.0 * t2 / n:5.2f} %)")
    print(f"    chunks BASCULANT sous la troncature          {bascule:>6}  ({100.0 * bascule / n:5.2f} %)")
    print(f"    surcoût de « Page: n » : médiane {surcout[n // 2]} jetons · max {surcout[-1]}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--candidat", required=True, metavar="SIG", help="signature du corpus candidat")
    p.add_argument("--troncature", action="store_true",
                   help="recalcule la nuisance du §10 sur le candidat (charge le tokenizer)")
    a = p.parse_args()
    sys.exit(controle(a.candidat, a.troncature))


if __name__ == "__main__":
    main()
