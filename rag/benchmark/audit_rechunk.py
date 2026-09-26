"""Audit du re-découpage — reconnaissance en **lecture seule**, signature ``5530cba145``.

Ce n'est pas le chantier de représentation : c'est la mesure de faisabilité qui doit le
précéder. Rien n'est créé, rien n'est ré-embarqué, rien n'est écrit en production, aucun
appel LLM, aucun banc rejoué, **et Qdrant n'est jamais ouvert** — la collection embarquée
est mono-processus et le serveur MCP la tient.

La matière première est ``data/processed/ingested/<dossier>/blocks.jsonl`` : les blocs
MinerU, versionnés, dont le ``sha256`` figure au registre.

Les cinq questions, et pourquoi elles se posent avant le chantier
----------------------------------------------------------------
1. **Peut-on re-découper sans rejouer MinerU ?** Si oui, le chantier coûte des secondes de
   calcul, pas des heures de GPU, et il devient annulable par construction — le corpus
   servi n'est pas la source, il en est une dérivation reproductible.
2. **Qui possède le défaut d'étiquette du §3.3 de ``docs/STRATEGIE.md`` ?** L'audit du
   6 septembre a mesuré sa prévalence (59,4 % de la population vérifiable) sans en nommer
   la cause. Cette étape-ci la nomme et la chiffre : ``--etiquettes``.
3. **Qui possède la coupure de phrase ?** ``RAPPORT-AUDIT-DECOUPE-2026-09-05.md`` §0.2
   l'attribue au chunker (« une frontière était disponible dans 99 % des cas »).
   ``--frontieres`` mesure la part déjà présente dans le **bloc** que le chunker reçoit.
4. **Des offsets exacts sont-ils dérivables ?** ``docs/STRATEGIE.md`` §2 et §3.1 en font la
   seule voie propre pour lever un jour la borne des 1 200 caractères. ``--offsets`` mesure
   sur quelle part du corpus ils sont exacts, et nomme les exceptions.
5. **Que coûte un préfixe plus long ?** Le texte embarqué est tronqué à 1 024 jetons.
   ``--troncature`` mesure ce qui est déjà perdu et ce qu'un préfixe plus gros perdrait.

Deux étapes de plus servent le protocole plutôt que le diagnostic : ``--identifiants``
(les pièges d'identité au re-découpage) et ``--ancrage`` (les 182 ancres d'or survivent-elles
au candidat ? — contrôle de survie de l'**instrument**, pas mesure de performance).

Les deux états comparés, et pourquoi le chunker est recopié ici
---------------------------------------------------------------
``variante_make_chunks`` est une copie de ``src/parsing/canonical_chunker.make_chunks`` portant
**deux** interrupteurs, qui décrivent les deux états entre lesquels tout ce module compare :

``REFERENCE = (False, False)``  le corpus versionné ``5530cba145``, tel qu'il est servi ;
``CANDIDAT  = (True,  True)``   la **correction commune** — c'est-à-dire C1 du
                                pré-enregistrement : ``flush()`` avant la mise à jour d'état
                                sur un titre, et le chemin publié tronqué à la section quand
                                l'ancêtre la contredit.

**Depuis le 6 septembre 2026, la production porte la correction** : le dépôt *est* C1, et cette
copie sert à mesurer l'écart avec l'état d'avant, qui n'existe plus qu'en tant que fichiers
versionnés. ``--reproduction`` porte donc **deux** gardes, et elles ne disent pas la même
chose : **A**, ``REFERENCE`` reproduit ``chunks.jsonl`` à l'octet sur les 421 documents ;
**B**, ``CANDIDAT`` est identique à ``canonical_chunker.make_chunks``. Sans B, cette copie
dériverait de la production sans un bruit et ses chiffres cesseraient de décrire ce qui sera
mesuré. La règle de troncature, elle, n'est pas dupliquée du tout : elle a un seul domicile,
``canonical_chunker.chemin_publie``, que ce module importe.

Ce que cet audit ne dit pas
---------------------------
- Il ne mesure **aucun** classement, aucun nDCG, aucune couverture de réponse. Il ne peut
  donc ni valider ni invalider le chantier : il en mesure la faisabilité et le coût.
- Les détecteurs de coupure de phrase sont ceux de ``audit_decoupage`` et gardent leur
  précision publiée (3/4 au début, 2/4 en fin) ; les comptes absolus se lisent avec elle.
- **Les tableaux sont ici en HTML brut**, alors que le corpus servi les voit en Markdown
  (``tables/tables-markdown-v1.json``, overlay indexé par ``chunk_id``). Un chiffre de
  frontière calculé ici sur les tableaux n'est **pas** comparable à ceux du corpus servi ;
  ``--frontieres`` sépare donc toujours tableaux et non-tableaux.

    .venv/bin/python rag/benchmark/audit_rechunk.py --reproduction
    .venv/bin/python rag/benchmark/audit_rechunk.py --etiquettes
    .venv/bin/python rag/benchmark/audit_rechunk.py --frontieres
    .venv/bin/python rag/benchmark/audit_rechunk.py --offsets
    .venv/bin/python rag/benchmark/audit_rechunk.py --identifiants
    .venv/bin/python rag/benchmark/audit_rechunk.py --chemin
    .venv/bin/python rag/benchmark/audit_rechunk.py --titres-perdus
    .venv/bin/python rag/benchmark/audit_rechunk.py --troncature      # charge le tokenizer
    .venv/bin/python rag/benchmark/audit_rechunk.py --ancrage
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import pathlib
import random
import re
import statistics
import sys

HERE = pathlib.Path(__file__).resolve().parent
RACINE = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(RACINE))

import audit_contexte_structure as acs  # noqa: E402  — le classeur d'étiquettes A/A-/B/C
import audit_decoupage as ad  # noqa: E402  — les détecteurs de coupure de phrase
import src.parsing.canonical_chunker as production  # noqa: E402  — le chunker servi
from src.parsing.canonical_chunker import chemin_publie  # noqa: E402  — la règle a UN domicile
from src.parsing.mineru_adapter import page_roles  # noqa: E402
from src.parsing.models import CanonicalBlock, CanonicalChunk, CanonicalDocument  # noqa: E402

INGESTED = RACINE / "data" / "processed" / "ingested"
SEPARATEUR = "\n\n"
#: Longueur maximale du texte embarqué, en jetons (``rag/quant_rag.MAX_LENGTH``).
MAX_LENGTH = 1024
GRAINE = 20260906


# ------------------------------------------------------------------ lecture des blocs


def documents() -> list[pathlib.Path]:
    return sorted(p for p in INGESTED.iterdir() if p.is_dir())


def charger(dossier: pathlib.Path) -> tuple[CanonicalDocument, list[CanonicalBlock], list[dict]]:
    """``document.json`` + ``blocks.jsonl`` reconstitués, et ``chunks.jsonl`` tel quel."""
    d = json.loads((dossier / "document.json").read_text(encoding="utf-8"))
    doc = CanonicalDocument(
        document_id=d["document_id"], source_path=d.get("source_path", ""),
        filename=d.get("filename", ""), sha256=d.get("sha256", ""), title=d.get("title"),
        subtitle=d.get("subtitle"), authors=d.get("authors") or [], editors=d.get("editors") or [],
        publication_year=d.get("publication_year"), edition=d.get("edition"),
        publisher=d.get("publisher"), page_count=d.get("page_count", 0),
        parser_backend=d.get("parser_backend", ""), parser_version=d.get("parser_version", ""),
        metadata=d.get("metadata") or {})
    blocs = []
    with (dossier / "blocks.jsonl").open(encoding="utf-8") as f:
        for ligne in f:
            b = json.loads(ligne)
            blocs.append(CanonicalBlock(
                block_id=b["block_id"], document_id=b["document_id"], page_idx=b["page_idx"],
                block_type=b["block_type"], text=b["text"], bbox=b.get("bbox"),
                heading_level=b.get("heading_level"), parent_heading=b.get("parent_heading"),
                content_type=b["content_type"], raw_text=b.get("raw_text"), part=b.get("part"),
                chapter=b.get("chapter"), section=b.get("section"), subsection=b.get("subsection"),
                image_refs=b.get("image_refs") or [], metadata=b.get("metadata") or {}))
    stockes = [json.loads(l) for l in (dossier / "chunks.jsonl").open(encoding="utf-8") if l.strip()]
    return doc, blocs, stockes


# ------------------------------------------------------------------ le chunker candidat


def _jetons(s: str) -> int:
    return len(re.findall(r"\w+|[^\w\s]", s, flags=re.UNICODE))


def _cid(*v) -> str:
    return "chunk-" + hashlib.sha1("\x1f".join(map(str, v)).encode()).hexdigest()[:16]


def _pid(*v) -> str:
    return "parent-" + hashlib.sha1("\x1f".join(map(str, v)).encode()).hexdigest()[:16]


def variante_make_chunks(doc, blocks, target=800, hard_max=1200,
                         ordre_flush=False, chemin_tronque=False):
    """Copie de ``canonical_chunker.make_chunks`` avec **deux** interrupteurs.

    Ils décrivent les deux états entre lesquels ce module compare :

    ``(False, False)``  **la référence versionnée** — le corpus `5530cba145` tel qu'il est
                        servi. ``--reproduction`` exige qu'elle reproduise ``chunks.jsonl``
                        à l'octet sur les 421 documents.
    ``(True, True)``    **la correction commune**, c'est-à-dire C1 — et ``--reproduction``
                        exige qu'elle soit identique à ``canonical_chunker.make_chunks``,
                        qui la porte depuis le 6 septembre 2026. Sans ce second contrôle,
                        cette copie dériverait de la production sans un bruit.

    ``ordre_flush``     ``flush()`` avant la mise à jour d'état sur un titre.
    ``chemin_tronque``  le chemin publié s'arrête à la section quand l'ancêtre la contredit
                        (``canonical_chunker.chemin_publie``, dont c'est le seul domicile).
    """
    roles = page_roles(blocks, doc.page_count)
    state = {"part": None, "chapter": None, "section": None, "subsection": None,
             "chapter_key": None, "chapter_title_key": None, "document_region": "body"}
    groups: list = []
    current: list = []

    def flush():
        nonlocal current
        if current:
            groups.append((dict(state), list(current)))
            current = []

    expanded: list = []
    for b in blocks:
        if b.content_type == "table" and _jetons(b.text) > hard_max:
            rows = re.findall(r"<tr.*?</tr>", b.text, flags=re.I | re.S)
            if rows:
                prefix = b.text.split("<table", 1)[0].strip()
                acc, n = prefix, 0
                table_parent = _pid(doc.document_id, "table", b.block_id)

                def _part(texte: str, index: int) -> CanonicalBlock:
                    return CanonicalBlock(
                        block_id=f"{b.block_id}-part{index}", document_id=b.document_id,
                        page_idx=b.page_idx, block_type="table", text=texte, bbox=b.bbox,
                        heading_level=None, parent_heading=b.parent_heading, content_type="table",
                        raw_text=texte, part=b.part, chapter=b.chapter, section=b.section,
                        subsection=b.subsection, image_refs=list(b.image_refs),
                        metadata={**b.metadata, "table_parent_id": table_parent,
                                  "table_full_text": b.text, "table_part": index})

                for row in rows:
                    candidat = acc + "\n" + row
                    if _jetons(candidat) > target and acc != prefix:
                        expanded.append(_part(acc, n))
                        n += 1
                        acc = prefix + "\n" + row
                    else:
                        acc = candidat
                if acc != prefix:
                    expanded.append(_part(acc, n))
                continue
        if b.content_type in ("text", "heading") and _jetons(b.text) > hard_max:
            mots = b.text.split()
            taille = min(500, target)
            for n in range(0, len(mots), taille):
                morceau = " ".join(mots[n:n + taille])
                expanded.append(CanonicalBlock(
                    block_id=f"{b.block_id}-part{n // taille}", document_id=b.document_id,
                    page_idx=b.page_idx, block_type=b.block_type, text=morceau, bbox=b.bbox,
                    heading_level=b.heading_level, parent_heading=b.parent_heading,
                    content_type=b.content_type, raw_text=morceau, part=b.part, chapter=b.chapter,
                    section=b.section, subsection=b.subsection, image_refs=list(b.image_refs),
                    metadata=dict(b.metadata)))
        else:
            expanded.append(b)

    for b in expanded:
        if b.metadata.get("document_region") in ("references", "bibliography", "backmatter"):
            region = b.metadata["document_region"]
            if state["document_region"] != region:
                flush()
                state.update(part=None, chapter=None, section=None, subsection=None,
                             chapter_key=None, chapter_title_key=None, document_region=region)
            if b.content_type in ("header", "footer") or b.block_type in ("page_number", "page_footnote"):
                continue
            current.append(b)
            if _jetons("\n".join(x.text for x in current)) >= target:
                flush()
            continue
        state["document_region"] = "body"
        if not roles.get(b.page_idx, ("content", True))[1]:
            flush(); current.append(b); flush(); continue
        if b.content_type in ("header", "footer") or b.block_type in ("page_number", "page_footnote"):
            continue
        if b.content_type == "table":
            flush(); current.append(b); flush(); continue
        if b.content_type in ("heading", "title"):
            if ordre_flush:
                flush()                       # <<< la correction : le tampon garde son étiquette
            if any((b.part, b.chapter, b.section, b.subsection)):
                state.update(part=b.part, chapter=b.chapter, section=b.section, subsection=b.subsection)
                state["chapter_key"] = b.metadata.get("structural_identity_key", state.get("chapter_key"))
                state["chapter_title_key"] = b.metadata.get("structural_title_key", state.get("chapter_title_key"))
            else:
                niveau = b.heading_level or 2
                if re.match(r"(?i)^(chapter|appendix)\b", b.text):
                    state.update(chapter=b.text, section=None, subsection=None)
                elif niveau <= 1:
                    if re.match(r"(?i)^part\b", b.text):
                        state.update(part=b.text, chapter=None, section=None, subsection=None)
                    else:
                        state.update(chapter=b.text, section=None, subsection=None)
                elif niveau == 2:
                    state["section"] = b.text
                    state["subsection"] = None
                else:
                    state["subsection"] = b.text
            if not ordre_flush:
                flush()
            current.append(b)
            continue
        if (current and _jetons("\n".join(x.text for x in current)) + _jetons(b.text) > target
                and b.content_type not in ("table", "equation")):
            flush()
        current.append(b)
        if _jetons("\n".join(x.text for x in current)) >= target:
            flush()
    flush()

    chunks: list[CanonicalChunk] = []
    for i, (st, bs) in enumerate(groups):
        text = "\n\n".join(b.text for b in bs if b.text).strip()
        if not text:
            continue
        pages = [b.page_idx for b in bs]
        types = {b.content_type for b in bs}
        ct = next(iter(types)) if len(types) == 1 else "mixed"
        eligible = (all(roles.get(p, ("content", True))[1] for p in pages)
                    and all(b.metadata.get("document_region", "body")
                            not in ("references", "bibliography", "backmatter", "index") for b in bs))
        table_parent = next((b.metadata.get("table_parent_id") for b in bs if b.metadata.get("table_parent_id")), None)
        chapter_key = st.get("chapter_key") or st.get("chapter")
        parent = (table_parent or _pid(doc.document_id, "chapter", chapter_key) if st.get("chapter")
                  else table_parent or _pid(doc.document_id, "section", st.get("part"), st.get("section")) if st.get("section")
                  else table_parent or _pid(doc.document_id, "part", st.get("part")) if st.get("part")
                  else table_parent or _pid(doc.document_id, "document"))
        if chemin_tronque:
            title_path, chapitre_ecarte = chemin_publie(
                st.get("part"), st.get("chapter"), st.get("section"), st.get("subsection"))
        else:
            title_path = " > ".join(x for x in (st.get("part"), st.get("chapter"),
                                                st.get("section"), st.get("subsection")) if x)
            chapitre_ecarte = None
        chunks.append(CanonicalChunk(
            chunk_id=_cid(doc.document_id, i, text), document_id=doc.document_id, parent_id=parent,
            previous_chunk_id=None, next_chunk_id=None, page_start=min(pages), page_end=max(pages),
            part=st.get("part"), chapter=st.get("chapter"), section=st.get("section"),
            subsection=st.get("subsection"), text=text, title_path=title_path or None,
            content_type=ct, token_count=_jetons(text), rag_eligible=eligible,
            image_refs=sorted({r for b in bs for r in b.image_refs}),
            metadata={"block_ids": [b.block_id for b in bs],
                      "page_roles": {str(p): roles[p][0] for p in sorted(set(pages))},
                      "table_parent_id": table_parent,
                      "structural_identity_key": chapter_key if st.get("chapter") else None,
                      "structural_title_key": st.get("chapter_title_key") if st.get("chapter") else None,
                      "canonical_structural_key": chapter_key if st.get("chapter") else None,
                      "document_region": bs[0].metadata.get("document_region", "body"),
                      "chapitre_ecarte": chapitre_ecarte}))

    fusionnes: list[CanonicalChunk] = []
    for c in chunks:
        p = fusionnes[-1] if fusionnes else None
        if (p and c.rag_eligible and p.rag_eligible and c.content_type in ("text", "mixed")
                and p.content_type in ("text", "mixed") and c.parent_id == p.parent_id
                and _jetons(p.text + "\n\n" + c.text) <= target):
            p.text += "\n\n" + c.text
            p.page_end = max(p.page_end, c.page_end)
            p.token_count = _jetons(p.text)
            p.content_type = "mixed" if p.content_type != c.content_type else p.content_type
            p.image_refs = sorted(set(p.image_refs + c.image_refs))
            p.metadata["block_ids"] += c.metadata.get("block_ids", [])
        else:
            fusionnes.append(c)
    for i, c in enumerate(fusionnes):
        c.previous_chunk_id = fusionnes[i - 1].chunk_id if i else None
        c.next_chunk_id = fusionnes[i + 1].chunk_id if i + 1 < len(fusionnes) else None
    return fusionnes, roles


#: Les deux états comparés par ce module : (ordre_flush, chemin_tronque).
REFERENCE = (False, False)
CANDIDAT = (True, True)


def corpus(etat: tuple[bool, bool], eligibles_seulement: bool = True) -> list[dict]:
    """Le corpus re-découpé, en mémoire, sous forme de dictionnaires."""
    ordre_flush, chemin_tronque = etat
    out = []
    for dossier in documents():
        doc, blocs, _ = charger(dossier)
        cs, _ = variante_make_chunks(doc, blocs, ordre_flush=ordre_flush,
                                     chemin_tronque=chemin_tronque)
        out.extend(c.as_dict() for c in cs if c.rag_eligible or not eligibles_seulement)
    return out


# ------------------------------------------------------------------ 1. reproduction


CHAMPS_COMPARES = ("chunk_id", "text", "part", "chapter", "section", "subsection", "title_path",
                   "page_start", "page_end", "parent_id", "content_type", "rag_eligible",
                   "token_count", "previous_chunk_id", "next_chunk_id")


def _comparer(gauche: list[dict], droite: list[dict]) -> dict | str:
    """Diff champ à champ de deux découpages d'un même document."""
    if len(gauche) != len(droite):
        return f"{len(gauche)} / {len(droite)} chunks"
    diff = collections.Counter()
    for g, d in zip(gauche, droite):
        for champ in CHAMPS_COMPARES:
            if g.get(champ) != d.get(champ):
                diff[champ] += 1
    return dict(diff)


def step_reproduction() -> None:
    """Les deux gardes de cette copie du chunker, et elles ne disent pas la même chose.

    **A — la référence versionnée.** ``(False, False)`` doit reproduire ``chunks.jsonl`` à
    l'octet : c'est ce qui prouve que le corpus servi est une dérivation reproductible des
    blocs, et donc que le re-découpage ne demande ni MinerU ni GPU.

    **B — la dérive.** ``(True, True)`` doit être identique à
    ``canonical_chunker.make_chunks``, qui porte la correction commune depuis le 6 septembre
    2026. Sans ce contrôle, cette copie s'écarterait de la production sans un bruit, et tous
    les chiffres des autres étapes cesseraient de décrire ce qui sera mesuré.
    """
    ok_a, ok_b, echecs = 0, 0, []
    for dossier in documents():
        doc, blocs, stockes = charger(dossier)
        ref, _ = variante_make_chunks(doc, blocs, ordre_flush=REFERENCE[0], chemin_tronque=REFERENCE[1])
        diff_a = _comparer(stockes, [c.as_dict() for c in ref])
        if diff_a:
            echecs.append(("A", dossier.name, diff_a))
        else:
            ok_a += 1
        cand, _ = variante_make_chunks(doc, blocs, ordre_flush=CANDIDAT[0], chemin_tronque=CANDIDAT[1])
        prod, _ = production.make_chunks(doc, blocs)
        diff_b = _comparer([c.as_dict() for c in prod], [c.as_dict() for c in cand])
        if diff_b:
            echecs.append(("B", dossier.name, diff_b))
        else:
            ok_b += 1
    n = len(documents())
    print(f"documents                                            {n:>6}")
    print(f"  A  référence (False,False) == chunks.jsonl versionné {ok_a:>6}  ({100.0 * ok_a / n:5.1f} %)")
    print(f"  B  candidat (True,True) == chunker de production     {ok_b:>6}  ({100.0 * ok_b / n:5.1f} %)")
    for garde, nom, detail in echecs[:10]:
        print(f"    ÉCHEC {garde}  {nom}  {detail}")
    if echecs:
        print("\n  A échoue : les blocs ne suffisent plus, ou la référence a bougé.\n"
              "  B échoue : cette copie a dérivé du chunker de production.\n"
              "  Dans les deux cas, tout chiffre des autres étapes est à jeter jusqu'à correction.")


# ------------------------------------------------------------------ 2. étiquettes


def _stats_etiquettes(chunks: list[dict]) -> collections.Counter:
    c = collections.Counter()
    for ch in chunks:
        v = acs.classer(ch)
        if v is None:
            c["non vérifiable"] += 1
            continue
        c[v[0]] += 1
        if v[0] == "C":
            c["C/" + v[1].get("sens", "?")] += 1
    return c


def step_etiquettes() -> None:
    for etat, nom in ((REFERENCE, "RÉFÉRENCE — corpus versionné 5530cba145"),
                      (CANDIDAT, "CANDIDAT C1 — correction commune")):
        chunks = corpus(etat)
        c = _stats_etiquettes(chunks)
        verif = c["A"] + c["A-"] + c["B"] + c["C"]
        tailles = sorted(len(x["text"]) for x in chunks)
        mediane = tailles[len(tailles) // 2]
        print(f"\n  {nom}")
        print(f"    chunks rag_eligible                          {len(chunks):>6}")
        print(f"    caractères : médiane {mediane}  ·  < 250 car. "
              f"{sum(1 for t in tailles if t < 250)}  ·  > 4 000 car. {sum(1 for t in tailles if t > 4000)}")
        print(f"    POPULATION VÉRIFIABLE                        {verif:>6}  "
              f"({100.0 * verif / len(chunks):4.1f} % du corpus)")
        for cle, libelle in (("A", "s'ouvre sur le titre étiqueté"),
                             ("A-", "étiquette = 1er titre, ouverture avant"),
                             ("B", "étiquette plus loin dans le chunk"),
                             ("C", "étiquette absente du chunk")):
            print(f"      {cle:<3} {libelle:<40} {c[cle]:>6}  ({100.0 * c[cle] / verif:5.1f} %)")
        for sens in ("en avance", "en retard", "intercalée", "illisible"):
            if c["C/" + sens]:
                print(f"            · {sens:<34} {c['C/' + sens]:>6}")
        faux = c["A-"] + c["B"] + c["C"]
        print(f"      NE s'ouvre PAS sur sa section             {faux:>6}  ({100.0 * faux / verif:5.1f} %)")


# ------------------------------------------------------------------ 3. frontières


def step_frontieres() -> None:
    """Qui possède la coupure de phrase : le bloc reçu, ou le chunker ?"""
    c = collections.Counter()
    for dossier in documents():
        doc, blocs, stockes = charger(dossier)
        par_id = {b.block_id: b for b in blocs}
        for b in blocs:
            if b.content_type != "text":
                continue
            c["blocs_prose"] += 1
            faux = {"content_type": "text", "text": b.text}
            c["bloc_coupure_debut"] += ad.est_coupure_debut(faux)
            c["bloc_coupure_fin"] += ad.est_coupure_fin(faux)
        for ch in stockes:
            if not ch.get("rag_eligible"):
                continue
            table = ch["content_type"] == "table"
            c["chunks_table" if table else "chunks_autre"] += 1
            if ad.est_coupure_fin(ch):
                c["cf_table" if table else "cf_autre"] += 1
            if not ad.est_coupure_debut(ch):
                continue
            c["cd_table" if table else "cd_autre"] += 1
            c["cd"] += 1
            ids = (ch.get("metadata") or {}).get("block_ids") or []
            premier = par_id.get(ids[0]) if ids else None
            if premier is None:
                c["cd_bloc_synthetique"] += 1
            elif ad.est_coupure_debut({"content_type": premier.content_type, "text": premier.text}):
                c["cd_deja_dans_le_bloc"] += 1
            else:
                c["cd_creee_par_le_chunker"] += 1

    print(f"blocs de prose                                    {c['blocs_prose']:>6}")
    print(f"  commençant en pleine phrase                     {c['bloc_coupure_debut']:>6}  "
          f"({100.0 * c['bloc_coupure_debut'] / c['blocs_prose']:5.1f} %)")
    print(f"  finissant en pleine phrase                      {c['bloc_coupure_fin']:>6}  "
          f"({100.0 * c['bloc_coupure_fin'] / c['blocs_prose']:5.1f} %)")
    print()
    print(f"chunks servis, hors tableaux                      {c['chunks_autre']:>6}")
    print(f"  coupure de début                                {c['cd_autre']:>6}  "
          f"({100.0 * c['cd_autre'] / c['chunks_autre']:5.1f} %)")
    print(f"  coupure de fin                                  {c['cf_autre']:>6}  "
          f"({100.0 * c['cf_autre'] / c['chunks_autre']:5.1f} %)")
    print(f"chunks-tableaux (HTML brut ici, Markdown au service — non comparable) "
          f"{c['chunks_table']:>6}, coupure de fin {c['cf_table']}")
    print()
    print(f"PROPRIÉTAIRE de la coupure de début ({c['cd']} chunks) :")
    for cle, libelle in (("cd_deja_dans_le_bloc", "déjà présente dans le bloc reçu"),
                         ("cd_creee_par_le_chunker", "créée par le chunker"),
                         ("cd_bloc_synthetique", "bloc synthétique (tableau/texte redécoupé)")):
        print(f"  {libelle:<46} {c[cle]:>6}  ({100.0 * c[cle] / c['cd']:5.1f} %)")


# ------------------------------------------------------------------ 4. offsets


def step_offsets() -> None:
    """Le texte d'un chunk est-il exactement reconstructible depuis les blocs ?

    Le texte canonique d'un document est la concaténation ordonnée du texte de **tous** ses
    blocs, séparés par ``\\n\\n``. Un chunk saute des blocs (en-têtes, pieds, numéros de
    page) : son offset n'est donc pas un intervalle mais une **liste** d'intervalles.
    """
    c = collections.Counter()
    for dossier in documents():
        doc, blocs, stockes = charger(dossier)
        spans, morceaux, pos = {}, [], 0
        for b in blocs:
            t = b.text or ""
            spans[b.block_id] = (pos, pos + len(t))
            morceaux.append(t)
            pos += len(t) + len(SEPARATEUR)
        texte = SEPARATEUR.join(morceaux)
        for ch in stockes:
            c["chunks"] += 1
            ids = (ch.get("metadata") or {}).get("block_ids") or []
            if not ids:
                c["sans_blocs"] += 1
                continue
            if any(b not in spans for b in ids):
                c["bloc_synthetique"] += 1
                racines = {b.rsplit("-part", 1)[0] for b in ids if b not in spans}
                if all(r in spans for r in racines):
                    c["synthetique_parent_connu"] += 1
                continue
            recons = SEPARATEUR.join(x for x in (texte[s:e] for s, e in (spans[b] for b in ids)) if x)
            if recons.strip() == ch["text"].strip():
                c["multi_intervalle_exact"] += 1
            elif texte[spans[ids[0]][0]:spans[ids[-1]][1]].strip() == ch["text"].strip():
                c["intervalle_unique"] += 1
            else:
                c["echec"] += 1
    n = c["chunks"]
    print(f"chunks (tous, éligibles ou non)                   {n:>6}")
    for cle, libelle in (("multi_intervalle_exact", "offsets EXACTS (liste d'intervalles)"),
                         ("intervalle_unique", "offsets exacts (intervalle unique)"),
                         ("bloc_synthetique", "bloc synthétique — offset au bloc parent"),
                         ("synthetique_parent_connu", "  dont parent retrouvé"),
                         ("sans_blocs", "sans block_ids"),
                         ("echec", "NON reconstructible")):
        print(f"  {libelle:<44} {c[cle]:>6}  ({100.0 * c[cle] / n:5.2f} %)")


# ------------------------------------------------------------------ 5. identifiants


def step_identifiants() -> None:
    """Les pièges d'identité qu'un re-découpage tend à qui compare deux corpus.

    ``chunk_id = sha1(document_id ‖ rang du groupe ‖ texte)`` est calculé **avant** l'étape
    de fusion, qui modifie le texte sans recalculer l'identifiant. Deux corpus peuvent donc
    partager un ``chunk_id`` et servir des textes différents.
    """
    ref = {c["chunk_id"]: c for c in corpus(REFERENCE, eligibles_seulement=False)}
    cand = {c["chunk_id"]: c for c in corpus(CANDIDAT, eligibles_seulement=False)}
    communs = set(ref) & set(cand)
    texte_different = [k for k in communs if ref[k]["text"] != cand[k]["text"]]
    etiquette_differente = [k for k in communs
                            if (ref[k].get("title_path") or "") != (cand[k].get("title_path") or "")]
    par_texte = collections.defaultdict(list)
    for c in ref.values():
        par_texte[c["text"]].append(c)
    texte_egal_etiquette_changee = sum(
        1 for c in cand.values()
        if any((r.get("title_path") or "") != (c.get("title_path") or "") for r in par_texte.get(c["text"], ())))
    print(f"chunks — référence {len(ref)}   candidat {len(cand)}   identifiants communs {len(communs)}")
    print(f"  MÊME chunk_id, texte DIFFÉRENT                  {len(texte_different):>6}")
    print(f"  MÊME chunk_id, title_path DIFFÉRENT             {len(etiquette_differente):>6}")
    print(f"  texte identique, étiquette changée              {texte_egal_etiquette_changee:>6}")
    print()
    print("  Conséquence, et c'est la seule qui compte pour le chantier : le déclencheur d'un")
    print("  ré-embarquement ne peut être ni le chunk_id ni le texte du chunk. C'est la chaîne")
    print("  réellement plongée — « Document: … / Path: … / texte » — qui doit être hachée.")


# ------------------------------------------------------------------ 6. chemin


def step_chemin(n_exemples: int = 12) -> None:
    """Cohérence interne du fil d'Ariane, et **concentration** par document."""
    for etat, nom in ((REFERENCE, "RÉFÉRENCE"), (CANDIDAT, "CANDIDAT C1")):
        c = collections.Counter()
        par_document = collections.Counter()
        exemples = []
        for ch in corpus(etat):
            if (ch.get("metadata") or {}).get("chapitre_ecarte"):
                c["chemin_tronque"] += 1
            ns = acs.numero_etiquette(ch.get("section") or "")
            nc = acs.numero_etiquette(ch.get("chapter") or "")
            if ns is None or nc is None:
                continue
            c["comparables"] += 1
            if ns.split(".")[0] == nc.split(".")[0]:
                c["accord"] += 1
            else:
                c["incoherent"] += 1
                par_document[ch["document_id"]] += 1
                exemples.append((ch.get("chapter"), ch.get("section")))
        base = c["comparables"]
        print(f"\n  {nom} : chapitre ET section numérotés {base}  ·  "
              f"incohérents {c['incoherent']} ({100.0 * c['incoherent'] / base:.1f} %)")
        print(f"    chemins publiés TRONQUÉS à la section          {c['chemin_tronque']:>6}")
        if etat is CANDIDAT:
            print("    — le compte d'incohérences ne bouge pas, et c'est normal : il se lit sur les")
            print("      champs `chapter`/`section`, qui restent de la PROVENANCE. Ce qui change est")
            print("      ce qui est PUBLIÉ dans `title_path` — donc plongé, servi et indexé.")
        tete = par_document.most_common(3)
        part = sum(n for _, n in tete)
        print(f"    {len(par_document)} documents concernés ; les 3 premiers en portent "
              f"{part} ({100.0 * part / c['incoherent']:.0f} %)")
        if etat is CANDIDAT:
            random.Random(GRAINE).shuffle(exemples)
            for chap, sect in exemples[:n_exemples]:
                print(f"      chapitre {str(chap)[:52]:<52} section {str(sect)[:44]}")


# ------------------------------------------------------------------ 6 bis. titres perdus


def step_titres_perdus(n_exemples: int = 8) -> None:
    """Combien de titres numérotés MinerU a-t-il livrés comme du texte ordinaire ?

    C'est ce qui reste après la correction d'ordre : un chunk classé ``C`` par
    ``--etiquettes`` porte alors une étiquette **juste pour son ouverture** et enjambe une
    frontière que le parseur n'a pas vue. Cette étape en mesure le gisement — et publie
    ses faux positifs, parce qu'un compte sans précision est un chiffre décoratif.
    """
    c = collections.Counter()
    exemples = []
    for dossier in documents():
        _doc, blocs, _ = charger(dossier)
        for b in blocs:
            if b.content_type == "heading":
                c["blocs_heading"] += 1
                continue
            if b.content_type != "text":
                continue
            c["blocs_texte"] += 1
            lignes = [l.strip() for l in (b.text or "").splitlines() if l.strip()]
            if not lignes:
                continue
            titre = len(lignes[0]) <= acs.LONGUEUR_TITRE_MAX and acs.TITRE.match(lignes[0])
            if titre and len(lignes) == 1:
                c["bloc_entier_est_un_titre"] += 1
                exemples.append(lignes[0])
            elif titre:
                c["bloc_commence_par_un_titre"] += 1
            elif any(len(l) <= acs.LONGUEUR_TITRE_MAX and acs.TITRE.match(l) for l in lignes[1:]):
                c["titre_enfoui_dans_le_bloc"] += 1
    total = c["bloc_entier_est_un_titre"] + c["bloc_commence_par_un_titre"] + c["titre_enfoui_dans_le_bloc"]
    print(f"blocs livrés comme titres par MinerU               {c['blocs_heading']:>6}")
    print(f"blocs de prose                                     {c['blocs_texte']:>6}")
    for cle, libelle in (("bloc_entier_est_un_titre", "le bloc entier EST un titre numéroté"),
                         ("bloc_commence_par_un_titre", "le bloc commence par un titre numéroté"),
                         ("titre_enfoui_dans_le_bloc", "un titre numéroté est enfoui dans le bloc")):
        print(f"  {libelle:<44} {c[cle]:>6}")
    print(f"  GISEMENT récupérable                             {total:>6}  "
          f"({100.0 * total / c['blocs_texte']:4.1f} % des blocs de prose)")
    random.Random(GRAINE).shuffle(exemples)
    print("\n  échantillon — la précision de ce détecteur n'est PAS mesurée :")
    for e in exemples[:n_exemples]:
        print(f"    {e[:100]}")


# ------------------------------------------------------------------ 7. troncature


def step_troncature() -> None:
    """Ce que la fenêtre de 1 024 jetons perd déjà, et ce qu'un préfixe plus gros perdrait.

    Lit le corpus **servi** (``rows.jsonl`` + ``imported-rows.jsonl``, overlays appliqués),
    pas le corpus re-découpé : la question porte sur la recette de plongement en vigueur.
    """
    from transformers import AutoTokenizer

    import corpus_overlay

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-Embedding-0.6B")
    titres = {d: e["clean_title"]
              for d, e in json.loads(corpus_overlay.TITLES.read_text(encoding="utf-8"))["documents"].items()}
    sources = (RACINE / "data" / "embeddings" / "ingested-all-qwen3-06b" / "rows.jsonl",
               RACINE / "rag" / "ingestion" / "imported-rows.jsonl")
    vus, longueurs, prefixes = set(), [], []
    for source in sources:
        if not source.exists():
            continue
        for ligne in source.open(encoding="utf-8"):
            if not ligne.strip():
                continue
            r = json.loads(ligne)
            ch, doc = r["chunk"], r["document"]
            if ch["chunk_id"] in vus:
                continue
            texte = corpus_overlay.apply(ch["document_id"], ch["chunk_id"], ch.get("text") or "")
            if texte is None:
                continue
            vus.add(ch["chunk_id"])
            titre = titres.get(ch["document_id"]) or (doc.get("title") or "")
            prefixe = f"Document: {titre}\nPath: {ch.get('title_path') or titre}\n\n"
            longueurs.append(len(tok.encode(prefixe + texte.strip(), add_special_tokens=False)))
            prefixes.append(len(tok.encode(prefixe, add_special_tokens=False)))
    n = len(longueurs)
    longueurs.sort()

    def q(p: float) -> int:
        return longueurs[min(n - 1, int(p * n))]

    print(f"chunks servis                                     {n:>6}")
    print(f"  texte embarqué (jetons) : médiane {q(.5)}  p90 {q(.9)}  p99 {q(.99)}  max {longueurs[-1]}")
    print(f"  préfixe Document+Path   : médiane {statistics.median(prefixes)}  max {max(prefixes)}")
    for marge in (0, 10, 20, 40, 80, 160):
        perdus = sum(1 for x in longueurs if x + marge > MAX_LENGTH)
        libelle = "TRONQUÉS aujourd'hui" if marge == 0 else f"tronqués si le préfixe grossit de {marge:>3} jetons"
        print(f"  {libelle:<48} {perdus:>6}  ({100.0 * perdus / n:5.1f} %)")


# ------------------------------------------------------------------ 8. ancrage


def step_ancrage() -> None:
    """Les 182 ancres d'or survivent-elles au candidat ?

    Contrôle de survie de l'**instrument**. Aucun classement, aucun nDCG : ce chiffre ne
    peut ni valider ni invalider le chantier. Réserve à écrire à côté de tout résultat :
    les tableaux sont ici en HTML brut et les ancres ont été construites sur le texte
    **servi** (Markdown), de sorte que le bras de référence n'est pas le contrôle
    d'identité exact du dépôt — c'est la **différence** entre les deux bras qui se lit.
    """
    import gold_ancrage as ga

    charge = ga.charger_ancres()
    ancres = charge["ancres"]
    docs = {a["document_id"] for a in ancres.values()}
    print(f"{len(ancres)} ancres réparties sur {len(docs)} documents\n")
    for etat, nom in ((REFERENCE, "A — référence (corpus versionné)"),
                      (CANDIDAT, "B — candidat C1 (correction commune)")):
        nouveaux = [c for c in corpus(etat) if c["document_id"] in docs]
        r = ga.resume(ga.ancrer(ancres, nouveaux))
        print(f"  {nom}")
        print(f"    candidats {len(nouveaux)}   n={r['n']}  ok={r['ok']}  partiel={r['partiel']}  "
              f"perdu={r['perdu']}  dispersés={r['disperses']}")
        print(f"    couverture de l'union : min {r['couverture_union_min']} · "
              f"médiane {r['couverture_union_mediane']}\n")


# ------------------------------------------------------------------ CLI


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--reproduction", action="store_true")
    p.add_argument("--etiquettes", action="store_true")
    p.add_argument("--frontieres", action="store_true")
    p.add_argument("--offsets", action="store_true")
    p.add_argument("--identifiants", action="store_true")
    p.add_argument("--chemin", action="store_true")
    p.add_argument("--titres-perdus", action="store_true")
    p.add_argument("--troncature", action="store_true")
    p.add_argument("--ancrage", action="store_true")
    a = p.parse_args()
    etapes = [(a.reproduction, step_reproduction), (a.etiquettes, step_etiquettes),
              (a.frontieres, step_frontieres), (a.offsets, step_offsets),
              (a.identifiants, step_identifiants), (a.chemin, step_chemin),
              (a.titres_perdus, step_titres_perdus),
              (a.troncature, step_troncature), (a.ancrage, step_ancrage)]
    if not any(actif for actif, _ in etapes):
        p.print_help()
        return
    for actif, fonction in etapes:
        if actif:
            fonction()


if __name__ == "__main__":
    main()
