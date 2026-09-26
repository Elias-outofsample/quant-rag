"""Extraction des métadonnées bibliographiques des 258 documents (titre, auteurs, année).

Quatre sources indépendantes, consolidées avec provenance et conflits explicites :

  S1 filename      nom du fichier PDF : identifiant arXiv (YYMM.NNNNN), nommage « à tirets
                   doubles » (Titre -- Auteurs -- Année -- Éditeur), préfixe de date.
                   Piège mesuré : le préfixe ``2026-08-03_`` de 105 fichiers est une
                   **date de téléchargement**, pas une date de publication (38 de ces
                   papiers portent un tampon arXiv de 2010 à 2025). Il est ignoré.
  S2 arxiv_stamp   tampon « arXiv:1107.4632v3 [q-fin.RM] 16 Jul 2012 » dans les deux
                   premières pages : année de *cette version* (publication_year) ; l'identifiant
                   donne l'année de première soumission (first_year).
  S3 pdf_embedded  dictionnaire Info et XMP du PDF (pypdf) : /Title, /Author, dc:creator,
                   /CreationDate. 222 PDF présents sur 258. Qualité inégale : 'option41.dvi',
                   'CSUR5101-04', 'prhansen' sont rejetés par des règles de plausibilité.
  S4 llm_first_page Mistral (mistral-small) lit le texte de la première page tel que MinerU
                   l'a extrait (blocks.jsonl) — disponible pour les 258 documents, y compris
                   les 36 sans PDF — plus les lignes « © / edition / published » des pages
                   suivantes. Sortie JSON : titre, auteurs, année (avec citation de la preuve),
                   première année, support, type de document.

Deux compléments : une seconde passe LLM sur les lignes datées des vingt premières pages pour
les documents restés sans année (page de copyright des livres), et ``overrides.json`` pour
les corrections manuelles (source « manual », connaissance externe au corpus).

Le résultat est ``documents-metadata-v1.json`` : par document, la valeur retenue pour
chaque champ, sa source, les candidats écartés, les conflits, ce qui reste à renseigner.

    python rag/metadata/extract_metadata.py            # passe complète (LLM mis en cache)
    python rag/metadata/extract_metadata.py --no-llm   # règles seules, sans appel réseau
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import warnings
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
INGESTED = ROOT / "data" / "processed" / "ingested"
PAPERS = ROOT / "data" / "papers"
OUT = Path(__file__).resolve().parent / "documents-metadata-v1.json"
sys.path.insert(0, str(ROOT / "rag" / "benchmark"))  # llm.py : client Mistral mis en cache

#: Instrument de référence des métadonnées : Mistral, comme le banc. Surchargeable par
#: ``QUANT_RAG_METADATA_LLM``, à la convention des ``QUANT_RAG_*`` du banc — ``llm.complete``
#: route seul un modèle « gemini » vers Google, il n'y a que la variable à brancher.
#: **Tout écart au défaut doit apparaître dans le fichier de résultats** (champ ``llm.model``)
#: et dans la provenance des documents concernés : un titre extrait par un autre modèle
#: n'est pas comparable à ceux qui l'ont été par Mistral.
LLM_MODEL = os.environ.get("QUANT_RAG_METADATA_LLM", "mistral-small-latest")
YEAR_MIN, YEAR_MAX = 1960, 2026

_YEAR = re.compile(r"(?<!\d)(19[6-9]\d|20[0-2]\d)(?!\d)(?!\.\d{4})")
_ARXIV_ID_FN = re.compile(r"(?<!\d)(\d{2})(\d{2})\.(\d{4,5})(v\d+)?")
_ARXIV_STAMP = re.compile(
    r"arXiv:\s*(\d{2})(\d{2})\.(\d{4,5})(v\d+)?\s*\[([^\]]+)\]\s*(\d{1,2})\s+([A-Za-z]{3})\w*\s+(\d{4})", re.I)
_DATE_PREFIX = re.compile(r"^(\d{4})(?:-(\d{2})-(\d{2}))?[_-]")
_DATEISH = re.compile(r"©|copyright|first published|published by|first edition|second edition|third edition|"
                      r"\bedition\b|isbn|printed in|all rights reserved|this version|first version|"
                      r"\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\b",
                      re.I)
_PARTICLES = {"de", "van", "von", "der", "den", "di", "da", "le", "la", "del", "della", "du", "des", "ter", "ten"}


# --------------------------------------------------------------------------- utilitaires

_LIGATURES = (
    (re.compile(r"\b([Dd])iferen"), r"\1ifferen"), (re.compile(r"\b([Ee])ficien"), r"\1fficien"),
    (re.compile(r"\b([Ii]n)eficien"), r"\1efficien"), (re.compile(r"\b([Cc])oeficien"), r"\1oefficien"),
    (re.compile(r"\b([Ee])fect"), r"\1ffect"), (re.compile(r"\b([Ss])uficien"), r"\1ufficien"),
    (re.compile(r"\b([Oo])fer"), r"\1ffer"), (re.compile(r"¨([aouAOU])"), lambda m: {"a": "ä", "o": "ö", "u": "ü", "A": "Ä", "O": "Ö", "U": "Ü"}[m.group(1)]),
)


def clean_title(text: str | None) -> str | None:
    """Espaces, ponctuation de bord, et les ligatures « ff/fi » que MinerU perd (Diferential, Eficient)."""
    if not text:
        return None
    text = re.sub(r"\s+", " ", str(text)).strip(" .—-")
    for pattern, repl in _LIGATURES:
        text = pattern.sub(repl, text)
    return text or None


def _fold(text: str) -> str:
    import unicodedata

    return "".join(ch for ch in unicodedata.normalize("NFKD", text.lower()) if not unicodedata.combining(ch))


def similar(a: str, b: str) -> float:
    import difflib

    return difflib.SequenceMatcher(None, re.sub(r"[^a-z0-9]", "", _fold(a)), re.sub(r"[^a-z0-9]", "", _fold(b))).ratio()


def plausible_title(text: str | None) -> bool:
    if not text or len(text) < 8:
        return False
    if re.search(r"\.(dvi|tex|docx?|pdf|indd|qxd)$", text, re.I):
        return False
    if re.match(r"^(untitled|microsoft word|doi:|https?:|www\.)", text, re.I):
        return False
    if re.match(r"^[A-Za-z]{2,8}[_\d\s.-]+$", text):          # CSUR5101-04, JFQ_2200087 968..1004
        return False
    if text.count("_") >= 2:                                    # mesfin_2026_structural…
        return False
    if not re.search(r"[A-Za-z]{3,}\s+[A-Za-z]{2,}", text) and not re.fullmatch(r"[A-Za-z][A-Za-z\-']{5,}", text):
        return False                                             # un mot seul reste possible : « Backtesting »
    return True


def plausible_author(name: str) -> bool:
    name = name.strip()
    if len(name) < 5 or len(name) > 60:
        return False
    low = name.lower()
    if re.search(r"faculty|staff|research\b|admin|user\b|owner|author\b|unknown|editor\b|department|university|"
                 r"institute|contribution|\bwith\b|\bby\b|\.com|\.edu|@|\d", low):
        return False
    if name.isupper() and len(name.split()) >= 4:                # ligne de couverture en capitales
        return False
    words = [w for w in re.split(r"[\s,]+", name) if w]
    if len(words) < 2:
        return False
    return sum(w[:1].isupper() for w in words) >= 2


def split_authors(raw: str) -> list[str]:
    raw = raw.replace(" and ", ";").replace(" & ", ";")
    parts = [p.strip() for p in re.split(r"\s*;\s*", raw) if p.strip()]
    out = []
    parts = [clean_name(p) for p in parts]
    for part in parts:                                           # « Jim Gatheral, Nassim Taleb » : virgule = séparateur
        pieces = [p.strip() for p in part.split(",") if p.strip()]  # si chaque côté est un nom complet
        out.extend(pieces if len(pieces) >= 2 and all(" " in p for p in pieces) else [part])
    return out


def clean_name(name: str) -> str:
    name = re.sub(r"[\[\]\"'‘’“”*†‡§¶]+", "", str(name))
    name = re.sub(r"\s*\d+\s*$", "", name)                       # marques de note « Jacquier 1 »
    return re.sub(r"\s+", " ", name).strip(" ,;")


def surname(name: str) -> str:
    name = clean_name(name)
    if "," in name:                                              # « Lopez de Prado, Marcos »
        return name.split(",")[0].strip()
    words = name.split()
    if not words:
        return name
    i = len(words) - 1
    particle = False
    while i - 1 >= 1 and words[i - 1].lower() in _PARTICLES:
        i -= 1
        particle = True
    if particle and i - 1 >= 1:                                  # « Marcos López de Prado » → « López de Prado »
        i -= 1
    return " ".join(words[i:])


def short_ref(authors: list[str], year: int | None, title: str | None) -> str:
    y = str(year) if year else "s.d."
    names = [surname(a) for a in authors]
    if not names:
        return f"{(title or '?')[:60]} ({y})"
    if len(names) == 1:
        return f"{names[0]} ({y})"
    if len(names) == 2:
        return f"{names[0]} & {names[1]} ({y})"
    return f"{names[0]} et al. ({y})"


def surname_set(authors: list[str]) -> set[str]:
    return {re.sub(r"[^a-z]", "", _fold(surname(a)).split()[-1]) for a in authors if a.strip()}


# --------------------------------------------------------------------------- S1 : nom de fichier

def parse_filename(filename: str, batch_prefixes: set[str]) -> dict:
    stem = filename[:-4] if filename.lower().endswith(".pdf") else filename
    out: dict = {"source": "filename", "title": None, "authors": [], "year": None, "first_year": None,
                 "edition": None, "kind": None, "arxiv_id": None, "download_batch": False}
    m = _ARXIV_ID_FN.search(stem)
    if m and 7 <= int(m.group(1)) <= 26:
        out["arxiv_id"] = f"{m.group(1)}{m.group(2)}.{m.group(3)}"
        out["first_year"] = 2000 + int(m.group(1))
        out["kind"] = "arxiv-id"
    if " -- " in stem:                                           # Titre -- Auteurs -- Année -- Éditeur
        parts = [p.strip() for p in stem.split(" -- ")]
        title = re.sub(r"\s*_\s+_\s*", ": ", parts[0]).replace("_ ", ": ").replace(" _", ":")
        title = re.split(r"\s*\((Wiley|Financial Engineering|Global Financial)", title)[0].strip(" :")
        out["title"] = clean_title(title)
        if len(parts) > 1:
            authors = re.sub(r"^\[edited by\]\s*", "", parts[1]).replace("_", ".")
            out["authors"] = [a for a in split_authors(authors) if plausible_author(a)]
        years = [int(y) for p in parts[2:] for y in _YEAR.findall(p)]
        if years:
            out["year"], out["first_year"] = max(years), (min(years) if min(years) < max(years) else None)
        ed = re.search(r"\b(\d)(?:st|nd|rd|th)\b|\b(first|second|third|fourth)\b", " ".join(parts[2:]), re.I)
        if ed:
            out["edition"] = ed.group(0)
        out["kind"] = "double-dash"
        return out
    m = _DATE_PREFIX.match(stem)
    if m:
        prefix = stem[:m.end() - 1]
        out["download_batch"] = prefix in batch_prefixes
        if not out["download_batch"]:
            out["year"] = int(m.group(1))
        out["kind"] = out["kind"] or ("date-prefix" if m.group(2) else "year-prefix")
        return out
    if out["kind"] != "arxiv-id":
        ys = [int(y) for y in _YEAR.findall(stem)]
        if ys:
            out["year"] = ys[0]
            out["kind"] = "year-in-name"
    return out


# --------------------------------------------------------------------------- S2 : tampon arXiv

def parse_arxiv_stamp(text: str) -> dict | None:
    m = _ARXIV_STAMP.search(text)
    if not m:
        return None
    return {"source": "arxiv_stamp", "arxiv_id": f"{m.group(1)}{m.group(2)}.{m.group(3)}",
            "version": m.group(4) or "v1", "category": m.group(5),
            "year": int(m.group(8)), "first_year": 2000 + int(m.group(1)),
            "stamp": m.group(0)}


# --------------------------------------------------------------------------- S3 : PDF (Info + XMP)

def parse_pdf(path: Path) -> dict:
    out: dict = {"source": "pdf_embedded", "present": path.exists(), "title": None, "authors": [],
                 "creation_year": None, "raw": {}}
    if not path.exists():
        return out
    from pypdf import PdfReader
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            reader = PdfReader(str(path))
            info = {k.strip("/"): str(v).strip() for k, v in (reader.metadata or {}).items() if v and str(v).strip()}
            xmp = reader.xmp_metadata
    except Exception as error:  # PDF abîmé : on ne bloque pas la passe
        out["raw"] = {"error": repr(error)[:100]}
        return out
    out["raw"] = {k: info[k][:160] for k in ("Title", "Author", "CreationDate", "Producer") if k in info}
    title = clean_title(info.get("Title"))
    authors: list[str] = []
    if xmp is not None:
        try:
            if xmp.dc_creator:
                authors = [clean_name(a) for a in xmp.dc_creator if clean_name(a)]
                out["raw"]["xmp_creator"] = authors[:8]
            if not title and xmp.dc_title:
                title = clean_title(next(iter(xmp.dc_title.values()), None))
            for attr in ("xmp_create_date", "dc_date"):
                value = getattr(xmp, attr, None)
                value = value[0] if isinstance(value, list) and value else value
                if value and not info.get("CreationDate") and hasattr(value, "year"):
                    info["CreationDate"] = f"D:{value.year:04d}"
                    break
        except Exception:
            pass
    if not authors and info.get("Author"):
        authors = split_authors(info["Author"])
    authors = [clean_name(a) for a in authors if clean_name(a)]
    out["title"] = title if plausible_title(title) else None
    out["authors"] = [a for a in authors if plausible_author(a)]
    m = re.match(r"D:(\d{4})", info.get("CreationDate", ""))
    if m and YEAR_MIN <= int(m.group(1)) <= YEAR_MAX:
        out["creation_year"] = int(m.group(1))
    return out


# --------------------------------------------------------------------------- S4 : LLM sur la première page

def first_page_text(blocks_path: Path) -> tuple[str, str]:
    """(texte de la première page utile, lignes datées des pages suivantes)."""
    pages: dict[int, list[str]] = {}
    with blocks_path.open(encoding="utf-8") as handle:
        for line in handle:
            b = json.loads(line)
            if b["page_idx"] > 6:
                break
            t = re.sub(r"\s+", " ", b.get("text") or "").strip()
            if t:
                pages.setdefault(b["page_idx"], []).append(t)
    first = ""
    for idx in sorted(pages):
        first = "\n".join(pages[idx])
        if len(first) >= 40:
            break
    dated = []
    for idx in sorted(pages):
        for t in pages[idx]:
            if _DATEISH.search(t) and len(dated) < 12:
                dated.append(f"[p.{idx + 1}] {t[:240]}")
    return first[:3800], "\n".join(dated)


_PROMPT = """You extract bibliographic metadata from the first page of a document (a research paper, book, thesis or report in quantitative finance), exactly as printed. Use ONLY the text provided; never guess from general knowledge and never take a year from the reference list.

Return a JSON object with these keys:
- "title": the document's title (with subtitle after a colon if printed), or null
- "authors": list of author full names as printed ("Given Surname"), without affiliations, footnote marks or degrees; [] if none printed
- "publication_year": the year this version/edition was published or dated (arXiv stamp date, journal issue, copyright, "this version", thesis date, edition year), as an integer, or null if no date appears in the text
- "year_evidence": the exact text fragment that carries that year, or null
- "first_year": an earlier year if the text says this is a later version/edition of an earlier work ("first version", "first edition", original copyright), else null
- "venue": journal, conference, publisher, repository (e.g. "Econometrica", "arXiv", "SSRN", "Wiley") or null
- "doc_type": one of "journal-article", "preprint", "working-paper", "book", "book-chapter", "thesis", "report", "lecture-notes", "other"
- "edition": e.g. "2nd", "Second Edition", or null
- "confidence": "high" if title, authors and year are all printed clearly; "medium" if one is missing or ambiguous; "low" otherwise

FIRST PAGE:
<<<
{first}
>>>

DATED LINES FROM THE FOLLOWING PAGES (copyright page, edition notice; may be empty):
<<<
{dated}
>>>"""


def llm_extract(first: str, dated: str) -> dict | None:
    import llm

    messages = [{"role": "user", "content": _PROMPT.format(first=first, dated=dated or "(none)")}]
    parsed = llm.complete_json(messages, model=LLM_MODEL, max_tokens=600)
    if not isinstance(parsed, dict):
        return None
    out = {"source": "llm_first_page", "model": LLM_MODEL}
    out["title"] = clean_title(parsed.get("title"))
    authors = parsed.get("authors") or []
    out["authors"] = [clean_name(a) for a in authors if isinstance(a, str) and plausible_author(clean_name(a))]
    for key in ("publication_year", "first_year"):
        v = parsed.get(key)
        out[key] = int(v) if isinstance(v, (int, float)) and YEAR_MIN <= int(v) <= YEAR_MAX else None
    out["year_evidence"] = (parsed.get("year_evidence") or None) and str(parsed["year_evidence"])[:200]
    # « 2020 Mathematics Subject Classification » n'est pas une date : preuve nue + MSC dans le texte → rejet.
    if out["publication_year"] and re.fullmatch(r"\D{0,3}\d{4}\D{0,3}", (out["year_evidence"] or "").strip()) \
            and re.search(rf"{out['publication_year']}\s+Mathematics Subject Classification|MSC\s*\(?{out['publication_year']}", first):
        out["rejected"] = f"année {out['publication_year']} = Mathematics Subject Classification"
        out["publication_year"], out["year_evidence"] = None, None
    out["venue"] = clean_title(parsed.get("venue"))
    out["doc_type"] = parsed.get("doc_type") if isinstance(parsed.get("doc_type"), str) else None
    out["edition"] = clean_title(parsed.get("edition"))
    out["confidence"] = parsed.get("confidence") if parsed.get("confidence") in ("high", "medium", "low") else "low"
    return out


_DEEP_PROMPT = """Below are lines mentioning dates, copyright, edition or publication, taken from the first pages of a document (a book, paper or thesis in quantitative finance), each prefixed by its page number. Using ONLY these lines, give the publication year of THIS edition/version of the document.

Return a JSON object: {{"publication_year": int or null, "year_evidence": exact fragment or null, "first_year": earlier year of a first edition/version if stated, else null}}. Ignore years that belong to cited works, data samples or biographies. If no publication date is stated, return null.

LINES:
<<<
{lines}
>>>"""


def dated_lines(blocks_path: Path, max_page: int = 20, max_lines: int = 40) -> str:
    out = []
    with blocks_path.open(encoding="utf-8") as handle:
        for line in handle:
            b = json.loads(line)
            if b["page_idx"] >= max_page or len(out) >= max_lines:
                break
            t = re.sub(r"\s+", " ", b.get("text") or "").strip()
            if t and _YEAR.search(t) and (_DATEISH.search(t) or b["page_idx"] <= 4):
                out.append(f"[p.{b['page_idx'] + 1}] {t[:200]}")
    return "\n".join(out)


def llm_year_deep(blocks_path: Path) -> dict | None:
    import llm

    lines = dated_lines(blocks_path)
    if not lines:
        return None
    parsed = llm.complete_json([{"role": "user", "content": _DEEP_PROMPT.format(lines=lines)}],
                               model=LLM_MODEL, max_tokens=200)
    if not isinstance(parsed, dict):
        return None
    year = parsed.get("publication_year")
    evidence = str(parsed.get("year_evidence") or "")
    # Une plage « 2000–2024 » est un échantillon de données, pas une date d'édition.
    usable = bool(_DATEISH.search(evidence) or "©" in evidence or "arxiv" in evidence.lower()) \
        and not re.search(r"\d{4}\s*[–\-]\s*\d{4}", evidence)
    if not (isinstance(year, (int, float)) and YEAR_MIN <= int(year) <= YEAR_MAX and usable):
        return {"source": "llm_deep_pages", "publication_year": None, "rejected_evidence": evidence[:100] or None,
                "lines": lines.count("\n") + 1}
    first = parsed.get("first_year")
    return {"source": "llm_deep_pages", "publication_year": int(year), "year_evidence": str(parsed["year_evidence"])[:200],
            "first_year": int(first) if isinstance(first, (int, float)) and YEAR_MIN <= int(first) < int(year) else None,
            "lines": lines.count("\n") + 1}


# --------------------------------------------------------------------------- consolidation

def consolidate(doc: dict, fn: dict, stamp: dict | None, pdf: dict, llm: dict | None) -> dict:
    conflicts: list[str] = []
    todo: list[str] = []

    # --- année : candidats par ordre de confiance
    candidates: list[tuple[str, int]] = []
    if stamp:
        candidates.append(("arxiv_stamp", stamp["year"]))
    if llm and llm.get("publication_year") and llm.get("year_evidence"):
        candidates.append(("llm_first_page", llm["publication_year"]))
    if fn["kind"] == "double-dash" and fn["year"]:
        candidates.append(("filename_dashed", fn["year"]))
    if fn["year"] and fn["kind"] != "double-dash":
        candidates.append(("filename", fn["year"]))
    if pdf.get("creation_year"):
        candidates.append(("pdf_creation_date", pdf["creation_year"]))
    if fn.get("arxiv_id") and fn.get("first_year") and not candidates:   # tampon arXiv illisible (texte vertical)
        candidates.append(("filename_arxiv_id", fn["first_year"]))
    year_source, year = (candidates[0] if candidates else (None, None))
    strong = {"arxiv_stamp", "llm_first_page", "filename_dashed", "filename"}
    agreeing = [s for s, y in candidates if y == year and s in strong]
    for s, y in candidates[1:]:
        if y != year and s in strong and abs(y - year) > (1 if s == "filename" else 0):
            conflicts.append(f"année : {year_source}={year} vs {s}={y}")
    if year is None:
        todo.append("année absente de toutes les sources")
    year_confidence = ("high" if len(agreeing) >= 2 else
                       "medium" if year_source in ("arxiv_stamp", "llm_first_page", "filename_dashed") else
                       "low" if year else None)

    first_year = None
    llm_year = (llm or {}).get("publication_year") if (llm or {}).get("year_evidence") else None
    for cand in sorted(c for c in ((stamp or {}).get("first_year"), (llm or {}).get("first_year"),
                                   fn.get("first_year"), llm_year if stamp else None) if c):
        if cand and year and cand < year:
            first_year = cand
            break

    # --- titre
    title, title_source = None, None
    llm_title, pdf_title = (llm or {}).get("title"), pdf.get("title")
    if llm_title and pdf_title and plausible_title(pdf_title) and similar(llm_title, pdf_title) >= 0.85:
        title, title_source = pdf_title, "pdf_embedded"          # même titre, typographie du PDF (ligatures, accents)
    else:
        for src, value in (("llm_first_page", llm_title), ("pdf_embedded", pdf_title),
                           ("filename_dashed", fn.get("title") if fn["kind"] == "double-dash" else None)):
            if value and plausible_title(value):
                title, title_source = value, src
                break
    if title is None:
        title, title_source = re.sub(r"[_-]+", " ", Path(doc["filename"]).stem).strip(), "filename_stem"
        todo.append("titre non extrait (nom de fichier utilisé)")
    if llm_title and pdf_title and plausible_title(pdf_title) and similar(llm_title, pdf_title) < 0.85:
        conflicts.append(f"titre : llm={llm_title[:50]!r} vs pdf={pdf_title[:50]!r}")

    # --- auteurs : la liste la plus complète parmi celles qui se recoupent
    lists = [("pdf_embedded", pdf.get("authors") or []), ("llm_first_page", (llm or {}).get("authors") or []),
             ("filename_dashed", fn.get("authors") or [])]
    lists = [(s, a) for s, a in lists if a]
    authors, authors_source = [], None
    if lists:
        authors_source, authors = lists[0]
        for s, a in lists[1:]:
            s1, s2 = surname_set(authors), surname_set(a)
            if s1 == s2 and s == "llm_first_page" and len(a) == len(authors):
                authors, authors_source = a, s                    # texte imprimé : garde les diacritiques
            elif s1 <= s2 and len(a) > len(authors):
                authors, authors_source = a, s
            elif not (s1 <= s2 or s2 <= s1):
                conflicts.append(f"auteurs : {authors_source}={[surname(x) for x in authors][:4]} vs {s}={[surname(x) for x in a][:4]}")
    if not authors:
        todo.append("auteurs absents de toutes les sources")
    authors_agree = len({s for s, a in lists if surname_set(a) & surname_set(authors)}) >= 2

    confidence = ("high" if year_confidence == "high" and authors and authors_agree else
                  "medium" if year and authors else "low")
    return {
        "document_id": doc["document_id"], "filename": doc["filename"], "page_count": doc.get("page_count"),
        "title": title, "authors": authors, "publication_year": year, "first_year": first_year,
        "venue": (llm or {}).get("venue") or ("arXiv" if stamp else None),
        "doc_type": (llm or {}).get("doc_type") or ("preprint" if stamp else None),
        "edition": (llm or {}).get("edition") or fn.get("edition"),
        "arxiv_id": (stamp or {}).get("arxiv_id") or fn.get("arxiv_id"),
        "short_ref": short_ref(authors, year, title),
        "provenance": {"title": title_source, "authors": authors_source, "year": year_source,
                       "year_candidates": [{"source": s, "year": y} for s, y in candidates],
                       "year_confidence": year_confidence, "authors_agree": authors_agree},
        "confidence": confidence, "conflicts": conflicts, "todo": todo,
        # Provenance minimale, ajoutée le 8 septembre 2026 par le chantier
        # `reprise-ingestion`. DÉRIVÉE, jamais saisie : voir rag/metadata/source_class.py,
        # dont la règle s'accorde 150 fois sur 150 avec les provenances déclarées à la main
        # dans les rapports de lot. Elle n'entre ni dans registry_digest ni dans
        # titles_digest : l'ajouter ne renomme aucun index.
        # `licence` est RÉSERVÉ et laissé à None — usage personnel et local, aucun chantier
        # de droits n'est ouvert. Le champ existe pour que la question ait une place le jour
        # où elle se posera, pas pour être rempli aujourd'hui.
        "source_class": _classer_provenance(doc, stamp),
        "licence": None,
        "sources": {"filename": fn, "arxiv_stamp": stamp, "pdf_embedded": pdf, "llm_first_page": llm},
    }


def _classer_provenance(doc: dict, stamp: dict | None) -> str:
    """`arxiv` | `ssrn` | `livre` | `autre`, par la règle du corpus. Import tardif :
    ``source_class`` vit à côté de ce module et n'a aucune dépendance."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import source_class as sc

    return sc.classer(doc.get("filename"), doc.get("page_count"), stamp)


# --------------------------------------------------------------------------- passe complète

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-llm", action="store_true", help="règles seules, aucun appel réseau")
    parser.add_argument("--limit", type=int, default=0, help="ne traiter que N documents (test)")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()

    folders = sorted(p for p in INGESTED.iterdir() if p.is_dir())
    docs = [json.loads((p / "document.json").read_text(encoding="utf-8")) | {"_folder": p} for p in folders]
    if args.limit:
        docs = docs[:args.limit]
    prefixes = Counter(m.group(0)[:-1] for d in docs if (m := _DATE_PREFIX.match(d["filename"])) and m.group(2))
    batch_prefixes = {p for p, n in prefixes.items() if n >= 5}   # 105 fichiers « 2026-08-03_ »

    def work(doc: dict) -> dict:
        fn = parse_filename(doc["filename"], batch_prefixes)
        first, dated = first_page_text(doc["_folder"] / "blocks.jsonl")
        stamp = parse_arxiv_stamp(first + "\n" + dated)
        pdf = parse_pdf(PAPERS / doc["filename"])
        llm = None if args.no_llm else llm_extract(first, dated)
        return consolidate(doc, fn, stamp, pdf, llm)

    started = datetime.now(timezone.utc)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        rows = []
        for i, row in enumerate(pool.map(work, docs), 1):
            rows.append(row)
            print(f"  {i}/{len(docs)}  {row['short_ref'][:50]:50} {row['confidence']}", flush=True)

    folders = {d["document_id"]: d["_folder"] for d in docs}
    if not args.no_llm:                                          # seconde passe : documents sans année
        for row in rows:
            if row["publication_year"]:
                continue
            deep = llm_year_deep(folders[row["document_id"]] / "blocks.jsonl")
            row["sources"]["llm_deep_pages"] = deep
            if deep and deep.get("publication_year"):
                row["publication_year"] = deep["publication_year"]
                row["first_year"] = row["first_year"] or deep.get("first_year")
                row["provenance"].update(year="llm_deep_pages", year_confidence="medium")
                row["provenance"]["year_candidates"].append({"source": "llm_deep_pages", "year": deep["publication_year"]})
                row["todo"] = [t for t in row["todo"] if not t.startswith("année")]
                print(f"  seconde passe : {row['filename'][:50]} → {deep['publication_year']} ({deep['year_evidence'][:50]!r})")

    overrides_path = Path(__file__).resolve().parent / "overrides.json"
    overrides = json.loads(overrides_path.read_text(encoding="utf-8")) if overrides_path.exists() else {}
    for row in rows:
        fix = overrides.get(row["filename"])
        if not fix:
            continue
        for field in ("title", "authors", "publication_year", "first_year", "venue", "doc_type"):
            if field in fix:
                row[field] = fix[field]
                key = "year" if field == "publication_year" else field
                if key in row["provenance"]:
                    row["provenance"][key] = "manual"
                if field == "publication_year":
                    row["provenance"]["year_confidence"] = "medium"
                    row["provenance"]["year_candidates"].append({"source": "manual", "year": fix[field]})
                row["todo"] = [t for t in row["todo"] if not t.startswith({"publication_year": "année", "authors": "auteurs", "title": "titre"}.get(field, "\0"))]
        row["sources"]["manual"] = fix
        if row["publication_year"] and row["authors"] and row["confidence"] == "low":
            row["confidence"] = "medium"

    for row in rows:                                             # référence courte finale
        row["short_ref"] = short_ref(row["authors"], row["publication_year"], row["title"])

    n = len(rows)
    summary = {
        "documents": n,
        "publication_year": sum(1 for r in rows if r["publication_year"]),
        "first_year": sum(1 for r in rows if r["first_year"]),
        "authors": sum(1 for r in rows if r["authors"]),
        "title_not_from_filename": sum(1 for r in rows if r["provenance"]["title"] != "filename_stem"),
        "year_source": dict(Counter(r["provenance"]["year"] or "none" for r in rows)),
        "year_confidence": dict(Counter(r["provenance"]["year_confidence"] or "none" for r in rows)),
        "title_source": dict(Counter(r["provenance"]["title"] for r in rows)),
        "authors_source": dict(Counter(r["provenance"]["authors"] or "none" for r in rows)),
        "confidence": dict(Counter(r["confidence"] for r in rows)),
        "with_conflicts": sum(1 for r in rows if r["conflicts"]),
        "with_todo": sum(1 for r in rows if r["todo"]),
        "download_batch_files": sum(1 for r in rows if r["sources"]["filename"]["download_batch"]),
        "pdf_present": sum(1 for r in rows if r["sources"]["pdf_embedded"]["present"]),
        "year_histogram": dict(sorted(Counter(r["publication_year"] for r in rows if r["publication_year"]).items())),
    }
    stats = {}
    if not args.no_llm:
        import llm as _llm
        stats = _llm.stats()
    args.out.write_text(json.dumps({
        "version": "documents-metadata-v1", "created_at": started.isoformat(timespec="seconds"),
        "llm": None if args.no_llm else {"model": LLM_MODEL, **stats},
        "policy": {
            "publication_year": "année de la version/édition présente dans le corpus (tampon arXiv de la version, "
                                "numéro de revue, page de copyright de l'édition)",
            "first_year": "année antérieure connue (première soumission arXiv, première édition, « first version »)",
            "year_precedence": ["arxiv_stamp", "llm_first_page", "filename_dashed", "filename", "pdf_creation_date",
                            "filename_arxiv_id", "llm_deep_pages (documents restés sans année)", "manual (overrides.json)"],
            "download_batch": "un préfixe de date partagé par ≥ 5 fichiers est une date de téléchargement : ignoré",
        },
        "summary": summary, "documents": rows,
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    print("écrit :", args.out)


if __name__ == "__main__":
    main()
