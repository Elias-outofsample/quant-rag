"""HTML de tableau (sortie MinerU) -> Markdown, sans dépendance externe.

MinerU rend chaque tableau en HTML brut (``<table><tr><td>…``), parfois avec ``colspan`` /
``rowspan``, des ``<sub>``/``<sup>`` dans les cellules et des formules ``$…$``. Le chunker
(``canonical_chunker.py``) découpe les grands tableaux par ``<tr>`` ; les parties suivantes
n'ont ni ``<table>`` ni ligne d'en-tête. Ce module convertit :

- un tableau complet ou un fragment de lignes en tableau Markdown (``| a | b |``) ;
- avec un en-tête fourni par l'appelant pour les fragments (``header=``), pris sur la
  première partie du même tableau ;
- en gardant la légende ``Table: …`` qui précède le HTML, les formules, et en échappant ``|``.

Pour brancher la conversion à la source (``mineru_adapter._entry_text``), il faut aussi
adapter le découpage du chunker, qui cherche des ``<tr>`` : découper sur les lignes du
Markdown (une ligne = une rangée) en répétant l'en-tête et le séparateur sur chaque partie.
"""
from __future__ import annotations

import html
import re
from html.parser import HTMLParser

_HAS_ROW = re.compile(r"<tr\b", re.I)
_CONTINUED = re.compile(r"^\W*(continued|cont\.?|suite)\W*$", re.I)


class _RowParser(HTMLParser):
    """Collecte les rangées : liste de listes de cellules, colspan développé, rowspan propagé."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: list[list[str]] = []
        self._row: list[str] | None = None
        self._cell: list[str] | None = None
        self._span = 1
        self._pending: dict[int, tuple[str, int]] = {}  # colonne -> (texte, rangées restantes) pour rowspan

    def handle_starttag(self, tag, attrs):
        tag = tag.lower()
        if tag == "tr":
            self._row = []
            self._fill_rowspans()
        elif tag in ("td", "th"):
            if self._row is None:
                self._row = []
            self._fill_rowspans()
            attributes = dict(attrs)
            self._cell = []
            self._span = max(int(str(attributes.get("colspan", "1")).strip() or 1), 1)
            self._rowspan = max(int(str(attributes.get("rowspan", "1")).strip() or 1), 1)
        elif tag == "br" and self._cell is not None:
            self._cell.append(" ")

    def handle_endtag(self, tag):
        tag = tag.lower()
        if tag in ("td", "th") and self._cell is not None and self._row is not None:
            text = _clean("".join(self._cell))
            column = len(self._row)
            self._row.append(text)
            for _ in range(self._span - 1):
                self._row.append("")
            if self._rowspan > 1:
                for offset in range(self._span):
                    self._pending[column + offset] = (text if offset == 0 else "", self._rowspan - 1)
            self._cell = None
        elif tag == "tr" and self._row is not None:
            self.rows.append(self._row)
            self._row = None

    def handle_data(self, data):
        if self._cell is not None:
            self._cell.append(data)

    def _fill_rowspans(self) -> None:
        """Insère, à la position courante, les cellules héritées d'un rowspan de la rangée du dessus."""
        if self._row is None or not self._pending:
            return
        column = len(self._row)
        while column in self._pending:
            text, remaining = self._pending.pop(column)
            self._row.append(text)
            if remaining > 1:
                self._pending[column] = (text, remaining - 1)
            column += 1

    def close(self):
        super().close()
        if self._row is not None:
            self.rows.append(self._row)
            self._row = None


def _clean(text: str) -> str:
    text = html.unescape(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.replace("|", "\\|")


def parse_rows(fragment: str) -> list[list[str]]:
    parser = _RowParser()
    parser.feed(fragment)
    parser.close()
    rows = [r for r in parser.rows if any(c.strip() for c in r)]
    # rangées « continued » insérées par MinerU aux sauts de page
    return [r for r in rows if not (len([c for c in r if c.strip()]) == 1 and _CONTINUED.match(next(c for c in r if c.strip())))]


def rows_to_markdown(rows: list[list[str]], header: list[str] | None = None) -> str:
    if header is not None:
        body = rows
    else:
        header, body = (rows[0], rows[1:]) if rows else ([], [])
    width = max([len(header)] + [len(r) for r in body]) if (header or body) else 0
    if width == 0:
        return ""
    def line(cells: list[str]) -> str:
        cells = list(cells) + [""] * (width - len(cells))
        return "| " + " | ".join(c if c.strip() else " " for c in cells) + " |"
    out = [line(header), "|" + "---|" * width]
    out.extend(line(r) for r in body)
    return "\n".join(out)


def split_caption(text: str) -> tuple[str, str]:
    """(légende avant le HTML, fragment HTML)."""
    match = re.search(r"<(table|tr|td|th)\b", text, re.I)
    if not match:
        return text, ""
    return text[:match.start()].strip(), text[match.start():]


def header_of(text: str) -> list[str] | None:
    """Première rangée d'un tableau complet, pour l'appliquer à ses fragments."""
    _, fragment = split_caption(text)
    rows = parse_rows(fragment) if fragment else []
    return rows[0] if rows else None


def html_table_to_markdown(text: str, header: list[str] | None = None) -> str:
    """Convertit un chunk-tableau ; texte inchangé s'il ne contient aucune rangée."""
    if not _HAS_ROW.search(text) and not re.search(r"<t[dh]\b", text, re.I):
        return text
    caption, fragment = split_caption(text)
    rows = parse_rows(fragment)
    if not rows:
        # tableau vide (cellules sans texte) ou légende seule : on rend la légende débarrassée des
        # balises <sub>/<sup> que l'OCR de MinerU sème dans les petites capitales.
        cleaned = _clean(re.sub(r"<[^>]+>", " ", re.sub(r"</?su[bp]>", "", text, flags=re.I))).replace("\\|", "|")
        return cleaned if cleaned else text
    if header is not None and rows and [c.lower() for c in rows[0]] == [c.lower() for c in header]:
        header = None  # l'en-tête est déjà la première rangée du fragment
    table = rows_to_markdown(rows, header)
    caption = _clean(re.sub(r"<[^>]+>", " ", re.sub(r"</?su[bp]>", "", caption, flags=re.I))).replace("\\|", "|") if caption else ""
    return (caption + "\n\n" + table) if caption else table
