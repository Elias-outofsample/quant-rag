"""Conversion HTML de tableau (MinerU) -> Markdown : cas réels du corpus."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from parsing.table_markdown import header_of, html_table_to_markdown, parse_rows  # noqa: E402


def test_full_table_with_caption():
    text = 'Table: Params\n<table><tr><td>Step</td><td>Z</td></tr><tr><td>0</td><td>—</td></tr></table>'
    assert html_table_to_markdown(text) == "Table: Params\n\n| Step | Z |\n|---|---|\n| 0 | — |"


def test_pipe_is_escaped_and_entities_decoded():
    text = '<table><tr><td>a|b</td><td>P&amp;L</td></tr><tr><td>1</td><td>2</td></tr></table>'
    assert html_table_to_markdown(text) == "| a\\|b | P&L |\n|---|---|\n| 1 | 2 |"


def test_fragment_receives_header_and_drops_continued_row():
    text = '<tr><td colspan="8">. continued</td></tr> <tr><td>Year</td><td>28</td></tr>'
    assert html_table_to_markdown(text, header=["A", "B"]) == "| A | B |\n|---|---|\n| Year | 28 |"


def test_header_not_duplicated_when_fragment_already_starts_with_it():
    text = '<tr><td>A</td><td>B</td></tr><tr><td>1</td><td>2</td></tr>'
    assert html_table_to_markdown(text, header=["A", "B"]) == "| A | B |\n|---|---|\n| 1 | 2 |"


def test_colspan_and_rowspan_expand_to_cells():
    text = '<table><tr><td colspan="2">H</td></tr><tr><td rowspan="2">R</td><td>x</td></tr><tr><td>y</td></tr></table>'
    assert parse_rows(text) == [["H", ""], ["R", "x"], ["R", "y"]]


def test_sub_sup_ocr_noise_in_caption_is_flattened():
    text = 'Table: T<sub>a</sub>bl<sub>e</sub> 6<sub>:</sub> Corr\n<table><tr><td>Inv</td><td>x</td></tr><tr><td>0.32</td><td>0.07</td></tr></table>'
    assert html_table_to_markdown(text).startswith("Table: Table 6: Corr\n\n| Inv | x |")


def test_empty_table_keeps_only_caption():
    text = 'Table: Trading Instrument Summary\n<tr><td></td><td></td></tr>'
    assert html_table_to_markdown(text) == "Table: Trading Instrument Summary"


def test_plain_text_untouched():
    assert html_table_to_markdown("no table here") == "no table here"


def test_header_of_full_table():
    assert header_of('Table: c\n<table><tr><td>A</td><td>B</td></tr><tr><td>1</td><td>2</td></tr></table>') == ["A", "B"]
