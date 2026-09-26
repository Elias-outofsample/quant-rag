"""The corpus-free demo runs, and shows what it claims to show."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "rag"))

import demo  # noqa: E402


def test_demo_tour(capsys):
    results = demo.demo()
    out = capsys.readouterr().out
    assert "Quote verification" in out

    # every raw cut breaks something, no contract cut does, and each announces what remains
    for cut in results["cuts"]:
        assert cut["raw"], cut["label"]
        assert cut["contract"] == [], cut["label"]
        assert cut["remaining"] > 0

    quotes = results["citations"]
    assert quotes["copied from the passage"]["trouve"] is True
    assert quotes["retyped: spacing, hyphen, capitals"]["trouve"] is True
    retyped, copied = quotes["retyped: spacing, hyphen, capitals"], quotes["copied from the passage"]
    assert retyped["offsets"] == copied["offsets"]
    assert quotes["one word changed (more -> less)"]["trouve"] is False
    assert quotes["one word changed (more -> less)"]["plus_proche"]["ressemblance"] > 0.9
    assert quotes["stitched from two places"]["trouve"] is False

    periods = results["periods"]
    between = periods["volatility targeting, according to papers published between 2015 and 2020"]
    assert (between["year_min"], between["year_max"]) == (2015, 2020)
    assert periods["optimal execution with square-root impact"] is None
