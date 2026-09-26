"""``quant_rag.period_bounds`` : la clause de période, et seulement elle.

    .venv/bin/python -m pytest rag/benchmark/test_period_bounds.py -q
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import quant_rag  # noqa: E402


def test_template_of_the_bench():
    got = quant_rag.period_bounds("How many records do we need, according to sources published in 2022 or earlier?")
    assert got["year_max"] == 2022 and got["year_min"] is None
    assert got["query"] == "How many records do we need?"
    got = quant_rag.period_bounds("Which estimator wins, according to sources published in 2025 or later?")
    assert got["year_min"] == 2025 and got["year_max"] is None
    assert got["query"] == "Which estimator wins?"


def test_other_phrasings():
    assert quant_rag.period_bounds("papers published before 2010 on cointegration")["year_max"] == 2010
    assert quant_rag.period_bounds("what does the literature since 2020 say about rough volatility?")["year_min"] == 2020
    got = quant_rag.period_bounds("sources published between 2015 and 2020 on market impact")
    assert (got["year_min"], got["year_max"]) == (2015, 2020)
    assert quant_rag.period_bounds("pre-2008 research on liquidity")["year_max"] == 2008
    assert quant_rag.period_bounds("Kelly criterion, d'après les travaux publiés avant 2000")["year_max"] == 2000
    assert quant_rag.period_bounds("volatilité rugueuse selon les sources depuis 2023")["year_min"] == 2023


def test_content_years_never_match():
    for text in (
        "What are the annual EVT-based tail index estimates for the S&P 500 from 1962 to 1987?",
        "Which Brazilian stock tickers were screened for cointegration from 1997 to 2023?",
        "What was the reported theta value for S&P 500 options in the 2003 CBOE Annual Report?",
        "What is the reported arbitrage prediction accuracy in Heston (2000) for the daily S&P 500 options dataset?",
        "highest yearly Sharpe ratio after fees using China A-share stock data between Jan 2022 and Dec 2024",
        "how did option prices' reaction to net demand change after options started trading on multiple exchanges in 1999?",
        "What Hurst exponent value did Gatheral and Oomen (2007) report for S&P 500 realized volatility?",
        "the Fed's 2022 Financial Stability Report volatility estimate",
    ):
        assert quant_rag.period_bounds(text) is None, text


def test_every_bench_question():
    """15/15 questions datées de v3 avec les bornes d'or ; 0 faux positif ailleurs (v3 + v1)."""
    v3 = [json.loads(l) for l in (HERE / "questions-v3.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    v1 = [json.loads(l) for l in (HERE / "questions-v1.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    for item in v3:
        got = quant_rag.period_bounds(item["question"])
        if item["kind"] == "dated":
            assert got is not None, item["qid"]
            assert {k: v for k, v in got.items() if k in ("year_min", "year_max") and v is not None} == item["filters"], item["qid"]
            assert got["query"].rstrip("?") == item["retrieval_query"].rstrip("?"), (item["qid"], got["query"], item["retrieval_query"])
        else:
            assert got is None, (item["qid"], got)
    for item in v1:
        assert quant_rag.period_bounds(item["question"]) is None, item["qid"]
