"""Thirty-second tour of the served surface, without the corpus.

The corpus is not distributed, so this script runs the real serving functions on one
synthetic document written for the purpose: the output contract (``contrat``), quote
verification (``citation``), the period parser and the exact-token detector
(``quant_rag``). Only the storage layer is replaced: the canonical text of the document
comes from memory instead of the parsed corpus, and the Qdrant fallback is switched off.

    python rag/demo.py
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "src"))

import citation  # noqa: E402
import contrat  # noqa: E402
import quant_rag  # noqa: E402

DOC_ID = "doc-demo-000000000001"
TEXT = r"""3. A MEAN-REVERTING SPREAD

We model the log-spread $s_t$ between two cointegrated assets as an Ornstein-Uhlenbeck process. Its speed of mean reversion $\kappa$ sets the half-life of a deviation, and the long-run level $\theta$ is where the spread returns after a shock.

$$
ds_t = \kappa (\theta - s_t)\,dt + \sigma\, dW_t, \qquad \text{half-life} = \frac{\ln 2}{\kappa}
$$

A larger $\kappa$ shortens the half-life, so the same entry threshold is crossed more often and each position is held for less time. The table below reports the estimates on three simulated pairs.

Table: Table 2. Estimated parameters on simulated pairs.

| pair | kappa | theta | sigma | half-life (days) |
|---|---|---|---|---|
| A | 0.35 | 0.00 | 0.12 | 1.98 |
| B | 0.08 | 0.02 | 0.09 | 8.66 |
| C | 0.02 | -0.01 | 0.05 | 34.66 |

The estimation error on $\kappa$ decreases with the length of the sample, which is why a pair is only traded after two years of history.
"""


def _use_in_memory_document() -> None:
    """Point ``citation`` at the synthetic document instead of the parsed corpus."""
    sha = hashlib.sha256(TEXT.encode("utf-8")).hexdigest()
    citation.ancrage.texte_canonique_du_document = lambda doc: TEXT if doc == DOC_ID else None
    citation._sha_du_document = lambda doc: sha if doc == DOC_ID else None
    citation._couvertures = lambda doc: ()
    citation._chercher_dans_le_servi = lambda *args, **kwargs: None
    citation._document_brut.cache_clear()
    citation._document_normalise.cache_clear()


def what_breaks(text: str, position: int) -> list[str]:
    """What a cut of ``text`` at ``position`` leaves broken, judged on the text itself."""
    head = text[:position]
    broken = []
    if head.count("$$") % 2:
        broken.append("an open $$ block")
    elif not contrat.math_desequilibree(text) and contrat._dollars_simples(head) % 2:
        broken.append("an open $ formula")
    if contrat._dans_une_ligne_de_tableau(text, position):
        broken.append("half a table row")
    if contrat._dans_un_mot(text, position):
        broken.append("half a word")
    return broken


def tail(text: str, width: int = 44) -> str:
    return "…" + text[-width:].replace("\n", "⏎")


def section(title: str) -> None:
    print(f"\n{title}\n{'─' * len(title)}")


def demo() -> dict:
    """Run the tour; return what a test needs to check."""
    _use_in_memory_document()
    results: dict = {"cuts": [], "citations": {}}

    section("1. The output contract: a passage arrives whole, or cleanly interrupted")
    caps = {
        "inside the display formula": TEXT.index(r"\qquad"),
        "inside a table row": TEXT.index("| B |") + 12,
        "inside a word": TEXT.index("threshold") + 4,
    }
    for label, cap in caps.items():
        rendered, remaining = contrat.couper(TEXT, cap)
        raw_broken = what_breaks(TEXT, cap)
        safe_broken = what_breaks(TEXT, len(rendered))
        results["cuts"].append({"label": label, "raw": raw_broken, "contract": safe_broken,
                                "remaining": remaining})
        print(f"cap {cap:>4} ({label})")
        print(f"   raw cut      {tail(TEXT[:cap])!s:<50} breaks: {', '.join(raw_broken) or 'nothing'}")
        print(f"   contract cut {tail(rendered)!s:<50} breaks: {', '.join(safe_broken) or 'nothing'}"
              f" · {remaining} characters announced as remaining")
    print("   marker appended (the server speaks French):",
          contrat.marqueur(results["cuts"][-1]["remaining"], "chunk-demo").strip())
    print("   quality flags  :", {k: v for k, v in contrat.drapeaux_qualite(TEXT).items() if v})

    section("2. Quote verification, without a language model")
    sentence = "A larger $\\kappa$ shortens the half-life, so the same entry threshold is crossed more often"
    retyped = (sentence.replace("half-life", "half\u2011life")          # non-breaking hyphen
               .replace("the same", "the  same").replace("more", "MORE"))
    quotes = {
        "copied from the passage": sentence,
        "retyped: spacing, hyphen, capitals": retyped,
        "one word changed (more -> less)": sentence.replace("more often", "less often"),
        "stitched from two places": ("the long-run level $\\theta$ is where the spread returns"
                                     " after two years"),
    }
    for label, quote in quotes.items():
        verdict = citation.verify_citation(DOC_ID, quote)
        results["citations"][label] = verdict
        if verdict["trouve"]:
            detail = f"found · offsets {verdict['offsets']} · {verdict['methode']}"
        else:
            closest = verdict.get("plus_proche")
            detail = ("NOT found" + (f" · closest passage: similarity {closest['ressemblance']},"
                                     f" {closest['mots_differents']} word edits" if closest else ""))
        print(f"   {label:<36} {detail}")
    anchor = contrat.ligne_ancre({"document_id": DOC_ID, "ancrage_intervalles": [[0, len(TEXT)]],
                                  "ancrage_granularite": "exacte",
                                  "doc_text_sha256": hashlib.sha256(TEXT.encode()).hexdigest()})
    print("   anchor line of a served passage:", anchor)

    section("3. The query side: publication periods and exact tokens")
    for question in ("pairs trading, according to sources published before 2010",
                     "rough volatility papers since 2020",
                     "volatility targeting, according to papers published between 2015 and 2020",
                     "optimal execution with square-root impact"):
        bounds = quant_rag.period_bounds(question)
        results.setdefault("periods", {})[question] = bounds
        if bounds:
            print(f"   {question!r}\n      -> year_min={bounds['year_min']} year_max={bounds['year_max']},"
                  f" embedded query: {bounds['query']!r}")
        else:
            print(f"   {question!r}\n      -> no period: the whole corpus is searched")
    for question in ("Fukasawa SVI no-arbitrage conditions B2 B3", "why does rough volatility fit the smile"):
        tokens = quant_rag.exact_tokens(question)
        results.setdefault("tokens", {})[question] = tokens
        print(f"   exact tokens in {question!r}: {tokens['n_exact']}"
              f" {[t for k in ('acronyms', 'proper', 'numeric') for t in tokens[k]]}")
    print("\n`auto` mode stays dense whatever the count: the measurements behind that choice are in"
          " docs/evaluation.md.")
    return results


if __name__ == "__main__":
    demo()
