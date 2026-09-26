# The output contract

`rag/contrat.py` is the single source of what the served surface returns. It makes three
promises, each measured. Its version (`1.0.0`) is distinct from the server's and is written
into every response and every log line, so that an old trace stays readable.

## 1. A passage arrives whole, or cleanly interrupted

No served passage is cut inside a `$…$` formula, a `$$…$$` block, a Markdown table row or
a word.

Before the contract there were **five** truncation windows on the served surface, all
literal and mutually inconsistent: `search_documents` 2,500 characters, `search_graph`
2,500, `connect_entities` 1,200, `timeline` 300 and `get_passage` 6,000. The last one was
written `[:max_characters]` and escaped the regular-expression sweep that guarded the
others. At 2,500 characters, 25.8 % of the corpus passages were truncated, and 72.6 % of
those cuts broke something: 1,179 inside an open `$$` block, 277 inside a `$…$` formula, 29
in the middle of a table row, 3,792 in the middle of a word.

Measured on what the server actually returned to the 155 benchmark questions: **178 of the
760 served passages carried a forbidden cut; after the contract, 0.** Context grew from
8,631 to 9,844 characters per question (+14.1 %).

This is a fidelity fix, not a claimed accuracy gain. A separate experiment had measured that
widening the window does not improve answer coverage beyond what production already served
(2,400 → 4,500 characters: +0.031, 95 % CI [−0.061, +0.123]), and the contract does not
claim otherwise.

| Constant | Value | Governs |
|---|---:|---|
| `PASSAGE_CHARACTERS` | 10,000 | a search passage, and the benchmark window |
| `PASSAGE_ENTIER_CHARACTERS` | 15,000 | `get_passage`: the passage and one neighbour on each side |
| `APERCU_CHARACTERS` | 300 | `timeline`: a declared one-line preview, the only exception to the promise |

Why 10,000 and not 6,000: the longest served passage is a 9,384-character table, precisely
the material the contract protects. The mean served passage is 1,734.7 characters with a
6,000 cap and 1,734.9 with no cap at all, so the higher cap costs two tenths of a character
per passage. For `get_passage`, the old 6,000 window truncated 30.2 % of passage-plus-
neighbours windows (median 4,520, p95 8,850, maximum 13,982 characters); the new one
truncates none.

How the cut works (`couper`): walk back from the cap to the last position that is outside
any open formula or block, not inside a table row and not inside a word; prefer a sentence
or paragraph boundary if one lies within a tenth of the window (at most 400 characters);
never return an empty passage; append a marker saying how many characters remain and how to
read them (`get_passage` with the chunk id). A text whose own markup is unbalanced (127
passages, mostly dollar signs used as currency in tables) is flagged `math_unbalanced`
instead of being "protected".

Quality flags are deterministic and never repair the text (repairing would change the
frozen corpus signature): `has_math`, `math_unbalanced`, `has_control_chars`,
`has_html_tags`, `has_table`, `truncated`.

## 2. A citation is verifiable

Every passage carries an anchor line:

```text
ancre   : doc-96437e7c996bb218 · caractères 10861–13884 (exacte) · sha256:c41346f98382
```

document, character offsets in the canonical text of the document, granularity (`exacte` =
the offsets delimit the passage word for word; `bloc` = the block it came from), and the
SHA-256 of the canonical text. 100 % of the 26,120 served passages are anchored, 86.9 % at
the exact granularity.

`verify_citation(document_id, quote)` answers yes or no **without a language model**
(`rag/citation.py`):

1. **Deterministic normalisation** of the quote and of the canonical text: NFKC (ligatures
   back to letters), `<sup>`/`<sub>` tags removed, dashes and quotes unified, whitespace
   collapsed, case folded. Every normalised character keeps the index of its origin, so a
   match is returned as offsets in the canonical text.
2. **Exact search** of the normalised quote. This step alone decides `trouve` (found).
3. **Tolerant search** by word window when the exact one fails. It never says found: it says
   how far the closest passage is (similarity, number of differing words) and where, above a
   0.60 similarity floor. A quote with one changed word is not "almost true"; it is false,
   and knowing which word differs is what makes the verdict useful.

Table passages are served in their Markdown rendering, which is not a substring of the
canonical text; the tool also searches the served text and says which one matched.

Measured by `benchmark/controle_citation.py`: **50 of 50 real quotes found; 0 of 50 quotes
altered by a single word accepted.** Example from the running server: a real quote
is found at offsets [13312, 13463] in 83 ms; the same quote with *minimize* changed to
*maximize* returns `trouve: false`, with the closest passage at similarity 0.917.

## 3. Everything is traceable

`request_id`, `server_version`, `contrat_version`, `corpus_signature`, `config_hash` and
`caller` open every served response **and** every line of the decision log; the same request
id links the two.

```text
trace: request_id=8a7bdb351ec64fbb · serveur 1.4.0 · contrat 1.0.0 · corpus e1bdf36e2e · config aa5531c56ba9 · fenêtre 10000 c.
```

A number copied from a response without its trace line cannot be re-read six months later;
with it, the exact server, contract, corpus and configuration are known.

## What the contract did not change

The ranking. Every contract change is checked against unchanged retrieval figures on the
benchmark (the same nDCG@10 on both benches, `dense_matrix --check` agreement, same corpus
signature), and the tests guard it: a regular-expression sweep fails if a literal truncation
reappears in `mcp_server.py`, `graph_search.py` or `quant_rag.py`, and every rendering site
must go through `contrat`.
