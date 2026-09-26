# Evaluation

This document describes how retrieval and answers are measured, and what the measurements
decided. Conventions used throughout:

- **nDCG@10** at chunk level unless stated otherwise; R@k = recall at k.
- **Paired differences**: both arms see the same questions, so the per-question difference
  is resampled; brackets give the 95 % bootstrap interval; `*` marks an interval that
  excludes zero.
- **v1** = 25 known-item questions, **v2** = 50 generated questions, **v3** = 150 generated
  questions (130 with an answer in the corpus), **pooled** = v1 + v3, 155 questions.

The scripts live in [`rag/benchmark/`](../rag/benchmark/README.md). Question sets and
per-run result files are derived from the corpus and are not distributed.

## 1. Why the first benchmark could not be trusted

The first benchmark (v1) had 25 questions written by hand while looking at the passage that
answers them. Such a question copies its answer's vocabulary. Measured as an IDF-weighted
leak,

```text
leak = sum of IDF of the question tokens that appear in the gold passage
       -----------------------------------------------------------------
       sum of IDF of all informative tokens of the question
```

v1 has a median leak of **0.533**, a 75th percentile of 0.689 and a maximum of **0.916**: on
its worst question, 92 % of the informative content of the question is literally in the
passage. Retrieval on such questions is string matching. v1 also measured retrieval only: a
system that finds the right passage and then answers wrongly, or invents when it finds
nothing, scored perfectly.

## 2. Building questions against the known-item bias

To have a ground truth, a question must be written *from* a passage, and that leaks. The
leak cannot be removed; it can be lowered and **measured**. Four barriers in series
(`generate_questions.py`):

1. **Practitioner framing.** The writer sees the passage but must ask the question as it is
   asked *before* the answer is known: describe the difficulty, do not name the solution.
2. **Blind laundering.** A second call receives **only the question**, never the passage.
   Without the source in view, it cannot copy its vocabulary. This is the central mechanism.
3. **Objective gate.** Above a leak of **0.42**, the first quartile of v1, the question goes
   back to laundering with the offending terms named; two rounds, then it is dropped. A
   floor of 0.10 stops over-laundering (a question that shares no content word with its
   passage no longer tests retrieval), and laundering keeps the candidate closest to the
   ceiling within the band.
4. **Verification.** A verifier reads question and passage and decides whether the question
   is answerable, self-contained and **singularising**: would many other passages of the
   corpus answer it equally well? This removes questions that are true everywhere and
   therefore useful nowhere.

A real example, before and after laundering:

```text
draft (leak 0.775)
  "Under what conditions does a delta-hedged option portfolio financed at the
   risk-free rate accrue a profit proportional to gamma and realized variance?"
kept (leak 0.399)
  "When you delta-hedge an options portfolio using borrowed cash, what makes it
   make extra money if stock prices barely move?"
```

**The rejection rate is a result in itself.** On v3, about a third of text, multi-document
and dated drafts survive, but only **12 %** of table drafts and **5 %** of exact-token
drafts. The dominant reason for tables is the verifier: a question about a table rarely
stands without the table in view, or it would fit a hundred other tables. The rate is the
same (12 %) whether tables are stored as HTML or as Markdown, which corrected an earlier
reading that blamed the HTML.

### v3 question families

| Family | n | What it probes | Gold passages |
|---|---:|---|---:|
| `single` | 52 | ordinary retrieval and synthesis | 1 |
| `table` | 30 | table passages (a fifth of the corpus) | 1 |
| `multi` | 23 | synthesis across two distinct documents | 2 |
| `dated` | 15 | the publication-year filter; bounds chosen to remove 46–91 % of the dated corpus | 1–2 |
| `exact` | 10 | a single exact token (acronym, proper name) in a natural-language question | 1 |
| `negative` | 20 | the corpus does **not** hold the answer: abstention expected | 0 |

The 50 v2 questions are kept byte for byte inside v3: they are the fixed point that makes
runs comparable.

## 3. Negative questions: proving an absence without asking the system

A negative question names a source whose claim is absent from the corpus, close to what the
corpus covers ("What Hurst exponent value did Gatheral and Oomen (2007) report for S&P 500
realized volatility?": rough volatility is everywhere in the corpus, that attribution is
not).

- **Lexical proof.** The LLM provides three to five spellings of the source; **all** must be
  absent from the normalised corpus text. A token test would not do: both tokens of a
  common author name can occur in unrelated documents while the phrase occurs nowhere.
- **The retriever may only refute.** The strongest configuration is queried and an LLM
  judges whether a returned passage really answers. A yes disqualifies the question; a no
  proves nothing, because that would ask the system under test to certify its own
  ignorance.
- **One source, one question.** Failures on the same absent source are correlated and would
  move the family score in a block.

## 4. Two score families, never merged

- **Retrieval, no judge.** Gold passages are fixed at generation time; recall@k, MRR and
  nDCG@10 depend on the corpus only, and stay valid when a judge model is updated. nDCG uses
  graded gains: 3 for a gold passage, 1 for another passage of the gold document (answers
  can straddle a chunk boundary), capped at two per document, so that returning ten passages
  of the right document is never worth as much as finding the right one.
- **Generation, judged.** *Groundedness* (faithful to the passages shown?) and *coverage*
  (are the reference facts present?) are graded 0–2 by an LLM judge. The reference facts
  (2–4 atomic statements) are extracted when the question is generated, before any answer
  exists, so the judge checks a list rather than grading an open answer.
- **Abstention belongs to neither.** The generator has an explicit exit token,
  `INSUFFICIENT_EVIDENCE`; counting abstentions is a string comparison.
- **The hinge.** Each row records whether a gold passage was among the passages shown to the
  generator (a membership test, no judge). Coverage *when the gold was shown* isolates the
  generation stage from retrieval.

## 5. Measuring the judge instead of trusting it

- **Sentinel items**, graded blind in the same stream, whose correct grade is known:
  `gold_verbatim` (the gold passage copied), `off_topic` (the answer to another question),
  `unsupported_fluent` (an answer written without passages), `refusal_on_answerable`,
  `refusal_on_negative`. The share graded within the expected band is the judge's accuracy:
  **0.889 (16/18)** on the v3 baseline, with the two failure modes a lenient judge lets
  through (`off_topic`, `unsupported_fluent`) at 1.00.
- **Noise floor.** A sample is graded twice at temperature 0.3 (deliberately above the 0.0
  used for grading, to overstate the noise); disagreement gives the floor under which a
  coverage difference is not a result: **0.083** on the v3 baseline.
- Pointwise grading only (no position bias), different models to answer and to judge
  (`mistral-small` answers, `mistral-medium` judges), answers capped at 130 words, a 0–2
  scale with the justification written before the grade.
- **Stated residual bias:** generator and judge are from the same model family (a stronger
  model was not available on the free tier). The sentinels bound this bias; they do not
  remove it.

## 6. Instruments are proven before they are used

- `dense_matrix.py` rebuilds the vector matrix in memory so that several experiments can run
  in parallel despite the embedded Qdrant lock. `--check` compares its top 50 with Qdrant's:
  **155/155 identical, in the same order**. The check found a real discrepancy on the way
  (production drops heading passages after the top 50; without reproducing it, 43 top-10
  lists differed).
- `compare_v1_v2.py` replays the v1 questions in the current benchmark code and recovers the
  published figures exactly (maximum deviation 0.00). It runs every time: if it drifts, the
  benchmark is broken, and that is known before anything else is read.
- `eval_router.py` checks that the router reproduces **exactly** the ranking of the path it
  chooses (0 divergences on 155): the router chooses, it modifies nothing.

## 7. Results

### 7.1 The result that changed the design

Same configurations, same metrics, same code, two question sets:

| Configuration | v1 | v2 | Change |
|---|---:|---:|---:|
| dense | 0.853 | **0.476** | −0.377 |
| BM25 | 0.720 | 0.110 | −0.610 |
| RRF (dense + BM25) | 0.833 | 0.383 | −0.450 |
| **RRF + rerank** | **0.935** | 0.335 | −0.600 |

The best configuration on v1 is the second worst on v2: RRF + rerank against dense on v2,
**−0.140 [−0.240, −0.043]\***. The explanation fits in two questions. v1 asks *"What MATLAB
code implements a time-series momentum strategy with a 250-day lookback and a 25-day holding
period?"*: three rare exact tokens, and BM25 finds the target without understanding
anything. v2 asks *"if I keep the portfolio's overall risk level and each individual
holding's risk level the same, how does the total cash I have in the market change when I
spread it across more positions?"*: nothing to match, BM25 falls to R@1 = 0.00 (32 misses
out of 40), RRF dilutes the dense signal with noise, and the reranker, applied to a diluted
pool, makes it worse. Checked to rule out a bug: BM25 returns its 50 candidates and none
loses its text.

The conclusion is not that BM25 is useless: the value of the lexical signal depends entirely
on the query distribution, and v1's was maximally favourable to it.

### 7.2 The router

A rule-based router was calibrated on the 65 questions of v1 + v2, over every signal
available without an LLM (IDF, exact tokens, dense confidence, BM25 score and margin,
dense–BM25 overlap, combinations). The retained rule, *hybrid if the query holds at least
two exact tokens*, was the only one whose lower bound stayed above zero on both benches:

| Configuration | v1 | v2 | Pooled | Router vs … |
|---|---:|---:|---:|---|
| dense | 0.853 | 0.476 | 0.621 | +0.021 [+0.002, +0.050]\* |
| RRF + rerank | 0.935 | 0.335 | 0.566 | +0.076 [+0.002, +0.150]\* |
| **router** | 0.885 | 0.490 | **0.642** | — |

The 150-question v3 bench was built to confirm it, and it did not:

| Bench | n | dense | RRF + rerank | router | router − dense |
|---|---:|---:|---:|---:|---|
| v1 | 25 | 0.853 | 0.935 | 0.885 | +0.032 [+0.000, +0.088] |
| **v3** | 130 | **0.521** | 0.412 | 0.503 | **−0.019 [−0.041, +0.002]** |
| v3, the 90 new questions | 90 | — | — | — | −0.027 [−0.059, +0.003] |

Table questions are full of exact tokens (tickers, years, "S&P 500"), so a third of them
were routed to the hybrid path, which is weak on tables (−0.072 on that family). No rule
computable without an LLM beat dense on v3; a perfect router would have added +0.055, so
the signal exists but lies in no statistic of the query or of the top-k. **Decision:
`auto` is dense**; hybrid stays available on request for remembered identifiers, the
detector and the decision log stay, and one constant restores the rule if a larger bench
supports it.

### 7.3 Fusion, reranking and rewriting

Fusion of dense and BM25 without a reranker, 14 variants: **none adopted**; plain RRF costs
−0.088 [−0.124, −0.054]\* pooled, and the loss grows monotonically with the weight given to
the lexical side. On 16 of the 17 questions that fusion pushes out of the
top 10, the target is absent from BM25's top 50: BM25 does not compete, it dilutes.

Rerankers on the same top-50 pool (v3, 130 questions, clean-title vectors):

| Pool | Reranker | v1 | v3 | Δ v3 vs pool | Latency per query |
|---|---|---:|---:|---|---:|
| dense | none (production) | 0.897 | 0.549 | — | 0.2 s |
| dense | `bge-reranker-base` | 0.913 | 0.458 | −0.092 [−0.148, −0.033]\* | 1.45 s |
| dense | `bge-reranker-v2-m3` | 0.876 | 0.556 | +0.007 [−0.047, +0.063] | 4.9 s |
| dense | `Qwen3-Reranker-0.6B` | 0.912 | **0.620** | **+0.070 [+0.018, +0.125]\*** | 15.0 s (p50) |
| RRF | `bge-reranker-base` (hybrid mode) | 0.935 | 0.419 | −0.049 [−0.096, −0.000] | 1.6 s |
| RRF | `bge-reranker-v2-m3` | 0.917 | 0.529 | +0.062 [+0.017, +0.107]\* | 5.4 s |

Qwen3-Reranker is the only one that improves the production path significantly without
hurting v1 (R@1 0.25 → 0.37, 43 net gains against 24 net losses). At 15 s per query it is
not a default. A budget study followed: scores are pointwise and batch-stable, so reranking
only the head of the pool can be derived from one run (checked: 155/155 identical nDCG).
Reranking the top 30 is indistinguishable from reranking all 50 (−0.017 [−0.044, +0.005])
for half the time; the top 10 keeps half the gain for a sixth of the time (3.3 s). What it
buys is an *order*, not recall: the 26 questions whose gold is outside the top 50 stay
missed. **Verdict:** documented, not shipped; a future "deep search" mode has a measured
candidate. A latency figure was also corrected on the way: the first measurement was
inflated by a full-vocabulary logits tensor (1.5 GB per batch in fp16) that sent the machine
to swap; with `logits_to_keep=1` the median is 15.0 s, and the quality figures are
unchanged.

LLM query rewriting and HyDE (dense matrix, same filters, v3):

| Configuration | v1 | v3 | Δ v3 vs dense | Misses v3 |
|---|---:|---:|---|---:|
| **dense** | 0.853 | **0.521** | — | 37 |
| HyDE passage alone | 0.748 | 0.451 | −0.070 [−0.119, −0.021]\* | 41 |
| mean of query and HyDE vectors | 0.843 | 0.538 | +0.017 [−0.019, +0.056] | 29 |
| RRF of query and 3 reformulations | 0.739 | 0.509 | −0.013 [−0.051, +0.025] | 24 |
| mean of the 4 vectors | 0.810 | 0.520 | −0.002 [−0.035, +0.032] | 30 |

No rewriting beats dense robustly, and every RRF variant loses on v1. The failure mode of
HyDE on a specialised corpus showed up verbatim: an acronym the model does not know is
invented rather than preserved ("MCP functions", Model Context Protocol, became "Mean-CVaR
Portfolio", and the hypothetical passage went confidently the wrong way). **Rejected as a
default, kept as an experiment.**

### 7.4 What was adopted

**Clean titles in the embedding.** Re-embedding every passage behind its consolidated title
instead of the export title:

| | v1 (25) | v3 (130) | Pooled (155) |
|---|---:|---:|---:|
| export titles | 0.853 | 0.521 | 0.575 |
| **clean titles** | **0.897** | **0.549** | **0.606** |
| Δ | +0.045 [−0.014, +0.136] | +0.028 [−0.001, +0.057] | **+0.031 [+0.003, +0.059]\*** |
| gold outside the top 50 | 2 → 1 | 37 → 25 | 39 → 26 |

The only experiment of its round that loses on no bench. Where the export title was
aberrant (HTML, or over 200 characters; 16 questions) the gain is +0.153 [+0.012, +0.323]\*;
elsewhere +0.017. After the collection was updated, the production evaluation reproduced to
the thousandth the figures the dense matrix had predicted.

**Publication-year filter.** On the 15 dated questions, same query with and without the
bounds:

| Configuration | With filter | Without | Δ |
|---|---:|---:|---|
| dense | 0.566 | 0.466 | **+0.100 [+0.029, +0.184]\*** |
| RRF + rerank | 0.452 | 0.260 | +0.193 [+0.080, +0.320]\* |

**Period clause parsed in code.** Three simulated callers on 155 questions: the raw question
(date clause included, no filter), the question passed through `period_bounds` (clause
removed, bounds applied), and the ideal (information need plus gold filter). The parser
found the gold bounds on 15/15 dated questions with **0 false positives** on the 160 others;
on dated questions nDCG@10 went from 0.468 (raw) to 0.552, +0.084 [+0.002, +0.178]\*, against
0.566 for the ideal filter. The remaining gap comes from two questions where a single
trailing question mark moves the gold passage (rank 7 → 12 on one of them): the embedding
is sensitive to punctuation, and the benchmark measures the query as submitted.

**Graph kept apart.** Fusing the passages that mention entities recognised in the question:
RRF with dense −0.105 [−0.135, −0.074]\* on v3 (5 gains, 55 losses), boosting them inside
the dense pool −0.029 [−0.051, −0.007]\*. Only the `exact` family gains (+0.058, n = 10, not
significant). A rare, named entity helps; a generic one ("market", "model") hurts. The graph
is served as exploration tools, not as a ranking signal.

### 7.5 Answers: wrongful abstention, measured without a judge

On the first v3 generation baseline, 58 % of answerable questions ended in
`INSUFFICIENT_EVIDENCE`. The judge-free metric `gold_present_but_abstained` (gold passage
shown to the generator, generator abstained) was **24/57 = 0.421**. Reading the 24 cases
showed two causes: dated questions whose passage headers did not carry the year, and two
prompt rules that contradicted each other (abstain vs give a partial answer). Two patches,
measured separately on the same retrieval:

| | Baseline | + year in passage header | + prompt v2 |
|---|---:|---:|---:|
| `gold_present_but_abstained` | 24/57 = 0.421 | 22/57 = 0.386 | **3/57 = 0.053** |
| paired Δ vs baseline | — | −0.035 [−0.123, +0.053] | **−0.368 [−0.509, −0.228]\*** |
| wrongful abstention, all answerable | 0.585 | 0.569 | 0.215 |
| judged coverage (0–2) | 0.362 | 0.377 | 0.569 [0.43, 0.71] |
| negatives: abstention / fabrication | 20/20 · 0 | 20/20 · 0 | **20/20 · 0** |

The guard held: on the 20 questions whose absence is proven, 20 abstentions and 0
fabrications, before and after. The price is where it was expected: when the gold is not
shown, the generator now answers with what it has more often, and the judge counts 7
ungrounded answers instead of 4.

### 7.6 The deterministic v4 bench and its placebo

v4 (`banc_v4.py`, `familles_v4.py`) removes the judge wherever a program can decide:
`formula` (45 questions; a LaTeX expression is canonicalised before comparison),
`table_cell` (100; a cell value is normalised and scored by a deterministic scorer),
`negative_voisine` (34; the on-topic passage is retrieved, but the quantity asked for is
provably absent from it, from the 50 candidates and from the whole corpus: the system must
notice it by itself) and `negative_v3` (20), 199 in all. It refuses to compare two runs that
differ in any governing field: corpus signature, served window, judge window, models,
question-set version, answer-prompt version. A full run costs about 0.73 USD.

**A placebo arm was drawn**: the same answer contract under another name, a fresh call
cache, the same 199 questions, nothing else changed.

| Quantity | Δ at **zero** change | 95 % CI |
|---|---:|---|
| identical answers | 110 / 199 (55.3 %) | |
| `negative_voisine`, fabrications | **+0.1176** | **[+0.029, +0.235]**, excludes zero |
| `table_cell`, score | −0.0200 | [−0.070, +0.030] |
| `formula`, score | +0.0222 | [+0.000, +0.067] |
| `negative_v3`, everything | 0.0000 | [0.000, 0.000] |

The +0.1176 [+0.029, +0.235] is **exactly** the only significant result that an earlier
batch had reported for `negative_voisine`: it is reproduced by an arm with no variable. The
thresholds that had been computed assuming a deterministic generator were withdrawn for the
negative families. Three rules followed: draw both arms in the same run; a difference whose
interval overlaps the placebo's is not a result, for gains as for losses; and look at
`negative_v3` first, the only family insensitive to regeneration noise.

## 8. What these benchmarks do not measure

Written down so that nobody makes them say it.

- The questions are written by an LLM from corpus passages; they do not follow the
  distribution of a real user's questions.
- The lexical leak is reduced and measured, not removed (median about 0.3–0.4 on v2).
- Pairing documents for multi-document questions is lexical (rare shared vocabulary; using
  the retriever would be circular), which slightly favours BM25: the conservative direction
  for the claim tested.
- A negative question can be answerable under other words: the absence of the *phrase* is
  proven, not the absence of the fact.
- The date clause is a template ("according to sources published in 2022 or earlier"); a
  user would say "recent" or "before the crisis", and the benchmark measures what happens
  after the caller has translated it.
- One answer generator: configurations of retrieval are compared at a constant generator;
  generators are not compared. In production, the generator is the agent calling the MCP
  tools, not the benchmark's prompt.
- The `exact` family has 10 questions: it settles one clear case (a single exact token does
  not call for hybrid retrieval), not more.
- Coverage figures published before the served window was aligned with production (it was
  1,600 characters in the benchmark while production served more) are floors, and are not
  compared with later ones.
