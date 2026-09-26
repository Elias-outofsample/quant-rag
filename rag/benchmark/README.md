# Benchmark harness

About a hundred scripts, one per question asked of the system. Methodology and results are
in [docs/evaluation.md](../../docs/evaluation.md); this page is a map. File names and
docstrings are in French, the working language of the project.

Every script reads the private corpus state (question sets, vectors, overlays) and writes a
JSON result file next to itself; none of those files are distributed. The tests in
`tests/` run without them, except the ones listed in [`../conftest.py`](../conftest.py).

## Core

| Script | Role |
|---|---|
| `pipeline.py` | the system under test: retrieval configurations and answer generation |
| `corpus.py` | corpus access and lexical tooling: leak score, proof of absence of a phrase |
| `metrics.py` | retrieval metrics (graded nDCG, recall, MRR), no LLM |
| `judge.py` | LLM grading, and the measurement of the judge: sentinels, noise floor |
| `llm.py` | the only network exit of the harness; token and cost accounting from `prix-modeles.json` |
| `experiment.py` | shared tooling of retrieval experiments: questions, paired intervals, gains and losses |
| `dense_matrix.py` | dense retrieval without Qdrant, proven identical to it (`--check`) |
| `generate_questions.py`, `verify_questions.py` | the question factory; re-verification after a corpus change |
| `run_benchmark.py`, `compare_e2e.py` | end-to-end runs (retrieval, generation, grading) and their per-question comparison |
| `compare_v1_v2.py`, `evaluate.py`, `eval_hybrid.py` | the known-item bench, replayed in the current code |

## Retrieval

| Script | Question |
|---|---|
| `calibrate_router.py`, `eval_router.py`, `eval_routage_exact.py` | can a rule route between dense and hybrid? oracle ceiling |
| `eval_fusion.py` | dense + BM25 fusion variants on the two frozen benches |
| `eval_protocol.py`, `test_period_bounds.py` | which usage rules belong in code: period parsing, graph fusion |
| `eval_rewrite.py` | LLM query rewriting and HyDE against production dense |
| `eval_rerankers.py`, `eval_reranking.py`, `eval_reranker_reponse.py` | stronger cross-encoders on the same pool; does reranking change the answer or only the ranking? |
| `eval_rerank_budget.py`, `eval_rerank_latency.py` | how many candidates to rerank; honest latency |
| `eval_reclassement_selectif.py`, `eval_recherche_approfondie.py` | selective reranking; a "deep search" surface |
| `eval_pool_recall.py`, `eval_pool_rerank.py`, `eval_shadow.py` | recall ceiling of each candidate generator; shadow reranking on real queries |
| `apport_graphe.py`, `eval_angle_mort.py` | what the entity graph adds to answers; questions whose context changes silently |

## Answers

| Script | Question |
|---|---|
| `abstention.py`, `eval_answer_gap.py` | wrongful abstention without a judge; what a missing gold passage costs |
| `eval_characters.py`, `eval_fenetre_generateur.py`, `audit_fenetre_generateur.py` | how much of each passage the generator should see |
| `eval_fenetre_contexte.py`, `eval_assemblage.py` | assembling the context |
| `format_reponse_v3.py`, `garde_reponse.py`, `verdict_reponse.py` | the answer contract: is it followed? non-regression guard; pre-registered verdict |
| `placebo_reponse.py`, `controle_derive_generateur.py` | regeneration noise: the placebo arm |
| `eval_rotation_reponse.py`, `garde_rotation.py`, `recensement_rotation.py` | effect of a corpus rotation at answer level |

## The deterministic v4 bench

| Script | Role |
|---|---|
| `banc_v4.py` | one command, one JSON, and a refusal to compare runs whose governing fields differ |
| `familles_v4.py` | builds the `formula`, `table_cell` and `negative_voisine` families |
| `latex_norme.py`, `score_formule.py`, `selection_formules.py` | canonical LaTeX; formula scoring; candidate selection |
| `norme_valeurs.py`, `score_tableau.py`, `selection_tableaux.py` | canonical numeric values; table-cell scoring and its rejects |
| `score_citation.py` | does `[n]` point to a passage that states what is claimed? no judge |
| `survie_questions.py`, `resolution_banc.py`, `asymetrie_seuil.py`, `diagnostic_formules.py` | question survival across corpus changes; the smallest effect the bench can see |

## Corpus representation

| Script | Question |
|---|---|
| `audit_decoupage.py`, `audit_rechunk.py`, `audit_contexte_structure.py`, `pilote_granularite.py` | chunk boundaries and granularity |
| `recollage.py`, `audit_recollage.py`, `bancs_latex.py`, `sonde_latex.py` | re-joining LaTeX split by the PDF parser |
| `sonde_troncature.py`, `eval_debalisage.py` | the embedding window; stripping `<sub>`/`<sup>` tags |
| `eval_representation.py`, `verdict_representation.py`, `eval_dense_candidat.py`, `controle_identite_c1c2.py` | a candidate collection against production, with a pre-registered hierarchy of criteria |
| `audit_pertes.py`, `reancrage_characters.py`, `gold_ancrage.py` | where the served path loses questions; gold anchored on text rather than chunk ids |

## Output contract and infrastructure

| Script | Role |
|---|---|
| `mesure_contrat.py`, `controle_citation.py` | what the server serves to the 155 questions; 100 citations, 50 altered by one word |
| `check_bm25.py`, `check_readers.py`, `build_bm25_titles.py` | the served index matches the corpus state; corpus invariants |
| `parite_backend.py`, `verif_exactitude_serveur.py`, `latence_backends.py` | embedded vs server Qdrant: same ranking? latency |
| `demo_concurrence.py`, `temoin_m4.py`, `echelle_reconstructions.py` | concurrency; rebuild from artefacts; cost at 1,000 and 3,000 documents |
| `controles_operationnels.py`, `portes_fonctionnelles.py` | every failure must fail loudly |

## Human-validated cohort

`human-v1/` builds review packets from the corpus, validates the reviewers' decisions and
checks the cohort before it is inserted in a bench (`cohorte1.py`, `paquet_revue.py`,
`valider.py`, `verifier_cohorte1.py`, `offsets.py`).
