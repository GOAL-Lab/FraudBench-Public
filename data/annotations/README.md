# `data/annotations/`

Annotations used during FraudBench dataset curation. We do **not** redistribute the underlying Westlaw holding text; only the annotation metadata (IDs, labels, votes), prompt templates, and industry-jury outputs used to elicit them.

## `fraud_presence/`

Holding-level fraud-presence annotations. Each row is keyed by `holding_id = "{case_id}-{holding_idx}-{key}"`, where `key` is the West key number (`k8` = fraudulent misrepresentation, `k15` = fraudulent concealment, `k94` = misrepresentation under contract law).

| File | Description |
| --- | --- |
| `holdings_llm_annotations.jsonl` | LLM-judge annotations for every candidate holding. Columns include per-model run scores (`scores_<model>`, `overall_score_<model>` for `gpt-4o`, `claude-sonnet-4-20250514`, `gemini-2.5-flash`), the cross-model pooled scores `all_scores`, and the final majority-vote label `overall_score`. |
| `holdings_human_annotations.jsonl` | Human expert annotations for a stratified subset of holdings (~330 rows). Columns: `holding_id, case_id, case_citation, key, holding_idx, year, is_fraud_human` with `is_fraud_human` in {`TRUE`, `FALSE`, `INCONCLUSIVE`}. Used to validate the LLM-judge pipeline. |
| `binary_fraud_eval_system_prompt.txt` | System-prompt template used by the LLM judges. Contains placeholders (`!{FRAUD_TYPE}`, `${EXAMPLE_TRUE}`, etc.) that are filled in per West key. |
| `binary_fraud_eval_prompt_vars.json` | Per-West-key variable values that fill in the prompt template (legal definitions, alternate names, "not-this-doctrine" foils). |

## `party_types/`

| File | Description |
| --- | --- |
| `PROMPT_PARTY_CLASSIFICATION.txt` | Prompt template used to classify each named party as `INDIVIDUAL` vs. `BUSINESS`. The resulting party-type labels are baked into `dataset_public.jsonl` (under each scenario's `seed.plaintiff_is_org` / `seed.defendant_is_org`). |

## `industry_categories/`

NAICS-sector annotations for each evaluation scenario. A three-model LLM jury (the models in `fraudbench.utils.globals.MODELS`) classifies every scenario; authors then review the votes and select the final sector label that is baked into `dataset_public.jsonl` as the top-level `naics_sector` field.

| File | Description |
| --- | --- |
| `PROMPT_INDUSTRY_CLASSIFICATION.txt` | Prompt template that asks an LLM to map a scenario (its structured legal facts) to one of the 20 canonical NAICS sector labels. |
| `generate.py` | `jsonargparse` CLI that runs the three-judge annotation pipeline. Every scenario is scored by each model in `globals.MODELS`; the script records per-judge votes, a strict two-of-three majority, and a `no_majority_vote` flag. Per-judge raw-response checkpoints are written beside the run log, and rerunning against an existing output preserves its author-resolved `final_score` values. Example: `uv run data/annotations/industry_categories/generate.py data/final/dataset_public.jsonl --out industry_annotations.json`. |
| `industry_annotations.json` | Jury output for all 159 scenarios. Each record has `summary_id`, the nested `seed`, `judge_scores` (`[{model, industry}, …]`), `majority_vote`, `no_majority_vote`, and `final_score` (the author-resolved sector copied into `dataset_public.jsonl.naics_sector`; may differ from `majority_vote` when judges disagree or when authors override). |

## `llm_judge/`

Human annotations of LLM benchmark responses, used to validate the LLM-as-jury grading pipeline (the public dataset's `scoring_criteria_*` rubrics). Consumed by `analysis/tables/table_judge_irr.py` and `analysis/validation/llm_judge_validation.py` via `fraudbench.analysis.table.judge_irr.load_human_annotations`.

`human_annotations.jsonl` holds every human rating, one JSON object per line. It contains 1,770 ratings on 600 items (200 per LLM setting) from 7 annotators; 498 items have 3 ratings, 66 have 2, 36 have 4. An item may be rated by several annotators, but each annotator rates a given item at most once. Annotators are identified only by a randomly assigned numeric ID; their free-text comments and the presentation order were removed before release.

| Field | Description |
| --- | --- |
| `annotator_id` | Anonymized annotator identifier: a randomly assigned integer from 1 to 7. Required. |
| `setting` | Benchmark setting of the rated response (`closed`, `open` or `agentic`). Informational; matches `obs_data.parquet`. |
| `sample_id` | Response identifier joined to `obs_data.parquet` (`sample_id`). Required. |
| `eval_id`, `run_id` | Provenance of the rated response. Informational. |
| `prompt`, `llm_response`, `fraud_criteria` | The prompt, the model's response and the rubric shown to the annotator. Informational. |
| `fraud_score`, `refusal_score`, `options_score` | The human's boolean rating for each metric. All three are required on every row. |

The validation population covers the closed, open and agentic LLM settings only; human-baseline responses are excluded from the judge IRR table and validation entirely. The validation sample used for post-stratification defaults to every item with at least one human rating. Pass `--validated-samples <csv>` (a `Sample ID` column) to the table script, or `--validated_samples_path` to the validation script, to supply an explicit list instead.
