# Typed decision models study

[Research guide](../README.md) · [Project overview](../../README.md)

Commands for the study that tests typed decision models (Jev, Clef, Clef-flash)
as replacements for the LLM audit and judging stages of MMLU-Pro-Open. The
study plan and TODO timeline live in the paper repository's README; task IDs
below (S3, S5, B1, I2) refer to it.

Install the optional dependency once:

```bash
uv sync --extra tdm
```

## Item tables (S5)

Downloads the released datasets at the revisions cited in the dataset paper and
writes one table per stage: `filter.jsonl` (12,032 questions), `rewrite.jsonl`
(8,454 rewrites) and `judging.jsonl` (42,270 graded responses with their text).
Each row carries the released pipeline's decision and any human labels.

```bash
uv run localgate-tdm items --out output/tdm-items
```

## Baseline against human labels (B1)

Scores the released pipeline against the existing audits with stratum
weighting and stratified bootstrap intervals. No model calls. With rater 2's
self-containedness answers reversed, as in the paper, it reproduces 4.46%
meaning change and 2.16% not self-contained.

```bash
uv run localgate-tdm baseline --items output/tdm-items \
  --reverse-self-contained 2 --out output/tdm-baseline.json
```

## Numeric matcher (I2)

Applies the rubric's 1% rule in code when reference and answer both parse as a
single quantity, and returns `undecided` otherwise. Unit tests:

```bash
uv run python -m unittest discover tests   # numeric matcher and decision client
uv run localgate-tdm numeric --items output/tdm-items   # smoke test on calibration items
```

## Schema-constrained generation (G1)

`--output-format json` makes the answering model write `{"steps": [...],
"final_answer": {"text", "value", "unit"}}`, enforced during decoding by vLLM.
Output cut off at the token limit or invalid against the schema is recorded as
`answer_status: no_answer` and never sent to a judge. Free generation stays the
default, so the released runs are unchanged; the two formats cannot share a run
directory. The dry run prints the manifest, the vLLM chat message and the
equivalent request for an OpenAI-compatible API (`response_format: json_schema`),
without loading a model:

```bash
uv run localgate-data generate open --corpus <corpus.jsonl> --out output/json-pilot \
  --output-format json --limit 5 --dry-run
```

## LLM pipeline v2 (L1)

The study reruns the original LLM pipeline with prompts updated to the M1 definitions
(`research/data/prompts_v2.py`). The released prompts stay frozen as v1 and remain
the default, so every released record still validates.

```bash
uv run localgate-data convert judge --prompts v2 ...     # filter, readmission, rewrite
uv run localgate-grade --prompts v2 ...                  # judge panel
```

v2 adds `answer_key_ok` to the filter output and keeps long cases and passages word
for word: the model rewrites only the final question sentence, and a rewrite much
shorter than a long original gets `length_flag`. Digests: converter v1 `93057f756115`,
v2 `87b2e56f8c37`; grading v1 `e2c597632659`, v2 `feb67c68c95a`.

## Instruments (I1)

`research/tdm/instruments.py` holds the frozen v0 drafts and the v1 drafts written from
the prompt review (`research/docs/tdm-prompt-review.md`): `filter-v1` (adds
`answer_key_usable`), `rewrite-audit-v1` (asks for missing context as a defect),
`judge-choice-v1`, `judge-nouls-v1`, and `-final` versions of both that grade the
structured final answer only (D12). Variants for the consistency checks, passed with
`--variants`: `flipped` (Nouls), `rotated` (Choice options), `context` (a neutral
sentence appended to the response) and `wrong_reference` (a distractor option as the
reference). An instrument skips variants it does not have.

## Decision calls (S3)

Sends a draft instrument to a pinned model and appends every call to a JSONL
log that resumes after interruption. `--dry-run` builds the requests without
keys or network access. Keys come from the environment only.

```bash
uv run localgate-tdm decide --instrument judge-nouls-v0 --items output/tdm-items \
  --model openrouter:typesafe/jev-1.13-20260917 \
  --variants base flipped --human-labelled-only --dry-run \
  --out output/tdm-runs/judge-nouls-v0.jevdry.jsonl
```

| Model spec | Needs |
|---|---|
| `openrouter:typesafe/jev-1.13-20260917` | `OPENROUTER_API_KEY` |
| `workers-ai:@cf/cloudflare/clef` | `CLOUDFLARE_ACCOUNT_ID`, `CLOUDFLARE_API_TOKEN` |
| `workers-ai:@cf/cloudflare/clef-flash` | same |
| `readout:<model>` (open readout, any OpenAI-compatible API with log-probabilities) | `READOUT_BASE_URL` (default: local vLLM), `READOUT_API_KEY` |

> [!WARNING]
> Live calls cost money. Do not run on test items until the instruments are
> frozen and the preregistration is submitted (tasks I6, P2). Until the gold
> dev/test split exists (H2), live runs should use `--ids` with a dev list.
