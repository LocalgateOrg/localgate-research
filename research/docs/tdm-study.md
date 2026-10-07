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
uv run python -m unittest discover tests
uv run localgate-tdm numeric --items output/tdm-items   # smoke test on calibration items
```

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

> [!WARNING]
> Live calls cost money. Do not run on test items until the instruments are
> frozen and the preregistration is submitted (tasks I6, P2). Until the gold
> dev/test split exists (H2), live runs should use `--ids` with a dev list.
