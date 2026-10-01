# Questions and grading

[Research guide](../README.md) · [Project overview](../../README.md)

This guide follows the questions from their original multiple-choice form to
open-ended answers and reference labels.

Conversion retained 8,454 of the 12,032 MMLU-Pro questions. Removing answer
options requires preserving both what a question asks and whether it can stand
alone. The text of the original correct option remains the grading reference.
The [conversion protocol](https://huggingface.co/datasets/localgate/mmlu-pro-open/blob/6bc069a7b1b83d93ee352dadc9fb49b59cb4dc5d/protocol.md)
contains the two filtering templates, rewrite template and response schemas.
The converter used `deepseek-v4-flash` at temperature 0, with schema validation
and up to three output-validation retries. A preliminary AI-assisted comparison
of three converter models on 93 questions, supervised by one author, found that
Flash changed six filtering decisions between repeated runs (6.5%), compared
with 16 for Mistral Small (17.2%), supporting its selection.

## Generation settings

Both runs served the standard Gemma-4-E2B-it checkpoint through vLLM with an
8,192-token context limit, a 6,144-token output limit, temperature 1.0,
top-p 0.95 and top-k 64. They used the default chat template without a system
message. The standard checkpoint differs from a quantised consumer-device
version, so the success estimates concern this serving setup.

| Setting | Screening | Study reference |
|---|---|---|
| Questions | 8,454 | 280 |
| Generations per question | 5 | 30 |
| Responses | 42,270 | 8,400 |
| Seeds | 5042–5046 | 6042–6071 |
| vLLM | 0.25.1 | 0.29.0 |
| PyTorch | 2.11.0 | 2.13.0 |
| Transformers | 5.14.1 | 5.17.0 |

The disjoint seeds provide fresh draws, but the software environments also
changed. Differences between the two sets of estimates cannot therefore be
attributed to the number of generations alone.

## Human validation and reference uncertainty

Three authors assessed 248 original questions for answerability without options
and suitability of the original answer as a single grading reference. They then
assessed 152 retained rewrites for preserved meaning and self-containedness.
The paper reports the sampling design and its consequences for interpretation.

| Rewrite group | Retained population | Reviewed | Meaning rejected | Self-containedness rejected |
|---|---:|---:|---:|---:|
| Directly accepted | 7,704 | 96 | 4 | 2 |
| Accepted after reconsideration | 573 | 26 | 2 | 1 |
| Borderline, then retained | 177 | 30 | 2 | 0 |

Population weighting gives rejection estimates of **4.46% for meaning** and
**2.16% for self-containedness**, with Fleiss’ κ of 0.572 and 0.121 respectively.
The latter incorporates one annotator’s confirmed reversal of yes/no responses.
The [audit dataset](https://huggingface.co/datasets/localgate/audit-results/tree/a4dcb995522f89cc7c0b695624731c94cfb3df1d)
preserves the original ratings and uncorrected summaries. The
[audit command](questions-and-grading.md#conversion-and-judge-audits) applies the confirmed correction
only to self-containedness; the meaning assessment is unchanged.
The 5% acceptance rule concerned a point estimate, not an upper confidence bound.
One further meaning rejection in the first group would raise its weighted
estimate to approximately 5.41%.

A third validation compared three authors’ judgements of 100 model responses
with the automated grading panel. The panel agreed with the human majority on
90 responses (Scott’s π = 0.79992), narrowly missing the later local protocol’s
0.80 gate. It produced 44 true positives, 46 true negatives, 2 false positives
and 8 false negatives. In the bulk output, two majority decisions changed,
reducing agreement on the same examples to 88% (π = 0.7596). This is a consistency
comparison on calibration examples, not independent validation. The rubric and
sampling qualifications remain in the paper.

## Prepare questions and generate answers

Begin with [installation and input access](getting-started.md). All commands run from the repository root.

> [!WARNING]
> Live conversion and judge commands make paid API calls. Replaying saved logs and reducing completed verdicts do not.

The `localgate-data` commands prepare the inputs used by grading, training and
analysis. Start from the [versioned resources](../../README.md#data-and-models) to
repeat the reported calculations. Creating new answers requires a suitable GPU;
conversion and grading use paid APIs only when their live commands are invoked.

The original multiple-choice logs can be reduced into question-level counts.
Completed conversion logs can likewise be assembled without calling a model:

```bash
uv run localgate-data closed-labels /path/to/multiple-choice-run \
  --out output/labels.jsonl

uv run localgate-data convert replay \
  --judge-log /path/to/convert_judge.jsonl \
  --rescore-log /path/to/convert_rescore.jsonl \
  --rewrite-log /path/to/convert_rewrite.jsonl \
  --out output/converted.jsonl
```

For a new conversion, prepare an answer key from an explicit revision of
[TIGER-Lab/MMLU-Pro](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro), then
check its IDs and option texts against the labelled corpus. Set
`MMLU_PRO_REVISION` to the dataset commit corresponding to your inputs.
The conversion stages preserve the filter, rescoring and answer-blind rewrite
prompts. Each live stage writes an append-only log and resumes completed items.

```bash
uv sync --python 3.13 --extra corpus --locked
uv run localgate-data corpus prepare \
  --revision "$MMLU_PRO_REVISION" --out output/answer_key.jsonl
uv run localgate-data corpus check \
  --answer-key output/answer_key.jsonl --labels output/labels.jsonl

# These three commands make paid converter calls.
uv run localgate-data convert judge \
  --labels output/labels.jsonl --answer-key output/answer_key.jsonl \
  --out output/convert_judge.jsonl
uv run localgate-data convert rescore \
  --labels output/labels.jsonl --answer-key output/answer_key.jsonl \
  --judge-log output/convert_judge.jsonl --out output/convert_rescore.jsonl
uv run localgate-data convert rewrite \
  --labels output/labels.jsonl --judge-log output/convert_judge.jsonl \
  --rescore-log output/convert_rescore.jsonl --out output/convert_rewrite.jsonl
```

Use `convert replay` with those three logs to assemble the converted corpus.
The converter defaults to `deepseek:deepseek-v4-flash`; provider credentials
belong in the local environment.

Generation uses separate environments because the two recorded runs used
different inference-library versions. The small requirements files pin their
principal runtime dependencies. Use these environments for local generation
with trusted inputs. They are separate from the classifier and ordinary
analysis installation. Set `GEMMA_REVISION` to the full model commit SHA
for the checkpoint being evaluated; new runs record that revision.

```bash
uv venv output/generation-initial-env --python 3.13
uv pip install --python output/generation-initial-env/bin/python \
  -r research/data/environments/initial.txt
uv venv output/generation-study-env --python 3.13
uv pip install --python output/generation-study-env/bin/python \
  -r research/data/environments/study.txt

# Run from the repository root. Remove --dry-run to generate new answers.
output/generation-initial-env/bin/python -m research.data generate mc \
  --revision "$GEMMA_REVISION" --out output/multiple-choice-run --dry-run
output/generation-initial-env/bin/python -m research.data generate open \
  --corpus /path/to/converted.jsonl --profile initial \
  --revision "$GEMMA_REVISION" --out output/open-run --dry-run
output/generation-study-env/bin/python -m research.data generate open \
  --corpus /path/to/study_prompts_corpus.jsonl --profile study \
  --revision "$GEMMA_REVISION" --out output/study-run --dry-run
```

Dry runs construct the requests without loading a model or writing results.
Live generation requires a full model commit SHA. Custom model code is disabled
by default; `--trust-remote-code` explicitly enables it when needed for a trusted
checkpoint.
The initial open-ended profile uses five seeds, 5042–5046; the study profile
uses thirty, 6042–6071. Multiple-choice generation uses five seeds, 42–46.
The commands validate saved run settings and completed records before resuming.
New stochastic generations need not reproduce the recorded answers exactly.
The generation dependency sets resolve independently; package verification uses
saved responses and offline request construction, not a fresh GPU generation run.

## Grading

The [grading implementation](../grading/judge.py) includes the actual rubric,
structured verdict schema, provider settings and three-judge reduction.
Existing verdict files can be reduced without provider access:

```bash
uv run localgate-grade \
  --corpus /path/to/open-corpus.jsonl \
  --closed-labels /path/to/closed-labels.jsonl \
  --panel /path/to/judge1.jsonl /path/to/judge2.jsonl /path/to/judge3.jsonl \
  --expected-k 5 --out output/labels_open.jsonl
```

To obtain new verdicts, configure credentials for the chosen provider and run
one judge per invocation. This makes paid model calls. For the GPT-OSS panel
member, the supplied settings use low reasoning effort. The generation-run
directory must contain `rep*/generations.jsonl` files:

```bash
uv run localgate-grade \
  --judge bedrock:openai.gpt-oss-120b-1:0 \
  --corpus /path/to/open-corpus.jsonl \
  --generations /path/to/generation-run \
  --reasoning-effort low --output-mode prompted \
  --out output/judges
```

Use `--expected-k 30` when reducing the study generations. See `--help` for
resuming incomplete judge runs and panel options. Credentials belong in the
local environment, never in source files.

## Prepare audit samples

```bash
uv run localgate-data audit-sample \
  --corpus /path/to/converted.jsonl --labels /path/to/labels.jsonl \
  --out output/conversion_sample
uv run localgate-data calibration-sample \
  --generations /path/to/five-generation-run --corpus /path/to/converted.jsonl \
  --out output/calibration_sample
```

The audit draw produces blank rating packages and their question key. Its
stratum quotas yield 248 original questions and 152 retained rewrites for the
study inputs. The calibration draw produces 100 responses from 90 questions.
Completed ratings are separate inputs to the
[audit analyses](#conversion-and-judge-audits).

## Compare converter models

The conversion command reads the original blind question order, human verdicts
and two filtering runs per model. Repeat `--model` for each model being compared.

```bash
uv run localgate-analysis conversion \
  --blind-order /path/to/blind_order.json \
  --human-verdicts /path/to/human_verdicts.json \
  --model flash /path/to/flash-run1.jsonl /path/to/flash-run2.jsonl \
  --model mistral /path/to/mistral-run1.jsonl /path/to/mistral-run2.jsonl \
  --model pro /path/to/pro-run1.jsonl /path/to/pro-run2.jsonl \
  --output output/conversion.json
```

## Conversion and judge audits

The `conversion-audit` command analyses the three raters' decisions on the 248
original questions and the 152 retained rewrites. Population weights are
calculated from the converted corpus.

The reported self-containedness result uses the confirmed reversal of Noah's
yes/no column. The correction below names that column and the SHA-256 of the
original rating file. A mismatching file is rejected. Only the in-memory ratings
change; the report records the correction and the source file is left intact.
Omitting the option analyses the original labels instead.

```bash
uv run localgate-analysis conversion-audit \
  --corpus /path/to/converted.jsonl --key /path/to/audit/KEY.json \
  --stage1 federico=/path/to/audit/verdicts_federico.json \
  --stage1 noah=/path/to/audit/verdicts_noah.json \
  --stage1 samuel=/path/to/audit/verdicts_samuel.json \
  --stage2 federico=/path/to/audit/conversion_federico.json \
  --stage2 noah=/path/to/audit/conversion_noah.json \
  --stage2 samuel=/path/to/audit/conversion_samuel.json \
  --reverse-column noah:self_contained:a7257e43494a1c0119a87b4be87f280d343906e939c9ca832bb94adcd34e776c \
  --output output/conversion_audit.json

uv run localgate-analysis judge-calibration \
  --items /path/to/calibration/items.jsonl \
  --human federico=/path/to/calibration/annotate_federico.json \
  --human noah=/path/to/calibration/annotate_noah.json \
  --human samuel=/path/to/calibration/annotate_samuel.json \
  --judges /path/to/original/judge1.jsonl /path/to/original/judge2.jsonl /path/to/original/judge3.jsonl \
  --output output/judge_calibration.json
```

The calibration command reports each judge and the panel against the human
majority. Use the original calibration verdicts for the 90% result. Repeating
the command with the three bulk-grading files selects the same calibration
responses and gives the 88% consistency result. Neither invocation makes model
calls. A completed report with a failed agreement gate returns exit status `1`;
this is the expected outcome for both reported panels, whose Scott's π is below
0.80. Invalid inputs are rejected separately.

## Reference-label simulation

The reference-label simulation fits the five-generation counts on a 201-point
binomial-mixture grid and uses seed `20260728`. Its simulated weighted agreement
rounds to .899. Add `--closed-labels /path/to/labels.jsonl` to include complete,
retained and dropped corpus counts in its summary. This command makes no model calls.

```bash
uv run localgate-analysis reference \
  --labels /path/to/labels_open.jsonl \
  --kappa-ceiling --study-k 30 --study-n 280 --sims 6000 \
  --out output/reference.json
```

## Next steps

The resulting labels feed [study sampling](human-study.md#draw-and-assign-study-questions)
and [classifier preparation](classifier.md#selected-classifier-refit).
