# Human study

[Research guide](../README.md) · [Project overview](../../README.md)

## Registration and study materials

The [preregistration PDF](../../paper/LARP_Preregistration_Human_Judgment_of_Local_AI_Model_Solvability.pdf) is included alongside the submitted paper.
The [OSF registration](https://osf.io/zx5ej/overview?view_only=71c641fc77a5464aa207c976dfc2f492)
planned 20 reference generations and a 100-judgement validation sample stratified
by agreement scores. The completed study used 30 fresh generations and instead
stratified the validation sample by category, response length and list style.
The panel-agreement gate was added after preregistration. The reported power
calculation assumes equal frequencies across the six success bands, rather than
the empirical frequencies proposed for the planned preregistration update.

H1 and its adequacy threshold were preregistered, but cohort selection and the
bootstrap procedure, along with background analyses, were chosen during analysis.
H2 and H3, background
associations, classifier error limits and electricity scenarios are exploratory.

The [study materials](../../study/README.md) preserve source excerpts and participant
wording, including the description of local execution as private and less
energy-intensive. The study was approved by its supervisor, David Elsweiler.
The [separate study application](https://github.com/LocalgateOrg/Study_LocalModel)
is the runnable interface; participant submissions are a separate input.

## Human forecasts

The primary analysis uses one complete forecast set for each of 30 assigned
participant IDs, giving three forecasts per question and 840 forecasts in total.
Of 34 submissions, three additional complete sets and one incomplete set were
excluded. P04's complete set is included, and missing background fields do not
exclude a participant's forecasts. The supporting analysis also evaluates the
later complete P29 submission as a sensitivity check.

Agreement compares the median human forecast with the reference success band.
Routing instead uses the majority decision at the probability boundary of one
half. The question bootstrap uses 10,000 resamples with seed 20260921 and keeps
the recruited participant group fixed. When rater removal leaves two
forecasts, separate lower- and upper-median calculations retain that ambiguity.
The paper gives the agreement, directional-error and robustness results.

### Background associations

These exploratory analyses ask whether background characteristics relate to
forecast agreement or to overestimation. Bias is the mean signed difference
between a participant’s forecast bands and reference bands, so positive values
indicate overestimation. Each row below fits one predictor to the available
responses using ordinary least squares with HC3 covariance and t-based 95%
intervals, with residual degrees of freedom.

<details>
<summary>Background-association estimates</summary>

| Predictor | Valid n | Agreement slope, κw [95% CI] | Bias slope, bands [95% CI] |
|---|---:|---:|---:|
| Age | 29 | −0.003 [−0.021, 0.015] | −0.022 [−0.086, 0.043] |
| Qualification | 25 | −0.016 [−0.086, 0.055] | 0.029 [−0.349, 0.408] |
| Student status | 27 | −0.038 [−0.338, 0.263] | −0.186 [−0.453, 0.082] |
| GenAI use | 28 | −0.010 [−0.047, 0.027] | 0.021 [−0.194, 0.236] |
| Gender | 29 | 0.017 [−0.066, 0.100] | −0.138 [−0.610, 0.333] |

</details>

All intervals span zero and admit effects in either direction. Models combining
age, qualification, student status and GenAI use were also inconclusive, with
23 complete cases in the primary cohort and 22 when substituting the later P29
submission. These small, imbalanced samples do not establish an absence of
background associations.

Age is measured in years. Completed high-school, bachelor and master/equivalent
qualifications are coded 0, 1 and 2. GenAI use runs from less than weekly to
daily on a 0–4 scale. Student status includes employed and doctoral students,
with employee-only as the comparison. The separate gender model codes recorded
female as 0 and male as 1. These are linear-trend models for the ordinal
predictors, not separate effects for every category. HC3 follows
[MacKinnon and White](https://doi.org/10.1016/0304-4076(85)90158-7).

<details>
<summary>Education coding counts</summary>

| Completed qualification used in regressions | Count | Percentage |
|---|---:|---:|
| High school | 2 | 8.0% of 25 coded responses |
| Bachelor | 15 | 60.0% of 25 coded responses |
| Master or equivalent | 8 | 32.0% of 25 coded responses |
| Nonblank, not coded as completed | 4 | 13.3% of the 30-person cohort |
| Not reported | 1 | 3.3% of the 30-person cohort |

</details>

This predictor differs from the paper’s descriptive education summary, which
includes current study and has 28 coded responses. Three current-bachelor-only
responses and one ambiguous response remain uncoded for completed qualification.
No forecast is excluded because a background field is missing. The
[background-analysis code](../analysis/) is included, although recalculating these
results requires participant responses that are not distributed.

## Draw and assign study questions

After panel reduction produces `labels_open.jsonl`, the study draw selects
questions from the supplied test split. Assignment then distributes each
question to three participants. The split is supplied with the versioned data
and is validated again during classifier preparation.

```bash
uv run localgate-data study-draw \
  --labels /path/to/labels_open.jsonl --split /path/to/split.json \
  --out output/study_prompts.jsonl
uv run localgate-data study-assign \
  --prompts output/study_prompts.jsonl --corpus /path/to/converted.jsonl \
  --out output/study_assignments
```

## Calculate human results

> [!IMPORTANT]
> Participant submissions and background responses are not distributed. These analyses also require the separate 30-generation study labels; the public checkout alone is not sufficient.

Complete [installation](getting-started.md#installation) and use the [output-directory conventions](getting-started.md#output-directories).

The human command calculates descriptive results, inference, robustness and
background associations together. It reads participant files named `P01.csv`
or `larp_P01.csv`, with one complete forecast set for each ID from P01 to P30.
Supplying `--screening`, `--corpus`
and `--allocation` together also writes `human_sample.json`; none is required for
the primary analysis.

```bash
uv run localgate-analysis human \
  --responses-dir /path/to/participant-submissions \
  --labels /path/to/study-labels-k30.jsonl \
  --output-dir "$PWD/output/human"
```

The command writes `human_descriptive.json`, `human_inference.json`,
`human_robustness.json` and `human_background.json`. The descriptive report
feeds the paired [classifier comparison](classifier.md#calculate-classifier-results),
[electricity scenarios](electricity.md#calculate-electricity-scenarios) and
[figure generation](figures.md#regenerate-the-figures).

## Power calculation

Power defaults to equal success-band frequencies and 6,000 repetitions with
seed `20260814`. Passing `--labels` instead uses the supplied empirical
frequencies. This command makes no model calls.

```bash
uv run localgate-analysis power --out output/power.json
```

## Related methods

[Reference-label simulation](questions-and-grading.md#reference-label-simulation)
examines generation-sampling variation. The [human-analysis entry point](../analysis/human.py)
coordinates separate descriptive, inferential, robustness and background modules.
