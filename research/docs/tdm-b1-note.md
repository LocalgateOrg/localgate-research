# B1 · The released pipeline against the human labels

Internal note for decision meeting M1. No model was called; everything comes from the released MMLU-Pro-Open data and its human audits. Reproduce with:

```bash
uv run localgate-tdm baseline --items output/tdm-items --reverse-self-contained 2 --out output/tdm-baseline-v2.json
```

The run reproduces the dataset paper's 4.46% meaning-change and 2.16% not-self-contained rates. All intervals are 95%: stratified bootstrap (10,000 resamples, seed 20261007) for population figures, Wilson intervals for small groups.

## Filter (Stage 1, 241 audited questions with a human majority)

| | Estimate | 95% CI |
|---|---|---|
| Accepted questions humans judge convertible (precision) | 91.6% | 86.5–95.9% |
| Rejected questions humans judge convertible | 52.7% | 40.4–64.6% |
| Recall (convertible questions the filter keeps) | 80.4% | |
| Population share where filter and humans disagree | **21.6%** | 16.8–26.7% |
| Scott's π, population-weighted | **0.42** | 0.28–0.56 |
| Scott's π, audit sample unweighted | 0.27 | |

By cell, agreement between the converter and the human majority:

| Stratum, decision | Audited / population | Agreement | 95% CI |
|---|---|---|---|
| kept_judge, kept | 93 / 7,704 | 93.5% | 86.6–97.0% |
| kept_rescore, kept | 26 / 573 | 65.4% | 46.2–80.6% |
| borderline, kept | 30 / 177 | 90.0% | 74.4–96.5% |
| borderline, dropped | 18 / 106 | 16.7% | 5.8–39.2% |
| drop_rescore, dropped | 62 / 3,456 | 48.4% | 36.4–60.6% |
| drop_rewrite, dropped | 12 / 16 | 25.0% | 8.9–53.2% |

Almost all of the filter's error is in the 3,456 questions rejected after rescoring: humans would have kept about half (roughly 1,880 questions in all). The filter is the weakest stage and has the most room for a cheaper, calibrated decision to add questions. The drop_rescore estimate rests on 62 audits.

## Rewrites (Stage 2, 152 audited rewrites)

| | Weighted failure rate | 95% CI |
|---|---|---|
| Meaning changed | 4.46% | 1.21–8.59% |
| Not self-contained (rater 2 reversed, as in the paper) | 2.16% | 0.00–5.01% |

Failures are rare, so the audit sample has few positives: 8 meaning changes and 3 self-containedness failures. Self-containedness also had κ = 0.121 between raters (D2).

## Grading (100 human-graded responses)

The panel agrees with the human majority on 88% (π 0.76). Its 12 errors are 10 false negatives and 2 false positives, so **the panel is too strict** rather than too lenient. Per judge: gpt-oss 94% (π 0.88), qwen 89%, ministral 88%.

| Reference type | n | Panel accuracy | 95% CI | π |
|---|---|---|---|---|
| numeric | 39 | 94.9% | 83.1–98.6% | 0.88 |
| short text (1–2 words) | 24 | 91.7% | 74.2–97.7% | 0.82 |
| text (3–10 words) | 24 | **70.8%** | 50.8–85.1% | **0.32** |
| long text (>10 words) | 13 | 92.3% | 66.7–98.6% | 0.83 |

| Response length (output tokens) | n | Panel accuracy | 95% CI |
|---|---|---|---|
| short (≤331) | 34 | 100% | 89.8–100% |
| medium (332–846) | 33 | 78.8% | 62.2–89.3% |
| long (>846) | 33 | 84.8% | 69.1–93.3% |

Errors concentrate in mid-length text references and in medium and long responses. Groups are small; read this as where to oversample (H5), not as estimates. By category the samples are 5–15 responses each and show nothing reliable.

On the full 42,270 responses the three judges split on 9.2% of short-text, 10.2% of numeric and 18.8% of long-text references (`panel_disagreement`). The paper's Method section groups these by word count; the item tables group "text" separately. Align the two before the numbers go in the paper (talk/open-questions.md #1).

## Inputs for the power analysis (H4, decision D3)

The paired non-inferiority test needs the share of gold items where candidate and baseline disagree (ψ). Before any candidate exists, the baseline's own error rate e bounds it: about 2e(1−e) if the two err independently, less if their errors overlap.

| Stage | Baseline error e | 95% CI | ψ if independent | n at 5-pt margin | n at 10-pt margin |
|---|---|---|---|---|---|
| Filter | 21.6% | 16.8–26.7% | 0.34 | ~1,060 | ~270 |
| Grading | 12.0% | 7.0–19.8% | 0.21 | ~660 | ~170 |
| Rewrite meaning | 4.5% | 1.2–8.6% | 0.085 | ~270 | ~70 |
| Rewrite self-contained | 2.2% | 0.0–5.0% | 0.042 | ~130 | ~35 |

n is the rough size for one-sided α = 0.025 and 80% power when the candidate is truly as good as the baseline: n ≈ (z₀.₉₇₅ + z₀.₈)² ψ / δ². H4 should replace this with an exact McNemar calculation and the margin chosen for D15.

## What this means for M1

- **D3 (gold-set size):** "about 400 per stage" is enough for the rewrite audit but not for the filter or grading at a 5-point margin if errors are independent. Options: a 10-point margin for the filter, which errs on 1 question in 5 already; a larger filter and grading gold set; or treat the filter as exploratory. Because grading errors concentrate in mid-length text references and longer responses, oversampling those (H5) adds information per label.
- **D2 (self-containedness):** with a 2.2% failure rate and κ = 0.121, a 152-item gold set holds about 3 failures. Even with a better definition, this criterion cannot be tested with useful precision on a few hundred items. That supports dropping it, or keeping it as descriptive only.
- **D1 (majority vs adjudicated):** 7 Stage-1 items have no majority; adjudication matters most for the drop_rescore cell, where human agreement with the converter is near 50%.
