# B2 · Released data against the dataset paper

Checked on 9 October 2026 against the public `localgate` Hugging Face datasets at the revisions the item tables pin. Row counts come from the Hugging Face datasets-server API; card text from each dataset's README.

## 1. The 50,670 graded responses

The released data holds **42,270** graded responses: 8,454 questions × 5 answers.

| Dataset | What it holds | Rows |
|---|---|---|
| `mmlu-pro-open-gemma-generations` | generations, all splits | 29,255 + 4,285 + 8,275 + 455 = **42,270** |
| `judge-panel-verdicts` | judgements (3 per response) | 126,810 = 42,270 × 3 |
| `judge-panel-verdicts` | consensus labels | 8,454 |

The paper's 50,670 is 42,270 + 8,400. The 8,400 are the study run: 280 questions × 30 answers, seeds 6042–6071, vLLM 0.29.0 (`research/docs/questions-and-grading.md`). The code reads them as `labels_study_k30.jsonl` (`research/analysis/input_validation.py`). **Neither those responses nor their roughly 25,200 judge verdicts are in any public dataset**, and no copy exists in this checkout.

Two ways to make paper and data agree:

1. **Release the study run (preferred):** add a `study_k30` split to `mmlu-pro-open-gemma-generations` and to `judge-panel-verdicts`, and publish `labels_study_k30`. The forecasting results rest on these labels, so they are worth releasing. Needs the files from whoever ran the study generation and grading.
2. **Correct the text:** "42,270 released responses (k = 5), plus 8,400 study-run responses (280 × 30) used for the reference labels and not released".

## 2. Panel agreement with the human grades: 90 or 88 of 100

The paper and the `audit-results` card report that the panel agreed with the human majority on **90 of 100** (π = 0.79992). The same 100 responses, scored with the verdicts in the **bulk** release (`judge-panel-verdicts`), give **88 of 100** (π = 0.7596).

Both are correct; they come from two different judge runs on identical inputs. The calibration run (`audit-results/calibration`, 300 calls) re-graded the 100 responses. Compared with the bulk run on the same input:

| Judge | Same verdict | Changed | Changed match ↔ not match |
|---|---|---|---|
| ministral-3-14b | 98 | 2 | 2 |
| gpt-oss-120b | 98 | 2 | 1 |
| qwen3-next-80b | 97 | 3 | 1 |

So **7 of 300 verdicts (2.3%) changed between two calls on identical input**, and four of those flipped the binary outcome, enough to move the panel from 90 to 88.

- **For the paper:** say which run the 90/100 comes from, and that the released bulk decisions agree on 88/100. The study's baseline (task B1) uses the released bulk decisions, so it reports 88.
- **For this study (decision D5):** the LLM baseline is not repeatable either. Repeated-call flip rates for the decision models should be compared with this 2.3%, not with zero.

## 3. Dataset cards against the paper

| Check | Result |
|---|---|
| Counts: 12,032 questions, 8,454 retained, 126,810 verdicts | match the paper |
| Conversion tokens: card 57,621,206; paper "about 58M" (30.7M in, 27.0M out) | consistent (rounding) |
| Panel: three-way unanimity 79.19%, binary 87.4%, Fleiss κ 0.83 | match |
| Calibration 90/100, π 0.79992 | see section 2 |
| **"The repository is private. Authenticate…"** | **wrong on all six cards**: `mmlu-pro-open`, `mmlu-pro-open-gemma-generations`, `judge-panel-verdicts`, `audit-results`, `mmlu-pro-gemma-traces` and `mmlu-pro-closed` are all public. Replace with plain `load_dataset(...)` instructions. |

## Still open

- Filter-vs-human agreement for the dataset paper. The numbers are ready in `research/docs/tdm-b1-note.md`: precision 91.6% (86.5–95.9%), rejections humans accept 52.7% (40.4–64.6%), recall 80.4%, weighted π 0.42. Paper edits are on hold for now.
- Card edits and the study-run release need write access to the `localgate` organisation on Hugging Face.
