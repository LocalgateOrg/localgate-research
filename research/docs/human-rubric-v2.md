# Human rubric v2: one flowchart per pass

Task H3. Replaces the 1,059-word audit rubric for all human labelling in the rerun: re-adjudicating the existing labels (H2) and the fresh gold (H7). It follows the M1 definitions, the same rules as the LLM pipeline v2 (`research/data/prompts_v2.py`) and the decision-model instruments (`research/tdm/instruments.py`, v1).

How it is shown in the annotation apps (H6):

- **The flowchart for the current pass is always visible** beside the item. The worked examples sit one click away.
- **In every pass, "yes" means "fine".** The old rubric mixed the two polarities, and one rater answered a whole question on a flipped scale.
- **Every outcome has one invented example.** None is taken or paraphrased from the existing audit or calibration items, so those items stay clean for checking label reliability.
- **"Unsure" is allowed, but use it rarely.** Unsure items go to adjudication.

---

## Pass 1 · The original question (filter)

You see the multiple-choice question, its options and the correct option.

```
1. Is the answer key usable?
   no  → it looks wrong, lacks units, is a fragment, names option labels ("I and III"),
         or answers a different question. Click "answer key problem" and say why.
         Then still answer 2.
2. Delete the options. Would an expert give this answer, and no other equally correct one?
   no  → it excludes (NOT, EXCEPT), points at options (all of the above), ranks them
         ("best", "most likely" over a list or a case), or asks for one example of many
   yes → numbers within 1%, by a standard textbook method
   yes → explain / define, if this is THE standard textbook answer
   otherwise → yes
```

*(110 words)*

**Examples.** All invented.

| Question | Reference | Key usable? | Converts? | Why |
|---|---|---|---|---|
| Which of the following is not a primary colour of light? | yellow | yes | **no** | excludes |
| Which treatment is most appropriate for this patient? (after a case) | antibiotic X | yes | **no** | ranks options over a case |
| An example of a mammal that lays eggs is | platypus | yes | **no** | one of many (echidnas too) |
| A $2,500 deposit at 3.1% simple interest earns in one year | $78 | yes | **yes** | $77.50 is within 1% |
| Explain why the sky is blue. | the standard textbook account (Rayleigh scattering) | yes | **yes** | the standard answer |
| A car covers 120 km in 2 hours; its average speed is | 6 km/h | **no**: should be 60 km/h | **yes** | the question is fine; the key is not |
| The speed of sound in air at 20 °C is | 343 | **no**: unit missing | **yes** | |

---

## Pass 2 · The rewrite

You see the original question, its options, the correct option, and the open-ended rewrite.

```
1. Same question?
   Would someone answering the rewrite give the same answer as for the original?
   no  → a case, passage, table, data or condition was dropped;
         something was added that the original never said;
         or it became easier or harder
   yes → only references to the options were removed, or the wording changed
2. Is everything it refers to included?
   no  → it points at options, a list, a passage, a table or an earlier problem that is not there
   yes → everything it mentions is in the rewrite (it may still be vague: that was pass 1)
```

*(99 words)*

**Examples.** All invented.

| Original | Rewrite | Same? | Included? |
|---|---|---|---|
| (a 10-line contract dispute) Who wins the case? | Who wins the case? (dispute missing) | **no** | **no** |
| Which of the following is true about photosynthesis? | Which statement about light reactions is true? | **no**: narrowed toward the answer | yes |
| The capital of France is | What is the capital of France? | yes | yes |
| (passage) From the passage, one may infer that | Based on the passage, what can one infer? (passage kept) | yes | yes |

---

## Pass 3 · Grading a response

You see the question, the reference answer and the model's response.

```
1. Find the final answer: what the response concludes, not what it considers on the way.
   "X, though some argue Y" → the final answer is X.
   No conclusion (cut off, empty, only restates, answers another question) → no_answer.
   Two answers as equally final ("X or Y, depending on …") → no_match.
2. Compare with the reference. The reference is right even if you disagree.
   match    → same fact; wording, language and extra correct detail do not matter;
              numbers within 1% in any unit
   no_match → a different fact, or a fact the reference requires is missing
```

*(97 words)*

**Examples.** All invented.

| Reference | Final answer | Verdict | Why |
|---|---|---|---|
| dog | a Labrador | **match** | extra correct detail |
| ad hominem | attacking the person instead of the argument | **match** | another word for the same thing |
| 1,500 m/s | about 1,509 m/s | **match** | 0.6% apart |
| 2 km | 2,000 m | **match** | the same value in other units |
| 6 kg | between 5 and 7 kg | **no_match** | a range is not a value |
| Paris | "Paris, although a few sources point to Lyon" | **match** | a conclusion with a caveat |
| mitosis | "either mitosis or meiosis, depending on the cell" | **no_match** | two answers given as final |
| 100 °C | (stops mid-calculation) | **no_answer** | cut off |

---

## Calibration round (H3, last step)

All three authors label the same 30 items: 10 per pass, from the dev set, and not the 7 items with no majority. Compute Fleiss' κ per question. If any question is below 0.4, revise its wording once and repeat with 30 new items; if it stays below 0.4, report it as descriptive only.
