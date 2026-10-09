# Prompt review: where humans and the pipeline disagree, and what to change

Input for task I1 (instruments), H3 (self-containedness definition) and H6 (annotation apps). Based on every audited item where the human majority and the released pipeline disagree:

- **Filter:** 74 of the 241 Stage-1 items with a human majority.
- **Rewrites:** the 8 meaning changes and the self-containedness splits among the 152 Stage-2 items.
- **Grading:** the 12 of 100 calibration responses where the panel and the humans disagree.

The converter's stated reasons come from `conversion_record`; the judges' reasons from `judge-panel-verdicts`.

Prompt lengths today:

| Prompt | Who reads it | Words |
|---|---|---|
| Filter (`SYSTEM_JUDGE`) | converter | 813 |
| Readmission (`SYSTEM_RESCORE`) | converter | 630 |
| Rewrite (`SYSTEM_REWRITE`) | converter | 260 |
| Grading rubric | 3 judges | 261 |
| Audit rubric, both passes | human annotators | 1,059 |

Every number below can be rechecked with the commands in `research/docs/tdm-study.md`.

---

## 1. Filter: five ways the definition of "converts" is unclear

### F1 · Numbers: the filter rejects answers the grader would accept

The filter rejects calculation questions whenever the exact result depends on rounding or on a standard textbook method. Humans accepted these, most of them unanimously. Examples:

| Item | What the filter said | Humans |
|---|---|---|
| 429 premium on a $60,000 policy, reference "$124" | exact value is $123.90, so "$124 depends on unstated rounding" | yes, yes, yes |
| 585 homeowners premium, reference "$207" | exact value is $207.36 | yes, yes, yes |
| 12014 heat loss from a duct, reference "93.64 Btu/hr-ft" | "omits emissivity, correlation…" | yes, yes, yes |
| 6910 Durbin–Watson 1.53, reference "inconclusive" | "depends on the significance level" | yes, yes, yes |
| 3814 EPR lines, reference "3" | "standard counting gives 21" | yes, yes, yes |

$123.90 and $124 differ by 0.08%, and the grading rubric accepts anything within 1%. The two stages use different standards for the same numbers.

**Change:** tell the filter what the grader accepts. "Numbers within 1% count as the same answer. Assume standard textbook methods and conventions unless the question says otherwise. Reject only if a *different standard method* gives an answer more than 1% away."

### F2 · Wrong answer keys are not a "convertibility" question (9 items)

Humans said yes and the filter said no, but the real problem is that the reference answer is wrong or mislabelled:

| Item | Problem |
|---|---|
| 4486 | 0.5 M and 0.25 M H₂SO₄ mixed gives 0.35 M; the reference says "35 M" |
| 4188 | the data give [NH₃] = 6.7 × 10⁻⁴ M; the reference says 6.7 × 10⁻³ |
| 9333 | "Fermi temperature in eV", reference 81,600, which is in kelvin |
| 3801 | asks for litres of water; the reference is "603.0 g" |
| 8743 | asks for x₃; the reference is x₄ |
| 8617 | asks for "the number" of positive real zeros; the reference is a Descartes'-rule pair |
| 2207, 10730, 6021 | the reference is disputed as a claim about the world (the filter's reading; not verified) |

Humans judged the *question* (convertible), the filter judged the *item* (would a knowledgeable person give this reference?). Neither instruction says which.

**Change:** make "the answer key looks wrong" a separate outcome in all three places (filter, decision-model instrument, human rubric). Items flagged this way go to a fix-or-drop list, not into the convertibility error rate. Today these 9 items count as filter errors.

### F3 · Explanation and definition questions: humans are inconsistent too (12 items)

"Differentiate breathing and respiration", "What is the distinction between an aptitude and an achievement test?", "What prevents the stomach from digesting itself?", "Explain four evolutionary events in the Paleozoic". The filter rejects these as "many correct explanations". Humans accepted most of them 2–1 or 3–0.

The opposite direction also happens. "Crowding out", "What is collusion?" and "Use capital deepening to explain…" were kept by the filter, and humans rejected them 2–1 or 3–0.

The rule "a different *fact* would answer it just as correctly" does not say what to do when the reference is one standard textbook explanation among several.

**Change (a decision for the team, see D2/D6):**

> An explain/describe/define question converts **if** the reference gives the standard textbook answer **and** a grader could check a response against it point by point. It fails if the reference is one of several equally standard answers, or is only a fragment of a full explanation.

Two worked examples (one each way) should go in both the human rubric and the instrument.

### F4 · The readmission pass is too lenient on superlatives and inferences

| Cell | Agreement with humans |
|---|---|
| kept_judge (accepted on the first pass) | 93.5% |
| **kept_rescore (readmitted by the second pass)** | **65.4%** (9 of 26 audited readmissions rejected by humans) |

The failures are mostly "best/most" questions and "what can be inferred from the passage":

- 1146 "most applicable basis for the constitutional challenge"
- 1634 "best defense"
- 2716 "best predictor"
- 5896 "most reasonable action"
- 6592 "most suggestive"
- 4879, 4888 passage inferences

The rescore prompt's long carve-out ("A superlative is not automatically a ranking…", 109 words) and "the middle is where most doubtful questions belong" pull these up to scores of 5–9.

**Change:** keep the carve-out to one sentence with the K-ary-tree example, and add a counter-example: "'best defense' / 'most likely cause' over a legal or clinical vignette ranks options and fails". For the decision models this becomes moot: one probability replaces both passes (D6), and the threshold is set on dev data.

### F5 · "An example of X" questions

Zellweger "is a peroxisomal disorder", "a principal source of Earth's internal energy", "a specific kind of fallacy". The filter is right by the rule (open set), but humans accepted 2–1. Annotators seem to accept when the reference is the *most famous* member. No change to the rule; add one worked example to the human rubric so annotators apply it the same way.

### Length

The filter prompt is 813 words: three rules, seven illustrations, five "does not fail" notes and seven examples. The decision models weigh option labels more than definitions (Azizi et al., 2026), so most of that text would be ignored there anyway. For the instruments, the rules belong in **short criteria labels** (≤ 40 words each), not long instructions.

---

## 2. Rewrites

### R1 · Rewrites that drop the case or passage (largest single problem)

6 of the 8 meaning changes the audit found are rewrites that kept only the last sentence of a long question. Item 1029 is a full traffic-accident case, rewritten as "What is the appropriate judgment in the case?" with no facts left.

This can be counted on all 8,454 rewrites:

| | Rewrites |
|---|---|
| Original ≥ 400 characters | 1,599 |
| Rewrite < 30% of the original's length | **245 (2.9% of the dataset)** |
| …of those, in the audit sample | 6, **all 6 judged meaning-changed** |
| By subject | law 167, history 49 (mostly "This question refers to the following information"), health 20 |

The rewrite prompt already says "Never compress a case description". The model ignores it on long stems.

**Change (cheap and certain):**

1. **In code, not in the prompt:** keep the passage or case text verbatim, and let the model rewrite only the final question sentence.
2. **Flag any rewrite under 50% of a long original's length.** A rule like this costs nothing and would have caught every one of these.
3. The 245 rewrites should be fixed for MMLU-Pro-Open v1.1 (task W4). Their effect on the published meaning-change rate is worth checking: the audit drew 6 of them and found 6 failures.

### R2 · Self-containedness: the question itself confused annotators

- **Rater 2 disagreed with both other raters on 136 of 152 items**, the pattern of a reversed scale (the paper corrects for it). The question "can it be answered without the options?" puts yes = good, while pass one's questions put yes = good for a different property. Annotators had to keep switching what a "yes" meant.
- **The remaining splits are mostly not about self-containedness.** For example, 7143 "What is collusion?", whose rewrite is identical to the original, was marked not self-contained by rater 1. Annotators were re-judging *whether the answer is unique* (pass one again) rather than whether the rewrite points at missing options or context.

**Change for H3:**

> Ask the defect directly, with yes always meaning "there is a problem": **"Does the rewrite point at something the reader cannot see? (options, a list, a passage, a table, an earlier problem)"**

Give three examples:

- **yes:** "Which of the following is a prime?" (points at a list)
- **yes:** "Based on the passage, what did Gorbachev argue?" with the passage dropped
- **no:** "What is collusion?" (may be vague, but that is pass one's question, not this one)

---

## 3. Grading: the panel's 12 errors against the humans

| Pattern | Items | Judge behaviour |
|---|---|---|
| **G1 Numbers within 1% called different** | 9325:4 (921 vs 924, 0.3% apart), 608:0 ($605.92 vs $606) | ministral and qwen state "differs by more than 1%"; all three rejected $605.92 |
| **G2 Unit conversion done wrong** | 11702:1 (1.67 × 10⁻⁸ J/m³ vs 1/6 erg/m³, which are equal) | qwen converted wrongly; ministral called it a contradiction |
| **G3 Wording penalised** | 11248:1 ("Irrelevance" vs "irrelevant conclusion"), 2581:1 ("history" vs "a simultaneous event") | ministral: "omits the reference's specific term" |
| **G4 Hedged endings** | 10483:0 ("all three work… if forced, Two's complement"), 897:2, 5183:3, 7361:4 | judges read the last sentence; humans read the overall conclusion; humans themselves split 2–1 |
| **G5 Reference names option labels** | 10483:0 "I, II, and III", 6908:3 "(i) and (iii) only" | the grader must map labels the response never saw |

**Changes:**

- **G1 and G2 → code, not prompts.** Arithmetic tolerance and unit conversion are where LLM judges fail. The numeric matcher (I2) decides these exactly; decision D11 should make it the default for numeric references, not just an ablation.
- **G3 → reword the no_match label.** "Contradicts or *omits part of* the reference" invites strictness on wording. Use: "states a different fact, or leaves out a fact the reference requires (a different word for the same thing is not leaving it out)". Add the rubric's own example ("Labrador" matches "dog") to the instrument's label, where these models actually read it.
- **G4 → one explicit rule** for humans and models alike: "The final answer is what the response concludes. A conclusion with a caveat ('X, though Y is sometimes argued') commits to X. Two or more answers offered as equally final ('X or Y depending on…') is hedging: no_match." Structured JSON answers (G1, decision D9) remove most of these cases.
- **G5 → check the reference at the filter stage.** The audit rubric's second pass-one question (`answer_stands_alone`) catches exactly this, but nothing in the pipeline asks it. Add it as a second Noul to the filter instrument.

---

## 4. For the human annotation apps (H3, H6)

The rubric is 1,059 words across two passes. Annotators did ~15 hours each, and drift from fatigue is a known risk (H7). Suggested shape:

1. **One flowchart per pass, at most about 120 words**, always visible beside the item. The full rubric with examples sits one click away.
2. **"Yes" always means the same thing within a pass**: "yes = there is a problem", or always "yes = fine", never mixed.
3. **Every outcome in the flowchart has one example item.**
4. **A separate "the answer key looks wrong" button** that does not count as a convertibility verdict (F2).

Draft flowchart for pass one, to edit at M1:

```
Does the question exclude ("NOT", "EXCEPT") or point at the other options ("all of the above")?
   yes → does not convert
Is the reference answer itself wrong or unusable alone (bare number without units, "I and III")?
   yes → flag "answer key" (separate button) and move on
Without options, would an expert give this answer?
   - numbers: within 1%, standard textbook method → yes
   - "best/most" over a vignette or a list → no
   - "an example of X" where many examples exist → no
   - explain/define: yes only if this is THE standard textbook answer
   → converts / does not convert
```

---

## 5. What changes in the I1 instruments (draft v1)

Implemented in `research/tdm/instruments.py` as `*-v1` next to the unchanged `*-v0`, so both can be compared on dev data (I4):

- **Filter v1:** criteria labels carry the 1% / standard-method rule (F1), the superlative-over-a-vignette counter-example (F4) and the explanation rule (F3). Adds a second Noul, `answer_key_usable` (F2, G5).
- **Rewrite v1:** "same question" explicitly fails when a case, passage or data was dropped (R1). "Self-contained" is asked as the defect (R2).
- **Judging v1:** reworded no_match label (G3); explicit hedging rule (G4). Numbers are left to the matcher arm (G1, G2).

Plus the I1 consistency variants:

- **Choice:** answer options rotated.
- **Grading:** a neutral sentence added to the response (context flip).
- **Grading:** a wrong reference taken from a distractor option.
- **Grading:** final answer only vs full response (D12).
