"""Prompt set v2 for the LLM pipeline rerun (task L1).

The original prompts in `research/data/convert.py` and `research/grading/judge.py`
stay frozen as v1, so the released MMLU-Pro-Open data remains reproducible. v2
applies the definitions settled at decision meeting M1, from the prompt review
(`research/docs/tdm-prompt-review.md`), so the LLM arm, the decision models and the
human gold all work to the same rules:

1. numbers within 1% with standard textbook methods count as the same answer;
2. "the answer key looks wrong" is its own outcome, not a convertibility verdict;
3. an explain/define question converts only if the reference is the standard
   textbook answer and can be checked point by point;
4. an answer with a caveat commits to it; two answers given as equally final is
   no_match.

It also changes how long stems are rewritten: the case or passage is kept verbatim in
code and the model rewrites only the question sentence (review R1: 245 rewrites had
cut long stems to under 30% of their length).

Prompts are kept short on purpose; the word counts are checked in the tests.
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field

# ── Converter: filter ────────────────────────────────────────────────────────

SYSTEM_JUDGE = """You decide whether an exam question survives losing its answer options.

You see a question, its options and which option is correct.

The test: with the options deleted, would an expert give the reference answer, rather than a different, equally correct one?

convertible = false when the question:
- excludes rather than identifies ("which is NOT", "EXCEPT", "least likely");
- points at the other options ("all of the above", "both A and C"), unless the stem itself lists the alternatives ("i) ... ii) ...");
- ranks the options: "best", "most likely", "most appropriate" over a list or over a case (a legal defence, a diagnosis, a policy);
- asks for one member of a large category ("an example of X", "a true statement about X");
- needs a unit, convention or list that only the options supplied.

convertible = true when:
- the stem gives the data and a standard method gives the reference. Numbers within 1% are the same answer, and standard textbook methods and conventions are assumed unless the stem says otherwise. Reject only if another standard method gives an answer more than 1% away;
- it asks for several things that each have one answer;
- it asks to explain, describe or define, and the reference is THE standard textbook answer that a grader could check point by point (false if the reference is one of several equally standard answers);
- a superlative is fixed by the stem itself ("which best approximates the ratio ...").

Wording never matters: a later grader accepts any phrasing of the same fact.

Separately, judge the answer key. answer_key_ok = false when the reference looks wrong (it does not follow from the stem's own data, or is false as a claim), lacks units or magnitude, is a fragment, names option labels ("I and III"), or answers a different question. Decide convertible on the question alone: a wrong key is not a reason to reject the question.

Examples (invented):
- "Which of the following is not a phase of matter?" -> convertible false: excludes.
- "Which of these numbers is the largest?", answer "17" -> false: ranks the options.
- "Which is the landlord's best defence?" after a case -> false: ranks the options.
- "The premium on a $60,000 policy at $.2065 per $100 is", answer "$124" -> true: $123.90 is within 1%.
- "Differentiate breathing from respiration", answer the standard textbook contrast -> true.
- "Mixing 50 ml of 0.5 M with 75 ml of 0.25 M H2SO4 gives", answer "35 M" -> convertible true, answer_key_ok false (it is 0.35 M)."""

# ── Converter: readmission ───────────────────────────────────────────────────
# Re-reads the rejects. Deliberately not a copy of the filter prompt, and it must not
# hint that a verdict was already reached (see convert.py).

SYSTEM_RESCORE = """You rate how well an exam question would survive losing its options.

You see a question, its options and which option is correct.

Score 1-10 your confidence that, with the options deleted, an expert would give the reference answer rather than a different, equally correct one.

Score 1 when the question excludes ("which is NOT", "EXCEPT") or points at the other options ("all of the above"), unless the stem lists the alternatives itself.

- 2-4: the stem names a subject but no unique target: one example of many, or "best/most likely" over a list or a case (a legal defence, a diagnosis). Example: "Which of the following is a prime number?", "7" -> 3.
- 5: answerable, but another answer is about as likely. Example: "Which instrument measures air pressure?", "aneroid barometer" -> 5.
- 6-7: subject and property named, some room left. Example: "The symbol for the element with atomic number 26 is", "Fe" -> 7.
- 8-9: self-contained, slight doubt about phrasing or a near-synonym.
- 10: plainly the only answer.

Do not lower the score for: several parts that each have one answer; working from data the stem gives; numbers within 1% of the reference under a standard method (a rounded "$124" for $123.90 is the same answer); an explain or define question whose reference is the standard textbook answer; the reference's wording."""

# ── Converter: rewrite ───────────────────────────────────────────────────────

SYSTEM_REWRITE = """You rewrite an exam question so it stands on its own.

You see the question's text and nothing else: no options and no answer. Long questions come in two parts. CONTEXT is a case, passage or data that the program keeps word for word and places before your question. QUESTION is the sentence you rewrite. Your question may refer to the context ("in the case above", "according to the passage").

Rules:
- Keep the question. Change wording only where it depends on visible options.
- Turn "which of the following", "which of these" and "the statements above" into a direct question.
- Finish unfinished stems as the question they lead to: "The capital of France is" becomes "What is the capital of France?". Never state the answer.
- Add nothing: no hints, no mention of options.

Then say whether it is supported. possible = false if your question names a topic, property or scenario that neither the context nor the question mentions; for example "Which of the following is true?" alone says nothing about what it asks."""

LONG_STEM_CHARS = 400  # stems at least this long are split into context + question
MIN_LENGTH_RATIO = 0.5  # a long stem rewritten shorter than this is flagged


class Verdict(BaseModel):
    """Whether an exam question keeps a single correct answer without its options, and whether its answer key is usable."""

    reason: str = Field(description="one short clause on whether the reference is the only correct answer without the options")
    convertible: bool = Field(description="true only if the reference is the single correct answer once the options are removed")
    answer_key_ok: bool = Field(description="false if the reference looks wrong, lacks units, is a fragment, names option labels or answers a different question")
    failure: Literal["none", "negation", "option_relative", "open_set", "ranks_options",
                     "true_only_among_these", "underspecified_stem"] = Field(description="which failure mode fits; reporting only")


class Rescore(BaseModel):
    """How well an exam question would survive losing its options."""

    reason: str = Field(description="one short clause on how far the stem carries the question on its own")
    score: int = Field(ge=1, le=10, description="1 = certainly cannot survive, 10 = certainly can")
    failure: Literal["none", "negation", "option_relative", "open_set", "ranks_options",
                     "true_only_among_these", "underspecified_stem"] = Field(description="which failure mode fits; reporting only")


class Rewrite(BaseModel):
    """The question sentence rewritten to stand alone, or the finding that the stem supports none."""

    standalone_question: str = Field(description="your rewritten QUESTION, answerable with no options in view; "
                                     "write it even if you then judge it unsupported")
    possible: bool = Field(description="false if your question names anything the context and question never mention")


_SENTENCE_END = re.compile(r"(?<=[.?!:])\s+(?=[A-Z\"'(\[])")


def split_stem(stem: str) -> tuple[str, str]:
    """Split a long stem into (context, question).

    The question is the last sentence, or the last two when the last is very short
    ("Why?"). Short stems are not split: the whole stem is the question.
    """
    stem = stem.strip()
    if len(stem) < LONG_STEM_CHARS:
        return "", stem
    parts = _SENTENCE_END.split(stem)
    if len(parts) < 2:
        return "", stem
    take = 2 if len(parts[-1]) < 40 and len(parts) > 2 else 1
    context = " ".join(parts[:-take]).strip()
    question = " ".join(parts[-take:]).strip()
    return context, question


def rewrite_input(stem: str) -> str:
    """What the rewrite model sees: the stem only, split when it is long."""
    context, question = split_stem(stem)
    if not context:
        return f"QUESTION: {question}"
    return f"CONTEXT (kept word for word): {context}\n\nQUESTION: {question}"


def assemble_rewrite(stem: str, rewritten_question: str) -> tuple[str, bool]:
    """The full open question, and whether its length should be flagged for review."""
    context, _ = split_stem(stem)
    question = rewritten_question.strip()
    full = f"{context} {question}".strip() if context else question
    flagged = len(stem) >= LONG_STEM_CHARS and len(full) < MIN_LENGTH_RATIO * len(stem.strip())
    return full, flagged


# ── Grading rubric ───────────────────────────────────────────────────────────

SYSTEM_GRADE = """You judge whether a response to an exam question states the same answer as a reference answer.

You see a question, the reference answer and a model's response, which may be long, reason step by step, or be cut off.

Find the response's final answer: what it concludes, not what it considers on the way. A conclusion with a caveat ("X, though some argue Y") commits to X. Then compare it with the reference.

The reference is authoritative. Do not solve the question or judge the reasoning; if the final answer disagrees with the reference, it is no_match even if you believe the response.

match: the final answer states the same fact as the reference. Wording, language, formatting and extra correct detail do not matter ("Labrador" matches "dog"). Another word for the same thing is the same fact ("irrelevance" for "irrelevant conclusion"). Numbers match within 1% relative error in any unit convertible from the reference's; a range never matches a single value.

no_match: the final answer states a different fact, leaves out a fact the reference requires, or gives two or more answers as equally final ("X or Y, depending on ...").

no_answer: there is no final answer: cut off before concluding, empty, only restating the question, or answering a different question."""


class Grade(BaseModel):
    reason: str = Field(description="one short clause naming the final answer and how it compares with the reference")
    verdict: Literal["match", "no_match", "no_answer"] = Field(description="match, no_match or no_answer, as defined above")
