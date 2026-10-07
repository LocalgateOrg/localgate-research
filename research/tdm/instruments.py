"""DRAFT instruments: what a typed decision model sees for each pipeline stage.

An instrument turns one item-table row into a request body in the shared Jev /
Clef Decisions format: a ``state`` plus named ``questions`` of type noul, choice
or score. The same instrument is sent unchanged to every candidate model.

Wording is a first draft for task I1 and is expected to change on the dev split.
Rules from the original rubric are written into the criteria labels themselves,
because labels outweigh definitions in these models (Azizi et al., 2026). Every
binary question has a ``flipped`` twin with reversed polarity for the
consistency check; the hash of an instrument covers its questions and template.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Callable


@dataclass(frozen=True)
class Instrument:
    name: str
    stage: str  # which item table it reads: filter, rewrite, judging
    state: Callable[[dict], dict]
    questions: dict
    flipped: dict = field(default_factory=dict)  # question name -> flipped question
    notes: str = ""

    def body(self, row: dict, variant: str = "base") -> dict:
        if variant not in ("base", "flipped"):
            raise ValueError(f"unknown variant {variant!r}")
        questions = self.questions
        if variant == "flipped":
            questions = {k: self.flipped.get(k, q) for k, q in self.questions.items()}
        return {"state": self.state(row), "questions": questions}

    def digest(self) -> str:
        blob = json.dumps(
            {"name": self.name, "questions": self.questions, "flipped": self.flipped, "notes": self.notes},
            sort_keys=True,
            ensure_ascii=False,
        )
        return hashlib.sha256(blob.encode()).hexdigest()[:12]


def _options(row: dict) -> str:
    return "\n".join(f"{chr(65 + i)}. {o}" for i, o in enumerate(row.get("options") or []))


# ── Filter: does the question survive losing its options? (D6) ───────────────

FILTER = Instrument(
    name="filter-v0",
    stage="filter",
    state=lambda r: {
        "question": r["question"],
        "options": _options(r),
        "reference_answer": r["reference_answer"],
    },
    questions={
        "single_answer_without_options": {
            "type": "noul",
            "instructions": (
                "Imagine a knowledgeable person sees only the question, with no answer options. "
                "Would they give the reference answer, rather than a different, equally correct answer?"
            ),
            "criteria": {
                "true": "Yes: the reference answer is the single correct answer even without the options "
                "(wording of the answer may differ; data given in the question still counts).",
                "false": "No: without the options the question excludes (NOT, EXCEPT), refers to the options "
                "(all/none of the above), ranks the listed options, or admits many correct answers.",
            },
        }
    },
    flipped={
        "single_answer_without_options": {
            "type": "noul",
            "instructions": (
                "Imagine a knowledgeable person sees only the question, with no answer options. "
                "Would the question become unanswerable or admit a different, equally correct answer?"
            ),
            "criteria": {
                "true": "Yes: without the options the question excludes (NOT, EXCEPT), refers to the options "
                "(all/none of the above), ranks the listed options, or admits many correct answers.",
                "false": "No: the reference answer is the single correct answer even without the options "
                "(wording of the answer may differ; data given in the question still counts).",
            },
        }
    },
    notes="Probability replaces both the judge pass and the 1-10 rescore (D6).",
)

# ── Rewrite audit: same question? self-contained? (D8) ───────────────────────

REWRITE = Instrument(
    name="rewrite-audit-v0",
    stage="rewrite",
    state=lambda r: {
        "original_question": r["original_question"],
        "original_options": _options(r),
        "reference_answer": r["reference_answer"],
        "rewritten_question": r["open_question"],
    },
    questions={
        "same_question": {
            "type": "noul",
            "instructions": "Does the rewritten question ask for the same thing as the original, "
            "so that the reference answer is still the correct answer?",
            "criteria": {
                "true": "Same question: the rewrite only removes references to the options or rephrases.",
                "false": "Changed question: the rewrite adds, drops or alters information so the answer changes.",
            },
        },
        "self_contained": {
            "type": "noul",
            "instructions": "Can the rewritten question be understood and answered without ever seeing "
            "the original options?",
            "criteria": {
                "true": "Self-contained: everything needed to answer is in the rewritten question.",
                "false": "Not self-contained: it still points at options, a list or context that is missing.",
            },
        },
    },
    flipped={
        "same_question": {
            "type": "noul",
            "instructions": "Does the rewritten question ask for something different from the original, "
            "so that the reference answer may no longer be correct?",
            "criteria": {
                "true": "Changed question: the rewrite adds, drops or alters information so the answer changes.",
                "false": "Same question: the rewrite only removes references to the options or rephrases.",
            },
        },
        "self_contained": {
            "type": "noul",
            "instructions": "Does the rewritten question still depend on the original options or on "
            "missing context?",
            "criteria": {
                "true": "Not self-contained: it still points at options, a list or context that is missing.",
                "false": "Self-contained: everything needed to answer is in the rewritten question.",
            },
        },
    },
    notes="D8 open: whether the audit should see options and reference (here: yes, as the human auditors did).",
)

# ── Judging: one 3-way Choice, and the 2-Noul decomposition (D10, D12) ───────

_RUBRIC_MATCH = (
    "match: the final committed answer states the same fact as the reference answer "
    "(wording, language and extra correct detail do not matter; numbers within 1%; a range never matches a value)"
)
_RUBRIC_NO_MATCH = (
    "no_match: the final answer contradicts or omits part of the reference, or hedges between different answers"
)
_RUBRIC_NO_ANSWER = (
    "no_answer: the response never commits to a final answer (cut off, empty, or answers a different question)"
)


def _judging_state(r: dict) -> dict:
    # D12: final_answer only once generations are structured; full response for now.
    return {
        "question": r["question"],
        "reference_answer": r["reference_answer"],
        "response": r.get("final_answer") or r["response"],
    }


JUDGE_CHOICE = Instrument(
    name="judge-choice-v0",
    stage="judging",
    state=_judging_state,
    questions={
        "verdict": {
            "type": "choice",
            "instructions": "Compare the response's final committed answer with the reference answer. "
            "The reference is authoritative: do not solve the question yourself.",
            "criteria": {"match": _RUBRIC_MATCH, "no_match": _RUBRIC_NO_MATCH, "no_answer": _RUBRIC_NO_ANSWER},
        }
    },
    notes="Option rotation is the consistency variant for Choice; see client.rotate_choice.",
)

JUDGE_NOULS = Instrument(
    name="judge-nouls-v0",
    stage="judging",
    state=_judging_state,
    questions={
        "commits": {
            "type": "noul",
            "instructions": "Does the response commit to one final answer?",
            "criteria": {
                "true": "Commits: it states one final answer (even if it is wrong).",
                "false": "Does not commit: cut off, empty, only restates the question, or hedges between answers.",
            },
        },
        "matches_reference": {
            "type": "noul",
            "instructions": "Does the response's final answer state the same fact as the reference answer? "
            "The reference is authoritative: do not solve the question yourself.",
            "criteria": {
                "true": "Same fact as the reference (wording, language and extra correct detail do not matter; "
                "numbers within 1%).",
                "false": "Different fact, missing part of the reference, a range for a single value, or no answer.",
            },
        },
    },
    flipped={
        "matches_reference": {
            "type": "noul",
            "instructions": "Does the response's final answer differ from the reference answer? "
            "The reference is authoritative: do not solve the question yourself.",
            "criteria": {
                "true": "Different fact, missing part of the reference, a range for a single value, or no answer.",
                "false": "Same fact as the reference (wording, language and extra correct detail do not matter; "
                "numbers within 1%).",
            },
        },
    },
    notes="Verdict: no_answer if commits < t1, else match if matches_reference > t2 (thresholds tuned on dev).",
)

INSTRUMENTS = {i.name: i for i in (FILTER, REWRITE, JUDGE_CHOICE, JUDGE_NOULS)}
