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


# Consistency variants (paper §3.5, task I1):
#   flipped          every Noul asked with reversed polarity
#   rotated          every Choice with its options in a different order
#   context          one neutral sentence appended to the graded response
#   wrong_reference  the reference replaced by a plausible wrong answer (a distractor
#                    option); a grader that still says "match" grades from its own knowledge
VARIANTS = ("base", "flipped", "rotated", "context", "wrong_reference")
NEUTRAL_SENTENCE = "This answer was written without access to reference material."


@dataclass(frozen=True)
class Instrument:
    name: str
    stage: str  # which item table it reads: filter, rewrite, judging
    state: Callable[[dict], dict]
    questions: dict
    flipped: dict = field(default_factory=dict)  # question name -> flipped question
    notes: str = ""

    def supports(self, variant: str) -> bool:
        if variant == "base":
            return True
        if variant == "flipped":
            return bool(self.flipped)
        if variant == "rotated":
            return any(q["type"] == "choice" for q in self.questions.values())
        if variant in ("context", "wrong_reference"):
            return self.stage == "judging"
        return False

    def body(self, row: dict, variant: str = "base") -> dict:
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant {variant!r}")
        if not self.supports(variant):
            raise ValueError(f"{self.name} has no {variant!r} variant")
        questions = self.questions
        if variant == "flipped":
            questions = {k: self.flipped.get(k, q) for k, q in self.questions.items()}
        if variant == "rotated":
            questions = {k: _rotate(q) if q["type"] == "choice" else q for k, q in questions.items()}
        if variant == "context":
            row = {**row, "response": row["response"].rstrip() + "\n\n" + NEUTRAL_SENTENCE}
            if row.get("final_answer"):
                row["final_answer"] = row["final_answer"].rstrip() + " " + NEUTRAL_SENTENCE
        if variant == "wrong_reference":
            if not row.get("wrong_reference"):
                raise ValueError(f"row {row.get('key')} has no wrong_reference")
            row = {**row, "reference_answer": row["wrong_reference"]}
        return {"state": self.state(row), "questions": questions}

    def digest(self) -> str:
        blob = json.dumps(
            {"name": self.name, "questions": self.questions, "flipped": self.flipped, "notes": self.notes},
            sort_keys=True,
            ensure_ascii=False,
        )
        return hashlib.sha256(blob.encode()).hexdigest()[:12]


def _rotate(question: dict) -> dict:
    """The same Choice question with its options moved one place (last becomes first)."""
    names = list(question["criteria"])
    order = names[-1:] + names[:-1]
    return {**question, "criteria": {n: question["criteria"][n] for n in order}}


def _flip(question: dict, instructions: str) -> dict:
    """The same Noul with the opposite polarity: true and false swap meanings."""
    return {"type": "noul", "instructions": instructions,
            "criteria": {"true": question["criteria"]["false"], "false": question["criteria"]["true"]}}


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
    # Frozen: this text is part of the v0 hash. Rotation is the "rotated" variant.
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

# ═══ v1 drafts (9 Oct 2026) ═════════════════════════════════════════════════
# Rewritten from research/docs/tdm-prompt-review.md. Each rule lives in a short
# criteria label (at most ~40 words) rather than in long instructions, because
# labels outweigh definitions in these models. v0 stays for comparison on dev.

_FILTER_YES = (
    "Yes, one answer. Numbers within 1% and standard textbook methods count as the same answer; "
    "an explain or define question counts if the reference is the standard textbook answer."
)
_FILTER_NO = (
    "No. It excludes (NOT, EXCEPT), points at the options (all of the above), ranks them "
    "(best or most likely over a list or case), asks for one example of many, or needs a unit only the options gave."
)
_KEY_OK = "Usable: it answers the question asked, with units where needed, and looks correct."
_KEY_BAD = (
    "Not usable: it looks wrong, lacks units or magnitude, is a fragment, names option labels "
    "(\"I and III\"), or answers a different question."
)
_filter_single = {
    "type": "noul",
    "instructions": "Remove the answer options. Would an expert asked this question give the reference answer, "
    "rather than a different, equally correct one?",
    "criteria": {"true": _FILTER_YES, "false": _FILTER_NO},
}
_filter_key = {
    "type": "noul",
    "instructions": "Read on its own, can the reference answer serve as the correct answer that responses are graded against?",
    "criteria": {"true": _KEY_OK, "false": _KEY_BAD},
}
FILTER_V1 = Instrument(
    name="filter-v1",
    stage="filter",
    state=FILTER.state,
    questions={"single_answer_without_options": _filter_single, "answer_key_usable": _filter_key},
    flipped={
        "single_answer_without_options": _flip(_filter_single, "Remove the answer options. Would an expert give a "
                                               "different, equally correct answer, or be unable to answer?"),
        "answer_key_usable": _flip(_filter_key, "Read on its own, is the reference answer unusable as the correct "
                                   "answer that responses are graded against?"),
    },
    notes="Review F1-F5, G5: 1% and standard methods; explanation rule; superlatives over a case fail; "
    "answer-key problems are a separate question, not a convertibility verdict.",
)

_rewrite_same = {
    "type": "noul",
    "instructions": "Would a reader answering the rewritten question give the same answer as for the original?",
    "criteria": {
        "true": "Same: only references to the options were removed or the wording changed.",
        "false": "Changed: a case, passage, data or condition was dropped, something was added, "
        "or the question became easier or harder.",
    },
}
_rewrite_points = {
    "type": "noul",
    "instructions": "Does the rewritten question point at something the reader cannot see?",
    "criteria": {
        "true": "Yes: options, a list, a passage, a table or an earlier problem that is not included.",
        "false": "No: everything it refers to is in the rewritten question (it may still be vague; that is a different check).",
    },
}
REWRITE_V1 = Instrument(
    name="rewrite-audit-v1",
    stage="rewrite",
    state=REWRITE.state,
    questions={"same_question": _rewrite_same, "points_at_missing": _rewrite_points},
    flipped={
        "same_question": _flip(_rewrite_same, "Would a reader answering the rewritten question give a different "
                               "answer than for the original?"),
        "points_at_missing": _flip(_rewrite_points, "Is everything the rewritten question refers to included in it?"),
    },
    notes="Review R1, R2: dropped cases and passages count as changed; self-containedness asked as the defect "
    "(points_at_missing = true is a failure, the opposite polarity of v0's self_contained).",
)

_V1_MATCH = (
    "match: the final answer states the same fact as the reference. Wording, language and extra correct detail "
    "do not matter (\"Labrador\" matches \"dog\"); numbers within 1%."
)
_V1_NO_MATCH = (
    "no_match: it states a different fact, leaves out a fact the reference requires (another word for the same "
    "thing is not leaving it out), or gives two answers as equally final."
)
_V1_NO_ANSWER = "no_answer: no final answer: cut off, empty, only restates the question, or answers a different question."
_V1_INSTRUCTIONS = (
    "Compare the response's conclusion with the reference. The reference is authoritative: do not solve the "
    "question. A conclusion with a caveat still commits to it."
)


def _judging_state_final(r: dict) -> dict:
    """Grader input for D12: the structured final answer only (about 250 tokens)."""
    if not r.get("final_answer"):
        raise ValueError(f"row {r.get('key')} has no final_answer (needs schema-constrained generation, G1)")
    return {"question": r["question"], "reference_answer": r["reference_answer"], "final_answer": r["final_answer"]}


def _judge_choice_v1(name: str, state) -> Instrument:
    return Instrument(
        name=name, stage="judging", state=state,
        questions={"verdict": {"type": "choice", "instructions": _V1_INSTRUCTIONS,
                               "criteria": {"match": _V1_MATCH, "no_match": _V1_NO_MATCH, "no_answer": _V1_NO_ANSWER}}},
        notes="Review G3, G4: no_match no longer says 'omits'; hedging rule in the labels. Numbers: matcher arm (D11).",
    )


_commits = {
    "type": "noul",
    "instructions": "Does the response reach one final answer? A conclusion with a caveat counts.",
    "criteria": {
        "true": "Commits: it concludes with one answer, even if the answer is wrong.",
        "false": "Does not commit: cut off, empty, only restates the question, or gives two answers as equally final.",
    },
}
_matches = {
    "type": "noul",
    "instructions": "Does the final answer state the same fact as the reference? The reference is authoritative: "
    "do not solve the question.",
    "criteria": {
        "true": "Same fact. Wording, language and extra correct detail do not matter (\"Labrador\" matches \"dog\"); "
        "numbers within 1%.",
        "false": "Different fact, or a fact the reference requires is missing (another word for the same thing is "
        "not missing).",
    },
}


def _judge_nouls_v1(name: str, state) -> Instrument:
    return Instrument(
        name=name, stage="judging", state=state,
        questions={"commits": _commits, "matches_reference": _matches},
        flipped={
            "commits": _flip(_commits, "Does the response fail to reach one final answer? A conclusion with a "
                             "caveat still counts as an answer."),
            "matches_reference": _flip(_matches, "Does the final answer state a different fact from the reference? "
                                       "The reference is authoritative: do not solve the question."),
        },
        notes="Verdict: no_answer if commits < t1, else match if matches_reference > t2 (thresholds tuned on dev).",
    )


JUDGE_CHOICE_V1 = _judge_choice_v1("judge-choice-v1", _judging_state)
JUDGE_NOULS_V1 = _judge_nouls_v1("judge-nouls-v1", _judging_state)
# D12: the same questions over the structured final answer only.
JUDGE_CHOICE_V1_FINAL = _judge_choice_v1("judge-choice-v1-final", _judging_state_final)
JUDGE_NOULS_V1_FINAL = _judge_nouls_v1("judge-nouls-v1-final", _judging_state_final)

INSTRUMENTS = {i.name: i for i in (
    FILTER, REWRITE, JUDGE_CHOICE, JUDGE_NOULS,
    FILTER_V1, REWRITE_V1, JUDGE_CHOICE_V1, JUDGE_NOULS_V1, JUDGE_CHOICE_V1_FINAL, JUDGE_NOULS_V1_FINAL,
)}
