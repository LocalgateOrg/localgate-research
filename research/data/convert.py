"""Convert labelled questions through judge, rescore, and answer-blind rewrite stages."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import random
import re
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field
from pydantic_ai import Agent, PromptedOutput

from research.analysis.data_io import InputError, read_records, sha256
from research.data import corpus
from research.grading import batch, providers

# Converter and grading panel must use different model lineages.
DEFAULT_MODEL = "deepseek:deepseek-v4-flash"
READMIT_AT = 5
# Keep limited runs reproducible.
LIMIT_SEED = 20260801
# Allow two complete transport retry cycles before recording an item timeout.
ITEM_TIMEOUT_SECONDS = 2 * (providers.REQUEST_TIMEOUT + providers.MAX_WAIT_SECONDS)

# Phrases that only mean something while the options are on the page. Used to report
# suspect rewrites to the audit, never to change a verdict.
DANGLING = re.compile(
    r"\b(?:of the following|of these|following statements|"
    r"statements above|listed above|options above)\b",
    re.IGNORECASE,
)

# Named failure modes, for reporting only. Recorded *after* the verdict in every schema
# below: fields are generated in order, and a label placed first would drive the decision
# instead of describing it. Nothing in this module reads `failure` to decide anything.
Failure = Literal[
    "none",  # nothing disqualifying
    "negation",  # excludes rather than identifies: NOT, EXCEPT, is false
    "option_relative",  # "all of the above", "both A and C"
    "open_set",  # one member of a large category: an example of X
    "ranks_options",  # best / closest / greatest, over the list not the world
    "true_only_among_these",  # true of these ten choices, false as a claim about the world
    "underspecified_stem",  # leans on a unit, convention, or list only the options gave
]

LABEL_INSTRUCTION = """
Last, after your verdict, name the single failure mode that best fits, purely so the
run can be summarised. Use "none" when nothing disqualifying applies. This label records
what you already decided; it must not change the decision.

  negation | option_relative | open_set | ranks_options | true_only_among_these |
  underspecified_stem | none"""


SYSTEM_JUDGE = (
    """You decide whether an exam question survives losing its answer options.

You are given a question, its answer options, and which option is correct.

Apply one test. If a knowledgeable person saw this question with no options at all,
would they give the reference answer — rather than a different, equally correct one?

Answer true only when the reference answer is the question's single correct answer.
Answer false when the question, stripped of its options, admits many correct answers or
stops meaning anything without them.

Two kinds of question fail by construction, whatever else is true of them. Answer false
without further thought:

- It excludes rather than identifies: "which is NOT", "EXCEPT", "is false", "least
  likely". Every object in the world outside the answer set then answers it correctly.
  This holds however natural the question reads and however obvious the intended answer.
- Its answer only means something beside the others: "all of the above", "none of the
  above", "both A and C". There is no proposition left to ask for. This turns on where
  the alternatives live. An answer like "ii and iv" names real propositions when the
  stem itself sets them out ("i) ... ii) ... iii) ..."), and names nothing when they
  existed only as options.

Otherwise apply the test. Questions commonly fail it by asking for one member of a large
category ("an example of X", "a characteristic of X", "a true statement about X", where
the reference is one of many); by ranking the options ("best", "closest", "greatest",
which ranks the list rather than the world); by having an answer true among these ten
choices but false as a claim about the world; or by leaning on something only the
options supplied — a unit, a convention, or a list the answer selects from. These are
illustrations of the test, not a checklist: judge by the test itself.

Note what does *not* fail the test:

- A question may ask for several things at once and still have one correct answer:
  "calculate the efficiency and the reheat factor", "name the three broad groups of X".
  Judge whether the answer is unique, not whether it is simple.
- Working the answer out from figures the stem supplies is not leaning on the options.
  A question that sets out its own data and asks what follows from it keeps that data
  once the list is gone.
- The reference answer's *wording* does not matter. A later stage compares responses to
  it and accepts any phrasing of the same fact, so an answer that could be worded many
  ways is fine. What disqualifies a question is that a *different fact* would answer it
  just as correctly.
- Clumsy phrasing in the question does not matter either; a later stage rewords it. But
  a question is not merely clumsy when its phrasing is the only thing giving it one
  answer — "which of the following statements about X is true" is a large category, not
  a wording problem.
- A superlative is not automatically a ranking. Ask what it is comparing. "Which is the
  most persuasive argument", where ten arguments are supplied and one is strongest among
  them, ranks the list and fails. "Which best approximates the ratio of nonterminal nodes
  to total nodes in a complete K-ary tree", answer "1/K", does not: the ratio has one
  value, and "best approximates" hedges the rounding rather than comparing candidates.
  The test is whether deleting the options leaves the superlative with anything to rank.
  If the stem itself determines the quantity, it does not.

Examples, invented for illustration and not drawn from any dataset:
- "Which of the following is not a phase of matter?" -> false. It excludes.
- "Which of the following statements about photosynthesis is true?" -> false. Many true
  statements about photosynthesis exist.
- "Which of these numbers is the largest?", answer "17" -> false. It ranks the options.
- "Which of the following caused the most deaths in the 1800s?", answer "cholera" ->
  false if cholera is merely the worst of the ten listed, not of all causes.
- "Which of the following are the three primary colours of light?", answer "red, green,
  blue" -> true. One canonical answer; only the phrasing tied it to the list.
- "Which of these are noble gases? i) helium ii) nitrogen iii) argon", answer "i and
  iii" -> true. The candidates are in the question, so removing the options removes
  nothing the reader needed.
- "The boiling point of water at sea level is", answer "100 degrees Celsius" -> true.
"""
    + LABEL_INSTRUCTION
)

# Deliberately not a copy of the judge prompt: fed the same instructions the model
# reproduces the same rejections instead of catching the mistaken ones, which is the
# whole reason this pass exists.
SYSTEM_RESCORE = (
    """You rate how well an exam question would survive losing its options.

You are given a question, its answer options, and which option is correct.

Apply one test. If a knowledgeable person saw this question with no options at all,
would they give the reference answer — rather than a different, equally correct one?

Score 1-10 for your confidence that the reference answer is the question's single
correct answer once the options are gone.

Two kinds of question score 1, however natural they read and however obvious the
intended answer:

- It excludes rather than identifies: "which is NOT", "EXCEPT", "is false", "least
  likely". Every object in the world outside the answer set answers it correctly.
- Its answer only means something beside the others: "all of the above", "both A and C".
  There is no proposition left to ask for. Where the alternatives live decides this: an
  answer like "ii and iv" names real propositions when the stem itself sets them out
  ("i) ... ii) ... iii) ..."), and names nothing when they existed only as options.

Otherwise every point on the scale has a meaning. Use whichever fits; the middle is
where most doubtful questions belong, not a region to be avoided.

- 2-4: the stem names a subject but not a unique target — it asks for one member of a
  large category ("an example of X", "a true statement about X"), ranks the options
  ("best", "closest"), or leans on something only the options supplied. Example: "Which
  of the following is a prime number?", answer "7" -> 3.
- 5: answerable from the stem, but a different answer is about as likely as the
  reference. Example: "Which of the following instruments measures air pressure?",
  answer "aneroid barometer" -> 5, since "barometer" alone or "manometer" compete.
- 6-7: the stem names its subject and the property wanted, with some room left. Example:
  "The chemical symbol for the element with atomic number 26 is", answer "Fe" -> 7.
- 8-9: self-contained, with slight doubt about phrasing or about a near-synonymous
  alternative. Example: "What gas do plants absorb during photosynthesis?", answer
  "carbon dioxide" -> 9.
- 10: self-contained, and the reference is plainly the only answer; only the phrasing
  ever tied it to the list. Example: "The boiling point of water at sea level is",
  answer "100 degrees Celsius" -> 10.

Four things do not lower the score. A question asking for several quantities, or an
answer with several parts, is not thereby uncertain — judge whether the answer is
unique, not whether it is simple. Working the answer out from figures the stem supplies
is not leaning on the options either; a question that sets out its own data and asks
what follows from it keeps that data once the list is gone. Nor does the reference
answer's wording matter: a later stage accepts any phrasing of the same fact. What
disqualifies a question is that a *different fact* would answer it just as correctly.

Nor is a superlative automatically a ranking — ask what it compares. "Which is the most
persuasive argument", with ten arguments supplied and one strongest among them, ranks the
list and scores low. "Which best approximates the ratio of nonterminal nodes to total
nodes in a complete K-ary tree", answer "1/K", does not: the ratio has one value and
"best approximates" hedges the rounding. If the stem itself determines the quantity, the
superlative has nothing left to rank once the options are gone.
"""
    + LABEL_INSTRUCTION
)

SYSTEM_REWRITE = """You rewrite an exam question so it stands on its own.

You are given the question stem and nothing else. There are no answer options and no
correct answer, because your rewrite must be built from the stem alone.

One principle governs the rewrite: a reader must be able to answer your version exactly
as they would have answered the original, knowing nothing about the options.

That means:
- Preserve the question. Change wording only where the original depends on options
  being visible.
- Remove references to the list. "Which of the following", "which of these", "the
  statements above" point at nothing now; turn them into a direct question.
- Preserve everything the reader needs: facts, numbers, scenario details, data tables.
  Never compress a case description.
- Complete no sentences. An unfinished stem becomes the question it was leading to:
  "The capital of France is" becomes "What is the capital of France?". Never state the
  answer instead.
- Add nothing: no hints, and no mention that options ever existed.

Having written it, say whether it is supported: is every word of your question traceable
to the stem? If it names a topic, a property, or a scenario the stem never mentioned,
you supplied that yourself, and the question is now yours rather than the one that was
asked. Then possible=false, whatever you wrote.

That happens with stems like "Which of the following is true?", where nothing but the
options ever said what the question was about. It is rare. Most stems, including
unfinished ones, already contain everything their rewrite needs."""


# Field order, class docstrings and descriptions are part of the frozen model input.


# Deliberately carries no field to write a question into: a model that cannot draft a
# rewrite cannot let one justify its verdict.
class Verdict(BaseModel):
    """Whether an exam question keeps a single correct answer without its options."""

    reason: str = Field(
        description="one short clause weighing whether the reference "
        "answer is the only correct one without the options"
    )
    convertible: bool = Field(
        description="true only if the reference answer is the single correct answer "
        "once the options are removed"
    )
    failure: Failure = Field(description="which failure mode fits; reporting only")


# Pass 2 re-reads the pool pass 1 rejected, so nothing here may hint that a verdict was
# already reached: a model told the item was rejected agrees with the rejection this
# pass exists to overturn.
class Rescore(BaseModel):
    """How well an exam question would survive losing its options."""

    reason: str = Field(
        description="one short clause weighing how far the stem carries the question on its own"
    )
    score: int = Field(ge=1, le=10, description="1 = certainly cannot survive, 10 = certainly can")
    failure: Failure = Field(description="which failure mode fits; reporting only")


class Rewrite(BaseModel):
    """A question rewritten to stand alone, or the finding that the stem supports none."""

    standalone_question: str = Field(
        description="your best attempt at the question, answerable with no options in "
        "view; write it even if you then judge it unsupported"
    )
    possible: bool = Field(
        description="false if the question you just wrote names anything the stem never "
        "mentioned; the question is then discarded"
    )


def prompt_version() -> str:
    """Digest the complete model instrument stored on every stage record.

    The schemas are prompt text too: `PromptedOutput` sends each model's JSON schema,
    including its docstring and field descriptions, in the instructions.
    """
    schemas = (
        json.dumps(model.model_json_schema(), sort_keys=True)
        for model in (Verdict, Rescore, Rewrite)
    )
    joined = "\0".join((SYSTEM_JUDGE, SYSTEM_RESCORE, SYSTEM_REWRITE, *schemas))
    return hashlib.sha256(joined.encode()).hexdigest()[:12]


PROMPT_VERSION = prompt_version()


def require_retrying(model: object, spec: str) -> None:
    """Refuse to run a batch on a bare `provider:model` string.

    A plain string hands the transport to Pydantic AI's default, silently bypassing the
    retry settings providers.py exists to apply; a paid batch pays for the failures.
    """
    if isinstance(model, str):
        raise SystemExit(
            f"  {spec} resolves to a bare model string with no retrying transport. "
            "Add a retrying transport in research.grading.providers before running it."
        )


def check_provenance(paths: list[Path], model_name: str) -> None:
    """Require every successful record to use this model and prompt instrument."""
    for path in paths:
        for record in batch.load(path).values():
            if "error" in record:
                continue
            if (record.get("prompts"), record.get("model")) != (PROMPT_VERSION, model_name):
                raise SystemExit(
                    f"  {path} holds records from {record.get('model')} / prompts "
                    f"{record.get('prompts')}, but this run is {model_name} / "
                    f"{PROMPT_VERSION}. Move or delete the file to start fresh."
                )


def prompt_for(row: dict, answer_text: str) -> str:
    options = "\n".join(f"  {chr(65 + i)}. {opt}" for i, opt in enumerate(row["options"]))
    return f"Question: {row['question']}\n\nOptions:\n{options}\n\nCorrect answer: {answer_text}"


def stem_only(row: dict) -> str:
    """The rewrite pass's whole input: the stem, with no options and no answer.

    Withholding them makes an answer-driven rewrite impossible rather than merely
    forbidden: a rewriter that can see the answer reaches for it whenever the stem is
    too thin, while with the stem alone a thin stem simply has no rewrite.
    """
    return f"Question: {row['question']}"


def finalise(
    row: dict,
    answer_text: str,
    *,
    convertible: bool,
    reason: str,
    stage: str,
    score: int | None = None,
    failure: str = "none",
    model: str = "",
) -> dict:
    """The stored record for one item, before its rewrite is attached.

    `list_style` marks the preregistered stratum, computed from the source item's
    original phrasing; it is not an instruction to the converter.
    """
    record = {
        "question_id": row["question_id"],
        "category": row["category"],
        "list_style": row["list_style"],
        # The stem travels along so the audit reads one file, and so the answer-leak
        # check can tell a leak from an echo of the stem's own words.
        "question": row["question"].strip(),
        "stage": stage,
        "convertible": convertible,
        "reason": reason,
        # Reporting only; nothing decides anything from it.
        "failure": failure,
        "open_question": "",
        # Travels with the item: it is what the judge panel grades against.
        "reference_answer": answer_text,
        # Which instrument produced this verdict.
        "model": model,
        "prompts": PROMPT_VERSION,
    }
    if score is not None:
        record["score"] = score
    return record


async def ask(agent: Agent, prompt: str):
    """One model call, bounded in wall-clock time. See `ITEM_TIMEOUT_SECONDS`."""
    async with asyncio.timeout(ITEM_TIMEOUT_SECONDS):
        return await agent.run(prompt)


def tokens_of(result) -> dict:
    """Input/output token counts for one call.

    `usage` is a property on this version of pydantic-ai and was a method on earlier
    ones; both are accepted so an upgrade cannot silently cost a batch.
    """
    usage = result.usage
    if callable(usage):
        usage = usage()
    return {
        "in": getattr(usage, "input_tokens", 0) or 0,
        "out": getattr(usage, "output_tokens", 0) or 0,
    }


async def run_judge(
    rows: list[dict],
    answer_key: dict[int, dict],
    agent: Agent,
    out_path: Path,
    concurrency: int,
    retry_failed: bool,
    model_name: str = "",
) -> dict[str, dict]:
    async def work(row: dict) -> dict:
        answer = answer_key[row["question_id"]]["answer_text"]
        result = await ask(agent, prompt_for(row, answer))
        record = finalise(
            row,
            answer,
            convertible=result.output.convertible,
            reason=result.output.reason,
            stage="judge",
            failure=result.output.failure,
            model=model_name,
        )
        record["tokens"] = tokens_of(result)
        return record

    return await batch.run(
        rows,
        lambda row: str(row["question_id"]),
        work,
        out_path,
        concurrency=concurrency,
        label="items (judge)",
        retry_failed=retry_failed,
    )


async def run_rescore(
    rows: list[dict],
    answer_key: dict[int, dict],
    agent: Agent,
    out_path: Path,
    concurrency: int,
    retry_failed: bool,
    model_name: str = "",
) -> dict[str, dict]:
    async def work(row: dict) -> dict:
        answer = answer_key[row["question_id"]]["answer_text"]
        result = await ask(agent, prompt_for(row, answer))
        record = finalise(
            row,
            answer,
            convertible=result.output.score >= READMIT_AT,
            reason=result.output.reason,
            stage="rescore",
            score=result.output.score,
            failure=result.output.failure,
            model=model_name,
        )
        record["tokens"] = tokens_of(result)
        return record

    return await batch.run(
        rows,
        lambda row: str(row["question_id"]),
        work,
        out_path,
        concurrency=concurrency,
        label="rejects (rescore)",
        retry_failed=retry_failed,
    )


async def run_rewrite(
    rows: list[dict],
    agent: Agent,
    out_path: Path,
    concurrency: int,
    retry_failed: bool,
    model_name: str = "",
) -> dict[str, dict]:
    async def work(row: dict) -> dict:
        # No answer_key lookup: this pass is deliberately blind to the answer.
        result = await ask(agent, stem_only(row))
        question = result.output.standalone_question.strip()
        return {
            "question_id": row["question_id"],
            "open_question": question if result.output.possible else "",
            "model": model_name,
            "prompts": PROMPT_VERSION,
            "tokens": tokens_of(result),
        }

    return await batch.run(
        rows,
        lambda row: str(row["question_id"]),
        work,
        out_path,
        concurrency=concurrency,
        label="survivors (rewrite)",
        retry_failed=retry_failed,
    )


def merge(judged: dict[str, dict], rescored: dict[str, dict]) -> list[dict]:
    """One verdict per item: the judge's, unless the re-score pass re-read it."""
    final = []
    for key, record in judged.items():
        if "error" not in record and not record["convertible"]:
            record = rescored.get(key, record)
        final.append(record)
    return final


def attach(verdicts: list[dict], rewrites: dict[str, dict]) -> list[dict]:
    """Put each survivor's rewrite on its record; an item without one is dropped.

    The rewrite pass sees only the stem, so it is the last stage able to notice that a
    stem poses no question at all — the failure the first two passes let through. Its
    empty return is therefore a verdict, and overrides the earlier "convertible".
    """
    out = []
    for record in verdicts:
        if "error" in record or not record["convertible"]:
            out.append(record)
            continue
        rewrite = rewrites.get(str(record["question_id"]), {})
        if "error" in rewrite:
            # Its own stage value: the audit samples by deciding stage, and an HTTP
            # failure left carrying `stage: judge` would count against the judge.
            record = {
                **record,
                "convertible": False,
                "stage": "rewrite_error",
                "reason": f"rewrite failed ({record['reason']})",
            }
        elif not rewrite.get("open_question"):
            record = {
                **record,
                "convertible": False,
                "stage": "rewrite",
                "reason": "stem alone poses no answerable question",
            }
        else:
            record = {**record, "open_question": rewrite["open_question"]}
        out.append(record)
    return out


def summarise(records: list[dict]) -> dict[str, object]:
    """Report the conversion rate and anything the audit should read first.

    Returns the headline numbers as well as printing them. Deliberately no plausibility
    band on the rate: what the rate should be is the audit's finding, not a constant in
    this file.
    """
    done = [r for r in records if "error" not in r]
    failed = len(records) - len(done)
    convertible = [r for r in done if r["convertible"]]
    empty = [r for r in convertible if not r["open_question"].strip()]

    print(f"\n  {len(done):,} decided, {failed:,} failed")
    if not done:
        raise SystemExit("  nothing succeeded")
    rate = len(convertible) / len(done)
    print(f"  convertible: {len(convertible):,} ({rate:.1%})")
    print(f"  dropped:     {len(done) - len(convertible):,}")

    rescored = [r for r in done if r["stage"] == "rescore"]
    if rescored:
        readmitted = sum(1 for r in rescored if r["convertible"])
        print(
            f"  rescore:     {readmitted:,}/{len(rescored):,} rejects readmitted "
            f"({readmitted / len(rescored):.1%})"
        )

    counted = [r["tokens"] for r in done if r.get("tokens")]
    if counted:
        tin, tout = sum(t["in"] for t in counted), sum(t["out"] for t in counted)
        print(
            f"  tokens:      {tin / 1e6:.2f}M in, {tout / 1e6:.2f}M out "
            f"across {len(counted):,} verdicts"
        )

    labelled = [r for r in done if r.get("failure")]
    if labelled:
        counts: dict[str, int] = {}
        for r in labelled:
            counts[r["failure"]] = counts.get(r["failure"], 0) + 1
        shown = ", ".join(f"{k} {v}" for k, v in sorted(counts.items(), key=lambda kv: -kv[1]))
        print(f"  failure modes: {shown}")

    by_style = {}
    for r in done:
        key = "list-style" if r["list_style"] else "plain"
        seen, ok = by_style.get(key, (0, 0))
        by_style[key] = (seen + 1, ok + bool(r["convertible"]))
    for key, (seen, ok) in sorted(by_style.items()):
        print(f"    {key:11s} {ok:,}/{seen:,} convertible ({ok / seen:.1%})")

    # Reported, never acted on. A rewrite that still points at the options cannot be
    # answered without them, and one containing the reference answer reveals it instead
    # of asking for it. Whether the filter is good enough is the audit's finding.
    stuck = [r for r in convertible if DANGLING.search(r["open_question"])]
    if stuck:
        print(
            f"  WARNING: {len(stuck):,} rewrites still refer to the options — "
            f"audit first: {[r['question_id'] for r in stuck[:10]]}"
        )
    leaked = [
        r
        for r in convertible
        if len(r["reference_answer"].strip()) >= 8
        and r["reference_answer"].strip().lower() in r["open_question"].lower()
        and r["reference_answer"].strip().lower() not in r["question"].lower()
    ]
    if leaked:
        print(
            f"  WARNING: {len(leaked):,} rewrites contain the reference answer — "
            f"audit first: {[r['question_id'] for r in leaked[:10]]}"
        )

    # A record marked usable while carrying no question contradicts itself and would
    # reach generation as a blank prompt.
    if empty:
        raise SystemExit(f"  {len(empty):,} items marked convertible with an empty rewrite")

    # Counted apart from `failed`: a rewrite that never returned is stored as a drop
    # and would read as a rejection nobody made. See `attach`.
    rewrite_errors = sum(1 for r in done if r["stage"] == "rewrite_error")
    if rewrite_errors:
        print(
            f"  {rewrite_errors:,} items dropped because their rewrite call failed, "
            f"not because they were rejected"
        )

    return {
        "decided": len(done),
        "failed": failed,
        "convertible": len(convertible),
        "rate": rate,
        "dangling": len(stuck),
        "answer_leaks": len(leaked),
        "rewrite_errors": rewrite_errors,
        "prompts": PROMPT_VERSION,
    }


def require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise SystemExit(f"{label} file does not exist: {path}")


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def load_rows(labels_path: Path, limit: int | None = None) -> list[dict]:
    require_file(labels_path, "labels")
    rows = read_records(labels_path)
    corpus.index_rows(rows, "labels")
    if not rows:
        raise SystemExit("labels contain no questions")
    for row in rows:
        if type(row.get("question_id")) is not int or row["question_id"] < 0:
            raise SystemExit("labels require non-negative integer question IDs")
        if not isinstance(row.get("question"), str) or not row["question"].strip():
            raise SystemExit("labels require nonempty question text")
        if not isinstance(row.get("options"), list) or not row["options"] or any(
            not isinstance(option, str) for option in row["options"]
        ):
            raise SystemExit("labels require a nonempty list of answer options")
        if not isinstance(row.get("category"), str) or type(row.get("list_style")) is not bool:
            raise SystemExit("labels require a category and boolean list_style")
    if limit is not None:
        rows = random.Random(LIMIT_SEED).sample(rows, min(limit, len(rows)))
    return rows


def load_inputs(
    labels_path: Path, answer_key_path: Path, limit: int | None = None
) -> tuple[list[dict], dict[int, dict]]:
    corpus.check(answer_key_path, labels_path)
    rows = load_rows(labels_path, limit)
    answer_key = corpus.load(answer_key_path)
    absent = sorted({row["question_id"] for row in rows} - set(answer_key))
    if absent:
        raise SystemExit(
            f"{len(absent):,} items have no answer key entry (first few: {absent[:5]})"
        )
    return rows, answer_key


def load_stage(
    path: Path, label: str, *, repair_truncated_tail: bool = False
) -> dict[str, dict]:
    require_file(path, label)
    stage = label.split()[0]
    records: dict[str, dict] = {}
    raw = path.read_bytes()
    lines = raw.split(b"\n")
    stage_rows: list[dict] = []
    truncate_at: int | None = None
    for number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            is_unterminated_tail = number == len(lines) and not raw.endswith(b"\n")
            if repair_truncated_tail and is_unterminated_tail:
                truncate_at = len(raw) - len(line)
                break
            raise InputError(f"invalid JSONL {path}:{number}") from exc
        if not isinstance(record, dict):
            raise InputError(f"JSONL row must be an object: {path}:{number}")
        stage_rows.append(record)

    for record in stage_rows:
        key = record.get("key")
        if not isinstance(key, str) or not key:
            raise SystemExit(f"{path}: stage record requires a string key")
        if key in records and "error" not in records[key]:
            raise SystemExit(f"{path}: duplicate completed stage record {key}")
        if "error" not in record:
            if str(record.get("question_id")) != key:
                raise SystemExit(f"{path}: stage key and question ID differ")
            if stage == "rewrite":
                if record.get("stage", "rewrite") != "rewrite" or not isinstance(record.get("open_question"), str):
                    raise SystemExit(f"{path}: invalid rewrite record {key}")
            else:
                if record.get("stage") != stage or type(record.get("convertible")) is not bool:
                    raise SystemExit(f"{path}: invalid {stage} record {key}")
                for field in ("question", "reference_answer", "reason", "category", "failure"):
                    if not isinstance(record.get(field), str):
                        raise SystemExit(f"{path}: {key} lacks {field}")
                if stage == "rescore" and (
                    type(record.get("score")) is not int or not 1 <= record["score"] <= 10
                    or record["convertible"] != (record["score"] >= READMIT_AT)
                ):
                    raise SystemExit(f"{path}: inconsistent rescoring decision {key}")
        records[key] = record
    if truncate_at is not None:
        with path.open("r+b") as stream:
            stream.truncate(truncate_at)
    return records


def bind_run(args: argparse.Namespace, input_paths: tuple[Path, ...]) -> None:
    """Bind a resumable live conversion log to its inputs and effective settings."""
    path = args.out.with_suffix(args.out.suffix + ".manifest.json")
    validate_distinct_output(path, input_paths)
    expected = {
        "stage": args.command, "model": args.model, "prompts": PROMPT_VERSION,
        "temperature": 0.0, "readmit_at": READMIT_AT, "output_retries": 3,
        "limit": args.limit, "limit_seed": LIMIT_SEED,
        "input_sha256": [sha256(source) for source in input_paths],
    }
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != expected:
            raise SystemExit(f"{path}: saved conversion inputs or settings differ")
    else:
        if args.out.exists() and args.out.stat().st_size:
            raise SystemExit(f"{args.out}: cannot resume a nonempty log without its input manifest")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(expected, indent=2) + "\n")
    if args.out.exists():
        load_stage(args.out, args.command + " log", repair_truncated_tail=True)


def check_source_rows(records: dict[str, dict], rows: list[dict], answer_key: dict | None = None) -> None:
    """Require completed decisions to describe the supplied original questions."""
    source = {str(row["question_id"]): row for row in rows}
    for key, record in records.items():
        if key not in source or "error" in record:
            continue
        row = source[key]
        if (
            record["question"] != row["question"].strip()
            or record["category"] != row["category"]
            or record.get("list_style") is not row["list_style"]
        ):
            raise SystemExit(f"stage log and input question differ for {key}")
        if answer_key is not None and record["reference_answer"] != answer_key[row["question_id"]]["answer_text"]:
            raise SystemExit(f"stage log and reference answer differ for {key}")


def validate_selected_judges(judged: dict[str, dict], wanted: set[str]) -> None:
    """Require a completed judge decision for every selected source row."""
    missing = wanted - set(judged)
    if missing:
        raise SystemExit(
            f"judge log is missing {len(missing):,} selected rows "
            f"(first few: {sorted(missing)[:5]})"
        )
    failed = {key for key in wanted if "error" in judged[key]}
    if failed:
        raise SystemExit(f"judge log has {len(failed):,} failed selected rows")


def validate_output(path: Path, input_paths: list[Path] | tuple[Path, ...] = ()) -> None:
    resolved = path.resolve()
    if any(resolved == input_path.resolve() for input_path in input_paths):
        raise SystemExit(f"output is the same path as an input: {path}")
    if path.exists():
        raise SystemExit(f"output already exists: {path}")


def validate_distinct_output(path: Path, input_paths: list[Path] | tuple[Path, ...]) -> None:
    resolved = path.resolve()
    if any(resolved == input_path.resolve() for input_path in input_paths):
        raise SystemExit(f"output is the same path as an input: {path}")


def validate_stage_coverage(
    judged: dict[str, dict],
    rescored: dict[str, dict],
    rewrites: dict[str, dict] | None = None,
) -> None:
    if not judged:
        raise SystemExit("judge log has no records")
    failed_judges = {key for key, record in judged.items() if "error" in record}
    if failed_judges:
        raise SystemExit(f"judge log has {len(failed_judges):,} failed required rows")

    required_rescores = {
        key for key, record in judged.items() if not record["convertible"]
    }
    if set(rescored) - required_rescores:
        raise SystemExit("rescore log contains questions not rejected by the initial filter")
    missing_rescores = required_rescores - set(rescored)
    if missing_rescores:
        raise SystemExit(
            f"missing {len(missing_rescores):,} required rescore rows "
            f"(first few: {sorted(missing_rescores)[:5]})"
        )
    failed_rescores = {
        key for key in required_rescores if "error" in rescored[key]
    }
    if failed_rescores:
        raise SystemExit(f"rescore log has {len(failed_rescores):,} failed required rows")

    for key in required_rescores:
        for field in ("question", "reference_answer", "category", "list_style"):
            if judged[key].get(field) != rescored[key].get(field):
                raise SystemExit(f"judge/rescore source identity differs for {key}: {field}")
    if rewrites is None:
        return
    verdicts = merge(judged, rescored)
    required_rewrites = {
        record["key"] for record in verdicts if record["convertible"]
    }
    if set(rewrites) - required_rewrites:
        raise SystemExit("rewrite log contains questions not retained by filtering")
    missing_rewrites = required_rewrites - set(rewrites)
    if missing_rewrites:
        raise SystemExit(
            f"missing {len(missing_rewrites):,} required rewrite rows "
            f"(first few: {sorted(missing_rewrites)[:5]})"
        )
    failed_rewrites = {
        key for key in required_rewrites if "error" in rewrites[key]
    }
    if failed_rewrites:
        raise SystemExit(f"rewrite log has {len(failed_rewrites):,} failed required rows")


def build_agent(model_name: str, stage: str) -> Agent:
    model = providers.resolve(model_name)
    require_retrying(model, model_name)
    output_type, system_prompt = {
        "judge": (Verdict, SYSTEM_JUDGE),
        "rescore": (Rescore, SYSTEM_RESCORE),
        "rewrite": (Rewrite, SYSTEM_REWRITE),
    }[stage]
    return Agent(
        model,
        output_type=PromptedOutput(output_type),
        system_prompt=system_prompt,
        model_settings={"temperature": 0.0},
        retries={"output": 3},
    )


def replay(
    judge_path: Path,
    rescore_path: Path,
    rewrite_path: Path,
    model_name: str = DEFAULT_MODEL,
) -> list[dict]:
    """Rebuild the converted corpus from completed stage logs without model calls."""
    judged = load_stage(judge_path, "judge log")
    rescored = load_stage(rescore_path, "rescore log")
    rewrites = load_stage(rewrite_path, "rewrite log")
    check_provenance([judge_path, rescore_path, rewrite_path], model_name)
    validate_stage_coverage(judged, rescored, rewrites)
    final = attach(merge(judged, rescored), rewrites)
    return sorted((row for row in final if "error" not in row), key=lambda row: row["question_id"])


def write_records(path: Path, records: list[dict], *, input_paths: tuple[Path, ...] = ()) -> None:
    validate_output(path, input_paths)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        stream.write("".join(json.dumps(record) + "\n" for record in records))


def add_live_options(parser: argparse.ArgumentParser, *, answer_key: bool) -> None:
    parser.add_argument("--labels", type=Path, required=True)
    if answer_key:
        parser.add_argument("--answer-key", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--limit", type=positive_int)
    parser.add_argument("--concurrency", type=positive_int, default=batch.DEFAULT_CONCURRENCY)
    parser.add_argument("--retry-failed", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True)

    judge_parser = commands.add_parser("judge", help="run the initial verdict stage")
    add_live_options(judge_parser, answer_key=True)

    rescore_parser = commands.add_parser("rescore", help="rescore rejected judge rows")
    add_live_options(rescore_parser, answer_key=True)
    rescore_parser.add_argument("--judge-log", type=Path, required=True)

    rewrite_parser = commands.add_parser("rewrite", help="rewrite surviving stems")
    add_live_options(rewrite_parser, answer_key=False)
    rewrite_parser.add_argument("--judge-log", type=Path, required=True)
    rewrite_parser.add_argument("--rescore-log", type=Path, required=True)

    replay_parser = commands.add_parser("replay", help="assemble completed stage logs")
    replay_parser.add_argument("--judge-log", type=Path, required=True)
    replay_parser.add_argument("--rescore-log", type=Path, required=True)
    replay_parser.add_argument("--rewrite-log", type=Path, required=True)
    replay_parser.add_argument("--out", type=Path, required=True)
    replay_parser.add_argument("--model", default=DEFAULT_MODEL)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "replay":
        inputs = (args.judge_log, args.rescore_log, args.rewrite_log)
        validate_output(args.out, inputs)
        records = replay(
            args.judge_log, args.rescore_log, args.rewrite_log, model_name=args.model
        )
        write_records(args.out, records, input_paths=inputs)
        print(f"wrote {len(records):,} records -> {args.out}")
        summarise(records)
        return 0

    if args.command == "judge":
        validate_distinct_output(args.out, (args.labels, args.answer_key))
        rows, answer_key = load_inputs(args.labels, args.answer_key, args.limit)
        check_provenance([args.out], args.model)
        bind_run(args, (args.labels, args.answer_key))
        agent = build_agent(args.model, "judge")
        asyncio.run(
            run_judge(
                rows,
                answer_key,
                agent,
                args.out,
                args.concurrency,
                args.retry_failed,
                args.model,
            )
        )
        return 0

    if args.command == "rescore":
        validate_distinct_output(args.out, (args.labels, args.answer_key, args.judge_log))
        rows, answer_key = load_inputs(args.labels, args.answer_key, args.limit)
        judged = load_stage(args.judge_log, "judge log")
        check_provenance([args.judge_log, args.out], args.model)
        wanted = {str(row["question_id"]) for row in rows}
        validate_selected_judges(judged, wanted)
        rejected = {
            key
            for key in wanted
            if not judged[key]["convertible"]
        }
        selected = [row for row in rows if str(row["question_id"]) in rejected]
        check_source_rows(judged, rows, answer_key)
        if selected:
            bind_run(args, (args.labels, args.answer_key, args.judge_log))
            agent = build_agent(args.model, "rescore")
            asyncio.run(
                run_rescore(
                    selected,
                    answer_key,
                    agent,
                    args.out,
                    args.concurrency,
                    args.retry_failed,
                    args.model,
                )
            )
        else:
            print("no rejected judge rows to rescore")
        return 0

    validate_distinct_output(args.out, (args.labels, args.judge_log, args.rescore_log))
    rows = load_rows(args.labels, args.limit)
    judged = load_stage(args.judge_log, "judge log")
    rescored = load_stage(args.rescore_log, "rescore log")
    check_provenance([args.judge_log, args.rescore_log, args.out], args.model)
    wanted = {str(row["question_id"]) for row in rows}
    validate_selected_judges(judged, wanted)
    selected_judges = {key: judged[key] for key in wanted}
    selected_rescores = {key: record for key, record in rescored.items() if key in wanted}
    validate_stage_coverage(selected_judges, selected_rescores)
    verdicts = [record for record in merge(judged, rescored) if record["key"] in wanted]
    survivor_ids = {
        str(record["question_id"])
        for record in verdicts
        if "error" not in record and record["convertible"]
    }
    selected = [row for row in rows if str(row["question_id"]) in survivor_ids]
    check_source_rows(judged, rows)
    check_source_rows(rescored, rows)
    if selected:
        bind_run(args, (args.labels, args.judge_log, args.rescore_log))
        agent = build_agent(args.model, "rewrite")
        asyncio.run(
            run_rewrite(
                selected,
                agent,
                args.out,
                args.concurrency,
                args.retry_failed,
                args.model,
            )
        )
    else:
        print("no surviving rows to rewrite")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
