"""Grade open-ended responses and combine three judges into success-count labels.

Each judge writes a separate resumable log. Panel reduction retains three-way verdicts
and maps no_answer to incorrect; prompts, schema and settings remain fingerprinted.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import tempfile
from collections import Counter
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from research.grading.panel import (
    build_panel,
    panel_agreement,
    panel_labels,
    verdict_binary,
)

# Prompt text, field order and schema descriptions are frozen model inputs.

SYSTEM_GRADE = """\
You judge whether a response to an exam question states the same answer as a
reference answer.

You are given a question, the reference answer, and a model's response. The
response may be long, may reason step by step, and may be cut off.

Apply one test. Find the response's final answer — the answer it commits to,
not answers it merely considers along the way — and decide whether that final
answer states the same fact as the reference.

The reference answer is authoritative. Do not re-solve the question, and do
not judge the response's reasoning; if the response's final answer disagrees
with the reference, the verdict is no_match even if you believe the response.

A response matches when its final answer contains everything the reference
states. Wording never matters: paraphrase, a different language, different
formatting, or extra correct detail alongside the reference content are all
still a match ("Labrador" matches reference "dog"). For numeric answers, the
response must agree with the reference to within 1% relative error, in any
unit convertible from the reference's; a range does not match a specific
value, even if the range contains it.

A response is no_match when it commits to a final answer that contradicts or
omits part of the reference, or when it hedges between several different
final answers.

A response is no_answer when it never commits to a final answer: it is cut
off before concluding, answers a different question than the one asked, is
empty, or only restates the question. Nothing it says contradicts the
reference; it simply never answers."""


def tokens_of(result) -> dict:
    """Input/output token counts for one pydantic-ai call."""
    usage = result.usage
    if callable(usage):
        usage = usage()
    return {
        "in": getattr(usage, "input_tokens", 0) or 0,
        "out": getattr(usage, "output_tokens", 0) or 0,
    }


class Grade(BaseModel):
    """Whether a response's final answer states the same fact as the reference answer."""

    # Field order is generation order: the clause is written before the verdict so the
    # verdict can depend on it, not the other way round.
    reason: str = Field(
        description="one short clause naming the response's final "
        "answer and weighing it against the reference"
    )
    verdict: Literal["match", "no_match", "no_answer"] = Field(
        description="match only if the final answer contains everything the reference "
        "states; no_answer only if the response never commits to one"
    )


def prompt_version() -> str:
    """Hash the system prompt and response schema for grading-run consistency checks."""
    schema = json.dumps(Grade.model_json_schema(), sort_keys=True)
    return hashlib.sha256(f"{SYSTEM_GRADE}\0{schema}".encode()).hexdigest()[:12]


PROMPT_VERSION = prompt_version()


def _grading_version(system: str, grade: type) -> str:
    schema = json.dumps(grade.model_json_schema(), sort_keys=True)
    return hashlib.sha256(f"{system}\0{schema}".encode()).hexdigest()[:12]


def _prompt_sets() -> dict[str, tuple[str, type, str]]:
    """name -> (system prompt, schema, digest). v1 is frozen; v2 is task L1."""
    from research.data import prompts_v2

    return {
        "v1": (SYSTEM_GRADE, Grade, PROMPT_VERSION),
        "v2": (prompts_v2.SYSTEM_GRADE, prompts_v2.Grade, _grading_version(prompts_v2.SYSTEM_GRADE, prompts_v2.Grade)),
    }


PROMPT_SETS = _prompt_sets()
ACTIVE_PROMPTS = "v1"  # set once from --prompts before grading


# A response with no visible characters cannot be graded and must not cost three judge
# calls. Anything else — however short or strange — goes to the judges: cheap heuristics
# beyond emptiness would be a second, unvalidated grader.
def is_ungradable(response: str) -> bool:
    return not response.strip()


# The user template has a separate fingerprint from the system prompt and schema.
USER_TEMPLATE = (
    "Question: {question}\n\nReference answer: {reference}\n\nResponse to judge:\n{response}"
)
TEMPLATE_VERSION = hashlib.sha256(USER_TEMPLATE.encode()).hexdigest()[:12]


def user_prompt(question: str, reference: str, response: str) -> str:
    """The grading call's user message. The reference sits before the response so the
    judge reads the yardstick before the thing measured."""
    return USER_TEMPLATE.format(question=question, reference=reference, response=response)


# ── Inputs: the generation run joined to the corpus ──────────────────────────


def load_references(corpus_path: Path) -> dict[int, dict]:
    """question_id -> {open_question, reference_answer} for convertible items."""
    references = {}
    with corpus_path.open() as stream:
        for line in stream:
            row = json.loads(line)
            if row.get("convertible"):
                references[row["question_id"]] = {
                    "question": (row.get("open_question") or "").strip(),
                    "reference": (row.get("reference_answer") or "").strip(),
                }
    return references


def load_generation_items(run_dir: Path, corpus_path: Path) -> list[dict]:
    """Every (item, repeat) to grade, joined with its question and reference.

    A generation whose question is missing from the corpus is a corpus/run mismatch and
    stops the join — grading it against nothing would produce a verdict about nothing.
    """
    references = load_references(corpus_path)
    items = []
    rep_dirs = sorted(run_dir.glob("rep*"), key=lambda path: int(path.name.removeprefix("rep")))
    if not rep_dirs:
        raise SystemExit(f"no rep*/ directories under {run_dir}")
    for rep_dir in rep_dirs:
        rep = int(rep_dir.name.removeprefix("rep"))
        with (rep_dir / "generations.jsonl").open() as stream:
            for line in stream:
                record = json.loads(line)
                qid = record["question_id"]
                if qid not in references:
                    raise SystemExit(
                        f"question_id {qid} in {rep_dir} has no convertible corpus row — "
                        "the generation run and the corpus file do not match."
                    )
                items.append(
                    {
                        "question_id": qid,
                        "rep": rep,
                        "question": references[qid]["question"],
                        "reference": references[qid]["reference"],
                        "response": record["text"],
                        "finish_reason": record.get("finish_reason"),
                    }
                )
    return items


def item_key(item: dict) -> str:
    return f"{item['question_id']}:{item['rep']}"


# ── Provenance ───────────────────────────────────────────────────────────────

MAX_OUTPUT_TOKENS = 400  # a clause + JSON needs ~150; the cap guards temp-0 loops


def check_provenance(
    path: Path, judge: str, output_mode: str = "prompted", reasoning_effort: str | None = None
) -> None:
    """Refuse to resume a file produced by a different prompt, judge, or output mode.

    The output mode is part of the instrument: prompted JSON and forced tool calls
    deliver the same schema by different mechanisms, and a file graded half one way is
    two instruments wearing one name."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "prompts" not in record:
            # Error rows carry no provenance fields (batch.py writes only key +
            # error). One landing first must not poison every later resume.
            continue
        found = (
            record.get("prompts"),
            record.get("judge"),
            record.get("output_mode", "prompted"),
            record.get("reasoning_effort"),
        )
        # The template fingerprint is optional; when present, it must match.
        if record.get("template") not in (None, TEMPLATE_VERSION):
            raise SystemExit(
                f"{path} holds records from user-template {record.get('template')} "
                f"but the current template digests to {TEMPLATE_VERSION} — the "
                "user message changed; do not mix instruments in one file"
            )
        if found != (PROMPT_SETS[ACTIVE_PROMPTS][2], judge, output_mode, reasoning_effort):
            raise SystemExit(
                f"{path} holds records from judge={record.get('judge')} "
                f"prompts={record.get('prompts')} mode={found[2]} "
                f"effort={found[3]}, but this run is judge={judge} "
                f"prompts={PROMPT_SETS[ACTIVE_PROMPTS][2]} mode={output_mode} "
                f"effort={reasoning_effort}. Move or delete the file to start fresh."
            )
        return  # one record is enough; the file is append-only under one instrument


def judge_slug(spec: str) -> str:
    """A filesystem-safe name for the judge's output file."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", spec)


def positive_int(value: str) -> int:
    """Parse a strictly positive integer for bounded worker counts."""
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


# Output modes and reasoning settings used for judge calibration and bulk grading.
CALIBRATED_SETTINGS = {
    "bedrock:mistral.ministral-3-14b-instruct": ("prompted", None),
    "bedrock:openai.gpt-oss-120b-1:0": ("prompted", "low"),
    "bedrock:qwen.qwen3-next-80b-a3b": ("prompted", None),
}


def require_calibrated_settings(spec: str, output_mode: str, reasoning_effort: str | None) -> None:
    """Require the calibrated output mode and reasoning effort for each listed judge."""
    calibrated = CALIBRATED_SETTINGS.get(spec)
    if calibrated is None:
        return
    if (output_mode, reasoning_effort) != calibrated:
        raise SystemExit(
            f"{spec} was calibrated with output_mode={calibrated[0]} "
            f"reasoning_effort={calibrated[1]}, but this run passes "
            f"({output_mode}, {reasoning_effort}). Use the calibration settings."
        )


# ── The grading run ──────────────────────────────────────────────────────────


def require_bedrock_for_effort(spec: str, reasoning_effort: str | None) -> None:
    """The effort setting rides `bedrock_additional_model_requests_fields`, which
    every non-Bedrock transport silently drops — while the value would still be
    written into each record's provenance. A record must never claim a setting
    the model did not receive."""
    if reasoning_effort and not spec.startswith("bedrock:"):
        raise SystemExit(
            f"--reasoning-effort only reaches Bedrock models; {spec!r} would ignore "
            "it while the output records still claimed it. Drop the flag or use a "
            "bedrock: spec."
        )


# Two LaTeX habits break JSON strings. `\$1000`: the backslash escapes a literal
# character, so drop it. `\(`, `\pi`, `\overline{21}`: the backslash starts a command,
# so double it and the text keeps the formula the judge wrote. `\\` is already valid and
# passes through; `\b` and `\f` are treated as LaTeX (`\beta`, `\frac`) — nobody
# means backspace or form feed in a grading reason.
INVALID_JSON_ESCAPES = re.compile(r'(\\\\)|\\(?=[$%&#_])|(\\)(?![\\"/nrtu])')


def repair_escapes(text: str) -> str:
    """Rewrite invalid JSON escapes in reply text; the verdict field is never touched."""

    def fix(match: re.Match) -> str:
        if match.group(1):
            return match.group(1)  # proper escaped backslash
        if match.group(2):
            return "\\\\"  # LaTeX command: keep the backslash, escaped
        return ""  # LaTeX-escaped literal: bare character

    return INVALID_JSON_ESCAPES.sub(fix, text)


class EscapeRepairModel:
    """Wrap a provider to repair LaTeX escapes before validating its JSON response."""

    def __new__(cls, wrapped):
        from pydantic_ai.models.wrapper import WrapperModel

        class _Repair(WrapperModel):
            repairs = 0

            async def request(self, messages, model_settings, model_request_parameters):
                response = await self.wrapped.request(
                    messages, model_settings, model_request_parameters
                )
                for part in response.parts:
                    content = getattr(part, "content", None)
                    if isinstance(content, str):
                        repaired = repair_escapes(content)
                        if repaired != content:
                            part.content = repaired
                            type(self).repairs += 1
                return response

        return _Repair(wrapped)


async def run_judge(
    spec: str,
    run_dir: Path,
    corpus_path: Path,
    out_dir: Path,
    concurrency: int,
    retry_failed: bool,
    output_mode: str = "prompted",
    reasoning_effort: str | None = None,
) -> None:
    from pydantic_ai import Agent, PromptedOutput

    from research.grading import batch, providers

    require_bedrock_for_effort(spec, reasoning_effort)
    out_path = out_dir / f"{judge_slug(spec)}.jsonl"
    check_provenance(out_path, spec, output_mode, reasoning_effort)
    items = load_generation_items(run_dir, corpus_path)

    # Prompted mode supplies the response schema in the instructions; tool mode
    # supplies it through a tool definition. Both use the same grading schema.
    system_prompt, grade_schema, prompts_version = PROMPT_SETS[ACTIVE_PROMPTS]
    output = PromptedOutput(grade_schema) if output_mode == "prompted" else grade_schema
    model = EscapeRepairModel(providers.resolve(spec))
    settings: dict = {"temperature": 0.0, "max_tokens": MAX_OUTPUT_TOKENS}
    if reasoning_effort:
        # Send the reasoning setting through Bedrock and record it with each verdict.
        settings["bedrock_additional_model_requests_fields"] = {
            "reasoning_effort": reasoning_effort
        }
    agent = Agent(
        model,
        output_type=output,
        system_prompt=system_prompt,
        model_settings=settings,
        retries={"output": 3},
    )

    async def work(item: dict) -> dict:
        base = {
            "question_id": item["question_id"],
            "rep": item["rep"],
            "judge": spec,
            "prompts": prompts_version,
            "template": TEMPLATE_VERSION,
            "output_mode": output_mode,
            "reasoning_effort": reasoning_effort,
        }
        if is_ungradable(item["response"]):
            return {
                **base,
                "verdict": "no_answer",
                "verdict_binary": 0,
                "reason": "empty response",
                "short_circuit": True,
            }
        result = await agent.run(user_prompt(item["question"], item["reference"], item["response"]))
        grade = result.output
        tokens = tokens_of(result)
        record = {
            **base,
            "verdict": grade.verdict,
            "verdict_binary": verdict_binary(grade.verdict),
            "reason": grade.reason,
            "tokens": tokens,
            # The generation's own stop signal: the ground truth behind the
            # judge's no_answer opinion (truncation vs genuine non-answer).
            "finish_reason": item.get("finish_reason"),
        }
        if tokens.get("out", 0) >= MAX_OUTPUT_TOKENS:
            # A capped response may have been cut mid-JSON and only survived via a
            # retry; flag it so the temp-0-loop guard is auditable.
            record["hit_token_cap"] = True
        return record

    await batch.run(
        items,
        item_key,
        work,
        out_path,
        concurrency=concurrency,
        label="gradings",
        retry_failed=retry_failed,
    )
    if model.repairs:
        print(f"  repaired invalid JSON escapes in {model.repairs} replies")


def _atomic_write_text(path: Path, text: str, *, overwrite: bool) -> None:
    """Replace one output atomically after its complete contents reach disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        if overwrite:
            os.replace(temporary, path)
        else:
            try:
                os.link(temporary, path)
            except FileExistsError as exc:
                raise SystemExit(
                    f"{path} appeared while panel outputs were being written; "
                    "refusing to overwrite it without --force"
                ) from exc
    finally:
        temporary.unlink(missing_ok=True)


def _panel_output_paths(
    judge_files: list[Path],
    corpus_path: Path,
    closed_labels: Path,
    out: Path,
    agreement_out: Path | None,
    *,
    force: bool,
) -> Path:
    """Validate panel paths before either output is created or replaced."""
    report_file = agreement_out or (out.parent / "judge_agreement.json")
    resolved_out = out.resolve()
    resolved_report = report_file.resolve()
    if resolved_out == resolved_report:
        raise SystemExit("labels and agreement report must use different paths")

    inputs = [*judge_files, corpus_path, closed_labels]
    for output_name, output_path, resolved_output in (
        ("labels", out, resolved_out),
        ("agreement report", report_file, resolved_report),
    ):
        for input_path in inputs:
            if resolved_output == input_path.resolve():
                raise SystemExit(
                    f"{output_name} path {output_path} collides with input {input_path}"
                )
        if output_path.exists() and not force:
            raise SystemExit(f"{output_path} exists — --force to overwrite panel outputs")
    return report_file


def write_panel(
    judge_files: list[Path],
    corpus_path: Path,
    closed_labels: Path,
    out: Path,
    *,
    expected_k: int = 5,
    agreement_out: Path | None = None,
    allow_partial: bool = False,
    force: bool = False,
) -> None:
    """Validate panel coverage, then write labels and the agreement report."""
    report_file = _panel_output_paths(
        judge_files,
        corpus_path,
        closed_labels,
        out,
        agreement_out,
        force=force,
    )
    panel = build_panel(judge_files)
    with closed_labels.open() as stream:
        covariates = {row["question_id"]: row for row in map(json.loads, stream)}
    labels = panel_labels(panel, covariates)
    # A question absent from ALL THREE files never enters the join, so
    # completeness is checked against the corpus, not against the votes.
    with corpus_path.open() as stream:
        convertible = sum(1 for line in stream if json.loads(line).get("convertible"))
    short_k = [row["question_id"] for row in labels if row["k"] != expected_k]
    if (len(labels) != convertible or short_k) and not allow_partial:
        raise SystemExit(
            f"panel covers {len(labels)} questions but the corpus has "
            f"{convertible} convertible items, and {len(short_k)} have k != {expected_k} "
            f"(first: {short_k[:5]}) — a question judged by nobody vanishes "
            "silently, so the join refuses; --allow-partial to override "
            "for a partial dataset"
        )
    agreement = panel_agreement(panel)
    labels_text = "".join(json.dumps(label) + "\n" for label in labels)
    report_text = json.dumps(agreement, indent=2) + "\n"
    # Publish the labels file of record last. Both files are individually atomic,
    # and a failure while staging either cannot leave a truncated JSON document.
    _atomic_write_text(report_file, report_text, overwrite=force)
    _atomic_write_text(out, labels_text, overwrite=force)
    spread = Counter(label["correct_count"] for label in labels)
    print(f"  items: {len(labels)}  spread: {dict(sorted(spread.items()))}")
    print(
        f"  inter-judge Fleiss kappa: binary {agreement['fleiss_kappa_binary']}, "
        f"three-way {agreement['fleiss_kappa_three_way']}  "
        f"(unanimous {agreement['unanimous_share']:.1%})"
    )
    print(f"  wrote {out} and {report_file}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Modes and conditional requirements:
  Offline panel mode: pass --panel JUDGE_FILE JUDGE_FILE JUDGE_FILE,
  --closed-labels CLOSED_LABELS, --corpus CORPUS, and --out OUTPUT. This only
  joins existing judge files and makes no provider calls.

  Live judge mode: pass --judge PROVIDER:MODEL, --generations RUN_DIR,
  --corpus CORPUS, and --out OUTPUT_DIR. The selected provider's credentials
  must already be available in the environment or provider configuration.

Use --panel for offline aggregation or --judge for live grading. A nonempty
--panel takes precedence if both are supplied. Live grading requires
--generations, a run directory containing rep*/generations.jsonl files.""",
    )
    parser.add_argument("--judge", help="provider:model spec for ONE judge")
    parser.add_argument("--generations", type=Path)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="directory for per-judge files, or the labels JSONL path with --panel",
    )
    parser.add_argument(
        "--panel", nargs="*", type=Path, help="three judge files to join into per-question labels"
    )
    parser.add_argument("--concurrency", type=positive_int, default=8)
    parser.add_argument("--prompts", choices=("v1", "v2"), default="v1",
                        help="v1: the released rubric (default); v2: the M1 definitions (task L1)")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument(
        "--output-mode",
        choices=("prompted", "tool"),
        default="prompted",
        help="how the response schema reaches the model: prompted instructions or a tool definition",
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=("low", "medium", "high"),
        default=None,
        help="reasoning-model effort (gpt-oss calibration setting: low); recorded with each verdict",
    )
    parser.add_argument(
        "--closed-labels",
        type=Path,
        default=None,
        help="closed-corpus labels supplying category/list_style for the panel label rows",
    )
    parser.add_argument(
        "--expected-k",
        type=int,
        default=5,
        help="panel only: expected repeats per item (default 5)",
    )
    parser.add_argument(
        "--agreement-out",
        type=Path,
        default=None,
        help="panel only: path for the agreement report JSON (default judge_agreement.json in out directory)",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="panel only: generate partial-dataset labels for exploratory inspection, "
        "allowing incomplete corpus coverage or repeat counts that differ from --expected-k",
    )
    parser.add_argument(
        "--force", action="store_true", help="panel only: overwrite existing panel outputs"
    )
    args = parser.parse_args(argv)
    global ACTIVE_PROMPTS
    ACTIVE_PROMPTS = args.prompts

    if args.panel:
        if args.closed_labels is None:
            parser.error("--closed-labels is required with --panel")
        out = (
            args.out
            if args.out.suffix == ".jsonl"
            else (args.out / "labels_open.jsonl")
        )
        write_panel(
            args.panel,
            args.corpus,
            args.closed_labels,
            out,
            expected_k=args.expected_k,
            agreement_out=args.agreement_out,
            allow_partial=args.allow_partial,
            force=args.force,
        )
        return 0

    if not args.judge:
        raise SystemExit("either --judge or --panel is required")
    if args.generations is None:
        parser.error("--generations is required with --judge")
    require_calibrated_settings(args.judge, args.output_mode, args.reasoning_effort)
    asyncio.run(
        run_judge(
            args.judge,
            args.generations,
            args.corpus,
            args.out,
            args.concurrency,
            args.retry_failed,
            args.output_mode,
            args.reasoning_effort,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
