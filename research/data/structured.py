"""Schema-constrained generation: answers as JSON, reasoning first (task G1, decision D9).

New generations can be forced into one JSON object during decoding: an array of
reasoning ``steps`` first, then a ``final_answer`` with ``text``, ``value`` and
``unit``. Grading then compares a short final answer with the reference instead
of searching a long response, and numeric answers can be checked in code
(research/tdm/numeric.py). Output cut off by the token limit, or not valid
against the schema, is ``no_answer`` by rule and never reaches a judge.

The same schema serves two back ends:

- vLLM's offline engine (``structured_outputs_params``), as used by
  ``localgate-data generate open --output-format json``;
- any OpenAI-compatible API that supports JSON-schema response formats
  (``openai_request``), for running the answering model on a hosted endpoint.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

SCHEMA_VERSION = "answer-v1"

# Property order matters: constrained decoders emit properties in schema order,
# so the steps are written before the model commits to a final answer.
ANSWER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "steps": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Reasoning, one step per item, before the final answer.",
        },
        "final_answer": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "The answer as you would state it."},
                "value": {"type": ["number", "null"], "description": "The number, if the answer is one quantity."},
                "unit": {"type": ["string", "null"], "description": "Its unit, if any."},
            },
            "required": ["text", "value", "unit"],
            "additionalProperties": False,
        },
    },
    "required": ["steps", "final_answer"],
    "additionalProperties": False,
}

# Appended to the question so the model knows what the constrained format is for.
INSTRUCTION = (
    "\n\nAnswer in JSON: first your reasoning as a list of steps, then a final_answer "
    "with the answer as text and, if it is a single quantity, its numeric value and unit."
)


def schema_digest() -> str:
    blob = json.dumps({"version": SCHEMA_VERSION, "schema": ANSWER_SCHEMA, "instruction": INSTRUCTION},
                      sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


def prompt(question: str) -> str:
    return question + INSTRUCTION


def structured_outputs_params():
    """vLLM sampling-parameter keyword for the schema (vLLM >= 0.11 or older)."""
    try:
        from vllm.sampling_params import StructuredOutputsParams

        return {"structured_outputs": StructuredOutputsParams(json=ANSWER_SCHEMA)}
    except ImportError:
        from vllm.sampling_params import GuidedDecodingParams

        return {"guided_decoding": GuidedDecodingParams(json=ANSWER_SCHEMA)}


def openai_request(model: str, question: str, **sampling: Any) -> dict[str, Any]:
    """Chat-completions payload that enforces the schema on a hosted endpoint."""
    return {
        "model": model,
        "messages": [{"role": "user", "content": prompt(question)}],
        "response_format": {"type": "json_schema",
                            "json_schema": {"name": SCHEMA_VERSION, "schema": ANSWER_SCHEMA, "strict": True}},
        **sampling,
    }


def parse(text: str, finish_reason: str | None) -> dict[str, Any]:
    """Apply the no_answer rule and return the final answer when there is one.

    Returns ``{"status": "ok", "final_answer": {...}, "steps": n}`` or
    ``{"status": "no_answer", "reason": ...}``.
    """
    if finish_reason == "length":
        return {"status": "no_answer", "reason": "cut off at the token limit"}
    try:
        data = json.loads(text)
    except (TypeError, json.JSONDecodeError):
        return {"status": "no_answer", "reason": "not valid JSON"}
    problem = _check(data)
    if problem:
        return {"status": "no_answer", "reason": problem}
    answer = data["final_answer"]
    if not answer["text"].strip() and answer["value"] is None:
        return {"status": "no_answer", "reason": "empty final answer"}
    return {"status": "ok", "final_answer": answer, "steps": len(data["steps"])}


def _check(data: Any) -> str | None:
    """Validate against ANSWER_SCHEMA without a JSON-schema dependency."""
    if not isinstance(data, dict) or set(data) != {"steps", "final_answer"}:
        return "top level must have exactly steps and final_answer"
    if not isinstance(data["steps"], list) or not all(isinstance(s, str) for s in data["steps"]):
        return "steps must be a list of strings"
    answer = data["final_answer"]
    if not isinstance(answer, dict) or set(answer) != {"text", "value", "unit"}:
        return "final_answer must have exactly text, value and unit"
    if not isinstance(answer["text"], str):
        return "final_answer.text must be a string"
    if answer["value"] is not None and (isinstance(answer["value"], bool) or not isinstance(answer["value"], (int, float))):
        return "final_answer.value must be a number or null"
    if answer["unit"] is not None and not isinstance(answer["unit"], str):
        return "final_answer.unit must be a string or null"
    return None


def answer_string(answer: dict[str, Any]) -> str:
    """One string for the numeric matcher: value and unit when present, else the text."""
    if answer.get("value") is not None:
        return f"{answer['value']:g} {answer.get('unit') or ''}".strip()
    return answer.get("text", "")
