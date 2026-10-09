"""Send instruments to typed decision models and log every call (task S3).

Model specs name a provider and a pinned model:

  openrouter:typesafe/jev-1.13-20260917   Jev through OpenRouter's Decisions API
  workers-ai:@cf/cloudflare/clef           Clef on Cloudflare Workers AI
  workers-ai:@cf/cloudflare/clef-flash     Clef-flash on Cloudflare Workers AI
  readout:<model>                          an ordinary LLM read out over option letters

Jev and Clef take the same body (``state`` + ``questions``); Workers AI wraps
its answer under ``result``. The open readout (task S4, decision D4) sends each
question to any OpenAI-compatible chat endpoint (a hosted API or vLLM), asks for
one option letter and reads the next-token log-probabilities over the letters.
Keys are read from the environment only: OPENROUTER_API_KEY;
CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN; READOUT_BASE_URL and
READOUT_API_KEY.

Logs are append-only JSONL, one record per (item, variant). A run resumes by
skipping completed keys and refuses a log written with a different instrument
hash or model, as the grading module does. ``--dry-run`` builds and logs the
request bodies without sending anything and needs no keys.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from datetime import UTC, datetime
from pathlib import Path

from research.tdm.instruments import INSTRUMENTS, Instrument

PROVIDERS = {
    "openrouter": "https://openrouter.ai/api/alpha/decisions",
    "workers-ai": "https://api.cloudflare.com/client/v4/accounts/{account}/ai/run/{model}",
    "readout": "{base}/chat/completions",
}
READOUT_DEFAULT_BASE = "http://127.0.0.1:8000/v1"  # vLLM's OpenAI-compatible server
READOUT_SYSTEM = (
    "You answer one typed question about the state you are given. "
    "Reply with exactly one option letter and nothing else."
)


class ProviderError(RuntimeError):
    """A provider returned an error or an answer that does not fit the request."""


def parse_spec(spec: str) -> tuple[str, str]:
    provider, _, model = spec.partition(":")
    if provider not in PROVIDERS or not model:
        raise SystemExit(f"model spec must be provider:model with provider in {sorted(PROVIDERS)}: {spec!r}")
    if model.endswith("latest"):
        raise SystemExit("refusing a floating alias; pin a dated snapshot (decision D5)")
    return provider, model


def request(provider: str, model: str, body: dict) -> tuple[str, dict, dict]:
    """URL, headers and JSON payload for one call."""
    if provider == "openrouter":
        key = os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise SystemExit("OPENROUTER_API_KEY is not set")
        return PROVIDERS[provider], {"Authorization": f"Bearer {key}"}, {"model": model, **body}
    account = os.environ.get("CLOUDFLARE_ACCOUNT_ID")
    token = os.environ.get("CLOUDFLARE_API_TOKEN")
    if not account or not token:
        raise SystemExit("CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN must be set")
    url = PROVIDERS[provider].format(account=account, model=model)
    return url, {"Authorization": f"Bearer {token}"}, body


def readout_prompt(state: dict, question: dict) -> tuple[list[dict], dict[str, str]]:
    """Chat messages for one question, and the option letter for each criterion."""
    if question["type"] not in ("noul", "choice"):
        raise ProviderError(f"the open readout supports noul and choice questions, not {question['type']}")
    letters = {name: chr(65 + i) for i, name in enumerate(question["criteria"])}
    lines = [f"## {key.replace('_', ' ')}\n{value}" for key, value in state.items()]
    lines += ["", f"Question: {question['instructions']}", "", "Options:"]
    lines += [f"{letters[name]}. {label}" for name, label in question["criteria"].items()]
    lines += ["", f"Answer with one letter: {', '.join(letters.values())}."]
    return [{"role": "system", "content": READOUT_SYSTEM}, {"role": "user", "content": "\n".join(lines)}], letters


def readout_probabilities(top_logprobs: list[dict], letters: dict[str, str]) -> tuple[dict[str, float], float]:
    """Probability per criterion from the first token's top log-probabilities.

    Tokens are matched after stripping whitespace, case-insensitively, so " A"
    and "a" both count for A. Returns the renormalised distribution and the
    probability mass the letters had before renormalising (low mass means the
    model wanted to say something else).
    """
    mass = {name: 0.0 for name in letters}
    by_letter = {letter: name for name, letter in letters.items()}
    for entry in top_logprobs:
        name = by_letter.get(entry["token"].strip().upper())
        if name is not None:
            mass[name] += math.exp(entry["logprob"])
    total = sum(mass.values())
    if total == 0:
        raise ProviderError("none of the option letters is among the top log-probabilities")
    return {name: p / total for name, p in mass.items()}, total


def send_readout(model: str, body: dict, retries: int = 4) -> dict:
    import httpx

    base = os.environ.get("READOUT_BASE_URL", READOUT_DEFAULT_BASE).rstrip("/")
    key = os.environ.get("READOUT_API_KEY", "")
    url = PROVIDERS["readout"].format(base=base)
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    answers, usage, latency, served = {}, {"input_tokens": 0, "output_tokens": 0}, 0.0, None
    for name, question in body["questions"].items():
        messages, letters = readout_prompt(body["state"], question)
        payload = {"model": model, "messages": messages, "max_tokens": 1, "temperature": 0,
                   "logprobs": True, "top_logprobs": 20}
        for attempt in range(retries + 1):
            started = time.perf_counter()
            try:
                response = httpx.post(url, headers=headers, json=payload, timeout=60)
                if response.status_code in (429, 500, 502, 503, 504) and attempt < retries:
                    time.sleep(2**attempt)
                    continue
                response.raise_for_status()
                break
            except httpx.TransportError:
                if attempt == retries:
                    raise
                time.sleep(2**attempt)
        latency += time.perf_counter() - started
        data = response.json()
        served = data.get("model", served)
        choice = data["choices"][0]
        content = (choice.get("logprobs") or {}).get("content") or []
        if not content:
            raise ProviderError("the endpoint returned no log-probabilities; it must support logprobs")
        probs, coverage = readout_probabilities(content[0]["top_logprobs"], letters)
        answers[name] = {"type": question["type"], "probabilities": probs, "letter_mass": round(coverage, 6),
                         "sampled": (choice.get("message") or {}).get("content")}
        u = data.get("usage") or {}
        usage["input_tokens"] += u.get("prompt_tokens") or 0
        usage["output_tokens"] += u.get("completion_tokens") or 0
    return {"answers": answers, "usage": usage, "served_model": served, "latency_s": round(latency, 4)}


def unwrap(provider: str, payload: dict, questions: dict) -> dict:
    """Return {answers, usage, served_model} and check every question was answered."""
    if provider == "workers-ai":
        if not payload.get("success", True):
            raise ProviderError(f"workers-ai error: {payload.get('errors')}")
        payload = payload.get("result", payload)
    answers = payload.get("answers")
    if not isinstance(answers, dict) or set(answers) != set(questions):
        raise ProviderError(f"answers do not match questions: {sorted(answers or {})} vs {sorted(questions)}")
    for name, question in questions.items():
        if answers[name].get("type") != question["type"]:
            raise ProviderError(f"{name}: answered as {answers[name].get('type')}, asked as {question['type']}")
    return {"answers": answers, "usage": payload.get("usage"), "served_model": payload.get("model")}


def send(provider: str, model: str, body: dict, retries: int = 4) -> dict:
    import httpx

    if provider == "readout":
        return send_readout(model, body, retries)
    url, headers, payload = request(provider, model, body)
    for attempt in range(retries + 1):
        started = time.perf_counter()
        try:
            response = httpx.post(url, headers=headers, json=payload, timeout=60)
            latency = time.perf_counter() - started
            if response.status_code in (429, 500, 502, 503, 504) and attempt < retries:
                time.sleep(2**attempt)
                continue
            response.raise_for_status()
            return {**unwrap(provider, response.json(), body["questions"]), "latency_s": round(latency, 4)}
        except httpx.TransportError:
            if attempt == retries:
                raise
            time.sleep(2**attempt)
    raise ProviderError("unreachable")


def record_key(row: dict, variant: str) -> str:
    item = row.get("key") or str(row["question_id"])
    return f"{item}|{variant}"


def check_log(path: Path, instrument: Instrument, spec: str) -> set[str]:
    """Completed keys; refuse a log from another instrument version or model."""
    done = set()
    if not path.exists():
        return done
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("instrument_hash") != instrument.digest() or record.get("model") != spec:
            raise SystemExit(
                f"{path}:{number} was written by {record.get('instrument')}@{record.get('instrument_hash')} "
                f"with {record.get('model')}; refusing to mix instruments or models in one log"
            )
        if "error" not in record:
            done.add(record["key"])
    return done


def run(
    instrument: Instrument,
    rows: list[dict],
    spec: str,
    out: Path,
    variants: list[str],
    dry_run: bool,
) -> dict:
    provider, model = parse_spec(spec)
    done = check_log(out, instrument, spec)
    counts = {"sent": 0, "skipped": 0, "errors": 0, "input_tokens": 0, "cost": 0.0}
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as log:
        for row in rows:
            for variant in variants:
                if variant == "flipped" and not instrument.flipped:
                    continue
                key = record_key(row, variant)
                if key in done:
                    counts["skipped"] += 1
                    continue
                body = instrument.body(row, variant)
                record = {
                    "key": key,
                    "variant": variant,
                    "instrument": instrument.name,
                    "instrument_hash": instrument.digest(),
                    "model": spec,
                    "request_sha256": hashlib.sha256(
                        json.dumps(body, sort_keys=True, ensure_ascii=False).encode()
                    ).hexdigest()[:16],
                    "called_at": datetime.now(UTC).isoformat(timespec="seconds"),
                }
                if dry_run:
                    record["dry_run_body"] = body
                else:
                    try:
                        record.update(send(provider, model, body))
                        usage = record.get("usage") or {}
                        counts["input_tokens"] += usage.get("input_tokens") or 0
                        counts["cost"] += usage.get("cost") or 0.0
                    except Exception as exc:  # logged and retried on the next resume
                        record["error"] = f"{type(exc).__name__}: {exc}"
                        counts["errors"] += 1
                log.write(json.dumps(record, ensure_ascii=False) + "\n")
                log.flush()
                counts["sent"] += 1
    return counts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--instrument", choices=sorted(INSTRUMENTS), required=True)
    parser.add_argument("--items", type=Path, required=True, help="output directory of `localgate-tdm items`")
    parser.add_argument("--model", required=True, help="provider:model, e.g. openrouter:typesafe/jev-1.13-20260917 or readout:<model>")
    parser.add_argument("--out", type=Path, required=True, help="append-only JSONL log")
    parser.add_argument("--variants", nargs="+", default=["base"], choices=["base", "flipped"])
    parser.add_argument("--human-labelled-only", action="store_true", help="only rows that carry human labels")
    parser.add_argument("--ids", type=Path, help="file with one item key or question_id per line (e.g. a dev split)")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dry-run", action="store_true", help="build and log requests without sending")
    args = parser.parse_args(argv)

    instrument = INSTRUMENTS[args.instrument]
    table = args.items / f"{instrument.stage}.jsonl"
    rows = [json.loads(line) for line in table.open(encoding="utf-8") if line.strip()]
    if args.human_labelled_only:
        label = {"filter": "human_votes", "rewrite": "human_same_question", "judging": "human_verdicts"}
        rows = [r for r in rows if r.get(label[instrument.stage]) is not None]
    if args.ids:
        wanted = {line.strip() for line in args.ids.read_text().splitlines() if line.strip()}
        rows = [r for r in rows if str(r.get("key") or r["question_id"]) in wanted]
    if args.limit:
        rows = rows[: args.limit]
    counts = run(instrument, rows, args.model, args.out, args.variants, args.dry_run)
    print(f"{instrument.name}@{instrument.digest()} on {args.model}: {counts}")
    return 0
