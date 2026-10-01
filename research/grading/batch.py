"""Run bounded async work, append results, and resume from recorded item keys.

Transport retries belong to the provider. This runner stops on sustained failures;
completed calls without a saved record may need to be repeated after interruption.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable, Iterable
from pathlib import Path

DEFAULT_CONCURRENCY = 8
# Stop early when no request succeeds.
FAIL_FAST_AFTER = 20
# Also stop sustained failures after earlier successful requests.
ABORT_AFTER_CONSECUTIVE = 30
PROGRESS_EVERY = 200


class BatchAborted(RuntimeError):
    """Everything failed early, so the run stopped rather than paying to continue."""


def load(path: Path) -> dict[str, dict]:
    """Records already on disk, keyed by their `key` field.

    A truncated final line is skipped rather than raised on: that is the normal result of
    a run being killed mid-write, and the item it belonged to is simply redone.
    """
    if not path.exists():
        return {}
    records = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "key" in record:
            records[record["key"]] = record
    return records


def failed_keys(path: Path) -> set[str]:
    """Keys whose recorded result was an error, so a retry pass can target them."""
    return {key for key, record in load(path).items() if "error" in record}


def ensure_final_newline(path: Path) -> None:
    """Separate future records from an unterminated line without removing any bytes."""
    if path.exists() and path.stat().st_size:
        with path.open("rb") as stream:
            stream.seek(-1, 2)
            needs_newline = stream.read(1) != b"\n"
        if needs_newline:
            with path.open("a") as stream:
                stream.write("\n")


async def run(
    items: Iterable[dict],
    key_of: Callable[[dict], str],
    work: Callable[[dict], Awaitable[dict]],
    out_path: Path,
    *,
    concurrency: int = DEFAULT_CONCURRENCY,
    label: str = "items",
    retry_failed: bool = False,
) -> dict[str, dict]:
    """Apply `work` to every item not already recorded, and return every record.

    `work` returns the fields to store for its item, or raises. A raising item is recorded
    with its error and does not stop the rest, so one malformed row cannot cost a whole
    run — but errors are counted, reported, and left in the file for the caller to gate on
    and for `retry_failed` to pick up next time.
    """
    items = list(items)
    done = load(out_path)
    failed = {key for key, record in done.items() if "error" in record}
    skip = set(done) - (failed if retry_failed else set())
    todo = [item for item in items if key_of(item) not in skip]

    print(f"  {len(items):,} {label}: {len(skip):,} done, {len(todo):,} to do")
    if not todo:
        return done

    out_path.parent.mkdir(parents=True, exist_ok=True)
    ensure_final_newline(out_path)
    # Concurrency is bounded here rather than on the model, so there is one place that
    # decides how much work is in flight.
    limiter = asyncio.Semaphore(concurrency)
    write_lock = asyncio.Lock()
    counts = {"ok": 0, "failed": 0, "consecutive": 0}
    # Every item is scheduled at once, so raising is not enough to stop the queue — the
    # coroutines still waiting would each go on to spend a call. They check this first and
    # return immediately once it is set.
    aborted: list[BatchAborted] = []

    with out_path.open("a") as handle:

        async def process(item: dict) -> dict:
            key = key_of(item)
            if aborted:
                raise aborted[0]
            async with limiter:
                if aborted:
                    raise aborted[0]
                try:
                    record = {"key": key, **await work(item)}
                except asyncio.CancelledError:
                    raise
                except Exception as exc:  # noqa: BLE001 - recorded, then gated on above
                    record = {"key": key, "error": f"{type(exc).__name__}: {exc}"}

            async with write_lock:
                handle.write(json.dumps(record) + "\n")
                handle.flush()
                bad = "error" in record
                counts["failed" if bad else "ok"] += 1
                counts["consecutive"] = counts["consecutive"] + 1 if bad else 0
                finished = counts["ok"] + counts["failed"]
                if finished % PROGRESS_EVERY == 0 or finished == len(todo):
                    print(f"    {finished:,}/{len(todo):,}  ({counts['failed']:,} failed)")
                if counts["ok"] == 0 and counts["failed"] >= FAIL_FAST_AFTER:
                    aborted.append(
                        BatchAborted(
                            f"first {counts['failed']} items all failed — stopping before "
                            f"spending more. Last error: {record['error']}"
                        )
                    )
                    raise aborted[0]
                if counts["consecutive"] >= ABORT_AFTER_CONSECUTIVE:
                    aborted.append(
                        BatchAborted(
                            f"{counts['consecutive']} consecutive failures after "
                            f"{counts['ok']:,} successes — stopping. Everything already "
                            f"recorded is kept; resume once the cause is fixed. "
                            f"Last error: {record['error']}"
                        )
                    )
                    raise aborted[0]
            return record

        # `return_exceptions` so that every coroutine has finished before the `with`
        # closes the handle. Letting gather propagate leaves siblings mid-flight, and
        # the first thing they do on waking is write to a file that is no longer open —
        # losing records on the one path where the record matters most.
        results = await asyncio.gather(*(process(item) for item in todo), return_exceptions=True)

    for record in results:
        if not isinstance(record, BaseException):
            done[record["key"]] = record
    raised = [r for r in results if isinstance(r, BaseException)]
    if raised:
        raise raised[0]

    if counts["failed"]:
        print(f"  {counts['failed']:,} failed; rerun with --retry-failed to redo only those")
    return done
