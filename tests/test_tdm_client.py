"""Unit tests for the decision client (task S3). No network: providers are faked.

Run: uv run python -m unittest discover tests
"""

import json
import math
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from research.tdm import client
from research.tdm.instruments import FILTER, JUDGE_CHOICE, JUDGE_NOULS

FILTER_ROWS = [
    {"question_id": 101, "question": "What is the SI unit of force?", "options": ["newton", "joule", "watt"],
     "reference_answer": "newton"},
    {"question_id": 102, "question": "Which of the following is NOT a noble gas?", "options": ["neon", "argon", "nitrogen"],
     "reference_answer": "nitrogen"},
    {"question_id": 103, "question": "What is 2 + 2?", "options": ["3", "4", "5"], "reference_answer": "4"},
]
JUDGING_ROW = {"key": "7:0", "question_id": 7, "question": "Boiling point of water at 1 atm in °C?",
               "reference_answer": "100", "response": "It boils at 100 °C."}


def fake_answer(questions: dict) -> dict:
    """What a Decisions endpoint returns: one typed answer per question."""
    return {"answers": {name: {"type": q["type"], "probability": 0.8} for name, q in questions.items()},
            "usage": {"input_tokens": 120, "cost": 0.000005}, "model": "typesafe/jev-1.13-20260917"}


class SpecTests(unittest.TestCase):
    def test_known_providers(self):
        self.assertEqual(client.parse_spec("openrouter:typesafe/jev-1.13-20260917"), ("openrouter", "typesafe/jev-1.13-20260917"))
        self.assertEqual(client.parse_spec("readout:qwen3-32b"), ("readout", "qwen3-32b"))

    def test_rejects_unknown_provider_and_floating_alias(self):
        with self.assertRaises(SystemExit):
            client.parse_spec("bedrock:some-model")
        with self.assertRaises(SystemExit):
            client.parse_spec("openrouter:typesafe/jev-latest")

    def test_openrouter_request_needs_key_and_carries_model(self):
        body = FILTER.body(FILTER_ROWS[0])
        with mock.patch.dict(os.environ, {}, clear=True), self.assertRaises(SystemExit):
            client.request("openrouter", "typesafe/jev-1.13-20260917", body)
        with mock.patch.dict(os.environ, {"OPENROUTER_API_KEY": "test"}, clear=True):
            url, headers, payload = client.request("openrouter", "typesafe/jev-1.13-20260917", body)
        self.assertTrue(url.startswith("https://openrouter.ai/"))
        self.assertEqual(payload["model"], "typesafe/jev-1.13-20260917")
        self.assertEqual(payload["questions"], body["questions"])


class UnwrapTests(unittest.TestCase):
    def test_workers_ai_wrapper(self):
        q = FILTER.questions
        out = client.unwrap("workers-ai", {"success": True, "result": fake_answer(q)}, q)
        self.assertEqual(set(out["answers"]), set(q))
        self.assertEqual(out["usage"]["input_tokens"], 120)

    def test_rejects_missing_or_mistyped_answers(self):
        q = JUDGE_NOULS.questions
        partial = {"answers": {"commits": {"type": "noul"}}}
        with self.assertRaises(client.ProviderError):
            client.unwrap("openrouter", partial, q)
        wrong_type = {"answers": {name: {"type": "choice"} for name in q}}
        with self.assertRaises(client.ProviderError):
            client.unwrap("openrouter", wrong_type, q)
        with self.assertRaises(client.ProviderError):
            client.unwrap("workers-ai", {"success": False, "errors": ["quota"]}, q)


class ReadoutTests(unittest.TestCase):
    def test_prompt_lists_every_criterion_with_a_letter(self):
        question = JUDGE_CHOICE.questions["verdict"]
        messages, letters = client.readout_prompt({"question": "q", "reference_answer": "r", "response": "x"}, question)
        self.assertEqual(letters, {"match": "A", "no_match": "B", "no_answer": "C"})
        for letter in "ABC":
            self.assertIn(f"\n{letter}. ", messages[1]["content"])

    def test_score_questions_are_refused(self):
        with self.assertRaises(client.ProviderError):
            client.readout_prompt({}, {"type": "score", "instructions": "", "criteria": {}})

    def test_probabilities_renormalise_over_letters(self):
        letters = {"true": "A", "false": "B"}
        top = [{"token": " A", "logprob": math.log(0.6)}, {"token": "b", "logprob": math.log(0.2)},
               {"token": "The", "logprob": math.log(0.2)}]
        probs, mass = client.readout_probabilities(top, letters)
        self.assertAlmostEqual(probs["true"], 0.75)
        self.assertAlmostEqual(probs["false"], 0.25)
        self.assertAlmostEqual(mass, 0.8)

    def test_no_letter_in_top_logprobs_is_an_error(self):
        with self.assertRaises(client.ProviderError):
            client.readout_probabilities([{"token": "Sure", "logprob": -0.1}], {"true": "A", "false": "B"})

    def test_send_readout_asks_each_question_and_sums_usage(self):
        def reply(*_args, json, **_kwargs):
            self.assertEqual(json["max_tokens"], 1)
            self.assertTrue(json["logprobs"])
            data = {"model": "qwen3-32b", "usage": {"prompt_tokens": 300, "completion_tokens": 1},
                    "choices": [{"message": {"content": "A"},
                                 "logprobs": {"content": [{"top_logprobs": [
                                     {"token": "A", "logprob": math.log(0.9)}, {"token": "B", "logprob": math.log(0.1)}]}]}}]}
            return mock.Mock(status_code=200, json=lambda: data, raise_for_status=lambda: None)

        with mock.patch("httpx.post", side_effect=reply) as post:
            out = client.send_readout("qwen3-32b", JUDGE_NOULS.body(JUDGING_ROW))
        self.assertEqual(post.call_count, 2)  # one call per question
        self.assertAlmostEqual(out["answers"]["commits"]["probabilities"]["true"], 0.9)
        self.assertEqual(out["usage"]["input_tokens"], 600)
        self.assertEqual(out["served_model"], "qwen3-32b")


class RunTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.log = Path(self.dir.name) / "run.jsonl"
        self.spec = "openrouter:typesafe/jev-1.13-20260917"

    def tearDown(self):
        self.dir.cleanup()

    def records(self):
        return [json.loads(line) for line in self.log.read_text().splitlines()]

    def test_dry_run_logs_bodies_with_provenance(self):
        counts = client.run(FILTER, FILTER_ROWS, self.spec, self.log, ["base", "flipped"], dry_run=True)
        self.assertEqual(counts["sent"], 6)
        rec = self.records()[0]
        self.assertEqual(rec["instrument_hash"], FILTER.digest())
        self.assertEqual(rec["model"], self.spec)
        self.assertIn("state", rec["dry_run_body"])
        self.assertEqual(len(rec["request_sha256"]), 16)
        self.assertIn("called_at", rec)

    def test_resume_skips_done_and_retries_errors(self):
        calls = []

        def flaky(provider, model, body):
            calls.append(body["state"]["question"])
            if len(calls) == 2:
                raise client.ProviderError("temporary")
            return {**client.unwrap(provider, fake_answer(body["questions"]), body["questions"]), "latency_s": 0.2}

        with mock.patch.object(client, "send", side_effect=flaky):
            first = client.run(FILTER, FILTER_ROWS, self.spec, self.log, ["base"], dry_run=False)
            second = client.run(FILTER, FILTER_ROWS, self.spec, self.log, ["base"], dry_run=False)
        self.assertEqual(first["errors"], 1)
        self.assertEqual(second["skipped"], 2)  # the two successes
        self.assertEqual(second["sent"], 1)  # the failed item is retried
        ok = [r for r in self.records() if "error" not in r]
        self.assertEqual(len(ok), 3)
        # tokens, cost, latency and the full answer (with its probabilities) are on every record
        for r in ok:
            self.assertEqual(r["usage"]["input_tokens"], 120)
            self.assertIn("probability", next(iter(r["answers"].values())))
            self.assertIn("latency_s", r)
        self.assertEqual(first["input_tokens"], 240)
        self.assertAlmostEqual(first["cost"], 0.00001)

    def test_refuses_to_mix_models_or_instruments_in_one_log(self):
        client.run(FILTER, FILTER_ROWS[:1], self.spec, self.log, ["base"], dry_run=True)
        with self.assertRaises(SystemExit):
            client.run(FILTER, FILTER_ROWS[:1], "workers-ai:@cf/cloudflare/clef", self.log, ["base"], dry_run=True)
        with self.assertRaises(SystemExit):
            client.run(JUDGE_NOULS, [JUDGING_ROW], self.spec, self.log, ["base"], dry_run=True)

    def test_flipped_variant_only_for_instruments_that_have_one(self):
        counts = client.run(JUDGE_CHOICE, [JUDGING_ROW], self.spec, self.log, ["base", "flipped"], dry_run=True)
        self.assertEqual(counts["sent"], 1)


if __name__ == "__main__":
    unittest.main()
