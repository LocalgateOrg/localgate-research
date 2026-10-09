"""Schema-constrained generation (task G1). No model is loaded.

Run: uv run python -m unittest discover tests
"""

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

from research.data import generate, structured
from research.tdm.numeric import match


def answer(steps, text, value=None, unit=None):
    return json.dumps({"steps": steps, "final_answer": {"text": text, "value": value, "unit": unit}})


class SchemaTests(unittest.TestCase):
    def test_reasoning_comes_before_the_answer(self):
        self.assertEqual(list(structured.ANSWER_SCHEMA["properties"]), ["steps", "final_answer"])
        self.assertEqual(structured.ANSWER_SCHEMA["required"], ["steps", "final_answer"])

    def test_final_answer_fields(self):
        fields = structured.ANSWER_SCHEMA["properties"]["final_answer"]
        self.assertEqual(list(fields["properties"]), ["text", "value", "unit"])
        self.assertFalse(fields["additionalProperties"])

    def test_openai_request_enforces_the_schema(self):
        req = structured.openai_request("google/gemma-4-E2B-it", "What is 2+2?", temperature=1.0)
        self.assertEqual(req["response_format"]["type"], "json_schema")
        self.assertTrue(req["response_format"]["json_schema"]["strict"])
        self.assertIs(req["response_format"]["json_schema"]["schema"], structured.ANSWER_SCHEMA)
        self.assertTrue(req["messages"][0]["content"].startswith("What is 2+2?"))
        self.assertEqual(req["temperature"], 1.0)

    def test_digest_is_stable(self):
        self.assertEqual(structured.schema_digest(), structured.schema_digest())
        self.assertEqual(len(structured.schema_digest()), 12)


class NoAnswerRuleTests(unittest.TestCase):
    def test_valid_answer(self):
        out = structured.parse(answer(["Use dT/dP = ΔV/ΔS."], "-3.5 °C", -3.5, "°C"), "stop")
        self.assertEqual(out["status"], "ok")
        self.assertEqual(out["final_answer"]["value"], -3.5)
        self.assertEqual(out["steps"], 1)

    def test_cut_off_is_no_answer_even_if_parseable(self):
        self.assertEqual(structured.parse(answer([], "4", 4), "length")["status"], "no_answer")

    def test_invalid_json_is_no_answer(self):
        for text in ('{"steps": ["a"], "final_answer": {"text": "4"', "", "The answer is 4."):
            with self.subTest(text=text):
                self.assertEqual(structured.parse(text, "stop")["status"], "no_answer")

    def test_schema_violations_are_no_answer(self):
        bad = [
            json.dumps({"final_answer": {"text": "4", "value": 4, "unit": None}}),  # no steps
            json.dumps({"steps": "one", "final_answer": {"text": "4", "value": 4, "unit": None}}),
            json.dumps({"steps": [], "final_answer": {"text": "4", "value": "4", "unit": None}}),
            json.dumps({"steps": [], "final_answer": {"text": "4", "value": True, "unit": None}}),
            json.dumps({"steps": [], "final_answer": {"text": "4", "value": 4, "unit": None, "extra": 1}}),
            answer([], "  ", None),  # nothing committed
        ]
        for text in bad:
            with self.subTest(text=text):
                self.assertEqual(structured.parse(text, "stop")["status"], "no_answer")

    def test_answer_string_feeds_the_numeric_matcher(self):
        out = structured.parse(answer(["…"], "about 3.0e-19 joules", 3.0e-19, "J"), "stop")
        self.assertEqual(match("3.03 × 10^-19 joule", structured.answer_string(out["final_answer"]))[0], "match")
        self.assertEqual(structured.answer_string({"text": "Postmodern ethics", "value": None, "unit": None}),
                         "Postmodern ethics")


class GeneratorTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.corpus = Path(self.dir.name) / "corpus.jsonl"
        rows = [{"question_id": 7, "convertible": True, "category": "physics", "open_question": "What is 2 + 2?"},
                {"question_id": 9, "convertible": False, "category": "law", "open_question": None}]
        self.corpus.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    def tearDown(self):
        self.dir.cleanup()

    def manifest(self, output_format):
        rows = generate.load_open_corpus(self.corpus)
        return generate.build_open_manifest(rows, model="m", revision=None, trust_remote_code=False, limit=None,
                                            repeats=1, seed_base=1, versions={}, energy_telemetry=None,
                                            output_format=output_format)

    def test_free_manifest_is_unchanged(self):
        free = self.manifest("free")
        self.assertEqual(free["task"], "openended-label")
        self.assertNotIn("structured_output", free)

    def test_formats_cannot_share_a_run_directory(self):
        problems = generate.check_manifest_compatible(self.manifest("free"), self.manifest("json"))
        self.assertTrue(any(p.startswith("task") for p in problems))
        self.assertTrue(any(p.startswith("structured_output") for p in problems))
        self.assertEqual(generate.check_manifest_compatible(self.manifest("json"), self.manifest("json")), [])

    def test_dry_run_builds_requests_without_vllm(self):
        sys.modules.pop("vllm", None)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = generate.main(["open", "--corpus", str(self.corpus), "--out", str(Path(self.dir.name) / "run"),
                                  "--output-format", "json", "--dry-run"])
        self.assertEqual(code, 0)
        self.assertNotIn("vllm", sys.modules)
        report = json.loads(out.getvalue())
        self.assertEqual(report["manifest"]["structured_output"]["digest"], structured.schema_digest())
        self.assertTrue(report["example"]["vllm_chat_message"]["content"].endswith(structured.INSTRUCTION))
        self.assertEqual(report["example"]["openai_compatible_request"]["response_format"]["type"], "json_schema")
        self.assertFalse(Path(self.dir.name, "run").exists())


if __name__ == "__main__":
    unittest.main()
