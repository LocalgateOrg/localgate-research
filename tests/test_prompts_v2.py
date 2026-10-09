"""Prompt set v2 for the LLM pipeline rerun (task L1). No model is called.

Run: uv run python -m unittest discover tests
"""

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from research.data import convert, prompts_v2
from research.grading import judge

CASE = ("Wilshire Street is a public thoroughfare, designated as a one-way street for northbound traffic. "
        "Wilshire and Figueroa Streets intersect at right angles. The intersection is controlled by traffic lights. "
        "A businessman was driving his car east on Figueroa Street and did not see the traffic light. He entered the "
        "intersection when the light was red for eastbound traffic. A woman, in violation of statute, was proceeding "
        "south on Wilshire Street and struck the businessman's car. Which of the following is the appropriate judgment?")


class FrozenV1Tests(unittest.TestCase):
    def test_released_instruments_unchanged(self):
        # These digests are stamped on every released conversion record and verdict.
        self.assertEqual(convert.PROMPT_SETS["v1"].version, "93057f756115")
        self.assertEqual(judge.PROMPT_SETS["v1"][2], "e2c597632659")

    def test_v1_is_the_default(self):
        args = convert.build_parser().parse_args(["judge", "--labels", "l", "--answer-key", "a", "--out", "o"])
        self.assertEqual(args.prompts, "v1")


class V2ContentTests(unittest.TestCase):
    def test_v2_is_shorter(self):
        pairs = [(convert.SYSTEM_JUDGE, prompts_v2.SYSTEM_JUDGE), (convert.SYSTEM_RESCORE, prompts_v2.SYSTEM_RESCORE),
                 (convert.SYSTEM_REWRITE, prompts_v2.SYSTEM_REWRITE), (judge.SYSTEM_GRADE, prompts_v2.SYSTEM_GRADE)]
        for old, new in pairs:
            with self.subTest(new=new[:40]):
                self.assertLess(len(new.split()), len(old.split()))
                self.assertLessEqual(len(new.split()), 450)

    def test_m1_definitions_are_in_the_prompts(self):
        self.assertIn("within 1%", prompts_v2.SYSTEM_JUDGE)
        self.assertIn("answer_key_ok", prompts_v2.SYSTEM_JUDGE)
        self.assertIn("THE standard textbook answer", prompts_v2.SYSTEM_JUDGE)
        self.assertIn("within 1%", prompts_v2.SYSTEM_RESCORE)
        self.assertIn("caveat", prompts_v2.SYSTEM_GRADE)
        self.assertIn("equally final", prompts_v2.SYSTEM_GRADE)
        self.assertNotIn("omits", prompts_v2.SYSTEM_GRADE)

    def test_verdict_reports_the_answer_key_after_the_decision(self):
        fields = list(prompts_v2.Verdict.model_fields)
        self.assertEqual(fields, ["reason", "convertible", "answer_key_ok", "failure"])

    def test_v2_has_its_own_digests(self):
        self.assertNotEqual(convert.PROMPT_SETS["v2"].version, convert.PROMPT_SETS["v1"].version)
        self.assertNotEqual(judge.PROMPT_SETS["v2"][2], judge.PROMPT_SETS["v1"][2])


class StemSplitTests(unittest.TestCase):
    def test_short_stems_are_not_split(self):
        self.assertEqual(prompts_v2.split_stem("The capital of France is"), ("", "The capital of France is"))

    def test_long_case_keeps_the_facts_as_context(self):
        context, question = prompts_v2.split_stem(CASE)
        self.assertTrue(context.startswith("Wilshire Street"))
        self.assertIn("struck the businessman's car.", context)
        self.assertEqual(question, "Which of the following is the appropriate judgment?")

    def test_a_very_short_last_sentence_takes_the_one_before(self):
        stem = CASE.replace(" Which of the following is the appropriate judgment?", " Who pays? Why?")
        context, question = prompts_v2.split_stem(stem)
        self.assertEqual(question, "Who pays? Why?")

    def test_rewrite_input_marks_the_parts(self):
        text = prompts_v2.rewrite_input(CASE)
        self.assertTrue(text.startswith("CONTEXT (kept word for word): Wilshire"))
        self.assertIn("\n\nQUESTION: Which of the following", text)
        self.assertEqual(prompts_v2.rewrite_input("The capital of France is"), "QUESTION: The capital of France is")

    def test_assembled_rewrite_keeps_the_case(self):
        full, flagged = prompts_v2.assemble_rewrite(CASE, "What is the appropriate judgment in the case above?")
        self.assertTrue(full.startswith("Wilshire Street"))
        self.assertTrue(full.endswith("in the case above?"))
        self.assertFalse(flagged)

    def test_a_long_stem_rewritten_much_shorter_is_flagged(self):
        stem = "Given " + "a long chain of numerical data, " * 20 + "find x"  # one sentence: cannot be split
        full, flagged = prompts_v2.assemble_rewrite(stem, "Find x.")
        self.assertEqual(full, "Find x.")
        self.assertTrue(flagged)


class FakeAgent:
    """Answers like a rewrite model and records what it was asked."""

    def __init__(self, question):
        self.question, self.prompts = question, []

    async def run(self, prompt):
        self.prompts.append(prompt)
        usage = SimpleNamespace(input_tokens=10, output_tokens=5)
        return SimpleNamespace(output=prompts_v2.Rewrite(standalone_question=self.question, possible=True), usage=usage)


class RewriteRunTests(unittest.TestCase):
    def tearDown(self):
        convert.use_prompts("v1")

    def run_rewrite(self, prompt_set, agent):
        convert.use_prompts(prompt_set)
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "rewrite.jsonl"
            row = {"question_id": 1029, "question": CASE}
            asyncio.run(convert.run_rewrite([row], agent, out, concurrency=1, retry_failed=False, model_name="m"))
            return [json.loads(line) for line in out.read_text().splitlines()]

    def test_v2_rewrites_only_the_question_and_keeps_the_case(self):
        agent = FakeAgent("What is the appropriate judgment in the case above?")
        (record,) = self.run_rewrite("v2", agent)
        self.assertIn("CONTEXT (kept word for word)", agent.prompts[0])
        self.assertTrue(record["open_question"].startswith("Wilshire Street"))
        self.assertEqual(record["prompts"], convert.PROMPT_SETS["v2"].version)
        self.assertNotIn("length_flag", record)

    def test_v1_behaves_as_released(self):
        agent = FakeAgent("What is the appropriate judgment in the case?")
        (record,) = self.run_rewrite("v1", agent)
        self.assertEqual(agent.prompts[0], f"Question: {CASE}")
        self.assertEqual(record["open_question"], "What is the appropriate judgment in the case?")
        self.assertEqual(record["prompts"], "93057f756115")


if __name__ == "__main__":
    unittest.main()
