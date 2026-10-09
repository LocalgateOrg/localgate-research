"""Instruments and consistency variants (task I1). No model is called.

Run: uv run python -m unittest discover tests
"""

import json
import tempfile
import unittest
from pathlib import Path

from research.tdm import client
from research.tdm.instruments import (
    FILTER, INSTRUMENTS, JUDGE_CHOICE, JUDGE_CHOICE_V1, JUDGE_NOULS, JUDGE_NOULS_V1, JUDGE_NOULS_V1_FINAL,
    NEUTRAL_SENTENCE, REWRITE, REWRITE_V1, FILTER_V1,
)

FILTER_ROW = {"question_id": 101, "question": "What is the SI unit of force?", "options": ["newton", "joule", "watt"],
              "reference_answer": "newton"}
REWRITE_ROW = {"question_id": 5, "original_question": "Which of the following is a noble gas?",
               "options": ["neon", "nitrogen"], "reference_answer": "neon", "open_question": "Name a noble gas."}
JUDGING_ROW = {"key": "7:0", "question_id": 7, "question": "Boiling point of water at 1 atm in °C?",
               "reference_answer": "100", "response": "It boils at 100 °C.", "wrong_reference": "90"}


class FrozenTests(unittest.TestCase):
    def test_v0_hashes_unchanged(self):
        # Logged runs refer to these; changing v0 would orphan them.
        self.assertEqual(JUDGE_NOULS.digest(), "558e7365f438")
        self.assertEqual(JUDGE_CHOICE.digest(), "80233d116c08")

    def test_every_instrument_has_a_unique_hash(self):
        digests = [i.digest() for i in INSTRUMENTS.values()]
        self.assertEqual(len(digests), len(set(digests)))


class V1Tests(unittest.TestCase):
    def test_labels_are_short(self):
        for name, instrument in INSTRUMENTS.items():
            if not name.endswith(("-v1", "-v1-final")):
                continue
            for question in instrument.questions.values():
                for label in question["criteria"].values():
                    with self.subTest(instrument=name, label=label[:40]):
                        self.assertLessEqual(len(label.split()), 40)

    def test_filter_v1_asks_about_the_answer_key_separately(self):
        self.assertEqual(set(FILTER_V1.questions), {"single_answer_without_options", "answer_key_usable"})
        self.assertIn("1%", FILTER_V1.questions["single_answer_without_options"]["criteria"]["true"])

    def test_rewrite_v1_counts_dropped_context_as_changed(self):
        self.assertIn("passage", REWRITE_V1.questions["same_question"]["criteria"]["false"])

    def test_judging_v1_no_longer_penalises_wording(self):
        no_match = JUDGE_CHOICE_V1.questions["verdict"]["criteria"]["no_match"]
        self.assertNotIn("omits", no_match)
        self.assertIn("same thing", no_match)


class VariantTests(unittest.TestCase):
    def test_flipped_swaps_the_meaning_of_true_and_false(self):
        for instrument in (FILTER_V1, REWRITE_V1, JUDGE_NOULS_V1):
            for name, question in instrument.questions.items():
                flipped = instrument.flipped[name]
                with self.subTest(instrument=instrument.name, question=name):
                    self.assertEqual(flipped["criteria"]["true"], question["criteria"]["false"])
                    self.assertEqual(flipped["criteria"]["false"], question["criteria"]["true"])
                    self.assertNotEqual(flipped["instructions"], question["instructions"])

    def test_rotated_changes_choice_order_only(self):
        base = JUDGE_CHOICE_V1.body(JUDGING_ROW)["questions"]["verdict"]["criteria"]
        rotated = JUDGE_CHOICE_V1.body(JUDGING_ROW, "rotated")["questions"]["verdict"]["criteria"]
        self.assertEqual(list(base), ["match", "no_match", "no_answer"])
        self.assertEqual(list(rotated), ["no_answer", "match", "no_match"])
        self.assertEqual(dict(base), dict(rotated))

    def test_context_adds_one_neutral_sentence_to_the_response(self):
        body = JUDGE_NOULS_V1.body(JUDGING_ROW, "context")
        self.assertTrue(body["state"]["response"].endswith(NEUTRAL_SENTENCE))
        self.assertEqual(body["state"]["reference_answer"], "100")

    def test_wrong_reference_replaces_the_reference(self):
        body = JUDGE_NOULS_V1.body(JUDGING_ROW, "wrong_reference")
        self.assertEqual(body["state"]["reference_answer"], "90")
        with self.assertRaises(ValueError):
            JUDGE_NOULS_V1.body({**JUDGING_ROW, "wrong_reference": None}, "wrong_reference")

    def test_unsupported_variants_are_refused(self):
        self.assertFalse(FILTER.supports("rotated"))
        self.assertFalse(REWRITE.supports("context"))
        self.assertFalse(JUDGE_CHOICE.supports("flipped"))
        with self.assertRaises(ValueError):
            FILTER.body(FILTER_ROW, "context")

    def test_final_answer_input_for_d12(self):
        body = JUDGE_NOULS_V1_FINAL.body({**JUDGING_ROW, "final_answer": "100 °C"})
        self.assertEqual(body["state"], {"question": JUDGING_ROW["question"], "reference_answer": "100",
                                         "final_answer": "100 °C"})
        with self.assertRaises(ValueError):
            JUDGE_NOULS_V1_FINAL.body(JUDGING_ROW)

    def test_wrong_reference_is_a_stable_distractor(self):
        options = ["100 °C", "90 °C", "212 °C"]
        picks = {client.wrong_reference(JUDGING_ROW, options, "100 °C") for _ in range(5)}
        self.assertEqual(len(picks), 1)
        self.assertIn(picks.pop(), {"90 °C", "212 °C"})
        self.assertIsNone(client.wrong_reference(JUDGING_ROW, ["100 °C"], "100 °C"))


class RunWithVariantsTests(unittest.TestCase):
    def test_dry_run_with_all_variants_skips_what_an_instrument_lacks(self):
        with tempfile.TemporaryDirectory() as d:
            log = Path(d) / "run.jsonl"
            spec = "openrouter:typesafe/jev-1.13-20260917"
            counts = client.run(JUDGE_CHOICE_V1, [JUDGING_ROW], spec, log,
                                ["base", "flipped", "rotated", "context", "wrong_reference"], dry_run=True)
            self.assertEqual(counts["sent"], 4)  # no flipped variant for a Choice
            variants = [json.loads(line)["variant"] for line in log.read_text().splitlines()]
            self.assertEqual(variants, ["base", "rotated", "context", "wrong_reference"])


if __name__ == "__main__":
    unittest.main()
