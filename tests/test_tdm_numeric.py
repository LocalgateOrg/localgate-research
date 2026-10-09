"""Unit tests for the 1% numeric matcher. Run: uv run python -m unittest discover tests"""

import json
import unittest
from pathlib import Path

from research.tdm.numeric import match, parse_quantity

FIXTURE = Path(__file__).parent / "fixtures" / "numeric_calibration.json"


class ParseTests(unittest.TestCase):
    def test_plain_and_currency(self):
        self.assertEqual(parse_quantity("$60.00").value, 60.0)
        self.assertEqual(parse_quantity("1,234.5").value, 1234.5)

    def test_scientific(self):
        self.assertAlmostEqual(parse_quantity("1.5 × 10^3 J").value, 1500.0)
        self.assertAlmostEqual(parse_quantity("2.0e-3 m").value, 0.002)

    def test_fraction(self):
        self.assertAlmostEqual(parse_quantity("3/4").value, 0.75)

    def test_latex_as_written_in_the_released_references(self):
        q = parse_quantity("-3.5 $^{\\circ} \\mathrm{C}$")
        self.assertEqual((q.value, q.unit, q.dimension), (-3.5, "C", "temperature"))
        q = parse_quantity("0.46$\\mathrm{~J}$")
        self.assertEqual((q.value, q.dimension), (0.46, "energy"))
        self.assertAlmostEqual(parse_quantity("3.0 \\times 10^-19").value, 3.0e-19)
        self.assertIsNone(parse_quantity("-994.3 $\\mathrm{~kJ} \\mathrm{mol}^{-1}$"))  # compound unit: left to the TDM

    def test_unit_words(self):
        self.assertEqual(parse_quantity("3.03 × 10^-19 joule").dimension, "energy")

    def test_only_ascii_digits(self):
        # "How is the Arabic numeral for 2 written?" -> "٢" is a different glyph, not the number 2.
        self.assertIsNone(parse_quantity("٢"))
        self.assertIsNone(parse_quantity("２"))

    def test_rejects_multi_part_and_ranges(self):
        self.assertIsNone(parse_quantity("7 • 11"))
        self.assertIsNone(parse_quantity("5 to 7 kg"))
        self.assertIsNone(parse_quantity("photosynthesis"))


class MatchTests(unittest.TestCase):
    def cases(self, rows):
        for reference, answer, expected in rows:
            with self.subTest(reference=reference, answer=answer):
                self.assertEqual(match(reference, answer)[0], expected)

    def test_tolerance(self):
        self.cases([
            ("93.64 Btu/hr-ft", "93.6 Btu/hr-ft", "match"),
            ("100", "100.9", "match"),
            ("100", "101.5", "no_match"),
            ("0", "0", "match"),
        ])

    def test_unit_conversion(self):
        self.cases([
            ("146 g", "0.146 kg", "match"),
            ("2 km", "2000 m", "match"),
            ("2 km", "2 kg", "no_match"),
            ("1 atm", "101.3 kPa", "match"),
        ])

    def test_temperature(self):
        self.cases([("20.6°C", "20.6 °C", "match"), ("100°C", "212 °F", "match"), ("100°C", "90 °C", "no_match")])

    def test_percent(self):
        self.cases([("45%", "0.45", "match"), ("45%", "45 %", "match")])

    def test_range_never_matches(self):
        self.assertEqual(match("6 kg", "between 5 and 7 kg")[0], "no_match")

    def test_text_is_undecided(self):
        self.cases([("Postmodern ethics", "postmodern ethics", "undecided"), ("18", "eighteen", "undecided")])


class CalibrationTests(unittest.TestCase):
    """The 39 human-graded calibration responses with numeric references.

    Answers are the last stated answer extracted from free text, so most stay
    undecided; the point is that whatever the matcher does decide agrees with
    the human majority. Regenerate the fixture from `localgate-tdm items`.
    """

    def test_decisions_agree_with_humans(self):
        items = json.loads(FIXTURE.read_text(encoding="utf-8"))
        self.assertEqual(len(items), 39)
        decided = 0
        for item in items:
            verdict, why = match(item["reference"], item["answer"])
            if verdict == "undecided":
                continue
            decided += 1
            with self.subTest(key=item["key"], reference=item["reference"], answer=item["answer"]):
                self.assertEqual(verdict == "match", item["human_match"], why)
        self.assertGreaterEqual(decided, 12)  # regression floor: 12/39 decided on 9 Oct 2026

    def test_known_pairs(self):
        cases = [("$606", "605.92", "match"), ("924.0", "921", "match"), ("6", "4", "no_match"),
                 ("3.03 × 10^-19 joule", "3.0 \\times 10^-19 J", "match"), ("176°F", "80 °C", "match"),
                 ("2", "٢", "undecided")]
        for reference, answer, expected in cases:
            with self.subTest(reference=reference, answer=answer):
                self.assertEqual(match(reference, answer)[0], expected)


if __name__ == "__main__":
    unittest.main()
