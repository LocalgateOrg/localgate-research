"""Unit tests for the 1% numeric matcher. Run: uv run python -m unittest discover tests"""

import unittest

from research.tdm.numeric import match, parse_quantity


class ParseTests(unittest.TestCase):
    def test_plain_and_currency(self):
        self.assertEqual(parse_quantity("$60.00").value, 60.0)
        self.assertEqual(parse_quantity("1,234.5").value, 1234.5)

    def test_scientific(self):
        self.assertAlmostEqual(parse_quantity("1.5 × 10^3 J").value, 1500.0)
        self.assertAlmostEqual(parse_quantity("2.0e-3 m").value, 0.002)

    def test_fraction(self):
        self.assertAlmostEqual(parse_quantity("3/4").value, 0.75)

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


if __name__ == "__main__":
    unittest.main()
