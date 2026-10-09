"""Deterministic numeric matching under the grading rubric's 1% rule (task I2).

The rubric: a numeric final answer matches when it agrees with the reference to
within 1% relative error, in any unit convertible from the reference's; a range
never matches a point value. This module applies that rule in code when both
sides parse cleanly, and returns ``undecided`` otherwise so the decision falls
back to the typed decision model. It is an extra arm of the study (decision
D11), not an assumed part of the pipeline.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

TOLERANCE = 0.01

# Base units with their dimension and factor to the SI base. Prefixed forms are
# generated below; symbols are matched case-sensitively except where noted.
BASE_UNITS: dict[str, tuple[str, float]] = {
    "m": ("length", 1.0), "g": ("mass", 1e-3), "s": ("time", 1.0), "J": ("energy", 1.0),
    "W": ("power", 1.0), "N": ("force", 1.0), "Pa": ("pressure", 1.0), "V": ("voltage", 1.0),
    "A": ("current", 1.0), "Hz": ("frequency", 1.0), "mol": ("amount", 1.0), "L": ("volume", 1e-3),
    "eV": ("energy", 1.602176634e-19), "cal": ("energy", 4.184), "Ω": ("resistance", 1.0),
    "ohm": ("resistance", 1.0), "ohms": ("resistance", 1.0), "C": ("charge", 1.0),
    "F": ("capacitance", 1.0), "H": ("inductance", 1.0), "T": ("magnetic_field", 1.0),
    "M": ("concentration", 1.0), "atm": ("pressure", 101_325.0), "bar": ("pressure", 1e5),
    "min": ("time", 60.0), "h": ("time", 3600.0), "hr": ("time", 3600.0), "hours": ("time", 3600.0),
    "in": ("length", 0.0254), "ft": ("length", 0.3048), "lb": ("mass", 0.45359237),
    "lbm": ("mass", 0.45359237), "psi": ("pressure", 6894.757), "Btu": ("energy", 1055.06),
    "kWh": ("energy", 3.6e6), "hp": ("power", 745.7),
    "joule": ("energy", 1.0), "joules": ("energy", 1.0), "meter": ("length", 1.0), "meters": ("length", 1.0),
    "second": ("time", 1.0), "seconds": ("time", 1.0), "gram": ("mass", 1e-3), "grams": ("mass", 1e-3),
}
PREFIXES = {"G": 1e9, "M": 1e6, "k": 1e3, "c": 1e-2, "m": 1e-3, "μ": 1e-6, "µ": 1e-6, "u": 1e-6, "n": 1e-9, "p": 1e-12}
PREFIXABLE = {"m", "g", "s", "J", "W", "N", "Pa", "V", "A", "Hz", "mol", "L", "eV", "cal", "Ω", "C", "F", "H", "T", "M"}
TEMPERATURES = {"°C": "C", "°F": "F", "K": "K", "C°": "C", "degrees Celsius": "C", "degrees Fahrenheit": "F", "kelvin": "K"}
CURRENCY = re.compile(r"^[$€£]")

# ASCII digits only: Python's \d also matches ٢ or ２, and a question about how a
# numeral is written must not be decided by converting the glyph.
NUMBER = r"[-+−]?(?:[0-9]{1,3}(?:,[0-9]{3})+|[0-9]+)(?:\.[0-9]+)?|[-+−]?\.[0-9]+"
SCIENTIFIC = rf"(?P<mant>{NUMBER})\s*(?:[eE](?P<exp1>[-+−]?[0-9]+)|[x×*·]\s*10\s*\^?\s*\{{?(?P<exp2>[-+−]?[0-9]+)\}}?)"
FRACTION = r"(?P<num>[-+−]?[0-9]+)\s*/\s*(?P<den>[0-9]+)"
RANGE = re.compile(rf"(?:{NUMBER})\s*(?:-|–|—|to|and)\s*(?:{NUMBER})")


@dataclass(frozen=True)
class Quantity:
    value: float
    unit: str | None  # normalised unit string as written, or None
    dimension: str | None
    si_value: float | None  # value in SI base units when the unit is known
    percent: bool = False


def _units() -> dict[str, tuple[str, float]]:
    table = dict(BASE_UNITS)
    for base in PREFIXABLE:
        dimension, factor = BASE_UNITS[base]
        for prefix, scale in PREFIXES.items():
            table.setdefault(prefix + base, (dimension, factor * scale))
    return table


UNITS = _units()


def _clean(text: str) -> str:
    """Undo the LaTeX the released references and responses are written in.

    ``-3.5 $^{\\circ} \\mathrm{C}$`` becomes ``-3.5 °C`` and ``3.0 \\times 10^-19`` becomes
    ``3.0 × 10^-19``; anything that is not markup is left alone.
    """
    text = text.replace("$", "")
    text = re.sub(r"\\(?:mathrm|text|mathbf|operatorname|rm)\s*\{([^{}]*)\}", r"\1", text)
    text = re.sub(r"\^\s*\{?\s*\\circ\s*\}?|\\circ|\\degree", "°", text)
    text = re.sub(r"\\(?:times|cdot)", "×", text)
    text = re.sub(r"\\[,;!: ]|~", " ", text)
    text = re.sub(r"°\s+([CF])\b", r"°\1", text)
    return re.sub(r"\s+", " ", text).strip()


def _number(text: str) -> float:
    return float(text.replace(",", "").replace("−", "-"))


def parse_quantity(text: str) -> Quantity | None:
    """Parse one number with an optional unit. Returns None for anything else.

    None covers text with no number, with several numbers (multi-part answers),
    and ranges, which the rubric never accepts as a point value.
    """
    text = _clean(text or "").rstrip(".").strip()
    text = CURRENCY.sub("", text).strip()
    if not text or RANGE.search(text):
        return None
    match = re.match(rf"^(?:{SCIENTIFIC}|{FRACTION}|(?P<plain>{NUMBER}))\s*(?P<rest>.*)$", text)
    if not match:
        return None
    if match.group("mant") is not None:
        exponent = match.group("exp1") or match.group("exp2")
        value = _number(match.group("mant")) * 10 ** int(exponent.replace("−", "-"))
    elif match.group("num") is not None:
        value = float(Fraction(int(match.group("num").replace("−", "-")), int(match.group("den"))))
    else:
        value = _number(match.group("plain"))
    rest = match.group("rest").strip()
    if re.search(r"\d", rest):
        return None  # a second number: multi-part or compound unit with exponents
    if rest == "%":
        return Quantity(value, "%", "ratio", value / 100, percent=True)
    if not rest:
        return Quantity(value, None, None, None)
    for written, scale in TEMPERATURES.items():
        if rest == written:
            return Quantity(value, scale, "temperature", _kelvin(value, scale))
    unit = rest.rstrip(".")
    if unit in UNITS:
        dimension, factor = UNITS[unit]
        return Quantity(value, unit, dimension, value * factor)
    return Quantity(value, unit.lower(), None, None)


def _kelvin(value: float, scale: str) -> float:
    if scale == "C":
        return value + 273.15
    if scale == "F":
        return (value - 32) * 5 / 9 + 273.15
    return value


def _close(a: float, b: float) -> bool:
    if b == 0:
        return abs(a) < 1e-12
    return abs(a - b) / abs(b) <= TOLERANCE


def match(reference: str, answer: str) -> tuple[str, str]:
    """Return (verdict, reason) with verdict in match / no_match / undecided."""
    ref = parse_quantity(reference)
    if ref is None:
        return "undecided", "reference is not a single numeric quantity"
    if answer and RANGE.search(answer) and parse_quantity(answer) is None:
        return "no_match", "answer is a range; a range never matches a point value"
    ans = parse_quantity(answer)
    if ans is None:
        return "undecided", "answer is not a single numeric quantity"
    if ref.dimension == "temperature" or ans.dimension == "temperature":
        if ref.dimension != ans.dimension:
            return "undecided", "temperature against a non-temperature unit"
        # Relative error on an interval scale is defined on the value as written.
        if ref.unit == ans.unit:
            return ("match" if _close(ans.value, ref.value) else "no_match"), "same temperature scale"
        return ("match" if abs(ans.si_value - ref.si_value) <= TOLERANCE * abs(ref.value) else "no_match"), "converted temperature"
    if ref.dimension and ans.dimension:
        if ref.dimension != ans.dimension:
            return "no_match", f"dimension {ans.dimension} differs from {ref.dimension}"
        return ("match" if _close(ans.si_value, ref.si_value) else "no_match"), "converted units"
    if ref.percent != ans.percent and (ref.unit or ans.unit):
        # 45% against 0.45: compare as ratios
        r = ref.si_value if ref.percent else ref.value
        a = ans.si_value if ans.percent else ans.value
        return ("match" if _close(a, r) else "no_match"), "percent compared as ratio"
    if ref.unit and ans.unit and ref.unit != ans.unit:
        return "undecided", f"unknown units {ans.unit!r} vs {ref.unit!r}"
    return ("match" if _close(ans.value, ref.value) else "no_match"), "same or missing unit"


# ── Check against the human-graded calibration sample ────────────────────────

FINAL_LINE = re.compile(r"(?:final answer|answer is|≈|=)\s*[:\s]*\**\s*(?P<ans>[^\n*]+)", re.IGNORECASE)


def last_stated_answer(response: str) -> str:
    """Rough proxy for a free-text response's final answer: the last 'answer is / =' clause.

    Only used to sanity-check the matcher on the released free-text responses;
    new generations carry final_answer as a structured field.
    """
    bold = re.findall(r"\*\*([^*]+)\*\*", response or "")
    found = FINAL_LINE.findall(response or "")
    candidate = bold[-1] if bold else (found[-1] if found else "")
    candidate = re.sub(r"\\\$|\$|\\boxed\{|\\mathbf\{|\\text\{[^}]*\}|[{}]", "", candidate).strip()
    return candidate


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--items", type=Path, required=True, help="output directory of `localgate-tdm items`")
    args = parser.parse_args(argv)
    rows = [json.loads(line) for line in (args.items / "judging.jsonl").open() if line.strip()]
    references = {
        json.loads(line)["question_id"]: json.loads(line)["reference_answer"]
        for line in (args.items / "filter.jsonl").open()
    }
    graded = [r for r in rows if r["human_match"] is not None and r["reference_type"] == "numeric"]
    outcome = {"match": [0, 0], "no_match": [0, 0], "undecided": [0, 0]}
    for r in graded:
        verdict, _ = match(references[r["question_id"]], last_stated_answer(r["response"]))
        outcome[verdict][0] += 1
        outcome[verdict][1] += int((verdict == "match") == r["human_match"]) if verdict != "undecided" else 0
    decided = outcome["match"][0] + outcome["no_match"][0]
    agree = outcome["match"][1] + outcome["no_match"][1]
    print(f"numeric-reference calibration responses: {len(graded)}")
    for verdict, (n, ok) in outcome.items():
        print(f"  {verdict:9s} {n:3d}" + (f"  agrees with humans on {ok}" if verdict != "undecided" else ""))
    if decided:
        print(f"decided {decided}/{len(graded)}, agreement with human majority {agree}/{decided}")
    print("Final answers here are extracted heuristically from free text; treat as a smoke test.")
    return 0
