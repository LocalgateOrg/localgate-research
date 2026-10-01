"""Reproduce the final oracle-cloud electricity analysis without model calls."""

import argparse
import csv
import importlib.metadata
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import NamedTuple

import numpy as np

from research.analysis import input_validation
from research.analysis.data_io import sha256, write_json

ECOLOGITS_VERSION = "0.11.1"
CATALOG_SHA256 = "d02e621cfc0006610a4b144aa8888f1de878c57b15b4c63d21d95a9879bbaf5d"
EXPECTED_GENERATION_SHA256 = {
    "gemma-4-E2B-it_openlabel/rep0/generations.jsonl": "298b6362f593796a453992a7a8ec7c8ff18360b6803f4d03bd3058e33fb78ac2",
    "gemma-4-E2B-it_openlabel/rep1/generations.jsonl": "f3d927e308b13a75821ea48aeb92f7c9418ec2a94bcac11780bd44a84cf1e210",
    "gemma-4-E2B-it_openlabel/rep2/generations.jsonl": "5b42cdf64c5290b7afac38d28915c0e2085a1cd23974bd0a6b6a7187bbe4f613",
    "gemma-4-E2B-it_openlabel/rep3/generations.jsonl": "f56e3bd1a668f770085804a317a58767e0d115cf5da28ac85b7a90943652eb84",
    "gemma-4-E2B-it_openlabel/rep4/generations.jsonl": "d133062202661e1fd0d051852fb4a26df294d336910dc7a94105b42ff30bde8b",
    "gemma-4-E2B-it_study_k30/rep0/generations.jsonl": "0e0fa9fe401f55e4fbc95a86865dd08ad54db333476d34796d43530ac985cb24",
    "gemma-4-E2B-it_study_k30/rep1/generations.jsonl": "913d0bad47ecc67168582739bffe512e1913b664f75d6fa815f8e2557817674e",
    "gemma-4-E2B-it_study_k30/rep2/generations.jsonl": "0c1463648b88f27bab5c75a5076e4da6236fedf22435b6783dca3dc0e1528b00",
    "gemma-4-E2B-it_study_k30/rep3/generations.jsonl": "6c51a5729d590268fdb28ed91d0e44204416d7dd34459a211738cecbb2faa41a",
    "gemma-4-E2B-it_study_k30/rep4/generations.jsonl": "d1e0cc2424e933db37ddb28f4db7dac87bd2f1e6af340fd952501b307b0abf3d",
    "gemma-4-E2B-it_study_k30/rep5/generations.jsonl": "90badfe83d0f30a76fd2545b43c3b5cd962a35bef08a90ef87090b6f89462660",
    "gemma-4-E2B-it_study_k30/rep6/generations.jsonl": "3d3920b3103c371022d6baa8a758c242415fe504b254f8772f763d36f37d75d6",
    "gemma-4-E2B-it_study_k30/rep7/generations.jsonl": "f900b048794a329991817e5495711675ca5654c7a5649aae6d5cd42bb5c87f04",
    "gemma-4-E2B-it_study_k30/rep8/generations.jsonl": "ed6d4ca23938af644ad72e2cf06d89bad96df7fe7af199e052280623ddc46429",
    "gemma-4-E2B-it_study_k30/rep9/generations.jsonl": "0fa21422454678f71d800e93f275a1ba77b5383e4e2abb06a4d1d679ce328205",
    "gemma-4-E2B-it_study_k30/rep10/generations.jsonl": "42c7f5448be7fc23c2ff91fe11b58c3142ed781c9dfb0463b1dc2d599b6f0630",
    "gemma-4-E2B-it_study_k30/rep11/generations.jsonl": "fbe62e9a9db4117f90ad24008f2074c6432968c4b12985557824638a4717c181",
    "gemma-4-E2B-it_study_k30/rep12/generations.jsonl": "5aea016b56fe9be3f2664dd7ee39fd311c82ce1ceb98c9c8b48dff3c8aa45de5",
    "gemma-4-E2B-it_study_k30/rep13/generations.jsonl": "9dcb9108ec0c653ba0a7c996e3b178d5142924842f2ad35a071bd856e0718dca",
    "gemma-4-E2B-it_study_k30/rep14/generations.jsonl": "70948c86dc3604e99b6dd0287866535ed0b4c6d9f43fe312810920b15feb99b6",
    "gemma-4-E2B-it_study_k30/rep15/generations.jsonl": "6845842d3fef9146d4578356bbeacd87849c7f4a9883eaae57ef20a3b641adf0",
    "gemma-4-E2B-it_study_k30/rep16/generations.jsonl": "3a91cc67bba381960cd4eb8814201a69baee1b9e09feacb461308041ab588c6a",
    "gemma-4-E2B-it_study_k30/rep17/generations.jsonl": "5179a6173e0d39dfb4933d8f0965e539511a66c45b3f4dd2638f57aebeb7c63c",
    "gemma-4-E2B-it_study_k30/rep18/generations.jsonl": "cd32670b71ad5c8c4d5afcacd408639603250f0f9de64eaebbe10dbe2b967b81",
    "gemma-4-E2B-it_study_k30/rep19/generations.jsonl": "8a9c2e8ca4ce2a55468c5927159e4aa4358daaa066de584f7a60df16de42a742",
    "gemma-4-E2B-it_study_k30/rep20/generations.jsonl": "689b67d6d7ea235465a71ffa7988ed0506a52b1027ca771d2c56b73586fca360",
    "gemma-4-E2B-it_study_k30/rep21/generations.jsonl": "d255598250b5ec256c94667568d1a57184e0bd133a21502af428b5eeeddc19e5",
    "gemma-4-E2B-it_study_k30/rep22/generations.jsonl": "e31aaf6ca8e8258fab83f5166d3a91e13928a765416efebf94de5b677e24e8a6",
    "gemma-4-E2B-it_study_k30/rep23/generations.jsonl": "253fe47218baacfc96306a2c5e9c67b7bd3cb66a88a783dbb8ac2c3bf56f9cfd",
    "gemma-4-E2B-it_study_k30/rep24/generations.jsonl": "bdc21711c07a5e68b9b402e7a854e81d563300ff414877fb0a38f77dae435b96",
    "gemma-4-E2B-it_study_k30/rep25/generations.jsonl": "4696e0f9a64a3630d4c323b55241dd8b3551b087fae8f50d17d6191ef12cf582",
    "gemma-4-E2B-it_study_k30/rep26/generations.jsonl": "06e5499ed7757dd665e093e11e514b4cd8828ea64e21cfadf6d35b96413f6b3c",
    "gemma-4-E2B-it_study_k30/rep27/generations.jsonl": "1a60f68801355f013f00654bad502be03c317af54ebeca8981e0006cb6afcac0",
    "gemma-4-E2B-it_study_k30/rep28/generations.jsonl": "27a54277de70e1fa487ec770b8f535a1f729ae0e2c8d94dab8bda00e453bcceb",
    "gemma-4-E2B-it_study_k30/rep29/generations.jsonl": "bacc0689c8ca770d48ea12fa2af7c54ca046ee07ddaba3182e7d6ef04946e18e",
}
LOCAL_MODEL = ("google_genai", "gemma-4-26b-a4b-it")
DEFAULT_CLOUD_MODEL = "gpt-5.5-pro-2026-04-23"
DEFAULT_CLASSIFIER_WH = 0.00022133699853226423
SEED = 20260929
DEFAULT_REPETITIONS = 10_000
MAX_ATTEMPTS = 6  # Initial attempt plus zero to five extra retries.
LENGTH_MULTIPLIERS = (0.5, 1, 2)
COHORTS = (
    ("heldout1375", "heldout1375_predictions.jsonl", "gemma-4-E2B-it_openlabel", 1375, 5, 5),
    ("study280", "study280_predictions.jsonl", "gemma-4-E2B-it_study_k30", 280, 30, 30),
)
MMLU_PRO_CATEGORIES = (
    "biology",
    "business",
    "chemistry",
    "computer science",
    "economics",
    "engineering",
    "health",
    "history",
    "law",
    "math",
    "other",
    "philosophy",
    "physics",
    "psychology",
)
EXPECTED_HUMAN_VALIDATION = {
    "attention_checks_passed": 30,
    "participants": 30,
    "prompts": 280,
    "ratings_per_prompt": 3,
    "substantive_ratings": 840,
}


class Item(NamedTuple):
    question_id: int
    probability: float
    correct_count: int
    k: int
    mean_output_tokens: float


def json_lines(path: Path):
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(value, dict):
                raise input_validation.ValidationError(f"{path}:{line_number}: expected object")
            yield value


def valid_id(value: object) -> bool:
    return type(value) is int and value >= 0


def load_predictions(path: Path, k: int, expected_count: int):
    predictions: dict[int, tuple[float, int]] = {}
    categories: dict[int, str] = {}
    for row in json_lines(path):
        question_id = row.get("question_id")
        probability = row.get("probability")
        correct_count = row.get("correct_count")
        category = row.get("category")
        if not valid_id(question_id):
            raise ValueError(f"{path}: invalid question_id {question_id!r}")
        if question_id in predictions:
            raise ValueError(f"{path}: duplicate question_id {question_id}")
        if (
            type(probability) not in (int, float)
            or not math.isfinite(probability)
            or not 0 <= probability <= 1
        ):
            raise ValueError(f"{path}: invalid probability for {question_id}")
        if (
            row.get("k") != k
            or type(correct_count) is not int
            or not 0 <= correct_count <= k
        ):
            raise ValueError(f"{path}: invalid k/count for {question_id}")
        if category not in MMLU_PRO_CATEGORIES:
            raise ValueError(f"{path}: invalid category for {question_id}: {category!r}")
        predictions[question_id] = (float(probability), correct_count)
        categories[question_id] = category
    if len(predictions) != expected_count:
        raise ValueError(
            f"{path}: expected {expected_count} unique predictions, got {len(predictions)}"
        )
    if set(categories.values()) != set(MMLU_PRO_CATEGORIES):
        raise ValueError(f"{path}: category names do not match the MMLU-Pro cohorts")
    return predictions, categories


def load_lengths(
    paths: list[Path], expected_ids: set[int], repetitions: int
) -> dict[int, float]:
    if len(paths) != repetitions:
        raise ValueError(f"expected {repetitions} generation files, got {len(paths)}")
    totals: defaultdict[int, int] = defaultdict(int)
    for path in paths:
        seen: set[int] = set()
        for row in json_lines(path):
            question_id = row.get("question_id")
            output_tokens = row.get("output_tokens")
            if not valid_id(question_id) or question_id in seen:
                raise ValueError(f"{path}: invalid or duplicate generation ID {question_id!r}")
            if type(output_tokens) is not int or output_tokens < 0:
                raise ValueError(f"{path}: invalid output_tokens for {question_id}")
            seen.add(question_id)
            if question_id in expected_ids:
                totals[question_id] += output_tokens
        missing = expected_ids - seen
        if missing:
            raise ValueError(f"{path}: generation/prediction ID join mismatch: missing={len(missing)}")
    return {question_id: totals[question_id] / repetitions for question_id in expected_ids}


def load_human_decisions(path: Path, items: list[Item]) -> dict[int, bool]:
    """Load validated human forecasts from the private analysis report."""
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("validation") != EXPECTED_HUMAN_VALIDATION:
        raise ValueError("unexpected human cohort counts")
    expected = {item.question_id: item for item in items}
    decisions: dict[int, bool] = {}
    per_prompt = report.get("per_prompt")
    if not isinstance(per_prompt, list):
        raise input_validation.ValidationError("human report has no per_prompt list")
    for row in per_prompt:
        try:
            question_id = int(row["prompt_id"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("invalid human-report prompt ID") from exc
        if question_id in decisions or question_id not in expected:
            raise ValueError("human/classifier ID join mismatch")
        item = expected[question_id]
        if (row.get("correct_count"), row.get("k")) != (item.correct_count, item.k):
            raise ValueError("human/classifier reference labels differ")
        ratings = row.get("ratings")
        if (
            not isinstance(ratings, list)
            or len(ratings) != 3
            or len({rating.get("participant_id") for rating in ratings}) != 3
        ):
            raise ValueError("expected three distinct human ratings")
        levels = [rating.get("rating_level") for rating in ratings]
        if any(type(level) is not int or not 0 <= level <= 5 for level in levels):
            raise ValueError("invalid human rating")
        route_local = sum(level >= 3 for level in levels) >= 2
        if row.get("human_majority_local") is not route_local:
            raise ValueError("human majority differs from individual ratings")
        decisions[question_id] = route_local
    if set(decisions) != set(expected):
        raise ValueError("human/classifier ID join mismatch")
    return decisions


def energy_wh_range(impacts) -> tuple[float, float]:
    if impacts.energy is None or impacts.errors:
        raise ValueError(f"EcoLogits produced no energy: {impacts.errors}")
    if impacts.energy.unit != "kWh":
        raise ValueError(f"unexpected EcoLogits energy unit {impacts.energy.unit}")
    value = impacts.energy.value
    low = value.min if hasattr(value, "min") else value
    high = value.max if hasattr(value, "max") else value
    if not all(math.isfinite(number) and number >= 0 for number in (low, high)) or low > high:
        raise ValueError(f"invalid EcoLogits energy range {value}")
    return low * 1000, high * 1000


def validate_cloud_model(catalogue: Path, model_name: str) -> tuple[str, str]:
    """Require an exact OpenAI model entry in the pinned EcoLogits catalogue."""
    data = json.loads(catalogue.read_text(encoding="utf-8"))
    supported = {
        (entry.get("provider"), entry.get("name"))
        for entry in data.get("models", [])
        if entry.get("type") == "model"
    }
    model = ("openai", model_name)
    if model not in supported:
        raise ValueError(f"unsupported EcoLogits OpenAI model: {model_name}")
    return model


def estimate_costs(items: list[Item], cloud_model: tuple[str, str]):
    from ecologits.tracers.utils import llm_impacts

    cache: dict[tuple[str, str, int], tuple[float, float]] = {}

    def costs(model: tuple[str, str], multiplier: float) -> np.ndarray:
        estimates = []
        for item in items:
            tokens = math.floor(item.mean_output_tokens * multiplier + 0.5)
            key = (*model, tokens)
            if key not in cache:
                impacts = llm_impacts(*model, tokens, math.inf, "WOR")
                cache[key] = energy_wh_range(impacts)
            estimates.append(cache[key])
        result = np.asarray(estimates)
        if result.shape != (len(items), 2) or not np.isfinite(result).all() or (result < 0).any():
            raise ValueError("EcoLogits returned invalid per-question energy estimates")
        return result

    local = costs(LOCAL_MODEL, 1)
    if not np.array_equal(local[:, 0], local[:, 1]):
        raise ValueError("the pinned local proxy unexpectedly returns a range")
    cloud = {multiplier: costs(cloud_model, multiplier) for multiplier in LENGTH_MULTIPLIERS}
    return local[:, 0], cloud


def expected_attempts(probabilities: np.ndarray, limit: int) -> np.ndarray:
    """A call is needed whenever all preceding attempts failed, including at p=0."""
    return sum((1 - probabilities) ** attempt for attempt in range(limit))


def first_success(
    rng: np.random.Generator, probabilities: np.ndarray, repetitions: int
) -> np.ndarray:
    """Simulate independent attempts; seven denotes failure through all six attempts."""
    successes = rng.random((repetitions, len(probabilities), MAX_ATTEMPTS)) < probabilities[
        None, :, None
    ]
    return np.where(successes.any(axis=2), successes.argmax(axis=2) + 1, MAX_ATTEMPTS + 1)


def summaries(
    exact: np.ndarray,
    simulated: np.ndarray,
    indices: np.ndarray,
    variance: np.ndarray,
) -> tuple[float, float, float]:
    samples = simulated[:, indices].mean(axis=1)
    expected = float(exact[indices].mean())
    mean = float(samples.mean())
    # Use model variance: rare failures can be absent from finite simulations,
    # so an empirical standard error of zero would give a misleading check.
    standard_error = (
        math.sqrt(float(np.maximum(variance[indices], 0).sum()) / simulated.shape[0])
        / len(indices)
    )
    # This checks simulation arithmetic, not uncertainty in the scientific inputs.
    if abs(mean - expected) > 8 * standard_error + 1e-10:
        raise ArithmeticError("Monte Carlo mean differs from its exact expectation")
    return expected, mean, standard_error


def oracle_rows(
    items: list[Item],
    labels: np.ndarray,
    policies: dict[str, np.ndarray],
    local_wh: np.ndarray,
    cloud_wh: dict[float, np.ndarray],
    classifier_wh: float,
    cloud_model: tuple[str, str],
    rng: np.random.Generator,
    repetitions: int,
):
    """Compare fixed routing policies with recognized-failure cloud fallback."""
    probabilities = np.asarray([item.correct_count / item.k for item in items])
    local_first = first_success(rng, probabilities, repetitions)
    groups = {"overall": np.arange(len(items))}
    groups.update(
        {category: np.flatnonzero(labels == category) for category in MMLU_PRO_CATEGORIES}
    )

    for multiplier, bounds in cloud_wh.items():
        for bound_index, bound_name in enumerate(("lower", "upper")):
            cloud = bounds[:, bound_index]
            for group, indices in groups.items():
                if group != "overall" and multiplier != 1:
                    continue
                baseline = float(cloud[indices].mean())
                yield {
                    "category": group,
                    "n": len(indices),
                    "policy": "all_cloud",
                    "scenario": "all_cloud",
                    "local_retries": None,
                    "cloud_provider": cloud_model[0],
                    "cloud_model": cloud_model[1],
                    "cloud_length_multiplier": multiplier,
                    "cloud_estimate": bound_name,
                    "classifier_wh_per_request": 0.0,
                    "local_wh_per_attempt": float(local_wh[indices].mean()),
                    "local_routing_fraction": 0.0,
                    "expected_wh_per_question": baseline,
                    "all_cloud_wh_per_question": baseline,
                    "wh_saved_per_question": 0.0,
                    "electricity_relative_to_cloud": 1.0,
                    "electricity_saved_fraction": 0.0,
                    "expected_final_success": 1.0,
                    "all_cloud_final_success": 1.0,
                    "mc_wh_per_question": baseline,
                    "mc_wh_standard_error": 0.0,
                    "mc_final_success": 1.0,
                    "mc_success_standard_error": 0.0,
                }

            for policy, decisions in policies.items():
                selected = np.asarray(decisions, dtype=bool)
                if selected.shape != probabilities.shape:
                    raise ValueError(f"{policy}: routing decision count differs from cohort")
                overheads = (classifier_wh, 0.0) if policy == "classifier" else (0.0,)
                overheads = tuple(dict.fromkeys(overheads))

                # Setting A commits to one local answer for selected questions.
                commit_quality = np.where(selected, probabilities, 1.0)
                commit_energy = np.where(selected, local_wh, cloud)
                mc_commit_quality = (~selected)[None, :] | (local_first == 1)
                mc_commit_energy = np.broadcast_to(commit_energy, mc_commit_quality.shape)
                for group, indices in groups.items():
                    if group != "overall" and multiplier != 1:
                        continue
                    success, mc_success, se_success = summaries(
                        commit_quality,
                        mc_commit_quality,
                        indices,
                        np.where(selected, probabilities * (1 - probabilities), 0.0),
                    )
                    energy, mc_energy, se_energy = summaries(
                        commit_energy, mc_commit_energy, indices, np.zeros(len(items))
                    )
                    baseline = float(cloud[indices].mean())
                    for overhead in overheads:
                        total = energy + overhead
                        yield {
                            "category": group,
                            "n": len(indices),
                            "policy": policy,
                            "scenario": "commit_local",
                            "local_retries": 0,
                            "cloud_provider": cloud_model[0],
                            "cloud_model": cloud_model[1],
                            "cloud_length_multiplier": multiplier,
                            "cloud_estimate": bound_name,
                            "classifier_wh_per_request": overhead,
                            "local_wh_per_attempt": float(local_wh[indices].mean()),
                            "local_routing_fraction": float(selected[indices].mean()),
                            "expected_wh_per_question": total,
                            "all_cloud_wh_per_question": baseline,
                            "wh_saved_per_question": baseline - total,
                            "electricity_relative_to_cloud": total / baseline,
                            "electricity_saved_fraction": 1 - total / baseline,
                            "expected_final_success": success,
                            "all_cloud_final_success": 1.0,
                            "mc_wh_per_question": mc_energy + overhead,
                            "mc_wh_standard_error": se_energy,
                            "mc_final_success": mc_success,
                            "mc_success_standard_error": se_success,
                        }

                # Setting B recognizes local failures and uses one perfect cloud fallback.
                for local_limit in range(1, MAX_ATTEMPTS + 1):
                    fail = np.where(selected, (1 - probabilities) ** local_limit, 1.0)
                    attempts = selected * expected_attempts(probabilities, local_limit)
                    attempts_squared = selected * sum(
                        (2 * attempt + 1) * (1 - probabilities) ** attempt
                        for attempt in range(local_limit)
                    )
                    exact_energy = attempts * local_wh + fail * cloud
                    exact_energy_squared = (
                        attempts_squared * local_wh**2
                        + fail * cloud**2
                        + 2 * selected * local_limit * local_wh * fail * cloud
                    )
                    mc_attempts = np.minimum(local_first, local_limit) * selected
                    mc_fallback = (~selected)[None, :] | (local_first > local_limit)
                    mc_energy_values = mc_attempts * local_wh + mc_fallback * cloud
                    for group, indices in groups.items():
                        if group != "overall" and multiplier != 1:
                            continue
                        energy, mc_energy, se_energy = summaries(
                            exact_energy,
                            mc_energy_values,
                            indices,
                            exact_energy_squared - exact_energy**2,
                        )
                        baseline = float(cloud[indices].mean())
                        for overhead in overheads:
                            total = energy + overhead
                            yield {
                                "category": group,
                                "n": len(indices),
                                "policy": policy,
                                "scenario": "fallback",
                                "local_retries": local_limit - 1,
                                "cloud_provider": cloud_model[0],
                                "cloud_model": cloud_model[1],
                                "cloud_length_multiplier": multiplier,
                                "cloud_estimate": bound_name,
                                "classifier_wh_per_request": overhead,
                                "local_wh_per_attempt": float(local_wh[indices].mean()),
                                "local_routing_fraction": float(selected[indices].mean()),
                                "expected_wh_per_question": total,
                                "all_cloud_wh_per_question": baseline,
                                "wh_saved_per_question": baseline - total,
                                "electricity_relative_to_cloud": total / baseline,
                                "electricity_saved_fraction": 1 - total / baseline,
                                "expected_final_success": 1.0,
                                "all_cloud_final_success": 1.0,
                                "mc_wh_per_question": mc_energy + overhead,
                                "mc_wh_standard_error": se_energy,
                                "mc_final_success": 1.0,
                                "mc_success_standard_error": 0.0,
                            }


def load_cohort(
    prediction_path: Path,
    generations: Path,
    cohort: tuple[str, str, str, int, int, int],
) -> tuple[list[Item], np.ndarray]:
    name, _filename, generation_directory, count, k, repeats = cohort
    predictions, categories = load_predictions(prediction_path, k, count)
    paths = [
        generations / generation_directory / f"rep{repeat}" / "generations.jsonl"
        for repeat in range(repeats)
    ]
    lengths = load_lengths(paths, set(predictions), repeats)
    items = [
        Item(question_id, *predictions[question_id], k, lengths[question_id])
        for question_id in sorted(predictions)
    ]
    if any(
        not math.isfinite(item.mean_output_tokens) or item.mean_output_tokens < 0
        for item in items
    ):
        raise ValueError(f"{name}: invalid mean output-token count")
    labels = np.asarray([categories[item.question_id] for item in items])
    return items, labels


def run(
    output: Path,
    study_predictions: Path,
    heldout_predictions: Path,
    data_dir: Path,
    generations: Path,
    human_report: Path,
    classifier_wh: float = DEFAULT_CLASSIFIER_WH,
    cloud_model: str = DEFAULT_CLOUD_MODEL,
    repetitions: int = DEFAULT_REPETITIONS,
) -> None:
    """Write aggregate final-oracle CSVs; private per-question inputs stay private."""
    from ecologits import __file__ as eco_init

    output = Path(output)
    prediction_paths = {"study280": Path(study_predictions), "heldout1375": Path(heldout_predictions)}
    data_dir = Path(data_dir)
    generations = Path(generations)
    human_report = Path(human_report)
    if output.exists():
        raise ValueError(f"output already exists: {output}")
    if not math.isfinite(classifier_wh) or classifier_wh < 0:
        raise ValueError("classifier energy must be finite and nonnegative")
    if type(repetitions) is not int or repetitions <= 0:
        raise ValueError("repetitions must be a positive integer")
    if importlib.metadata.version("ecologits") != ECOLOGITS_VERSION:
        raise ValueError(f"energy analysis requires ecologits=={ECOLOGITS_VERSION}")
    catalogue = Path(eco_init).parent / "data/models.json"
    if sha256(catalogue) != CATALOG_SHA256:
        raise ValueError("use the unmodified EcoLogits 0.11.1 release catalogue")
    model = validate_cloud_model(catalogue, cloud_model)

    expected_hashes = {
        catalogue: CATALOG_SHA256,
    }
    for _, _, generation_directory, _, _, repeats in COHORTS:
        for repeat in range(repeats):
            relative = f"{generation_directory}/rep{repeat}/generations.jsonl"
            expected_hashes[generations / relative] = EXPECTED_GENERATION_SHA256[relative]
    for path, expected_hash in expected_hashes.items():
        if sha256(path) != expected_hash:
            raise ValueError(f"input differs from the pinned study data: {path}")
    # The human command may add or reorder non-analytical metadata. Its cohort,
    # references, ratings and derived routes are validated by load_human_decisions.
    # Pin the supplied bytes only for the duration of this run to catch races.
    input_hashes = {**expected_hashes, human_report: sha256(human_report)}
    for path in prediction_paths.values():
        input_hashes[path] = sha256(path)
    for name in ("converted.jsonl", "labels_open.jsonl", "labels_study_k30.jsonl", "study_prompts_corpus.jsonl", "split.json"):
        path = data_dir / name
        input_hashes[path] = sha256(path)
    groups = input_validation.prepare_inputs(data_dir)
    input_validation.load_predictions(prediction_paths["study280"], groups.study)
    input_validation.load_predictions(prediction_paths["heldout1375"], groups.remainder)

    prepared = []
    seen_ids: set[int] = set()
    for cohort in COHORTS:
        items, labels = load_cohort(prediction_paths[cohort[0]], generations, cohort)
        ids = {item.question_id for item in items}
        overlap = seen_ids & ids
        if overlap:
            raise ValueError(f"the two cohorts overlap on {len(overlap)} question IDs")
        seen_ids.update(ids)
        policies = {
            "classifier": np.asarray([item.probability >= 0.5 for item in items]),
        }
        if cohort[0] == "study280":
            human = load_human_decisions(human_report, items)
            policies["human_aggregate"] = np.asarray(
                [human[item.question_id] for item in items]
            )
        local, cloud = estimate_costs(items, model)
        prepared.append((cohort[0], items, labels, policies, local, cloud))

    results = []
    for cohort_index, (name, items, labels, policies, local, cloud) in enumerate(
        prepared
    ):
        rng = np.random.default_rng(np.random.SeedSequence([SEED, cohort_index]))
        rows = list(
            oracle_rows(
                items,
                labels,
                policies,
                local,
                cloud,
                classifier_wh,
                model,
                rng,
                repetitions,
            )
        )
        results.append((name, len(items), rows))
    for path, expected_hash in input_hashes.items():
        if sha256(path) != expected_hash:
            raise ValueError(f"input changed during calculation: {path}")

    output.mkdir(mode=0o700, parents=True)
    if output.stat().st_mode & 0o077:
        raise ValueError("output directory must be private")
    for name, question_count, rows in results:
        with (output / f"{name}_oracle.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(
            f"{name}: {question_count} questions, {len(rows)} oracle rows; "
            "exact/Monte Carlo checks passed"
        )
    input_names = {
        catalogue: "ecologits/models.json",
        human_report: "human_report",
        prediction_paths["study280"]: "study_predictions",
        prediction_paths["heldout1375"]: "heldout_predictions",
        **{data_dir / name: f"data/{name}" for name in (
            "converted.jsonl", "labels_open.jsonl", "labels_study_k30.jsonl",
            "study_prompts_corpus.jsonl", "split.json",
        )},
        **{
            generations / relative: f"generations/{relative}"
            for relative in EXPECTED_GENERATION_SHA256
        },
    }
    write_json(output / "inputs.json", {
        "input_files_sha256": {
            input_names[path]: digest for path, digest in input_hashes.items()
        },
        "cloud_model": cloud_model,
        "classifier_wh": classifier_wh,
        "repetitions": repetitions,
        "seed": SEED,
        "ecologits_version": ECOLOGITS_VERSION,
    })
    print(output)


def positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def finite_nonnegative(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a finite, nonnegative number") from exc
    if not math.isfinite(parsed) or parsed < 0:
        raise argparse.ArgumentTypeError("must be a finite, nonnegative number")
    return parsed


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="new private output directory")
    parser.add_argument("--study-predictions", type=Path, required=True)
    parser.add_argument("--heldout-predictions", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--generation-results-dir", type=Path, required=True)
    parser.add_argument("--human-report", type=Path, required=True)
    parser.add_argument(
        "--classifier-wh",
        type=finite_nonnegative,
        default=DEFAULT_CLASSIFIER_WH,
        help=f"classifier electricity in Wh/request (default: {DEFAULT_CLASSIFIER_WH})",
    )
    parser.add_argument(
        "--cloud-model",
        default=DEFAULT_CLOUD_MODEL,
        help=f"EcoLogits OpenAI model (default: {DEFAULT_CLOUD_MODEL})",
    )
    parser.add_argument(
        "--repetitions",
        type=positive_int,
        default=DEFAULT_REPETITIONS,
        help=f"Monte Carlo repetitions (default: {DEFAULT_REPETITIONS})",
    )
    args = parser.parse_args(argv)
    run(
        args.output,
        args.study_predictions,
        args.heldout_predictions,
        args.data_dir,
        args.generation_results_dir,
        args.human_report,
        args.classifier_wh,
        args.cloud_model,
        args.repetitions,
    )


if __name__ == "__main__":
    main()
