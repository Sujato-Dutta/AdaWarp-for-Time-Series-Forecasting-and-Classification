"""Shared, dependency-light utilities for MVPF paper analyses."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import random
import statistics
from typing import Iterable, Sequence


ROOT = Path(__file__).resolve().parent.parent
HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"

DATASETS = ("ETTh1", "ETTh2", "ETTm2", "Weather", "Electricity", "Traffic")
HORIZONS = (96, 192, 336, 720)
METHODS = (
    "MVPF",
    "DLinear",
    "PatchTST",
    "TimesNet",
    "iTransformer",
    "TimeMixer",
    "FEDformer",
    "VPNet",
)
METRICS = ("mse", "mae")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[dict[str, object]], fields: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        ordered: list[str] = []
        for row in rows:
            for key in row:
                if key not in ordered:
                    ordered.append(key)
        fields = ordered
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def rank_values(values: Sequence[float]) -> list[float]:
    """Average ranks for ascending values, including exact ties."""

    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        average_rank = 0.5 * ((start + 1) + end)
        for position in range(start, end):
            ranks[order[position]] = average_rank
        start = end
    return ranks


def _regularized_gamma_q(shape: float, value: float) -> float:
    """Upper regularized incomplete gamma Q(shape, value)."""

    if value < 0.0 or shape <= 0.0:
        raise ValueError("Gamma arguments must be positive.")
    if value == 0.0:
        return 1.0
    epsilon = 3e-14
    tiny = 1e-300
    log_term = -value + shape * math.log(value) - math.lgamma(shape)
    if value < shape + 1.0:
        term = 1.0 / shape
        total = term
        ap = shape
        for _ in range(10000):
            ap += 1.0
            term *= value / ap
            total += term
            if abs(term) <= abs(total) * epsilon:
                break
        p_value = total * math.exp(log_term)
        return max(0.0, min(1.0, 1.0 - p_value))

    b = value + 1.0 - shape
    c = 1.0 / tiny
    d = 1.0 / max(abs(b), tiny)
    if b < 0:
        d = -d
    h = d
    for iteration in range(1, 10000):
        an = -iteration * (iteration - shape)
        b += 2.0
        d = an * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + an / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) <= epsilon:
            break
    return max(0.0, min(1.0, math.exp(log_term) * h))


def chi_square_survival(statistic: float, degrees_of_freedom: int) -> float:
    return _regularized_gamma_q(0.5 * degrees_of_freedom, 0.5 * statistic)


def friedman_test(blocks: Sequence[Sequence[float]]) -> dict[str, float]:
    if not blocks:
        raise ValueError("Friedman test requires at least one block.")
    treatment_count = len(blocks[0])
    if treatment_count < 2 or any(len(block) != treatment_count for block in blocks):
        raise ValueError("Every Friedman block must contain the same treatments.")
    block_count = len(blocks)
    rank_matrix = [rank_values(block) for block in blocks]
    average_ranks = [
        statistics.mean(row[index] for row in rank_matrix)
        for index in range(treatment_count)
    ]
    statistic = (
        12.0 * block_count / (treatment_count * (treatment_count + 1.0))
        * sum(rank * rank for rank in average_ranks)
        - 3.0 * block_count * (treatment_count + 1.0)
    )
    tie_sum = 0.0
    for block in blocks:
        counts: dict[float, int] = {}
        for value in block:
            counts[value] = counts.get(value, 0) + 1
        tie_sum += sum(count**3 - count for count in counts.values() if count > 1)
    correction = 1.0 - tie_sum / (
        block_count * treatment_count * (treatment_count**2 - 1.0)
    )
    if correction > 0:
        statistic /= correction
    degrees = treatment_count - 1
    return {
        "statistic": statistic,
        "degrees_of_freedom": float(degrees),
        "p_value": chi_square_survival(statistic, degrees),
        "tie_correction": correction,
    }


def wilcoxon_signed_rank(differences: Iterable[float], tolerance: float = 1e-12) -> dict[str, float]:
    all_values = [float(value) for value in differences]
    values = [value for value in all_values if abs(value) > tolerance]
    zeros = sum(1 for value in all_values if abs(value) <= tolerance)
    if not values:
        return {
            "n": 0.0,
            "zeros": float(zeros),
            "w_plus": 0.0,
            "w_minus": 0.0,
            "statistic": 0.0,
            "p_value": 1.0,
            "rank_biserial": 0.0,
        }
    absolute = [abs(value) for value in values]
    ranks = rank_values(absolute)
    w_plus = sum(rank for rank, value in zip(ranks, values) if value > 0)
    w_minus = sum(rank for rank, value in zip(ranks, values) if value < 0)
    scaled_ranks = [int(round(2.0 * rank)) for rank in ranks]
    total = sum(scaled_ranks)
    observed = int(round(2.0 * w_plus))
    distribution = {0: 1}
    for rank in scaled_ranks:
        updated = dict(distribution)
        for subtotal, count in distribution.items():
            updated[subtotal + rank] = updated.get(subtotal + rank, 0) + count
        distribution = updated
    observed_distance = abs(observed - 0.5 * total)
    extreme = sum(
        count
        for subtotal, count in distribution.items()
        if abs(subtotal - 0.5 * total) >= observed_distance - 1e-12
    )
    p_value = min(1.0, extreme / (2 ** len(scaled_ranks)))
    rank_total = w_plus + w_minus
    effect = (w_plus - w_minus) / rank_total if rank_total else 0.0
    return {
        "n": float(len(values)),
        "zeros": float(zeros),
        "w_plus": w_plus,
        "w_minus": w_minus,
        "statistic": min(w_plus, w_minus),
        "p_value": p_value,
        "rank_biserial": effect,
    }


def holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(p_values, key=p_values.get)
    count = len(ordered)
    adjusted: dict[str, float] = {}
    running = 0.0
    for position, name in enumerate(ordered):
        candidate = min(1.0, (count - position) * p_values[name])
        running = max(running, candidate)
        adjusted[name] = running
    return adjusted


def percentile(values: Sequence[float], probability: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    location = (len(ordered) - 1) * probability
    lower = math.floor(location)
    upper = math.ceil(location)
    if lower == upper:
        return ordered[lower]
    weight = location - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def paired_bootstrap_ci(
    values: Sequence[float],
    *,
    statistic: str = "median",
    samples: int = 10000,
    seed: int = 20260710,
) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    rng = random.Random(seed)
    function = statistics.median if statistic == "median" else statistics.mean
    estimates = []
    for _ in range(samples):
        resample = [values[rng.randrange(len(values))] for _ in values]
        estimates.append(float(function(resample)))
    return percentile(estimates, 0.025), percentile(estimates, 0.975)


def wins_ties_losses(candidate: Sequence[float], reference: Sequence[float], tolerance: float = 1e-12) -> tuple[int, int, int]:
    wins = ties = losses = 0
    for candidate_value, reference_value in zip(candidate, reference):
        difference = candidate_value - reference_value
        if abs(difference) <= tolerance:
            ties += 1
        elif difference < 0:
            wins += 1
        else:
            losses += 1
    return wins, ties, losses


