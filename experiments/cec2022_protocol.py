"""CEC2022 competition run protocol and result-file writer.

Pass an optimizer callable with signature
``optimizer(objective, lower_bounds, upper_bounds, max_fes, seed)``. The
objective wrapper counts evaluations, records checkpoints, and stops at the
specified target or function-evaluation budget.
"""

import csv
import json
import math
import re
from functools import lru_cache
from pathlib import Path
from platform import platform, processor, python_version
from time import perf_counter
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np

from benchmarks.cec2022 import (
    LOWER_BOUND,
    OPTIMAL_VALUES,
    SUPPORTED_DIMENSIONS,
    cec2022,
)


RUNS_PER_PROBLEM = 30
TARGET_ERROR = 1.0e-8
MAX_FES_BY_DIMENSION = {10: 200_000, 20: 1_000_000}
_SEED_FILE = Path(__file__).parents[1] / "benchmarks" / "cec2022_data" / "Rand_Seeds.txt"


def evaluation_checkpoints(dimension, max_fes=None):
    """Return the 16 report checkpoints, using k=0,...,15."""
    if dimension not in SUPPORTED_DIMENSIONS:
        raise ValueError(f"dimension must be one of {SUPPORTED_DIMENSIONS}")
    if max_fes is None:
        max_fes = MAX_FES_BY_DIMENSION[dimension]
    if max_fes <= 0:
        raise ValueError("max_fes must be positive")
    return tuple(
        math.floor(dimension ** (k / 5.0 - 3.0) * max_fes)
        for k in range(16)
    )


@lru_cache(maxsize=1)
def _official_seeds():
    seeds = np.asarray(np.loadtxt(_SEED_FILE), dtype=np.uint32).reshape(-1)
    if seeds.size != 1000:
        raise ValueError("The official Rand_Seeds.txt must contain 1,000 seeds")
    return seeds


def competition_seed(function_number, dimension, run_id):
    """Return the official numeric seed for a zero-based run id (0 to 29)."""
    if function_number not in OPTIMAL_VALUES:
        raise ValueError("function_number must be between 1 and 12")
    if dimension not in SUPPORTED_DIMENSIONS:
        raise ValueError(f"dimension must be one of {SUPPORTED_DIMENSIONS}")
    if not 0 <= run_id < RUNS_PER_PROBLEM:
        raise ValueError("run_id must be in [0, 29]")
    zero_based_index = ((dimension // 10) * function_number * RUNS_PER_PROBLEM + run_id + 1 - RUNS_PER_PROBLEM) % 1000
    return int(_official_seeds()[zero_based_index])


class _StopTrial(Exception):
    """Internal control flow for the target and FE-budget limits."""


def run_trial(optimizer, function_number, dimension, run_id):
    """Run one seeded trial and return 16 errors, FEterm, and the seed.

    ``optimizer`` receives a scalar objective callable, lower/upper bound
    vectors, the CEC2022 MaxFES, and the numeric seed. It must call the
    objective for each candidate evaluation and should use the supplied seed
    to initialize its random number generator.
    """
    if dimension not in MAX_FES_BY_DIMENSION:
        raise ValueError(f"dimension must be one of {SUPPORTED_DIMENSIONS}")
    if function_number not in OPTIMAL_VALUES:
        raise ValueError("function_number must be between 1 and 12")

    max_fes = MAX_FES_BY_DIMENSION[dimension]
    checkpoints = evaluation_checkpoints(dimension, max_fes)
    seed = competition_seed(function_number, dimension, run_id)
    best_error = math.inf
    error_history = [None] * len(checkpoints)
    evaluations = 0
    next_checkpoint = 0

    def objective(candidate):
        nonlocal best_error, evaluations, next_checkpoint
        if evaluations >= max_fes:
            raise _StopTrial
        value = cec2022(candidate, function_number)
        evaluations += 1
        best_error = min(best_error, max(0.0, value - OPTIMAL_VALUES[function_number]))

        while next_checkpoint < len(checkpoints) and evaluations >= checkpoints[next_checkpoint]:
            error_history[next_checkpoint] = best_error
            next_checkpoint += 1

        if best_error < TARGET_ERROR:
            while next_checkpoint < len(error_history):
                error_history[next_checkpoint] = TARGET_ERROR
                next_checkpoint += 1
            raise _StopTrial
        return value

    lower_bounds = np.full(dimension, LOWER_BOUND, dtype=float)
    upper_bounds = np.full(dimension, -LOWER_BOUND, dtype=float)
    try:
        optimizer(objective, lower_bounds, upper_bounds, max_fes, seed)
    except _StopTrial:
        pass

    if evaluations < max_fes and best_error >= TARGET_ERROR:
        raise RuntimeError(
            f"Optimizer stopped after {evaluations} evaluations without reaching "
            f"the target or using MaxFES={max_fes}"
        )
    if any(value is None for value in error_history):
        raise RuntimeError("A trial ended before all required checkpoints were recorded")
    return {
        "errors": tuple(float(value) for value in error_history),
        "FEterm": evaluations,
        "seed": seed,
        "final_error": float(best_error),
    }


def write_result_file(path, function_number, dimension, algorithm_name, runs):
    """Write the required 17-by-30 matrix as a tab-separated text file."""
    if len(runs) != RUNS_PER_PROBLEM:
        raise ValueError(f"exactly {RUNS_PER_PROBLEM} runs are required for the CEC2022 file format")
    checkpoints = evaluation_checkpoints(dimension)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
        writer.writerow(["Measure", *(f"Run{index + 1}" for index in range(RUNS_PER_PROBLEM))])
        for checkpoint_index, fes in enumerate(checkpoints):
            writer.writerow([
                f"Error@FES={fes}",
                *(f"{run['errors'][checkpoint_index]:.16e}" for run in runs),
            ])
        writer.writerow(["FEterm", *(str(run["FEterm"]) for run in runs)])


def run_cec2022_suite(optimizer, algorithm_name, output_dir="results/cec2022", progress_callback=None, resume=False):
    """Run all 24 cases, optionally reusing valid result matrices from disk."""
    if not algorithm_name or not re.fullmatch(r"[A-Za-z0-9_-]+", algorithm_name):
        raise ValueError("algorithm_name may contain only letters, digits, hyphens, and underscores")
    output_dir = Path(output_dir)
    summary_rows = []

    for dimension in SUPPORTED_DIMENSIONS:
        for function_number in range(1, 13):
            result_path = output_dir / f"{algorithm_name}_{function_number}_{dimension}.txt"
            if resume and result_path.is_file():
                with result_path.open("r", newline="", encoding="utf-8") as stream:
                    rows = list(csv.reader(stream, delimiter="\t"))
                if (len(rows) == 18 and all(len(row) == RUNS_PER_PROBLEM + 1 for row in rows)
                        and rows[0][1:] == [f"Run{i}" for i in range(1, RUNS_PER_PROBLEM + 1)]
                        and rows[-1][0] == "FEterm"):
                    final_errors = np.asarray([float(value) for value in rows[-2][1:]], dtype=float)
                    summary_rows.append((
                        function_number, dimension,
                        float(np.min(final_errors)), float(np.max(final_errors)),
                        float(np.median(final_errors)), float(np.mean(final_errors)),
                        float(np.std(final_errors, ddof=1)),
                    ))
                    print(f"Reusing {result_path}", flush=True)
                    continue
            checkpoint_path = output_dir / f".{algorithm_name}_{function_number}_{dimension}.partial.json"
            runs = []
            if resume and checkpoint_path.is_file():
                try:
                    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
                    if (checkpoint.get("function") == function_number
                            and checkpoint.get("dimension") == dimension
                            and len(checkpoint.get("runs", [])) < RUNS_PER_PROBLEM):
                        runs = checkpoint["runs"]
                        print(f"Resuming {result_path} at run {len(runs) + 1}", flush=True)
                except (OSError, ValueError, TypeError):
                    runs = []
            for run_id in range(len(runs), RUNS_PER_PROBLEM):
                result = run_trial(optimizer, function_number, dimension, run_id)
                runs.append({"errors": list(result["errors"]), "FEterm": result["FEterm"],
                             "seed": result["seed"], "final_error": result["final_error"]})
                temp_checkpoint = checkpoint_path.with_suffix(".tmp")
                temp_checkpoint.write_text(json.dumps({
                    "function": function_number, "dimension": dimension, "runs": runs,
                }), encoding="utf-8")
                temp_checkpoint.replace(checkpoint_path)
                if progress_callback is not None:
                    progress_callback(function_number, dimension, run_id + 1, result)
            write_result_file(result_path, function_number, dimension, algorithm_name, runs)
            checkpoint_path.unlink(missing_ok=True)
            # k=15 is the MaxFES checkpoint. Early target hits are represented
            # there by the required 1e-8 sentinel, not the smaller hit value.
            final_errors = np.asarray([run["errors"][-1] for run in runs], dtype=float)
            summary_rows.append((
                function_number,
                dimension,
                float(np.min(final_errors)),
                float(np.max(final_errors)),
                float(np.median(final_errors)),
                float(np.mean(final_errors)),
                float(np.std(final_errors, ddof=1)),
            ))

    summary_path = output_dir / f"{algorithm_name}_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["function", "dimension", "best", "worst", "median", "mean", "std"])
        writer.writerows(summary_rows)
    archive_path = output_dir / f"{algorithm_name}_CEC2022_results.zip"
    with ZipFile(archive_path, "w", compression=ZIP_DEFLATED) as archive:
        for dimension in SUPPORTED_DIMENSIONS:
            for function_number in range(1, 13):
                filename = f"{algorithm_name}_{function_number}_{dimension}.txt"
                archive.write(output_dir / filename, arcname=filename)
    return summary_rows


def measure_cec2022_complexity(optimizer, dimension):
    """Measure T0, T1, mean T2, and (T2-T1)/T0 as specified by the report.

    T0 is the prescribed 200,000-iteration arithmetic loop. T1 times 200,000
    evaluations of F1. T2 times five complete optimizer runs, each capped at
    200,000 evaluations of F1. Unlike competition trials, these timing runs do
    not stop at the 1e-8 target.
    """
    if dimension not in SUPPORTED_DIMENSIONS:
        raise ValueError(f"dimension must be one of {SUPPORTED_DIMENSIONS}")
    timing_fes = 200_000
    x = 0.55
    start = perf_counter()
    for _ in range(timing_fes):
        x = x + x
        x = x / 2.0
        x = x * x
        x = math.sqrt(x)
        x = math.log(x)
        x = math.exp(x)
        x = x / (x + 2.0)
    t0 = perf_counter() - start

    candidate = np.full(dimension, 0.5, dtype=float)
    start = perf_counter()
    for _ in range(timing_fes):
        cec2022(candidate, 1)
    t1 = perf_counter() - start

    lower_bounds = np.full(dimension, LOWER_BOUND, dtype=float)
    upper_bounds = np.full(dimension, -LOWER_BOUND, dtype=float)
    t2_values = []
    timing_seeds = [competition_seed(1, dimension, run_id) for run_id in range(5)]
    for seed in timing_seeds:
        evaluations = 0

        def objective(point):
            nonlocal evaluations
            if evaluations >= timing_fes:
                raise _StopTrial
            value = cec2022(point, 1)
            evaluations += 1
            return value

        start = perf_counter()
        try:
            optimizer(objective, lower_bounds, upper_bounds, timing_fes, seed)
        except _StopTrial:
            pass
        elapsed = perf_counter() - start
        if evaluations != timing_fes:
            raise RuntimeError(
                f"Complexity run used {evaluations} FEs; expected {timing_fes}"
            )
        t2_values.append(elapsed)

    mean_t2 = float(np.mean(t2_values))
    return {
        "dimension": dimension,
        "T0_seconds": t0,
        "T1_seconds": t1,
        "T2_seconds_each": tuple(t2_values),
        "T2_mean_seconds": mean_t2,
        "(T2-T1)/T0": (mean_t2 - t1) / t0,
        "language": f"Python {python_version()}",
        "system": platform(),
        "processor": processor(),
        "timing_fes": timing_fes,
    }


def write_complexity_report(path, measurements):
    """Write the report's T0/T1/T2 fields and machine details as a text file."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(f"Dimension: {measurements['dimension']}\n")
        stream.write(f"T0/T1/T2 iterations or evaluations: {measurements['timing_fes']}\n")
        stream.write(f"Language: {measurements['language']}\n")
        stream.write(f"System: {measurements['system']}\n")
        stream.write(f"Processor: {measurements['processor']}\n")
        stream.write(f"T0 (seconds): {measurements['T0_seconds']:.10f}\n")
        stream.write(f"T1 (seconds): {measurements['T1_seconds']:.10f}\n")
        for run_number, elapsed in enumerate(measurements["T2_seconds_each"], start=1):
            stream.write(f"T2 run {run_number} (seconds): {elapsed:.10f}\n")
        stream.write(f"T2 mean (seconds): {measurements['T2_mean_seconds']:.10f}\n")
        stream.write(f"(T2-T1)/T0: {measurements['(T2-T1)/T0']:.10f}\n")
