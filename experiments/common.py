import random
import time
import statistics
import csv

from metrics import (
    hypervolume_2d,
    igd,
    gd,
    spacing
)
from benchmarks.zdt import ZDT_REFERENCE_POINTS

def save_summary_metrics(raw_results, output_file):
    benchmarks = sorted(set(result["benchmark"] for result in raw_results))

    with open(output_file, "w", newline="") as file:
        writer = csv.writer(file)

        writer.writerow([
            "benchmark",
            "hv_mean", "hv_std",
            "igd_mean", "igd_std",
            "gd_mean", "gd_std",
            "spacing_mean", "spacing_std",
            "runtime_mean", "runtime_std"
        ])

        for benchmark in benchmarks:
            benchmark_results = [
                result for result in raw_results
                if result["benchmark"] == benchmark
            ]

            hv_values = [
                result["hypervolume"]
                for result in benchmark_results
            ]

            igd_values = [
                result["igd"]
                for result in benchmark_results
            ]

            gd_values = [
                result["gd"]
                for result in benchmark_results
            ]

            spacing_values = [
                result["spacing"]
                for result in benchmark_results
            ]

            runtime_values = [
                result["runtime"]
                for result in benchmark_results
            ]

            writer.writerow([
                benchmark,

                statistics.mean(hv_values),
                statistics.stdev(hv_values),

                statistics.mean(igd_values),
                statistics.stdev(igd_values),

                statistics.mean(gd_values),
                statistics.stdev(gd_values),

                statistics.mean(spacing_values),
                statistics.stdev(spacing_values),

                statistics.mean(runtime_values),
                statistics.stdev(runtime_values)
            ])

def run_algorithm(
    algorithm_class,
    algorithm_parameters,
    objective_function,
    bounds_function,
    pareto_front_function,
    benchmark_name,
    seed
):

    # Reproducibility
    random.seed(seed)

    number_of_variables = algorithm_parameters[
        "number_of_variables"
    ]

    lower_bounds, upper_bounds = bounds_function(
        number_of_variables
    )

    # Create algorithm
    algorithm = algorithm_class(
        objective_function=objective_function,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        **algorithm_parameters
    )

    # Measure execution time
    start_time = time.perf_counter()

    population, pareto_front = algorithm.run()

    end_time = time.perf_counter()

    runtime = end_time - start_time

    # Obtained Pareto front
    obtained_points = [
        individual.objectives
        for individual in pareto_front
    ]

    # True Pareto front
    true_front = pareto_front_function()

    true_points = []

    for true_f1, true_f2 in true_front:
        for i in range(len(true_f1)):
            true_points.append(
                [true_f1[i], true_f2[i]]
            )

    # Metrics
    reference_point = ZDT_REFERENCE_POINTS[
        benchmark_name
    ]

    hv_value = hypervolume_2d(
        obtained_points,
        reference_point
    )

    igd_value = igd(
        obtained_points,
        true_points
    )

    gd_value = gd(
        obtained_points,
        true_points
    )

    spacing_value = spacing(
        obtained_points
    )

    return (
        hv_value,
        igd_value,
        gd_value,
        spacing_value,
        runtime
    )

def run_multiple(
    algorithm_class,
    algorithm_parameters,
    objective_function,
    bounds_function,
    pareto_front_function,
    benchmark_name,
    algorithm_name,
    number_of_runs=3
):

    hypervolumes = []
    igd_values = []
    gd_values = []
    spacing_values = []
    runtimes = []

    run_results = []

    for seed in range(1, number_of_runs + 1):

        hv, igd_value, gd_value, spacing_value, runtime = run_algorithm(
            algorithm_class,
            algorithm_parameters,
            objective_function,
            bounds_function,
            pareto_front_function,
            benchmark_name,
            seed
        )

        run_results.append({
            "benchmark": benchmark_name,
            "algorithm": algorithm_name,
            "seed": seed,
            "hypervolume": hv,
            "igd": igd_value,
            "gd": gd_value,
            "spacing": spacing_value,
            "runtime": runtime
        })

        hypervolumes.append(hv)
        igd_values.append(igd_value)
        gd_values.append(gd_value)
        spacing_values.append(spacing_value)
        runtimes.append(runtime)

        print(
            f"Run {seed}/{number_of_runs} | "
            f"HV: {hv:.6f} | "
            f"IGD: {igd_value:.6f} | "
            f"GD: {gd_value:.6f} | "
            f"Spacing: {spacing_value:.6f} | "
            f"Time: {runtime:.4f}s"
        )

    gd_mean = statistics.mean(gd_values)

    gd_std = (
        statistics.stdev(gd_values)
        if number_of_runs > 1
        else 0.0
    )

    spacing_mean = statistics.mean(spacing_values)

    spacing_std = (
        statistics.stdev(spacing_values)
        if number_of_runs > 1
        else 0.0
    )
    
    return {
        "runs": run_results,

        "hv_mean": statistics.mean(hypervolumes),
        "hv_std": statistics.stdev(hypervolumes)
            if number_of_runs > 1 else 0.0,

        "igd_mean": statistics.mean(igd_values),
        "igd_std": statistics.stdev(igd_values)
            if number_of_runs > 1 else 0.0,

        "gd_mean": gd_mean,
        "gd_std": gd_std,

        "spacing_mean": spacing_mean,
        "spacing_std": spacing_std,

        "runtime_mean": statistics.mean(runtimes),
        "runtime_std": statistics.stdev(runtimes)
            if number_of_runs > 1 else 0.0
    }