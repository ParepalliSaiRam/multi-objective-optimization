import matplotlib.pyplot as plt

from algorithms.nsga2 import NSGA2

from benchmarks.zdt import (
    zdt1,
    zdt1_bounds,
    zdt1_pareto_front,

    zdt2,
    zdt2_bounds,
    zdt2_pareto_front,

    zdt3,
    zdt3_bounds,
    zdt3_pareto_front,

    zdt4,
    zdt4_bounds,
    zdt4_pareto_front,

    zdt6,
    zdt6_bounds,
    zdt6_pareto_front,
    ZDT_REFERENCE_POINTS
)

from experiments.common import (
    run_multiple,
    save_summary_metrics
)
NSGA2_PARAMETERS = {
    "population_size": 100,
    "generations": 250,
    "number_of_variables": 30,
    "crossover_probability": 0.9,
    "mutation_probability": 1.0 / 30,
    "eta_c": 20,
    "eta_m": 20
}

from experiments.results import save_results

BENCHMARKS = [
    (
        "ZDT1",
        zdt1,
        zdt1_bounds,
        zdt1_pareto_front
    ),
    (
        "ZDT2",
        zdt2,
        zdt2_bounds,
        zdt2_pareto_front
    ),
    (
        "ZDT3",
        zdt3,
        zdt3_bounds,
        zdt3_pareto_front
    ),
    (
        "ZDT4",
        zdt4,
        zdt4_bounds,
        zdt4_pareto_front
    ),
    (
        "ZDT6",
        zdt6,
        zdt6_bounds,
        zdt6_pareto_front
    )
]

def main():
    all_results = []
    for (
        benchmark_name,
        objective_function,
        bounds_function,
        pareto_front_function
    ) in BENCHMARKS:

        print(
            f"\n\n========== {benchmark_name} =========="
        )

        results = run_multiple(
            algorithm_class=NSGA2,
            algorithm_parameters=NSGA2_PARAMETERS,
            objective_function=objective_function,
            bounds_function=bounds_function,
            pareto_front_function=pareto_front_function,
            benchmark_name=benchmark_name,
            algorithm_name="NSGA-II",
            number_of_runs=10
        )

        all_results.extend(results["runs"])

        save_results(
            results["runs"]
        )

        print("\nSummary:")

        print(
            f"HV: "
            f"{results['hv_mean']:.6f} "
            f"+/- {results['hv_std']:.6f}"
        )

        print(
            f"IGD: "
            f"{results['igd_mean']:.6f} "
            f"+/- {results['igd_std']:.6f}"
        )

        print(
            f"GD: "
            f"{results['gd_mean']:.6f} "
            f"+/- {results['gd_std']:.6f}"
        )

        print(
            f"Spacing: "
            f"{results['spacing_mean']:.6f} "
            f"+/- {results['spacing_std']:.6f}"
        )

        print(
            f"Runtime: "
            f"{results['runtime_mean']:.4f}s "
            f"+/- {results['runtime_std']:.4f}s"
        )

    save_summary_metrics(
        all_results,
        "results/summary_metrics.csv"
    )

if __name__ == "__main__":
    main()