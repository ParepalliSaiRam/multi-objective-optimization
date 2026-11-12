from nsga2 import NSGA2
from functools import partial

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
from benchmarks.dtlz import (
    dtlz1, dtlz2, dtlz3, dtlz4, dtlz5, dtlz6, dtlz7,
    dtlz_bounds, dtlz_pareto_front,
)
from benchmarks.classic import CLASSIC_BENCHMARKS, bounds as classic_bounds, reference_front as classic_reference_front

from experiments.common import (
    run_multiple,
    save_summary_metrics
)
NSGA2_PARAMETERS = {
    "population_size": 100,
    "generations": 1000,
    "number_of_variables": 30,
    "crossover_probability": 0.9,
    "mutation_probability": 1.0 / 30,
    "eta_c": 20,
    "eta_m": 20
}

NUMBER_OF_RUNS = 5
DTLZ_OBJECTIVES = 3
DTLZ_REFERENCE_DIVISIONS = 50

from experiments.results import save_results
from visualization.plots import save_pareto_front_plot

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
    ),
    *(
        (
            f"DTLZ{index}",
            partial(function, number_of_objectives=DTLZ_OBJECTIVES),
            dtlz_bounds,
            partial(dtlz_pareto_front, f"DTLZ{index}", DTLZ_REFERENCE_DIVISIONS, DTLZ_OBJECTIVES),
        )
        for index, function in enumerate((dtlz1, dtlz2, dtlz3, dtlz4, dtlz5, dtlz6, dtlz7), start=1)
    ),
    *(
        (
            name,
            function,
            partial(classic_bounds, bounds_name),
            partial(classic_reference_front, bounds_name),
            constraint_function,
        )
        for name, function, bounds_name, constraint_function in CLASSIC_BENCHMARKS
    ),
]

def main():
    all_results = []
    benchmark_fronts = []
    for (
        benchmark_name,
        objective_function,
        bounds_function,
        pareto_front_function,
        *extra
    ) in BENCHMARKS:

        print(
            f"\n\n========== {benchmark_name} =========="
        )

        constraint_function = extra[0] if extra else None
        if benchmark_name in {"ZDT1", "ZDT2", "ZDT3"}:
            number_of_variables = 30
        elif benchmark_name in {"ZDT4", "ZDT6"}:
            number_of_variables = 10
        elif benchmark_name.startswith("DTLZ"):
            number_of_variables = 12
        else:
            number_of_variables = len(bounds_function(1)[0])
        benchmark_parameters = {
            **NSGA2_PARAMETERS,
            "number_of_variables": number_of_variables,
            "mutation_probability": 1.0 / number_of_variables,
        }
        if benchmark_name.startswith("DTLZ"):
            # DTLZ's 3-objective fronts need more population diversity, and
            # broader mutation helps exploration on the multimodal variants.
            benchmark_parameters.update(population_size=200, eta_m=10)
        if benchmark_name.startswith("DTLZ"):
            # DTLZ variants need a longer search to converge and cover their
            # three-objective fronts consistently.
            benchmark_parameters["generations"] = 3000
        results = run_multiple(
            algorithm_class=NSGA2,
            algorithm_parameters=benchmark_parameters,
            objective_function=objective_function,
            bounds_function=bounds_function,
            pareto_front_function=pareto_front_function,
            benchmark_name=benchmark_name,
            algorithm_name="NSGA-II",
            number_of_runs=NUMBER_OF_RUNS,
            constraint_violation_function=constraint_function,
        )

        all_results.extend(results["runs"])
        benchmark_fronts.append(
            (benchmark_name, pareto_front_function, results["obtained_fronts"])
        )

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

    print("\nSaving benchmark Pareto-front plots...")
    for benchmark_name, pareto_front_function, obtained_fronts in benchmark_fronts:
        plot_path = save_pareto_front_plot(
            benchmark_name,
            pareto_front_function(),
            obtained_fronts,
            output_directory="results/plots/benchmarks_30",
        )
        print(f"{benchmark_name}: {plot_path}")

if __name__ == "__main__":
    main()
