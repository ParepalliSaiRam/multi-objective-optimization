"""Run NSGA-II on the supplied 24-problem CEC2020 MMO suite."""

import argparse
from functools import partial
from pathlib import Path

from scipy.io import loadmat

from benchmarks.cec2020_mmo import CEC2020_MMO_PROBLEMS, evaluate_mmo
from experiments.common import run_multiple, save_summary_metrics
from experiments.results import save_results
from nsga2 import NSGA2
from visualization.plots import save_pareto_front_plot, save_pareto_set_plot

ROOT = Path(__file__).resolve().parents[1]
REFERENCE_DIR = ROOT / "MM_testfunctions" / "Reference_PSPF_data"


def _make_case(problem):
    reference_path = REFERENCE_DIR / f"{problem.name}_Reference_PSPF_data.mat"
    if not reference_path.is_file():
        raise FileNotFoundError(
            f"Missing CEC2020 reference PS/PF data: {reference_path}. "
            "Restore the supplied Reference_PSPF_data folder to generate comparisons."
        )
    data = loadmat(reference_path)
    reference_front = data["PF"]
    reference_set = data["PS"]
    return (
        problem.name,
        partial(evaluate_mmo, problem=problem.number),
        lambda n, p=problem: (list(p.lower_bounds), list(p.upper_bounds)),
        lambda front=reference_front: front,
        lambda pareto_set=reference_set: pareto_set,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=3, help="independent runs per problem (default: 3)")
    parser.add_argument("--population-size", type=int, default=100)
    parser.add_argument("--generations", type=int, default=200)
    parser.add_argument("--problems", nargs="*", help="problem names such as MMF1 MMF14; defaults to all 24")
    args = parser.parse_args(argv)
    if args.runs < 1 or args.population_size < 2 or args.generations < 1:
        parser.error("runs and generations must be positive, and population size must be at least 2")

    selected = [p for p in CEC2020_MMO_PROBLEMS if args.problems is None or p.name in args.problems]
    unknown = set(args.problems or ()) - {p.name for p in CEC2020_MMO_PROBLEMS}
    if unknown:
        parser.error("unknown problem name(s): " + ", ".join(sorted(unknown)))
    if not selected:
        parser.error("select at least one problem")

    all_results = []
    for problem in selected:
        name, objective, bounds, reference, reference_set = _make_case(problem)
        params = {
            "population_size": args.population_size,
            "generations": args.generations,
            "number_of_variables": problem.variables,
            "crossover_probability": 0.9,
            "mutation_probability": 1.0 / problem.variables,
            "eta_c": 20,
            "eta_m": 20,
        }
        print(f"\n========== {name} ==========")
        result = run_multiple(
            NSGA2, params, objective, bounds, reference, name, "NSGA-II", args.runs
        )
        all_results.extend(result["runs"])
        save_results(result["runs"], str(ROOT / "results" / "data" / "cec2020_mmo_results.csv"))
        plot_path = save_pareto_front_plot(
            name, reference(), result["obtained_fronts"],
            output_directory="results/plots/mmo_24",
        )
        print(f"Reference and obtained fronts: {plot_path}")
        set_path = save_pareto_set_plot(
            name, reference_set(), result["obtained_sets"],
            output_directory="results/plots/mmo_24",
        )
        print(f"Reference and obtained Pareto sets: {set_path}")

    save_summary_metrics(all_results, str(ROOT / "results" / "cec2020_mmo_summary.csv"))
    print(f"\nSummary saved to {ROOT / 'results' / 'cec2020_mmo_summary.csv'}")


if __name__ == "__main__":
    main()
