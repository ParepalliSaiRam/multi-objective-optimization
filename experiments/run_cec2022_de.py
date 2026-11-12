"""Run the full CEC2022 suite using the project's Differential Evolution."""

import json
import argparse
from pathlib import Path

from experiments.cec2022_protocol import (
    measure_cec2022_complexity,
    run_cec2022_suite,
    write_complexity_report,
)
from optimizers.differential_evolution import (
    DEFAULT_DE_PARAMETERS,
    DifferentialEvolution,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", action="store_true", help="reuse valid completed result matrices")
    args = parser.parse_args()
    output_dir = Path("results/cec2022")
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "DE_parameters.json").open("w", encoding="utf-8") as stream:
        json.dump(DEFAULT_DE_PARAMETERS, stream, indent=2)
        stream.write("\n")

    def report_progress(function_number, dimension, run_number, result):
        print(
            f"F{function_number:02d} D={dimension} Run {run_number}/30 "
            f"FEterm={result['FEterm']} error={result['final_error']:.6e}",
            flush=True,
        )

    optimizer = DifferentialEvolution()
    run_cec2022_suite(
        optimizer, "DE", output_dir, progress_callback=report_progress,
        resume=args.resume,
    )
    for dimension in (10, 20):
        print(f"Measuring CEC complexity at D={dimension}...", flush=True)
        measurements = measure_cec2022_complexity(optimizer, dimension)
        write_complexity_report(
            output_dir / f"DE_complexity_{dimension}.txt", measurements
        )


if __name__ == "__main__":
    main()
