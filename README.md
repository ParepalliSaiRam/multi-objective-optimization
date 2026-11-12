# Multi-objective Optimization

Run the NSGA-II benchmark suite with:

```bash
python -m experiments.run_nsga2
```

The suite contains these 30 problems:
ZDT1-ZDT4, ZDT6; Binh1, Binh3, Deb, Deb2, Fonseca, Fonseca2, Kursawe,
Laumanns, Murata, Quagliarella, Rendon, Rendon2, Schaffer, Schaffer2,
Srinivas, Tanaka, Viennet, Viennet2; and DTLZ1-DTLZ7.

The experiment uses 5 independent runs. Most benchmarks use a population of
100 and 1,000 generations (100,100 candidates per run). DTLZ1-DTLZ7 use a
population of 200, mutation distribution index 10, and 3,000 generations
(600,200 candidates per run) to improve 3-objective front coverage and
exploration. DTLZ uses 12 variables and 3
objectives, with 50 divisions for reference fronts. Other functions use their
standard variable counts and bounds.

For each benchmark, the run saves a side-by-side obtained/reference front plot
under `results/plots/benchmarks_30/`, using 3D plots when the benchmark has
three objectives.
Summary metrics are saved in
`results/summary_metrics.csv`; raw run metrics are appended to
`results/data/results.csv`.

Create the combined comparison report from the current plots with:

```bash
python -m experiments.create_benchmark_comparison_pdf
```

The 30-problem report is saved to `output/pdf/benchmark_comparison.pdf`.

## CEC 2022 single-objective benchmarks

CEC2022 is a separate suite of 12 **single-objective** minimization functions.
Their domains are `[-100, 100]`
with the competition dimensions 10 and 20. The objective implementation and
official shift, rotation, and shuffle data are available through
`benchmarks/cec2022.py` and `benchmarks/cec2022_data/`. The existing NSGA-II
suite remains multi-objective; these scalar functions are provided separately
so they can be used with a single-objective optimizer.

The numeric input data was copied from the official CEC2022 reference
repository, [P-N-Suganthan/2022-SO-BO](https://github.com/P-N-Suganthan/2022-SO-BO).
The implementation exposes each function as a scalar objective callable:

```python
from benchmarks.cec2022 import cec2022, make_cec2022

value = cec2022(candidate, function_number=1)  # F1, dimension 10 or 20
objective = make_cec2022(6)                    # callable for F6
```

`CEC2022_BENCHMARKS` lists all 12 function numbers, names, known optima, and
common bounds. The objective values include each function's published bias
(300 through 2700, depending on the function).

The competition protocol and output writer are in
`experiments/cec2022_protocol.py`. Supply a single-objective optimizer callable
with signature `optimizer(objective, lower_bounds, upper_bounds, max_fes,
seed)` to `run_cec2022_suite(optimizer, "AlgorithmName")`. It runs 30 seeded
trials for all 12 functions at both dimensions, enforces the published FE
budget and target, records the 16 `D^(k/5-3) * MaxFES` error checkpoints and
`FEterm`, then writes the 24 named tab-separated text matrices, a summary CSV,
and a ZIP containing the 24 submission files under `results/cec2022/`. The
official seed values are included with the other CEC2022 input data.

Call `measure_cec2022_complexity(optimizer, dimension)` and
`write_complexity_report(path, measurements)` to produce the separate T0/T1/T2
timing measurements and machine/language details requested by the report.

The selected single-objective optimizer is Differential Evolution,
DE/rand/1/bin, with population 100, F=0.5, and CR=0.9. It uses uniform
initialization, clipping at the bounds, and greedy one-to-one selection. These
settings are shared across all 12 functions and both dimensions; no
problem-specific tuning is performed. Start the complete CEC2022 experiment
with:

```bash
python -m experiments.run_cec2022_de
```

This executes 30 runs for every function/dimension pair and writes the chosen
DE settings to `results/cec2022/DE_parameters.json`. After the benchmark runs,
it also measures and writes the requested T0/T1/T2 complexity reports for both
dimensions. The CEC numeric seeds are preserved; Python's NumPy PCG64 generator
is deterministic but does not produce the same random stream as MATLAB's
Twister generator.

The report has internal inconsistencies: its summary table gives Composition
Function 2 four components, while its detailed definition gives three. This
implementation follows the detailed definition and reference package (three).
The listed Hybrid Function 2 partition vector also has seven proportions for
six components; this implementation uses the six-component partition from the
reference package.
The report's detailed equations also differ from parts of the reference C
implementation for F3, F4, F5, and the F7 component used in Hybrid Function 2;
those functions follow the report equations here.

## CEC 2020 multimodal multiobjective benchmarks

The 24 CEC2020 MMO entries listed by the supplied MATLAB example are available
through `benchmarks/cec2020_mmo.py`. Each problem has published bounds and
supports vector or batch evaluation. For example:

```python
from benchmarks.cec2020_mmo import CEC2020_MMO_PROBLEMS, evaluate_mmo, make_mmo

problem = CEC2020_MMO_PROBLEMS[0]  # MMF1 and its bounds
objectives = evaluate_mmo([2.0, 0.0], problem.number)
objective = make_mmo("MMF14", objectives=3)
```

The suite includes the base functions, expanded/local variants, and scalable
MMF14-MMF16 variants from the CEC2020 benchmark package. The runner loads the
reference PS/PF `.mat` files from
`MM_testfunctions/Reference_PSPF_data/`; all 24 files required by this
project's problem list are included there.

Run NSGA-II against all 24 problems and compare each obtained front with its
included reference front using:

```bash
python -m experiments.run_cec2020_mmo
```

The runner defaults to 3 runs, a population of 100, and 200 generations, and
writes metrics and plots under `results/`. Adjust the workload or run selected
problems with options such as `--generations 50 --runs 1 --problems MMF1 MMF14`.
Each run now saves both Pareto-front plots (objective space) and Pareto-set
plots (decision space), comparing NSGA-II's obtained solutions with the
reference data under `results/plots/mmo_24/`. After running the suite, create
the combined PS/PF PDF with:

```bash
python -m experiments.create_cec2020_mmo_comparison_pdf
```

The 24-page document is saved to `output/pdf/CEC2020_MMO_PS_PF_Comparison.pdf`.
