import csv
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import numpy as np

from benchmarks.cec2022 import OPTIMAL_VALUES, SUPPORTED_DIMENSIONS, cec2022
from experiments.cec2022_protocol import (
    MAX_FES_BY_DIMENSION,
    RUNS_PER_PROBLEM,
    competition_seed,
    evaluation_checkpoints,
    run_trial,
    run_cec2022_suite,
    write_result_file,
)


class TestCEC2022Benchmarks(unittest.TestCase):
    def test_all_functions_reach_published_optimum_at_shift(self):
        data_dir = Path(__file__).parents[1] / "benchmarks" / "cec2022_data"
        for dimension in SUPPORTED_DIMENSIONS:
            for function_number, expected in OPTIMAL_VALUES.items():
                with self.subTest(function=function_number, dimension=dimension):
                    shifts = np.loadtxt(data_dir / f"shift_data_{function_number}.txt")
                    candidate = np.asarray(shifts).reshape(-1, dimension)[0]
                    self.assertAlmostEqual(cec2022(candidate, function_number), expected, places=8)

    def test_protocol_checkpoints_and_seed_index(self):
        data_dir = Path(__file__).parents[1] / "benchmarks" / "cec2022_data"
        seeds = np.asarray(np.loadtxt(data_dir / "Rand_Seeds.txt"), dtype=int)
        self.assertEqual(evaluation_checkpoints(10)[0], 200)
        self.assertEqual(evaluation_checkpoints(10)[-1], MAX_FES_BY_DIMENSION[10])
        self.assertEqual(evaluation_checkpoints(20)[0], 125)
        self.assertEqual(evaluation_checkpoints(20)[-1], MAX_FES_BY_DIMENSION[20])
        self.assertEqual(competition_seed(1, 10, 0), seeds[1])

    def test_target_termination_and_17_by_30_output(self):
        data_dir = Path(__file__).parents[1] / "benchmarks" / "cec2022_data"
        dimension = 10
        shifts = np.loadtxt(data_dir / "shift_data_1.txt")
        optimum = np.asarray(shifts).reshape(-1, dimension)[0]

        def optimizer(objective, lower, upper, max_fes, seed):
            objective(optimum)

        result = run_trial(optimizer, 1, dimension, 0)
        self.assertEqual(result["FEterm"], 1)
        self.assertEqual(result["errors"], (1.0e-8,) * 16)

        project_temp = Path(__file__).parents[1] / "tmp"
        with tempfile.TemporaryDirectory(dir=project_temp) as temp_dir:
            path = Path(temp_dir) / "DE_1_10.txt"
            write_result_file(path, 1, 10, "DE", [result] * RUNS_PER_PROBLEM)
            with path.open(encoding="utf-8") as stream:
                rows = list(csv.reader(stream, delimiter="\t"))
            self.assertEqual(len(rows), 18)
            self.assertEqual(len(rows[1:]), 17)
            self.assertTrue(all(len(row) == 31 for row in rows))

    def test_suite_writes_all_named_files_and_submission_zip(self):
        def optimizer(objective, lower, upper, max_fes, seed):
            objective(np.zeros_like(lower))

        with tempfile.TemporaryDirectory(dir=Path(__file__).parents[1] / "tmp") as temp_dir:
            with patch(
                "experiments.cec2022_protocol.cec2022",
                side_effect=lambda x, function_number: OPTIMAL_VALUES[function_number],
            ):
                summary = run_cec2022_suite(optimizer, "DE", temp_dir)
            output_dir = Path(temp_dir)
            result_files = sorted(output_dir.glob("DE_*_*.txt"))
            self.assertEqual(len(result_files), 24)
            self.assertEqual(len(summary), 24)
            self.assertTrue(all(np.allclose(row[2:6], (1.0e-8,) * 4) for row in summary))
            self.assertTrue(all(np.isclose(row[6], 0.0) for row in summary))
            with result_files[0].open(encoding="utf-8") as stream:
                self.assertEqual(len(list(csv.reader(stream, delimiter="\t"))), 18)
            archive_path = output_dir / "DE_CEC2022_results.zip"
            with zipfile.ZipFile(archive_path) as archive:
                self.assertEqual(len(archive.namelist()), 24)


if __name__ == "__main__":
    unittest.main()
