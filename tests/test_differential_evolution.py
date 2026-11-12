import unittest

import numpy as np
from unittest.mock import patch

from experiments.cec2022_protocol import run_trial
from optimizers.differential_evolution import DifferentialEvolution


class TestDifferentialEvolution(unittest.TestCase):
    def test_optimizer_respects_budget_bounds_and_seed(self):
        lower = np.full(5, -5.0)
        upper = np.full(5, 5.0)

        def run_once():
            calls = []

            def objective(x):
                calls.append(np.asarray(x).copy())
                return float(np.dot(x, x))

            result = DifferentialEvolution(population_size=20)(
                objective, lower, upper, max_fes=103, seed=2718
            )
            return result, calls

        first, first_calls = run_once()
        second, second_calls = run_once()
        self.assertEqual(first["evaluations"], 103)
        self.assertEqual(len(first_calls), 103)
        self.assertTrue(np.all(first["x"] >= lower))
        self.assertTrue(np.all(first["x"] <= upper))
        self.assertEqual(first["value"], second["value"])
        self.assertTrue(np.array_equal(first["x"], second["x"]))
        self.assertTrue(all(np.all(x >= lower) and np.all(x <= upper) for x in first_calls))
        self.assertEqual(len(second_calls), 103)

    def test_parameters_are_validated(self):
        with self.assertRaises(ValueError):
            DifferentialEvolution(population_size=3)
        with self.assertRaises(ValueError):
            DifferentialEvolution(differential_weight=0.0)
        with self.assertRaises(ValueError):
            DifferentialEvolution(crossover_rate=1.1)

    def test_de_runs_through_cec_protocol_with_exact_budget(self):
        with patch.dict(
            "experiments.cec2022_protocol.MAX_FES_BY_DIMENSION",
            {10: 100, 20: 100},
        ):
            result = run_trial(DifferentialEvolution(), 1, 10, 0)
        self.assertEqual(result["FEterm"], 100)
        self.assertEqual(len(result["errors"]), 16)


if __name__ == "__main__":
    unittest.main()
