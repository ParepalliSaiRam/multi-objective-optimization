import unittest

from algorithms.nsga2 import NSGA2
from benchmarks.zdt import zdt1, zdt1_bounds
from algorithms.nsga2.individual import Individual
from algorithms.nsga2.operators import sbx_crossover
from algorithms.nsga2.operators import polynomial_mutation


class TestNSGA2(unittest.TestCase):

    def test_nsga2_runs(self):

        number_of_variables = 30

        lower_bounds, upper_bounds = zdt1_bounds(
            number_of_variables
        )

        algorithm = NSGA2(
            population_size=20,
            generations=5,
            number_of_variables=30,
            objective_function=zdt1,
            lower_bounds=[0.0] * 30,
            upper_bounds=[1.0] * 30,
            crossover_probability=0.9,
            mutation_probability=1.0 / 30,
            eta_c=20,
            eta_m=20
        )

        population, pareto_front = algorithm.run()

        self.assertEqual(
            len(population),
            20
        )

        self.assertGreater(
            len(pareto_front),
            0
        )

        for individual in population:

            self.assertEqual(
                len(individual.objectives),
                2
            )

            for x in individual.variables:
                self.assertGreaterEqual(x, 0.0)
                self.assertLessEqual(x, 1.0)

        def test_sbx_respects_bounds(self):

            parent1 = Individual(
                [0.2, 0.4, 0.6]
            )

            parent2 = Individual(
                [0.8, 0.6, 0.3]
            )

            lower_bounds = [0.0, 0.0, 0.0]
            upper_bounds = [1.0, 1.0, 1.0]

            child1, child2 = sbx_crossover(
                parent1,
                parent2,
                lower_bounds,
                upper_bounds,
                probability=1.0,
                eta_c=20
            )

            for child in [child1, child2]:

                for value in child:

                    self.assertGreaterEqual(
                        value,
                        0.0
                    )

                    self.assertLessEqual(
                        value,
                        1.0
                    )

        def test_mutation_respects_bounds(self):

            variables = [
                0.2,
                0.5,
                0.8,
                0.1
            ]

            lower_bounds = [0.0] * 4
            upper_bounds = [1.0] * 4

            mutated = polynomial_mutation(
                variables,
                lower_bounds,
                upper_bounds,
                probability=1.0,
                eta_m=20
            )

            for value in mutated:

                self.assertGreaterEqual(
                    value,
                    0.0
                )

                self.assertLessEqual(
                    value,
                    1.0
                )

if __name__ == "__main__":
    unittest.main()