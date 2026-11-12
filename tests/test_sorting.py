import unittest

from nsga2.individual import Individual
from nsga2.sorting import dominates
from nsga2.sorting import fast_non_dominated_sort


class TestDominance(unittest.TestCase):

    def test_dominates(self):
        a = Individual([0.0])
        b = Individual([0.0])

        a.objectives = [1.0, 2.0]
        b.objectives = [2.0, 3.0]

        self.assertTrue(dominates(a, b))
        self.assertFalse(dominates(b, a))

    def test_no_dominance(self):
        a = Individual([0.0])
        b = Individual([0.0])

        a.objectives = [1.0, 4.0]
        b.objectives = [2.0, 3.0]

        self.assertFalse(dominates(a, b))
        self.assertFalse(dominates(b, a))

    def test_equal_solutions_do_not_dominate(self):
        a = Individual([0.0])
        b = Individual([0.0])

        a.objectives = [1.0, 2.0]
        b.objectives = [1.0, 2.0]

        self.assertFalse(dominates(a, b))
        self.assertFalse(dominates(b, a))

        def test_fast_non_dominated_sort(self):

            points = [
                [1.0, 4.0],
                [2.0, 3.0],
                [3.0, 2.0],
                [4.0, 1.0],
                [3.0, 4.0],
                [4.0, 4.0]
            ]

            population = []

            for point in points:
                individual = Individual([0.0])
                individual.objectives = point
                population.append(individual)

            fronts = fast_non_dominated_sort(population)

            # First four points are non-dominated.
            self.assertEqual(len(fronts[0]), 4)

            # The remaining points should be in later fronts.
            self.assertEqual(len(fronts), 3)

            for individual in fronts[0]:
                self.assertEqual(individual.rank, 1)

if __name__ == "__main__":
    unittest.main()
