import unittest

from nsga2.individual import Individual
from nsga2.sorting import calculate_crowding_distance


class TestCrowdingDistance(unittest.TestCase):

    def test_boundary_points_are_infinite(self):

        points = [
            [0.0, 4.0],
            [1.0, 3.0],
            [2.0, 2.0],
            [3.0, 1.0]
        ]

        front = []

        for point in points:
            individual = Individual([0.0])
            individual.objectives = point
            front.append(individual)

        calculate_crowding_distance(front)

        self.assertEqual(
            front[0].crowding_distance,
            float("inf")
        )

        self.assertEqual(
            front[-1].crowding_distance,
            float("inf")
        )

    def test_internal_points_have_finite_distance(self):

        points = [
            [0.0, 4.0],
            [1.0, 3.0],
            [2.0, 2.0],
            [3.0, 1.0]
        ]

        front = []

        for point in points:
            individual = Individual([0.0])
            individual.objectives = point
            front.append(individual)

        calculate_crowding_distance(front)

        for individual in front[1:-1]:
            self.assertTrue(
                individual.crowding_distance < float("inf")
            )


if __name__ == "__main__":
    unittest.main()
