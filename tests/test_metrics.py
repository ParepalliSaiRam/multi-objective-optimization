import unittest

from metrics import (
    hypervolume_2d,
    igd,
    gd,
    spacing
)

class TestMetrics(unittest.TestCase):

    def test_simple_hypervolume(self):

        points = [
            [1.0, 4.0],
            [2.0, 3.0],
            [3.0, 2.0]
        ]

        reference_point = [4.0, 5.0]

        hv = hypervolume_2d(
            points,
            reference_point
        )

        self.assertAlmostEqual(
            hv,
            6.0
        )

    def test_igd_zero_for_identical_fronts(self):

        true_front = [
            [0.0, 1.0],
            [0.5, 0.5],
            [1.0, 0.0]
        ]

        obtained_front = [
            [0.0, 1.0],
            [0.5, 0.5],
            [1.0, 0.0]
        ]

        value = igd(
            obtained_front,
            true_front
        )

        self.assertAlmostEqual(
            value,
            0.0
        )

    def test_points_outside_reference_are_ignored(self):

        points = [
            [1.0, 4.0],
            [2.0, 3.0],
            [10.0, 10.0]
        ]

        reference_point = [4.0, 5.0]

        hv = hypervolume_2d(
            points,
            reference_point
        )

        self.assertAlmostEqual(
            hv,
            5.0
        )

    def test_gd_zero_for_identical_fronts(self):

        true_front = [
            [0.0, 1.0],
            [0.5, 0.5],
            [1.0, 0.0]
        ]

        obtained_front = [
            [0.0, 1.0],
            [0.5, 0.5],
            [1.0, 0.0]
        ]

        value = gd(
            obtained_front,
            true_front
        )

        self.assertAlmostEqual(
            value,
            0.0
        )

    def test_spacing_zero_for_equally_spaced_points(self):

        points = [
            [0.0, 1.0],
            [0.5, 0.5],
            [1.0, 0.0]
        ]

        value = spacing(points)

        self.assertAlmostEqual(
            value,
            0.0
        )


if __name__ == "__main__":
    unittest.main()