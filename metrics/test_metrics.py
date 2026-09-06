import unittest

from metrics.metrics import hypervolume_2d


class TestHypervolume(unittest.TestCase):

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


if __name__ == "__main__":
    unittest.main()