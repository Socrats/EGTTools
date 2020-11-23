import unittest

import numpy as np

from egttools.analytical import replicator_equation
from egttools.utils import find_saddle_type_and_gradient_direction


class UtilsTestCase(unittest.TestCase):
    def setUp(self) -> None:
        nb_points = 101
        strategy_i = np.linspace(0, 1, num=nb_points, dtype=np.float64)
        strategy_j = 1 - strategy_i
        states = np.array((strategy_i, strategy_j)).T

        # Payoff matrix
        v, d, t = 2, 3, 1
        payoffs = np.array([
            [(v - d) / 2, v],
            [0, (v / 2) - t],
        ])

        # Calculate gradient
        self.gradients = np.array([replicator_equation(states[i], payoffs)[0] for i in range(len(states))])
        epsilon = 1e-3
        self.saddle_points_idx = np.where((self.gradients <= epsilon) & (self.gradients >= -epsilon))[0]

    def test_find_saddle_type_and_gradient_direction(self):
        """
        This test is used to check if the types of saddle points are correctly
        identified and if the arrow directions are correct.
        """

        saddle_type, gradient_direction = find_saddle_type_and_gradient_direction(self.gradients,
                                                                                  self.saddle_points_idx)

        self.assertEqual(saddle_type[0], False)  # First
        self.assertEqual(saddle_type[2], False)  # Last saddle points are not stable
        self.assertEqual(saddle_type[1], True)  # Middle saddle point is stable

        # check that there are only 2 arrows
        self.assertEqual(gradient_direction.shape, (2, 2))
        # check tha the first arrows goes from 0 to the stable point
        self.assertEqual(tuple(gradient_direction[0]), (0., 0.79))
        # check tha the second goes from 1 to the stable point
        self.assertEqual(tuple(gradient_direction[1]), (1., 0.81))

    def tearDown(self) -> None:
        del self.gradients


def suite():
    test_suite = unittest.TestSuite()
    test_suite.addTest(UtilsTestCase('test_find_saddle_type_and_gradient_direction'))
    return test_suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(suite())
