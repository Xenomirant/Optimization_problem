import unittest
import optimizer._optimizers as opt
import numpy as np
import sympy as sym

class NeutonMethodTest(unittest.TestCase):

    def setUp(self) -> None:
        self.optimizer = opt.Neuton

    def test_simple_func(self):
        optimizer = self.optimizer(x0=np.array([1, 2]), function=sym.parse_expr("x1**2 + x2**2 + 5"))
        optimizer.minimize()
        self.assertAlmostEquals(optimizer.minimum, 5.0)

    def test_hard_func(self):
        x1, x2 = sym.symbols("x1 x2")

        fact1a = (x1 + x2 + 1) ** 2
        fact1b = 19 - 14 * x1 + 3 * x1 ** 2 - 14 * x2 + 6 * x1 * x2 + 3 * x2 ** 2
        fact1 = 1 + fact1a * fact1b

        fact2a = (2 * x1 - 3 * x2) ** 2
        fact2b = 18 - 32 * x1 + 12 * x1 ** 2 + 48 * x2 - 36 * x1 * x2 + 27 * x2 ** 2
        fact2 = 30 + fact2a * fact2b

        func = fact1 * fact2
        optimizer = self.optimizer(x0 = np.array([.05, -.9]), function=func)
        optimizer.minimize()
        self.assertAlmostEquals(optimizer.minimum, 3)


if __name__ == '__main__':
    unittest.main()
