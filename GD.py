import numpy as np
import sympy
import sympy as sym
from typing import *


class GD:
    """
    Computes minimum of the specified function using Stochastic Gradient descent method
    :param x0: np.arange,
    :param function: sympy.Expr
    :param method: str = ["armijo"]
    :param kwargs:
    """

    def __init__(self, x0: np.array, function: sym.Expr, eps: float = 1e-7, max_steps: int = 1000,
                 method: str = "armijo", **kwargs: object) -> None:

        if not isinstance(function, sympy.Expr):
            raise ValueError("Function must be a valid sympy expression!")

        if x0.size < len(function.free_symbols):
            raise ValueError(f"Function has more dimentions: {len(function.free_symbols)} "
                             f"than specified for x0: {x0.size}")

        self.__get_free_symbols = lambda x: sorted(x.free_symbols, key=lambda s: s.name)

        self.__x0 = x0
        self.x = x0
        self.function = function
        self.vars = self.__get_free_symbols(function)
        self.max_steps = max_steps
        # self._function_lambdified = sym.lambdify(
        #     self._get_free_symbols(function), function, "numpy"
        #)
        self.method = method
        self.eps = eps

        self.gradient = []
        self.steps = []

        for var in self.vars:
            self.gradient.append(-function.diff(var))

        # self._gradient_lambdified = [(sym.lambdify(self._get_free_symbols(expr), expr, "numpy"),
        #                               self._get_free_symbols(expr))
        #                              for expr in self.gradient]

        match method:
            case "armijo":
                self.lr = self.armijo_lr
            case _:
                raise NotImplemented("Method does not exist or is not implemented")

        return

    def lr(self, x, *args, **kwargs):
        pass

    def armijo_lr(self, x, **kwargs):
        """
        Computes lr using armijo's rule
        :param x: np.array - current point in R^n
        :param kwargs: alpha (>0); 0 < eps, delta < 1
        :return: lr: float
        """

        alpha = kwargs.get("alpha", 3e-2)
        eps = kwargs.get("eps", .5)
        delta = kwargs.get("delta", .5)

        def armijo_inequality() -> bool:

            grad = np.array([self.__substitute(diff, x) for diff in self.gradient])

            left = self.__substitute(self.function, x + alpha*grad)
            print(left)
            right = self.__substitute(self.function, x) + eps*alpha*(-grad.T)@grad
            print(right)
            return left <= right

        while not armijo_inequality():
            alpha = delta*alpha

        print(alpha)

        return alpha


    def __substitute(self, function: sympy.Expr, x: np.array) -> np.array:

        return function.subs(
            dict(
                zip(
                    self.vars, x
                )
            )
        )


    def step(self, x: np.array, **kwargs) -> np.array:

        grad = np.array([self.__substitute(diff, x) for diff in self.gradient])

        return x + self.lr(x, **kwargs)*grad

    def minimize(self) -> np.array:
        pass


x = np.array([-1, 1, 2])

x1, x2, x3 = sym.symbols("x1 x2 x3")

print(type(x1))

expr = 2*x1**2 + x1*x2 + 3*x2**2 + x3**3

opt = SGD(x, expr)

print(f'Grad = {[grad.subs(dict(zip(opt.vars, x))) for grad in opt.gradient]}')

opt.armijo_lr(x)

