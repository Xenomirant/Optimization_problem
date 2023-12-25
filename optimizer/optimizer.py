import numpy as np
import typing
from . import _parsers
from . import _optimizers


class Optimizer:

    def __init__(self, optimizer: typing.Union[typing.Type[_optimizers.AbstractOptimizer], str],
                 parser: _parsers.SympyParser = _parsers.SympyParser()) -> None:

        self.parser = parser

        if not (isinstance(optimizer, type) or isinstance(optimizer, str)):
            raise ValueError("Can't use optimizer. It's set wrong or not implemented")

        self.optimizer_cls = self.__set_optimizer(optimizer)
        self.optimizer = None

    @staticmethod
    def __set_optimizer(optimizer: str) -> typing.Type[_optimizers.AbstractOptimizer]:

        match optimizer:
            case "GradientDescent" | "GD":
                optimizer = _optimizers.GD
            case "Neuton" | "N":
                optimizer = _optimizers.Neuton
            case "ConjugateGradient" | "CG":
                optimizer = _optimizers.ConjugateGradient
            case "ConditionalGradient" | "ConG":
                optimizer = _optimizers.ConditionalGradient
            case "QuadPenalty" | "QP":
                optimizer = _optimizers.QuadPenalty
            case _:
                raise ValueError("Optimizer is not yet implemented")

        return optimizer

    def set_parser(self, parser: _parsers.Parser) -> None:
        self.parser = parser

    def optimize(self, x0: np.array, function: typing.Any, *args, eps: float = 1e-7, max_steps: int = 1000,
                 show: bool = True, **kwargs) -> typing.Optional[typing.List]:

        expression = self.parser(function)

        if "constraints" in kwargs:
            constraints = [self.parser(expr) for expr in kwargs.get("constraints")]
            del kwargs["constraints"]
        else:
            constraints = []

        self.optimizer = self.optimizer_cls(x0=x0, function=expression, *args, eps=eps, max_steps=max_steps,
                                            constraints=constraints, **kwargs)

        self.optimizer()

        if show:
            return self.optimizer.path
        return None

    def show(self, **kwargs) -> None:

        self.optimizer.plot(**kwargs)

        return None
