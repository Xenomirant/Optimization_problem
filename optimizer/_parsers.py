import sympy as sym
from abc import ABC, abstractmethod
import typing
import functools


class Parser(ABC):

    @abstractmethod
    def parse(self, expression: typing.Any) -> sym.Expr:
        pass

    def __call__(self, expression: typing.Any) -> sym.Expr:
        return self.parse(expression)


class SympyParser(Parser):

    @functools.singledispatchmethod
    def parse(self, expression: typing.Union[sym.Expr, str]):
        return expression

    @parse.register(sym.Expr)
    def _(self, expression: sym.Expr):
        return expression

    @parse.register(str)
    def _(self, expression: str):
        return sym.parse_expr(expression)