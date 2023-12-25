import unittest
import sympy as sym
from optimizer._parsers import SympyParser


class SympyParserTest(unittest.TestCase):

    def setUp(self) -> None:
        self.parser = SympyParser()
    def test_string(self):

        string = "x**2 + y**2 + 5"
        parsed_string = self.parser.parse(string)
        self.assertIsInstance(parsed_string, sym.Expr)

    def test_incorrect_string(self):

        incorrect_string = "x!? 32y"
        self.assertRaises((SyntaxError, ValueError, AttributeError), self.parser.parse, incorrect_string)


if __name__ == '__main__':
    unittest.main()
