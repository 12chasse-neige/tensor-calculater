"""Exact, subprocess-bounded reference proof for Kerr Ricci entries and scalar.

This development helper reduces rational functions modulo sin²(theta) +
cos²(theta) - 1.  It performs no numerical substitution or sampling, and applies
where the original metric and expressions are defined.
"""

import json
import sys

import sympy as sp


def exact_trigonometric_zero(expression):
    theta = sp.Symbol("theta")
    sine, cosine = sp.Dummy("sine"), sp.Dummy("cosine")
    expression = expression.xreplace({sp.sin(theta): sine, sp.cos(theta): cosine})
    other_symbols = sorted(expression.free_symbols - {sine, cosine}, key=str)
    generators = (sine, cosine, *other_symbols)
    identity = sp.Poly(sine**2 + cosine**2 - 1, *generators)
    reduced = []
    # Reducing terms separately prevents giant common-denominator expansions.
    # Polynomial remainders are exact congruences under the trigonometric identity.
    for term in sp.Add.make_args(expression):
        numerator, denominator = term.as_numer_denom()
        numerator = sp.Poly(numerator, *generators).rem(identity).as_expr()
        denominator = sp.Poly(denominator, *generators).rem(identity).as_expr()
        if denominator == 0:
            raise ValueError("Reference denominator vanishes under the trigonometric identity")
        reduced.append(sp.cancel(numerator / denominator))
    return sp.cancel(sp.Add(*reduced)) == 0


if __name__ == "__main__":
    expressions = json.load(sys.stdin)
    namespace = {name: sp.Symbol(name) for name in ["t", "r", "theta", "phi", "M", "a"]}
    checks = [
        exact_trigonometric_zero(sp.sympify(item["expression"], locals=namespace))
        for item in expressions
    ]
    print(json.dumps({"all_zero": all(checks), "expression_count": len(checks)}))
