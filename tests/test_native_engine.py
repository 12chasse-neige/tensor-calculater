"""Independent mathematical and protocol checks for the native tensor worker.

SymPy is a development dependency only.  The packaged application does not use
it.  Run with ``.venv/bin/python -m unittest discover -s tests -v`` after building
``build/core/tensor-worker``; TENSOR_WORKER_PATH can select another executable.
"""

from __future__ import annotations

import functools
import itertools
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import unittest

import sympy as sp
from sympy.parsing.sympy_parser import convert_xor, parse_expr, standard_transformations


ROOT = Path(__file__).resolve().parents[1]
WORKER = Path(os.environ.get("TENSOR_WORKER_PATH", ROOT / "build/core/tensor-worker"))
TIMEOUT = float(os.environ.get("TENSOR_TEST_TIMEOUT", "180"))
OUTPUTS = ["inverse_metric", "christoffel", "riemann", "ricci", "ricci_scalar", "kretschmann"]
TENSOR_RANKS = {"inverse_metric": 2, "christoffel": 3, "riemann": 4, "ricci": 2}
TRANSFORMATIONS = standard_transformations + (convert_xor,)


def request(coordinates, metric, *, scalars="", functions="", outputs=None, request_id="reference"):
    return {
        "version": 1,
        "id": request_id,
        "coordinates": coordinates,
        "scalars": scalars,
        "functions": functions,
        "metric": metric,
        "outputs": OUTPUTS if outputs is None else outputs,
    }


def kerr_request(outputs, *, request_id="kerr"):
    return request(
        "t,r,theta,phi",
        "[-(1-2*M*r/(r^2+a^2*cos(theta)^2)),0,0,-2*M*a*r*sin(theta)^2/(r^2+a^2*cos(theta)^2)],"
        "[0,(r^2+a^2*cos(theta)^2)/(r^2-2*M*r+a^2),0,0],"
        "[0,0,r^2+a^2*cos(theta)^2,0],"
        "[-2*M*a*r*sin(theta)^2/(r^2+a^2*cos(theta)^2),0,0,"
        "(r^2+a^2+2*M*a^2*r*sin(theta)^2/(r^2+a^2*cos(theta)^2))*sin(theta)^2]",
        scalars="M,a", outputs=outputs, request_id=request_id,
    )


def run_lines(lines, *, timeout=TIMEOUT):
    if not WORKER.is_file():
        raise AssertionError(f"Build the native worker first: {WORKER}")
    completed = subprocess.run(
        [str(WORKER)], input="\n".join(lines) + "\n", text=True,
        capture_output=True, timeout=timeout, check=False,
    )
    if completed.returncode != 0:
        raise AssertionError(
            f"Worker exited {completed.returncode}:\n{completed.stderr}\n{completed.stdout}"
        )
    try:
        events = [json.loads(line) for line in completed.stdout.splitlines() if line.strip()]
    except json.JSONDecodeError as exc:
        raise AssertionError(f"Worker stdout must contain JSON lines only: {completed.stdout}") from exc
    if not events:
        raise AssertionError(f"Worker returned no events. stderr: {completed.stderr}")
    for event in events:
        if event.get("type") not in {"progress", "result", "error"}:
            raise AssertionError(f"Unexpected event: {event}")
    return events


@functools.lru_cache(maxsize=None)
def cached_result(encoded_request):
    req = json.loads(encoded_request)
    events = run_lines([encoded_request])
    errors = [event for event in events if event["type"] == "error"]
    if errors:
        raise AssertionError(f"Calculation failed: {errors}")
    results = [event for event in events if event["type"] == "result"]
    if len(results) != 1:
        raise AssertionError(f"Expected one result, received {events}")
    for event in events:
        if event.get("id") != req["id"]:
            raise AssertionError(f"Request id was not preserved: {event}")
        if event.get("version") != 1:
            raise AssertionError(f"Wrong response protocol version: {event}")
    return results[0]["result"]


def calculate(req):
    return cached_result(json.dumps(req, sort_keys=True))


def expression_namespace(req):
    names = [name.strip() for name in (req["coordinates"] + "," + req["scalars"]).split(",") if name.strip()]
    namespace = {name: sp.Symbol(name) for name in names}
    # Function declarations can contain commas inside their argument lists.
    import re
    for name in re.findall(r"([A-Za-z_]\w*)\s*\(", req["functions"]):
        namespace[name] = sp.Function(name)
    namespace.update({"Derivative": sp.Derivative, "Subs": sp.Subs, "abs": sp.Abs})
    return namespace


def parse_output(text, req):
    return parse_expr(text, local_dict=expression_namespace(req), transformations=TRANSFORMATIONS)


def tensor(result, key, req):
    selected = next(item for item in result["tensors"] if item["key"] == key)
    return {tuple(item["indices"]): parse_output(item["expression"], req) for item in selected["components"]}


def scalar(result, key, req):
    selected = next(item for item in result["scalars"] if item["key"] == key)
    return parse_output(selected["expression"], req)


def value(components, *indices):
    return components.get(tuple(indices), sp.S.Zero)


def reference_connection(metric, coords):
    """The Levi-Civita definition, without reusing production code."""
    n = len(coords)
    inverse = metric.inv()
    gamma = {}
    for a, b, c in itertools.product(range(n), repeat=3):
        expr = sum(
            inverse[a, d] * (
                sp.diff(metric[d, c], coords[b]) + sp.diff(metric[d, b], coords[c])
                - sp.diff(metric[b, c], coords[d])
            ) / 2 for d in range(n)
        )
        gamma[a, b, c] = sp.simplify(expr)
    return inverse, gamma


class NativeEngineTests(unittest.TestCase):
    maxDiff = None

    def assertEquivalent(self, actual, expected, label=""):
        # Cancel each additive term before combining denominators.  This remains
        # an exact symbolic proof, and avoids a huge common-denominator expansion
        # for native outputs whose rational factors have not been cancelled yet.
        reduced_terms = [sp.cancel(term) for term in sp.Add.make_args(actual - expected)]
        difference = sp.cancel(sp.Add(*reduced_terms))
        if difference != 0:
            difference = sp.simplify(sp.trigsimp(difference))
        self.assertEqual(difference, 0, f"{label}: actual {actual}, expected {expected}")

    def assertTensorZero(self, components):
        for indices, expr in components.items():
            self.assertEquivalent(expr, 0, str(indices))

    def assertSchema(self, result, req):
        n = len(req["coordinates"].split(","))
        self.assertEqual(result["coordinates"], [s.strip() for s in req["coordinates"].split(",")])
        self.assertIsInstance(result["convention"], str)
        self.assertTrue(result["convention"])
        self.assertIsInstance(result["warnings"], list)
        self.assertGreaterEqual(result["elapsed_seconds"], 0)
        self.assertTrue(math.isfinite(result["elapsed_seconds"]))
        returned = set()
        for item in result["tensors"]:
            key = item["key"]
            self.assertNotIn(key, returned)
            returned.add(key)
            rank = TENSOR_RANKS[key]
            self.assertEqual(item["rank"], rank)
            self.assertEqual(item["shape"], [n] * rank)
            for field in ["name", "symbol", "variance"]:
                self.assertTrue(item[field])
            seen = set()
            for component in item["components"]:
                indices = tuple(component["indices"])
                self.assertEqual(len(indices), rank)
                self.assertNotIn(indices, seen)
                seen.add(indices)
                self.assertTrue(all(isinstance(i, int) and 0 <= i < n for i in indices))
                for field in ["expression", "latex", "zero_status"]:
                    self.assertIsInstance(component[field], str)
                    self.assertTrue(component[field])
        for item in result["scalars"]:
            self.assertNotIn(item["key"], returned)
            returned.add(item["key"])
            for field in ["name", "symbol", "expression", "latex"]:
                self.assertTrue(item[field])
        self.assertEqual(returned, set(req["outputs"]), "Return only explicitly selected outputs")

    def test_flat_spherical_coordinates(self):
        req = request("t, r, theta, phi", "diag(-1, 1, r^2, r^2*sin(theta)^2)")
        result = calculate(req)
        self.assertSchema(result, req)
        r, theta = sp.symbols("r theta")
        inverse = tensor(result, "inverse_metric", req)
        for i, expected in enumerate([-1, 1, 1/r**2, 1/(r**2*sp.sin(theta)**2)]):
            self.assertEquivalent(value(inverse, i, i), expected)
        gamma = tensor(result, "christoffel", req)
        self.assertEquivalent(value(gamma, 1, 2, 2), -r)
        self.assertEquivalent(value(gamma, 1, 3, 3), -r*sp.sin(theta)**2)
        self.assertEquivalent(value(gamma, 2, 1, 2), 1/r)
        self.assertEquivalent(value(gamma, 2, 3, 3), -sp.sin(theta)*sp.cos(theta))
        self.assertEquivalent(value(gamma, 3, 2, 3), sp.cot(theta))
        self.assertTensorZero(tensor(result, "riemann", req))
        self.assertTensorZero(tensor(result, "ricci", req))
        self.assertEquivalent(scalar(result, "ricci_scalar", req), 0)
        self.assertEquivalent(scalar(result, "kretschmann", req), 0)

    def sphere(self):
        req = request("theta, phi", "[[L^2,0],[0,L^2*sin(theta)^2]]", scalars="L")
        return req, calculate(req)

    def test_sphere_curvature_and_preserved_sign(self):
        req, result = self.sphere()
        self.assertSchema(result, req)
        L, theta = sp.symbols("L theta")
        ricci = tensor(result, "ricci", req)
        self.assertEquivalent(value(ricci, 0, 0), -1)
        self.assertEquivalent(value(ricci, 1, 1), -sp.sin(theta)**2)
        self.assertEquivalent(scalar(result, "ricci_scalar", req), -2/L**2)
        self.assertEquivalent(scalar(result, "kretschmann", req), 4/L**4)
        self.assertEquivalent(value(tensor(result, "riemann", req), 0, 1, 0, 1), -sp.sin(theta)**2)

    def test_negated_kerr_retains_closed_form_invariant(self):
        req = kerr_request(["kretschmann"])
        baseline = calculate(req)
        rows = req["metric"].replace("\n", "").split("],[")
        req["metric"] = ",".join("[" + ",".join("-(" + entry + ")" for entry in row.strip("[]").split(",")) + "]" for row in rows)
        req["riemann_sign"] = -1
        result = calculate(req)
        self.assertEqual(result["scalars"], baseline["scalars"])
        self.assertTrue(any("recognized closed form" in warning for warning in result["warnings"]))

    def test_selectable_curvature_and_metric_signs(self):
        L = sp.Symbol("L")
        for metric_sign in (1, -1):
            for curvature_sign in (1, -1):
                req = request("theta, phi", f"[{metric_sign}*L^2,0],[0,{metric_sign}*L^2*sin(theta)^2]", scalars="L")
                req["riemann_sign"] = curvature_sign
                result = calculate(req)
                self.assertEquivalent(scalar(result, "ricci_scalar", req), -2*metric_sign*curvature_sign/L**2)
                self.assertEquivalent(scalar(result, "kretschmann", req), 4/L**4)
                self.assertEquivalent(value(tensor(result, "ricci", req), 0, 0), -curvature_sign)
                direct = dict(req, outputs=["ricci_scalar"])
                self.assertEquivalent(scalar(calculate(direct), "ricci_scalar", direct), -2*metric_sign*curvature_sign/L**2)
        req["riemann_sign"] = 0
        self.assertTrue(any(event["type"] == "error" for event in run_lines([json.dumps(req)])))

    def test_connection_metric_compatibility_and_tensor_identities(self):
        req, result = self.sphere()
        L, theta, phi = sp.symbols("L theta phi")
        coords = (theta, phi)
        metric = sp.diag(L**2, L**2*sp.sin(theta)**2)
        inverse, expected_gamma = reference_connection(metric, coords)
        actual_inverse = tensor(result, "inverse_metric", req)
        gamma = tensor(result, "christoffel", req)
        R = tensor(result, "riemann", req)
        ricci = tensor(result, "ricci", req)
        for a, b in itertools.product(range(2), repeat=2):
            self.assertEquivalent(value(actual_inverse, a, b), inverse[a, b])
            self.assertEquivalent(value(ricci, a, b), sum(value(R, c, a, c, b) for c in range(2)))
        for a, b, c in itertools.product(range(2), repeat=3):
            self.assertEquivalent(value(gamma, a, b, c), expected_gamma[a, b, c])
            self.assertEquivalent(value(gamma, a, b, c), value(gamma, a, c, b))
            covariant_derivative = sp.diff(metric[a, b], coords[c]) - sum(
                value(gamma, d, c, a)*metric[d, b] + value(gamma, d, c, b)*metric[a, d]
                for d in range(2)
            )
            self.assertEquivalent(covariant_derivative, 0, "metric compatibility")
        def lower(a, b, c, d):
            return sum(metric[a, e]*value(R, e, b, c, d) for e in range(2))
        for a, b, c, d in itertools.product(range(2), repeat=4):
            self.assertEquivalent(lower(a,b,c,d), -lower(b,a,c,d), "first pair antisymmetry")
            self.assertEquivalent(lower(a,b,c,d), -lower(a,b,d,c), "last pair antisymmetry")
            self.assertEquivalent(lower(a,b,c,d), lower(c,d,a,b), "pair exchange")
            self.assertEquivalent(value(R,a,b,c,d)+value(R,a,c,d,b)+value(R,a,d,b,c), 0, "Bianchi identity")

    def test_off_diagonal_flat_coordinate_transform(self):
        # X=u, Y=v+u² transforms dX²+dY² to this non-diagonal metric.
        req = request("u, v", "Matrix([[1+4*u^2,2*u],[2*u,1]])")
        result = calculate(req)
        self.assertSchema(result, req)
        u = sp.Symbol("u")
        inverse = tensor(result, "inverse_metric", req)
        expected_inverse = sp.Matrix([[1,-2*u],[-2*u,1+4*u**2]])
        for i,j in itertools.product(range(2), repeat=2):
            self.assertEquivalent(value(inverse,i,j), expected_inverse[i,j])
        gamma = tensor(result, "christoffel", req)
        for a,b,c in itertools.product(range(2), repeat=3):
            self.assertEquivalent(value(gamma,a,b,c), 2 if (a,b,c)==(1,0,0) else 0)
        self.assertTensorZero(tensor(result, "riemann", req))
        self.assertTensorZero(tensor(result, "ricci", req))
        self.assertEquivalent(scalar(result, "ricci_scalar", req), 0)
        self.assertEquivalent(scalar(result, "kretschmann", req), 0)

    def test_off_diagonal_curved_metric_contraction(self):
        # theta=u, phi=v+u²: invariants of a sphere survive the coordinate change.
        req = request("u, v", "[[L^2*(1+4*u^2*sin(u)^2),2*u*L^2*sin(u)^2],[2*u*L^2*sin(u)^2,L^2*sin(u)^2]]", scalars="L")
        result = calculate(req)
        L = sp.Symbol("L")
        self.assertEquivalent(scalar(result, "ricci_scalar", req), -2/L**2)
        self.assertEquivalent(scalar(result, "kretschmann", req), 4/L**4)

    def test_schwarzschild_vacuum_and_kretschmann(self):
        req = request("t, r, theta, phi", "[-(1-2M/r),0,0,0],\n[0,1/(1-2M/r),0,0],\n[0,0,r^2,0],\n[0,0,0,r^2*sin(theta)^2]", scalars="M")
        result = calculate(req)
        self.assertSchema(result, req)
        M,r = sp.symbols("M r")
        self.assertTensorZero(tensor(result, "ricci", req))
        self.assertEquivalent(scalar(result, "ricci_scalar", req), 0)
        self.assertEquivalent(scalar(result, "kretschmann", req), 48*M**2/r**6)
        self.assertTrue(tensor(result, "riemann", req), "Vacuum still has nonzero tidal curvature")

    def test_reissner_nordstrom_invariants(self):
        req = request("t, r, theta, phi", "diag(-(1-2*M/r+Q^2/r^2),1/(1-2*M/r+Q^2/r^2),r^2,r^2*sin(theta)^2)", scalars="M,Q", outputs=["ricci", "ricci_scalar", "kretschmann"])
        result = calculate(req)
        self.assertSchema(result, req)
        M,Q,r = sp.symbols("M Q r")
        self.assertEquivalent(scalar(result, "ricci_scalar", req), 0)
        self.assertEquivalent(scalar(result, "kretschmann", req), 48*M**2/r**6-96*M*Q**2/r**7+56*Q**4/r**8)
        self.assertTrue(tensor(result, "ricci", req), "Electromagnetic stress produces nonzero Ricci components")

    def test_kerr_closed_form_invariant_is_bounded_and_reported(self):
        req = kerr_request(["kretschmann"], request_id="kerr-invariant")
        events = run_lines([json.dumps(req)], timeout=10)
        self.assertFalse(any(event["type"] == "error" for event in events), events)
        results = [event for event in events if event["type"] == "result"]
        self.assertEqual(len(results), 1, events)
        result = results[0]["result"]
        self.assertSchema(result, req)
        M,a,r,theta = sp.symbols("M a r theta")
        c = sp.cos(theta)
        expected = 48*M**2*(r**6-15*a**2*r**4*c**2+15*a**4*r**2*c**4-a**6*c**6)/(r**2+a**2*c**2)**6
        actual = scalar(result, "kretschmann", req)
        self.assertEquivalent(actual, expected)
        self.assertEquivalent(actual.subs(a,0), 48*M**2/r**6)
        self.assertTrue(any("closed form" in warning.lower() for warning in result["warnings"]))
        self.assertFalse(any(event.get("stage") == "riemann" for event in events),
                         "Recognized Kerr scalar should not compute the full Riemann tensor")

    def test_kerr_general_curvature_path_is_exact_vacuum(self):
        req = kerr_request(["riemann", "ricci", "ricci_scalar"], request_id="kerr-vacuum")
        events = run_lines([json.dumps(req)], timeout=10)
        self.assertFalse(any(event["type"] == "error" for event in events), events)
        results = [event for event in events if event["type"] == "result"]
        self.assertEqual(len(results), 1, events)
        result = results[0]["result"]
        self.assertSchema(result, req)
        self.assertTrue(any(event.get("stage") == "christoffel" for event in events))
        self.assertTrue(any(event.get("stage") == "riemann" for event in events))
        self.assertTrue(any(event.get("stage") == "ricci" for event in events))
        # Conservatively stored unresolved components are allowed.  Independently
        # prove each and the scalar are zero in a bounded reference process.
        ricci = next(item for item in result["tensors"] if item["key"] == "ricci")
        scalar_output = next(item for item in result["scalars"] if item["key"] == "ricci_scalar")
        expressions = ricci["components"] + [scalar_output]
        proof = subprocess.run(
            [sys.executable, str(ROOT / "tests/kerr_vacuum_proof.py")],
            input=json.dumps(expressions), text=True, capture_output=True,
            timeout=10, check=False,
        )
        self.assertEqual(proof.returncode, 0, proof.stderr)
        validation = json.loads(proof.stdout)
        self.assertTrue(validation["all_zero"], validation)
        self.assertEqual(validation["expression_count"], len(expressions))

    def test_flrw_arbitrary_function_and_repeated_derivatives(self):
        req = request("t,r,theta,phi", "diag(-1,a(t)^2/(1-k*r^2),a(t)^2*r^2,a(t)^2*r^2*sin(theta)^2)", scalars="k", functions="a(t)", outputs=["ricci_scalar"])
        result = calculate(req)
        self.assertSchema(result, req)
        t,k = sp.symbols("t k")
        a = sp.Function("a")(t)
        expected = -6*(sp.diff(a,t,2)/a + sp.diff(a,t)**2/a**2 + k/a**2)
        self.assertEquivalent(scalar(result, "ricci_scalar", req), expected)

    def test_multivariate_unknown_functions(self):
        req = request("t,r", "diag(-exp(2*Phi(t,r)),exp(2*Lambda(t,r)))", functions="Phi(t,r),Lambda(t,r)", outputs=["inverse_metric", "christoffel"])
        result = calculate(req)
        self.assertSchema(result, req)
        t,r = sp.symbols("t r")
        Phi,Lambda = sp.Function("Phi")(t,r),sp.Function("Lambda")(t,r)
        metric = sp.diag(-sp.exp(2*Phi),sp.exp(2*Lambda))
        inverse, gamma = reference_connection(metric, (t,r))
        actual_inverse = tensor(result, "inverse_metric", req)
        actual_gamma = tensor(result, "christoffel", req)
        for i,j in itertools.product(range(2), repeat=2):
            self.assertEquivalent(value(actual_inverse,i,j), inverse[i,j])
        for indices, expected in gamma.items():
            self.assertEquivalent(value(actual_gamma,*indices), expected, str(indices))

    def test_nonzero_curvature_at_old_zero_samples_is_retained(self):
        x = sp.Symbol("x")
        sampled_roots = [sp.Rational(2,3),sp.Rational(5,7),sp.Rational(11,13),sp.Rational(17,19)]
        second_derivative = sp.prod(x-root for root in sampled_roots)
        f = 1 + sp.integrate(sp.integrate(second_derivative,x),x)
        req = request("x,y", f"diag(1,({f})^2)", outputs=["ricci"])
        result = calculate(req)
        actual = value(tensor(result, "ricci", req),0,0)
        expected = second_derivative/f
        self.assertEquivalent(actual,expected)
        for sample in sampled_roots:
            self.assertEquivalent(expected.subs(x,sample),0)
        self.assertNotEqual(actual.subs(x,0),0, "Sampling zeros must not erase exact nonzero expressions")

    def test_tiny_exact_curvature_is_retained(self):
        req = request("x,y", "diag(1,(1+x^2/10^30)^2)", outputs=["ricci", "ricci_scalar"])
        result = calculate(req)
        x = sp.Symbol("x")
        epsilon = sp.Rational(1,10**30)
        self.assertEquivalent(value(tensor(result,"ricci",req),0,0),2*epsilon/(1+epsilon*x**2))
        self.assertEquivalent(scalar(result,"ricci_scalar",req),4*epsilon/(1+epsilon*x**2))

    def test_selected_outputs_do_not_leak_dependencies(self):
        for selected in OUTPUTS:
            with self.subTest(output=selected):
                req = request("x,y", "diag(1,x^2)", outputs=[selected], request_id=selected)
                self.assertSchema(calculate(req),req)

    def test_invalid_input_returns_structured_errors(self):
        valid = request("x,y", "diag(1,1)", outputs=["inverse_metric"])
        variants = {
            "empty coordinates": {"coordinates":""},
            "duplicate coordinates": {"coordinates":"x,x"},
            "invalid coordinate": {"coordinates":"x,2y"},
            "wrong dimension": {"metric":"diag(1,1,1)"},
            "nonsymmetric": {"metric":"[[1,x],[0,1]]"},
            "singular": {"metric":"[[1,1],[1,1]]"},
            "unknown symbol": {"metric":"diag(1,z)"},
            "malformed rows": {"metric":"[[1,0],[0,1]"},
            "unknown output": {"outputs":["unrecognized_tensor"]},
            "unsupported version": {"version":2},
            "coordinate scalar conflict": {"scalars":"x"},
            "function coordinate conflict": {"functions":"x(y)"},
            "undeclared function": {"metric":"diag(1,a(x))"},
            "function wrong argument": {"functions":"a(x)","metric":"diag(1,a(y))"},
        }
        for label, change in variants.items():
            with self.subTest(input=label):
                req = valid | change | {"id":label}
                events = run_lines([json.dumps(req)])
                errors = [event for event in events if event["type"]=="error"]
                self.assertEqual(len(errors),1,events)
                self.assertFalse(any(event["type"]=="result" for event in events))
                self.assertEqual(errors[0].get("id"),label)
                message = errors[0].get("message") or errors[0].get("error",{}).get("message")
                self.assertIsInstance(message,str)
                self.assertTrue(message)

    def test_worker_recovers_after_bad_request_and_preserves_request_ids(self):
        first = request("x,y","diag(1,1)",outputs=["inverse_metric"],request_id="first")
        second = request("x,y","diag(1,x^2)",outputs=["ricci_scalar"],request_id="second")
        events = run_lines(["{malformed-json",json.dumps(first),json.dumps(second)])
        self.assertTrue(any(event["type"]=="error" for event in events))
        results = [event for event in events if event["type"]=="result"]
        self.assertEqual([event["id"] for event in results],["first","second"])
        self.assertEquivalent(scalar(results[1]["result"],"ricci_scalar",second),0)

    def test_bounded_rejection_of_extreme_expressions_and_infinities(self):
        invalid_expressions = {
            "negative exponent int minimum": "1e-2147483648",
            "positive exponent int maximum": "1e2147483647",
            "excessive parenthesis nesting": "("*129 + "1" + ")"*129,
            "excessive recursive power nesting": "1^"*129 + "1",
            "numeric power expansion size": "(((2^1000)^1000)^1000)^1000",
            "logarithm of zero": "log(0)",
            "square root of infinity": "sqrt(log(0))",
            "square root of division by zero": "sqrt(1/0)",
        }
        for label, expression in invalid_expressions.items():
            with self.subTest(expression=label):
                req = request("x,y",f"diag(1,{expression})",outputs=["inverse_metric"],request_id=label)
                events = run_lines([json.dumps(req)], timeout=10)
                self.assertFalse(any(event["type"] == "result" for event in events), events)
                errors = [event for event in events if event["type"] == "error"]
                self.assertEqual(len(errors), 1, events)
                self.assertEqual(errors[0].get("id"), label)
                self.assertIsInstance(errors[0].get("message"), str)
                self.assertTrue(errors[0]["message"])


if __name__ == "__main__":
    unittest.main()
