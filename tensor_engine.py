from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Callable

import sympy as sp
from sympy.printing.latex import LatexPrinter
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)


IDENTIFIER_RE = re.compile(r"^[A-Za-z_]\w*$")
RESERVED_NAMES = {
    "Matrix",
    "diag",
    "diff",
    "Derivative",
    "Rational",
    "S",
    "I",
    "E",
    "pi",
}
PARSER_TRANSFORMATIONS = tuple(
    transform
    for transform in standard_transformations
    if transform.__name__ != "auto_symbol"
) + (implicit_multiplication_application, convert_xor)
SAFE_GLOBALS = {
    "__builtins__": {},
    "Integer": sp.Integer,
    "Float": sp.Float,
    "Rational": sp.Rational,
    "Add": sp.Add,
    "Mul": sp.Mul,
    "Pow": sp.Pow,
    "factorial": sp.factorial,
}


EXAMPLES = {
    "平直时空（球坐标）": {
        "coords": "t, r, theta, phi",
        "scalars": "",
        "functions": "",
        "metric": (
            "[-1, 0, 0, 0],\n"
            "[0, 1, 0, 0],\n"
            "[0, 0, r^2, 0],\n"
            "[0, 0, 0, r^2*sin(theta)^2]"
        ),
    },
    "Schwarzschild（M）": {
        "coords": "t, r, theta, phi",
        "scalars": "M",
        "functions": "",
        "metric": (
            "[-(1 - 2M/r), 0, 0, 0],\n"
            "[0, 1/(1 - 2M/r), 0, 0],\n"
            "[0, 0, r^2, 0],\n"
            "[0, 0, 0, r^2*sin(theta)^2]"
        ),
    },
    "Reissner-Nordstrom（M, Q）": {
        "coords": "t, r, theta, phi",
        "scalars": "M, Q",
        "functions": "",
        "metric": (
            "[-(1 - 2*M/r + Q^2/r^2), 0, 0, 0],\n"
            "[0, 1/(1 - 2*M/r + Q^2/r^2), 0, 0],\n"
            "[0, 0, r^2, 0],\n"
            "[0, 0, 0, r^2*sin(theta)^2]"
        ),
    },
    "Kerr（M, a）": {
        "coords": "t, r, theta, phi",
        "scalars": "M, a",
        "functions": "",
        "metric": (
            "[-(1 - 2*M*r/(r^2 + a^2*cos(theta)^2)), 0, 0,"
            " -2*M*a*r*sin(theta)^2/(r^2 + a^2*cos(theta)^2)],\n"
            "[0, (r^2 + a^2*cos(theta)^2)/(r^2 - 2*M*r + a^2), 0, 0],\n"
            "[0, 0, r^2 + a^2*cos(theta)^2, 0],\n"
            "[-2*M*a*r*sin(theta)^2/(r^2 + a^2*cos(theta)^2), 0, 0,"
            " (r^2 + a^2 + 2*M*a^2*r*sin(theta)^2/(r^2 + a^2*cos(theta)^2))*sin(theta)^2]"
        ),
    },
    "FLRW（a(t), k）": {
        "coords": "t, r, theta, phi",
        "scalars": "k",
        "functions": "a(t)",
        "metric": (
            "[-1, 0, 0, 0],\n"
            "[0, a(t)^2/(1 - k*r^2), 0, 0],\n"
            "[0, 0, a(t)^2*r^2, 0],\n"
            "[0, 0, 0, a(t)^2*r^2*sin(theta)^2]"
        ),
    },
}


ProgressCallback = Callable[[str], None] | None


@dataclass(frozen=True)
class ParsedInputs:
    coords: tuple[sp.Symbol, ...]
    scalar_symbols: tuple[sp.Symbol, ...]
    function_defs: tuple[tuple[str, tuple[str, ...]], ...]
    metric: sp.Matrix


@dataclass(frozen=True)
class TensorComponents:
    name: str
    symbol: str
    shape: tuple[int, ...]
    components: dict[tuple[int, ...], sp.Expr]
    variance: str


@dataclass(frozen=True)
class GeometryResult:
    parsed: ParsedInputs
    inverse_metric: sp.Matrix
    christoffel: TensorComponents
    riemann: TensorComponents
    ricci: TensorComponents
    ricci_scalar: sp.Expr
    kretschmann: sp.Expr | None
    elapsed_seconds: float


def split_top_level_csv(text: str) -> list[str]:
    """Split comma-separated input while respecting brackets and parentheses."""
    items: list[str] = []
    start = 0
    stack: list[str] = []
    opening = {"(": ")", "[": "]", "{": "}"}
    closing = {")": "(", "]": "[", "}": "{"}

    for index, char in enumerate(text):
        if char in opening:
            stack.append(char)
        elif char in closing:
            if not stack or stack[-1] != closing[char]:
                raise ValueError("括号不匹配，请检查输入。")
            stack.pop()
        elif char == "," and not stack:
            item = text[start:index].strip()
            if item:
                items.append(item)
            start = index + 1

    if stack:
        raise ValueError("括号不匹配，请检查输入。")

    item = text[start:].strip()
    if item:
        items.append(item)
    return items


def validate_identifier(name: str, label: str) -> None:
    if not IDENTIFIER_RE.match(name):
        raise ValueError(f"{label} '{name}' 不是合法的 Python/SymPy 名称。")
    if name in RESERVED_NAMES:
        raise ValueError(f"{label} '{name}' 是保留名称，请换一个名字。")


def parse_coordinate_symbols(coord_str: str) -> tuple[sp.Symbol, ...]:
    coord_names = split_top_level_csv(coord_str.strip())
    if not coord_names:
        raise ValueError("请输入至少一个坐标符号。")

    seen: set[str] = set()
    for name in coord_names:
        validate_identifier(name, "坐标")
        if name in seen:
            raise ValueError(f"坐标 '{name}' 重复。")
        seen.add(name)

    return tuple(sp.symbols(coord_names))


def _add_math_namespace(local_dict: dict[str, object]) -> None:
    math_names = [
        "sin",
        "cos",
        "tan",
        "cot",
        "sec",
        "csc",
        "asin",
        "acos",
        "atan",
        "sinh",
        "cosh",
        "tanh",
        "exp",
        "sqrt",
        "log",
        "ln",
        "Abs",
        "Rational",
        "Integer",
        "Float",
        "S",
        "pi",
        "E",
        "I",
        "diff",
        "Derivative",
        "Matrix",
        "diag",
        "zeros",
        "ones",
        "eye",
    ]
    for name in math_names:
        if name == "ln":
            local_dict[name] = sp.log
        elif hasattr(sp, name):
            local_dict[name] = getattr(sp, name)


def build_namespace(
    coords: tuple[sp.Symbol, ...], scalar_str: str, func_str: str
) -> tuple[dict[str, object], tuple[sp.Symbol, ...], tuple[tuple[str, tuple[str, ...]], ...]]:
    local_dict: dict[str, object] = {str(coord): coord for coord in coords}
    _add_math_namespace(local_dict)

    scalar_symbols: list[sp.Symbol] = []
    scalar_names = split_top_level_csv(scalar_str.strip()) if scalar_str.strip() else []
    for name in scalar_names:
        validate_identifier(name, "标量常量")
        if name in local_dict:
            raise ValueError(f"标量常量 '{name}' 与已有坐标或函数名冲突。")
        symbol = sp.Symbol(name)
        local_dict[name] = symbol
        scalar_symbols.append(symbol)

    function_defs: list[tuple[str, tuple[str, ...]]] = []
    function_items = split_top_level_csv(func_str.strip()) if func_str.strip() else []
    coord_name_set = {str(coord) for coord in coords}
    for item in function_items:
        match = re.match(r"^([A-Za-z_]\w*)\s*\((.*)\)$", item)
        if not match:
            raise ValueError(f"无法解析自定义函数 '{item}'。格式应为: 函数名(变量名)")

        func_name = match.group(1)
        validate_identifier(func_name, "自定义函数")
        if func_name in local_dict:
            raise ValueError(f"自定义函数 '{func_name}' 与已有坐标、常量或函数名冲突。")

        arg_names = split_top_level_csv(match.group(2))
        if not arg_names:
            raise ValueError(f"自定义函数 '{item}' 至少需要一个坐标参数。")

        for arg in arg_names:
            if arg not in coord_name_set:
                raise ValueError(
                    f"自定义函数 '{item}' 的参数 '{arg}' 不在坐标列表中。"
                )

        local_dict[func_name] = sp.Function(func_name)
        function_defs.append((func_name, tuple(arg_names)))

    return local_dict, tuple(scalar_symbols), tuple(function_defs)


def parse_metric_expression(metric_str: str, local_dict: dict[str, object]) -> sp.Matrix:
    metric_str = normalize_metric_input(metric_str)
    if not metric_str:
        raise ValueError("度规矩阵不能为空。")

    try:
        metric_obj = parse_expr(
            metric_str,
            local_dict=local_dict,
            global_dict=SAFE_GLOBALS,
            transformations=PARSER_TRANSFORMATIONS,
            evaluate=True,
        )
    except NameError as exc:
        missing = re.search(r"name '([^']+)' is not defined", str(exc))
        if missing:
            name = missing.group(1)
            raise ValueError(
                f"表达式中的 '{name}' 未定义。若它是常量，请加入“标量常量”；"
                "若它是未知函数，请加入“自定义函数”。"
            ) from exc
        raise ValueError(f"无法解析度规矩阵: {exc}") from exc
    except Exception as exc:
        raise ValueError(f"无法解析度规矩阵: {exc}") from exc

    if isinstance(metric_obj, sp.MatrixBase):
        metric_mat = sp.Matrix(metric_obj)
    elif isinstance(metric_obj, (list, tuple)):
        metric_mat = sp.Matrix(metric_obj)
    else:
        raise ValueError("度规矩阵必须是二维方阵，例如 [[g00, g01], [g10, g11]]。")

    return metric_mat


def _looks_like_metric_row(text: str) -> bool:
    stripped = text.strip()
    return stripped.startswith("[") and stripped.endswith("]")


def normalize_metric_input(metric_str: str) -> str:
    """Accept either [[...], [...]] or the UI-friendly row-only matrix form."""
    text = metric_str.strip()
    if not text:
        return ""

    try:
        top_level_items = split_top_level_csv(text)
    except ValueError:
        return text

    if len(top_level_items) > 1 and all(
        _looks_like_metric_row(item) for item in top_level_items
    ):
        return "[" + ",\n".join(item.strip() for item in top_level_items) + "]"

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    row_lines = [line[:-1].rstrip() if line.endswith(",") else line for line in lines]
    is_old_wrapped_matrix = text.startswith("[[") and text.endswith("]]")
    if (
        len(row_lines) > 1
        and not is_old_wrapped_matrix
        and all(_looks_like_metric_row(line) for line in row_lines)
    ):
        return "[" + ",\n".join(row_lines) + "]"

    return text


def parse_gr_inputs(
    coord_str: str, scalar_str: str, func_str: str, metric_str: str
) -> ParsedInputs:
    coords = parse_coordinate_symbols(coord_str)
    local_dict, scalar_symbols, function_defs = build_namespace(
        coords, scalar_str, func_str
    )
    metric = parse_metric_expression(metric_str, local_dict)

    if metric.rows != metric.cols:
        raise ValueError("度规矩阵必须是方阵。")
    if metric.rows != len(coords):
        raise ValueError(
            f"坐标数量({len(coords)})与度规矩阵维度({metric.rows})不匹配。"
        )

    return ParsedInputs(coords, scalar_symbols, function_defs, metric)


def prefer_simpler(original: sp.Expr, candidate: sp.Expr) -> sp.Expr:
    if candidate == 0:
        return sp.S.Zero
    try:
        if sp.count_ops(candidate) <= sp.count_ops(original) * 1.15 + 4:
            return candidate
    except Exception:
        return candidate
    return original


def fast_simplify(expr: sp.Expr, *, deep: bool = False) -> sp.Expr:
    """Keep simplification predictable; full simplify is reserved for scalars."""
    expr = sp.sympify(expr)
    if expr == 0:
        return sp.S.Zero

    for simplifier in (sp.cancel, sp.factor_terms):
        try:
            candidate = simplifier(expr)
            expr = prefer_simpler(expr, candidate)
        except Exception:
            pass

    try:
        candidate = sp.trigsimp(expr)
        expr = prefer_simpler(expr, candidate)
    except Exception:
        pass

    try:
        if sp.count_ops(expr) <= 120 and expr.has(
            sp.sin, sp.cos, sp.tan, sp.cot, sp.sec, sp.csc
        ):
            candidate = sp.trigsimp(expr, method="fu")
            expr = prefer_simpler(expr, candidate)
    except Exception:
        pass

    if deep:
        try:
            candidate = sp.simplify(expr)
            expr = prefer_simpler(expr, candidate)
        except Exception:
            pass

    return sp.S.Zero if expr == 0 else expr


def _is_time_symbol(symbol: sp.Basic) -> bool:
    return isinstance(symbol, sp.Symbol) and str(symbol) in {"t", "tau"}


class DotDerivativeLatexPrinter(LatexPrinter):
    def _print_Derivative(self, expr: sp.Derivative) -> str:
        if len(expr.variable_count) == 1:
            variable, order = expr.variable_count[0]
            if _is_time_symbol(variable) and order in {1, 2}:
                return self._print_time_derivative(expr.expr, order)
        return super()._print_Derivative(expr)

    def _print_time_derivative(self, expr: sp.Expr, order: int) -> str:
        if getattr(expr, "is_Function", False):
            name = self._print(sp.Symbol(expr.func.__name__))
            accent = r"\dot" if order == 1 else r"\ddot"
            dotted_name = rf"{accent}{{{name}}}"
            args = ",".join(self._print(arg) for arg in expr.args)
            return rf"{dotted_name}{{\left({args}\right)}}"

        printed = self._print(expr)
        if order == 1:
            return rf"\dot{{{printed}}}"
        return rf"\ddot{{{printed}}}"


DOT_DERIVATIVE_LATEX_PRINTER = DotDerivativeLatexPrinter()


def latex_expression(expr: sp.Expr) -> str:
    try:
        return DOT_DERIVATIVE_LATEX_PRINTER.doprint(fast_simplify(expr, deep=False))
    except Exception:
        return DOT_DERIVATIVE_LATEX_PRINTER.doprint(expr)


class TensorCalculator:
    def __init__(self, parsed: ParsedInputs, progress: ProgressCallback = None):
        self.parsed = parsed
        self.coords = parsed.coords
        self.metric = parsed.metric
        self.n = len(self.coords)
        self.progress = progress or (lambda _message: None)
        self._metric_derivative_cache: dict[tuple[int, int, int], sp.Expr] = {}
        self._expr_derivative_cache: dict[tuple[sp.Expr, int], sp.Expr] = {}

    def _set_progress(self, message: str) -> None:
        self.progress(message)

    def _metric_derivative(self, i: int, j: int, coord_index: int) -> sp.Expr:
        key = (i, j, coord_index)
        if key not in self._metric_derivative_cache:
            value = self.metric[i, j]
            if value == 0:
                derivative = sp.S.Zero
            else:
                derivative = fast_simplify(sp.diff(value, self.coords[coord_index]))
            self._metric_derivative_cache[key] = derivative
        return self._metric_derivative_cache[key]

    def _derivative(self, expr: sp.Expr, coord_index: int) -> sp.Expr:
        if expr == 0:
            return sp.S.Zero
        key = (expr, coord_index)
        if key not in self._expr_derivative_cache:
            self._expr_derivative_cache[key] = fast_simplify(
                sp.diff(expr, self.coords[coord_index])
            )
        return self._expr_derivative_cache[key]

    @staticmethod
    def _matrix_nonzero_by_row(matrix: sp.Matrix) -> list[list[tuple[int, sp.Expr]]]:
        entries: list[list[tuple[int, sp.Expr]]] = []
        for i in range(matrix.rows):
            row: list[tuple[int, sp.Expr]] = []
            for j in range(matrix.cols):
                value = fast_simplify(matrix[i, j])
                if value != 0:
                    row.append((j, value))
            entries.append(row)
        return entries

    @staticmethod
    def _store_component(
        components: dict[tuple[int, ...], sp.Expr],
        key: tuple[int, ...],
        value: sp.Expr,
        *,
        deep: bool = False,
    ) -> None:
        simplified = fast_simplify(value, deep=deep)
        if simplified != 0:
            components[key] = simplified

    def _inverse_metric(self) -> sp.Matrix:
        try:
            inverse = self.metric.inv()
        except Exception as exc:
            raise ValueError("无法构造逆度规；请检查度规是否非奇异。") from exc
        return inverse.applyfunc(lambda value: fast_simplify(value, deep=False))

    def _christoffel(self, inverse_metric: sp.Matrix) -> TensorComponents:
        components: dict[tuple[int, int, int], sp.Expr] = {}
        inverse_rows = self._matrix_nonzero_by_row(inverse_metric)

        for rho in range(self.n):
            for mu in range(self.n):
                for nu in range(mu, self.n):
                    total = sp.S.Zero
                    for sigma, inverse_value in inverse_rows[rho]:
                        inner = (
                            self._metric_derivative(sigma, nu, mu)
                            + self._metric_derivative(sigma, mu, nu)
                            - self._metric_derivative(mu, nu, sigma)
                        )
                        if inner != 0:
                            total += inverse_value * inner
                    value = total / 2
                    simplified = fast_simplify(value)
                    if simplified != 0:
                        components[(rho, mu, nu)] = simplified
                        if mu != nu:
                            components[(rho, nu, mu)] = simplified

        return TensorComponents(
            name="克里斯托费尔联络",
            symbol="Gamma",
            shape=(self.n, self.n, self.n),
            components=components,
            variance="ull",
        )

    def _riemann(self, christoffel: TensorComponents) -> TensorComponents:
        gamma = christoffel.components
        components: dict[tuple[int, int, int, int], sp.Expr] = {}

        def gamma_value(a: int, b: int, c: int) -> sp.Expr:
            return gamma.get((a, b, c), sp.S.Zero)

        for rho in range(self.n):
            for sigma in range(self.n):
                for mu in range(self.n):
                    for nu in range(mu + 1, self.n):
                        total = self._derivative(
                            gamma_value(rho, nu, sigma), mu
                        ) - self._derivative(gamma_value(rho, mu, sigma), nu)

                        for alpha in range(self.n):
                            left = gamma_value(rho, mu, alpha) * gamma_value(
                                alpha, nu, sigma
                            )
                            right = gamma_value(rho, nu, alpha) * gamma_value(
                                alpha, mu, sigma
                            )
                            if left != 0 or right != 0:
                                total += left - right

                        value = fast_simplify(total)
                        if value != 0:
                            components[(rho, sigma, mu, nu)] = value
                            components[(rho, sigma, nu, mu)] = -value

        return TensorComponents(
            name="黎曼曲率张量",
            symbol="Riemann",
            shape=(self.n, self.n, self.n, self.n),
            components=components,
            variance="ulll",
        )

    def _ricci(self, riemann: TensorComponents) -> TensorComponents:
        components: dict[tuple[int, int], sp.Expr] = {}
        r = riemann.components
        for sigma in range(self.n):
            for nu in range(self.n):
                total = sp.S.Zero
                for rho in range(self.n):
                    total += r.get((rho, sigma, rho, nu), sp.S.Zero)
                self._store_component(components, (sigma, nu), total, deep=True)

        return TensorComponents(
            name="里奇张量",
            symbol="Ricci",
            shape=(self.n, self.n),
            components=components,
            variance="ll",
        )

    def _ricci_scalar(
        self, inverse_metric: sp.Matrix, ricci: TensorComponents
    ) -> sp.Expr:
        inverse_rows = self._matrix_nonzero_by_row(inverse_metric)
        total = sp.S.Zero
        for mu, row in enumerate(inverse_rows):
            for nu, inverse_value in row:
                ricci_value = ricci.components.get((mu, nu), sp.S.Zero)
                if ricci_value != 0:
                    total += inverse_value * ricci_value
        return fast_simplify(total, deep=True)

    def _lower_riemann(self, riemann: TensorComponents) -> dict[tuple[int, ...], sp.Expr]:
        lowered: dict[tuple[int, int, int, int], sp.Expr] = {}
        metric_rows = self._matrix_nonzero_by_row(self.metric)
        for (rho, beta, gamma, delta), value in riemann.components.items():
            for alpha, metric_value in metric_rows[rho]:
                key = (alpha, beta, gamma, delta)
                lowered[key] = lowered.get(key, sp.S.Zero) + metric_value * value

        return {
            key: simplified
            for key, value in lowered.items()
            if (simplified := fast_simplify(value)) != 0
        }

    def _kretschmann(
        self, inverse_metric: sp.Matrix, riemann: TensorComponents
    ) -> sp.Expr:
        lowered = self._lower_riemann(riemann)
        if not lowered:
            return sp.S.Zero

        inverse_rows = self._matrix_nonzero_by_row(inverse_metric)
        total = sp.S.Zero
        for (a, b, c, d), lower_value in lowered.items():
            raised_value = sp.S.Zero
            for e, gae in inverse_rows[a]:
                for f, gbf in inverse_rows[b]:
                    for g, gcg in inverse_rows[c]:
                        for h, gdh in inverse_rows[d]:
                            other = lowered.get((e, f, g, h), sp.S.Zero)
                            if other != 0:
                                raised_value += gae * gbf * gcg * gdh * other
            if raised_value != 0:
                total += lower_value * raised_value

        return fast_simplify(total, deep=True)

    def calculate(self, *, include_kretschmann: bool = True) -> GeometryResult:
        start = time.perf_counter()
        self._set_progress("正在构造逆度规...")
        inverse_metric = self._inverse_metric()

        self._set_progress("正在计算克里斯托费尔联络...")
        christoffel = self._christoffel(inverse_metric)

        self._set_progress("正在计算黎曼曲率张量...")
        riemann = self._riemann(christoffel)

        self._set_progress("正在收缩里奇张量...")
        ricci = self._ricci(riemann)

        self._set_progress("正在计算标量曲率...")
        ricci_scalar = self._ricci_scalar(inverse_metric, ricci)

        kretschmann = None
        if include_kretschmann:
            self._set_progress("正在计算 Kretschmann 标量...")
            kretschmann = self._kretschmann(inverse_metric, riemann)

        elapsed = time.perf_counter() - start
        self._set_progress(f"计算完成，用时 {elapsed:.2f} 秒")
        return GeometryResult(
            parsed=self.parsed,
            inverse_metric=inverse_metric,
            christoffel=christoffel,
            riemann=riemann,
            ricci=ricci,
            ricci_scalar=ricci_scalar,
            kretschmann=kretschmann,
            elapsed_seconds=elapsed,
        )


def calculate_geometry(
    parsed: ParsedInputs,
    *,
    include_kretschmann: bool = True,
    progress: ProgressCallback = None,
) -> GeometryResult:
    return TensorCalculator(parsed, progress=progress).calculate(
        include_kretschmann=include_kretschmann
    )
