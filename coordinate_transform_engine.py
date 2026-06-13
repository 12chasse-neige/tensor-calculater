from __future__ import annotations

import re
from dataclasses import dataclass

import sympy as sp
from sympy.parsing.sympy_parser import parse_expr

from tensor_engine import (
    PARSER_TRANSFORMATIONS,
    SAFE_GLOBALS,
    ParsedInputs,
    _add_math_namespace,
    fast_simplify,
    parse_coordinate_symbols,
    split_top_level_csv,
    text_expression,
    validate_identifier,
)


@dataclass(frozen=True)
class CoordinateTransformResult:
    old_coords: tuple[sp.Symbol, ...]
    new_coords: tuple[sp.Symbol, ...]
    forward_map: tuple[sp.Expr, ...]
    inverse_map: tuple[sp.Expr, ...]
    jacobian: sp.Matrix
    metric: sp.Matrix


@dataclass(frozen=True)
class CoordinateTransformExample:
    name: str
    source_coords: str
    source_scalars: str
    source_functions: str
    source_metric: str
    new_coords: str
    transform: str
    description: str
    auto_solve_supported: bool = True


SCHWARZSCHILD_METRIC_TEXT = (
    "[-(1 - 2*M/r), 0, 0, 0],\n"
    "[0, 1/(1 - 2*M/r), 0, 0],\n"
    "[0, 0, r^2, 0],\n"
    "[0, 0, 0, r^2*sin(theta)^2]"
)


TRANSFORM_EXAMPLES: dict[str, CoordinateTransformExample] = {
    "Eddington-Finkelstein 时间坐标": CoordinateTransformExample(
        name="Eddington-Finkelstein 时间坐标",
        source_coords="t, r, theta, phi",
        source_scalars="M",
        source_functions="",
        source_metric=SCHWARZSCHILD_METRIC_TEXT,
        new_coords="v, r, theta, phi",
        transform=(
            "v = t + 2*M*log(r/(2*M) - 1)\n"
            "r = r\n"
            "theta = theta\n"
            "phi = phi"
        ),
        description=(
            "老师课上常用的 EF 时间坐标写法；若要 null advanced coordinate，"
            "可改为 v = t + r + 2*M*log(r/(2*M) - 1)。"
        ),
    ),
    "Rindler 标准正变换": CoordinateTransformExample(
        name="Rindler 标准正变换",
        source_coords="t, x",
        source_scalars="",
        source_functions="",
        source_metric="[-1, 0],\n[0, 1]",
        new_coords="tau, rho",
        transform=(
            "tau = log((x + t)/(x - t))/2\n"
            "rho = sqrt(x^2 - t^2)"
        ),
        description=(
            "右 Rindler 楔区的标准正变换。自动求逆会遇到正负分支选择，"
            "当前版本会提示不能唯一求逆。"
        ),
        auto_solve_supported=False,
    ),
    "Kruskal-Szekeres 标准正变换": CoordinateTransformExample(
        name="Kruskal-Szekeres 标准正变换",
        source_coords="t, r, theta, phi",
        source_scalars="M",
        source_functions="",
        source_metric=SCHWARZSCHILD_METRIC_TEXT,
        new_coords="T, X, theta, phi",
        transform=(
            "T = sqrt(r/(2*M) - 1)*exp(r/(4*M))*sinh(t/(4*M))\n"
            "X = sqrt(r/(2*M) - 1)*exp(r/(4*M))*cosh(t/(4*M))\n"
            "theta = theta\n"
            "phi = phi"
        ),
        description=(
            "Schwarzschild 外区常见 Kruskal-Szekeres 正变换。逆变换含 LambertW "
            "并有区域分支，当前版本不自动求逆。"
        ),
        auto_solve_supported=False,
    ),
}


def _forward_namespace(parsed: ParsedInputs) -> dict[str, object]:
    local_dict: dict[str, object] = {str(coord): coord for coord in parsed.coords}
    _add_math_namespace(local_dict)

    for scalar in parsed.scalar_symbols:
        name = str(scalar)
        if name in local_dict:
            raise ValueError(f"标量常量 '{name}' 与旧坐标名冲突。")
        local_dict[name] = scalar

    for function_name, _args in parsed.function_defs:
        if function_name in local_dict:
            raise ValueError(f"自定义函数 '{function_name}' 与旧坐标或常量名冲突。")
        local_dict[function_name] = sp.Function(function_name)

    return local_dict


def _validate_new_coordinate_names(
    parsed: ParsedInputs, new_coords: tuple[sp.Symbol, ...]
) -> None:
    scalar_names = {str(symbol) for symbol in parsed.scalar_symbols}
    function_names = {name for name, _args in parsed.function_defs}
    for coord in new_coords:
        name = str(coord)
        if name in scalar_names:
            raise ValueError(f"新坐标 '{name}' 与标量常量冲突。")
        if name in function_names:
            raise ValueError(f"新坐标 '{name}' 与自定义函数名冲突。")


def _assignment_items(transform_text: str) -> list[str]:
    prepared = transform_text.strip()
    if not prepared:
        raise ValueError("请输入坐标变换，例如 v = t + r, r = r。")

    items: list[str] = []
    for raw_line in re.split(r"[;\n]", prepared):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        for item in split_top_level_csv(line):
            cleaned = item.strip().rstrip(",;").strip()
            if cleaned:
                items.append(cleaned)
    return items


def _parse_transform_expr(text: str, local_dict: dict[str, object]) -> sp.Expr:
    try:
        return parse_expr(
            text,
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
                f"坐标变换表达式中的 '{name}' 未定义。若它是常量，请加入“标量常量”。"
            ) from exc
        raise ValueError(f"无法解析坐标变换表达式: {exc}") from exc
    except Exception as exc:
        raise ValueError(f"无法解析坐标变换表达式: {exc}") from exc


def parse_forward_assignments(
    new_coords: tuple[sp.Symbol, ...],
    transform_text: str,
    local_dict: dict[str, object],
) -> tuple[sp.Expr, ...]:
    new_coord_names = {str(coord) for coord in new_coords}
    assignments: dict[str, sp.Expr] = {}

    for item in _assignment_items(transform_text):
        if "=" not in item:
            raise ValueError(f"坐标变换 '{item}' 缺少等号，应写成 新坐标 = 旧坐标表达式。")
        lhs, rhs = (part.strip() for part in item.split("=", 1))
        validate_identifier(lhs, "新坐标")
        if lhs not in new_coord_names:
            raise ValueError(f"坐标变换左侧 '{lhs}' 不在新坐标列表中。")
        if lhs in assignments:
            raise ValueError(f"新坐标 '{lhs}' 的变换重复。")
        if not rhs:
            raise ValueError(f"新坐标 '{lhs}' 的右侧表达式不能为空。")
        assignments[lhs] = _parse_transform_expr(rhs, local_dict)

    missing = [str(coord) for coord in new_coords if str(coord) not in assignments]
    if missing:
        raise ValueError("缺少新坐标的变换: " + ", ".join(missing))

    return tuple(assignments[str(coord)] for coord in new_coords)


def _temporary_symbols(coords: tuple[sp.Symbol, ...], prefix: str) -> tuple[sp.Dummy, ...]:
    return tuple(sp.Dummy(f"{prefix}_{index}_{coord}") for index, coord in enumerate(coords))


def solve_inverse_map(
    old_coords: tuple[sp.Symbol, ...],
    new_coords: tuple[sp.Symbol, ...],
    forward_map: tuple[sp.Expr, ...],
) -> tuple[sp.Expr, ...]:
    old_internal = _temporary_symbols(old_coords, "old")
    new_internal = _temporary_symbols(new_coords, "new")
    old_to_internal = dict(zip(old_coords, old_internal))
    new_to_external = dict(zip(new_internal, new_coords))

    forward_internal = tuple(expr.subs(old_to_internal) for expr in forward_map)
    equations = [
        sp.Eq(new_symbol, expression)
        for new_symbol, expression in zip(new_internal, forward_internal)
    ]

    try:
        raw_solutions = sp.solve(equations, old_internal, dict=True, simplify=False)
    except Exception as exc:
        raise ValueError("该坐标变换不能唯一求逆；请检查输入或改用更简单的变换。") from exc

    complete_solutions: list[tuple[sp.Expr, ...]] = []
    for solution in raw_solutions:
        if not all(symbol in solution for symbol in old_internal):
            continue
        inverse_internal = tuple(fast_simplify(solution[symbol]) for symbol in old_internal)
        if any(expression.has(*old_internal) for expression in inverse_internal):
            continue
        inverse_external = tuple(
            fast_simplify(expression.subs(new_to_external), deep=True)
            for expression in inverse_internal
        )
        complete_solutions.append(inverse_external)

    if len(complete_solutions) != 1:
        raise ValueError("该坐标变换不能唯一求逆；请检查输入或改用更简单的变换。")

    return complete_solutions[0]


def _transform_metric_from_inverse(
    parsed: ParsedInputs,
    new_coords: tuple[sp.Symbol, ...],
    inverse_map: tuple[sp.Expr, ...],
) -> tuple[sp.Matrix, sp.Matrix]:
    substitutions = {
        old_coord: new_expr for old_coord, new_expr in zip(parsed.coords, inverse_map)
    }
    substituted_metric = parsed.metric.applyfunc(lambda value: value.subs(substitutions))
    jacobian = sp.Matrix(
        [
            [fast_simplify(sp.diff(old_expr, new_coord)) for new_coord in new_coords]
            for old_expr in inverse_map
        ]
    )
    transformed = jacobian.T * substituted_metric * jacobian
    transformed = transformed.applyfunc(lambda value: fast_simplify(value, deep=True))
    return jacobian, transformed


def transform_metric(
    parsed: ParsedInputs,
    new_coord_text: str,
    transform_text: str,
) -> CoordinateTransformResult:
    new_coords = parse_coordinate_symbols(new_coord_text)
    if len(new_coords) != len(parsed.coords):
        raise ValueError(
            f"新坐标数量({len(new_coords)})与旧度规维度({parsed.metric.rows})不匹配。"
        )
    _validate_new_coordinate_names(parsed, new_coords)

    local_dict = _forward_namespace(parsed)
    forward_map = parse_forward_assignments(new_coords, transform_text, local_dict)
    inverse_map = solve_inverse_map(parsed.coords, new_coords, forward_map)
    jacobian, transformed = _transform_metric_from_inverse(parsed, new_coords, inverse_map)

    return CoordinateTransformResult(
        old_coords=parsed.coords,
        new_coords=new_coords,
        forward_map=forward_map,
        inverse_map=inverse_map,
        jacobian=jacobian,
        metric=transformed,
    )


def metric_matrix_to_input_text(metric: sp.Matrix) -> str:
    rows: list[str] = []
    for row_index in range(metric.rows):
        entries = [
            text_expression(metric[row_index, col_index])
            for col_index in range(metric.cols)
        ]
        rows.append("[" + ", ".join(entries) + "]")
    return ",\n".join(rows)
