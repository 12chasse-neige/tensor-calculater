from __future__ import annotations

import re
from dataclasses import dataclass

import sympy as sp
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

from tensor_engine import split_top_level_csv, validate_identifier


@dataclass(frozen=True)
class FormulaBlock:
    kind: str
    title: str
    latex: str = ""
    fallback: str = ""


@dataclass(frozen=True)
class PerturbationResult:
    action_name: str
    convention: str
    blocks: tuple[FormulaBlock, ...]


@dataclass(frozen=True)
class ExpansionDisplay:
    symbol_latex: str
    fallback_name: str
    coeff_latex: tuple[str, str, str]


EPS = sp.Symbol("epsilon")
SQRTG = sp.Symbol("sqrtg")
GINV = sp.Symbol("gInv")
GCOV = sp.Symbol("gCov")
R = sp.Symbol("R")
RICCI = sp.Symbol("Ricci")
RIEMANN = sp.Symbol("Riemann")
RICCI2 = sp.Symbol("Ricci2")
K = sp.Symbol("K")

ETA_INV = sp.Symbol("etaInv")
ETA_COV = sp.Symbol("etaCov")
H_INV = sp.Symbol("hInv")
H_COV = sp.Symbol("hCov")
H_INV2 = sp.Symbol("hInv2")
H1 = sp.Symbol("H1")
H2 = sp.Symbol("H2")
R1 = sp.Symbol("R1")
R2 = sp.Symbol("R2bulk")
RICCI1 = sp.Symbol("Ricci1")
RICCI_TENSOR2 = sp.Symbol("RicciTensor2")
RIEMANN1 = sp.Symbol("Riemann1")
RIEMANN_TENSOR2 = sp.Symbol("RiemannTensor2")
RICCI1SQ = sp.Symbol("Ricci1Sq")
RIEMANN1SQ = sp.Symbol("Riemann1Sq")

KNOWN_SYMBOLS = {
    "sqrtg": SQRTG,
    "sqrt_g": SQRTG,
    "sqrt_minus_g": SQRTG,
    "g": GINV,
    "gInv": GINV,
    "ginv": GINV,
    "gUU": GINV,
    "g_up": GINV,
    "metricInv": GINV,
    "inverse_metric": GINV,
    "gCov": GCOV,
    "gcov": GCOV,
    "gDD": GCOV,
    "g_down": GCOV,
    "metricCov": GCOV,
    "R": R,
    "RicciScalar": R,
    "Ricci": RICCI,
    "Ricci2": RICCI2,
    "RicciSq": RICCI2,
    "Riemann": RIEMANN,
    "Riemann2": K,
    "RiemannSq": K,
    "K": K,
    "Kretschmann": K,
}
MATH_NAMES = {
    "sqrt": sp.sqrt,
    "sin": sp.sin,
    "cos": sp.cos,
    "tan": sp.tan,
    "exp": sp.exp,
    "log": sp.log,
    "ln": sp.log,
    "pi": sp.pi,
    "E": sp.E,
}
SAFE_GLOBALS = {
    "__builtins__": {},
    "Integer": sp.Integer,
    "Float": sp.Float,
    "Rational": sp.Rational,
    "Add": sp.Add,
    "Mul": sp.Mul,
    "Pow": sp.Pow,
}
PARSER_TRANSFORMATIONS = standard_transformations + (
    implicit_multiplication_application,
    convert_xor,
)
RESERVED_IDENTIFIERS = set(KNOWN_SYMBOLS) | set(MATH_NAMES)
RICCI_C_DEFINITION_LATEX = (
    r"C_{\rho\mu\nu}="
    r"\partial_\mu h_{\rho\nu}+\partial_\nu h_{\rho\mu}"
    r"-\partial_\rho h_{\mu\nu},\qquad "
    r"C^\rho{}_{\mu\nu}=\eta^{\rho\sigma}C_{\sigma\mu\nu}"
)
RICCI1_LATEX = (
    r"\frac{1}{2}\left("
    r"\partial_\mu\partial_\nu h"
    r"+\partial^\rho\partial_\rho h_{\mu\nu}"
    r"-\partial_\rho\partial_\mu h^\rho{}_\nu"
    r"-\partial_\rho\partial_\nu h^\rho{}_\mu\right)"
)
RICCI2_LATEX = (
    r"\frac{1}{2}\partial_\rho\left(h^{\rho\sigma}C_{\sigma\nu\mu}\right)"
    r"-\frac{1}{2}\partial_\nu\left(h^{\rho\sigma}"
    r"\partial_\mu h_{\rho\sigma}\right)"
    r"+\frac{1}{4}C^\rho{}_{\nu\alpha}C^\alpha{}_{\rho\mu}"
    r"-\frac{1}{4}\left(\partial_\alpha h\right)C^\alpha{}_{\nu\mu}"
)
LATEX_NAMES = {
    EPS: r"\epsilon",
    SQRTG: r"\sqrt{-g}",
    GINV: r"g^{\mu\nu}",
    GCOV: r"g_{\mu\nu}",
    ETA_INV: r"\eta^{\mu\nu}",
    ETA_COV: r"\eta_{\mu\nu}",
    H_INV: r"h^{\mu\nu}",
    H_COV: r"h_{\mu\nu}",
    H_INV2: r"h^\mu{}_\rho h^{\rho\nu}",
    R: "R",
    RICCI: r"R_{\mu\nu}",
    RIEMANN: r"R_{\mu\nu\rho\sigma}",
    RICCI2: r"R_{\mu\nu}R^{\mu\nu}",
    K: r"R_{\mu\nu\rho\sigma}R^{\mu\nu\rho\sigma}",
    H1: r"H_1",
    H2: r"H_2",
    R1: r"R^{(1)}",
    R2: r"R^{(2)}_{\mathrm{bulk}}",
    RICCI1: RICCI1_LATEX,
    RICCI_TENSOR2: RICCI2_LATEX,
    RIEMANN1: r"R^{(1)}_{\mu\nu\rho\sigma}",
    RIEMANN_TENSOR2: r"R^{(2)}_{\mu\nu\rho\sigma}",
    RICCI1SQ: r"R^{(1)}_{\mu\nu}R_{(1)}^{\mu\nu}",
    RIEMANN1SQ: r"R^{(1)}_{\mu\nu\rho\sigma}R_{(1)}^{\mu\nu\rho\sigma}",
}


def _latex(expr: sp.Expr) -> str:
    try:
        return sp.latex(expr, symbol_names=LATEX_NAMES)
    except Exception:
        return sp.latex(expr)


def _ordered_series_latex(c0: sp.Expr, c1: sp.Expr, c2: sp.Expr) -> str:
    pieces: list[str] = []
    for order, coeff in enumerate((c0, c1, c2)):
        if coeff == 0:
            continue
        term = coeff if order == 0 else sp.Mul(EPS**order, coeff, evaluate=False)
        if pieces and term.could_extract_minus_sign():
            pieces.append("- " + _latex(-term))
        elif pieces:
            pieces.append("+ " + _latex(term))
        else:
            pieces.append(_latex(term))
    return " ".join(pieces) if pieces else "0"


def _generic_coeff_latex(symbol_latex: str) -> tuple[str, str, str]:
    return tuple(
        rf"\left({symbol_latex}\right)^{{({order})}}" for order in range(3)
    )


def _expansion_display(expr: sp.Expr) -> ExpansionDisplay:
    exact_displays: dict[sp.Symbol, ExpansionDisplay] = {
        SQRTG: ExpansionDisplay(
            r"\sqrt{-g}",
            "sqrtg",
            _generic_coeff_latex(r"\sqrt{-g}"),
        ),
        GINV: ExpansionDisplay(
            r"g^{\mu\nu}",
            "gInv",
            _generic_coeff_latex(r"g^{\mu\nu}"),
        ),
        GCOV: ExpansionDisplay(
            r"g_{\mu\nu}",
            "gCov",
            _generic_coeff_latex(r"g_{\mu\nu}"),
        ),
        R: ExpansionDisplay(
            "R",
            "R",
            (r"R^{(0)}", r"R^{(1)}", r"R^{(2)}"),
        ),
        RICCI: ExpansionDisplay(
            r"R_{\mu\nu}",
            "Ricci",
            (
                r"R^{(0)}_{\mu\nu}",
                r"R^{(1)}_{\mu\nu}",
                r"R^{(2)}_{\mu\nu}",
            ),
        ),
        RIEMANN: ExpansionDisplay(
            r"R_{\mu\nu\rho\sigma}",
            "Riemann",
            (
                r"R^{(0)}_{\mu\nu\rho\sigma}",
                r"R^{(1)}_{\mu\nu\rho\sigma}",
                r"R^{(2)}_{\mu\nu\rho\sigma}",
            ),
        ),
        RICCI2: ExpansionDisplay(
            r"R_{\mu\nu}R^{\mu\nu}",
            "Ricci2",
            _generic_coeff_latex(r"R_{\mu\nu}R^{\mu\nu}"),
        ),
        K: ExpansionDisplay(
            r"R_{\mu\nu\rho\sigma}R^{\mu\nu\rho\sigma}",
            "K",
            _generic_coeff_latex(
                r"R_{\mu\nu\rho\sigma}R^{\mu\nu\rho\sigma}"
            ),
        ),
    }
    if isinstance(expr, sp.Symbol) and expr in exact_displays:
        return exact_displays[expr]

    if expr.has(SQRTG):
        return ExpansionDisplay(
            r"\mathcal{L}",
            "L",
            (
                r"\mathcal{L}^{(0)}",
                r"\mathcal{L}^{(1)}",
                r"\mathcal{L}^{(2)}",
            ),
        )

    tensor_symbols = (GINV, GCOV, RICCI, RIEMANN)
    if any(expr.has(symbol) for symbol in tensor_symbols):
        return ExpansionDisplay(
            r"\mathcal{T}",
            "T",
            (
                r"\mathcal{T}^{(0)}",
                r"\mathcal{T}^{(1)}",
                r"\mathcal{T}^{(2)}",
            ),
        )

    return ExpansionDisplay(
        r"\mathcal{Q}",
        "Q",
        (
            r"\mathcal{Q}^{(0)}",
            r"\mathcal{Q}^{(1)}",
            r"\mathcal{Q}^{(2)}",
        ),
    )


def _preprocess_action(text: str) -> str:
    text = text.strip()
    text = re.sub(r"sqrt\s*\(\s*-\s*g\s*\)", "sqrtg", text)
    text = re.sub(r"sqrt\s*\(\s*-\s*detg\s*\)", "sqrtg", text)
    text = text.replace("√(-g)", "sqrtg")
    return text


def _nonnegative_integer_power(power: sp.Expr) -> int | None:
    if power.is_integer and power.is_nonnegative:
        return int(power)
    return None


def _normalize_tensor_contractions(expr: sp.Expr) -> sp.Expr:
    """Normalize tensor shorthand products before the scalar epsilon expansion."""
    expanded = sp.expand(expr)

    def has_metric_ricci_pair(term: sp.Expr) -> bool:
        if not isinstance(term, sp.Mul):
            return False
        powers = term.as_powers_dict()
        g_power = _nonnegative_integer_power(powers.get(GINV, sp.S.Zero))
        ricci_power = _nonnegative_integer_power(powers.get(RICCI, sp.S.Zero))
        return bool(g_power and ricci_power)

    def reduce_metric_ricci_pair(term: sp.Expr) -> sp.Expr:
        powers = term.as_powers_dict()
        g_power = _nonnegative_integer_power(powers.get(GINV, sp.S.Zero))
        ricci_power = _nonnegative_integer_power(powers.get(RICCI, sp.S.Zero))
        if not g_power or not ricci_power:
            return term
        pairs = min(g_power, ricci_power)
        return sp.simplify(term * R**pairs / (GINV**pairs * RICCI**pairs))

    contracted = expanded.replace(has_metric_ricci_pair, reduce_metric_ricci_pair)
    return contracted.xreplace(
        {
            RICCI**2: RICCI2,
            RIEMANN**2: K,
        }
    )


def _scalar_symbols(scalar_text: str) -> dict[str, sp.Symbol]:
    symbols: dict[str, sp.Symbol] = {}
    if not scalar_text.strip():
        return symbols

    for name in split_top_level_csv(scalar_text):
        validate_identifier(name, "自定义标量")
        if name in RESERVED_IDENTIFIERS:
            raise ValueError(f"自定义标量 '{name}' 与内置几何量冲突。")
        symbols[name] = sp.Symbol(name)
    return symbols


def _identifiers(text: str) -> set[str]:
    return set(re.findall(r"\b[A-Za-z_]\w*\b", text))


def parse_action_density(
    action_density: str, scalar_text: str = ""
) -> tuple[sp.Expr, tuple[str, ...]]:
    prepared = _preprocess_action(action_density)
    if not prepared:
        raise ValueError("请输入待展开表达式，例如 Ricci、gInv 或 sqrtg*(gInv*Ricci + alpha*R^2)。")

    local_dict: dict[str, object] = {}
    local_dict.update(MATH_NAMES)
    local_dict.update(KNOWN_SYMBOLS)
    local_dict.update(_scalar_symbols(scalar_text))

    for name in sorted(_identifiers(prepared)):
        if name in local_dict or name in {"O"}:
            continue
        validate_identifier(name, "自动识别的标量")
        local_dict[name] = sp.Symbol(name)

    try:
        expr = parse_expr(
            prepared,
            local_dict=local_dict,
            global_dict=SAFE_GLOBALS,
            transformations=PARSER_TRANSFORMATIONS,
            evaluate=True,
        )
    except TypeError as exc:
        raise ValueError("暂不支持 f(R) 这类未知函数；请把它写成展开后的代数组合。") from exc
    except Exception as exc:
        raise ValueError(f"无法解析待展开表达式: {exc}") from exc

    function_atoms = [atom for atom in expr.atoms(sp.Function) if atom.func not in MATH_NAMES.values()]
    if function_atoms:
        raise ValueError("暂不支持未知函数形式的作用量；请使用标量符号或代数组合。")

    scalar_names = tuple(
        sorted(
            str(symbol)
            for symbol in expr.free_symbols
            if symbol
            not in {
                SQRTG,
                GINV,
                GCOV,
                R,
                RICCI,
                RIEMANN,
                RICCI2,
                K,
            }
        )
    )
    return _normalize_tensor_contractions(expr), scalar_names


def expand_action_density(expr: sp.Expr) -> tuple[sp.Expr, sp.Expr, sp.Expr]:
    replacements = {
        SQRTG: 1 + EPS * H1 + EPS**2 * H2,
        GINV: ETA_INV - EPS * H_INV + EPS**2 * H_INV2,
        GCOV: ETA_COV + EPS * H_COV,
        R: EPS * R1 + EPS**2 * R2,
        RICCI: EPS * RICCI1 + EPS**2 * RICCI_TENSOR2,
        RIEMANN: EPS * RIEMANN1 + EPS**2 * RIEMANN_TENSOR2,
        RICCI2: EPS**2 * RICCI1SQ,
        K: EPS**2 * RIEMANN1SQ,
    }
    expanded = sp.expand(expr.subs(replacements))

    try:
        series = sp.series(expanded, EPS, 0, 3).removeO()
    except Exception as exc:
        raise ValueError("该表达式在平直背景附近不能稳定展开到二阶。") from exc

    for term in sp.Add.make_args(series):
        power = term.as_powers_dict().get(EPS, sp.S.Zero)
        if power.is_number and power < 0:
            raise ValueError("该表达式含有曲率的负幂，在平直背景 R=0 附近奇异。")

    return (
        sp.simplify(series.coeff(EPS, 0)),
        sp.simplify(series.coeff(EPS, 1)),
        sp.simplify(series.coeff(EPS, 2)),
    )


def _formula_blocks_for_definitions() -> tuple[FormulaBlock, ...]:
    return (
        FormulaBlock("heading", "几何量定义"),
        FormulaBlock("formula", "", r"H_1=\frac{1}{2}h", "H1 = h/2"),
        FormulaBlock(
            "formula",
            "",
            (
                r"H_2=\frac{1}{8}h^2"
                r"-\frac{1}{4}h_{\mu\nu}h^{\mu\nu}"
            ),
            "H2 = h^2/8 - h_mn h^mn/4",
        ),
        FormulaBlock(
            "formula",
            "",
            (
                r"R^{(1)}=\partial^\mu\partial_\mu h"
                r"-\partial_\mu\partial_\nu h^{\mu\nu}"
            ),
            "R1 = d^m d_m h - d_m d_n h^mn",
        ),
        FormulaBlock(
            "formula",
            "",
            RICCI_C_DEFINITION_LATEX,
            "C_rmn = d_m h_rn + d_n h_rm - d_r h_mn",
        ),
        FormulaBlock(
            "formula",
            "",
            rf"R^{{(1)}}_{{\mu\nu}}={RICCI1_LATEX}",
            "linear Ricci tensor",
        ),
        FormulaBlock(
            "formula",
            "",
            rf"R^{{(2)}}_{{\mu\nu}}={RICCI2_LATEX}",
            "quadratic Ricci tensor",
        ),
        FormulaBlock(
            "formula",
            "",
            (
                r"R^{(1)}_{\mu\nu\rho\sigma}=\frac{1}{2}\left("
                r"\partial_\rho\partial_\mu h_{\nu\sigma}"
                r"+\partial_\sigma\partial_\nu h_{\mu\rho}"
                r"-\partial_\rho\partial_\nu h_{\mu\sigma}"
                r"-\partial_\sigma\partial_\mu h_{\nu\rho}\right)"
            ),
            "linear Riemann tensor",
        ),
        FormulaBlock(
            "formula",
            "",
            (
                r"R_{\mu\nu\rho\sigma}=\epsilon R^{(1)}_{\mu\nu\rho\sigma}"
                r"+\epsilon^2R^{(2)}_{\mu\nu\rho\sigma}+O(\epsilon^3)"
            ),
            "Riemann_mnrs = eps Riemann1_mnrs + eps^2 Riemann2_mnrs + O(eps^3)",
        ),
        FormulaBlock(
            "formula",
            "",
            (
                r"\mathcal{L}^{(2)}_{\mathrm{EH}}"
                r"=-\frac{1}{4}\partial_\lambda h_{\mu\nu}\partial^\lambda h^{\mu\nu}"
                r"+\frac{1}{2}\partial_\mu h^{\mu\nu}\partial^\lambda h_{\lambda\nu}"
                r"-\frac{1}{2}\partial_\mu h^{\mu\nu}\partial_\nu h"
                r"+\frac{1}{4}\partial_\lambda h\,\partial^\lambda h"
            ),
            "Fierz-Pauli bulk quadratic density",
        ),
        FormulaBlock(
            "formula",
            "",
            (
                r"R^{(2)}_{\mathrm{bulk}}\doteq "
                r"\mathcal{L}^{(2)}_{\mathrm{EH}}-\frac{1}{2}hR^{(1)}"
            ),
            "R2_bulk = L_EH2 - h R1 / 2, up to total derivatives",
        ),
        FormulaBlock(
            "formula",
            "",
            (
                r"g^{\mu\nu}=\eta^{\mu\nu}-\epsilon h^{\mu\nu}"
                r"+\epsilon^2h^\mu{}_\rho h^{\rho\nu}+O(\epsilon^3)"
            ),
            "g^mn = eta^mn - eps h^mn + eps^2 h^m_r h^rn + O(eps^3)",
        ),
        FormulaBlock(
            "formula",
            "",
            (
                r"R_{\mu\nu}=\epsilon R^{(1)}_{\mu\nu}"
                r"+\epsilon^2R^{(2)}_{\mu\nu}+O(\epsilon^3),\qquad "
                r"g^{\mu\nu}R_{\mu\nu}=R"
            ),
            "Ricci_mn = eps Ricci1_mn + eps^2 Ricci2_mn + O(eps^3)",
        ),
    )


def custom_action_perturbation(
    action_density: str = "sqrtg*R", scalar_text: str = ""
) -> PerturbationResult:
    expr, scalar_names = parse_action_density(action_density, scalar_text)
    q0, q1, q2 = expand_action_density(expr)
    display = _expansion_display(expr)
    coefficient_fallbacks = (
        f"{display.fallback_name}0 = {q0}",
        f"{display.fallback_name}1 = {q1}",
        f"{display.fallback_name}2 = {q2}",
    )

    blocks: list[FormulaBlock] = [
        FormulaBlock("heading", "扰动设定"),
        FormulaBlock(
            "formula",
            "",
            (
                r"g_{\mu\nu}=\eta_{\mu\nu}+\epsilon h_{\mu\nu},\qquad "
                r"\eta_{\mu\nu}=\mathrm{diag}(-1,1,1,1)"
            ),
            "g_mn = eta_mn + eps h_mn",
        ),
        FormulaBlock(
            "formula",
            "",
            rf"{display.symbol_latex}={_latex(expr)}",
            f"{display.fallback_name} = {expr}",
        ),
    ]
    if expr.has(RICCI):
        blocks.append(
            FormulaBlock(
                "formula",
                "",
                RICCI_C_DEFINITION_LATEX,
                "C_rmn = d_m h_rn + d_n h_rm - d_r h_mn",
            )
        )

    blocks.extend(
        [
            FormulaBlock("heading", "二阶展开"),
            FormulaBlock(
                "formula",
                "",
                (
                    rf"{display.symbol_latex}="
                    rf"{_ordered_series_latex(q0, q1, q2)}+O(\epsilon^3)"
                ),
                (
                    f"{display.fallback_name} = {q0} + epsilon*({q1}) "
                    f"+ epsilon^2*({q2}) + O(epsilon^3)"
                ),
            ),
            FormulaBlock(
                "formula",
                "",
                rf"{display.coeff_latex[0]}={_latex(q0)}",
                coefficient_fallbacks[0],
            ),
            FormulaBlock(
                "formula",
                "",
                rf"{display.coeff_latex[1]}={_latex(q1)}",
                coefficient_fallbacks[1],
            ),
            FormulaBlock(
                "formula",
                "",
                rf"{display.coeff_latex[2]}={_latex(q2)}",
                coefficient_fallbacks[2],
            ),
        ]
    )

    if scalar_names:
        blocks.append(FormulaBlock("heading", "用户标量"))
        blocks.append(
            FormulaBlock(
                "formula",
                "",
                ", ".join(sp.latex(sp.Symbol(name)) for name in scalar_names),
                ", ".join(scalar_names),
            )
        )

    blocks.extend(_formula_blocks_for_definitions())
    blocks.append(
        FormulaBlock(
            "text",
            "输入支持单个张量或标量密度：sqrtg 或 sqrt(-g)、R、Ricci、Riemann、g/gInv/gUU、gCov/gDD、Ricci2/Ricci^2、K/Riemann2/Riemann^2，以及标量符号；gInv*Ricci 会按 g^{μν}R_{μν}=R 处理，重复指标用 η 升降并求和，R2_bulk 按舍去总导数后的 bulk 形式展示。",
        )
    )

    return PerturbationResult(
        action_name="Custom action",
        convention=(
            "eta=(-,+,+,+), R^lambda_{mu nu kappa} = "
            "Gamma^lambda_{mu nu,kappa} - Gamma^lambda_{mu kappa,nu} + ..."
        ),
        blocks=tuple(blocks),
    )


def calculate_perturbative_action(
    action_density: str = "sqrtg*R", scalar_text: str = ""
) -> PerturbationResult:
    return custom_action_perturbation(action_density, scalar_text)
