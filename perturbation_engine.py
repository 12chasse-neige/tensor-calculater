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


EPS = sp.Symbol("epsilon")
SQRTG = sp.Symbol("sqrtg")
R = sp.Symbol("R")
RICCI = sp.Symbol("Ricci")
RIEMANN = sp.Symbol("Riemann")
RICCI2 = sp.Symbol("Ricci2")
K = sp.Symbol("K")

H1 = sp.Symbol("H1")
H2 = sp.Symbol("H2")
R1 = sp.Symbol("R1")
R2 = sp.Symbol("R2bulk")
RICCI1 = sp.Symbol("Ricci1")
RIEMANN1 = sp.Symbol("Riemann1")
RICCI1SQ = sp.Symbol("Ricci1Sq")
RIEMANN1SQ = sp.Symbol("Riemann1Sq")

KNOWN_SYMBOLS = {
    "sqrtg": SQRTG,
    "sqrt_g": SQRTG,
    "sqrt_minus_g": SQRTG,
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
LATEX_NAMES = {
    EPS: r"\epsilon",
    SQRTG: r"\sqrt{-g}",
    R: "R",
    RICCI: r"R_{\mu\nu}",
    RIEMANN: r"R_{\mu\nu\rho\sigma}",
    RICCI2: r"R_{\mu\nu}R^{\mu\nu}",
    K: r"R_{\mu\nu\rho\sigma}R^{\mu\nu\rho\sigma}",
    H1: r"H_1",
    H2: r"H_2",
    R1: r"R^{(1)}",
    R2: r"R^{(2)}_{\mathrm{bulk}}",
    RICCI1: r"R^{(1)}_{\mu\nu}",
    RIEMANN1: r"R^{(1)}_{\mu\nu\rho\sigma}",
    RICCI1SQ: r"R^{(1)}_{\mu\nu}R_{(1)}^{\mu\nu}",
    RIEMANN1SQ: r"R^{(1)}_{\mu\nu\rho\sigma}R_{(1)}^{\mu\nu\rho\sigma}",
}


def _latex(expr: sp.Expr) -> str:
    try:
        return sp.latex(expr, symbol_names=LATEX_NAMES)
    except Exception:
        return sp.latex(expr)


def _preprocess_action(text: str) -> str:
    text = text.strip()
    text = re.sub(r"sqrt\s*\(\s*-\s*g\s*\)", "sqrtg", text)
    text = re.sub(r"sqrt\s*\(\s*-\s*detg\s*\)", "sqrtg", text)
    text = text.replace("√(-g)", "sqrtg")
    return text


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
        raise ValueError("请输入作用量密度，例如 sqrtg*R 或 sqrtg*(R + alpha*R^2)。")

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
        raise ValueError(f"无法解析作用量密度: {exc}") from exc

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
                R,
                RICCI,
                RIEMANN,
                RICCI2,
                K,
            }
        )
    )
    return expr, scalar_names


def expand_action_density(expr: sp.Expr) -> tuple[sp.Expr, sp.Expr, sp.Expr]:
    replacements = {
        SQRTG: 1 + EPS * H1 + EPS**2 * H2,
        R: EPS * R1 + EPS**2 * R2,
        RICCI: EPS * RICCI1,
        RIEMANN: EPS * RIEMANN1,
        RICCI2: EPS**2 * RICCI1SQ,
        K: EPS**2 * RIEMANN1SQ,
    }
    expanded = sp.expand(expr.subs(replacements))

    try:
        series = sp.series(expanded, EPS, 0, 3).removeO()
    except Exception as exc:
        raise ValueError("该作用量在平直背景附近不能稳定展开到二阶。") from exc

    for term in sp.Add.make_args(series):
        power = term.as_powers_dict().get(EPS, sp.S.Zero)
        if power.is_number and power < 0:
            raise ValueError("该作用量含有曲率的负幂，在平直背景 R=0 附近奇异。")

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
            (
                r"R^{(1)}_{\mu\nu}=\frac{1}{2}\left("
                r"\partial_\mu\partial_\nu h"
                r"+\partial^\rho\partial_\rho h_{\mu\nu}"
                r"-\partial_\rho\partial_\mu h^\rho{}_\nu"
                r"-\partial_\rho\partial_\nu h^\rho{}_\mu\right)"
            ),
            "linear Ricci tensor",
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
    )


def custom_action_perturbation(
    action_density: str = "sqrtg*R", scalar_text: str = ""
) -> PerturbationResult:
    expr, scalar_names = parse_action_density(action_density, scalar_text)
    l0, l1, l2 = expand_action_density(expr)

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
            rf"\mathcal{{L}}={_latex(expr)}",
            f"L = {expr}",
        ),
        FormulaBlock("heading", "二阶展开"),
        FormulaBlock(
            "formula",
            "",
            (
                r"S=\int d^4x\,\left["
                r"\mathcal{L}^{(0)}+\epsilon\mathcal{L}^{(1)}"
                r"+\epsilon^2\mathcal{L}^{(2)}+O(\epsilon^3)\right]"
            ),
            "S = int d^4x [L0 + eps L1 + eps^2 L2 + O(eps^3)]",
        ),
        FormulaBlock("formula", "", rf"\mathcal{{L}}^{{(0)}}={_latex(l0)}", f"L0 = {l0}"),
        FormulaBlock("formula", "", rf"\mathcal{{L}}^{{(1)}}={_latex(l1)}", f"L1 = {l1}"),
        FormulaBlock("formula", "", rf"\mathcal{{L}}^{{(2)}}={_latex(l2)}", f"L2 = {l2}"),
    ]

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
            "输入支持 sqrtg 或 sqrt(-g)、R、Ricci2/Ricci^2、K/Riemann2/Riemann^2，以及标量符号；重复指标用 η 升降并求和，R2_bulk 按舍去总导数后的 bulk 形式展示。",
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
