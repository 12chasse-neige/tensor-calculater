import io
import os
import re
import tempfile
import tkinter as tk
from tkinter import messagebox, scrolledtext, ttk

os.environ.setdefault(
    "MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "gr_calculator_matplotlib")
)

import matplotlib

matplotlib.use("Agg")

import sympy as sp
from einsteinpy.symbolic import (
    ChristoffelSymbols,
    MetricTensor,
    RicciScalar,
    RicciTensor,
    RiemannCurvatureTensor,
)
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    standard_transformations,
    parse_expr,
)
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from PIL import Image, ImageTk


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
    t for t in standard_transformations if t.__name__ != "auto_symbol"
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
            "[[-1, 0, 0, 0],\n"
            " [0, 1, 0, 0],\n"
            " [0, 0, r^2, 0],\n"
            " [0, 0, 0, r^2*sin(theta)^2]]"
        ),
    },
    "Schwarzschild（M）": {
        "coords": "t, r, theta, phi",
        "scalars": "M",
        "functions": "",
        "metric": (
            "[[-(1 - 2M/r), 0, 0, 0],\n"
            " [0, 1/(1 - 2M/r), 0, 0],\n"
            " [0, 0, r^2, 0],\n"
            " [0, 0, 0, r^2*sin(theta)^2]]"
        ),
    },
    "FLRW（a(t), k）": {
        "coords": "t, r, theta, phi",
        "scalars": "k",
        "functions": "a(t)",
        "metric": (
            "[[-1, 0, 0, 0],\n"
            " [0, a(t)^2/(1 - k*r^2), 0, 0],\n"
            " [0, 0, a(t)^2*r^2, 0],\n"
            " [0, 0, 0, a(t)^2*r^2*sin(theta)^2]]"
        ),
    },
}


def split_top_level_csv(text):
    """Split comma-separated input while respecting brackets and parentheses."""
    items = []
    start = 0
    depth = 0
    pairs = {"(": ")", "[": "]", "{": "}"}
    closing = {")": "(", "]": "[", "}": "{"}
    stack = []

    for index, char in enumerate(text):
        if char in pairs:
            stack.append(char)
            depth += 1
        elif char in closing:
            if not stack or stack[-1] != closing[char]:
                raise ValueError("括号不匹配，请检查输入。")
            stack.pop()
            depth -= 1
        elif char == "," and depth == 0:
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


def validate_identifier(name, label):
    if not IDENTIFIER_RE.match(name):
        raise ValueError(f"{label} '{name}' 不是合法的 Python/SymPy 名称。")
    if name in RESERVED_NAMES:
        raise ValueError(f"{label} '{name}' 是保留名称，请换一个名字。")


def parse_coordinate_symbols(coord_str):
    coord_names = split_top_level_csv(coord_str.strip())
    if not coord_names:
        raise ValueError("请输入至少一个坐标符号。")

    seen = set()
    for name in coord_names:
        validate_identifier(name, "坐标")
        if name in seen:
            raise ValueError(f"坐标 '{name}' 重复。")
        seen.add(name)

    return list(sp.symbols(coord_names))


def add_math_namespace(local_dict):
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
        "simplify",
    ]
    for name in math_names:
        if name == "ln":
            local_dict[name] = sp.log
        elif hasattr(sp, name):
            local_dict[name] = getattr(sp, name)


def build_namespace(coords, scalar_str, func_str):
    local_dict = {str(coord): coord for coord in coords}
    add_math_namespace(local_dict)

    scalar_symbols = []
    scalar_names = split_top_level_csv(scalar_str.strip()) if scalar_str.strip() else []
    for name in scalar_names:
        validate_identifier(name, "标量常量")
        if name in local_dict:
            raise ValueError(f"标量常量 '{name}' 与已有坐标或函数名冲突。")
        symbol = sp.Symbol(name)
        local_dict[name] = symbol
        scalar_symbols.append(symbol)

    function_defs = []
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

    return local_dict, scalar_symbols, function_defs


def parse_metric_expression(metric_str, local_dict):
    if not metric_str.strip():
        raise ValueError("度规矩阵不能为空。")

    try:
        metric_obj = parse_expr(
            metric_str,
            local_dict=local_dict,
            global_dict=SAFE_GLOBALS,
            transformations=PARSER_TRANSFORMATIONS,
            evaluate=False,
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


def parse_gr_inputs(coord_str, scalar_str, func_str, metric_str):
    coords = parse_coordinate_symbols(coord_str)
    local_dict, scalar_symbols, function_defs = build_namespace(
        coords, scalar_str, func_str
    )
    metric_mat = parse_metric_expression(metric_str, local_dict)

    if metric_mat.rows != metric_mat.cols:
        raise ValueError("度规矩阵必须是方阵。")
    if metric_mat.rows != len(coords):
        raise ValueError(
            f"坐标数量({len(coords)})与度规矩阵维度({metric_mat.rows})不匹配。"
        )
    if metric_mat.det() == 0:
        raise ValueError("度规矩阵行列式为 0，无法构造逆度规。")

    return coords, scalar_symbols, function_defs, metric_mat


class GRCalculatorApp:
    """广义相对论符号计算器 GUI。"""

    def __init__(self, root):
        self.root = root
        self.root.title("GR 曲率张量计算器")
        self.root.geometry("1050x880")
        self.create_widgets()

    def create_widgets(self):
        input_frame = ttk.LabelFrame(self.root, text="输入设置", padding=10)
        input_frame.pack(fill="x", padx=10, pady=5)
        input_frame.columnconfigure(1, weight=1)

        ttk.Label(input_frame, text="坐标符号:").grid(
            row=0, column=0, sticky="w", padx=5
        )
        self.coord_entry = ttk.Entry(input_frame, width=60)
        self.coord_entry.grid(row=0, column=1, sticky="we", padx=5, pady=2)

        ttk.Label(input_frame, text="标量常量:").grid(
            row=1, column=0, sticky="w", padx=5
        )
        self.scalar_entry = ttk.Entry(input_frame, width=60)
        self.scalar_entry.grid(row=1, column=1, sticky="we", padx=5, pady=2)

        ttk.Label(input_frame, text="自定义函数:").grid(
            row=2, column=0, sticky="w", padx=5
        )
        self.func_entry = ttk.Entry(input_frame, width=60)
        self.func_entry.grid(row=2, column=1, sticky="we", padx=5, pady=2)

        ttk.Label(input_frame, text="示例:").grid(
            row=3, column=0, sticky="w", padx=5
        )
        example_frame = ttk.Frame(input_frame)
        example_frame.grid(row=3, column=1, sticky="we", padx=5, pady=2)
        example_frame.columnconfigure(0, weight=1)
        self.example_var = tk.StringVar(value="平直时空（球坐标）")
        self.example_combo = ttk.Combobox(
            example_frame,
            textvariable=self.example_var,
            values=list(EXAMPLES.keys()),
            state="readonly",
        )
        self.example_combo.grid(row=0, column=0, sticky="we")
        ttk.Button(
            example_frame,
            text="载入示例",
            command=self.load_selected_example,
        ).grid(row=0, column=1, padx=(8, 0))

        ttk.Label(
            input_frame,
            text="度规矩阵:",
        ).grid(row=4, column=0, sticky="nw", padx=5, pady=5)
        self.metric_text = scrolledtext.ScrolledText(
            input_frame, width=86, height=10, wrap=tk.WORD
        )
        self.metric_text.grid(row=4, column=1, sticky="we", padx=5, pady=5)

        hint = (
            "输入提示：逗号分隔；常量如 M, G, c 写入“标量常量”；"
            "函数如 a(t), Phi(t, r) 写入“自定义函数”；"
            "表达式支持 2M/r、r^2、sin(theta)、diag(...)。"
        )
        ttk.Label(input_frame, text=hint, foreground="#555555").grid(
            row=5, column=0, columnspan=2, sticky="w", padx=5, pady=(0, 4)
        )

        action_frame = ttk.Frame(self.root)
        action_frame.pack(fill="x", padx=10, pady=6)
        self.calc_btn = ttk.Button(
            action_frame, text="计算所有曲率张量", command=self.calculate
        )
        self.calc_btn.pack(side="left")
        self.status_var = tk.StringVar(value="就绪")
        ttk.Label(action_frame, textvariable=self.status_var).pack(
            side="left", padx=12
        )

        output_frame = ttk.LabelFrame(self.root, text="计算结果", padding=10)
        output_frame.pack(fill="both", expand=True, padx=10, pady=5)

        self.notebook = ttk.Notebook(output_frame)
        self.notebook.pack(fill="both", expand=True)

        self.tab_summary = scrolledtext.ScrolledText(self.notebook, wrap=tk.WORD)
        self.tab_christoffel = scrolledtext.ScrolledText(
            self.notebook, wrap=tk.WORD
        )
        self.tab_riemann = scrolledtext.ScrolledText(self.notebook, wrap=tk.WORD)
        self.tab_ricci = scrolledtext.ScrolledText(self.notebook, wrap=tk.WORD)
        self.tab_ricciscalar = scrolledtext.ScrolledText(
            self.notebook, wrap=tk.WORD
        )

        self.notebook.add(self.tab_summary, text="输入摘要")
        self.notebook.add(self.tab_christoffel, text="克里斯托费尔联络")
        self.notebook.add(self.tab_riemann, text="黎曼曲率张量")
        self.notebook.add(self.tab_ricci, text="里奇张量")
        self.notebook.add(self.tab_ricciscalar, text="标量曲率")

        self.load_example("平直时空（球坐标）")

    def load_selected_example(self):
        self.load_example(self.example_var.get())

    def load_example(self, name):
        example = EXAMPLES[name]
        self.coord_entry.delete(0, tk.END)
        self.coord_entry.insert(0, example["coords"])
        self.scalar_entry.delete(0, tk.END)
        self.scalar_entry.insert(0, example["scalars"])
        self.func_entry.delete(0, tk.END)
        self.func_entry.insert(0, example["functions"])
        self.metric_text.delete("1.0", tk.END)
        self.metric_text.insert("1.0", example["metric"])
        self.status_var.set(f"已载入示例：{name}")

    def parse_inputs(self):
        coord_str = self.coord_entry.get()
        scalar_str = self.scalar_entry.get()
        func_str = self.func_entry.get()
        metric_str = self.metric_text.get("1.0", tk.END)
        return parse_gr_inputs(coord_str, scalar_str, func_str, metric_str)

    def set_text(self, widget, text):
        widget.delete("1.0", tk.END)
        widget.insert("1.0", text)

    def clear_outputs(self):
        for widget in [
            self.tab_summary,
            self.tab_christoffel,
            self.tab_riemann,
            self.tab_ricci,
            self.tab_ricciscalar,
        ]:
            self.set_text(widget, "")

    def format_summary(self, coords, scalar_symbols, function_defs, metric_mat):
        scalar_text = ", ".join(str(symbol) for symbol in scalar_symbols) or "无"
        function_text = (
            ", ".join(
                f"{name}({', '.join(args)})" for name, args in function_defs
            )
            or "无"
        )
        lines = [
            "坐标:",
            "  " + ", ".join(str(coord) for coord in coords),
            "",
            "标量常量:",
            "  " + scalar_text,
            "",
            "自定义函数:",
            "  " + function_text,
            "",
            f"度规矩阵 ({metric_mat.rows} x {metric_mat.cols}):",
        ]
        for row in metric_mat.tolist():
            lines.append("  " + str([sp.simplify(item) for item in row]))
        return "\n".join(lines)

    def format_tensor(self, tensor, name="张量", symbol="T"):
        if hasattr(tensor, "tensor"):
            arr = tensor.tensor()
        else:
            arr = tensor

        if isinstance(arr, (sp.MutableDenseNDimArray, sp.ImmutableDenseNDimArray)):
            shape = arr.shape
            if len(shape) == 2:
                lines = [f"{name} (二维，形状 {shape}):"]
                for i in range(shape[0]):
                    row_str = [str(sp.simplify(arr[i, j])) for j in range(shape[1])]
                    lines.append("  " + " , ".join(row_str))
            elif len(shape) == 3:
                lines = [f"{name} (三维，形状 {shape})，非零分量:"]
                found = False
                for k in range(shape[0]):
                    for i in range(shape[1]):
                        for j in range(shape[2]):
                            val = sp.simplify(arr[k, i, j])
                            if val != 0:
                                lines.append(f"  {symbol}^{k}_{i}{j} = {val}")
                                found = True
                if not found:
                    lines.append("  所有分量为零。")
            elif len(shape) == 4:
                lines = [f"{name} (四维，形状 {shape})，非零分量:"]
                found = False
                for i in range(shape[0]):
                    for j in range(shape[1]):
                        for k in range(shape[2]):
                            for l in range(shape[3]):
                                val = sp.simplify(arr[i, j, k, l])
                                if val != 0:
                                    lines.append(f"  {symbol}^{i}_{j}{k}{l} = {val}")
                                    found = True
                if not found:
                    lines.append("  所有分量为零。")
            else:
                lines = [str(arr)]
            return "\n".join(lines)

        if isinstance(arr, sp.Basic):
            return f"{name}: {sp.simplify(arr)}"
        return str(arr)

    def set_status(self, text):
        self.status_var.set(text)
        self.root.update_idletasks()

    def calculate(self):
        try:
            coords, scalar_symbols, function_defs, metric_mat = self.parse_inputs()
        except Exception as exc:
            messagebox.showerror("输入错误", str(exc))
            self.status_var.set("输入错误")
            return

        self.calc_btn.configure(state="disabled")
        self.clear_outputs()
        self.set_text(
            self.tab_summary,
            self.format_summary(coords, scalar_symbols, function_defs, metric_mat),
        )

        try:
            self.set_status("正在构造度规张量...")
            metric = MetricTensor(metric_mat.tolist(), coords, config="ll")

            self.set_status("正在计算克里斯托费尔联络...")
            ch = ChristoffelSymbols.from_metric(metric)
            self.set_text(
                self.tab_christoffel,
                self.format_tensor(ch, "克里斯托费尔联络", "Γ"),
            )

            self.set_status("正在计算黎曼曲率张量...")
            rm = RiemannCurvatureTensor.from_christoffels(ch)
            self.set_text(
                self.tab_riemann,
                self.format_tensor(rm, "黎曼曲率张量", "R"),
            )

            self.set_status("正在计算里奇张量...")
            rc = RicciTensor.from_riemann(rm)
            self.set_text(self.tab_ricci, self.format_tensor(rc, "里奇张量", "R"))

            self.set_status("正在计算标量曲率...")
            try:
                rs = RicciScalar.from_riccitensor(rc)
            except AttributeError:
                try:
                    rs = RicciScalar.from_metric(metric)
                except AttributeError:
                    rs = RicciScalar(rc)

            if hasattr(rs, "expr"):
                scalar_expr = sp.simplify(rs.expr)
            elif hasattr(rs, "tensor"):
                arr = rs.tensor()
                if isinstance(
                    arr, (sp.MutableDenseNDimArray, sp.ImmutableDenseNDimArray)
                ):
                    scalar_expr = sp.simplify(arr[()])
                else:
                    scalar_expr = sp.simplify(arr)
            else:
                scalar_expr = sp.simplify(rs)

            self.set_text(self.tab_ricciscalar, f"标量曲率 (R): {scalar_expr}")
            self.set_status("计算完成")
        except Exception as exc:
            messagebox.showerror("计算错误", f"计算出错:\n{exc}")
            self.status_var.set("计算错误")
        finally:
            self.calc_btn.configure(state="normal")


def main():
    root = tk.Tk()
    GRCalculatorApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
