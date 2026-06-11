import io
import os
import re
import tempfile
import tkinter as tk
from tkinter import messagebox, scrolledtext, ttk

FONT_CACHE_DIR = os.path.join(tempfile.gettempdir(), "gr_calculator_matplotlib")
os.makedirs(FONT_CACHE_DIR, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", FONT_CACHE_DIR)
os.environ.setdefault("XDG_CACHE_HOME", FONT_CACHE_DIR)

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


def latex_symbol(symbol):
    if symbol == "Γ":
        return r"\Gamma"
    return symbol


def latex_expression(expr):
    try:
        return sp.latex(sp.simplify(expr))
    except Exception:
        return sp.latex(expr)


def render_latex_to_pil(latex, font_size=17, dpi=160):
    fig = Figure(figsize=(1, 0.35), dpi=dpi)
    fig.patch.set_facecolor("white")
    canvas = FigureCanvasAgg(fig)
    fig.text(0, 0, f"${latex}$", fontsize=font_size, color="#111111")

    buffer = io.BytesIO()
    fig.savefig(
        buffer,
        format="png",
        dpi=dpi,
        bbox_inches="tight",
        pad_inches=0.08,
        facecolor="white",
    )
    buffer.seek(0)
    return Image.open(buffer).copy()


class FormulaOutput(ttk.Frame):
    """Scrollable container for rendered formulas and short text notes."""

    def __init__(self, parent):
        super().__init__(parent)
        self.images = []
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        self.canvas = tk.Canvas(self, bg="white", highlightthickness=0)
        self.v_scroll = ttk.Scrollbar(
            self, orient="vertical", command=self.canvas.yview
        )
        self.h_scroll = ttk.Scrollbar(
            self, orient="horizontal", command=self.canvas.xview
        )
        self.canvas.configure(
            yscrollcommand=self.v_scroll.set,
            xscrollcommand=self.h_scroll.set,
        )

        self.content = tk.Frame(self.canvas, bg="white")
        self.window_id = self.canvas.create_window(
            (0, 0), window=self.content, anchor="nw"
        )
        self.content.bind("<Configure>", self.update_scroll_region)
        self.canvas.bind("<Enter>", self.bind_mousewheel)
        self.canvas.bind("<Leave>", self.unbind_mousewheel)

        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.v_scroll.grid(row=0, column=1, sticky="ns")
        self.h_scroll.grid(row=1, column=0, sticky="ew")

    def bind_mousewheel(self, _event):
        self.canvas.bind_all("<MouseWheel>", self.on_mousewheel)
        self.canvas.bind_all("<Button-4>", self.on_linux_scroll_up)
        self.canvas.bind_all("<Button-5>", self.on_linux_scroll_down)

    def unbind_mousewheel(self, _event):
        self.canvas.unbind_all("<MouseWheel>")
        self.canvas.unbind_all("<Button-4>")
        self.canvas.unbind_all("<Button-5>")

    def on_mousewheel(self, event):
        direction = -1 if event.delta > 0 else 1
        self.canvas.yview_scroll(direction * 3, "units")

    def on_linux_scroll_up(self, _event):
        self.canvas.yview_scroll(-3, "units")

    def on_linux_scroll_down(self, _event):
        self.canvas.yview_scroll(3, "units")

    def update_scroll_region(self, _event=None):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def clear(self):
        for child in self.content.winfo_children():
            child.destroy()
        self.images.clear()
        self.canvas.xview_moveto(0)
        self.canvas.yview_moveto(0)
        self.update_scroll_region()

    def add_heading(self, text):
        label = tk.Label(
            self.content,
            text=text,
            bg="white",
            fg="#111111",
            anchor="w",
            justify="left",
            font=("TkDefaultFont", 12, "bold"),
        )
        label.pack(anchor="w", fill="x", padx=12, pady=(12, 4))

    def add_text(self, text):
        label = tk.Label(
            self.content,
            text=text,
            bg="white",
            fg="#333333",
            anchor="w",
            justify="left",
            font=("TkDefaultFont", 10),
        )
        label.pack(anchor="w", fill="x", padx=12, pady=2)

    def add_formula(self, latex, fallback=None):
        try:
            image = render_latex_to_pil(latex)
            photo = ImageTk.PhotoImage(image)
            self.images.append(photo)
            label = tk.Label(self.content, image=photo, bg="white")
        except Exception:
            label = tk.Label(
                self.content,
                text=fallback or latex,
                bg="white",
                fg="#333333",
                anchor="w",
                justify="left",
                font=("Menlo", 12),
            )
        label.pack(anchor="w", padx=18, pady=5)

    def set_entries(self, entries):
        self.clear()
        if not entries:
            self.add_text("无输出。")
            return

        for entry in entries:
            kind = entry[0]
            if kind == "heading":
                self.add_heading(entry[1])
            elif kind == "formula":
                self.add_formula(entry[1], entry[2] if len(entry) > 2 else None)
            else:
                self.add_text(entry[1])
        self.update_scroll_region()

    def set_text(self, text):
        self.set_entries([("text", text)])


def tensor_render_entries(tensor, name="张量", symbol="T"):
    if hasattr(tensor, "tensor"):
        arr = tensor.tensor()
    else:
        arr = tensor

    entries = []
    display_symbol = latex_symbol(symbol)

    if isinstance(arr, (sp.MutableDenseNDimArray, sp.ImmutableDenseNDimArray)):
        shape = arr.shape
        entries.append(("heading", f"{name}，形状 {shape}"))

        if len(shape) == 2:
            found = False
            for i in range(shape[0]):
                for j in range(shape[1]):
                    value = sp.simplify(arr[i, j])
                    if value != 0:
                        latex = (
                            rf"{display_symbol}_{{{i}{j}}}"
                            rf" = {latex_expression(value)}"
                        )
                        entries.append(
                            ("formula", latex, f"{symbol}_{i}{j} = {value}")
                        )
                        found = True
            if not found:
                entries.append(("text", "所有分量为零。"))
        elif len(shape) == 3:
            found = False
            for k in range(shape[0]):
                for i in range(shape[1]):
                    for j in range(shape[2]):
                        value = sp.simplify(arr[k, i, j])
                        if value != 0:
                            latex = (
                                rf"{display_symbol}^{{{k}}}_{{{i}{j}}}"
                                rf" = {latex_expression(value)}"
                            )
                            entries.append(
                                ("formula", latex, f"{symbol}^{k}_{i}{j} = {value}")
                            )
                            found = True
            if not found:
                entries.append(("text", "所有分量为零。"))
        elif len(shape) == 4:
            found = False
            for i in range(shape[0]):
                for j in range(shape[1]):
                    for k in range(shape[2]):
                        for l in range(shape[3]):
                            value = sp.simplify(arr[i, j, k, l])
                            if value != 0:
                                latex = (
                                    rf"{display_symbol}^{{{i}}}_{{{j}{k}{l}}}"
                                    rf" = {latex_expression(value)}"
                                )
                                entries.append(
                                    (
                                        "formula",
                                        latex,
                                        f"{symbol}^{i}_{j}{k}{l} = {value}",
                                    )
                                )
                                found = True
            if not found:
                entries.append(("text", "所有分量为零。"))
        else:
            entries.append(("text", str(arr)))
        return entries

    if isinstance(arr, sp.Basic):
        entries.append(("formula", rf"{display_symbol} = {latex_expression(arr)}"))
    else:
        entries.append(("text", str(arr)))
    return entries


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

        self.tab_summary = FormulaOutput(self.notebook)
        self.tab_christoffel = FormulaOutput(self.notebook)
        self.tab_riemann = FormulaOutput(self.notebook)
        self.tab_ricci = FormulaOutput(self.notebook)
        self.tab_ricciscalar = FormulaOutput(self.notebook)

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
        if hasattr(widget, "set_text"):
            widget.set_text(text)
        else:
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
            if hasattr(widget, "clear"):
                widget.clear()
            else:
                self.set_text(widget, "")

    def format_summary(self, coords, scalar_symbols, function_defs, metric_mat):
        entries = [("heading", "坐标")]
        coord_latex = r"\left(" + ", ".join(sp.latex(coord) for coord in coords) + r"\right)"
        entries.append(
            (
                "formula",
                coord_latex,
                "(" + ", ".join(str(coord) for coord in coords) + ")",
            )
        )

        entries.append(("heading", "标量常量"))
        if scalar_symbols:
            entries.append(
                (
                    "formula",
                    ", ".join(sp.latex(symbol) for symbol in scalar_symbols),
                    ", ".join(str(symbol) for symbol in scalar_symbols),
                )
            )
        else:
            entries.append(("text", "无"))

        entries.append(("heading", "自定义函数"))
        if function_defs:
            for name, args in function_defs:
                arg_symbols = [sp.Symbol(arg) for arg in args]
                function_expr = sp.Function(name)(*arg_symbols)
                entries.append(
                    (
                        "formula",
                        sp.latex(function_expr),
                        f"{name}({', '.join(args)})",
                    )
                )
        else:
            entries.append(("text", "无"))

        entries.append(("heading", f"度规矩阵 ({metric_mat.rows} x {metric_mat.cols})"))
        found_metric_component = False
        for i in range(metric_mat.rows):
            for j in range(metric_mat.cols):
                value = sp.simplify(metric_mat[i, j])
                if value != 0:
                    entries.append(
                        (
                            "formula",
                            rf"g_{{{i}{j}}} = {latex_expression(value)}",
                            f"g_{i}{j} = {value}",
                        )
                    )
                    found_metric_component = True
        if not found_metric_component:
            entries.append(("text", "所有分量为零。"))
        return entries

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
        self.tab_summary.set_entries(
            self.format_summary(coords, scalar_symbols, function_defs, metric_mat)
        )

        try:
            self.set_status("正在构造度规张量...")
            metric = MetricTensor(metric_mat.tolist(), coords, config="ll")

            self.set_status("正在计算克里斯托费尔联络...")
            ch = ChristoffelSymbols.from_metric(metric)
            self.tab_christoffel.set_entries(
                tensor_render_entries(ch, "克里斯托费尔联络", "Γ")
            )

            self.set_status("正在计算黎曼曲率张量...")
            rm = RiemannCurvatureTensor.from_christoffels(ch)
            self.tab_riemann.set_entries(
                tensor_render_entries(rm, "黎曼曲率张量", "R")
            )

            self.set_status("正在计算里奇张量...")
            rc = RicciTensor.from_riemann(rm)
            self.tab_ricci.set_entries(tensor_render_entries(rc, "里奇张量", "R"))

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

            self.tab_ricciscalar.set_entries(
                [
                    ("heading", "标量曲率"),
                    (
                        "formula",
                        rf"R = {latex_expression(scalar_expr)}",
                        f"R = {scalar_expr}",
                    ),
                ]
            )
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
