from __future__ import annotations

import io
import os
import queue
import tempfile
import threading
import tkinter as tk
from tkinter import messagebox, scrolledtext, ttk
from tkinter import font as tkfont

FONT_CACHE_DIR = os.path.join(tempfile.gettempdir(), "gr_calculator_matplotlib")
os.makedirs(FONT_CACHE_DIR, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", FONT_CACHE_DIR)
os.environ.setdefault("XDG_CACHE_HOME", FONT_CACHE_DIR)

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams.update(
    {
        "font.family": "STIXGeneral",
        "mathtext.fontset": "stix",
        "mathtext.default": "it",
    }
)

import sympy as sp
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from PIL import Image, ImageTk

from coordinate_transform_engine import (
    CoordinateTransformResult,
    CoordinateTransformExample,
    TRANSFORM_EXAMPLES,
    metric_matrix_to_input_text,
    transform_metric,
)
from perturbation_engine import FormulaBlock, calculate_perturbative_action
from tensor_engine import (
    EXAMPLES,
    GeometryResult,
    ParsedInputs,
    TensorComponents,
    calculate_geometry,
    fast_simplify,
    latex_expression,
    parse_gr_inputs,
    text_expression,
)


APP_BG = "#f5f6fa"
PANEL_BG = "#f5f6fa"
FIELD_BG = "#ffffff"
TEXT = "#172033"
MUTED = "#687386"
BORDER = "#cfd6e6"
ACCENT = "#265bbd"
ERROR = "#ad2f38"
CODE_FONT = ("Menlo", 11)


def configure_fonts(root: tk.Tk) -> None:
    defaults = {
        "TkDefaultFont": ("Helvetica Neue", 13),
        "TkTextFont": ("Helvetica Neue", 13),
        "TkMenuFont": ("Helvetica Neue", 13),
        "TkHeadingFont": ("Helvetica Neue", 13, "bold"),
        "TkFixedFont": ("Menlo", 12),
    }
    for name, config in defaults.items():
        try:
            tkfont.nametofont(name).configure(family=config[0], size=config[1])
            if len(config) > 2:
                tkfont.nametofont(name).configure(weight=config[2])
        except tk.TclError:
            pass


def configure_style(root: tk.Tk) -> None:
    configure_fonts(root)
    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except tk.TclError:
        pass

    root.configure(bg=APP_BG)
    style.configure("App.TFrame", background=APP_BG)
    style.configure("Panel.TFrame", background=PANEL_BG)
    style.configure("Output.TFrame", background=APP_BG)
    style.configure("TLabel", background=PANEL_BG, foreground=TEXT)
    style.configure("Muted.TLabel", background=PANEL_BG, foreground=MUTED)
    style.configure("Status.TLabel", background=APP_BG, foreground=MUTED)
    style.configure("Title.TLabel", background=PANEL_BG, foreground=TEXT, font=("Helvetica Neue", 18, "bold"))
    style.configure("TButton", padding=(12, 7))
    style.configure("Accent.TButton", padding=(14, 8), foreground="#ffffff", background=ACCENT)
    style.map(
        "Accent.TButton",
        background=[("active", "#1d4fb7"), ("disabled", "#90a7db")],
        foreground=[("disabled", "#edf2ff")],
    )
    style.configure("TNotebook", background=APP_BG, borderwidth=0)
    style.configure("TNotebook.Tab", padding=(14, 8), background="#e4e8f1", foreground=TEXT)
    style.map("TNotebook.Tab", background=[("selected", PANEL_BG)])
    style.configure("TCheckbutton", background=PANEL_BG, foreground=TEXT)
    style.configure("TCombobox", padding=4)
    style.configure("TLabelframe", background=PANEL_BG, bordercolor=BORDER)
    style.configure("TLabelframe.Label", background=PANEL_BG, foreground=TEXT)


def render_latex_to_pil(latex: str, font_size: int = 14, dpi: int = 150) -> Image.Image:
    fig = Figure(figsize=(1, 0.32), dpi=dpi)
    fig.patch.set_facecolor(PANEL_BG)
    canvas = FigureCanvasAgg(fig)
    fig.text(0, 0, f"${latex}$", fontsize=font_size, color=TEXT)

    buffer = io.BytesIO()
    fig.savefig(
        buffer,
        format="png",
        dpi=dpi,
        bbox_inches="tight",
        pad_inches=0.05,
        facecolor=PANEL_BG,
    )
    buffer.seek(0)
    return Image.open(buffer).copy()


class FormulaOutput(ttk.Frame):
    def __init__(self, parent: tk.Widget):
        super().__init__(parent, style="Output.TFrame")
        self.images: list[ImageTk.PhotoImage] = []
        self.latex_lines: list[str] = []
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        self.canvas = tk.Canvas(self, bg=APP_BG, highlightthickness=0)
        self.v_scroll = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.h_scroll = ttk.Scrollbar(self, orient="horizontal", command=self.canvas.xview)
        self.canvas.configure(
            yscrollcommand=self.v_scroll.set,
            xscrollcommand=self.h_scroll.set,
        )

        self.content = tk.Frame(self.canvas, bg=APP_BG)
        self.window_id = self.canvas.create_window((0, 0), window=self.content, anchor="nw")
        self.content.bind("<Configure>", self.update_scroll_region)
        self.canvas.bind("<Enter>", self.bind_mousewheel)
        self.canvas.bind("<Leave>", self.unbind_mousewheel)

        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.v_scroll.grid(row=0, column=1, sticky="ns")
        self.h_scroll.grid(row=1, column=0, sticky="ew")

    def bind_mousewheel(self, _event: tk.Event) -> None:
        self.canvas.bind_all("<MouseWheel>", self.on_mousewheel)
        self.canvas.bind_all("<Button-4>", self.on_linux_scroll_up)
        self.canvas.bind_all("<Button-5>", self.on_linux_scroll_down)

    def unbind_mousewheel(self, _event: tk.Event) -> None:
        self.canvas.unbind_all("<MouseWheel>")
        self.canvas.unbind_all("<Button-4>")
        self.canvas.unbind_all("<Button-5>")

    def on_mousewheel(self, event: tk.Event) -> None:
        direction = -1 if event.delta > 0 else 1
        self.canvas.yview_scroll(direction * 3, "units")

    def on_linux_scroll_up(self, _event: tk.Event) -> None:
        self.canvas.yview_scroll(-3, "units")

    def on_linux_scroll_down(self, _event: tk.Event) -> None:
        self.canvas.yview_scroll(3, "units")

    def update_scroll_region(self, _event: tk.Event | None = None) -> None:
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def clear(self) -> None:
        for child in self.content.winfo_children():
            child.destroy()
        self.images.clear()
        self.latex_lines.clear()
        self.canvas.xview_moveto(0)
        self.canvas.yview_moveto(0)
        self.update_scroll_region()

    def add_heading(self, text: str) -> None:
        label = tk.Label(
            self.content,
            text=text,
            bg=APP_BG,
            fg=TEXT,
            anchor="w",
            justify="left",
            font=("Helvetica Neue", 13, "bold"),
        )
        label.pack(anchor="w", fill="x", padx=14, pady=(14, 4))

    def add_text(self, text: str, *, color: str = MUTED) -> None:
        label = tk.Label(
            self.content,
            text=text,
            bg=APP_BG,
            fg=color,
            anchor="w",
            justify="left",
            wraplength=960,
            font=("Helvetica Neue", 11),
        )
        label.pack(anchor="w", fill="x", padx=16, pady=2)

    def add_formula(self, latex: str, fallback: str | None = None) -> None:
        row = tk.Frame(
            self.content,
            bg=PANEL_BG,
            highlightbackground=BORDER,
            highlightthickness=1,
        )
        row.pack(anchor="w", fill="x", padx=14, pady=3)
        try:
            image = render_latex_to_pil(latex)
            photo = ImageTk.PhotoImage(image)
            self.images.append(photo)
            label = tk.Label(row, image=photo, bg=PANEL_BG)
            self.latex_lines.append(latex)
        except Exception:
            label = tk.Label(
                row,
                text=fallback or latex,
                bg=PANEL_BG,
                fg=TEXT,
                anchor="w",
                justify="left",
                font=("Menlo", 10),
                wraplength=1100,
            )
        label.pack(anchor="w", padx=9, pady=6)

    def set_entries(self, entries: list[tuple[str, str] | tuple[str, str, str]]) -> None:
        self.clear()
        if not entries:
            self.add_text("无输出。")
            return

        for entry in entries:
            kind = entry[0]
            if kind == "heading":
                self.add_heading(entry[1])
            elif kind == "formula":
                fallback = entry[2] if len(entry) > 2 else None
                self.add_formula(entry[1], fallback)
            elif kind == "error":
                self.add_text(entry[1], color=ERROR)
            else:
                self.add_text(entry[1])
        self.update_scroll_region()

    def set_blocks(self, blocks: tuple[FormulaBlock, ...]) -> None:
        entries: list[tuple[str, str] | tuple[str, str, str]] = []
        for block in blocks:
            if block.kind == "heading":
                entries.append(("heading", block.title))
            elif block.kind == "formula":
                entries.append(("formula", block.latex, block.fallback))
            elif block.kind == "text":
                entries.append(("text", block.title))
        self.set_entries(entries)

    def set_text(self, text: str) -> None:
        self.set_entries([("text", text)])


def latex_tensor_symbol(tensor: TensorComponents) -> str:
    if tensor.symbol == "Gamma":
        return r"\Gamma"
    if tensor.symbol == "Riemann":
        return "R"
    if tensor.symbol == "Ricci":
        return "R"
    return tensor.symbol


def latex_component_label(tensor: TensorComponents, key: tuple[int, ...]) -> str:
    symbol = latex_tensor_symbol(tensor)
    if tensor.variance == "ull":
        return rf"{symbol}^{{{key[0]}}}_{{{key[1]}{key[2]}}}"
    if tensor.variance == "ulll":
        return rf"{symbol}^{{{key[0]}}}_{{{key[1]}{key[2]}{key[3]}}}"
    if tensor.variance == "ll":
        return rf"{symbol}_{{{key[0]}{key[1]}}}"
    return symbol


def tensor_entries(tensor: TensorComponents) -> list[tuple[str, str] | tuple[str, str, str]]:
    entries: list[tuple[str, str] | tuple[str, str, str]] = [
        ("heading", f"{tensor.name}，非零分量 {len(tensor.components)} 个，形状 {tensor.shape}")
    ]
    if not tensor.components:
        entries.append(("text", "所有分量为零。"))
        return entries

    for key in sorted(tensor.components):
        value = tensor.components[key]
        latex = rf"{latex_component_label(tensor, key)} = {latex_expression(value)}"
        fallback = f"{latex_component_label(tensor, key)} = {text_expression(value)}"
        entries.append(("formula", latex, fallback))
    return entries


def matrix_component_entries(
    matrix: sp.Matrix,
    *,
    heading: str,
    symbol: str,
    variance: str = "ll",
) -> list[tuple[str, str] | tuple[str, str, str]]:
    entries: list[tuple[str, str] | tuple[str, str, str]] = [("heading", heading)]
    found = False
    for i in range(matrix.rows):
        for j in range(matrix.cols):
            value = fast_simplify(matrix[i, j])
            if value != 0:
                label = (
                    rf"{symbol}^{{{i}{j}}}"
                    if variance == "uu"
                    else rf"{symbol}_{{{i}{j}}}"
                )
                entries.append(
                    (
                        "formula",
                        rf"{label} = {latex_expression(value)}",
                        f"{label} = {text_expression(value)}",
                    )
                )
                found = True
    if not found:
        entries.append(("text", "所有分量为零。"))
    return entries


class GRCalculatorApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("GR Tensor Calculator")
        self.root.geometry("1340x860")
        self.root.minsize(1160, 720)
        configure_style(root)

        self.geometry_cache: dict[tuple[str, str, str, str, bool], GeometryResult] = {}
        self.worker: threading.Thread | None = None
        self.messages: queue.Queue[tuple[str, object]] = queue.Queue()
        self.busy_buttons: list[ttk.Button] = []
        self.last_transform_result: CoordinateTransformResult | None = None
        self.last_transform_metric_text = ""
        self.active_transform_example: CoordinateTransformExample | None = None

        self.create_widgets()
        self.load_example("平直时空（球坐标）")
        self.load_transform_example(
            "Eddington-Finkelstein 时间坐标",
            include_source=False,
        )
        self.render_perturbation()

    def create_widgets(self) -> None:
        self.root.columnconfigure(0, minsize=520)
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)
        self.root.rowconfigure(1, minsize=36)

        left = ttk.Frame(self.root, style="Panel.TFrame", padding=(18, 16))
        left.grid(row=0, column=0, sticky="nsew")
        left.columnconfigure(0, weight=1)
        left.rowconfigure(1, weight=1)

        title = ttk.Label(left, text="GR Tensor Calculator", style="Title.TLabel")
        title.grid(row=0, column=0, sticky="w", pady=(0, 14))

        self.input_notebook = ttk.Notebook(left)
        self.input_notebook.grid(row=1, column=0, sticky="nsew")

        geometry_tab = ttk.Frame(self.input_notebook, style="Panel.TFrame", padding=(10, 12))
        geometry_tab.columnconfigure(0, weight=1)
        geometry_tab.rowconfigure(9, weight=1)
        self.input_notebook.add(geometry_tab, text="曲率")
        self.create_geometry_inputs(geometry_tab)

        transform_tab = ttk.Frame(self.input_notebook, style="Panel.TFrame", padding=(10, 12))
        transform_tab.columnconfigure(0, weight=1)
        transform_tab.rowconfigure(5, weight=1)
        self.input_notebook.add(transform_tab, text="坐标变换")
        self.create_transform_inputs(transform_tab)

        perturbation_tab = ttk.Frame(self.input_notebook, style="Panel.TFrame", padding=(10, 12))
        perturbation_tab.columnconfigure(0, weight=1)
        self.input_notebook.add(perturbation_tab, text="扰动作用量")
        self.create_perturbation_inputs(perturbation_tab)

        right = ttk.Frame(self.root, style="App.TFrame", padding=(14, 14))
        right.grid(row=0, column=1, sticky="nsew")
        right.columnconfigure(0, weight=1)
        right.rowconfigure(0, weight=1)

        self.output_notebook = ttk.Notebook(right)
        self.output_notebook.grid(row=0, column=0, sticky="nsew")
        self.outputs: dict[str, FormulaOutput] = {}
        for key, label in (
            ("summary", "摘要"),
            ("christoffel", "Γ"),
            ("riemann", "Riemann"),
            ("ricci", "Ricci"),
            ("scalars", "标量"),
            ("transform", "坐标变换"),
            ("perturbation", "扰动作用量"),
        ):
            output = FormulaOutput(self.output_notebook)
            self.outputs[key] = output
            self.output_notebook.add(output, text=label)

        status_bar = ttk.Frame(self.root, style="App.TFrame", padding=(14, 0))
        status_bar.grid(row=1, column=0, columnspan=2, sticky="ew")
        status_bar.columnconfigure(0, weight=1)
        self.status_var = tk.StringVar(value="就绪")
        ttk.Label(status_bar, textvariable=self.status_var, style="Status.TLabel").grid(
            row=0, column=0, sticky="w"
        )

    def create_geometry_inputs(self, parent: ttk.Frame) -> None:
        ttk.Label(parent, text="示例").grid(row=0, column=0, sticky="w")
        example_row = ttk.Frame(parent, style="Panel.TFrame")
        example_row.grid(row=1, column=0, sticky="ew", pady=(3, 12))
        example_row.columnconfigure(0, weight=1)
        self.example_var = tk.StringVar(value="平直时空（球坐标）")
        self.example_combo = ttk.Combobox(
            example_row,
            textvariable=self.example_var,
            values=list(EXAMPLES.keys()),
            state="readonly",
        )
        self.example_combo.grid(row=0, column=0, sticky="ew")
        ttk.Button(example_row, text="载入", command=self.load_selected_example).grid(
            row=0, column=1, padx=(8, 0)
        )

        self.coord_entry = self.add_entry(parent, "坐标符号", 2, "t, r, theta, phi")
        self.scalar_entry = self.add_entry(parent, "标量常量", 4, "M, G, c")
        self.func_entry = self.add_entry(parent, "自定义函数", 6, "a(t), Phi(t, r)")

        ttk.Label(parent, text="度规矩阵").grid(row=8, column=0, sticky="w", pady=(4, 3))
        self.metric_text = scrolledtext.ScrolledText(
            parent,
            width=62,
            height=16,
            wrap=tk.NONE,
            font=CODE_FONT,
            bg=FIELD_BG,
            fg=TEXT,
            insertbackground=TEXT,
            relief="solid",
            borderwidth=1,
            highlightthickness=1,
            highlightbackground=BORDER,
            highlightcolor=ACCENT,
        )
        self.metric_text.grid(row=9, column=0, sticky="nsew")

        options = ttk.Frame(parent, style="Panel.TFrame")
        options.grid(row=10, column=0, sticky="ew", pady=(12, 8))
        options.columnconfigure(0, weight=1)
        self.include_k_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            options,
            text="Kretschmann 标量",
            variable=self.include_k_var,
        ).grid(row=0, column=0, sticky="w")

        action_row = ttk.Frame(parent, style="Panel.TFrame")
        action_row.grid(row=11, column=0, sticky="ew")
        action_row.columnconfigure(0, weight=1)
        self.calculate_btn = ttk.Button(
            action_row,
            text="计算曲率",
            style="Accent.TButton",
            command=self.calculate_curvature,
        )
        self.calculate_btn.grid(row=0, column=0, sticky="ew")
        self.busy_buttons.append(self.calculate_btn)

    def create_transform_inputs(self, parent: ttk.Frame) -> None:
        ttk.Label(parent, text="变换示例").grid(row=0, column=0, sticky="w")
        example_row = ttk.Frame(parent, style="Panel.TFrame")
        example_row.grid(row=1, column=0, sticky="ew", pady=(3, 12))
        example_row.columnconfigure(0, weight=1)
        self.transform_example_var = tk.StringVar(
            value="Eddington-Finkelstein 时间坐标"
        )
        self.transform_example_combo = ttk.Combobox(
            example_row,
            textvariable=self.transform_example_var,
            values=list(TRANSFORM_EXAMPLES.keys()),
            state="readonly",
        )
        self.transform_example_combo.grid(row=0, column=0, sticky="ew")
        ttk.Button(
            example_row,
            text="载入",
            command=self.load_selected_transform_example,
        ).grid(row=0, column=1, padx=(8, 0))

        self.transform_coord_entry = self.add_entry(
            parent,
            "新坐标符号",
            2,
            "v, r, theta, phi",
        )

        ttk.Label(parent, text="新坐标关于旧坐标的表达式").grid(
            row=4, column=0, sticky="w", pady=(4, 3)
        )
        self.transform_text = scrolledtext.ScrolledText(
            parent,
            width=62,
            height=14,
            wrap=tk.NONE,
            font=CODE_FONT,
            bg=FIELD_BG,
            fg=TEXT,
            insertbackground=TEXT,
            relief="solid",
            borderwidth=1,
            highlightthickness=1,
            highlightbackground=BORDER,
            highlightcolor=ACCENT,
        )
        self.transform_text.grid(row=5, column=0, sticky="nsew")

        hint = "输入方向固定为：新坐标 = 旧坐标的函数；程序会自动求逆后变换度规。"
        ttk.Label(parent, text=hint, style="Muted.TLabel", wraplength=440).grid(
            row=6, column=0, sticky="ew", pady=(8, 10)
        )

        action_row = ttk.Frame(parent, style="Panel.TFrame")
        action_row.grid(row=7, column=0, sticky="ew")
        action_row.columnconfigure(1, weight=1)
        self.transform_btn = ttk.Button(
            action_row,
            text="变换度规",
            style="Accent.TButton",
            command=self.calculate_metric_transform,
        )
        self.transform_btn.grid(row=0, column=1, sticky="ew", padx=(0, 4))
        self.busy_buttons.append(self.transform_btn)
        ttk.Button(
            action_row,
            text="载入曲率页",
            command=self.load_transformed_metric,
        ).grid(row=0, column=2, sticky="ew", padx=(4, 0))

    def add_entry(
        self, parent: ttk.Frame, label: str, row: int, placeholder: str
    ) -> ttk.Entry:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=(0, 3))
        entry = ttk.Entry(parent)
        entry.grid(row=row + 1, column=0, sticky="ew", pady=(0, 10))
        entry.insert(0, placeholder)
        return entry

    def create_perturbation_inputs(self, parent: ttk.Frame) -> None:
        parent.rowconfigure(1, weight=1)

        ttk.Label(parent, text="作用量密度").grid(row=0, column=0, sticky="w")
        self.action_text = scrolledtext.ScrolledText(
            parent,
            width=42,
            height=7,
            wrap=tk.WORD,
            font=CODE_FONT,
            bg=FIELD_BG,
            fg=TEXT,
            insertbackground=TEXT,
            relief="solid",
            borderwidth=1,
            highlightthickness=1,
            highlightbackground=BORDER,
            highlightcolor=ACCENT,
        )
        self.action_text.grid(row=1, column=0, sticky="nsew", pady=(3, 12))
        self.action_text.insert(
            "1.0",
            "sqrtg*(R + alpha*R^2 + beta*Ricci2 + gamma*K + V)",
        )

        self.action_scalar_entry = self.add_entry(
            parent,
            "标量/常量",
            2,
            "alpha, beta, gamma, V",
        )

        convention = (
            "可用 sqrtg 或 sqrt(-g)、R、Ricci2/Ricci^2、K/Riemann2/Riemann^2。"
        )
        ttk.Label(parent, text=convention, style="Muted.TLabel", wraplength=360).grid(
            row=4, column=0, sticky="ew", pady=(0, 14)
        )

        self.perturbation_btn = ttk.Button(
            parent,
            text="展开到二阶",
            style="Accent.TButton",
            command=self.render_perturbation,
        )
        self.perturbation_btn.grid(row=5, column=0, sticky="ew")
        self.busy_buttons.append(self.perturbation_btn)

    def load_selected_example(self) -> None:
        self.load_example(self.example_var.get())

    def load_example(self, name: str) -> None:
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

    def load_selected_transform_example(self) -> None:
        self.load_transform_example(self.transform_example_var.get(), include_source=True)

    def load_transform_example(self, name: str, *, include_source: bool = True) -> None:
        example = TRANSFORM_EXAMPLES[name]
        self.active_transform_example = example
        self.transform_example_var.set(example.name)

        if include_source:
            self.coord_entry.delete(0, tk.END)
            self.coord_entry.insert(0, example.source_coords)
            self.scalar_entry.delete(0, tk.END)
            self.scalar_entry.insert(0, example.source_scalars)
            self.func_entry.delete(0, tk.END)
            self.func_entry.insert(0, example.source_functions)
            self.metric_text.delete("1.0", tk.END)
            self.metric_text.insert("1.0", example.source_metric)

        self.transform_coord_entry.delete(0, tk.END)
        self.transform_coord_entry.insert(0, example.new_coords)
        self.transform_text.delete("1.0", tk.END)
        self.transform_text.insert("1.0", example.transform)
        self.last_transform_result = None
        self.last_transform_metric_text = ""

        if example.auto_solve_supported:
            self.outputs["transform"].set_entries([("text", example.description)])
            self.status_var.set(f"已载入坐标变换示例：{example.name}")
        else:
            self.outputs["transform"].set_entries(
                [
                    ("heading", example.name),
                    ("text", example.description),
                    (
                        "text",
                        "此示例用于展示标准写法；当前自动求逆不会直接计算它。",
                    ),
                ]
            )
            self.status_var.set(f"已载入暂不自动求逆的示例：{example.name}")

    def geometry_input_key(self) -> tuple[str, str, str, str, bool]:
        return (
            self.coord_entry.get().strip(),
            self.scalar_entry.get().strip(),
            self.func_entry.get().strip(),
            self.metric_text.get("1.0", tk.END).strip(),
            self.include_k_var.get(),
        )

    def parse_inputs(self) -> ParsedInputs:
        return parse_gr_inputs(
            self.coord_entry.get(),
            self.scalar_entry.get(),
            self.func_entry.get(),
            self.metric_text.get("1.0", tk.END),
        )

    def set_busy(self, busy: bool) -> None:
        state = "disabled" if busy else "normal"
        for button in self.busy_buttons:
            button.configure(state=state)

    def calculate_curvature(self) -> None:
        if self.worker and self.worker.is_alive():
            self.status_var.set("已有计算正在运行。")
            return

        try:
            parsed = self.parse_inputs()
        except Exception as exc:
            messagebox.showerror("输入错误", str(exc))
            self.status_var.set("输入错误")
            return

        key = self.geometry_input_key()
        cached = self.geometry_cache.get(key)
        if cached:
            self.render_geometry(cached)
            self.status_var.set(f"使用缓存结果，用时 {cached.elapsed_seconds:.2f} 秒")
            return

        self.clear_geometry_outputs()
        self.outputs["summary"].set_entries(self.summary_entries(parsed))
        self.output_notebook.select(self.outputs["summary"])
        self.set_busy(True)
        self.status_var.set("开始计算...")

        def progress(message: str) -> None:
            self.messages.put(("status", message))

        def worker() -> None:
            try:
                result = calculate_geometry(
                    parsed,
                    include_kretschmann=self.include_k_var.get(),
                    progress=progress,
                )
                self.messages.put(("geometry_result", (key, result)))
            except Exception as exc:
                self.messages.put(("error", str(exc)))
            finally:
                self.messages.put(("done", None))

        self.worker = threading.Thread(target=worker, daemon=True)
        self.worker.start()
        self.root.after(80, self.drain_messages)

    def drain_messages(self) -> None:
        while True:
            try:
                kind, payload = self.messages.get_nowait()
            except queue.Empty:
                break

            if kind == "status":
                self.status_var.set(str(payload))
            elif kind == "geometry_result":
                key, result = payload
                self.geometry_cache[key] = result
                self.render_geometry(result)
            elif kind == "error":
                self.outputs["summary"].set_entries([("error", str(payload))])
                messagebox.showerror("计算错误", str(payload))
                self.status_var.set("计算错误")
            elif kind == "done":
                self.set_busy(False)

        if self.worker and self.worker.is_alive():
            self.root.after(80, self.drain_messages)

    def calculate_metric_transform(self) -> None:
        if self.is_active_unsupported_transform_example():
            self.render_unsupported_transform_example()
            return

        try:
            parsed = self.parse_inputs()
            result = transform_metric(
                parsed,
                self.transform_coord_entry.get(),
                self.transform_text.get("1.0", tk.END),
            )
        except Exception as exc:
            self.last_transform_result = None
            self.last_transform_metric_text = ""
            self.outputs["transform"].set_entries([("error", str(exc))])
            messagebox.showerror("坐标变换错误", str(exc))
            self.status_var.set("坐标变换失败")
            return

        self.last_transform_result = result
        self.last_transform_metric_text = metric_matrix_to_input_text(result.metric)
        self.outputs["transform"].set_entries(self.transform_entries(result))
        self.output_notebook.select(self.outputs["transform"])
        self.status_var.set("坐标变换完成")

    def is_active_unsupported_transform_example(self) -> bool:
        example = self.active_transform_example
        if example is None or example.auto_solve_supported:
            return False
        return (
            self.transform_coord_entry.get().strip() == example.new_coords.strip()
            and self.transform_text.get("1.0", tk.END).strip() == example.transform.strip()
        )

    def render_unsupported_transform_example(self) -> None:
        example = self.active_transform_example
        if example is None:
            return
        self.last_transform_result = None
        self.last_transform_metric_text = ""
        self.outputs["transform"].set_entries(
            [
                ("heading", example.name),
                ("text", example.description),
                (
                    "text",
                    "这个标准正变换当前不能可靠自动求逆；已避免直接调用求解器，以免界面长时间卡住。",
                ),
            ]
        )
        self.output_notebook.select(self.outputs["transform"])
        self.status_var.set("该示例暂不支持自动求逆")

    def load_transformed_metric(self) -> None:
        self.calculate_metric_transform()
        if self.last_transform_result is None:
            return

        coord_text = ", ".join(str(coord) for coord in self.last_transform_result.new_coords)
        self.coord_entry.delete(0, tk.END)
        self.coord_entry.insert(0, coord_text)
        self.metric_text.delete("1.0", tk.END)
        self.metric_text.insert("1.0", self.last_transform_metric_text)
        self.input_notebook.select(0)
        self.clear_geometry_outputs()
        self.outputs["summary"].set_entries(
            [("text", "已载入变换后的度规。点击“计算曲率”继续计算几何量。")]
        )
        self.status_var.set("已将变换后的度规载入曲率页")

    def clear_geometry_outputs(self) -> None:
        for key in ("summary", "christoffel", "riemann", "ricci", "scalars"):
            self.outputs[key].clear()

    def summary_entries(
        self, parsed: ParsedInputs
    ) -> list[tuple[str, str] | tuple[str, str, str]]:
        entries: list[tuple[str, str] | tuple[str, str, str]] = [("heading", "坐标")]
        coord_latex = r"x^\mu=\left(" + ", ".join(sp.latex(coord) for coord in parsed.coords) + r"\right)"
        entries.append(
            (
                "formula",
                coord_latex,
                "(" + ", ".join(str(coord) for coord in parsed.coords) + ")",
            )
        )

        entries.append(("heading", "标量常量"))
        if parsed.scalar_symbols:
            entries.append(
                (
                    "formula",
                    ", ".join(sp.latex(symbol) for symbol in parsed.scalar_symbols),
                    ", ".join(str(symbol) for symbol in parsed.scalar_symbols),
                )
            )
        else:
            entries.append(("text", "无"))

        entries.append(("heading", "自定义函数"))
        if parsed.function_defs:
            for name, args in parsed.function_defs:
                arg_symbols = [sp.Symbol(arg) for arg in args]
                function_expr = sp.Function(name)(*arg_symbols)
                entries.append(
                    (
                        "formula",
                        latex_expression(function_expr),
                        text_expression(function_expr),
                    )
                )
        else:
            entries.append(("text", "无"))

        entries.extend(
            matrix_component_entries(
                parsed.metric,
                heading=f"度规矩阵 gμν ({parsed.metric.rows} x {parsed.metric.cols})",
                symbol="g",
            )
        )
        return entries

    def render_geometry(self, result: GeometryResult) -> None:
        summary = self.summary_entries(result.parsed)
        summary.extend(
            matrix_component_entries(
                result.inverse_metric,
                heading="逆度规 g^{μν}",
                symbol="g",
                variance="uu",
            )
        )
        summary.append(("heading", "性能"))
        summary.append(("text", f"本次计算用时 {result.elapsed_seconds:.2f} 秒。"))
        self.outputs["summary"].set_entries(summary)
        self.outputs["christoffel"].set_entries(tensor_entries(result.christoffel))
        self.outputs["riemann"].set_entries(tensor_entries(result.riemann))
        self.outputs["ricci"].set_entries(tensor_entries(result.ricci))
        self.outputs["scalars"].set_entries(self.scalar_entries(result))
        self.status_var.set(f"计算完成，用时 {result.elapsed_seconds:.2f} 秒")

    def transform_entries(
        self, result: CoordinateTransformResult
    ) -> list[tuple[str, str] | tuple[str, str, str]]:
        entries: list[tuple[str, str] | tuple[str, str, str]] = [("heading", "输入的正变换 y(x)")]
        for new_coord, expression in zip(result.new_coords, result.forward_map):
            entries.append(
                (
                    "formula",
                    rf"{sp.latex(new_coord)} = {latex_expression(expression)}",
                    f"{new_coord} = {text_expression(expression)}",
                )
            )

        entries.append(("heading", "自动求得的逆变换 x(y)"))
        for old_coord, expression in zip(result.old_coords, result.inverse_map):
            entries.append(
                (
                    "formula",
                    rf"{sp.latex(old_coord)} = {latex_expression(expression)}",
                    f"{old_coord} = {text_expression(expression)}",
                )
            )

        entries.extend(
            matrix_component_entries(
                result.jacobian,
                heading=r"Jacobian 矩阵 Jμa = ∂xμ/∂ya",
                symbol="J",
            )
        )
        entries.extend(
            matrix_component_entries(
                result.metric,
                heading=f"新度规矩阵 g'μν ({result.metric.rows} x {result.metric.cols})",
                symbol="g'",
            )
        )
        entries.append(("heading", "可载入的矩阵文本"))
        entries.append(("text", self.last_transform_metric_text or metric_matrix_to_input_text(result.metric)))
        return entries

    def scalar_entries(
        self, result: GeometryResult
    ) -> list[tuple[str, str] | tuple[str, str, str]]:
        entries: list[tuple[str, str] | tuple[str, str, str]] = [("heading", "标量曲率")]
        entries.append(
            (
                "formula",
                rf"R = {latex_expression(result.ricci_scalar)}",
                f"R = {text_expression(result.ricci_scalar)}",
            )
        )
        entries.append(("heading", "Kretschmann 标量"))
        if result.kretschmann is None:
            entries.append(("text", "本次未计算。"))
        else:
            entries.append(
                (
                    "formula",
                    (
                        r"R_{\mu\nu\rho\sigma}R^{\mu\nu\rho\sigma}"
                        rf" = {latex_expression(result.kretschmann)}"
                    ),
                    f"K = {text_expression(result.kretschmann)}",
                )
            )
        return entries

    def render_perturbation(self) -> None:
        try:
            result = calculate_perturbative_action(
                self.action_text.get("1.0", tk.END),
                self.action_scalar_entry.get(),
            )
        except Exception as exc:
            self.outputs["perturbation"].set_entries([("error", str(exc))])
            self.status_var.set("扰动作用量生成失败")
            return
        self.outputs["perturbation"].set_blocks(result.blocks)
        self.status_var.set("已生成扰动作用量展开")


def main() -> None:
    root = tk.Tk()
    GRCalculatorApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
