# GR 曲率张量计算器

这是一个基于 Tkinter 的广义相对论符号计算 GUI，使用 SymPy 和
EinsteinPy 计算度规对应的克里斯托费尔联络、黎曼曲率张量、里奇张量和标量曲率。

## 运行

项目环境保存在当前目录的 `.venv` 中，后续会继续保留。

```bash
./run_ui.sh
```

如果需要重建环境：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
./run_ui.sh
```

## 输入格式

- 坐标符号：用逗号分隔，例如 `t, r, theta, phi`。
- 标量常量：用逗号分隔，例如 `M, G, c`。
- 自定义函数：写成坐标的函数，例如 `a(t)` 或 `Phi(t, r)`。
- 度规矩阵：使用 Python/SymPy 风格的方阵列表。

表达式支持更接近手写公式的写法，例如 `2M/r`、`r^2`、`sin(theta)`、
`sqrt(...)`、`diff(...)`、`Matrix(...)` 和 `diag(...)`。

## 内置示例

UI 中提供了三个示例：

- 球坐标下的平直时空
- 含标量常量 `M` 的 Schwarzschild 度规
- 含尺度因子 `a(t)` 和曲率参数 `k` 的 FLRW 度规

例如 Schwarzschild 度规可以这样输入：

```text
坐标符号: t, r, theta, phi
标量常量: M
自定义函数:
度规矩阵:
[[-(1 - 2M/r), 0, 0, 0],
 [0, 1/(1 - 2M/r), 0, 0],
 [0, 0, r^2, 0],
 [0, 0, 0, r^2*sin(theta)^2]]
```

## 注意

复杂度规的符号化简可能很慢。建议先从内置示例开始，再逐步加入新的常量、
函数或非对角项。
