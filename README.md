# GR Tensor Calculator

这是一个纯 Python / SymPy 的广义相对论符号计算 GUI。它可以从用户输入的度规计算
非零的 Christoffel 联络、Riemann 张量、Ricci 张量、Ricci 标量和
Kretschmann 标量，也可以把用户输入的几何作用量密度在
`g_{\mu\nu}=\eta_{\mu\nu}+\epsilon h_{\mu\nu}` 下展开到二阶。

## 运行

项目环境保存在当前目录的 `.venv` 中：

```bash
./run_ui.sh
```

如果需要重建环境：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
./run_ui.sh
```

## 曲率计算输入

- 坐标符号：用逗号分隔，例如 `t, r, theta, phi`。
- 标量常量：用逗号分隔，例如 `M, G, c`。
- 自定义函数：写成坐标的函数，例如 `a(t)` 或 `Phi(t, r)`。
- 度规矩阵：每行输入一个矩阵行，例如 `[g00, g01, ...]`；旧的
  `[[...], [...]]` 外层括号写法仍然兼容。

表达式支持接近手写公式的写法，例如 `2M/r`、`r^2`、`sin(theta)`、
`sqrt(...)`、`diff(...)`、`Matrix(...)` 和 `diag(...)`。

内置示例包括：

- 球坐标下的平直时空
- Schwarzschild 度规
- Reissner-Nordstrom 度规
- Kerr 度规（Boyer-Lindquist 坐标）
- FLRW 度规

Schwarzschild 示例：

```text
坐标符号: t, r, theta, phi
标量常量: M
自定义函数:
度规矩阵:
[-(1 - 2M/r), 0, 0, 0],
[0, 1/(1 - 2M/r), 0, 0],
[0, 0, r^2, 0],
[0, 0, 0, r^2*sin(theta)^2]
```

## 功能说明

- 曲率计算使用本地 SymPy 引擎，不依赖 Mathematica notebook 或 EinsteinPy。
- Riemann 张量约定为
  `R^lambda_{mu nu kappa} = Gamma^lambda_{mu nu,kappa} - Gamma^lambda_{mu kappa,nu} + Gamma^lambda_{alpha kappa} Gamma^alpha_{mu nu} - Gamma^lambda_{alpha nu} Gamma^alpha_{mu kappa}`，
  即代码中的索引顺序 `(lambda, mu, nu, kappa)` 表示
  `R^lambda_{mu nu kappa}`。
- 引擎会缓存偏导数、利用 Christoffel 和 Riemann 的对称性，并只保存非零分量。
- UI 的 LaTeX 输出使用较小的 STIX 数学字体渲染，适合窗口模式阅读。
- 扰动作用量页支持自定义标量密度，例如：

```text
sqrtg*(R + alpha*R^2 + beta*Ricci2 + gamma*K + V)
```

其中 `sqrtg` 也可写作 `sqrt(-g)`；`R` 是 Ricci 标量，
`Ricci2`/`Ricci^2` 表示 `R_{\mu\nu}R^{\mu\nu}`，
`K`/`Riemann2`/`Riemann^2` 表示
`R_{\mu\nu\rho\sigma}R^{\mu\nu\rho\sigma}`。其它名字会按用户标量或常量处理。
展开采用 `η=(-,+,+,+)`，重复指标用 `η` 升降并求和；二阶 `R` 的 bulk 形式会按
舍去总导数后的表达式展示。

## 注意

复杂非对角度规的符号化简仍可能比对角度规慢。建议先从内置示例开始，再逐步加入新的常量、
函数或非对角项。默认会计算 Kretschmann 标量；内置 Kerr 度规会使用已知闭式表达式，
但 Riemann 分量本身仍需要较多符号计算。如果只是检查低阶张量分量，可以在 UI 中关闭
Kretschmann 标量。
