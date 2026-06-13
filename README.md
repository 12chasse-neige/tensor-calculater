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
- 坐标变换页可以从“曲率”页当前输入的原度规出发，输入新坐标和坐标变换，
  返回变换后的新度规，并可一键载入曲率页继续计算。
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

## 坐标变换输入

坐标变换页读取“曲率”页中的原坐标、常量、自定义函数和原度规。输入约定为：

- 新坐标符号：用逗号分隔，例如 `v, r, theta, phi`。
- 输入方向可选：
  - `新坐标 = 旧坐标函数（自动求逆）`，例如 `v = t + r`。
  - `旧坐标 = 新坐标函数（直接计算）`，例如 `t = v - r`。
- 默认使用 `新坐标 = 旧坐标函数`；如果自动求逆不唯一或求不出，可以切换到
  `旧坐标 = 新坐标函数`，直接输入你知道的逆变换。
- 坐标名必须是合法的 Python/SymPy 标识符，不能直接写 `t'`；建议写作
  `tp`、`t_prime`、`v` 等。

程序使用

```text
g'_{ab}(y) = (∂x^μ/∂y^a)(∂x^ν/∂y^b) g_{μν}(x(y))
```

生成新度规。点击“载入曲率页”后，可以继续用现有“计算曲率”按钮计算新坐标下的几何量。

Schwarzschild 到 Eddington-Finkelstein 时间坐标的输入示例：

```text
新坐标符号:
v, r, theta, phi

新坐标关于旧坐标的表达式:
v = t + 2*M*log(r/(2*M) - 1)
r = r
theta = theta
phi = phi
```

对应的新度规会包含非零交叉项 `g_vr = g_rv = 2M/r`，并且
`g_rr = 1 + 2M/r`。如果使用 null advanced coordinate
`v = t + r + 2*M*log(r/(2*M) - 1)`，则会得到 `g_vr = g_rv = 1`
和 `g_rr = 0`。

坐标变换页的示例下拉还包含可直接计算的逆变换示例：

- Schwarzschild 近视界 Rindler 近似（Rindler 有时会误拼成 Rinder）：
  这个示例的原度规不是完整 Schwarzschild 度规，而是先在 `r≈2M` 和小角片
  `sin(theta)≈theta` 下近似为
  ```text
  ds^2 ≈ -(r-2M)/(2M) dt^2 + 2M/(r-2M) dr^2
         + (2M)^2(dtheta^2 + theta^2 dphi^2)
  ```
  ```text
  输入方向:
  旧坐标 = 新坐标函数（直接计算）

  旧坐标关于新坐标的表达式:
  t = 4*M*omega
  r = 2*M + rho^2/(8*M)
  theta = sqrt(x^2 + y^2)/(2*M)
  phi = atan(y/x)
  ```
  这等价于课堂里的
  `rho≈2*sqrt(2*M*(r-2*M))`、`omega=t/(4*M)`、
  `x=2*M*theta*cos(phi)`、`y=2*M*theta*sin(phi)`，会得到
  `ds^2≈-rho^2 d omega^2 + d rho^2 + dx^2 + dy^2`。其中
  `phi = atan(y/x)` 只是局部角片中的一个分支写法。
- Kruskal-Szekeres 坐标：
  ```text
  输入方向:
  旧坐标 = 新坐标函数（直接计算）

  旧坐标关于新坐标的表达式:
  t = 4*M*atanh(T/X)
  r = 2*M*(1 + LambertW((X^2 - T^2)/E))
  theta = theta
  phi = phi
  ```
  这里采用常见外区分支；程序按给定逆变换计算，不额外判断物理区域。
