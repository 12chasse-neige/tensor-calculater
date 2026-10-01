# Tensor Calculator for macOS

A native macOS app for exact symbolic calculations from a coordinate metric.
The interface uses SwiftUI, an AppKit metric editor, and native SwiftMath
typesetting. A separate C++20 worker uses **SymEngine 0.14.0** to calculate:

- Inverse metric and Christoffel symbols
- Riemann and Ricci tensors
- Ricci scalar and Kretschmann scalar

The native app contains only the tensor calculation workflow. Its runtime
does not require Python, Homebrew, a terminal, or an internet connection.
The obsolete Python application has been removed. Python scripts under `scripts/` and `tests/` are development tools only.

## Download version 0.1.0

Download the Apple Silicon app from the [v0.1.0 release](https://github.com/12chasse-neige/tensor-calculater/releases/tag/v0.1.0).
Extract the ZIP and move **Tensor Calculator.app** to Applications. The binary
requires macOS 15 or later and is signed ad hoc; it has not been notarized by
Apple. The release includes SHA-256 checksums and [release notes](docs/releases/v0.1.0.md).

## Build and run

The current bundle requires macOS 15 or later. Building requires Xcode or its
command-line tools with Swift 5.9+,
CMake 3.24+, GMP, and Python 3 for the development packaging script. Network
access is needed on the first build to fetch the pinned dependencies.

```sh
brew install cmake ninja gmp
./scripts/build_macos.sh
open "dist/Tensor Calculator.app"
```

`./run_ui.sh` opens the built app, building it first if necessary. Re-run the
build script after editing source. The build uses the current Mac's CPU
architecture. It bundles and relinks non-system libraries, includes third-party
license notices, and signs the local app ad hoc. The minimum macOS version is
derived from the UI, worker, and bundled libraries (the installed GMP build
currently requires macOS 15). To use a Developer ID identity,
set `CODE_SIGN_IDENTITY` when building. Notarization is a separate release step.

## Using the app

Choose a preset or enter coordinates, constants, unknown functions, and a
**symmetric covariant metric** in the same coordinate order. Select the desired
outputs and click Calculate (`⌘↩`). Cancel (`⌘.`) stops the worker; the app remains
available for another calculation.

Schwarzschild, in geometric units:

```text
Coordinates: t, r, theta, phi
Constants: M
Functions:
Metric:
[-(1 - 2M/r), 0, 0, 0],
[0, 1/(1 - 2M/r), 0, 0],
[0, 0, r^2, 0],
[0, 0, 0, r^2*sin(theta)^2]
```

Built-in presets: spherical-coordinate flat spacetime, Schwarzschild,
Reissner–Nordström, Kerr, FLRW with arbitrary `a(t)`, and a two-sphere.
Results can be searched and copied as expressions or LaTeX. Save/open
`.tensorcalc` documents with native File menu commands. Documents preserve the
inputs that produced the saved results; editing inputs marks those results as
belonging to the previous calculation.

### Input grammar

- ASCII symbol names; explicitly declare constants and unknown functions.
- Exact integers, fractions, decimal numbers, and scientific notation. Decimal
  literals are interpreted as exact rationals (for example `0.1 = 1/10`).
- `+`, `-`, `*`, `/`, `^` or `**`, parentheses, and implicit multiplication such
  as `2M` and `2(r+1)`.
- `sin`, `cos`, `tan`, `cot`, `sec`, `csc`, `asin`, `acos`, `atan`, `sinh`,
  `cosh`, `tanh`, `exp`, `sqrt`, `log`/`ln`, `Abs`/`abs`, and constants `pi`,
  `E`, `I`.
- Unknown functions declared as `a(t)` or `Phi(t,r)`. Function calls respect
  their declared coordinate dependencies and preserve symbolic derivatives.
- `diff(expression, coordinate[, order])`, mixed derivatives such as
  `diff(Phi(t,r),t,r)`, `Derivative(...)`, and `Rational(p,q)`.
- A row per line, comma-separated rows, nested `[[...],[...]]`,
  `Matrix([[...],[...]])`, or `diag(g00,g11,...)`.

This is a restricted mathematical grammar, not arbitrary Python syntax. The
worker accepts 1–8 coordinates. Inputs have bounded nesting, literal size,
numeric powers, and derivative orders so malformed expressions can be rejected
before excessive work. Singular, asymmetric, undefined, and nonfinite metrics
return input errors. Symbolically conditional inverses apply only where the
metric and expressions are defined.

## Mathematical conventions and simplification

The native engine preserves the original calculator's convention:

```text
R^rho_{sigma mu nu} = d_nu Gamma^rho_{mu sigma}
                   - d_mu Gamma^rho_{nu sigma}
                   + Gamma^rho_{nu alpha} Gamma^alpha_{mu sigma}
                   - Gamma^rho_{mu alpha} Gamma^alpha_{nu sigma}

Ricci_{sigma nu} = sum_rho R^rho_{sigma rho nu}
```

This convention gives `R = -2/L^2` for a two-sphere of radius `L`. Component
indices in the worker protocol are zero-based; the app displays coordinate
labels. Inverse metric, connection, Riemann, and Ricci index positions are
`uu`, `ull`, `ulll`, and `ll`, where `u` means upper and `l` means lower.

Only **proven symbolic zeros** are omitted. Numerical samples never establish
a symbolic zero. Bounded rational/trigonometric normalization can leave
mathematically zero components unresolved; such expressions remain visible.
The engine calculates only dependencies needed by the selected outputs and
caches metric/expression derivatives during each job. Kerr's recognized
Kretschmann invariant uses the existing closed-form shortcut, explicitly
reported in the result. Other selected Kerr tensors use the general algorithms
and can produce large expressions.

## Development and verification

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$(brew --prefix)"
cmake --build build --target tensor-worker --parallel
python3 -m venv .venv
.venv/bin/python -m pip install sympy==1.14.0
.venv/bin/python -m unittest discover -s tests -v
swift test --package-path native
```

SymPy is a **test-only** dependency. Mathematical tests use analytic invariants
and an independent SymPy implementation to check metric inverses, metric
compatibility, tensor symmetries, the Bianchi identity, Ricci contractions,
non-diagonal metrics, arbitrary-function derivatives, and exact-zero
regressions. Native tests cover streaming results, document provenance,
typesetting, and cancellation/recovery.

After packaging, also run the native model against the bundled engine:

```sh
TENSOR_INTEGRATION_WORKER_PATH="$PWD/dist/Tensor Calculator.app/Contents/MacOS/tensor-worker" \
  swift test --package-path native
```

Set `TENSOR_SNAPSHOT_DIR` to an output directory in that command to render the
metric editor, formula results, and long-expression preview for visual checks.

For an unbundled development UI, set `TENSOR_WORKER_PATH` to the absolute path
of `build/core/tensor-worker` before running the Swift executable.

### Source layout

- `core/`: reusable C++ tensor library and JSON-lines worker
- `native/`: Swift macOS app and native integration tests
- `tests/`: independent mathematical/protocol reference checks
- `scripts/`: reproducible build and app packaging

The worker accepts a version-1 JSON request on each stdin line and writes
progress, result, or error events to stdout. The native app starts a worker for
each calculation; cooperative cancellation is backed by forced termination
when an algebra operation does not return promptly. See
[the protocol specification](docs/worker-protocol.md) for message formats.


### LaTeX input and sign controls

The metric editor accepts bracketed rows, `diag(...)`, or LaTeX `matrix`,
`pmatrix`, and `bmatrix` environments. Use **Use LaTeX** to convert an existing
matrix. The preview next to the editor shows the selected overall metric sign.
For example, with coordinates `theta, phi` and constant `L`:

```latex
\begin{pmatrix}
L^{2} & 0 \\
0 & L^{2}\sin^{2}{\theta}
\end{pmatrix}
```

Supported input includes fractions (`\frac{a}{b}`), square roots (`\sqrt{a}`),
Greek symbol names, explicit function arguments (`\sin{\theta}` or
`\sin(\theta)`), powers, and `\cdot` / `\times`. Unsupported commands produce
an input error. This is a mathematical subset of LaTeX, not a TeX interpreter.

The metric buttons show **(− + + +)** and **(+ − − −)** for four-dimensional
spacetime. Enter the matrix in the first convention; the second multiplies all
entries by −1. For other dimensions, the buttons show **g** and **−g**.
The two-sphere preset is positive definite. Choosing another preset preserves
both sign choices, and the highlighted row follows the selected preset.

The **Riemann** buttons display LaTeX and select either derivative order; the
complete definition is displayed below them. Christoffel and Riemann indices
use separate upper and lower slots, such as `\Gamma^{\alpha}{}_{\beta\,\gamma}`
and `R^{\mu}{}_{\nu\,\rho\,\sigma}`. Component order is preserved, and Copy
LaTeX includes the component label and its expression. Ricci always contracts `R^a_bad`. Reversing the
curvature convention negates Riemann, Ricci, and the Ricci scalar, while leaving
Christoffel symbols and the Kretschmann scalar unchanged. The default preserves
the previous calculator convention. Both selections are saved in documents;
older documents open with the original defaults.
