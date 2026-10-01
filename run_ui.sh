#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
if [[ "$(uname -s)" == "Darwin" ]]; then
    if [[ ! -x "dist/Tensor Calculator.app/Contents/MacOS/TensorCalculator" ]]; then
        ./scripts/build_macos.sh
    fi
    exec open "dist/Tensor Calculator.app"
fi
# The previous Python interface remains available on other platforms.
exec .venv/bin/python GR_caculator.py
