#!/bin/bash
# Developer build tool; the resulting application needs no Python or Homebrew.
set -euo pipefail

tensor_root="$(cd "$(dirname "$0")/.." && pwd)"
tensor_build_dir="${BUILD_DIR:-$tensor_root/build}"
tensor_dist_dir="${DIST_DIR:-$tensor_root/dist}"
tensor_sign_identity="${CODE_SIGN_IDENTITY:--}"
tensor_package_only=false

case "${1:-}" in
    --package-only) tensor_package_only=true ;;
    --help|-h)
        cat <<'USAGE'
Usage: scripts/build_macos.sh [--package-only]

Builds the C++ engine and Swift UI, then creates dist/Tensor Calculator.app.
Developer requirements: macOS, Xcode command-line tools, CMake, Homebrew GMP,
and Python 3 (used to assemble the bundle; never shipped with the app).

Optional environment variables:
  BUILD_DIR          CMake build directory (default: project/build)
  DIST_DIR           Output directory (default: project/dist)
  BUILD_JOBS         Parallel build jobs (default: up to 8 logical CPUs)
  CODE_SIGN_IDENTITY  Signing identity (default: '-' for local ad-hoc signing)

--package-only reassembles already-built Release artifacts.
USAGE
        exit 0 ;;
    "") ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
esac

if [ "$(uname -s)" != Darwin ]; then
    echo "The native application must be built on macOS." >&2
    exit 1
fi
for tensor_tool in cmake swift python3 brew xcrun; do
    if ! command -v "$tensor_tool" >/dev/null 2>&1; then
        echo "Missing developer tool: $tensor_tool" >&2
        exit 1
    fi
done
tensor_brew_prefix="$(brew --prefix)"
tensor_gmp_prefix="$(brew --prefix gmp)"
if [ ! -f "$tensor_gmp_prefix/include/gmp.h" ]; then
    echo "GMP development files are missing. Install the Homebrew gmp package." >&2
    exit 1
fi
tensor_jobs="${BUILD_JOBS:-$(sysctl -n hw.logicalcpu)}"
if [ -z "${BUILD_JOBS:-}" ] && [ "$tensor_jobs" -gt 8 ]; then tensor_jobs=8; fi
mkdir -p "$tensor_build_dir/swift-clang-cache"
export CLANG_MODULE_CACHE_PATH="$tensor_build_dir/swift-clang-cache"

if [ "$tensor_package_only" = false ]; then
    cmake -S "$tensor_root" -B "$tensor_build_dir" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_OSX_DEPLOYMENT_TARGET=14.0 \
        -DCMAKE_PREFIX_PATH="$tensor_brew_prefix${CMAKE_PREFIX_PATH:+;$CMAKE_PREFIX_PATH}"
    cmake --build "$tensor_build_dir" --parallel "$tensor_jobs" --target tensor-worker
    swift build --package-path "$tensor_root/native" --configuration release --jobs "$tensor_jobs"
fi

# SwiftPM layouts differ between toolchain versions; never hard-code .build paths.
tensor_swift_bin="$(swift build --package-path "$tensor_root/native" --configuration release --show-bin-path)"
python3 "$tensor_root/scripts/bundle_macos.py" \
    --project-root "$tensor_root" \
    --build-dir "$tensor_build_dir" \
    --swift-bin-path "$tensor_swift_bin" \
    --output-dir "$tensor_dist_dir" \
    --gmp-prefix "$tensor_gmp_prefix" \
    --sign-identity "$tensor_sign_identity"
