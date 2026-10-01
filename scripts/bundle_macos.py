#!/usr/bin/env python3
"""Assemble and sign a standalone macOS bundle using only Python's standard library.

The Python interpreter is a developer tool; the app ships only Swift/C++ executables,
the native equation fonts, and transitive non-system dynamic libraries.
"""

from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path
import plistlib
import shutil
import subprocess
import tempfile


def run(*arguments: str, capture: bool = False) -> str:
    result = subprocess.run(arguments, check=True, text=True,
                            stdout=subprocess.PIPE if capture else None)
    return result.stdout.strip() if capture else ""


def dependencies(binary: Path) -> list[str]:
    lines = run("/usr/bin/otool", "-L", str(binary), capture=True).splitlines()[1:]
    return [line.strip().split(" (", 1)[0] for line in lines if line.strip()]


def dylib_id(binary: Path) -> str | None:
    if binary.suffix != ".dylib":
        return None
    lines = run("/usr/bin/otool", "-D", str(binary), capture=True).splitlines()
    return lines[1].strip() if len(lines) > 1 else None


def minimum_macos_version(binary: Path) -> str:
    """A copied dependency can require a newer OS than the app build target."""
    versions = ["14.0"]
    command = ""
    for line in run("/usr/bin/otool", "-l", str(binary), capture=True).splitlines():
        fields = line.strip().split()
        if len(fields) == 2 and fields[0] == "cmd":
            command = fields[1]
        elif len(fields) == 2 and ((command == "LC_BUILD_VERSION" and fields[0] == "minos")
                                  or (command == "LC_VERSION_MIN_MACOSX" and fields[0] == "version")):
            versions.append(fields[1])
    return max(versions, key=lambda value: tuple(int(part) for part in value.split(".")))


def rpaths(binary: Path) -> list[str]:
    lines = run("/usr/bin/otool", "-l", str(binary), capture=True).splitlines()
    paths = []
    for index, line in enumerate(lines):
        if line.strip() == "cmd LC_RPATH":
            for candidate in lines[index + 1:index + 4]:
                if candidate.strip().startswith("path "):
                    paths.append(candidate.strip()[5:].split(" (offset", 1)[0])
    return paths


def is_system_dependency(dependency: str) -> bool:
    return dependency.startswith(("/usr/lib/", "/System/Library/", "/Library/Apple/System/Library/"))


def resolve_dependency(dependency: str, source: Path, executable_directory: Path) -> Path:
    def expand(path: str) -> Path:
        path = path.replace("@loader_path", str(source.parent)).replace("@executable_path", str(executable_directory))
        return Path(path)

    if dependency.startswith("@rpath/"):
        relative = dependency[len("@rpath/"):]
        candidates = [expand(path) / relative for path in rpaths(source)]
        candidates += [source.parent / relative, executable_directory / relative]
    else:
        candidates = [expand(dependency)]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise RuntimeError(f"Cannot locate dynamic dependency {dependency!r} required by {source}")


def copy_dynamic_libraries(entrypoints: list[tuple[Path, Path]], frameworks: Path) -> list[Path]:
    """Walk the original dependency graph and rewrite every packaged load command.

    Executables use @executable_path; library-to-library links use @loader_path,
    so a copied application remains usable after moving it to another machine.
    """
    copied: dict[Path, Path] = {}
    claimed_names: dict[str, Path] = {}
    queue = [(source.resolve(), target, source.parent.resolve()) for source, target in entrypoints]
    for source, target, executable_directory in queue:
        own_id = dylib_id(source)
        for dependency in dependencies(source):
            if dependency == own_id or is_system_dependency(dependency):
                continue
            original = resolve_dependency(dependency, source, executable_directory)
            if original.suffix != ".dylib":
                raise RuntimeError(f"Unsupported non-system framework dependency: {original}")
            if original not in copied:
                if original.name in claimed_names and claimed_names[original.name] != original:
                    raise RuntimeError(f"Two different dynamic libraries use the same filename: {original.name}")
                claimed_names[original.name] = original
                destination = frameworks / original.name
                shutil.copy2(original, destination)
                destination.chmod(0o755)
                copied[original] = destination
                queue.append((original, destination, executable_directory))
            destination = copied[original]
            replacement = ("@loader_path/" if target.parent == frameworks else "@executable_path/../Frameworks/") + destination.name
            run("/usr/bin/install_name_tool", "-change", dependency, replacement, str(target))
        if own_id:
            run("/usr/bin/install_name_tool", "-id", "@rpath/" + target.name, str(target))

    # A clean launch must never depend on the build machine's Homebrew directory.
    for binary in [target for _, target in entrypoints] + list(copied.values()):
        own_id = dylib_id(binary)
        for dependency in dependencies(binary):
            if dependency != own_id and not is_system_dependency(dependency) and not dependency.startswith(("@loader_path/", "@executable_path/../Frameworks/")):
                raise RuntimeError(f"Unbundled dependency remains in {binary}: {dependency}")
    return list(copied.values())


def cmake_source_directory(build_dir: Path, keys: list[str], fallback: str) -> Path:
    """Respect both FetchContent defaults and explicit local source overrides."""
    cache = build_dir / "CMakeCache.txt"
    values: dict[str, str] = {}
    if cache.is_file():
        for raw_line in cache.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith(("#", "//")) or "=" not in line:
                continue
            declaration, value = line.split("=", 1)
            values[declaration.split(":", 1)[0]] = value
    for key in keys:
        if values.get(key):
            return Path(values[key]).resolve()
    return build_dir / "_deps" / fallback


def copy_licenses(project_root: Path, build_dir: Path, gmp_prefix: Path, destination: Path) -> None:
    destination.mkdir()
    swiftmath = project_root / "native/.build/checkouts/SwiftMath"
    symengine = cmake_source_directory(build_dir, ["symengine_SOURCE_DIR", "FETCHCONTENT_SOURCE_DIR_SYMENGINE"], "symengine-src")
    json_source = cmake_source_directory(build_dir, ["json_SOURCE_DIR", "nlohmann_json_SOURCE_DIR", "FETCHCONTENT_SOURCE_DIR_JSON"], "json-src")
    entries = [
        (symengine / "LICENSE", "SymEngine-LICENSE.txt"),
        (json_source / "LICENSE.MIT", "nlohmann-json-LICENSE.txt"),
        (swiftmath / "LICENSE", "SwiftMath-LICENSE.txt"),
        (swiftmath / "Sources/SwiftMath/mathFonts.bundle/GUST-FONT-LICENSE.txt", "Fonts-GUST-LICENSE.txt"),
        (swiftmath / "Sources/SwiftMath/mathFonts.bundle/OFL.txt", "Fonts-OFL.txt"),
        (swiftmath / "Sources/SwiftMath/mathFonts.bundle/LICENSE", "Fonts-LICENSE.txt"),
        (gmp_prefix / "COPYING", "GMP-COPYING.txt"),
        (gmp_prefix / "COPYING.LESSERv3", "GMP-COPYING.LESSERv3.txt"),
    ]
    for source, name in entries:
        if not source.is_file():
            raise RuntimeError(f"Required dependency license is missing: {source}")
        shutil.copy2(source, destination / name)
    gmp_version = gmp_prefix.resolve().name
    notice = f"""Tensor Calculator bundled third-party components

SymEngine 0.14.0: https://github.com/symengine/symengine
nlohmann/json 3.12.0: https://github.com/nlohmann/json
SwiftMath 1.7.3 and its bundled mathematical fonts:
https://github.com/mgriebling/SwiftMath
GMP {gmp_version}: https://gmplib.org/
GMP corresponding source: https://gmplib.org/download/gmp/gmp-{gmp_version}.tar.xz

The license texts and font notices accompany this file. GMP is dynamically
linked from Contents/Frameworks; it is not statically incorporated into the
application. Its library may be replaced with a compatible build. A modified
local app can be re-signed with: codesign --force --deep --sign - <app path>.

The app has no Python runtime dependency. Python is used only by developer
build and reference-verification tools outside the distributed application.
"""
    (destination / "NOTICE.txt").write_text(notice, encoding="utf-8")


def sign(path: Path, identity: str) -> None:
    arguments = ["/usr/bin/codesign", "--force", "--sign", identity]
    if identity != "-":
        arguments += ["--options", "runtime", "--timestamp"]
    run(*arguments, str(path))


def bundle(args: argparse.Namespace) -> Path:
    project_root = args.project_root.resolve()
    build_dir = args.build_dir.resolve()
    swift_bin = args.swift_bin_path.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    gui = swift_bin / "TensorCalculator"
    worker = build_dir / "core/tensor-worker"
    fonts_bundle = swift_bin / "SwiftMath_SwiftMath.bundle"
    for artifact in [gui, worker, fonts_bundle]:
        if not artifact.exists():
            raise RuntimeError(f"Release artifact is missing: {artifact}. Run scripts/build_macos.sh first.")

    with tempfile.TemporaryDirectory(prefix=".tensor-package-", dir=output) as staging_directory:
        app = Path(staging_directory) / "Tensor Calculator.app"
        contents = app / "Contents"
        executables = contents / "MacOS"
        frameworks = contents / "Frameworks"
        resources = contents / "Resources"
        for directory in [executables, frameworks, resources]:
            directory.mkdir(parents=True)
        bundled_gui = executables / "TensorCalculator"
        bundled_worker = executables / "tensor-worker"
        shutil.copy2(gui, bundled_gui)
        shutil.copy2(worker, bundled_worker)
        bundled_gui.chmod(0o755)
        bundled_worker.chmod(0o755)
        shutil.copytree(fonts_bundle, resources / fonts_bundle.name)
        libraries = copy_dynamic_libraries([(gui, bundled_gui), (worker, bundled_worker)], frameworks)

        gui_architectures = set(run("/usr/bin/lipo", "-archs", str(bundled_gui), capture=True).split())
        for binary in [bundled_worker] + libraries:
            architectures = set(run("/usr/bin/lipo", "-archs", str(binary), capture=True).split())
            if not gui_architectures.issubset(architectures):
                raise RuntimeError(f"Architecture mismatch: UI {gui_architectures}, {binary.name} {architectures}")

        minimum_os = max(
            (minimum_macos_version(binary) for binary in [bundled_gui, bundled_worker] + libraries),
            key=lambda value: tuple(int(part) for part in value.split(".")),
        )

        iconset = Path(staging_directory) / "AppIcon.iconset"
        run("/usr/bin/xcrun", "swift", str(project_root / "scripts/make_icon.swift"), str(iconset))
        run("/usr/bin/iconutil", "--convert", "icns", str(iconset), "--output", str(resources / "AppIcon.icns"))

        info = {
            "CFBundleName": "Tensor Calculator",
            "CFBundleDisplayName": "Tensor Calculator",
            "CFBundleIdentifier": "org.tensorcalculator.mac",
            "CFBundleExecutable": "TensorCalculator",
            "CFBundleIconFile": "AppIcon.icns",
            "CFBundlePackageType": "APPL",
            "CFBundleShortVersionString": "0.1.0",
            "CFBundleVersion": "1",
            "LSMinimumSystemVersion": minimum_os,
            "LSApplicationCategoryType": "public.app-category.education",
            "NSHighResolutionCapable": True,
            "NSPrincipalClass": "NSApplication",
            "CFBundleDocumentTypes": [{
                "CFBundleTypeName": "Tensor Calculation",
                "CFBundleTypeRole": "Editor",
                "LSHandlerRank": "Owner",
                "LSItemContentTypes": ["org.tensorcalculator.calculation"],
            }],
            "UTExportedTypeDeclarations": [{
                "UTTypeIdentifier": "org.tensorcalculator.calculation",
                "UTTypeDescription": "Tensor Calculation",
                "UTTypeConformsTo": ["public.json"],
                "UTTypeTagSpecification": {
                    "public.filename-extension": ["tensorcalc"],
                    "public.mime-type": "application/json",
                },
            }],
        }
        with (contents / "Info.plist").open("wb") as file:
            plistlib.dump(info, file, sort_keys=False)
        (contents / "PkgInfo").write_bytes(b"APPL????")
        copy_licenses(project_root, build_dir, args.gmp_prefix.resolve(), resources / "Licenses")
        build_info = {
            "version": "0.1.0",
            "architectures": sorted(gui_architectures),
            "minimum_macos_version": minimum_os,
            "built_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "dependencies": {"SymEngine": "0.14.0", "nlohmann/json": "3.12.0", "SwiftMath": "1.7.3"},
            "bundled_libraries": [library.name for library in libraries],
        }
        (resources / "build-info.json").write_text(json.dumps(build_info, indent=2) + "\n", encoding="utf-8")
        run("/usr/bin/plutil", "-lint", str(contents / "Info.plist"))
        for library in libraries:
            sign(library, args.sign_identity)
        sign(bundled_worker, args.sign_identity)
        sign(resources / fonts_bundle.name, args.sign_identity)
        sign(app, args.sign_identity)
        run("/usr/bin/codesign", "--verify", "--deep", "--strict", str(app))

        destination = output / app.name
        backup = output / ".Tensor Calculator.previous.app"
        if backup.exists():
            shutil.rmtree(backup)
        if destination.exists():
            destination.rename(backup)
        try:
            app.rename(destination)
        except Exception:
            if backup.exists():
                backup.rename(destination)
            raise
        if backup.exists():
            shutil.rmtree(backup)
        return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--swift-bin-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gmp-prefix", type=Path, required=True)
    parser.add_argument("--sign-identity", default="-")
    args = parser.parse_args()
    print(f"Application ready: {bundle(args)}")


if __name__ == "__main__":
    main()
