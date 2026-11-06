"""Dependency and data-path diagnostics; never changes the Conda environment.

Examples:
    python run.py doctor
    python run.py doctor --check-data
    python run.py doctor --check-data --strict
    python run.py doctor --json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_CONDA_ENV = "gsrefine"
MODULES = {
    "numpy": "numpy", "scipy": "scipy", "yaml": "PyYAML",
    "trimesh": "trimesh", "shapely": "shapely", "open3d": "open3d",
    "plyfile": "plyfile", "skimage": "scikit-image", "pytest": "pytest",
}


def inspect_project(config: str | None = None, check_data: bool = False):
    """Return a nonmutating, serializable health report."""
    conda = os.environ.get("CONDA_DEFAULT_ENV")
    report = {
        "project_root": str(ROOT),
        "expected_conda_env": EXPECTED_CONDA_ENV,
        "active_conda_env": conda,
        "python_executable": sys.executable,
        "python_version": sys.version.split()[0],
        "python_expected_3_10": sys.version_info[:2] == (3, 10),
        "package_available": {name: importlib.util.find_spec(name) is not None for name in MODULES},
        "node_executable": shutil.which("node"),
        "npm_executable": shutil.which("npm"),
        "playcanvas_installed": (ROOT / "ui/node_modules/playcanvas/build/playcanvas.mjs").is_file(),
        "splat_transform_installed": (ROOT / "ui/node_modules/.bin/splat-transform").is_file(),
        "node_version": None,
        "node_22_compatible": False,
        "source_tree": {name: (ROOT / name).exists() for name in (
            "run.py", "environment.yml", "configs", "gsrefine/roi/core.py",
            "gsrefine/geometry/poisson.py", "gsrefine/web/server.py",
            "ui/app.mjs", "tests", "integrations")},
        "checks": [],
    }
    if report['node_executable']:
        try:
            version = subprocess.check_output([report['node_executable'], '--version'], text=True, timeout=5).strip().lstrip('v')
            report['node_version'] = version
            major = int(version.split('.')[0])
            report['node_22_compatible'] = (22 <= major < 23)
        except (subprocess.SubprocessError, ValueError, OSError):
            pass
    if not report['node_22_compatible']:
        report['checks'].append({'level': 'warning', 'message': 'Node.js 22 is required for full-resolution SOG conversion; run bash bootstrap.sh --sync-deps'})
    if not report['splat_transform_installed']:
        report['checks'].append({'level': 'warning', 'message': 'SplatTransform is missing; full-resolution SOG display cache cannot be generated'})
    if conda and conda != EXPECTED_CONDA_ENV:
        report["checks"].append({"level": "warning", "message": f"Active Conda environment is '{conda}', expected '{EXPECTED_CONDA_ENV}'"})
    if not report["python_expected_3_10"]:
        report["checks"].append({"level": "warning", "message": "Project environment.yml specifies Python 3.10"})
    for pkg, available in report["package_available"].items():
        if not available:
            report["checks"].append({"level": "error", "message": f"Python dependency missing: {pkg} ({MODULES[pkg]})"})
    if not report["playcanvas_installed"]:
        report["checks"].append({"level": "warning", "message": "PlayCanvas project-local dependency missing; browser UI will not start"})
    for path, present in report["source_tree"].items():
        if not present:
            report["checks"].append({"level": "error", "message": f"Missing source component: {path}"})
    if check_data:
        target = Path(config) if config else ROOT / "configs/songziguanyin.yaml"
        if not target.is_absolute():
            target = (Path.cwd() / target).resolve()
        report["config_file"] = str(target)
        if not target.is_file():
            report["checks"].append({"level": "error", "message": f"Config not found: {target}"})
        elif importlib.util.find_spec("yaml") is None:
            report["checks"].append({"level": "error", "message": "PyYAML missing; cannot inspect config paths"})
        else:
            import yaml
            cfg = yaml.safe_load(target.read_text(encoding="utf-8")) or {}
            paths = cfg.get("paths") or {}
            status = {}
            # Only INPUTS are checked: never create/modify them.
            for key in ("gaussian_ply", "poisson_raw", "clean_sfm_npz"):
                value = paths.get(key)
                if value:
                    p = Path(value).expanduser()
                    if not p.is_absolute():
                        p = (ROOT / p).resolve()
                    status[key] = {"path": str(p), "exists": p.is_file()}
                    if not p.is_file():
                        report["checks"].append({"level": "error", "message": f"Input missing: {key} -> {p}"})
                else:
                    status[key] = {"path": None, "exists": False}
                    report["checks"].append({"level": "error", "message": f"Input path not configured: {key}"})
            report["data_inputs"] = status
            report["source_model_scope"] = cfg.get("source_model_scope", "unknown")
            if report["source_model_scope"] != "full_original":
                report["checks"].append({"level": "warning", "message": "Model is not declared full_original; previously AABB-cropped Gaussians cannot be recovered"})
    report["error_count"] = sum(item["level"] == "error" for item in report["checks"])
    report["warning_count"] = sum(item["level"] == "warning" for item in report["checks"])
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None, help="Config to inspect with --check-data")
    ap.add_argument("--check-data", action="store_true", help="Also verify real scene input files, without reading large data")
    ap.add_argument("--no-config", action="store_true", help="Do not inspect scene input paths (default unless --check-data)")
    ap.add_argument("--strict", action="store_true", help="Exit nonzero if any hard error is detected")
    ap.add_argument("--json", action="store_true", help="Machine-readable report")
    args = ap.parse_args()
    if args.no_config and args.check_data:
        ap.error("--no-config and --check-data are mutually exclusive")
    report = inspect_project(args.config, check_data=args.check_data)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(f"GSSceneRefine diagnostic (read-only)\n  root: {report['project_root']}\n  Python: {report['python_version']} ({report['python_executable']})\n  Conda: {report['active_conda_env'] or 'not identified'} (expected gsrefine)")
        print(f"  Python dependencies: {sum(report['package_available'].values())}/{len(report['package_available'])} available")
        print(f"  PlayCanvas dependency: {'OK' if report['playcanvas_installed'] else 'not installed'}")
        print(f"  Node.js: {report['node_version'] or 'missing'} (required: 22.x)")
        print(f"  SplatTransform dependency: {'OK' if report['splat_transform_installed'] else 'not installed'}")
        if "data_inputs" in report:
            for key, data in report["data_inputs"].items():
                print(f"  {key}: {'FOUND' if data['exists'] else 'MISSING'} {data['path']}")
        for entry in report["checks"]:
            print(f"  {entry['level'].upper()}: {entry['message']}")
        print(f"  Result: {report['error_count']} error(s), {report['warning_count']} warning(s)")
    return int(args.strict and report["error_count"] > 0)


if __name__ == "__main__":
    raise SystemExit(main())
