#!/usr/bin/env python3
"""Replay a completed Stage 1 with its saved ROI and prepared geometry, in fresh output roots."""
import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import yaml

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from gsrefine.roi.core import file_sha256
from gsrefine.roi_init.results import write_atomic
from gsrefine.roi_init.workflow import output_for_roi, run_once


def read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def comparison(original, replay):
    left, right = (read(d / "metadata/manifest.json") for d in (original, replay))
    a, b = left["stage2_inputs"], right["stage2_inputs"]
    checks = {"source_hashes": left["input_sha256"] == right["input_sha256"],
              "final_ply_bytes": (file_sha256(a["gaussian_ply"]) == left["output_sha256"]
                                  == right["output_sha256"] == file_sha256(b["gaussian_ply"]))}
    for key in ("r3_local_safety_npz", "sfm_roi_npz"):
        with np.load(a[key], allow_pickle=False) as x, np.load(b[key], allow_pickle=False) as y:
            checks[key + ".fields"] = set(x.files) == set(y.files)
            for name in sorted(set(x.files) & set(y.files)):
                first, second = x[name], y[name]
                if name == "source_local_safety_sha256":
                    # Archives may have different container metadata. Verify the
                    # digest links in each run; the handoff arrays are compared above.
                    ok = all(str(v.item()) == file_sha256(src["r3_local_safety_npz"])
                             for v, src in ((first, a), (second, b)))
                else:
                    ok = (first.dtype == second.dtype and first.shape == second.shape
                          and first.tobytes() == second.tobytes())
                checks[key + "." + name] = ok
    return checks


def compare_run(source, parent, baseline):
    old = read(source / "metadata/manifest.json")
    if old.get("status") != "completed":
        raise ValueError("A completed Stage-1 source manifest is required")
    config = (source / "metadata/effective_scene.yaml").resolve(strict=True)
    roi = Path(old["stage2_inputs"]["roi_json"]).resolve(strict=True)
    if file_sha256(roi) != old["input_sha256"]["roi"]:
        raise ValueError("The saved ROI differs from the completed run")
    cfg = yaml.safe_load(config.read_text(encoding="utf8"))
    for key, field in (("gaussian_ply", "source_full_gaussian_ply"),
                       ("poisson_raw", "poisson_mesh"), ("clean_sfm_npz", "clean_sfm_npz")):
        if Path(cfg["paths"][key]).resolve(strict=True) != Path(old["stage2_inputs"][field]).resolve(strict=True):
            raise ValueError(f"Input path differs from saved run: {key}")
    mode = (cfg.get("module1") or {}).get("mode", "reuse")
    if mode not in ("reuse", "prepared"):
        raise ValueError("Use the effective prepared geometry configuration from the completed run")
    timings, outputs = {}, {}
    if baseline == "saved-v1":
        timing = read(source / "metadata/stage1_timing.json")
        if timing.get("accelerated") is not True or "handoff_and_publication" in timing:
            raise ValueError("saved-v1 needs the optimized V1 run directory with its original timing report")
        if set(timing.get("mask_sha256", {})) != {"R0", "core", "safety_added", "R3"}:
            raise ValueError("The saved V1 run is missing its four mask digests")
        timings["reference"] = timing
        outputs["reference"] = source
    parent.mkdir(parents=True, exist_ok=True)
    job = Path(tempfile.mkdtemp(prefix="stage1_ab_", dir=parent))
    print("Output:", job, flush=True)
    variants = ("optimized",) if baseline == "saved-v1" else ("optimized", "reference")
    for variant in variants:
        root = job / variant
        subprocess.run([sys.executable, str(Path(__file__).resolve()), str(source),
                        "--worker", variant, "--output-root", str(root), "--baseline", baseline], check=True)
        out = output_for_roi(config, roi, root)
        outputs[variant] = out
        timings[variant] = read(out / "metadata/stage1_timing.json")
    checks = comparison(source, outputs["optimized"])
    paired = (dict(checks) if baseline == "saved-v1"
              else comparison(outputs["reference"], outputs["optimized"]))
    for key in ("R0", "core", "safety_added", "R3"):
        first, second = (timings[k]["mask_sha256"].get(key) for k in ("reference", "optimized"))
        paired["mask." + key] = bool(first) and first == second
    fast, ref = (timings[k]["elapsed_to_publication_s"] for k in ("optimized", "reference"))
    ok = all(checks.values()) and all(paired.values())
    result = {"status": "PASS" if ok else "MISMATCH", "source_run": str(source),
              "reference_mode": baseline, "optimized_version": 2,
              "historical_output_checks": checks, "paired_checks": paired,
              "timings": timings, "seconds_saved": ref-fast, "speedup": ref/fast,
              "optimized_ply": str(outputs["optimized"] / "final/gaussian_roi.ply")}
    report = job / "comparison.json"
    write_atomic(report, result)
    print(f"{result['status']}  reference={ref:.3f}s  optimized={fast:.3f}s  "
          f"saved={ref-fast:.3f}s  speedup={ref/fast:.3f}x")
    print(f"{'ROI confirmed through publication':34} {'reference/s':>12} {'optimized/s':>12}")
    for phase in timings["optimized"]["phases_s"]:
        print(f"{phase:34} {timings['reference']['phases_s'][phase]:12.3f}"
              f" {timings['optimized']['phases_s'][phase]:12.3f}")
    print("Baseline:", baseline, "  optimized: v2")
    if baseline == "saved-v1":
        print("Reference time is from the saved V1 run; disk cache and load may differ.")
    detail = timings["optimized"]["handoff_and_publication"]
    for label, entry in detail["steps"].items():
        print(f"{label:36} {entry['exclusive_s']:8.3f}s exclusive")
    print("Same-version file digest hits:", detail["hash_cache_hits"])
    print("Optimized safety:", timings["optimized"]["safety"])
    print("Report:", report)
    print("PLY:", result["optimized_ply"])
    if not ok:
        print("Failed checks:", [f"{run}.{key}" for run, group in (("historic", checks),
                                ("paired", paired)) for key, good in group.items() if not good])
        return 1
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("source_run", type=Path, help="Original completed roi_NNN directory")
    ap.add_argument("--output-root", type=Path, help="Parent of a NEW comparison directory")
    ap.add_argument("--baseline", choices=("exhaustive", "v1", "saved-v1"), default="exhaustive",
                    help="v1 reruns both selectors. saved-v1 replays only v2 against a saved accelerated v1 run")
    ap.add_argument("--worker", choices=("reference", "optimized"), help=argparse.SUPPRESS)
    args = ap.parse_args()
    source = args.source_run.expanduser().resolve(strict=True)
    if args.worker:
        config = source / "metadata/effective_scene.yaml"
        result = read(source / "metadata/workflow_result.json")
        roi = read(source / "metadata/manifest.json")["stage2_inputs"]["roi_json"]
        run_once(config, roi, args.output_root,
                 reuse_scene_roi=result.get("reused_roi_geometry", False),
                 accelerate_stage1=args.worker == "optimized" or args.baseline == "v1",
                 optimization_version=2 if args.worker == "optimized" else 1)
        return 0
    return compare_run(source, (args.output_root or source.parent/"stage1_comparisons").resolve(), args.baseline)


if __name__ == "__main__":
    sys.exit(main())
