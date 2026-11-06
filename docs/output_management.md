# GSSceneRefine: Output directories and data ownership

This document describes the stable contract for the four-module `roi-init` / `roi-auto` workflow. Research-only commands (`roi-study`, `roi-debug`, etc.) remain independent and require an explicit output path when used for a separate experiment.

## Basic rules

- The Git repository contains source code, templates, tests and documentation; the **user-provided `--output-dir` is the single run-output root**.
- Input COLMAP data and Gaussian PLY are **read-only**, even if they live outside the output root.
- Each scene has one reusable SfM/Poisson reference and one ROI annotation collection. Individual Gaussian methods produce separate final models and masks; a mask from one method is never transferred to another.
- **`roi-auto --output-dir` is a root directory**, exactly like `roi-init --output-dir`, not the path for an individual ROI run. The software computes the run subdirectory.
- Existing scene + method + ROI results are not overwritten silently.
- Success is the validated `metadata/manifest.json` plus `metadata/workflow_result.json`; a PLY alone is not proof of successful completion.
- Module 3 R3's scientific calculations are independent of the output layout.

## Root layout

```text
<user output root>/
└── <scene>/
    ├── geometry/
    │   ├── cleaned_sfm.npz
    │   ├── poisson_surface.ply
    │   ├── orientation/               # orientation cache
    │   ├── metadata/                  # cache identity + quality records
    │   ├── runtime/                   # immutable per-method effective configs
    │   └── logs/
    ├── roi/
    │   ├── manifest.json             # confirmed annotation catalogue
    │   └── roi_001.json              # reusable 2D polygon constraints
    └── methods/
        ├── absgs/
        │   └── roi_001/
        │       ├── final/gaussian_roi.ply
        │       ├── metadata/
        │       │   ├── roi.json
        │       │   ├── effective_scene.yaml
        │       │   ├── manifest.json
        │       │   ├── processing_report.json
        │       │   └── workflow_result.json
        │       └── logs/
        │           ├── processing.log
        │           └── workflow_progress.json
        └── vanilla/
            └── roi_001/
                ├── final/gaussian_roi.ply
                ├── metadata/
                └── logs/
```

The resulting `metadata/manifest.json` provides explicit `stage2_inputs` for subsequent anomaly removal: `gaussian_ply`, `roi_json`, `poisson_mesh`, and `clean_sfm_npz`. A `gaussian_roi.ply` is an ROI extraction, **not yet a cleaned/anomaly-free Gaussian scene**.

## Diagnostics and temporary files

By default, the automatic four-stage workflow exports the final PLY and compact metadata. Internal masks, candidate classification PLYs, tables, and intermediate indices are written to a **managed temporary directory**, then discarded when processing is complete. `--keep-diagnostics` retains those research outputs under `diagnostics/surfaces`, `diagnostics/masks`, `diagnostics/geometry` and `diagnostics/reports` without changing any algorithm thresholds or output Gaussian set.

Module 1 retains reusable geometry and its quality/identity metadata. It is not a temporary directory and must not be deleted between methods. Module 2's display-only Poisson preview cache is disposable and removed on browser handoff. An interrupted run may have an incomplete `.work-*` folder; the next invocation for that **specific run** cleans managed stale work folders only. Never recursively purge the entire user output root.

## New scene

```bash
conda activate gsrefine
cd /home/xth/GSSceneRefine
python run.py roi-init \
  --config configs/songziguanyin.yaml \
  --scene bazhaifeng --method absgs \
  --sfm-dir /path/to/bazhaifeng/COLMAP/sparse \
  --gaussian-ply /path/to/bazhaifeng/absgs/point_cloud.ply \
  --output-dir /mnt/f/danxia_experiment_archive/GSSceneRefine_results
```

## Same scene, different Gaussian method

```bash
python run.py roi-auto \
  --config configs/songziguanyin.yaml \
  --scene bazhaifeng --method vanilla \
  --sfm-dir /path/to/bazhaifeng/COLMAP/sparse \
  --gaussian-ply /path/to/bazhaifeng/vanilla/point_cloud.ply \
  --roi /mnt/f/danxia_experiment_archive/GSSceneRefine_results/bazhaifeng/roi/roi_001.json \
  --reuse-scene-roi \
  --output-dir /mnt/f/danxia_experiment_archive/GSSceneRefine_results
```

Old valid ROI files under `runs/roi_init/` are still readable via the explicit `--roi` argument if their COLMAP frame and orientation source match. A legacy ROI's `annotation_metadata.model_scope: full_reconstruction` remains valid even if the newer `source_model_scope` field was never written. Cropped/AABB models cannot be disguised as full-scene reconstructions.

## Existing repository `runs/`

No new four-stage invocation should write outputs inside the code checkout. The patch intentionally **does not delete** old `runs/`, since it may contain confirmed ROIs or SfM/Poisson caches used as sources by older experiments. To clean a checkout, first back up or migrate any required ROI JSON/Poisson/SfM and verify the final `manifest.json` references, then explicitly archive the old `runs/` elsewhere. Do not run a blanket `rm -rf` against an unexamined tree.

## Reproducibility

The runtime `effective_scene.yaml` retains exact algorithm parameters, the output manifest specifies input and output identities, and the compact processing report includes the Gaussian counts. `R0/R3` remain internal algorithm stage names in diagnostics; the public model is always `gaussian_roi.ply`.
