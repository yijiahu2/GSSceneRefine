# GSSceneRefine – stage2 open-surface A+B main-only experiment (2026-11-10)

## Purpose

This update adds a **new, separate, experimental `open-ab` phase** to the existing `stage2-cleanup` entry point. It does not replace or rewrite previous `check`, `seal`, `sdf` or `export` workflows. The historical sealing cache revision stays unchanged.

**Experiment scope is ONLY the largest observed Direct-R3 Poisson component** (`poisson_roi_main.ply`, Songzi Guanyin: 15,389 faces). The other 19 components are **completely ignored**, not deleted from source data, not consulted for distances, and not consulted for spatial classification. As a result, a candidate corresponding to those components can potentially be removed; this is an explicit limitation of the current experiment.

## Decision rule

- Previous distance threshold remains unchanged: `delta = full_core_SfM_P95(main) + P10--P90_trimmed_mean(3 * exp(max(log_scales_0_1_2)))`. The epsilon remains `1e-5`.
- A: query **exact unsigned** Gaussian-centre-to-main-triangle distances with Open3D. Near-threshold Gaussians are kept.
- B: **only for d > delta + epsilon**, compute fast generalized winding numbers on the original, consistently oriented, open main surface (libigl). Count independent camera centers whose COLMAP optical axis points towards the candidate and whose camera-to-candidate ray is not occluded by a main Poisson triangle (Open3D). This is **geometric line-of-sight evidence**, not photographic proof of empty space.
- Delete **directly**, without a second anomaly classifier, when `abs(w) <= winding_outside_max (default 0.10)` **AND** `unoccluded_camera_votes >= 2`. Otherwise keep. `abs(w) >= 0.85` is an internal candidate and always kept; intermediate values are uncertain and always kept.
- A valid closed solid is **not** claimed. Open-surface winding numbers are not an exact inside/outside certificate. This is an explicitly conservative, reproducible experiment, not ground truth.

## Installation

Copy the ZIP to the Windows Edge download folder, then execute in WSL:

```bash
cd /home/xth/GSSceneRefine
conda activate gsrefine
unzip -o /mnt/f/Edge/GSSceneRefine_stage2_open_AB_main_only_20261110.zip -d /home/xth/GSSceneRefine
python -m pip install libigl
```

`libigl` is the only new dependency. It is installed in the **existing** `gsrefine` environment; no parallel environment. API: https://libigl.github.io/libigl-python-bindings/api/igl/ .

## Run (Songzi Guanyin existing data)

```bash
ROI_RUN=/mnt/f/danxia_experiment_archive/GSSceneRefine_results/songziguanyin/methods/absgs/roi_001

python run.py stage2-cleanup \
  --manifest "$ROI_RUN/metadata/manifest.json" \
  --phase open-ab \
  --preseal-report "$ROI_RUN/cleanup/geometry/direct_r3_sdf/e79ac66ff1af70f592e214bc/preseal_diagnostics.json" \
  --output-dir "$ROI_RUN/cleanup/preview/open_ab_main_v1"
```

Change `--output-dir` for a new attempt. Existing runs are never overwritten. If conservative winding classification removes almost nothing, review the camera reference / winding bands first; for a controlled ablation you may pass `--winding-outside-max 0.15` or `--min-unobstructed-cameras 3`, keeping the **distance threshold unchanged**. Do not claim an exterior classification as ground truth solely from one approximate winding value.

## New outputs (only in the selected `--output-dir`)

- `gaussian_roi_clean.ply`: trial cleaned Gaussian PLY, with original stored attributes preserved, *not* the project official final.
- `classification_masks.npz`: per-Gaussian `keep`, `remove`, `over_threshold`, `inside_candidate`, `unknown_kept`, `distance_main`, `winding`, `visible_camera_votes`.
- `open_ab_main_report.json`: immutable input hashes, P95 and Gaussian radius statistics, actual threshold, class counts, camera baseline, and method limitations.

No `poisson_roi_direct_r3.ply`, `poisson_roi_main.ply`, `poisson_roi_auxiliary.ply`, Stage-1 `gaussian_roi.ply`, or Stage-2 official final is modified. No added skirt or bottom. Does not depend on `--phase seal` success.

## Validation / precautions

1. The preseal report MUST match the manifest and verified R3 source; full core SfM must have been used, and main surface must be intact and topologically eligible.
2. Inspect removed Gaussians in the NPZ/Ply and render several standardized novel views. Conservative classification can still make mistakes in unobserved areas; the method is **an experiment**, not a certified semantic separator.
3. The source Gaussian is ROI Stage-1 output; the 19 auxiliary components are excluded from the reference surface **by experiment design only**.
4. Default `winding_outside_max=0.10` is intentionally conservative for an open surface. Camera reference winding values are printed in the report; if fewer than 50% of registered cameras fall in the exterior winding band, the program refuses pruning to avoid uncalibrated sign assumptions.

## Regression tests

```bash
python -m pytest -q tests/test_stage2_direct_r3_pipeline.py tests/test_stage2_main_auxiliary.py tests/test_stage2_seal_projection_fallback.py tests/test_stage2_open_ab_main.py
```

The 31 local tests used synthetic geometry and mocks for Open3D/libigl. Actual Songzi Guanyin 3DGS results require running this command on the user's verified WSL data; no result is claimed yet.
