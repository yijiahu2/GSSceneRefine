# Estimating gravity-aligned views from original COLMAP camera geometry

## Why changing a `+Z` checkbox is not scientifically sound

A monocular SfM reconstruction is known only up to a 3D similarity transform;
there is no intrinsic world-up. The Gaussian model inherits this frame. For a
natural, cliff-dominated landmark, PCA of all 3D points or a single global
Poisson plane is not reliably a terrain-horizontal plane.

## Current implementation (conditional UAV assumption)

When the camera gimbal largely eliminates roll, the camera's horizontal image
axis lies approximately in a global horizontal plane even for variable yaw and
pitch. In COLMAP `images.bin`, q maps world->camera, so

- camera right vector: `r_i = R(q_i)^T [1,0,0]`
- camera forward axis: `f_i = R(q_i)^T [0,0,1]`
- camera center: `C_i = -R(q_i)^T t_i`

Estimate `u` with `||u||=1` minimizing `sum w_i (r_i · u)^2`, using smallest
covariance eigenvector and 7 robust IRLS iterations. Sign is chosen by the
majority of cameras down-looking: `median(f_i · u) < 0`.

Acceptance requires diversity of camera-right directions, median/P90
right-axis residual below 12°/28° and sufficient optical sign support. These
are experimental heuristics, NOT a calibrated gravity confidence, and should
be tuned if acquisition conditions differ. Rejected estimates are cached and
reported but **never silently replaced by +Z**.

For a fixed scenario (Songzi Guanyin), all 3DGS methods must share the original
COLMAP frame identifier and the same orientation source SHA. ROI is a pure
orthographic projection; Gaussians, SfM and Poisson coordinates are unchanged.

## How to run (single environment)

```bash
conda activate gsrefine
cd /home/xth/GSSceneRefine
# One-time update of historical +z YAML; creates a backup:
python tools/migrate_orientation_auto.py --config configs/songziguanyin.yaml
# Show estimate and optional Poisson normal statistics:
python run.py orientation --config configs/songziguanyin.yaml --check-poisson
# Only if accepted, launch actual Windows WebGPU ROI workspaces:
python run.py roi-init --config configs/songziguanyin.yaml
```

The original `images.bin` is looked up at
`/home/xth/datasets/danxiashan/songziguanyin/sparse/images.bin` by default,
and can be overridden by `roi_init.orientation.colmap_images`.
Cached diagnostics are in `runs/orientation/`. If rejected, do not simply
manually set verified=true. Check acquisition metadata/GNSS, IMU/gimbal
alignment and actual COLMAP camera images. An independently verified vector
can be passed via `mode: manual, manual_up_vector: [ux,uy,uz],
manual_verified: true` with its provenance explained in the paper.

## Alternative literature paths

- Lhuillier (2023), Computer Vision and Image Understanding, DOI
  `10.1016/j.cviu.2023.103814`: camera path PCA and Hough transform votes of
  triangular surface normals; camera-height assumption may fail for UAV cliff
  flyovers, so in this project trajectory PCA and Poisson normal fractions are
  diagnostics only.
- COLMAP `model_orientation_aligner`: Manhattan/vanishing points usually assume
  strong man-made line structure, unreliable for irregular rocky landscapes.
- COLMAP `model_aligner`: if reliable image GPS data and geographic alignment
  exist, transform original reconstruction to the ENU reference or recover the
  original-frame world-up using the inverse similarity rotation.
- Pan et al. (ECCV 2024), gravity-aligned rotation averaging: when actual gravity
  measurements are available they can constrain rotation estimation robustly.

**Important limit:** if camera roll assumption fails and no independently
calibrated gravity/geo information exists, absolute vertical may be
unidentifiable from the monocular SfM reconstruction alone. The system must
report this rather than invent a direction.
