# GSSceneRefine

GSSceneRefine extracts a 3D Gaussian Splatting region of interest (ROI) from a complete Gaussian scene with a shared COLMAP SfM–Poisson reference. Its workflow is **SfM cleaning and Poisson reference → interactive polygon ROI → Poisson-guided local recovery and boundary protection → verified Gaussian ROI export**. This is ROI extraction, *not* final removal of all abnormal Gaussians.

The project uses the single WSL repository `/home/xth/GSSceneRefine` and Conda environment `gsrefine`. Scientific R0/R3 algorithms retain their internal stage identifiers; exported models have readable names.

## Quick start: input and output paths supplied by the user

```bash
conda activate gsrefine
cd /home/xth/GSSceneRefine

python run.py roi-init \
  --config configs/songziguanyin.yaml \
  --scene songziguanyin --method absgs \
  --sfm-dir /home/xth/datasets/danxiashan/songziguanyin/sparse \
  --gaussian-ply /mnt/f/danxia_experiment_archive/songziguanyin/final30k_D_absgs_accel/point_cloud/iteration_30000/point_cloud.ply \
  --output-dir /mnt/f/danxia_experiment_archive/GSSceneRefine_results
```

- `--sfm-dir`: original COLMAP folder containing `points3D.bin` and `images.bin` (or use `--colmap-points` and `--colmap-images` instead).
- `--gaussian-ply`: **complete** GSPL-compatible Gaussian PLY, in the same original COLMAP world coordinates.
- `--output-dir`: one root for **all** newly generated four-module geometry caches, ROI definitions, models, metadata and logs. It must be a directory; the software creates scene and method subfolders.
- `--config`: reusable algorithm parameter template. The original YAML is not overwritten; effective runtime configs live under the user output root.

When Module 2 confirms a polygon ROI in Edge, the browser service closes safely; WSL continues to display Module 3/4 progress and the final saved path.

## Reuse the same ROI with a different 3DGS method

```bash
python run.py roi-auto \
  --config configs/songziguanyin.yaml \
  --scene songziguanyin --method vanilla \
  --sfm-dir /home/xth/datasets/danxiashan/songziguanyin/sparse \
  --gaussian-ply /mnt/f/danxia_experiment_archive/songziguanyin/final30k_A_vanilla/point_cloud/iteration_30000/point_cloud.ply \
  --roi /mnt/f/danxia_experiment_archive/GSSceneRefine_results/songziguanyin/roi/roi_001.json \
  --reuse-scene-roi \
  --output-dir /mnt/f/danxia_experiment_archive/GSSceneRefine_results
```

This reuses **only** the confirmed 2D polygon constraints and the shared SfM–Poisson geometry. Gaussian masks and final PLYs are independently computed for the new method. Older confirmed ROI JSONs can be passed explicitly via `--roi` when their scene orientation matches the current COLMAP data; the older `model_scope` field remains supported.

## Output organization

```text
GSSceneRefine_results/
└── songziguanyin/
    ├── geometry/
    │   ├── cleaned_sfm.npz
    │   └── poisson_surface.ply
    ├── roi/
    │   └── roi_001.json
    └── methods/
        ├── absgs/roi_001/
        │   ├── final/gaussian_roi.ply
        │   ├── metadata/manifest.json
        │   ├── metadata/processing_report.json
        │   └── logs/processing.log
        └── vanilla/roi_001/
            └── final/gaussian_roi.ply
```

The output manifest has explicit input paths for the *subsequent anomaly cleanup stage*: the Gaussian ROI, ROI polygon JSON, trusted Poisson surface and cleaned SfM. Output naming never presents `gaussians_final_R3.ply` as a public result, although R3 remains the internal final recovery method.

The automatic workflow normally keeps only the final model, reusable geometry/ROI, essential reports and logs; its temporary masks and debug meshes are discarded after a validated run. For reproducible in-depth study, add **`--keep-diagnostics`** to `roi-init` or `roi-auto`. This preserves candidate surfaces, masks and tables under `diagnostics/` without changing the Gaussian model.

All generated four-module outputs are under `--output-dir`. The old repository `runs/` is not used by new automatic invocations and is **not deleted automatically**: it may still hold valuable ROIs or source references. Keep or archive those only after confirming no experiments depend on them.

See [output directory contract](docs/output_management.md), [interactive ROI workflow](docs/roi_init.md), and [algorithm overview](docs/architecture.md).

## Verification

```bash
conda activate gsrefine
cd /home/xth/GSSceneRefine
python -m pytest -q tests
(cd ui && npm test)
```

To test on a new scene, ensure source input geometry is not cropped, SfM/camera data belong to the same COLMAP reconstruction, and the full Gaussian PLY is expressed in that original world frame. Successful automated tests do not substitute for real-scene visual validation.
