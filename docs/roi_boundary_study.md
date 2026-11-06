# ROI boundary-topology and multiscale Poisson candidate study

**Status:** experimental method and ablation study, not a validated automatic semantic segmentation algorithm. Runs entirely inside the existing `GSSceneRefine` project and `gsrefine` environment. No new project directory or environment. Original `.ply`, `.npz`, ROI `.json`, and `configs/*.yaml` are read-only. Original `roi-refine` behavior remains unchanged.

## Goal

1. Preserve all bounded, confidence-supported exterior Poisson candidate faces **before** the former whole-area rejection.
2. Group candidates by (view, polygon edge, triangle-face adjacency component), identify the ROI intersection trace and location, and compute local relative area, gravity-referenced height, base azimuth coverage, and multi-scale surface growth curves.
3. Conservatively triage patches into `normal`, `suspect`, or `uncertain` without treating connected terrain as an automatically restorable target.
4. Run an R0–R3 controlled comparison with exactly the same source Gaussian, SfM, Poisson and ROI.

## Commands (Songzi Guanyin ROI002)

```bash
conda activate gsrefine
cd /home/xth/GSSceneRefine

# Examine geometry first, avoiding costly Gaussian guard computation.
python run.py roi-study \
  --config configs/songziguanyin.yaml \
  --roi runs/roi_init/songziguanyin_absgs_roi_002.json \
  --output-dir runs/roi_tests/songziguanyin_roi002_boundary \
  --no-ablation

# After reviewing candidate geometry, run R0–R3 masks and reports.
python run.py roi-study \
  --config configs/songziguanyin.yaml \
  --roi runs/roi_init/songziguanyin_absgs_roi_002.json \
  --output-dir runs/roi_tests/songziguanyin_roi002_boundary

# Optional (large files): R0/R1/R2/R3 Gaussian models for SuperSplat.
python run.py roi-study \
  --config configs/songziguanyin.yaml \
  --roi runs/roi_init/songziguanyin_absgs_roi_002.json \
  --output-dir runs/roi_tests/songziguanyin_roi002_boundary \
  --export-ply
```

All outputs are placed below **the same existing project `runs/` tree**. No new source code folders outside the project.

If `--roi` is omitted, `roi-study` reads the latest confirmed scene ROI through the existing `roi_init` index. For reproducible ablations, specifying `--roi ..._002.json` is strongly recommended. The `study_report.json` records SHA-256 hashes of all inputs.

## Important output files

| Filename | Interpretation |
| --- | --- |
| `study_report.json` | Input hashes, 32.90%-style pre-budget area ratio, candidate classification areas, R0–R3 counts and caveats |
| `roi_boundary_trace.ply` | Point samples of ROI/Poisson boundary crossings for CloudCompare; **approximate** |
| `boundary_traces.npz` | Native COLMAP xyz trace sample positions and corresponding orthographic polygon edge indices |
| `exterior_all_colored.ply` | All pre-budget candidates colored by triage label; CloudCompare/Blender mesh layer |
| `exterior_normal.ply` | Gray: expected continuous base background candidates (high-confidence geometric evidence only) |
| `exterior_suspect.ply` | Red: suspected locally clipped extension candidates |
| `exterior_uncertain.ply` | Orange: insufficient evidence; not auto-restored in R2/R3 |
| `candidate_groups.csv` | Area, location, contact edge, height, wrapping, multiscale area growth and labels for each patch |
| `candidate_groups.json` | Complete machine-readable patch table and multidimensional growth curve arrays |
| `candidate_faces.npz` | Original Poisson triangle indices, group assignments and confidence |
| `keep_mask_R0.npy` | Initial hard ROI, identical Gaussian row ordering |
| `keep_mask_R1.npy` | All bounded Poisson candidates + global directional Gaussian safety |
| `keep_mask_R2.npy` | Suspected candidates only + global directional Gaussian safety |
| `keep_mask_R3.npy` | Suspected candidates only + local adaptive Gaussian safety |
| `gaussians_R*.ply` | Optional full-resolution research visualization exports, only with `--export-ply` |

**Visualization:** Open PLY Gaussian models in SuperSplat and open the separate triangle-mesh Poisson candidate layers in CloudCompare or Blender. SuperSplat does not necessarily display triangular mesh PLYs; use an appropriate mesh viewer. ROI intersected with the original Poisson is not its AABB or gravity OBB.

## R0/R1/R2/R3 and automatic uncertainty

- `R0`: Gaussian centers falling inside every user orthographic polygon.
- `R1`: the former bounded confident exterior candidate set (without a global area hard rejection) and Gaussian global-directional scale band.
- `R2`: only geometric **suspect** exterior patches, using the same global directional band.
- `R3`: the same suspects, with local directional Gaussian covariance and SfM-spacing uncertainty.
- **Uncertain is not negative ground truth:** conservative automatic R2/R3 may restore nothing if available geometry is insufficient. The result must be reported, not silently converted to a successful recovery. This is the key falsifiable research question of the first real-scene test.

For a supervised **reference / manual ablation**, first inspect `candidate_groups.json` and create:

```json
{
  "labels": {
    "v00_e01_c000060": "suspect"
  }
}
```

Use actual stable IDs from your current `candidate_groups.json`; the displayed ID above is an **example only**. Save at `runs/roi_tests/songziguanyin_roi002_boundary/review_labels.json` and repeat with:

```bash
python run.py roi-study \
  --config configs/songziguanyin.yaml \
  --roi runs/roi_init/songziguanyin_absgs_roi_002.json \
  --output-dir runs/roi_tests/songziguanyin_roi002_manual_reference \
  --labels runs/roi_tests/songziguanyin_roi002_boundary/review_labels.json
```

A manually overridden result is a **manual-reference variant**, never an autonomous algorithm result. Group IDs depend on input Poisson topology and should not be reused across remeshing.

## Optional experimental knobs

Under the existing scene YAML, `roi_study:` is **optional**. These defaults are provided automatically; changing them is NOT needed to run ROI002.

```yaml
roi_study:
  growth_steps: 6                 # multi-scale geodesic distance samples
  growth_multiple: 3.0           # diagnostic maximum = 3 x initial outside search
  max_growth_groups: 32          # largest groups evaluated; others uncertain
  base_height_quantile: 0.30     # height normalized between SfM/Poisson credible P05-P95
  high_height_quantile: 0.65
  min_ring_coverage: 0.45
  max_suspect_area_fraction: 0.25
```

These are **initial hypotheses, not validated universal thresholds**. Freeze parameters in cross-scene experiments and run independent ablations rather than adjusting until the desired test passes.

## Known limitations and interpretation of 32.90%

1. The previous 32.90% corresponds to `sum(area_faces[credible & outside_candidate]) / sum(area_faces[credible & inside])` **before the global area budget**. It is a *surface area ratio in SfM model units squared*, not a fraction of visible screen area or known target rock area. The new study deliberately reproduces this initial ratio and then splits the numerator by spatial patch.
2. Existing candidate and inside labels are based on **face centroids**. `roi_boundary_trace.ply` is a sampled intersection of mesh edges with orthographic polygon edges, not a watertight exact mesh Boolean. Boundary triangles straddling multiple constraints may be assigned approximately; numbers are preliminary. A future publication-quality exact-clipped-area method may be needed if boundary aliasing is material.
3. The geodesic metric uses triangle-centroid graph shortest paths rather than an exact intrinsic surface geodesic. Growth curves are analyzed only within a limited region and the 32 largest groups by default.
4. A connected background slope can be adjacent to a legitimate clipped rock. Broad base wrapping and persistent area growth supply **evidence**, not semantic proof. A small isolated background rock may mimic a local truncated target. Report the `uncertain` fraction explicitly and assess target recall/background leakage using independently labeled reference renderings.
5. The original Songzi Guanyin AABB model contains only 2,501,530 Gaussians. R1–R3 cannot reconstruct Gaussians removed in the earlier AABB experiment.
6. The optional `--export-ply` can produce multiple large files. Use `--no-ablation` first. No changes are made to the scene configuration or saved ROI.

## Proposed independent verification

- Check `initial_exterior_ratio` against 0.3290 for the ROI002 inputs, before assessing boundary groups. If different, inspect input SHA hashes and code versions.
- Overlay the initial R0 Gaussian reconstruction with ROI002 in SuperSplat; inspect top, back, left, right and base.
- Examine `exterior_all_colored.ply` together with original Poisson in CloudCompare/Blender. Distinguish valid missed rock and expected continuous base terrain manually; do not treat auto triage as ground truth.
- Plot per-patch `growth_scales` versus `growth_curve_area` from JSON. Identify plateau due to original mesh boundary/holes separately from a real bounded surface end.
- Compare R0–R3 against independent annotated rock recall and background Gaussian inclusion; also record runtime, memory, added Gaussian counts and uncertain-patch area.
- Repeat on a different ROI and another Danxiashan scene without retuning, after the ROI002 diagnostic succeeds.
