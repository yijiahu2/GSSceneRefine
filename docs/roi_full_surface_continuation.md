# Complete exterior Poisson context + local Gaussian recovery

## Purpose

`python run.py roi-study` now distinguishes **full external geometry** from a **bounded local recovery candidate**. This fixes the earlier interpretation error in which the thin, fragmented exterior candidate band was mistaken for all of the Poisson surface outside the ROI.

- `exterior_raw_full.ply`: all raw Poisson triangles with centroids outside the ROI; no confidence cutoff, no geodesic search limit. Mesh connectivity follows the original Poisson triangle topology.
- `exterior_credible_full.ply`: all outside Poisson triangles passing the SfM-derived confidence threshold; no outside strip limit. It may still look fragmented because a binary confidence threshold can remove internal surface faces.
- `exterior_candidate_local.ply`: **original, unchanged**, confidence-, outside-distance-, geodesic-distance- and ROI-interior-seed-constrained candidate. ONLY this mask can be considered for Gaussian recovery.
- `exterior_normal.ply`, `exterior_suspect.ply`, `exterior_uncertain.ply`: spatial classification subsets of the bounded local candidate. These are **not** the full background surface.

## Analysis method

For full raw Poisson faces outside the dual-orthographic ROI, construct an edge-sharing face graph. Graph weights are approximate center-to-center lengths with an extra travel penalty on faces below the confidence threshold. Low-confidence faces can help diagnose whether a surface appears connected in the original geometry, but their area **does not contribute to trustworthy external area** and they **cannot authorize Gaussian recovery**.

For each of the largest outside candidate connected components, calculate cumulative raw and reliable exterior area versus increasing (approximate face-graph geodesic) distance up to a configurable fraction of the scene diagonal. This analysis is NOT limited to the small initial ROI search strip. Classify a candidate component as expected normal continuation only if both original across-boundary/base evidence and supplementary full-scale persistence/support checks hold. High-altitude/top/terminating candidates and manual labels remain protected from blanket background classification. Where full-scale evidence is not calculated or insufficient, label uncertain.

The resulting areas use **triangle centroid membership** for the ROI and are explicitly approximate near the ROI boundary; strict polygon-clipped triangle surface integrals remain for subsequent improvement. The "raw" full exterior may include unreliable Poisson artifacts. None of the geometric classification labels constitutes semantic ground truth.

## New report fields

`study_report.json` -> `candidate.full_exterior`:
- `raw_exterior_area_centroid_approx`
- `credible_exterior_area_centroid_approx`
- `credible_internal_area_centroid_approx`
- `full_credible_exterior_to_internal_ratio`
- `bounded_candidate_area`
- `bounded_to_full_credible_exterior_fraction`
- `full_growth_scales_model_units`

`full_exterior_components.json`: per-component raw/credible growth curves, full/local area ratios, outer-half growth, far reliable-support fraction, whether evaluated.

`background_components.json`: same old boundary topology evidence augmented by `full_exterior`, `full_checks`, and final candidate-component decision.

## Use on ROI002

```bash
conda activate gsrefine
cd /home/xth/GSSceneRefine
python -m pytest -q tests
python run.py roi-study \
  --config configs/songziguanyin.yaml \
  --roi runs/roi_init/songziguanyin_absgs_roi_002.json \
  --output-dir runs/roi_tests/songziguanyin_roi002_boundary \
  --no-ablation
```

Visually overlay `exterior_raw_full.ply`, `exterior_credible_full.ply`, and `exterior_candidate_local.ply` with the original Poisson or Gaussian scene using CloudCompare/Blender. The full exterior diagnostic is for analysis only, not a proposal to restore the full exterior Gaussian field.

## Optional configuration in scene YAML

```yaml
roi_study:
  full_growth_max_diag_frac: 0.65
  full_growth_steps: 9
  full_growth_max_components: 16
  low_confidence_path_penalty: 2.0
  full_normal_area_multiplier: 2.0
  full_normal_growth_fraction: 0.12
  full_normal_reliable_far_fraction: 0.35
```

All settings are optional. These default thresholds are research hypotheses needing sensitivity/ablation evaluation; they are not calibrated universal physical laws. No existing ROI, model or environment file is changed by this patch. The main algorithm keeps original Gaussian masks and recovery logic.

`full_exterior_faces.npz` preserves source Poisson face indices and confidence values for audit and cross-run comparisons. The first two exported full PLYs are visualization products; this NPZ keeps their exact source index mapping (under the current centroid-based ROI definition).
