# GSSceneRefine / ROI module 3: R3-v2 integration instructions

## Environment / constraints

Target existing repository: `/home/xth/GSSceneRefine` (WSL Ubuntu 22.04). Existing conda env: `gsrefine`. Do not create new project or environment. This ZIP adds only one algorithm file and one unit-test file; it does **not** overwrite `run.py`, `study.py`, `roi-finalize`, or existing project configs. The engine has been tested with synthetic data, not against the user's full WSL scene data.

## Research contract (non-negotiable)

- Core geometry recovery **must finish before** safety-distance expansion. No Gaussian-scale safety distance may creep into the core-recovery selection criteria. In the first stage, preserve R0 and recover the mesh-identified suspect patch with **geometric** matching tolerances (Poisson-to-Gaussian fitting error only; audit independently). Preserve existing normal-exterior identification, do not relabel everything or introduce SAM/SDF cleaning.
- After core reconstruction, regenerate the **actual full core boundary** and apply kNN 5%-each-tail trimmed average Gaussian **longest principal-axis sigma** to an implicit metric SDF outside the complete core boundary. Use all original Task1 AABB Gaussians for kNN, without screening out abnormal points. All points in computed shell are kept.
- Do not treat old R2 or old R3 output as a validated geometry-only core. Both use protective radii in existing code paths. Refactor the old geometry step to create `core_mask_R3v2.npy` independently.
- `roi.distances(xyz)` seen in previous project source has positive-inside distances to 2 polygon views; `-min(roi.distances(...))` may be a **signed inside/outside proxy**, but it is NOT necessarily a true metric Euclidean SDF at corners. Implement a **metric signed distance to the actual intersection of both extruded polygons**, or generate a local implicit SDF with narrow-band refinement and a proven/respectable discretization error bound. The interface `original_roi_sdf` in `two_stage_sdf.py` REQUIRES metric distance for exterior. Do not silently use the proxy in production.
- `CoreImplicitField` represents recovered Gaussian core locations as small balls whose *geometric support radius* is separate from the safety band. Derive/support this radius from Poisson triangle resolution, discretization pitch or spatial Gaussian-center sampling. Ensure connected core geometry can be recovered and disjoint fragments are not bridged accidentally. A high-precision representation may replace the ball union if needed; preserve the two-stage API and tests.
- Default k=128, trim=0.05 each tail (6 values removed at each end of 128), factor=1; config is tunable, not claimed scientifically optimal. Standard 3DGS `scale_0/1/2` are log-sigma and should be exponentiated once. Largest axis sigma = max(exp(scale_j)); quaternion rotation is irrelevant for *largest eigenvalue*, but would matter for directional scale (not requested here).
- `CoreImplicitField` evaluates a continuous implicit distance with no output mesh. The external geometric distance of the original ROI + recovered ball union is exact only if original ROI SDF is metric and geometric core has been represented correctly. Do not confuse it with a reconstructed physical Poisson surface SDF. Avoid reporting a hard guarantee for rendered splat support from a center-based inclusion test.
- `safety_band_after_core` is a reference implementation for automatic local boundary projection + kNN + inclusion, chunking to limit peak memory. Audit projection failures; if many occur, improve boundary point computation, and do NOT silently accept approximated boundary statistics as a proven guarantee.
- Analyze all boundary segments (not just the top/suspect area); do not use any location-specific logic. Stage 2 should not exclude background or abnormal Gaussians.

## Integration in GSSceneRefine

1. Inspect current `gsrefine/roi/study.py` and the `local_protection` / `add_guard_gaussians` implementation. Keep historical R0-R3 intact for comparison. Add `R3v2` as new option in the existing `roi-study` pipeline; do not delete or rename R3 files.
2. Stage A: from existing `analysis['label_masks']['suspect']`, use mesh geometry to select newly restored core Gaussians **without invoking any Gaussian covariance safety dilation**. Save `core_added_mask_R3v2.npy`, `core_mask_R3v2.npy`, a count and geometric tolerance diagnostic, and a PLY for visual comparison.
3. Stage B: generate a valid metric signed distance to **R0 union Stage-A core geometry**. The supplied engine uses original metric R0 SDF union a geometric support envelope around recovered Gaussian centers. If that approximation poorly represents the recovered Poisson body, replace it with a more faithful implicit geometric support representation, not with the pre-repair ROI. Feed the whole Gaussian dataset into `safety_band_after_core(...)` after Stage A is done, with `major_axes` extracted from original PLY.
4. Export `safety_added_mask_R3v2.npy` and `keep_mask_R3v2.npy` (`core_mask | safety_added_mask`). The masks must have same row order and length as original Gaussian PLY. Save `final_gaussian_report_R3v2.json`. Update `roi-finalize --method R3v2` to accept the new mask, preserving all source PLY attributes and source order. Do not change or recompute masks during export.
5. Save a separate per-boundary/at-top diagnostic: selected positions, closest reconstructed-core distance, KNN 5% trimmed sigma, accepted safe zone indicator, and projection residual. Report failure count and implicit-field geometric support pitch. If using grid-based SDF, use a conservative numerical error bound `sqrt(3)*pitch/2` for cell-center distance sampling plus a verified interpolation error. If using direct continuous evaluation as in this engine, report that no voxel interpolation is used, but geometry approximation / boundary projection errors still exist.
6. Validate inclusion/exclusion for random boundary samples and for a deliberately clipped island in a synthetic dataset. Assert baseline is always preserved, core and safety added masks disjoint, final mask equals union, staged Gaussian counts reconcile, kNN includes outliers, and no position-specific branch exists. Test K=64/128/256 for stability without inventing an optimal value.
7. Run existing tests, then a separate ROI002 run output directory, export new PLY, and visually compare against the input AABB and old R3. Do not claim the algorithm has been visually validated until inspected in SuperSplat.

## Recommended Codex working commands

```bash
conda activate gsrefine
cd /home/xth/GSSceneRefine
python -m pytest -q tests/test_two_stage_sdf.py
python -m pytest -q tests
# Once integrated:
python run.py roi-study \
  --config configs/songziguanyin.yaml \
  --roi runs/roi_init/songziguanyin_absgs_roi_002.json \
  --output-dir runs/roi_tests/songziguanyin_roi002_R3v2
python run.py roi-finalize \
  --study-dir runs/roi_tests/songziguanyin_roi002_R3v2 \
  --config configs/songziguanyin.yaml --method R3v2
```

## Scientific limitation

A band that includes **all Gaussian centers** at a distance <= local trimmed sigma does not guarantee the complete tails of every unbounded Gaussian function; that is a different criterion. The requested long-axis mean is a sensible, testable buffer proxy, but its completeness and redundancy should be evaluated on known synthetic truncations and free-view renders.
