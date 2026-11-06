# GSSceneRefine · Module 3 simplified ROI study (2026-11-05)

**Aim.** Respect the user's manually drawn rough ROI. Do not re-segment the entire 3D scene: exclude expected terrain/background extension along the ROI perimeter; review the remaining **local** candidate surfaces jointly by **area ratio, mesh connectivity and distance**; only then perform bounded Gaussian recovery.

## Algorithm and mathematical definitions

Input: original open Poisson mesh `M`, confirmed two-orthographic-polygons ROI `Ω₀`, Gaussian PLY, and (only if R1–R3 is requested) cleaned SfM for the Gaussian guard width.

1. **Temporarily trust every original Poisson face.** The default `roi-study` mode sets `C(f)=1` and skips `geometry_confidence()` completely. `roi-refine` is unchanged. No Poisson mesh smoothing, closing or new surface reconstruction is performed. The previous scored study can be requested explicitly with `roi_study.poisson_confidence_enabled: true` as a historical comparison, not the default.
2. **Analyze complete ROI exterior.** Define `M_out={f∈M: centroid(f)∉Ω₀}`. Build the outside-only adjacency graph from the **original** mesh, never traversing an inside face to connect two exterior components. Measure **each entire exterior component once**, its area `A_full,j`, its directly intersecting ROI boundary faces, and which saved orthographic polygon edges it reaches. Also report `A_in=∑_{centroid(f)∈Ω₀} Area(f)`.
3. **Identify normal expected perimeter continuation.** For each exterior component that intersects the ROI, measure: `A_full,j/A_in`, `A_full,j/A_local,j` (full exterior vs finite boundary candidate), number of distinct ROI edges, lower/base candidate area fraction, and horizontal azimuth coverage. Normal-background **evidence is joint**, not an area-only threshold. Even when the full exterior component is background-like, **top-edge and higher local candidate patches are not blanket-rejected**.
4. **Review bounded local possible miscuts.** Preserve existing `mesh_external_candidates` with ROI-near projected distance, original mesh connectivity to inside seed faces, and limited approximate surface-geodesic distance. For each locally connected patch `k` measure `R_k=A_ext,k/A_in`, `max outside projection violation`, contact angular spread, and actual interior-to-exterior mesh adjacency. A patch becomes `suspect` and therefore eligible for Gaussian guards **only when several geometric conditions hold together**. Otherwise it remains `uncertain` and is not automatically recovered. `normal` patches are excluded.
5. **Gaussian protection.** `R0` keeps every Gaussian center inside `Ω₀`. `R1` considers the **complete bounded local candidate** as a control, while `R2/R3` consider only the geometrically accepted `suspect` subset. Existing Gaussian covariance/directional sigma and SfM-spacing guard widths are reused. `R2/R3` never remove `R0` Gaussians. Full exterior surfaces are *for interpretation only*, never a Gaussian restore mask.

These are interpretable *geometric hypotheses*, not ground-truth rock vs terrain classifications. Exact clipping of triangles along 3D ROI boundaries is **not yet implemented**; face centroids give an area approximation. Poisson boundary traces are sampled mesh-edge crossings. For a paper, validate these approximations and the classification against manually checked 3D regions and held-out scenes.

## Files and compatibility

The update only replaces `gsrefine/roi/study.py` and introduces `gsrefine/roi/simple_geometry.py` plus tests and this document. It does **not** replace `core.py`, original PLYs, ROI JSONs, configuration files, module 2, the full-exterior helper, or the `roi-refine` implementation. Existing `roi-study` CLI command remains available. The historical growth-based study can be rerun with the *opt-in* config key `roi_study.legacy_growth_classifier: true`; don't combine its diagnostics with the new method in the same report.

The following outputs retain their names:

- `exterior_raw_full.ply` and `exterior_credible_full.ply`: **same face set** in the default all-trusted experiment; colors differ.
- `exterior_candidate_local.ply`: finite bounded ROI exterior candidates.
- `exterior_normal.ply`, `exterior_suspect.ply`, `exterior_uncertain.ply`: geometric local candidate classes. Empty categories do **not** leave an old PLY behind.
- `candidate_groups.csv/json`: per local patch, including `area_to_internal_fraction`, `inside_mesh_connected`, `max_outside_violation`, associated complete exterior component, classification and reason.
- `background_components.csv/json` and `full_exterior_components.json`: each **unique entire exterior component** with one area value and joint-evidence flags. No repeated full-area calculation from different seed groups. Growth curves are **not** used in the default classifier.
- `study_report.json`: totals, approximation/assumption flags and R0–R3 (if enabled).

If `--no-ablation` is used, don't interpret any older `keep_mask_R1/R2/R3.npy` files left in a reused output directory as results of the current run. Use a fresh **results subfolder**, not a new project directory.

## In-place installation (one project, one Conda env)

Save the ZIP to `F:\Edge`, then in WSL:

```bash
conda activate gsrefine
cd /home/xth/GSSceneRefine
unzip -o /mnt/f/Edge/GSSceneRefine_roi_simple_geometry_update.zip -d /home/xth/
python -m pytest -q tests
python run.py roi-study --help
```

First analyze only the Songzi Guanyin ROI002:

```bash
python run.py roi-study \
  --config configs/songziguanyin.yaml \
  --roi runs/roi_init/songziguanyin_absgs_roi_002.json \
  --output-dir runs/roi_tests/songziguanyin_roi002_simple_geometry \
  --no-ablation
```

Look at `study_report.json` -> `poisson_face_scoring.enabled` (must be `false`), `candidate.algorithm` (must be `simple_geometry_v1`), `candidate.areas_by_label`, and `background_components.json`. All-face assumption changes the denominator, so **do not expect the historical scored 32.90% ratio to remain unchanged**.

After visually checking `exterior_normal`, `exterior_suspect`, `exterior_uncertain` against the full Poisson and Gaussian scene:

```bash
python run.py roi-study \
  --config configs/songziguanyin.yaml \
  --roi runs/roi_init/songziguanyin_absgs_roi_002.json \
  --output-dir runs/roi_tests/songziguanyin_roi002_simple_geometry
```

This writes R0–R3 Boolean masks; add `--export-ply` only if large Gaussian PLYs are wanted. For full-source recovery, remember the current configured input is **AABB-cropped**: Gaussians already deleted by that earlier AABB crop cannot be restored from this file.

## Conservative adjustable parameters (optional)

Default values are defined in `simple_geometry._test_config`. They are **starting experimental tolerances**, not scientifically calibrated universal constants. Avoid fitting them to Songzi Guanyin alone; use other held-out Danxiashan scenes and independently verified surfaces. You can override a value under `roi_study:` in your existing scene YAML *later*, but this patch intentionally does **not** edit any user configuration.

`background_min_full_to_internal`, `background_min_full_to_local`, `background_min_contact_edges`, `background_min_base_fraction`, `background_min_ring_coverage`, `background_max_local_height`, `local_max_area_to_internal`, `local_relaxed_area_to_internal`, `local_relaxed_distance_fraction`, `local_max_contact_coverage`.

## Validation checks

- `∑ areas_by_label = initial_exterior_candidate_area` within numeric tolerance.
- `normal`, `suspect`, `uncertain` label masks disjoint and subset of local candidate.
- Sum of all distinct full-exterior component areas must **not** count the same mesh faces multiple times.
- `R2` and `R3` Gaussian masks must contain the entire `R0` mask.
- Global full exterior never becomes automatic Gaussian recovery.
- Manual `--labels` overrides remain supported and are explicitly indicated in the output.
