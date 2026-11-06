
## Experimental ROI boundary-topology study — 2026-11-05
- Added `roi-study` CLI in the existing project root. Reuses the existing initial exterior candidate extraction and Gaussian protection functions.
- Added native COLMAP-coordinate ROI/Poisson boundary-trace sampling, boundary-source grouping, base-wrapping diagnostics and multiscale geodesic surface-area curves.
- Added explainable three-way patch triage (normal / suspect / uncertain) and R0–R3 research masks with optional PLY exports.
- Added synthetic tests, optional reviewer labels, SHA256 provenance and research limitations. Existing ROI and configs untouched.
## 2026-11-04 — ROI restoration risk budgets

- Corrected a scientifically misleading all-or-nothing 10% exterior-surface-area veto: percentage now flags bounded candidates as provisional for visual inspection by default; strict `reject` policy remains opt-in.
- Corrected independent 12% added-Gaussian all-or-nothing veto in the same way.
- Added connected surface-patch diagnostic statistics and `exterior_candidate_surface.ply` / `exterior_candidate_faces.npz` outputs.
- Added explicit `--export-ply` on `roi-refine`; preliminary flagged exports receive `_PROVISIONAL` suffix.
- Existing `gsrefine` Conda environment, user-edited configuration, ROI JSONs, and SfM/Poisson/Gaussian sources are not modified.

## 2026-11-04 — ROI initialization / refinement handoff

- Module 2 always saves mathematically valid confirmed polygons as new sequential ROI JSON files.
- Module 2 does not reject a valid ROI for subject truncation; semantic/geometric recovery belongs to the following module.
- Manifest schema 1 gains backward-compatible `scene_latest`; the original `scene_canonical` is preserved.
- `roi-refine` defaults to latest scene ROI across methods sharing one verified COLMAP reconstruction; `--roi` pins historical versions and `--show-roi` inspects the selection.
- Browser load-existing uses latest ROI **only on request**; every new session starts blank.
- No changes to original Gaussian PLY, Poisson data, Conda environment, or legacy stages.

# Changelog

This file tracks meaningful source changes. Git commits and tags, not cloned folders, preserve code revisions.

## Full-scene SOG display & dependency consolidation (current)

- SplatTransform 3.10.3 is now a required UI package; Node.js 22 is the single-environment runtime baseline.
- Add `bootstrap.sh --sync-deps` to install/update only needed Node and npm packages in-place.
- Add `ui --build-display-cache --cache-device cpu|auto` for first-run one-command cache setup.
- Extend read-only `doctor` checks to Node.js and SplatTransform.
- Keep manual PLY fallback and display-only sampling as explicit debugging options; no redundant source trees.

## Browser model-load acceleration (previous)

- Add optional reusable bundled SOG visualization cache via MIT-licensed SplatTransform.
- Load complete SOG in PlayCanvas when available, keep original full-precision PLY for science.
- Add HTTP ETag/Range/browser caching for localhost model transfers and tests.
- Preserve one Conda environment (`gsrefine`) and project folder; no environment.yml change.
- Document exact local cache build/launch commands and open-source attribution.

## Architecture hardening

- Preserve one project root `/home/xth/GSSceneRefine` and one `gsrefine` Conda environment.
- Keep all existing production/test source module names and entrypoints stable.
- Make `bootstrap.sh` safe for pre-existing Conda environments: no automatic `conda env update --prune`.
- Add read-only `python run.py doctor` and optional `--check-data` checks.
- Document module ownership, in-place update policies, local config isolation, migration gates and data interfaces.
- No changes to `environment.yml`, scientific ROI computation or the three legacy Poisson/SDF workflows.

## Initial integrated project

- GeoGuard-ROI geometric core and `hard/global/local` ablation modes.
- Dual orthographic PlayCanvas workspace and ROI JSON export.
- Optional standalone SfM filtering and Poisson preparation prototypes.
- Stable CLI `run.py`, synthetic Python/JS tests and local HTTP handoff.

## Module 2 — native Windows ROI initializer (2026-11-04)

- Added independent `python run.py roi-init --config ...` WSL launcher.
- Added a PowerShell-only, native Windows, loopback model HTTP server with one-file allowlist and Range support.
- Switched compact PlayCanvas UI to real WebGPU creation (`AppBase` + `GSplatHandler`); one shared GSplat asset for the top/front cameras.
- Used verified COLMAP vertical-axis configuration; UI can no longer silently change it.
- Added exact orthographic multi-polygon Gaussian-center intersection preview on full source PLY and sparse point overlay.
- Added confirmed atomic JSON writing to a common folder, scene/method/number naming, and scene-wide manifest/canonical ROI reuse.
- Exposed the new `roi-refine` handoff command; left previous geometry/Python and old Poisson stages unchanged.
- Kept SOG preprocessing and the old UI server as optional legacy tools rather than deleting working prior functions before real-scene regression testing.

## Module 2 automatic COLMAP orientation (2026-11-04)

- Replace manual `+z` / `verified` assumption with arbitrary original-COLMAP unit up vector.
- Estimate robustly from registered roll-stabilised UAV camera-right axes, resolve sign from down-looking optical axes, and report acceptance diagnostics; fail closed on insufficient evidence.
- Cache orientation per source COLMAP frame, with original binary SHA-256 provenance. Never rotate the source Gaussian, SfM, or Poisson geometry.
- Add optional camera-trajectory and raw-Poisson-face-normal diagnostics (`run.py orientation --check-poisson`); neither is falsely treated as ground truth for complex cliffs.
- Update browser top/front orthographic projection and ROI manifests to save precise `up_vector` and coordinate provenance.
- Add a narrow, backed-up, comment-preserving config migration tool; keep user model paths and all historical experiment data unchanged.
