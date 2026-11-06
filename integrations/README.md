# Verified prior workflows (not integrated in current release)

The following three user-verified workflows are **not bundled or executed** by v0.3 and should not be replaced or rewritten before validating their source, configuration, and real-scene results:

1. `songziguanyin_colmap_poisson` — original COLMAP SfM cleaning and Poisson reconstruction.
2. `poisson_stage2_unified` — sealed Poisson and shared SDF offset logic.
3. `poisson_stage3_compare` — Gaussian/Poisson/SDF cleanup comparisons.

Future consolidation target:

| Existing workflow | Future package location | Data interface |
| --- | --- | --- |
| `songziguanyin_colmap_poisson` | `gsrefine.geometry` | Original COLMAP -> clean SfM -> observed Poisson + confidence |
| `poisson_stage2_unified` | `gsrefine.cleanup` | Observed mesh -> sealed geometry + SDF reference |
| `poisson_stage3_compare` | `gsrefine.cleanup` | Gaussian + geometry -> cleanup masks, PLY, diagnostics |

Before any migration, inventory each script and configuration, hash and freeze source files, reproduce its output in the original environment, then run independently under `gsrefine` and compare output metrics. Avoid claiming integration until those checks pass. No old conda environment is required for current GSSceneRefine code.
