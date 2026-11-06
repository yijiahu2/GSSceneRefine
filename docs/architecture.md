# GSSceneRefine architecture and module contracts

**A single stable repository and one `gsrefine` Conda environment** for complete 3DGS natural scene refinement.

## Logical data flow (modules, NOT version/stage folders)

```text
Original COLMAP points3D/images       Original GSPL/3DGS checkpoint
                |                                  |
                v                                  v
     gsrefine.geometry                      full Gaussian PLY
   SfM filtering + Poisson                      (one-time export)
                |                                  |
         original observed mesh                   |
                +---------------+------------------+
                                v
               gsrefine.roi_init.server / ui
        Module 1 model load → Module 2 orthographic ROI confirm
                                |
                    auto workflow dispatcher
                                v
                   gsrefine.roi.study
        Module 3 Poisson local geometry recovery, then R3
              KNN-trimmed full-boundary safety band
                                |
                                v
                   gsrefine.roi.finalize
          Module 4 validated Gaussian subset PLY
                                |
                                v
                  gsrefine.cleanup (FUTURE)
            sealed SDF / internal artifact cleanup
                                |
                                v
                 gsrefine.evaluation (FUTURE)
             quality / completeness / efficiency
```

## Stable interfaces

| Artifact | Schema/contract | Owner |
| --- | --- | --- |
| Clean SfM | `.npz`: `xyz` in original COLMAP world coordinates; optional `error`, `track_length`, `point_id` | geometry |
| Poisson observed mesh | `.ply` triangle mesh, same coordinates as SfM, no artificial skirt/base used for ROI protection | geometry |
| Gaussian model | GSPL native binary little-endian `.ply`, positions + log scales + quaternions | external Gaussian exporter / ROI reader |
| User ROI | `.json` containing `coordinate_system=original_COLMAP_world`, `polygon_units=COLMAP_model_units_not_pixels`, `views[]` records each with `name`, `origin[3]`, `axes[2][3]`, `polygon[][2]` | web/UI |
| ROI keep mask | `.npy` boolean vector in **input Gaussian PLY vertex order** | roi |
| ROI guard surface samples | `.npz` with `points` and `radii` in COLMAP units | roi |
| Run report | `report.json` + `config_snapshot.yaml`, hashes and count diagnostics | roi / future modules |

**Important:** The generated `.npy` mask belongs to the particular input PLY row order. It cannot automatically index the original training checkpoint after resampling, reordering, or external filtering. Record source hash, input scope and coordinate transform.

## Code ownership and future modules

- `gsrefine/geometry`: reproducible SfM filtering, trust/confidence and original observed Poisson surfaces. Previously verified `songziguanyin_colmap_poisson` remains external until migrated.
- `gsrefine/roi`: GeoGuard-ROI spatial selection, bounded trusted-surface extension, model-specific Gaussian safety envelope, preserve-original polygon union.
- `gsrefine/web`, `ui`: local dual workspace front/back; only render and annotate, not decide science. Local browser fetches PLY from the loopback server.
- `gsrefine/cleanup`: reserved for previously validated `poisson_stage2_unified` and `poisson_stage3_compare`; **not bundled yet**.
- `gsrefine/evaluation`: future reference/free-view comparison and quantitative analysis; don't invent metrics before implementation.
- `integrations`: migration mapping and scientific regression checklist, NOT a copy of old code.

Current `run.py` commands include `roi-init` (Modules 1+2; auto-launches Modules 3+4 after confirmation), `roi-auto` (replay one confirmed ROI), `roi-study` (individual scientific experiment), `roi-finalize`, `ui` (legacy), `roi`, `sfm-clean`, `poisson`, `doctor`.

## Invariants

- One repo `/home/xth/GSSceneRefine`; one environment `gsrefine`.
- Original COLMAP and Gaussian coordinates may not be assumed to align with geographic vertical. View axes must be calibrated and serialized.
- All methods use the same underlying input reference; preview-only display decimation cannot affect official Gaussian mask.
- Scientific output and code are stored separately (`runs` vs package); program never modifies original inputs.
- Version history through Git commits/tags, not separate duplicate source trees.
- The `GeoGuard-ROI` method is a component of `GSSceneRefine`, not the umbrella project itself.

See [`development.md`](development.md) for the in-place update process and testing gates.
