# GSSceneRefine: four-module architecture

This repository has one code root and one Conda environment (`gsrefine`).

```text
Raw COLMAP points3D.bin + matching images.bin    FULL Gaussian PLY
               |                                      |
               +----------------------+---------------+
                                      |
                         Module 1: geometry.prepare
          COLMAP observation/reprojection/SOR filtering
          normals, Screened Poisson, spatial QA and cache
                                      |
               runs/geometry/<scene>/active_scene.yaml
                                      |
                       Module 2: roi_init.server + UI
                      top / front orthographic ROI
                          confirmed ROI JSON
                                      |
                    workflow.py persistent task manager
                                      |
                      Module 3: roi.study, R3 only
                 local Poisson core -> KNN-trimmed SDF
                   safe margin on all core boundaries
                                      |
                       Module 4: roi.finalize
                     final Gaussian subset PLY
```

- Modules 1 and 2 share an effective runtime config, with identical original Gaussian PLY source and original COLMAP coordinate system. Cached Module-1 geometry and the original source assets are never edited as a side effect of ROI selection.
- `run.py roi-init --colmap-points ... --gaussian-ply ...` creates `runs/geometry/<scene>/input_scene.yaml`, executes or reuses Module 1, then starts Module 2. The tracked scene config remains unchanged.
- Module 1 produces `02_sfm_clean_colmap_metadata.npz` and `04_poisson_raw.ply`, plus validation/caching reports. This reference mesh is not the final renderable result and passing geometry QA is not a per-triangle truth guarantee.
- The browser only confirms polygons; the backend's R3 uses full original PLY vertex order. An annotation recorded on AABB cannot silently become a confirmed annotation for full original PLY.
- Module 3 unchanged scientific recovery: two-stage core then all-boundary safety, K=128, 5% trimmed tails, implicit field with a bounded sampling error. Module 4 exports PLY and checks final counts.
- Workflow steps and true success/failure are written to `workflow_progress.json` and `workflow_result.json` under the ROI result directory. The UI polls `/api/workflow-status` and can reconnect.
- `roi-study` R0–R3 ablation, `roi-finalize` manual export, `sfm-clean` and `poisson` stand-alone commands are retained.
- The only scientific output at the end of Stage 1 is the **Gaussian ROI**, not a Poisson mesh. Internal anomaly cleanup is reserved for Stage 2.

See `README.md` for commands and input contract, `docs/development.md` for source and release rules.
