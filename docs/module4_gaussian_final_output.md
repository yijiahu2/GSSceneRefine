# Module 4 — Gaussian ROI extraction and reproducible output

This is the **fourth module of Stage 1**, not the downstream Poisson–SDF internal artifact cleanup.

## Algorithm contract

For the Gaussian indices in the **same original binary PLY**:

- `K0 = keep_mask_R0`: all Gaussian centers accepted by the user-drawn ROI.
- `K1`, `K2`, `K3`: saved by `roi-study`, each including every index in `K0` plus a specific bounded geometry-protection set.
- The chosen final method `Rm` outputs **exactly** `K_m` of the source PLY, in source order and without changing any fields, including RGB/SH, opacity, scale, quaternion, or nonstandard scalar vertex properties.
- `K_m = K0 ∪ Δ_m` is validated; the added mask must be `Δ_m = K_m \ K0`. The finalization step **does not rerun Poisson classification** and **does not apply SDF cleanup**.

No Gaussian is created or synthesized. Missing Gaussians already removed from the **input AABB PLY** cannot be recovered; to recover them the entire, unmodified training-source Gaussian PLY is required.

## One-shot Stage-1 workflow (modules 3 + 4)

```bash
conda activate gsrefine
cd /home/xth/GSSceneRefine
python run.py roi-study \
  --config configs/songziguanyin.yaml \
  --roi runs/roi_init/songziguanyin_absgs_roi_002.json \
  --output-dir runs/roi_tests/songziguanyin_roi002_R0_R3
```

Without additional flags, `roi-study` evaluates R0–R3 then **automatically exports only the R3 full-scene Gaussian PLY** to:

```
runs/roi_tests/songziguanyin_roi002_R0_R3/gaussians_final_R3.ply
```

If you need the four comparison PLYs, add `--export-ply`; this uses more storage. To choose R2 as the automatically exported final scene add `--final-method R2`. To compute masks only, add `--no-final-export`. If you use `--no-ablation`, there are no R1–R3 masks and therefore no R3 export.

## Export from an **already completed** R0–R3 experiment (no recomputation)

```bash
python run.py roi-finalize \
  --study-dir runs/roi_tests/songziguanyin_roi002_R0_R3 \
  --config configs/songziguanyin.yaml \
  --method R3
```

Choose `--method R0`, `R1`, or `R2` for comparisons, e.g.:

```bash
python run.py roi-finalize \
  --study-dir runs/roi_tests/songziguanyin_roi002_R0_R3 \
  --config configs/songziguanyin.yaml \
  --method R0
```

`roi-finalize` reads `study_report.json` and the existing `keep_mask_Rm.npy` and `added_mask_Rm.npy`. It refuses a changed source PLY (SHA256 mismatch), a changed ROI JSON, inconsistent mask length/count, missing original R0 Gaussians, or inconsistent added masks. For new reports, it also verifies the SHA256 of the masks. Older experiments without per-mask SHA256 remain compatible through count and subset checks.

You may choose another output PLY pathname using `--output /mnt/f/.../your_scene.ply`, but this does not change which Gaussian indices are selected.

## Module 4 output files

| Filename | Meaning |
|---|---|
| `gaussians_final_R3.ply` | **Full kept Gaussian scene** from source model. Open in SuperSplat. |
| `keep_mask_R3.npy` | Boolean mask over original PLY rows. |
| `added_mask_R3.npy` | Rows restored outside the original R0 ROI. |
| `kept_indices_R3.npy` | Source row indices of all final Gaussians. |
| `restored_indices_R3.npy` | Source row indices of newly restored Gaussians only. |
| `final_gaussian_report_R3.json` | Input/output hashes, counts, retention ratio, source scope, validation flags. |
| `study_report.json` | Earlier ROI classification and R0–R3 study; now includes a link to the final scene. |

## Validation gates

1. `|K_m| == |K0| + |Δ_m|`.
2. No `K0` Gaussian is discarded by module 4.
3. Input PLY SHA256, row count and ROI reference are consistent with the study.
4. Output PLY has the **same field layout** as input; sampled records are byte-for-byte unchanged.
5. Output and masks retain source row order and indices.
6. This output represents **geometry-only Stage 1**. It should not be described as cleaned of interior hallucinations. Future Poisson–SDF cleanup is downstream and independently testable.

## Current research caveat

With `source_model_scope: aabb_baseline_only` and `paths.gaussian_ply: .../roi_aabb.ply`, all restored Gaussians must already exist in the Task-1 AABB model. The final PLY is a valid **AABB-baseline** scene, not an extraction from the original 5.38-million-Gaussian checkpoint.
