# GSSceneRefine v8.2 / Stage-2 audit timer hotfix

## Root cause

The installed v8.1 source has `with measured('engineering_cross_stage_and_risk_audit'):` in
`gsrefine/cleanup/local_region_pipeline.py`. The function `measured` is a decorator,
not a context manager. Running it in a `with` raises `AttributeError: __enter__`
after expensive classification and auxiliary protection.

The correction is to import `timed` from `stage2_runtime` at **module level** and use
`with timed('engineering_cross_stage_and_risk_audit'):`. `measured` remains used for
function-call instrumentation. No decision masks or mathematical rules are changed.

## Apply

```bash
cd /home/xth/GSSceneRefine
conda activate gsrefine
unzip -o /mnt/f/Edge/GSSceneRefine_v8_2_stage2_audit_hotfix.zip -d /home/xth/GSSceneRefine
python tools/gsrefine_v8/hotfix_timing_v8_2.py --check
python tools/gsrefine_v8/hotfix_timing_v8_2.py
python -m py_compile gsrefine/cleanup/local_region_pipeline.py
ROI_RUN="/mnt/f/danxia_experiment_archive/GSSceneRefine_benchmark/GSSceneRefine_results/lfls/methods/gsplat_accel/roi_003"
python tools/gsrefine_v8/hotfix_timing_v8_2.py --preflight-manifest "$ROI_RUN/metadata/manifest.json"
```

## Resume Stage 2 on the completed LFLS roi_003 (not Stage 1)

```bash
ROI_RUN="/mnt/f/danxia_experiment_archive/GSSceneRefine_benchmark/GSSceneRefine_results/lfls/methods/gsplat_accel/roi_003"
python run.py pipeline stage2 --manifest "$ROI_RUN/metadata/manifest.json" --accept-auto-exterior
```

**Current limitation:** The official stage2 runner creates a **new** stage2 run folder
for a retry. It reuses verified persistent geometry, but the `original-region`
classification and auxiliary distance lookup may recompute. The previous
failed directory is not advertised as a completed / checkpointed classification.
Do not delete the original Stage-1 result or reuse an unverified partial
classification. Real LFLS completion must be verified from the generated
`stage2_final_report.json` and `engineering_audit.json`.
