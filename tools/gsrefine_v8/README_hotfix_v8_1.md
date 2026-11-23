# GSSceneRefine v8.1 — Stage2 measured variable shadowing fix

The v8 patch inserts `from gsrefine.cleanup.stage2_runtime import measured` inside `local_region_pipeline.run`, after an earlier `measured(...)` invocation. Python then compiles `measured` as a local for that whole function and raises `UnboundLocalError` at the first use.

This hotfix removes only that function-local import, leaving the existing top-level import, all original method decisions, and output paths unchanged. It explicitly checks the source anchor, verifies the resulting Python AST, makes a source backup, and is safe to run twice.

Install with `unzip -o /mnt/f/Edge/GSSceneRefine_v8_1_stage2_hotfix.zip -d /home/xth/GSSceneRefine`.

Run from `/home/xth/GSSceneRefine`:

```
python tools/gsrefine_v8/hotfix_measured_v8_1.py --check
python tools/gsrefine_v8/hotfix_measured_v8_1.py
python -m py_compile gsrefine/cleanup/local_region_pipeline.py
```

Do NOT rerun Stage1. Resume from `/mnt/f/danxia_experiment_archive/GSSceneRefine_benchmark/GSSceneRefine_results/lfls/methods/gsplat_accel/roi_003/metadata/manifest.json` with `python run.py pipeline stage2 --manifest ... --accept-auto-exterior`.
