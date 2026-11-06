# Scene configuration

**Use `configs/default.yaml` for all new scenes and Gaussian methods.** It is an algorithm-parameter template with no paths. Give the original COLMAP reconstruction, full Gaussian PLY, scene/method labels and output root on the command line. The program generates a run-specific effective YAML in `<output-root>/<scene>/geometry/runtime/` without changing `configs/default.yaml`.

```bash
python run.py roi-init \
  --config configs/default.yaml \
  --scene bazhaifeng --method absgs \
  --sfm-dir /path/to/bazhaifeng/colmap/sparse \
  --gaussian-ply /path/to/bazhaifeng/absgs/point_cloud.ply \
  --output-dir /path/to/GSSceneRefine_results
```

To reuse a confirmed ROI under a second Gaussian method (only for the **same COLMAP reference frame**):

```bash
python run.py roi-auto \
  --config configs/default.yaml \
  --scene bazhaifeng --method vanilla \
  --sfm-dir /path/to/bazhaifeng/colmap/sparse \
  --gaussian-ply /path/to/bazhaifeng/vanilla/point_cloud.ply \
  --roi /path/to/GSSceneRefine_results/bazhaifeng/roi/roi_001.json \
  --reuse-scene-roi \
  --output-dir /path/to/GSSceneRefine_results
```

- `configs/songziguanyin.yaml` remains only for historical reference, **not** the general entrypoint. It contains fixed local paths and legacy flags.
- `configs/roi_init.example.yaml` is an **incomplete module-2 fragment**, not a full pipeline configuration.
- `--scene` is the stable identity of a COLMAP reconstruction. Different SfM reference frames must never share the same scene ID, even when the physical landscape is the same.
- `--method` labels a Gaussian model independently from the scene geometry; models must be aligned to that COLMAP world frame.
- All generated data, including working configs, geometry caches, ROI definitions, exports and logs, is organized under `--output-dir`. Do not check in live results.
- The default numerical settings are a reproducible **baseline**, not a guarantee of optimal cross-scene surface reconstruction or recovery. Tune experimentally in a deliberate config copy if needed, recording the resulting effective runtime YAML.
