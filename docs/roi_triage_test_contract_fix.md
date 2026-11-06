# ROI triage test contract synchronization

This incremental update changes only one assertion in `tests/test_roi_boundary_study.py`.

## Why

The older test expected `triage()` to return `normal` from only one
side-boundary observation (`height_quantile=.15`, `angular_coverage=.8`,
unsaturated growth). The current geometry method deliberately defers background
classification to `component_background_evidence()` using real Poisson face
connectivity and multi-edge, base, azimuth and growth evidence.

The correct single-edge outcome is `uncertain`, not `normal`.

## Impact

No production code, scene data, ROI files, configuration values, numerical
thresholds, or Conda environment are changed. The high local clipped-top
`'suspect'` assertion is retained.

## Validate

```bash
cd /home/xth/GSSceneRefine
python -m pytest -q tests/test_roi_boundary_study.py tests/test_roi_background_components.py tests/test_roi_full_surface_import_contract.py
python -m pytest -q tests
```

The patch is safe to apply in-place after backing up your test file.
