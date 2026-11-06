# ROI full-exterior import compatibility fix

## Symptoms
`pytest` stops during collection with:
`ImportError: cannot import name 'component_background_evidence' from 'gsrefine.roi.candidate_classification'`

## Cause
`gsrefine/roi/study.py` in the full-exterior update imports the component-aware
helper functions added in the preceding background-continuation update. The
full-exterior ZIP did not bundle that previously updated dependency.

## Contents
- `gsrefine/roi/candidate_classification.py`: latest verified background-component classifier, with both `component_background_evidence` and `apply_component_background_evidence`.
- `tests/test_roi_full_surface_import_contract.py`: safeguards the import contract and ensures one-sided boundary contact is not misclassified as surrounding background.

## Install (original location, no new project or environment)

```bash
conda activate gsrefine
cd /home/xth/GSSceneRefine
unzip -o /mnt/f/Edge/GSSceneRefine_roi_full_surface_import_fix.zip -d /home/xth/
python -m pytest -q tests
python run.py roi-study --config configs/songziguanyin.yaml --roi runs/roi_init/songziguanyin_absgs_roi_002.json --output-dir runs/roi_tests/songziguanyin_roi002_boundary --no-ablation
```

The update does **not** modify core.py, study.py, full_surface.py, ROI JSON,
configs, test data, or any Gaussian/Poisson model. Existing CSV/PLY results
are overwritten only when you explicitly re-run `roi-study`.

If you are not sure the zip is present in F:\Edge, copy/download it there first.
