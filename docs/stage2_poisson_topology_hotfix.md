# Stage-2 Poisson topology hotfix (2026-11-07)

## Scope and failure

Local error during `python run.py stage2-cleanup --manifest ... --no-export`:

```text
Neither skirt/base direction passed topology+ALL-camera checks
Mesh has 1 non-manifold edges (>2 faces); cannot safely skirt.
```

The affected first attempt used an unset/incorrect `ROI_RUN`. The second traceback confirms the manifest is now accessible. Neither `up_sign` trial reached a valid camera-outside test because both failed in the same geometric-topology precondition.

## Changes (only the Stage-2 files)

- `gsrefine/geometry/sdf_sealing.py`: `prepare_open_poisson()` now sanitizes an **in-memory copy** of the stage-1 reference by merging coincident vertices to the same 8-decimal world-coordinate precision as the historical validated stage-2 Poisson code; removes degenerate and exact duplicate triangle faces; records before/after non-manifold edges. No ambiguous face is removed, and no unknown cavities are auto-filled.
- `gsrefine/cleanup/direct_sdf.py`: calls topology sanitation *after* measuring SfM-to-raw-Poisson P95, *before* selecting the dominant mesh component and attempting both skirt/base directions. Adds topology diagnostics to the sealed geometry report, and includes the hotfix algorithm revision and weld precision in the cache key.
- `tests/test_stage2_cleanup.py`: adds tests for a duplicate-triangle case causing exactly one non-manifold edge and for a geometrically ambiguous three-distinct-triangle case that must still stop.

This hotfix does **not** modify stage-1 meshes, ROI polygons or Gaussian records. It does not change the mathematical threshold `P95 + P10–P90 trimmed mean 3σ Gaussian major radius` or the direct SDF classifier. It does **not** read GitHub or any remote repository.

## Installation to the same project / environment

Before applying the ZIP, optionally inspect `git status --short` and back up locally modified versions of the two Stage-2 Python files. The archive does not include or replace `run.py`, ROI algorithms, or your environment specification. Apply it **in place**:

```bash
conda activate gsrefine
cd /home/xth/GSSceneRefine
unzip -o /mnt/f/Edge/GSSceneRefine_stage2_topology_fix.zip -d /home/xth/GSSceneRefine
python -m pytest -q tests/test_stage2_cleanup.py
```

Optional fast local topology precheck (without calculating SfM P95):

```bash
python - "$ROI_RUN/metadata/manifest.json" <<'PY'
import json, sys, trimesh
from pathlib import Path
from gsrefine.geometry.sdf_sealing import prepare_open_poisson, choose_main_component, boundary_loops
manifest = json.loads(Path(sys.argv[1]).read_text(encoding='utf8'))
source = Path(manifest['stage2_inputs']['poisson_mesh'])
raw = trimesh.load(source, process=False)
working, report = prepare_open_poisson(raw)
print('Topology cleanup:', json.dumps(report, ensure_ascii=False, indent=2))
main, comp = choose_main_component(working, min_fraction=.85)
print('Main component:', comp['main_face_fraction'])
loops, diag = boundary_loops(main)
print('Boundary loops:', len(loops), json.dumps(diag, ensure_ascii=False))
PY
```

Then run your previous command without modifying any Stage-1 outputs:

```bash
python run.py stage2-cleanup --manifest "$ROI_RUN/metadata/manifest.json" --no-export
```

If another topology or projection error appears, **do not** use mesh-wide blanket face deletion, `--force`, or disable water-tightness/camera checks. Supply the updated error and topology report. No real Songzi Guanyin mesh was included in the code archive, so real-scene success still needs to be validated on the user's machine.
