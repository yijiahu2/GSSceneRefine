# GSSceneRefine — Long-term development contract

> Canonical WSL root: `/home/xth/GSSceneRefine`  
> One Conda environment: `gsrefine`  
> Repo identity: `GSSceneRefine`  
> Scientific ROI method: `GeoGuard-ROI` (a module, not a separate repo).

## 1. Do NOT duplicate project roots or environments

- All future features and fixes are **edits to files in this repository**. No `GSSceneRefine_v1/`, `stage2_new/`, `project_final2/` or stage-specific Conda environments.
- Use **Git commits and tags** for version history; do not version by copying entire code directories.
- Never overwrite `/home/xth/datasets/`, `/mnt/f/danxia_experiment_archive/` or previous experiment output as a side effect of a code update.
- A new script belongs under one existing module (`gsrefine/geometry`, `roi`, `cleanup`, `web`, future `evaluation`); do not add new root-level scripts for each experiment.
- Code imports must use `gsrefine.*` and stable entrypoint `python run.py <command>`. No imports from personal Conda or other project source directories.

## 2. Stable naming and file responsibilities

| Stable path | Role | Naming policy |
| --- | --- | --- |
| `run.py` | Public CLI dispatcher, current and future subcommands | Keep permanent |
| `environment.yml` | Single reproducible `gsrefine` dependency declaration | Update in place when project dependencies change |
| `bootstrap.sh` | Idempotent environment setup and smoke test | Update only required missing/outdated runtime packages in-place |
| `gsrefine/geometry/` | Original COLMAP/SfM quality filter, Poisson, geometry confidence | New methods under descriptive filenames; no stage-prefixed file tree |
| `gsrefine/roi/core.py` | Existing GeoGuard-ROI distance, surface protection, Gaussian masks | Keep CLI/API working; later factor helpers out without breaking the wrapper |
| `gsrefine/web/server.py` | Local ROI workspace HTTP backend | Keep endpoint contracts stable |
| `ui/` | PlayCanvas frontend, orthographic ROI editor | Browser-only, no scientific data mutation |
| `gsrefine/cleanup/` | Future Poisson/SDF external and internal artifact removal | Reserved; not implemented by the current project |
| `gsrefine/evaluation/` | Future scene integrity, rendering, efficiency and ablation evaluation | Add only when implemented |
| `configs/*.yaml` | Tracked reproducible **examples/defaults** | Never store user machine secrets or one-off absolute data assumptions as canonical |
| `configs/local/*.yaml` | **Untracked** machine-specific configurations | Copy and edit examples locally; preserve across updates |
| `integrations/` | Migration register for previous validated scripts | Source is NOT migrated until equivalence checks complete |
| `data/` | Optional data landing area (normally external paths) | Never commit huge original assets |
| `runs/<scene>/<experiment>/` | Derived masks, geometries, reports, logs | Generated output, not implementation |

Existing scripts `songziguanyin_colmap_poisson`, `poisson_stage2_unified`, and
`poisson_stage3_compare` are **external validated legacy workflows** for now.
Do not rename or change their previous results when preparing a new feature.

## 3. Changes without environment recreation

**Normal code update:** replace existing source in place and run tests. If dependencies change, maintain the SAME `gsrefine` environment; install, upgrade, or remove packages as necessary. No per-stage environment or duplicate source tree.

```bash
cd /home/xth/GSSceneRefine
conda activate gsrefine
python run.py doctor --check-data --config configs/local/songziguanyin.yaml
python -m pytest -q tests
node --test ui/tests/*.test.mjs
```

To initialize a local config once, use
`mkdir -p configs/local && cp configs/songziguanyin.yaml configs/local/songziguanyin.yaml`.
Then always pass `--config configs/local/songziguanyin.yaml` to UI/ROI commands.
Tracked example configs remain stable while local paths persist.

**`bootstrap.sh` behavior:** `--check` is read-only. Default and `--sync-deps` maintain required Node.js 22 and PlayCanvas/SplatTransform packages in the existing `gsrefine` environment, never recreating it. `--update-env` applies the complete `environment.yml` without pruning unrelated packages. Dependencies are driven by the project, not by a rule forbidding environment maintenance.

## 4. Source and experiment validation gate

Every meaningful change must satisfy:

1. Syntax and unit tests pass (`pytest` + JS tests).
2. CLI and browser ROI JSON format remain compatible or a documented schema version is provided.
3. Same input and parameters yield the same keep masks (or numerical tolerances and drift are documented).
4. No source asset is silently changed: preserve original SfM IDs, Gaussian vertex ordering, PLY properties, coordinate system and units.
5. GPU display sampling/cropping is VISUALIZATION ONLY, never scientific input.
6. Outputs include config snapshot, input hashes and method/run metadata. Run directory should be a distinct named **experiment output**, not a new code directory.
7. For changes in Poisson/SDF or ROI protection, compare against the last validated real-scene results before promoting the new behavior as default.
8. Code updates and scientific experiment results are separate claims; synthetic tests cannot prove real scene quality.

## 5. Git workflow

```bash
cd /home/xth/GSSceneRefine
git init                     # one time only if repo is not already initialized
git status --short
git add README.md docs gsrefine ui run.py bootstrap.sh environment.yml configs tests integrations tools .gitignore
git commit -m "chore: establish stable GSSceneRefine project architecture"
```

Subsequent changes: edit existing files, run tests, `git diff`, then `git commit`.
Use tags like `v0.3.1` for a release, **not** a new folder called `v0.3.1`.
Do not commit `runs/`, `data/`, `configs/local/` or `ui/node_modules/`.
If obsolete files or duplicated implementations cause confusion, remove them after confirming no active imports and passing regression tests; use `git rm` to record removal. If files must be renamed, use `git mv` and preserve a compatibility import/CLI alias until callers are updated.

## 6. Later migration of the three existing Poisson/SDF workflows

For each workflow: locate original entrypoint, config, input metadata and verified outputs; record SHA-256; run original scientific regression; migrate source to `gsrefine/geometry` or `gsrefine/cleanup`; adapt dependencies in ONE `gsrefine` environment; compare critical metrics/masks; only then register a stable command in `run.py` and retire old execution paths.

**No migration is claimed in the current release.** Details in `integrations/README.md`.
