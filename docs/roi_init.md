# Module 2 — user-guided dual-orthographic ROI initialization

## Purpose and boundaries

Module 2 owns **only** initial ROI annotation, intersection preview/validation, and confirmed output. It must not apply Poisson-based subject restoration or edit source Gaussians. The core formula is

```
Omega_0 = intersection_over_views( inverse_projection(P_view) )
keep_i = 1 if ALL views contain the projection of Gaussian center_i
```

The preview count is computed over **all Gaussian centers** in the configured native full-resolution PLY; the 450 green points are a sparse visual *indicator* and not a final clipped rendering. The final Gaussian protection/repair is a separate module (`roi-refine`).

## Stable paths and environment

- Project root: `/home/xth/GSSceneRefine`
- Single WSL Conda environment: `gsrefine`
- Downloads: `/mnt/f/Edge`
- One output folder for all scenes and methods: `/home/xth/GSSceneRefine/runs/roi_init/`
- Existing known-good Poisson/SDF programs are **not** changed in this patch.

## Configure the scene

For an existing working scene configuration, **do not overwrite it with a new YAML**. One-time append the ROI settings (the tool makes a backup and preserves your comments):

```bash
conda activate gsrefine
cd /home/xth/GSSceneRefine
python tools/activate_roi_init.py --config configs/songziguanyin.yaml --scene songziguanyin --method absgs
```

Then edit `configs/songziguanyin.yaml` to verify `scene.name`, `scene.method`, `paths.gaussian_ply`, optional `paths.poisson_raw`, `roi_init.output_dir`, service ports and particularly `roi_init.orientation`. The existing configured AABB PLY and Poisson remain the test inputs; `roi_init` inherits these paths by default.

**Coordinate update:** COLMAP has no fixed world-up axis. Module 2 now estimates a three-component `up_vector` from original COLMAP camera poses, checking axis diversity, residuals and optical sign before opening. The legacy `orientation.up_axis` / `orientation.verified` fields are no longer used in auto mode. See `docs/orientation_auto.md`.

Use automatic estimation of arbitrary COLMAP world-up. After the one-time backed-up configuration migration, the relevant YAML should look like:

```yaml
roi_init:
  orientation:
    mode: auto
    colmap_images: /home/xth/datasets/danxiashan/songziguanyin/sparse/images.bin
    frame_id: songziguanyin_colmap_original
```

The `frame_id` must identify the actual **shared COLMAP world reconstruction**, not just a scene nickname; the same string across methods is a declaration, not proof of geometric registration. Check camera/SfM alignment when switching sources. The estimated vector is then read-only in the browser, preventing silent changes after drawing. The resulting ROI JSON records the original COLMAP coordinates and projection axes; no UI pose is baked into the source PLY.

## Start module 2 (single command)

```bash
conda activate gsrefine
cd /home/xth/GSSceneRefine
python run.py roi-init --config configs/songziguanyin.yaml
```

Open **Windows** Edge at `http://127.0.0.1:8765`.

The WSL launcher starts a separate **native Windows PowerShell single-file TCP HTTP service** on `127.0.0.1:8766`, using Windows's built-in PowerShell and **no Windows Conda/Python/Node installation**. The Windows file service returns only `/model.ply` for the **exact** PLY specified in the config. It refuses all other filenames and paths, binds only to loopback, checks Host and Origin, supports HTTP ranges, and never writes to the source PLY. The HTML, UI JavaScript and small JSON endpoints remain on the WSL server (`127.0.0.1:8765`). This avoids sending the large PLY through WSL's file server or performing CPU PLY→SOG conversion.

The rendering uses the official PlayCanvas WebGPU GSplat handler, as a compact subset of the same open source stack used by SuperSplat. This patch does **not** vendor the full SuperSplat editor; it reuses the efficient engine concepts while keeping only ROI UI features. Real rendering throughput on RTX 3090 has **not** been verified in this execution environment.

### Browser workflow

1. Inspect automatic orientation diagnostics and check the top/front render visually; never force +Z when the camera support is weak.
2. Draw **one** top polygon, and **one to four** elevation polygons from the eight 45° candidates.
3. Click **① 交集核查** to compute exact full-PLY Gaussian center intersection and show a sparse location overlay. Severe empty ROI is blocked.
4. Review that important parts of the subject are visible. The sparse green points are only a diagnostic, and even a normal count cannot guarantee structure preservation.
5. Click **② 确认并保存ROI**. The server atomically writes the JSON and updates the shared manifest; no manual Save As and no heavy core analysis is run.

For method comparison, configure the same scene name and verified COLMAP frame but another `scene.method`. Any confirmed ROI from this scene is reusable across registered methods. New sessions start with blank polygons; the latest saved ROI may be loaded explicitly. Each confirmation automatically creates a new numbered file. The original scene reference stays unchanged; **module 3 defaults to the latest confirmed scene ROI**. There is no need to set `roi_init.allow_new_revision`.

## Outputs

```
runs/roi_init/
  songziguanyin_absgs_roi_001.json
  songziguanyin_absgs_roi_002.json           # next confirmed annotation
  bazhaifeng_citygaussian_roi_001.json       # different scene
  manifest.json                              # scene_canonical + scene_latest + records
  windows_model_service.log                  # runtime log, not source code
```

The JSON contains views (`name`, `origin`, two orthonormal `axes`, and model-unit `polygon`), original COLMAP frame metadata, source PLY and orientation provenance, and the full-model Gaussian-center intersection count. The `manifest.json` separately stores the first scene reference (`scene_canonical`) and last confirmed scene ROI (`scene_latest`). The downstream command reads `scene_latest` by default and checks COLMAP registration consistency.

## Pass to subsequent geometry module

```bash
python run.py roi-refine --config configs/songziguanyin.yaml
```

This calls the unchanged ROI geometry core with the scene's latest confirmed JSON. It must never substitute a model from a different coordinate frame. Use `--show-roi` to inspect selection without running the core algorithm.

## Windows-native file service diagnostics

From Windows PowerShell (if the WSL launcher cannot start the native service automatically):

```powershell
# First check if the configured file server is already running:
Invoke-WebRequest http://127.0.0.1:8766/health -UseBasicParsing

# Test the file's beginning; must show PLY header bytes (model path remains private):
# HTTP Range is supported. To debug, use Edge F12 > Network > model.ply.
```

In WSL, check `runs/roi_init/windows_model_service.log` if startup fails. `powershell.exe` interop, `wslpath -w`, and the Windows loopback listener must all be available. Windows firewall, mirrored networking, and localhost forwarding can affect behavior; if the file service cannot be reached from Edge, check the PowerShell log and available ports. Avoid running legacy `--build-display-cache --cache-device cpu` for module 2.

## Quality gates and remaining validation

Automatic tests verify automatic orientation inference and failure checks, orthographic coordinate round-trips, polygon self-intersection rejection, exact center counts, HTTP save, canonical ROI reuse across 3DGS methods, and atomic outputs. They cannot validate actual Windows PowerShell/Edge WebGPU performance, Poisson anatomical correctness, or a true GPU-clipped preview. Those require your real Songzi Guanyin AABB model and local Windows machine. A point-center intersection by itself cannot detect all cases where a Gaussian ellipsoid extends across a boundary.


### 前端升级后出现 `Cannot read properties of undefined (reading 'slice')`

这通常表示浏览器仍在使用旧版 `up_axis` 的 JavaScript，而当前自动定向服务只提供 `coordinate_frame.up_vector`。结束旧 `roi-init` 服务后重新启动，使用 Edge 的 `Ctrl+Shift+R` 强制刷新。当前项目为 `index.html`、`app.mjs` 和 `geometry.mjs` 禁止缓存，并在模块URL中设置兼容性版本标识；不会改变原始PLY文件的缓存/加载策略。如果仍报错，应检查F12 Console中的完整调用栈及 `/api/meta` 的 `coordinate_frame`。不要把 `+Z` 写回配置来绕过报错。


## Poisson projection and confirmed-save feedback (UI update)

- The Poisson checkbox loads a *sampled, unverified raw Poisson* mesh once and adds a
  visible **orthographic translucent projection/wireframe diagnostic** above the
  Gaussian display. It also attempts to render the 3D mesh. The 2D guide does NOT
  have per-pixel 3D occlusion and must **not** be used as a trustworthy ground-truth
  surface or as a stage-3 subject-protection verdict. Disable with the checkbox.
- Clicking **Confirm and Save ROI** now opens a clear modal with the server-confirmed
  WSL output path, retained center count, and a Copy Path button. Only a successful
  server ACK with a file existing on disk clears both view polygons and drafts.
  Errors are displayed without clearing the polygons. Click *View Saved Record*
  to query the scene's latest confirmed ROI and its original reference.
- CLI check, independent of the browser:

```bash
conda activate gsrefine
cd /home/xth/GSSceneRefine
python run.py roi-status --config configs/songziguanyin.yaml
```

- **Deliberate facade miscut test:** Module 2 accepts a geometrically valid ROI
  even if it excludes visible target structure. This is a task for the next
  geometry-protection module. Every confirmed annotation gets the next numbered
  filename without replacing earlier files. To run the next module on the latest
  confirmed ROI:

```bash
python run.py roi-refine --config configs/songziguanyin.yaml
```

  Inspect `runs/songziguanyin/roi_core/report.json` and the three masks; compare
  `hard` with `global` and `local`. The AABB debug input cannot restore Gaussians
  previously deleted by AABB pruning, and a larger retained count alone does not
  prove that a specific miscut cliff structure was recovered.

### Isolated miscut experiment without overwriting baseline outputs

If you wish to reproduce an older deliberate test ROI rather than the latest,
run it with an explicit ROI path and separate results folder. This does not edit
the saved ROI manifest or original PLY. Example (replace `_001` with verified output name):

```bash
python run.py roi-refine \
  --config configs/songziguanyin.yaml \
  --roi /home/xth/GSSceneRefine/runs/roi_init/songziguanyin_absgs_roi_001.json \
  --output-dir runs/roi_tests/songziguanyin_misclip_001 \
  --modes hard,global,local
```

The synthetic end-to-end regression demonstrates restoring Gaussian centers
next to a short artificially clipped surface without following a *disconnected*
background. This **does not prove** successful real Songzi cliff recovery: inspect
`report.json`, the masks and independent rendered views, and note that connected
background is harder to disambiguate.


## Module 2 → module 3: latest confirmed ROI by default

Module 2 judges only mathematical correctness: valid polygons, actual orthonormal
projections, consistent original COLMAP coordinates, and nonempty Gaussian-center
intersection. Whether a polygon cuts a meaningful feature is **NOT** a Module 2
rejection rule. This is evaluated by the subsequent geometry protection module.

Each successful `Confirm and Save` writes an immutable, incrementally numbered
`{scene}_{method}_roi_{index:03d}.json` in the SAME shared `runs/roi_init/` directory,
regardless of earlier ROI files. The first saved file remains the scene's reference
`scene_canonical`; the new `scene_latest` in `manifest.json` advances on every confirmed
save. The latest file is selected by manifest *confirmation order*, not by filename
or operating-system file modification time. Different methods can share it only
when they use the same verified COLMAP frame and camera-pose source.

```bash
python run.py roi-status --config configs/songziguanyin.yaml
python run.py roi-refine --config configs/songziguanyin.yaml --show-roi
# Default geometry computation uses the latest confirmed scene ROI:
python run.py roi-refine --config configs/songziguanyin.yaml
# Reproduce an older experiment by pinning its exact input:
python run.py roi-refine --config configs/songziguanyin.yaml \
  --roi /home/xth/GSSceneRefine/runs/roi_init/songziguanyin_absgs_roi_001.json
```

The CLI prints the resolved full input path at startup. For reproducible publications,
retain the exact ROI JSON pathname and file SHA-256 in experiment reports. A newer
GUI confirmation changes only which ROI is selected *by default*, never modifies
previous ROI JSONs or previous algorithm results. Existing manifests with only
`scene_canonical` and `records` are migrated in memory on first read; new keys
are introduced on the next confirmed save, without deleting existing records.
