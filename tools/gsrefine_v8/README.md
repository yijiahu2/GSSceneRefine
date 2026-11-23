# GSSceneRefine v8 统一工程更新：直接解压到项目根目录

**基础版本：** Git commit `1ebc9c06013b29f5fa71cfdeb1b0754cbd670123`。  
**适用环境：** `/home/xth/GSSceneRefine` / conda `gsrefine`。  
**正式算法：** 阶段一R3不变；阶段二全局原始Poisson最大连通主体判侧 + R3局部最大连通主体测距；无全局测距A/B。

## 1. 安装：ZIP内部无多余父目录

```bash
cd /home/xth/GSSceneRefine
conda activate gsrefine
unzip -o /mnt/f/Edge/GSSceneRefine_unified_v8_official.zip -d /home/xth/GSSceneRefine
python tools/gsrefine_v8/install.py --check
python tools/gsrefine_v8/install.py --apply
python tools/gsrefine_v8/install.py --verify
python -m compileall -q gsrefine
python run.py pipeline --help
```

**为什么还需要执行一次 `install.py --apply`？** 本包不能安全地把现有源码文件在未知的本地工作树上盲目覆盖；直接解压已经把正式入口和更新工具放到项目正确相对路径，安装器进一步检查Git基线和源码锚点，再**原地一次性更新**已有源代码，避免覆盖不匹配的版本。它会在项目同级目录自动备份修改前的文件。`--check`不修改源码。无需切换仓库、无需手工打补丁。

如本机HEAD不是指定基线、或存在除本包以外的未提交改动，安装器将拒绝操作，切勿用 `git reset --hard` 强行恢复。

## 2. 正式统一入口（三个模式）

以后均使用 `python run.py pipeline ...`。程序默认 `original-region`，需要显式指定外侧种子选择（可选 `--accept-auto-exterior`，属于暂定假设；或 `--exterior-seed X Y Z`）。

### A. 从头完整流程（模块1 → Edge手绘模块2 → 阶段一R3 → 阶段二清理）

```bash
python run.py pipeline full \
  --config configs/default.yaml \
  --scene lfls --method gsplat_accel \
  --sfm-dir "$SFM_DIR" --gaussian-ply "$FULL_PLY" \
  --output-dir "$OUTPUT_ROOT" \
  --accept-auto-exterior
```

### B. 从已有阶段一完整成果继续（**不重新执行阶段一**）

```bash
python run.py pipeline stage2 \
  --manifest "$ROI_RUN/metadata/manifest.json" \
  --accept-auto-exterior
```

`ROI_RUN` 必须指向 `.../lfls/methods/gsplat_accel/roi_001` 这类**已经完成**的结果目录。仅追加新的阶段二尝试，保留原阶段一结果。

### C. 已完成Edge手绘，只有正式ROI JSON（跳过Edge，执行模块3+4与阶段二）

```bash
python run.py pipeline from-roi \
  --config configs/default.yaml \
  --scene lfls --method gsplat_accel \
  --sfm-dir "$SFM_DIR" --gaussian-ply "$FULL_PLY" \
  --output-dir "$OUTPUT_ROOT" \
  --roi "$ROI_JSON" \
  --accept-auto-exterior
```

`ROI_JSON` 必须是已确认、不可变、与当前COLMAP坐标系和Gaussian源模型匹配的手绘ROI文件，例如 `.../lfls/roi/roi_001.json`。不需要重新手绘；若之前已经产出阶段一manifest，应选择B而不是C。跨方法复用同一ROI时显式增加 `--reuse-scene-roi` 并保证同一COLMAP原始坐标系。

### 可选：复现LFLS无辅助保护设置

在上述任意命令末尾加 `--no-aux-protection`。默认**启用**辅助保护。若在禁用辅助保护时还希望统计其反事实，可同时增加 `--audit-aux-ablation`（计算时间可能增加，且不改变正式删除掩码）。

## 3. 完整更新的范围

正式代码更新执行后，集成：Poisson/SfM原始曲面几何支持诊断（只读）、预览距离缓存跨次复用、阶段一永久科学掩码及哈希、SfM 1×/2×投影失败交叉表、严格V3 handoff、正式`sealless`几何入口、原始Poisson审计工具迁入库、默认`original-region`、阶段二Gaussian归属/外侧种子风险/辅助保护消融审计、计时和顶层质量警告。历史封底、测试及SfM回填功能保留作显式历史工具，不通过本版正式调用路径自动执行。

## 4. 检查与回归

```bash
python tools/gsrefine_v8/test_patch_components.py
python tools/gsrefine_v8/test_runner.py
python tools/gsrefine_v8/install.py --verify
python -m compileall -q gsrefine
```

上述检查只能验证安装与局部逻辑。**还需在用户实际LFLS数据上完成数值等价、跨场景和冷/热缓存的耗时对照；本包不宣称已得到实验加速或Gaussian删除正确性的证据。**

### 注意

- 所有阶段仍共用原来的Conda环境 `gsrefine`，无须建立新环境。
- ZIP可直接解压至项目根目录；源码更新器仍必需运行一次，因为GitHub连接只有读取权限，无法安全地为未经访问的本地工作树生成完全可信的覆盖文件。
- 已发布的阶段一、二成果不应被覆盖。阶段二尝试单独生成目录。
