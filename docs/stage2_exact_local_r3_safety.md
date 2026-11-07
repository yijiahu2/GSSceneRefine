# GSSceneRefine：阶段一 R3 逐位置安全距离 ×2 提取净化 SfM（严格版）

> 2026-11-07。此包是针对 `GSSceneRefine_stage2_sfm_double_safety.zip` 的**增量补丁**。以用户的 `/home/xth/GSSceneRefine` 本地代码为权威，不访问 GitHub，不修改 Stage-1 R3 Gaussian 选择与导出，不修改 Stage-2 的 `delta = P95 + Gaussian 3σ 截尾均值` 判据和 `sdf <= delta+epsilon` 判别。

## 1. 技术定义（此次确认）

阶段一 R3 在空间位置 `x` 原本使用的局部安全距离是 `d_R3(x)`。记确认的双正交候选 ROI 在世界坐标 `x` 的有符号距离为 `s_ROI(x)`（内正外负，采用既有 `roi_signed` 函数）。

- `gaussian_roi.ply`：**完全不改**；仍是阶段一原有 R3 算法选中的 Gaussian。
- `sfm_roi.npz` 的核心点：净化 SfM 中 `s_ROI(x) >= -d_R3(x)`；对应已有空间 1× 安全包络。这里不是重跑 Gaussian 选择，也不声称核心 SfM 与带 Poisson 连续性保护的 Gaussian R3 掩码完全相同。
- `sfm_roi.npz` 的重建点：净化 SfM 中 `s_ROI(x) >= -2*d_R3(x)`；用于局部 Poisson 重建的点集，包含 1× 核心点及额外 1× 支撑点。
- 不引入全局平均距离、不重算 `P95` 来估算 Stage-1 R3 安全距离、不访问原始 COLMAP `points3D.bin`。
- 阶段二 Poisson 拟合 P95 只用 `in_core_roi==true` 的 SfM；Gaussian 3σ 模型截尾均值及最终 SDF 判别式不变。重建外扩区不自动判定为可信目标表面。

**不等价之处：** 上述 SfM 核心/支撑区复用 `d_R3(x)` 的**空间缓冲规则**；阶段一 Gaussian R3 还包含可信曲面连续性和局部主体保护等条件，所以不能把 SfM 核心掩码冒充成最终 Gaussian R3 掩码。真实的边界是否一致仍由 Stage-1 源代码验证。

## 2. 代码文件

- `gsrefine/geometry/roi_sfm_local.py`：新增纯粹的逐位置距离保存、已净化 SfM 提取与 SHA256 验证；不会调用原始 SfM，也不会新建估计 d 的算法。
- `gsrefine/cleanup/direct_sdf.py`：优先使用 Stage-1 精确 `sfm_roi.npz`，缺失时仅允许凭**已经计算的**精确逐位置 `d_R3(x)` 文件补生成。彻底禁止原有标量退路。原 SDF 公式与 Poisson 封底算法原样保留。预览报告改为按输入签名命名，避免与之前的旧版结果覆盖冲突。
- `tools/apply_exact_sfm_handoff.py`：守护修改 `gsrefine/roi_init/output_layout.py` 中**此前版本已安装的** SfM 发布钩子；不会修改 R3 算法。
- `tools/install_exact_sfm_patch.py`：先检验旧版文件 SHA256 与 Stage-1 publisher 结构，再备份和原位安装；不对未知本地代码覆盖。
- `tests/test_stage2_exact_local_safety.py`：逐点 1×/2×、SfM 元数据保留、来源哈希、阶段一发布、阶段二复用、旧数据补生成、拒绝标量回退等回归测试。

## 3. 阶段一接入的唯一必要步骤（需要实际 R3 计算函数）

**重要限制：** 这个对话提供的历史补丁中**没有**你 WSL 当前最新版 `gsrefine/roi/study.py` 或等效的 R3 安全距离计算实现，因此无法可靠自动把一个未知函数调用插入到实际 R3 计算点。不能凭经验虚构变量名。下面是已实现的稳定接口；需要把其中的 `existing_r3_local_safety_function` 对接到阶段一真实计算过 `d_R3(x)` 的函数，不能替换为平均值或重新拟合。

在阶段一安全距离已经计算完成、工作目录尚未清理时，增加：

```python
from gsrefine.geometry.roi_sfm_local import save_stage1_r3_local_safety

save_stage1_r3_local_safety(
    work_dir=work,                     # 阶段一真实临时工作目录
    clean_sfm_npz=clean_sfm_npz,      # 阶段一净化后的 SfM NPZ
    roi_json=roi_json,                # 当前 ROI 的确认版 JSON
    safety_at=existing_r3_local_safety_function,  # 实际 R3 d(x)，输入 Nx3，输出 N 个 d
)
```

如果该轮 R3 本来就已经对**同一份净化 SfM 的所有点**计算过距离数组，无需重复求值，直接改为：

```python
save_stage1_r3_local_safety(
    work_dir=work, clean_sfm_npz=clean_sfm_npz,
    roi_json=roi_json,
    per_point_safety=already_computed_r3_distances_in_clean_sfm_row_order,
)
```

必须是 N 个**逐位置**距离，不能传标量。保存时同时记录净化 SfM 文件和 ROI 文件 SHA256；原 SfM 点坐标和属性一字节不改。发布器将 `stage2_ready_r3_local_safety.npz` 留存为 `metadata/stage1_r3_local_safety.npz`，并从这份精确数据得到 `final/sfm_roi.npz`、`metadata/roi_sfm_report.json`，在 manifest 的 `stage2_inputs` 增加 `sfm_roi_npz`、`r3_local_safety_npz`。若阶段一没产生这个文件，发布器自动跳过 SfM 输出，让 Stage-1 Gaussian 导出继续正常完成，绝不触发旧全局近似。第二阶段则明确提示缺少精确 d(x)，而不是错误地使用替代值。

## 4. 当前旧 ROI 的兼容规则

对 `/mnt/f/danxia_experiment_archive/GSSceneRefine_results/songziguanyin/methods/absgs/roi_001`：

- 如果 Stage-1 结果中已有符合 v2 且哈希正确的 `sfm_roi.npz`，直接使用。
- 如果只有本地保存的 Stage-1 精确 `metadata/stage1_r3_local_safety.npz`，Stage-2 可以独立从现有**净化** SfM 中补算，生成 `cleanup/inputs/sfm_roi_local_v2.npz`。这是你要求的“阶段一没有提取 SfM，则阶段二自己补生成”。
- 若只有之前旧版的**标量近似** `sfm_roi.npz`，则拒绝复用，避免将其伪装成 2× 局部安全。
- 如果 Stage-1 没留下逐位置 `d_R3(x)` 也没有可运行的 R3 位置安全距离函数，那么**无法仅凭最终 Gaussian PLY 恢复每个位置原来的 d(x)**。此时不能谎称 Stage-2 可以精确生成；必须从 Stage-1 现有 R3 代码**仅补计算该局部安全场**（不重跑 Gaussian 训练或完整阶段一流程），再由 Stage-2 生成 SfM。你可以把本地 R3 对应源码提供给我，以完成这个最后的接口适配。

## 5. 安全安装（不要直接 `unzip -o` 覆盖未知源码）

先将 ZIP 下载到 Windows Edge 下载目录，WSL 中执行：

```bash
cd /home/xth/GSSceneRefine
conda activate gsrefine

# 原有 Stage-2 文件必须是上一版：
sha256sum gsrefine/cleanup/direct_sdf.py
# 预期：c4c6949527151860b0dd226dd35ad49bcdfe32c97ca9f6219f6fdddf401032d8

# 在临时目录展开而不是直接覆盖仓库
mkdir -p /tmp/gsrefine_exact_sfm_patch
unzip -o /mnt/f/Edge/GSSceneRefine_stage2_exact_local_safety.zip \
  -d /tmp/gsrefine_exact_sfm_patch

# 先做结构与版本兼容检查
python /tmp/gsrefine_exact_sfm_patch/tools/install_exact_sfm_patch.py \
  --project-root /home/xth/GSSceneRefine \
  --payload-root /tmp/gsrefine_exact_sfm_patch \
  --dry-run

# 确认结果没有错误后执行原位更新
python /tmp/gsrefine_exact_sfm_patch/tools/install_exact_sfm_patch.py \
  --project-root /home/xth/GSSceneRefine \
  --payload-root /tmp/gsrefine_exact_sfm_patch

python -m pytest -q \
  tests/test_stage2_exact_local_safety.py \
  tests/test_stage2_cleanup.py \
  tests/test_stage2_roi_sfm_double_safety.py
```

已知不兼容的 Stage-1 publisher 或用户修改过的 Stage-2 源码将直接拒绝更新，并将报告错误；不要强行 `unzip -o` 覆盖。

## 6. 运行最新送子观音 ROI

```bash
ROI_RUN=/mnt/f/danxia_experiment_archive/GSSceneRefine_results/songziguanyin/methods/absgs/roi_001

python run.py stage2-cleanup \
  --manifest "$ROI_RUN/metadata/manifest.json" \
  --surface-only
```

如果当前 `roi_001` 的 Stage-1 还没有保存真正逐位置 `d(x)`，这条命令会明确报 `Missing exact Stage-1 per-location R3 safety field`。这不是重建/封底错误，而是必要的接口数据缺失。

一旦真实 Stage-1 R3 安全场已由相同算法补写到某个文件，可指定：

```bash
python run.py stage2-cleanup \
  --manifest "$ROI_RUN/metadata/manifest.json" \
  --stage1-local-safety-field /绝对路径/stage1_r3_local_safety.npz \
  --surface-only
```

**不允许**以 `--safety-distance 0.08` 替代：此参数已停用。

预览日志将打印 `cleanup/preview/local_poisson_preview_<signature>.json`，实际曲面保存在 `cleanup/geometry/<signature>/poisson_roi.ply`；此时不封底、不清理 Gaussian。真实送子观音数据仍需视检并通过拓扑与相机核验。

## 7. 测试事实

在提供的历史补丁源代码、合成净化 SfM 和本地接口占位的测试环境中，原有回归与新增测试共 34 项通过；本环境**无法访问你 WSL 中的最新 R3 源码、COLMAP/真实 SfM/大规模 Gaussian**。因此不声称已在真实送子观音完成局部 d(x) 接入或 3D Poisson 验收，也不声称现有 Stage-1 核心代码已自动被修改。
