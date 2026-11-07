# GSSceneRefine Stage 2 累计升级补丁（修复安装器版本链）

## 错误原因（已从此前交付 ZIP 逐字节确认）

用户当前 `direct_sdf.py` SHA256 为 `db169555800191c9f976a1264845c38b783a32b1b9d62e6db1fa2aabd2fa3856`，准确对应此前提供的 `GSSceneRefine_stage2_roi_clip_fix.zip`。随后给出的 `GSSceneRefine_stage2_exact_local_safety.zip` 安装器只接受中间版本 `GSSceneRefine_stage2_sfm_double_safety.zip` 中的 SHA256 `c4c69495...`，因此报“diverged”。**该报错由我交付的更新包版本链不完整导致，不是用户改动源码。**

此外，最新 `direct_sdf.py` 引用 `gsrefine.geometry.roi_poisson`，该依赖仅随中间升级包发布，因此仅添加一个哈希并不够。

## 本补丁内容

- 集成此前 double-safety 包内 `gsrefine/geometry/roi_poisson.py`。
- 集成 exact-local 包内 `gsrefine/geometry/roi_sfm_local.py` 与 `gsrefine/cleanup/direct_sdf.py`。
- 新安装器直接认可 `db169555...`、`c4c69495...`、`9eb3e986...` 三个**已交付版本**，不再强制逐包升级。
- 阶段一 `gsrefine/roi_init/output_layout.py` 如未安装 SfM 发布钩子则插入精确版；若已有旧版近似钩子则替换；已是精确版则不重复修改。原有 R3 核心计算/导出程序不动。
- `--dry-run` 先检测完整依赖和发布器接口。实际安装保留文件备份，失败回滚已写入项。
- 不下载仓库，不触碰用户场景数据，不修改现有 `gaussian_roi.ply`、原始 Poisson、净化 SfM 和 SDF 判据。

## 安装

ZIP 放到 Windows Edge 下载目录后（WSL `/mnt/f/Edge`），执行：

```bash
cd /home/xth/GSSceneRefine
conda activate gsrefine

mkdir -p /tmp/gsrefine_cumulative_patch
unzip -o /mnt/f/Edge/GSSceneRefine_stage2_cumulative_install_fix.zip -d /tmp/gsrefine_cumulative_patch

python /tmp/gsrefine_cumulative_patch/tools/install_exact_sfm_patch.py \
  --project-root /home/xth/GSSceneRefine \
  --dry-run

python /tmp/gsrefine_cumulative_patch/tools/install_exact_sfm_patch.py \
  --project-root /home/xth/GSSceneRefine

python -m pytest -q \
  tests/test_stage2_cumulative_installer.py \
  tests/test_stage2_exact_local_safety.py \
  tests/test_stage2_cleanup.py
```

预期 dry-run `detected_previous_version: roi_clip_fix`；若本地实际目录中其他已交付的阶段一发布器布局不同，将报告具体不匹配项，不执行任何更改。

## 仍存在的 R3 数据接线条件

本补丁**修复从当前已经提供的版本直接安装最新版**的问题，而不虚称已经对接未上传的阶段一 `d_R3(x)` 内部计算代码。最新阶段一交接接口只在已有准确的逐位置 `d_R3(x)` 数据时保存 `sfm_roi.npz`；它不回退到全局平均值。旧的 `roi_001` 如果没有保存逐位置安全距离，需要基于用户本地 Stage-1 R3 计算源码补写对应的保存钩子。缺少该安全场时 `--surface-only` 仍会明确报出前置数据缺失，不能用更换安装器解决。

## 不变量

- Gaussian 使用既有 R3 最终结果，1× 安全距离与 R3 原有连续性规则不变。
- SfM 只来自 Stage-1 **cleaned SfM**，针对同一位置逐点使用 `2*d_R3(x)`；核心区按 `d_R3(x)` 标记。
- Poisson 由局部 SfM 重建（不再对完整网格反复隐式三角裁剪）。
- SDF 的 `P95 + Gaussian 3σ 截尾均值` 阈值公式不变。
