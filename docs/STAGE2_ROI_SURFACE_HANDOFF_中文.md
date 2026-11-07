# GSSceneRefine 第二阶段：最终 ROI 可信曲面裁剪与阶段间数据复用

> 2026-11-07 · 本地原位更新补丁 · 仅针对现有单仓库 `/home/xth/GSSceneRefine`、Conda 环境 `gsrefine`。
>
> 本补丁按照用户本地已安装的 `GSSceneRefine_stage2_topology_fix.zip` 代码接口制作，**没有从 GitHub 更新代码**。首次使用前校验本地脚本版本，避免覆盖更新过的本地源码。

## 核心修订

旧第二阶段：阶段一完整 `poisson_mesh` → 直接尝试封闭 → 对阶段一 `gaussian_roi.ply` 做 SDF。

新第二阶段：阶段一原始可信 `poisson_mesh` **＋已确认 ROI** → 优先复用阶段一的可核验最终 R3 可信 Poisson 面片记录 → 如不存在，则第二阶段单独提取 ROI 可信曲面并保守推断边界保护 → 在**裁剪后的观测曲面**上估算 ROI SfM 拟合 P95 → 局部拓扑诊断、Poisson 封闭、相机校验 → 模型专属直接 SDF → 全属性 Gaussian 输出。

**不重新构建 SfM–Poisson、不重新训练 3DGS、不重复执行阶段一 R0/R3 Gaussian 筛选。** 原始可信 Poisson、ROI、阶段一模型不修改。

## 三层复用优先级

| 优先级 | 数据来源 | 实际行为 |
|---|---|---|
| 1 | 阶段一 manifest 中已记录 `gaussian_ply`、`poisson_mesh`、`clean_sfm_npz`、`roi_json` | 全部直接读取并按 SHA256 校验，不重新计算 |
| 2 | 阶段一如果额外保存了已经计算好的、与原 Poisson **面片行序**对应的最终 R3 可信曲面掩码 | 直接复用该掩码；不执行阶段二的保护范围推定 |
| 3 | 当前或历史阶段一结果没有上述面片掩码 | 第二阶段只补算 ROI 局部可信曲面，不重新进行全局 SfM 净化、Poisson 重建或 Gaussian ROI 恢复；明确标记 `inferred_r3_gaussian_support` |

**重要边界：** 阶段一 Gaussian R3 掩码不是 Poisson 面片掩码。阶段一的 `gaussian_roi.ply` 不足以精确反推 R3 面片集合。当前未获得你的本地最新版 `gsrefine/roi/study.py` 完整源码，因此补丁不擅自假定阶段一已导出了原始 Poisson 面片的最终 R3 掩码，也不修改阶段一数学流程。

## 新增接口与现有代码归并

- `gsrefine/geometry/roi_surface.py`：从已确认的双正交多边形提取原始可信 Poisson 的 ROI 局部曲面；相交三角形通过共享边的分段线性零交点裁切。对于 R3 外侧保护，只有在同一 Poisson 连续曲面、有限测地距离、净化 SfM 局部支持和 Gaussian 空间支持条件同时成立时才保守纳入。此路径属于**推定保护范围**，不等同于阶段一真实 R3 面片判定。较粗网格的凸凹交叉无法由当前线性近似安全判别时将停止。
- `gsrefine/geometry/stage1_handoff.py`：定义已经计算好的最终 R3 可信 Poisson 原始面片掩码的可验证读写接口。面片掩码需有 SHA256 与明确的 `final_r3_trusted_roi_surface_faces` 语义。
- `tools/apply_stage2_handoff.py`：在现有 **本地** `gsrefine/roi_init/output_layout.py` 中插入仅负责保存阶段一已有面片记录的轻量钩子，同时在 `configs/default.yaml` 增加通用参数。插入前必须识别已知结构，不匹配就拒绝修改；将原文件备份到 `/tmp/gsrefine_stage2_source_backups`，绝不新建仓库或覆盖 stage1 算法。
- `gsrefine/cleanup/direct_sdf.py`：改成先 ROI 曲面后封闭；优先读取 manifest 中 `stage2_inputs.roi_surface_faces`，没有则补算。缓存签名包含 ROI、模型、Poisson、SfM、COLMAP 相机、配置与曲面来源身份。
- `gsrefine/geometry/sdf_sealing.py`：**不在此次补丁中覆盖**；继续复用本地拓扑修复后的封闭算法。多边界环、背景主体混入、非流形、相机外部检查不通过时拒绝自动导出，不擅自删除真实面片。

### 阶段一应如何生成精确交接记录？

若阶段一研究代码确实在计算过程中已经得到**原始可信 Poisson 面片行序上最终 R3 主体范围的 bool mask**，可在该掩码产生的同一工作区调用：

```python
from gsrefine.geometry.stage1_handoff import save_computed_stage1_surface_faces
save_computed_stage1_surface_faces(
    work, final_r3_trusted_poisson_face_mask,
    paths['poisson_raw'], roi,
    face_count=len(original_poisson.faces))
```

然后已有阶段一 `publish()` 保存钩子会把 `stage2_ready_roi_surface_faces.npz` 保留为 `metadata/trusted_roi_surface_faces.npz`，并将其加入 `metadata/manifest.json` 的 `stage2_inputs.roi_surface_faces`。这一步只写入阶段一**已计算**的数据，不重新计算 R3。**若当前 `roi.study` 尚未提供这个精确面片数组，不应强行构造并打上“已认证”标签；第二阶段走第3级补算。**

## 本地原位更新

补丁不含现有 `run.py`、`sdf_sealing.py`、`environment.yml` 和任何原始/实验数据。建议先核对 `direct_sdf.py` 是之前拓扑热修复版。旧热修复版 SHA256：

`00c2929070fe3e7bbd76267aadd790c124ee0c30036836264944989018f85731`

```bash
cd /home/xth/GSSceneRefine
conda activate gsrefine
sha256sum gsrefine/cleanup/direct_sdf.py
# 若与上述 SHA 不同，先审阅本地修改，勿直接覆盖。

unzip -o /mnt/f/Edge/GSSceneRefine_stage2_roi_handoff.zip -d /home/xth/GSSceneRefine

# 只改本地已有 Stage-1 输出保存钩子及通用配置，不触碰其算法
python tools/apply_stage2_handoff.py --project-root /home/xth/GSSceneRefine

python -m pytest -q tests/test_stage2_cleanup.py tests/test_stage2_stage1_writer.py
python run.py stage2-cleanup --help
```

首次在真实当前阶段一目录中先**只提取曲面**：

```bash
ROI_RUN=/mnt/f/danxia_experiment_archive/GSSceneRefine_results/songziguanyin/methods/absgs/roi_001

python run.py stage2-cleanup \
    --manifest "$ROI_RUN/metadata/manifest.json" \
    --surface-only
```

这一步只在 `<ROI_RUN>/cleanup/preview/` 输出 `trusted_roi_poisson.ply` 和 `roi_surface_report.json`，**不封底、不剪枝、不覆盖阶段一模型**。请检查 ROI 曲面是否保留真实外伸岩体、是否混入远处背景、是否存在无法闭合的多个开口。

通过可视化后进行封底和 SDF 诊断：

```bash
python run.py stage2-cleanup \
    --manifest "$ROI_RUN/metadata/manifest.json" \
    --no-export
```

封闭检查通过后再执行正式导出。若来源是**推定的** R3 曲面，必须额外明确授权（这是防止不可靠自动清理的安全闸门）：

```bash
python run.py stage2-cleanup \
    --manifest "$ROI_RUN/metadata/manifest.json" \
    --allow-inferred-guard-export
```

如果使用经过来源校验的阶段一精确面片记录，则不需要此参数。无法通过封闭拓扑和相机检查时应停止诊断，不要放宽条件强制导出。

## 输出规范

```text
<output_dir>/<scene>/
├── geometry/                            # 阶段一场景共享几何，不修改
├── roi/                                 # 阶段一ROI，不修改
└── methods/<method>/<roi_id>/
    ├── final/gaussian_roi.ply           # 阶段一最终ROI，不修改
    ├── metadata/manifest.json           # 阶段一来源，不修改（新任务可包含曲面面片记录）
    └── cleanup/
        ├── preview/trusted_roi_poisson.ply  # 明确请求 --surface-only 时生成
        ├── cache/<signature>/
        │   ├── trusted_roi_poisson.ply  # 按 ROI 裁剪后的观测曲面
        │   ├── sealed_poisson.ply       # 封闭 Poisson 约束体
        │   └── sealing_report.json      # 共享/局部/验证证据
        ├── final/gaussian_roi.ply        # 第二阶段正式成果
        ├── metadata/cleanup_summary.json
        ├── metadata/manifest.json
        └── diagnostics/                 # 仅 --keep-diagnostics 输出大数组
```

正式文件名都使用 `snake_case`，实验数据全部留在用户指定的 `--output-dir` 内。失败时不删除用户任何历史数据。

## 验收条件

1. 确认阶段一输入的 4 种数据 SHA256 均一致；不存在重新重建 SfM/Poisson 行为。
2. `roi_crop.selection_mode` 区分 `certified_stage1_surface_faces` 与 `inferred_r3_gaussian_support`。
3. 当前完整 Poisson 被裁剪为 ROI 可信曲面后再进行 `prepare_open_poisson` 和 `seal_main`。
4. SfM 点到真实裁剪后 Poisson 的 P95 统计不使用人工裙边/底面。
5. 封闭体水密、有效体积、相机全处于外部；否则停止。
6. 沿用模型专属 `delta = ROI_SfM_P95 + Gaussian_3σ_major_axis_trimmed_mean(P10–P90)` 和 `d_s <= delta + ε`。
7. 输出 Gaussian 逐行属性字节保留；不跨模型复用掩码；阶段一正式 PLY 不改动。
8. 本补丁仅做轻量单元测试，**未运行送子观音真实场景**；在本地实测和视觉核验前不能宣称最终清理效果已通过验证。
