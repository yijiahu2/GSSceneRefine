# GSSceneRefine 阶段二正式无封底流程（V3，2026-11-12）

## 修复目标

修复在 **sengmaofeng / EfficientGS / roi_003** 触发的 `ValueError: Preseal main surface did not pass existing mesh/SfM safeguards` 所暴露的旧流程干扰：旧的 `open_ab_main._validated_inputs` 把单一简单开放边界环、完整绕向、人工封底资格 `preseal_eligible` 当成了正式ROI—Poisson Parity清理的必要条件。**现在正式阶段二没有人工封底，不应该继承这些封底特有的门槛。**

本补丁把历史封底相关条件从正式调用链中隔离，保留历史源码以复现旧消融，但 **`stage2_onepass` 与 `roi_barrier_parity` 不再导入历史 `open_ab_main` 的预审函数或距离配置**。这里 `preseal_diagnostics.json` 是已有主辅几何缓存的历史文件名，不代表要执行封底。

## 正式技术流程（不变）

1. 读取第一阶段最终恢复的连续R3 ROI、原生 Gaussian PLY、净化 SfM 和原始全场景 Poisson，检验来源。
2. 按真正的 R3 连续空间从原始 Poisson 直接裁剪，得到局部主曲面（用于距离）和全部辅助小分量（可选择用于保护）。
3. **无封底几何预审**：核验主辅PLY、来源SHA256、全量核心SfM P95、P95≤ROI对角线5%以及有效三角面。局部主曲面可以有多个开放边界环，不再要求适合裙边封底或水密。
4. 用 **原始全场景 Poisson 的最大连通主体** 与最终 R3 构建空间隔断；闭合交线、粗细网格连通分区、真实三角面跨区域奇偶和跨尺度一致性检查仍然执行，失败则拒绝。
5. 计算全部Gaussian到局部主曲面三角面的无符号距离；阈值保持 `delta = P95(SfM→主曲面) + P10–P90截尾均值(3×Gaussian最大主轴σ)`。
6. 仅对超阈值Gaussian做真实Poisson三角面短射线XOR侧别查询；未决、冲突一律不导出。
7. 对外部删除候选按所有辅助分量距离≤同一delta保护；或加`--no-aux-protection`进行无辅助保护消融。原生逐记录复制，最终输出 **`gaussian_roi_clean.ply`**（SuperSplat兼容）。

**重要：**并非放宽一切检查。完整SfM拟合真实性、P95上限、曲面来源、R3验证、墙体隔断以及Parity冲突检验均未取消。自动顶部外部种子仍是研究性假设，建议实景确认后使用 `--exterior-seed X Y Z`。

## 修改文件

- 新增 `gsrefine/cleanup/stage2_geometry_validation.py`：统一无封底主曲面验证，报告旧封底资格但**不把它作为门槛**。
- 更新 `gsrefine/cleanup/roi_barrier_parity.py`：只使用新预审和独立的距离参数，保留新版跨区域Parity空间隔断。
- 更新 `gsrefine/cleanup/stage2_onepass.py`：自动准备及复用正式主辅曲面，新增 `diagnostics/sealless_geometry_validation.json`；多个**等价**的几何记录自动去重，不同记录依然要求研究者明确选择。
- 测试：`tests/test_stage2_sealless_geometry.py`，以及适配后的 `tests/test_stage2_onepass.py`。

不修改 `run.py`、阶段一、现有原始Poisson、ROI PLY、老的 `direct_r3_pipeline.py` 与 `open_ab_main.py`。历史 `--phase seal/sdf/export` 只作为实验回溯，不属于正式一键执行链。

## 安装

```bash
cd /home/xth/GSSceneRefine
conda activate gsrefine
unzip -o /mnt/f/Edge/GSSceneRefine_stage2_sealless_unified_V3_20261112.zip -d /home/xth/GSSceneRefine
python -m pytest -q tests/test_stage2_sealless_geometry.py tests/test_stage2_onepass.py tests/test_barrier_separation_gate_v2.py
```

## 僧帽峰 EfficientGS / roi_003：一条命令重新运行

此前 `stage2_final_v1` 已留下失败日志，改用新的输出目录，避免覆盖：

```bash
ROI_RUN=/mnt/f/danxia_experiment_archive/GSSceneRefine_results/sengmaofeng/methods/EfficientGS/roi_003
python -m gsrefine.cleanup.stage2_onepass \
  --manifest "$ROI_RUN/metadata/manifest.json" \
  --accept-auto-exterior \
  --output-dir "$ROI_RUN/cleanup/stage2_final_v2"
```

注意：`roi_003` 是新画的ROI，不能套用 `roi_002` 的R3空间隔断或分类缓存；新流程会独立核验来源。没有把任何场景、Gaussians数量或辅助分量数量写死。

成功后重要成果：

```text
roi_003/cleanup/stage2_final_v2/
├── gaussian_roi_clean.ply                # 唯一正式完整ROI Gaussian模型
├── stage2_final_report.json              # 阈值/来源/删除和保护统计
└── diagnostics/
    ├── sealless_geometry_validation.json # 新增，预审结果
    ├── final_classification_masks.npz
    └── parity/
        ├── region_separation_gate.json   # 空间隔断结果
        └── roi_parity_report.json
```

需要无辅助保护消融时，加 `--no-aux-protection`，并使用另一个新的 `--output-dir`。

## 如果仍然报错

- **`SfM-to-main surface fit FAILED`**：真正的P95超过安全上限，报错中会给出P95和ROI尺度上限；需要检查SfM、ROI和可信Poisson，而不是调低安全要求。
- **`SHA256 mismatch` / `DIFFERENT Stage-1`**：数据来源或缓存不一致，必须修正输入。
- **`region separation did not pass`**：真实的原始Poisson—R3空间隔断仍没有得到足够数值证据，可查看 `region_separation_gate.json`；不会为了某个场景放行。
- **`odd_cycle_conflicts` / `unresolved`**：Parity或Gaussian短射线未通过，停止正式导出。
- **`Several DIFFERENT authenticated ...`**：同一个ROI存在不同有效几何版本，需指定审核过的 `--preseal-report`，防止意外复用。

本补丁在本地完成代码语法、模拟网格、安全检查和接口回归；完整僧帽峰真实数据需在你的WSL上运行。不能保证任何ROI必然通过几何/空间有效性检查，也不能在未核验实景前宣称模型完全清理干净。
