# 阶段二局部主体分区清理

适用于 GSSceneRefine 提交 20806e903704d5aab40d9d31e8fed12ec8fbf640 及该提交上的阶段二 perf_v3 更新。此压缩包包含从该提交运行新分类器所需的累计改动。

## 从现有 ROI 重跑

```bash
conda activate gsrefine
cd /home/xth/GSSceneRefine
unzip -o /mnt/f/Edge/GSSceneRefine_stage2_local_region_v4.zip -d /home/xth/GSSceneRefine
ROI_RUN=/mnt/f/danxia_experiment_archive/GSSceneRefine_benchmark/GSSceneRefine_results/lfls/methods/gsplat_accel/roi_002
python run.py full-pipeline \
  --resume-manifest "$ROI_RUN/metadata/manifest.json" \
  --stage2-name stage2_local_region_v4 \
  --classifier local-region \
  --accept-auto-exterior \
  --reclassify \
  --keep-classification-previews
```

本命令使用该 ROI 的现有阶段一成果。自动外侧假设为沿项目向上方向的最高 R3 壳层节点位于外侧。如果现场几何与这个假设不符，请用经核实的外侧种子，参见命令帮助中的 exterior seed 参数。

## 判定与输出

最大局部主体同时用于分区核验和主体距离判定。程序在最终 R3 内审计两级连通关系以及原始三角面穿越。如果出现绕行，程序细化离散核验，检查局部主体边缘和原始面证据。只有原始主体提供支持且重裁后重新通过检查，才采用重裁几何并重算全部核心 SfM 点的 P95。若仍存在开口，尝试排除有绕行证据的邻域，并在剩余区域独立核验。有限分辨率数值核验可遗漏非常细小的开口。

核验后，粗细体素共享面的连通关系用于传播标签。内侧安全归类的体素整格保留。已确定为外侧且整格超阈值的体素直接进入候选。可能跨安全带的外侧 Gaussian 批量测距；超过阈值且归属未知的 Gaussian 才回退判侧。删除前执行辅助分量保护。达到细分预算时采用保守查询。无法核验的 Gaussian 保留在原位。

在 `cleanup/stage2_local_region_v4/` 查看 `stage2_final_report.json`。完整核验输出 `gaussian_roi_clean.ply`，局部核验输出 `gaussian_roi_partial.ply`，完全没有可核验范围时报告指向原阶段一 ROI。三种状态依次为 `COMPLETED_EXPERIMENTAL_CLEAN_ROI_GAUSSIAN`、`PARTIAL_VERIFIED_ROI_CLEANUP`、`NO_VERIFIED_CLEANUP_SCOPE`。

`diagnostics/scope.json` 记录采用的主体、核验尝试与范围；`diagnostics/final_classification_masks.npz` 的掩码按阶段一 ROI PLY 原始行序保存；`diagnostics/stage2_timing.csv` 同时记录含子步骤时间与独占时间。全流程比较需要同一阶段一 manifest、相同诊断输出及可比缓存状态。初次试用建议使用独立的阶段二目录以保留旧版成果。
