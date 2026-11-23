# 阶段二自适应体素分类更新

适用于 GSSceneRefine 提交 `20806e903704d5aab40d9d31e8fed12ec8fbf640`。更新位于正式入口 `run.py full-pipeline` 下的超阈值 Gaussian 中心分类步骤。使用已有 `gsrefine` 环境。

## 原位更新并从 LFLS 阶段一结果启动

```bash
conda activate gsrefine
cd /home/xth/GSSceneRefine
unzip -o /mnt/f/Edge/GSSceneRefine_stage2_adaptive_voxel.zip -d /home/xth/GSSceneRefine

ROI_RUN=/mnt/f/danxia_experiment_archive/GSSceneRefine_benchmark/GSSceneRefine_results/lfls/methods/gsplat_accel/roi_001
python run.py full-pipeline \
  --resume-manifest "$ROI_RUN/metadata/manifest.json" \
  --stage2-name stage2_voxel_v1 \
  --classifier adaptive-voxel \
  --accept-auto-exterior \
  --reclassify \
  --keep-classification-previews
```

`--resume-manifest` 读取已完成的阶段一。原有机制可复用来源校验通过的几何成果。`--reclassify` 重新计算本轮分类，适用于测速。每次运行创建独立输出目录，首选目录已存在时自动递增版本。完整路径见终端最后一行和 `$ROI_RUN/metadata/full_pipeline_status.json`。对新场景按原有 full-pipeline 命令从阶段一启动即可，默认使用 adaptive-voxel。

## 分类与结果核对

- `adaptive-voxel` 用原始全局 Poisson 最大主体三角面作空间分界，复用已经确定内外方向的导航节点。只在能确认属于 R3 且不会穿越分界面的范围内共享标签。边界只细分含有待分类点的格子，难以判定的候选使用原短射线路径。达到局部深度或密度限制时同样回退。
- `base-voxel` 在导航网格的基础尺度查表，剩余候选走短射线。`short-ray` 对每个超阈值候选执行原算法。这两个选项可通过同一条命令进行消融。各次使用相同的 manifest、外侧方向和诊断开关。
- 日志 `Adaptive voxels depth=...` 给出每层处理数量。`Adaptive voxel lookup ...; short-ray fallback ...` 给出查表和逐点回退数量。格子代表点可能额外需要少量短射线，计入 `representative_queries`。总耗时包含分界面查询索引构建、体素分类、代表点和回退计算。
- `stage2_final_report.json` 的 `classification_performance` 保存以上统计和 `total_seconds`。若复用了已完成分类，统计属于产生该分类的原运行。`diagnostics/parity/classification_masks.npz` 保存逐行侧别，`diagnostics/final_classification_masks.npz` 保存辅助保护后的掩膜，主结果为 `gaussian_roi_clean.ply`。
- 体素只改变侧别的求解方式。P95、Gaussian 尺度、安全阈值、粗细网格审计、外侧种子、辅助曲面保护以及未决候选阻止正式导出的代码继续使用原流程。
