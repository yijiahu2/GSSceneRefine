# GSSceneRefine 阶段二性能更新

适配提交 `20806e903704d5aab40d9d31e8fed12ec8fbf640` 和其上的自适应体素 v1 更新。压缩包中的路径相对于项目根目录。

```bash
conda activate gsrefine
cd /home/xth/GSSceneRefine
unzip -o /mnt/f/Edge/GSSceneRefine_stage2_perf_v2.zip -d /home/xth/GSSceneRefine
ROI_RUN=/mnt/f/danxia_experiment_archive/GSSceneRefine_benchmark/GSSceneRefine_results/lfls/methods/gsplat_accel/roi_002
python run.py full-pipeline \
  --resume-manifest "$ROI_RUN/metadata/manifest.json" \
  --stage2-name stage2_perf_v2 --classifier adaptive-voxel \
  --accept-auto-exterior --reclassify --keep-classification-previews
```

## 结果核对

```bash
OUT="$ROI_RUN/cleanup/stage2_perf_v2"
cat "$OUT/diagnostics/stage2_timing.csv"
cmp "$ROI_RUN/cleanup/stage2_voxel_v1/gaussian_roi_clean.ply" "$OUT/gaussian_roi_clean.ply" && echo '新旧 PLY 逐字节一致'
```

阶段二的秒表从 onepass 入口计到最终报告组装，包含输出 PLY。CSV 的独占秒数可相加，含子步骤秒数用于单独查看某一模块。每层体素、代表点和逐点回退的分类器计时仍在 diagnostics/parity/roi_parity_report.json。复用已有几何时，跳过的准备步骤不会产生计时。

主曲面距离使用三角网格距离的 1-Lipschitz 性质。T 为原判据的 delta 加 epsilon。以 d 表示体素中心到曲面的距离，r 为体素半对角线加数值余量。若 d+r ≤ T，整格均在阈值内。若 d−r > T，整格均超过阈值。其余行继续细分或逐点计算。辅助曲面以按 delta 扩张的分量包围盒排除远点，再对剩余候选计算三角面距离。

使用上下界判定的行在 distance_main 或 candidate_aux_distances 中写 NaN。分析全体点的精确距离分布时，在上述命令前设置 GSREFINE_EXACT_DISTANCES=1，保留 --reclassify 并换一个新的 --stage2-name。分类和保护使用相应的布尔掩膜。
