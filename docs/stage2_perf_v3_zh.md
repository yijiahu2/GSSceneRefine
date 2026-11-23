# 阶段二几何校验复用与短射线计时

基于公开提交 `20806e903704d5aab40d9d31e8fed12ec8fbf640` 和阶段二 v2 更新。压缩包路径相对于项目根目录。使用现有 gsrefine 环境。

```bash
conda activate gsrefine
cd /home/xth/GSSceneRefine
unzip -o /mnt/f/Edge/GSSceneRefine_stage2_perf_v3.zip -d /home/xth/GSSceneRefine
ROI_RUN=/mnt/f/danxia_experiment_archive/GSSceneRefine_benchmark/GSSceneRefine_results/lfls/methods/gsplat_accel/roi_002
python run.py full-pipeline --resume-manifest "$ROI_RUN/metadata/manifest.json" \
  --stage2-name stage2_perf_v3 --classifier adaptive-voxel \
  --accept-auto-exterior --reclassify --keep-classification-previews
python tools/compare_stage2_times.py "$ROI_RUN/cleanup/stage2_perf_v2" "$ROI_RUN/cleanup/stage2_perf_v3"
```

## 如何读计时

`diagnostics/stage2_timing.csv` 覆盖 onepass 启动到最终报告组装。每行列出调用次数、含子步骤的耗时和独占耗时。父、子步骤有重叠，汇总时使用独占秒数。旧版和新版比较时用同一个 ROI、同一份几何和审计缓存、相同的诊断输出选项。比较命令将输出总耗时、分类耗时和二进制 PLY 核对结果。

| 字段 | 计时范围 |
|---|---|
| read_file_sha256 | 实际打开文件并计算 SHA-256，在不同调用位置累计 |
| validate_local_geometry | 对当前来源版本首次完成局部主体三角面及辅助分量校验 |
| r3_source_read_and_authentication | 原始数据读取、R3 交接数据与输入来源校验 |
| r3_core_field_reconstruction | 主体恢复、构建核心场、准备 Gaussian 尺度 |
| r3_full_gaussian_tree | 从全场景 Gaussian 坐标构建近邻查询树 |
| r3_handoff_consistency_check | 重算选定 SfM 点并与交接数据核对 |
| ray_navigation_tree_build / ray_nearest_nodes | 短射线导航树构建 / 查询近邻节点 |
| ray_origin_r3_check / ray_segment_r3_check | 射线起点归属 / 检查线段上采样点的 R3 归属 |
| ray_triangle_intersections / ray_intersection_parsing | 原三角面求交及结果传回数组 / 奇偶与退化交点解析 |

射线子步骤为所有候选查询合计，代表点与最终回退耗时分别保存在最终报告的 classification_performance 中。导航建图和审计本身对三角面的测试使用 navigation_ 前缀。文件摘要与已验证的几何对象限当次进程使用。来源路径、设备、inode、大小、纳秒级修改时间或状态变更时间有变化时重新读取。下一次运行重新校验首次读取的文件。
