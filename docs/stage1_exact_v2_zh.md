# 阶段一等价加速 V2

适用基线为 GSSceneRefine 20806e9 和阶段二 original-region v7，更新包包含阶段一 V1 的必要文件。日常 full-pipeline 自动使用 V2。原有 --keep-diagnostics 仍计算完整候选半径统计。包内路径相对于项目根目录。

## 在别传禅寺复核

本次 A/B 已经留下 V1 的四个掩码摘要、运行时间和正式产物。下列命令只运行一次 V2。原 ROI 和原成果均保留，V2 在新目录导出正式 PLY 和供阶段二使用的 manifest。

```bash
conda activate gsrefine
cd /home/xth/GSSceneRefine
unzip -o /mnt/f/Edge/GSSceneRefine_stage1_exact_v2.zip -d /home/xth/GSSceneRefine
METHOD_RUN=/mnt/f/danxia_experiment_archive/GSSceneRefine_results/biechuanchansi/methods/Mip-Splatting
V1_RUN="$METHOD_RUN/stage1_comparisons/stage1_ab_cq08srjd/optimized/biechuanchansi/methods/Mip-Splatting/roi_005"
python tools/compare_stage1_fixed_roi.py "$V1_RUN" --baseline saved-v1 --output-root "$METHOD_RUN/stage1_comparisons"
```

saved-v1 逐项对比真实 PLY 文件摘要、四层掩码摘要、净化点的索引、坐标、局部半径、隐式场值、投影残差及其他交接字段。SfM 文件中引用上游文件的摘要按各自文件实测核验。历史时间来自原 V1，机器负载及操作系统文件缓存可能不同。需要新进程重新测两版时，使用原始 ROI 成果目录：

```bash
python tools/compare_stage1_fixed_roi.py "$METHOD_RUN/roi_005" --baseline v1
```

## 修改与计时

Gaussian 和净化 SfM 的边界投影沿用 V1 的有限差分步长、迭代次数和收敛标准。距离查询省去法向生成，六个差分位置共享一次最近邻查询。用包围盒给出的保守距离下界排除必定无法胜过 ROI 场值的采样球，SfM 首轮投影复用同批刚计算的场值。所有净化点仍逐点得到局部半径和残差。

同一次运行中，输入及成果的实际 SHA-256 按路径、设备、inode、大小、纳秒 mtime 和 ctime 复用。发布前核对输入版本和最终 PLY 摘要。发布步骤只解读校验一次原交接记录，筛选 SfM 时复用其数组。新复制的记录先校验内容再发布。沿用原有文件格式、压缩和内外 1×/2×选择规则。

结果在 comparison.json 中。timings.optimized.safety.projection_s 记录 Gaussian 投影，recovery_substeps_s.sfm_handoff_s 记录整个 SfM 交接，phases_s.verification_and_publication 记录最终校验发布。handoff_and_publication.steps 单列 SfM 读取、场值、投影、近邻、归档、SHA-256、复制验证、筛选等耗时。inclusive_s 包含子项，exclusive_s 扣除子项。npz_stream_write 是缓冲文件写入调用耗时，归档独占时间含序列化与压缩。文件系统写回和实际物理磁盘吞吐无法由此单独测量。
