# 阶段一等价加速与固定 ROI 对照

适用于 GSSceneRefine 20806e9 加当前阶段二 v7。本包改变的是已确认 ROI 到正式 R3 的计算。输入 Gaussian、Poisson、净化 SfM 和 ROI 仍由现有流程校验。

```bash
conda activate gsrefine
cd /home/xth/GSSceneRefine
unzip -o /mnt/f/Edge/GSSceneRefine_stage1_exact_v1.zip -d /home/xth/GSSceneRefine
python tools/compare_stage1_fixed_roi.py /mnt/f/danxia_experiment_archive/GSSceneRefine_results/biechuanchansi/methods/Mip-Splatting/roi_005
```

脚本从原目录读取已完成的 manifest、成果和有效配置。在同级 stage1_comparisons/stage1_ab_* 下先运行优化版、再运行逐点精算对照版，两次各用一个新进程和独立输出根。两次都执行模块三、模块四及 SfM 交接。原实验的 ROI、几何、Gaussian 和参数相同。首次准备、ROI 绘制和阶段二不计入本对照。系统文件缓存会影响两次的磁盘耗时。

终端输出分段墙钟时间、精算次数和新 PLY 路径。comparison.json 同时记录两版计时、输入哈希、与原成果的 PLY 及交接数组比较、两版四层掩码哈希比较。PASS 表示所有检查通过，MISMATCH 会逐项列出。比较过程要保存两份额外的完整 ROI PLY 和 SfM 成果。可用 --output-root 指定其他磁盘作为实验父目录。

普通 full-pipeline 和 roi-auto 命令自动使用加速逻辑。原命令中如指定 --keep-diagnostics，将计算全体候选的真实局部半径，保留完整诊断。快速模式在 metadata/stage1_timing.json 中记录提前保留、提前排除和精算的点数。phases_s 是互斥的墙钟区间。recovery_substeps_s 为其内部子项，projection_s 和 knn_s 又包含在 safety_total_s 中。safety 阶段区间也含 SfM 交接和导出前的整理工作。计时终点为成果发布完成。

快速模式在核心区域外沿用 R3 原有隐式场。全模型最小和最大的 k 项分别给出任意 k 邻域截尾均值的下界和上界。浮点边界留有保守余量。界限不能判定的点继续使用原来的投影和局部截尾均值。完整 SfM 交接逐点计算。科研诊断中未实际计算的全体候选半径分位数不会用估计值填充。
