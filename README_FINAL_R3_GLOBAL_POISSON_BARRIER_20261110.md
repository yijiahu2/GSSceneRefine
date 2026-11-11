# GSSceneRefine — 最终R3 ROI × 原始全场景Poisson 空间隔断实验（只读诊断）

**目的**：以第一阶段已经确认的**最终恢复R3 ROI**为容器，以**原始全场景Poisson的真实三角面**为隔断，测试是否能在ROI内将上方外部连通区与下方空间有效分开；**不**使用初始俯视/立面多边形代替最终R3，**不**使用已裁剪的`poisson_roi_main.ply`作隔断，**不**修改三角面，**不**进行Gaussian删除。

## 1. 输入和几何定义

- 原始全场景Poisson：由`manifest.json`溯源至`stage2_inputs.poisson_mesh`，严格核对SHA256；只选其中**最大原始连通分量**作为主体隔断源，其他原始独立分量不参与本次主体实验。
- 最终ROI：从已核验的`stage1_r3_local_safety_v3.npz`与阶段一原始输入，复现**恢复后的连续R3 1×空间包含函数**，并与保存的数据进行抽样一致性核查。
- 初始ROI多边形：**只用于计算诊断扫描窗口的保守外包界**，不用于决定ROI成员身份或曲面交线。扫描窗口同时包含最终R3 Gaussian的空间范围，并留出边距。
- `preseal_diagnostics.json`只提供已有实验来源一致性，不作为壁面来源。
- 原始源Poisson可能含非流形、洞及大尺度非主体附属；当前只把其最大原始连通分量作为测试隔断，并不代表源曲面天然水密或语义上完美。

## 2. 算法干了什么

1. 在**原始全场景Poisson的三角顶点**直接计算最终R3函数值；原始三角网格的边两端若跨越R3=0，则在该**原始边**上使用真实R3函数二分求交，默认22轮；保存交点、残差、交线段和漏检风险统计。
2. 采用有空间分辨率上限的三维网格作为**灌水通路的采样图**，每条可能通路都调用最终R3连续函数检查是否在ROI中，并在**原始Poisson三角面上做线段与三角面穿越测试**。因此墙不是体素化后的Poisson，也不是旧裁剪Poisson。
3. 用经阶段一确认的场景上方向，在R3容器的高处边壳采样外部种子、在低处边壳采样“下方测试点”。搜索有没有穿过ROI内未被Poisson阻挡的连通路径。搜索到则保存`*_leak_path.xyz`（可在CloudCompare加载为点序列）；没有搜索到**只能说明当前网格分辨率下暂未发现漏水**，不等于已经数学证明密封。
4. 可选抽样读取之前V2的`classification_masks.npz`并输出CSV，查看旧版超阈值Gaussian在本次灌水图的空间位置。**不输出新的清理PLY**。

**精度说明（必须重视）**：最终R3是非线性隐式空间，且可能在同一原始边/三角面内出现多个交点。程序通过真实R3函数在原始Poisson边上数值求根；对原始三角面进行直接线段射线阻挡检查，绝不拿已经裁剪的三角面代替原始面。但在未知全局Lipschitz界时，不能宣称获得数学上无遗漏的“解析精确Boolean相交”。报告同时提供端点求根残差、交线中点残差、相同符号边的中点漏检抽样指标。如果存在明显异常，**不能进入自动清理**，应增加自适应三角面细分/严格拓扑检查。

## 3. 安装

Windows Edge下载ZIP后，WSL中：

```bash
cd /home/xth/GSSceneRefine
conda activate gsrefine
unzip -o /mnt/f/Edge/GSSceneRefine_final_R3_global_Poisson_barrier_audit_20261110.zip -d /home/xth/GSSceneRefine
python -m tools.audit_final_r3_global_poisson_barrier --help
```

复用当前项目安装好的`numpy scipy trimesh open3d`、阶段一R3重放函数和相关依赖；无须新建Conda环境。

## 4. 先运行“原始全场景Poisson × 最终R3 ROI实际交线”测试

```bash
ROI_RUN=/mnt/f/danxia_experiment_archive/GSSceneRefine_results/songziguanyin/methods/absgs/roi_001
PRESEAL="$ROI_RUN/cleanup/geometry/direct_r3_sdf/e79ac66ff1af70f592e214bc/preseal_diagnostics.json"

python -m tools.audit_final_r3_global_poisson_barrier \
  --manifest "$ROI_RUN/metadata/manifest.json" \
  --preseal-report "$PRESEAL" \
  --intersections-only \
  --output-dir "$ROI_RUN/cleanup/diagnostics/final_r3_global_intersection_v1"
```

输出：`roi_global_poisson_barrier_report.json`、`global_poisson_final_r3_intersection.npz`、`intersection_segments.csv`、`intersection_points.ply`（如果存在严格穿越边）。

应重点查看 `intersection.strict_crossing_original_edges`、`root_field_abs_residual`、`linear_segment_midpoint_R3_residual`、`same_sign_midpoint_sign_change_samples` 和 `original_faces_with_one_intersected_edge` 等字段。数字异常说明交线可能存在非线性遗漏，必须谨慎。

## 5. 再运行完整“隔断 / 灌水 / 漏水”测试

完整测试会重新核验第一阶段R3并计算几何交线；请指定新的空输出目录：

```bash
python -m tools.audit_final_r3_global_poisson_barrier \
  --manifest "$ROI_RUN/metadata/manifest.json" \
  --preseal-report "$PRESEAL" \
  --masks "$ROI_RUN/cleanup/preview/open_ab_v2_main/classification_masks.npz" \
  --coarse-voxel 0.30 --fine-voxel 0.17 \
  --coarse-max-cells 95000 --fine-max-cells 250000 \
  --roi-link-samples 3 \
  --output-dir "$ROI_RUN/cleanup/diagnostics/final_r3_global_flood_v1"
```

如果你在V2使用了别的`classification_masks.npz`路径，修改`--masks`参数；它只是比较辅助，可以**直接删除`--masks`整行**，不会影响核心隔断诊断。

输出：`coarse_connectivity.npz`、`fine_connectivity.npz`、`*_leak_path.xyz`（发现通路时）以及主报告。诊断只写到`--output-dir`，不覆盖原始Poisson和阶段一Gaussian。

## 6. 如何判断结果

- **`TOP_TO_BOTTOM_PATH_FOUND_SAMPLED_GRAPH`**：当前有限采样图发现一条从ROI上方到低处、没有与原始可信三角面相交的通路；应查看`*_leak_path.xyz`。这是“可能漏水”的有力证据，但低处是否是真正岩体内部，仍需对照验证。
- **`NO_TOP_TO_BOTTOM_PATH_SEEN_NOT_PROVEN_SEALED`**：当前网格没找到可走通路；**不能声称已证实密封**，因为薄缝、未捕捉到的曲面局部结构可能被采样遗漏。
- **`WINDOW_TRUNCATED`**：最终R3空间触及诊断窗口边界，测试结论无效；应增大`--padding`重新检查。
- **`BLOCKED_FLOOD_AUDIT_ERROR`**：保留已有交线与错误报告，但拒绝输出“清理成功”的成果。

### 当前算法限制

- 根据最终R3边界计算实际交线与原始Poisson三角面穿越的方案，**避免了重新构造裙边、底面和裁剪Poisson产生的几何问题**；但不能替代严格拓扑证书。
- 此测试先行检查几何隔断，**不能拿灌水图直接对2,850,016个Gaussian删减**；只有验证上方外部种子、被隔开的内部种子以及曲面与ROI闭合关系后，才能接回原SDF规则。
- 如果最终R3 ROI本身不能分成两个连通空间，原因可能是开口或源Poisson不覆盖某处；增加体素精度不一定有用。报告中的交线与漏水路径用于定位。
- 本次仍只围绕主体：**不额外引入19个附属分量作为隔断**。
