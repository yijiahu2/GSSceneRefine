# GSSceneRefine：最终R3与原始Poisson最大主体空间关系专项诊断 V1

## 目的

本工具专门定位为何一个场景的 `roi_barrier_parity` 被两级 `TOP_TO_BOTTOM_PATH_FOUND_SAMPLED_GRAPH` 阻止。

**只读诊断，不改变正式阶段二算法、不更改粗细灌水硬性条件、不修改原始Poisson / ROI Gaussian / R3、不删除任何Gaussian。** 所有文件只写入指定的全新 `--output-dir`。

主要核对三项：

1. **真实交线拓扑：** 使用已经完成的“原始全场景Poisson最大主体 × 最终R3”交线NPZ，重新检查每条线段是否组成无断头、无分叉的闭合离散环。
2. **空间连通及测试种子：** 从既有粗、细灌水NPZ读取每个空间连通分量，重新以相同R3场景上方向与上、下10%边壳采样规则统计测试点分别落在哪个分量。对已保存的上下路径核查其节点所处区域，并导出直观的彩色空间证据。
3. **原始三角面侧别验证（默认开启）：** 对细网格两个最大连通区域之间的多组短连接，使用同一个经过来源核对的原始Poisson最大主体与R3函数，检验连接是否位于R3，计算**真实原始三角面射线穿越奇偶性**，判断这两个区域的采样位置是否有位于Poisson不同侧的证据；并独立复验旧“上下贯通路径”是否真的没有穿过原始三角面。

**注意：** 一组短线段的奇偶一致性只能支持/反驳局部两侧关系；不能据此证明整个ROI完全水密。不能将“上下连通”直接等同于“Poisson漏水”；也不能因发现两个分离的区域便宣称没有其他漏水通道。

## 安装

```bash
cd /home/xth/GSSceneRefine
conda activate gsrefine
unzip -o /mnt/f/Edge/GSSceneRefine_R3_wall_connectivity_diagnosis_V1_20261111.zip -d /home/xth/GSSceneRefine
python -m pytest -q tests/test_r3_wall_connectivity_diagnosis.py
```

## 僧帽峰一次执行

```bash
ROI_RUN=/mnt/f/danxia_experiment_archive/GSSceneRefine_results/sengmaofeng/methods/absgs/roi_002
AUDIT="$ROI_RUN/cleanup/stage2_final_v2/diagnostics/parity/geometry_audit"

python -m tools.audit_r3_wall_connectivity \
  --manifest "$ROI_RUN/metadata/manifest.json" \
  --audit-dir "$AUDIT" \
  --output-dir "$ROI_RUN/cleanup/diagnostics/r3_wall_connectivity_v1"
```

会直接复用原有 `coarse_connectivity.npz`、`fine_connectivity.npz`、`global_poisson_final_r3_intersection.npz` 和 `coarse/fine_leak_path.xyz`，不重新运行粗细网格的全局求交与灌水；默认会重新载入原始最大主体与R3场作少量真实三角面短射线检查。

若想先只看已有缓存中的空间区域、不重新载入原始几何，可以另外指定一个空目录，加 `--no-triangle-probe`。此模式无法判断最大两个区域是否位于原始Poisson两侧。

## 主要输出

位置：`$ROI_RUN/cleanup/diagnostics/r3_wall_connectivity_v1/`

| 文件 | 作用 | 查看程序 |
|---|---|---|
| `r3_wall_connectivity_diagnosis.json` | 交线闭合性、各区节点数、上/下种子分布、路径验证、三角面侧别统计 | 任意文本编辑器 |
| `original_global_poisson_largest_body.ply` | **用于阻隔的原始全场景Poisson最大主体**（约445368面），非R3裁剪曲面 | CloudCompare |
| `actual_R3_wall_intersection_samples.ply` | 原始三角面与最终R3边界的交线采样点（红色） | CloudCompare |
| `fine_connected_regions.ply` | 细网格所有可通行节点，按连通区域着色 | CloudCompare |
| `fine_sampled_R3_shell.ply` | 网格近似的最终R3边壳采样点（**不是精确ROI曲面**） | CloudCompare |
| `fine_upper_shell_seeds.ply` | 原灌水规则选择的高处测试点（蓝色） | CloudCompare |
| `fine_lower_shell_probes.ply` | 原灌水规则选择的低处测试点（橙色） | CloudCompare |
| `fine_top_bottom_witness.ply` | 既有上到下可通路径的密集可视化点（紫红色） | CloudCompare |
| `fine_cross_region_probe_endpoints.ply`、`fine_cross_region_probe_lines.ply` | 两个最大区域之间短线段的端点及路径 | CloudCompare |
| `cross_region_chord_parity.csv` | 每条侧别检测线段的原三角面奇偶判定及有效性 | 表格软件 |
| `coarse_*` | 同样的粗网格空间区域、种子、路径可视化 | CloudCompare |

**所有诊断PLY都是普通彩色XYZ点云/三角网格，目的为几何叠加检查，不能把它们当成SuperSplat Gaussian模型。** 既有正式`gaussian_roi_clean.ply`和阶段一`gaussian_roi.ply`不受影响。

## 推荐CloudCompare显示顺序

1. 载入 `original_global_poisson_largest_body.ply`，显示原始最大主体曲面。
2. 叠加红色的 `actual_R3_wall_intersection_samples.ply`，观察交线是否合理沿R3边界分布。
3. 叠加 `fine_connected_regions.ply`，观察不同色彩代表的两大片空间是否被Poisson隔开（连通分量编号本身无内外含义）。
4. 分别打开蓝色 `fine_upper_shell_seeds.ply` 与橙色 `fine_lower_shell_probes.ply`，确认所谓上部点是否分布在曲面两个侧别。
5. 打开紫红色 `fine_top_bottom_witness.ply`，检查它是否一直位于某一侧。结合JSON中的 `witness_all_links_no_triangle_crossings` 判断程序是否重新验证路径无穿越。

## 理解关键报告

- `intersection_topology.topology_validated=true` + `closed_cycles`：**数值提取的离散交线**符合闭合图拓扑；不是数学精确Boolean证明。
- `levels.fine.upper_seed_counts_by_region` 和 `lower_seed_counts_by_region`：显示原“上/下”测试点是否已落入共同空间区域。若共享，并不能单凭此判定真正漏水。
- `levels.fine.witness_all_nodes_same_region`：原始报告那条上下连通路径是否完全落在同一个离散连通区域。
- `original_triangle_region_probe.region_pair_status=DIFFERENT_SIDES_SUPPORTED_BY_TRIANGLE_PARITY`：支持两个最大空间区域局部位于Poisson不同侧；与“上/下种子混入同一侧”一起，可解释旧门槛为何出现假告警，但**不能据此直接放宽正式阶段二门槛**。
- `SAME_SIDE_SUPPORTED_BY_TRIANGLE_PARITY` / `MIXED_OR_INCONSISTENT_PARITY_NEEDS_REVIEW` / `TOO_FEW_RELIABLE_CHORDS`：需检查曲面拓扑、ROI几何或采样条件，不应默认通过。
- `witness_all_links_no_triangle_crossings=true`：先前连通路径逐段通过这次的R3采样和原始三角面奇偶核验；它可能是物理上的同侧路径，不是跨墙路径。

**后续判断**：若诊断证明原上/下种子选到了同一侧，应在所有场景统一研究“经过真实三角面侧别验证的外部/内部种子”验收方案，而不是给僧帽峰增加特例、也不是立即移除粗细空间隔断测试。

## 依赖及校验

- 需要现有 `gsrefine` 环境、先前已安装的 `tools.audit_final_r3_global_poisson_barrier` 和 `gsrefine.cleanup.roi_barrier_parity`。
- 校验已有的manifest、原始全场景Poisson、第一阶段ROI Gaussian和R3交接记录的来源摘要。
- 不运行阶段二Gaussian距离筛选、辅助分量保护和最终清理。
- 4项本地单元测试覆盖种子分量分布、PLY完整性、跨区域近邻选择和路径采样；尚未在用户僧帽峰原始大数据上执行。
