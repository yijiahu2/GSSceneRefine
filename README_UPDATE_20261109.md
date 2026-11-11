# GSSceneRefine 阶段二：R3 封底二维及三维自相交来源诊断

## 目的

**只做诊断，不修复、不覆盖、不重新计算任何 Poisson / R3 / Gaussian。** 需要区分：

1. 原 COLMAP Up 投影为何形成自交二维边界；二维交点的两条三维边在空间中是否真的相交，或仅因投影重叠；
2. 主体 `poisson_roi_main.ply` 在任何封底之前有无三角面自相交；
3. 正式 `seal_main` 人工封底生成后，相交究竟发生在 `main-main`、`main-skirt`、`main-cap`、`skirt-skirt`、`skirt-cap`、`cap-cap` 哪一层；
4. 候选封底的二维有效性与三维自相交状态是否矛盾：二维有效并不代表三维封底有效。

## 基于现有日志的**已知**结果（不作为新的现场诊断）

- 主体 15,389 个面片，623 条边构成一个简单闭合三维外围环；没有非流形边，面连通分量为 1。
- 初始 COLMAP Up 投影：`Self-intersection[4.55749149550133 -2.40136873359179]`。
- 5°–25°，共检查 80 个倾斜投影方向，其中 52 个在二维平面内合法。
- 之前最多 8 个候选方向各尝试正反封底：合法投影能生成水密单体，但所有封底的 Open3D `is_self_intersecting()` 为 true。
- 5° / 180° / 正方向案例：水密、有效体积、单连通体、1055 相机 100% 外部，但三维自相交，因而绝对不得用于有符号 SDF 清理。
- 以上日志**没有相交面片配对和空间位置**，无法从日志直接断定 `main-main` 或 `main-skirt` 等来源。诊断脚本会补足这部分信息。

## 原位安装及运行

无需安装新环境或依赖（使用项目已有 `open3d`、`trimesh`、`shapely`、`numpy`）。

```bash
cd /home/xth/GSSceneRefine
conda activate gsrefine
unzip -o /mnt/f/Edge/GSSceneRefine_R3_seal_2D_3D_intersection_audit_20261109.zip -d /home/xth/GSSceneRefine

ROI_RUN=/mnt/f/danxia_experiment_archive/GSSceneRefine_results/songziguanyin/methods/absgs/roi_001

python -m tools.audit_r3_seal_intersections \
  --preseal-report "$ROI_RUN/cleanup/geometry/direct_r3_sdf/e79ac66ff1af70f592e214bc/preseal_diagnostics.json" \
  --max-directions 8
```

`--preseal-report` 选择**刚才真实失败的签名目录**，不依赖多个签名目录的时间排序。如果需要加快检查，初次可将 `--max-directions 8` 改成 `4`，不影响原始 Up 与主体内部自相交诊断。

脚本不会调用 `run.py stage2-cleanup --phase seal` 或写入正式封闭实体，仅在选定签名目录下建立 `intersection_diagnostic/`。

## 输出说明

- `intersection_audit.json`：完整诊断、无损重建的候选封底面片对分类、每一类数量及样例空间位置；**定位XYZ是两个三角面重心的中点，仅便于可视化，不是精确的相交线坐标**。
- `original_up_2d_crossings.json`：原始 Up 的二维相交边界段 ID、二维交点、对应两段边的3D端点，并计算**同一投影交点处两条3D边的空间距离**（`source_3d_gap_at_projected_crossing`）。该值非零可帮助判断“二维投影重叠但三维边未碰撞”。
- `projection_2d_trials.csv`：原 Up + 80个倾斜方向的二维有效性、最小净空、交叉数量。
- `main_intrinsic_3d_pairs.csv`：封底之前原始主体的3D相交三角面 ID；若非空，先解决主体内在自交，不能仅调整裙边。
- `intersections_3d_pairs.csv`：原始主体 / 裙边 / 底面的相交三角面配对及来源。
- `debug_intersecting_faces_main.ply`、`debug_intersecting_faces_skirt.ply`、`debug_intersecting_faces_cap.ply`：第一个合格倾斜候选（正向）中有问题的各层三角面，可在 CloudCompare/Blender 中叠加原场景查看（只有该候选存在相关相交面时才输出）。

## 按报告决定下一步，而不是盲目放行

| 结果 | 说明 | 下一步候选 |
|---|---|---|
| 原 Up 投影相交，但3D边距明显大于数值误差 | 典型投影折叠；不等于3D边界有错 | 2D投影策略不适用，转非平面补片/局部裙边 |
| `main_intrinsic_self_intersection_pairs > 0` | 观察岩体面自身存在3D相交 | 优先原面局部鉴别和处理；封底参数无解 |
| `main-skirt` 为主 | 人工裙边穿过岩体 | 尝试局部变高度/避障裙边，仅修改人工几何 |
| `main-cap` 为主 | 人工底面切入岩体 | 调整局部底部拓扑，或改非平面封底 |
| `skirt-skirt` 为主 | 人工裙边互相穿越 | 改变延伸场与裁剪方式，避免垂直投影裙边 |
| `skirt-cap` 或 `cap-cap` 为主 | 人工底面/侧裙自相交 | 使用合法的空间补片/体积布尔边界，不强行放行 |
| 全部3D交叉计数为0但生产布尔仍为true | 需要检查 Open3D 版本/几何退化与数值精度 | 转 CGAL 几何精确谓词进行独立复核 |

注意：Open3D `get_self_intersecting_triangles()` 定位的是相交三角面对。它不是精确构造相交线段的工具；需要精确裁切相交三角面时，应选 CGAL 的精确内核。二维 `Shapely.make_valid()` 会改变多边形拓扑，**不得用其结果冒充原R3边界**。

## 文献和成熟工具（核对于2026-11-09）

1. CGAL Polygon Mesh Processing, *Self-intersections* / `self_intersections()`: https://doc.cgal.org/latest/Polygon_mesh_processing/ 。先检测每对真实相交面片。
2. Open3D TriangleMesh `get_self_intersecting_triangles()`：https://www.open3d.org/docs/release/python_api/open3d.geometry.TriangleMesh.html 。直接复用当前 gsrefine 环境，实现精细定位。
3. CGAL Hole Filling / `triangulate_hole()`：https://doc.cgal.org/latest/Polygon_mesh_processing/ 。支持非平面三维环的补片，但生成结果也必须验证是否与原表面相交。
4. Attene (2014), *Direct repair of self-intersecting meshes*, Graphical Models 76(6):658–668. https://doi.org/10.1016/j.gmod.2014.09.002 。局部切割/缝合思路，适合最终定位到少量面片时借鉴。
5. Jacobson et al. (2013), *Robust Inside-Outside Segmentation using Generalized Winding Numbers*, SIGGRAPH/TOG. https://igl.ethz.ch/projects/winding-number/ 。开放、自交网格的内外推断候选；需要相机/SfM约束避免错误内外。
6. Hu et al. (2020), *Fast Tetrahedral Meshing in the Wild*, ACM TOG 39(4). https://web.uvic.ca/~teseo/publications/ftetwild/downloads/2020-fTetWild.pdf 。如果最终局部显式补片不可行，可用容差包络内的鲁棒体积构造；注意它允许输入表面偏移，不符合用户“原观察三角面坐标完全不动”作为**首选**的要求。
7. Huang et al. (2020), *ManifoldPlus: A Robust and Scalable Watertight Manifold Surface Generation Method for Triangle Soups*. https://arxiv.org/abs/2005.11621 。基于体素占据和投影优化，属于重构方案，非首选。
8. Xie et al. (2026), *Fast and Exact Winding Numbers for Triangle Meshes*, ACM TOG 45(4), DOI:10.1145/3811339. https://research-explorer.ista.ac.at/record/22241 。快速广义绕数可作为开放网格有向空间判定的研究备选，但不是无校验的“直接SDF”。

**核心原则：** 若诊断明确只有人工裙边/底面导致的相交，优先修改人工封底；若原始主表面已经自相交，则不能通过换投影角度规避。所有修复后的候选仍须通过水密、非流形、真实三维自相交、相机外部及完整核心 SfM 拟合检查，才可进入 SDF。
