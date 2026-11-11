# GSSceneRefine — 最终R3空间隔断 + 射线奇偶 + 原SDF阈值清理（V1实装）

## 1. 本版实现内容

这是可以原位覆盖到 `/home/xth/GSSceneRefine` 的更新包，不建立新工程/新环境，不修改任何原始ROI、SfM、Poisson、Gaussian。只写用户指定的、尚为空的 `--output-dir`。优先使用已完成的最终R3—原始全局Poisson灌水审查；其他场景可以不传审查目录，由工具在输出目录内部自动执行同一审查。

**几何绝不混用**：

- 最终ROI：重放第一阶段**最终R3连续隐式函数**，验证其Stage1记录与来源哈希。
- 空间隔断：只使用**原始全场景Poisson的最大源连通主体**及其真实三角面；不使用局部裁剪面、人工封底/裙边。
- 距离：用 `poisson_roi_main.ply` 的原始三角面，通过 Open3D `compute_distance` 对**全部**Gaussian中心计算无符号最短距离。
- 阈值：维持 `δ=P95(全部核心SfM→可信Poisson拟合误差)+P10–P90截尾均值(3×Gaussian长半轴)`，`ε=1e-5`（与既有V2实验口径一致）。
- 删除：`remove_i = (d_i > δ+ε) AND (C_i == Outside)`；未超阈值不做射线分类；内部和不确定均保留。
- 不纳入19个局部附属Poisson小分量，也不纳入其他原始全局Poisson小分量。

## 2. 怎样得到Inside / Outside

先读取R3与原始Poisson审查成果 `fine_connectivity.npz`，它只充当**空间导航格子**；对导航格子之间的实际连线调用 `Open3D RaycastingScene.list_intersections`，对**原始Poisson真实三角面**统计所有有效穿越（0次为同侧，1次翻转，2次又翻转）。

采用带XOR势能的并查集，给导航节点建立相对外部起点的 `Outside=0` / `Inside=1` 标签，对不同路径奇偶冲突的连通分量整体标记 `Unknown=-1`。选定一个ROI上方外部候选起点（如有更可靠的实测外部坐标，可用 `--exterior-seed X Y Z` 指定）；外部身份在导出清理PLY前仍需人工核查。不能因为“某空间比较低”就默认它为内部。

对于全部Gaussian，仅距离超阈值候选执行附近K个**已标记导航节点到Gaussian中心的短线段**求交，并严格检查其是否仍落在最终R3内。**至少2条有效短线段结果一致**才判定内/外；路径出ROI、落在三角面共边/端点、几何交点重合或不同路径标签冲突，标记Unknown并保留。没有V2的大宽度未知保护带，也没有相机投票/绝对绕数分类。

**科学边界**：我们先前看到的623点/623段离散闭环与粗细两级未发现“漏水”，只构成数值几何先验，不是严格水密拓扑证明。因此该版自动清理必须明确声明为**待真实场景复核的实验结果**；程序不会把未能确定的点判成外部。

## 3. 安装方式（保持项目规范）

Windows Edge下载ZIP后，在WSL执行：

```bash
cd /home/xth/GSSceneRefine
conda activate gsrefine
unzip -o /mnt/f/Edge/GSSceneRefine_R3_barrier_parity_SDF_V1_20261110.zip -d /home/xth/GSSceneRefine
python -m gsrefine.cleanup.direct_r3_pipeline --help
```

只覆盖这次修改的 `direct_r3_pipeline.py`，并新增 `roi_barrier_parity.py`、单元测试、文档；附带已有的只读相交审查工具，方便新场景自动建立索引。使用原环境中的numpy、scipy、open3d、trimesh，不额外要求`igl` / `PyMaxflow`。

## 4. 送子观音：先只分类，绝不删除

```bash
ROI_RUN=/mnt/f/danxia_experiment_archive/GSSceneRefine_results/songziguanyin/methods/absgs/roi_001
PRESEAL="$ROI_RUN/cleanup/geometry/direct_r3_sdf/e79ac66ff1af70f592e214bc/preseal_diagnostics.json"
AUDIT="$ROI_RUN/cleanup/diagnostics/final_r3_global_flood_v1"

python -m gsrefine.cleanup.direct_r3_pipeline \
  --manifest "$ROI_RUN/metadata/manifest.json" \
  --phase open-ab \
  --b-method roi-barrier-parity \
  --preseal-report "$PRESEAL" \
  --barrier-audit-dir "$AUDIT" \
  --classification-only \
  --output-dir "$ROI_RUN/cleanup/preview/roi_parity_classify_v1"
```

如果你本地 `python run.py stage2-cleanup` 已经关联 `direct_r3_pipeline`，也可将上述 `python -m gsrefine.cleanup.direct_r3_pipeline` 换为 `python run.py stage2-cleanup`。

**注意：必须使用新的空目录**；不会覆盖之前V2的实验结果。

诊断输出：

- `roi_navigation_cache.npz`：已建立的可校验来源身份的空间奇偶导航索引；同一份ROI/Poisson/审查网格可复用。
- `classification_masks.npz`：`distance_main`、`over_threshold`、`occupancy_label`（0外/1内/-1不确定）、`outside_confident`、`inside_confident`、`uncertain_kept`、`remove`、`keep`；长度与原始Gaussian顺序一致。
- `outside_candidates.ply`、`inside_candidates.ply`、`uncertain_candidates.ply`：仅诊断，保留原Gaussian字段、顺序。空组不生成PLY。
- `roi_parity_report.json`：阈值、原始来源哈希、路径冲突、网格实际分辨率、计数、未知原因及状态。
- **不输出** `gaussian_roi_clean.ply`。

查看报告：

```bash
python -m json.tool "$ROI_RUN/cleanup/preview/roi_parity_classify_v1/roi_parity_report.json"
```

## 5. 人工复核分类结果后，再正式输出实验清理PLY

在检查实际外部起点、岩体/植被保护、天空残影、Inside/Outside/Unknown三个候选PLY之后：

```bash
python -m gsrefine.cleanup.direct_r3_pipeline \
  --manifest "$ROI_RUN/metadata/manifest.json" \
  --phase open-ab \
  --b-method roi-barrier-parity \
  --preseal-report "$PRESEAL" \
  --barrier-audit-dir "$AUDIT" \
  --index-cache "$ROI_RUN/cleanup/preview/roi_parity_classify_v1/roi_navigation_cache.npz" \
  --confirm-exterior \
  --output-dir "$ROI_RUN/cleanup/preview/roi_parity_clean_v1"
```

只有此命令才输出 `gaussian_roi_clean.ply`，且报告记录 `CLEANED_EXPERIMENTAL_EXTERIOR_APPROVED`。它仍是独立的阶段二实验结果，不覆盖第一阶段正式 `final/gaussian_roi.ply`。

## 6. 换新场景：同一算法自动建索引

新场景完成第一阶段最终R3、原始Poisson与主曲面预审后，不提供 `--barrier-audit-dir` 也可以直接运行：

```bash
python -m gsrefine.cleanup.direct_r3_pipeline \
  --manifest "$NEW_ROI_RUN/metadata/manifest.json" \
  --phase open-ab \
  --b-method roi-barrier-parity \
  --preseal-report "$NEW_PRESEAL" \
  --classification-only \
  --output-dir "$NEW_ROI_RUN/cleanup/preview/roi_parity_classify_v1"
```

此时完整R3—Poisson几何相交/两级灌水审查自动生成在 `--output-dir/geometry_audit/`，然后自动生成索引并分类。**新场景未必有623条交线或两个区域**；若审查发现不能可靠建立隔断、外部节点不存在、空间路径矛盾、源数据SHA不一致，程序将阻止正式删除或把无法确认的Gaussian标为Unknown，而不是硬套送子观音结果。为了避免错误标签，新场景建议优先使用 `--exterior-seed X Y Z` 提供实测确认的外部位置，否则默认使用最终R3上方壳层最顶部的单个候选节点，必须通过可视化确认身份。

## 7. 运行前的安装与验证

```bash
cd /home/xth/GSSceneRefine
python -m pytest -q tests/test_stage2_roi_barrier_parity.py
python -m gsrefine.cleanup.direct_r3_pipeline --help
```

**本次交付的验证范围**：在提供代码的环境内，已使用模拟三角面求交、合成网格、真实的623条交线数据与现有审查工具完成单元及接口测试；没有你的完整 `gaussian_roi.ply` / Poisson实体和WSL `gsrefine` 环境，**尚未跑过送子观音真实2,850,016个Gaussian的全流程**。该限制必须通过第一次 `--classification-only` 的真实运行和CloudCompare复核来消除。

**性能设计**：所有Gaussian只调用一次（分块）局部可信Poisson的精确距离；只有超阈值Gaussian需要K条短射线。导航索引可复用、不会逐Gaussian发射贯穿ROI的长射线。实测速度由场景几何和已安装Open3D决定，不预设具体耗时。
