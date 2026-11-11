# GSSceneRefine：19个辅助Poisson小分量距离保护消融（SuperSplat可视化版）

## 目的

**不重新进行空间分类，不引入SfM/语义检查，不修改原始数据。** 仅从已有V2的 `outside_candidates`（原始ROI中已被V2判为外部且超过主曲面距离阈值的Gaussian）提取距离**19个辅助Poisson分量任意一处原始三角面**不超过原阈值 δ 的Gaussian。

数学规则：`protected = V2_remove AND d(Gaussian center, poisson_roi_auxiliary.ply) <= V2_delta`。

**导出的 `aux_protected_gaussians.ply` 是完整GSPL/3DGS Gaussian二进制PLY**，保留原始 `x/y/z`、`f_dc_*`、所有 `f_rest_*`、`opacity`、`scale_*`、`rot_*` 和原始其他属性，可直接在 **SuperSplat** 中按Gaussian显示；不使用 `Open3D.write_point_cloud`，也不烘焙点颜色。

## 安装（只新增工具与测试，不改动现有算法）

Windows Edge下载到 `F:\Edge` 后执行：

```bash
cd /home/xth/GSSceneRefine
conda activate gsrefine
unzip -o /mnt/f/Edge/GSSceneRefine_Aux19_distance_protection_SuperSplat_20261111.zip -d /home/xth/GSSceneRefine
python -m pytest -q tests/test_aux_component_distance_ablation.py
```

## 运行（当前送子观音AbsGS）

```bash
ROI_RUN=/mnt/f/danxia_experiment_archive/GSSceneRefine_results/songziguanyin/methods/absgs/roi_001
PRESEAL="$ROI_RUN/cleanup/geometry/direct_r3_sdf/e79ac66ff1af70f592e214bc/preseal_diagnostics.json"

python -m tools.ablate_auxiliary_distance_protect \
  --manifest "$ROI_RUN/metadata/manifest.json" \
  --parity-dir "$ROI_RUN/cleanup/preview/roi_parity_classify_v2" \
  --preseal-report "$PRESEAL" \
  --expected-components 19 \
  --output-dir "$ROI_RUN/cleanup/preview/aux19_distance_protection_v1"
```

默认辅助网格是与`PRESEAL`同一目录下的 `poisson_roi_auxiliary.ply`，包含原先的19个小分量（预计160面）。如果用户的实际位置不同，可附加 `--aux-poisson /absolute/path/poisson_roi_auxiliary.ply`，但必须在同一次ROI运行的`cleanup`目录下。**不要使用全场景Poisson或者15,389面的主体作为辅助曲面。**

## 成果

| 文件 | 用途 |
|---|---|
| `aux_protected_gaussians.ply` | **直接导入SuperSplat**查看19个辅助分量保护下来的原始Gaussian |
| `aux_protected_roi_indices.npy` | 被保护Gaussian在原始ROI Gaussian中的行索引 |
| `aux_distance_protection_masks.npz` | V2删除候选掩码、保护掩码、剩余候选掩码、精确距离，方便下一轮消融 |
| `aux_protection_report.json` | δ、19分量数、保护与仍可删数量、输入签名 |

这次仅导出**被辅助分量保护的Gaussian**，不是最终清理的完整场景。若保护数量为0，导出的PLY为空；不应把它解释成错误的渲染结果。

## 数据与研究边界

- 主曲面距离阈值、V2内外分类、全部原始Gaussian属性原样不动。
- **不管**19个小分量位于主曲面哪一侧，只计算无符号距离。
- 19个小分量未验证均是真实结构；保护到的Gaussian仍需在SuperSplat/CloudCompare中判断是否属于乔木、岩体、或误保留的漂浮点。
- 相交距离用`open3d.t.geometry.RaycastingScene.compute_distance`对原始辅助三角面查询，不使用最近顶点距离。
- 不用旧实验的固定113,811数量或固定δ；两者从本次真实V2结果中读取与校验。保护边界严格采用`d_aux <= delta`。
- 输出目录必须是同一ROI运行`cleanup`下的新空目录；原始PLY、既有清理PLY、V2分类掩码都不会改变。
