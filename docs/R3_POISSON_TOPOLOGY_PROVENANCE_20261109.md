# GSSceneRefine — R3直接裁剪前后Poisson拓扑来源追溯（2026-11-09）

## 目标

只追溯原始未经修复的 `poisson_roi_direct_r3.ply` 中 **11 条非流形边**及**最大主体之外的19个小连通分量**，判断其在第一次全场景 `poisson_surface.ply` 中是否已存在，或是否由 R3 裁剪切断连接而产生。

不修改或重建 Poisson，不重复计算 R3 空间场，不改动 Stage-1 Gaussian，不封底、不做 SDF、不修改/覆盖任何先前成果；只读取现有全场景 Poisson、未经修复的 R3 直接裁剪候选、Stage-1 manifest、已验证的裁剪报告，输出 JSON 与两个小 CSV。

## 安装（现有 gsrefine 环境）

```bash
cd /home/xth/GSSceneRefine
conda activate gsrefine
unzip -o /mnt/f/Edge/GSSceneRefine_R3_Poisson_topology_provenance_20261109.zip -d /home/xth/GSSceneRefine
python -m pytest -q tests/test_trace_r3_poisson_origin.py
```

此更新只包含 `tools/trace_r3_poisson_origin.py`、一份独立测试和本文档。不覆盖现有主流程，无安装器。Open3D/Trimesh/Scipy 使用 gsrefine 现有依赖。

## 使用（送子观音 AbsGS roi_001）

```bash
ROI_RUN=/mnt/f/danxia_experiment_archive/GSSceneRefine_results/songziguanyin/methods/absgs/roi_001

python tools/trace_r3_poisson_origin.py \
  --manifest "$ROI_RUN/metadata/manifest.json"
```

程序通过第一阶段 manifest 和 Direct R3 裁剪的原有 SHA256 报告绑定输入，**严格使用未修复版** `poisson_roi_direct_r3.ply`。如该 ROI 下曾生成多份 Direct R3 裁剪报告（例如不同抽样设置），显式指定先前做可视化使用的那一份：

```bash
python tools/trace_r3_poisson_origin.py \
  --manifest "$ROI_RUN/metadata/manifest.json" \
  --clip-report "$ROI_RUN/cleanup/geometry/direct_r3_clip/<既有签名>/clip_comparison_report.json"
```

该脚本不需要任何 SfM 采样参数，因为本次仅比较网格三角面来源，并不计算 SfM 拟合误差。

## 结果位置

```text
roi_001/cleanup/geometry/direct_r3_provenance/<源数据签名>/
├── topology_provenance_report.json
├── nonmanifold_edge_origins.csv
└── minor_component_origins.csv
```

JSON 输出两个关键分组：

- `nonmanifold_edge_origin`: 逐条输出异常边的位置、对应原始面索引、原始边面片数、判断结果。
- `minor_component_origin`: 逐个输出小连通分量的面数、空间位置、对应全场景连通分量 ID、是否含切分产生的新顶点、判断结果。

## 如何解释来源类型

**非流形边：**

- `inherited_from_global_nonmanifold_edge`：裁剪边被证明位于原始 Poisson 的一条非流形边上；原始边就有3个及以上相邻面，属于**源网格已有**。
- `clip_or_overlapping_sheet_topology`：对应原始边只连接1或2个源面，但裁剪后该边非流形。可能是裁剪拼接或原始重叠面造成，需要进一步检查。
- `clip_generated_or_geometric_overlap`：无法找到对应的原始拓扑边，属于新产生或被裁剪暴露的几何重叠异常，但**不能仅据此证明裁剪算法代码错误**。
- `uncertain_parent_mapping`：源面匹配不唯一，不作强制归因。

**小连通分量：**

- `preexisting_separate_global_component`：在第一次全场景 Poisson 中就属于独立分量；不是 R3 裁剪把它与主岩体切断。
- `separated_by_R3_crop_or_clip_topology`：源面原本在全场景中与主岩体连通，R3 裁剪后变成独立分量；这可能是合理的边界截断，也可能需要检查局部裁剪，不自动视作错误。
- `uncertain_parent_mapping` / `mixed_source_components_needs_review`：缺乏唯一几何对应，不武断判断。

采用 Open3D 最近三角面查找及3个三角面内部采样点交叉验证，并检验裁剪后三角面是否完全落在原始三角面内。遇到重叠/共面原始曲面会明确标记不确定性；正常保留顶点坐标只受 PLY 浮点精度影响，默认几何容差由原始包围盒对角线自适应设置。

## 后续

请提供终端输出和 `topology_provenance_report.json`。有了11条边、19个分量的分类统计，才能选择真正有针对性的最小修复：源网格固有缺陷与裁剪切断连通结构应该分开处理。**本诊断不执行任何自动修复。**
