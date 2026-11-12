# GSSceneRefine：阶段二空间隔断验收 V2（通用代码更新）

## 1. 目的与修改边界

本更新仅修正 `ROI—原始全场景Poisson最大主体` 的**空间隔断验收**，保持原有R3恢复、原始三角面求交、粗/细灌水图的构建、XOR导航、Gaussian短射线、主曲面自适应阈值、辅助曲面保护、原生SuperSplat PLY导出算法不变。

旧程序：只要上下边壳测试点**有一个**空间连通，就判定隔断失败。僧帽峰粗网格顶部207点在区域0、1点在区域1；底部208点全部在区域1；细网格顶部626/3分处区域0/1，底部629点全部在区域1。这种“上下同侧正常连通”不应该单独触发失败。

新程序：保留原有粗/细网格，并以**连通区域的空间关系和原始Poisson三角面穿越奇偶**验证隔断。

## 2. 新的严格验收条件（两个尺度均执行）

1. 校验阶段一manifest、R3交接记录、原始全场景Poisson SHA；原始最大连通主体与R3交线必须构成闭合、无分叉的离散环；交线仍不是数学水密证明。
2. 粗、细网格报告均需完整、网格覆盖ROI且与各自NPZ形状、步长一致；逐网格读取 `passable` 和 `components`，保留原始上下连通结果作诊断，不再用上下连通作为独立的一票否决依据。
3. 按区域节点数量确定有实际规模的连通区域（默认：至少16个节点，且≥可通行节点1%；全部此类区域应覆盖≥95%的可通行空间）。数量不限定为2，但若空间过度破碎仍需人工复核。
4. 对有实际规模的不同空间区域选取分散的邻近节点对；要求连接处于原R3内部，并通过**原始全场景Poisson最大主体的三角面**计算有限线段穿越奇偶；每条已验证的区域关系至少8条可靠短线段，且可靠的奇偶结果必须一致。**不强制所有不同区域都为奇数关系**：同侧但不连通的区域可能出现偶数关系。
5. 构建区域级XOR关系图，要求所有实际规模区域关系有足够证据关联、无奇偶矛盾，并且存在得到三角面证明的两侧空间。
6. 粗/细网格的区域编号是局部编号，不应直接比较“区域0”；基于真实XYZ近邻匹配区域，并检查主要空间区域对应关系与相对XOR标签的一致性。无法可靠对应则拒绝。
7. 外部种子仍须确认。自动种子只作为研究性假设：不再取全体上部采样点中绝对最高点，而从**占主导的顶部连通区域**中选择最高点。若其占比不足70%则停止并要求显式 `--exterior-seed X Y Z`。外部/内部全局标签仍由现有XOR导航完成。
8. Gaussian分类仍须所有超阈值候选得到无冲突二分类；有UNKNOWN或冲突不导出最终PLY。

上述1%、95%、8条、70%属于**通用保守数值门槛**，不是地名/分量数/阈值的特例，也不是水密的数学证明；如后续消融需调整，应同时在所有场景上验证并注明版本，不在单个场景私自放宽。

## 3. 原位覆盖安装

```bash
cd /home/xth/GSSceneRefine
conda activate gsrefine
unzip -o /mnt/f/Edge/GSSceneRefine_stage2_region_separation_V2_20261111.zip -d /home/xth/GSSceneRefine
python -m pytest -q tests/test_barrier_separation_gate_v2.py tests/test_stage2_roi_barrier_parity.py tests/test_stage2_onepass.py
```

所需已有模块：`tools.audit_final_r3_global_poisson_barrier`、`gsrefine.geometry.r3_direct_poisson_clip`、`gsrefine.cleanup.sdf_common` 和其他原阶段一、阶段二模块。不创建新Conda环境，不替换完整仓库，不更改已有输入、V1/V2正式结果和既有几何缓存。

## 4. 僧帽峰：直接复用失败前已经完成的粗细网格灌水成果

```bash
ROI_RUN=/mnt/f/danxia_experiment_archive/GSSceneRefine_results/sengmaofeng/methods/absgs/roi_002
AUDIT="$ROI_RUN/cleanup/stage2_final_v2/diagnostics/parity/geometry_audit"

python -m gsrefine.cleanup.stage2_onepass \
  --manifest "$ROI_RUN/metadata/manifest.json" \
  --barrier-audit-dir "$AUDIT" \
  --accept-auto-exterior \
  --reclassify \
  --output-dir "$ROI_RUN/cleanup/stage2_final_v3"
```

其中 `--accept-auto-exterior` 只是**明确接受自动顶部种子是外部的研究性假设**，不是客观证实。建议结合原始Poisson、COLMAP相机位置检查该区域确实属于外部；如已有经确认的外部坐标，将其替换成 `--exterior-seed X Y Z`，并保留 `--reclassify`。

**若历史preseal版本不唯一**，继续添加 `--preseal-report "/当前ROI/cleanup/geometry/direct_r3_sdf/签名/preseal_diagnostics.json"`，指定与你已确认一致的曲面成果。切勿复制其他场景的签名。

**如果`stage2_final_v3`已经存在**，不要覆盖，改用新的结果目录（如`stage2_final_v4`）。

若尚无可复用隔断缓存或需要针对其他场景，可以省略`--barrier-audit-dir`。程序会在当前ROI清理目录中查找来源匹配且粗细数据齐全的历史成果，缺失时自动重新运行原始隔断审计；然后统一执行新验收。

### 可选：取消辅助Poisson保护消融

在同一新结果目录的命令中附加 `--no-aux-protection`，输出另一个完整的 `gaussian_roi_clean.ply`，**不是单独删除点预览**。不能与已存在正式输出目录混用。

## 5. 成果、诊断与验收

成功时：

```text
stage2_final_v3/
├── gaussian_roi_clean.ply                # 最终完整原生3DGS ROI场景，SuperSplat可看
├── stage2_final_report.json              # 删除/保护/保留数量与来源
└── diagnostics/
    ├── final_classification_masks.npz
    ├── aux_protected_gaussians.ply        # 仅诊断，若有保护点
    └── parity/
        ├── region_separation_gate.json    # 新的几何验收证据
        ├── roi_navigation_cache.npz
        ├── classification_masks.npz
        └── roi_parity_report.json
```

新验收失败时，`region_separation_gate.json` 记录阻断原因；一键总控仍会写出`stage2_failure_report.json`，且**不会导出正式`gaussian_roi_clean.ply`**。

### 为什么会重新分类？

之前V2的Parity分类已使用旧验收前提，不能无条件复用为新V3分类。版本指纹改为`GSSceneRefine.roi_barrier_parity_sdf.v3`；一键正式输出只接收通过新验收的V3分类报告。先前V2成果、消融及其结果均保留不变。

### 此次明确没有修改的内容

- 不修改`poisson_roi_main.ply`、`poisson_roi_auxiliary.ply`、原始Poisson或R3 Gaussian。
- 不修改自适应距离阈值`δ=SfM→Poisson P95 + P10–P90截尾的Gaussian三倍最大σ均值`。
- 不修改自适应短射线的候选内外分类规则；不修改辅助分量的距离保护逻辑。
- 不预设送子观音623条、巴寨峰635条、僧帽峰1,828条交线，不预设辅助分量数。
- 不以“粗细主要有两个区域”推断空间严格水密；本验收只表示数据支持**数值上两侧分离**。

## 6. 本地测试与限制

在构建环境中，针对新区域验收、三角面奇偶、旧上下连通误报、缺失缓存、同侧假分区拒绝、XOR矛盾及导航接口的合成测试已经通过。全量真实僧帽峰数据需在你的WSL机器中运行并审阅新报告后，才能判定V3是否正式通过；不提前宣称清理成功。
