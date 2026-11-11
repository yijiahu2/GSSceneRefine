# GSSceneRefine：第一次全场景 Poisson → R3 1×空间直接裁剪对照实验

## 目的与边界

本实验仅用于宋子观音 AbsGS `roi_001` 的独立候选曲面对照，不修改 GSSceneRefine 已有模块、不重建任何 Poisson、不封底、不运行 SDF、不写出或改写 Gaussian。通过同一个 Stage-1 R3 后恢复的隐式主体及 1×局部安全距离，以 **原始模块一 `poisson_surface.ply`** 为唯一输入曲面，执行**共享边切分**，而非根据三角面中心或 SfM 局部密度逐面删面。

具体集合为：`M_candidate = M_stage1_global_poisson ∩ {x: F_R3(x) <= r_R3(x) + epsilon}`，这里 `F_R3` 是阶段一的 `CoreImplicitField`，`r_R3` 是对完整 Gaussian 模型进行边界投影和 KNN 5% 截尾计算所得，**与 V3 Stage-1→SfM 使用同一函数和计算约束**。注意 `F_R3` 的外部距离含保守近似，输出不是精确的 Euclidean SDF、也不是语义准确的岩体真值。

`stage2_ready_r3_sfm_v3.npz` 保存的是 SfM **离散点**的空间评价；**不能直接拿 SfM 位置的距离对 Poisson 顶点插值**。新程序会只在内存中复用既有 Stage-1 算法重新构建 R3 隐式场，随后从 V3 已保存的全场景 SfM 样本中抽样比对 `F(x)` 和 `r(x)`，检查误差后才允许裁剪。

## 安装方式

ZIP 内只有三个文件：本模块、一个独立测试文件、一份说明。**不含安装器，也不替换现有代码。** 仅需：

```bash
cd /home/xth/GSSceneRefine
conda activate gsrefine
unzip -o /mnt/f/Edge/GSSceneRefine_R3_global_poisson_direct_clip_20261108.zip -d /home/xth/GSSceneRefine
python -m pytest -q tests/test_r3_direct_poisson_clip.py
```

## 送子观音 AbsGS 对照实验

```bash
ROI_RUN=/mnt/f/danxia_experiment_archive/GSSceneRefine_results/songziguanyin/methods/absgs/roi_001
python -m gsrefine.geometry.r3_direct_poisson_clip \
  --manifest "$ROI_RUN/metadata/manifest.json" \
  --fit-max-points 60000
```

运行程序会自动读取 `manifest.json` 中的 `poisson_mesh` 作为**第一次模块一全场景 Poisson 曲面**，并检查 `source_gaussian`、`roi_json`、`clean_sfm_npz` 及 `stage2_ready_r3_sfm_v3.npz` 的哈希。要求先已经成功执行 `python run.py stage2-sfm --manifest ...`；你之前生成的 V3 文件应已存在。无需重复执行局部 Poisson。

默认从 `cleanup/preview/local_poisson_preview_*.json` 自动识别用于对照的 `poisson_roi.ply` 和 `trusted_roi_poisson.ply`。若存在多个不同来源，显式指定：

```bash
python -m gsrefine.geometry.r3_direct_poisson_clip \
  --manifest "$ROI_RUN/metadata/manifest.json" \
  --compare-local "$ROI_RUN/cleanup/geometry/17a86a9b54af7b569155d319/poisson_roi.ply" \
  --compare-trusted "$ROI_RUN/cleanup/geometry/trusted_roi/a2f0072dd925e6767a732c7a/trusted_roi_poisson.ply" \
  --fit-max-points 60000
```

`--fit-max-points 60000` 以相同、确定性抽样的 **R3 1×核心 SfM** 对所有三张曲面计算**精确点到三角形**最近距离的 mean、median、P90、P95、P99；若要完全使用所有 450,874 个核心点，设为 `--fit-max-points 0`。此处 P95 只用于对照报告，不作为本实验 SDF 阈值。需要在 `gsrefine` 环境中安装 `open3d`（你的既有 Poisson 流程本身已依赖它）。

## 唯一新成果目录

```text
<roi_001>/cleanup/geometry/direct_r3_clip/<signature>/
  poisson_roi_direct_r3.ply
  clip_comparison_report.json
```

- `poisson_roi_direct_r3.ply` 是**开曲面候选**，不是 `trusted_roi_poisson.ply` 的修复版，也不是封闭实体。
- `clip_comparison_report.json` 记录所有上游哈希、真实 R3 样本回放误差、曲面面片数量、面面积、连通分量、非流形边、开放边界边数、以及各曲面对相同 SfM 样本的点面拟合统计。
- 产物目录按照输入哈希、算法版本和采样参数确定，重复运行将验证并复用。原始 PLY、NPZ、ROI、Gaussian 不改写。

## Blender 建议

三张网格分别加载，使用相同 COLMAP 世界坐标（不重新居中、不随意改对象缩放）。依次查看：

1. `poisson_roi.ply`（之前局部 SfM 二次 Poisson，外围平滑幕状曲面）；
2. `trusted_roi_poisson.ply`（在局部 Poisson 上按 SfM 支撑筛选，存在内部误删风险）；
3. **`poisson_roi_direct_r3.ply`**（从第一次全场景 Poisson 按 R3 1×裁剪，边界切分但不按 SfM 密度逐面删除）。

对比相同视角下的主体完整性、内部孔洞、外部延伸和原有 Poisson 的相应结构，结合报告中的 Open boundary edges、Connected components 和 SfM P95。注意 SfM 点到曲面的距离很小并不能证明曲面整体具有正确形态；优先保留真实岩体结构与经过验证的可视结果。

## 保守声明

这里只是小范围“直接裁剪”消融实验，不修改阶段二正式主线。`roi_surface._clip_piecewise_linear()` 采用**原三角面端点隐式值线性插值**，不是对非线性 R3 空间场每条边求解析交点；因此 ROI 小特征可能发生次面片级近似。代码会重新查询输出曲面上一批样本的 R3 真实值并在报告标记差异；本实验允许生成候选供人工比较，**不自动作为可信封底曲面**。

若最新 Stage-1 R3 回放和 V3 已保存的字段有明显不一致，程序拒绝输出，防止拿错误的 ROI 空间范围误裁剪。若源 POISSON 本身包含非目标结构或空洞，直接裁剪无法消除/修复这些原始问题。
