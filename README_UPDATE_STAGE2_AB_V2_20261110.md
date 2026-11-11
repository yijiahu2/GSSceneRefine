# GSSceneRefine — Stage 2 A+B V2（开放主体 + 双分辨率隐式空间）

**版本：2026-11-10；性质：已实现的实验代码更新包，尚未经送子观音真实数据执行验证。**

## 实验边界（不改变）

- 可信主体：从已通过 preseal 检查的 `poisson_roi_main.ply` 读取；送子观音主体为 7,999 顶点、15,389 三角面。绝不增加裙边、底面、重新Poisson或移动原始顶点。
- **仅最大主体**：其他19个Poisson分量完全不读取、不作为距离参考或保护几何。
- 距离A：Open3D计算**Gaussian中心到原始主体三角面的精确无符号距离**。
- 阈值A：`delta = full-core SfM->main P95 + P10～P90 trimmed mean(3*exp(Gaussian log scale max))`，`epsilon=1e-5`。旧实验报告当前应得 `0.09082496203482143 + 0.031859954171468384 = 0.12268491620628982`；**程序读取真实数据重新计算，绝不硬编码**。
- 外部B：COLMAP实际 `points3D.bin` 轨迹（image ID ↔ 三维点）+ 原始可信曲面定向种子 + 广义绕数柔性先验 + 两套不同分辨率的体素图割 + 相机观测连通性检查；不再采用旧版逐Gaussian最近12相机投票。
- **正式删除规则唯一**：`outside_confident` 且 `distance_to_original_main > delta+epsilon`。内部或不确定保留。无额外异常识别。
- 产生的空间边界是**有证据约束的估计性体积分类，并非数学认证的真实封闭岩体**；距原曲面和开放边界较近的位置保护为不确定。

## 代码修改清单

- `gsrefine/cleanup/implicit_space.py`（新增）：COLMAP真实观测轨迹、体素射线、源曲面内部/外部种子、广义绕数先验、PyMaxflow图割、空间连通性、双分辨率一致性和开放边界保护。
- `gsrefine/cleanup/open_ab_main.py`（修改）：保留旧版相机投票作为显式消融选项；默认执行V2。输出上界预览、分类掩码与统计报告、保留原始所有Gaussian PLY属性。
- `gsrefine/cleanup/direct_r3_pipeline.py`（修改）：沿用原来的`stage2-cleanup --phase open-ab`，新增B模式与体素/轨迹/诊断参数。
- `tests/test_stage2_open_ab_implicit.py`（新增）：含COLMAP二进制轨迹、距离规则、空间连通性、边界/射线和实际图割接线模拟测试。

未修改 `sdf_common.py`（阈值公式），未修改已有 `check / seal / sdf / export` 的实现及原始PLY。

## 安装（WSL现有环境/现有根目录）

把Zip下载到Windows Edge目录后：

```bash
cd /home/xth/GSSceneRefine
conda activate gsrefine
unzip -o /mnt/f/Edge/GSSceneRefine_stage2_open_AB_V2_implicit_20261110.zip -d /home/xth/GSSceneRefine

# libigl此前Open A+B已安装；如缺失，安装到同一个gsrefine环境
python -c "import igl; print('libigl OK')" || python -m pip install libigl
# 新增的图割依赖：已有numpy，无需让pip擅自改动整个数值栈
python -c "import maxflow; print('PyMaxflow OK')" || python -m pip install --no-deps PyMaxflow==1.3.2
python -c "import numpy,scipy,trimesh,open3d,igl,maxflow;print('A+B V2 dependency check OK')"
```

COLMAP官方二进制格式包含`images.bin`、`points3D.bin`，本实现需要它们位于同一sparse目录；如缺少`points3D.bin`，B层会拒绝自动清理，不能退回未经验证的相机射线代替。

## 推荐先运行：只检查A层距离上界（不需要重算B）

```bash
ROI_RUN=/mnt/f/danxia_experiment_archive/GSSceneRefine_results/songziguanyin/methods/absgs/roi_001

python run.py stage2-cleanup \
  --manifest "$ROI_RUN/metadata/manifest.json" \
  --phase open-ab \
  --b-method implicit-volume \
  --diagnostic-only \
  --preseal-report "$ROI_RUN/cleanup/geometry/direct_r3_sdf/e79ac66ff1af70f592e214bc/preseal_diagnostics.json" \
  --output-dir "$ROI_RUN/cleanup/preview/open_ab_v2_distance_upper"
```

`gaussian_distance_only_upper_bound.ply` **只是激进上界诊断**：对所有`distance>delta`的Gaussian删除（预计本场同输入约151,076个），可能会误删真实结构，**不可作为最终结果**。它仅用于观察大块残影是否能够由现有A层中心距离删除规则处理。输出包括`distance_main.npy`和`open_ab_v2_report.json`。

## 运行正式的A+B V2实验

```bash
ROI_RUN=/mnt/f/danxia_experiment_archive/GSSceneRefine_results/songziguanyin/methods/absgs/roi_001

python run.py stage2-cleanup \
  --manifest "$ROI_RUN/metadata/manifest.json" \
  --phase open-ab \
  --b-method implicit-volume \
  --preseal-report "$ROI_RUN/cleanup/geometry/direct_r3_sdf/e79ac66ff1af70f592e214bc/preseal_diagnostics.json" \
  --output-dir "$ROI_RUN/cleanup/preview/open_ab_v2_main"
```

注：正式模式**也会自动输出距离上界PLY**，方便同输入同阈值比较；如果磁盘不够，可只执行诊断优先判断。

默认：每次完整建两套有内存上限的**均匀体素网格**，粗网格上限96,000格，细网格上限240,000格（并不是已经实现自适应八叉树）；默认最多读取约130,000条实际COLMAP观测射线，局部记录基于P95安全余量。会自动放大体素间距以满足上限。可使用：

```bash
--max-volume-nodes 240000 --max-sfm-rays 130000
# --voxel-size 0.15 仅作为期望间距，实际间距必须服从网格容量上限
```

如果粗/细尺度无法区分、SfM射线数量不足、空间内部种子与观测自由空间冲突过多、或缺少必要依赖，实验会BLOCKED：仅留下A阶段诊断，不会产生声称清理成功的正式PLY。

### 可选消融：旧版逐Gaussian相机投票

```bash
# 与上面正式命令相同，但将 --b-method 改成 camera-legacy
# 且必须指定一个新的 --output-dir，不覆盖历史结果。
```

## 输出（只在指定 output-dir 中）

| 文件 | 说明 |
|---|---|
| `gaussian_roi_clean.ply` | **V2实验候选**，不是未经验证即可发布的正式结果。 |
| `gaussian_distance_only_upper_bound.ply` | 诊断上界，可能误删真实Gaussian。 |
| `distance_main.npy` | 与输入Gaussian行次序严格一致的原始曲面无符号距离。 |
| `classification_masks.npz` | `keep/remove/over_threshold/inside_confident/outside_confident/uncertain_kept/distance_main/signed_distance_proxy/occupancy_label`。 |
| `implicit_coarse_grid.npz`、`implicit_fine_grid.npz` | 空间标签、网格原点/实际分辨率，便于再现与调试。 |
| `open_ab_v2_report.json` | 来源SHA256、旧阈值分解、COLMAP真实观测轨迹数量、双尺度统计和删除数量。 |

不覆盖用户输入或阶段一`final/gaussian_roi.ply`，也不会在仓库根目录创建runs。默认拒绝覆盖已有非空输出目录。

## 如何判断本次研究是否取得了进展？

先用同一视角打开：旧版`open_ab_main_v1/gaussian_roi_clean.ply`、V2`open_ab_v2_main/gaussian_roi_clean.ply`、激进上界`gaussian_distance_only_upper_bound.ply`。V2应比旧B有更合理的外部空间覆盖，且不能破坏岩体真实主体。如果连“距离上界”仍有大面积残影，说明残影可能来自**中心距离未超阈值**的Gaussian（例如大椭球或不透明渲染残影），此时必须先诊断这类现象，而不能凭空调整已经冻结的delta或宣称B已解决。

## 测试与真实性边界

```bash
cd /home/xth/GSSceneRefine
python -m pytest -q tests/test_stage2_open_ab_implicit.py tests/test_stage2_open_ab_main.py \
    tests/test_stage2_direct_r3_pipeline.py tests/test_stage2_main_auxiliary.py \
    tests/test_stage2_seal_projection_fallback.py
```

更新包在构建容器已通过49项相关单元/合成回归测试（包括以NetworkX模拟mincut接口验证图割接线）；**由于构建容器不包含原始285万Gaussian和COLMAP完整工程，也没有Open3D/libigl/PyMaxflow原生库，未完成真实场景端到端执行或对应原生库的集成测试**。这次是可执行的研究实验版，不应宣称实际清理效果已经验证成功。
