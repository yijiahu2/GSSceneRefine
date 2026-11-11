# GSSceneRefine 阶段二：局部可信主体曲面 → Poisson 封底 → Direct SDF

版本：`GSSceneRefine.trusted_observed_roi_surface.v1.20261108`  
依赖：已经成功安装并测试 `GSSceneRefine_R3_SfM_bridge_v3_20261107.zip`。  
原位更新：**仅为仓库新增一个几何模块，安装器对现有 `gsrefine/cleanup/direct_sdf.py` 做少量锚点修改**。不会覆盖已经验收的 R3–SfM 模块、不修改已生成 Gaussian、SfM、Poisson 或阶段一清理成果。

## 一、实现范围

既有阶段二从 `sfm_roi_local_v3.npz` 直接重建 `poisson_roi.ply`，但 Poisson 外插（例如画面上超出橙色 SfM 分布的灰色外侧幕状表面）不能无条件认为是可信主体。本次接入：

1. **可信观测曲面筛选**：利用 V3 `in_core_roi` 的 R3 1×净化 SfM 作为主体空间支持种子；2×外圈 SfM 仍只用于 Screened Poisson 拟合，不得自动确认为目标。对局部 Poisson 三角面计算中心到最近核心 SfM 的距离，按局部 KNN 点距自适应限制，并拒绝跨度异常大的面片，防止跨空洞桥接。随后分析面片连通分量；只有主连通分量面积/面片占比足够，才允许自动进入下一阶段。筛选仍属于**几何候选**，不宣称语义真值。
2. **拓扑与封底**：在可信观测曲面上复用原有 `prepare_open_poisson`、`choose_main_component`、`seal_main` 和相机外部验证。多重要连通分量、非流形、多个孔洞边界、投影自交、相机验证失败都应中止，不任意删面、补洞、放宽门槛。
3. **可信曲面 P95**：核心 SfM 到筛选后的 *观测面片* 的精确点面距离（不包含裙边或底面），并检查其局部覆盖率。之后仍使用现有 `P95 + 截尾 Gaussian 3σ 最长半轴` 与 `signed SDF <= δ + ε`。
4. **无损 Gaussian**：只对阶段一 `final/gaussian_roi.ply` 进行 SDF 分类；现有 `GaussianPLY.export` 及属性校验流程不变。`--no-export` 先输出诊断；实际输出必须在查看候选曲面和封底后显式执行 `--allow-inferred-guard-export`。

注意：当前分割采用 *SfM 支撑与连接关系*，不是新训练的语义分割模型，也不是“任意孔洞自动填补”。如果 Poisson 与核心 SfM 根本不吻合，正确结果是停止并报告几何不可靠，而不是强行清理 Gaussian。

## 二、安装

下载 ZIP 到 Windows Edge 目录，在 WSL 运行：

```bash
cd /home/xth/GSSceneRefine
conda activate gsrefine

unzip -o /mnt/f/Edge/GSSceneRefine_stage2_trusted_cleanup_20261108.zip \
  -d /home/xth/GSSceneRefine

# 仅校验，不修改（可选）
python tools/apply_stage2_trusted_surface.py --project-root /home/xth/GSSceneRefine

# 在当前仓库中应用（只修改 direct_sdf.py；不会重新安装 Python 环境）
python tools/apply_stage2_trusted_surface.py --project-root /home/xth/GSSceneRefine --apply

python -m pytest -q tests/test_stage2_trusted_surface.py
```

**不能直接以 ZIP 覆盖你当前的 `direct_sdf.py`**：本包故意不携带该文件的历史副本，避免覆盖已经成功安装的 V3 R3–SfM 衔接。原位修改必须执行最后的 `--apply`。安装器不自动备份，不改动仓库其他源码；如果版本不匹配立即拒绝，不做猜测替换。

## 三、按既定阶段顺序验收——送子观音 AbsGS

```bash
ROI_RUN=/mnt/f/danxia_experiment_archive/GSSceneRefine_results/songziguanyin/methods/absgs/roi_001

# 1. 已完成的 V3 R3–SfM；仅在需要核对输入时执行
python run.py stage2-sfm --manifest "$ROI_RUN/metadata/manifest.json"

# 2. 复用先前的局部 Screened Poisson，新增可信主体提取；不封底、不删 Gaussian
python run.py stage2-cleanup --manifest "$ROI_RUN/metadata/manifest.json" --surface-only
```

程序会分别打印：

- `Original local Poisson:` 原始 `poisson_roi.ply` 路径（**不修改**）；
- `Trusted ROI Poisson preview:` 新生成 `trusted_roi_poisson.ply` 路径；
- `Preview report:` 日志及来源报告。

新的可信曲面与报告放在 `<ROI_RUN>/cleanup/geometry/trusted_roi/<signature>/`，并按输入文件 SHA256 与参数缓存。报告内容含原始/保留/不支持的三角面数量、支持面片主连通分量占比、R3 1×核心 SfM 点数、SfM 局部 KNN 距离统计、拓扑边界预检查等。将其与 `gaussian_roi.ply`、原始 `poisson_roi.ply` 比较，重点看顶部与左右侧 R3 恢复结构是否完整、幕后巨大外推面是否去除，以及真实岩体有无误删。

```bash
# 3. 只有可信曲面合理，才测试封底+SDF；不导出 Gaussian
python run.py stage2-cleanup --manifest "$ROI_RUN/metadata/manifest.json" --no-export
```

此命令将在允许的情况下生成 `cleanup/cache/<signature>/sealed_poisson.ply`、`sealing_report.json`，并输出 `cleanup/preview/cleanup_summary_<signature>.json`。必须检查水密性、法向/相机方向检查、P95/局部覆盖率与删除数量，未通过会停机；此时**绝不强行跳过验证**。

```bash
# 4. 仅在完整检查通过后，手动明确批准导出候选清理结果
python run.py stage2-cleanup \
  --manifest "$ROI_RUN/metadata/manifest.json" \
  --allow-inferred-guard-export
```

成功后：`<ROI_RUN>/cleanup/final/gaussian_roi.ply`；原有 `<ROI_RUN>/final/gaussian_roi.ply` 不变。请复核岩体、R3 补全区域、悬浮 Gaussian 与误删风险。不同方法运行 `roi-auto` 时复用同场景 ROI，但新建各自的 `methods/<method>/roi_001` 专属清理目录，不跨模型复用 Gaussian 缩放统计与分类结果。

## 四、可调参数（不强迫修改配置）

默认全都由内置 `gsrefine/geometry/trusted_roi.py` 提供，无需修改你的 `configs/default.yaml`。若不同森林/丹霞场景稀疏程度明显不同，可在原有 `stage2_trusted_surface` 配置项中按需覆盖，例如：

```yaml
stage2_trusted_surface:
  sfm_knn: 7
  surface_support_factor: 3.5
  spacing_cap_p90_factor: 2.5
  max_triangle_edge_factor: 10.0
  min_main_face_fraction: 0.85
  min_core_sfm_coverage: 0.95
  coverage_support_factor: 4.5
```

默认不建议调大支撑半径以“让程序通过”，那会重新放入 Poisson 外插面。`min_core_sfm_coverage` 是正式封底前的核心 SfM→可信观测面的逐点覆盖要求；默认 ≥95%。覆盖率不足应先研究真实局部 Poisson 法向/分布、ROI 或边界筛选，不应随意降低。

**特别注意历史 ROI 的配置：** 阶段二默认读取对应成果目录下 `metadata/effective_scene.yaml`，而非运行时临时修改的 `configs/default.yaml`。只有明确需要做跨阈值消融实验，才使用已支持的 `--config` 指定经过核验的有效配置文件。

## 五、验证与限制

- 在构建环境中的独立合成网格数据上，验证 R3 核心保留、独立外围幕状曲面移除、连续相接的外推曲面移除、2×背景点不被当作目标种子、参数拒绝、缓存可重复复用及修改器拒绝缺失 V3 接口等，共 8 项通过。
- 当前构建环境没有真实送子观音的 Gaussian / SfM / Poisson 大规模文件，也未安装 Open3D，**尚未运行真实案例 Poisson 拓扑封底与 Gaussian SDF 全流程**。真实运行时可能因曲面具有复杂洞、欠支撑或者相机核验失败而中止，不能宣称已获得最终干净的 Gaussian ROI。
- 阶段一原生 R3 算法以及 V3 SfM 提取接口不改动，阶段二原有 SDF 阈值公式和无损属性导出不改动。本包只增加几何支持筛选、接入位置、可靠性门槛及输出来源记录。
