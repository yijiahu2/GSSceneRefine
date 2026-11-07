# GSSceneRefine 第二阶段集成方案（增量覆盖补丁）

基线：GitHub `yijiahu2/GSSceneRefine`，`main` 提交 `9a9be5c71740c7f5d55bbf571d279a38ce390e27`（2026-11-06）。

**基本边界：第一阶段为唯一上游，第二阶段只处理 `final/gaussian_roi.ply`。** 绝不重新进行SfM清理、Poisson曲面重建、双正交ROI筛选或R3恢复。第一阶段导出的 `metadata/manifest.json` 是唯一正式入口；该文件记录真实Poisson、净化SfM、已确认的ROI及R3 Gaussian。所有导出均为原生全属性3DGS PLY。

## 1. 与现有仓库的归并关系

| 原压缩包的能力 | 现有仓库对应能力 | 处理方式 |
|---|---|---|
| 读取COLMAP、SfM净化、Poisson重建 | `gsrefine.geometry.prepare`、`sfm_clean`、`poisson` | 完全复用现有代码，不移入旧重建脚本 |
| 读取Gaussian PLY/保留全属性 | `gsrefine.roi.core.GaussianPLY`、`gsrefine.roi.finalize` | 复用已有全属性读写器，不引入第二套PLY实现 |
| ROI筛选、R3连续主体恢复 | `gsrefine.roi.study`、`two_stage_sdf`、模块四 | 不改已有算法 |
| COLMAP相机读取与竖直方向 | `gsrefine.geometry.orientation` | 复用相机姿态；使用已确认的ROI `up_vector`，而非硬编码+Z |
| Poisson主连通体识别、开放边界、裙边+封底 | 现有仓库无等价的正式SDF实体封闭 | 新增 `gsrefine/geometry/sdf_sealing.py`（基于历史已验证算法，仅提取必需函数） |
| SfM到真实Poisson表面的距离 | `gsrefine.roi.core.exact_sfm_to_poisson_distance_open3d` | 直接复用，P95在未封底Poisson上计算 |
| P10—P90 Gaussian 3σ长半轴截尾均值 | 第二阶段专有 | 新增到 `gsrefine/cleanup/direct_sdf.py` |
| 直接封闭Poisson连续SDF过滤 | 第二阶段专有 | 新增到 `gsrefine/cleanup/direct_sdf.py`，取消先提取外扩PLY步骤 |
| 原旧项目的脚本/配置/重复测试 | 不属于GSSceneRefine结构 | 不拷入仓库，不再依赖 `sfm_surface` 环境 |

## 2. 真正的数据通路

```text
GSSceneRefine 当前第一阶段（不修改）
  模块1：SfM/可信Poisson ──> geometry/poisson_surface.ply + cleaned_sfm.npz
  模块2：双正交ROI用户确认 ──> roi_001.json
  模块3：ROI局部连续主体恢复和全边界保护 ──> R3保留集合
  模块4：全属性导出 ──> methods/absgs/roi_001/final/gaussian_roi.ply
                              metadata/manifest.json
                                        │
                                        ▼
GSSceneRefine 第二阶段（本补丁，独立可重复验收）
  校验第一阶段manifest+源文件SHA256+相机坐标一致性
        ↓
  读取**既有**Poisson；检测主连通体与开边界；基于Stage1 up_vector试验两种封底方向
        ↓
  验证水密性、正体积、全部注册相机是否位于外部、法向与相机一致性
        ↓
  实体跨方法共用缓存，SfM对**未封底Poisson**统计P95
        ↓
  基于当前模型的R3 Gaussian统计3σ长半轴P10-P90截尾均值
        ↓
  delta = P95 + 当前模型截尾尺度均值
        ↓
  分块计算每个R3 Gaussian中心到原始尺度**封闭实体**的直接连续SDF
        ↓
  删除 iff 有限SDF > delta + epsilon；非有限SDF保守保留并标记
        ↓
  final/gaussian_roi_cleaned.ply + cleanup/分类掩码与阈值 + metadata/cleanup_manifest.json
```

**注意：** 本补丁会严格拦截无效封底、相机内外判别失败、可信Poisson拟合P95异常或来源不一致等情况。拒绝执行时不会伪造“清理成功”。第一阶段已完成的产物不会丢失。

## 3. 不新建仓库或环境的本地安装

安装前，在原仓库执行：

```bash
cd /home/xth/GSSceneRefine
git status --short
git rev-parse HEAD
# 建议备份变更过的源码及已有结果后再覆盖；不要使用 git reset --hard。
```

Windows Edge 下载补丁压缩包后，在现有 WSL 项目目录**原位**解压：

```bash
unzip -o \
  /mnt/f/Edge/GSSceneRefine_stage2.zip \
  -d /home/xth/GSSceneRefine
```

只使用原来的 `gsrefine`：

```bash
conda activate gsrefine
cd /home/xth/GSSceneRefine
python -m pip install -r gsrefine/cleanup/requirements.txt
python run.py stage2-cleanup --help
python -m pytest -q tests/test_stage2_cleanup.py
python -m pytest -q tests
```

如果已安装满足范围的 `numpy/scipy/open3d/trimesh`，无须再次重装。这里特别要求 **Shapely >= 2.1**，以使用约束三角化生成封底；本补丁已在原 `environment.yml` 中将 Shapely 最小版本提升至 2.1，其余依赖保持不变。不要切换旧 `sfm_surface` 或 `gspl` 环境。

## 4. 真实运行（先验收阶段一，再运行阶段二）

第一阶段按仓库原 README 命令启动，确认 ROI 后自动完成 R3 导出。找到相应输出的 `metadata/manifest.json`。例：

```bash
export RUN_ROOT=/mnt/f/danxia_experiment_archive/GSSceneRefine_results
export ROI_RUN="$RUN_ROOT/songziguanyin/methods/absgs/roi_001"

# 先只生成第二阶段诊断，不导出清理模型。允许随后正式执行。
python run.py stage2-cleanup \
  --manifest "$ROI_RUN/metadata/manifest.json" \
  --no-export

# 检查 ROI_RUN/cleanup_preview/ 的分类诊断和缓存检查通过后，正式清理：
python run.py stage2-cleanup \
  --manifest "$ROI_RUN/metadata/manifest.json"
```

默认从第一阶段的 `metadata/effective_scene.yaml` 读取 `roi_init.orientation.colmap_images`；如该路径缺失，可显式添加 `--colmap-images /.../sparse/images.bin`。会验证图片姿态文件与阶段一 ROI 保存时的方向哈希一致。

**输出**（属于原第一阶段的同一个 `roi_001` 目录，不复制仓库）：

```text
.../songziguanyin/
├── geometry/sealed_sdf/<场景签名>/sealed_poisson.ply
│   └── .../sealing_report.json          # 场景几何共用缓存
└── methods/absgs/roi_001/
    ├── final/gaussian_roi.ply            # 第一阶段R3，保持不变
    ├── final/gaussian_roi_cleaned.ply    # 第二阶段结果，新文件
    ├── metadata/manifest.json           # 第一阶段，保持不变
    ├── metadata/cleanup_manifest.json   # 第二阶段完成记录
    ├── cleanup/
    │   ├── signed_distances.npy
    │   ├── classification_masks.npz      # keep/remove/uncertain/nearest_artificial_face
    │   ├── removed_roi_local_indices.npy
    │   ├── sdf_threshold.json
    │   └── cleanup_summary.json
    └── cleanup_preview/                 # --no-export 单独保存的诊断
```

`removed_roi_local_indices.npy` 的索引是**第一阶段 Gaussian ROI PLY 的局部行号**，**不是**原始全场景训练 checkpoint 的行号。不同3DGS方法的掩码不可混用。

## 5. 论文与实验参数

当前正式规则保持用户技术稿的已验证模型级阈值，区别于阶段一的局部方向性感知ROI保护宽度：

\[
 a_i=3\max(\sigma_{i1},\sigma_{i2},\sigma_{i3}),\qquad
 \delta_m=P_{95}\big(d(P_{\mathrm{SfM}},M_{\mathrm{trust}})\big)+\overline a_{m,\mathrm{trim}(10\%-90\%)}.
\]

\[
 \mathcal G_{\mathrm{clean,ROI}}=\{G_i\in\mathcal G_{\mathrm{R3}}:d_s(\mu_i,M_{\mathrm{sealed}})\le\delta_m+10^{-5}\}.
\]

非有限SDF保守保留并记录；清理后PLY仍存储原生Gaussian字段，不将辅助Poisson网格用于渲染。默认使用当前**R3 Gaussian**重新估计尺度，因此送子观音历史**AABB版**AbsGS `0.0791926490` 和 E_depth `0.0802549075` 不会机械照搬，删除比例也不能直接与历史AABB版本同比。

选项通过第一阶段的场景配置 `stage2_cleanup` 节设置，不覆盖用户原始多边形，亦无需新增场景特定配置文件：

```yaml
stage2_cleanup:
  trim_fraction: 0.10
  sigma_multiplier: 3.0
  nsamples: 3
  chunk_size: 40000
  epsilon: 0.00001
  min_main_face_fraction: 0.85
  bottom_margin_fraction: 0.10
  min_camera_outside_fraction: 0.85
  min_camera_normal_agreement: 0.60
  max_p95_to_sfm_spacing: 3.0
  max_p95_to_roi_diagonal: 0.05
```

运行时读取实际 `metadata/effective_scene.yaml`；如果要调整参数，应在第一阶段启动之前将配置写进基础 YAML，并检查实际生效的 effective config。已完成的第一阶段运行仍可通过 `--config /path/to/effective_scene.yaml` 显式指定等价的完整配置，但请保证相机来源、ROI和输入数据未改变。

## 6. 科研验证与工程验收

1. **原位更新验证**：`git status --short` 仅出现与补丁对应的新增/修改；`python run.py roi-init --help` 和原阶段一 tests 继续通过。
2. **几何身份**：输入Poisson哈希等于阶段一 `manifest.input_sha256.poisson`；SfM哈希一致；COLMAP images哈希符合确认的ROI；输出Poisson是从既有可信表面构造的封闭实体，不重新执行Poisson重建。
3. **封闭体核验**：`is_watertight = True` 且 `is_volume = True`，两方向候选均有检查报告，注册相机外部比例及法向一致性达到门槛；未满足则停止。真实多连通体的保护需要独立检查主连通体占比和被排除分支。
4. **高斯属性一致**：被保留Gaussian的PLY行序与字节级属性不变，PLY字段布局、数量和文件大小全部校验；输入R3 PLY不被覆盖。
5. **误删/漏删复核**：可分别对 `gaussian_roi.ply` 和 `gaussian_roi_cleaned.ply` 在SuperSplat或GSPL同机位比较，尤其检查裙边附近及岩壁底部。已标记 `nearest_artificial_face` 的移除项需要重点检查。
6. **科研消融**：在固定ROI/R3、高斯模型、坐标及评估视角条件下，对比 ROI only、sealed no offset、sealed adaptive direct SDF；历史外扩PLY仅做实现一致性参考。多模型、多场景结果需逐一实测。

## 7. 本补丁的验证边界

本包可在本地用 Python 静态检查和无需Open3D的几何/单元测试。真实送子观音数据、Open3D GPU/CPU raycaster、原始相机姿态文件未包含在上传的源码压缩包中，因此**不能将本补丁描述为已经在你WSL上完成端到端真实场景验证**。第一次真实运行前建议先 `--no-export`，核对Poisson方向选择及P95警报。若前置Poisson存在多个真实主体、复杂多孔边界或压扁后的边界环自交，应人工评估而不是自动降低阈值。

## 8. 与未来一键串联的关系

当前采用更安全的**阶段一完成→第二阶段显式执行**方式。已有浏览器确认ROI至R3一键工作流不受影响。待首个实际场景验收通过，再考虑在 `workflow` 完成后自动调用第二阶段；自动调用应是可配置开关，且第二阶段失败不能覆盖、重命名或删除已验证的第一阶段成果。不要将新的第二阶段误写成原 `roi-auto` 的阶段三/四的一部分。
