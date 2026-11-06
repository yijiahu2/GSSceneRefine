# GSSceneRefine

GSSceneRefine 面向原始COLMAP SfM与完整3D Gaussian Splatting（3DGS）模型，提供**SfM净化与Poisson几何参考构建 → 人工双正交ROI粗选 → R3局部误截断自适应恢复 → 最终Gaussian PLY导出**的连续流程。唯一仓库 `/home/xth/GSSceneRefine`，唯一Conda环境 `gsrefine`。所有模块保持独立命令和可复现实验接口。

## 用户指定SfM输入、Gaussian输入与最终输出目录（推荐）

**首次对一个场景进行交互式ROI提取**：用户自行提供COLMAP稀疏模型目录（同时包含`points3D.bin`与`images.bin`）、完整Gaussian PLY路径、以及最终结果根目录。`--scene`和`--method`是用于隔离缓存与结果的简短标识，并非硬编码场景。

```bash
conda activate gsrefine
cd /home/xth/GSSceneRefine

python run.py roi-init \
  --config configs/songziguanyin.yaml \
  --scene bazhaifeng \
  --method absgs \
  --sfm-dir /真实路径/bazhaifeng/COLMAP/sparse/0 \
  --gaussian-ply /真实路径/bazhaifeng/absgs/point_cloud.ply \
  --output-dir /mnt/f/danxia_experiment_archive/GSSceneRefine_results
```

上述`--config`仅作为算法参数模板，输入模型与输出位置由命令参数覆盖；原配置文件不会改写。也支持分别使用`--colmap-points /.../points3D.bin --colmap-images /.../images.bin`代替`--sfm-dir`，两种指定方式不能混用。首次构建模块一可能需要较长时间；之后同一SfM源和参数复用缓存。

**输出路径说明**：对于交互式`roi-init`，`--output-dir`表示**最终ROI结果的根目录**。每次确认ROI后，模块三、四自动将`gaussians_final_R3.ply`和原有全部掩码、诊断PLY、报告输出到该根目录下的独立ROI子目录，而不是直接覆盖根目录中的一个PLY。例如：

```text
/mnt/f/danxia_experiment_archive/GSSceneRefine_results/
  bazhaifeng_absgs_roi_001/
    gaussians_final_R3.ply
    keep_mask_R3.npy
    core_mask_R3.npy
    safety_added_mask_R3.npy
    study_report.json
    workflow_result.json
    ...原有诊断文件保持不变
```

模块一净化SfM与Poisson缓存仍存放于项目的`runs/geometry/<scene>/`，模块二ROI JSON仍存放于`runs/roi_init/`；三者不与最终结果输出根目录混淆。最终结果会自动区分模型身份，避免不同3DGS方法覆盖彼此的PLY。

**同场景更换3DGS方法、复用已确认ROI时**，使用非交互式`roi-auto`。这里`--output-dir`是这一次运行的**确切结果目录**，而不是根目录：

```bash
python run.py roi-auto \
  --config configs/songziguanyin.yaml \
  --scene bazhaifeng \
  --method vanilla \
  --sfm-dir /真实路径/bazhaifeng/COLMAP/sparse/0 \
  --gaussian-ply /真实路径/bazhaifeng/vanilla/point_cloud.ply \
  --roi runs/roi_init/bazhaifeng_absgs_roi_001.json \
  --reuse-scene-roi \
  --output-dir /mnt/f/danxia_experiment_archive/GSSceneRefine_results/bazhaifeng_vanilla_roi001
```

只有明确属于同一COLMAP世界坐标系的不同方法才允许复用ROI几何，**绝不复用上一方法的Gaussian掩码**。手工指定的`roi-auto --output-dir`应当为当前模型预留独立目录，防止覆盖既有成果。

## 多场景与多3DGS方法复用（阶段一泛化）

所有工作仍在 `/home/xth/GSSceneRefine` 的同一代码仓库、同一 `gsrefine` 环境内进行，R3算法保持不变。本轮**保留原输出模式**：原有 `candidate_groups.json`、`exterior_*.ply`、`boundary_traces.npz`、`full_exterior_faces.npz`、`config_snapshot.json`、R0/R3掩码、两阶段恢复掩码、最终PLY、`kept_indices_R3.npy`、`restored_indices_R3.npy`、报告和运行日志等仍按原逻辑生成。自动模式仍只计算最终R3；需要R1/R2的完整消融时运行独立 `roi-study`。

**新场景首次启动（自动模块一 → 人工模块二 → 模块三/四）：**

```bash
python run.py roi-init \
  --config configs/songziguanyin.yaml \
  --scene bazhaifeng --method absgs \
  --colmap-points /实际场景/COLMAP/sparse/points3D.bin \
  --colmap-images /实际场景/COLMAP/sparse/images.bin \
  --gaussian-ply /实际场景/absgs/point_cloud.ply
```

`--scene` 是场景唯一标识，不能将不属于同一COLMAP世界坐标系的数据写到同一场景名下。场景A与B各自缓存模块一SfM/Poisson数据与ROI注释；同一场景中的方法A、B使用完全相同的SfM参考和Poisson曲面，但Gaussian输入及对应的保留掩码必须独立计算。默认不会擅自复用其他方法的ROI。

**同场景更换方法并复用已经确认的双正交ROI：**在第一种方法完整跑过模块一后，直接执行（不必重新人工圈选）：

```bash
python run.py roi-auto \
  --config configs/songziguanyin.yaml \
  --method vanilla \
  --gaussian-ply /真实路径/vanilla_full.ply \
  --roi runs/roi_init/songziguanyin_absgs_roi_005.json \
  --reuse-scene-roi
```

如果首次运行时使用的原始SfM不在配置里，也不在 `runs/geometry/<scene>/module1_manifest.json` 中，请额外提供 `--colmap-points` 和 `--colmap-images`。`--reuse-scene-roi` **必须显式启用**，并要求两种模型都是完整场景、使用同一COLMAP世界坐标系；源于旧AABB的ROI不能跨方法直接复用。复用的仅为双正交多边形几何，**不会复用另一方法的Gaussian掩码或统计结果**。

第二种方法的运行结果会使用带方法名和源模型标识的独立结果目录，不覆盖第一种方法；单一模型原先的编号ROI目录命名保持不变。模块一只缓存SfM/Poisson，但每次切换Gaussian都会重新进行对齐质量检查，随后分别运行R3和最终导出。

当前各方法需导出**兼容GSPL标准字段的binary little-endian Gaussian PLY**，至少包括 `x/y/z`、`scale_0..2`、`rot_0..3`；其他非标准格式需先转换，且Gaussian中心与COLMAP坐标必须一致。新场景所需SfM/Poisson质量、R3阈值仍需实测验证，不能把单一送子观音结果视为泛化精度结论。

## 模块与数据契约

1. **模块一：可信SfM–Poisson几何参考构建。** 输入原始COLMAP `points3D.bin`（以及匹配的`images.bin`用于方向估计）和完整Gaussian PLY。根据点的观测轨迹和重投影误差筛选，再进行空间离群点过滤、法向估计与一致性处理、Screened Poisson重建；输出净化SfM NPZ、原始Poisson PLY与数据完整性/拟合残差诊断。这里的“可信”表示经过可核查的几何质量检查，不表示每个面片均被证实为真实主体，也不执行模块三已停用的Poisson面片打分。
2. **模块二：交互式ROI粗选。** Windows Edge在完整Gaussian上做俯视和立面的正交多边形选择，确认并保存三维ROI JSON。仅此步骤需要人工粗选。
3. **模块三：R3自适应恢复。** 沿用已经验证成功的Poisson外围延伸识别；先恢复误截断核心，再依据**全部边界**局部Gaussian长轴的KNN两端各5%截尾均值、隐式距离场进行安全外扩。新流程只增加状态通知和最终计算模式，**不改变R3几何算法和权重**。
4. **模块四：最终PLY导出。** 保留原始Gaussian记录的所有属性与顶点顺序，输出`gaussians_final_R3.ply`，并对掩码、计数和ROI来源做一致性校验。本阶段不执行内部异常Gaussian清理。

当前自动处理只接受**完整Gaussian PLY**作为显示与导出输入；如只有训练CKPT，请先使用现有3DGS/GSPL导出流程转成保持所有Gaussian属性的PLY。初始COLMAP SfM、Gaussian、Poisson与ROI必须处于一致的原始COLMAP世界坐标系。

## 新场景或完整原始Gaussian：一条命令启动模块一和模块二

```bash
conda activate gsrefine
cd /home/xth/GSSceneRefine

python run.py roi-init \
  --config configs/songziguanyin.yaml \
  --colmap-points /home/xth/datasets/danxiashan/songziguanyin/sparse/points3D.bin \
  --gaussian-ply /请替换为实际路径/原始完整Gaussian模型.ply
```

`--gaussian-ply` **必须替换成实际完整重建Gaussian PLY路径**，不能填以前的`roi_aabb.ply`。如果同一场景的方向数据没有在配置文件中设置，添加 `--colmap-images /绝对路径/images.bin`；默认配置已经包含送子观音的`images.bin`位置。不要使用来自另一套未配准COLMAP重建的文件。

首次运行会根据源SfM和参数自动执行净化与Poisson重建，输出到`runs/geometry/songziguanyin/`，并生成按方法及模型路径区分的`active_scene__<method>__<source-id>.yaml`。之后相同输入和参数可复用缓存；若源SfM或Poisson参数改变，则重新计算模块一。原始场景配置文件及以前的AABB实验数据**不会被覆盖**。

命令完成模块一之后，终端会给出 `http://127.0.0.1:8765`。在Windows Edge打开，完成模块二粗ROI并点击“确认ROI并自动生成最终PLY”，系统会自动执行模块三/四。浏览器会显示实际执行阶段（不是虚构百分比）、报错以及最终PLY路径；刷新/重开浏览器可读取落盘的运行状态。

### 结果位置与验证

```text
runs/geometry/songziguanyin/
  02_sfm_clean_colmap_metadata.npz    # 净化后的SfM
  04_poisson_raw.ply                  # Poisson参考曲面（不作为最终渲染场景）
  sfm_clean_report.json
  geometry_quality_report.json
  module1_manifest.json
  module1_status.json
  active_scene__<method>__<source-id>.yaml  # 各方法独立的有效运行配置

runs/roi_tests/<已确认ROI文件名去掉.json>/
  gaussians_final_R3.ply
  core_mask_R3.npy
  safety_added_mask_R3.npy
  keep_mask_R3.npy
  study_report.json
  workflow_progress.json
  workflow_result.json
  workflow.log
```

若要手动对**完整Gaussian模型**的已确认ROI复跑后两个模块，必须使用该次模块一准备的模块一打印的`active_scene__<method>__<source-id>.yaml`，不要再使用指向旧AABB的原始配置：

```bash
python run.py roi-auto \
  --config runs/geometry/songziguanyin/active_scene__<method>__<source-id>.yaml \
  --roi runs/roi_init/这里填写新确认的ROI文件名.json
```

旧AABB ROI002/003不会自动当作完整PLY的已确认结果，也不会在新完整PLY的交互界面中作为当前ROI自动载入；旧实验和结果仍保留。切换模型后应重新确认ROI，以保证模型来源和Gaussian顶点索引正确对应。

## 兼容旧实验、独立工具及测试

仅执行 `python run.py roi-init --config configs/songziguanyin.yaml` 时，**保持原配置的已有SfM/Poisson和AABB验证基线**，不做自动重建。需要新数据的自动模块一，使用上述两个显式输入参数，或在场景YAML中配置`module1.mode: auto`、`module1.colmap_points`、`source_model_scope: full_reconstruction`与完整`paths.gaussian_ply`。

保留独立命令：`sfm-clean`、`poisson`、`scene-prepare`、`roi-init`、`roi-study`、`roi-finalize`及`roi-auto`。独立的R0–R3消融可继续使用`roi-study`，自动流水线只计算最终R3。

```bash
python -m pytest -q tests
node --test ui/tests/*.test.mjs
```

原始SfM和Gaussian数据、几何缓存与实验输出均不纳入Git仓库，参见`.gitignore`与`docs/development.md`。当前开源实现对模块一提供数据完整性和几何拟合诊断，**不能替代对真实场景Poisson曲面质量的人工核查**；先前AABB上验证成功的R3恢复效果，也不能直接作为完整原始Gaussian输入下已验证的结论。
