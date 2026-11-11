# GSSceneRefine Stage 2 — 单命令最终ROI Gaussian清理（V1）

> **本包仅在现有 `/home/xth/GSSceneRefine` 项目中新增两个正式 cleanup 模块和一套测试。不创建新项目，不移除旧实验代码，不覆盖任何输入PLY。**
>
> 代码已通过本地合成/接口测试；**尚未使用你的完整送子观音数据运行端到端验收**。

## 1. 确定的技术路线

1. **输入**：第一阶段最终R3 ROI的 `metadata/manifest.json`、原始全场景Poisson、R3 ROI Gaussian与SfM。所有路径由manifest及现有成果追溯，禁止程序中硬编码场景名或方法名。
2. **几何准备**：自动识别/验证R3 SfM交接数据、原始Poisson按最终R3直接裁剪后的曲面，以及主体/辅助分量划分；复用已经可信验证的缓存。未准备时调用GSSceneRefine已有的 `r3_sfm_bridge`、`r3_direct_poisson_clip`、`direct_r3_pipeline --phase check` 核心流程。
3. **空间隔断**：原始全场景Poisson最大连通主体与最终R3 ROI共同建立隔断。验证/复用空间交线和灌水诊断，并以原始三角面构建射线奇偶导航；不人工封底。
4. **全Gaussian距离**：全部Gaussian中心计算到 *R3内Poisson最大可信主体* 的无符号三角面距离。阈值 `δ = P95(SfM→main Poisson) + mean_trim_P10..P90(3·max Gaussian σ)`，由每次输入真实数据计算。仅距离大于 `δ+ε` 的Gaussian进入奇偶分类。
5. **外部候选**：仅 `d_main > δ+ε AND Outside` 构成原始删除候选；不确定分类会阻断正式输出，绝不按外部删除。
6. **辅助距离保护**：对全部R3辅助Poisson分量，无需内外判定，仅检查上一步拟删除的Gaussian到其原始三角面的距离。`d_aux <= δ` → 保留。**辅助分量个数由网格自动统计，可为0、19或其他值。**
7. **最终输出**：从第一阶段 `final/gaussian_roi.ply` 按最终保留掩码导出 **`gaussian_roi_clean.ply`**，包含完整原始Gaussian属性（SH/颜色、opacity、scale、rotation），可直接在SuperSplat中渲染。

最终删除公式：

`delete_i = (d_main_i > δ + ε) AND Outside_i AND NOT(d_aux_i <= δ)`

其中如果没有辅助曲面，则 `NOT(d_aux_i<=δ)` 对所有候选成立。

## 2. 下载并原位覆盖安装

将 `GSSceneRefine_stage2_onepass_clean_ROI_V1_20261111.zip` 下载到Windows的 `F:\Edge`，在WSL执行：

```bash
cd /home/xth/GSSceneRefine
conda activate gsrefine
unzip -o /mnt/f/Edge/GSSceneRefine_stage2_onepass_clean_ROI_V1_20261111.zip -d /home/xth/GSSceneRefine
python -m pytest -q tests/test_stage2_onepass.py
```

该ZIP包含：

- `gsrefine/cleanup/stage2_onepass.py`：阶段二单命令总控入口。
- `gsrefine/cleanup/auxiliary_protection.py`：正式辅助距离保护、V2分类掩码验证、SuperSplat原生PLY导出。
- `tests/test_stage2_onepass.py`：通用算法、原始字段保留、缓存隔离及模拟端到端测试。

**依赖现有已安装的** `gsrefine.cleanup.roi_barrier_parity`、`open_ab_main`、`direct_r3_pipeline`、R3桥接/裁剪、`tools.audit_final_r3_global_poisson_barrier`；没有重复打包这些成熟模块。仍在原有 `gsrefine` 环境运行。

## 3. 送子观音：一条命令完成阶段二

```bash
ROI_RUN=/mnt/f/danxia_experiment_archive/GSSceneRefine_results/songziguanyin/methods/absgs/roi_001

python -m gsrefine.cleanup.stage2_onepass \
  --manifest "$ROI_RUN/metadata/manifest.json" \
  --accept-auto-exterior \
  --output-dir "$ROI_RUN/cleanup/stage2_final_v1"
```

**说明**：`--accept-auto-exterior` 是显式接受“ROI最高边壳作为外部种子”的**实验性假设**。送子观音已完成外部候选可视化；其他场景务必核对外部朝向。它不是普遍正确的数学结论。若有已确认的实际外部点，可改用 `--exterior-seed X Y Z` 而不使用此标志。

总控自动检测满足哈希/来源校验的预处理成果，可能直接复用当前的 `preseal_diagnostics.json`、`final_r3_global_flood_v1` 和 `roi_parity_classify_v2`。如果没有合格缓存，则自动执行相应阶段。不同场景只改变manifest和output-dir，无需修改任何代码或参数数量。

### 新场景通用命令

```bash
python -m gsrefine.cleanup.stage2_onepass \
  --manifest "/新场景ROI运行目录/metadata/manifest.json" \
  --exterior-seed X Y Z \
  --output-dir "/新场景ROI运行目录/cleanup/stage2_final_v1"
```

外部种子坐标应位于最终R3 ROI内且**确认为可信Poisson外部**；没有确证种子时，可先使用 `--accept-auto-exterior` 进行实验性运行、查看结果。存在多套来源匹配但几何不同的缓存时程序会拒绝自动猜测，可通过 `--preseal-report` 或 `--parity-dir` 明确选择经过复核的一套。

如希望强制重新运行本场景的Gaussian分类，加上 `--reclassify`。也可通过 `--barrier-audit-dir`、`--index-cache` 复用现有完整隔断/导航结果。

## 4. 最终成果文件结构

```text
roi_001/
└── cleanup/
    ├── inputs/                                  # 既有R3 SfM共享缓存（缺失时自动生成）
    ├── geometry/                                # 既有R3直接裁剪/主辅分量共享缓存
    ├── diagnostics/                             # 可复用的已验证隔断缓存
    └── stage2_final_v1/                         # 本次指定 --output-dir
        ├── gaussian_roi_clean.ply               # ★ 唯一主成果：干净完整ROI 3DGS
        ├── stage2_final_report.json             # 完整来源和数量守恒报告
        └── diagnostics/
            ├── final_classification_masks.npz   # 最终掩码：删除、辅助保护、阈值
            ├── aux_protected_gaussians.ply       # 仅辅助保护点，可在SuperSplat查看
            └── parity/                          # 未使用旧分类缓存时才生成
                ├── classification_masks.npz
                ├── roi_navigation_cache.npz
                ├── roi_parity_report.json
                └── geometry_audit/              # 缺失已验证隔断缓存时才生成
```

仅主成果 `gaussian_roi_clean.ply` 是完整干净ROI场景；`aux_protected_gaussians.ply` 只是其中被辅助分量救回的一小部分用于对照。新生成的中间 outside/inside Gaussian预览PLY在最终成功后默认清除以控制空间（掩码、缓存和报告保留）；如需保留，加 `--keep-classification-previews`。

**注意现有GSSceneRefine代码约定**：阶段二SfM、直接R3裁剪和主辅分量通用缓存的首次创建，仍位于当前 `ROI_RUN/cleanup/inputs` 和 `ROI_RUN/cleanup/geometry`，它们依然处于第一阶段用户指定的结果根目录下。该包未擅自改写原有共享缓存位置；最终成品及本次特有诊断均保存于 `--output-dir`，项目源码不产生 `runs/`。若需要所有首次共享缓存也严格只能写到 `--output-dir`，需另行修改既有三个底层模块的缓存接口，不能把当前版宣称为完全满足该要求。

## 5. 研究、性能及安全边界

- **不预设**送子观音、AbsGS、623交点、19分量、151076个候选或 `δ=0.122685`；所有量均由数据生成。两个或多个独立可用空间区域均按其真实拓扑标记，但隔断失败会拒绝正式输出。
- 主体 Poisson用于距离和空间隔断（全场景最大主体用于墙；R3内主体用于距离）；辅助Poisson仅参与距离保护，**不进入隔断**，也不再次判断内外。
- 当前方案是此前已通过预览实验的基线固定版，**不是最终语义准确性证明**。19个辅助分量中可能有噪声，本版统一保护可能带来少量误保留。
- 不确定的超阈值Gaussian、冲突导航区域、错误来源哈希、错误/不完整Poisson隔断，均阻止正式输出；不会为了“无Uncertain”强行删除。
- 完成后报告将输出输入总数、超阈值数、原外部候选数、辅助保护数、最终删除数、最终保留数及完整来源哈希。必须满足 `N_input = N_keep + N_remove`。
- `--accept-auto-exterior` 只是允许使用**尚未由独立观测证实**的顶部外部种子；最终报告标记为实验性。
- 阶段二后续可围绕“可信Poisson完整性优化”和“异常高斯与误删高斯的识别”分别开展迭代；本版先固定可复现实验基线。

## 6. 本地验证结果

- 16项新总控模块单元/模拟集成测试、既有18项分类和辅助分量回归测试通过。
- 新增了2项关于多辅助分量数量和不同Gaussian方法缓存隔离的测试；合计**36项测试通过**。
- 本地测试环境没有安装完整CUDA/Open3D重建环境，测试覆盖流程与接口而非真实送子观音大场景端到端运行。**首次在WSL正式运行仍需检查报告和SuperSplat视觉质量。**
