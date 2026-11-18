# GSSceneRefine 项目工程统一与阶段二恢复（2026-11-14）

## 范围和原则

本次是**工程重构**，不改变研究已确认的 R3 恢复、Poisson 选取/空间分隔、SDF 自适应距离、Gaussian 删除判定、辅助组件保护等数学算法。

针对已同步的 GitHub `yijiahu2/GSSceneRefine`，统一以下行为：

1. **一个场景ROI只保留一个权威JSON**：Edge确认后保存为`<result_root>/<scene>/roi/roi_NNN.json`。新阶段一的`metadata/manifest.json -> stage2_inputs.roi_json`直接指向它，不再生成`<ROI_RUN>/metadata/roi.json`副本。`metadata/effective_scene.yaml`仍保留，属于必需的本次实际配置快照。
2. **修改检测**：阶段一发布时重核ROI的SHA256，阶段二统一通过已有`gsrefine.cleanup.sdf_common._load_stage1_manifest`验证ROI、Poisson、SfM、R3 Gaussian的哈希。若用户手动编辑已使用的ROI，停止复用旧结果。哈希校验是逻辑不可变约束，不宣称操作系统层面禁止手动编辑。
3. **历史数据兼容**：此前已生成、以`metadata/roi.json`为来源的历史阶段一成果，不迁移、不删除。仅在原始ROI仍存在、与历史快照字节一致、处理报告来源指向同一ROI时允许显式恢复。
4. **任务状态隔离**：新打开Edge会话时，不再把历史完成任务打印成此次确认任务的完成日志。历史任务状态查询能力保留。
5. **参数匹配的分类缓存**：在新的`roi_parity_report.json`额外记录`nearest_k`、`min_path_agree`、`roi_link_samples`、`seed_mode`。自动复用分类结果要求数值设置和外部种子方式匹配。以前没有这些字段的分类缓存不会被自动当作有效结果，需重新分类。几何缓存原有的来源校验、空间分隔算法均不变。
6. **阶段二重启**：一键命令增加`--resume-manifest`，明确指定已完成的某个阶段一manifest。不会启动Edge或阶段一。每次生成新`stage2_final_vN`目录，不覆盖失败或成功的旧实验。

## 安装

请将ZIP保存到Windows `F:\Edge`，然后在WSL的现有项目目录执行：

```bash
cd /home/xth/GSSceneRefine
conda activate gsrefine
unzip -o /mnt/f/Edge/GSSceneRefine_engineering_unification_v1_20261114.zip -d /home/xth/GSSceneRefine
python tools/install_engineering_unification_v1.py --check
python tools/install_engineering_unification_v1.py
python -m pytest -q tests/test_engineering_unification_v1.py tests/test_full_pipeline.py tests/test_stage2_onepass.py tests/test_stage2_roi_barrier_parity.py
```

`--check`会检查所有正式文件中的锚点、编译所有改动后的代码，任何匹配失败都不修改正式文件。正式安装会在`.git/gsrefine_patch_backups/`备份被替换的原文件；重新执行不重复修改。

注意：`unzip`只放入安装脚本、完整调度入口载荷、回归测试和说明文档；**不直接覆盖生产模块**，必须通过上述`--check`后才安装。

## 使用命令（原有方式不变）

### 新实验：从Edge开始，自动执行两阶段

```bash
python run.py full-pipeline \
  --config configs/default.yaml \
  --scene biechuanchansi \
  --method Mip-Splatting \
  --sfm-dir /home/xth/datasets/danxiashan/biechuanchansi/sparse \
  --gaussian-ply /mnt/f/danxia_experiment_archive/biechuanchansi/final30k_C_mip_accel/exports/point_cloud.ply \
  --output-dir /mnt/f/danxia_experiment_archive/GSSceneRefine_results \
  --accept-auto-exterior
```

### 从当前已完成的 roi_003 恢复阶段二，无须重画ROI

```bash
ROI_RUN=/mnt/f/danxia_experiment_archive/GSSceneRefine_results/biechuanchansi/methods/Mip-Splatting/roi_003

python run.py full-pipeline \
  --resume-manifest "$ROI_RUN/metadata/manifest.json" \
  --accept-auto-exterior
```

可以额外带`--stage2-name stage2_final_v1`（默认）。如果目录已存在，会创建下一个未使用的版本。可选添加`--output-dir <root>`来限制只能使用属于指定结果根目录的manifest。

注意：`--accept-auto-exterior`仅表示允许使用自动推测的外部空间种子，仍需研究者核查其物理合理性；如有已核验的种子，可使用`--exterior-seed X Y Z`代替。

### 原独立命令照常使用

`python run.py roi-init ...`、`python run.py roi-auto ...`、`python -m gsrefine.cleanup.stage2_onepass --manifest ... --output-dir ... --accept-auto-exterior`不变。

## 不改变与未实现的范围

- 无法从GPU进程内部任意步骤恢复：本次支持的是**从完成的阶段一恢复阶段二**。阶段二合格的共享Poisson/SfM/空间审计及分类缓存仍按原流程复用。
- 不会无条件复用缺少分类参数记录的旧分类缓存；重新计算优先保证数据一致性。
- 不删除已存在的`metadata/roi.json`历史副本或任何已完成的输出。
- 并未在用户WSL/GPU与所有真实场景运行最终的两阶段重建；随包回归测试验证的是工程控制流程与数据身份检查。
