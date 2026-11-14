# GSSceneRefine 阶段一 + 阶段二：一键执行

本更新仅增加新的 `python run.py full-pipeline` 命令，保留原来的：

- `python run.py roi-init ...`（阶段一，交互式画ROI）
- `python run.py roi-auto ...`（阶段一，复用已确认ROI）
- `python run.py stage2-sfm ...` / `python run.py stage2-cleanup ...`
- `python -m gsrefine.cleanup.stage2_onepass ...`（阶段二）

## 新命令

```bash
cd /home/xth/GSSceneRefine
conda activate gsrefine
python run.py full-pipeline \
  --config configs/default.yaml \
  --scene biechuanchansi \
  --method Mip-Splatting \
  --sfm-dir /home/xth/datasets/danxiashan/biechuanchansi/sparse \
  --gaussian-ply /mnt/f/danxia_experiment_archive/biechuanchansi/final30k_C_mip_accel/exports/point_cloud.ply \
  --output-dir /mnt/f/danxia_experiment_archive/GSSceneRefine_results \
  --accept-auto-exterior
```

### 运行步骤

1. 和原来完全相同，启动Edge交互界面，绘制/确认双正交ROI；原阶段一模块1–4运行。
2. 阶段一结束时，直接从 `WorkflowManager` 已验证完成的状态中获取实际ROI、结果目录、最终PLY路径，校验阶段一 `metadata/manifest.json` 的输入身份；绝不通过`glob`、`latest`或者最大ROI编号猜测目标。
3. 只有阶段一完成，才自动启动**原有** `gsrefine.cleanup.stage2_onepass`，不会改动其任何清理算法/判据。
4. 阶段二生成结果到当前ROI的 `cleanup/stage2_final_v1`。如该目录已存在，则自动使用`v2`、`v3`等空闲目录，保护之前的结果。
5. 将总状态保存在当前ROI的 `metadata/full_pipeline_status.json`；最终模型仍为 `cleanup/stage2_final_vN/gaussian_roi_clean.ply`。

**提示：** `--accept-auto-exterior`明确接受原阶段二的自动外部空间种子假设，仍然需要人工检查结果。若有核实过的种子，可改为 `--exterior-seed X Y Z`；两者不可同时使用。本命令不会偷偷放宽这道安全限制。

如果原 `configs/default.yaml` 的 `workflow.auto_after_confirm` 为 `false`，一键模式会在启动Edge前拒绝运行，不会悄悄跳过阶段一的自动计算。原命令行为不受影响。

### 安装

ZIP中只有新增文件，没有替换完整已有源码。先解压，再运行一次安装器，安装器会检查并在原版 `run.py`和`gsrefine/roi_init/server.py`中各插入一处很小的入口衔接。安装不会改动两阶段科学算法、Poisson、R3或历史结果。

```bash
cd /home/xth/GSSceneRefine
unzip -o /mnt/f/Edge/GSSceneRefine_full_pipeline_v1_20261113.zip -d /home/xth/GSSceneRefine
python tools/install_full_pipeline.py --check
python tools/install_full_pipeline.py
python -m pytest -q tests/test_full_pipeline.py
python run.py --help
```

`--check`仅读取源码，不写入。安装前检查源码标记和Python语法；无重复写入，修改前备份原有入口文件。已有其他命令保持原样。

## 边界和故障处置

- **没有绘制确认ROI**、阶段一失败或关闭窗口：阶段二不启动。
- **未验证的阶段一结果**：阶段二不启动。
- **阶段二遇到空间连通/计算盒子/种子/奇偶冲突失败**：保留诊断数据和`stage2_failure_report.json`，状态标记为失败，不宣称清理成功。
- 这是新增的流程编排，不改变阶段二的空间分隔条件；如某个旧场景仍触发阶段二数值错误，新命令不会掩盖该问题。
- 新命令不会删除已存在的 ROI、缓存、PLY 或结果；运行阶段一仍然遵守原有ROI不可覆盖约束。
