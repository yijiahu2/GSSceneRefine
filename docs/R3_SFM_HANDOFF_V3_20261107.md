# GSSceneRefine：R3 主体恢复后隐式场 → 净化 SfM（1×/2×）原位更新

## 本次范围

本包以 GitHub `yijiahu2/GSSceneRefine` main 的提交 `bad9ca5ac672ed63deac361efb2d9e5398287560` 为基线，只完成阶段一与阶段二输入的衔接，不改变：

- 双正交交互 ROI、R3 Stage-A 主体恢复、Stage-B Gaussian 1×自适应安全带和 Gaussian R3 输出掩码；
- 已有净化 SfM 数据、原始全场景 Poisson；
- Stage-2 局部 Poisson、封底及 Direct SDF 阈值 `delta = P95 + trimmed 3σ major axis`。

这里的 **R3 位置距离** 是原 Stage-1 `CoreImplicitField` 对 **恢复后主体** 的空间查询。SfM 核心为 `F_R3(x) <= r_R3(x)+eps`，局部 Poisson 支撑为 `F_R3(x) <= 2*r_R3(x)+eps`。不以初始 ROI 的 `roi_signed` 代替恢复后的主体，也不把 Gaussian R3 掩码等同于 SfM 掩码。

## 安装：先解包再原位应用（单个 shell 命令）

**注意：这个 ZIP 是版本校验的原位更新包，而不是未经校验直接覆盖大文件的 ZIP。** 解压只添加新模块、安装器、测试、说明；紧随其后的 `--apply` 才修改原有五个文件。它不会创建第二份项目、Conda 环境或项目内的备份/补丁目录。

```bash
unzip -o /mnt/f/Edge/GSSceneRefine_R3_SfM_bridge_v3_20261107.zip \
  -d /home/xth/GSSceneRefine && \
python /home/xth/GSSceneRefine/tools/apply_r3_sfm_handoff.py \
  --project-root /home/xth/GSSceneRefine --apply
```

若需预检，可先运行同一安装器但省略 `--apply`，它默认只检查不修改。安装器对真实代码采用 Git blob SHA 校验，发现意外变化则报错并停止，**不强制覆盖**。成功后重复执行具有幂等性。

## 最先验证（不进入 Poisson/封底/SDF）

```bash
cd /home/xth/GSSceneRefine
conda activate gsrefine
python -m pytest -q tests/test_stage1_r3_sfm_bridge_v3.py tests/test_r3_sfm_install_contract.py
```

针对已经存在的送子观音 Stage-1 R3 `roi_001`：

```bash
ROI_RUN=/mnt/f/danxia_experiment_archive/GSSceneRefine_results/songziguanyin/methods/absgs/roi_001
python run.py stage2-sfm --manifest "$ROI_RUN/metadata/manifest.json"
```

- 如果 manifest 中已包含 Stage-1 v3 的空间场与 SfM，直接校验并复用。
- 如果是旧版本、尚未生成精确空间场，则读取 Stage-1 manifest 中的原始全场景 Gaussian、可信 Poisson、确认 ROI 和净化 SfM，并调用原有 `study.analyze()`、`anchored_local_recovery_faces()`、`CoreImplicitField()`、`estimate_geometric_support_radius()` 等函数只重建 R3 所需空间支持。**不会重训 3DGS、不会重绘 ROI、不会重新执行 Poisson 重建、不会改写 Stage-1 的 Gaussian PLY。**
- 对旧版本的补算增加已保留 Gaussian 采样点一致性校验，发现源输入或计算结果不一致将停止。

旧 ROI 的 SfM 产出在 `cleanup/inputs/sfm_roi_local_v3.npz`，其逐位置 R3 记录保存在 `cleanup/inputs/stage2_ready_r3_sfm_v3.npz`；新 ROI 在 `final/sfm_roi.npz`，精确记录在 `metadata/stage1_r3_local_safety_v3.npz`，报告在 `metadata/roi_sfm_report_v3.json`。manifest 的 `stage2_inputs` 会记录新 ROI 的两份路径。

输出 `sfm_roi.npz` 中包括 `xyz`、原始净化 SfM 对应属性、`in_core_roi`、`in_r3_geometric_core`、`r3_core_field_distance`、`stage1_r3_local_safety_distance`、`build_outward_distance_local` 和 `source_row_indices_in_clean_sfm`，可逐点审计、复算 1×/2×选择。

## 接下来独立验收局部 Poisson（本次不修改算法）

只有核验 `sfm_roi.npz` 空间点选择正确后，再运行：

```bash
python run.py stage2-cleanup \
  --manifest "$ROI_RUN/metadata/manifest.json" \
  --surface-only
```

这会进入现有局部 Poisson，仅用于可视检查，不封底、不执行 SDF 清理。后续的可信主体选择、局部 Poisson 与封底质量改善仍是下一开发阶段，本包不宣称它们已完成。

## 变更文件

- 新增 `gsrefine/geometry/r3_sfm_bridge.py`：真实 R3 场读取、SfM 1×/2×选择、验签、旧 ROI 补算及独立 CLI。
- 原位修改 `gsrefine/roi/two_stage_sdf.py`：只在计算 R3 的原有 field/cfg 已确定后触发交接。
- 原位修改 `gsrefine/roi/study.py`：最终 R3 时传入 SfM 输入路径；Gaussian 筛选条件不动。
- 原位修改 `gsrefine/roi_init/output_layout.py`：先以临时输出 Gaussian 的内容哈希发布精确 SfM，后执行原有 `os.replace`，解决正式 PLY 尚未存在的问题。
- 原位修改 `gsrefine/cleanup/direct_sdf.py`：Stage-2 读取 v3，拒绝不含恢复主体语义的旧 v2 数据；阈值、封底和清理公式不动。
- 原位修改 `run.py`：添加只计算/核验 SfM 的 `stage2-sfm` 子命令。
- 新增两份相关回归测试；不覆盖原有测试。

## 限制与核验声明

构建端已对新模块执行 Python 编译检查，新增单元测试 6 项通过，安装器的 dry-run、应用、重复运行和异常版本拒绝已在受控测试环境验证。由于构建端无法读取用户 WSL 的真实送子观音 PLY/NPZ 和执行完整项目环境，**尚未在真实送子观音数据上运行完整 Stage-1/Stage-2 端到端实测**；首次 `stage2-sfm` 结果须核对实际 SfM 点云、R3 外伸岩体覆盖和报告来源哈希。
