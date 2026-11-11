# GSSceneRefine 阶段二：Direct R3 主体／附属曲面分层更新

日期：2026-11-09。适配上一版 `GSSceneRefine_R3_topology_transactional_fix_20261109.zip`，本更新直接取代其 **Stage-2 默认修复策略**，不删除旧版代码以便开展消融实验。

## 1. 改动范围

仅修改 `gsrefine/cleanup/direct_r3_pipeline.py`，新增 `gsrefine/geometry/r3_main_auxiliary.py`，更新两个测试文件。**不修改**模块一、保存的双正交 ROI、第一次全场景 Poisson、Direct R3 裁剪算法、Gaussian PLY、SfM 数据或 `run.py`。

1. **主体提取**：从经过SHA256来源核验的 `poisson_roi_direct_r3.ply` 出发，按三角面共享边的连通分量提取唯一最大连通分量，输出 `poisson_roi_main.ply`。不平滑、不移动顶点、不修补三角形、不重新 Poisson、不做顶点焊接。
2. **附属曲面保留**：其余所有面片原样作为 `poisson_roi_auxiliary.ply` 保存。提供 `main_aux_face_map.npz`，记录两层每个三角面在原始裁剪文件中的面编号，另存 `main_aux_partition_report.json` 和所有19个分量的独立观测证据。**没有任何面片被自动删除。**
3. **封底门槛**：仅主体进入已有的裙边／底面算法；必须通过单连通、唯一无分叉的开放边界环、0条非流形边、无顶点扇区分裂、主外围环坐标一致、完整核心 SfM 拟合质量阈值。封底后继续检查水密性、绕向、体积、单实体、COLMAP相机外侧比例及外法向一致性；**新增 Open3D 自相交、边流形、顶点流形检查**。不合格时 fail-closed。
4. **Direct SDF**：主封闭实体是唯一有符号距离来源。仅对主 SDF 准备删除的 Gaussian 计算到附属曲面的**无符号距离**。距离不大于 `delta` 的候选依据独立的多位置 SfM + Gaussian 支撑和到主体曲面分离性分为 `distributed_both_sources_and_separated` 与 `review` 两类，都暂时保留，分别计数、输出诊断；其他 Gaussian 遵循原主 SDF 判据。此举避免自动误删未确认的附属真实结构，但不是语义真实性证明。
5. **Sci reproducibility**：仍使用全量 R3 核心 SfM 进行正式拟合，`--fit-max-points` 默认0；新的 `REVISION` 生成独立缓存签名，避免复用旧版拓扑修复输出。`--phase sdf` 只出诊断，正式导出仍需显式 `--accept-provisional`。

## 2. 原位安装

将本 ZIP 使用 Windows Edge 下载到 `F:\Edge`，WSL中执行：

```bash
cd /home/xth/GSSceneRefine
conda activate gsrefine
unzip -o /mnt/f/Edge/GSSceneRefine_R3_main_auxiliary_20261109.zip -d /home/xth/GSSceneRefine
python -m pytest -q tests/test_stage2_direct_r3_pipeline.py tests/test_stage2_main_auxiliary.py
```

## 3. 首先只运行 check

```bash
ROI_RUN=/mnt/f/danxia_experiment_archive/GSSceneRefine_results/songziguanyin/methods/absgs/roi_001
python run.py stage2-cleanup --manifest "$ROI_RUN/metadata/manifest.json" --phase check
```

核对输出 JSON 和签名目录下的 `main_aux_partition_report.json`：

| 检查项目 | 送子观音原始诊断预期 | 说明 |
| --- | ---: | --- |
| `topology_original.mesh_faces` | 15,549 | 原始裁剪文件，不动 |
| `topology_partition.main_face_count` | 15,389 | 最大连通主体 |
| `topology_partition.auxiliary_face_count` | 160 | 独立保留 |
| `topology_partition.auxiliary_component_count` | 19 | 原始小分量 |
| `topology.nonmanifold_edges` | 0 | 只检查主体 |
| `topology.open_boundary_edges` | 623 | 只检查主体 |
| `topology.open_boundary_components` | 1 | 还必须简单闭环 |
| `topology_partition.original_r3_boundary_preserved` | true | 所有开放边的原始坐标一致 |
| `sfm_fit.fit_sfm_count` | 450,874 | 只在输入和环境未变化时如此 |
| `preseal_eligible` | 待真实检查 | 不是保证一定通过 |

如果 `status=ELIGIBLE_FOR_SEAL_TRIAL`，再逐级执行：

```bash
python run.py stage2-cleanup --manifest "$ROI_RUN/metadata/manifest.json" --phase seal
# 仅封底成功后
python run.py stage2-cleanup --manifest "$ROI_RUN/metadata/manifest.json" --phase sdf --keep-diagnostics
```

**不要在初次检查前直接运行 `export`。** 如需研究者人工核查后的暂定成果，可在确认 `sdf` 报告及辅助观测候选后，再显式通过 `--phase export --accept-provisional` 导出；该结果仍标记为 `PROVISIONAL_REQUIRES_VISUAL_VERIFICATION`。

## 4. 输出（写入用户指定的 ROI_RUN 结果目录）

```
roi_001/cleanup/geometry/direct_r3_sdf/<new_signature>/
    poisson_roi_main.ply
    poisson_roi_auxiliary.ply
    main_aux_face_map.npz
    main_aux_partition_report.json
    preseal_diagnostics.json
    sealed_poisson.ply              # 仅封底成功时
    sealing_report.json             # 仅封底成功时
```

SfM拟合统计、19个分量的空间覆盖和分离性、Gaussians在辅助曲面附近得到保护/需人工复核的数量，均写入JSON报告。

## 5. 保留的安全边界与局限

- 主体最大连通分量的判据只是**拓扑连通性**，不是“主体语义确认”；场景间应核对主体占比和来源。并列最大分量会阻断运行。
- 19个附属结构**没有删除**；观测支持不能证明其必然属于岩体。距离很近但证据不足的 Gaussian 只列为待人工复核，不冒险自动删除。
- 只有主封闭实体参与 SDF 符号定义，附属曲面只能提供局部无符号保护带。
- 新增的自相交检查调用 `open3d.geometry.TriangleMesh.is_self_intersecting()`；若用户现有Open3D版本不支持或发生错误，将按失败处理，不会跳过检查。
- 保持旧版完整 SfM P95 和 `delta` 的计算规则；既有参数的实验口径保持一致。
- 本包经过**34项合成和接口模拟测试**，但不能替代在用户机器上的真实送子观音 Poisson、450,874 SfM、Gaussian 和相机的检查。
