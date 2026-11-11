# GSSceneRefine 阶段二：统一 Direct R3 曲面与清理流程（2026-11-09）

## 核心决策

取消此前两条不再合理的正式算法路径：①局部净化 SfM 重新 Screened Poisson 得到 `poisson_roi.ply`；②根据逐三角面 SfM KNN 支撑筛选得到 `trusted_roi_poisson.ply`。本次删除与这两条旧路径相关的源代码、测试和过期安装器；**不删除用户磁盘中的旧 PLY / NPZ / 诊断 JSON**，以保留论文消融对比证据。

阶段二正式几何仅使用：**模块一第一次全场景 `poisson_surface.ply`，经已确认的 R3 恢复后 1× 隐式空间场直接裁剪而得的 `poisson_roi_direct_r3.ply`**。仍保留 `sfm_roi_local_v3.npz`，但用途调整为几何拟合与安全误差估计，不再进行第二次 Poisson。第一次全场景 Poisson、原始净化 SfM、阶段一 R3 Gaussian 及现有 ROI 不作改动。

## 各步骤与必须通过的关卡

| 顺序 | 阶段 | 操作 | 进入下一步条件 |
|---|---|---|---|
| A | 封底前 | 校验 Direct R3 候选 PLY / manifest / R3 V3 输入哈希；检查三角面连通性、边界环、非流形、绕序；采用相同核心 SfM 计算未封底曲面点–三角面的 Mean/P90/P95 | 拓扑具有一个连续的主体、单个简单边界环、无非流形；P95 未超过距离/场景范围安全阈值 |
| B | 封底操作 | 保守顶点清理和绕序修正；沿 COLMAP 验证 up 轴正反方向尝试现有裙边 + 约束 Delaunay 底面 | 单环边界投影有效；最终为水密、绕序一致、单体有效封闭网格 |
| C | 封底后 | 全部 COLMAP 相机的外部位置与法向一致性检查 | 相机验证全部达到现有阈值；否则输出失败报告，**不执行 SDF** |
| D | Direct SDF | 仅作用阶段一 `final/gaussian_roi.ply`；`delta_m = e95 + a_trim(3sigma, 10% each tail)`；保留 `d_s <= delta_m + epsilon` | 先执行无导出诊断，人工验证后才允许输出候选 PLY |

说明：`e95` 来自**未封底、实际观测曲面**的核心 SfM 点–三角面距离，而不是人工裙边、底面。步骤 A 中默认只抽样 60,000 点用于快速诊断；步骤 B/C/D **强制全部 1× R3 核心 SfM**计算官方 P95。`check` 通过仅表示可尝试封底，不能保证封底结果正确。封底仅支持单环情形；遇到多环、内洞、复杂投影边界或法向相机验证失败时，严格阻断，不自动修补真实岩体孔洞。

## 安装（WSL，唯一仓库和 gsrefine 环境）

```bash
cd /home/xth/GSSceneRefine
conda activate gsrefine
unzip -o /mnt/f/Edge/GSSceneRefine_stage2_R3_canonical_cleanup_20261109.zip -d /home/xth/GSSceneRefine
python tools/apply_stage2_r3_canonical.py --project-root /home/xth/GSSceneRefine --check
python tools/apply_stage2_r3_canonical.py --project-root /home/xth/GSSceneRefine --apply
python -m pytest -q tests/test_stage2_r3_canonical_install.py tests/test_r3_direct_poisson_clip.py tests/test_stage2_direct_r3_pipeline.py
```

安装器先做版本和依赖检查，再原位修订 `run.py` 和 `roi_init/output_layout.py`，最后删除明示的旧算法源文件、测试与安装器。重复运行 `--apply` 是幂等操作。若发现意外 `run.py` 或旧 Stage1 面片挂接修改，会停止，不强行覆盖；磁盘实验成果从不触及。原先 `run.py stage2-cleanup` **现在只指向 Direct R3**，不再保留并行的 `stage2-r3` 别名。`run.py stage2-sfm` 保持不变。

## 送子观音 ROI_001 分步验收命令

```bash
ROI_RUN=/mnt/f/danxia_experiment_archive/GSSceneRefine_results/songziguanyin/methods/absgs/roi_001
```

**封底前，只检查：**

```bash
python run.py stage2-cleanup --manifest "$ROI_RUN/metadata/manifest.json" --phase check --fit-max-points 60000
```

**通过检查后再尝试封底，并使用全部 SfM：**

```bash
python run.py stage2-cleanup --manifest "$ROI_RUN/metadata/manifest.json" --phase seal
```

**封底通过后执行 SDF 诊断，但不输出清理 PLY：**

```bash
python run.py stage2-cleanup --manifest "$ROI_RUN/metadata/manifest.json" --phase sdf
```

**充分核对报告后才允许输出候选结果：**

```bash
python run.py stage2-cleanup --manifest "$ROI_RUN/metadata/manifest.json" --phase export --accept-provisional
```

如果裁剪实验历史目录里存在多个 `clip_comparison_report.json`，必须显式添加 `--clip-report "<ROI_RUN>/cleanup/geometry/direct_r3_clip/<实际签名>/clip_comparison_report.json"`，防止错误选择。**本次会继续使用已有 Direct R3 实验结果，绝不强制重新裁剪或运行局部 Poisson。**

## 输出与失败保护

- 封底前：`<ROI_RUN>/cleanup/geometry/direct_r3_sdf/<signature>/preseal_diagnostics.json`
- 封底成功：`sealed_poisson.ply`、`sealing_report.json`
- 失败时：`failed_sealing_report.json`，状态 `BLOCKED_PRESEAL` / `BLOCKED_SEALING`；不删除 Gaussian
- SDF 无导出诊断：`<ROI_RUN>/cleanup/preview/direct_r3_<signature>/cleanup_report.json`
- 正式清理候选：`<ROI_RUN>/cleanup/final/gaussian_roi.ply`、`cleanup/metadata/manifest.json`
- 原始 Stage1 `final/gaussian_roi.ply`、第一次全场景 Poisson、R3 V3 SfM、历史 `poisson_roi` 与 `trusted_roi_poisson` 成果均保持原样

**科学限制：** Direct R3 的三角形裁剪仍使用共享边分段线性插值，不是非线性 R3 场的数学精确求交；SDF 是基于封底实体对 Gaussian *中心* 的分类，不代表完整椭球体积分。即使水密、相机检查通过，候选结果仍需视觉验证，标记 `PROVISIONAL_REQUIRES_VISUAL_VERIFICATION`，不能宣称“真实干净几何”已被独立认证。真实送子观音数据尚需用户 WSL 运行验收。
