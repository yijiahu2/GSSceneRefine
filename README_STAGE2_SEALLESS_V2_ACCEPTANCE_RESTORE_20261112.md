# GSSceneRefine 阶段二正式基线：恢复已成功的 V2 空间隔断验收，隔离历史封底接口

**适用**：正式 `gsrefine.cleanup.stage2_onepass` 无封底清理流程；跨场景通用，不使用场景名称、面数或分量数特例。

## 1. 本次改动范围（非常有限）

本包**只替换**：

- `gsrefine/cleanup/roi_barrier_parity.py`：以先前送子观音、巴寨峰成功使用的 `GSSceneRefine.roi_barrier_parity_sdf.v2` 为原始代码，完整保留闭合交线、粗/细灌水验收、XOR图、短射线分类、距离阈值、外部种子选择及掩码逻辑；把旧 `open_ab_main._validated_inputs` 替换为现有 `stage2_geometry_validation.validate_parity_geometry`，把原本放在 Open A+B 的三项距离参数改从相同数值的 `GAUSSIAN_DISTANCE_DEFAULTS` 读取；诊断PLY直接调用 `GaussianPLY.export()`，不再导入历史 `_v2_export`。
- `gsrefine/cleanup/stage2_onepass.py`：只恢复正式接收 **V2** 分类报告，并调整自动审计缓存查找：只复用粗细灌水**均已通过旧标准**且来源一致的报告。其余无封底几何准备、一键流程、辅助保护、数量守恒、SuperSplat导出、`--no-aux-protection` 原样保留。
- `tests/`：添加验收回归，调整旧测试对“失败灌水缓存也可复用”的过时预期。

**不替换、不修改**：阶段一、R3构建、原始Poisson裁剪、主辅分量提取、SfM拟合方法、灌水诊断实现、XOR/有限射线核算法、Gaussian距离公式、辅助Poisson保护、SuperSplat原生PLY导出。历史 `barrier_separation_gate.py` 保持仓库原样，但在正式 Stage 2 执行链中**不再引用**；历史封底代码同样保留作实验档案，不参与正式清理验收。

## 2. 正式验收标准（恢复送子观音、巴寨峰成功基线）

1. 复用来源、SHA256和最终R3函数一致性均通过的完整隔断审计。
2. 原始Poisson最大主体与最终R3的交线构成可验证的闭合离散环，无端点/分叉。
3. `flood.coarse.status` 和 `flood.fine.status` 都必须是 `NO_TOP_TO_BOTTOM_PATH_SEEN_NOT_PROVEN_SEALED`。
4. 两级ROI采样均不得接触诊断窗口边界。
5. 原V2射线奇偶导航与自适应短射线分类不变；奇偶冲突或未解析的候选阻止最终导出。
6. 最终仍按 `主曲面距离 > delta+epsilon AND Outside AND 未被辅助保护` 删除；输出 `gaussian_roi_clean.ply`。

**注意**：这是已成功运行过的保守数值验收规则，不是严格水密性证明。对于此前 `roi_002` 上下灌水两级均检测到贯通的情况，依然会被此标准阻止；本次不能据此宣称解决了 `roi_002` 的验收问题。

## 3. 安装（原位覆盖）

```bash
cd /home/xth/GSSceneRefine
conda activate gsrefine
unzip -o /mnt/f/Edge/GSSceneRefine_stage2_sealless_V2_gate_restore_20261112.zip -d /home/xth/GSSceneRefine
python -m pytest -q tests/test_stage2_baseline_restored.py tests/test_stage2_roi_barrier_parity.py tests/test_stage2_onepass.py tests/test_stage2_sealless_geometry.py
```

## 4. 僧帽峰 EfficientGS、重新绘制的 ROI_003：复用已完成的几何与灌水

此次日志显示 `roi_003` 原始Poisson最大主体的粗、细灌水报告都为 `NO_TOP_TO_BOTTOM_PATH_SEEN_NOT_PROVEN_SEALED`，且V3新验收的失败原因只在于错误要求“跨分辨率连通区域一对一匹配”。**不要重复灌水或修改网格分辨率**。使用当前完整审计成果和已校验的无封底几何：

```bash
ROI_RUN=/mnt/f/danxia_experiment_archive/GSSceneRefine_results/sengmaofeng/methods/EfficientGS/roi_003
PRESEAL="$ROI_RUN/cleanup/geometry/direct_r3_sdf/f92e0e657d27bad24081e7a2/preseal_diagnostics.json"
AUDIT="$ROI_RUN/cleanup/stage2_final_v2/diagnostics/parity/geometry_audit"

python -m gsrefine.cleanup.stage2_onepass \
  --manifest "$ROI_RUN/metadata/manifest.json" \
  --preseal-report "$PRESEAL" \
  --barrier-audit-dir "$AUDIT" \
  --accept-auto-exterior \
  --reclassify \
  --output-dir "$ROI_RUN/cleanup/stage2_final_v3"
```

输出主成果：

`$ROI_RUN/cleanup/stage2_final_v3/gaussian_roi_clean.ply`

这份原生Gaussian PLY保留SH、透明度、尺度和旋转信息，可在SuperSplat打开。

**重要提醒**：`--accept-auto-exterior` 采用旧V2最高R3边壳外部假设，仍须根据场景/相机检查。若已有人工确认的外部空间坐标，可改为 `--exterior-seed X Y Z`。若任何真实几何、安全距离或奇偶一致性检查仍不通过，程序应继续阻止导出；不能保证仅凭恢复旧规则就对每个ROI完成清理。

## 5. 验收与缓存注意

- **不要覆盖**之前的 `stage2_final_v1` 或 `stage2_final_v2` 失败记录；新实验放 `stage2_final_v3`。
- `--reclassify` 强制运行旧V2原版分类算法，避免误用V3区域分离实验的分类报告。
- `--barrier-audit-dir` 指向当前ROI已完成的粗、细灌水结果，减少重复执行和磁盘占用。
- 若不同场景/ROI/模型、Poisson或R3来源不同，不要交叉复用诊断索引或分类掩码。
- 输出 `stage2_final_report.json` 成功状态和完整 `gaussian_roi_clean.ply` 后，才视为本轮正式清理成功。
