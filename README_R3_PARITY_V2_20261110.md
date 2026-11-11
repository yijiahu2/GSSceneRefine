# GSSceneRefine R3 Barrier Parity V2 — 原位更新说明

## 本版修复的崩溃

V1 的 `_navigation_identity()` **定义要求5个位置参数，但调用只有4个**，所以在输出 `Audit accepted for sampled navigation` 后立即抛出 `TypeError`。V2 删除了多余且未使用的参数，并以可重复的来源哈希为索引校验依据。

无需重新计算原始ROI–Poisson交线和灌水报告。继续使用已完成的 `final_r3_global_flood_v1` 诊断目录即可。

## 空间查询完备性改进

- 三阶段扩展：先从附近最多8个已标记导航节点出发进行短射线精确Poisson求交；如果无法取得足够的可靠路径，再扩展至约24/96个节点（默认）；搜索半径由2.5倍体素边长逐步扩展至6倍和16倍。
- 真正通过每条短射线的，是**原始全场景Poisson最大主体真实三角面**的有限线段交点数量，不是用距离估算奇偶性，也不把体素格子当成曲面。
- 当路径变长时增加连续R3场的沿线采样，拒绝被检测到穿出ROI的路径（有限采样不是严格解析证明）。
- 需至少2条路径一致；有相反有效判定时不采用多数投票强行归类。
- 内外仍为二分类：`UNKNOWN=-1` 只表示数值/路径查询没有完成，**不是第三种几何空间**。
- `--classification-only` 始终保留诊断报告；正式清理时，任何超阈值Gaussian仍未解析，将写明阻塞原因并**拒绝导出** `gaussian_roi_clean.ply`，防止误删真实主体。
- 新版缓存带V2来源标识，旧V1缓存不再混用。空间导航缓存只依赖最终R3、原始全场景Poisson、审计网格与路径参数，不依赖Gaussian方法专用的局部距离曲面。

## 原位安装（不要新建项目）

```bash
cd /home/xth/GSSceneRefine
conda activate gsrefine
unzip -o /mnt/f/Edge/GSSceneRefine_R3_barrier_parity_SDF_V2_fix_20261110.zip -d /home/xth/GSSceneRefine
python -m pytest -q tests/test_stage2_roi_barrier_parity.py
```

## 送子观音重跑分类（建议用新的输出目录）

```bash
ROI_RUN=/mnt/f/danxia_experiment_archive/GSSceneRefine_results/songziguanyin/methods/absgs/roi_001
PRESEAL="$ROI_RUN/cleanup/geometry/direct_r3_sdf/e79ac66ff1af70f592e214bc/preseal_diagnostics.json"
AUDIT="$ROI_RUN/cleanup/diagnostics/final_r3_global_flood_v1"

python -m gsrefine.cleanup.direct_r3_pipeline \
  --manifest "$ROI_RUN/metadata/manifest.json" \
  --phase open-ab \
  --b-method roi-barrier-parity \
  --preseal-report "$PRESEAL" \
  --barrier-audit-dir "$AUDIT" \
  --classification-only \
  --output-dir "$ROI_RUN/cleanup/preview/roi_parity_classify_v2"
```

没有必要在这一步修改原有 `δ` 公式或任何原始文件。新目录下检查 `roi_parity_report.json` 中的 `short_ray.stages`、`stage_resolved`、`classified_uncertain`，并在CloudCompare中查看 `outside_candidates.ply` 与 `inside_candidates.ply`。

## 正式导出（分类检查无误后）

```bash
python -m gsrefine.cleanup.direct_r3_pipeline \
  --manifest "$ROI_RUN/metadata/manifest.json" \
  --phase open-ab \
  --b-method roi-barrier-parity \
  --preseal-report "$PRESEAL" \
  --barrier-audit-dir "$AUDIT" \
  --index-cache "$ROI_RUN/cleanup/preview/roi_parity_classify_v2/roi_navigation_cache.npz" \
  --confirm-exterior \
  --output-dir "$ROI_RUN/cleanup/preview/roi_parity_clean_v2"
```

只有当所有超阈值候选均完成可靠的Inside/Outside判定时才会写出正式结果；若状态为 `BLOCKED_INCOMPLETE_PARITY_CLASSIFICATION`，请查看同目录 `roi_parity_report.json` 与诊断PLY。不要以删除数量为目标强行把冲突点标为Outside。

## 兼容与验证范围

本包是 **V1上的小规模原位升级**，仅更新 `gsrefine/cleanup/roi_barrier_parity.py` 与该模块测试，沿用V1已有的 `direct_r3_pipeline.py` 和已安装的 `tools/audit_final_r3_global_poisson_barrier.py`。已通过本地14项含合成射线、闭合交线和回归测试；由于当前执行环境没有用户完整的送子观音Poisson/PLY和Open3D，**尚未完成真实场景端到端验证**。

该方法通过实际原始Poisson三角面测试穿越，不等于已数学严格证明开放曲面和隐式ROI之间完全水密。通过离散网格缓存不会自动获得100%可判定性；V2以自适应搜索改善覆盖率，并用正式输出阻塞来保证无法判断不会转化成误删。
