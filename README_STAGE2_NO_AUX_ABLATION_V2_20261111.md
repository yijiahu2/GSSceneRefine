# GSSceneRefine Stage 2：无辅助Poisson保护消融（V2）

## 目标

**只关闭辅助Poisson距离保护，不改变任何其他步骤**，与已完成的 `stage2_final_v1` 基线做单因素消融。代码不限制场景名称、Gaussian方法、ROI编号、辅助分量个数或阈值；只通过已完成报告、哈希和逐Gaussian掩码识别来源。

本包包含：

- `gsrefine/cleanup/ablate_no_aux.py`：**推荐，最快**。从已有完整基线结果直接导出“无辅助保护”的完整3DGS场景。无需重新Poisson、SfM、空间隔断、射线或距离计算。
- `gsrefine/cleanup/stage2_onepass.py`：现有总控原位更新，增加通用 `--no-aux-protection` 选项。未提供该选项时，原有保护逻辑不变。
- `tests/test_stage2_no_aux_ablation.py`：单因素掩码与安全导出回归测试。

不允许直接删除 `aux_protected_gaussians.ply` 文件来完成消融：它只是预览文件，真正需要移除的是原模型中对应的Gaussian。最终重导出必须从**第一阶段不可变 `final/gaussian_roi.ply`** 加载，保留所有颜色、SH、opacity、scale及rotation字段，生成可在SuperSplat查看的完整 `gaussian_roi_clean.ply`。

## 安装

```bash
cd /home/xth/GSSceneRefine
conda activate gsrefine
unzip -o /mnt/f/Edge/GSSceneRefine_stage2_no_aux_ablation_V2_20261111.zip -d /home/xth/GSSceneRefine
python -m pytest -q tests/test_stage2_no_aux_ablation.py
```

## 巴寨峰：推荐的最快消融（基于本次成功结果）

```bash
ROI_RUN=/mnt/f/danxia_experiment_archive/GSSceneRefine_results/bazhaifeng/methods/absgs/roi_001

python -m gsrefine.cleanup.ablate_no_aux \
  --baseline-report "$ROI_RUN/cleanup/stage2_final_v1/stage2_final_report.json" \
  --output-dir "$ROI_RUN/cleanup/stage2_no_aux_v1" \
  --export-diagnostics
```

主结果：

```
roi_001/cleanup/stage2_no_aux_v1/
├── gaussian_roi_clean.ply                    # 完整可视化3DGS ROI，高斯属性逐行保留
├── stage2_final_report.json                  # 单因素消融来源与对比报告
└── diagnostics/
    ├── final_classification_masks.npz         # 新保留/删除掩码
    ├── no_aux_removed_gaussians.ply          # 取消保护后全部删除候选（可选）
    └── inside_over_threshold.ply             # 超阈值但分类为内部的Gaussian（可选）
```

基线目录 `stage2_final_v1` 完全不修改。输出必须是新的空目录；若之前存在已有内容，程序拒绝覆盖。

## 通用一键流程：从R3输入直接关闭辅助保护

```bash
ROI_RUN=/mnt/f/danxia_experiment_archive/GSSceneRefine_results/bazhaifeng/methods/absgs/roi_001

python -m gsrefine.cleanup.stage2_onepass \
  --manifest "$ROI_RUN/metadata/manifest.json" \
  --preseal-report "$ROI_RUN/cleanup/geometry/direct_r3_sdf/1beda596ee355c0f4b32797d/preseal_diagnostics.json" \
  --parity-dir "$ROI_RUN/cleanup/stage2_final_v1/diagnostics/parity" \
  --accept-auto-exterior \
  --no-aux-protection \
  --output-dir "$ROI_RUN/cleanup/stage2_no_aux_onepass_v1"
```

上面`preseal`路径来自巴寨峰这次成功报告，专为此轮对照显式选定*同一个*主Poisson和距离阈值；新场景必须使用新场景自己的路径，**代码不包含该路径**。若新场景没有可复用分类结果，省略 `--parity-dir`，同一套总控会按流程计算。

**注意**：已有的 `--no-aux-preview` 仅控制是否导出辅助预览PLY，**并不会关闭保护**。本次新增、真正控制消融的是 `--no-aux-protection`。

## 数学定义

令当前R3 ROI下的Gaussian集合为 \(\mathcal{G}\)，通过原始最大主体Poisson得到的空间标签为 \(C_i\)，高斯中心到R3可信主体曲面的距离为 \(d_{i,\mathrm{main}}\)。阈值 \(\delta=P_{95}(E_{\rm SfM})+\overline{3\sigma_{\max}}_{P10-P90}\)。

- **现有有辅助保护基线**：\(R_i^{\mathrm{aux}}=\mathbf1(d_{i,\mathrm{main}}>\delta+\varepsilon)\mathbf1(C_i=\mathrm{Outside})\mathbf1(d_{i,\mathrm{aux}}>\delta)\)
- **无辅助保护消融**：\(R_i^{\mathrm{noaux}}=\mathbf1(d_{i,\mathrm{main}}>\delta+\varepsilon)\mathbf1(C_i=\mathrm{Outside})\)

因此 \(R_i^{\mathrm{noaux}}-R_i^{\mathrm{aux}}=1\) 仅可能对应“基线被辅助曲面保护”的Gaussian。

### 本次巴寨峰报告的预期数量

| 统计 | 有辅助保护基线 | 无辅助消融 |
|---|---:|---:|
| 最终ROI Gaussian | 2,111,662 | 2,111,662 |
| 适应性δ | 0.0825964742 | 同一值 |
| 超阈值 | 113,216 | 113,216 |
| 外部候选 | 70,592 | 70,592 |
| 受辅助保护 | 3,777 | 0 |
| 最终删除 | 66,815 | **70,592** |
| 最终保留 | 2,044,847 | **2,041,070** |

**上述无辅助数字是依据基线掩码的理论精确对照预期，真正导出的PLY需在用户WSL环境里运行验证。**

## 后续异常归因

如果无辅助保护版本中原右侧发亮伪影明显消失，说明辅助曲面可能保护了部分假结构；但对于左上较淡的雾状异常，如果依然存在，应继续检查：

1. `inside_over_threshold.ply`：是否有空间分类为内部、但视觉位于场景外的异常；若有，检查ROI外部种子与局部短射线结果。
2. 距离未超阈值者（本次1,998,446个中大部分是真实结构）：由于第一步距离判定已自动保留，不能只靠“空间外部”删除，需要单独研究主可信Poisson的完整性、局部距离误差和Gaussian异常特征。
3. 与辅助保护无关的其他真实植被，取消保护后也可能再次误删；因此**实验先对照可视化，不把无辅助版本自动替代正式基线**。

后续优先优化原始可信曲面完整性，特别是独立树冠、陡壁附着植被及孤立结构；辅助曲面是否可信可由SfM多视图支持、局部连通和真实观测进一步验证，但本次消融不引入这些额外规则，以保证单因素比较成立。
