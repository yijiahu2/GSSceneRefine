# GSSceneRefine v8.3：from-roi 独立重复实验入口

## 定位

这是 **已安装 GSSceneRefine v8（并修复 v8.1/v8.2）的增量更新包**，不是一个独立完整仓库。其唯一生产代码改动为 `gsrefine/roi_init/formal_pipeline.py`。不覆盖 `run.py`、`workflow.py`、`stage2_onepass.py`、`local_region_pipeline.py`、几何/高斯算法及既有阶段一成果。

`pipeline full` 与 `pipeline stage2` 的入口参数、调用链保持不变。三种模式均在启动前静态检查先前暴露的v8.1局部`measured`遮蔽错误、v8.2错误计时上下文；如尚未修复将直接提示对应热修复，避免浪费全量计算时间。

`pipeline from-roi` 从现有已确认的手绘 `roi_NNN.json` 启动：

1. 检查原始 ROI 的登记记录、双正交视图哈希、源模型及配准身份；有既有阶段一输出时核对完整 ROI SHA256。
2. 根据指定 `--config` 中的 SfM 净化、Poisson 参数和原始 `points3D.bin` 的 SHA256 验证共享模块一缓存；**不调用模块一重建**。默认优先搜索当前 `--output-dir/<scene>/geometry`，其次搜索原始 ROI 所在场景的 `geometry`。
3. 以原子方式预留唯一实验根目录 `repeat_0001`、`repeat_0002`……。已有成功与失败目录都不会重用。
4. 在新实验目录写入独立 `config/effective_scene.yaml`；只引用共享的、已验证的SfM/Poisson，原始 ROI JSON 不复制、不更改。
5. 直接调用现有 `gsrefine.roi_init.workflow.run_once()`，完整重新执行模块三R3恢复与模块四Gaussian ROI导出。
6. 校验新生成的 `metadata/manifest.json` 指向原始ROI且来源SHA256一致；随后使用该 `manifest` 启动正式阶段二 `full-pipeline --resume-manifest`。
7. 记录 `experiment.json`、阶段一/二运行状态、输入来源和时间统计；失败保留原目录。已有 `roi_003` 等历史输出保持不变。

算法锁定：双正交 ROI、局部主体恢复、R3自适应安全距离、SfM 1×/2×、原始Poisson全局判侧+局部测距、SfM P95+Gaussian尺度、短射线回退；没有改动数学规则。辅助保护仍依已有v8.2参数决定。

## 安装

```bash
cd /home/xth/GSSceneRefine
conda activate gsrefine

# 可选：提前备份当前入口源码
cp gsrefine/roi_init/formal_pipeline.py \
   gsrefine/roi_init/formal_pipeline_before_v8_3.py.bak

unzip -o /mnt/f/Edge/GSSceneRefine_from_roi_repeat_v8_3.zip \
  -d /home/xth/GSSceneRefine

python -m py_compile gsrefine/roi_init/formal_pipeline.py
python tools/gsrefine_v8/test_from_roi_repeat.py
python run.py pipeline from-roi --help
```

此 ZIP 按源码相对路径组织，解压到项目根即可覆盖入口；无需运行额外安装或补丁脚本。

## LFLS 示例（同一手绘 ROI，重复模块三+四及阶段二）

```bash
DATA=/mnt/f/danxia_experiment_archive/GSSceneRefine_benchmark
SFM="$DATA/GauUscene/LFLS/sparse/0"
PLY="$DATA/GSPL_training_outputs/GauUscene/gauu_lfls/iter30000_gsplat_accel_4x/point_cloud/iteration_30000/point_cloud.ply"
OUT="$DATA/GSSceneRefine_results"
ROI_JSON="$OUT/lfls/roi/roi_003.json"

python run.py pipeline from-roi \
  --config configs/default.yaml \
  --scene lfls --method gsplat_accel \
  --sfm-dir "$SFM" --gaussian-ply "$PLY" \
  --output-dir "$OUT" --roi "$ROI_JSON" \
  --accept-auto-exterior
```

重复调用原命令时使用不同 `repeat_NNNN`；不再报“Stage-1 manifest already exists”。实验路径：

```text
<OUT>/experiments/lfls/gsplat_accel/roi_003/repeat_0001/
├── experiment.json                  # 唯一实验登记、ROI SHA256、来源与耗时
├── config/effective_scene.yaml      # 此次实验专属有效配置
└── lfls/methods/gsplat_accel/roi_003/
    ├── metadata/manifest.json       # 此次阶段一的唯一权威manifest
    ├── metadata/stage1_timing.json
    ├── final/gaussian_roi.ply
    └── cleanup/stage2_original_region_v8/
        ├── stage2_final_report.json
        ├── gaussian_roi_clean.ply    # 仅在阶段二完整成功时生成
        └── diagnostics/
```

> 若阶段二返回 `PARTIAL_VERIFIED_ROI_CLEANUP` 或 `NO_VERIFIED_CLEANUP_SCOPE`，程序不会谎称已经完成全域清理，实验登记记录原样终态。辅助保护消融可以通过 `--no-aux-protection` 再运行一个独立实验。

## 重复运行之后继续阶段二

如果新实验阶段一成功，但阶段二失败，直接使用这一实验自己的 manifest 重试阶段二；不重新运行模块三、四：

```bash
EXP="$OUT/experiments/lfls/gsplat_accel/roi_003/repeat_0001/lfls/methods/gsplat_accel/roi_003"
python run.py pipeline stage2 --manifest "$EXP/metadata/manifest.json" --accept-auto-exterior
```

## 禁止的操作与验收条件

- 不要修改、重新保存或复制替换原始 `roi_003.json`。如视图哈希或既有阶段一 SHA256 不匹配，程序拒绝执行。
- `--config` 的模块一参数必须与原有缓存一致；如签名不符，程序拒绝自动重建Poisson，避免 `from-roi` 实际从模块一重算。
- 如需对另一3DGS方法复用同一ROI，可显式加 `--reuse-scene-roi`，仍须通过原有的帧、up向量及模型作用域校验。
- 可选的体积较大的阶段一/阶段二诊断PLY仅按原有 `--keep-diagnostics`/`--keep-classification-previews` 控制；阶段一和阶段二的核心审计文件保持原有v8行为。

11项离线模拟测试已覆盖：连续两次独立运行、失败后不重用、原始ROI/Poisson/SfM不改变、非法编辑拦截、历史SHA锚点、模块一缓存失效拦截、不同输出根复用、并发独立目录、`full`/`stage2` 路由保持不变。**未在真实 LFLS 数据上执行**，需要你在WSL中验证其完整运行、结果一致性与性能。
