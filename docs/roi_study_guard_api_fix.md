# roi-study 与 Gaussian guard 接口一致性修复

## 原因

`gsrefine/roi/study.py` 调用 `add_guard_gaussians(..., budget_policy='review')`；
此前旧版 `gsrefine/roi/core.py` 的该函数只有五个参数，因而触发
`TypeError: add_guard_gaussians() got an unexpected keyword argument 'budget_policy'`。
这属于跨更新包的**代码版本不一致**，不是输入数据、用户ROI或Python依赖问题。

## 本次变更

仅将 `gsrefine/roi/core.py` 同步为先前“自适应预算更新包”中已发布、与 `roi-study` 兼容的版本，并新增小型接口回归测试。
`review` 策略在超出全局保护比例时保留供验证的候选掩码，同时设置 `review_required`；`reject` 才会清空超预算保护结果。

本修复包不会改变配置、ROI JSON、Gaussian模型或Poisson文件。覆盖前建议对 `core.py` 做单独备份，尤其在此前手工修改过该文件的情况下。

## WSL安装

```bash
conda activate gsrefine
cd /home/xth/GSSceneRefine
mkdir -p runs/code_backups
cp -a gsrefine/roi/core.py runs/code_backups/core_before_guard_api_fix.py
unzip -o /mnt/f/Edge/GSSceneRefine_roi_study_guard_api_fix.zip -d /home/xth/
python -m pytest -q tests/test_guard_budget_api.py tests/test_roi_boundary_study.py tests/test_roi_extension_budget.py
python -m pytest -q tests
```

## 重试

```bash
python run.py roi-study \
  --config configs/songziguanyin.yaml \
  --roi runs/roi_init/songziguanyin_absgs_roi_002.json \
  --output-dir runs/roi_tests/songziguanyin_roi002_boundary
```

本修复只解决 `budget_policy` 接口兼容性，不表示真实送子观音候选分类或保护效果已经得到验证。
