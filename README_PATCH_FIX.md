# GSSceneRefine 阶段二更新脚本修复（2026-11-12）

## 修复内容

之前 `tools/apply_stage2_generalization.py` 中 `update_audit()` 使用整段源码的精确文本匹配，匹配字段缩进时误以为 `upper_shell_connected_regions` 等字段的缩进为29个空格，GitHub 2026-11-12的实际源码缩进不同，导致：

`remove top/bottom result fields: expected one source match, found 0; no files changed`

本修复使用结构化正则定位同一组连续字段，并保留严格的唯一匹配检查（必须且只能匹配一次）。不能找到唯一字段块时仍会中止，保证不执行错误替换。其余三项修复目标与原补丁一致：删除旧顶部/底部连通门槛、使用区域分隔V2、原始Poisson先选最大主体再R3裁剪及裁剪后再次选最大主体，小分量仅参与可选保护。

本包 **仅替换更新脚本**，不直接覆盖R3、Poisson、Gaussian文件，也不直接覆盖任何阶段二算法源文件。旧版更新脚本在预审阶段中止，尚未修改算法文件。

## WSL操作

```bash
cd /home/xth/GSSceneRefine
conda activate gsrefine
unzip -o /mnt/f/Edge/GSSceneRefine_stage2_generalization_updater_fix_20261112.zip -d /home/xth/GSSceneRefine
python tools/apply_stage2_generalization.py
python -m pytest -q tests/test_stage2_patch_source_contract.py tests/test_barrier_separation_gate_v2.py tests/test_stage2_roi_barrier_parity.py tests/test_stage2_onepass.py
```

不需要重新解压最初的完整更新包。首次运行之前如果最初的更新包未解压，则先解压最初完整包再执行上述两步。

## 验证范围

已在当前环境对实际报错的字段块分别使用28、29、32空格缩进构造输入，全部完成`update_audit`转换，且对缺失目标字段的负例继续拒绝修改。已通过Python编译检查。尚未在用户本机仓库运行全部文件更新及真实场景计算，不能据此认定阶段二模型清理已经通过。
