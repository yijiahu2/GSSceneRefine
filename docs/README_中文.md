# GSSceneRefine 阶段二三项过时测试修复（2026-11-12）

这是测试代码更新工具，不更改算法。上次 `git apply` 失败的根源是补丁对旧文件上下文敏感。本版使用 Python AST 解析顶层测试函数，按函数名定位，替换三个测试函数，保留其他文件及测试不变。

修复范围：

1. `test_auto_seed_requires_modal_top_component`：取消旧的 dominant_top_region 断言，改为核查单外部种子的实际字段、坐标及一致性。
2. `test_reuses_source_matched_barrier_audit`：允许缓存已经完成的区域连通图，不再以顶部—底部连通判定拒绝；仍检查采样窗口截断及源Poisson散列。
3. `test_geometry_preparation_reuses_validated_existing_records`：明确验证旧 clip-before-main preseal 拒绝，新 original-main-first 预审可复用。

安装并运行：

```bash
cd /home/xth/GSSceneRefine
conda activate gsrefine
unzip -o /mnt/f/Edge/GSSceneRefine_stage2_tests_astfix_20261112.zip -d /home/xth/GSSceneRefine
python tools/fix_stage2_tests_ast.py --check
python tools/fix_stage2_tests_ast.py
python -m pytest -q tests/test_stage2_patch_source_contract.py tests/test_barrier_separation_gate_v2.py tests/test_stage2_roi_barrier_parity.py tests/test_stage2_onepass.py
```

第一次验证不会修改任何文件。正式执行前会将原始测试文件备份至 `~/GSSceneRefine_test_backup_20261112/<UTC日期时间>/`。重复运行安全。更新脚本会先验证所有文件的函数存在且目标结果可解析，之后才写入。

**注意**：当前已在本会话中的静态样本上验证 AST 更新；尚未在用户 WSL 里运行真实 pytest。通过测试后再执行真实 ROI 实验。
