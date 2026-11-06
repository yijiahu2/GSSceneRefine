# GSSceneRefine: WSL终端四模块运行日志

## 统一工作流程

1. `roi-init`启动模块一：原始COLMAP SfM净化与Poisson曲面构建（命中缓存时复用）。
2. 启动模块二：Edge访问 `http://127.0.0.1:8765` 进行双正交多边形交互。首次检查可手工预览。
3. 点击“确认ROI”：服务端先落盘ROI JSON，核验后创建`roi-auto`计算子进程，并向浏览器返回成功响应。随后停止WSL 8765网页服务、终止当前命令启动的Windows模型显示服务，WSL终端接管进度。
4. WSL终端只打印真实计算阶段：模块三Poisson识别、主体恢复、全边界安全带，模块四PLY导出、结果校验。
5. 最终状态只有在`workflow_result.json`和`gaussians_final_R3.ply`验证通过后才报告成功，否则输出失败原因及`workflow.log`路径。
6. 运行命令退出后，可再运行`roi-init`打开新的交互工作区。不需要在浏览器里重新加载计算进度。

## 输入与输出

```bash
conda activate gsrefine
cd /home/xth/GSSceneRefine
python run.py roi-init \
  --config configs/songziguanyin.yaml \
  --scene songziguanyin --method absgs \
  --sfm-dir /你的COLMAP稀疏模型目录 \
  --gaussian-ply /你的完整Gaussian场景.ply \
  --output-dir /你指定的实验输出根目录
```

支持`--colmap-points`与`--colmap-images`替代`--sfm-dir`；各场景方法分开保存结果。不能使用不同坐标系的SfM与Gaussian。

## 恢复独立运行

ROI已确认但后续计算失败时，不必重画ROI，使用：

```bash
python run.py roi-auto --config /对应场景的有效配置.yaml \
  --roi /已经保存的ROI.json \
  --output-dir /希望保存结果的目录
```

保留原有所有研究输出与独立命令。此次不改变R3计算、掩码、PLY格式或已有实验目录。

## 重要说明

- 确认ROI后关闭的是服务端HTTP监听，不会强制关闭用户自己打开的Edge标签页；标签页显示确认结果，用户可随时关闭。
- `workflow.log`、`workflow_progress.json`及最终报告仍保存在各自输出目录中，便于复现和排错。
- 在浏览器完成确认前，不应关闭原来的WSL终端。确认后主进程持续监控工作进度，直到最终成功或失败才返回命令行。
- 浏览器`GET`成功请求不会再刷屏；HTTP错误仍可能打印，供排查。
