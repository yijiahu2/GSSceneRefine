# GSSceneRefine

GSSceneRefine 是面向 3D Gaussian Splatting (3DGS) 场景的交互式三维 ROI 提取与自适应结构保护项目。所有模块统一位于本仓库，使用同一个 `gsrefine` Conda 环境。

## 阶段一：四模块连续工作流

1. **模块一｜模型和几何数据加载**：从场景 YAML 加载原始坐标系下的 Gaussian PLY、COLMAP方向元数据，以及可选的原始 Poisson 曲面；Windows Edge 通过本机WebGPU浏览高斯场景。
2. **模块二｜交互粗选ROI**：在俯视和立面绘制正交多边形，核查初始Gaussian中心交集，点击“确认ROI”后保存有唯一编号和姿态校验信息的JSON。
3. **模块三｜Poisson约束恢复及R3保护**：分析ROI外正常外围延伸，识别疑似误截断主体；先进行几何恢复，再以K=128、两端各5%截尾的局部Gaussian长轴均值构建隐式自适应安全带。原始ROI全部保留。
4. **模块四｜Gaussian PLY导出**：直接复用输入PLY顶点属性及顺序，导出最终R3保留场景和校验报告。此阶段**不做内部异常Gaussian清理**。

### 启动（Windows 10 + WSL2 Ubuntu 22.04）

```bash
conda activate gsrefine
cd /home/xth/GSSceneRefine
python run.py roi-init --config configs/songziguanyin.yaml
```

在Windows Edge打开控制台显示的 `http://127.0.0.1:8765`，完成俯视与立面ROI粗选，点击 **确认ROI并自动生成最终PLY**。无需再手动启动 `roi-study` 或 `roi-finalize`。计算在WSL独立进程运行，页面显示正在执行/成功/失败状态和最终PLY路径。

默认配置 `workflow.auto_after_confirm: true`。每次确认ROI后，模型存入：

```text
runs/roi_tests/<已确认ROI文件名（不含.json）>/gaussians_final_R3.ply
```

同一目录下保存 `workflow.log`、`workflow_result.json`、`study_report.json`、`core_mask_R3.npy`、`safety_added_mask_R3.npy` 和最终顶点索引等。**输入 Gaussian PLY、Poisson 曲面以及原始ROI JSON 均不会被覆盖。**

### 单独复跑或做消融

```bash
# 已确认ROI的自动后续阶段（无需打开浏览器）
python run.py roi-auto --config configs/songziguanyin.yaml \
  --roi runs/roi_init/songziguanyin_absgs_roi_002.json

# 科研消融：独立运行R0-R3；默认生成最终R3 PLY
python run.py roi-study --config configs/songziguanyin.yaml \
  --roi runs/roi_init/songziguanyin_absgs_roi_002.json \
  --output-dir runs/roi_tests/songziguanyin_roi002_R0_R3

# 独立按已保存掩码导出
python run.py roi-finalize \
  --study-dir runs/roi_tests/songziguanyin_roi002_R0_R3 \
  --config configs/songziguanyin.yaml --method R3
```

如要只手动保存多边形（不自动处理），可在配置中设 `workflow.auto_after_confirm: false`。为了确保ROI和Gaussian数据一致，界面显示的原始PLY必须与模块三读取的PLY相同，并共享COLMAP原始世界坐标系。

**注意：** 使用已被Task1 AABB裁剪的Gaussian输入时，算法只能恢复该输入中现存的Gaussian，不能找回此前已删除的Gaussian。

## 环境、结构与测试

所有代码在固定项目 `/home/xth/GSSceneRefine` 中原位维护，不为算法调整创建新的仓库或Conda环境。依赖参见 `environment.yml`，Windows原生PLY服务与交互前端参见 `ui/` 和 `gsrefine/roi_init/windows.py`；首次安装时按 `bootstrap.sh` / 项目文档准备环境、运行 `cd ui && npm install`。

```bash
python -m pytest -q tests
node --test ui/tests/*.test.mjs
```

更多细节参见 `docs/architecture.md`。阶段二的内部异常Gaussian清理仍是独立后续研究，不包含在当前自动导出步骤中。
