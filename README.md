# stroboscopic_image_generator

用 SAM2 跟踪视频中的运动物体，输出频闪合成图像（stroboscopic composite image）。

## Demo

<video src="https://github.com/user-attachments/assets/7e9e4808-0e15-4ef1-9104-1e2c4e437b16"></video>

## 环境要求

- `uv` 包管理器
- NVIDIA GPU（推荐，支持 CUDA；CPU 也可运行但较慢）

安装 `uv`：https://docs.astral.sh/uv/getting-started/installation/

## 快速开始

```bash
# 1. 安装依赖（自动安装 CUDA 版 PyTorch）
uv sync

# 2. 放置视频到 video/ 目录

# 3. 启动
uv run python scripts/stroboscopic_gui.py --video video/example.MP4
```

首次运行会自动下载 SAM2 模型到 `~/.cache/huggingface/hub/`（约 352 MB），后续启动无需重新下载。

---

## 使用说明

### 工作流程

```
选点标记 ──▶ SAM2 跟踪 ──▶ 选帧 + 调整可见范围 ──▶ 保存合成图
 (多物体)    (前后双向)     (手动/间隔/范围)
```

1. **标记物体**：在视频帧上单击添加跟踪点，按 `N` 或点击 `[+ New Object]` 添加新物体，不同物体可在不同帧标记
2. **开始跟踪**：按 `Enter` 或点击 `[▶ Start Tracking]`，SAM2 同时向前+向后跟踪所有物体
3. **选取合成帧**：按 `K` 手动标记帧、按 `I` 设置间隔自动选取、按 `R` 框选范围
4. **调整可见性**：为每个物体设置可见时间范围（在合成图中该物体出现的时间段）
5. **预览 & 保存**：按 `V` 切换 mask/合成图/原图预览，按 `S` 保存

### 启动选项

```bash
uv run python scripts/stroboscopic_gui.py \
  --video video/example.MP4 \
  --out video/stroboscopic_gui.png \
  --process-fps 15 \
  --alpha 0.6 \
  --max-dim 1280
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--video` | `video/exp1_dwvp.MP4` | 输入视频路径 |
| `--out` | `video/stroboscopic_gui.png` | 输出图像路径 |
| `--process-fps` | 视频原始 fps | 处理帧率（降低可减少显存占用） |
| `--alpha` | `0.60` | 合成透明度 (0.0 ~ 1.0) |
| `--mask-threshold` | `0.0` | mask 阈值，越高边缘越紧 |
| `--dilate-kernel` | `5` | mask 膨胀核大小 (0 = 不膨胀) |
| `--min-area` | `300` | 最小连通区域面积（像素） |
| `--max-dim` | `1280` | 视频长边最大尺寸（降低可大幅减少显存） |
| `--device` | `auto` | 推理设备：`auto` / `cuda` / `cpu` / `mps` |
| `--hf-model-id` | `facebook/sam2.1-hiera-small` | Hugging Face 模型 ID |
| `--model-cfg` | — | 本地 SAM2 配置文件路径 |
| `--checkpoint` | — | 本地 SAM2 权重文件路径 |
| `--vos-optimized` | `False` | 启用 VOS 优化编译 |

### 键盘快捷键

| 按键 | 功能 |
|------|------|
| `←` `→` / 鼠标拖滑块 | 帧导航 |
| 鼠标左键 | 在当前帧为活跃物体添加跟踪点 |
| `1` ~ `9` | 切换活跃物体 |
| `N` | 新建物体 |
| `Backspace` | 删除最后一个点 |
| `R` | 清除当前物体所有点 |
| `Enter` | 开始跟踪 |
| `Esc` | 取消 / 中止跟踪 / 返回 |
| `K` | 标记/取消当前帧用于合成 |
| `I` | 应用间隔自动选取 |
| `Shift+K` | 范围选取模式 |
| `V` | 切换预览视图（mask / 合成图 / 原图） |
| `B` | 设置当前帧为背景 |
| `S` | 保存合成图像 |

### 控制面板

启动时会弹出两个窗口：

- **OpenCV 窗口**：视频画面 + 底部时间线 + Frame/Alpha 滑块
- **tkinter 控制面板**：物体管理、操作按钮、参数滑块、视图切换、快捷键提示

所有按钮操作和键盘快捷键**双向同步**——点击按钮等同于按下对应按键。

### 显存管理

- `--max-dim 1280`：视频长边缩放到 1280 像素以内再送入 SAM2，避免 OOM
- `--process-fps 15`：降低处理帧率可减少加载的帧数
- 长视频或低显存 GPU 建议同时降低这两个参数
- 中止跟踪后视频帧缓存会保留，重新跟踪无需重新加载

---

## 模型选择

默认使用 Hugging Face 模型 `facebook/sam2.1-hiera-small`（自动下载缓存）。

使用本地模型：

```bash
uv run python scripts/stroboscopic_gui.py \
  --video video/example.MP4 \
  --model-cfg configs/sam2.1/sam2.1_hiera_s.yaml \
  --checkpoint /path/to/sam2.1_hiera_small.pt
```

## 注意事项

- 背景图默认为第一帧（可按 `B` 重新指定）
- 长视频或高 FPS 建议先降低 `--process-fps`（如 `15`）或 `--max-dim`（如 `960`）以避免显存不足
- 需要桌面环境（无法在纯 terminal / headless 环境下运行）
- SAM2 模型下载可能需要代理，如遇网络问题请设置 `HF_ENDPOINT` 环境变量
