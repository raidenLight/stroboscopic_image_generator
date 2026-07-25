# stroboscopic_image_generator

用 SAM2 跟踪视频中的运动物体，输出频闪合成图像。

## Demo

<video src="https://github.com/user-attachments/assets/7e9e4808-0e15-4ef1-9104-1e2c4e437b16"></video>

## 环境要求

- `uv` 包管理器
- NVIDIA GPU（推荐，支持 CUDA；CPU 也可运行但较慢）

安装 `uv`：https://docs.astral.sh/uv/getting-started/installation/

## 快速开始

```bash
# 1. 安装依赖
uv sync

# 2. 启动
uv run python scripts/stroboscopic_gui.py --video <你的视频路径>
```

首次运行自动下载 SAM2 模型（约 352 MB）。

输出图像自动保存到项目 `results/` 目录，以原视频文件名命名（重名自动追加序号）。

## 使用说明

### 工作流程

1. **点击视频添加跟踪点** — 首次自动创建物体
2. **保存并预览 (P)** — 预览当前帧 mask，物体自动保存
3. **继续点击** — 自动创建下一个物体
4. **开始跟踪 (Enter)** — SAM2 增量追踪所有帧
5. **标记合成帧 (K/I/R)** — 手动/间隔/范围
6. **调整 alpha** — 渐变 + 背景 + 逐帧
7. **保存 (S / 💾按钮)** — 输出合成图到 `results/`

### 启动参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--video` | `video/exp1_dwvp.MP4` | 输入视频路径 |
| `--process-fps` | `10` | 处理帧率，降低可减少内存 |
| `--alpha` | `0.80` | 合成透明度 (0.0~1.0) |
| `--mask-threshold` | `0.5` | mask 阈值，越高越紧 |
| `--dilate-kernel` | `5` | mask 膨胀核 (0=不膨胀) |
| `--min-area` | `300` | 最小连通区域面积 |
| `--max-dim` | `1280` | 视频长边最大尺寸，`-1` 不缩放 |
| `--device` | `auto` | `auto`/`cuda`/`cpu`/`mps` |
| `--hf-model-id` | `facebook/sam2.1-hiera-small` | HuggingFace 模型 |
| `--offload-video-to-cpu` | `False` | 视频帧放 CPU 内存 |
| `--offload-state-to-cpu` | `False` | 预测器状态放 CPU 内存 |
| `--bg-align` | `True` | 背景帧间 ECC 对齐（去抖动） |

### 键盘快捷键

| 按键 | 功能 |
|------|------|
| 鼠标左键（视频） | 添加跟踪点 |
| 鼠标点击/拖拽（时间线） | 帧导航 |
| `←` `→` / `Ctrl+←→` | 逐帧 / 跳标记帧 |
| `P` | 保存并预览 |
| `Enter` | 开始跟踪 |
| `K` / `I` / `R` | 标记帧 / 间隔选取 / 范围选取 |
| `V` | 切换视图 (Mask/合成/原图) |
| `S` | 保存合成图 |
| `1`~`9` / `Tab` | 切换活跃物体 |
| `Backspace` | 删除最后一点 |

### 控制面板

| 区域 | 内容 |
|------|------|
| 状态栏 | 当前状态、帧号、活跃物体 |
| 物体 | 物体按钮 + 删除（彩色标识） |
| 操作 | 保存并预览 / 跟踪(双向/单向切换) / 选点模式 |
| 视图 | Mask/合成/原图切换、mask阈值、💾保存 |
| 帧选取 | K标记、R范围、清除、间隔选取 |
| Alpha | 首/末帧渐变、背景alpha、清除逐帧 |
| 合成帧表格 | 行选框(排除帧)、帧/时间、逐物体☑/☐、☑BG、alpha、删除 |
| 日志 | 操作日志、错误提示 |
| 快捷键栏 | 常用快捷键速查 + 重置按钮 |

所有按钮和键盘**双向同步**。

### 多帧背景融合

勾选多帧的 `☑BG` 列，自动融合为干净背景：

1. 标记若干帧 → 在帧列表中勾选它们的 `BG` 列
2. 各帧的非物体区域互补，逐像素择优填充
3. 相机轻微抖动时自动 ECC 对齐（`--no-bg-align` 可关闭）
4. 背景融合结果缓存，仅 BG 勾选变化时重算

### 内存管理

- `--process-fps 3` 可大幅降低内存（减少处理帧数）
- `--max-dim 640` 降低帧缓存和 mask 存储占用的内存
- `--offload-video-to-cpu` 将视频帧放 CPU（省 GPU 显存）
- 跟踪完成后自动释放 memory bank 回收显存
- 预览/跟踪 OOM 时降低参数重试

## 模型

默认 HuggingFace `facebook/sam2.1-hiera-small`。本地模型：

```bash
uv run python scripts/stroboscopic_gui.py --video <视频> \
  --model-cfg configs/sam2.1/sam2.1_hiera_s.yaml \
  --checkpoint /path/to/sam2.1_hiera_small.pt
```
