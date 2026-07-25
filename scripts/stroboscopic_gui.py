#!/usr/bin/env python3
"""
频闪图像生成器 — SAM2 多物体跟踪交互式 GUI（v2）

模块化架构：
  - gui_types.py   数据类型、枚举、颜色常量
  - gui_panel.py   tkinter 控制面板
  - gui_app.py     核心应用逻辑（跟踪、渲染、合成）
  - stroboscopic_gui.py  入口 + CLI 参数解析 + 视频预处理

双窗口架构：
  - OpenCV 窗口：视频帧 + 时间线 + mask/选点叠加
  - tkinter 面板：原生按钮、滑块（Python 内置，零额外依赖）

工作流程：
  1. 编辑：在任意帧标记物体 → 加点 → 预览 → 增量跟踪
  2. 跟踪：SAM2 模态追踪（仅脏物体）
  3. 编辑：标记合成帧 → 调整可见性 → 预览 → 保存
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

# 禁用 tqdm 后台监控线程，避免与 cv2 的 GIL 释放冲突导致 PyEval_RestoreThread 崩溃
import tqdm
tqdm.tqdm.monitor_interval = 0

# ── 复用原始脚本的 SAM2 辅助函数 ──
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from stroboscopic_image_generator import (  # noqa: E402
    build_predictor,
    get_frame_count,
    maybe_downsample_video,
    resolve_device,
    sam_inference_context,
)

from gui_app import StroboscopicGUI

ROOT_DIR = _THIS_DIR.parent
DEFAULT_VIDEO = ROOT_DIR / "video" / "exp1_dwvp.MP4"
DEFAULT_OUT = ROOT_DIR / "video" / "stroboscopic_gui.png"


# ===================================================================
# Argument parsing
# ===================================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SAM2 多物体跟踪频闪图像生成器 — 交互式 GUI v2",
    )
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO, help="输入视频路径")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="输出图像路径")
    parser.add_argument("--process-fps", type=float, default=10,
                        help="处理帧率，低于源帧率可减少内存占用")
    parser.add_argument("--alpha", type=float, default=0.80, help="合成透明度 (0.0~1.0)")
    parser.add_argument("--mask-threshold", type=float, default=0.5, help="mask 阈值，越高 mask 越紧")
    parser.add_argument("--dilate-kernel", type=int, default=5, help="mask 膨胀核大小 (0=不膨胀)")
    parser.add_argument("--min-area", type=int, default=300, help="最小连通区域面积（像素）")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu", "mps"), default="auto",
                        help="推理设备")
    parser.add_argument("--hf-model-id", type=str, default="facebook/sam2.1-hiera-small",
                        help="HuggingFace 模型 ID")
    parser.add_argument("--model-cfg", type=str, default=None, help="SAM2 配置文件路径")
    parser.add_argument("--checkpoint", type=Path, default=None, help="SAM2 权重文件路径")
    parser.add_argument("--vos-optimized", action="store_true", help="启用 VOS 优化编译")
    parser.add_argument("--offload-video-to-cpu", action=argparse.BooleanOptionalAction, default=False,
                        help="将视频帧放在 CPU 内存")
    parser.add_argument("--offload-state-to-cpu", action=argparse.BooleanOptionalAction, default=False,
                        help="将预测器状态放在 CPU 内存")
    parser.add_argument("--max-dim", type=int, default=1280,
                        help="帧最大尺寸（像素）。默认 1280。传 -1 表示不缩放。")

    args = parser.parse_args()
    if not 0.0 <= args.alpha <= 1.0:
        parser.error("--alpha must be between 0.0 and 1.0")
    if args.dilate_kernel < 0:
        parser.error("--dilate-kernel must be >= 0")
    if (args.model_cfg is None) != (args.checkpoint is None):
        parser.error("--model-cfg and --checkpoint must be specified together")
    if args.checkpoint is not None and not args.checkpoint.exists():
        parser.error(f"checkpoint not found: {args.checkpoint}")
    return args


# ===================================================================
# Video resize helper
# ===================================================================
def maybe_resize_video(src_video: Path, max_dim: int, tmp_dir: Path) -> tuple[Path, int, int]:
    """缩放视频使长边 ≤ max_dim，保持宽高比。返回 (新路径, w, h)。

    max_dim ≤ 0 时不缩放，直接返回原视频路径和尺寸。
    """
    cap = cv2.VideoCapture(str(src_video))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {src_video}")

    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    long_side = max(src_w, src_h)

    if max_dim <= 0 or long_side <= max_dim:
        cap.release()
        return src_video, src_w, src_h

    scale = max_dim / long_side
    new_w = int(src_w * scale)
    new_h = int(src_h * scale)
    new_w -= new_w % 2
    new_h -= new_h % 2

    out_path = tmp_dir / f"{src_video.stem}_resized_{new_w}x{new_h}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, src_fps, (new_w, new_h))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"无法打开写入器: {out_path}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        writer.write(resized)

    cap.release()
    writer.release()
    return out_path, new_w, new_h


# ===================================================================
# main
# ===================================================================
def main() -> None:
    args = parse_args()
    if not args.video.exists():
        raise RuntimeError(f"视频不存在: {args.video}")

    device_name = resolve_device(args.device)
    print(f"加载视频: {args.video}")
    print(f"推理设备: {device_name}")
    print("注意: SAM2 模型 100% 本地 GPU 运行。")
    print("首次运行将从 HuggingFace 下载约 150MB 模型到本地缓存。")
    print()
    print("交互指引:")
    print("  1. 点击视频 → 自动创建物体并添加跟踪点")
    print("  2. P 保存并预览 | Enter 开始跟踪 | 再次点击自动创建下一个物体")
    print("  3. K 标记帧 | I 间隔选取 | R 范围选取 | B 设背景")
    print("  4. V 切换视图 | S 保存 | ← → 导航 | Tab 切换物体")
    print()

    with tempfile.TemporaryDirectory(prefix="sam2_gui_") as tmp_dir:
        # Step 1: FPS 降采样
        processing_video, source_fps, was_downsampled = maybe_downsample_video(
            src_video=args.video, target_fps=args.process_fps, tmp_dir=Path(tmp_dir),
        )

        # Step 2: 空间缩放
        processing_video, vid_w, vid_h = maybe_resize_video(
            processing_video, args.max_dim, Path(tmp_dir),
        )

        cap = cv2.VideoCapture(str(processing_video))
        if not cap.isOpened():
            raise RuntimeError(f"无法打开视频: {processing_video}")

        try:
            fps = cap.get(cv2.CAP_PROP_FPS) or source_fps or 30.0
            n_frames = get_frame_count(cap)
            n_frames_mb = n_frames * vid_w * vid_h * 4 / (1024 * 1024)
            print(f"帧数: {n_frames}, 帧率: {fps:.2f}, 分辨率: {vid_w}x{vid_h}")
            print(f"预估帧内存: ~{n_frames_mb:.0f} MB")
            if was_downsampled:
                print(f"帧率降采样: {source_fps:.2f} → {fps:.2f}")

            device = resolve_device(args.device)
            predictor = build_predictor(args, device=device)
            gui = StroboscopicGUI(cap, predictor, processing_video,
                                  n_frames, fps, args, Path(tmp_dir),
                                  source_fps=source_fps)
            with sam_inference_context(device):
                try:
                    gui.run()
                except RuntimeError as e:
                    msg = str(e)
                    if any(kw in msg for kw in ("not enough memory", "OutOfMemory", "CUDA error")):
                        print("\n" + "=" * 60)
                        print("内存不足 — 请尝试以下方案：")
                        print("  1. 降低 --max-dim（如 --max-dim 720 或 640）")
                        print("  2. 降低 --process-fps（如 --process-fps 3）")
                        print("  3. 使用更小模型: --hf-model-id facebook/sam2.1-hiera-tiny")
                        print("  4. 仅 CPU: --device cpu（慢但省显存）")
                        print("=" * 60)
                    raise
        finally:
            cap.release()
            cv2.destroyAllWindows()
    print("完成。")


if __name__ == "__main__":
    main()
