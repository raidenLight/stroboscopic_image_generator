"""gui_types.py — 数据类型、枚举、常量定义。

独立模块，不依赖 tkinter / OpenCV / SAM2。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto


# ---------------------------------------------------------------------------
# GUI 状态枚举（v2：简化为仅两种状态）
# ---------------------------------------------------------------------------
class GUIState(Enum):
    EDIT = auto()       # 统一编辑态：加点、删物体、标记帧、调可见性、预览、保存
    TRACKING = auto()   # 模态跟踪态：进度条 + 可中止


# ---------------------------------------------------------------------------
# 跟踪物体数据类
# ---------------------------------------------------------------------------
@dataclass
class TrackObject:
    """SAM2 跟踪的一个物体。

    Attributes:
        obj_id: SAM2 内部 ID（从 1 开始）。
        color: BGR 颜色元组。
        color_hex: 颜色 hex 字符串（如 "#FF0000"）。
        name: 显示名称（如 "Obj1"）。
        seed_frame: 第一个跟踪点所在帧号。
        points: 跟踪点列表 [(x, y), ...]。
        vis_start: 合成图可见起始帧（None = 从第 0 帧开始）。
        vis_end: 合成图可见结束帧（None = 到最后一帧）。
        _dirty: 是否需要（重新）跟踪。新建/加了新点为 True，跟踪完成为 False。
    """
    obj_id: int
    color: tuple[int, int, int]
    color_hex: str
    name: str
    seed_frame: int = 0
    points: list[tuple[int, int]] = field(default_factory=list)
    vis_start: int | None = None
    vis_end: int | None = None
    _dirty: bool = True


# ---------------------------------------------------------------------------
# 物体颜色（12 色，支持更多物体）
# ---------------------------------------------------------------------------
OBJECT_COLORS: list[tuple[int, int, int]] = [
    (0, 0, 255),       # red
    (255, 0, 0),       # blue
    (0, 255, 0),       # green
    (0, 255, 255),     # yellow
    (255, 255, 0),     # cyan
    (255, 0, 255),     # magenta
    (0, 165, 255),     # orange
    (255, 255, 255),   # white
    (128, 0, 128),     # purple
    (255, 192, 203),   # pink
    (0, 215, 255),     # gold
    (180, 130, 70),    # sky blue
]

OBJECT_COLOR_HEX: list[str] = [
    "#FF0000", "#0000FF", "#00FF00", "#FFFF00",
    "#00FFFF", "#FF00FF", "#FFA500", "#FFFFFF",
    "#800080", "#FFC0CB", "#00D7FF", "#B48246",
]


def get_object_color(obj_id: int) -> tuple[tuple[int, int, int], str]:
    """根据 obj_id 返回 (BGR颜色, hex颜色)。超 12 个时循环复用。"""
    idx = (obj_id - 1) % len(OBJECT_COLORS)
    return OBJECT_COLORS[idx], OBJECT_COLOR_HEX[idx]


# ---------------------------------------------------------------------------
# 窗口 & 控件常量
# ---------------------------------------------------------------------------
WINDOW_NAME = "Stroboscopic Generator (SAM2)"
TRACKBAR_FRAME = "Frame"
TRACKBAR_ALPHA = "Alpha"
TIMELINE_H = 60   # v2: 从 40 增加到 60
