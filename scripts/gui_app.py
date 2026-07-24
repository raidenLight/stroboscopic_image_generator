"""gui_app.py — StroboscopicGUI 核心应用逻辑。

状态管理、跟踪、渲染、合成、键盘/鼠标处理。
依赖 gui_types（数据类型）和 gui_panel（控制面板）。
"""

from __future__ import annotations

import contextlib
import tkinter as tk
from typing import TYPE_CHECKING

import cv2
import numpy as np

from gui_types import (
    GUIState,
    TrackObject,
    get_object_color,
    WINDOW_NAME,
    TRACKBAR_FRAME,
    TRACKBAR_ALPHA,
    TIMELINE_H,
)

if TYPE_CHECKING:
    from gui_panel import ControlPanel


# ===================================================================
# Frame read helpers
# ===================================================================
def read_frame_at_fast(cap, frame_idx: int, cache: dict[int, np.ndarray]) -> np.ndarray:
    """读取视频帧，带 LRU 风格缓存（最多 50 帧）。"""
    if frame_idx in cache:
        return cache[frame_idx].copy()

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    if not ret:
        return np.zeros((int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480),
                         int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640), 3), dtype=np.uint8)

    if len(cache) >= 50:
        # 删除最旧的条目
        oldest = next(iter(cache))
        del cache[oldest]
    cache[frame_idx] = frame.copy()
    return frame.copy()


# ===================================================================
# Main GUI class
# ===================================================================
class StroboscopicGUI:
    """交互式频闪合成，SAM2 多物体跟踪。"""

    def __init__(self, cap, predictor, video_path, n_frames, fps, args, tmp_dir, source_fps=None):
        self.cap = cap
        self.video_path = video_path
        self.n_frames = n_frames
        self.fps = fps
        self.source_fps = fps  # 原始视频帧率，用于显示时间戳
        frame = read_frame_at_fast(cap, 0, {})
        if frame.size == 0:
            raise RuntimeError("无法读取第一帧。")
        self.h, self.w = frame.shape[:2]

        self.predictor = predictor
        self.inference_state = None

        # ── 多物体（v2: 无默认物体）──
        self.objects: list[TrackObject] = []
        self.active_obj_idx: int = -1

        # ── 跟踪 ──
        self.masks: dict[int, dict[int, np.ndarray]] = {}  # frame_idx -> {obj_id: mask}
        self._tracked_obj_ids: set[int] = set()
        self._tracking_generator = None
        self._tracking_direction = "forward"
        self._tracking_pass = 0
        self._tracking_total_passes = 0
        self._tracking_frame_count = 0
        self._tracking_total_frames = 0

        # ── 状态 ──
        self.state = GUIState.EDIT
        self.current_frame_idx: int = 0
        self.composite_frames: set[int] = set()
        self.background_frame_idx: int = 0
        self._range_start: int | None = None
        self.viz_mode = "mask"
        self._preview_dirty = True
        self._preview_cache: np.ndarray | None = None
        self.frame_overrides: dict[int, dict[int, bool]] = {}
        self._excluded_frames: set[int] = set()    # 行勾选排除的帧（仍在列表中但灰色）
        self._data_version: int = 0                 # 数据变更版本号，触发面板重建

        # ── Alpha：渐变 + 逐帧覆盖 ──
        self.alpha_start: float = args.alpha
        self.alpha_end: float = args.alpha
        self.per_frame_alpha: dict[int, float] = {}

        # ── 跟踪后隐藏选点 ──
        self._show_points_overlay: bool = True

        # ── 帧缓存 ──
        self._frame_cache: dict[int, np.ndarray] = {}

        # ── 单帧预览 ──
        self._preview_mask: np.ndarray | None = None
        self._preview_mask_obj_id: int | None = None

        # ── 状态消息 ──
        self.status_message = ""
        self.status_timer = 0
        self.status_color = (0, 255, 255)  # 默认黄色

        # ── 引导覆盖层 ──
        self._show_onboarding = True

        # ── trackbar ──
        self._trackbar_locked = False

        self.args = args
        self.tmp_dir = tmp_dir
        self.panel: ControlPanel | None = None

        # ── 中止回滚 snapshot ──
        self._pre_tracking_masks: dict[int, dict[int, np.ndarray]] = {}

        # ── 模态对话框保护锁（tick 期间有 messagebox 弹出时跳过渲染/重建，避免卡死）──
        self._in_modal: bool = False

    # ==================================================================
    # 物体管理
    # ==================================================================
    def _create_object(self, seed_frame: int) -> TrackObject:
        obj_id = max([o.obj_id for o in self.objects], default=0) + 1
        color, color_hex = get_object_color(obj_id)
        obj = TrackObject(
            obj_id=obj_id, color=color, color_hex=color_hex,
            name=f"Obj{obj_id}", seed_frame=seed_frame, _dirty=True,
        )
        self.objects.append(obj)
        self.active_obj_idx = len(self.objects) - 1
        return obj

    def _remove_object(self, obj: TrackObject) -> None:
        """删除物体：清理 objects 列表、masks、frame_overrides、composite_frames 中的关联数据。"""
        oid = obj.obj_id
        # 清理 masks
        for fidx in list(self.masks.keys()):
            self.masks[fidx].pop(oid, None)
            if not self.masks[fidx]:
                del self.masks[fidx]
        # 清理 frame_overrides
        for fidx in list(self.frame_overrides.keys()):
            self.frame_overrides[fidx].pop(oid, None)
            if not self.frame_overrides[fidx]:
                del self.frame_overrides[fidx]
        # 从列表移除
        idx = self.objects.index(obj)
        self.objects.remove(obj)
        # 调整活跃索引
        if not self.objects:
            self.active_obj_idx = -1
        elif self.active_obj_idx >= len(self.objects):
            self.active_obj_idx = len(self.objects) - 1
        self._preview_dirty = True

    def active_object(self) -> TrackObject | None:
        if 0 <= self.active_obj_idx < len(self.objects):
            return self.objects[self.active_obj_idx]
        return None

    # ==================================================================
    # Action dispatch
    # ==================================================================
    def action(self, name: str, *args) -> None:
        """中央 action 分发器。所有 UI 控件（键盘/按钮）路由到此。"""
        if name == "quit":
            self._quit_flag = True

        elif name == "new_object":
            self._create_object(seed_frame=self.current_frame_idx)
            obj = self.active_object()
            self._show_onboarding = False
            self._set_status(f"已创建 {obj.name}（种子帧 {self.current_frame_idx}）", "info")

        elif name == "select_object":
            idx = args[0] if args else 0
            if 0 <= idx < len(self.objects):
                # 自动清理空物体（切换到另一物体时删除空物体）
                old = self.active_object()
                if old and not old.points and len(self.objects) > 1:
                    self._remove_object(old)
                    idx = min(idx, len(self.objects) - 1)
                self.active_obj_idx = idx

        elif name == "delete_object":
            idx = args[0] if args else self.active_obj_idx
            if 0 <= idx < len(self.objects):
                obj = self.objects[idx]
                self._remove_object(obj)
                self._set_status(f"已删除 {obj.name}", "warn")

        elif name == "clear_points":
            obj = self.active_object()
            if obj:
                obj.points.clear()
                obj._dirty = True

        elif name == "start_tracking":
            if self.state != GUIState.EDIT:
                return
            dirty = [o for o in self.objects if o._dirty and o.points]
            if not dirty:
                self._set_status("所有物体均已追踪，无需重复。", "warn")
                return
            try:
                self._start_tracking()
            except (RuntimeError, MemoryError) as e:
                self._handle_oom(str(e))

        elif name == "abort_tracking":
            if self.state == GUIState.TRACKING:
                self._tracking_generator = None
                # 回滚：恢复追踪前的 mask 快照
                if self._pre_tracking_masks is not None:
                    self.masks = self._pre_tracking_masks
                    self._pre_tracking_masks = {}
                if self.inference_state is not None:
                    with contextlib.suppress(Exception):
                        self.predictor.reset_state(self.inference_state)
                self.state = GUIState.EDIT
                self._show_points_overlay = True
                self._preview_mask = None          # 清除预览残影
                n_preserved = sum(len(m) for m in self.masks.values())
                self._set_status(f"跟踪已中止。{n_preserved} 个已有 mask 已保留。", "warn")

        elif name == "mark_frame":
            if self.state == GUIState.EDIT:
                if self.current_frame_idx in self.composite_frames:
                    self.composite_frames.discard(self.current_frame_idx)
                else:
                    self.composite_frames.add(self.current_frame_idx)
                self._preview_dirty = True

        elif name == "prev_marked":
            frames = sorted(self.composite_frames)
            for f in reversed(frames):
                if f < self.current_frame_idx:
                    self.current_frame_idx = f
                    self._preview_dirty = True
                    break

        elif name == "next_marked":
            frames = sorted(self.composite_frames)
            for f in frames:
                if f > self.current_frame_idx:
                    self.current_frame_idx = f
                    self._preview_dirty = True
                    break

        elif name == "clear_all_marked":
            self.composite_frames.clear()
            self.frame_overrides.clear()
            self._excluded_frames.clear()
            self._data_version += 1
            self._preview_dirty = True
            self._set_status("已清除所有标记帧。", "info")

        elif name == "toggle_frame_object_at":
            # 在帧列表中切换指定帧的指定物体
            fidx = args[0] if args else self.current_frame_idx
            oid = args[1] if len(args) > 1 else None
            if oid is not None:
                if fidx not in self.frame_overrides:
                    self.frame_overrides[fidx] = {}
                cur = self.frame_overrides[fidx].get(oid)
                self.frame_overrides[fidx][oid] = not (cur if cur is not None else True)
                self._data_version += 1
                self._preview_dirty = True

        elif name == "set_per_frame_alpha":
            fidx = args[0] if args else self.current_frame_idx
            val = args[1] if len(args) > 1 else None
            if val is not None and 0.0 <= val <= 1.0:
                self.per_frame_alpha[fidx] = val
                self._data_version += 1
                self._preview_dirty = True

        elif name == "set_alpha_start":
            val = args[0] if args else None
            if val is not None and 0.0 <= val <= 1.0:
                self.alpha_start = val
                self._preview_dirty = True

        elif name == "set_alpha_end":
            val = args[0] if args else None
            if val is not None and 0.0 <= val <= 1.0:
                self.alpha_end = val
                self._preview_dirty = True

        elif name == "reset_per_frame_alphas":
            self.per_frame_alpha.clear()
            self._preview_dirty = True
            self._set_status("逐帧 alpha 已重置。", "info")

        elif name == "range_select":
            if self.state == GUIState.EDIT:
                if self._range_start is None:
                    self._range_start = self.current_frame_idx
                    self._set_status(f"范围起点: {self._range_start}。再按一次 R 设终点。", "info")
                else:
                    start, end = sorted([self._range_start, self.current_frame_idx])
                    for f in range(start, end + 1):
                        self.composite_frames.add(f)
                    self._range_start = None
                    self._set_status(f"范围 {start}→{end}：已添加 {end - start + 1} 帧。", "info")
                    self._preview_dirty = True

        elif name == "apply_interval":
            if self.state == GUIState.EDIT:
                try:
                    interval_sec = float(self.panel._interval_str.get()) if self.panel else 1.5
                    r_start = int(self.panel._range_start_str.get()) if self.panel else 0
                    r_end = int(self.panel._range_end_str.get()) if self.panel else self.n_frames - 1
                except ValueError:
                    self._set_status("间隔参数格式错误", "error")
                    return
                r_start = max(0, min(r_start, self.n_frames - 1))
                r_end = max(0, min(r_end, self.n_frames - 1))
                if r_start > r_end:
                    r_start, r_end = r_end, r_start
                fps = max(self.fps, 1)
                step = max(1, int(round(interval_sec * fps)))
                count = 0
                for f in range(r_start, r_end + 1, step):
                    self.composite_frames.add(f)
                    count += 1
                self._set_status(f"间隔 {interval_sec}秒 ({r_start}→{r_end})：已添加 {count} 帧。", "info")
                self._preview_dirty = True

        elif name == "set_bg":
            self.background_frame_idx = self.current_frame_idx
            self._preview_dirty = True
            self._set_status(f"背景设为第 {self.current_frame_idx} 帧", "info")

        elif name in ("view_mask", "view_composite", "view_original"):
            self.viz_mode = name.split("_")[1]
            self._preview_dirty = True

        elif name == "preview_frame":
            self._do_single_frame_preview()

        elif name == "save":
            if not self.composite_frames:
                self._set_status("没有标记任何帧！请用 K 键或间隔选择来标记帧。", "error")
                return
            msg = self._composite_and_save()
            self._set_status(msg, "success")

        elif name == "restart":
            # 直接重置，无确认对话框（避免 messagebox 卡死 + 误触概率极低）
            if self.inference_state is not None:
                with contextlib.suppress(Exception):
                    self.predictor.reset_state(self.inference_state)
                self.inference_state = None
            self.objects.clear()
            self.active_obj_idx = -1
            self.masks.clear()
            self._tracked_obj_ids.clear()
            self.composite_frames.clear()
            self.frame_overrides.clear()
            self._excluded_frames.clear()
            self._data_version += 1
            self.background_frame_idx = 0
            self._range_start = None
            self._preview_dirty = True
            self._preview_cache = None
            self._preview_mask = None
            self._frame_cache.clear()
            self.viz_mode = "mask"
            self._show_onboarding = True
            self._show_points_overlay = True
            self.per_frame_alpha.clear()
            self.alpha_start = self.args.alpha
            self.alpha_end = self.args.alpha
            self.state = GUIState.EDIT
            self._set_status("已重新开始。", "info")

    # ==================================================================
    # 状态消息
    # ==================================================================
    def _set_status(self, msg: str, level: str = "info") -> None:
        """设置状态消息（终端打印中文，OpenCV 覆盖层不显示中文避免乱码）。"""
        self.status_message = msg
        # 终端显示完整中文消息
        prefix = {"error": "[ERROR]", "warn": "[WARN]", "success": "[OK]", "info": "[INFO]"}
        print(f"{prefix.get(level, '[INFO]')} {msg}")
        # 同步到控制面板日志栏
        if self.panel:
            self.panel.log(msg)
        if level == "error":
            self.status_color = (0, 0, 255)
            self.status_timer = 180
        elif level == "warn":
            self.status_color = (0, 200, 255)
            self.status_timer = 120
        elif level == "success":
            self.status_color = (0, 255, 0)
            self.status_timer = 150
        else:
            self.status_color = (0, 255, 255)
            self.status_timer = 90

    # ==================================================================
    # Main loop
    # ==================================================================
    def run(self) -> None:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.createTrackbar(TRACKBAR_FRAME, WINDOW_NAME, 0, max(0, self.n_frames - 1),
                           self._on_trackbar_frame)
        cv2.setMouseCallback(WINDOW_NAME, self._on_mouse)
        self._set_trackbar(TRACKBAR_FRAME, self.current_frame_idx)

        from gui_panel import ControlPanel
        self.panel = ControlPanel(self)
        self._quit_flag = False

        def tick() -> None:
            if self._quit_flag:
                self.panel.destroy()
                cv2.destroyWindow(WINDOW_NAME)
                return

            # 模态对话框期间跳过渲染+面板同步，避免 messagebox 嵌套事件循环中
            # OpenCV 操作或 UI 重建导致 tkinter 卡死
            if self._in_modal:
                if not self._quit_flag:
                    self.panel.root.after(50, tick)
                return

            try:
                if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                    self.panel.destroy()
                    cv2.destroyWindow(WINDOW_NAME)
                    return
            except cv2.error:
                self.panel.destroy()
                return

            # 键盘输入
            key_raw = cv2.pollKey()
            if key_raw >= 0:
                key = key_raw & 0xFF
                self._handle_keyboard(key, key_raw)

            # ★ 先同步面板：确保状态变化触发的控件创建/销毁在渲染前完成，
            # 避免 _advance_tracking() 访问尚未创建的 progress/label 控件。
            if self.panel:
                self.panel.sync_from_gui()

            # 渲染
            if self.state == GUIState.TRACKING:
                canvas = self._render_tracking()
            else:
                canvas = self._render_edit()

            cv2.imshow(WINDOW_NAME, canvas)

            # 状态消息计时
            if self.status_timer > 0:
                self.status_timer -= 1
                if self.status_timer == 0:
                    self.status_message = ""

            if not self._quit_flag:
                self.panel.root.after(33, tick)

        self.panel.root.after(100, tick)
        self.panel.root.mainloop()
        cv2.destroyAllWindows()

    # ==================================================================
    # Keyboard handler
    # ==================================================================
    def _handle_keyboard(self, key: int, key_raw: int) -> None:
        ctrl = (key_raw & 0xE00000) != 0  # 检测 Ctrl 修饰键

        if key_raw == 27:  # Esc
            if self.state == GUIState.TRACKING:
                self.action("abort_tracking")
            else:
                self.action("quit")
            return

        # ── 导航 ──
        if key_raw in (81, 65361, 2424832):  # ←
            if ctrl:
                self.action("prev_marked")
            else:
                self.current_frame_idx = (self.current_frame_idx - 1) % self.n_frames
                self._set_trackbar(TRACKBAR_FRAME, self.current_frame_idx)
                self._preview_dirty = True
                self._preview_mask = None  # 移动帧清除预览
        elif key_raw in (83, 65363, 2555904):  # →
            if ctrl:
                self.action("next_marked")
            else:
                self.current_frame_idx = (self.current_frame_idx + 1) % self.n_frames
                self._set_trackbar(TRACKBAR_FRAME, self.current_frame_idx)
                self._preview_dirty = True
                self._preview_mask = None

        # ── 数字键选择物体 ──
        if ord("1") <= key <= ord("9"):
            idx = key - ord("1")
            if idx < len(self.objects):
                self.action("select_object", idx)

        # ── N: 新建物体 ──
        elif key == ord("n") or key == ord("N"):
            self.action("new_object")

        # ── Tab: 切换物体 ──
        elif key == 9:
            if self.objects:
                self.active_obj_idx = (self.active_obj_idx + 1) % len(self.objects)

        # ── EDIT 态专用键 ──
        if self.state == GUIState.EDIT:
            if key == 8:  # Backspace
                obj = self.active_object()
                if obj and obj.points:
                    obj.points.pop()
                    obj._dirty = True
                    self._preview_dirty = True
            elif key == 127:  # Delete
                self.action("delete_object", self.active_obj_idx)
            elif key in (13, 10, 32):  # Enter / Space
                self.action("start_tracking")
            elif key == ord("k") or key == ord("K"):
                self.action("mark_frame")
            elif key == ord("b") or key == ord("B"):
                self.action("set_bg")
            elif key == ord("v") or key == ord("V"):
                cycle = {"mask": "composite", "composite": "original", "original": "mask"}
                self.action("view_" + cycle[self.viz_mode])
            elif key == ord("i") or key == ord("I"):
                self.action("apply_interval")
            elif key == ord("s") or key == ord("S"):
                self.action("save")
            elif key == ord("p") or key == ord("P"):
                self.action("preview_frame")
            elif key == ord("r") or key == ord("R"):
                self.action("range_select")
        # ── TRACKING 态专用键 ──
        elif self.state == GUIState.TRACKING:
            if key_raw == 27:
                self.action("abort_tracking")

    # ==================================================================
    # Trackbar callbacks
    # ==================================================================
    def _on_trackbar_frame(self, value: int) -> None:
        if self._trackbar_locked or self.state == GUIState.TRACKING:
            return
        if value != self.current_frame_idx:
            self.current_frame_idx = value
            self._preview_dirty = True
            self._preview_mask = None

    def _on_trackbar_alpha(self, value: int) -> None:
        self.args.alpha = value / 100.0
        self._preview_dirty = True

    def _set_trackbar(self, name: str, value: int) -> None:
        self._trackbar_locked = True
        cv2.setTrackbarPos(name, WINDOW_NAME, value)
        self._trackbar_locked = False

    # ==================================================================
    # Mouse
    # ==================================================================
    def _on_mouse(self, event: int, x: int, y: int, _flags: int, _param: object) -> None:
        if self.state != GUIState.EDIT:
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            if 0 <= x < self.w and 0 <= y < self.h:
                # 首次点击：仅关闭引导覆盖层，不添加点
                if self._show_onboarding:
                    self._show_onboarding = False
                    return
                # 如果无物体，自动创建一个
                if not self.objects:
                    self.action("new_object")
                obj = self.active_object()
                if obj:
                    # 第一个点 → 设置 seed_frame
                    if not obj.points:
                        obj.seed_frame = self.current_frame_idx
                    obj.points.append((x, y))
                    obj._dirty = True
                    self._preview_dirty = True

    # ==================================================================
    # Frame helpers
    # ==================================================================
    def _current_frame(self) -> np.ndarray:
        return read_frame_at_fast(self.cap, self.current_frame_idx, self._frame_cache)

    # ==================================================================
    # EDIT rendering
    # ==================================================================
    def _render_edit(self) -> np.ndarray:
        frame = self._current_frame()

        if self.viz_mode == "composite":
            # 合成视图：仅显示合成结果，不叠加当前帧/选点/预览mask
            canvas = self._get_composite_preview()
        elif self.viz_mode == "mask":
            canvas = self._draw_all_masks(frame)
            # mask 视图下叠加跟踪点和预览mask
            if self._show_points_overlay:
                self._draw_points_overlay(canvas)
            if self._preview_mask is not None:
                overlay = np.zeros_like(canvas, dtype=np.uint8)
                overlay[self._preview_mask] = (0, 255, 120)
                cv2.addWeighted(overlay, 0.35, canvas, 1.0, 0, canvas)
            if self._show_onboarding and not self.objects:
                self._draw_onboarding(canvas)
        else:
            canvas = frame.copy()
            # 原图视图下叠加跟踪点和预览mask
            if self._show_points_overlay:
                self._draw_points_overlay(canvas)
            if self._preview_mask is not None:
                overlay = np.zeros_like(canvas, dtype=np.uint8)
                overlay[self._preview_mask] = (0, 255, 120)
                cv2.addWeighted(overlay, 0.35, canvas, 1.0, 0, canvas)
            if self._show_onboarding and not self.objects:
                self._draw_onboarding(canvas)

        self._draw_status_bar(canvas)
        canvas = _draw_timeline_on_canvas(self, canvas)
        return canvas

    def _draw_points_overlay(self, canvas: np.ndarray) -> None:
        """仅绘制当前活跃物体的跟踪点。"""
        active = self.active_object()
        if active is None:
            return
        b, g, r = int(active.color[0]), int(active.color[1]), int(active.color[2])
        for i, (px, py) in enumerate(active.points, start=1):
            cv2.circle(canvas, (px, py), 7, (b, g, r), -1)
            cv2.circle(canvas, (px, py), 20, (b, g, r), 2)
            cv2.putText(canvas, str(i), (px + 8, py - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (b, g, r), 2, cv2.LINE_AA)

    def _draw_onboarding(self, canvas: np.ndarray) -> None:
        """首次启动引导覆盖层。"""
        overlay = canvas.copy()
        cv2.rectangle(overlay, (0, 0), (self.w, self.h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.55, canvas, 0.45, 0, canvas)

        lines = [
            ("Click on video to mark objects to track", 1.1),
            ("N: New object  |  Enter: Start tracking", 0.7),
            ("P: Preview  |  K: Mark frame  |  S: Save", 0.6),
            ("Use control panel on the right for all actions", 0.6),
        ]
        cy = self.h // 2 - 40
        for text, scale in lines:
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
            cx = (self.w - tw) // 2
            cv2.putText(canvas, text, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX,
                        scale, (255, 255, 255), 2, cv2.LINE_AA)
            cy += th + 20

    # ==================================================================
    # TRACKING
    # ==================================================================
    def _start_tracking(self) -> None:
        dirty = [o for o in self.objects if o._dirty and o.points]
        if not dirty:
            return

        # snapshot 已有 mask（用于中止回滚）
        self._pre_tracking_masks = {
            fidx: dict(obj_masks)
            for fidx, obj_masks in self.masks.items()
        }

        self.state = GUIState.TRACKING
        self._show_points_overlay = False  # 跟踪开始后隐藏彩色选点
        self._preview_mask = None          # 清除旧的预览残影

        if self.inference_state is not None:
            self.predictor.reset_state(self.inference_state)
            self._set_status(f"增量跟踪 {len(dirty)} 个物体中...", "info")
        else:
            self._set_status("加载视频帧到内存...", "info")
            self.inference_state = self.predictor.init_state(
                video_path=str(self.video_path),
                offload_video_to_cpu=self.args.offload_video_to_cpu,
                offload_state_to_cpu=self.args.offload_state_to_cpu,
            )

        # 注册所有有点的物体（SAM2 要求预先注册）
        all_with_points = [o for o in self.objects if o.points]
        for obj in all_with_points:
            points_np = np.array(obj.points, dtype=np.float32)
            labels_np = np.ones(len(obj.points), dtype=np.int32)
            self.predictor.add_new_points_or_box(
                inference_state=self.inference_state,
                frame_idx=obj.seed_frame,
                obj_id=obj.obj_id,
                points=points_np,
                labels=labels_np,
            )

        self._tracked_obj_ids = {o.obj_id for o in all_with_points}
        self._tracking_pass = 0
        self._tracking_total_frames = self.n_frames  # forward+backward 各覆盖一半，总计约 n 帧
        self._tracking_frame_count = 0
        min_seed = min(o.seed_frame for o in all_with_points)

        self._tracking_direction = "forward"
        self._tracking_generator = self.predictor.propagate_in_video(
            self.inference_state, start_frame_idx=min_seed, reverse=False,
        )
        self._tracking_total_passes = 1 + sum(
            1 for o in all_with_points if o.seed_frame > min_seed
        )

    def _advance_tracking(self) -> bool:
        if self._tracking_generator is None:
            return False

        try:
            frame_idx, obj_ids, video_res_masks = next(self._tracking_generator)
            fidx = int(frame_idx)
            if fidx not in self.masks:
                self.masks[fidx] = {}
            for i, oid in enumerate(obj_ids):
                if oid in self._tracked_obj_ids:
                    mask_logits = video_res_masks[i]
                    if mask_logits.ndim == 3:
                        mask_logits = mask_logits[0]
                    mask = (mask_logits > self.args.mask_threshold).detach().cpu().numpy()
                    if mask.any():
                        self.masks[fidx][oid] = mask

            self._tracking_frame_count += 1
            self.current_frame_idx = fidx
            self._set_trackbar(TRACKBAR_FRAME, self.current_frame_idx)

            if self.panel and hasattr(self.panel, "progress_var"):
                pct = min(100, 100 * self._tracking_frame_count / max(self._tracking_total_frames, 1))
                try:
                    self.panel.progress_var.set(pct)
                    self.panel.lbl_progress.configure(text=f"{pct:.0f}% ({self._tracking_direction})")
                except tk.TclError:
                    pass  # 控件在重建中被销毁，跳过本次更新

            return True
        except StopIteration:
            self._tracking_pass += 1
            all_with_points = [o for o in self.objects if o.points]
            min_seed = min(o.seed_frame for o in all_with_points)
            late_objects = [o for o in all_with_points if o.seed_frame > min_seed]

            if self._tracking_pass == 1:
                self._tracking_direction = "backward"
                self._tracking_generator = self.predictor.propagate_in_video(
                    self.inference_state, start_frame_idx=min_seed, reverse=True,
                )
                return self._advance_tracking()
            elif late_objects and self._tracking_pass - 2 < len(late_objects):
                obj = late_objects[self._tracking_pass - 2]
                self._tracking_direction = f"backward({obj.name})"
                self._tracking_generator = self.predictor.propagate_in_video(
                    self.inference_state, start_frame_idx=obj.seed_frame, reverse=True,
                )
                return self._advance_tracking()
            else:
                # 完成
                self._tracking_generator = None
                self._pre_tracking_masks = {}
                for obj in self.objects:
                    if obj._dirty and obj.points:
                        obj._dirty = False
                self.state = GUIState.EDIT
                self._show_points_overlay = True
                self._preview_mask = None          # 清除预览残影
                self.current_frame_idx = min_seed
                self._set_trackbar(TRACKBAR_FRAME, self.current_frame_idx)
                total_masks = sum(len(m) for m in self.masks.values())
                self._set_status(f"跟踪完成。{total_masks} 个 mask，覆盖 {len(self.masks)} 帧。", "success")
                self._preview_dirty = True
                return False

    def _render_tracking(self) -> np.ndarray:
        self._advance_tracking()
        frame = self._current_frame()
        canvas = self._draw_all_masks(frame)
        self._draw_status_bar(canvas)

        if self.state == GUIState.TRACKING:
            pct = min(100, 100 * self._tracking_frame_count / max(self._tracking_total_frames, 1))
            bar_x, bar_y, bar_w, bar_h = 20, self.h - 30, self.w - 40, 12
            cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (80, 80, 80), -1)
            fill_w = int(bar_w * pct / 100)
            if fill_w > 0:
                cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h),
                              (0, 220, 0), -1)
            cv2.putText(canvas, f"Pass {self._tracking_pass + 1}/{max(self._tracking_total_passes, 1)}",
                        (bar_x, bar_y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
        return canvas

    # ==================================================================
    # 单帧预览
    # ==================================================================
    def _do_single_frame_preview(self) -> None:
        obj = self.active_object()
        if not obj or not obj.points:
            self._set_status("请先为活跃物体添加跟踪点。", "warn")
            return

        if self.state == GUIState.TRACKING:
            return

        try:
            if self.inference_state is None:
                self._set_status("首次预览，加载视频帧...", "info")
                self.inference_state = self.predictor.init_state(
                    video_path=str(self.video_path),
                    offload_video_to_cpu=self.args.offload_video_to_cpu,
                    offload_state_to_cpu=self.args.offload_state_to_cpu,
                )

            # ★ 不 reset_state！直接注册当前物体的 prompt（SAM2 支持多物体共存）
            # reset_state 会清除所有已注册物体，迫使已跟踪物体重新来过
            points_np = np.array(obj.points, dtype=np.float32)
            labels_np = np.ones(len(obj.points), dtype=np.int32)
            self.predictor.add_new_points_or_box(
                inference_state=self.inference_state,
                frame_idx=obj.seed_frame,
                obj_id=obj.obj_id,
                points=points_np,
                labels=labels_np,
            )

            # 单步传播获取当前帧 mask
            gen = self.predictor.propagate_in_video(
                self.inference_state,
                start_frame_idx=self.current_frame_idx,
                reverse=False,
            )
            try:
                fidx, obj_ids, video_res_masks = next(gen)
            except StopIteration:
                self._set_status("预览失败：当前帧无法生成 mask。", "error")
                return
            finally:
                gen.close() if hasattr(gen, "close") else None

            for i, oid in enumerate(obj_ids):
                if oid == obj.obj_id:
                    mask_logits = video_res_masks[i]
                    if mask_logits.ndim == 3:
                        mask_logits = mask_logits[0]
                    m = (mask_logits > self.args.mask_threshold).detach().cpu().numpy()
                    if m.any():
                        self._preview_mask = m
                        self._preview_mask_obj_id = obj.obj_id
                        self._set_status(f"预览: {obj.name} | 帧 {self.current_frame_idx}", "success")
                    else:
                        self._preview_mask = None
                        self._set_status("预览：mask 为空，请调整跟踪点。", "warn")
                    break

        except Exception as e:
            self._set_status(f"预览失败: {e}", "error")

    # ==================================================================
    # 绘制
    # ==================================================================
    def _draw_all_masks(self, frame: np.ndarray) -> np.ndarray:
        canvas = frame.copy()
        frame_masks = self.masks.get(self.current_frame_idx, {})
        for obj in self.objects:
            mask = frame_masks.get(obj.obj_id)
            if mask is None or not mask.any():
                continue
            b, g, r = int(obj.color[0]), int(obj.color[1]), int(obj.color[2])
            overlay = np.zeros_like(frame, dtype=np.uint8)
            overlay[mask] = (b, g, r)
            cv2.addWeighted(overlay, 0.30, canvas, 1.0, 0, canvas)
        return canvas

    def _draw_status_bar(self, canvas: np.ndarray) -> None:
        bar_h = 32
        overlay = canvas.copy()
        cv2.rectangle(overlay, (0, 0), (self.w, bar_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, canvas, 0.45, 0, canvas)

        n_objs = len(self.objects)
        active = self.active_object()
        active_name = active.name if active else "—"

        if self.state == GUIState.TRACKING:
            n_masks = sum(len(m) for m in self.masks.values())
            info = (f"TRACKING | Frame {self.current_frame_idx}/{self.n_frames}"
                    f" | Objs: {n_objs} | Masks: {n_masks}")
        else:
            marked = "K" if self.current_frame_idx in self.composite_frames else "-"
            fps_str = f"fps={self.fps:.0f}" if self.args.process_fps else ""
            alpha_str = f"alpha={self.get_frame_alpha(self.current_frame_idx):.2f}"
            info = (f"EDIT | Frame {self.current_frame_idx}/{self.n_frames}"
                    f" | Active: {active_name} | Marked: {len(self.composite_frames)}"
                    f" | BG: {self.background_frame_idx} | [{marked}]"
                    f" | {alpha_str} {fps_str}")

        cv2.putText(canvas, info, (10, 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (220, 220, 220), 1, cv2.LINE_AA)

    # ==================================================================
    # Composite
    # ==================================================================
    def _visible_objects_at(self, frame_idx: int) -> list[TrackObject]:
        result = []
        for obj in self.objects:
            start = obj.vis_start if obj.vis_start is not None else 0
            end = obj.vis_end if obj.vis_end is not None else self.n_frames
            in_range = start <= frame_idx <= end
            override = self.frame_overrides.get(frame_idx, {}).get(obj.obj_id)
            if override is not None:
                in_range = override
            if in_range and self.masks.get(frame_idx, {}).get(obj.obj_id) is not None:
                result.append(obj)
        return result

    def _render_composite(self) -> np.ndarray:
        bg = read_frame_at_fast(self.cap, self.background_frame_idx, self._frame_cache)
        canvas = bg.astype(np.float32)

        for fidx in sorted(self.composite_frames):
            if fidx in self._excluded_frames:
                continue
            for obj in self._visible_objects_at(fidx):
                mask = self.masks.get(fidx, {}).get(obj.obj_id)
                if mask is None or not mask.any():
                    continue
                frame = read_frame_at_fast(self.cap, fidx, self._frame_cache)
                from stroboscopic_image_generator import clean_mask
                mask_clean = clean_mask(
                    mask=mask, min_area=self.args.min_area,
                    dilate_kernel=self.args.dilate_kernel,
                    seed_xys=obj.points if fidx == obj.seed_frame else None,
                )
                if not mask_clean.any():
                    continue
                m = mask_clean.astype(np.float32)[..., None]
                fa = self.get_frame_alpha(fidx)
                canvas = (canvas * (1.0 - fa * m)
                          + frame.astype(np.float32) * (fa * m))

        # 种子帧 100% 不透明置顶
        for obj in self.objects:
            seed_mask = self.masks.get(obj.seed_frame, {}).get(obj.obj_id)
            if seed_mask is not None and seed_mask.any():
                from stroboscopic_image_generator import clean_mask
                seed_mask_clean = clean_mask(
                    mask=seed_mask, min_area=self.args.min_area,
                    dilate_kernel=self.args.dilate_kernel, seed_xys=obj.points,
                )
                if seed_mask_clean.any():
                    seed_frame = read_frame_at_fast(self.cap, obj.seed_frame, self._frame_cache)
                    m0 = seed_mask_clean.astype(np.float32)[..., None]
                    canvas = canvas * (1.0 - m0) + seed_frame.astype(np.float32) * m0

        return np.clip(canvas, 0, 255).astype(np.uint8)

    def _get_composite_preview(self) -> np.ndarray:
        if not self._preview_dirty and self._preview_cache is not None:
            return self._preview_cache.copy()
        preview = self._render_composite()
        self._preview_cache = preview
        self._preview_dirty = False
        return preview.copy()

    def _composite_and_save(self) -> str:
        out_img = self._render_composite()
        self.args.out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(self.args.out), out_img)
        file_size = self.args.out.stat().st_size / 1024
        return (
            f"已保存到: {self.args.out}\n"
            f"合成帧数: {len(self.composite_frames)}\n"
            f"物体数: {len(self.objects)}\n"
            f"文件大小: {file_size:.1f} KB"
        )

    # ==================================================================
    # OOM 恢复
    # ==================================================================
    def _handle_oom(self, error_msg: str) -> None:
        """OOM 时保留已有 mask 回到 EDIT，弹窗建议降参数。"""
        self._tracking_generator = None
        self.state = GUIState.EDIT
        if self.inference_state is not None:
            with contextlib.suppress(Exception):
                self.predictor.reset_state(self.inference_state)
            self.inference_state = None
        mem_mb = self.n_frames * self.w * self.h * 4 / (1024 * 1024)
        detail = (
            f"内存不足（需约 {mem_mb:.0f} MB）。\n\n"
            f"已有 mask 已保留，可以先保存当前结果。\n\n"
            f"当前: --max-dim {self.args.max_dim}, --process-fps {self.args.process_fps or '默认'}\n"
            f"分辨率: {self.w}x{self.h}, 帧数: {self.n_frames}\n\n"
            f"建议: --max-dim 640 --process-fps 3"
        )
        if self.panel:
            self._in_modal = True
            try:
                self.panel.show_error("内存不足 (OOM)", detail)
            finally:
                self._in_modal = False
        self._set_status("内存不足，已有 mask 已保留。请降低参数重试。", "error")

    # ── Alpha 系统 ──
    def get_frame_alpha(self, frame_idx: int) -> float:
        """返回指定帧的 alpha 值（逐帧覆盖 > 渐变插值）"""
        if frame_idx in self.per_frame_alpha:
            return self.per_frame_alpha[frame_idx]
        frames = sorted(self.composite_frames)
        if not frames or len(frames) == 1:
            return self.alpha_start
        if frame_idx <= frames[0]:
            return self.alpha_start
        if frame_idx >= frames[-1]:
            return self.alpha_end
        t = (frame_idx - frames[0]) / (frames[-1] - frames[0])
        return self.alpha_start + t * (self.alpha_end - self.alpha_start)

    def close(self) -> None:
        if self.inference_state is not None:
            with contextlib.suppress(Exception):
                self.predictor.reset_state(self.inference_state)
            self.inference_state = None


# ===================================================================
# Timeline drawer (enhanced: 60px, 3 layers)
# ===================================================================
def _draw_timeline_on_canvas(gui: StroboscopicGUI, canvas: np.ndarray) -> np.ndarray:
    h, w = canvas.shape[:2]
    n = gui.n_frames
    out = np.zeros((h + TIMELINE_H, w, 3), dtype=np.uint8)
    out[:h, :] = canvas
    y0 = h

    # ── Layer 1 (top 10px): object visibility range bars ──
    bar_h = TIMELINE_H // 3 - 1
    for obj in gui.objects:
        start = obj.vis_start if obj.vis_start is not None else 0
        end = obj.vis_end if obj.vis_end is not None else n - 1
        x1 = int(w * start / max(n, 1))
        x2 = int(w * (end + 1) / max(n, 1))
        b, g, r = int(obj.color[0]), int(obj.color[1]), int(obj.color[2])
        cv2.rectangle(out, (x1, y0 + 1), (x2, y0 + 1 + bar_h), (b, g, r), -1)

    # ── Layer 2 (middle 24px): per-frame status ──
    mid_y0 = y0 + bar_h + 2
    mid_h = TIMELINE_H - bar_h - 16
    for fidx in range(n):
        x_start = int(w * fidx / max(n, 1))
        x_end = int(w * (fidx + 1) / max(n, 1))
        x_end = max(x_end, x_start + 1)

        if fidx in gui.composite_frames:
            color = (0, 200, 80)  # 绿：标记
        elif fidx in gui.masks and gui.masks[fidx]:
            color = (60, 60, 60)   # 深灰：有 mask
        else:
            color = (40, 40, 40)   # 灰：空
        cv2.rectangle(out, (x_start, mid_y0), (x_end, mid_y0 + mid_h), color, -1)

    # 标记帧的绿色三角
    for fidx in sorted(gui.composite_frames):
        tri_x = int(w * (fidx + 0.5) / max(n, 1))
        tri_y = mid_y0
        pts = np.array([[tri_x - 3, tri_y], [tri_x + 3, tri_y], [tri_x, tri_y + 5]], np.int32)
        cv2.fillPoly(out, [pts], (0, 255, 0))

    # ── Layer 3 (bottom 14px): indicators ──
    btm_y = mid_y0 + mid_h

    # 背景指示器（橙色）
    bg_x = int(w * gui.background_frame_idx / max(n, 1))
    cv2.line(out, (bg_x, y0), (bg_x, y0 + TIMELINE_H), (255, 150, 50), 2)

    # 当前位置（白色）
    cur_x = int(w * (gui.current_frame_idx + 0.5) / max(n, 1))
    cv2.line(out, (cur_x, y0), (cur_x, y0 + TIMELINE_H), (255, 255, 255), 2)

    # 帧号标签
    cv2.putText(out, f"F{gui.current_frame_idx}", (cur_x + 4, y0 + TIMELINE_H - 3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)

    # 标记帧计数
    marked_info = f"Marked: {len(gui.composite_frames)}"
    cv2.putText(out, marked_info, (5, y0 + TIMELINE_H - 3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1, cv2.LINE_AA)

    return out
