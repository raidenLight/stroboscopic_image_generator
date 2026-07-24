"""gui_panel.py — tkinter 控制面板。

所有 UI 控件、状态同步、键盘转发、对话框。
独立于 OpenCV/SAM2 业务逻辑，通过 gui.action() 解耦。
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk, messagebox
from typing import TYPE_CHECKING

from gui_types import GUIState

if TYPE_CHECKING:
    from gui_app import StroboscopicGUI

# 面板初始尺寸
PANEL_WIDTH = 360
PANEL_HEIGHT = 700


class ControlPanel:
    """浮动 tkinter 控制面板 — 零额外依赖（Python 内置）。"""

    def __init__(self, gui: StroboscopicGUI):
        self.gui = gui
        self.root = tk.Tk()
        self.root.title("控制面板 — 频闪图像生成器")
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.resizable(True, True)
        self.root.geometry(f"{PANEL_WIDTH}x{PANEL_HEIGHT}+50+50")
        self.root.attributes("-topmost", True)
        self.root.lift()
        self.root.after(3000, lambda: self.root.attributes("-topmost", False))

        self.root.bind("<Key>", self._on_tk_key)

        style = ttk.Style(self.root)
        style.theme_use("clam")

        # 状态跟踪，避免不必要的重建
        self._last_state: GUIState | None = None
        self._last_active_idx: int = -1
        self._last_obj_count: int = -1
        self._last_marked_count: int = -1
        self._interval_value = tk.DoubleVar(value=1.5)

        self._build_static()
        self._build_dynamic()
        self.root.update()

    # ==================================================================
    # 静态控件（始终存在）
    # ==================================================================
    def _build_static(self) -> None:
        """构建不随状态变化的控件。"""
        # ── 状态 ──
        frm = ttk.LabelFrame(self.root, text="状态", padding=5)
        frm.pack(fill=tk.X, padx=5, pady=2)
        self.lbl_state = ttk.Label(frm, text="编辑", font=("", 10, "bold"))
        self.lbl_state.pack(anchor=tk.W)
        self.lbl_frame = ttk.Label(frm, text="帧: 0 / 0")
        self.lbl_frame.pack(anchor=tk.W)
        self.lbl_active = ttk.Label(frm, text="活跃: —")
        self.lbl_active.pack(anchor=tk.W)

        # ── 物体 ──
        self.frm_objects = ttk.LabelFrame(self.root, text="物体", padding=5)
        self.frm_objects.pack(fill=tk.X, padx=5, pady=2)
        self.obj_grid_frame = ttk.Frame(self.frm_objects)
        self.obj_grid_frame.pack(fill=tk.X)
        self.btn_new_obj = ttk.Button(
            self.frm_objects, text="+ 新建物体 (N)",
            command=lambda: self.gui.action("new_object"),
        )
        self.btn_new_obj.pack(fill=tk.X, pady=2)

        # ── 操作 ──
        self.frm_actions = ttk.LabelFrame(self.root, text="操作", padding=5)
        self.frm_actions.pack(fill=tk.X, padx=5, pady=2)
        self.actions_inner = ttk.Frame(self.frm_actions)
        self.actions_inner.pack(fill=tk.X)

        # ── 帧选取 ──
        self.frm_framesel = ttk.LabelFrame(self.root, text="帧选取", padding=5)
        self.frm_framesel.pack(fill=tk.X, padx=5, pady=2)
        self.framesel_inner = ttk.Frame(self.frm_framesel)
        self.framesel_inner.pack(fill=tk.X)

        # ── 合成帧列表 ──
        self.frm_marked = ttk.LabelFrame(self.root, text="合成帧", padding=5)
        self.frm_marked.pack(fill=tk.X, padx=5, pady=2)
        self.marked_inner = ttk.Frame(self.frm_marked)
        self.marked_inner.pack(fill=tk.X)

        # ── 视图模式 ──
        self.frm_view = ttk.LabelFrame(self.root, text="视图", padding=5)
        self.frm_view.pack(fill=tk.X, padx=5, pady=2)
        self.view_var = tk.StringVar(value="mask")
        view_inner = ttk.Frame(self.frm_view)
        view_inner.pack(fill=tk.X)
        for mode, label in [("mask", "Mask覆盖"), ("composite", "合成预览"), ("original", "原始帧")]:
            ttk.Radiobutton(
                view_inner, text=label, variable=self.view_var, value=mode,
                command=lambda m=mode: self.gui.action("view_" + m),
            ).pack(side=tk.LEFT, padx=3)

        # ── 可见范围占位（动态重建） ──
        self.frm_vis = ttk.LabelFrame(self.root, text="可见范围", padding=5)

        # ── 快捷键 ──
        frm_keys = ttk.LabelFrame(self.root, text="快捷键", padding=5)
        frm_keys.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(
            frm_keys,
            text="←→ 导航 | Ctrl+←→ 跳标记帧 | P 预览",
            font=("", 8),
        ).pack(anchor=tk.W)
        ttk.Label(
            frm_keys,
            text="N 新建 | 1-9 选物体 | Del 删物体 | Backspace 删点",
            font=("", 8),
        ).pack(anchor=tk.W)
        ttk.Label(
            frm_keys,
            text="K 标记帧 | I 间隔 | R 范围 | B 背景 | V 视图 | S 保存",
            font=("", 8),
        ).pack(anchor=tk.W)

        # ── 底部 ──
        bottom = ttk.Frame(self.root)
        bottom.pack(fill=tk.X, padx=5, pady=5)
        self.btn_restart = ttk.Button(
            bottom, text="↺ 重置全部",
            command=lambda: self.gui.action("restart"),
        )
        self.btn_restart.pack(side=tk.LEFT, padx=2)
        ttk.Button(
            bottom, text="退出 (Esc)",
            command=lambda: self.gui.action("quit"),
        ).pack(side=tk.RIGHT, padx=2)

    # ==================================================================
    # 动态控件（随状态/物体/标记帧变化重建）
    # ==================================================================
    def _build_dynamic(self) -> None:
        """(重)构建依赖当前状态的控件。"""
        self._last_state = self.gui.state
        self._last_active_idx = self.gui.active_obj_idx
        self._last_obj_count = len(self.gui.objects)

        # 清空动态区域
        for w in self.actions_inner.winfo_children():
            w.destroy()
        for w in self.framesel_inner.winfo_children():
            w.destroy()
        for w in self.marked_inner.winfo_children():
            w.destroy()

        if self.gui.state == GUIState.TRACKING:
            self._build_tracking_actions()
            self._build_vis_range(editable=False)
        else:
            self._build_edit_actions()
            self._build_frame_selection()
            self._build_marked_list()
            self._build_vis_range(editable=True)

    # ── EDIT 操作按钮 ──
    def _build_edit_actions(self) -> None:
        r = ttk.Frame(self.actions_inner)
        r.pack(fill=tk.X)
        ttk.Button(r, text="👁 单帧预览 (P)",
                   command=lambda: self.gui.action("preview_frame")).pack(side=tk.LEFT, padx=2)
        ttk.Button(r, text="✕ 清除选点 (Backspace)",
                   command=lambda: self.gui.action("clear_points")).pack(side=tk.LEFT, padx=2)

        # 开始跟踪按钮：脏物体存在时绿色高亮
        dirty = [o for o in self.gui.objects if o._dirty and o.points]
        if dirty:
            btn = tk.Button(
                self.actions_inner,
                text=f"▶ 开始跟踪 ({len(dirty)} 物体待追踪)",
                bg="#4CAF50", fg="white", font=("", 10, "bold"),
                relief=tk.RAISED,
                command=lambda: self.gui.action("start_tracking"),
            )
        else:
            btn = tk.Button(
                self.actions_inner,
                text="▶ 开始跟踪",
                bg="#cccccc", fg="#888888", font=("", 10),
                relief=tk.FLAT, state=tk.DISABLED,
            )
        btn.pack(fill=tk.X, pady=4, ipady=4)

    # ── TRACKING 操作按钮 ──
    def _build_tracking_actions(self) -> None:
        self.progress_var = tk.DoubleVar(value=0)
        self.progress_bar = ttk.Progressbar(
            self.actions_inner, variable=self.progress_var, length=260, mode="determinate",
        )
        self.progress_bar.pack(fill=tk.X, pady=2)
        self.lbl_progress = ttk.Label(self.actions_inner, text="0%")
        self.lbl_progress.pack()
        ttk.Button(
            self.actions_inner, text="⏹ 中止 (Esc)",
            command=lambda: self.gui.action("abort_tracking"),
        ).pack(fill=tk.X, pady=3)

    # ── 帧选取 ──
    def _build_frame_selection(self) -> None:
        # K 标记
        marked = self.gui.current_frame_idx in self.gui.composite_frames
        ttk.Button(
            self.framesel_inner,
            text="✓ 已标记 (K 取消)" if marked else "K 标记此帧",
            command=lambda: self.gui.action("mark_frame"),
        ).pack(fill=tk.X, pady=1)

        # R 范围
        rng_text = "R 范围选择"
        if self.gui._range_start is not None:
            rng_text = f"R 范围: {self.gui._range_start} → ?"
        ttk.Button(
            self.framesel_inner, text=rng_text,
            command=lambda: self.gui.action("range_select"),
        ).pack(fill=tk.X, pady=1)

        # 间隔
        frm_int = ttk.Frame(self.framesel_inner)
        frm_int.pack(fill=tk.X, pady=2)
        ttk.Label(frm_int, text="间隔:").pack(side=tk.LEFT)
        scale = ttk.Scale(
            frm_int, from_=0.1, to=10.0, variable=self._interval_value,
            orient=tk.HORIZONTAL, length=120,
            command=self._on_interval_change,
        )
        scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=3)
        self.lbl_interval_val = ttk.Label(frm_int, textvariable=self._interval_value, width=4)
        self.lbl_interval_val.pack(side=tk.LEFT)

        frm_int2 = ttk.Frame(self.framesel_inner)
        frm_int2.pack(fill=tk.X, pady=1)
        fps = max(self.gui.fps, 1)
        n_selected = int(self.gui.n_frames / max(fps * self._interval_value.get(), 0.1))
        self.lbl_interval_count = ttk.Label(frm_int2, text=f"~{n_selected} 帧")
        self.lbl_interval_count.pack(side=tk.LEFT)
        ttk.Button(
            frm_int2, text="应用间隔 (I)",
            command=lambda: self.gui.action("apply_interval"),
        ).pack(side=tk.RIGHT)

        # 背景
        ttk.Button(
            self.framesel_inner,
            text=f"B 设为背景 (当前: 帧{self.gui.background_frame_idx})",
            command=lambda: self.gui.action("set_bg"),
        ).pack(fill=tk.X, pady=1)

    # ── 标记帧列表 ──
    def _build_marked_list(self) -> None:
        frames = sorted(self.gui.composite_frames)
        count = len(frames)

        # 计数 + 快捷按钮
        top = ttk.Frame(self.marked_inner)
        top.pack(fill=tk.X)
        ttk.Label(top, text=f"已标记 {count} 帧").pack(side=tk.LEFT)

        if count == 0:
            ttk.Label(self.marked_inner, text="（暂无标记帧）", foreground="gray").pack()
            return

        # Listbox（最多显示 6 行，超出可滚动）
        display_count = min(count, 6)
        list_h = max(display_count, 1) * 22
        lb_frame = ttk.Frame(self.marked_inner)
        lb_frame.pack(fill=tk.X)

        self.marked_listbox = tk.Listbox(
            lb_frame, height=display_count, exportselection=False,
            font=("Consolas", 9),
        )
        self.marked_listbox.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # 滚动条
        if count > 6:
            sb = ttk.Scrollbar(lb_frame, orient=tk.VERTICAL, command=self.marked_listbox.yview)
            sb.pack(side=tk.RIGHT, fill=tk.Y)
            self.marked_listbox.configure(yscrollcommand=sb.set)

        # 填充数据
        obj_map = {o.obj_id: o for o in self.gui.objects}
        for fidx in frames:
            shown = []
            for oid, obj in obj_map.items():
                override = self.gui.frame_overrides.get(fidx, {}).get(oid)
                mask_exists = self.gui.masks.get(fidx, {}).get(oid) is not None
                if not mask_exists:
                    continue
                if override is False:
                    shown.append(f"○{obj.name}")
                else:
                    shown.append(f"●{obj.name}")
            text = f"帧{fidx:>5d}  {' '.join(shown) if shown else '(无物体)'}"
            self.marked_listbox.insert(tk.END, text)

        # 操作按钮行
        btn_row = ttk.Frame(self.marked_inner)
        btn_row.pack(fill=tk.X, pady=2)
        ttk.Button(
            btn_row, text="✕ 删除选中帧",
            command=self._delete_selected_marked,
        ).pack(side=tk.LEFT, padx=1)
        ttk.Button(
            btn_row, text="▶ 跳转",
            command=self._jump_selected_marked,
        ).pack(side=tk.LEFT, padx=1)
        ttk.Button(
            btn_row, text="清空全部",
            command=lambda: self.gui.action("clear_all_marked"),
        ).pack(side=tk.LEFT, padx=1)
        ttk.Button(
            btn_row, text="◀ 上一个",
            command=lambda: self.gui.action("prev_marked"),
        ).pack(side=tk.RIGHT, padx=1)
        ttk.Button(
            btn_row, text="下一个 ▶",
            command=lambda: self.gui.action("next_marked"),
        ).pack(side=tk.RIGHT, padx=1)

        # 物体叠加切换
        ttk.Button(
            self.marked_inner, text="切换当前帧物体叠加 (O)",
            command=lambda: self.gui.action("toggle_frame_object"),
        ).pack(fill=tk.X, pady=1)

    def _delete_selected_marked(self) -> None:
        sel = self.marked_listbox.curselection()
        if sel:
            frames = sorted(self.gui.composite_frames)
            if sel[0] < len(frames):
                self.gui.composite_frames.discard(frames[sel[0]])
                self.gui._preview_dirty = True

    def _jump_selected_marked(self) -> None:
        sel = self.marked_listbox.curselection()
        if sel:
            frames = sorted(self.gui.composite_frames)
            if sel[0] < len(frames):
                self.gui.current_frame_idx = frames[sel[0]]
                self.gui._preview_dirty = True

    def _on_interval_change(self, *args) -> None:
        """间隔滑块实时更新预估帧数标签。"""
        fps = max(self.gui.fps, 1)
        n_sel = int(self.gui.n_frames / max(fps * self._interval_value.get(), 0.1))
        if hasattr(self, "lbl_interval_count"):
            self.lbl_interval_count.configure(text=f"~{n_sel} 帧")

    # ── 可见范围 ──
    def _build_vis_range(self, editable: bool) -> None:
        """构建活跃物体的可见范围控件。"""
        self.frm_vis.pack_forget()
        for w in self.frm_vis.winfo_children():
            w.destroy()

        if not self.gui.objects:
            self.frm_vis.pack(fill=tk.X, padx=5, pady=2, after=self.frm_view)
            ttk.Label(self.frm_vis, text="无物体").pack()
            return

        obj = self.gui.active_object()
        if obj is None:
            self.frm_vis.pack(fill=tk.X, padx=5, pady=2, after=self.frm_view)
            return

        self.frm_vis.configure(text=f"可见范围 ({obj.name})")
        ttk.Label(self.frm_vis, text=f"活跃: {obj.name}", foreground=obj.color_hex).pack(anchor=tk.W)

        n = max(self.gui.n_frames - 1, 1)
        self.vis_start_var = tk.IntVar(value=obj.vis_start if obj.vis_start is not None else 0)
        self.vis_end_var = tk.IntVar(value=obj.vis_end if obj.vis_end is not None else n)

        frm1 = ttk.Frame(self.frm_vis)
        frm1.pack(fill=tk.X)
        ttk.Label(frm1, text="起始:").pack(side=tk.LEFT)
        s1 = ttk.Scale(
            frm1, from_=0, to=n, variable=self.vis_start_var,
            orient=tk.HORIZONTAL, length=160,
        )
        s1.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=3)
        if editable:
            s1.configure(command=self._on_vis_change)
        else:
            s1.configure(state=tk.DISABLED)
        ttk.Label(frm1, textvariable=self.vis_start_var, width=5).pack(side=tk.LEFT)

        frm2 = ttk.Frame(self.frm_vis)
        frm2.pack(fill=tk.X)
        ttk.Label(frm2, text="结束:").pack(side=tk.LEFT)
        s2 = ttk.Scale(
            frm2, from_=0, to=n, variable=self.vis_end_var,
            orient=tk.HORIZONTAL, length=160,
        )
        s2.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=3)
        if editable:
            s2.configure(command=self._on_vis_change)
        else:
            s2.configure(state=tk.DISABLED)
        ttk.Label(frm2, textvariable=self.vis_end_var, width=5).pack(side=tk.LEFT)

        if editable:
            ttk.Button(
                self.frm_vis, text="应用范围",
                command=lambda: self.gui.action("apply_vis_range"),
            ).pack(fill=tk.X, pady=2)
            ttk.Button(
                self.frm_vis, text="重置为全部帧",
                command=lambda: self.gui.action("reset_vis_range"),
            ).pack(fill=tk.X)

        self.frm_vis.pack(fill=tk.X, padx=5, pady=2, after=self.frm_view)

    def _on_vis_change(self, *args) -> None:
        """可见范围滑块拖动时实时写入物体 + 标记预览脏。"""
        obj = self.gui.active_object()
        if obj and hasattr(self, "vis_start_var") and hasattr(self, "vis_end_var"):
            obj.vis_start = self.vis_start_var.get()
            obj.vis_end = self.vis_end_var.get()
            self.gui._preview_dirty = True

    # ==================================================================
    # 同步 & 重建
    # ==================================================================
    def sync_from_gui(self) -> None:
        """每帧调用：同步状态标签 + 必要时重建动态控件。"""
        gui = self.gui

        # 检测是否需要重建
        need_rebuild = (
            gui.state != self._last_state
            or gui.active_obj_idx != self._last_active_idx
            or len(gui.objects) != self._last_obj_count
            or len(gui.composite_frames) != self._last_marked_count
        )

        if need_rebuild:
            self._rebuild_object_buttons()
            self._build_dynamic()
            self._last_marked_count = len(gui.composite_frames)

        # 状态标签（仅在非 TRACKING 时显示"编辑"）
        state_text = "跟踪中" if gui.state == GUIState.TRACKING else "编辑"
        self.lbl_state.configure(text=state_text)
        self.lbl_frame.configure(text=f"帧: {gui.current_frame_idx} / {gui.n_frames}")

        if gui.active_object():
            obj = gui.active_object()
            n_pts = len(obj.points)
            dirty_mark = " +" if obj._dirty and gui.state != GUIState.TRACKING else ""
            self.lbl_active.configure(
                text=f"活跃: {obj.name} ({n_pts}点, 种子帧{obj.seed_frame}){dirty_mark}",
            )
        else:
            self.lbl_active.configure(text="活跃: —")

        # 物体按钮高亮
        for i, btn in enumerate(getattr(self, "_obj_btns", [])):
            if i == gui.active_obj_idx:
                btn.configure(bg="white", relief=tk.RAISED, font=("", 9, "bold"))
            else:
                btn.configure(bg="#e0e0e0", relief=tk.FLAT, font=("", 9))

        # 删除按钮状态
        for i, del_btn in enumerate(getattr(self, "_del_btns", [])):
            del_btn.configure(state=tk.DISABLED if i == gui.active_obj_idx and gui.state == GUIState.TRACKING else tk.NORMAL)

        # 同步视图单选按钮
        if self.view_var.get() != gui.viz_mode:
            self.view_var.set(gui.viz_mode)

        # 同步可见范围滑块值（仅在滑块不存在于 vis 对象中时 — 避免覆盖用户拖拽）
        if gui.active_object() and hasattr(self, "vis_start_var"):
            obj = gui.active_object()
            n = max(gui.n_frames - 1, 1)
            expected_start = obj.vis_start if obj.vis_start is not None else 0
            expected_end = obj.vis_end if obj.vis_end is not None else n
            # 仅在"不是因滑块拖拽改变"时覆盖（_on_vis_change 会先写入 obj）
            if self.vis_start_var.get() != obj.vis_start or self.vis_end_var.get() != obj.vis_end:
                pass  # 滑块已写入 obj，不需要反向覆盖

    def _rebuild_object_buttons(self) -> None:
        """重建物体按钮网格（2 列布局，含删除按钮）。"""
        for w in self.obj_grid_frame.winfo_children():
            w.destroy()
        self._obj_btns = []
        self._del_btns = []

        COLS = 2
        for i, obj in enumerate(self.gui.objects):
            row = i // COLS
            col = (i % COLS) * 2  # 每物体占 2 列（按钮 + 删除）

            is_active = (i == self.gui.active_obj_idx)
            n_pts = len(obj.points)

            # 状态文字
            if obj._dirty and n_pts > 0:
                label = f"● {obj.name} ({n_pts}点)+"
            elif n_pts > 0:
                label = f"✓ {obj.name} ({n_pts}点)"
            else:
                label = f"○ {obj.name} (空)"

            btn = tk.Button(
                self.obj_grid_frame,
                text=label,
                fg=obj.color_hex,
                bg="white" if is_active else "#e0e0e0",
                activebackground="#c0c0ff",
                relief=tk.RAISED if is_active else tk.FLAT,
                font=("", 9, "bold") if is_active else ("", 9),
                command=lambda idx=i: self.gui.action("select_object", idx),
            )
            btn.grid(row=row, column=col, padx=2, pady=2, sticky="ew")
            self._obj_btns.append(btn)

            del_btn = tk.Button(
                self.obj_grid_frame,
                text="✕",
                fg="#cc0000",
                bg="#ffe0e0" if is_active else "#e0e0e0",
                activebackground="#ffcccc",
                relief=tk.FLAT,
                font=("", 9),
                command=lambda idx=i: self.gui.action("delete_object", idx),
            )
            del_btn.grid(row=row, column=col + 1, padx=1, pady=2)
            self._del_btns.append(del_btn)

        # 使 2 列等宽
        for c in range(COLS * 2):
            self.obj_grid_frame.columnconfigure(c, weight=1)

    # ==================================================================
    # 对话框
    # ==================================================================
    def show_error(self, title: str, msg: str = "") -> None:
        messagebox.showerror(title, msg if msg else title)

    def show_info(self, title: str, msg: str = "") -> None:
        messagebox.showinfo(title, msg if msg else title)

    def confirm(self, title: str, msg: str) -> bool:
        return messagebox.askyesno(title, msg)

    def toggle_frame_objects_dialog(self) -> list[int] | None:
        """弹出对话框让用户选择当前帧中要显示的物体。返回选中的 obj_id 列表。"""
        fidx = self.gui.current_frame_idx
        available = []
        for obj in self.gui.objects:
            if self.gui.masks.get(fidx, {}).get(obj.obj_id) is not None:
                available.append(obj)

        if not available:
            messagebox.showinfo("提示", f"帧 {fidx} 没有可用的物体 mask。")
            return None

        # 简单实现：用多个 askyesno 逐个确认
        result = []
        for obj in available:
            current = self.gui.frame_overrides.get(fidx, {}).get(obj.obj_id)
            default = True if current is None else current
            if messagebox.askyesno(
                f"帧 {fidx} — {obj.name}",
                f"{obj.name} 在此帧参与叠加？",
                default="yes" if default else "no",
            ):
                result.append(obj.obj_id)
        return result

    # ==================================================================
    # 键盘转发
    # ==================================================================
    def _on_tk_key(self, event: tk.Event) -> None:
        """将 tkinter 窗口的键盘事件转发到 GUI。"""
        gui = self.gui
        key = event.keysym
        char = event.char
        ctrl = event.state & 0x4  # Control 键

        if key == "Escape":
            gui.action("quit")
        elif key == "Return" or key == "space":
            if gui.state == GUIState.EDIT:
                gui.action("start_tracking")
        elif key == "BackSpace":
            obj = gui.active_object()
            if obj and obj.points:
                obj.points.pop()
                obj._dirty = True
                gui._preview_dirty = True
        elif key == "Delete":
            gui.action("delete_object", gui.active_obj_idx)
        elif key in ("Left", "Right", "Up", "Down"):
            if ctrl:
                gui.action("prev_marked" if key == "Left" else "next_marked")
            elif key == "Left":
                gui.current_frame_idx = (gui.current_frame_idx - 1) % gui.n_frames
                gui._preview_dirty = True
            elif key == "Right":
                gui.current_frame_idx = (gui.current_frame_idx + 1) % gui.n_frames
                gui._preview_dirty = True
        elif char.lower() == "k":
            gui.action("mark_frame")
        elif char.lower() == "b":
            gui.action("set_bg")
        elif char.lower() == "v":
            cycle = {"mask": "composite", "composite": "original", "original": "mask"}
            gui.action("view_" + cycle[gui.viz_mode])
        elif char.lower() == "i":
            gui.action("apply_interval")
        elif char.lower() == "s":
            gui.action("save")
        elif char.lower() == "n":
            gui.action("new_object")
        elif char.lower() == "p":
            gui.action("preview_frame")
        elif char.lower() == "r":
            gui.action("range_select")
        elif char.lower() == "o":
            gui.action("toggle_frame_object")
        elif char.lower() == "[":
            obj = gui.active_object()
            if obj:
                obj.vis_start = gui.current_frame_idx
                gui._preview_dirty = True
        elif char.lower() == "]":
            obj = gui.active_object()
            if obj:
                obj.vis_end = gui.current_frame_idx
                gui._preview_dirty = True
        elif char.lower() == "\\":
            gui.action("reset_vis_range")
        elif char in "123456789":
            idx = int(char) - 1
            if idx < len(gui.objects):
                gui.action("select_object", idx)
        elif key == "Tab":
            if gui.objects:
                gui.active_obj_idx = (gui.active_obj_idx + 1) % len(gui.objects)

    def _on_close(self) -> None:
        self.gui.action("quit")

    def destroy(self) -> None:
        try:
            self.root.destroy()
        except tk.TclError:
            pass
