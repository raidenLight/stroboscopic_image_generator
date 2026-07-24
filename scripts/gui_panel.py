"""gui_panel.py — tkinter 控制面板。"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk, messagebox
from typing import TYPE_CHECKING

from gui_types import GUIState

if TYPE_CHECKING:
    from gui_app import StroboscopicGUI

PANEL_WIDTH = 460
PANEL_HEIGHT = 760


class ControlPanel:
    def __init__(self, gui: StroboscopicGUI):
        self.gui = gui
        self.root = tk.Tk()
        self.root.title("控制面板 — 频闪图像生成器")
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.resizable(True, True)
        self.root.minsize(360, 760)
        self.root.geometry(f"{PANEL_WIDTH}x{PANEL_HEIGHT}+50+50")
        self.root.bind_all("<Key>", self._on_tk_key)
        # 初始化时短暂置顶，确保面板不被 OpenCV 窗口遮挡
        self.root.attributes('-topmost', True)
        self.root.after(200, lambda: self.root.attributes('-topmost', False))

        style = ttk.Style(self.root)
        style.theme_use("clam")

        self._last_state: GUIState | None = None
        self._last_active_idx: int = -1
        self._last_obj_count: int = -1
        self._last_marked_count: int = -1
        self._last_data_version: int = -1
        self._interval_str = tk.StringVar(value="1.5")
        self._range_start_str = tk.StringVar(value="0")
        self._range_end_str = tk.StringVar(value=str(max(gui.n_frames - 1, 0)))
        self._alpha_start_str = tk.StringVar(value="1.00")
        self._alpha_end_str = tk.StringVar(value="1.00")

        self._build_static()
        self._build_dynamic()
        self.root.update()

    # ==================================================================
    # 日志
    # ==================================================================
    def log(self, msg: str) -> None:
        """向底部日志窗追加一条日志。"""
        if hasattr(self, "_log_text"):
            try:
                self._log_text.configure(state=tk.NORMAL)
                self._log_text.insert(tk.END, msg + "\n")
                self._log_text.see(tk.END)
                # 限制行数，保留最近 50 行
                line_count = int(self._log_text.index('end-1c').split('.')[0])
                if line_count > 50:
                    self._log_text.delete('1.0', f'{line_count - 50}.0')
                self._log_text.configure(state=tk.DISABLED)
            except tk.TclError:
                pass

    # ==================================================================
    # 静态控件（只创建一次）
    # ==================================================================
    def _build_static(self) -> None:
        # ★ 合成帧:日志:快捷键 = 3:2:1 固定比例，仅合成帧可弹性扩展
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_rowconfigure(6, weight=1, minsize=234)       # Treeview — 3 份
        self.root.grid_rowconfigure(7, weight=0, minsize=156)       # 日志栏 — 2 份
        self.root.grid_rowconfigure(8, weight=0, minsize=78)        # 快捷键 — 1 份

        # ── 状态（Row 0）──
        frm = ttk.LabelFrame(self.root, text="状态", padding=3)
        frm.grid(row=0, column=0, sticky="ew", padx=3, pady=1)
        self.lbl_state = ttk.Label(frm, text="编辑", font=("", 10, "bold"))
        self.lbl_state.pack(anchor=tk.W)
        lf = ttk.Frame(frm)
        lf.pack(fill=tk.X)
        self.lbl_frame = ttk.Label(lf, text="帧: 0 / 0")
        self.lbl_frame.pack(side=tk.LEFT)
        self.lbl_active = ttk.Label(lf, text="活跃: —")
        self.lbl_active.pack(side=tk.RIGHT)

        # ── 物体（Row 1）──
        self.frm_objects = ttk.LabelFrame(self.root, text="物体", padding=3)
        self.frm_objects.grid(row=1, column=0, sticky="ew", padx=3, pady=1)
        self.obj_grid_frame = ttk.Frame(self.frm_objects)
        self.obj_grid_frame.pack(fill=tk.X)

        # ── 操作（Row 2）──
        self.frm_actions = ttk.LabelFrame(self.root, text="操作", padding=3)
        self.frm_actions.grid(row=2, column=0, sticky="ew", padx=3, pady=1)
        self.actions_inner = ttk.Frame(self.frm_actions)
        self.actions_inner.pack(fill=tk.X)

        # ── 视图（Row 3）──
        self.frm_view = ttk.LabelFrame(self.root, text="视图", padding=3)
        self.frm_view.grid(row=3, column=0, sticky="ew", padx=3, pady=1)
        self.view_inner = ttk.Frame(self.frm_view)
        self.view_inner.pack(fill=tk.X)

        # ── 帧选取（Row 4）──
        self.frm_select = ttk.LabelFrame(self.root, text="帧选取", padding=3)
        self.frm_select.grid(row=4, column=0, sticky="ew", padx=3, pady=1)
        self.select_inner = ttk.Frame(self.frm_select)
        self.select_inner.pack(fill=tk.X)

        # ── Alpha（Row 5）──
        self.frm_alpha = ttk.LabelFrame(self.root, text="Alpha 渐变", padding=3)
        self.frm_alpha.grid(row=5, column=0, sticky="ew", padx=3, pady=1)
        self.alpha_inner = ttk.Frame(self.frm_alpha)
        self.alpha_inner.pack(fill=tk.X)

        # ── 合成帧列表（Row 6, weight=1 — 吸收所有弹性压缩）──
        self.frm_marked = ttk.LabelFrame(self.root, text="合成帧", padding=3)
        self.frm_marked.grid(row=6, column=0, sticky="nsew", padx=3, pady=1)
        self.marked_inner = ttk.Frame(self.frm_marked)
        self.marked_inner.pack(fill=tk.BOTH, expand=True)

        # ── 日志栏（Row 7, minsize=170 — 不可压缩）──
        log_frame = tk.Frame(self.root, bg="#f0f0e0", relief=tk.SUNKEN, bd=1)
        log_frame.grid(row=7, column=0, sticky="ew", padx=3, pady=(1, 0))
        self._log_text = tk.Text(log_frame, height=10, font=("微软雅黑", 8), fg="#444444",
                                 bg="#f0f0e0", wrap=tk.WORD, state=tk.DISABLED, relief=tk.FLAT,
                                 bd=0, padx=3, pady=1)
        log_sb = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self._log_text.yview)
        self._log_text.configure(yscrollcommand=log_sb.set)
        self._log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        log_sb.pack(side=tk.RIGHT, fill=tk.Y)

        # ── 快捷键 + 操作按钮（Row 8, minsize=48 — 不可压缩）──
        bottom = ttk.Frame(self.root)
        bottom.grid(row=8, column=0, sticky="ew", padx=3, pady=1)
        r1 = ttk.Frame(bottom)
        r1.pack(fill=tk.X)
        ttk.Label(r1, text="P预览 K标记 I间隔 R范围 B背景  V视图 S保存",
                   font=("", 8)).pack(side=tk.LEFT)
        self.btn_restart = ttk.Button(r1, text="↺ 重置",
                                       command=lambda: self.gui.action("restart"))
        self.btn_restart.pack(side=tk.RIGHT, padx=2)
        r2 = ttk.Frame(bottom)
        r2.pack(fill=tk.X)
        ttk.Label(r2, text="Enter跟踪  ←→导航  Ctrl+←→跳标记  Tab切换物体  Backspace回退",
                   font=("", 8)).pack(side=tk.LEFT)

    # ==================================================================
    # 动态控件（每次 rebuild 重建）
    # ==================================================================
    def _build_dynamic(self) -> None:
        self._last_state = self.gui.state
        self._last_active_idx = self.gui.active_obj_idx
        self._last_obj_count = len(self.gui.objects)

        for w in self.actions_inner.winfo_children(): w.destroy()
        for w in self.view_inner.winfo_children(): w.destroy()
        for w in self.select_inner.winfo_children(): w.destroy()
        for w in self.alpha_inner.winfo_children(): w.destroy()
        for w in self.marked_inner.winfo_children(): w.destroy()

        if self.gui.state == GUIState.TRACKING:
            self._build_tracking_actions()
        else:
            self._build_edit_actions()
            self._build_view_section()
            self._build_select_section()
            self._build_alpha_section()
            self._build_marked_list()

    # ── EDIT 操作（预览 + 跟踪 + 选点模式 同行）──
    def _build_edit_actions(self) -> None:
        r = ttk.Frame(self.actions_inner)
        r.pack(fill=tk.X)
        ttk.Button(r, text="👁 保存并预览", command=lambda: self.gui.action("preview_frame")).pack(
            side=tk.LEFT, padx=1, fill=tk.X, expand=True)

        dirty = [o for o in self.gui.objects if o._dirty and o.points]
        if dirty:
            btn = tk.Button(r, text=f"▶ 跟踪({len(dirty)})", bg="#4CAF50",
                            fg="white", font=("", 9, "bold"), relief=tk.RAISED,
                            command=lambda: self.gui.action("start_tracking"))
        else:
            btn = tk.Button(r, text="▶ 跟踪", bg="#cccccc", fg="#888888",
                            font=("", 9), relief=tk.FLAT, state=tk.DISABLED)
        btn.pack(side=tk.LEFT, padx=1, fill=tk.X, expand=True, ipady=2)
        self._pt_btn = ttk.Button(r, command=lambda: self.gui.action("toggle_point_mode"))
        self._pt_btn.pack(side=tk.LEFT, padx=1, fill=tk.X, expand=True)

    # ── 视图 ──
    def _build_view_section(self) -> None:
        self.view_var = tk.StringVar(value=self.gui.viz_mode)
        for mode, label in [("mask", "Mask"), ("composite", "合成"), ("original", "原图")]:
            ttk.Radiobutton(self.view_inner, text=label, variable=self.view_var, value=mode,
                            command=lambda m=mode: self.gui.action("view_" + m)).pack(side=tk.LEFT, padx=3)
        ttk.Separator(self.view_inner, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=4)
        ttk.Label(self.view_inner, text="mask阈值:").pack(side=tk.LEFT)
        self._threshold_str = tk.StringVar(value="0.20")
        ttk.Entry(self.view_inner, textvariable=self._threshold_str, width=4).pack(side=tk.LEFT, padx=2)
        ttk.Button(self.view_inner, text="✓", width=2, command=self._apply_threshold).pack(side=tk.LEFT)

    def _apply_threshold(self):
        try:
            v = float(self._threshold_str.get())
            if 0.0 <= v <= 1.0:
                self.gui.args.mask_threshold = v
                self.gui._preview_dirty = True
                self.log(f"mask阈值={v:.2f}")
        except ValueError:
            pass

    # ── 帧选取 ──
    def _build_select_section(self) -> None:
        # 标记 / 背景 / 范围 / 清除 (一行)
        row1 = ttk.Frame(self.select_inner)
        row1.pack(fill=tk.X, pady=1)
        self._mark_btn = ttk.Button(row1, text="K 标记", command=lambda: self.gui.action("mark_frame"))
        self._mark_btn.pack(side=tk.LEFT, padx=1, fill=tk.X, expand=True)
        ttk.Button(row1, text="B 背景", command=lambda: self.gui.action("set_bg")).pack(side=tk.LEFT, padx=1, fill=tk.X, expand=True)
        self._range_btn = ttk.Button(row1, text="R 范围", command=lambda: self.gui.action("range_select"))
        self._range_btn.pack(side=tk.LEFT, padx=1, fill=tk.X, expand=True)
        ttk.Button(row1, text="清除标记", command=lambda: self.gui.action("clear_all_marked")).pack(side=tk.LEFT, padx=1, fill=tk.X, expand=True)

        # 间隔选择
        row2 = ttk.Frame(self.select_inner)
        row2.pack(fill=tk.X, pady=1)
        ttk.Label(row2, text="起始帧:").pack(side=tk.LEFT)
        ttk.Entry(row2, textvariable=self._range_start_str, width=5).pack(side=tk.LEFT, padx=1)
        ttk.Label(row2, text="中止帧:").pack(side=tk.LEFT, padx=(3, 0))
        ttk.Entry(row2, textvariable=self._range_end_str, width=5).pack(side=tk.LEFT, padx=1)
        ttk.Label(row2, text="间隔(秒):").pack(side=tk.LEFT, padx=(3, 0))
        ttk.Entry(row2, textvariable=self._interval_str, width=4).pack(side=tk.LEFT, padx=1)
        ttk.Button(row2, text="应用(I)", command=lambda: self.gui.action("apply_interval")).pack(side=tk.LEFT, padx=2)

    # ── Alpha（单行）──
    def _build_alpha_section(self) -> None:
        self._alpha_start_str.set(f"{self.gui.alpha_start:.2f}")
        self._alpha_end_str.set(f"{self.gui.alpha_end:.2f}")
        self._bg_alpha_str = tk.StringVar(value=f"{self.gui.background_alpha:.2f}")

        r = ttk.Frame(self.alpha_inner)
        r.pack(fill=tk.X)
        ttk.Label(r, text="首帧:").pack(side=tk.LEFT)
        e1 = ttk.Entry(r, textvariable=self._alpha_start_str, width=4)
        e1.pack(side=tk.LEFT, padx=1)
        e1.bind("<Return>", lambda e: self._apply_alpha_gradient())
        ttk.Label(r, text="末帧:").pack(side=tk.LEFT, padx=(3, 0))
        e2 = ttk.Entry(r, textvariable=self._alpha_end_str, width=4)
        e2.pack(side=tk.LEFT, padx=1)
        e2.bind("<Return>", lambda e: self._apply_alpha_gradient())
        ttk.Button(r, text="✓", width=2, command=self._apply_alpha_gradient).pack(side=tk.LEFT, padx=2)
        ttk.Label(r, text="背景:").pack(side=tk.LEFT, padx=(6, 0))
        e_bg = ttk.Entry(r, textvariable=self._bg_alpha_str, width=4)
        e_bg.pack(side=tk.LEFT, padx=1)
        e_bg.bind("<Return>", lambda e: self._apply_bg_alpha())
        ttk.Button(r, text="✓", width=2, command=self._apply_bg_alpha).pack(side=tk.LEFT, padx=1)
        ttk.Button(r, text="↩清除", command=lambda: self.gui.action("reset_per_frame_alphas")).pack(side=tk.RIGHT, padx=(4, 0))

    def _apply_bg_alpha(self):
        try:
            v = float(self._bg_alpha_str.get())
            if 0.0 <= v <= 1.0:
                self.gui.action("set_bg_alpha", v)
                self.log(f"背景 alpha={v:.2f}")
        except ValueError:
            pass

    def _apply_alpha_gradient(self):
        """应用首/末帧 alpha 渐变设置。"""
        try:
            a0 = float(self._alpha_start_str.get())
            a1 = float(self._alpha_end_str.get())
            if 0.0 <= a0 <= 1.0 and 0.0 <= a1 <= 1.0:
                self.gui.alpha_start = a0
                self.gui.alpha_end = a1
                self.gui._data_version += 1   # ★ 触发帧列表重建，刷新 alpha 列
                self.gui._preview_dirty = True
                self.log(f"Alpha 渐变: {a0:.2f} → {a1:.2f}")
        except ValueError:
            pass

    def _popup_alpha_editor(self, tree, item, fidx, col_idx):
        """弹出小窗编辑逐帧 alpha 值。"""
        top = tk.Toplevel(self.root)
        top.title(f"帧{fidx} alpha")
        top.geometry("160x70+%d+%d" % (self.root.winfo_x() + 150, self.root.winfo_y() + 200))
        top.transient(self.root)
        top.grab_set()
        cur = self.gui.get_frame_alpha(fidx)
        var = tk.StringVar(value=f"{cur:.2f}")
        e = ttk.Entry(top, textvariable=var, width=6, font=("", 12))
        e.pack(pady=6)
        e.select_range(0, tk.END)
        e.focus_set()
        def _apply():
            try:
                v = float(var.get())
                if 0.0 <= v <= 1.0:
                    self.gui.action("set_per_frame_alpha", fidx, v)
                    # 更新 Treeview 单元格
                    vals = list(tree.item(item, "values"))
                    vals[col_idx] = f"{v:.2f}"
                    tree.item(item, values=vals)
                    top.destroy()
            except ValueError:
                pass
        ttk.Button(top, text="确定", command=_apply).pack()
        e.bind("<Return>", lambda e: _apply())

    # ── TRACKING ──
    def _build_tracking_actions(self) -> None:
        self.progress_var = tk.DoubleVar(value=0)
        self.progress_bar = ttk.Progressbar(self.actions_inner, variable=self.progress_var,
                                             length=260, mode="determinate")
        self.progress_bar.pack(fill=tk.X, pady=2)
        self.lbl_progress = ttk.Label(self.actions_inner, text="0%")
        self.lbl_progress.pack()
        ttk.Button(self.actions_inner, text="⏹ 中止 (Esc)",
                   command=lambda: self.gui.action("abort_tracking")).pack(fill=tk.X, pady=2)

    # ── 合成帧列表（Treeview 表格）──
    def _build_marked_list(self) -> None:
        import re
        frames = sorted(self.gui.composite_frames)
        objs = self.gui.objects
        n_objs = len(objs)

        # 所有行：标记帧 + 背景帧
        all_rows = list(frames)
        bg = self.gui.background_frame_idx
        if bg not in all_rows:
            all_rows.append(bg)
        all_rows.sort()

        if not all_rows:
            ttk.Label(self.marked_inner, text="（暂无标记帧，按 K 标记）", foreground="gray").pack()
            return

        # ── 列定义: [行选] [帧标签] [obj1] [obj2]... [α] [✕] ──
        columns = ["row"] + ["frame"] + [f"obj_{o.obj_id}" for o in objs] + ["alpha", "del"]
        tree = ttk.Treeview(self.marked_inner, columns=columns, show="headings",
                            selectmode="browse", height=6)

        # Col 0: 行选框
        tree.heading("row", text="")
        tree.column("row", width=20, anchor=tk.CENTER, stretch=True, minwidth=18)
        tree.heading("frame", text=f"帧 / 时间 ({len(frames)})")
        tree.column("frame", width=80, anchor=tk.W, stretch=True, minwidth=60)
        for obj in objs:
            col = f"obj_{obj.obj_id}"
            tree.heading(col, text=f"☑ {obj.name}",
                         command=lambda oid=obj.obj_id: self._col_toggle_all(tree, oid))
            tree.column(col, width=46, anchor=tk.CENTER, stretch=True, minwidth=36)
        tree.heading("alpha", text="α")
        tree.column("alpha", width=46, anchor=tk.CENTER, stretch=True, minwidth=38)
        tree.heading("del", text="")
        tree.column("del", width=26, anchor=tk.CENTER, stretch=True, minwidth=22)

        # Scrollbar
        sb = ttk.Scrollbar(self.marked_inner, orient=tk.VERTICAL, command=tree.yview)
        tree.configure(yscrollcommand=sb.set)
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb.pack(side=tk.RIGHT, fill=tk.Y)

        # ── 填充数据 ──
        fps = max(self.gui.fps, 1)
        for fidx in all_rows:
            t_sec = fidx / fps
            ts = f"{int(t_sec // 60)}min {t_sec % 60:.2f}s"
            is_bg = (fidx == bg)
            is_excluded = fidx in self.gui._excluded_frames
            label = f"帧{fidx} {ts}" + (" [BG]" if is_bg else "") + (" [排除]" if is_excluded else "")

            # 行选 ☑/☐
            row_val = "☐" if is_excluded else "☑"
            values = [row_val, label]
            for obj in objs:
                has_mask = self.gui.masks.get(fidx, {}).get(obj.obj_id) is not None
                override = self.gui.frame_overrides.get(fidx, {}).get(obj.obj_id)
                checked = override if override is not None else True
                values.append("☑" if (has_mask and checked) else ("☐" if has_mask else "—"))
            cur_a = self.gui.get_frame_alpha(fidx)
            values.append(f"{cur_a:.2f}")
            values.append("✕")

            item_id = tree.insert("", tk.END, values=values)
            if is_excluded:
                tree.item(item_id, tags=("excluded",))
            elif is_bg:
                tree.item(item_id, tags=("bg",))

        tree.tag_configure("excluded", foreground="#aaaaaa")
        tree.tag_configure("bg", foreground="#CC6600")

        # ── 点击处理 ──
        def _on_click(event):
            region = tree.identify_region(event.x, event.y)
            if region != "cell":
                return
            col = tree.identify_column(event.x)
            item = tree.identify_row(event.y)
            if not item:
                return
            values = tree.item(item, "values")
            if not values:
                return
            label = values[1]  # 帧标签在第1列（0=行选）
            m = re.match(r"帧(\d+)", label)
            if not m:
                return
            fidx = int(m.group(1))
            col_idx = int(col[1:]) - 1  # "#0" → 0

            if col_idx == 0:
                # ★ 行选框：toggle _excluded_frames
                if fidx in self.gui._excluded_frames:
                    self.gui._excluded_frames.discard(fidx)
                else:
                    self.gui._excluded_frames.add(fidx)
                self.gui._data_version += 1
                self.gui._preview_dirty = True
            elif 2 <= col_idx <= n_objs + 1:
                # ★ 物体列：toggle 可见性
                obj = objs[col_idx - 2]
                self.gui.action("toggle_frame_object_at", fidx, obj.obj_id)
                has_mask = self.gui.masks.get(fidx, {}).get(obj.obj_id) is not None
                override = self.gui.frame_overrides.get(fidx, {}).get(obj.obj_id)
                checked = override if override is not None else True
                vals = list(tree.item(item, "values"))
                vals[col_idx] = "☑" if (has_mask and checked) else ("☐" if has_mask else "—")
                tree.item(item, values=vals)
            elif col_idx == n_objs + 2:
                # Alpha 列：弹出编辑
                self._popup_alpha_editor(tree, item, fidx, col_idx)
            elif col_idx == n_objs + 3:
                # 删除
                self.gui.composite_frames.discard(fidx)
                self.gui._data_version += 1
                self.gui._preview_dirty = True

        def _on_double(event):
            region = tree.identify_region(event.x, event.y)
            if region != "cell":
                return
            item = tree.identify_row(event.y)
            if not item:
                return
            label = tree.item(item, "values")[1]
            m = re.match(r"帧(\d+)", label)
            if m:
                self._jump_to_frame(int(m.group(1)))

        tree.bind("<ButtonRelease-1>", _on_click)
        tree.bind("<Double-1>", _on_double)
        def _tree_wheel(e):
            tree.yview_scroll(int(-1 * e.delta / 120), "units")
        tree.bind("<MouseWheel>", _tree_wheel)

    def _col_toggle_all(self, tree, obj_id: int) -> None:
        """列头点击：全选/全不选该物体的所有帧。"""
        # 判断当前是否全部选中
        all_checked = True
        for fidx in self.gui.composite_frames:
            has_mask = self.gui.masks.get(fidx, {}).get(obj_id) is not None
            if not has_mask:
                continue
            override = self.gui.frame_overrides.get(fidx, {}).get(obj_id)
            if override is not None and not override:
                all_checked = False
                break
            if override is None:
                continue  # 默认 True
        # 全不选 or 部分选 → 全选；全选 → 全不选
        new_val = not all_checked
        for fidx in self.gui.composite_frames:
            if self.gui.masks.get(fidx, {}).get(obj_id) is not None:
                if fidx not in self.gui.frame_overrides:
                    self.gui.frame_overrides[fidx] = {}
                self.gui.frame_overrides[fidx][obj_id] = new_val
        self.gui._data_version += 1
        self.gui._preview_dirty = True

    def _jump_to_frame(self, fidx: int) -> None:
        """双击帧列表跳转到该帧。"""
        self.gui.current_frame_idx = fidx
        self.gui._preview_dirty = True
        self.gui._preview_mask = None

    def _toggle_all_rows(self, show: bool) -> None:
        if not show:
            # 保存一份以便后续"全选"恢复（最佳努力）
            self._saved_composite = set(self.gui.composite_frames)
            self._saved_overrides = {f: dict(o) for f, o in self.gui.frame_overrides.items()}
            self.gui.composite_frames.clear()
            self.gui.frame_overrides.clear()
        elif hasattr(self, "_saved_composite") and self._saved_composite:
            self.gui.composite_frames = self._saved_composite
            self.gui.frame_overrides = self._saved_overrides
            self._saved_composite = set()
            self._saved_overrides = {}
        self.gui._preview_dirty = True

    def _toggle_all_frames(self, obj_id: int, show: bool) -> None:
        for fidx in self.gui.composite_frames:
            if self.gui.masks.get(fidx, {}).get(obj_id) is not None:
                if fidx not in self.gui.frame_overrides:
                    self.gui.frame_overrides[fidx] = {}
                self.gui.frame_overrides[fidx][obj_id] = show
        self.gui._preview_dirty = True
        self.gui._data_version += 1

    def _apply_frame_alpha(self, fidx, entry):
        try:
            val = float(entry.get())
            if 0.0 <= val <= 1.0:
                self.gui.action("set_per_frame_alpha", fidx, val)
                self.log(f"帧{fidx} alpha={val:.2f}")
        except ValueError:
            pass

    # ==================================================================
    # 同步
    # ==================================================================
    def sync_from_gui(self) -> None:
        gui = self.gui
        need_rebuild = (
            gui.state != self._last_state
            or gui.active_obj_idx != self._last_active_idx
            or len(gui.objects) != self._last_obj_count
            or len(gui.composite_frames) != self._last_marked_count
            or gui._data_version != self._last_data_version
        )
        if need_rebuild:
            self._rebuild_object_buttons()
            self._build_dynamic()
            self._last_marked_count = len(gui.composite_frames)
            self._last_data_version = gui._data_version

        state_text = "跟踪中" if gui.state == GUIState.TRACKING else "编辑"
        self.lbl_state.configure(text=state_text)
        fps_info = f" proc-fps={gui.args.process_fps:.0f}" if gui.args.process_fps else ""
        self.lbl_frame.configure(text=f"帧: {gui.current_frame_idx}/{gui.n_frames}{fps_info}")

        if gui.active_object():
            obj = gui.active_object()
            n_pts = len(obj.points)
            dirty_mark = " +" if obj._dirty and gui.state != GUIState.TRACKING else ""
            self.lbl_active.configure(text=f"活跃: {obj.name} ({n_pts}点, 种子{obj.seed_frame}){dirty_mark}")
        else:
            self.lbl_active.configure(text="活跃: —")

        for i, btn in enumerate(getattr(self, "_obj_btns", [])):
            btn.configure(bg="white", relief=tk.RAISED, font=("", 9, "bold")) if i == gui.active_obj_idx else \
                btn.configure(bg="#e0e0e0", relief=tk.FLAT, font=("", 9))

        for i, del_btn in enumerate(getattr(self, "_del_btns", [])):
            del_btn.configure(state=tk.DISABLED if i == gui.active_obj_idx and gui.state == GUIState.TRACKING else tk.NORMAL)

        if hasattr(self, "view_var") and self.view_var.get() != gui.viz_mode:
            self.view_var.set(gui.viz_mode)

        if hasattr(self, "_mark_btn"):
            marked = gui.current_frame_idx in gui.composite_frames
            try:
                self._mark_btn.configure(text="✓ K取消" if marked else "K 标记")
            except tk.TclError:
                pass

        # 选点模式按钮
        if hasattr(self, "_pt_btn"):
            try:
                if gui._point_mode_active:
                    self._pt_btn.configure(text="⏹ 退出选点")
                else:
                    self._pt_btn.configure(text="▶ 开始选点")
            except tk.TclError:
                pass

        # R 按钮：显示范围起点状态
        if hasattr(self, "_range_btn"):
            try:
                if gui._range_start is not None:
                    self._range_btn.configure(text=f"R 终点(从{gui._range_start})")
                elif self._range_btn.cget("text") != "R 范围":
                    self._range_btn.configure(text="R 范围")
            except tk.TclError:
                pass

    def _rebuild_object_buttons(self) -> None:
        for w in self.obj_grid_frame.winfo_children(): w.destroy()
        self._obj_btns = []
        self._del_btns = []

        COLS = 2
        for i, obj in enumerate(self.gui.objects):
            row, col = i // COLS, (i % COLS) * 2
            is_active = (i == self.gui.active_obj_idx)
            n_pts = len(obj.points)

            st = "+" if obj._dirty and n_pts else ("✓" if n_pts else "○")
            label = f"{st} {obj.name}"
            if n_pts:
                label += f" ({n_pts}p)"

            btn = tk.Button(self.obj_grid_frame, text=label, fg=obj.color_hex,
                            bg="white" if is_active else "#e0e0e0", activebackground="#c0c0ff",
                            relief=tk.RAISED if is_active else tk.FLAT,
                            font=("", 8, "bold") if is_active else ("", 8),
                            command=lambda idx=i: self.gui.action("select_object", idx))
            btn.grid(row=row, column=col, padx=1, pady=1, sticky="ew")
            self._obj_btns.append(btn)

            del_btn = tk.Button(self.obj_grid_frame, text="✕", fg="#cc0000",
                                bg="#ffe0e0" if is_active else "#e0e0e0", activebackground="#ffcccc",
                                relief=tk.FLAT, font=("", 8),
                                command=lambda idx=i: self.gui.action("delete_object", idx))
            del_btn.grid(row=row, column=col + 1, padx=1, pady=1)
            self._del_btns.append(del_btn)

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

    # ==================================================================
    # 键盘转发（bind_all → 所有控件共享）
    # ==================================================================
    def _on_tk_key(self, event: tk.Event) -> None:
        gui = self.gui
        key = event.keysym; char = event.char; ctrl = event.state & 0x4

        # 在 Entry 中打字时不拦截
        if isinstance(event.widget, ttk.Entry):
            return  # 让 Entry 正常处理输入

        if key == "Return":
            # Enter 只在非 Entry 控件时触发跟踪
            if gui.state == GUIState.EDIT and not isinstance(event.widget, ttk.Entry):
                gui.action("start_tracking")
        elif key == "BackSpace":
            if not isinstance(event.widget, ttk.Entry):
                obj = gui.active_object()
                if obj and obj.points:
                    obj.points.pop(); obj._dirty = True; gui._preview_dirty = True
        elif key == "Delete":
            if not isinstance(event.widget, ttk.Entry):
                gui.action("delete_object", gui.active_obj_idx)
        elif key in ("Left", "Right"):
            if ctrl:
                gui.action("prev_marked" if key == "Left" else "next_marked")
            else:
                gui.current_frame_idx = (gui.current_frame_idx + (-1 if key == "Left" else 1)) % gui.n_frames
                gui._preview_dirty = True
        elif char.lower() == "k": gui.action("mark_frame")
        elif char.lower() == "b": gui.action("set_bg")
        elif char.lower() == "v":
            cycle = {"mask": "composite", "composite": "original", "original": "mask"}
            gui.action("view_" + cycle[gui.viz_mode])
        elif char.lower() == "i": gui.action("apply_interval")
        elif char.lower() == "s": gui.action("save")
        elif char.lower() == "n": gui.action("new_object")
        elif char.lower() == "p": gui.action("preview_frame")
        elif char.lower() == "r": gui.action("range_select")
        elif char in "123456789":
            idx = int(char) - 1
            if idx < len(gui.objects): gui.action("select_object", idx)
        elif key == "Tab":
            if gui.objects: gui.active_obj_idx = (gui.active_obj_idx + 1) % len(gui.objects)

    def _on_close(self) -> None:
        self.gui.action("quit")

    def destroy(self) -> None:
        try:
            self.root.destroy()
        except tk.TclError:
            pass
