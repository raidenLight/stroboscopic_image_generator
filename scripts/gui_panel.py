"""gui_panel.py — tkinter 控制面板。"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk, messagebox
from typing import TYPE_CHECKING

from gui_types import GUIState

if TYPE_CHECKING:
    from gui_app import StroboscopicGUI

PANEL_WIDTH = 400
PANEL_HEIGHT = 600


class ControlPanel:
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

        self._last_state: GUIState | None = None
        self._last_active_idx: int = -1
        self._last_obj_count: int = -1
        self._last_marked_count: int = -1
        self._interval_str = tk.StringVar(value="1.5")
        self._alpha_start_str = tk.StringVar(value=f"{gui.args.alpha:.2f}")
        self._alpha_end_str = tk.StringVar(value=f"{gui.args.alpha:.2f}")

        self._build_static()
        self._build_dynamic()
        self.root.update()

    # ==================================================================
    # 静态控件
    # ==================================================================
    def _build_static(self) -> None:
        # ── 状态（固定顶部）──
        frm = ttk.LabelFrame(self.root, text="状态", padding=3)
        frm.pack(fill=tk.X, padx=3, pady=1)
        self.lbl_state = ttk.Label(frm, text="编辑", font=("", 10, "bold"))
        self.lbl_state.pack(anchor=tk.W)
        lf = ttk.Frame(frm)
        lf.pack(fill=tk.X)
        self.lbl_frame = ttk.Label(lf, text="帧: 0 / 0")
        self.lbl_frame.pack(side=tk.LEFT)
        self.lbl_active = ttk.Label(lf, text="活跃: —")
        self.lbl_active.pack(side=tk.RIGHT)

        # ── 物体 ──
        self.frm_objects = ttk.LabelFrame(self.root, text="物体", padding=3)
        self.frm_objects.pack(fill=tk.X, padx=3, pady=1)
        self.obj_grid_frame = ttk.Frame(self.frm_objects)
        self.obj_grid_frame.pack(fill=tk.X)
        self.btn_new_obj = ttk.Button(self.frm_objects, text="+ 新建物体 (N)",
                                       command=lambda: self.gui.action("new_object"))
        self.btn_new_obj.pack(fill=tk.X, pady=1)

        # ── 操作 ──
        self.frm_actions = ttk.LabelFrame(self.root, text="操作", padding=3)
        self.frm_actions.pack(fill=tk.X, padx=3, pady=1)
        self.actions_inner = ttk.Frame(self.frm_actions)
        self.actions_inner.pack(fill=tk.X)

        # ── 视图 + 帧选取（合并一行）──
        self.frm_tools = ttk.LabelFrame(self.root, text="视图 & 帧选取", padding=3)
        self.frm_tools.pack(fill=tk.X, padx=3, pady=1)
        self.tools_inner = ttk.Frame(self.frm_tools)
        self.tools_inner.pack(fill=tk.X)

        # ── Alpha ──
        self.frm_alpha = ttk.LabelFrame(self.root, text="Alpha 渐变", padding=3)
        self.frm_alpha.pack(fill=tk.X, padx=3, pady=1)
        self.alpha_inner = ttk.Frame(self.frm_alpha)
        self.alpha_inner.pack(fill=tk.X)

        # ── 合成帧列表（展开占满剩余空间）──
        self.frm_marked = ttk.LabelFrame(self.root, text="合成帧", padding=3)
        self.frm_marked.pack(fill=tk.BOTH, expand=True, padx=3, pady=1)
        self.marked_inner = ttk.Frame(self.frm_marked)
        self.marked_inner.pack(fill=tk.BOTH, expand=True)

        # ── 可见范围 ──
        self.frm_vis = ttk.LabelFrame(self.root, text="可见范围", padding=3)

        # ── 底部（固定）──
        bottom = ttk.Frame(self.root)
        bottom.pack(fill=tk.X, padx=3, pady=2)
        ttk.Label(bottom, text="K标记 | I间隔 | R范围 | B背景 | V视图 | ←→导航",
                   font=("", 7)).pack(side=tk.LEFT, anchor=tk.S)
        self.btn_restart = ttk.Button(bottom, text="↺ 重置全部",
                                       command=lambda: self.gui.action("restart"))
        self.btn_restart.pack(side=tk.RIGHT, padx=2)
        ttk.Button(bottom, text="退出 (Esc)",
                   command=lambda: self.gui.action("quit")).pack(side=tk.RIGHT, padx=2)

    # ==================================================================
    # 动态控件
    # ==================================================================
    def _build_dynamic(self) -> None:
        self._last_state = self.gui.state
        self._last_active_idx = self.gui.active_obj_idx
        self._last_obj_count = len(self.gui.objects)

        for w in self.actions_inner.winfo_children(): w.destroy()
        for w in self.tools_inner.winfo_children(): w.destroy()
        for w in self.alpha_inner.winfo_children(): w.destroy()
        for w in self.marked_inner.winfo_children(): w.destroy()

        if self.gui.state == GUIState.TRACKING:
            self._build_tracking_actions()
            self._build_vis_range(editable=False)
        else:
            self._build_edit_actions()
            self._build_tools_section()
            self._build_alpha_section()
            self._build_marked_list()
            self._build_vis_range(editable=True)

    # ── EDIT 操作 ──
    def _build_edit_actions(self) -> None:
        r = ttk.Frame(self.actions_inner)
        r.pack(fill=tk.X)
        ttk.Button(r, text="👁 预览 (P)", command=lambda: self.gui.action("preview_frame")).pack(side=tk.LEFT, padx=1)
        ttk.Button(r, text="✕ 清点 (Backspace)", command=lambda: self.gui.action("clear_points")).pack(side=tk.LEFT, padx=1)

        dirty = [o for o in self.gui.objects if o._dirty and o.points]
        if dirty:
            btn = tk.Button(self.actions_inner, text=f"▶ 开始跟踪 ({len(dirty)} 物体)", bg="#4CAF50",
                            fg="white", font=("", 10, "bold"), relief=tk.RAISED,
                            command=lambda: self.gui.action("start_tracking"))
        else:
            btn = tk.Button(self.actions_inner, text="▶ 开始跟踪", bg="#cccccc", fg="#888888",
                            font=("", 10), relief=tk.FLAT, state=tk.DISABLED)
        btn.pack(fill=tk.X, pady=2, ipady=2)

    # ── 视图 + 帧选取 ──
    def _build_tools_section(self) -> None:
        # 视图单选
        self.view_var = tk.StringVar(value=self.gui.viz_mode)
        vf = ttk.Frame(self.tools_inner)
        vf.pack(fill=tk.X, pady=1)
        for mode, label in [("mask", "Mask"), ("composite", "合成"), ("original", "原图")]:
            ttk.Radiobutton(vf, text=label, variable=self.view_var, value=mode,
                            command=lambda m=mode: self.gui.action("view_" + m)).pack(side=tk.LEFT, padx=2)

        # 标记 + 范围 + 背景 (一行)
        sf = ttk.Frame(self.tools_inner)
        sf.pack(fill=tk.X, pady=1)
        self._mark_btn = ttk.Button(sf, text="K 标记", command=lambda: self.gui.action("mark_frame"))
        self._mark_btn.pack(side=tk.LEFT, padx=1, fill=tk.X, expand=True)
        ttk.Button(sf, text="B 背景", command=lambda: self.gui.action("set_bg")).pack(side=tk.LEFT, padx=1, fill=tk.X, expand=True)
        ttk.Button(sf, text="R 范围", command=lambda: self.gui.action("range_select")).pack(side=tk.LEFT, padx=1, fill=tk.X, expand=True)

        # 间隔
        intf = ttk.Frame(self.tools_inner)
        intf.pack(fill=tk.X, pady=1)
        ttk.Label(intf, text="间隔(秒):").pack(side=tk.LEFT)
        ttk.Entry(intf, textvariable=self._interval_str, width=5).pack(side=tk.LEFT, padx=2)
        ttk.Button(intf, text="应用 (I)", command=lambda: self.gui.action("apply_interval")).pack(side=tk.LEFT, padx=2)

    # ── Alpha ──
    def _build_alpha_section(self) -> None:
        r = ttk.Frame(self.alpha_inner)
        r.pack(fill=tk.X)
        ttk.Label(r, text="首帧:").pack(side=tk.LEFT)
        e1 = ttk.Entry(r, textvariable=self._alpha_start_str, width=5)
        e1.pack(side=tk.LEFT, padx=2)
        e1.bind("<Return>", lambda e: self._apply_alpha("set_alpha_start", self._alpha_start_str))
        ttk.Label(r, text="末帧:").pack(side=tk.LEFT, padx=(4, 0))
        e2 = ttk.Entry(r, textvariable=self._alpha_end_str, width=5)
        e2.pack(side=tk.LEFT, padx=2)
        e2.bind("<Return>", lambda e: self._apply_alpha("set_alpha_end", self._alpha_end_str))
        ttk.Button(r, text="应用", command=self._apply_alpha_gradient).pack(side=tk.LEFT, padx=4)
        ttk.Button(r, text="重置逐帧", command=lambda: self.gui.action("reset_per_frame_alphas")).pack(side=tk.RIGHT)

    def _apply_alpha(self, action_name, var):
        try:
            val = float(var.get())
            if 0.0 <= val <= 1.0:
                self.gui.action(action_name, val)
        except ValueError:
            pass

    def _apply_alpha_gradient(self):
        try:
            a0 = float(self._alpha_start_str.get())
            a1 = float(self._alpha_end_str.get())
            if 0.0 <= a0 <= 1.0 and 0.0 <= a1 <= 1.0:
                self.gui.alpha_start = a0
                self.gui.alpha_end = a1
                self.gui._preview_dirty = True
        except ValueError:
            pass

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

    # ── 合成帧列表（Canvas 滚动 + checkbox 网格，展开填满）──
    def _build_marked_list(self) -> None:
        frames = sorted(self.gui.composite_frames)
        count = len(frames)
        objs = self.gui.objects
        n_objs = len(objs)

        # 所有行：标记帧 + 背景帧
        all_rows = list(frames)
        if self.gui.background_frame_idx not in all_rows:
            all_rows.append(self.gui.background_frame_idx)
        all_rows.sort()
        n_rows = len(all_rows)

        if n_rows == 0:
            ttk.Label(self.marked_inner, text="（暂无标记帧，按 K 标记）", foreground="gray").pack()
            return

        # Canvas + scrollbar（填满 frm_marked 空间）
        cvs = tk.Canvas(self.marked_inner, highlightthickness=0)
        sb = ttk.Scrollbar(self.marked_inner, orient=tk.VERTICAL, command=cvs.yview)
        inner = ttk.Frame(cvs)
        inner.bind("<Configure>", lambda e: cvs.configure(scrollregion=cvs.bbox("all")))
        cvs.create_window((0, 0), window=inner, anchor="nw")
        cvs.configure(yscrollcommand=sb.set)
        cvs.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb.pack(side=tk.RIGHT, fill=tk.Y)

        # 绑定滚轮
        def _wheel(e):
            cvs.yview_scroll(int(-1 * e.delta / 120), "units")
        cvs.bind("<MouseWheel>", _wheel)
        inner.bind("<MouseWheel>", _wheel)
        cvs.bind("<Enter>", lambda e: cvs.focus_set())

        # ── 头行 ──
        ttk.Label(inner, text=f"帧/时间 ({count})", font=("", 8, "bold"), anchor=tk.W).grid(row=0, column=0, sticky="w")
        for j, obj in enumerate(objs):
            cb_var = tk.BooleanVar(value=True)
            cb = ttk.Checkbutton(inner, variable=cb_var)
            cb.grid(row=0, column=j + 1)
            cb.configure(command=lambda oid=obj.obj_id, v=cb_var: self._toggle_all_frames(oid, v.get()))
            ttk.Label(inner, text=obj.name[:3], foreground=obj.color_hex, font=("", 6)).grid(
                row=0, column=j + 1, sticky="s", pady=(14, 0))
        ttk.Label(inner, text="α", width=4, anchor=tk.CENTER, font=("", 8, "bold")).grid(row=0, column=n_objs + 1)

        # ── 数据行 ──
        for i, fidx in enumerate(all_rows):
            row = i + 1
            t_sec = fidx / max(self.gui.fps, 1)
            ts = f"{int(t_sec // 60)}:{int(t_sec % 60):02d}"
            is_bg = (fidx == self.gui.background_frame_idx)
            label_text = f" 帧{fidx} {ts}s" + (" [BG]" if is_bg else "")
            ttk.Label(inner, text=label_text, foreground="#CC6600" if is_bg else "black",
                      font=("", 8, "bold" if is_bg else "normal")).grid(row=row, column=0, sticky="w")

            for j, obj in enumerate(objs):
                has_mask = self.gui.masks.get(fidx, {}).get(obj.obj_id) is not None
                override = self.gui.frame_overrides.get(fidx, {}).get(obj.obj_id)
                checked = override if override is not None else True
                if has_mask:
                    var = tk.BooleanVar(value=checked)
                    cb = ttk.Checkbutton(inner, variable=var)
                    cb.grid(row=row, column=j + 1)
                    cb.configure(command=lambda f=fidx, o=obj.obj_id: (
                        setattr(self.gui, '_preview_dirty', True),
                        self.gui.action("toggle_frame_object_at", f, o)
                    ))
                else:
                    ttk.Label(inner, text="—", font=("", 7)).grid(row=row, column=j + 1)

            # Alpha
            cur_a = self.gui.get_frame_alpha(fidx)
            a_var = tk.StringVar(value=f"{cur_a:.2f}")
            a_entry = ttk.Entry(inner, width=5, textvariable=a_var, font=("", 8))
            a_entry.grid(row=row, column=n_objs + 1, padx=1)
            a_entry.bind("<Return>", lambda e, f=fidx: self._apply_frame_alpha(f, e.widget))

            # 删除
            ttk.Button(inner, text="✕", width=2, command=lambda f=fidx: (
                self.gui.composite_frames.discard(f),
                setattr(self.gui, '_preview_dirty', True)
            )).grid(row=row, column=n_objs + 2)

    def _toggle_all_frames(self, obj_id: int, show: bool) -> None:
        for fidx in self.gui.composite_frames:
            if self.gui.masks.get(fidx, {}).get(obj_id) is not None:
                if fidx not in self.gui.frame_overrides:
                    self.gui.frame_overrides[fidx] = {}
                self.gui.frame_overrides[fidx][obj_id] = show
        self.gui._preview_dirty = True

    def _apply_frame_alpha(self, fidx, entry):
        try:
            val = float(entry.get())
            if 0.0 <= val <= 1.0:
                self.gui.action("set_per_frame_alpha", fidx, val)
        except ValueError:
            pass

    # ── 可见范围 ──
    def _build_vis_range(self, editable: bool) -> None:
        self.frm_vis.pack_forget()
        for w in self.frm_vis.winfo_children(): w.destroy()

        if not self.gui.objects or self.gui.active_object() is None:
            self.frm_vis.pack(fill=tk.X, padx=3, pady=1, after=self.frm_marked)
            ttk.Label(self.frm_vis, text="无活跃物体").pack()
            return

        obj = self.gui.active_object()
        self.frm_vis.configure(text=f"可见范围 ({obj.name})")
        n = max(self.gui.n_frames - 1, 1)
        self.vis_start_var = tk.IntVar(value=obj.vis_start if obj.vis_start is not None else 0)
        self.vis_end_var = tk.IntVar(value=obj.vis_end if obj.vis_end is not None else n)

        for label, var_attr in [("始", "vis_start_var"), ("末", "vis_end_var")]:
            frm = ttk.Frame(self.frm_vis)
            frm.pack(fill=tk.X)
            ttk.Label(frm, text=label, width=2).pack(side=tk.LEFT)
            scale = ttk.Scale(frm, from_=0, to=n, variable=getattr(self, var_attr),
                              orient=tk.HORIZONTAL, length=240)
            scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
            if not editable:
                scale.configure(state=tk.DISABLED)
            else:
                scale.configure(command=self._on_vis_change)
            ttk.Label(frm, textvariable=getattr(self, var_attr), width=4).pack(side=tk.LEFT)

        if editable:
            bf = ttk.Frame(self.frm_vis)
            bf.pack(fill=tk.X, pady=1)
            ttk.Button(bf, text="应用范围", command=lambda: self.gui.action("apply_vis_range")).pack(side=tk.LEFT, padx=1)
            ttk.Button(bf, text="重置全部帧", command=lambda: self.gui.action("reset_vis_range")).pack(side=tk.LEFT, padx=1)

        self.frm_vis.pack(fill=tk.X, padx=3, pady=1, after=self.frm_marked)

    def _on_vis_change(self, *args) -> None:
        obj = self.gui.active_object()
        if obj and hasattr(self, "vis_start_var") and hasattr(self, "vis_end_var"):
            obj.vis_start = self.vis_start_var.get()
            obj.vis_end = self.vis_end_var.get()
            self.gui._preview_dirty = True

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
        )
        if need_rebuild:
            self._rebuild_object_buttons()
            self._build_dynamic()
            self._last_marked_count = len(gui.composite_frames)

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
            self._mark_btn.configure(text="✓ K取消" if marked else "K 标记")

        if gui.state == GUIState.EDIT:
            if self._alpha_start_str.get() != f"{gui.alpha_start:.2f}":
                self._alpha_start_str.set(f"{gui.alpha_start:.2f}")
            if self._alpha_end_str.get() != f"{gui.alpha_end:.2f}":
                self._alpha_end_str.set(f"{gui.alpha_end:.2f}")

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
    # 键盘转发
    # ==================================================================
    def _on_tk_key(self, event: tk.Event) -> None:
        gui = self.gui
        key = event.keysym; char = event.char; ctrl = event.state & 0x4

        if key == "Escape":
            gui.action("quit")
        elif key == "Return" or key == "space":
            if gui.state == GUIState.EDIT:
                gui.action("start_tracking")
        elif key == "BackSpace":
            obj = gui.active_object()
            if obj and obj.points:
                obj.points.pop(); obj._dirty = True; gui._preview_dirty = True
        elif key == "Delete":
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
        elif char.lower() == "[" and gui.active_object():
            gui.active_object().vis_start = gui.current_frame_idx; gui._preview_dirty = True
        elif char.lower() == "]" and gui.active_object():
            gui.active_object().vis_end = gui.current_frame_idx; gui._preview_dirty = True
        elif char.lower() == "\\": gui.action("reset_vis_range")
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
