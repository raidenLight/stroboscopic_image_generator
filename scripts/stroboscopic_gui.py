#!/usr/bin/env python3
"""
Interactive GUI for stroboscopic image generation with SAM2 multi-object tracking.

Dual-window architecture:
  - OpenCV window: video display + timeline + mask/point overlays
  - tkinter panel: native buttons, sliders, labels for all controls (zero extra deps)

Workflow:
  1. SETUP:  mark one or more objects on any frames, click points, press Start.
  2. TRACKING: SAM2 tracks all objects forward+backward simultaneously.
  3. SELECTION: mark frames (K / interval / range), set visibility per object,
                toggle preview, save composite.
"""

from __future__ import annotations

import argparse
import contextlib
import sys
import tempfile
import tkinter as tk
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from tkinter import ttk
from typing import Callable

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Import reusable helpers from the original script
# ---------------------------------------------------------------------------
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from stroboscopic_image_generator import (  # noqa: E402
    build_predictor,
    clean_mask,
    find_object_index,
    get_frame_count,
    logits_to_mask,
    maybe_downsample_video,
    read_frame_at,
    resolve_device,
    sam_inference_context,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ROOT_DIR = _THIS_DIR.parent
DEFAULT_VIDEO = ROOT_DIR / "video" / "exp1_dwvp.MP4"
DEFAULT_OUT = ROOT_DIR / "video" / "stroboscopic_gui.png"

WINDOW_NAME = "Stroboscopic Generator (SAM2)"
TRACKBAR_FRAME = "Frame"
TRACKBAR_ALPHA = "Alpha"

TIMELINE_H = 40

OBJECT_COLORS: list[tuple[int, int, int]] = [
    (0, 0, 255),     # red
    (255, 0, 0),     # blue
    (0, 255, 0),     # green
    (0, 255, 255),   # yellow
    (255, 255, 0),   # cyan
    (255, 0, 255),   # magenta
    (0, 165, 255),   # orange
    (255, 255, 255), # white
]
OBJECT_COLOR_HEX = [
    "#FF0000", "#0000FF", "#00FF00", "#FFFF00",
    "#00FFFF", "#FF00FF", "#FFA500", "#FFFFFF",
]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
class GUIState(Enum):
    SETUP = auto()
    TRACKING = auto()
    SELECTION = auto()
    SAVE = auto()


@dataclass
class TrackObject:
    obj_id: int
    color: tuple[int, int, int]
    color_hex: str
    name: str
    seed_frame: int = 0
    points: list[tuple[int, int]] = field(default_factory=list)
    vis_start: int | None = None   # None = from first frame
    vis_end: int | None = None     # None = to last frame


# ---------------------------------------------------------------------------
# tkinter Control Panel
# ---------------------------------------------------------------------------
class ControlPanel:
    """Floating tkinter window with native controls for all GUI actions."""

    def __init__(self, gui: 'StroboscopicGUI'):
        self.gui = gui
        self.root = tk.Tk()
        self.root.title("Controls — Stroboscopic Generator")
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.resizable(True, True)
        self.root.geometry("320x620+50+50")  # positioned away from video
        self.root.attributes('-topmost', True)  # always visible
        self.root.lift()
        # NOTE: do NOT focus_force() — let OpenCV keep keyboard focus
        # Release topmost after 3s so it doesn't block other apps
        self.root.after(3000, lambda: self.root.attributes('-topmost', False))

        # Bind keyboard events -> forward to GUI so both windows work
        self.root.bind('<Key>', self._on_tk_key)

        # Style
        style = ttk.Style(self.root)
        style.theme_use("clam")

        # State tracking for widget rebuild avoidance
        self._last_state: GUIState | None = None
        self._last_active_idx: int = -1
        self._last_obj_count: int = -1
        self._interval_value = tk.DoubleVar(value=1.5)

        self._build_static()
        self._build_dynamic()
        self.root.update()

    # ------------------------------------------------------------------
    # Static widgets (always present)
    # ------------------------------------------------------------------
    def _build_static(self) -> None:
        # -- Status --
        frm = ttk.LabelFrame(self.root, text="Status", padding=5)
        frm.pack(fill=tk.X, padx=5, pady=2)
        self.lbl_state = ttk.Label(frm, text="SETUP", font=("", 10, "bold"))
        self.lbl_state.pack(anchor=tk.W)
        self.lbl_frame = ttk.Label(frm, text="Frame: 0 / 0")
        self.lbl_frame.pack(anchor=tk.W)
        self.lbl_active = ttk.Label(frm, text="Active: —")
        self.lbl_active.pack(anchor=tk.W)

        # -- Objects --
        self.frm_objects = ttk.LabelFrame(self.root, text="Objects", padding=5)
        self.frm_objects.pack(fill=tk.X, padx=5, pady=2)
        self.obj_buttons_frame = ttk.Frame(self.frm_objects)
        self.obj_buttons_frame.pack(fill=tk.X)
        self.btn_new_obj = ttk.Button(self.frm_objects, text="+ New Object",
                                       command=lambda: self.gui.action("new_object"))
        self.btn_new_obj.pack(fill=tk.X, pady=2)

        # -- Actions --
        self.frm_actions = ttk.LabelFrame(self.root, text="Actions", padding=5)
        self.frm_actions.pack(fill=tk.X, padx=5, pady=2)
        self.actions_frame = ttk.Frame(self.frm_actions)
        self.actions_frame.pack(fill=tk.X)

        # -- View mode --
        self.frm_view = ttk.LabelFrame(self.root, text="View", padding=5)
        self.frm_view.pack(fill=tk.X, padx=5, pady=2)
        self.view_var = tk.StringVar(value="mask")
        self.view_frame = ttk.Frame(self.frm_view)
        self.view_frame.pack(fill=tk.X)
        for mode, label in [("mask", "Mask"), ("composite", "Composite"), ("original", "Original")]:
            ttk.Radiobutton(self.view_frame, text=label, variable=self.view_var,
                            value=mode, command=lambda m=mode: self.gui.action("view_" + m)
                            ).pack(side=tk.LEFT, padx=3)

        # -- Keyboard hints --
        frm_keys = ttk.LabelFrame(self.root, text="Keyboard Shortcuts", padding=5)
        frm_keys.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(frm_keys, text="←→ Nav  |  K Mark  |  B BG  |  V View  |  S Save",
                   font=("", 8)).pack(anchor=tk.W)
        ttk.Label(frm_keys, text="N New  |  1-9 Obj  |  Enter Track  |  Esc Quit",
                   font=("", 8)).pack(anchor=tk.W)

        # -- Quit --
        ttk.Button(self.root, text="Quit (Esc)", command=lambda: self.gui.action("quit"))\
            .pack(fill=tk.X, padx=5, pady=5)

    # ------------------------------------------------------------------
    # Dynamic widgets (rebuild on state change)
    # ------------------------------------------------------------------
    def _build_dynamic(self) -> None:
        """(Re)build widgets that depend on current state."""
        self._last_state = self.gui.state
        self._last_active_idx = self.gui.active_obj_idx
        self._last_obj_count = len(self.gui.objects)

        # Clear dynamic frames
        for w in self.actions_frame.winfo_children():
            w.destroy()

        state = self.gui.state

        if state == GUIState.SETUP:
            self._build_setup_actions()
            self._build_vis_range(editable=False)
        elif state == GUIState.TRACKING:
            self._build_tracking_actions()
        elif state == GUIState.SELECTION:
            self._build_selection_actions()
            self._build_vis_range(editable=True)
        elif state == GUIState.SAVE:
            self._build_selection_actions()
            self._build_vis_range(editable=True)

    def _build_setup_actions(self) -> None:
        ttk.Button(self.actions_frame, text="✕ Clear Points",
                   command=lambda: self.gui.action("clear_points"))\
            .pack(fill=tk.X, pady=1)
        btn = ttk.Button(self.actions_frame, text="▶ Start Tracking",
                          command=lambda: self.gui.action("start_tracking"))
        btn.pack(fill=tk.X, pady=3)
        # Disable if no objects have points
        if not any(obj.points for obj in self.gui.objects):
            btn.configure(state=tk.DISABLED)

    def _build_tracking_actions(self) -> None:
        self.progress_var = tk.DoubleVar(value=0)
        self.progress_bar = ttk.Progressbar(
            self.actions_frame, variable=self.progress_var, length=200, mode="determinate"
        )
        self.progress_bar.pack(fill=tk.X, pady=2)
        self.lbl_progress = ttk.Label(self.actions_frame, text="0%")
        self.lbl_progress.pack()
        ttk.Button(self.actions_frame, text="Abort (Esc)",
                   command=lambda: self.gui.action("abort_tracking"))\
            .pack(fill=tk.X, pady=2)

    def _build_selection_actions(self) -> None:
        # Mark frame
        marked = self.gui.current_frame_idx in self.gui.composite_frames
        mark_text = "✓ Mark Frame" if marked else "K Mark Frame"
        ttk.Button(self.actions_frame, text=mark_text,
                   command=lambda: self.gui.action("mark_frame"))\
            .pack(fill=tk.X, pady=1)

        # Range selection
        rng_text = "R Range Select"
        if self.gui._range_start is not None:
            rng_text = f"R Range: {self.gui._range_start}→?"
        ttk.Button(self.actions_frame, text=rng_text,
                   command=lambda: self.gui.action("range_select"))\
            .pack(fill=tk.X, pady=1)

        # Interval
        frm_int = ttk.Frame(self.actions_frame)
        frm_int.pack(fill=tk.X, pady=2)
        ttk.Label(frm_int, text="Interval:").pack(side=tk.LEFT)
        scale = ttk.Scale(frm_int, from_=0.1, to=10.0, variable=self._interval_value,
                          orient=tk.HORIZONTAL, length=120)
        scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=3)
        ttk.Label(frm_int, textvariable=self._interval_value, width=4).pack(side=tk.LEFT)

        frm_int2 = ttk.Frame(self.actions_frame)
        frm_int2.pack(fill=tk.X, pady=1)
        fps = max(self.gui.fps, 1)
        n_selected = int(self.gui.n_frames / max(fps * self._interval_value.get(), 0.1))
        self.lbl_interval_count = ttk.Label(frm_int2, text=f"~{n_selected} frames")
        self.lbl_interval_count.pack(side=tk.LEFT)
        ttk.Button(frm_int2, text="Apply Interval",
                   command=lambda: self.gui.action("apply_interval"))\
            .pack(side=tk.RIGHT)

        # Background
        ttk.Button(self.actions_frame, text=f"B Set BG (current: {self.gui.background_frame_idx})",
                   command=lambda: self.gui.action("set_bg"))\
            .pack(fill=tk.X, pady=1)

        # Save & Restart
        ttk.Button(self.actions_frame, text="S Save Composite",
                   command=lambda: self.gui.action("save"))\
            .pack(fill=tk.X, pady=3)
        ttk.Button(self.actions_frame, text="↺ Restart",
                   command=lambda: self.gui.action("restart"))\
            .pack(fill=tk.X, pady=1)

        # Composite count
        ttk.Label(self.actions_frame,
                  text=f"Marked: {len(self.gui.composite_frames)} frames")\
            .pack(anchor=tk.W, pady=2)

    def _build_vis_range(self, editable: bool) -> None:
        """Build visibility range controls for the active object."""
        if hasattr(self, 'frm_vis'):
            self.frm_vis.destroy()
        self.frm_vis = ttk.LabelFrame(self.root, text="Visibility Range", padding=5)
        self.frm_vis.pack(fill=tk.X, padx=5, pady=2, after=self.frm_actions)

        if not self.gui.objects:
            ttk.Label(self.frm_vis, text="No objects").pack()
            return

        obj = self.gui.active_object()
        ttk.Label(self.frm_vis, text=f"Active: {obj.name}", foreground=obj.color_hex)\
            .pack(anchor=tk.W)

        n = max(self.gui.n_frames - 1, 1)
        self.vis_start_var = tk.IntVar(value=obj.vis_start if obj.vis_start is not None else 0)
        self.vis_end_var = tk.IntVar(value=obj.vis_end if obj.vis_end is not None else n)

        frm1 = ttk.Frame(self.frm_vis)
        frm1.pack(fill=tk.X)
        ttk.Label(frm1, text="Start:").pack(side=tk.LEFT)
        s1 = ttk.Scale(frm1, from_=0, to=n, variable=self.vis_start_var,
                       orient=tk.HORIZONTAL, length=160)
        s1.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=3)
        if not editable:
            s1.configure(state=tk.DISABLED)
        ttk.Label(frm1, textvariable=self.vis_start_var, width=5).pack(side=tk.LEFT)

        frm2 = ttk.Frame(self.frm_vis)
        frm2.pack(fill=tk.X)
        ttk.Label(frm2, text="End:  ").pack(side=tk.LEFT)
        s2 = ttk.Scale(frm2, from_=0, to=n, variable=self.vis_end_var,
                       orient=tk.HORIZONTAL, length=160)
        s2.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=3)
        if not editable:
            s2.configure(state=tk.DISABLED)
        ttk.Label(frm2, textvariable=self.vis_end_var, width=5).pack(side=tk.LEFT)

        if editable:
            def apply_vis():
                self.gui.action("apply_vis_range")
            ttk.Button(self.frm_vis, text="Apply Range",
                       command=apply_vis).pack(fill=tk.X, pady=2)
            ttk.Button(self.frm_vis, text="Reset to Full Range",
                       command=lambda: self.gui.action("reset_vis_range"))\
                .pack(fill=tk.X)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def update(self) -> bool:
        """Sync widget states from GUI and process tk events. Called per-frame.
        Returns False if the panel window was closed."""
        gui = self.gui

        # Check if tkinter window was closed
        try:
            self.root.winfo_exists()
        except tk.TclError:
            return False

        # Check if state changed → rebuild dynamic widgets
        if (gui.state != self._last_state
                or gui.active_obj_idx != self._last_active_idx
                or len(gui.objects) != self._last_obj_count):
            self._rebuild_object_buttons()
            self._build_dynamic()

        # Update status labels
        state_names = {GUIState.SETUP: "SETUP", GUIState.TRACKING: "TRACKING",
                       GUIState.SELECTION: "SELECTION", GUIState.SAVE: "SAVED"}
        self.lbl_state.configure(text=state_names.get(gui.state, "—"))
        self.lbl_frame.configure(
            text=f"Frame: {gui.current_frame_idx} / {gui.n_frames}"
        )

        if gui.active_object():
            obj = gui.active_object()
            n_pts = len(obj.points)
            self.lbl_active.configure(
                text=f"Active: {obj.name} ({n_pts} pts, seed fr {obj.seed_frame})"
            )
        else:
            self.lbl_active.configure(text="Active: —")

        # Button highlights for object buttons
        for i, btn in enumerate(getattr(self, '_obj_btns', [])):
            if i == gui.active_obj_idx:
                btn.configure(style="Accent.TButton" if hasattr(self, '_accent_style') else "")
            else:
                btn.configure(style="TButton")

        # Sync view radio
        self.view_var.set(gui.viz_mode)

        # Sync visibility sliders if in SELECTION
        if gui.state == GUIState.SELECTION and gui.active_object():
            obj = gui.active_object()
            n = max(gui.n_frames - 1, 1)
            if hasattr(self, 'vis_start_var'):
                val = obj.vis_start if obj.vis_start is not None else 0
                if self.vis_start_var.get() != val:
                    self.vis_start_var.set(val)
            if hasattr(self, 'vis_end_var'):
                val = obj.vis_end if obj.vis_end is not None else n
                if self.vis_end_var.get() != val:
                    self.vis_end_var.set(val)

        # Update interval label
        if gui.state == GUIState.SELECTION and hasattr(self, 'lbl_interval_count'):
            fps = max(gui.fps, 1)
            n_sel = int(gui.n_frames / max(fps * self._interval_value.get(), 0.1))
            self.lbl_interval_count.configure(text=f"~{n_sel} frames")

        # Process tk events
        try:
            self.root.update_idletasks()
            self.root.update()
        except tk.TclError:
            return False
        return True

    def _rebuild_object_buttons(self) -> None:
        """Rebuild object selection buttons."""
        for w in self.obj_buttons_frame.winfo_children():
            w.destroy()
        self._obj_btns = []
        for i, obj in enumerate(self.gui.objects):
            idx = i
            btn = tk.Button(
                self.obj_buttons_frame,
                text=f"● {obj.name}",
                fg=obj.color_hex,
                bg="white" if i == self.gui.active_obj_idx else "#e0e0e0",
                activebackground="#c0c0ff",
                relief=tk.RAISED if i == self.gui.active_obj_idx else tk.FLAT,
                font=("", 9, "bold") if i == self.gui.active_obj_idx else ("", 9),
                command=lambda i=idx: self.gui.action("select_object", i),
            )
            btn.pack(side=tk.LEFT, padx=2, pady=2, ipadx=8)
            self._obj_btns.append(btn)

    def show_error(self, msg: str) -> None:
        """Show an error dialog."""
        from tkinter import messagebox
        messagebox.showerror("Error", msg)

    def _on_tk_key(self, event: tk.Event) -> None:
        """Forward keyboard events from tkinter window to the GUI."""
        # Map tkinter key events to the same action names used by OpenCV key handler
        key = event.keysym
        char = event.char

        if key == 'Escape':
            self.gui.action("quit")
        elif key == 'Return' or key == 'space':
            if self.gui.state == GUIState.SETUP:
                self.gui.action("start_tracking")
            elif self.gui.state == GUIState.SELECTION:
                self.gui.action("restart")
        elif key == 'BackSpace' or key == 'Delete':
            obj = self.gui.active_object()
            if obj and obj.points:
                obj.points.pop()
        elif key in ('Left', 'Right', 'Up', 'Down'):
            if key == 'Left':
                self.gui.current_frame_idx = (self.gui.current_frame_idx - 1) % self.gui.n_frames
            elif key == 'Right':
                self.gui.current_frame_idx = (self.gui.current_frame_idx + 1) % self.gui.n_frames
            self.gui._preview_dirty = True
        elif char.lower() == 'k':
            self.gui.action("mark_frame")
        elif char.lower() == 'b':
            self.gui.action("set_bg")
        elif char.lower() == 'v':
            cycle = {"mask": "composite", "composite": "original", "original": "mask"}
            self.gui.action("view_" + cycle[self.gui.viz_mode])
        elif char.lower() == 'i':
            self.gui.action("apply_interval")
        elif char.lower() == 's':
            self.gui.action("save")
        elif char.lower() == 'n':
            self.gui.action("new_object")
        elif char.lower() == 'r':
            self.gui.action("clear_points")
        elif char.lower() == '[':
            obj = self.gui.active_object()
            if obj:
                obj.vis_start = self.gui.current_frame_idx
                self.gui._preview_dirty = True
        elif char.lower() == ']':
            obj = self.gui.active_object()
            if obj:
                obj.vis_end = self.gui.current_frame_idx
                self.gui._preview_dirty = True
        elif char.lower() == '\\':
            self.gui.action("reset_vis_range")
        elif char in '123456789':
            idx = int(char) - 1
            if idx < len(self.gui.objects):
                self.gui.action("select_object", idx)
        elif key == 'Tab':
            if self.gui.objects:
                self.gui.active_obj_idx = (self.gui.active_obj_idx + 1) % len(self.gui.objects)

    def _on_close(self) -> None:
        self.gui.action("quit")

    def destroy(self) -> None:
        try:
            self.root.destroy()
        except tk.TclError:
            pass


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactive GUI for stroboscopic image generation with SAM2."
    )
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--process-fps", type=float, default=None)
    parser.add_argument("--alpha", type=float, default=0.60)
    parser.add_argument("--mask-threshold", type=float, default=0.0)
    parser.add_argument("--dilate-kernel", type=int, default=5)
    parser.add_argument("--min-area", type=int, default=300)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu", "mps"), default="auto")
    parser.add_argument("--hf-model-id", type=str, default="facebook/sam2.1-hiera-small")
    parser.add_argument("--model-cfg", type=str, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--vos-optimized", action="store_true")
    parser.add_argument("--offload-video-to-cpu", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--offload-state-to-cpu", action=argparse.BooleanOptionalAction, default=False)

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


# ---------------------------------------------------------------------------
# Main GUI class
# ---------------------------------------------------------------------------
class StroboscopicGUI:
    """Interactive stroboscopic compositing with multi-object SAM2 tracking."""

    def __init__(self, cap, predictor, video_path, n_frames, fps, args, tmp_dir):
        self.cap = cap
        self.video_path = video_path
        self.n_frames = n_frames
        self.fps = fps
        ret, frame = read_frame_at(cap, 0)
        if not ret:
            raise RuntimeError("Cannot read first frame.")
        self.h, self.w = frame.shape[:2]

        self.predictor = predictor
        self.inference_state = None

        # --- multi-object ---
        self.objects: list[TrackObject] = []
        self.active_obj_idx: int = -1
        # Create default object 1
        self._create_object(seed_frame=0)

        # --- tracking ---
        self.masks: dict[int, dict[int, np.ndarray]] = {}  # frame_idx -> {obj_id: mask}
        self._tracking_generator = None
        self._tracking_direction = "forward"
        self._tracking_pass = 0
        self._tracking_total_passes = 0
        self._tracking_frame_count = 0
        self._tracking_total_frames = 0

        # --- selection ---
        self.state = GUIState.SETUP
        self.current_frame_idx: int = 0
        self.composite_frames: set[int] = set()
        self.background_frame_idx: int = 0
        self._range_start: int | None = None
        self.viz_mode = "mask"
        self._preview_dirty = True
        self._preview_cache: np.ndarray | None = None
        self.frame_overrides: dict[int, dict[int, bool]] = {}

        # --- status ---
        self.status_message = ""
        self.status_timer = 0

        # --- trackbar ---
        self._trackbar_locked = False

        self.args = args
        self.tmp_dir = tmp_dir

        # --- control panel ---
        self.panel: ControlPanel | None = None

    # ===================================================================
    # Object management
    # ===================================================================
    def _create_object(self, seed_frame: int) -> TrackObject:
        obj_id = len(self.objects) + 1
        color_idx = (obj_id - 1) % len(OBJECT_COLORS)
        obj = TrackObject(
            obj_id=obj_id,
            color=OBJECT_COLORS[color_idx],
            color_hex=OBJECT_COLOR_HEX[color_idx],
            name=f"Obj{obj_id}",
            seed_frame=seed_frame,
        )
        self.objects.append(obj)
        self.active_obj_idx = len(self.objects) - 1
        return obj

    def active_object(self) -> TrackObject | None:
        if 0 <= self.active_obj_idx < len(self.objects):
            return self.objects[self.active_obj_idx]
        return None

    # ===================================================================
    # Action dispatch (called by both tkinter buttons and keyboard)
    # ===================================================================
    def action(self, name: str, *args) -> None:
        """Central action dispatcher. All UI controls route through here."""
        if name == "quit":
            self._quit_flag = True
        elif name == "new_object":
            self._create_object(seed_frame=self.current_frame_idx)
            self.status_message = f"Created {self.active_object().name} at frame {self.current_frame_idx}"
            self.status_timer = 60
        elif name == "select_object":
            idx = args[0] if args else 0
            if 0 <= idx < len(self.objects):
                self.active_obj_idx = idx
        elif name == "clear_points":
            obj = self.active_object()
            if obj:
                obj.points.clear()
        elif name == "start_tracking":
            if self.state == GUIState.SETUP and any(o.points for o in self.objects):
                self._start_tracking()
        elif name == "abort_tracking":
            if self.state == GUIState.TRACKING:
                self._tracking_generator = None
                self.masks.clear()
                self.state = GUIState.SETUP
                self.status_message = "Tracking aborted."
                self.status_timer = 60
        elif name == "mark_frame":
            if self.state == GUIState.SELECTION:
                if self.current_frame_idx in self.composite_frames:
                    self.composite_frames.discard(self.current_frame_idx)
                else:
                    self.composite_frames.add(self.current_frame_idx)
                self._preview_dirty = True
        elif name == "range_select":
            if self.state == GUIState.SELECTION:
                if self._range_start is None:
                    self._range_start = self.current_frame_idx
                    self.status_message = f"Range start: {self._range_start}. Click again for end."
                    self.status_timer = 120
                else:
                    start, end = sorted([self._range_start, self.current_frame_idx])
                    for f in range(start, end + 1):
                        self.composite_frames.add(f)
                    self._range_start = None
                    self.status_message = f"Range {start}→{end}: {end - start + 1} frames added."
                    self.status_timer = 90
                    self._preview_dirty = True
        elif name == "apply_interval":
            if self.state == GUIState.SELECTION:
                interval_sec = self.panel._interval_value.get()
                fps = max(self.fps, 1)
                step = max(1, int(round(interval_sec * fps)))
                count = 0
                for f in range(0, self.n_frames, step):
                    self.composite_frames.add(f)
                    count += 1
                self.status_message = f"Interval {interval_sec}s: {count} frames added."
                self.status_timer = 90
                self._preview_dirty = True
        elif name == "set_bg":
            if self.state == GUIState.SELECTION:
                self.background_frame_idx = self.current_frame_idx
                self._preview_dirty = True
                self.status_message = f"Background: frame {self.current_frame_idx}"
                self.status_timer = 60
        elif name in ("view_mask", "view_composite", "view_original"):
            self.viz_mode = name.split("_")[1]
        elif name == "apply_vis_range":
            obj = self.active_object()
            if obj and self.state == GUIState.SELECTION:
                obj.vis_start = self.panel.vis_start_var.get()
                obj.vis_end = self.panel.vis_end_var.get()
                self._preview_dirty = True
                self.status_message = f"{obj.name} visible: {obj.vis_start}→{obj.vis_end}"
                self.status_timer = 60
        elif name == "reset_vis_range":
            obj = self.active_object()
            if obj:
                obj.vis_start = None
                obj.vis_end = None
                self._preview_dirty = True
                self.status_message = f"{obj.name} visible: all frames"
                self.status_timer = 60
        elif name == "save":
            if self.state == GUIState.SELECTION:
                if not self.composite_frames:
                    self.status_message = "No frames marked! Use K or Interval to mark frames."
                    self.status_timer = 90
                    return
                self.state = GUIState.SAVE
                self.status_message = self._composite_and_save()
                self.status_timer = 120
                # Immediately return to SELECTION
                self.state = GUIState.SELECTION
        elif name == "restart":
            if self.inference_state is not None:
                with contextlib.suppress(Exception):
                    self.predictor.reset_state(self.inference_state)
                self.inference_state = None
            self.masks.clear()
            self.composite_frames.clear()
            self.background_frame_idx = 0
            self._range_start = None
            self.frame_overrides.clear()
            self._preview_dirty = True
            self._preview_cache = None
            self.viz_mode = "mask"
            self.state = GUIState.SETUP
            self.status_message = "Restarted. Mark objects and press Start."
            self.status_timer = 90

    # ===================================================================
    # Main loop
    # ===================================================================
    def run(self) -> None:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.createTrackbar(TRACKBAR_FRAME, WINDOW_NAME, 0, max(0, self.n_frames - 1),
                           self._on_trackbar_frame)
        cv2.createTrackbar(TRACKBAR_ALPHA, WINDOW_NAME, int(self.args.alpha * 100), 100,
                           self._on_trackbar_alpha)
        cv2.setMouseCallback(WINDOW_NAME, self._on_mouse)
        self._set_trackbar(TRACKBAR_FRAME, self.current_frame_idx)
        self._set_trackbar(TRACKBAR_ALPHA, int(self.args.alpha * 100))

        # Create tkinter control panel
        self.panel = ControlPanel(self)
        self._quit_flag = False

        try:
            while not self._quit_flag:
                if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                    break

                key_raw = cv2.waitKeyEx(20)
                key = key_raw & 0xFF
                self._last_key_raw = key_raw
                self._handle_keyboard(key, key_raw)

                # Dispatch by state
                if self.state == GUIState.SETUP:
                    canvas = self._render_setup()
                elif self.state == GUIState.TRACKING:
                    canvas = self._render_tracking()
                elif self.state == GUIState.SELECTION:
                    canvas = self._render_selection()
                else:
                    canvas = self._render_save()

                cv2.imshow(WINDOW_NAME, canvas)

                if self.status_timer > 0:
                    self.status_timer -= 1
                    if self.status_timer == 0:
                        self.status_message = ""

                # Sync tkinter panel
                if self.panel:
                    if not self.panel.update():
                        break  # panel window was closed
        finally:
            if self.panel:
                self.panel.destroy()
            cv2.destroyWindow(WINDOW_NAME)
            cv2.waitKey(1)

    # ===================================================================
    # Keyboard handler
    # ===================================================================
    def _handle_keyboard(self, key: int, key_raw: int) -> None:
        if key_raw == 27:  # Esc
            self.action("quit")
            return

        # Navigation (arrow keys)
        if key_raw in (81, 65361, 2424832):
            self.current_frame_idx = (self.current_frame_idx - 1) % self.n_frames
            self._set_trackbar(TRACKBAR_FRAME, self.current_frame_idx)
            self._preview_dirty = True
        elif key_raw in (83, 65363, 2555904):
            self.current_frame_idx = (self.current_frame_idx + 1) % self.n_frames
            self._set_trackbar(TRACKBAR_FRAME, self.current_frame_idx)
            self._preview_dirty = True

        # Number keys: select object
        if ord('1') <= key <= ord('9'):
            idx = key - ord('1')
            if idx < len(self.objects):
                self.action("select_object", idx)
        elif key == ord('n') or key == ord('N'):
            self.action("new_object")
        elif key == 9:  # Tab
            if self.objects:
                self.active_obj_idx = (self.active_obj_idx + 1) % len(self.objects)

        # State-specific keys
        if self.state == GUIState.SETUP:
            if key in (8, 127):  # Backspace / Delete
                obj = self.active_object()
                if obj and obj.points:
                    obj.points.pop()
            elif key == ord('r') or key == ord('R'):
                self.action("clear_points")
            elif key in (13, 10, 32):  # Enter / Space
                self.action("start_tracking")
        elif self.state == GUIState.TRACKING:
            if key == 27:
                self.action("abort_tracking")
        elif self.state == GUIState.SELECTION:
            if key == ord('k') or key == ord('K'):
                self.action("mark_frame")
            elif key == ord('b') or key == ord('B'):
                self.action("set_bg")
            elif key == ord('v') or key == ord('V'):
                cycle = {"mask": "composite", "composite": "original", "original": "mask"}
                self.action("view_" + cycle[self.viz_mode])
            elif key == ord('i') or key == ord('I'):
                self.action("apply_interval")
            elif key == ord('s') or key == ord('S'):
                self.action("save")
            elif key in (13, 10):  # Enter
                self.action("restart")
            elif key == ord('['):
                obj = self.active_object()
                if obj:
                    obj.vis_start = self.current_frame_idx
                    self._preview_dirty = True
            elif key == ord(']'):
                obj = self.active_object()
                if obj:
                    obj.vis_end = self.current_frame_idx
                    self._preview_dirty = True
            elif key == ord('\\'):
                self.action("reset_vis_range")

    # ===================================================================
    # Trackbar callbacks
    # ===================================================================
    def _on_trackbar_frame(self, value: int) -> None:
        if self._trackbar_locked or self.state == GUIState.TRACKING:
            return
        if value != self.current_frame_idx:
            self.current_frame_idx = value
            self._preview_dirty = True

    def _on_trackbar_alpha(self, value: int) -> None:
        self.args.alpha = value / 100.0
        self._preview_dirty = True

    def _set_trackbar(self, name: str, value: int) -> None:
        self._trackbar_locked = True
        cv2.setTrackbarPos(name, WINDOW_NAME, value)
        self._trackbar_locked = False

    # ===================================================================
    # Mouse
    # ===================================================================
    def _on_mouse(self, event: int, x: int, y: int, _flags: int, _param: object) -> None:
        if self.state != GUIState.SETUP:
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            # Check if click is on video area (not timeline)
            if 0 <= x < self.w and 0 <= y < self.h:
                obj = self.active_object()
                if obj:
                    obj.points.append((x, y))

    # ===================================================================
    # Frame helpers
    # ===================================================================
    def _current_frame(self) -> np.ndarray:
        ret, frame = read_frame_at(self.cap, self.current_frame_idx)
        return frame.copy() if ret else np.zeros((self.h, self.w, 3), dtype=np.uint8)

    # ===================================================================
    # SETUP rendering
    # ===================================================================
    def _render_setup(self) -> np.ndarray:
        frame = self._current_frame()
        canvas = frame.copy()

        # Draw all objects' points
        active = self.active_object()
        for obj in self.objects:
            is_active = (obj is active)
            for i, (x, y) in enumerate(obj.points, start=1):
                b, g, r = int(obj.color[0]), int(obj.color[1]), int(obj.color[2])
                color_bgr = (b, g, r)
                if is_active:
                    cv2.circle(canvas, (x, y), 7, color_bgr, -1)
                    cv2.circle(canvas, (x, y), 20, color_bgr, 2)
                    cv2.putText(canvas, str(i), (x + 8, y - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_bgr, 2, cv2.LINE_AA)
                else:
                    cv2.circle(canvas, (x, y), 7, color_bgr, 2)  # hollow
                    cv2.putText(canvas, obj.name, (x + 12, y - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color_bgr, 1, cv2.LINE_AA)

        self._draw_status_bar(canvas, "SETUP")
        return canvas

    # ===================================================================
    # TRACKING
    # ===================================================================
    def _start_tracking(self) -> None:
        self.masks.clear()
        self.state = GUIState.TRACKING
        self.status_message = "Initializing SAM2..."

        self.inference_state = self.predictor.init_state(
            video_path=str(self.video_path),
            offload_video_to_cpu=self.args.offload_video_to_cpu,
            offload_state_to_cpu=self.args.offload_state_to_cpu,
        )

        # Register all objects at their seed frames
        objects_with_points = [o for o in self.objects if o.points]
        for obj in objects_with_points:
            points_np = np.array(obj.points, dtype=np.float32)
            labels_np = np.ones(len(obj.points), dtype=np.int32)
            self.predictor.add_new_points_or_box(
                inference_state=self.inference_state,
                frame_idx=obj.seed_frame,
                obj_id=obj.obj_id,
                points=points_np,
                labels=labels_np,
            )

        self._tracked_obj_ids = [o.obj_id for o in objects_with_points]
        self._tracking_pass = 0
        self._tracking_total_frames = self.n_frames * 2  # estimate
        self._tracking_frame_count = 0
        min_seed = min(o.seed_frame for o in objects_with_points)

        # First pass: forward from min_seed
        self._tracking_direction = "forward"
        self._tracking_generator = self.predictor.propagate_in_video(
            self.inference_state, start_frame_idx=min_seed, reverse=False,
        )
        self._tracking_total_passes = 1 + sum(
            1 for o in objects_with_points if o.seed_frame > min_seed
        )

    def _advance_tracking(self) -> bool:
        if self._tracking_generator is None:
            return False

        try:
            frame_idx, obj_ids, video_res_masks = next(self._tracking_generator)
            fidx = int(frame_idx)

            if fidx not in self.masks:
                self.masks[fidx] = {}

            for i, obj_id in enumerate(obj_ids):
                if obj_id in self._tracked_obj_ids:
                    mask_logits = video_res_masks[i]
                    if mask_logits.ndim == 3:
                        mask_logits = mask_logits[0]
                    mask = (mask_logits > self.args.mask_threshold).detach().cpu().numpy()
                    if mask.any():
                        self.masks[fidx][obj_id] = mask

            self._tracking_frame_count += 1
            self.current_frame_idx = fidx
            self._set_trackbar(TRACKBAR_FRAME, self.current_frame_idx)

            # Update panel progress
            if self.panel and hasattr(self.panel, 'progress_var'):
                pct = min(100, 100 * self._tracking_frame_count / max(self._tracking_total_frames, 1))
                self.panel.progress_var.set(pct)
                self.panel.lbl_progress.configure(text=f"{pct:.0f}%  ({self._tracking_direction})")

            return True
        except StopIteration:
            self._tracking_pass += 1
            # Additional backward passes for objects seeded after min_seed
            objects_with_points = [o for o in self.objects if o.points]
            min_seed = min(o.seed_frame for o in objects_with_points)
            late_objects = [o for o in objects_with_points if o.seed_frame > min_seed]

            if self._tracking_pass == 1:
                # Second pass: backward from min_seed
                self._tracking_direction = "backward"
                self._tracking_generator = self.predictor.propagate_in_video(
                    self.inference_state, start_frame_idx=min_seed, reverse=True,
                )
                return self._advance_tracking()
            elif late_objects:
                # Third+ passes: backward from each late object's seed
                obj = late_objects[self._tracking_pass - 2]
                self._tracking_direction = f"backward({obj.name})"
                self._tracking_generator = self.predictor.propagate_in_video(
                    self.inference_state, start_frame_idx=obj.seed_frame, reverse=True,
                )
                return self._advance_tracking()
            else:
                self._tracking_generator = None
                self.state = GUIState.SELECTION
                self.current_frame_idx = min_seed
                self._set_trackbar(TRACKBAR_FRAME, self.current_frame_idx)
                total_masks = sum(len(m) for m in self.masks.values())
                self.status_message = f"Tracking done. {total_masks} masks across {len(self.masks)} frames."
                self.status_timer = 120
                self._preview_dirty = True
                return False

    def _render_tracking(self) -> np.ndarray:
        self._advance_tracking()
        frame = self._current_frame()
        canvas = self._draw_all_masks(frame)
        self._draw_status_bar(canvas, "TRACKING")

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

    # ===================================================================
    # SELECTION rendering
    # ===================================================================
    def _render_selection(self) -> np.ndarray:
        frame = self._current_frame()

        if self.viz_mode == "composite":
            canvas = self._get_composite_preview()
        elif self.viz_mode == "mask":
            canvas = self._draw_all_masks(frame)
        else:
            canvas = frame.copy()

        self._draw_status_bar(canvas, "SELECTION")
        canvas = _draw_timeline_on_canvas(self, canvas)
        return canvas

    def _render_save(self) -> np.ndarray:
        return self._render_selection()

    # ===================================================================
    # Drawing helpers
    # ===================================================================
    def _draw_all_masks(self, frame: np.ndarray) -> np.ndarray:
        """Overlay all objects' masks in their respective colors."""
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

    def _draw_status_bar(self, canvas: np.ndarray, state_label: str) -> None:
        bar_h = 32
        overlay = canvas.copy()
        cv2.rectangle(overlay, (0, 0), (self.w, bar_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, canvas, 0.45, 0, canvas)

        n_objs = len(self.objects)
        active = self.active_object()
        active_name = active.name if active else "—"

        if self.state == GUIState.SETUP:
            info = (f"{state_label} | Frame {self.current_frame_idx}/{self.n_frames}"
                    f" | Active: {active_name} | Objects: {n_objs}")
        elif self.state == GUIState.TRACKING:
            n_masks = sum(len(m) for m in self.masks.values())
            info = (f"{state_label} | Frame {self.current_frame_idx}/{self.n_frames}"
                    f" | Objects: {n_objs} | Masks: {n_masks}")
        else:
            marked = "K" if self.current_frame_idx in self.composite_frames else "-"
            info = (f"{state_label} | Frame {self.current_frame_idx}/{self.n_frames}"
                    f" | Marked: {len(self.composite_frames)} | BG: {self.background_frame_idx}"
                    f" | [{marked}]")

        cv2.putText(canvas, info, (10, 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (220, 220, 220), 1, cv2.LINE_AA)

        if self.status_message:
            cv2.putText(canvas, self.status_message, (10, bar_h + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)

    # ===================================================================
    # Composite
    # ===================================================================
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
        ret, bg = read_frame_at(self.cap, self.background_frame_idx)
        if not ret:
            bg = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        canvas = bg.astype(np.float32)

        for fidx in sorted(self.composite_frames):
            for obj in self._visible_objects_at(fidx):
                mask = self.masks.get(fidx, {}).get(obj.obj_id)
                if mask is None or not mask.any():
                    continue
                ret, frame = read_frame_at(self.cap, fidx)
                if not ret:
                    continue
                mask_clean = clean_mask(
                    mask=mask, min_area=self.args.min_area,
                    dilate_kernel=self.args.dilate_kernel,
                    seed_xys=obj.points if fidx == obj.seed_frame else None,
                )
                if not mask_clean.any():
                    continue
                m = mask_clean.astype(np.float32)[..., None]
                canvas = (canvas * (1.0 - self.args.alpha * m)
                          + frame.astype(np.float32) * (self.args.alpha * m))

        # Crisp seed frames on top (full opacity)
        for obj in self.objects:
            seed_mask = self.masks.get(obj.seed_frame, {}).get(obj.obj_id)
            if seed_mask is not None and seed_mask.any():
                seed_mask_clean = clean_mask(
                    mask=seed_mask, min_area=self.args.min_area,
                    dilate_kernel=self.args.dilate_kernel, seed_xys=obj.points,
                )
                if seed_mask_clean.any():
                    ret, seed_frame = read_frame_at(self.cap, obj.seed_frame)
                    if ret:
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
        return f"Saved: {self.args.out}  ({len(self.composite_frames)} frames)"

    def close(self) -> None:
        if self.inference_state is not None:
            with contextlib.suppress(Exception):
                self.predictor.reset_state(self.inference_state)
            self.inference_state = None


# ===================================================================
# Timeline drawer
# ===================================================================
def _draw_timeline_on_canvas(gui: StroboscopicGUI, canvas: np.ndarray) -> np.ndarray:
    h, w = canvas.shape[:2]
    n = gui.n_frames
    out = np.zeros((h + TIMELINE_H, w, 3), dtype=np.uint8)
    out[:h, :] = canvas
    y0 = h

    # Per-frame vertical strips
    for fidx in range(n):
        x_start = int(w * fidx / max(n, 1))
        x_end = int(w * (fidx + 1) / max(n, 1))
        x_end = max(x_end, x_start + 1)

        if fidx in gui.composite_frames:
            color = (0, 200, 80)        # green: marked
        elif fidx in gui.masks and gui.masks[fidx]:
            color = (60, 60, 60)        # dark gray: has mask
        else:
            color = (40, 40, 40)        # gray: no mask
        cv2.rectangle(out, (x_start, y0), (x_end, y0 + TIMELINE_H), color, -1)

    # Object visibility range bars (top half of timeline)
    bar_h = TIMELINE_H // 2 - 2
    for obj in gui.objects:
        start = obj.vis_start if obj.vis_start is not None else 0
        end = obj.vis_end if obj.vis_end is not None else n - 1
        x1 = int(w * start / max(n, 1))
        x2 = int(w * (end + 1) / max(n, 1))
        b, g, r = int(obj.color[0]), int(obj.color[1]), int(obj.color[2])
        cv2.rectangle(out, (x1, y0 + 2), (x2, y0 + 2 + bar_h), (b, g, r), -1)

    # Background indicator (orange)
    bg_x = int(w * gui.background_frame_idx / max(n, 1))
    cv2.line(out, (bg_x, y0), (bg_x, y0 + TIMELINE_H), (255, 150, 50), 2)

    # Current position (white)
    cur_x = int(w * gui.current_frame_idx / max(n, 1))
    cv2.line(out, (cur_x, y0), (cur_x, y0 + TIMELINE_H), (255, 255, 255), 2)

    # Legend
    cv2.putText(out, "gray=no mask  dark=has mask  green=marked  colors=obj range  orange=bg  white=pos",
                (5, y0 + TIMELINE_H - 8), cv2.FONT_HERSHEY_SIMPLEX,
                0.3, (160, 160, 160), 1, cv2.LINE_AA)
    return out


# ===================================================================
# main()
# ===================================================================
def main() -> None:
    args = parse_args()
    if not args.video.exists():
        raise RuntimeError(f"Video not found: {args.video}")

    device_name = resolve_device(args.device)
    print(f"Loading video: {args.video}")
    print(f"Device: {device_name}")
    print("Note: SAM2 model runs 100% locally on your GPU.")
    print("First run will download ~150MB model from HuggingFace to local cache.")

    with tempfile.TemporaryDirectory(prefix="sam2_gui_") as tmp_dir:
        processing_video, source_fps, was_downsampled = maybe_downsample_video(
            src_video=args.video, target_fps=args.process_fps, tmp_dir=Path(tmp_dir),
        )
        cap = cv2.VideoCapture(str(processing_video))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {processing_video}")

        try:
            fps = cap.get(cv2.CAP_PROP_FPS) or source_fps or 30.0
            n_frames = get_frame_count(cap)
            print(f"Frames: {n_frames}, FPS: {fps:.2f}")
            if was_downsampled:
                print(f"Downsampled: {source_fps:.2f} → {fps:.2f}")

            device = resolve_device(args.device)
            predictor = build_predictor(args, device=device)
            gui = StroboscopicGUI(cap, predictor, processing_video,
                                  n_frames, fps, args, Path(tmp_dir))
            with sam_inference_context(device):
                gui.run()
        finally:
            cap.release()
            cv2.destroyAllWindows()
    print("Done.")


if __name__ == "__main__":
    main()
