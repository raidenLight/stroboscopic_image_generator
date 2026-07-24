#!/usr/bin/env python3
"""
Interactive GUI for stroboscopic image generation with SAM2 tracking.

Workflow:
  1. SETUP:  drag the trackbar / arrows to find the right frame, click points
             on the target object, press Enter to start SAM2 tracking.
  2. TRACKING: watch masks propagate frame-by-frame (Esc to abort).
  3. SELECTION: scrub through frames, press K to mark frames for compositing,
                B to set the background, V to toggle composite preview.
  4. SAVE: press S to composite & save, then continue editing or quit.
"""

from __future__ import annotations

import argparse
import contextlib
import sys
import tempfile
from enum import Enum, auto
from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Make the sibling script importable so we can reuse its helpers.
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
TRACKBAR_NAME = "Frame"

# OpenCV key codes that represent Enter / Space
_ENTER_KEYS = {13, 10, 32}

# Timeline bar height (drawn at the bottom of the canvas)
TIMELINE_H = 40


# ---------------------------------------------------------------------------
# State machine
# ---------------------------------------------------------------------------
class GUIState(Enum):
    SETUP = auto()
    TRACKING = auto()
    SELECTION = auto()
    SAVE = auto()


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactive GUI for stroboscopic image generation with SAM2."
    )

    # --- I/O ---
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)

    # --- Processing ---
    parser.add_argument(
        "--process-fps", type=float, default=None,
        help="FPS for SAM2 processing. Lower than source FPS reduces memory.",
    )
    parser.add_argument("--alpha", type=float, default=0.60)
    parser.add_argument("--mask-threshold", type=float, default=0.0)
    parser.add_argument("--dilate-kernel", type=int, default=5)
    parser.add_argument("--min-area", type=int, default=300)

    # --- Model ---
    parser.add_argument("--device", choices=("auto", "cuda", "cpu", "mps"), default="auto")
    parser.add_argument("--obj-id", type=int, default=1)
    parser.add_argument("--hf-model-id", type=str, default="facebook/sam2.1-hiera-small")
    parser.add_argument("--model-cfg", type=str, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--vos-optimized", action="store_true")
    parser.add_argument(
        "--offload-video-to-cpu", action=argparse.BooleanOptionalAction, default=True,
    )
    parser.add_argument(
        "--offload-state-to-cpu", action=argparse.BooleanOptionalAction, default=False,
    )

    args = parser.parse_args()

    # Validation
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
    """Interactive OpenCV GUI for SAM2-based stroboscopic image composition."""

    def __init__(
        self,
        cap: cv2.VideoCapture,
        predictor,
        video_path: Path,
        n_frames: int,
        fps: float,
        args: argparse.Namespace,
        tmp_dir: Path,
    ):
        # --- video ---
        self.cap = cap
        self.video_path = video_path
        self.n_frames = n_frames
        self.fps = fps
        ret, frame = read_frame_at(cap, 0)
        if not ret:
            raise RuntimeError("Cannot read first frame.")
        self.h, self.w = frame.shape[:2]

        # --- SAM2 ---
        self.predictor = predictor
        self.inference_state = None

        # --- tracking data ---
        self.masks: dict[int, np.ndarray] = {}          # frame_idx → bool mask
        self._tracking_generator = None
        self._tracking_direction = "forward"
        self._tracking_frame_count = 0
        self._tracking_total = 0

        # --- user selections ---
        self.state = GUIState.SETUP
        self.current_frame_idx: int = 0
        self.points: list[tuple[int, int]] = []
        self.seed_frame_idx: int = 0
        self.composite_frames: set[int] = set()
        self.background_frame_idx: int = 0

        # --- preview cache ---
        self.viz_mode = "mask"          # "mask" | "composite" | "original"
        self._preview_dirty = True
        self._preview_cache: np.ndarray | None = None

        # --- status ---
        self.status_message = ""
        self.status_timer = 0           # frames remaining for status message

        # --- trackbar guard (prevents re-entrant updates) ---
        self._trackbar_locked = False

        # --- args ---
        self.args = args
        self.tmp_dir = tmp_dir

    # =======================================================================
    # Public entry points
    # =======================================================================

    def run(self) -> None:
        """Main GUI event loop."""
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.createTrackbar(TRACKBAR_NAME, WINDOW_NAME, 0, max(0, self.n_frames - 1),
                           self._on_trackbar)
        cv2.setMouseCallback(WINDOW_NAME, self._on_mouse)

        # Initialize trackbar position
        self._set_trackbar(self.current_frame_idx)

        try:
            while True:
                # ---- detect window close ----
                if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                    break

                key_raw = cv2.waitKeyEx(20)
                key = key_raw & 0xFF   # ASCII portion for letter / digit checks
                self._last_key_raw = key_raw  # for arrow-key navigation

                # ---- dispatch by state ----
                if self.state == GUIState.SETUP:
                    canvas = self._render_setup(key)
                elif self.state == GUIState.TRACKING:
                    canvas = self._render_tracking(key)
                elif self.state == GUIState.SELECTION:
                    canvas = self._render_selection(key)
                elif self.state == GUIState.SAVE:
                    canvas = self._render_save(key)
                else:
                    break  # shouldn't happen

                if canvas is None:       # QUIT signal
                    break

                cv2.imshow(WINDOW_NAME, canvas)

                # tick down status timer
                if self.status_timer > 0:
                    self.status_timer -= 1
                    if self.status_timer == 0:
                        self.status_message = ""
        finally:
            cv2.destroyWindow(WINDOW_NAME)
            cv2.waitKey(1)

    # =======================================================================
    # Trackbar helpers
    # =======================================================================

    def _on_trackbar(self, value: int) -> None:
        """Called by OpenCV when the user drags the trackbar."""
        if self._trackbar_locked:
            return
        if self.state in (GUIState.TRACKING,):
            return  # ignore user drags during tracking
        if value != self.current_frame_idx:
            self.current_frame_idx = value
            self._preview_dirty = True

    def _set_trackbar(self, idx: int) -> None:
        """Programmatically move the trackbar without triggering callback."""
        self._trackbar_locked = True
        cv2.setTrackbarPos(TRACKBAR_NAME, WINDOW_NAME, idx)
        self._trackbar_locked = False

    # =======================================================================
    # Frame helpers
    # =======================================================================

    def _current_frame(self) -> np.ndarray:
        ret, frame = read_frame_at(self.cap, self.current_frame_idx)
        return frame.copy() if ret else np.zeros((self.h, self.w, 3), dtype=np.uint8)

    # =======================================================================
    # Mouse callback
    # =======================================================================

    def _on_mouse(self, event: int, x: int, y: int, _flags: int, _param: object) -> None:
        if self.state != GUIState.SETUP:
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            if 0 <= x < self.w and 0 <= y < self.h:
                self.points.append((x, y))

    # =======================================================================
    # Navigation (shared by SETUP and SELECTION)
    # =======================================================================

    def _handle_navigation(self, key: int) -> bool:
        """Handle left/right navigation keys. Returns True if frame changed."""
        changed = False
        key_raw = getattr(self, '_last_key_raw', key)
        # Left:  81 (Linux), 65361 (Linux extended), 2424832 (Windows extended)
        # Right: 83 (Linux), 65363 (Linux extended), 2555904 (Windows extended)
        if key_raw in (81, 65361, 2424832):
            self.current_frame_idx = (self.current_frame_idx - 1) % self.n_frames
            changed = True
        elif key_raw in (83, 65363, 2555904):
            self.current_frame_idx = (self.current_frame_idx + 1) % self.n_frames
            changed = True
        if changed:
            self._set_trackbar(self.current_frame_idx)
            self._preview_dirty = True
        return changed

    def _handle_common_keys(self, key: int) -> bool:
        """Handle Esc quit key. Returns True if quit was triggered."""
        return key == 27  # Esc

    # =======================================================================
    # SETUP state
    # =======================================================================

    def _render_setup(self, key: int) -> np.ndarray | None:
        if self._handle_common_keys(key):
            return None

        self._handle_navigation(key)
        self._handle_setup_keys(key)

        frame = self._current_frame()
        canvas = self._draw_points(frame)
        self._draw_status_bar(canvas, "SETUP")
        return canvas

    def _handle_setup_keys(self, key: int) -> None:
        if key in (8, 127) and self.points:          # Backspace / Delete
            self.points.pop()
        elif key == ord("r") or key == ord("R"):     # R: reset points
            self.points.clear()
        elif key in _ENTER_KEYS:
            if len(self.points) >= 1:
                self.seed_frame_idx = self.current_frame_idx
                self.state = GUIState.TRACKING
                self.status_message = "Tracking..."
                self.start_tracking()

    def _draw_points(self, frame: np.ndarray) -> np.ndarray:
        """Draw numbered red circles for each selected point."""
        canvas = frame.copy()
        for i, (x, y) in enumerate(self.points, start=1):
            cv2.circle(canvas, (x, y), 7, (0, 0, 255), -1)
            cv2.circle(canvas, (x, y), 20, (0, 0, 255), 2)
            cv2.putText(canvas, str(i), (x + 8, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
        return canvas

    # =======================================================================
    # TRACKING state
    # =======================================================================

    def start_tracking(self) -> None:
        """Initialize SAM2 inference and seed with clicked points."""
        self.masks.clear()

        self.inference_state = self.predictor.init_state(
            video_path=str(self.video_path),
            offload_video_to_cpu=self.args.offload_video_to_cpu,
            offload_state_to_cpu=self.args.offload_state_to_cpu,
        )

        points_np = np.array(self.points, dtype=np.float32)
        labels_np = np.ones(len(self.points), dtype=np.int32)

        _, out_obj_ids, out_mask_logits = self.predictor.add_new_points_or_box(
            inference_state=self.inference_state,
            frame_idx=self.seed_frame_idx,
            obj_id=self.args.obj_id,
            points=points_np,
            labels=labels_np,
        )

        obj_idx = find_object_index(out_obj_ids, self.args.obj_id)
        if obj_idx is not None:
            self.masks[self.seed_frame_idx] = logits_to_mask(
                out_mask_logits, obj_idx, self.args.mask_threshold
            )

        # Start with forward propagation
        self._tracking_direction = "forward"
        self._tracking_generator = self.predictor.propagate_in_video(
            self.inference_state,
            start_frame_idx=self.seed_frame_idx,
            reverse=False,
        )
        self._tracking_frame_count = 0
        self._tracking_total = self.n_frames  # approximate (forward + backward)

    def _advance_tracking(self) -> bool:
        """Pull one frame from the active generator. Returns True if still tracking."""
        if self._tracking_generator is None:
            return False

        try:
            frame_idx, out_obj_ids, out_mask_logits = next(self._tracking_generator)
            obj_idx = find_object_index(out_obj_ids, self.args.obj_id)
            if obj_idx is not None:
                mask = logits_to_mask(out_mask_logits, obj_idx, self.args.mask_threshold)
                self.masks[int(frame_idx)] = mask
            self._tracking_frame_count += 1
            self.current_frame_idx = int(frame_idx)
            self._set_trackbar(self.current_frame_idx)
            return True
        except StopIteration:
            # Switch direction or finish
            if self._tracking_direction == "forward":
                self._tracking_direction = "backward"
                self._tracking_generator = self.predictor.propagate_in_video(
                    self.inference_state,
                    start_frame_idx=self.seed_frame_idx,
                    reverse=True,
                )
                return self._advance_tracking()  # try first backward frame
            else:
                self._tracking_generator = None
                self.state = GUIState.SELECTION
                self.current_frame_idx = self.seed_frame_idx
                self._set_trackbar(self.current_frame_idx)
                self.status_message = (
                    f"Tracking complete. {len(self.masks)} / {self.n_frames} frames have masks."
                )
                self.status_timer = 120  # show for ~2.4s at ~50 fps loop
                self._preview_dirty = True
                return False

    def _progress_pct(self) -> float:
        if self._tracking_total == 0:
            return 0.0
        return min(100.0, 100.0 * self._tracking_frame_count / self._tracking_total)

    def _render_tracking(self, key: int) -> np.ndarray | None:
        if self._handle_common_keys(key):
            return None

        # Esc aborts tracking
        if key == 27:
            self._tracking_generator = None
            self.masks.clear()
            self.state = GUIState.SETUP
            self.current_frame_idx = self.seed_frame_idx
            self._set_trackbar(self.current_frame_idx)
            self.status_message = "Tracking aborted."
            self.status_timer = 60
            return self._current_frame()

        still_tracking = self._advance_tracking()

        frame = self._current_frame()
        canvas = self._draw_mask_overlay(frame)
        self._draw_status_bar(canvas, "TRACKING")

        if still_tracking:
            pct = self._progress_pct()
            # Progress bar drawn at bottom of frame
            bar_x, bar_y, bar_w, bar_h = 20, self.h - 30, self.w - 40, 12
            cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                          (80, 80, 80), -1)
            fill_w = int(bar_w * pct / 100)
            if fill_w > 0:
                cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h),
                              (0, 220, 0), -1)
            cv2.putText(canvas, f"{pct:.0f}%  ({self._tracking_direction})",
                        (bar_x, bar_y - 6), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (200, 200, 200), 1, cv2.LINE_AA)
        return canvas

    # =======================================================================
    # SELECTION state
    # =======================================================================

    def _render_selection(self, key: int) -> np.ndarray | None:
        if self._handle_common_keys(key):
            return None

        self._handle_navigation(key)
        self._handle_selection_keys(key)

        frame = self._current_frame()

        if self.viz_mode == "composite":
            canvas = self._get_composite_preview()
        elif self.viz_mode == "mask":
            canvas = self._draw_mask_overlay(frame)
        else:  # "original"
            canvas = frame.copy()

        self._draw_status_bar(canvas, "SELECTION")
        # Attach timeline at the bottom (returns a taller image)
        canvas = _draw_timeline_on_canvas(self, canvas)
        return canvas

    def _handle_selection_keys(self, key: int) -> None:
        if key == ord("k") or key == ord("K"):
            if self.current_frame_idx in self.composite_frames:
                self.composite_frames.discard(self.current_frame_idx)
            else:
                self.composite_frames.add(self.current_frame_idx)
            self._preview_dirty = True
        elif key == ord("b") or key == ord("B"):
            self.background_frame_idx = self.current_frame_idx
            self._preview_dirty = True
            self.status_message = f"Background set to frame {self.current_frame_idx}"
            self.status_timer = 60
        elif key == ord("v") or key == ord("V"):
            # Cycle: mask → composite → original → mask
            cycle = {"mask": "composite", "composite": "original", "original": "mask"}
            self.viz_mode = cycle[self.viz_mode]
            labels = {
                "mask": "Preview: mask overlay (V to switch)",
                "composite": "Preview: composite (V to switch)",
                "original": "Preview: original (V to switch)",
            }
            self.status_message = labels[self.viz_mode]
            self.status_timer = 60
        elif key == ord("s") or key == ord("S"):
            if not self.composite_frames:
                self.status_message = "No frames marked! Press K to mark frames for compositing."
                self.status_timer = 90
                return
            self.state = GUIState.SAVE
            self.status_message = self._composite_and_save()
            self.status_timer = 120
        elif key in _ENTER_KEYS:
            # Restart: go back to SETUP
            self.predictor.reset_state(self.inference_state)
            self.inference_state = None
            self.masks.clear()
            self.composite_frames.clear()
            self.background_frame_idx = 0
            self._preview_dirty = True
            self._preview_cache = None
            self.viz_mode = "mask"
            self.state = GUIState.SETUP
            self.status_message = "Restarted. Click points and press Enter to track."
            self.status_timer = 90

    # =======================================================================
    # SAVE state
    # =======================================================================

    def _render_save(self, key: int) -> np.ndarray | None:
        if self._handle_common_keys(key):
            return None

        # Any other key returns to SELECTION
        if key >= 0 and key != 255:
            self.state = GUIState.SELECTION
            return self._render_selection(-1)

        frame = self._current_frame()
        canvas = self._draw_mask_overlay(frame)
        self._draw_status_bar(canvas, "SAVED")
        canvas = _draw_timeline_on_canvas(self, canvas)
        return canvas

    # =======================================================================
    # Drawing helpers
    # =======================================================================

    def _draw_mask_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Overlay tracked mask as semi-transparent green on the frame."""
        mask = self.masks.get(self.current_frame_idx)
        if mask is None or not mask.any():
            return frame.copy()

        canvas = frame.copy()
        overlay = np.zeros_like(frame, dtype=np.uint8)
        overlay[mask] = (0, 255, 0)
        cv2.addWeighted(overlay, 0.35, canvas, 1.0, 0, canvas)
        return canvas

    def _draw_status_bar(self, canvas: np.ndarray, state_label: str) -> None:
        """Draw a semi-transparent status bar at the top of the canvas."""
        bar_h = 32
        # Darken the top band
        overlay = canvas.copy()
        cv2.rectangle(overlay, (0, 0), (self.w, bar_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, canvas, 0.5, 0, canvas)

        if self.state == GUIState.SETUP:
            info = (f"{state_label} | Frame {self.current_frame_idx}/{self.n_frames}"
                    f" | Points: {len(self.points)}")
            if self.points:
                info += f" | Seed: {self.current_frame_idx}"
        elif self.state == GUIState.TRACKING:
            info = (f"{state_label} | Frame {self.current_frame_idx}/{self.n_frames}"
                    f" | Masks: {len(self.masks)} | Dir: {self._tracking_direction}")
        elif self.state == GUIState.SELECTION:
            is_marked = "K" if self.current_frame_idx in self.composite_frames else "-"
            is_bg = "BG" if self.current_frame_idx == self.background_frame_idx else "-"
            info = (f"{state_label} | Frame {self.current_frame_idx}/{self.n_frames}"
                    f" | Marked: {len(self.composite_frames)}"
                    f" | BG: {self.background_frame_idx}"
                    f" | Masks: {len(self.masks)}/{self.n_frames}"
                    f" | [{is_marked}][{is_bg}]")
        else:
            info = f"{state_label} | Frame {self.current_frame_idx}/{self.n_frames}"

        cv2.putText(canvas, info, (10, 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (220, 220, 220), 1, cv2.LINE_AA)

        # Status message (e.g. save confirmation) below the bar
        if self.status_message:
            cv2.putText(canvas, self.status_message, (10, bar_h + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)

    # =======================================================================
    # Composite preview & save
    # =======================================================================

    def _get_composite_preview(self) -> np.ndarray:
        """Return a rendering of the current composite (cached when possible)."""
        if not self._preview_dirty and self._preview_cache is not None:
            return self._preview_cache.copy()
        preview = self._render_composite()
        self._preview_cache = preview
        self._preview_dirty = False
        return preview.copy()

    def _render_composite(self) -> np.ndarray:
        """Blend all marked frames on top of the background, return as uint8 BGR."""
        ret, bg = read_frame_at(self.cap, self.background_frame_idx)
        if not ret:
            bg = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        canvas = bg.astype(np.float32)

        sorted_frames = sorted(self.composite_frames)
        for fidx in sorted_frames:
            mask = self.masks.get(fidx)
            if mask is None or not mask.any():
                continue
            ret, frame = read_frame_at(self.cap, fidx)
            if not ret:
                continue
            mask_clean = clean_mask(
                mask=mask,
                min_area=self.args.min_area,
                dilate_kernel=self.args.dilate_kernel,
                seed_xys=self.points if fidx == self.seed_frame_idx else None,
            )
            if not mask_clean.any():
                continue
            m = mask_clean.astype(np.float32)[..., None]
            canvas = (canvas * (1.0 - self.args.alpha * m)
                      + frame.astype(np.float32) * (self.args.alpha * m))

        # Crisp seed frame on top (100% opacity)
        seed_mask = self.masks.get(self.seed_frame_idx)
        if seed_mask is not None and seed_mask.any():
            seed_mask_clean = clean_mask(
                mask=seed_mask,
                min_area=self.args.min_area,
                dilate_kernel=self.args.dilate_kernel,
                seed_xys=self.points,
            )
            if seed_mask_clean.any():
                ret, seed_frame = read_frame_at(self.cap, self.seed_frame_idx)
                if ret:
                    m0 = seed_mask_clean.astype(np.float32)[..., None]
                    canvas = canvas * (1.0 - m0) + seed_frame.astype(np.float32) * m0

        return np.clip(canvas, 0, 255).astype(np.uint8)

    def _composite_and_save(self) -> str:
        """Composite, save to disk, and return a status message."""
        out_img = self._render_composite()
        self.args.out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(self.args.out), out_img)
        return f"Saved: {self.args.out}  ({len(self.composite_frames)} frames composited)"

    # =======================================================================
    # Cleanup
    # =======================================================================

    def close(self) -> None:
        if self.inference_state is not None:
            with contextlib.suppress(Exception):
                self.predictor.reset_state(self.inference_state)
            self.inference_state = None


# =======================================================================
# Standalone helpers
# =======================================================================

def _draw_timeline_on_canvas(gui: StroboscopicGUI, canvas: np.ndarray) -> np.ndarray:
    """Draw timeline on a copy that is TIMELINE_H px taller than the original frame."""
    h, w = canvas.shape[:2]
    n = gui.n_frames
    out = np.zeros((h + TIMELINE_H, w, 3), dtype=np.uint8)
    out[:h, :] = canvas

    y0 = h
    for fidx in range(n):
        x_start = int(w * fidx / max(n, 1))
        x_end = int(w * (fidx + 1) / max(n, 1))
        x_end = max(x_end, x_start + 1)

        if fidx in gui.composite_frames:
            color = (0, 200, 80)        # green: marked for compositing
        elif fidx in gui.masks:
            color = (60, 60, 60)        # dark gray: has mask
        else:
            color = (40, 40, 40)        # gray: no mask
        cv2.rectangle(out, (x_start, y0), (x_end, y0 + TIMELINE_H), color, -1)

    # Background indicator (orange line)
    bg_x = int(w * gui.background_frame_idx / max(n, 1))
    cv2.line(out, (bg_x, y0), (bg_x, y0 + TIMELINE_H), (255, 150, 50), 2)

    # Current position (white line)
    cur_x = int(w * gui.current_frame_idx / max(n, 1))
    cv2.line(out, (cur_x, y0), (cur_x, y0 + TIMELINE_H), (255, 255, 255), 2)

    # Legend
    cv2.putText(out, "gray=no mask  dark=has mask  green=marked  orange=bg  white=pos",
                (5, y0 + TIMELINE_H - 8), cv2.FONT_HERSHEY_SIMPLEX,
                0.35, (180, 180, 180), 1, cv2.LINE_AA)
    return out


# =======================================================================
# main()
# =======================================================================
def main() -> None:
    args = parse_args()

    if not args.video.exists():
        raise RuntimeError(f"Video not found: {args.video}")

    device_name = resolve_device(args.device)
    print(f"Loading video: {args.video}")
    print(f"Device: {device_name}")

    with tempfile.TemporaryDirectory(prefix="sam2_gui_") as tmp_dir:
        processing_video, source_fps, was_downsampled = maybe_downsample_video(
            src_video=args.video,
            target_fps=args.process_fps,
            tmp_dir=Path(tmp_dir),
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

            gui = StroboscopicGUI(
                cap=cap,
                predictor=predictor,
                video_path=processing_video,
                n_frames=n_frames,
                fps=fps,
                args=args,
                tmp_dir=Path(tmp_dir),
            )

            with sam_inference_context(device):
                gui.run()
        finally:
            cap.release()
            cv2.destroyAllWindows()

    print("Done.")


if __name__ == "__main__":
    main()
