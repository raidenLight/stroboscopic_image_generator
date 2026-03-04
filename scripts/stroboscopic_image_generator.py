#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import tempfile
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np
import torch
from sam2.build_sam import build_sam2_video_predictor
from sam2.sam2_video_predictor import SAM2VideoPredictor

ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_VIDEO = ROOT_DIR / "video" / "exp1_dwvp.MP4"
DEFAULT_OUT = ROOT_DIR / "video" / "stroboscopic_sam2.png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Track one clicked object with SAM2 and compose a stroboscopic image "
            "at fixed time intervals."
        )
    )
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--interval-sec", type=float, default=1.5)
    parser.add_argument(
        "--process-fps",
        type=float,
        default=None,
        help=(
            "FPS used for SAM2 processing. If lower than source FPS, an internal "
            "downsampled video is created to reduce memory usage."
        ),
    )
    parser.add_argument("--alpha", type=float, default=0.60)
    parser.add_argument(
        "--mask-threshold",
        type=float,
        default=0.0,
        help="Threshold for mask logits (higher gives tighter mask).",
    )
    parser.add_argument(
        "--dilate-kernel",
        type=int,
        default=5,
        help="Dilate kernel size for each tracked mask (0 disables dilation).",
    )
    parser.add_argument(
        "--min-area",
        type=int,
        default=300,
        help="Drop connected components smaller than this area in pixels.",
    )
    parser.add_argument(
        "--point",
        type=int,
        nargs=2,
        action="append",
        metavar=("X", "Y"),
        help="Skip GUI and use one or more seed points on frame 0. Repeatable.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu", "mps"),
        default="auto",
    )
    parser.add_argument("--obj-id", type=int, default=1)
    parser.add_argument(
        "--hf-model-id",
        type=str,
        default="facebook/sam2.1-hiera-small",
        help="Used when --model-cfg/--checkpoint are not provided.",
    )
    parser.add_argument(
        "--model-cfg",
        type=str,
        default=None,
        help="SAM2 config path (e.g. configs/sam2.1/sam2.1_hiera_s.yaml).",
    )
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument(
        "--vos-optimized",
        action="store_true",
        help="Enable full-model compile path in SAM2 video predictor.",
    )
    parser.add_argument(
        "--offload-video-to-cpu",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep video frames on CPU to reduce GPU memory usage.",
    )
    parser.add_argument(
        "--offload-state-to-cpu",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Offload predictor state to CPU (slower but less GPU memory).",
    )
    args = parser.parse_args()

    if not 0.0 <= args.alpha <= 1.0:
        parser.error("--alpha must be between 0.0 and 1.0")
    if args.interval_sec <= 0:
        parser.error("--interval-sec must be > 0")
    if args.process_fps is not None and args.process_fps <= 0:
        parser.error("--process-fps must be > 0")
    if args.dilate_kernel < 0:
        parser.error("--dilate-kernel must be >= 0")
    if (args.model_cfg is None) != (args.checkpoint is None):
        parser.error("--model-cfg and --checkpoint must be specified together")
    if args.checkpoint is not None and not args.checkpoint.exists():
        parser.error(f"checkpoint not found: {args.checkpoint}")
    return args


def resolve_device(requested: str) -> str:
    if requested != "auto":
        if requested == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but no CUDA device is available.")
        if requested == "mps":
            mps_ok = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
            if not mps_ok:
                raise RuntimeError("MPS was requested but is not available.")
        return requested

    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@contextlib.contextmanager
def sam_inference_context(device: str) -> Iterator[None]:
    with torch.inference_mode():
        if device.startswith("cuda"):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                yield
            return
        yield


def get_frame_count(cap: cv2.VideoCapture) -> int:
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n_frames > 0:
        return n_frames

    # Fallback for containers without frame count metadata.
    tmp = 0
    while True:
        ret, _ = cap.read()
        if not ret:
            break
        tmp += 1
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    return tmp


def read_frame_at(cap: cv2.VideoCapture, idx: int) -> tuple[bool, np.ndarray]:
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    return cap.read()


def maybe_downsample_video(
    src_video: Path,
    target_fps: float | None,
    tmp_dir: Path,
) -> tuple[Path, float, bool]:
    """
    Downsample video FPS for memory-friendly SAM2 processing.

    Returns:
        (video_path_for_processing, source_fps, was_downsampled)
    """
    cap = cv2.VideoCapture(str(src_video))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {src_video}")

    source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    if target_fps is None or target_fps >= source_fps - 1e-6:
        cap.release()
        return src_video, source_fps, False

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        cap.release()
        raise RuntimeError("Failed to read video size while preparing FPS downsampling.")

    out_path = tmp_dir / f"{src_video.stem}_fps_{target_fps:.2f}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, float(target_fps), (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot open temp writer for downsampled video: {out_path}")

    frame_idx = 0
    kept = 0
    next_keep_t = 0.0
    interval = 1.0 / float(target_fps)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        current_t = frame_idx / source_fps
        if current_t + 1e-9 >= next_keep_t:
            writer.write(frame)
            kept += 1
            next_keep_t += interval
        frame_idx += 1

    cap.release()
    writer.release()

    if kept == 0:
        raise RuntimeError("FPS downsampling produced zero frames.")

    return out_path, source_fps, True


def select_points_gui(frame: np.ndarray) -> list[tuple[int, int]]:
    window_name = "Select target points (left click)"
    selected: list[tuple[int, int]] = []

    def on_mouse(event: int, x: int, y: int, _flags: int, _param: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            selected.append((x, y))

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, on_mouse)

    while True:
        canvas = frame.copy()
        cv2.putText(
            canvas,
            "Left click:add point / Backspace:undo / R:reset / Enter:confirm / Esc:cancel",
            (20, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            f"selected points: {len(selected)}",
            (20, 64),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        for i, (x, y) in enumerate(selected, start=1):
            cv2.circle(canvas, (x, y), 7, (0, 0, 255), -1)
            cv2.circle(canvas, (x, y), 20, (0, 0, 255), 2)
            cv2.putText(
                canvas,
                str(i),
                (x + 8, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

        cv2.imshow(window_name, canvas)
        key = cv2.waitKey(20) & 0xFF

        if key in (13, 10, 32):  # Enter / Return / Space
            if selected:
                cv2.destroyWindow(window_name)
                return selected
        elif key in (8, 127):  # Backspace / Delete
            if selected:
                selected.pop()
        elif key in (ord("r"), ord("R")):
            selected.clear()
        elif key in (27, ord("q"), ord("Q")):
            cv2.destroyWindow(window_name)
            raise RuntimeError("Selection cancelled by user.")


def find_object_index(obj_ids: list[int], obj_id: int) -> int | None:
    for idx, current in enumerate(obj_ids):
        if int(current) == int(obj_id):
            return idx
    return None


def logits_to_mask(
    out_mask_logits: torch.Tensor,
    obj_idx: int,
    threshold: float,
) -> np.ndarray:
    obj_mask_logits = out_mask_logits[obj_idx]
    if obj_mask_logits.ndim == 3:
        obj_mask_logits = obj_mask_logits[0]
    return (obj_mask_logits > threshold).detach().cpu().numpy()


def build_predictor(args: argparse.Namespace, device: str) -> SAM2VideoPredictor:
    if args.model_cfg is not None and args.checkpoint is not None:
        return build_sam2_video_predictor(
            config_file=args.model_cfg,
            ckpt_path=str(args.checkpoint),
            device=device,
            vos_optimized=args.vos_optimized,
        )

    return SAM2VideoPredictor.from_pretrained(
        args.hf_model_id,
        device=device,
        vos_optimized=args.vos_optimized,
    )


def track_masks_on_sampled_frames(
    predictor: SAM2VideoPredictor,
    args: argparse.Namespace,
    video_path: Path,
    sample_idxs: set[int],
    point_xys: list[tuple[int, int]],
) -> dict[int, np.ndarray]:
    inference_state = predictor.init_state(
        video_path=str(video_path),
        offload_video_to_cpu=args.offload_video_to_cpu,
        offload_state_to_cpu=args.offload_state_to_cpu,
    )

    points = np.array(point_xys, dtype=np.float32)
    labels = np.ones((len(point_xys),), dtype=np.int32)
    _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=0,
        obj_id=args.obj_id,
        points=points,
        labels=labels,
    )

    obj_idx = find_object_index(out_obj_ids, args.obj_id)
    if obj_idx is None:
        raise RuntimeError(f"Object id {args.obj_id} not found in initial prediction output.")

    masks: dict[int, np.ndarray] = {}
    if 0 in sample_idxs:
        masks[0] = logits_to_mask(out_mask_logits, obj_idx, args.mask_threshold)

    for frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(
        inference_state
    ):
        idx = int(frame_idx)
        if idx not in sample_idxs:
            continue
        obj_idx = find_object_index(out_obj_ids, args.obj_id)
        if obj_idx is None:
            continue
        masks[idx] = logits_to_mask(out_mask_logits, obj_idx, args.mask_threshold)

    return masks


def clean_mask(
    mask: np.ndarray,
    min_area: int,
    dilate_kernel: int,
    seed_xys: list[tuple[int, int]] | None,
) -> np.ndarray:
    out = (mask > 0).astype(np.uint8)

    if min_area > 0 and out.any():
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(out, connectivity=8)
        filtered = np.zeros_like(out)

        # Keep components containing clicked points if available.
        if seed_xys:
            keep_labels: set[int] = set()
            for x, y in seed_xys:
                if 0 <= x < out.shape[1] and 0 <= y < out.shape[0]:
                    seed_label = int(labels[y, x])
                    if seed_label > 0 and int(stats[seed_label, cv2.CC_STAT_AREA]) >= min_area:
                        keep_labels.add(seed_label)
            for keep_label in keep_labels:
                filtered[labels == keep_label] = 1

        # If the seed component is not available, keep all sufficiently large components.
        if not filtered.any():
            for label in range(1, num_labels):
                if int(stats[label, cv2.CC_STAT_AREA]) >= min_area:
                    filtered[labels == label] = 1

        out = filtered

    if dilate_kernel > 0 and out.any():
        kernel = np.ones((dilate_kernel, dilate_kernel), dtype=np.uint8)
        out = cv2.dilate(out, kernel, iterations=1)

    return out.astype(bool)


def main() -> None:
    args = parse_args()
    if not args.video.exists():
        raise RuntimeError(f"Video not found: {args.video}")

    with tempfile.TemporaryDirectory(prefix="sam2_fps_") as tmp_dir:
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
            step = max(1, int(round(args.interval_sec * fps)))
            n_frames = get_frame_count(cap)
            sample_idxs = set(range(0, n_frames, step))
            sample_idxs.add(0)

            ret, first_frame = read_frame_at(cap, 0)
            if not ret:
                raise RuntimeError("Cannot read first frame.")

            if args.point is None:
                try:
                    point_xys = select_points_gui(first_frame)
                except cv2.error as exc:
                    raise RuntimeError(
                        "OpenCV GUI is unavailable. Use --point X Y (repeatable) to run headless."
                    ) from exc
            else:
                point_xys = [(int(x), int(y)) for x, y in args.point]
            if not point_xys:
                raise RuntimeError("At least one point prompt is required.")

            h, w = first_frame.shape[:2]
            for point_xy in point_xys:
                if not (0 <= point_xy[0] < w and 0 <= point_xy[1] < h):
                    raise RuntimeError(
                        f"Point {point_xy} is outside frame size (w={w}, h={h})."
                    )

            background = first_frame.copy()

            device = resolve_device(args.device)
            predictor = build_predictor(args, device=device)
            with sam_inference_context(device):
                masks = track_masks_on_sampled_frames(
                    predictor=predictor,
                    args=args,
                    video_path=processing_video,
                    sample_idxs=sample_idxs,
                    point_xys=point_xys,
                )

            canvas = background.astype(np.float32)
            for idx in sorted(sample_idxs, reverse=True):
                mask = masks.get(idx)
                if mask is None:
                    continue

                ret, frame = read_frame_at(cap, idx)
                if not ret:
                    continue

                mask = clean_mask(
                    mask=mask,
                    min_area=args.min_area,
                    dilate_kernel=args.dilate_kernel,
                    seed_xys=point_xys if idx == 0 else None,
                )
                if not mask.any():
                    continue

                m = mask.astype(np.float32)[..., None]
                canvas = (
                    canvas * (1.0 - args.alpha * m) + frame.astype(np.float32) * (args.alpha * m)
                )

            # Keep the initial pose crisp on top.
            if 0 in masks:
                start_mask = clean_mask(
                    mask=masks[0],
                    min_area=args.min_area,
                    dilate_kernel=args.dilate_kernel,
                    seed_xys=point_xys,
                )
                if start_mask.any():
                    m0 = start_mask.astype(np.float32)[..., None]
                    canvas = canvas * (1.0 - m0) + first_frame.astype(np.float32) * m0
        finally:
            cap.release()
            cv2.destroyAllWindows()

    out_img = np.clip(canvas, 0, 255).astype(np.uint8)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.out), out_img)
    print("saved:", args.out)
    if was_downsampled:
        print("processing_fps: {:.2f} -> {:.2f}".format(source_fps, fps))
    else:
        print("processing_fps: {:.2f} (no downsampling)".format(fps))
    print(
        "fps={:.2f}, interval={}s, step={} frames, n_frames={}, sampled_frames={}".format(
            fps,
            args.interval_sec,
            step,
            n_frames,
            len(sample_idxs),
        )
    )
    print("device:", device)
    print("click points:", point_xys)


if __name__ == "__main__":
    main()
