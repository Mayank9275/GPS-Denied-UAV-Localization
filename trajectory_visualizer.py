#!/usr/bin/env python3
"""
Standalone trajectory comparison video generator for UAV localization results.

This module keeps the localization pipeline untouched. It reads:
    - UAV frames
    - DJI SRT telemetry
    - satellite image
    - satellite world file (.pgw)
    - per-frame localization prediction files

It renders a split-screen video:
    - Left: current UAV frame
    - Right: fixed satellite map with GT and predicted trajectories
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
from pyproj import CRS, Transformer


SRT_BLOCK_PATTERN = re.compile(
    r"(?P<index>\d+)\s+"
    r"(?P<time_range>\d{2}:\d{2}:\d{2},\d{3}\s+-->\s+\d{2}:\d{2}:\d{2},\d{3})\s+"
    r"<font[^>]*>SrtCnt\s*:\s*(?P<srt_cnt>\d+),\s*DiffTime\s*:\s*(?P<diff_ms>\d+)ms\s+"
    r"(?P<timestamp>\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2},\d{3},\d{3})\s+"
    r"(?P<telemetry>.*?)"
    r"</font>",
    re.DOTALL,
)

FIELD_PATTERNS = {
    "latitude": re.compile(r"\[latitude:\s*([-+]?\d+(?:\.\d+)?)\]"),
    "longitude": re.compile(r"\[longitude:\s*([-+]?\d+(?:\.\d+)?)\]"),
    "altitude": re.compile(r"\[altitude:\s*([-+]?\d+(?:\.\d+)?)\]"),
}

FRAME_ID_PATTERN = re.compile(r"(\d+)(?!.*\d)")
DEFAULT_TEXT_COLOR = (255, 255, 255)
DEFAULT_TEXT_SHADOW = (0, 0, 0)
GT_COLOR = (0, 220, 0)
PRED_COLOR = (0, 0, 255)


@dataclass(frozen=True)
class TelemetryRecord:
    frame_id: int
    srt_index: int
    srt_counter: int
    time_range: str
    timestamp: str
    latitude: float
    longitude: float
    altitude: float


@dataclass(frozen=True)
class PredictionRecord:
    frame_id: int
    pixel_x: float
    pixel_y: float
    world_x: float
    world_y: float
    inliers: int | None = None
    source_path: str | None = None


@dataclass(frozen=True)
class MatchedFrame:
    frame_id: int
    frame_path: Path
    telemetry: dict[str, Any]
    prediction: PredictionRecord
    raw_srt_id: int
    raw_prediction_id: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a ground-truth vs prediction trajectory comparison video.",
    )
    parser.add_argument("--satellite-image", type=Path, required=True)
    parser.add_argument("--pgw-file", type=Path, required=True)
    parser.add_argument("--srt-file", type=Path, required=True)
    parser.add_argument("--frames-dir", type=Path, required=True)
    parser.add_argument(
        "--predictions-dir",
        type=Path,
        help="Directory containing per-frame JSON prediction files.",
    )
    parser.add_argument(
        "--prediction-files",
        type=Path,
        nargs="*",
        help="Explicit list of per-frame prediction JSON files.",
    )
    parser.add_argument(
        "--map-crs",
        type=str,
        default="EPSG:3857",
        help="CRS of the satellite map/world file coordinates.",
    )
    parser.add_argument(
        "--gps-crs",
        type=str,
        default="EPSG:4326",
        help="CRS of the DJI GPS telemetry coordinates.",
    )
    parser.add_argument(
        "--frame-glob",
        type=str,
        default="*.jpg",
        help="Glob used to collect UAV frames from --frames-dir.",
    )
    parser.add_argument(
        "--output-video",
        type=Path,
        default=Path("trajectory_comparison.mp4"),
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Override output FPS. Defaults to the SRT-derived rate when available.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional cap for debugging or short previews.",
    )
    parser.add_argument(
        "--trail-length",
        type=int,
        default=0,
        help="Number of history points to retain. Use 0 for the full trajectory.",
    )
    parser.add_argument(
        "--prediction-coord-type",
        choices=("auto", "pixel", "world"),
        default="auto",
        help="How to interpret prediction coordinates when loading JSON files.",
    )
    parser.add_argument(
        "--show-map-grid",
        action="store_true",
        help="Draw a subtle grid over the satellite view for visual inspection.",
    )
    parser.add_argument(
        "--srt-offset",
        type=int,
        default=None,
        help="Optional manual offset applied to raw SRT ids to align them with frame ids.",
    )
    parser.add_argument(
        "--prediction-offset",
        type=int,
        default=None,
        help="Optional manual offset applied to raw prediction ids to align them with frame ids.",
    )
    parser.add_argument(
        "--frame-step",
        type=int,
        default=1,
        help=(
            "Sampling interval between extracted UAV frames and original video frames. "
            "For every 10th-frame extraction, use --frame-step 10."
        ),
    )
    return parser.parse_args()


def read_pgw(path: Path) -> dict[str, float]:
    values: list[float] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if text:
                values.append(float(text))

    if len(values) != 6:
        raise ValueError(
            f"Expected 6 numeric lines in PGW file, got {len(values)} from {path}",
        )

    return {
        "a": values[0],
        "d": values[1],
        "b": values[2],
        "e": values[3],
        "c": values[4],
        "f": values[5],
    }


def world_to_pixel(world_x: float, world_y: float, pgw: dict[str, float]) -> tuple[float, float]:
    matrix = np.array(
        [
            [pgw["a"], pgw["b"]],
            [pgw["d"], pgw["e"]],
        ],
        dtype=np.float64,
    )
    offset = np.array([world_x - pgw["c"], world_y - pgw["f"]], dtype=np.float64)
    col_row = np.linalg.solve(matrix, offset)
    return float(col_row[0]), float(col_row[1])


def pixel_to_world(col: float, row: float, pgw: dict[str, float]) -> tuple[float, float]:
    world_x = pgw["a"] * float(col) + pgw["b"] * float(row) + pgw["c"]
    world_y = pgw["d"] * float(col) + pgw["e"] * float(row) + pgw["f"]
    return float(world_x), float(world_y)


def build_crs_transformer(source_crs: str, target_crs: str) -> Transformer:
    return Transformer.from_crs(
        CRS.from_user_input(source_crs),
        CRS.from_user_input(target_crs),
        always_xy=True,
    )


def latlon_to_world(
    latitude: float,
    longitude: float,
    transformer: Transformer,
) -> tuple[float, float]:
    world_x, world_y = transformer.transform(longitude, latitude)
    return float(world_x), float(world_y)


def parse_srt(srt_path: Path) -> list[TelemetryRecord]:
    text = srt_path.read_text(encoding="utf-8", errors="ignore")
    records: list[TelemetryRecord] = []

    for match in SRT_BLOCK_PATTERN.finditer(text):
        telemetry_text = match.group("telemetry")
        latitude = extract_numeric_field("latitude", telemetry_text)
        longitude = extract_numeric_field("longitude", telemetry_text)
        altitude = extract_numeric_field("altitude", telemetry_text)
        srt_counter = int(match.group("srt_cnt"))

        records.append(
            TelemetryRecord(
                frame_id=srt_counter,
                srt_index=int(match.group("index")),
                srt_counter=srt_counter,
                time_range=match.group("time_range"),
                timestamp=match.group("timestamp"),
                latitude=latitude,
                longitude=longitude,
                altitude=altitude,
            )
        )

    if not records:
        raise ValueError(f"No telemetry records were parsed from {srt_path}")

    return records


def extract_numeric_field(field_name: str, telemetry_text: str) -> float:
    pattern = FIELD_PATTERNS[field_name]
    match = pattern.search(telemetry_text)
    if match is None:
        raise ValueError(f"Missing [{field_name}: ...] in SRT telemetry block")
    return float(match.group(1))


def infer_srt_fps(records: list[TelemetryRecord]) -> float | None:
    if len(records) < 2:
        return None

    durations_ms: list[float] = []
    for record in records:
        time_range = record.time_range.split("-->")
        if len(time_range) != 2:
            continue
        start_ms = parse_srt_timestamp_to_ms(time_range[0].strip())
        end_ms = parse_srt_timestamp_to_ms(time_range[1].strip())
        if end_ms > start_ms:
            durations_ms.append(end_ms - start_ms)

    if not durations_ms:
        return None

    mean_duration_ms = sum(durations_ms) / len(durations_ms)
    if mean_duration_ms <= 0:
        return None

    return 1000.0 / mean_duration_ms


def parse_srt_timestamp_to_ms(timestamp: str) -> float:
    hours, minutes, seconds_millis = timestamp.split(":")
    seconds, millis = seconds_millis.split(",")
    total_ms = (
        int(hours) * 3600 * 1000
        + int(minutes) * 60 * 1000
        + int(seconds) * 1000
        + int(millis)
    )
    return float(total_ms)


def collect_frame_paths(frames_dir: Path, frame_glob: str) -> dict[int, Path]:
    frame_paths: dict[int, Path] = {}
    for path in sorted(frames_dir.glob(frame_glob)):
        if not path.is_file():
            continue
        frame_id = extract_frame_id(path.stem)
        frame_paths[frame_id] = path

    if not frame_paths:
        raise FileNotFoundError(
            f"No frame files matching '{frame_glob}' were found under {frames_dir}",
        )

    return frame_paths


def collect_prediction_paths(
    predictions_dir: Path | None,
    explicit_files: Iterable[Path] | None,
) -> list[Path]:
    files: list[Path] = []
    if predictions_dir is not None:
        files.extend(sorted(predictions_dir.glob("*.json")))
    if explicit_files is not None:
        files.extend(explicit_files)

    unique_files = sorted({path.resolve() for path in files})
    if not unique_files:
        raise FileNotFoundError(
            "No prediction JSON files were provided. Use --predictions-dir or --prediction-files.",
        )
    return unique_files


def load_predictions(
    prediction_paths: Iterable[Path],
    pgw: dict[str, float],
    coord_type: str,
) -> dict[int, PredictionRecord]:
    predictions: dict[int, PredictionRecord] = {}

    for path in prediction_paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        frame_id = infer_prediction_frame_id(data, path)
        prediction = prediction_from_json(
            frame_id=frame_id,
            data=data,
            pgw=pgw,
            coord_type=coord_type,
            source_path=str(path),
        )
        predictions[frame_id] = prediction

    if not predictions:
        raise ValueError("No usable prediction records were loaded.")

    return predictions


def infer_prediction_frame_id(data: dict[str, Any], path: Path) -> int:
    candidate_keys = (
        "frame_id",
        "frame_index",
        "frame_idx",
        "index",
    )
    for key in candidate_keys:
        if key in data:
            return int(data[key])

    if "uav_image" in data and data["uav_image"]:
        return extract_frame_id(Path(str(data["uav_image"])).stem)

    return extract_frame_id(path.stem)


def prediction_from_json(
    *,
    frame_id: int,
    data: dict[str, Any],
    pgw: dict[str, float],
    coord_type: str,
    source_path: str,
) -> PredictionRecord:
    if coord_type == "auto":
        if "pred_col" in data and "pred_row" in data:
            coord_type = "pixel"
        elif "pred_world_x" in data and "pred_world_y" in data:
            coord_type = "world"
        else:
            raise ValueError(
                f"Could not infer coordinate type for prediction JSON: {source_path}",
            )

    if coord_type == "pixel":
        pixel_x = float(read_first_available(data, ("pred_col", "predicted_x", "x", "col")))
        pixel_y = float(read_first_available(data, ("pred_row", "predicted_y", "y", "row")))
        world_x, world_y = pixel_to_world(pixel_x, pixel_y, pgw)
    elif coord_type == "world":
        world_x = float(read_first_available(data, ("pred_world_x", "predicted_x", "x", "world_x")))
        world_y = float(read_first_available(data, ("pred_world_y", "predicted_y", "y", "world_y")))
        pixel_x, pixel_y = world_to_pixel(world_x, world_y, pgw)
    else:
        raise ValueError(f"Unsupported prediction coordinate type: {coord_type}")

    inliers = data.get("best_inliers")
    if inliers is not None:
        inliers = int(inliers)

    return PredictionRecord(
        frame_id=frame_id,
        pixel_x=pixel_x,
        pixel_y=pixel_y,
        world_x=world_x,
        world_y=world_y,
        inliers=inliers,
        source_path=source_path,
    )


def read_first_available(data: dict[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        if key in data and data[key] is not None:
            return data[key]
    raise KeyError(f"None of the keys were found: {tuple(keys)}")


def extract_frame_id(text: str) -> int:
    match = FRAME_ID_PATTERN.search(text)
    if match is None:
        raise ValueError(f"Could not extract a frame id from '{text}'")
    return int(match.group(1))


def telemetry_to_pixel_records(
    telemetry_records: Iterable[TelemetryRecord],
    transformer: Transformer,
    pgw: dict[str, float],
) -> dict[int, dict[str, Any]]:
    pixel_records: dict[int, dict[str, Any]] = {}
    for record in telemetry_records:
        world_x, world_y = latlon_to_world(
            latitude=record.latitude,
            longitude=record.longitude,
            transformer=transformer,
        )
        pixel_x, pixel_y = world_to_pixel(world_x, world_y, pgw)
        pixel_records[record.frame_id] = {
            "telemetry": record,
            "raw_id": record.frame_id,
            "world_x": world_x,
            "world_y": world_y,
            "pixel_x": pixel_x,
            "pixel_y": pixel_y,
        }
    return pixel_records


def compute_error_meters(
    gt_world_x: float,
    gt_world_y: float,
    pred_world_x: float,
    pred_world_y: float,
) -> float:
    return float(math.hypot(pred_world_x - gt_world_x, pred_world_y - gt_world_y))


def resize_to_height(image: np.ndarray, target_height: int) -> np.ndarray:
    h, w = image.shape[:2]
    if h == target_height:
        return image
    scale = target_height / float(h)
    target_width = max(1, int(round(w * scale)))
    return cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_AREA)


def draw_polyline(
    canvas: np.ndarray,
    points: list[tuple[float, float]],
    color: tuple[int, int, int],
    thickness: int,
) -> None:
    if len(points) < 2:
        return
    poly = np.array(
        [[int(round(x)), int(round(y))] for x, y in points],
        dtype=np.int32,
    )
    cv2.polylines(canvas, [poly], False, color, thickness, cv2.LINE_AA)


def draw_current_point(
    canvas: np.ndarray,
    point: tuple[float, float],
    color: tuple[int, int, int],
    radius: int,
) -> None:
    px = int(round(point[0]))
    py = int(round(point[1]))
    cv2.circle(canvas, (px, py), radius, color, -1, cv2.LINE_AA)
    cv2.circle(canvas, (px, py), radius + 2, (255, 255, 255), 1, cv2.LINE_AA)


def draw_map_grid(canvas: np.ndarray, spacing: int = 200) -> None:
    h, w = canvas.shape[:2]
    grid_color = (60, 60, 60)
    for x in range(0, w, spacing):
        cv2.line(canvas, (x, 0), (x, h - 1), grid_color, 1, cv2.LINE_AA)
    for y in range(0, h, spacing):
        cv2.line(canvas, (0, y), (w - 1, y), grid_color, 1, cv2.LINE_AA)


def draw_text_block(
    image: np.ndarray,
    lines: list[str],
    origin: tuple[int, int] = (16, 28),
    line_height: int = 24,
) -> None:
    x, y = origin
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.62
    thickness = 1

    max_width = 0
    for line in lines:
        (text_width, _), _ = cv2.getTextSize(line, font, scale, thickness)
        max_width = max(max_width, text_width)

    box_height = line_height * len(lines) + 14
    overlay = image.copy()
    cv2.rectangle(
        overlay,
        (x - 10, y - 22),
        (x + max_width + 18, y - 22 + box_height),
        (0, 0, 0),
        -1,
    )
    cv2.addWeighted(overlay, 0.42, image, 0.58, 0.0, dst=image)

    for idx, line in enumerate(lines):
        baseline_y = y + idx * line_height
        cv2.putText(
            image,
            line,
            (x, baseline_y),
            font,
            scale,
            DEFAULT_TEXT_SHADOW,
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            line,
            (x, baseline_y),
            font,
            scale,
            DEFAULT_TEXT_COLOR,
            thickness,
            cv2.LINE_AA,
        )


def clamp_history(
    history: list[tuple[float, float]],
    trail_length: int,
) -> list[tuple[float, float]]:
    if trail_length <= 0 or len(history) <= trail_length:
        return history
    return history[-trail_length:]


def build_info_lines(
    *,
    matched_frame_id: int,
    telemetry: TelemetryRecord,
    prediction: PredictionRecord,
    error_meters: float,
) -> list[str]:
    return [
        f"Frame: {matched_frame_id}",
        f"SRT Counter: {telemetry.srt_counter}",
        f"Timestamp: {telemetry.timestamp}",
        f"Latitude: {telemetry.latitude:.6f}",
        f"Longitude: {telemetry.longitude:.6f}",
        f"GT Altitude: {telemetry.altitude:.2f} m",
        f"Predicted Pixel: ({prediction.pixel_x:.1f}, {prediction.pixel_y:.1f})",
        f"Localization Error: {error_meters:.2f} m",
        (
            f"PnP Inliers: {prediction.inliers}"
            if prediction.inliers is not None
            else "PnP Inliers: N/A"
        ),
    ]


def render_frame(
    *,
    uav_frame: np.ndarray,
    satellite_base: np.ndarray,
    gt_history: list[tuple[float, float]],
    pred_history: list[tuple[float, float]],
    gt_point: tuple[float, float],
    pred_point: tuple[float, float],
    matched_frame_id: int,
    telemetry: TelemetryRecord,
    prediction: PredictionRecord,
    error_meters: float,
    show_map_grid: bool,
) -> np.ndarray:
    satellite_panel = satellite_base.copy()
    if show_map_grid:
        draw_map_grid(satellite_panel)

    draw_polyline(satellite_panel, gt_history, GT_COLOR, thickness=3)
    draw_polyline(satellite_panel, pred_history, PRED_COLOR, thickness=3)
    draw_current_point(satellite_panel, gt_point, GT_COLOR, radius=6)
    draw_current_point(satellite_panel, pred_point, PRED_COLOR, radius=6)
    draw_text_block(
        satellite_panel,
        build_info_lines(
            matched_frame_id=matched_frame_id,
            telemetry=telemetry,
            prediction=prediction,
            error_meters=error_meters,
        ),
    )

    target_height = max(uav_frame.shape[0], satellite_panel.shape[0])
    left = resize_to_height(uav_frame, target_height)
    right = resize_to_height(satellite_panel, target_height)

    separator = np.full((target_height, 12, 3), 24, dtype=np.uint8)
    return np.hstack([left, separator, right])


def generate_video(
    *,
    matched_frames: list[MatchedFrame],
    satellite_image: np.ndarray,
    output_video: Path,
    fps: float,
    max_frames: int | None,
    trail_length: int,
    show_map_grid: bool,
) -> tuple[int, Path]:
    if not matched_frames:
        raise ValueError(
            "No matched frames were available for rendering.",
        )

    if max_frames is not None:
        matched_frames = matched_frames[:max_frames]

    gt_history: list[tuple[float, float]] = []
    pred_history: list[tuple[float, float]] = []
    writer: cv2.VideoWriter | None = None
    frame_count = 0

    for matched in matched_frames:
        frame_path = matched.frame_path
        uav_frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        if uav_frame is None:
            raise RuntimeError(f"Failed to read UAV frame: {frame_path}")

        telemetry_item = matched.telemetry
        prediction = matched.prediction

        gt_point = (telemetry_item["pixel_x"], telemetry_item["pixel_y"])
        pred_point = (prediction.pixel_x, prediction.pixel_y)
        gt_history.append(gt_point)
        pred_history.append(pred_point)

        visible_gt_history = clamp_history(gt_history, trail_length)
        visible_pred_history = clamp_history(pred_history, trail_length)

        error_meters = compute_error_meters(
            telemetry_item["world_x"],
            telemetry_item["world_y"],
            prediction.world_x,
            prediction.world_y,
        )

        composed_frame = render_frame(
            uav_frame=uav_frame,
            satellite_base=satellite_image,
            gt_history=visible_gt_history,
            pred_history=visible_pred_history,
            gt_point=gt_point,
            pred_point=pred_point,
            matched_frame_id=matched.frame_id,
            telemetry=telemetry_item["telemetry"],
            prediction=prediction,
            error_meters=error_meters,
            show_map_grid=show_map_grid,
        )

        if writer is None:
            output_video.parent.mkdir(parents=True, exist_ok=True)
            height, width = composed_frame.shape[:2]
            writer = cv2.VideoWriter(
                str(output_video),
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                (width, height),
            )
            if not writer.isOpened():
                raise RuntimeError(f"Failed to open video writer for {output_video}")

        writer.write(composed_frame)
        frame_count += 1

    if writer is None:
        raise RuntimeError("Video writer was never initialized.")

    writer.release()
    return frame_count, output_video


def compute_best_offset(reference_ids: Iterable[int], candidate_ids: Iterable[int]) -> int:
    reference_set = set(reference_ids)
    candidate_list = list(candidate_ids)
    if not reference_set or not candidate_list:
        return 0

    offset_counts: Counter[int] = Counter()
    for reference_id in reference_set:
        for candidate_id in candidate_list:
            offset_counts[reference_id - candidate_id] += 1

    best_overlap = -1
    best_offset = 0
    for offset, _ in offset_counts.most_common():
        overlap = sum(1 for candidate_id in candidate_list if candidate_id + offset in reference_set)
        if overlap > best_overlap or (overlap == best_overlap and abs(offset) < abs(best_offset)):
            best_overlap = overlap
            best_offset = offset

    return best_offset

from typing import TypeVar

T = TypeVar("T")

def apply_offset_to_mapping(items: dict[int, T], offset: int) -> dict[int, tuple[int, T]]:
    aligned: dict[int, tuple[int, T]] = {}
    for raw_id, item in items.items():
        aligned[raw_id + offset] = (raw_id, item)
    return aligned


def build_matched_frames(
    *,
    frame_paths: dict[int, Path],
    telemetry_pixels: dict[int, dict[str, Any]],
    predictions: dict[int, PredictionRecord],
    srt_offset: int | None,
    prediction_offset: int | None,
    frame_step: int,
) -> tuple[list[MatchedFrame], dict[str, int]]:
    if frame_step <= 0:
        raise ValueError(f"--frame-step must be a positive integer, got {frame_step}")

    frame_ids = sorted(frame_paths)
    telemetry_ids = sorted(telemetry_pixels)
    prediction_ids = sorted(predictions)

    target_srt_ids = [frame_id * frame_step for frame_id in frame_ids]
    resolved_srt_offset = (
        srt_offset
        if srt_offset is not None
        else compute_best_offset(target_srt_ids, telemetry_ids)
    )
    resolved_prediction_offset = (
        prediction_offset
        if prediction_offset is not None
        else compute_best_offset(frame_ids, prediction_ids)
    )

    aligned_telemetry_by_frame: dict[int, tuple[int, dict[str, Any]]] = {}
    for frame_id in frame_ids:
        target_srt_id = frame_id * frame_step
        raw_srt_id = target_srt_id - resolved_srt_offset
        telemetry_item = telemetry_pixels.get(raw_srt_id)
        if telemetry_item is not None:
            aligned_telemetry_by_frame[frame_id] = (raw_srt_id, telemetry_item)

    aligned_predictions = apply_offset_to_mapping(predictions, resolved_prediction_offset)

    matched_frame_ids = sorted(
        set(frame_ids) & set(aligned_telemetry_by_frame) & set(aligned_predictions)
    )
    matched_frames: list[MatchedFrame] = []
    for frame_id in matched_frame_ids:
        raw_srt_id, telemetry_item = aligned_telemetry_by_frame[frame_id]
        raw_prediction_id, prediction_item = aligned_predictions[frame_id]
        matched_frames.append(
            MatchedFrame(
                frame_id=frame_id,
                frame_path=frame_paths[frame_id],
                telemetry=telemetry_item,
                prediction=prediction_item,
                raw_srt_id=raw_srt_id,
                raw_prediction_id=raw_prediction_id,
            )
        )

    skipped_frames = len(frame_paths) - len(matched_frames)
    summary = {
        "uav_frames": len(frame_paths),
        "srt_entries": len(telemetry_pixels),
        "prediction_jsons": len(predictions),
        "matched_frames": len(matched_frames),
        "skipped_frames": skipped_frames,
        "srt_offset": resolved_srt_offset,
        "prediction_offset": resolved_prediction_offset,
        "frame_step": frame_step,
    }
    return matched_frames, summary


def print_match_summary(matched_frames: list[MatchedFrame], summary: dict[str, int]) -> None:
    print(f"UAV frames: {summary['uav_frames']}")
    print(f"SRT entries: {summary['srt_entries']}")
    print(f"Prediction JSONs: {summary['prediction_jsons']}")
    print(f"Successfully matched frames: {summary['matched_frames']}")
    print(f"Skipped frames: {summary['skipped_frames']}")
    print(f"Frame step used: {summary['frame_step']}")
    print(f"SRT offset used: {summary['srt_offset']}")
    print(f"Prediction offset used: {summary['prediction_offset']}")
    print("First five matched tuples:")

    for matched in matched_frames[:5]:
        print(f"Frame {matched.frame_id}")
        print(f"SRT {matched.raw_srt_id}")
        print(f"Prediction {matched.raw_prediction_id}")
        print("---")


def main() -> None:
    args = parse_args()

    satellite_image = cv2.imread(str(args.satellite_image), cv2.IMREAD_COLOR)
    if satellite_image is None:
        raise RuntimeError(f"Failed to read satellite image: {args.satellite_image}")

    pgw = read_pgw(args.pgw_file)
    telemetry = parse_srt(args.srt_file)
    transformer = build_crs_transformer(args.gps_crs, args.map_crs)
    telemetry_pixels = telemetry_to_pixel_records(telemetry, transformer, pgw)
    frame_paths = collect_frame_paths(args.frames_dir, args.frame_glob)
    prediction_paths = collect_prediction_paths(args.predictions_dir, args.prediction_files)
    predictions = load_predictions(
        prediction_paths=prediction_paths,
        pgw=pgw,
        coord_type=args.prediction_coord_type,
    )
    matched_frames, summary = build_matched_frames(
        frame_paths=frame_paths,
        telemetry_pixels=telemetry_pixels,
        predictions=predictions,
        srt_offset=args.srt_offset,
        prediction_offset=args.prediction_offset,
        frame_step=args.frame_step,
    )
    print_match_summary(matched_frames, summary)

    inferred_fps = infer_srt_fps(telemetry)
    fps = args.fps if args.fps is not None else (inferred_fps or 30.0)

    frame_count, output_video = generate_video(
        matched_frames=matched_frames,
        satellite_image=satellite_image,
        output_video=args.output_video,
        fps=fps,
        max_frames=args.max_frames,
        trail_length=args.trail_length,
        show_map_grid=args.show_map_grid,
    )

    print(f"Wrote {frame_count} frames to {output_video}")


if __name__ == "__main__":
    main()
