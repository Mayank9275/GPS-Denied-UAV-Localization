#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Batch-localize UAV frames and measure inference performance.

Run:
    python run_uav_folder.py

Outputs:
    outputs/<frame>.json
    outputs/<frame>_prediction.png
    outputs/<frame>_overlay.png
    outputs/<frame>_debug.png
    outputs/profiling/frame_timings.csv
    outputs/profiling/first_frame_profile.txt
"""

from __future__ import annotations

import cProfile
import csv
import inspect
import io
import json
import pstats
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from pyproj import Geod

from single_image_localization import (
    init_real_localization_context,
    localize_image,
)
from trajectory_visualizer import (
    TelemetryRecord,
    extract_frame_id,
    parse_srt,
)


# =============================================================================
# Paths
# =============================================================================
FRAMES_DIR = Path(r"Data\Vedio_2\Extracted-frames")
SATELLITE_MAP = Path(r"Data\Vedio_2\satellite_map.png")
PGW_FILE = Path(r"Data\Vedio_1\satellite_map.pgw")
SRT_FILE = Path(r"Data\Vedio_2\flight_telemetry.srt")

OUTPUT_DIR = Path(r"Data\Vedio_2\lol")
WORK_DIR = OUTPUT_DIR / "_work"
PROFILE_DIR = OUTPUT_DIR / "profiling"
TIMING_CSV = PROFILE_DIR / "frame_timings.csv"

SOURCE_VIDEO_FPS = 30.0
FRAME_EXTRACTION_STEP = 5


# =============================================================================
# Camera and flight configuration
# =============================================================================

ALTITUDE = 474.0

K_MATRIX = np.array(
    [
        [1066.0, 0.0, 960.0],
        [0.0, 1066.0, 540.0],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float32,
)

ROLL_DEG = 0.0
PITCH_DEG = 0.0
YAW_DEG = 0.0


# =============================================================================
# Localization configuration
# =============================================================================

YAML_CONFIG = "config_selectable_matchers.yaml"
DEVICE = "cuda"

RETRIEVAL_METHOD = "CAMP"
MATCHING_METHOD = "Roma"
DETERMINISTIC_SEED = 0

POSE_PRIORI = "unknown"
PRIOR_X = None
PRIOR_Y = None


# =============================================================================
# Performance configuration
# =============================================================================

# Match only the best unique satellite candidates.
MAX_MATCHING_CANDIDATES = 3

# Remove candidates that point to exactly the same satellite crop.
DEDUPLICATE_CANDIDATES = True

# Stop testing candidates when a sufficiently strong solution is found.
# Set to None to disable.
EARLY_STOP_INLIERS = 180

# Reduce PnP input correspondences if supported by localize_image.
# RoMa currently returns 3000 matches in your logs.
MAX_PNP_MATCHES = 1500

# Profile the first frame at Python-function level.
PROFILE_FIRST_FRAME = True
PROFILE_TOP_FUNCTIONS = 50


# =============================================================================
# Utility functions
# =============================================================================

def cuda_enabled() -> bool:
    return DEVICE.startswith("cuda") and torch.cuda.is_available()


def synchronize_cuda() -> None:
    """
    Wait until queued CUDA operations finish.

    CUDA execution is asynchronous, so synchronization is needed for
    accurate timing.
    """
    if cuda_enabled():
        torch.cuda.synchronize()


def validate_paths() -> None:
    required_paths = [
        FRAMES_DIR,
        SATELLITE_MAP,
        PGW_FILE,
        SRT_FILE,
        Path(YAML_CONFIG),
    ]

    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(f"Required path not found: {path}")


def list_frames(directory: Path) -> list[Path]:
    supported_extensions = {".jpg", ".jpeg", ".png"}

    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file()
        and path.suffix.lower() in supported_extensions
    )


def get_cuda_memory() -> dict[str, float]:
    if not cuda_enabled():
        return {
            "allocated_gb": 0.0,
            "reserved_gb": 0.0,
            "peak_allocated_gb": 0.0,
        }

    return {
        "allocated_gb": (
            torch.cuda.memory_allocated() / 1024**3
        ),
        "reserved_gb": (
            torch.cuda.memory_reserved() / 1024**3
        ),
        "peak_allocated_gb": (
            torch.cuda.max_memory_allocated() / 1024**3
        ),
    }


def initialize_context() -> tuple[Any, float]:
    synchronize_cuda()
    start = time.perf_counter()

    context = init_real_localization_context(
        satellite_map=str(SATELLITE_MAP),
        pgw_file=str(PGW_FILE),
        yaml_config=YAML_CONFIG,
        device=DEVICE,
        retrieval_method=RETRIEVAL_METHOD,
        matching_method=MATCHING_METHOD,
        pose_priori=POSE_PRIORI,
        deterministic_seed=DETERMINISTIC_SEED,
    )

    synchronize_cuda()
    elapsed = time.perf_counter() - start

    return context, elapsed


def filter_supported_arguments(
    function: Any,
    arguments: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    """
    Pass optional optimization arguments only when localize_image supports
    them. This avoids a TypeError with an older function implementation.
    """
    signature = inspect.signature(function)
    parameters = signature.parameters

    accepts_var_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )

    if accepts_var_kwargs:
        return arguments, []

    supported = {
        name: value
        for name, value in arguments.items()
        if name in parameters
    }

    unsupported = [
        name
        for name in arguments
        if name not in parameters
    ]

    return supported, unsupported


def build_localization_arguments(
    frame: Path,
    context: Any,
) -> tuple[dict[str, Any], list[str]]:
    stem = frame.stem

    standard_arguments = {
        "uav_image": str(frame),
        "satellite_map": str(SATELLITE_MAP),
        "pgw_file": str(PGW_FILE),
        "altitude": ALTITUDE,
        "k_matrix": K_MATRIX,
        "output_image": str(
            OUTPUT_DIR / f"{stem}_prediction.png"
        ),
        "output_overlay": str(
            OUTPUT_DIR / f"{stem}_overlay.png"
        ),
        "output_json": str(
            OUTPUT_DIR / f"{stem}.json"
        ),
        "debug_image": str(
            OUTPUT_DIR / f"{stem}_debug.png"
        ),
        "context": context,
        "yaml_config": YAML_CONFIG,
        "device": DEVICE,
        "retrieval_method": RETRIEVAL_METHOD,
        "matching_method": MATCHING_METHOD,
        "pose_priori": POSE_PRIORI,
        "deterministic_seed": DETERMINISTIC_SEED,
        "roll": ROLL_DEG,
        "pitch": PITCH_DEG,
        "yaw": YAW_DEG,
        "prior_x": PRIOR_X,
        "prior_y": PRIOR_Y,
        "work_dir": str(
            WORK_DIR / stem
        ),
    }

    optimization_arguments = {
        "deduplicate_candidates": DEDUPLICATE_CANDIDATES,
        "max_matching_candidates": MAX_MATCHING_CANDIDATES,
        "early_stop_inliers": EARLY_STOP_INLIERS,
        "max_pnp_matches": MAX_PNP_MATCHES,
        "return_timings": True,
    }

    supported_optimizations, unsupported = (
        filter_supported_arguments(
            localize_image,
            optimization_arguments,
        )
    )

    standard_arguments.update(supported_optimizations)

    return standard_arguments, unsupported


def profile_call(
    function: Any,
    output_path: Path,
) -> Any:
    profiler = cProfile.Profile()

    synchronize_cuda()
    profiler.enable()

    try:
        result = function()
    finally:
        synchronize_cuda()
        profiler.disable()

    stream = io.StringIO()

    stats = pstats.Stats(
        profiler,
        stream=stream,
    )

    stats.strip_dirs()
    stats.sort_stats("cumulative")
    stats.print_stats(PROFILE_TOP_FUNCTIONS)

    output_path.write_text(
        stream.getvalue(),
        encoding="utf-8",
    )

    return result


def save_timing_csv(rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "frame_index",
        "frame_name",
        "time_seconds",
        "running_average_seconds",
        "cuda_allocated_gb",
        "cuda_reserved_gb",
        "cuda_peak_allocated_gb",
        "success",
        "error",
    ]

    with TIMING_CSV.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)


def print_internal_timings(result: Any) -> None:
    """
    Print timing information if localize_image returns it.

    Supported result examples:
        {"timings": {...}}
        {"stage_timings": {...}}
    """
    if not isinstance(result, dict):
        return

    timings = (
        result.get("timings")
        or result.get("stage_timings")
    )

    if not isinstance(timings, dict):
        return

    print("Internal stage timings:")

    for stage_name, stage_time in timings.items():
        try:
            print(
                f"  {stage_name:<35}"
                f"{float(stage_time):>10.3f} seconds"
            )
        except (TypeError, ValueError):
            print(f"  {stage_name}: {stage_time}")


def build_telemetry_lookup(
    telemetry_records: list[TelemetryRecord],
) -> dict[int, TelemetryRecord]:
    return {
        int(record.srt_counter): record
        for record in telemetry_records
    }


def resolve_telemetry_record(
    frame: Path,
    telemetry_lookup: dict[int, TelemetryRecord],
    telemetry_records: list[TelemetryRecord],
) -> TelemetryRecord | None:
    extracted_frame_index = extract_frame_id(frame.stem)
    source_frame_index = extracted_frame_index * FRAME_EXTRACTION_STEP

    direct_match = telemetry_lookup.get(source_frame_index)
    if direct_match is not None:
        return direct_match

    one_based_match = telemetry_lookup.get(source_frame_index + 1)
    if one_based_match is not None:
        return one_based_match

    if 0 <= source_frame_index < len(telemetry_records):
        return telemetry_records[source_frame_index]

    return None


def compute_gps_error_m(
    predicted_latitude: float,
    predicted_longitude: float,
    gt_latitude: float,
    gt_longitude: float,
    geod: Geod,
) -> float:
    _, _, distance_m = geod.inv(
        float(predicted_longitude),
        float(predicted_latitude),
        float(gt_longitude),
        float(gt_latitude),
    )
    return float(distance_m)


def append_ground_truth_to_prediction_json(
    frame: Path,
    output_json: str,
    telemetry_lookup: dict[int, TelemetryRecord],
    telemetry_records: list[TelemetryRecord],
    geod: Geod,
) -> None:
    output_path = Path(output_json)
    if not output_path.exists():
        print(f"Warning: prediction JSON not found for {frame.name}: {output_path}")
        return

    telemetry_record = resolve_telemetry_record(
        frame=frame,
        telemetry_lookup=telemetry_lookup,
        telemetry_records=telemetry_records,
    )
    if telemetry_record is None:
        print(f"Warning: no matching SRT telemetry record found for {frame.name}")
        return

    with output_path.open("r", encoding="utf-8") as file:
        prediction = json.load(file)

    pred_latitude = prediction.get("pred_latitude")
    pred_longitude = prediction.get("pred_longitude")
    if pred_latitude is None or pred_longitude is None:
        print(f"Warning: predicted GPS fields are missing in {output_path.name}")
        return

    extracted_frame_index = extract_frame_id(frame.stem)
    source_frame_index = extracted_frame_index * FRAME_EXTRACTION_STEP
    gps_error_m = compute_gps_error_m(
        predicted_latitude=float(pred_latitude),
        predicted_longitude=float(pred_longitude),
        gt_latitude=telemetry_record.latitude,
        gt_longitude=telemetry_record.longitude,
        geod=geod,
    )

    prediction["gt_latitude"] = float(telemetry_record.latitude)
    prediction["gt_longitude"] = float(telemetry_record.longitude)
    prediction["gt_srt_frame_id"] = int(telemetry_record.srt_counter)
    prediction["source_video_frame_index"] = int(source_frame_index)
    prediction["source_video_time_seconds"] = float(
        source_frame_index / SOURCE_VIDEO_FPS
    )
    prediction["gps_error_m"] = gps_error_m

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(prediction, file, ensure_ascii=False, indent=2)

    print(
        f"GT GPS: latitude={telemetry_record.latitude:.8f}, "
        f"longitude={telemetry_record.longitude:.8f}, error={gps_error_m:.3f} m"
    )



# =============================================================================
# Main
# =============================================================================

def main() -> None:
    validate_paths()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    PROFILE_DIR.mkdir(parents=True, exist_ok=True)

    if DEVICE.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested, but PyTorch cannot access a CUDA GPU."
        )

    frames = list_frames(FRAMES_DIR)

    if not frames:
        raise FileNotFoundError(
            f"No JPG, JPEG, or PNG files found in {FRAMES_DIR}"
        )

    print("=" * 76)
    print("UAV LOCALIZATION")
    print("=" * 76)
    print(f"Frames:                 {len(frames)}")
    print(f"Device:                 {DEVICE}")
    print(f"Retrieval method:       {RETRIEVAL_METHOD}")
    print(f"Matching method:        {MATCHING_METHOD}")
    print(f"Maximum candidates:     {MAX_MATCHING_CANDIDATES}")
    print(f"Deduplicate candidates: {DEDUPLICATE_CANDIDATES}")
    print(f"Early-stop inliers:     {EARLY_STOP_INLIERS}")
    print(f"Maximum PnP matches:    {MAX_PNP_MATCHES}")

    if cuda_enabled():
        print(f"GPU:                    {torch.cuda.get_device_name(0)}")

    print("\nInitializing localization context...")

    context, context_time = initialize_context()
    telemetry_records = parse_srt(SRT_FILE)
    telemetry_lookup = build_telemetry_lookup(telemetry_records)
    geod = Geod(ellps="WGS84")

    print(
        f"Context initialized in {context_time:.3f} seconds."
    )

    timing_rows: list[dict[str, Any]] = []
    successful_times: list[float] = []

    unsupported_reported = False

    batch_start = time.perf_counter()

    with torch.inference_mode():
        for index, frame in enumerate(frames, start=1):
            print("\n" + "-" * 76)
            print(f"Frame {index}/{len(frames)}: {frame.name}")

            if cuda_enabled():
                torch.cuda.reset_peak_memory_stats()

            arguments, unsupported = build_localization_arguments(
                frame=frame,
                context=context,
            )

            if unsupported and not unsupported_reported:
                print(
                    "\nWarning: localize_image() does not yet support "
                    "these optimization arguments:"
                )

                for argument in unsupported:
                    print(f"  - {argument}")

                print(
                    "The batch script will still run, but the unsupported "
                    "optimizations require the internal patch shown below."
                )

                unsupported_reported = True

            def localization_call() -> Any:
                return localize_image(**arguments)

            success = True
            error_message = ""
            result = None

            synchronize_cuda()
            frame_start = time.perf_counter()

            try:
                if PROFILE_FIRST_FRAME and index == 1:
                    profile_path = (
                        PROFILE_DIR /
                        "first_frame_profile.txt"
                    )

                    result = profile_call(
                        function=localization_call,
                        output_path=profile_path,
                    )

                    print(f"Profile saved: {profile_path}")
                else:
                    result = localization_call()

            except Exception as error:
                success = False
                error_message = repr(error)
                print(f"Localization failed: {error}")

            synchronize_cuda()
            frame_time = time.perf_counter() - frame_start

            memory = get_cuda_memory()

            if success:
                successful_times.append(frame_time)
                print_internal_timings(result)
                append_ground_truth_to_prediction_json(
                    frame=frame,
                    output_json=arguments["output_json"],
                    telemetry_lookup=telemetry_lookup,
                    telemetry_records=telemetry_records,
                    geod=geod,
                )

            running_average = (
                sum(successful_times) / len(successful_times)
                if successful_times
                else 0.0
            )

            print(f"Frame time:      {frame_time:.3f} seconds")
            print(
                f"Running average: {running_average:.3f} seconds/frame"
            )

            if cuda_enabled():
                print(
                    "CUDA memory:     "
                    f"{memory['allocated_gb']:.3f} GB allocated, "
                    f"{memory['peak_allocated_gb']:.3f} GB peak"
                )

            timing_rows.append(
                {
                    "frame_index": index,
                    "frame_name": frame.name,
                    "time_seconds": round(frame_time, 6),
                    "running_average_seconds": round(
                        running_average,
                        6,
                    ),
                    "cuda_allocated_gb": round(
                        memory["allocated_gb"],
                        6,
                    ),
                    "cuda_reserved_gb": round(
                        memory["reserved_gb"],
                        6,
                    ),
                    "cuda_peak_allocated_gb": round(
                        memory["peak_allocated_gb"],
                        6,
                    ),
                    "success": success,
                    "error": error_message,
                }
            )

            save_timing_csv(timing_rows)

    synchronize_cuda()
    batch_time = time.perf_counter() - batch_start

    print("\n" + "=" * 76)
    print("SUMMARY")
    print("=" * 76)
    print(f"Context initialization: {context_time:.3f} seconds")
    print(f"Batch execution:        {batch_time:.3f} seconds")
    print(
        f"Successful frames:      "
        f"{len(successful_times)}/{len(frames)}"
    )

    if successful_times:
        print(
            f"Average frame time:     "
            f"{np.mean(successful_times):.3f} seconds"
        )
        print(
            f"Fastest frame time:     "
            f"{np.min(successful_times):.3f} seconds"
        )
        print(
            f"Slowest frame time:     "
            f"{np.max(successful_times):.3f} seconds"
        )

    print(f"\nTiming CSV: {TIMING_CSV}")
    print(f"Profile:    {PROFILE_DIR / 'first_frame_profile.txt'}")


if __name__ == "__main__":
    main()
