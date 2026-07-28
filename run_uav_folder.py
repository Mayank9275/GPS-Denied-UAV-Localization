#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch-localize every UAV frame in Data/my_own_data/frames/.
Custom Data of my UAV and Satellite Map png
Run:

    python run_uav_folder.py
"""

from pathlib import Path
import time

import numpy as np

from single_image_localization import init_real_localization_context, localize_image


FRAMES_DIR = Path(r"Data\my_own_data\frames")
SATELLITE_MAP = r"Data\my_own_data\satellite_map.png"
PGW_FILE = r"Data\my_own_data\satellite_map.pgw"
OUTPUT_DIR = Path("outputs")
ALTITUDE = 500.0
K = np.array(
    [
        [1066.0, 0.0, 960.0],
        [0.0, 1066.0, 540.0],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float32,
)

YAML_CONFIG = "config_selectable_matchers.yaml"
DEVICE = "cuda"
RETRIEVAL_METHOD = "CAMP"
MATCHING_METHOD = "Roma"
DETERMINISTIC_SEED = 0

POSE_PRIORI = "unknown"
ROLL_DEG = 0.0
PITCH_DEG = 0.0
YAW_DEG = 0.0
PRIOR_X = None
PRIOR_Y = None


def list_frames(frames_dir):
    exts = {".jpg", ".jpeg", ".png"}
    return sorted([p for p in Path(frames_dir).iterdir() if p.is_file() and p.suffix.lower() in exts])


def main():
    start = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    frames = list_frames(FRAMES_DIR)
    if not frames:
        raise FileNotFoundError(f"No jpg/jpeg/png frames found in {FRAMES_DIR}")

    context = init_real_localization_context(
        satellite_map=SATELLITE_MAP,
        pgw_file=PGW_FILE,
        yaml_config=YAML_CONFIG,
        device=DEVICE,
        retrieval_method=RETRIEVAL_METHOD,
        matching_method=MATCHING_METHOD,
        pose_priori=POSE_PRIORI,
        deterministic_seed=DETERMINISTIC_SEED,
    )

    total = len(frames)
    for index, frame in enumerate(frames, start=1):
        print(f"Processing {index}/{total} : {frame.name}")
        stem = frame.stem
        localize_image(
            uav_image=str(frame),
            satellite_map=SATELLITE_MAP,
            pgw_file=PGW_FILE,
            altitude=ALTITUDE,
            k_matrix=K,
            output_image=str(OUTPUT_DIR / f"{stem}_prediction.png"),
            output_overlay=str(OUTPUT_DIR / f"{stem}_overlay.png"),
            output_json=str(OUTPUT_DIR / f"{stem}.json"),
            debug_image=str(OUTPUT_DIR / f"{stem}_debug.png"),
            context=context,
            yaml_config=YAML_CONFIG,
            device=DEVICE,
            retrieval_method=RETRIEVAL_METHOD,
            matching_method=MATCHING_METHOD,
            pose_priori=POSE_PRIORI,
            deterministic_seed=DETERMINISTIC_SEED,
            roll=ROLL_DEG,
            pitch=PITCH_DEG,
            yaw=YAW_DEG,
            prior_x=PRIOR_X,
            prior_y=PRIOR_Y,
            work_dir=str(OUTPUT_DIR / "_work" / stem),
        )

    elapsed = time.time() - start
    print("Batch localization completed successfully.")
    print(f"Processed {total} images.")
    print("Output directory:")
    print(f"{OUTPUT_DIR}/")
    print(f"Total execution time: {elapsed:.2f} seconds")


if __name__ == "__main__":
    main()
