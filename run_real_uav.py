#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run single-image UAV localization on real DJI-style input.

Edit the variables below, then run:

    python run_real_uav.py
"""

import json

import numpy as np

from single_image_localization import localize_image


# ---------------------------------------------------------------------------
# User configuration
# ---------------------------------------------------------------------------

UAV_IMAGE = r"Data\my_own_data\frame.png"
SATELLITE_MAP = r"Data\my_own_data\satellite_map.png"
PGW_FILE = r"Data\my_own_data\satellite_map.pgw"
ALTITUDE = 500.0

# Values must correspond to the actual UAV image resolution you pass in.
K = np.array(
    [
        [1066.0, 0.0, 960.0],
        [0.0, 1066.0, 540.0],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float32,
)

OUTPUT_IMAGE = "real_uav_prediction.png"
OUTPUT_JSON = "real_uav_prediction.json"
OUTPUT_OVERLAY = "real_uav_overlay.png"
DEBUG_UAV_IMAGE = "debug_real_uav.png"

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


def main():
    result = localize_image(
        uav_image=UAV_IMAGE,
        satellite_map=SATELLITE_MAP,
        pgw_file=PGW_FILE,
        altitude=ALTITUDE,
        k_matrix=K,
        output_image=OUTPUT_IMAGE,
        output_overlay=OUTPUT_OVERLAY,
        output_json=OUTPUT_JSON,
        debug_image=DEBUG_UAV_IMAGE,
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
        work_dir="real_uav_work",
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
