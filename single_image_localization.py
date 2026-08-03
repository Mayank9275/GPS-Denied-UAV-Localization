#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Single UAV image -> satellite image localization wrapper.

This script reuses the repository's existing retrieve -> match -> PnP pipeline.
For AnyVisLoc data, the UAV image is loaded directly from metadata_npz["image"].
The optional --uav argument is only an override for future custom datasets.
"""

import argparse
import json
import time
import warnings
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
from pyproj import Transformer
import torch
import yaml

from avl_data import _as_string, _load_npz, _read_image_from_relpath
from avl_utils import (
    build_anyvisloc_reference_view,
    dumpRotateImage,
    ensure_dir,
    matching_init,
    normalize_reference_mode,
    retrieval_all_anyvisloc,
    retrieval_init,
    select_pose_anyvisloc,
    set_deterministic_inference,
    Match2Pos_all_anyvisloc,
    tensor_rgb_to_bgr_uint8,
)

warnings.filterwarnings("ignore")

_WORLD_TO_GPS_TRANSFORMER = Transformer.from_crs(
    "EPSG:3857",
    "EPSG:4326",
    always_xy=True,
)


def _read_bgr(path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return img


def _rgb_array_to_tensor(image_rgb):
    rgb = np.asarray(image_rgb)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"Expected RGB image with shape [H,W,3], got {rgb.shape}")
    rgb = rgb.astype(np.float32) / 255.0
    return torch.from_numpy(rgb).permute(2, 0, 1).contiguous()


def _world_3857_to_gps(world_x, world_y):
    try:
        lon, lat = _WORLD_TO_GPS_TRANSFORMER.transform(float(world_x), float(world_y))
    except Exception as exc:
        print(
            f"[warning] Failed to convert predicted world coordinates from EPSG:3857 to EPSG:4326: {exc}"
        )
        return None
    return lon, lat


def _infer_metadata_npz(uav_path, dataset_root, scene, sample_id):
    candidates = []
    dataset_root = Path(dataset_root)

    if sample_id:
        if scene:
            candidates.append(dataset_root / scene / f"{sample_id}.npz")
        candidates.extend(dataset_root.rglob(f"{sample_id}.npz"))

    if uav_path:
        uav_path = Path(uav_path)
        candidates.append(uav_path.with_suffix(".npz"))
        candidates.extend(dataset_root.rglob(f"{uav_path.stem}.npz"))

    seen = set()
    for candidate in candidates:
        candidate = Path(candidate)
        key = str(candidate.resolve()) if candidate.exists() else str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if candidate.exists():
            return candidate
    return None


def _first_existing_reference_json(scene_dir):
    refs = sorted(Path(scene_dir).glob("L??_reference.json"))
    if len(refs) == 1:
        return refs[0]
    return None


def _optional_float_array(s, key, shape, default):
    if key in s:
        return np.asarray(s[key], dtype=np.float32).reshape(shape)
    return np.asarray(default, dtype=np.float32).reshape(shape)


def _metadata_float(s, key, index, default=None):
    if key not in s:
        return default
    arr = np.asarray(s[key], dtype=np.float32).reshape(-1)
    if index >= arr.size:
        return default
    return float(arr[index])


def _make_sample_from_metadata(metadata_npz, uav_path, args):
    metadata_npz = Path(metadata_npz)
    s = _load_npz(metadata_npz)
    required = ["K"] if uav_path else ["image", "K"]
    missing = [k for k in required if k not in s]
    if missing:
        raise KeyError(f"Metadata NPZ is missing required fields: {missing}")

    if uav_path:
        image_bgr = _read_bgr(uav_path)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    else:
        image_rgb = s["image"]
        if image_rgb.size == 0:
            relpath = _as_string(s.get("image_relpath", ""))
            image_rgb = _read_image_from_relpath(metadata_npz, relpath)

    image_tensor = _rgb_array_to_tensor(image_rgb)

    image_size_np = (
        s["image_size"].astype(np.int32)
        if "image_size" in s and not uav_path
        else np.asarray(image_rgb.shape[:2], dtype=np.int32)
    )
    image_size = torch.from_numpy(image_size_np)

    scene_id = int(np.asarray(s["scene_id"]).reshape(-1)[0]) if "scene_id" in s else int(args.scene_id)
    scene_name = Path(metadata_npz).parent.name
    sample_id = _as_string(s["sample_id"]) if "sample_id" in s else metadata_npz.stem
    if args.reference_json:
        ref_json = Path(args.reference_json)
    elif "scene_id" in s:
        ref_json = Path(metadata_npz).parent / f"L{scene_id:02d}_reference.json"
    else:
        ref_json = _first_existing_reference_json(Path(metadata_npz).parent)
        if ref_json is None:
            raise FileNotFoundError(
                "Minimal NPZ has no scene_id. Pass --reference_json path/to/Lxx_reference.json."
            )

    altitude = _metadata_float(s, "xyz", 2, args.altitude)
    if altitude is None:
        raise ValueError("Minimal NPZ has no xyz[2]. Pass --altitude for custom UAV inference.")

    prior_x = _metadata_float(s, "xyz", 0, args.prior_x)
    prior_y = _metadata_float(s, "xyz", 1, args.prior_y)
    roll = _metadata_float(s, "euler_deg", 0, args.roll)
    pitch = _metadata_float(s, "euler_deg", 1, args.pitch)
    yaw = _metadata_float(s, "euler_deg", 2, args.yaw)

    return {
        "image": image_tensor,
        "K": torch.from_numpy(np.asarray(s["K"], dtype=np.float32)),
        "dist": torch.from_numpy(_optional_float_array(s, "dist", (-1,), np.zeros(5, dtype=np.float32))),
        "uav_downsample": float(np.asarray(s.get("uav_downsample", 1.0)).reshape(-1)[0]),
        "original_size": image_size.clone(),
        "image_size": image_size,
        "xyz": torch.tensor([prior_x, prior_y, altitude], dtype=torch.float32),
        "euler_deg": torch.tensor([roll, pitch, yaw], dtype=torch.float32),
        "sample_id": sample_id,
        "scene_id": scene_id,
        "scene_name": scene_name,
        "npz_path": str(metadata_npz),
        "reference": {
            "ref_json_path": str(ref_json),
            "ref_path": str(ref_json),
            "scene_dir": str(Path(metadata_npz).parent),
        },
    }


def _load_reference_with_user_satellite(sample, satellite_path):
    view = build_anyvisloc_reference_view(sample, "satellite")
    user_satellite = _read_bgr(satellite_path)
    expected_h, expected_w = view["map"].shape[:2]
    actual_h, actual_w = user_satellite.shape[:2]
    if (actual_h, actual_w) != (expected_h, expected_w):
        print(
            "[warning] Satellite image size differs from reference metadata: "
            f"image={actual_w}x{actual_h}, metadata={expected_w}x{expected_h}. "
            "Pixel drawing will use the metadata resolution/origin."
        )
    view["map"] = user_satellite
    view["satellite_image_path"] = str(satellite_path)
    return view


def _sample_to_inference_pose(sample):
    image_size = sample["image_size"].detach().cpu().numpy().astype(np.int32).reshape(-1)
    h, w = int(image_size[0]), int(image_size[1])
    xyz = sample["xyz"].detach().cpu().numpy().astype(np.float32).reshape(3)
    euler = sample["euler_deg"].detach().cpu().numpy().astype(np.float32).reshape(3)
    K = sample["K"].detach().cpu().numpy().astype(np.float32)
    return {
        "x": float(xyz[0]),
        "y": float(xyz[1]),
        "z": float(xyz[2]),
        "rel_alt": float(xyz[2]),
        "roll": float(euler[0]),
        "pitch": float(euler[1]),
        "yaw": float(euler[2]),
        "width": int(w),
        "height": int(h),
        "K": K,
        "sample_id": str(sample.get("sample_id", "")),
        "scene_name": str(sample.get("scene_name", "")),
        "npz_path": str(sample.get("npz_path", "")),
    }


def _local_xy_to_pixel(x, y, map_resolution, map_origin_local):
    res = np.asarray(map_resolution, dtype=np.float32).reshape(-1)
    origin = np.asarray(map_origin_local, dtype=np.float32).reshape(-1)
    if res.size < 2 or origin.size < 2:
        raise ValueError("map_resolution and map_origin_local must each have at least 2 values")
    col = (float(x) - float(origin[0])) / max(float(res[0]), 1e-12)
    row = (float(y) - float(origin[1])) / max(float(res[1]), 1e-12)
    return col, row


def valid_pose(x, y, z, map_width, map_height):
    pose = np.asarray([x, y, z], dtype=np.float64)

    if not np.all(np.isfinite(pose)):
        return False

    if abs(x) > 1e6 or abs(y) > 1e6 or abs(z) > 1e6:
        return False

    if x < 0 or y < 0:
        return False

    if x >= map_width or y >= map_height:
        return False

    return True


def _filter_retrieved_candidates(
    row_starts,
    col_starts,
    patch_h,
    patch_w,
    *,
    deduplicate_candidates=True,
    max_matching_candidates=3,
):
    retrieved_candidates = []
    for row, col in zip(row_starts, col_starts):
        retrieved_candidates.append(
            {
                "row": int(row),
                "col": int(col),
                "height": int(patch_h),
                "width": int(patch_w),
            }
        )

    if deduplicate_candidates:
        unique_candidates = []
        seen = set()

        for candidate in retrieved_candidates:
            key = (
                int(candidate["row"]),
                int(candidate["col"]),
                int(candidate["height"]),
                int(candidate["width"]),
            )

            if key not in seen:
                seen.add(key)
                unique_candidates.append(candidate)

        retrieved_candidates = unique_candidates

    if max_matching_candidates is not None:
        retrieved_candidates = retrieved_candidates[: int(max_matching_candidates)]

    filtered_row_starts = [int(candidate["row"]) for candidate in retrieved_candidates]
    filtered_col_starts = [int(candidate["col"]) for candidate in retrieved_candidates]
    return retrieved_candidates, filtered_row_starts, filtered_col_starts


def _draw_prediction(satellite_bgr, pred_x, pred_y, view, output_path, draw_text=True):
    col, row = _local_xy_to_pixel(pred_x, pred_y, view["map_resolution"], view["map_origin_local"])
    out = satellite_bgr.copy()
    h, w = out.shape[:2]
    px = int(round(col))
    py = int(round(row))

    if not (0 <= px < w and 0 <= py < h):
        raise ValueError(
            f"Predicted pixel is outside satellite image: col={col:.2f}, row={row:.2f}, image={w}x{h}"
        )

    radius = max(10, int(round(min(w, h) * 0.012)))
    thickness = max(2, int(round(radius * 0.22)))
    cv2.circle(out, (px, py), radius, (0, 0, 255), thickness, cv2.LINE_AA)
    cv2.drawMarker(
        out,
        (px, py),
        (0, 0, 255),
        markerType=cv2.MARKER_CROSS,
        markerSize=radius * 2,
        thickness=thickness,
        line_type=cv2.LINE_AA,
    )

    if draw_text:
        label = f"Pred UAV ({pred_x:.2f}, {pred_y:.2f})"
        org = (min(px + radius + 8, max(w - 10, 0)), max(py - radius - 8, 20))
        if org[0] > w - 220:
            org = (max(px - radius - 220, 5), org[1])
        cv2.putText(out, label, org, cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, label, org, cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    if not cv2.imwrite(str(output_path), out):
        raise RuntimeError(f"Failed to write output image: {output_path}")
    return col, row


def read_pgw(path):
    values = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if text:
                values.append(float(text))
    if len(values) != 6:
        raise ValueError(f"Expected 6 numeric lines in PGW file, got {len(values)}: {path}")
    return {
        "a": values[0],
        "d": values[1],
        "b": values[2],
        "e": values[3],
        "c": values[4],
        "f": values[5],
    }


def pixel_to_world(col, row, pgw):
    world_x = pgw["a"] * float(col) + pgw["b"] * float(row) + pgw["c"]
    world_y = pgw["d"] * float(col) + pgw["e"] * float(row) + pgw["f"]
    return float(world_x), float(world_y)


def _draw_real_prediction(satellite_bgr, col, row, output_path, label=None):
    out = satellite_bgr.copy()
    h, w = out.shape[:2]
    px = int(round(float(col)))
    py = int(round(float(row)))
    within_bounds = 0 <= px < w and 0 <= py < h
    if not within_bounds:
        warning = f"Predicted pixel outside image: ({col:.1f}, {row:.1f})"
        cv2.putText(out, warning, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, warning, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 1, cv2.LINE_AA)
        ensure_dir(Path(output_path).parent)
        if not cv2.imwrite(str(output_path), out):
            raise RuntimeError(f"Failed to write output image: {output_path}")
        return out, False

    radius = max(10, int(round(min(w, h) * 0.012)))
    thickness = max(2, int(round(radius * 0.22)))
    cv2.circle(out, (px, py), radius, (0, 0, 255), thickness, cv2.LINE_AA)
    cv2.drawMarker(
        out,
        (px, py),
        (0, 0, 255),
        markerType=cv2.MARKER_CROSS,
        markerSize=radius * 2,
        thickness=thickness,
        line_type=cv2.LINE_AA,
    )

    if label:
        org = (min(px + radius + 8, max(w - 10, 0)), max(py - radius - 8, 20))
        if org[0] > w - 260:
            org = (max(px - radius - 260, 5), org[1])
        cv2.putText(out, label, org, cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, label, org, cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    ensure_dir(Path(output_path).parent)
    if not cv2.imwrite(str(output_path), out):
        raise RuntimeError(f"Failed to write output image: {output_path}")
    return out, True


def _compute_prediction_confidence(best_inliers, pnp_input_count):
    best_inliers = int(best_inliers)
    pnp_input_count = int(pnp_input_count)
    if best_inliers <= 0 or pnp_input_count <= 0:
        return 0.0, 0.0

    inlier_ratio = float(best_inliers) / float(max(pnp_input_count, 1))
    confidence_pct = 100.0 * (
        0.7 * inlier_ratio + 0.3 * min(float(best_inliers) / 80.0, 1.0)
    )
    confidence_pct = float(np.clip(confidence_pct, 0.0, 100.0))
    return confidence_pct, inlier_ratio


def build_real_reference_view(satellite_bgr, pgw):
    h, w = satellite_bgr.shape[:2]
    if abs(float(pgw["b"])) > 1e-9 or abs(float(pgw["d"])) > 1e-9:
        print("[warning] Rotated PGW terms are non-zero. Existing localization assumes an axis-aligned map.")

    map_resolution = np.array([abs(float(pgw["a"])), abs(float(pgw["e"]))], dtype=np.float32)
    map_origin_local = np.array([0.0, 0.0], dtype=np.float32)
    flat_dsm = np.zeros((h, w), dtype=np.float32)

    return {
        "reference_mode": "satellite",
        "map": satellite_bgr,
        "dsm": flat_dsm,
        "map_resolution": map_resolution,
        "dsm_resolution": map_resolution.copy(),
        "map_origin_local": map_origin_local,
        "dsm_origin_local": map_origin_local.copy(),
        "pgw": pgw,
    }


def build_real_opt(
    *,
    device="cuda",
    pose_priori="unknown",
    sample_id="real_uav",
    match_keypoints=3000,
    deterministic_seed=0,
):
    repo_root = Path(__file__).resolve().parent
    return SimpleNamespace(
        device=device,
        reference_mode="satellite",
        visualize=False,
        pose_priori=pose_priori,
        strategy="Topn_opt",
        PnP_method="P3P",
        resize_ratio=0.4,
        selectable_code_dir=str(repo_root / "Matching_Models" / "Sparse_matchers"),
        selectable_module="selectable_sparse_matcher",
        match_keypoints=match_keypoints,
        gim_lg_ckpt=str(repo_root / "Matching_Models" / "Sparse_matchers" / "weights" / "gim_lightglue_100h.ckpt"),
        minima_lg_ckpt=str(repo_root / "Matching_Models" / "Sparse_matchers" / "weights" / "minima_lightglue.pth"),
        patch_scale=1.0,
        min_patch_size_m=0.0,
        max_patch_size_m=0.0,
        draw_pnp_inlier_ratio=0.0,
        draw_pnp_inlier_seed=0,
        ref_feature_cache_dir=None,
        disable_ref_feature_cache=True,
        pnp_reproj_error=8.0,
        pnp_iterations=2000,
        pnp_confidence=0.999,
        deterministic_seed=int(deterministic_seed),
        debug_trace=True,
        _current_scene_name="real_uav",
        _current_sample_id=sample_id,
    )


def build_real_inference_pose(
    uav_bgr,
    map_resolution,
    map_shape,
    *,
    k_matrix,
    altitude,
    sample_id,
    roll=0.0,
    pitch=0.0,
    yaw=0.0,
    prior_x=None,
    prior_y=None,
):
    h, w = uav_bgr.shape[:2]
    map_h, map_w = map_shape[:2]
    prior_x = float(prior_x) if prior_x is not None else float(map_w * map_resolution[0] * 0.5)
    prior_y = float(prior_y) if prior_y is not None else float(map_h * map_resolution[1] * 0.5)
    return {
        "x": prior_x,
        "y": prior_y,
        "z": float(altitude),
        "rel_alt": float(altitude),
        "roll": float(roll),
        "pitch": float(pitch),
        "yaw": float(yaw),
        "width": int(w),
        "height": int(h),
        "K": np.asarray(k_matrix, dtype=np.float32),
        "sample_id": str(sample_id),
        "scene_name": "real_uav",
        "npz_path": "",
    }


def init_real_localization_context(
    *,
    satellite_map,
    pgw_file,
    yaml_config="config_selectable_matchers.yaml",
    device="cuda",
    retrieval_method="CAMP",
    matching_method="Roma",
    pose_priori="unknown",
    deterministic_seed=0,
):
    set_deterministic_inference(deterministic_seed)
    satellite_bgr = _read_bgr(satellite_map)
    pgw = read_pgw(pgw_file)
    with Path(yaml_config).open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    config["DEVICE"] = device

    opt = build_real_opt(device=device, pose_priori=pose_priori, deterministic_seed=deterministic_seed)
    view = build_real_reference_view(satellite_bgr, pgw)

    method_dict = {"retrieval_method": retrieval_method}
    method_dict = retrieval_init(method_dict, config)
    method_dict["matching_method"] = matching_method
    method_dict = matching_init(method_dict, opt, config)
    method_dict["deterministic_seed"] = int(deterministic_seed)

    return {
        "satellite_bgr": satellite_bgr,
        "pgw": pgw,
        "config": config,
        "opt": opt,
        "view": view,
        "method_dict": method_dict,
    }


def localize_image(
    *,
    uav_image,
    satellite_map,
    pgw_file,
    altitude,
    k_matrix,
    output_image,
    output_overlay,
    output_json,
    debug_image,
    context=None,
    yaml_config="config_selectable_matchers.yaml",
    device="cuda",
    retrieval_method="CAMP",
    matching_method="Roma",
    pose_priori="unknown",
    deterministic_seed=0,
    roll=0.0,
    pitch=0.0,
    yaw=0.0,
    prior_x=None,
    prior_y=None,
    work_dir="real_uav_work",
    deduplicate_candidates=True,
    max_matching_candidates=3,
    early_stop_inliers=180,
):
    set_deterministic_inference(deterministic_seed)
    t0 = time.time()
    uav_bgr = _read_bgr(uav_image)
    ensure_dir(Path(debug_image).parent)
    cv2.imwrite(str(debug_image), uav_bgr)

    if context is None:
        context = init_real_localization_context(
            satellite_map=satellite_map,
            pgw_file=pgw_file,
            yaml_config=yaml_config,
            device=device,
            retrieval_method=retrieval_method,
            matching_method=matching_method,
            pose_priori=pose_priori,
            deterministic_seed=deterministic_seed,
        )

    satellite_bgr = context["satellite_bgr"]
    pgw = context["pgw"]
    config = context["config"]
    opt = context["opt"]
    view = context["view"]
    method_dict = context["method_dict"]

    sample_id = Path(uav_image).stem
    opt._current_sample_id = sample_id
    inference_pose = build_real_inference_pose(
        uav_bgr,
        view["map_resolution"],
        satellite_bgr.shape,
        k_matrix=k_matrix,
        altitude=altitude,
        sample_id=sample_id,
        roll=roll,
        pitch=pitch,
        yaw=yaw,
        prior_x=prior_x,
        prior_y=prior_y,
    )

    if opt.pose_priori == "yp":
        ref_map, mat_rotation = dumpRotateImage(view["map"], inference_pose["yaw"])
    else:
        ref_map = view["map"]
        mat_rotation = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)

    ensure_dir(work_dir)
    (
        _ir_order,
        row_starts,
        col_starts,
        _pde_list,
        patch_h,
        patch_w,
        fine_scale,
        retrieval_time,
    ) = retrieval_all_anyvisloc(
        ref_map,
        uav_bgr,
        inference_pose,
        view["map_resolution"],
        view["map_origin_local"],
        mat_rotation,
        str(work_dir),
        opt,
        config,
        method_dict,
    )

    retrieved_candidates, row_starts, col_starts = _filter_retrieved_candidates(
        row_starts,
        col_starts,
        patch_h,
        patch_w,
        deduplicate_candidates=deduplicate_candidates,
        max_matching_candidates=max_matching_candidates,
    )
    method_dict["retrieved_candidates"] = retrieved_candidates

    XYZ_list, inliers_list, pnp_input_count_list, match_time, pnp_time = Match2Pos_all_anyvisloc(
        opt,
        config,
        uav_bgr,
        fine_scale,
        np.asarray(k_matrix, dtype=np.float32),
        ref_map,
        view["dsm"],
        row_starts,
        col_starts,
        patch_h,
        patch_w,
        str(work_dir),
        method_dict,
        mat_rotation,
        view["map_resolution"],
        view["map_origin_local"],
        view["dsm_resolution"],
        view["dsm_origin_local"],
        dist=np.zeros(5, dtype=np.float32),
        truePos=None,
        early_stop_inliers=early_stop_inliers,
    )

    pred_loc, pnp_success, best_index = select_pose_anyvisloc(XYZ_list, inliers_list)
    if not pnp_success:
        raise RuntimeError("PnP failed; no valid predicted UAV position was produced.")
    if bool(getattr(opt, "debug_trace", False)):
        selected_row = None if best_index is None else int(row_starts[best_index])
        selected_col = None if best_index is None else int(col_starts[best_index])
        print(
            f"[trace] selected patch: index={best_index}, row={selected_row}, col={selected_col}, "
            f"best_inliers={0 if best_index is None else int(inliers_list[best_index])}"
        )

    pred_x = float(pred_loc["x"])
    pred_y = float(pred_loc["y"])
    pred_z = None if pred_loc.get("z") is None else float(pred_loc["z"])
    best_inliers = 0 if best_index is None else int(inliers_list[best_index])
    best_pnp_input_count = 0 if best_index is None else int(pnp_input_count_list[best_index])
    confidence_pct, inlier_ratio = _compute_prediction_confidence(best_inliers, best_pnp_input_count)
    satellite_h, satellite_w = satellite_bgr.shape[:2]
    if bool(getattr(opt, "debug_trace", False)):
        print(
            f"[trace] map before pixel conversion: map_resolution={view['map_resolution'].tolist()}, "
            f"map_origin_local={view['map_origin_local'].tolist()}"
        )
    pred_col, pred_row = _local_xy_to_pixel(pred_x, pred_y, view["map_resolution"], view["map_origin_local"])
    if bool(getattr(opt, "debug_trace", False)):
        inside = 0 <= pred_col < satellite_w and 0 <= pred_row < satellite_h
        print(
            f"[trace] final prediction: local_x={pred_x}, local_y={pred_y}, local_z={pred_z}, "
            f"pixel_col={pred_col}, pixel_row={pred_row}, "
            f"satellite_size=({satellite_w}, {satellite_h}), inside={inside}"
        )
    world_x, world_y = pixel_to_world(pred_col, pred_row, pgw)

    label = f"UAV ({pred_col:.1f}, {pred_row:.1f}) | Conf {confidence_pct:.1f}%"
    marked, prediction_within_map = _draw_real_prediction(
        satellite_bgr, pred_col, pred_row, output_image, label=label
    )
    overlay = cv2.addWeighted(satellite_bgr, 0.70, marked, 0.30, 0.0)
    ensure_dir(Path(output_overlay).parent)
    cv2.imwrite(str(output_overlay), overlay)

    result = {
        "uav_image": str(uav_image),
        "satellite_map": str(satellite_map),
        "pgw_file": str(pgw_file),
        "output_image": str(output_image),
        "output_overlay": str(output_overlay),
        "debug_uav_image": str(debug_image),
        "altitude_m": float(altitude),
        "K": np.asarray(k_matrix, dtype=np.float32).tolist(),
        "pred_x_local_m": pred_x,
        "pred_y_local_m": pred_y,
        "pred_z_local_m": pred_z,
        "pred_col": float(pred_col),
        "pred_row": float(pred_row),
        "pred_world_x": float(world_x),
        "pred_world_y": float(world_y),
        "prediction_within_map": bool(prediction_within_map),
        "world_file_affine": pgw,
        "best_index": None if best_index is None else int(best_index),
        "best_inliers": best_inliers,
        "best_pnp_input_count": best_pnp_input_count,
        "inlier_ratio": inlier_ratio,
        "confidence_pct": confidence_pct,
        "retrieved_candidates": retrieved_candidates,
        "retrieval_time_s": float(retrieval_time),
        "match_time_s": [float(x) for x in match_time],
        "pnp_time_s": [float(x) for x in pnp_time],
        "total_time_s": float(time.time() - t0),
    }
    gps_coords = _world_3857_to_gps(world_x, world_y)
    if gps_coords is not None:
        lon, lat = gps_coords
        result["pred_latitude"] = float(lat)
        result["pred_longitude"] = float(lon)
        print(f"Predicted GPS: latitude={lat:.8f}, longitude={lon:.8f}")
    ensure_dir(Path(output_json).parent)
    with Path(output_json).open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    return result


def _build_opt(args):
    repo_root = Path(__file__).resolve().parent
    return SimpleNamespace(
        device=args.device,
        reference_mode="satellite",
        visualize=False,
        pose_priori=args.pose_priori,
        strategy=args.strategy,
        PnP_method=args.PnP_method,
        resize_ratio=args.resize_ratio,
        selectable_code_dir=str(repo_root / "Matching_Models" / "Sparse_matchers"),
        selectable_module="selectable_sparse_matcher",
        match_keypoints=args.match_keypoints,
        gim_lg_ckpt=str(repo_root / "Matching_Models" / "Sparse_matchers" / "weights" / "gim_lightglue_100h.ckpt"),
        minima_lg_ckpt=str(repo_root / "Matching_Models" / "Sparse_matchers" / "weights" / "minima_lightglue.pth"),
        patch_scale=args.patch_scale,
        min_patch_size_m=args.min_patch_size_m,
        max_patch_size_m=args.max_patch_size_m,
        draw_pnp_inlier_ratio=0.0,
        draw_pnp_inlier_seed=0,
        ref_feature_cache_dir=None,
        disable_ref_feature_cache=True,
        pnp_reproj_error=args.pnp_reproj_error,
        pnp_iterations=args.pnp_iterations,
        pnp_confidence=args.pnp_confidence,
        deterministic_seed=int(args.deterministic_seed),
        _current_scene_name="single_image",
        _current_sample_id="single_image",
    )


def get_parse():
    parser = argparse.ArgumentParser(description="Localize one UAV image on one satellite image.")
    parser.add_argument("--uav", default=None, type=str, help="Optional UAV jpg/png override. By default the UAV image is read from --metadata_npz.")
    parser.add_argument("--satellite", required=True, type=str, help="Path to satellite map jpg/png image.")
    parser.add_argument("--output", default="output_prediction.png", type=str, help="Output marked satellite image.")
    parser.add_argument("--metadata_npz", default=None, type=str, help="AnyVisLoc sample NPZ supplying K/dist/pose metadata.")
    parser.add_argument("--reference_json", default=None, type=str, help="Lxx_reference.json supplying satellite map resolution/origin and DSM.")
    parser.add_argument("--dataset_root", default="./Data/AnyVisLoc", type=str, help="Used to infer metadata_npz if omitted.")
    parser.add_argument("--scene", default=None, type=str, help="Optional scene name used when inferring metadata_npz.")
    parser.add_argument("--sample_id", default=None, type=str, help="Optional sample id used when inferring metadata_npz.")
    parser.add_argument("--scene_id", default=0, type=int, help="Fallback scene id when metadata_npz has no scene_id.")
    parser.add_argument("--altitude", default=None, type=float, help="Required fallback altitude when metadata_npz has no xyz[2].")
    parser.add_argument("--prior_x", default=0.0, type=float, help="Optional retrieval prior x when metadata_npz has no xyz[0].")
    parser.add_argument("--prior_y", default=0.0, type=float, help="Optional retrieval prior y when metadata_npz has no xyz[1].")
    parser.add_argument("--roll", default=0.0, type=float, help="Fallback roll when metadata_npz has no euler_deg.")
    parser.add_argument("--pitch", default=0.0, type=float, help="Fallback pitch when metadata_npz has no euler_deg.")
    parser.add_argument("--yaw", default=0.0, type=float, help="Fallback yaw when metadata_npz has no euler_deg.")
    parser.add_argument("--yaml", default="config_selectable_matchers.yaml", type=str, help="Model config yaml.")
    parser.add_argument("--retrieval_method", default="CAMP", type=str)
    parser.add_argument("--matching_method", default="Roma", type=str)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--pose_priori", default="yp", choices=["yp", "p", "unknown"])
    parser.add_argument("--strategy", default="Topn_opt", type=str)
    parser.add_argument("--PnP_method", default="P3P", type=str)
    parser.add_argument("--resize_ratio", default=0.4, type=float)
    parser.add_argument("--match_keypoints", default=3000, type=int)
    parser.add_argument("--patch_scale", default=1.0, type=float)
    parser.add_argument("--min_patch_size_m", default=0.0, type=float)
    parser.add_argument("--max_patch_size_m", default=0.0, type=float)
    parser.add_argument("--pnp_reproj_error", default=8.0, type=float)
    parser.add_argument("--pnp_iterations", default=2000, type=int)
    parser.add_argument("--pnp_confidence", default=0.999, type=float)
    parser.add_argument("--deterministic_seed", default=0, type=int)
    parser.add_argument("--no_text", action="store_true", help="Do not draw coordinate text on output image.")
    return parser.parse_args()


def main():
    args = get_parse()
    set_deterministic_inference(args.deterministic_seed)
    uav_path = Path(args.uav) if args.uav else None
    satellite_path = Path(args.satellite)
    if uav_path is not None and not uav_path.exists():
        raise FileNotFoundError(f"UAV image not found: {uav_path}")
    if not satellite_path.exists():
        raise FileNotFoundError(f"Satellite image not found: {satellite_path}")

    metadata_npz = Path(args.metadata_npz) if args.metadata_npz else _infer_metadata_npz(
        uav_path,
        args.dataset_root,
        args.scene,
        args.sample_id,
    )
    if metadata_npz is None:
        raise FileNotFoundError(
            "Cannot infer metadata NPZ. Pass --metadata_npz path/to/Lxx_yyyy.npz, "
            "or pass --sample_id with optional --scene so it can be found under --dataset_root."
        )

    with open(args.yaml, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    config["DEVICE"] = args.device

    opt = _build_opt(args)
    opt.reference_mode = normalize_reference_mode("satellite")

    sample = _make_sample_from_metadata(metadata_npz, uav_path, args)
    opt._current_scene_name = sample["scene_name"]
    opt._current_sample_id = sample["sample_id"]

    inference_pose = _sample_to_inference_pose(sample)
    K = sample["K"].detach().cpu().numpy().astype(np.float32)
    dist = sample["dist"].detach().cpu().numpy().astype(np.float32)
    uav_bgr = tensor_rgb_to_bgr_uint8(sample["image"])
    cv2.imwrite("debug_uav.png", uav_bgr)

    view = _load_reference_with_user_satellite(sample, satellite_path)
    ref_map0 = view["map"]
    dsm_map0 = view["dsm"]

    if opt.pose_priori == "yp":
        ref_map, mat_rotation = dumpRotateImage(ref_map0, inference_pose["yaw"])
    else:
        ref_map = ref_map0
        mat_rotation = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)

    method_dict = {"retrieval_method": args.retrieval_method}
    method_dict = retrieval_init(method_dict, config)
    method_dict["matching_method"] = args.matching_method
    method_dict = matching_init(method_dict, opt, config)
    method_dict["deterministic_seed"] = int(args.deterministic_seed)

    work_dir = Path(args.output).resolve().parent / "_single_image_localization_work"
    ensure_dir(work_dir)

    t0 = time.time()
    (
        _ir_order,
        row_starts,
        col_starts,
        _pde_list,
        patch_h,
        patch_w,
        fine_scale,
        retrieval_time,
    ) = retrieval_all_anyvisloc(
        ref_map,
        uav_bgr,
        inference_pose,
        view["map_resolution"],
        view["map_origin_local"],
        mat_rotation,
        str(work_dir),
        opt,
        config,
        method_dict,
    )

    XYZ_list, inliers_list, pnp_input_count_list, match_time, pnp_time = Match2Pos_all_anyvisloc(
        opt,
        config,
        uav_bgr,
        fine_scale,
        K,
        ref_map,
        dsm_map0,
        row_starts,
        col_starts,
        patch_h,
        patch_w,
        str(work_dir),
        method_dict,
        mat_rotation,
        view["map_resolution"],
        view["map_origin_local"],
        view["dsm_resolution"],
        view["dsm_origin_local"],
        dist=dist,
        truePos=None,
    )

    pred_loc, pnp_success, best_index = select_pose_anyvisloc(
        XYZ_list,
        inliers_list,
    )
    if not pnp_success:
        raise RuntimeError("PnP failed; no valid predicted UAV position was produced.")

    pred_x = float(pred_loc["x"])
    pred_y = float(pred_loc["y"])
    best_inliers = 0 if best_index is None else int(inliers_list[best_index])
    best_pnp_input_count = 0 if best_index is None else int(pnp_input_count_list[best_index])
    confidence_pct, inlier_ratio = _compute_prediction_confidence(best_inliers, best_pnp_input_count)
    col, row = _draw_prediction(
        _read_bgr(satellite_path),
        pred_x,
        pred_y,
        view,
        args.output,
        draw_text=not args.no_text,
    )

    result = {
        "uav": None if uav_path is None else str(uav_path),
        "uav_source": "metadata_npz.image" if uav_path is None else "uav_override",
        "satellite": str(satellite_path),
        "metadata_npz": str(metadata_npz),
        "output": str(Path(args.output)),
        "pred_x": pred_x,
        "pred_y": pred_y,
        "pred_z": None if pred_loc.get("z") is None else float(pred_loc["z"]),
        "pred_col": float(col),
        "pred_row": float(row),
        "pred_error_m_from_metadata_gt": None,
        "best_index": None if best_index is None else int(best_index),
        "best_inliers": best_inliers,
        "best_pnp_input_count": best_pnp_input_count,
        "inlier_ratio": inlier_ratio,
        "confidence_pct": confidence_pct,
        "retrieval_time_s": float(retrieval_time),
        "match_time_s": [float(x) for x in match_time],
        "pnp_time_s": [float(x) for x in pnp_time],
        "total_time_s": float(time.time() - t0),
    }
    result_path = Path(args.output).with_suffix(".json")
    with result_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
