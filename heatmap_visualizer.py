#!/usr/bin/env python3
"""
Standalone localization confidence heatmap renderer.

This module reads a saved localization JSON result and produces a
publication-quality heatmap overlay on the satellite image. It generates a
single, clean Gaussian hotspot centered at the final PnP-verified prediction.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

DEFAULT_VIDEO2_PREDICTION_DIR = Path("Data") / "Vedio_2" / "lol"
DEFAULT_VIDEO2_HEATMAP_DIR = Path("Data") / "Vedio_2" / "heatmap"


def _read_bgr(path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return img


def _load_result_json(path):
    with Path(path).open("r", encoding="utf-8") as f:
        data = json.load(f)
    if "pred_col" not in data or "pred_row" not in data:
        raise KeyError(
            "Result JSON must contain 'pred_col' and 'pred_row'. "
            "Please use a valid localization output file."
        )
    return data


def _read_pgw(path):
    values = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if text:
                values.append(float(text))

    if len(values) != 6:
        raise ValueError(f"Expected 6 numeric lines in PGW file, got {len(values)} from {path}")

    return {
        "a": values[0],
        "d": values[1],
        "b": values[2],
        "e": values[3],
        "c": values[4],
        "f": values[5],
    }





def _build_crs_transformer(source_crs, target_crs):
    from pyproj import CRS, Transformer

    return Transformer.from_crs(
        CRS.from_user_input(source_crs),
        CRS.from_user_input(target_crs),
        always_xy=True,
    )


def _latlon_to_world(latitude, longitude, transformer):
    world_x, world_y = transformer.transform(longitude, latitude)
    return float(world_x), float(world_y)


def _resolve_satellite_path(result, result_json_path):
    """Resolves the satellite image path from the result data."""
    # Check for 'satellite_map' key as per user's JSON structure.
    if "satellite_map" in result and result["satellite_map"]:
        return Path(result["satellite_map"])

    # Fallback to the old key for compatibility.
    if "satellite_image_path" in result and result["satellite_image_path"]:
        return Path(result["satellite_image_path"])

    raise KeyError(
        f"Could not automatically resolve satellite path from '{result_json_path.name}'. "
        "JSON is missing 'satellite_map' or 'satellite_image_path', or the path is empty. "
        "Use the --satellite argument to specify it manually."
    )


def _resolve_world_file_affine(result, satellite_path):
    affine = result.get("world_file_affine")
    if isinstance(affine, dict):
        required = ("a", "d", "b", "e", "c", "f")
        if all(key in affine for key in required):
            return {key: float(affine[key]) for key in required}

    for key in ("pgw_file", "world_file", "world_file_path", "satellite_pgw"):
        if key in result and result[key]:
            candidate = Path(result[key])
            if candidate.exists():
                return _read_pgw(candidate)

    for candidate in (
        satellite_path.with_suffix(".pgw"),
        satellite_path.with_suffix(".wld"),
    ):
        if candidate.exists():
            return _read_pgw(candidate)

    return None


def _resolve_ground_truth_point(
    result,
    satellite_path,
    map_crs,
    gps_crs,
):
    if "gt_col" in result and "gt_row" in result:
        return float(result["gt_col"]), float(result["gt_row"])

    if "gt_pixel_x" in result and "gt_pixel_y" in result:
        return float(result["gt_pixel_x"]), float(result["gt_pixel_y"])

    pgw = _resolve_world_file_affine(result, satellite_path)
    if pgw is None:
        return None

    if "gt_world_x" in result and "gt_world_y" in result:
        return _world_to_pixel(float(result["gt_world_x"]), float(result["gt_world_y"]), pgw)

    if "gt_latitude" in result and "gt_longitude" in result:
        transformer = _build_crs_transformer(gps_crs, map_crs)
        world_x, world_y = _latlon_to_world(
            float(result["gt_latitude"]),
            float(result["gt_longitude"]),
            transformer,
        )
        return _world_to_pixel(world_x, world_y, pgw)

    return None


def _estimate_heatmap_geometry(base_bgr, pred_col, pred_row, sigma_scale):
    """Estimates a practical search-area shape from local image structure."""
    gray = cv2.cvtColor(base_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    h, w = gray.shape[:2]
    sigma = min(h, w) * sigma_scale
    radius = int(max(24, sigma * 3.5))

    x0 = max(0, int(pred_col) - radius)
    x1 = min(w, int(pred_col) + radius + 1)
    y0 = max(0, int(pred_row) - radius)
    y1 = min(h, int(pred_row) + radius + 1)

    patch = gray[y0:y1, x0:x1]
    if patch.size == 0:
        return sigma * 1.25, sigma * 0.80, 0.0

    gx = cv2.Sobel(patch, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(patch, cv2.CV_32F, 0, 1, ksize=3)
    weights = cv2.GaussianBlur(gx * gx + gy * gy, (0, 0), sigmaX=max(1.0, sigma * 0.5))

    yy, xx = np.mgrid[y0:y1, x0:x1]
    xx = xx.astype(np.float32) - float(pred_col)
    yy = yy.astype(np.float32) - float(pred_row)

    weight_sum = float(weights.sum())
    if weight_sum <= 1e-6:
        return sigma * 1.25, sigma * 0.80, 0.0

    cov_xx = float((weights * xx * xx).sum() / weight_sum)
    cov_yy = float((weights * yy * yy).sum() / weight_sum)
    cov_xy = float((weights * xx * yy).sum() / weight_sum)
    covariance = np.array([[cov_xx, cov_xy], [cov_xy, cov_yy]], dtype=np.float32)

    eigvals, eigvecs = np.linalg.eigh(covariance)
    major_index = int(np.argmax(eigvals))
    minor_index = 1 - major_index

    major = float(max(eigvals[major_index], 1e-6))
    minor = float(max(eigvals[minor_index], 1e-6))
    anisotropy = float(np.clip(np.sqrt(major / minor), 1.15, 1.85))

    major_vec = eigvecs[:, major_index]
    angle = float(np.arctan2(major_vec[1], major_vec[0]))

    sigma_major = sigma * np.sqrt(anisotropy)
    sigma_minor = sigma / np.sqrt(anisotropy)
    return sigma_major, sigma_minor, angle


def _build_heatmap(shape, pred_col, pred_row, sigma_scale, base_bgr=None):
    """Builds a practical, slightly elongated 2D Gaussian heatmap."""
    h, w = shape[:2]
    sigma = min(h, w) * sigma_scale
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    if base_bgr is None:
        g = np.exp(-((x - pred_col) ** 2 + (y - pred_row) ** 2) / (2 * sigma ** 2))
    else:
        sigma_x, sigma_y, angle = _estimate_heatmap_geometry(base_bgr, pred_col, pred_row, sigma_scale)
        x_shift = x.astype(np.float32) - float(pred_col)
        y_shift = y.astype(np.float32) - float(pred_row)
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        x_rot = x_shift * cos_a + y_shift * sin_a
        y_rot = -x_shift * sin_a + y_shift * cos_a
        g = np.exp(-((x_rot ** 2) / (2 * sigma_x ** 2) + (y_rot ** 2) / (2 * sigma_y ** 2)))
    # Normalize to 0-1 range
    if g.max() > 0:
        g /= g.max()
    return g


def _build_blue_background(base_bgr):
    """Builds the blue paper-style background without any hotspot blending."""
    # Convert satellite to grayscale
    gray = cv2.cvtColor(base_bgr, cv2.COLOR_BGR2GRAY)

    # Make a blue background
    blue = np.zeros_like(base_bgr)
    blue[:, :, 0] = gray          # Blue
    blue[:, :, 1] = gray * 0.45   # Green
    blue[:, :, 2] = gray * 0.20   # Red

    blue = np.clip(blue, 0, 255).astype(np.uint8)
    return blue


def _apply_hotspot_overlay(background_bgr, heatmap, hotspot_alpha):
    """Overlays a saturated heatmap hotspot on top of a finalized background."""
    heat_color = cv2.applyColorMap(
        (heatmap * 255).astype(np.uint8),
        cv2.COLORMAP_JET
    )

    # A slightly steeper curve keeps the red/yellow core vivid while preserving
    # the green/cyan transition around the falloff.
    alpha_mask = np.power(heatmap, 0.45) * hotspot_alpha
    alpha_mask = np.clip(alpha_mask, 0.0, 1.0)

    # Guarantee a fully saturated hotspot core so the center never gets washed out.
    alpha_mask = np.maximum(alpha_mask, (heatmap >= 0.965).astype(np.float32))
    alpha_mask = alpha_mask[:, :, None]

    result = (
        background_bgr.astype(np.float32) * (1.0 - alpha_mask)
        + heat_color.astype(np.float32) * alpha_mask
    )

    return result.astype(np.uint8), heat_color



def render_heatmap_visualization(
    result_json,
    output_png=None,
    satellite_override=None,
    alpha=0.8,
    sigma_scale=0.04,
    map_crs="EPSG:3857",
    gps_crs="EPSG:4326",
):
    result_json_path = Path(result_json)
    result = _load_result_json(result_json_path)
    satellite_path = Path(satellite_override) if satellite_override else _resolve_satellite_path(result, result_json_path)
    base_bgr = _read_bgr(satellite_path)

    pred_col = float(result["pred_col"])
    pred_row = float(result["pred_row"])

    # 1. Build a single Gaussian heatmap at the prediction
    heatmap = _build_heatmap(base_bgr.shape, pred_col, pred_row, sigma_scale=float(sigma_scale), base_bgr=base_bgr)
    heatmap = cv2.GaussianBlur(heatmap, (0, 0), sigmaX=8)
    # 2. Finalize the blue background first, then overlay the hotspot.
    blue_background = _build_blue_background(base_bgr)
    hotspot_alpha = max(1.0, float(alpha))
    final_image, _ = _apply_hotspot_overlay(blue_background, heatmap, hotspot_alpha=hotspot_alpha)

    gt_point = _resolve_ground_truth_point(
        result,
        satellite_path=satellite_path,
        map_crs=map_crs,
        gps_crs=gps_crs,
    )
    if gt_point is not None:
        gt_x, gt_y = gt_point
        if 0 <= gt_x < final_image.shape[1] and 0 <= gt_y < final_image.shape[0]:
            cv2.circle(final_image, (int(round(gt_x)), int(round(gt_y))), 7, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.circle(final_image, (int(round(gt_x)), int(round(gt_y))), 5, (0, 220, 0), -1, cv2.LINE_AA)

    # 3. Add text box with score, inliers, prediction, and error.
    image_h, image_w = final_image.shape[:2]
    
    # Use keys from user's JSON structure: confidence_pct, gps_error_m
    score_val = result.get("confidence_pct", "N/A")
    inliers_val = result.get("pnp_inliers", "N/A")
    error_m_val = result.get("gps_error_m", "N/A")

    # Format text lines
    text_lines = [
        f"Score: {score_val:.2f}%" if isinstance(score_val, (int, float)) else f"Score: {score_val}",
        f"Pred: ({pred_col:.1f}, {pred_row:.1f})",
        f"Error: {error_m_val:.2f}m" if isinstance(error_m_val, (int, float)) else f"Error: {error_m_val}",
    ]

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    font_thickness = 1
    text_color = (255, 255, 255)  # White text
    box_alpha = 0.6  # Opacity for the background box

    # Calculate text box dimensions
    text_sizes = [cv2.getTextSize(line, font, font_scale, font_thickness)[0] for line in text_lines]
    line_height = text_sizes[0][1] + 5  # height of a line of text + padding
    box_w = max(w for w, h in text_sizes) + 20  # 10px padding on each side
    box_h = (line_height * len(text_lines)) + 15  # top/bottom padding
    
    # Position the box in the bottom right corner
    box_x = image_w - box_w - 10
    box_y = image_h - box_h - 10

    # Create a separate layer for the rectangle
    overlay = final_image.copy()

    # Draw the semi-transparent rectangle on the overlay
    cv2.rectangle(overlay, (box_x, box_y), (box_x + box_w, box_y + box_h), (0, 0, 0), -1)
    
    # Blend the overlay with the image
    cv2.addWeighted(overlay, box_alpha, final_image, 1 - box_alpha, 0, final_image)

    # Add text on top of the blended image
    current_y = box_y + line_height + 5  # Start y for the first line of text
    for line in text_lines:
        cv2.putText(final_image, line, (box_x + 10, current_y), font, font_scale, text_color, font_thickness, cv2.LINE_AA)
        current_y += line_height

    # 4. Save the final image
    if output_png is not None:
        output_png = Path(output_png)
    else:
        output_png = DEFAULT_VIDEO2_HEATMAP_DIR / f"{result_json_path.stem}_heatmap.png"
    output_png.parent.mkdir(parents=True, exist_ok=True)
    
    if not cv2.imwrite(str(output_png), final_image):
        raise RuntimeError(f"Failed to write heatmap visualization: {output_png}")

    summary = {
        "result_json": str(result_json_path),
        "satellite_image": str(satellite_path),
        "output_png": str(output_png),
        "pred_col": pred_col,
        "pred_row": pred_row,
    }
    return summary


def _parse_args():
    parser = argparse.ArgumentParser(description="Render localization confidence heatmaps from saved localization output.")
    parser.add_argument(
        "--input",
        dest="input_path",
        default=str(DEFAULT_VIDEO2_PREDICTION_DIR),
        help=(
            "Path to a single localization JSON file or a directory of them. "
            f"Default: '{DEFAULT_VIDEO2_PREDICTION_DIR}'"
        ),
    )
    parser.add_argument(
        "--output_dir",
        default=str(DEFAULT_VIDEO2_HEATMAP_DIR),
        help=f"Output directory for heatmaps. Default: '{DEFAULT_VIDEO2_HEATMAP_DIR}'",
    )
    parser.add_argument("--satellite", default=None, help="Optional satellite image override.")
    parser.add_argument("--alpha", type=float, default=0.7, help="Overlay alpha for the heatmap colors.")
    parser.add_argument("--sigma_scale", type=float, default=0.02, help="Scale for Gaussian sigma relative to image size.")
    parser.add_argument("--map_crs", default="EPSG:3857", help="CRS used by the satellite map/world file.")
    parser.add_argument("--gps_crs", default="EPSG:4326", help="CRS used by gt_latitude/gt_longitude.")
    return parser.parse_args()


def main():
    args = _parse_args()
    
    input_path = Path(args.input_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if input_path.is_dir():
        json_files = sorted(input_path.glob("*.json"))
        if not json_files:
            print(f"No JSON files found in directory: {input_path}")
            return
    elif input_path.is_file() and input_path.suffix == '.json':
        json_files = [input_path]
    else:
        raise FileNotFoundError(f"Input path is not a valid JSON file or directory: {input_path}")

    print(f"Found {len(json_files)} JSON files to process.")

    for json_file in json_files:
        try:
            output_png = output_dir / f"{json_file.stem}_heatmap.png"
            summary = render_heatmap_visualization(
                result_json=str(json_file),
                output_png=str(output_png),
                satellite_override=args.satellite,
                alpha=args.alpha,
                sigma_scale=args.sigma_scale,
                map_crs=args.map_crs,
                gps_crs=args.gps_crs,
            )
            print(f"Successfully generated heatmap for {json_file.name}")
        except (FileNotFoundError, KeyError, RuntimeError) as e:
            print(f"Could not process {json_file.name}: {e}")

if __name__ == "__main__":
    main()
