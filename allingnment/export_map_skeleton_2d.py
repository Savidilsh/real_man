#!/usr/bin/env python3
"""Export 2D map debug image with occupied obstacles (red) and free-space GVD (blue)."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

from align_las_to_map import load_ros_map


def alpha_blend(base_rgb: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], alpha: float) -> np.ndarray:
    out = base_rgb.copy().astype(np.float32)
    c = np.array(color, dtype=np.float32)
    m = mask.astype(bool)
    if np.any(m):
        out[m] = (1.0 - alpha) * out[m] + alpha * c
    return np.clip(out, 0, 255).astype(np.uint8)


def prune_small_components(mask: np.ndarray, min_component_px: int) -> np.ndarray:
    if min_component_px <= 1:
        return mask
    labels, n = ndimage.label(mask, structure=np.ones((3, 3), dtype=np.uint8))
    if n == 0:
        return mask
    counts = np.bincount(labels.ravel())
    keep = counts >= int(min_component_px)
    keep[0] = False
    return keep[labels]


def keep_largest_component(mask: np.ndarray) -> np.ndarray:
    labels, n = ndimage.label(mask, structure=np.ones((3, 3), dtype=np.uint8))
    if n == 0:
        return np.zeros_like(mask, dtype=bool)
    counts = np.bincount(labels.ravel())
    if counts.shape[0] <= 1:
        return np.zeros_like(mask, dtype=bool)
    largest_label = int(np.argmax(counts[1:]) + 1)
    return labels == largest_label


def build_free_space_gvd_mask(
    occupied: np.ndarray,
    unknown: np.ndarray,
    site_mode: str,
    min_clearance_px: float,
    min_component_px: int,
) -> tuple[np.ndarray, np.ndarray]:
    # Known traversable space only.
    free = (~occupied) & (~unknown)
    if not np.any(free):
        return np.zeros_like(occupied, dtype=bool), np.zeros_like(occupied, dtype=np.float32)

    # ESDF-like distance and nearest-obstacle feature transform.
    dist_to_obstacle, nearest = ndimage.distance_transform_edt(~occupied, return_indices=True)
    if site_mode == "pixel":
        site_r = nearest[0].astype(np.int64)
        site_c = nearest[1].astype(np.int64)
        site_id = site_r * occupied.shape[1] + site_c
    elif site_mode == "component":
        labels, _ = ndimage.label(occupied, structure=np.ones((3, 3), dtype=np.uint8))
        site_id = labels[nearest[0], nearest[1]].astype(np.int64)
    else:
        raise ValueError(f"Unknown site_mode: {site_mode}")

    # Voronoi ridge candidate: neighborhood has multiple nearest obstacle sites.
    min_site = ndimage.minimum_filter(site_id, size=3, mode="nearest")
    max_site = ndimage.maximum_filter(site_id, size=3, mode="nearest")
    site_change = min_site != max_site

    gvd = free & site_change & (dist_to_obstacle >= float(min_clearance_px))
    gvd = prune_small_components(gvd, min_component_px=min_component_px)
    return gvd, dist_to_obstacle.astype(np.float32, copy=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export 2D map overlay with red occupied + blue free-space GVD")
    parser.add_argument("--map-yaml", type=Path, default=Path("2D_map/map.yaml"))
    parser.add_argument("--out-overlay", type=Path, default=Path("outputs/map_overlay_red_gvd_blue_2d.png"))
    parser.add_argument("--out-gvd", type=Path, default=Path("outputs/map_gvd_blue_2d.png"))
    parser.add_argument(
        "--voronoi-min-clearance-px",
        type=float,
        default=2.0,
        help="Minimum ESDF clearance (pixels) for GVD points",
    )
    parser.add_argument(
        "--robot-width-m",
        type=float,
        default=1.0,
        help="Required robot width in meters. GVD keeps only points with clearance >= width/2.",
    )
    parser.add_argument(
        "--voronoi-site-mode",
        type=str,
        default="component",
        choices=["component", "pixel"],
        help="Voronoi site model: obstacle components (Hydra-like) or raw obstacle pixels",
    )
    parser.add_argument(
        "--voronoi-min-component-px",
        type=int,
        default=8,
        help="Remove tiny disconnected GVD components smaller than this size",
    )
    parser.add_argument("--gvd-thickness", type=int, default=2, help="Blue GVD thickness in pixels")
    parser.add_argument(
        "--keep-main-component",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep only the largest connected GVD component",
    )
    parser.add_argument("--red-alpha", type=float, default=0.70)
    parser.add_argument("--blue-alpha", type=float, default=1.0)
    args = parser.parse_args()

    ros_map = load_ros_map(args.map_yaml.resolve())
    base_gray = np.array(Image.open(ros_map.image_path).convert("L"), dtype=np.uint8)

    occupied = ros_map.occupied
    unknown = ros_map.unknown

    required_clearance_px = max(0.0, float(args.robot_width_m) / (2.0 * float(ros_map.resolution)))
    effective_min_clearance_px = max(float(args.voronoi_min_clearance_px), required_clearance_px)

    gvd, dist = build_free_space_gvd_mask(
        occupied=occupied,
        unknown=unknown,
        site_mode=str(args.voronoi_site_mode),
        min_clearance_px=effective_min_clearance_px,
        min_component_px=int(args.voronoi_min_component_px),
    )

    gvd_px_before_main = int(np.sum(gvd))
    if bool(args.keep_main_component) and np.any(gvd):
        gvd = keep_largest_component(gvd)

    if args.gvd_thickness > 1:
        gvd = ndimage.binary_dilation(gvd, iterations=max(0, args.gvd_thickness - 1))

    base_rgb = np.repeat(base_gray[:, :, None], 3, axis=2)
    overlay = alpha_blend(base_rgb, occupied, (255, 0, 0), float(args.red_alpha))
    overlay = alpha_blend(overlay, gvd, (0, 0, 255), float(args.blue_alpha))

    gvd_img = np.full((base_gray.shape[0], base_gray.shape[1], 3), 255, dtype=np.uint8)
    gvd_img[gvd] = np.array([0, 0, 255], dtype=np.uint8)

    args.out_overlay.parent.mkdir(parents=True, exist_ok=True)
    args.out_gvd.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(overlay, mode="RGB").save(args.out_overlay)
    Image.fromarray(gvd_img, mode="RGB").save(args.out_gvd)

    print(f"Map image: {ros_map.image_path}")
    print(f"Overlay saved: {args.out_overlay.resolve()}")
    print(f"GVD-only saved: {args.out_gvd.resolve()}")
    print(f"Occupied pixels: {int(np.sum(occupied))}")
    print(f"Known free pixels: {int(np.sum((~occupied) & (~unknown)))}")
    print(f"Voronoi site mode: {args.voronoi_site_mode}")
    print(f"Map resolution: {ros_map.resolution:.4f} m/px")
    print(f"Robot width requirement: {float(args.robot_width_m):.3f} m")
    print(f"Required clearance radius: {required_clearance_px:.2f} px ({required_clearance_px * ros_map.resolution:.3f} m)")
    print(f"Effective min clearance used: {effective_min_clearance_px:.2f} px ({effective_min_clearance_px * ros_map.resolution:.3f} m)")
    print(f"Keep main component: {bool(args.keep_main_component)} | GVD px before main-filter={gvd_px_before_main}")
    print(f"GVD pixels: {int(np.sum(gvd))}")
    print(f"GVD distance range on graph (px): min={float(np.min(dist[gvd])) if np.any(gvd) else 0.0:.2f}, "
          f"max={float(np.max(dist[gvd])) if np.any(gvd) else 0.0:.2f}")


if __name__ == "__main__":
    main()
