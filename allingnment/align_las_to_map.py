#!/usr/bin/env python3
"""
Align a LAS point cloud to a ROS occupancy map (map.yaml + image) in x, y, yaw.

Pipeline:
1) Load ROS map and build obstacle distance field.
2) Read LAS point records in chunks (robust to mismatched LAS header point_count).
3) Estimate floor height from sampled Z values.
4) Keep a floor-relative height band, downsample in XY, optimize x/y/yaw.
5) Export aligned points to .ply and pose report to JSON.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import yaml
from PIL import Image
from scipy import ndimage, optimize

import laspy


@dataclass
class RosMap:
    occupied: np.ndarray
    unknown: np.ndarray
    resolution: float
    origin: np.ndarray
    image_path: Path


@dataclass
class LasMeta:
    path: Path
    offset_to_points: int
    point_size: int
    dtype: np.dtype
    scales: np.ndarray
    offsets: np.ndarray
    total_points: int
    header_point_count: int
    has_color: bool
    tail_bytes: int


_FLOAT_RE = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")


def load_ros_map(yaml_path: Path) -> RosMap:
    with yaml_path.open("r", encoding="utf-8") as f:
        meta = yaml.safe_load(f)

    image_path = Path(meta["image"])
    if not image_path.is_absolute():
        image_path = (yaml_path.parent / image_path).resolve()

    gray = np.array(Image.open(image_path).convert("L"), dtype=np.float32) / 255.0

    resolution = float(meta["resolution"])
    origin = np.array(meta["origin"], dtype=np.float64)
    negate = int(meta.get("negate", 0))
    occ_th = float(meta.get("occupied_thresh", 0.65))
    free_th = float(meta.get("free_thresh", 0.196))

    occ_prob = gray if negate else (1.0 - gray)
    occupied = occ_prob >= occ_th
    free = occ_prob <= free_th
    unknown = ~(occupied | free)

    return RosMap(
        occupied=occupied,
        unknown=unknown,
        resolution=resolution,
        origin=origin,
        image_path=image_path,
    )


def build_distance_field(occupied: np.ndarray, resolution: float, clip_dist_m: float) -> np.ndarray:
    # Distance to nearest occupied pixel; points on obstacles get 0.
    dist_px = ndimage.distance_transform_edt(~occupied)
    dist_m = dist_px * resolution
    if clip_dist_m > 0:
        dist_m = np.minimum(dist_m, clip_dist_m)
    return dist_m.astype(np.float32)


def load_las_meta(path: Path) -> LasMeta:
    with laspy.open(path) as lf:
        h = lf.header
        dtype = np.dtype(h.point_format.dtype())
        point_size = int(h.point_format.size)
        header_point_count = int(h.point_count)
        scales = np.array(h.scales, dtype=np.float64)
        offsets = np.array(h.offsets, dtype=np.float64)
        offset_to_points = int(h.offset_to_point_data)

    file_size = os.path.getsize(path)
    available_bytes = max(0, file_size - offset_to_points)
    usable_bytes = (available_bytes // point_size) * point_size
    total_points = usable_bytes // point_size
    tail_bytes = available_bytes - usable_bytes

    names = set(dtype.names or ())
    has_color = {"red", "green", "blue"}.issubset(names)

    return LasMeta(
        path=path,
        offset_to_points=offset_to_points,
        point_size=point_size,
        dtype=dtype,
        scales=scales,
        offsets=offsets,
        total_points=int(total_points),
        header_point_count=header_point_count,
        has_color=has_color,
        tail_bytes=int(tail_bytes),
    )


def iter_las_chunks(meta: LasMeta, chunk_points: int) -> Iterator[tuple[np.ndarray, np.ndarray | None]]:
    with meta.path.open("rb") as f:
        f.seek(meta.offset_to_points)
        remaining = meta.total_points

        while remaining > 0:
            want = min(chunk_points, remaining)
            raw = f.read(want * meta.point_size)
            if len(raw) < meta.point_size:
                break

            usable = (len(raw) // meta.point_size) * meta.point_size
            if usable <= 0:
                break

            arr = np.frombuffer(raw[:usable], dtype=meta.dtype)
            n = arr.shape[0]
            if n == 0:
                break

            xyz_i = np.column_stack((arr["X"], arr["Y"], arr["Z"]))
            xyz = xyz_i.astype(np.float64, copy=False) * meta.scales + meta.offsets

            colors = None
            if meta.has_color:
                c = np.column_stack((arr["red"], arr["green"], arr["blue"]))
                if c.dtype.itemsize > 1:
                    c = np.clip(c / 256.0, 0, 255).astype(np.uint8)
                else:
                    c = c.astype(np.uint8, copy=False)
                colors = c

            yield xyz, colors
            remaining -= n


def estimate_floor(
    meta: LasMeta,
    chunk_points: int,
    floor_percentile: float,
    z_sample_per_chunk: int,
    rng: np.random.Generator,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    z_samples: list[np.ndarray] = []
    mins = np.array([np.inf, np.inf, np.inf], dtype=np.float64)
    maxs = np.array([-np.inf, -np.inf, -np.inf], dtype=np.float64)

    for xyz, _ in iter_las_chunks(meta, chunk_points=chunk_points):
        mins = np.minimum(mins, xyz.min(axis=0))
        maxs = np.maximum(maxs, xyz.max(axis=0))

        if xyz.shape[0] <= z_sample_per_chunk:
            z_samples.append(xyz[:, 2].copy())
        else:
            idx = rng.choice(xyz.shape[0], size=z_sample_per_chunk, replace=False)
            z_samples.append(xyz[idx, 2].copy())

    if not z_samples:
        raise RuntimeError("No points could be read from LAS file.")

    z_all = np.concatenate(z_samples, axis=0)
    floor_z = float(np.percentile(z_all, floor_percentile))
    return floor_z, mins, maxs, z_all


def collect_band_points(
    meta: LasMeta,
    chunk_points: int,
    band_z_min: float,
    band_z_max: float,
    keep_prob: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray | None, int, int]:
    xyz_parts: list[np.ndarray] = []
    color_parts: list[np.ndarray] = []
    total_candidates = 0
    total_kept = 0

    for xyz, colors in iter_las_chunks(meta, chunk_points=chunk_points):
        mask = (xyz[:, 2] >= band_z_min) & (xyz[:, 2] <= band_z_max)
        if not np.any(mask):
            continue

        cand = xyz[mask]
        cand_colors = colors[mask] if colors is not None else None
        total_candidates += cand.shape[0]

        if keep_prob < 1.0:
            keep = rng.random(cand.shape[0]) < keep_prob
            if not np.any(keep):
                continue
            cand = cand[keep]
            if cand_colors is not None:
                cand_colors = cand_colors[keep]

        xyz_parts.append(cand.astype(np.float32, copy=False))
        if cand_colors is not None:
            color_parts.append(cand_colors.astype(np.uint8, copy=False))
        total_kept += cand.shape[0]

    if not xyz_parts:
        return np.zeros((0, 3), dtype=np.float32), None, 0, 0

    xyz_out = np.concatenate(xyz_parts, axis=0)
    colors_out = np.concatenate(color_parts, axis=0) if color_parts else None
    return xyz_out, colors_out, total_candidates, total_kept


def voxel_downsample_xy(
    xyz: np.ndarray,
    colors: np.ndarray | None,
    voxel_size_m: float,
) -> tuple[np.ndarray, np.ndarray | None]:
    if xyz.shape[0] == 0 or voxel_size_m <= 0:
        return xyz, colors

    keys = np.floor(xyz[:, :2] / voxel_size_m).astype(np.int32)
    _, keep_idx = np.unique(keys, axis=0, return_index=True)
    keep_idx.sort()

    out_xyz = xyz[keep_idx]
    out_colors = colors[keep_idx] if colors is not None else None
    return out_xyz, out_colors


def transform_xy(xy: np.ndarray, tx: float, ty: float, yaw: float) -> np.ndarray:
    c = math.cos(yaw)
    s = math.sin(yaw)
    rot = np.array([[c, -s], [s, c]], dtype=np.float64)
    return xy @ rot.T + np.array([tx, ty], dtype=np.float64)


def world_to_pixel(xy_world: np.ndarray, resolution: float, origin: np.ndarray, height: int) -> tuple[np.ndarray, np.ndarray]:
    x0, y0 = float(origin[0]), float(origin[1])
    cols = (xy_world[:, 0] - x0) / resolution
    rows = (height - 1) - (xy_world[:, 1] - y0) / resolution
    return rows, cols


def score_pose(
    params: np.ndarray,
    xy: np.ndarray,
    dist_field_m: np.ndarray,
    resolution: float,
    origin: np.ndarray,
    outside_penalty_m: float,
    min_valid_fraction: float,
    object_xy: np.ndarray | None = None,
    object_dist_field_m: np.ndarray | None = None,
    object_weight: float = 0.0,
    object_min_valid_fraction: float = 0.0,
) -> float:
    tx, ty, yaw = float(params[0]), float(params[1]), float(params[2])
    xy_t = transform_xy(xy, tx, ty, yaw)

    h, w = dist_field_m.shape
    rows, cols = world_to_pixel(xy_t, resolution=resolution, origin=origin, height=h)

    valid = (rows >= 0.0) & (rows < (h - 1)) & (cols >= 0.0) & (cols < (w - 1))
    valid_count = int(valid.sum())
    if valid_count < max(500, int(min_valid_fraction * xy.shape[0])):
        return 1e6 + outside_penalty_m

    sampled = ndimage.map_coordinates(
        dist_field_m,
        [rows[valid], cols[valid]],
        order=1,
        mode="constant",
        cval=outside_penalty_m,
    )

    median_cost = float(np.median(sampled))
    q75 = float(np.quantile(sampled, 0.75))
    oob_ratio = 1.0 - (valid_count / float(xy.shape[0]))

    base_cost = median_cost + 0.35 * q75 + 0.6 * outside_penalty_m * oob_ratio

    if (
        object_weight <= 0.0
        or object_xy is None
        or object_dist_field_m is None
        or object_xy.shape[0] == 0
    ):
        return base_cost

    obj_t = transform_xy(object_xy, tx, ty, yaw)
    obj_rows, obj_cols = world_to_pixel(obj_t, resolution=resolution, origin=origin, height=h)
    obj_valid = (obj_rows >= 0.0) & (obj_rows < (h - 1)) & (obj_cols >= 0.0) & (obj_cols < (w - 1))
    obj_valid_count = int(obj_valid.sum())
    obj_required = max(1, int(object_min_valid_fraction * object_xy.shape[0]))
    if obj_valid_count < obj_required:
        obj_cost = outside_penalty_m + 0.5 * outside_penalty_m
    else:
        obj_sampled = ndimage.map_coordinates(
            object_dist_field_m,
            [obj_rows[obj_valid], obj_cols[obj_valid]],
            order=1,
            mode="constant",
            cval=outside_penalty_m,
        )
        obj_median = float(np.median(obj_sampled))
        obj_q75 = float(np.quantile(obj_sampled, 0.75))
        obj_oob_ratio = 1.0 - (obj_valid_count / float(object_xy.shape[0]))
        obj_cost = obj_median + 0.35 * obj_q75 + 0.8 * outside_penalty_m * obj_oob_ratio

    return base_cost + float(object_weight) * obj_cost


def write_ply(path: Path, xyz: np.ndarray, colors: np.ndarray | None = None) -> None:
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("xyz must be (N, 3)")

    n = int(xyz.shape[0])
    path.parent.mkdir(parents=True, exist_ok=True)

    if colors is None:
        colors = np.full((n, 3), 200, dtype=np.uint8)
    elif colors.shape != (n, 3):
        raise ValueError("colors must be (N, 3)")

    verts = np.empty(
        n,
        dtype=[
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ],
    )
    verts["x"] = xyz[:, 0].astype(np.float32, copy=False)
    verts["y"] = xyz[:, 1].astype(np.float32, copy=False)
    verts["z"] = xyz[:, 2].astype(np.float32, copy=False)
    verts["red"] = colors[:, 0].astype(np.uint8, copy=False)
    verts["green"] = colors[:, 1].astype(np.uint8, copy=False)
    verts["blue"] = colors[:, 2].astype(np.uint8, copy=False)

    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    )

    with path.open("wb") as f:
        f.write(header.encode("ascii"))
        verts.tofile(f)


def _write_vertex_chunk(f, xyz: np.ndarray, colors: np.ndarray | None) -> None:
    if xyz.shape[0] == 0:
        return
    if colors is None:
        colors = np.full((xyz.shape[0], 3), 200, dtype=np.uint8)

    verts = np.empty(
        xyz.shape[0],
        dtype=[
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ],
    )
    verts["x"] = xyz[:, 0].astype(np.float32, copy=False)
    verts["y"] = xyz[:, 1].astype(np.float32, copy=False)
    verts["z"] = xyz[:, 2].astype(np.float32, copy=False)
    verts["red"] = colors[:, 0].astype(np.uint8, copy=False)
    verts["green"] = colors[:, 1].astype(np.uint8, copy=False)
    verts["blue"] = colors[:, 2].astype(np.uint8, copy=False)
    verts.tofile(f)


def build_esdf_skeleton_mask(
    occupied: np.ndarray,
    min_dist_px: float,
    nms_size: int,
) -> np.ndarray:
    if not np.any(occupied):
        return np.zeros_like(occupied, dtype=bool)

    if nms_size < 3:
        nms_size = 3
    if nms_size % 2 == 0:
        nms_size += 1

    # ESDF inside obstacles: distance to nearest non-obstacle pixel.
    dist_inside = ndimage.distance_transform_edt(occupied)
    if min_dist_px > 0:
        valid_core = dist_inside >= float(min_dist_px)
    else:
        valid_core = occupied

    local_max = dist_inside == ndimage.maximum_filter(dist_inside, size=nms_size, mode="nearest")
    not_flat = dist_inside > ndimage.minimum_filter(dist_inside, size=3, mode="nearest")
    skeleton = occupied & valid_core & local_max & not_flat

    # Fallback for very thin obstacles where strict ridge test becomes empty.
    if not np.any(skeleton):
        skeleton = occupied & local_max

    # Final cleanup: keep only points with at least one obstacle neighbor.
    neigh = ndimage.convolve(skeleton.astype(np.uint8), np.ones((3, 3), dtype=np.uint8), mode="constant", cval=0)
    skeleton = skeleton & (neigh >= 2)
    return skeleton


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
    free = (~occupied) & (~unknown)
    if not np.any(free):
        return np.zeros_like(occupied, dtype=bool), np.zeros_like(occupied, dtype=np.float32)

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

    min_site = ndimage.minimum_filter(site_id, size=3, mode="nearest")
    max_site = ndimage.maximum_filter(site_id, size=3, mode="nearest")
    ridge = min_site != max_site

    gvd = free & ridge & (dist_to_obstacle >= float(min_clearance_px))
    gvd = prune_small_components(gvd, min_component_px=min_component_px)
    return gvd, dist_to_obstacle.astype(np.float32, copy=False)


def build_object_guidance_distance_field(
    occupied: np.ndarray,
    unknown: np.ndarray,
    resolution: float,
    overlay_mode: str,
    skeleton_min_dist_px: float,
    skeleton_nms_size: int,
    gvd_site_mode: str,
    gvd_min_clearance_px: float,
    gvd_min_component_px: int,
    robot_width_m: float,
    keep_main_component: bool,
) -> tuple[np.ndarray | None, str]:
    if overlay_mode == "skeleton_esdf":
        path_mask = build_esdf_skeleton_mask(
            occupied=occupied,
            min_dist_px=skeleton_min_dist_px,
            nms_size=skeleton_nms_size,
        )
        label = "skeleton"
    elif overlay_mode == "occupied_plus_skeleton":
        path_mask = build_esdf_skeleton_mask(
            occupied=occupied,
            min_dist_px=skeleton_min_dist_px,
            nms_size=skeleton_nms_size,
        )
        label = "skeleton"
    elif overlay_mode == "occupied_plus_gvd":
        required_clearance_px = max(0.0, float(robot_width_m) / (2.0 * float(resolution)))
        effective_min_clearance_px = max(float(gvd_min_clearance_px), required_clearance_px)
        path_mask, _ = build_free_space_gvd_mask(
            occupied=occupied,
            unknown=unknown,
            site_mode=gvd_site_mode,
            min_clearance_px=effective_min_clearance_px,
            min_component_px=gvd_min_component_px,
        )
        label = "gvd"
    else:
        return None, "none"

    if not np.any(path_mask):
        return None, label
    if keep_main_component:
        path_mask = keep_largest_component(path_mask)

    dist_px = ndimage.distance_transform_edt(~path_mask)
    dist_m = (dist_px * float(resolution)).astype(np.float32, copy=False)
    return dist_m, label


def mask_to_world_xy(mask: np.ndarray, resolution: float, origin: np.ndarray, pixel_stride: int) -> np.ndarray:
    rows, cols = np.nonzero(mask)
    if rows.size == 0:
        return np.zeros((0, 2), dtype=np.float32)

    if pixel_stride > 1:
        keep = ((rows % pixel_stride) == 0) & ((cols % pixel_stride) == 0)
        rows = rows[keep]
        cols = cols[keep]
        if rows.size == 0:
            return np.zeros((0, 2), dtype=np.float32)

    h = mask.shape[0]
    x = float(origin[0]) + (cols.astype(np.float64) + 0.5) * float(resolution)
    y = float(origin[1]) + (h - 0.5 - rows.astype(np.float64)) * float(resolution)
    return np.column_stack((x, y)).astype(np.float32)


def _parse_floats(text: str) -> list[float]:
    return [float(m.group(0)) for m in _FLOAT_RE.finditer(text)]


def load_object_xy(path: Path) -> np.ndarray:
    if not path.exists():
        return np.zeros((0, 2), dtype=np.float32)

    xy: list[tuple[float, float]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue

            centroid_match = re.search(r"centroid\s*=\s*\[([^\]]+)\]", s, flags=re.IGNORECASE)
            if centroid_match:
                nums = _parse_floats(centroid_match.group(1))
            else:
                bracket_match = re.search(r"\[([^\]]+)\]", s)
                if bracket_match:
                    nums = _parse_floats(bracket_match.group(1))
                else:
                    nums = _parse_floats(s)

            if len(nums) >= 3:
                x, y = float(nums[-3]), float(nums[-2])
            elif len(nums) >= 2:
                x, y = float(nums[0]), float(nums[1])
            else:
                continue

            if np.isfinite(x) and np.isfinite(y):
                xy.append((x, y))

    if not xy:
        return np.zeros((0, 2), dtype=np.float32)
    return np.array(xy, dtype=np.float32)


def world_xy_to_mask_rc(
    xy: np.ndarray,
    resolution: float,
    origin: np.ndarray,
    mask_shape: tuple[int, int],
) -> np.ndarray:
    if xy.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.int64)

    h, w = mask_shape
    cols_f = (xy[:, 0].astype(np.float64, copy=False) - float(origin[0])) / float(resolution) - 0.5
    rows_f = (float(h) - 0.5) - (xy[:, 1].astype(np.float64, copy=False) - float(origin[1])) / float(resolution)

    cols = np.rint(cols_f).astype(np.int64)
    rows = np.rint(rows_f).astype(np.int64)

    inside = (rows >= 0) & (rows < h) & (cols >= 0) & (cols < w)
    if not np.any(inside):
        return np.zeros((0, 2), dtype=np.int64)
    return np.column_stack((rows[inside], cols[inside])).astype(np.int64, copy=False)


def filter_path_mask_by_object_xy(
    path_mask: np.ndarray,
    resolution: float,
    origin: np.ndarray,
    object_xy_world: np.ndarray,
    filter_radius_m: float,
) -> tuple[np.ndarray, dict[str, float | int | bool]]:
    stats: dict[str, float | int | bool] = {
        "applied": False,
        "radius_m": float(max(0.0, filter_radius_m)),
        "object_points_total": int(object_xy_world.shape[0]),
        "object_points_in_map": 0,
        "object_points_matched": 0,
        "path_components_total": 0,
        "path_components_kept": 0,
        "path_pixels_before": int(np.sum(path_mask)),
        "path_pixels_after": int(np.sum(path_mask)),
    }

    if not np.any(path_mask) or object_xy_world.shape[0] == 0:
        return path_mask, stats

    radius_px = float(max(0.0, filter_radius_m)) / max(1e-12, float(resolution))
    object_rc = world_xy_to_mask_rc(object_xy_world, resolution=resolution, origin=origin, mask_shape=path_mask.shape)
    stats["object_points_in_map"] = int(object_rc.shape[0])

    if object_rc.shape[0] == 0:
        stats["applied"] = True
        stats["path_pixels_after"] = 0
        return np.zeros_like(path_mask, dtype=bool), stats

    labels, n_labels = ndimage.label(path_mask, structure=np.ones((3, 3), dtype=np.uint8))
    stats["path_components_total"] = int(n_labels)
    if n_labels == 0:
        stats["applied"] = True
        stats["path_pixels_after"] = 0
        return np.zeros_like(path_mask, dtype=bool), stats

    dist_to_path, nearest_idx = ndimage.distance_transform_edt(~path_mask, return_indices=True)

    keep_labels: set[int] = set()
    matched_points = 0
    for r, c in object_rc:
        if float(dist_to_path[r, c]) > radius_px:
            continue
        rr = int(nearest_idx[0, r, c])
        cc = int(nearest_idx[1, r, c])
        label_id = int(labels[rr, cc])
        if label_id <= 0:
            continue
        keep_labels.add(label_id)
        matched_points += 1

    if keep_labels:
        keep_ids = np.fromiter(keep_labels, dtype=labels.dtype, count=len(keep_labels))
        filtered = np.isin(labels, keep_ids)
    else:
        filtered = np.zeros_like(path_mask, dtype=bool)

    stats["applied"] = True
    stats["object_points_matched"] = int(matched_points)
    stats["path_components_kept"] = int(len(keep_labels))
    stats["path_pixels_after"] = int(np.sum(filtered))
    return filtered, stats


def build_map_obstacle_cloud(
    occupied: np.ndarray,
    unknown: np.ndarray,
    resolution: float,
    origin: np.ndarray,
    band_z_min: float,
    band_z_max: float,
    overlay_mode: str,
    z_placement: str,
    z_layers: int,
    pixel_stride: int,
    skeleton_min_dist_px: float,
    skeleton_nms_size: int,
    gvd_site_mode: str,
    gvd_min_clearance_px: float,
    gvd_min_component_px: int,
    robot_width_m: float,
    red_thickness_px: int,
    blue_thickness_px: int,
    object_filter_xy: np.ndarray | None,
    object_filter_radius_m: float,
    keep_main_component: bool,
) -> tuple[np.ndarray, np.ndarray, dict[str, float | int | bool]]:
    if z_layers < 1:
        z_layers = 1
    if pixel_stride < 1:
        pixel_stride = 1

    if red_thickness_px < 1:
        red_thickness_px = 1
    if blue_thickness_px < 1:
        blue_thickness_px = 1

    red = np.array([255, 0, 0], dtype=np.uint8)
    blue = np.array([0, 0, 255], dtype=np.uint8)
    green = np.array([0, 255, 0], dtype=np.uint8)

    red_mask = np.zeros_like(occupied, dtype=bool)
    blue_mask = np.zeros_like(occupied, dtype=bool)
    path_filter_stats: dict[str, float | int | bool] = {
        "applied": False,
        "radius_m": float(max(0.0, object_filter_radius_m)),
        "object_points_total": int(0 if object_filter_xy is None else object_filter_xy.shape[0]),
        "object_points_in_map": 0,
        "object_points_matched": 0,
        "object_points_overlay": 0,
        "path_components_total": 0,
        "path_components_kept": 0,
        "path_pixels_before": 0,
        "path_pixels_after": 0,
    }

    if overlay_mode == "skeleton_esdf":
        skeleton = build_esdf_skeleton_mask(
            occupied=occupied,
            min_dist_px=skeleton_min_dist_px,
            nms_size=skeleton_nms_size,
        )
        red_mask = skeleton
    elif overlay_mode == "occupied":
        red_mask = occupied
    elif overlay_mode == "occupied_plus_skeleton":
        skeleton = build_esdf_skeleton_mask(
            occupied=occupied,
            min_dist_px=skeleton_min_dist_px,
            nms_size=skeleton_nms_size,
        )
        red_mask = occupied
        blue_mask = skeleton
    elif overlay_mode == "occupied_plus_gvd":
        required_clearance_px = max(0.0, float(robot_width_m) / (2.0 * float(resolution)))
        effective_min_clearance_px = max(float(gvd_min_clearance_px), required_clearance_px)
        gvd, _ = build_free_space_gvd_mask(
            occupied=occupied,
            unknown=unknown,
            site_mode=gvd_site_mode,
            min_clearance_px=effective_min_clearance_px,
            min_component_px=gvd_min_component_px,
        )
        red_mask = occupied
        blue_mask = gvd
    else:
        raise ValueError(f"Unknown overlay_mode: {overlay_mode}")

    if object_filter_xy is not None and object_filter_xy.shape[0] > 0:
        if overlay_mode == "skeleton_esdf":
            red_mask, path_filter_stats = filter_path_mask_by_object_xy(
                path_mask=red_mask,
                resolution=resolution,
                origin=origin,
                object_xy_world=object_filter_xy,
                filter_radius_m=object_filter_radius_m,
            )
        elif overlay_mode in ("occupied_plus_skeleton", "occupied_plus_gvd"):
            blue_mask, path_filter_stats = filter_path_mask_by_object_xy(
                path_mask=blue_mask,
                resolution=resolution,
                origin=origin,
                object_xy_world=object_filter_xy,
                filter_radius_m=object_filter_radius_m,
            )

    if keep_main_component:
        if overlay_mode == "skeleton_esdf" and np.any(red_mask):
            red_mask = keep_largest_component(red_mask)
        elif overlay_mode in ("occupied_plus_skeleton", "occupied_plus_gvd") and np.any(blue_mask):
            blue_mask = keep_largest_component(blue_mask)

    if red_thickness_px > 1 and np.any(red_mask):
        red_mask = ndimage.binary_dilation(red_mask, iterations=red_thickness_px - 1)
    if blue_thickness_px > 1 and np.any(blue_mask):
        blue_mask = ndimage.binary_dilation(blue_mask, iterations=blue_thickness_px - 1)

    if np.any(blue_mask):
        # Keep full occupied map in red, but carve skeleton pixels out so they
        # appear cleanly as blue (no z-fighting at identical coordinates).
        red_mask = red_mask & (~blue_mask)

    mask_color_pairs: list[tuple[np.ndarray, np.ndarray]] = []
    if np.any(red_mask):
        mask_color_pairs.append((red_mask, red))
    if np.any(blue_mask):
        mask_color_pairs.append((blue_mask, blue))

    if z_placement == "bottom":
        z_values = np.array([band_z_min], dtype=np.float64)
    elif z_placement == "middle":
        z_values = np.array([0.5 * (band_z_min + band_z_max)], dtype=np.float64)
    elif z_placement == "top":
        z_values = np.array([band_z_max], dtype=np.float64)
    elif z_placement == "full_band":
        if z_layers == 1 or band_z_max <= band_z_min:
            z_values = np.array([0.5 * (band_z_min + band_z_max)], dtype=np.float64)
        else:
            z_values = np.linspace(band_z_min, band_z_max, num=z_layers, dtype=np.float64)
    else:
        raise ValueError(f"Unknown z_placement: {z_placement}")

    object_overlay_xy = np.zeros((0, 2), dtype=np.float32)
    if object_filter_xy is not None and object_filter_xy.shape[0] > 0:
        h, w = occupied.shape
        x = object_filter_xy[:, 0].astype(np.float64, copy=False)
        y = object_filter_xy[:, 1].astype(np.float64, copy=False)
        min_x = float(origin[0])
        max_x = float(origin[0]) + float(w) * float(resolution)
        min_y = float(origin[1])
        max_y = float(origin[1]) + float(h) * float(resolution)
        inside = (x >= min_x) & (x < max_x) & (y >= min_y) & (y < max_y)
        if np.any(inside):
            object_overlay_xy = object_filter_xy[inside].astype(np.float32, copy=False)
            path_filter_stats["object_points_overlay"] = int(object_overlay_xy.shape[0])

    xyz_parts: list[np.ndarray] = []
    color_parts: list[np.ndarray] = []
    for mask, rgb in mask_color_pairs:
        xy = mask_to_world_xy(mask=mask, resolution=resolution, origin=origin, pixel_stride=pixel_stride)
        if xy.shape[0] == 0:
            continue

        n_xy = xy.shape[0]
        xyz = np.empty((n_xy * z_values.shape[0], 3), dtype=np.float32)
        for i, z in enumerate(z_values):
            s = i * n_xy
            e = s + n_xy
            xyz[s:e, :2] = xy
            xyz[s:e, 2] = np.float32(z)

        colors = np.tile(rgb[None, :], (xyz.shape[0], 1))
        xyz_parts.append(xyz)
        color_parts.append(colors)

    if object_overlay_xy.shape[0] > 0:
        object_xyz = np.empty((object_overlay_xy.shape[0], 3), dtype=np.float32)
        object_xyz[:, :2] = object_overlay_xy
        object_xyz[:, 2] = np.float32(0.5 * (band_z_min + band_z_max))
        object_colors = np.tile(green[None, :], (object_xyz.shape[0], 1))
        xyz_parts.append(object_xyz)
        color_parts.append(object_colors)

    if not xyz_parts:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8), path_filter_stats
    return np.concatenate(xyz_parts, axis=0), np.concatenate(color_parts, axis=0), path_filter_stats


def stream_write_aligned_full_ply(
    path: Path,
    meta: LasMeta,
    chunk_points: int,
    tx: float,
    ty: float,
    yaw: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {meta.total_points}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    )

    with path.open("wb") as f:
        f.write(header.encode("ascii"))

        for xyz, colors in iter_las_chunks(meta, chunk_points=chunk_points):
            xy_t = transform_xy(xyz[:, :2], tx=tx, ty=ty, yaw=yaw)
            out_xyz = np.empty_like(xyz, dtype=np.float32)
            out_xyz[:, :2] = xy_t.astype(np.float32, copy=False)
            out_xyz[:, 2] = xyz[:, 2].astype(np.float32, copy=False)
            colors_chunk = colors.astype(np.uint8, copy=False) if colors is not None else None
            _write_vertex_chunk(f, out_xyz, colors_chunk)


def stream_write_full_with_map_overlay_ply(
    path: Path,
    meta: LasMeta,
    chunk_points: int,
    tx: float,
    ty: float,
    yaw: float,
    map_xyz: np.ndarray,
    map_colors: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    total_vertices = int(meta.total_points + map_xyz.shape[0])

    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {total_vertices}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    )

    with path.open("wb") as f:
        f.write(header.encode("ascii"))

        for xyz, colors in iter_las_chunks(meta, chunk_points=chunk_points):
            xy_t = transform_xy(xyz[:, :2], tx=tx, ty=ty, yaw=yaw)
            out_xyz = np.empty_like(xyz, dtype=np.float32)
            out_xyz[:, :2] = xy_t.astype(np.float32, copy=False)
            out_xyz[:, 2] = xyz[:, 2].astype(np.float32, copy=False)
            colors_chunk = colors.astype(np.uint8, copy=False) if colors is not None else None
            _write_vertex_chunk(f, out_xyz, colors_chunk)

        _write_vertex_chunk(f, map_xyz, map_colors)


def main() -> None:
    parser = argparse.ArgumentParser(description="Align LAS cloud to ROS map and export aligned .ply")
    parser.add_argument("--map-yaml", type=Path, default=Path("2D_map/map.yaml"), help="Path to ROS map YAML")
    parser.add_argument(
        "--las",
        type=Path,
        default=Path("pointcloud/RoboticsInstitute-pointcloud-fine.las"),
        help="Path to LAS point cloud",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"), help="Directory for outputs")
    parser.add_argument(
        "--output-ply",
        type=Path,
        default=None,
        help="Aligned band cloud output .ply (default: <output-dir>/aligned_band_cloud.ply)",
    )
    parser.add_argument(
        "--output-full-ply",
        type=Path,
        default=None,
        help="Optional full aligned cloud .ply (can be very large)",
    )
    parser.add_argument(
        "--output-overlay-ply",
        type=Path,
        default=None,
        help="Optional combined full cloud + red map obstacle band .ply",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Alignment report JSON (default: <output-dir>/alignment_result.json)",
    )

    parser.add_argument("--chunk-points", type=int, default=1_000_000, help="LAS chunk size")
    parser.add_argument("--seed", type=int, default=7, help="Random seed")

    parser.add_argument("--floor-percentile", type=float, default=2.0, help="Percentile for floor estimation")
    parser.add_argument("--z-band-min", type=float, default=0.2, help="Meters above floor (min)")
    parser.add_argument("--z-band-max", type=float, default=2.0, help="Meters above floor (max)")

    parser.add_argument(
        "--z-sample-per-chunk",
        type=int,
        default=50_000,
        help="Number of Z samples per chunk for floor estimation",
    )
    parser.add_argument(
        "--prevoxel-target-points",
        type=int,
        default=1_200_000,
        help="Target random sample count before voxel downsampling",
    )
    parser.add_argument("--voxel-size", type=float, default=0.08, help="XY voxel size in meters")
    parser.add_argument("--max-opt-points", type=int, default=25_000, help="Max points used by optimizer")

    parser.add_argument("--dist-clip-m", type=float, default=3.0, help="Distance-field clip distance in meters")
    parser.add_argument("--search-margin-m", type=float, default=8.0, help="Extra translation search margin")
    parser.add_argument("--de-maxiter", type=int, default=45, help="Differential evolution max iterations")
    parser.add_argument("--de-popsize", type=int, default=18, help="Differential evolution population size")
    parser.add_argument("--powell-maxiter", type=int, default=250, help="Powell max iterations")
    parser.add_argument(
        "--map-overlay-mode",
        type=str,
        default="occupied_plus_gvd",
        choices=["skeleton_esdf", "occupied", "occupied_plus_skeleton", "occupied_plus_gvd"],
        help="Map overlay geometry: skeleton-only, occupied-only, occupied+skeleton, or occupied+GVD",
    )
    parser.add_argument(
        "--map-overlay-z-placement",
        type=str,
        default="bottom",
        choices=["bottom", "middle", "top", "full_band"],
        help="Where to place map overlay in Z: bottom/middle/top of band, or full band extrusion",
    )
    parser.add_argument("--map-overlay-z-layers", type=int, default=1, help="Z slices for full_band overlay mode")
    parser.add_argument(
        "--map-overlay-pixel-stride",
        type=int,
        default=1,
        help="Subsample occupied map pixels for overlay (1 keeps all)",
    )
    parser.add_argument(
        "--map-skeleton-min-dist-px",
        type=float,
        default=1.0,
        help="Minimum ESDF distance (pixels) for skeleton points",
    )
    parser.add_argument(
        "--map-skeleton-nms-size",
        type=int,
        default=3,
        help="Neighborhood size for ESDF ridge non-maximum suppression",
    )
    parser.add_argument(
        "--map-gvd-site-mode",
        type=str,
        default="component",
        choices=["component", "pixel"],
        help="Voronoi site model for GVD: obstacle components (recommended) or raw obstacle pixels",
    )
    parser.add_argument(
        "--map-gvd-min-clearance-px",
        type=float,
        default=0.0,
        help="Minimum ESDF clearance in pixels for GVD points before robot-width filter",
    )
    parser.add_argument(
        "--map-gvd-min-component-px",
        type=int,
        default=8,
        help="Minimum connected-component size for GVD cleanup",
    )
    parser.add_argument(
        "--map-overlay-robot-width-m",
        type=float,
        default=0.75,
        help="Robot width (meters). For GVD mode, keeps only branches with clearance >= width/2",
    )
    parser.add_argument(
        "--map-overlay-red-thickness-px",
        type=int,
        default=2,
        help="Dilation thickness for red map lines in pixels",
    )
    parser.add_argument(
        "--map-overlay-blue-thickness-px",
        type=int,
        default=3,
        help="Dilation thickness for blue map lines in pixels",
    )
    parser.add_argument(
        "--map-keep-main-component",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep only the largest connected path component (blue graph in GVD/skeleton overlay modes)",
    )
    parser.add_argument(
        "--object-coordinates",
        type=Path,
        default=Path("pointcloud/detections.txt"),
        help="Object coordinate file used to filter skeleton/GVD paths (default: pointcloud/detections.txt)",
    )
    parser.add_argument(
        "--object-coordinates-frame",
        type=str,
        default="las",
        choices=["las", "map"],
        help="Frame for object coordinates: 'las' (raw cloud frame) or 'map' (already aligned map frame)",
    )
    parser.add_argument(
        "--object-path-filter-radius-m",
        type=float,
        default=1.0,
        help="Keep only skeleton/GVD components within this XY distance (meters) of object coordinates",
    )
    parser.add_argument(
        "--object-align-weight",
        type=float,
        default=1.25,
        help="Weight for object-anchor alignment term during optimization (0 disables object-guided alignment)",
    )
    parser.add_argument(
        "--object-align-min-valid-fraction",
        type=float,
        default=0.3,
        help="Minimum in-map fraction required for object anchors in optimization scoring",
    )

    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    map_yaml = args.map_yaml.resolve()
    las_path = args.las.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    output_ply = args.output_ply.resolve() if args.output_ply else (output_dir / "aligned_band_cloud.ply")
    output_full = args.output_full_ply.resolve() if args.output_full_ply else None
    output_overlay = args.output_overlay_ply.resolve() if args.output_overlay_ply else None
    output_json = args.output_json.resolve() if args.output_json else (output_dir / "alignment_result.json")
    object_coordinates_path = args.object_coordinates.resolve()
    object_xy_raw = load_object_xy(object_coordinates_path)

    ros_map = load_ros_map(map_yaml)
    dist_field = build_distance_field(ros_map.occupied, ros_map.resolution, args.dist_clip_m)

    las_meta = load_las_meta(las_path)
    if las_meta.total_points <= 0:
        raise RuntimeError("No readable points found in LAS file.")

    print(f"Map loaded: {ros_map.image_path} shape={ros_map.occupied.shape} res={ros_map.resolution:.4f} m/px")
    print(
        f"LAS readable points: {las_meta.total_points:,} | header point_count: {las_meta.header_point_count:,} | "
        f"tail_bytes: {las_meta.tail_bytes}"
    )
    if object_xy_raw.shape[0] > 0:
        print(
            f"Loaded {object_xy_raw.shape[0]} object coordinates from {object_coordinates_path} "
            f"(frame={args.object_coordinates_frame})"
        )
    elif object_coordinates_path.exists():
        print(f"No valid object coordinates parsed from: {object_coordinates_path}")
    else:
        print(f"Object coordinate file not found: {object_coordinates_path}")

    floor_z, mins, maxs, z_sample = estimate_floor(
        las_meta,
        chunk_points=args.chunk_points,
        floor_percentile=args.floor_percentile,
        z_sample_per_chunk=args.z_sample_per_chunk,
        rng=rng,
    )
    band_min = floor_z + float(args.z_band_min)
    band_max = floor_z + float(args.z_band_max)

    est_band_ratio = float(np.mean((z_sample >= band_min) & (z_sample <= band_max)))
    est_band_points = max(1, int(est_band_ratio * las_meta.total_points))
    keep_prob = min(1.0, args.prevoxel_target_points / float(est_band_points))

    print(
        f"LAS bounds xyz min={mins.round(3)} max={maxs.round(3)} | floor_z~{floor_z:.3f} | "
        f"band=[{band_min:.3f}, {band_max:.3f}]"
    )
    print(f"Estimated band points: {est_band_points:,} | keep_prob={keep_prob:.4f}")

    band_xyz, band_colors, band_candidates, band_kept = collect_band_points(
        las_meta,
        chunk_points=args.chunk_points,
        band_z_min=band_min,
        band_z_max=band_max,
        keep_prob=keep_prob,
        rng=rng,
    )
    if band_xyz.shape[0] == 0:
        raise RuntimeError(
            "Height band produced no points. Try broader z-band values, e.g. --z-band-min 0.0 --z-band-max 3.0"
        )

    down_xyz, down_colors = voxel_downsample_xy(band_xyz, band_colors, voxel_size_m=args.voxel_size)
    opt_xy = down_xyz[:, :2].astype(np.float64, copy=False)

    if opt_xy.shape[0] > args.max_opt_points:
        idx = rng.choice(opt_xy.shape[0], size=args.max_opt_points, replace=False)
        opt_xy = opt_xy[idx]

    h, w = ros_map.occupied.shape
    map_min_x = float(ros_map.origin[0])
    map_max_x = float(ros_map.origin[0] + w * ros_map.resolution)
    map_min_y = float(ros_map.origin[1])
    map_max_y = float(ros_map.origin[1] + h * ros_map.resolution)

    centroid = opt_xy.mean(axis=0)
    bounds = [
        (map_min_x - centroid[0] - args.search_margin_m, map_max_x - centroid[0] + args.search_margin_m),
        (map_min_y - centroid[1] - args.search_margin_m, map_max_y - centroid[1] + args.search_margin_m),
        (-math.pi, math.pi),
    ]

    print(
        f"Band points: candidates={band_candidates:,}, kept={band_kept:,}, voxel_downsampled={down_xyz.shape[0]:,}, "
        f"opt_points={opt_xy.shape[0]:,}"
    )
    print("Starting optimization (differential evolution -> Powell)...")

    object_xy_for_opt = (
        object_xy_raw.astype(np.float64, copy=False)
        if args.object_coordinates_frame == "las"
        else np.zeros((0, 2), dtype=np.float64)
    )
    object_guidance_dist, object_guidance_type = build_object_guidance_distance_field(
        occupied=ros_map.occupied,
        unknown=ros_map.unknown,
        resolution=ros_map.resolution,
        overlay_mode=args.map_overlay_mode,
        skeleton_min_dist_px=args.map_skeleton_min_dist_px,
        skeleton_nms_size=args.map_skeleton_nms_size,
        gvd_site_mode=args.map_gvd_site_mode,
        gvd_min_clearance_px=args.map_gvd_min_clearance_px,
        gvd_min_component_px=args.map_gvd_min_component_px,
        robot_width_m=args.map_overlay_robot_width_m,
        keep_main_component=bool(args.map_keep_main_component),
    )
    object_alignment_enabled = (
        float(args.object_align_weight) > 0.0
        and object_xy_for_opt.shape[0] > 0
        and object_guidance_dist is not None
    )
    if object_alignment_enabled:
        print(
            f"Object-guided alignment enabled: points={object_xy_for_opt.shape[0]} | "
            f"guidance={object_guidance_type} | weight={float(args.object_align_weight):.3f}"
        )
    elif object_xy_raw.shape[0] > 0 and args.object_coordinates_frame != "las":
        print("Object-guided alignment skipped (object coordinates are in map frame, expected LAS frame).")
    elif object_xy_raw.shape[0] > 0 and object_guidance_dist is None:
        print(f"Object-guided alignment skipped (no usable {object_guidance_type} guidance mask on map).")

    score_args = (
        opt_xy,
        dist_field,
        ros_map.resolution,
        ros_map.origin,
        float(args.dist_clip_m),
        0.15,
        object_xy_for_opt if object_alignment_enabled else None,
        object_guidance_dist if object_alignment_enabled else None,
        float(args.object_align_weight) if object_alignment_enabled else 0.0,
        float(args.object_align_min_valid_fraction),
    )

    coarse = optimize.differential_evolution(
        score_pose,
        bounds=bounds,
        args=score_args,
        maxiter=args.de_maxiter,
        popsize=args.de_popsize,
        seed=args.seed,
        polish=False,
        workers=1,
        updating="deferred",
    )

    fine = optimize.minimize(
        score_pose,
        x0=coarse.x,
        args=score_args,
        method="Powell",
        bounds=bounds,
        options={"maxiter": args.powell_maxiter, "xtol": 1e-4, "ftol": 1e-4},
    )

    if fine.success:
        best = fine.x
        best_cost = float(fine.fun)
        method = "powell"
    else:
        best = coarse.x
        best_cost = float(coarse.fun)
        method = "differential_evolution"

    tx, ty, yaw = float(best[0]), float(best[1]), float(best[2])

    aligned_xyz = down_xyz.copy()
    aligned_xyz[:, :2] = transform_xy(down_xyz[:, :2].astype(np.float64), tx=tx, ty=ty, yaw=yaw).astype(np.float32)

    write_ply(output_ply, aligned_xyz, down_colors)

    if output_full is not None:
        print(f"Writing full aligned cloud to: {output_full}")
        stream_write_aligned_full_ply(
            path=output_full,
            meta=las_meta,
            chunk_points=args.chunk_points,
            tx=tx,
            ty=ty,
            yaw=yaw,
        )

    map_overlay_points = 0
    map_overlay_red_points = 0
    map_overlay_blue_points = 0
    map_overlay_green_points = 0
    map_frame_object_xy = np.zeros((0, 2), dtype=np.float32)
    map_overlay_filter_stats: dict[str, float | int | bool] = {
        "applied": False,
        "radius_m": float(max(0.0, args.object_path_filter_radius_m)),
        "object_points_total": int(object_xy_raw.shape[0]),
        "object_points_in_map": 0,
        "object_points_matched": 0,
        "object_points_overlay": 0,
        "path_components_total": 0,
        "path_components_kept": 0,
        "path_pixels_before": 0,
        "path_pixels_after": 0,
    }
    if output_overlay is not None:
        if object_xy_raw.shape[0] > 0:
            if args.object_coordinates_frame == "las":
                map_frame_object_xy = transform_xy(
                    object_xy_raw.astype(np.float64, copy=False),
                    tx=tx,
                    ty=ty,
                    yaw=yaw,
                ).astype(np.float32, copy=False)
            else:
                map_frame_object_xy = object_xy_raw.astype(np.float32, copy=False)

        map_xyz, map_colors, map_overlay_filter_stats = build_map_obstacle_cloud(
            occupied=ros_map.occupied,
            unknown=ros_map.unknown,
            resolution=ros_map.resolution,
            origin=ros_map.origin,
            band_z_min=band_min,
            band_z_max=band_max,
            overlay_mode=args.map_overlay_mode,
            z_placement=args.map_overlay_z_placement,
            z_layers=args.map_overlay_z_layers,
            pixel_stride=args.map_overlay_pixel_stride,
            skeleton_min_dist_px=args.map_skeleton_min_dist_px,
            skeleton_nms_size=args.map_skeleton_nms_size,
            gvd_site_mode=args.map_gvd_site_mode,
            gvd_min_clearance_px=args.map_gvd_min_clearance_px,
            gvd_min_component_px=args.map_gvd_min_component_px,
            robot_width_m=args.map_overlay_robot_width_m,
            red_thickness_px=args.map_overlay_red_thickness_px,
            blue_thickness_px=args.map_overlay_blue_thickness_px,
            object_filter_xy=map_frame_object_xy,
            object_filter_radius_m=args.object_path_filter_radius_m,
            keep_main_component=bool(args.map_keep_main_component),
        )
        map_overlay_points = int(map_xyz.shape[0])
        if map_colors.shape[0] > 0:
            map_overlay_red_points = int(
                np.sum((map_colors[:, 0] == 255) & (map_colors[:, 1] == 0) & (map_colors[:, 2] == 0))
            )
            map_overlay_blue_points = int(
                np.sum((map_colors[:, 0] == 0) & (map_colors[:, 1] == 0) & (map_colors[:, 2] == 255))
            )
            map_overlay_green_points = int(
                np.sum((map_colors[:, 0] == 0) & (map_colors[:, 1] == 255) & (map_colors[:, 2] == 0))
            )
        print(
            "Writing combined overlay cloud (full + map overlay) to: "
            f"{output_overlay} | mode={args.map_overlay_mode} | z={args.map_overlay_z_placement} | "
            f"red points={map_overlay_red_points:,} | blue graph points={map_overlay_blue_points:,} | "
            f"green object points={map_overlay_green_points:,}"
        )
        if bool(map_overlay_filter_stats.get("applied", False)):
            print(
                "Path filter: "
                f"objects_in_map={int(map_overlay_filter_stats.get('object_points_in_map', 0))} | "
                f"matched={int(map_overlay_filter_stats.get('object_points_matched', 0))} | "
                f"components_kept={int(map_overlay_filter_stats.get('path_components_kept', 0))}/"
                f"{int(map_overlay_filter_stats.get('path_components_total', 0))}"
            )
        stream_write_full_with_map_overlay_ply(
            path=output_overlay,
            meta=las_meta,
            chunk_points=args.chunk_points,
            tx=tx,
            ty=ty,
            yaw=yaw,
            map_xyz=map_xyz,
            map_colors=map_colors,
        )

    report = {
        "inputs": {
            "map_yaml": str(map_yaml),
            "map_image": str(ros_map.image_path),
            "las": str(las_path),
        },
        "outputs": {
            "aligned_band_ply": str(output_ply),
            "aligned_full_ply": str(output_full) if output_full else None,
            "aligned_overlay_ply": str(output_overlay) if output_overlay else None,
        },
        "map": {
            "shape_hw": [int(h), int(w)],
            "resolution_m_per_px": float(ros_map.resolution),
            "origin": [float(v) for v in ros_map.origin.tolist()],
            "occupied_ratio": float(np.mean(ros_map.occupied)),
            "unknown_ratio": float(np.mean(ros_map.unknown)),
        },
        "las": {
            "readable_points": int(las_meta.total_points),
            "header_point_count": int(las_meta.header_point_count),
            "tail_bytes": int(las_meta.tail_bytes),
            "has_color": bool(las_meta.has_color),
            "bounds_min_xyz": [float(v) for v in mins.tolist()],
            "bounds_max_xyz": [float(v) for v in maxs.tolist()],
        },
        "preprocessing": {
            "floor_percentile": float(args.floor_percentile),
            "estimated_floor_z": float(floor_z),
            "z_band_min_abs": float(band_min),
            "z_band_max_abs": float(band_max),
            "estimated_band_ratio": float(est_band_ratio),
            "estimated_band_points": int(est_band_points),
            "band_candidates": int(band_candidates),
            "band_kept_after_random_sampling": int(band_kept),
            "voxel_size_m": float(args.voxel_size),
            "downsampled_points": int(down_xyz.shape[0]),
            "optimizer_points": int(opt_xy.shape[0]),
            "keep_probability": float(keep_prob),
        },
        "optimization": {
            "method_selected": method,
            "coarse_cost": float(coarse.fun),
            "fine_cost": float(fine.fun),
            "best_cost": float(best_cost),
            "pose": {
                "tx_m": tx,
                "ty_m": ty,
                "yaw_rad": yaw,
                "yaw_deg": float(np.degrees(yaw)),
            },
            "bounds": {
                "tx": [float(bounds[0][0]), float(bounds[0][1])],
                "ty": [float(bounds[1][0]), float(bounds[1][1])],
                "yaw": [float(bounds[2][0]), float(bounds[2][1])],
            },
            "object_guided_alignment_enabled": bool(object_alignment_enabled),
            "object_guidance_type": str(object_guidance_type),
            "object_align_weight": float(args.object_align_weight),
            "object_align_min_valid_fraction": float(args.object_align_min_valid_fraction),
            "object_align_points_used": int(object_xy_for_opt.shape[0] if object_alignment_enabled else 0),
        },
        "overlay": {
            "enabled": bool(output_overlay),
            "map_overlay_total_points": int(map_overlay_points),
            "red_map_points": int(map_overlay_red_points),
            "red_points": int(map_overlay_red_points),
            "blue_graph_points": int(map_overlay_blue_points),
            "blue_skeleton_points": int(map_overlay_blue_points),
            "green_object_points": int(map_overlay_green_points),
            "map_overlay_mode": str(args.map_overlay_mode),
            "map_overlay_z_placement": str(args.map_overlay_z_placement),
            "map_overlay_z_layers": int(args.map_overlay_z_layers),
            "map_overlay_pixel_stride": int(args.map_overlay_pixel_stride),
            "map_skeleton_min_dist_px": float(args.map_skeleton_min_dist_px),
            "map_skeleton_nms_size": int(args.map_skeleton_nms_size),
            "map_gvd_site_mode": str(args.map_gvd_site_mode),
            "map_gvd_min_clearance_px": float(args.map_gvd_min_clearance_px),
            "map_gvd_min_component_px": int(args.map_gvd_min_component_px),
            "map_overlay_robot_width_m": float(args.map_overlay_robot_width_m),
            "map_overlay_red_thickness_px": int(args.map_overlay_red_thickness_px),
            "map_overlay_blue_thickness_px": int(args.map_overlay_blue_thickness_px),
            "map_keep_main_component": bool(args.map_keep_main_component),
            "object_coordinates_file": str(object_coordinates_path),
            "object_coordinates_frame": str(args.object_coordinates_frame),
            "object_coordinates_count": int(object_xy_raw.shape[0]),
            "object_path_filter_radius_m": float(args.object_path_filter_radius_m),
            "object_path_filter_applied": bool(map_overlay_filter_stats.get("applied", False)),
            "object_points_in_map": int(map_overlay_filter_stats.get("object_points_in_map", 0)),
            "object_points_matched": int(map_overlay_filter_stats.get("object_points_matched", 0)),
            "object_points_overlay": int(map_overlay_filter_stats.get("object_points_overlay", 0)),
            "path_components_total": int(map_overlay_filter_stats.get("path_components_total", 0)),
            "path_components_kept": int(map_overlay_filter_stats.get("path_components_kept", 0)),
            "path_pixels_before_filter": int(map_overlay_filter_stats.get("path_pixels_before", 0)),
            "path_pixels_after_filter": int(map_overlay_filter_stats.get("path_pixels_after", 0)),
        },
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("Alignment complete.")
    print(f"Pose: tx={tx:.4f} m, ty={ty:.4f} m, yaw={math.degrees(yaw):.3f} deg")
    print(f"Cost: {best_cost:.4f}")
    print(f"Aligned band PLY: {output_ply}")
    if output_full is not None:
        print(f"Aligned full PLY: {output_full}")
    if output_overlay is not None:
        print(f"Overlay full+map PLY: {output_overlay}")
    print(f"Report JSON: {output_json}")


if __name__ == "__main__":
    main()
