#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np

from common import palette, read_json, write_json, write_point_cloud


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Point2Graph-style room segmentation from a preprocessed point cloud. "
            "This implements the paper's room-layer idea because the public "
            "Point2Graph repo releases object detection/classification code, not "
            "a callable room detector."
        )
    )
    parser.add_argument("--preprocess-manifest", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--grid-size", type=float, default=0.10)
    parser.add_argument("--slice-count", type=int, default=16)
    parser.add_argument("--z-min-quantile", type=float, default=0.08)
    parser.add_argument("--z-max-quantile", type=float, default=0.92)
    parser.add_argument("--wall-min-height", type=float, default=0.35)
    parser.add_argument("--wall-max-height", type=float, default=2.30)
    parser.add_argument("--min-component-area", type=float, default=3.0)
    parser.add_argument("--wall-persistence", type=float, default=0.28)
    parser.add_argument("--footprint-dilate-cells", type=int, default=3)
    parser.add_argument("--wall-dilate-cells", type=int, default=2)
    parser.add_argument("--free-close-cells", type=int, default=1)
    parser.add_argument("--assignment-dilate-cells", type=int, default=3)
    parser.add_argument("--polygon-rays", type=int, default=72)
    parser.add_argument("--max-rooms", type=int, default=200)
    parser.add_argument("--preview-ply", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def point_to_grid(coord: np.ndarray, origin_xy: np.ndarray, grid_size: float, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    col = np.floor((coord[:, 0] - origin_xy[0]) / grid_size).astype(np.int64)
    row = np.floor((coord[:, 1] - origin_xy[1]) / grid_size).astype(np.int64)
    row = np.clip(row, 0, shape[0] - 1)
    col = np.clip(col, 0, shape[1] - 1)
    return row, col


def occupancy_from_points(
    coord: np.ndarray,
    origin_xy: np.ndarray,
    grid_size: float,
    shape: tuple[int, int],
    min_count: int = 1,
) -> np.ndarray:
    row, col = point_to_grid(coord, origin_xy, grid_size, shape)
    grid = np.zeros(shape, dtype=np.int32)
    np.add.at(grid, (row, col), 1)
    return grid >= int(min_count)


def dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    radius = int(radius)
    if radius <= 0:
        return mask.copy()
    padded = np.pad(mask, radius, mode="constant", constant_values=False)
    out = np.zeros_like(mask, dtype=bool)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            out |= padded[
                radius + dy : radius + dy + mask.shape[0],
                radius + dx : radius + dx + mask.shape[1],
            ]
    return out


def erode(mask: np.ndarray, radius: int) -> np.ndarray:
    radius = int(radius)
    if radius <= 0:
        return mask.copy()
    padded = np.pad(mask, radius, mode="constant", constant_values=True)
    out = np.ones_like(mask, dtype=bool)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            out &= padded[
                radius + dy : radius + dy + mask.shape[0],
                radius + dx : radius + dx + mask.shape[1],
            ]
    return out


def close(mask: np.ndarray, radius: int) -> np.ndarray:
    return erode(dilate(mask, radius), radius)


def fill_holes(mask: np.ndarray) -> np.ndarray:
    outside = np.zeros_like(mask, dtype=bool)
    queue: deque[tuple[int, int]] = deque()
    h, w = mask.shape
    for x in range(w):
        if not mask[0, x]:
            outside[0, x] = True
            queue.append((0, x))
        if not mask[h - 1, x]:
            outside[h - 1, x] = True
            queue.append((h - 1, x))
    for y in range(h):
        if not mask[y, 0]:
            outside[y, 0] = True
            queue.append((y, 0))
        if not mask[y, w - 1]:
            outside[y, w - 1] = True
            queue.append((y, w - 1))

    while queue:
        y, x = queue.popleft()
        for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            if ny < 0 or nx < 0 or ny >= h or nx >= w or outside[ny, nx] or mask[ny, nx]:
                continue
            outside[ny, nx] = True
            queue.append((ny, nx))
    return mask | ~outside


def connected_components(mask: np.ndarray) -> tuple[np.ndarray, list[tuple[int, int]]]:
    labels = np.full(mask.shape, -1, dtype=np.int32)
    components: list[tuple[int, int]] = []
    h, w = mask.shape
    component_id = 0
    for y in range(h):
        for x in range(w):
            if not mask[y, x] or labels[y, x] >= 0:
                continue
            labels[y, x] = component_id
            queue = deque([(y, x)])
            count = 0
            while queue:
                cy, cx = queue.popleft()
                count += 1
                for ny, nx in (
                    (cy - 1, cx),
                    (cy + 1, cx),
                    (cy, cx - 1),
                    (cy, cx + 1),
                    (cy - 1, cx - 1),
                    (cy - 1, cx + 1),
                    (cy + 1, cx - 1),
                    (cy + 1, cx + 1),
                ):
                    if ny < 0 or nx < 0 or ny >= h or nx >= w:
                        continue
                    if not mask[ny, nx] or labels[ny, nx] >= 0:
                        continue
                    labels[ny, nx] = component_id
                    queue.append((ny, nx))
            components.append((component_id, count))
            component_id += 1
    return labels, components


def raster_polygon(mask: np.ndarray, origin_xy: np.ndarray, grid_size: float, rays: int) -> list[list[float]]:
    rows, cols = np.where(mask)
    if len(rows) == 0:
        return []
    xy = np.column_stack(
        (
            origin_xy[0] + (cols.astype(np.float32) + 0.5) * grid_size,
            origin_xy[1] + (rows.astype(np.float32) + 0.5) * grid_size,
        )
    )
    if len(xy) < 3:
        return rectangle_polygon(xy)
    center = np.mean(xy, axis=0)
    vec = xy - center
    angle = np.arctan2(vec[:, 1], vec[:, 0])
    radius = np.linalg.norm(vec, axis=1)
    bins = np.floor(((angle + np.pi) / (2.0 * np.pi)) * int(rays)).astype(np.int32)
    bins = np.clip(bins, 0, int(rays) - 1)

    picked = []
    for b in range(int(rays)):
        idx = np.flatnonzero(bins == b)
        if len(idx) == 0:
            continue
        farthest = idx[int(np.argmax(radius[idx]))]
        picked.append(xy[farthest])
    if len(picked) < 3:
        return rectangle_polygon(xy)
    return [[float(x), float(y)] for x, y in picked]


def rectangle_polygon(xy: np.ndarray) -> list[list[float]]:
    if len(xy) == 0:
        return []
    x0, y0 = np.min(xy, axis=0)
    x1, y1 = np.max(xy, axis=0)
    return [[float(x0), float(y0)], [float(x1), float(y0)], [float(x1), float(y1)], [float(x0), float(y1)]]


def build_border_map(coord: np.ndarray, origin_xy: np.ndarray, shape: tuple[int, int], args: argparse.Namespace) -> tuple[np.ndarray, dict[str, Any]]:
    z = coord[:, 2]
    floor_z = float(np.quantile(z, 0.02))
    z_min = floor_z + float(args.wall_min_height)
    z_max = floor_z + float(args.wall_max_height)
    explicit_wall_band_count = int(np.count_nonzero((z >= z_min) & (z <= z_max)))
    if explicit_wall_band_count < 500:
        z_min = float(np.quantile(z, args.z_min_quantile))
        z_max = float(np.quantile(z, args.z_max_quantile))
    if z_max <= z_min:
        z_min = float(z.min())
        z_max = float(z.max())
    edges = np.linspace(z_min, z_max, int(args.slice_count) + 1, dtype=np.float32)
    slice_hits = np.zeros(shape, dtype=np.float32)
    valid_slices = 0
    slice_stats = []
    for i in range(len(edges) - 1):
        lo, hi = float(edges[i]), float(edges[i + 1])
        in_slice = (coord[:, 2] >= lo) & (coord[:, 2] < hi)
        if int(np.count_nonzero(in_slice)) < 100:
            continue
        occ = occupancy_from_points(coord[in_slice], origin_xy, args.grid_size, shape, min_count=1)
        ratio = float(np.count_nonzero(occ) / occ.size)
        if ratio < 0.0005 or ratio > 0.75:
            continue
        slice_hits += occ.astype(np.float32)
        valid_slices += 1
        slice_stats.append({"z_min": lo, "z_max": hi, "occupied_ratio": ratio})

    if valid_slices == 0:
        persistence = np.zeros(shape, dtype=np.float32)
    else:
        persistence = slice_hits / float(valid_slices)

    height = z - floor_z
    low_band = coord[(height >= 0.15) & (height <= 0.75)]
    upper_band = coord[(height >= max(1.10, float(args.wall_min_height))) & (height <= float(args.wall_max_height))]
    low_occ = occupancy_from_points(low_band, origin_xy, args.grid_size, shape, min_count=1) if len(low_band) else np.zeros(shape, dtype=bool)
    upper_occ = occupancy_from_points(upper_band, origin_xy, args.grid_size, shape, min_count=1) if len(upper_band) else np.zeros(shape, dtype=bool)
    object_like = low_occ & ~dilate(upper_occ, 1)

    border = persistence >= float(args.wall_persistence)
    border = border & ~((persistence < float(args.wall_persistence) + 0.10) & object_like)
    border = close(dilate(border, int(args.wall_dilate_cells)), 1)
    debug = {
        "floor_z": floor_z,
        "z_min": z_min,
        "z_max": z_max,
        "wall_min_height": float(args.wall_min_height),
        "wall_max_height": float(args.wall_max_height),
        "explicit_wall_band_count": explicit_wall_band_count,
        "valid_slices": valid_slices,
        "slice_stats": slice_stats,
        "wall_persistence": float(args.wall_persistence),
        "object_like_cells": int(np.count_nonzero(object_like)),
    }
    debug_maps = {
        "wall_persistence": persistence.astype(np.float32),
        "wall_border": border.astype(np.uint8),
        "object_like": object_like.astype(np.uint8),
    }
    return border, {**debug, "maps": debug_maps}


def assign_point_indices(
    coord: np.ndarray,
    room_labels: np.ndarray,
    component_id: int,
    origin_xy: np.ndarray,
    grid_size: float,
) -> np.ndarray:
    row, col = point_to_grid(coord, origin_xy, grid_size, room_labels.shape)
    return np.flatnonzero(room_labels[row, col] == int(component_id)).astype(np.int64)


def write_preview(path: Path, coord: np.ndarray, room_entries: list[dict[str, Any]], indices_payload: dict[str, np.ndarray]) -> None:
    colors = np.full((len(coord), 3), 145, dtype=np.uint8)
    pal = palette(max(len(room_entries), 1))
    for i, room in enumerate(room_entries):
        key = room["point_indices_key"]
        colors[indices_payload[key]] = pal[i]
    write_point_cloud(path, coord, colors)


def main() -> None:
    args = parse_args()
    work_dir = Path(args.work_dir).expanduser().resolve()
    rooms_dir = work_dir / "02_rooms"
    rooms_dir.mkdir(parents=True, exist_ok=True)
    rooms_json = rooms_dir / "rooms.json"
    if rooms_json.exists() and not args.force:
        print(f"[rooms] Reusing {rooms_json}")
        return

    preprocess = read_json(Path(args.preprocess_manifest))
    cloud = np.load(Path(preprocess["cloud_npz"]), allow_pickle=False)
    coord = np.asarray(cloud["coord"], dtype=np.float32)
    if len(coord) == 0:
        raise ValueError("Preprocessed point cloud is empty.")

    xy_min = coord[:, :2].min(axis=0) - float(args.grid_size) * 4.0
    xy_max = coord[:, :2].max(axis=0) + float(args.grid_size) * 4.0
    shape = (
        int(np.ceil((xy_max[1] - xy_min[1]) / float(args.grid_size))) + 1,
        int(np.ceil((xy_max[0] - xy_min[0]) / float(args.grid_size))) + 1,
    )

    density_occ = occupancy_from_points(coord, xy_min, args.grid_size, shape, min_count=1)
    footprint = fill_holes(dilate(density_occ, int(args.footprint_dilate_cells)))
    border, debug = build_border_map(coord, xy_min, shape, args)
    debug_maps = debug.pop("maps", {})
    free = footprint & ~border
    free = close(free, int(args.free_close_cells))
    labels, components = connected_components(free)

    min_cells = max(1, int(float(args.min_component_area) / (float(args.grid_size) ** 2)))
    kept = sorted(
        ((cid, count) for cid, count in components if count >= min_cells),
        key=lambda item: item[1],
        reverse=True,
    )[: int(args.max_rooms)]

    room_label_map = np.full(labels.shape, -1, dtype=np.int32)
    rooms = []
    indices_payload: dict[str, np.ndarray] = {}
    for room_number, (component_id, cell_count) in enumerate(kept, start=1):
        component_mask = labels == component_id
        assignment_mask = dilate(component_mask, int(args.assignment_dilate_cells)) & footprint
        room_id = f"room_{room_number:03d}"
        key = f"{room_id}_indices"
        point_indices = assign_point_indices(coord, assignment_mask.astype(np.int32), 1, xy_min, args.grid_size)
        if len(point_indices) == 0:
            continue
        indices_payload[key] = point_indices
        room_label_map[assignment_mask] = room_number - 1
        polygon = raster_polygon(component_mask, xy_min, float(args.grid_size), int(args.polygon_rays))
        rows, cols = np.where(component_mask)
        bbox_min = [
            float(xy_min[0] + cols.min() * float(args.grid_size)),
            float(xy_min[1] + rows.min() * float(args.grid_size)),
        ]
        bbox_max = [
            float(xy_min[0] + (cols.max() + 1) * float(args.grid_size)),
            float(xy_min[1] + (rows.max() + 1) * float(args.grid_size)),
        ]
        rooms.append(
            {
                "room_id": room_id,
                "room_type": "room",
                "polygon_xy": polygon,
                "bbox_min_xy": bbox_min,
                "bbox_max_xy": bbox_max,
                "area_m2": float(cell_count * float(args.grid_size) ** 2),
                "point_count": int(len(point_indices)),
                "point_indices_npz": "room_point_indices.npz",
                "point_indices_key": key,
                "method": "point2graph_style_border_enhanced_density",
            }
        )

    if not rooms:
        print("[rooms] No room regions found; writing a single floor fallback room.")
        room_id = "room_001"
        key = f"{room_id}_indices"
        indices_payload[key] = np.arange(len(coord), dtype=np.int64)
        rooms = [
            {
                "room_id": room_id,
                "room_type": "floor",
                "polygon_xy": rectangle_polygon(coord[:, :2]),
                "bbox_min_xy": [float(coord[:, 0].min()), float(coord[:, 1].min())],
                "bbox_max_xy": [float(coord[:, 0].max()), float(coord[:, 1].max())],
                "area_m2": float((coord[:, 0].max() - coord[:, 0].min()) * (coord[:, 1].max() - coord[:, 1].min())),
                "point_count": int(len(coord)),
                "point_indices_npz": "room_point_indices.npz",
                "point_indices_key": key,
                "method": "fallback_single_floor",
            }
        ]

    indices_npz = rooms_dir / "room_point_indices.npz"
    np.savez_compressed(indices_npz, **indices_payload)
    if debug_maps:
        np.savez_compressed(rooms_dir / "room_debug_maps.npz", **debug_maps)
    preview_ply = ""
    if args.preview_ply:
        preview_path = rooms_dir / "rooms_preview.ply"
        write_preview(preview_path, coord, rooms, indices_payload)
        preview_ply = str(preview_path)

    payload = {
        "source": "Point2Graph-style room layer",
        "note": (
            "The public zimingluo/Point2Graph repository was inspected and contains "
            "the object detection/classification module, not a callable room detector. "
            "This stage implements the paper/project room-segmentation idea: Z slicing, "
            "XY grid projection, border-enhanced density map, and connected room regions."
        ),
        "preprocess_manifest": str(Path(args.preprocess_manifest).resolve()),
        "point_indices_npz": str(indices_npz),
        "preview_ply": preview_ply,
        "grid": {
            "grid_size": float(args.grid_size),
            "origin_xy": [float(xy_min[0]), float(xy_min[1])],
            "shape": [int(shape[0]), int(shape[1])],
        },
        "debug": {
            **debug,
            "footprint_cells": int(np.count_nonzero(footprint)),
            "border_cells": int(np.count_nonzero(border)),
            "free_cells": int(np.count_nonzero(free)),
            "component_count": len(components),
            "kept_component_count": len(rooms),
            "min_component_area": float(args.min_component_area),
        },
        "rooms": rooms,
    }
    write_json(rooms_json, payload)
    np.save(rooms_dir / "room_label_map.npy", room_label_map)
    print(f"[rooms] detected {len(rooms)} room region(s)")
    print(f"[rooms] wrote {rooms_json}")


if __name__ == "__main__":
    main()
