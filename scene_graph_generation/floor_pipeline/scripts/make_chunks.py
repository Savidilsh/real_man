#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from common import (
    load_room_polygons,
    points_in_polygon_xy,
    read_json,
    save_chunk_npy,
    write_json,
)


class XYBoxCounter:
    def __init__(self, coord: np.ndarray) -> None:
        order = np.argsort(coord[:, 0])
        self.x = np.asarray(coord[:, 0], dtype=np.float32)[order]
        self.y = np.asarray(coord[:, 1], dtype=np.float32)[order]
        self.cache: dict[tuple[float, float, float, float], int] = {}

    def count(self, box: tuple[float, float, float, float]) -> int:
        key = tuple(round(float(v), 4) for v in box)
        cached = self.cache.get(key)
        if cached is not None:
            return cached
        x0, x1, y0, y1 = box
        start = int(np.searchsorted(self.x, x0, side="left"))
        end = int(np.searchsorted(self.x, x1, side="right"))
        y = self.y[start:end]
        count = int(np.count_nonzero((y >= y0) & (y <= y1)))
        self.cache[key] = count
        return count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create room-first Mosaic3D inputs from room polygons."
    )
    parser.add_argument("--preprocess-manifest", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--halo", type=float, default=0.8, help="Overlap/context around split room chunks.")
    parser.add_argument("--max-points", type=int, default=900000, help="Hard point limit per Mosaic3D input.")
    parser.add_argument(
        "--room-full-max-points",
        type=int,
        default=900000,
        help="Run a full room as one Mosaic3D input if the room has this many points or fewer.",
    )
    parser.add_argument("--min-points", type=int, default=1500, help="Skip room chunks with fewer points.")
    parser.add_argument(
        "--min-split-points",
        type=int,
        default=0,
        help=(
            "Preferred minimum point count for chunks created by splitting an oversized room. "
            "Small full-room jobs are not affected."
        ),
    )
    parser.add_argument("--min-tile-size", type=float, default=1.0, help="Smallest split core size in meters.")
    parser.add_argument("--room-polygons", default="", help="Rooms JSON from Point2Graph-style detection.")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def box_mask(coord: np.ndarray, box: tuple[float, float, float, float]) -> np.ndarray:
    x0, x1, y0, y1 = box
    return (
        (coord[:, 0] >= x0)
        & (coord[:, 0] <= x1)
        & (coord[:, 1] >= y0)
        & (coord[:, 1] <= y1)
    )


def center_weights(coord: np.ndarray, core_box: tuple[float, float, float, float]) -> np.ndarray:
    x0, x1, y0, y1 = core_box
    cx = (x0 + x1) * 0.5
    cy = (y0 + y1) * 0.5
    hx = max((x1 - x0) * 0.5, 1e-6)
    hy = max((y1 - y0) * 0.5, 1e-6)
    radius = np.maximum(np.abs(coord[:, 0] - cx) / hx, np.abs(coord[:, 1] - cy) / hy)
    weights = np.full(len(coord), 0.05, dtype=np.float32)
    inside = radius <= 1.0
    weights[inside] = 0.35 + 0.65 * (1.0 - radius[inside])
    return weights


def build_area_specs(coord: np.ndarray, rooms: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rooms:
        return [
            {
                "room_id": "floor",
                "room_type": "floor",
                "point_mask": np.ones(len(coord), dtype=bool),
                "bbox": xy_bbox(coord),
                "polygon_xy": [],
            }
        ]

    specs = []
    xy = coord[:, :2]
    index_cache: dict[str, Any] = {}
    for room in rooms:
        mask = room_point_mask(coord, xy, room, index_cache)
        if not np.any(mask):
            continue
        room_coord = coord[mask]
        specs.append(
            {
                "room_id": room["room_id"],
                "room_type": room["room_type"],
                "point_mask": mask,
                "polygon_xy": room.get("polygon_xy", []),
                "bbox": xy_bbox(room_coord),
            }
        )
    return specs


def xy_bbox(coord: np.ndarray) -> tuple[float, float, float, float]:
    return (
        float(coord[:, 0].min()),
        float(coord[:, 0].max()),
        float(coord[:, 1].min()),
        float(coord[:, 1].max()),
    )


def room_point_mask(
    coord: np.ndarray,
    xy: np.ndarray,
    room: dict[str, Any],
    index_cache: dict[str, Any],
) -> np.ndarray:
    npz_path = str(room.get("point_indices_npz") or "")
    npz_key = str(room.get("point_indices_key") or "")
    if npz_path and npz_key:
        if npz_path not in index_cache:
            index_cache[npz_path] = np.load(npz_path)
        data = index_cache[npz_path]
        if npz_key in data:
            indices = np.asarray(data[npz_key], dtype=np.int64)
            indices = indices[(indices >= 0) & (indices < len(coord))]
            mask = np.zeros(len(coord), dtype=bool)
            mask[indices] = True
            return mask
    return points_in_polygon_xy(xy, room["polygon_xy"])


def split_room_boxes(
    coord: np.ndarray,
    area_coord: np.ndarray,
    core_box: tuple[float, float, float, float],
    halo: float,
    max_points: int,
    min_tile_size: float,
    counter: XYBoxCounter,
) -> list[tuple[float, float, float, float]]:
    core_mask = box_mask(area_coord, core_box)
    if not np.any(core_mask):
        return []

    x0, x1, y0, y1 = core_box
    halo_box = (x0 - halo, x1 + halo, y0 - halo, y1 + halo)
    if counter.count(halo_box) <= int(max_points):
        return [core_box]

    width = x1 - x0
    height = y1 - y0
    if max(width, height) <= float(min_tile_size):
        return [core_box]

    core_points = area_coord[core_mask]
    if width >= height:
        mid = float(np.median(core_points[:, 0]))
        if mid <= x0 + 1e-4 or mid >= x1 - 1e-4:
            mid = (x0 + x1) * 0.5
        children = [(x0, mid, y0, y1), (mid, x1, y0, y1)]
    else:
        mid = float(np.median(core_points[:, 1]))
        if mid <= y0 + 1e-4 or mid >= y1 - 1e-4:
            mid = (y0 + y1) * 0.5
        children = [(x0, x1, y0, mid), (x0, x1, mid, y1)]

    boxes: list[tuple[float, float, float, float]] = []
    for child in children:
        boxes.extend(split_room_boxes(coord, core_points, child, halo, max_points, min_tile_size, counter))
    return boxes


def box_union(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    return (min(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), max(a[3], b[3]))


def box_xy_gap(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> float:
    dx = max(0.0, max(b[0] - a[1], a[0] - b[1]))
    dy = max(0.0, max(b[2] - a[3], a[2] - b[3]))
    return float((dx * dx + dy * dy) ** 0.5)


def halo_point_count(
    counter: XYBoxCounter,
    box: tuple[float, float, float, float],
    halo: float,
) -> int:
    x0, x1, y0, y1 = box
    halo_box = (x0 - halo, x1 + halo, y0 - halo, y1 + halo)
    return counter.count(halo_box)


def expand_box(
    box: tuple[float, float, float, float],
    margin: float,
    bounds: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    return (
        max(bounds[0], box[0] - margin),
        min(bounds[1], box[1] + margin),
        max(bounds[2], box[2] - margin),
        min(bounds[3], box[3] + margin),
    )


def expand_context_box_for_min_points(
    counter: XYBoxCounter,
    context_box: tuple[float, float, float, float],
    min_points: int,
    max_points: int,
    bounds: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    """Grow a split chunk's context so tiny split jobs still have enough points."""

    target_min_points = int(min_points) + 512
    if min_points <= 0 or counter.count(context_box) >= target_min_points:
        return context_box

    max_margin = max(bounds[1] - bounds[0], bounds[3] - bounds[2])
    best_box = context_box
    best_count = counter.count(context_box)
    low = 0.0
    high = 0.25
    high_count = best_count
    while high < max_margin:
        candidate = expand_box(context_box, high, bounds)
        high_count = counter.count(candidate)
        if high_count >= target_min_points:
            break
        if high_count <= max_points and high_count > best_count:
            best_box = candidate
            best_count = high_count
        low = high
        high *= 2.0

    if high_count < target_min_points:
        candidate = expand_box(context_box, max_margin, bounds)
        count = counter.count(candidate)
        return candidate if target_min_points <= count <= max_points else best_box

    for _ in range(28):
        mid = (low + high) * 0.5
        candidate = expand_box(context_box, mid, bounds)
        count = counter.count(candidate)
        if count < target_min_points:
            low = mid
            if count <= max_points and count > best_count:
                best_box = candidate
                best_count = count
            continue
        high = mid
        if count <= max_points:
            best_box = candidate
            best_count = count

    if best_count >= target_min_points:
        return best_box
    return best_box


def merge_small_split_boxes(
    counter: XYBoxCounter,
    boxes: list[tuple[float, float, float, float]],
    halo: float,
    min_points: int,
    max_points: int,
) -> list[tuple[float, float, float, float]]:
    """Merge split boxes where possible so oversized rooms avoid tiny chunks."""

    if min_points <= 0 or not boxes:
        return boxes

    boxes = list(boxes)
    target = (float(min_points) + float(max_points)) * 0.5
    max_iterations = max(1, min(100, len(boxes) * 2))
    iterations = 0
    while True:
        if iterations >= max_iterations:
            remaining = sum(
                1 for box in boxes if halo_point_count(counter, box, halo) < min_points
            )
            if remaining:
                print(
                    f"[chunks] warning: stopped small-box merge after {iterations} iterations; "
                    f"{remaining} split box(es) remain below {min_points:,} points",
                    flush=True,
                )
            break
        counts = [halo_point_count(counter, box, halo) for box in boxes]
        small = [i for i, count in enumerate(counts) if count < min_points]
        if not small:
            break

        best: tuple[tuple[int, float, float, float], int, int, tuple[float, float, float, float]] | None = None
        for i in small:
            for j, other in enumerate(boxes):
                if i == j:
                    continue
                merged = box_union(boxes[i], other)
                merged_count = halo_point_count(counter, merged, halo)
                if merged_count > max_points:
                    continue
                reaches_min = 0 if merged_count >= min_points else 1
                gap = box_xy_gap(boxes[i], other)
                target_delta = abs(float(merged_count) - target)
                score = (reaches_min, gap, target_delta, -float(merged_count))
                if best is None or score < best[0]:
                    best = (score, i, j, merged)

        if best is None:
            break

        _, i, j, merged = best
        keep = [box for index, box in enumerate(boxes) if index not in {i, j}]
        keep.append(merged)
        boxes = keep
        iterations += 1

    return boxes


def largest_room_boxes(
    coord: np.ndarray,
    area_coord: np.ndarray,
    core_box: tuple[float, float, float, float],
    halo: float,
    max_points: int,
    min_split_points: int,
    min_tile_size: float,
    counter: XYBoxCounter,
) -> list[tuple[float, float, float, float]]:
    """Return large overlapping boxes for a room that is too big for one run."""

    boxes = split_room_boxes(coord, area_coord, core_box, halo, max_points, min_tile_size, counter)
    return merge_small_split_boxes(counter, boxes, halo, min_split_points, max_points)


def quantile_split_boxes(
    area_coord: np.ndarray,
    core_box: tuple[float, float, float, float],
    max_points: int,
    min_tile_size: float,
) -> list[tuple[float, float, float, float]]:
    """Fast split for large leftover/common areas.

    Common areas are often sparse and irregular. Splitting by core point count is
    much cheaper than repeatedly testing halo density over the full floor.
    """

    if len(area_coord) == 0:
        return []
    if len(area_coord) <= max_points:
        return [core_box]

    x0, x1, y0, y1 = core_box
    width = x1 - x0
    height = y1 - y0
    if max(width, height) <= float(min_tile_size):
        return [core_box]

    if width >= height:
        axis = 0
        mid = float(np.median(area_coord[:, axis]))
        if mid <= x0 + 1e-4 or mid >= x1 - 1e-4:
            mid = (x0 + x1) * 0.5
        left_mask = area_coord[:, axis] <= mid
        children = [
            (area_coord[left_mask], (x0, mid, y0, y1)),
            (area_coord[~left_mask], (mid, x1, y0, y1)),
        ]
    else:
        axis = 1
        mid = float(np.median(area_coord[:, axis]))
        if mid <= y0 + 1e-4 or mid >= y1 - 1e-4:
            mid = (y0 + y1) * 0.5
        left_mask = area_coord[:, axis] <= mid
        children = [
            (area_coord[left_mask], (x0, x1, y0, mid)),
            (area_coord[~left_mask], (x0, x1, mid, y1)),
        ]

    boxes: list[tuple[float, float, float, float]] = []
    for child_coord, child_box in children:
        boxes.extend(quantile_split_boxes(child_coord, child_box, max_points, min_tile_size))
    return boxes


def choose_limited_indices(
    chunk_idx: np.ndarray,
    core_idx: np.ndarray,
    max_points: int,
    seed: int,
) -> tuple[np.ndarray, bool]:
    if len(chunk_idx) <= max_points:
        return np.sort(chunk_idx), False

    rng = np.random.default_rng(seed)
    in_core = np.isin(chunk_idx, core_idx, assume_unique=False)
    core_in_chunk = chunk_idx[in_core]
    halo_in_chunk = chunk_idx[~in_core]

    if len(core_in_chunk) >= max_points:
        picked = rng.choice(core_in_chunk, size=max_points, replace=False)
        return np.sort(picked.astype(np.int64)), True

    remaining = max_points - len(core_in_chunk)
    if len(halo_in_chunk) > remaining:
        halo_in_chunk = rng.choice(halo_in_chunk, size=remaining, replace=False)
    picked = np.concatenate([core_in_chunk, halo_in_chunk]).astype(np.int64)
    return np.sort(picked), True


def write_chunk(
    chunks_dir: Path,
    name: str,
    coord: np.ndarray,
    color: np.ndarray,
    chunk_idx: np.ndarray,
    core_idx: np.ndarray,
    weights: np.ndarray,
    core_box: tuple[float, float, float, float],
    halo_box: tuple[float, float, float, float],
    area: dict[str, Any],
    chunk_kind: str,
    limited: bool,
) -> dict[str, Any]:
    chunk_idx = chunk_idx.astype(np.int64)
    weights = np.asarray(weights, dtype=np.float32)
    order = np.argsort(chunk_idx)
    chunk_idx = chunk_idx[order]
    weights = weights[order]
    core_idx = np.sort(core_idx.astype(np.int64))
    local_coord = coord[chunk_idx]
    local_color = color[chunk_idx]
    is_core = np.isin(chunk_idx, core_idx, assume_unique=False)
    if len(weights) != len(chunk_idx):
        raise ValueError("weights must have the same length as chunk_idx")

    cloud_path = chunks_dir / f"{name}.npy"
    meta_path = chunks_dir / f"{name}_meta.npz"
    save_chunk_npy(cloud_path, local_coord, local_color)
    np.savez_compressed(
        meta_path,
        global_index=chunk_idx,
        center_weight=weights.astype(np.float32),
        is_core=is_core.astype(bool),
        core_box=np.asarray(core_box, dtype=np.float32),
        halo_box=np.asarray(halo_box, dtype=np.float32),
        room_id=np.asarray(area["room_id"]),
        chunk_kind=np.asarray(chunk_kind),
    )
    return {
        "chunk_id": name,
        "chunk_kind": chunk_kind,
        "room_id": area["room_id"],
        "room_type": area["room_type"],
        "cloud_path": str(cloud_path),
        "meta_path": str(meta_path),
        "point_count": int(len(chunk_idx)),
        "core_point_count": int(np.count_nonzero(is_core)),
        "room_point_count": int(np.count_nonzero(area["point_mask"])),
        "limited_to_max_points": bool(limited),
        "core_box": [float(v) for v in core_box],
        "halo_box": [float(v) for v in halo_box],
    }


def main() -> None:
    args = parse_args()
    work_dir = Path(args.work_dir).expanduser().resolve()
    chunks_dir = work_dir / "02_chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = chunks_dir / "chunks_manifest.json"
    if manifest_path.exists() and not args.force:
        print(f"[chunks] Reusing {manifest_path}")
        return

    preprocess = read_json(Path(args.preprocess_manifest))
    cloud_npz = Path(preprocess["cloud_npz"])
    data = np.load(cloud_npz)
    coord = np.asarray(data["coord"], dtype=np.float32)
    color = np.asarray(data["color"], dtype=np.float32)
    rooms = load_room_polygons(args.room_polygons)
    area_specs = build_area_specs(coord, rooms)
    counter = XYBoxCounter(coord)
    floor_box = xy_bbox(coord)

    entries = []
    chunk_id = 0
    for area_number, area in enumerate(area_specs, start=1):
        area_mask = area["point_mask"]
        room_idx = np.flatnonzero(area_mask).astype(np.int64)
        if len(room_idx) < int(args.min_points):
            print(
                f"[chunks] {area_number}/{len(area_specs)} skip {area['room_id']} "
                f"({len(room_idx):,} points)",
                flush=True,
            )
            continue

        area_coord = coord[room_idx]
        room_box = area["bbox"]
        print(
            f"[chunks] {area_number}/{len(area_specs)} {area['room_id']} "
            f"{len(room_idx):,} points",
            flush=True,
        )
        if len(room_idx) <= int(args.room_full_max_points):
            name = f"chunk_{chunk_id:06d}"
            entries.append(
                write_chunk(
                    chunks_dir=chunks_dir,
                    name=name,
                    coord=coord,
                    color=color,
                    chunk_idx=room_idx,
                    core_idx=room_idx,
                    weights=np.ones(len(room_idx), dtype=np.float32),
                    core_box=room_box,
                    halo_box=room_box,
                    area=area,
                    chunk_kind="room_full",
                    limited=False,
                )
            )
            chunk_id += 1
            continue

        if str(area["room_id"]).startswith("common_area"):
            boxes = quantile_split_boxes(
                area_coord=area_coord,
                core_box=room_box,
                max_points=int(args.max_points),
                min_tile_size=float(args.min_tile_size),
            )
        else:
            boxes = largest_room_boxes(
                coord=coord,
                area_coord=area_coord,
                core_box=room_box,
                halo=float(args.halo),
                max_points=int(args.max_points),
                min_split_points=int(args.min_split_points),
                min_tile_size=float(args.min_tile_size),
                counter=counter,
            )
        print(
            f"[chunks] {area['room_id']} split into {len(boxes):,} candidate box(es)",
            flush=True,
        )
        for core_box in boxes:
            x0, x1, y0, y1 = core_box
            halo_box = (x0 - args.halo, x1 + args.halo, y0 - args.halo, y1 + args.halo)
            halo_box = expand_context_box_for_min_points(
                counter=counter,
                context_box=halo_box,
                min_points=int(args.min_split_points),
                max_points=int(args.max_points),
                bounds=floor_box,
            )
            core_idx = np.flatnonzero(area_mask & box_mask(coord, core_box)).astype(np.int64)
            chunk_idx = np.flatnonzero(box_mask(coord, halo_box)).astype(np.int64)
            if len(chunk_idx) < int(args.min_points) or len(core_idx) == 0:
                continue
            chunk_idx, limited = choose_limited_indices(
                chunk_idx, core_idx, int(args.max_points), seed=chunk_id + 17
            )
            local_coord = coord[chunk_idx]
            is_core = box_mask(local_coord, core_box) & area_mask[chunk_idx]
            weights = center_weights(local_coord, core_box)
            weights[~is_core] = np.minimum(weights[~is_core], 0.05)

            name = f"chunk_{chunk_id:06d}"
            entries.append(
                write_chunk(
                    chunks_dir=chunks_dir,
                    name=name,
                    coord=coord,
                    color=color,
                    chunk_idx=chunk_idx,
                    core_idx=core_idx,
                    weights=weights,
                    core_box=core_box,
                    halo_box=halo_box,
                    area=area,
                    chunk_kind="room_split",
                    limited=limited,
                )
            )
            chunk_id += 1
        print(f"[chunks] total written so far: {chunk_id:,}", flush=True)

    manifest = {
        "preprocess_manifest": str(Path(args.preprocess_manifest).resolve()),
        "cloud_npz": str(cloud_npz),
        "chunks_dir": str(chunks_dir),
        "chunk_count": len(entries),
        "strategy": "room_full_else_largest_overlapping_splits",
        "halo": float(args.halo),
        "max_points": int(args.max_points),
        "room_full_max_points": int(args.room_full_max_points),
        "min_points": int(args.min_points),
        "min_split_points": int(args.min_split_points),
        "room_polygons": str(Path(args.room_polygons).resolve()) if args.room_polygons else "",
        "chunks": entries,
    }
    write_json(manifest_path, manifest)
    print(f"[chunks] wrote {len(entries):,} Mosaic3D input(s) to {chunks_dir}")
    print(f"[chunks] full-room jobs={sum(1 for item in entries if item['chunk_kind'] == 'room_full'):,}")
    print(f"[chunks] split-room jobs={sum(1 for item in entries if item['chunk_kind'] == 'room_split'):,}")
    print(f"[chunks] manifest={manifest_path}")


if __name__ == "__main__":
    main()
