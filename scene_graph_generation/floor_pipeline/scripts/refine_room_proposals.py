#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from common import palette, read_json, write_json, write_point_cloud
from detect_rooms_point2graph import (
    assign_point_indices,
    build_border_map,
    close,
    connected_components,
    dilate,
    fill_holes,
    occupancy_from_points,
    point_to_grid,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Refine Point2Graph-style room proposals with wall-stable geometry: "
            "cleanup, wall-based split, weak-boundary merge, polygon simplification, "
            "and wall-line snapping."
        )
    )
    parser.add_argument("--preprocess-manifest", required=True)
    parser.add_argument("--rooms-json", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--grid-size", type=float, default=0.0, help="0 means reuse the proposal grid size.")
    parser.add_argument("--slice-count", type=int, default=8)
    parser.add_argument("--wall-min-height", type=float, default=0.35)
    parser.add_argument("--wall-max-height", type=float, default=2.30)
    parser.add_argument("--wall-persistence", type=float, default=0.32)
    parser.add_argument("--wall-dilate-cells", type=int, default=1)
    parser.add_argument("--cleanup-close-cells", type=int, default=1)
    parser.add_argument("--split-wall-dilate-cells", type=int, default=1)
    parser.add_argument("--split-assignment-dilate-cells", type=int, default=2)
    parser.add_argument("--split-min-area", type=float, default=8.0)
    parser.add_argument("--merge-gap-cells", type=int, default=2)
    parser.add_argument("--merge-min-open-boundary", type=float, default=0.75)
    parser.add_argument("--merge-max-wall-ratio", type=float, default=0.18)
    parser.add_argument("--min-room-area", type=float, default=3.0)
    parser.add_argument("--assignment-dilate-cells", type=int, default=2)
    parser.add_argument("--snap-distance", type=float, default=0.30)
    parser.add_argument("--snap-min-support", type=float, default=1.0)
    parser.add_argument("--simplify-epsilon", type=float, default=0.20)
    parser.add_argument(
        "--assign-all-points",
        action="store_true",
        help=(
            "After confident room masks are refined, assign every inference point "
            "to exactly one room/corridor/common-area owner. This keeps detection "
            "masks separate from final point ownership."
        ),
    )
    parser.add_argument(
        "--leftover-component-min-area",
        type=float,
        default=8.0,
        help="Minimum m^2 for an unowned free-space component to become a corridor/common area.",
    )
    parser.add_argument(
        "--leftover-corridor-ratio",
        type=float,
        default=2.5,
        help="Elongation ratio used to name large leftover components as corridors.",
    )
    parser.add_argument(
        "--leftover-corridor-max-width",
        type=float,
        default=3.0,
        help="Maximum short-side width in meters for a leftover component to be named corridor.",
    )
    parser.add_argument("--max-rooms", type=int, default=250)
    parser.add_argument("--preview-ply", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def polygon_area(poly: np.ndarray) -> float:
    if len(poly) < 3:
        return 0.0
    x = poly[:, 0]
    y = poly[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) * 0.5)


def grid_shape_from_coord(coord: np.ndarray, origin_xy: np.ndarray, grid_size: float) -> tuple[int, int]:
    xy_max = coord[:, :2].max(axis=0) + grid_size * 4.0
    return (
        int(np.ceil((xy_max[1] - origin_xy[1]) / grid_size)) + 1,
        int(np.ceil((xy_max[0] - origin_xy[0]) / grid_size)) + 1,
    )


def polygon_to_mask(
    polygon: list[list[float]] | np.ndarray,
    origin_xy: np.ndarray,
    grid_size: float,
    shape: tuple[int, int],
) -> np.ndarray:
    poly = np.asarray(polygon, dtype=np.float64)
    if len(poly) < 3:
        return np.zeros(shape, dtype=bool)
    cols = np.rint((poly[:, 0] - origin_xy[0]) / grid_size).astype(np.int32)
    rows = np.rint((poly[:, 1] - origin_xy[1]) / grid_size).astype(np.int32)
    pts = np.column_stack((cols, rows)).reshape(-1, 1, 2)
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 1)
    return mask.astype(bool)


def mask_to_polygon(
    mask: np.ndarray,
    origin_xy: np.ndarray,
    grid_size: float,
    simplify_epsilon_m: float,
) -> list[list[float]]:
    mask_u8 = mask.astype(np.uint8)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    contour = max(contours, key=cv2.contourArea)
    epsilon_cells = max(1.0, float(simplify_epsilon_m) / float(grid_size))
    approx = cv2.approxPolyDP(contour, epsilon_cells, True)
    pts = approx.reshape(-1, 2)
    if len(pts) < 3:
        rows, cols = np.where(mask)
        if len(rows) == 0:
            return []
        pts = np.asarray(
            [
                [cols.min(), rows.min()],
                [cols.max() + 1, rows.min()],
                [cols.max() + 1, rows.max() + 1],
                [cols.min(), rows.max() + 1],
            ],
            dtype=np.float32,
        )
    poly = np.column_stack(
        (
            origin_xy[0] + pts[:, 0].astype(np.float64) * grid_size,
            origin_xy[1] + pts[:, 1].astype(np.float64) * grid_size,
        )
    )
    return remove_duplicate_vertices(poly).tolist()


def remove_duplicate_vertices(poly: np.ndarray) -> np.ndarray:
    if len(poly) <= 1:
        return poly
    kept = [poly[0]]
    for point in poly[1:]:
        if np.linalg.norm(point - kept[-1]) > 1e-6:
            kept.append(point)
    if len(kept) > 2 and np.linalg.norm(kept[0] - kept[-1]) <= 1e-6:
        kept.pop()
    return np.asarray(kept, dtype=np.float64)


def cleanup_mask(mask: np.ndarray, close_cells: int) -> np.ndarray:
    cleaned = fill_holes(close(mask, int(close_cells)))
    labels, comps = connected_components(cleaned)
    if not comps:
        return cleaned
    keep = np.zeros_like(cleaned, dtype=bool)
    for cid, count in comps:
        if count >= 4:
            keep |= labels == cid
    return keep


def load_room_masks(
    rooms_payload: dict[str, Any],
    rooms_json: Path,
    coord: np.ndarray,
    origin_xy: np.ndarray,
    grid_size: float,
    shape: tuple[int, int],
) -> list[dict[str, Any]]:
    rooms = rooms_payload.get("rooms", rooms_payload if isinstance(rooms_payload, list) else [])
    xy = coord[:, :2]
    masks = []
    index_cache: dict[str, Any] = {}
    for i, room in enumerate(rooms):
        room_id = str(room.get("room_id") or room.get("id") or f"room_{i + 1:03d}")
        room_type = str(room.get("room_type") or room.get("type") or "room")
        mask = np.zeros(shape, dtype=bool)

        npz_text = str(room.get("point_indices_npz") or "").strip()
        npz_key = str(room.get("point_indices_key") or "").strip()
        if npz_text and npz_key:
            npz_path = Path(npz_text).expanduser()
            if not npz_path.is_absolute():
                npz_path = rooms_json.parent / npz_path
            if npz_path.exists():
                if str(npz_path) not in index_cache:
                    index_cache[str(npz_path)] = np.load(npz_path)
                data = index_cache[str(npz_path)]
                if npz_key in data:
                    indices = np.asarray(data[npz_key], dtype=np.int64)
                    indices = indices[(indices >= 0) & (indices < len(coord))]
                    if len(indices):
                        row, col = point_to_grid(coord[indices], origin_xy, grid_size, shape)
                        mask[row, col] = True

        if not np.any(mask):
            polygon = room.get("polygon_xy") or room.get("polygon") or []
            mask = polygon_to_mask(polygon, origin_xy, grid_size, shape)

        if np.any(mask):
            masks.append(
                {
                    "source_room_ids": [room_id],
                    "room_type": room_type,
                    "mask": mask,
                    "original_point_count": int(room.get("point_count") or 0),
                }
            )
    return masks


def split_room_mask(room: dict[str, Any], wall_map: np.ndarray, args: argparse.Namespace, grid_size: float) -> list[dict[str, Any]]:
    mask = room["mask"]
    min_cells = max(1, int(float(args.split_min_area) / (grid_size**2)))
    blocking = dilate(wall_map, int(args.split_wall_dilate_cells))
    free = close(mask & ~blocking, 1)
    labels, comps = connected_components(free)
    kept = sorted((item for item in comps if item[1] >= min_cells), key=lambda item: item[1], reverse=True)
    if len(kept) <= 1:
        return [room]

    assigned = np.zeros_like(mask, dtype=bool)
    out = []
    for cid, _count in kept:
        comp = labels == cid
        repaired = dilate(comp, int(args.split_assignment_dilate_cells)) & mask & ~assigned
        repaired = cleanup_mask(repaired, int(args.cleanup_close_cells))
        if int(np.count_nonzero(repaired)) < min_cells:
            continue
        assigned |= repaired
        out.append(
            {
                "source_room_ids": list(room["source_room_ids"]),
                "room_type": room["room_type"],
                "mask": repaired,
                "split_from": list(room["source_room_ids"]),
            }
        )
    return out if out else [room]


def merge_rooms(
    rooms: list[dict[str, Any]],
    wall_map: np.ndarray,
    args: argparse.Namespace,
    grid_size: float,
) -> list[dict[str, Any]]:
    n = len(rooms)
    if n <= 1:
        return rooms
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    min_open_cells = max(1, int(float(args.merge_min_open_boundary) / max(grid_size, 1e-6)))
    gap = int(args.merge_gap_cells)
    for i in range(n):
        mask_i = rooms[i]["mask"]
        for j in range(i + 1, n):
            mask_j = rooms[j]["mask"]
            adjacency = (dilate(mask_i, gap) & mask_j) | (mask_i & dilate(mask_j, gap))
            open_cells = int(np.count_nonzero(adjacency))
            if open_cells < min_open_cells:
                continue
            wall_ratio = float(np.count_nonzero(wall_map & adjacency) / max(open_cells, 1))
            if wall_ratio <= float(args.merge_max_wall_ratio):
                union(i, j)

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)

    merged = []
    for members in groups.values():
        mask = np.zeros_like(rooms[members[0]]["mask"], dtype=bool)
        source_ids: list[str] = []
        for idx in members:
            mask |= rooms[idx]["mask"]
            source_ids.extend(rooms[idx]["source_room_ids"])
        merged.append(
            {
                "source_room_ids": sorted(set(source_ids)),
                "room_type": rooms[members[0]]["room_type"],
                "mask": cleanup_mask(mask, int(args.cleanup_close_cells)),
                "merged_from": sorted(set(source_ids)) if len(members) > 1 else [],
            }
        )
    return merged


def snap_polygon_to_walls(
    polygon: list[list[float]],
    wall_map: np.ndarray,
    origin_xy: np.ndarray,
    grid_size: float,
    snap_distance_m: float,
    min_support_m: float,
) -> list[list[float]]:
    poly = np.asarray(polygon, dtype=np.float64)
    if len(poly) < 3:
        return polygon
    snap_cells = max(0, int(round(float(snap_distance_m) / float(grid_size))))
    if snap_cells <= 0:
        return polygon
    min_support = max(2, int(round(float(min_support_m) / float(grid_size))))
    h, w = wall_map.shape
    adjusted = poly.copy()

    def best_col(x_world: float, y0: float, y1: float) -> float | None:
        col0 = int(round((x_world - origin_xy[0]) / grid_size))
        r0 = int(np.clip(np.floor((min(y0, y1) - origin_xy[1]) / grid_size), 0, h - 1))
        r1 = int(np.clip(np.ceil((max(y0, y1) - origin_xy[1]) / grid_size), 0, h - 1))
        c0 = max(0, col0 - snap_cells)
        c1 = min(w - 1, col0 + snap_cells)
        if r1 < r0 or c1 < c0:
            return None
        support = wall_map[r0 : r1 + 1, c0 : c1 + 1].sum(axis=0)
        if support.size == 0 or int(support.max()) < min_support:
            return None
        return float(origin_xy[0] + (c0 + int(np.argmax(support))) * grid_size)

    def best_row(y_world: float, x0: float, x1: float) -> float | None:
        row0 = int(round((y_world - origin_xy[1]) / grid_size))
        c0 = int(np.clip(np.floor((min(x0, x1) - origin_xy[0]) / grid_size), 0, w - 1))
        c1 = int(np.clip(np.ceil((max(x0, x1) - origin_xy[0]) / grid_size), 0, w - 1))
        r0 = max(0, row0 - snap_cells)
        r1 = min(h - 1, row0 + snap_cells)
        if r1 < r0 or c1 < c0:
            return None
        support = wall_map[r0 : r1 + 1, c0 : c1 + 1].sum(axis=1)
        if support.size == 0 or int(support.max()) < min_support:
            return None
        return float(origin_xy[1] + (r0 + int(np.argmax(support))) * grid_size)

    for i in range(len(poly)):
        j = (i + 1) % len(poly)
        p0 = adjusted[i]
        p1 = adjusted[j]
        dx = abs(float(p1[0] - p0[0]))
        dy = abs(float(p1[1] - p0[1]))
        if dy > 0 and dx / dy < 0.35:
            snapped = best_col(float((p0[0] + p1[0]) * 0.5), float(p0[1]), float(p1[1]))
            if snapped is not None:
                adjusted[i, 0] = snapped
                adjusted[j, 0] = snapped
        elif dx > 0 and dy / dx < 0.35:
            snapped = best_row(float((p0[1] + p1[1]) * 0.5), float(p0[0]), float(p1[0]))
            if snapped is not None:
                adjusted[i, 1] = snapped
                adjusted[j, 1] = snapped

    adjusted = remove_duplicate_vertices(adjusted)
    if len(adjusted) < 3 or polygon_area(adjusted) <= 0.5:
        return polygon
    return adjusted.tolist()


def write_preview(path: Path, coord: np.ndarray, rooms: list[dict[str, Any]], indices_payload: dict[str, np.ndarray]) -> None:
    colors = np.full((len(coord), 3), 145, dtype=np.uint8)
    pal = palette(max(len(rooms), 1))
    for i, room in enumerate(rooms):
        colors[indices_payload[room["point_indices_key"]]] = pal[i]
    write_point_cloud(path, coord, colors)


def mask_bbox_xy(mask: np.ndarray, origin_xy: np.ndarray, grid_size: float) -> tuple[list[float], list[float]]:
    rows, cols = np.where(mask)
    if len(rows) == 0:
        return [float(origin_xy[0]), float(origin_xy[1])], [float(origin_xy[0]), float(origin_xy[1])]
    return (
        [
            float(origin_xy[0] + cols.min() * grid_size),
            float(origin_xy[1] + rows.min() * grid_size),
        ],
        [
            float(origin_xy[0] + (cols.max() + 1) * grid_size),
            float(origin_xy[1] + (rows.max() + 1) * grid_size),
        ],
    )


def classify_leftover_area(mask: np.ndarray, origin_xy: np.ndarray, grid_size: float, args: argparse.Namespace) -> str:
    bbox_min, bbox_max = mask_bbox_xy(mask, origin_xy, grid_size)
    width = max(float(bbox_max[0] - bbox_min[0]), 1e-6)
    height = max(float(bbox_max[1] - bbox_min[1]), 1e-6)
    short_side = min(width, height)
    ratio = max(width, height) / short_side
    if ratio >= float(args.leftover_corridor_ratio) and short_side <= float(args.leftover_corridor_max_width):
        return "corridor"
    return "common_area"


def nearest_area_indices(coord_xy: np.ndarray, rooms: list[dict[str, Any]]) -> np.ndarray:
    bbox_min = np.asarray([room["bbox_min_xy"] for room in rooms], dtype=np.float32)
    bbox_max = np.asarray([room["bbox_max_xy"] for room in rooms], dtype=np.float32)
    centroids = (bbox_min + bbox_max) * 0.5
    out = np.zeros(len(coord_xy), dtype=np.int32)
    chunk_size = 200_000

    for start in range(0, len(coord_xy), chunk_size):
        stop = min(start + chunk_size, len(coord_xy))
        xy = coord_xy[start:stop].astype(np.float32)
        x = xy[:, 0]
        y = xy[:, 1]
        best_score = np.full(len(xy), np.inf, dtype=np.float32)
        best_idx = np.zeros(len(xy), dtype=np.int32)
        for room_idx in range(len(rooms)):
            dx = np.maximum(np.maximum(bbox_min[room_idx, 0] - x, x - bbox_max[room_idx, 0]), 0.0)
            dy = np.maximum(np.maximum(bbox_min[room_idx, 1] - y, y - bbox_max[room_idx, 1]), 0.0)
            bbox_dist2 = dx * dx + dy * dy
            centroid_dist2 = (x - centroids[room_idx, 0]) ** 2 + (y - centroids[room_idx, 1]) ** 2
            score = bbox_dist2 + centroid_dist2 * 1e-6
            update = score < best_score
            best_score[update] = score[update]
            best_idx[update] = room_idx
        out[start:stop] = best_idx
    return out


def main() -> None:
    args = parse_args()
    work_dir = Path(args.work_dir).expanduser().resolve()
    output_dir = work_dir / "02_rooms_refined"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_json = output_dir / "rooms.json"
    if output_json.exists() and not args.force:
        print(f"[room-refine] Reusing {output_json}")
        return

    preprocess_path = Path(args.preprocess_manifest).expanduser().resolve()
    rooms_json = Path(args.rooms_json).expanduser().resolve()
    preprocess = read_json(preprocess_path)
    proposals = read_json(rooms_json)
    cloud = np.load(Path(preprocess["cloud_npz"]), allow_pickle=False)
    coord = np.asarray(cloud["coord"], dtype=np.float32)
    if len(coord) == 0:
        raise ValueError("Preprocessed point cloud is empty.")

    proposal_grid = proposals.get("grid", {}) if isinstance(proposals, dict) else {}
    grid_size = float(args.grid_size) if float(args.grid_size) > 0 else float(proposal_grid.get("grid_size") or 0.10)
    if "origin_xy" in proposal_grid:
        origin_xy = np.asarray(proposal_grid["origin_xy"], dtype=np.float64)
    else:
        origin_xy = coord[:, :2].min(axis=0).astype(np.float64) - grid_size * 4.0
    if "shape" in proposal_grid:
        shape = (int(proposal_grid["shape"][0]), int(proposal_grid["shape"][1]))
    else:
        shape = grid_shape_from_coord(coord, origin_xy, grid_size)

    map_args = argparse.Namespace(
        grid_size=grid_size,
        slice_count=int(args.slice_count),
        z_min_quantile=0.08,
        z_max_quantile=0.92,
        wall_min_height=float(args.wall_min_height),
        wall_max_height=float(args.wall_max_height),
        wall_persistence=float(args.wall_persistence),
        wall_dilate_cells=int(args.wall_dilate_cells),
    )
    wall_map, wall_debug = build_border_map(coord, origin_xy, shape, map_args)
    wall_debug_maps = wall_debug.pop("maps", {})
    density_occ = occupancy_from_points(coord, origin_xy, grid_size, shape, min_count=1)
    footprint = fill_holes(dilate(density_occ, 2))

    proposal_masks = load_room_masks(proposals, rooms_json, coord, origin_xy, grid_size, shape)
    cleaned = []
    min_cells = max(1, int(float(args.min_room_area) / (grid_size**2)))
    for room in proposal_masks:
        mask = cleanup_mask(room["mask"] & footprint, int(args.cleanup_close_cells))
        room["mask"] = mask
        if int(np.count_nonzero(mask)) >= min_cells:
            cleaned.append(room)

    split_rooms: list[dict[str, Any]] = []
    for room in cleaned:
        split_rooms.extend(split_room_mask(room, wall_map, args, grid_size))
    merged_rooms = merge_rooms(split_rooms, wall_map, args, grid_size)
    merged_rooms = sorted(merged_rooms, key=lambda item: int(np.count_nonzero(item["mask"])), reverse=True)[: int(args.max_rooms)]

    rooms = []
    indices_payload: dict[str, np.ndarray] = {}
    claimed_points = np.zeros(len(coord), dtype=bool)
    point_owner_index = np.full(len(coord), -1, dtype=np.int32)
    room_label_map = np.full(shape, -1, dtype=np.int32)
    cell_owner_map = np.full(shape, -1, dtype=np.int32)
    for room_number, room in enumerate(merged_rooms, start=1):
        mask = cleanup_mask(room["mask"] & footprint, int(args.cleanup_close_cells))
        if int(np.count_nonzero(mask)) < min_cells:
            continue
        polygon = mask_to_polygon(mask, origin_xy, grid_size, float(args.simplify_epsilon))
        polygon = snap_polygon_to_walls(
            polygon,
            wall_map,
            origin_xy,
            grid_size,
            float(args.snap_distance),
            float(args.snap_min_support),
        )
        assignment_mask = dilate(mask, int(args.assignment_dilate_cells)) & footprint
        point_indices = assign_point_indices(coord, assignment_mask.astype(np.int32), 1, origin_xy, grid_size)
        point_indices = point_indices[~claimed_points[point_indices]]
        if len(point_indices) == 0:
            continue
        room_idx = len(rooms)
        claimed_points[point_indices] = True
        point_owner_index[point_indices] = room_idx

        rows, cols = np.where(mask)
        room_id = f"room_{room_idx + 1:03d}"
        key = f"{room_id}_indices"
        indices_payload[key] = point_indices.astype(np.int64)
        room_label_map[mask] = room_idx
        unowned_cells = assignment_mask & (cell_owner_map < 0)
        cell_owner_map[unowned_cells] = room_idx
        rooms.append(
            {
                "room_id": room_id,
                "room_type": room["room_type"],
                "polygon_xy": polygon,
                "bbox_min_xy": [
                    float(origin_xy[0] + cols.min() * grid_size),
                    float(origin_xy[1] + rows.min() * grid_size),
                ],
                "bbox_max_xy": [
                    float(origin_xy[0] + (cols.max() + 1) * grid_size),
                    float(origin_xy[1] + (rows.max() + 1) * grid_size),
                ],
                "area_m2": float(np.count_nonzero(mask) * grid_size**2),
                "point_count": int(len(point_indices)),
                "point_indices_npz": "room_point_indices.npz",
                "point_indices_key": key,
                "source_room_ids": room.get("source_room_ids", []),
                "merged_from": room.get("merged_from", []),
                "split_from": room.get("split_from", []),
                "method": "point2graph_style_refined_geometry",
            }
        )

    if not rooms:
        raise RuntimeError("Room refinement removed every proposal. Check room detector settings.")

    ownership_debug = {
        "assign_all_points": bool(args.assign_all_points),
        "initial_assigned_points": int(np.count_nonzero(claimed_points)),
        "leftover_area_count": 0,
        "leftover_area_points": 0,
        "nearest_assigned_points": 0,
    }
    if args.assign_all_points:
        leftover_free = footprint & ~wall_map & (cell_owner_map < 0)
        leftover_labels, leftover_components = connected_components(leftover_free)
        min_leftover_cells = max(1, int(float(args.leftover_component_min_area) / (grid_size**2)))
        kept_leftovers = sorted(
            (item for item in leftover_components if item[1] >= min_leftover_cells),
            key=lambda item: item[1],
            reverse=True,
        )
        area_counters = {"corridor": 0, "common_area": 0}
        for component_id, cell_count in kept_leftovers:
            mask = cleanup_mask((leftover_labels == component_id) & footprint, int(args.cleanup_close_cells))
            if int(np.count_nonzero(mask)) < min_leftover_cells:
                continue
            point_indices = assign_point_indices(coord, mask.astype(np.int32), 1, origin_xy, grid_size)
            point_indices = point_indices[~claimed_points[point_indices]]
            if len(point_indices) == 0:
                continue

            room_type = classify_leftover_area(mask, origin_xy, grid_size, args)
            area_counters[room_type] += 1
            room_id = f"{room_type}_{area_counters[room_type]:03d}"
            key = f"{room_id}_indices"
            room_idx = len(rooms)
            claimed_points[point_indices] = True
            point_owner_index[point_indices] = room_idx
            indices_payload[key] = point_indices.astype(np.int64)
            cell_owner_map[mask & (cell_owner_map < 0)] = room_idx
            room_label_map[mask] = room_idx

            bbox_min, bbox_max = mask_bbox_xy(mask, origin_xy, grid_size)
            polygon = mask_to_polygon(mask, origin_xy, grid_size, float(args.simplify_epsilon))
            area_m2 = float(cell_count * grid_size**2)
            rooms.append(
                {
                    "room_id": room_id,
                    "room_type": room_type,
                    "polygon_xy": polygon,
                    "bbox_min_xy": bbox_min,
                    "bbox_max_xy": bbox_max,
                    "area_m2": area_m2,
                    "point_count": int(len(point_indices)),
                    "point_indices_npz": "room_point_indices.npz",
                    "point_indices_key": key,
                    "source_room_ids": [],
                    "merged_from": [],
                    "split_from": [],
                    "method": "ownership_leftover_free_component",
                }
            )
            ownership_debug["leftover_area_count"] += 1
            ownership_debug["leftover_area_points"] += int(len(point_indices))

        remaining = np.flatnonzero(~claimed_points).astype(np.int64)
        if len(remaining):
            nearest = nearest_area_indices(coord[remaining, :2], rooms)
            for room_idx, room in enumerate(rooms):
                owned = remaining[nearest == room_idx]
                if len(owned) == 0:
                    continue
                key = room["point_indices_key"]
                indices_payload[key] = np.concatenate([indices_payload[key], owned.astype(np.int64)])
                point_owner_index[owned] = room_idx
                ownership_debug["nearest_assigned_points"] += int(len(owned))
            claimed_points[remaining] = True

        for room_idx, room in enumerate(rooms):
            key = room["point_indices_key"]
            unique_indices = np.unique(indices_payload[key].astype(np.int64))
            indices_payload[key] = unique_indices
            room["point_count"] = int(len(unique_indices))

    ownership_debug["final_assigned_points"] = int(np.count_nonzero(point_owner_index >= 0))
    ownership_debug["unassigned_points"] = int(np.count_nonzero(point_owner_index < 0))
    ownership_debug["coverage_ratio"] = float(np.count_nonzero(point_owner_index >= 0) / max(len(coord), 1))

    indices_npz = output_dir / "room_point_indices.npz"
    np.savez_compressed(indices_npz, **indices_payload)
    np.save(output_dir / "room_label_map.npy", room_label_map)
    np.save(output_dir / "room_owner_index.npy", point_owner_index)
    if wall_debug_maps:
        wall_debug_maps = {k: v for k, v in wall_debug_maps.items() if k not in {"wall_border", "footprint"}}
        np.savez_compressed(
            output_dir / "room_refine_debug_maps.npz",
            wall_border=wall_map.astype(np.uint8),
            footprint=footprint.astype(np.uint8),
            **wall_debug_maps,
        )

    preview_ply = ""
    if args.preview_ply:
        preview_path = output_dir / "rooms_preview.ply"
        write_preview(preview_path, coord, rooms, indices_payload)
        preview_ply = str(preview_path)

    payload = {
        "source": "Point2Graph-style room proposals refined with wall-stable geometry",
        "preprocess_manifest": str(preprocess_path),
        "proposal_rooms_json": str(rooms_json),
        "point_indices_npz": str(indices_npz),
        "preview_ply": preview_ply,
        "grid": {
            "grid_size": float(grid_size),
            "origin_xy": [float(origin_xy[0]), float(origin_xy[1])],
            "shape": [int(shape[0]), int(shape[1])],
        },
        "debug": {
            **wall_debug,
            "proposal_count": len(proposal_masks),
            "cleaned_count": len(cleaned),
            "split_count": len(split_rooms),
            "refined_count": len(rooms),
            "footprint_cells": int(np.count_nonzero(footprint)),
            "wall_cells": int(np.count_nonzero(wall_map)),
            "min_room_area": float(args.min_room_area),
            "split_min_area": float(args.split_min_area),
            "merge_max_wall_ratio": float(args.merge_max_wall_ratio),
            "snap_distance": float(args.snap_distance),
            "ownership": ownership_debug,
        },
        "rooms": rooms,
    }
    write_json(output_json, payload)
    print(f"[room-refine] proposals={len(proposal_masks)} split={len(split_rooms)} refined={len(rooms)}")
    print(
        "[room-refine] ownership "
        f"assigned={ownership_debug['final_assigned_points']:,}/{len(coord):,} "
        f"unassigned={ownership_debug['unassigned_points']:,}"
    )
    print(f"[room-refine] wrote {output_json}")


if __name__ == "__main__":
    main()
