#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import mimetypes
import re
import sys
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import numpy as np


APP_ROOT = Path(__file__).resolve().parent
STATIC_ROOT = APP_ROOT / "static"
REPO_ROOT = APP_ROOT.parents[1]
VENDOR_ROOT = REPO_ROOT / "mosaic3d_studio" / "static" / "vendor"


def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def safe_name(text: str, fallback: str = "object") -> str:
    value = " ".join(str(text).strip().split())
    value = "".join(ch if ch.isalnum() or ch in ("-", "_", " ") else "_" for ch in value)
    return value or fallback


def label_key(text: str) -> str:
    value = re.sub(r"[_\-]+", " ", str(text or "").strip().lower())
    value = re.sub(r"\s+", " ", value).strip()
    return value


def class_key_variants(text: str) -> set[str]:
    key = label_key(text)
    if not key:
        return set()
    variants = {key}
    parts = key.split()
    if len(parts) > 1 and parts[-1].isdigit():
        variants.add(" ".join(parts[:-1]))
    if key.endswith("ies") and len(key) > 3:
        variants.add(f"{key[:-3]}y")
    if key.endswith("es") and len(key) > 2:
        variants.add(key[:-2])
    if key.endswith("s") and len(key) > 1:
        variants.add(key[:-1])
    return {item for item in variants if item}


def readable_node_name(raw: str) -> str:
    return re.sub(r"\s+", "_", str(raw).strip())


def natural_sort_key(text: str) -> list:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", str(text))]


def room_id_value(room: dict) -> str:
    return str(room.get("room_id") or room.get("id") or room.get("name") or "room")


def bbox_xy_gap(a: dict, b: dict) -> float:
    amin = np.asarray(a.get("bbox_min_xy", []), dtype=np.float32)
    amax = np.asarray(a.get("bbox_max_xy", []), dtype=np.float32)
    bmin = np.asarray(b.get("bbox_min_xy", []), dtype=np.float32)
    bmax = np.asarray(b.get("bbox_max_xy", []), dtype=np.float32)
    if amin.shape[0] != 2 or amax.shape[0] != 2 or bmin.shape[0] != 2 or bmax.shape[0] != 2:
        apoly = np.asarray(a.get("polygon_xy", []), dtype=np.float32)
        bpoly = np.asarray(b.get("polygon_xy", []), dtype=np.float32)
        if len(apoly) < 3 or len(bpoly) < 3:
            return float("inf")
        amin, amax = np.min(apoly, axis=0), np.max(apoly, axis=0)
        bmin, bmax = np.min(bpoly, axis=0), np.max(bpoly, axis=0)
    dx = max(0.0, float(max(bmin[0] - amax[0], amin[0] - bmax[0])))
    dy = max(0.0, float(max(bmin[1] - amax[1], amin[1] - bmax[1])))
    return float((dx * dx + dy * dy) ** 0.5)


def room_neighbor_map(rooms: list[dict], max_gap: float = 0.75, max_neighbors: int = 6) -> dict[str, list[str]]:
    distances: dict[str, list[tuple[float, str]]] = {room_id_value(room): [] for room in rooms}
    for i, room_a in enumerate(rooms):
        room_a_id = room_id_value(room_a)
        for room_b in rooms[i + 1 :]:
            room_b_id = room_id_value(room_b)
            gap = bbox_xy_gap(room_a, room_b)
            if gap <= float(max_gap):
                distances.setdefault(room_a_id, []).append((gap, room_b_id))
                distances.setdefault(room_b_id, []).append((gap, room_a_id))

    neighbors = {}
    for room_id, items in distances.items():
        items = sorted(items, key=lambda item: (item[0], natural_sort_key(item[1])))[: int(max_neighbors)]
        neighbors[room_id] = [neighbor_id for _, neighbor_id in items]
    return neighbors


def is_asset_label(label: str) -> bool:
    key = label_key(label)
    return key in {
        "cabinet",
        "chair",
        "couch",
        "desk",
        "shelf",
        "sofa",
        "table",
        "workbench",
        "workstation",
    }


def build_readable_scene_graph(graph: dict) -> list[dict]:
    rooms = list(graph.get("rooms", []))
    objects = {str(item["id"]): item for item in graph.get("objects", [])}
    room_ids = [room_id_value(room) for room in rooms]
    room_lookup = {room_id_value(room): room for room in rooms}
    readable_room_neighbors = room_neighbor_map(rooms)

    room_children: dict[str, list[str]] = {room_id: [] for room_id in room_ids}
    for obj_id, item in objects.items():
        room_id = str(item.get("room_id", "floor"))
        room_children.setdefault(room_id, []).append(obj_id)

    supported_by: dict[str, str] = {}
    support_children: dict[str, list[str]] = {}
    for edge in graph.get("edges", []):
        if edge.get("relation") != "on_top_of":
            continue
        child = str(edge.get("source", ""))
        parent = str(edge.get("target", ""))
        if child not in objects or parent not in objects:
            continue
        if str(objects[child].get("room_id")) != str(objects[parent].get("room_id")):
            continue
        if not is_asset_label(str(objects[parent].get("label", ""))):
            continue
        supported_by.setdefault(child, parent)
        support_children.setdefault(parent, []).append(child)

    def object_sort_key(obj_id: str) -> list:
        return natural_sort_key(readable_node_name(objects.get(obj_id, {}).get("name", obj_id)))

    def object_entry(obj_id: str, seen: set[str] | None = None) -> dict:
        seen = set(seen or ())
        item = objects.get(obj_id, {})
        name = readable_node_name(str(item.get("name") or item.get("object_name") or obj_id))
        node_type = "asset" if is_asset_label(str(item.get("label", ""))) else "object"
        if obj_id in seen:
            return {"node_name": name, "node_type": node_type}
        seen.add(obj_id)

        entry: dict = {"node_name": name, "node_type": node_type}
        children = [
            child
            for child in sorted(support_children.get(obj_id, []), key=object_sort_key)
            if child not in seen
        ]
        if children or node_type == "asset":
            entry["child_nodes"] = [object_entry(child, seen=set(seen)) for child in children]
        return entry

    readable = []
    for room in sorted(rooms, key=lambda item: natural_sort_key(room_id_value(item))):
        room_id = room_id_value(room)
        direct_children = [
            obj_id
            for obj_id in room_children.get(room_id, [])
            if supported_by.get(obj_id) not in objects
            or str(objects.get(supported_by.get(obj_id), {}).get("room_id")) != room_id
        ]
        readable.append(
            {
                "node_name": readable_node_name(room_lookup.get(room_id, room).get("room_id", room_id)),
                "node_type": "room",
                "neighbor_nodes": [
                    readable_node_name(neighbor)
                    for neighbor in sorted(readable_room_neighbors.get(room_id, []), key=natural_sort_key)
                ],
                "child_nodes": [object_entry(obj_id) for obj_id in sorted(direct_children, key=object_sort_key)],
            }
        )
    return readable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Web QA viewer for Mosaic3D floor labels.")
    parser.add_argument(
        "--work-dir",
        default="",
        help="Pipeline run directory containing 04_stitched/.",
    )
    parser.add_argument("--predictions", default="", help="floor_mosaic3d_stitched_predictions.npz")
    parser.add_argument("--instances-npz", default="", help="floor_instances.npz")
    parser.add_argument("--instances-json", default="", help="floor_instances.json")
    parser.add_argument("--scene-graph", default="", help="scene_graph.json")
    parser.add_argument(
        "--load-latest-edits",
        action="store_true",
        help="With --work-dir, load the newest 05_interactive_edits/session_* outputs.",
    )
    parser.add_argument(
        "--edit-session",
        default="",
        help="With --work-dir, load this saved edit session path or session_* name.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8898)
    parser.add_argument("--max-points", type=int, default=6000000)
    parser.add_argument(
        "--export-viewer-sample",
        action="store_true",
        help="Write viewer_sample_instance_colorized.ply with exactly the sampled points shown in the UI.",
    )
    return parser.parse_args()


def palette(num_labels: int) -> np.ndarray:
    import colorsys

    hues = (np.arange(max(num_labels, 1)) * 0.618033988749895) % 1.0
    colors = [colorsys.hsv_to_rgb(float(hue), 0.65, 0.95) for hue in hues]
    return np.asarray(colors, dtype=np.float32)


def object_palette(num_labels: int) -> np.ndarray:
    """Bright, high-contrast palette for object instances."""

    import colorsys

    fixed = np.asarray(
        [
            [0.95, 0.10, 0.10],
            [0.10, 0.42, 1.00],
            [0.10, 0.78, 0.25],
            [1.00, 0.62, 0.05],
            [0.72, 0.18, 1.00],
            [0.00, 0.78, 0.78],
            [1.00, 0.10, 0.65],
            [0.72, 0.90, 0.06],
            [0.25, 0.22, 1.00],
            [1.00, 0.28, 0.12],
            [0.00, 0.95, 0.55],
            [0.95, 0.88, 0.10],
            [0.55, 0.35, 0.08],
            [0.40, 0.90, 1.00],
            [0.95, 0.45, 0.95],
            [0.52, 1.00, 0.42],
            [0.12, 0.68, 0.95],
            [1.00, 0.40, 0.40],
        ],
        dtype=np.float32,
    )
    if num_labels <= len(fixed):
        return fixed[: max(num_labels, 1)]

    colors = [color for color in fixed]
    remaining = num_labels - len(colors)
    for i in range(remaining):
        hue = (0.07 + i * 0.7548776662466927) % 1.0
        sat = (0.98, 0.84, 0.70)[i % 3]
        val = (1.00, 0.92, 0.82)[(i // 3) % 3]
        colors.append(colorsys.hsv_to_rgb(float(hue), sat, val))
    return np.asarray(colors, dtype=np.float32)


def points_in_polygon_xy(xy: np.ndarray, polygon: list[list[float]]) -> np.ndarray:
    if len(polygon) < 3:
        return np.zeros(len(xy), dtype=bool)
    poly = np.asarray(polygon, dtype=np.float64)
    x = xy[:, 0].astype(np.float64)
    y = xy[:, 1].astype(np.float64)
    inside = np.zeros(len(xy), dtype=bool)
    xj = poly[-1, 0]
    yj = poly[-1, 1]
    for xi, yi in poly:
        crosses = ((yi > y) != (yj > y)) & (
            x < (xj - xi) * (y - yi) / ((yj - yi) + 1e-12) + xi
        )
        inside ^= crosses
        xj = xi
        yj = yi
    return inside


def write_point_cloud(path: Path, coord: np.ndarray, color: np.ndarray) -> None:
    import open3d as o3d

    path.parent.mkdir(parents=True, exist_ok=True)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(coord, dtype=np.float64))
    rgb = np.asarray(color, dtype=np.float32)
    if rgb.max(initial=0) > 1.0:
        rgb = rgb / 255.0
    pcd.colors = o3d.utility.Vector3dVector(np.clip(rgb, 0.0, 1.0).astype(np.float64))
    o3d.io.write_point_cloud(str(path), pcd)


def saved_edit_session_paths(work_dir: Path, session_arg: str = "") -> dict[str, Path]:
    edits_root = work_dir / "05_interactive_edits"
    if session_arg:
        session = Path(session_arg).expanduser()
        if not session.is_absolute():
            session = edits_root / session_arg
        session = session.resolve()
    else:
        candidates = [
            path
            for path in edits_root.glob("session_*")
            if path.is_dir()
            and (path / "floor_instances.json").exists()
            and (path / "floor_instances.npz").exists()
            and (path / "scene_graph.json").exists()
        ]
        if not candidates:
            raise FileNotFoundError(f"No saved edit sessions found in {edits_root}")
        session = max(candidates, key=lambda path: (path / "edit_summary.json").stat().st_mtime if (path / "edit_summary.json").exists() else path.stat().st_mtime)

    required = {
        "instances_npz": session / "floor_instances.npz",
        "instances_json": session / "floor_instances.json",
        "scene_graph": session / "scene_graph_detailed.json",
    }
    if not required["scene_graph"].exists():
        if (session / "edited_scene_graph.json").exists():
            required["scene_graph"] = session / "edited_scene_graph.json"
        else:
            required["scene_graph"] = session / "scene_graph.json"
    missing = [name for name, path in required.items() if not path.exists()]
    if missing:
        joined = "\n".join(f"{name}: {required[name]}" for name in missing)
        raise FileNotFoundError(f"Saved edit session is incomplete:\n{joined}")
    return required


def resolve_inputs(args: argparse.Namespace) -> dict[str, Path]:
    if args.work_dir:
        work_dir = Path(args.work_dir).expanduser().resolve()
        stitched = work_dir / "04_stitched"
        scene_graph_default = stitched / "scene_graph_detailed.json"
        if not scene_graph_default.exists():
            scene_graph_default = stitched / "scene_graph.json"
        defaults = {
            "predictions": stitched / "floor_mosaic3d_stitched_predictions.npz",
            "instances_npz": stitched / "floor_instances.npz",
            "instances_json": stitched / "floor_instances.json",
            "scene_graph": scene_graph_default,
        }
        if args.load_latest_edits or args.edit_session:
            defaults.update(saved_edit_session_paths(work_dir, args.edit_session))
    else:
        defaults = {}

    paths = {
        "predictions": Path(args.predictions).expanduser().resolve()
        if args.predictions
        else defaults.get("predictions", Path()),
        "instances_npz": Path(args.instances_npz).expanduser().resolve()
        if args.instances_npz
        else defaults.get("instances_npz", Path()),
        "instances_json": Path(args.instances_json).expanduser().resolve()
        if args.instances_json
        else defaults.get("instances_json", Path()),
        "scene_graph": Path(args.scene_graph).expanduser().resolve()
        if args.scene_graph
        else defaults.get("scene_graph", Path()),
    }
    missing = [name for name, path in paths.items() if not path.exists()]
    if missing:
        joined = "\n".join(f"{name}: {paths[name]}" for name in missing)
        raise FileNotFoundError(f"Missing viewer input files:\n{joined}")
    return paths


def stratified_sample(
    point_count: int,
    max_points: int,
    instance_index: np.ndarray,
    seed: int = 13,
) -> np.ndarray:
    if max_points <= 0 or point_count <= max_points:
        return np.arange(point_count, dtype=np.int64)

    rng = np.random.default_rng(seed)
    selected_parts: list[np.ndarray] = []
    selected_count = 0

    valid = np.flatnonzero(instance_index >= 0)
    if len(valid):
        valid_instance = instance_index[valid]
        order = np.argsort(valid_instance, kind="stable")
        valid = valid[order]
        valid_instance = valid_instance[order]
        starts = np.r_[0, np.flatnonzero(np.diff(valid_instance)) + 1]
        ends = np.r_[starts[1:], len(valid_instance)]
        per_instance = max(30, min(450, max_points // max(len(starts) * 2, 1)))
        for start, end in zip(starts, ends):
            ids = valid[start:end]
            take = min(len(ids), per_instance)
            if take <= 0:
                continue
            if len(ids) > take:
                ids = rng.choice(ids, size=take, replace=False)
            selected_parts.append(ids.astype(np.int64))
            selected_count += int(take)
            if selected_count >= max_points:
                break

    selected = np.unique(np.concatenate(selected_parts)) if selected_parts else np.empty(0, dtype=np.int64)
    remaining = max_points - len(selected)
    if remaining > 0:
        if len(selected):
            mask = np.ones(point_count, dtype=bool)
            mask[selected] = False
            candidates = np.flatnonzero(mask)
        else:
            candidates = np.arange(point_count, dtype=np.int64)
        if len(candidates) > remaining:
            fill = rng.choice(candidates, size=remaining, replace=False)
        else:
            fill = candidates
        selected = np.unique(np.concatenate([selected, fill.astype(np.int64)]))

    if len(selected) > max_points:
        selected = rng.choice(selected, size=max_points, replace=False)
    return np.sort(selected.astype(np.int64))


class ViewerData:
    def __init__(self, paths: dict[str, Path], max_points: int):
        pred = np.load(paths["predictions"], allow_pickle=False)
        instances_npz = np.load(paths["instances_npz"], allow_pickle=False)
        instances_payload = json.loads(paths["instances_json"].read_text(encoding="utf-8"))
        scene_graph = json.loads(paths["scene_graph"].read_text(encoding="utf-8"))
        if isinstance(scene_graph, list):
            scene_graph = {
                "graph_type": "readable_room_list",
                "rooms": [],
                "edges": [],
                "readable_rooms": scene_graph,
            }

        self.coord = np.asarray(pred["coord"], dtype=np.float32)
        self.color = np.asarray(pred["color"], dtype=np.float32)
        self.pred_index = np.asarray(pred["pred_index"], dtype=np.int32)
        self.score = np.asarray(pred["score"], dtype=np.float32)
        self.labels = [str(x) for x in pred["labels"]]
        self.instance_index = np.asarray(instances_npz["instance_index"], dtype=np.int32)
        self.instance_params = dict(instances_payload.get("params", {}))
        self.instances = self._normalize_instances(list(instances_payload.get("instances", [])))
        self.scene_graph = scene_graph
        self.paths = paths
        self.max_points = int(max_points)
        self.manual_segment_count = 0
        self.last_preview_indices: np.ndarray | None = None
        self.last_preview_summary: dict | None = None
        self._points_binary_dirty = True
        self._points_binary_cache = b""

        if self.color.max(initial=0) > 1.0:
            self.color = self.color / 255.0
        self.color = np.clip(self.color, 0.0, 1.0).astype(np.float32)

        self.center = ((self.coord.min(axis=0) + self.coord.max(axis=0)) * 0.5).astype(np.float32)
        self.bbox_min = self.coord.min(axis=0).astype(float).tolist()
        self.bbox_max = self.coord.max(axis=0).astype(float).tolist()
        self.sample_index = stratified_sample(
            point_count=len(self.coord),
            max_points=int(max_points),
            instance_index=self.instance_index,
        )
        self.edit_dir = self._default_edit_dir()

    def export_viewer_sample(self) -> Path:
        predictions = self.paths["predictions"]
        root = predictions.parent / "viewer_exports"
        root.mkdir(parents=True, exist_ok=True)
        path = root / f"viewer_sample_{len(self.sample_index)}_points_instance_colorized.ply"
        inst_palette = self._instance_palette()
        sampled_inst = self.instance_index[self.sample_index]
        colors = np.full((len(self.sample_index), 3), 0.48, dtype=np.float32)
        assigned = sampled_inst >= 0
        if np.any(assigned):
            safe_inst = np.clip(sampled_inst[assigned], 0, len(inst_palette) - 1)
            colors[assigned] = inst_palette[safe_inst]
        write_point_cloud(path, self.coord[self.sample_index], colors)
        return path

    def _default_edit_dir(self) -> Path:
        predictions = self.paths["predictions"]
        if predictions.parent.name == "04_stitched":
            work_dir = predictions.parent.parent
            root = work_dir / "05_interactive_edits"
        else:
            root = predictions.parent / "interactive_edits"
        return root / f"session_{now_stamp()}"

    def _normalize_instances(self, rows: list[dict]) -> list[dict]:
        normalized = []
        for i, item in enumerate(rows):
            instance_id = int(item.get("instance_id", i))
            original = str(item.get("original_label") or item.get("edited_label") or f"instance_{instance_id}")
            edited = str(item.get("edited_label") or original)
            centroid = [float(x) for x in item.get("centroid", [0, 0, 0])]
            normalized.append(
                {
                    "instance_id": instance_id,
                    "semantic_label_index": int(item.get("semantic_label_index", -1)),
                    "semantic_label": str(item.get("semantic_label", "")),
                    "original_label": original,
                    "edited_label": edited,
                    "active": bool(item.get("active", True)),
                    "points": int(item.get("points", 0)),
                    "mean_score": float(item.get("mean_score", 0.0)),
                    "object_label_score": float(item.get("object_label_score", item.get("mean_score", 0.0))),
                    "object_label_margin": float(item.get("object_label_margin", 0.0)),
                    "label_candidates": list(item.get("label_candidates", [])),
                    "relabel_changed": bool(item.get("relabel_changed", False)),
                    "pre_relabel_semantic_label": str(item.get("pre_relabel_semantic_label", "")),
                    "pre_geometry_best_label": str(item.get("pre_geometry_best_label", "")),
                    "geometry_filter_applied": bool(item.get("geometry_filter_applied", False)),
                    "geometry_corrected_label": bool(item.get("geometry_corrected_label", False)),
                    "geometry_rejected_candidates": list(item.get("geometry_rejected_candidates", [])),
                    "centroid": centroid,
                    "bbox_min": [float(x) for x in item.get("bbox_min", [0, 0, 0])],
                    "bbox_max": [float(x) for x in item.get("bbox_max", [0, 0, 0])],
                    "room_id": str(item.get("room_id", "floor")),
                    "source": str(item.get("source", "mosaic3d")),
                    "qa_status": str(item.get("qa_status", "likely_correct")),
                    "qa_reasons": [str(x) for x in item.get("qa_reasons", [])],
                    "qa_metrics": dict(item.get("qa_metrics", {})),
                }
            )
        normalized.sort(key=lambda item: int(item["instance_id"]))
        return normalized

    def get_points_binary(self) -> bytes:
        if self._points_binary_dirty:
            self._points_binary_cache = self._build_points_binary()
            self._points_binary_dirty = False
        return self._points_binary_cache

    def _instance_palette(self) -> np.ndarray:
        row_max = max([int(item["instance_id"]) for item in self.instances] + [-1])
        assigned = self.instance_index[self.instance_index >= 0]
        assigned_max = int(assigned.max()) if len(assigned) else -1
        return object_palette(max(row_max, assigned_max, 0) + 2)

    def _build_points_binary(self) -> bytes:
        idx = self.sample_index
        coord = self.coord[idx] - self.center[None, :]
        original = self.color[idx]

        sem_palette = palette(len(self.labels))
        safe_pred = np.clip(self.pred_index[idx], 0, len(self.labels) - 1)
        semantic = sem_palette[safe_pred]

        inst_palette = self._instance_palette()
        sampled_inst = self.instance_index[idx]
        instance = np.full((len(idx), 3), 0.48, dtype=np.float32)
        assigned = sampled_inst >= 0
        if np.any(assigned):
            safe_inst = np.clip(sampled_inst[assigned], 0, len(inst_palette) - 1)
            instance[assigned] = inst_palette[safe_inst]
            inactive_ids = {
                int(item["instance_id"])
                for item in self.instances
                if not bool(item.get("active", True))
            }
            if inactive_ids:
                inactive = np.isin(sampled_inst, np.asarray(sorted(inactive_ids), dtype=np.int32))
                instance[inactive] = np.asarray([0.34, 0.36, 0.39], dtype=np.float32)

        payload = np.column_stack(
            [
                coord.astype(np.float32),
                original.astype(np.float32),
                semantic.astype(np.float32),
                instance.astype(np.float32),
                self.pred_index[idx].astype(np.float32),
                sampled_inst.astype(np.float32),
                self.score[idx].astype(np.float32),
            ]
        )
        return np.ascontiguousarray(payload.astype(np.float32)).tobytes()

    def _build_metadata(self, max_points: int | None = None) -> dict:
        max_points = self.max_points if max_points is None else int(max_points)
        instance_palette_values = self._instance_palette()
        instances = []
        for item in self.instances:
            centroid = np.asarray(item.get("centroid", [0.0, 0.0, 0.0]), dtype=np.float32)
            instance_id = int(item.get("instance_id", -1))
            color = [0.48, 0.48, 0.48]
            if 0 <= instance_id < len(instance_palette_values):
                color = instance_palette_values[instance_id].astype(float).tolist()
            instances.append(
                {
                    "instance_id": instance_id,
                    "name": str(item.get("edited_label") or item.get("original_label") or f"instance_{instance_id}"),
                    "semantic_label": str(item.get("semantic_label", "")),
                    "room_id": str(item.get("room_id", "floor")),
                    "original_label": str(item.get("original_label", "")),
                    "edited_label": str(item.get("edited_label", item.get("original_label", ""))),
                    "active": bool(item.get("active", True)),
                    "source": str(item.get("source", "mosaic3d")),
                    "points": int(item.get("points", 0)),
                    "mean_score": float(item.get("mean_score", 0.0)),
                    "object_label_score": float(item.get("object_label_score", item.get("mean_score", 0.0))),
                    "object_label_margin": float(item.get("object_label_margin", 0.0)),
                    "label_candidates": list(item.get("label_candidates", [])),
                    "relabel_changed": bool(item.get("relabel_changed", False)),
                    "pre_relabel_semantic_label": str(item.get("pre_relabel_semantic_label", "")),
                    "pre_geometry_best_label": str(item.get("pre_geometry_best_label", "")),
                    "geometry_filter_applied": bool(item.get("geometry_filter_applied", False)),
                    "geometry_corrected_label": bool(item.get("geometry_corrected_label", False)),
                    "geometry_rejected_candidates": list(item.get("geometry_rejected_candidates", [])),
                    "centroid": centroid.astype(float).tolist(),
                    "viewer_centroid": (centroid - self.center).astype(float).tolist(),
                    "bbox_min": [float(x) for x in item.get("bbox_min", [0, 0, 0])],
                    "bbox_max": [float(x) for x in item.get("bbox_max", [0, 0, 0])],
                    "color": color,
                    "qa_status": str(item.get("qa_status", "likely_correct")),
                    "qa_reasons": [str(x) for x in item.get("qa_reasons", [])],
                    "qa_metrics": dict(item.get("qa_metrics", {})),
                }
            )

        label_counts = {}
        unique, counts = np.unique(self.pred_index, return_counts=True)
        for label_id, count in zip(unique, counts):
            if 0 <= int(label_id) < len(self.labels):
                label_counts[self.labels[int(label_id)]] = int(count)

        return {
            "title": "Mosaic3D Label QA Viewer",
            "stride": 15,
            "point_count": int(len(self.coord)),
            "sample_count": int(len(self.sample_index)),
            "max_points": int(max_points),
            "center": self.center.astype(float).tolist(),
            "bbox_min": self.bbox_min,
            "bbox_max": self.bbox_max,
            "labels": self.labels,
            "label_counts": label_counts,
            "instances": instances,
            "rooms": self.scene_graph.get("rooms", []),
            "edge_count": len(self.scene_graph.get("edges", [])),
            "files": {name: str(path) for name, path in self.paths.items()},
            "edit_dir": str(self.edit_dir),
            "last_preview": self.last_preview_summary or {},
        }

    def metadata_payload(self) -> dict:
        return self._build_metadata()

    def _find_instance(self, instance_id: int) -> dict:
        for item in self.instances:
            if int(item["instance_id"]) == int(instance_id):
                return item
        raise IndexError(f"instance_id out of range: {instance_id}")

    def edit_instance(self, instance_id: int, edited_label: str) -> dict:
        item = self._find_instance(instance_id)
        edited = edited_label.strip() or str(item["original_label"])
        item["edited_label"] = edited
        return {"ok": True, "metadata": self.metadata_payload()}

    def set_instance_active(self, instance_id: int, active: bool) -> dict:
        item = self._find_instance(instance_id)
        item["active"] = bool(active)
        self._points_binary_dirty = True
        return {"ok": True, "metadata": self.metadata_payload()}

    def set_class_active(self, label: str, active: bool) -> dict:
        target_variants = class_key_variants(label)
        if not target_variants:
            raise ValueError("Class label is required")

        matched_ids = []
        for item in self.instances:
            item_variants = set()
            for field in ("semantic_label", "edited_label", "original_label"):
                item_variants.update(class_key_variants(str(item.get(field, ""))))
            if target_variants.isdisjoint(item_variants):
                continue
            item["active"] = bool(active)
            matched_ids.append(int(item["instance_id"]))

        if matched_ids:
            self._points_binary_dirty = True
        return {
            "ok": True,
            "label": label,
            "active": bool(active),
            "matched_count": len(matched_ids),
            "instance_ids": matched_ids,
            "metadata": self.metadata_payload(),
        }

    def reset_preview(self) -> dict:
        self.last_preview_indices = None
        self.last_preview_summary = None
        return {"ok": True, "metadata": self.metadata_payload()}

    def _sample_hits(self, global_indices: np.ndarray, max_hits: int = 200_000) -> list[int]:
        if len(global_indices) == 0:
            return []
        global_indices = np.unique(np.asarray(global_indices, dtype=np.int64))
        pos = np.searchsorted(self.sample_index, global_indices)
        valid = pos < len(self.sample_index)
        pos = pos[valid]
        global_indices = global_indices[valid]
        keep = self.sample_index[pos] == global_indices
        hits = pos[keep].astype(np.int32)
        if len(hits) > max_hits:
            rng = np.random.default_rng(37)
            hits = np.sort(rng.choice(hits, size=max_hits, replace=False).astype(np.int32))
        return hits.tolist()

    def _nearest_index(self, point: np.ndarray, search_radius: float) -> int:
        delta = self.coord - point[None, :]
        dist2 = np.einsum("ij,ij->i", delta, delta)
        idx = int(np.argmin(dist2))
        if float(dist2[idx]) > float(search_radius) ** 2:
            return idx
        return idx

    def _room_for_centroid(self, centroid: np.ndarray) -> str:
        xy = np.asarray(centroid[:2], dtype=np.float64)[None, :]
        for room in self.scene_graph.get("rooms", []):
            polygon = room.get("polygon_xy", [])
            if len(polygon) >= 3 and bool(points_in_polygon_xy(xy, polygon)[0]):
                return str(room.get("id") or room.get("room_id") or "floor")
        return "floor"

    def _unique_manual_label(self, label: str) -> str:
        base = safe_name(label, "object")
        used = {str(item.get("edited_label", "")) for item in self.instances}
        if base not in used:
            return base
        n = 1
        while f"{base}_{n}" in used:
            n += 1
        return f"{base}_{n}"

    def segment_preview(self, body: dict) -> dict:
        positives = np.asarray(body.get("positive_points") or [], dtype=np.float32)
        negatives = np.asarray(body.get("negative_points") or [], dtype=np.float32)
        if positives.ndim != 2 or positives.shape[1] != 3 or len(positives) == 0:
            raise ValueError("positive_points must contain at least one [x,y,z] point")

        radius = float(body.get("radius", 1.5))
        voxel_size = float(body.get("voxel_size", 0.12))
        negative_radius = float(body.get("negative_radius", max(voxel_size * 2.0, 0.20)))
        semantic_mode = str(body.get("semantic_mode", "same")).strip().lower()
        max_crop_points = int(body.get("max_crop_points", 450_000))
        max_preview_points = int(body.get("max_preview_points", 20_000))
        radius = max(0.15, min(radius, 10.0))
        voxel_size = max(0.03, min(voxel_size, 0.75))
        max_crop_points = max(20_000, min(max_crop_points, 800_000))
        max_preview_points = max(250, min(max_preview_points, 50_000))

        seed_point = positives[0]
        delta = self.coord - seed_point[None, :]
        dist2 = np.einsum("ij,ij->i", delta, delta)
        crop_mask = dist2 <= radius * radius
        local_global = np.flatnonzero(crop_mask).astype(np.int64)
        if len(local_global) == 0:
            raise ValueError("No points found near click. Increase segment radius.")

        local_dist2 = dist2[local_global]
        if len(local_global) > max_crop_points:
            order = np.argpartition(local_dist2, max_crop_points - 1)[:max_crop_points]
            local_global = local_global[order]
            local_dist2 = local_dist2[order]

        seed_local = int(np.argmin(local_dist2))
        seed_global = int(local_global[seed_local])
        seed_label = int(self.pred_index[seed_global])
        local_pred = self.pred_index[local_global]

        allowed = np.ones(len(local_global), dtype=bool)
        if semantic_mode != "all":
            if 0 <= seed_label < len(self.labels):
                other_label = self.labels.index("other") if "other" in self.labels else -999
                allowed = local_pred == seed_label
                if semantic_mode == "loose" and seed_label != other_label:
                    allowed |= local_pred == other_label

        if len(negatives):
            for neg in negatives:
                neg_delta = self.coord[local_global] - neg[None, :]
                neg_dist2 = np.einsum("ij,ij->i", neg_delta, neg_delta)
                allowed &= neg_dist2 > negative_radius * negative_radius

        allowed[seed_local] = True
        allowed_local = np.flatnonzero(allowed)
        if len(allowed_local) == 0:
            allowed_local = np.asarray([seed_local], dtype=np.int64)

        component_local = self._voxel_component(
            points=self.coord[local_global],
            allowed_local=allowed_local,
            seed_local=seed_local,
            voxel_size=voxel_size,
        )
        if len(component_local) < 5:
            near = local_dist2 <= max(voxel_size * 2.0, 0.15) ** 2
            component_local = np.flatnonzero(near)

        component_global = np.sort(local_global[component_local].astype(np.int64))
        if len(component_global) == 0:
            raise ValueError("Segment produced no points.")

        self.last_preview_indices = component_global
        pts = self.coord[component_global]
        centroid = pts.mean(axis=0)
        summary = {
            "points": int(len(component_global)),
            "centroid": centroid.astype(float).tolist(),
            "bbox_min": pts.min(axis=0).astype(float).tolist(),
            "bbox_max": pts.max(axis=0).astype(float).tolist(),
            "seed_label": self.labels[seed_label] if 0 <= seed_label < len(self.labels) else "unknown",
            "radius": radius,
            "voxel_size": voxel_size,
            "semantic_mode": semantic_mode,
        }
        self.last_preview_summary = summary
        preview_indices = component_global
        if len(preview_indices) > max_preview_points:
            rng = np.random.default_rng(39)
            preview_indices = np.sort(
                rng.choice(preview_indices, size=max_preview_points, replace=False).astype(np.int64)
            )
        preview_points = (self.coord[preview_indices] - self.center[None, :]).astype(np.float32)
        return {
            "ok": True,
            "preview": summary,
            "sample_hits": self._sample_hits(component_global),
            "preview_points": preview_points.astype(float).tolist(),
            "metadata": self.metadata_payload(),
        }

    def _voxel_component(
        self,
        points: np.ndarray,
        allowed_local: np.ndarray,
        seed_local: int,
        voxel_size: float,
    ) -> np.ndarray:
        allowed_set = set(int(x) for x in allowed_local.tolist())
        if seed_local not in allowed_set:
            allowed_set.add(seed_local)

        allowed_points = points[allowed_local]
        origin = allowed_points.min(axis=0)
        vox = np.floor((allowed_points - origin[None, :]) / voxel_size).astype(np.int32)
        voxel_to_allowed: dict[tuple[int, int, int], list[int]] = {}
        seed_key = None
        allowed_index_for_seed = 0
        for pos, local_id in enumerate(allowed_local):
            key = (int(vox[pos, 0]), int(vox[pos, 1]), int(vox[pos, 2]))
            voxel_to_allowed.setdefault(key, []).append(pos)
            if int(local_id) == int(seed_local):
                seed_key = key
                allowed_index_for_seed = pos
        if seed_key is None:
            return np.asarray([seed_local], dtype=np.int64)

        neighbors = [
            (dx, dy, dz)
            for dx in (-1, 0, 1)
            for dy in (-1, 0, 1)
            for dz in (-1, 0, 1)
            if not (dx == 0 and dy == 0 and dz == 0)
        ]
        visited_voxels = {seed_key}
        queue = [seed_key]
        component_positions: list[int] = []
        while queue:
            key = queue.pop()
            component_positions.extend(voxel_to_allowed.get(key, []))
            for dx, dy, dz in neighbors:
                nxt = (key[0] + dx, key[1] + dy, key[2] + dz)
                if nxt in visited_voxels or nxt not in voxel_to_allowed:
                    continue
                visited_voxels.add(nxt)
                queue.append(nxt)

        if allowed_index_for_seed not in component_positions:
            component_positions.append(allowed_index_for_seed)
        return allowed_local[np.asarray(component_positions, dtype=np.int64)]

    def accept_segment(self, label: str) -> dict:
        if self.last_preview_indices is None or len(self.last_preview_indices) < 5:
            raise ValueError("No segment preview is available to accept.")

        indices = np.asarray(self.last_preview_indices, dtype=np.int64)
        label = self._unique_manual_label(label or "manual_object")
        new_id = int(max([int(item["instance_id"]) for item in self.instances] + [-1]) + 1)
        old_ids = np.unique(self.instance_index[indices])
        self.instance_index[indices] = new_id

        pts = self.coord[indices]
        pred_ids = self.pred_index[indices]
        valid_pred = pred_ids[(pred_ids >= 0) & (pred_ids < len(self.labels))]
        if len(valid_pred):
            semantic_label_index = int(np.bincount(valid_pred.astype(np.int64)).argmax())
        else:
            semantic_label_index = -1
        semantic_label = self.labels[semantic_label_index] if semantic_label_index >= 0 else "manual"
        scores = self.score[indices]
        centroid = pts.mean(axis=0)
        row = {
            "instance_id": new_id,
            "semantic_label_index": semantic_label_index,
            "semantic_label": semantic_label,
            "original_label": label,
            "edited_label": label,
            "active": True,
            "points": int(len(indices)),
            "mean_score": float(np.mean(scores)) if len(scores) else 0.0,
            "centroid": centroid.astype(float).tolist(),
            "bbox_min": pts.min(axis=0).astype(float).tolist(),
            "bbox_max": pts.max(axis=0).astype(float).tolist(),
            "room_id": self._room_for_centroid(centroid),
            "source": "manual_segment",
        }
        self.instances.append(row)
        self.instances.sort(key=lambda item: int(item["instance_id"]))
        self.manual_segment_count += 1

        for old_id in old_ids:
            old_id = int(old_id)
            if old_id < 0:
                continue
            self._refresh_instance_stats(old_id)

        self.last_preview_indices = None
        self.last_preview_summary = None
        self._points_binary_dirty = True
        return {"ok": True, "new_instance": row, "metadata": self.metadata_payload()}

    def _refresh_instance_stats(self, instance_id: int) -> None:
        try:
            item = self._find_instance(instance_id)
        except IndexError:
            return
        idx = np.flatnonzero(self.instance_index == instance_id)
        item["points"] = int(len(idx))
        if len(idx) == 0:
            item["active"] = False
            return
        pts = self.coord[idx]
        item["centroid"] = pts.mean(axis=0).astype(float).tolist()
        item["bbox_min"] = pts.min(axis=0).astype(float).tolist()
        item["bbox_max"] = pts.max(axis=0).astype(float).tolist()
        item["mean_score"] = float(np.mean(self.score[idx]))

    def save_edits(self) -> dict:
        self.edit_dir.mkdir(parents=True, exist_ok=True)
        instances_json = self.edit_dir / "edited_instances.json"
        instances_csv = self.edit_dir / "edited_instances.csv"
        index_npz = self.edit_dir / "edited_instance_index.npz"
        scene_graph_detailed_json = self.edit_dir / "scene_graph_detailed.json"
        floor_instances_json = self.edit_dir / "floor_instances.json"
        floor_instances_npz = self.edit_dir / "floor_instances.npz"
        floor_scene_graph_json = self.edit_dir / "scene_graph.json"
        summary_json = self.edit_dir / "edit_summary.json"

        floor_instances_payload = {
            "instance_count": int(len(self.instances)),
            "params": self.instance_params,
            "instances": self.instances,
        }
        payload = {
            "version": 1,
            "saved_at": now_iso(),
            "source_files": {name: str(path) for name, path in self.paths.items()},
            "point_count": int(len(self.coord)),
            "pipeline_compatible_files": {
                "instances_json": str(floor_instances_json),
                "instances_npz": str(floor_instances_npz),
                "scene_graph_json": str(floor_scene_graph_json),
                "scene_graph_detailed_json": str(scene_graph_detailed_json),
            },
            "instances": self.instances,
        }
        instances_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        floor_instances_json.write_text(json.dumps(floor_instances_payload, indent=2), encoding="utf-8")
        with instances_csv.open("w", newline="", encoding="utf-8") as handle:
            fieldnames = [
                "instance_id",
                "original_label",
                "edited_label",
                "semantic_label",
                "room_id",
                "active",
                "points",
                "mean_score",
                "object_label_score",
                "object_label_margin",
                "relabel_changed",
                "pre_relabel_semantic_label",
                "source",
                "qa_status",
                "qa_reasons",
            ]
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(self.instances)

        np.savez_compressed(
            index_npz,
            instance_index=self.instance_index.astype(np.int32),
            instance_ids=np.asarray([int(item["instance_id"]) for item in self.instances], dtype=np.int32),
            labels=np.asarray([str(item["edited_label"]) for item in self.instances]),
            original_labels=np.asarray([str(item["original_label"]) for item in self.instances]),
            edited_labels=np.asarray([str(item["edited_label"]) for item in self.instances]),
            active=np.asarray([bool(item.get("active", True)) for item in self.instances]),
        )
        np.savez_compressed(floor_instances_npz, instance_index=self.instance_index.astype(np.int32))
        edited_graph = self._build_edited_scene_graph()
        readable_graph = build_readable_scene_graph(edited_graph)
        scene_graph_detailed_json.write_text(json.dumps(edited_graph, indent=2), encoding="utf-8")
        floor_scene_graph_json.write_text(json.dumps(readable_graph, indent=2), encoding="utf-8")

        active_count = sum(1 for item in self.instances if bool(item.get("active", True)) and int(item.get("points", 0)) > 0)
        manual_count = sum(1 for item in self.instances if str(item.get("source")) == "manual_segment")
        summary = {
            "version": 1,
            "saved_at": now_iso(),
            "edit_dir": str(self.edit_dir),
            "point_count": int(len(self.coord)),
            "instance_count": int(len(self.instances)),
            "active_instance_count": int(active_count),
            "manual_segment_count": int(manual_count),
            "files": {
                "instances_json": str(instances_json),
                "instances_csv": str(instances_csv),
                "instance_index_npz": str(index_npz),
                "scene_graph_json": str(floor_scene_graph_json),
                "scene_graph_detailed_json": str(scene_graph_detailed_json),
                "floor_instances_json": str(floor_instances_json),
                "floor_instances_npz": str(floor_instances_npz),
                "floor_scene_graph_json": str(floor_scene_graph_json),
                "summary_json": str(summary_json),
            },
        }
        summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return {"ok": True, "summary": summary, "metadata": self.metadata_payload()}

    def _build_edited_scene_graph(self) -> dict:
        rooms = list(self.scene_graph.get("rooms", []))
        nodes = [
            {
                "id": "floor",
                "name": "floor",
                "type": "floor",
                "label": "floor",
                "parent": None,
                "children": [str(room.get("id") or room.get("room_id")) for room in rooms],
            }
        ]
        edges = []
        room_children: dict[str, list[str]] = {
            str(room.get("id") or room.get("room_id")): [] for room in rooms
        }
        for room in rooms:
            room_id = str(room.get("id") or room.get("room_id"))
            room_node = dict(room)
            room_node["children"] = []
            nodes.append(room_node)
            edges.append({"source": "floor", "target": room_id, "relation": "contains"})

        objects = []
        active_object_ids_by_name: dict[str, str] = {}
        for item in self.instances:
            if not bool(item.get("active", True)) or int(item.get("points", 0)) <= 0:
                continue
            stable_id = f"instance_{int(item['instance_id']):06d}"
            room_id = str(item.get("room_id", "floor"))
            node = {
                "id": stable_id,
                "name": str(item.get("edited_label", stable_id)),
                "type": "object",
                "label": str(item.get("semantic_label", "")),
                "class_name": str(item.get("semantic_label", "")),
                "object_name": str(item.get("edited_label", stable_id)),
                "original_label": str(item.get("original_label", "")),
                "room_id": room_id,
                "parent": room_id,
                "instance_id": int(item["instance_id"]),
                "source": str(item.get("source", "mosaic3d")),
                "position": {
                    "x": float(item["centroid"][0]),
                    "y": float(item["centroid"][1]),
                    "z": float(item["centroid"][2]),
                },
                "coordinates": {
                    "x": float(item["centroid"][0]),
                    "y": float(item["centroid"][1]),
                },
                "centroid": [float(x) for x in item["centroid"]],
                "bbox_min": [float(x) for x in item["bbox_min"]],
                "bbox_max": [float(x) for x in item["bbox_max"]],
                "points": int(item.get("points", 0)),
                "mean_score": float(item.get("mean_score", 0.0)),
                "object_label_score": float(item.get("object_label_score", item.get("mean_score", 0.0))),
                "object_label_margin": float(item.get("object_label_margin", 0.0)),
                "label_candidates": list(item.get("label_candidates", [])),
                "relabel_changed": bool(item.get("relabel_changed", False)),
                "pre_relabel_semantic_label": str(item.get("pre_relabel_semantic_label", "")),
                "pre_geometry_best_label": str(item.get("pre_geometry_best_label", "")),
                "geometry_filter_applied": bool(item.get("geometry_filter_applied", False)),
                "geometry_corrected_label": bool(item.get("geometry_corrected_label", False)),
                "geometry_rejected_candidates": list(item.get("geometry_rejected_candidates", [])),
                "qa_status": str(item.get("qa_status", "likely_correct")),
                "qa_reasons": list(item.get("qa_reasons", [])),
                "qa_metrics": dict(item.get("qa_metrics", {})),
                "hierarchy": ["floor", room_id, stable_id],
            }
            objects.append(node)
            nodes.append(node)
            for name_key in {
                str(item.get("edited_label", "")),
                str(item.get("original_label", "")),
                str(node.get("name", "")),
                str(node.get("object_name", "")),
            }:
                if name_key:
                    active_object_ids_by_name.setdefault(name_key, stable_id)
            room_children.setdefault(room_id, []).append(stable_id)
            edges.append({"source": room_id, "target": stable_id, "relation": "contains"})

        source_objects = {
            str(item.get("id")): item
            for item in self.scene_graph.get("objects", [])
            if isinstance(item, dict)
        }
        existing_edges = {
            (str(edge.get("source")), str(edge.get("target")), str(edge.get("relation")))
            for edge in edges
        }
        for edge in self.scene_graph.get("edges", []):
            relation = str(edge.get("relation", ""))
            if relation not in {"near", "on_top_of"}:
                continue
            source_key = str(edge.get("source", ""))
            target_key = str(edge.get("target", ""))
            source_name = str(source_objects.get(source_key, {}).get("name") or source_key)
            target_name = str(source_objects.get(target_key, {}).get("name") or target_key)
            source_id = active_object_ids_by_name.get(source_name)
            target_id = active_object_ids_by_name.get(target_name)
            if not source_id or not target_id:
                continue
            edge_key = (source_id, target_id, relation)
            if edge_key in existing_edges:
                continue
            new_edge = {"source": source_id, "target": target_id, "relation": relation}
            if "distance_xy" in edge:
                new_edge["distance_xy"] = edge["distance_xy"]
            edges.append(new_edge)
            existing_edges.add(edge_key)

        for node in nodes:
            if node.get("type") == "room":
                room_id = str(node.get("id") or node.get("room_id"))
                node["children"] = sorted(room_children.get(room_id, []))

        return {
            "graph_type": "mosaic3d_interactive_edited_scene_graph",
            "root": "floor",
            "rooms": [node for node in nodes if node.get("type") == "room"],
            "objects": objects,
            "nodes": nodes,
            "edges": edges,
            "source_graph_type": self.scene_graph.get("graph_type", ""),
        }


def run_server(data: ViewerData, host: str, port: int) -> None:
    class ReusableThreadingHTTPServer(ThreadingHTTPServer):
        allow_reuse_address = True

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args) -> None:
            print(f"[viewer] {self.address_string()} - {fmt % args}")

        def send_bytes(self, body: bytes, content_type: str) -> None:
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def send_json(self, payload: dict, status: int = 200) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def read_json(self) -> dict:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length) if length > 0 else b"{}"
            if not raw:
                return {}
            return json.loads(raw.decode("utf-8"))

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            path = parsed.path
            if path in {"/", "/index.html"}:
                self.serve_file(STATIC_ROOT / "index.html")
                return
            if path == "/api/metadata":
                self.send_bytes(json.dumps(data.metadata_payload()).encode("utf-8"), "application/json")
                return
            if path == "/api/points.bin":
                self.send_bytes(data.get_points_binary(), "application/octet-stream")
                return
            if path.startswith("/static/"):
                self.serve_file(STATIC_ROOT / path.removeprefix("/static/"))
                return
            if path.startswith("/vendor/"):
                self.serve_file(VENDOR_ROOT / path.removeprefix("/vendor/"))
                return
            self.send_error(404, "Not found")

        def do_POST(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            try:
                body = self.read_json()
                if parsed.path == "/api/edit_instance":
                    self.send_json(data.edit_instance(int(body["instance_id"]), str(body.get("edited_label", ""))))
                    return
                if parsed.path == "/api/set_instance_active":
                    self.send_json(data.set_instance_active(int(body["instance_id"]), bool(body.get("active", True))))
                    return
                if parsed.path == "/api/set_class_active":
                    self.send_json(data.set_class_active(str(body.get("label", "")), bool(body.get("active", True))))
                    return
                if parsed.path == "/api/save_edits":
                    self.send_json(data.save_edits())
                    return
                if parsed.path == "/api/segment_preview":
                    self.send_json(data.segment_preview(body))
                    return
                if parsed.path == "/api/segment_accept":
                    self.send_json(data.accept_segment(str(body.get("label", "manual_object"))))
                    return
                if parsed.path == "/api/segment_reset":
                    self.send_json(data.reset_preview())
                    return
                self.send_error(404, "Not found")
            except Exception as exc:
                self.send_json({"ok": False, "error": str(exc)}, status=400)

        def serve_file(self, path: Path) -> None:
            path = path.resolve()
            roots = [STATIC_ROOT.resolve(), VENDOR_ROOT.resolve()]
            if not any(root == path or root in path.parents for root in roots) or not path.exists():
                self.send_error(404, "Not found")
                return
            content_type = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
            self.send_bytes(path.read_bytes(), content_type)

    try:
        httpd = ReusableThreadingHTTPServer((host, port), Handler)
    except OSError as exc:
        if exc.errno == 98:
            print(
                f"[viewer] Port {port} is already in use on {host}.\n"
                f"[viewer] Stop the old viewer process or rerun with another port, for example: --port {port + 1}",
                file=sys.stderr,
            )
            raise SystemExit(98) from exc
        raise
    print(f"[viewer] URL: http://{host}:{port}")
    meta = data.metadata_payload()
    print(f"[viewer] points: {meta['sample_count']:,}/{meta['point_count']:,}")
    print(f"[viewer] objects: {len(meta['instances']):,}")
    print(f"[viewer] edits: {meta['edit_dir']}")
    if getattr(data, "viewer_sample_export_path", None):
        print(f"[viewer] exported sample: {data.viewer_sample_export_path}")
    httpd.serve_forever()


def main() -> None:
    args = parse_args()
    paths = resolve_inputs(args)
    data = ViewerData(paths, max_points=args.max_points)
    if args.export_viewer_sample:
        data.viewer_sample_export_path = str(data.export_viewer_sample())
    run_server(data, args.host, args.port)


if __name__ == "__main__":
    main()
