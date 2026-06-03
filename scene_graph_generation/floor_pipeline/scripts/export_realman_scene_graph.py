#!/usr/bin/env python3
"""Export a Mosaic3D floor run into the Realman_SG scene graph format.

The output schema matches Realman_SG/scene_graph.py:
  - scene_graph.json
  - scene_graph_masks.npz
  - scene_graph_readable.json

This script keeps the Realman-style SceneGraph class structure, but adapts
object creation for Mosaic3D floor runs where instance geometry is already
stored in floor_instances.json and sparse point indices are stored in
floor_instances.npz.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import re
import shutil
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np


class NodeType:
    SCENE = "scene"
    ROOM = "room"
    REGION = "region"
    OBJECT = "object"
    PART = "part"
    ASSET = "asset"


class EdgeType:
    CONTAINS = "contains"
    SPATIAL = "spatial"
    FUNCTIONAL = "functional"


class SpatialRel:
    ON = "on"
    NEXT_TO = "next-to"
    INSIDE = "inside"
    ABOVE = "above"
    BELOW = "below"


@dataclass
class ObjectRecord:
    instance_id: int
    label: str
    centroid: list[float]
    bbox_min: list[float]
    bbox_max: list[float]
    color: list[float]
    attributes: dict[str, Any]


@dataclass
class RoomRecord:
    room_id: str
    label: str
    centroid: list[float]
    bbox_min: list[float]
    bbox_max: list[float]
    polygon_xy: list[list[float]]
    attributes: dict[str, Any]


def _as_list(value: Any) -> list[Any] | None:
    if value is None:
        return None
    if hasattr(value, "tolist"):
        return value.tolist()
    return list(value)


class SceneGraph:
    """Realman-style scene graph backed by a NetworkX DiGraph."""

    _ASSET_KEYWORDS = {
        "table",
        "desk",
        "bed",
        "chair",
        "couch",
        "sofa",
        "bench",
        "shelf",
        "bookshelf",
        "cabinet",
        "wardrobe",
        "drawer",
        "tv stand",
        "shoe rack",
        "rug",
        "workstation",
        "workbench",
    }

    def __init__(self, scene_label: str = "Scene"):
        self.G = nx.DiGraph()
        self._next_id = 0
        self._masks: dict[str, np.ndarray] = {}
        self.root_id = self._add_node(
            NodeType.SCENE,
            scene_label,
            centroid=None,
            bbox_min=None,
            bbox_max=None,
            color=None,
        )

    def _new_id(self) -> str:
        nid = f"n{self._next_id}"
        self._next_id += 1
        return nid

    def _add_node(
        self,
        node_type: str,
        label: str,
        centroid: Any = None,
        bbox_min: Any = None,
        bbox_max: Any = None,
        color: Any = None,
        attributes: dict[str, Any] | None = None,
    ) -> str:
        nid = self._new_id()
        self.G.add_node(
            nid,
            type=node_type,
            label=label,
            centroid=_as_list(centroid),
            bbox_min=_as_list(bbox_min),
            bbox_max=_as_list(bbox_max),
            color=_as_list(color),
            attributes=attributes or {},
        )
        return nid

    def add_object(
        self,
        label: str,
        mask_np: np.ndarray,
        xyz: np.ndarray,
        color: Any = None,
        parent_id: str | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> str:
        """Add an OBJECT node from a dense Realman/OpenIns3D boolean mask."""
        pts = xyz[mask_np]
        centroid = pts.mean(axis=0) if len(pts) > 0 else np.zeros(3)
        if len(pts) > 20:
            bbox_min = np.percentile(pts, 2, axis=0)
            bbox_max = np.percentile(pts, 98, axis=0)
        else:
            bbox_min = pts.min(axis=0) if len(pts) > 0 else np.zeros(3)
            bbox_max = pts.max(axis=0) if len(pts) > 0 else np.zeros(3)
        nid = self.add_object_from_geometry(
            label,
            centroid,
            bbox_min,
            bbox_max,
            color=color,
            parent_id=parent_id,
            attributes=attributes,
        )
        self._masks[nid] = np.flatnonzero(mask_np).astype(np.int32, copy=False)
        return nid

    def add_object_from_geometry(
        self,
        label: str,
        centroid: Any,
        bbox_min: Any,
        bbox_max: Any,
        color: Any = None,
        parent_id: str | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> str:
        """Add an OBJECT node from Mosaic3D's precomputed centroid/bbox."""
        nid = self._add_node(
            NodeType.OBJECT,
            label,
            centroid=centroid,
            bbox_min=bbox_min,
            bbox_max=bbox_max,
            color=color,
            attributes=attributes,
        )
        parent = parent_id if parent_id else self.root_id
        self.G.add_edge(parent, nid, type=EdgeType.CONTAINS, label="contains")
        return nid

    def add_part(
        self,
        label: str,
        mask_np: np.ndarray,
        xyz: np.ndarray,
        parent_object_id: str,
        color: Any = None,
        attributes: dict[str, Any] | None = None,
    ) -> str:
        pts = xyz[mask_np]
        centroid = pts.mean(axis=0) if len(pts) > 0 else np.zeros(3)
        bbox_min = pts.min(axis=0) if len(pts) > 0 else np.zeros(3)
        bbox_max = pts.max(axis=0) if len(pts) > 0 else np.zeros(3)
        nid = self._add_node(
            NodeType.PART,
            label,
            centroid=centroid,
            bbox_min=bbox_min,
            bbox_max=bbox_max,
            color=color,
            attributes=attributes,
        )
        self._masks[nid] = np.flatnonzero(mask_np).astype(np.int32, copy=False)
        self.G.add_edge(parent_object_id, nid, type=EdgeType.CONTAINS, label="contains")
        return nid

    def add_region(self, label: str, attributes: dict[str, Any] | None = None) -> str:
        nid = self._add_node(NodeType.REGION, label, attributes=attributes)
        self.G.add_edge(self.root_id, nid, type=EdgeType.CONTAINS, label="contains")
        return nid

    def add_room(self, room: RoomRecord) -> str:
        attributes = dict(room.attributes)
        attributes["room_id"] = room.room_id
        attributes["polygon_xy"] = room.polygon_xy
        nid = self._add_node(
            NodeType.ROOM,
            room.label,
            centroid=room.centroid,
            bbox_min=room.bbox_min,
            bbox_max=room.bbox_max,
            attributes=attributes,
        )
        self.G.add_edge(self.root_id, nid, type=EdgeType.CONTAINS, label="contains")
        return nid

    def add_spatial_edge(self, src_id: str, dst_id: str, rel_label: str) -> None:
        self.G.add_edge(src_id, dst_id, type=EdgeType.SPATIAL, label=rel_label)

    def add_functional_edge(self, src_id: str, dst_id: str, rel_label: str) -> None:
        self.G.add_edge(src_id, dst_id, type=EdgeType.FUNCTIONAL, label=rel_label)

    def set_mask_indices(self, nid: str, indices: np.ndarray) -> None:
        self._masks[nid] = np.asarray(indices, dtype=np.int32)

    def get_node(self, nid: str) -> dict[str, Any]:
        return dict(self.G.nodes[nid])

    def get_mask(self, nid: str) -> np.ndarray | None:
        return self._masks.get(nid)

    def get_children(self, nid: str) -> list[str]:
        return [
            v
            for _, v, data in self.G.out_edges(nid, data=True)
            if data.get("type") == EdgeType.CONTAINS
        ]

    def get_parent(self, nid: str) -> str | None:
        for u, _, data in self.G.in_edges(nid, data=True):
            if data.get("type") == EdgeType.CONTAINS:
                return u
        return None

    def get_objects(self) -> list[tuple[str, dict[str, Any]]]:
        return [
            (nid, data)
            for nid, data in self.G.nodes(data=True)
            if data.get("type") in {NodeType.OBJECT, NodeType.ASSET}
        ]

    def get_rooms(self) -> list[tuple[str, dict[str, Any]]]:
        return [
            (nid, data)
            for nid, data in self.G.nodes(data=True)
            if data.get("type") == NodeType.ROOM
        ]

    def compute_spatial_edges(
        self,
        overlap_thresh: float = 0.3,
        proximity_thresh: float = 2.5,
        vertical_gap_thresh: float = 0.15,
        max_next_to_per_node: int = 3,
        max_above_per_node: int = 8,
    ) -> None:
        """Compute Realman-style inside/on/above/next-to object edges."""
        objects = self.get_objects()
        if len(objects) < 2:
            return

        old_edges = [
            (u, v)
            for u, v, data in self.G.edges(data=True)
            if data.get("type") == EdgeType.SPATIAL
            and self.G.nodes[u].get("type") in {NodeType.OBJECT, NodeType.ASSET}
            and self.G.nodes[v].get("type") in {NodeType.OBJECT, NodeType.ASSET}
        ]
        self.G.remove_edges_from(old_edges)

        next_to_candidates: list[tuple[float, str, str]] = []
        above_candidates: list[tuple[float, str, str]] = []

        for i, (id_a, data_a) in enumerate(objects):
            if data_a.get("bbox_min") is None or data_a.get("bbox_max") is None:
                continue
            a_min = np.asarray(data_a["bbox_min"], dtype=float)
            a_max = np.asarray(data_a["bbox_max"], dtype=float)
            a_cen = np.asarray(data_a["centroid"], dtype=float)
            a_vol = max(1e-9, float(np.prod(a_max - a_min)))
            a_area = max(1e-9, float((a_max[0] - a_min[0]) * (a_max[1] - a_min[1])))
            a_z_span = max(1e-9, float(a_max[2] - a_min[2]))

            for id_b, data_b in objects[i + 1 :]:
                if data_b.get("bbox_min") is None or data_b.get("bbox_max") is None:
                    continue
                b_min = np.asarray(data_b["bbox_min"], dtype=float)
                b_max = np.asarray(data_b["bbox_max"], dtype=float)
                b_cen = np.asarray(data_b["centroid"], dtype=float)
                b_vol = max(1e-9, float(np.prod(b_max - b_min)))
                b_area = max(
                    1e-9, float((b_max[0] - b_min[0]) * (b_max[1] - b_min[1]))
                )
                b_z_span = max(1e-9, float(b_max[2] - b_min[2]))

                h_overlap_min = np.maximum(a_min[:2], b_min[:2])
                h_overlap_max = np.minimum(a_max[:2], b_max[:2])
                h_inter = float(np.maximum(0.0, h_overlap_max - h_overlap_min).prod())
                h_iou = h_inter / (a_area + b_area - h_inter + 1e-9)

                a_z_lo, a_z_hi = float(a_min[2]), float(a_max[2])
                b_z_lo, b_z_hi = float(b_min[2]), float(b_max[2])
                z_overlap = max(0.0, min(a_z_hi, b_z_hi) - max(a_z_lo, b_z_lo))

                a_in_b = h_inter / (a_area + 1e-9)
                b_in_a = h_inter / (b_area + 1e-9)
                added = False

                if (
                    a_in_b > 0.8
                    and z_overlap / a_z_span > 0.8
                    and b_vol / a_vol > 3.0
                ):
                    self.add_spatial_edge(id_a, id_b, SpatialRel.INSIDE)
                    added = True
                elif (
                    b_in_a > 0.8
                    and z_overlap / b_z_span > 0.8
                    and a_vol / b_vol > 3.0
                ):
                    self.add_spatial_edge(id_b, id_a, SpatialRel.INSIDE)
                    added = True

                if not added and h_iou > overlap_thresh:
                    gap_a_on_b = a_z_lo - b_z_hi
                    gap_b_on_a = b_z_lo - a_z_hi

                    if abs(gap_a_on_b) < vertical_gap_thresh and a_cen[2] > b_cen[2]:
                        self.add_spatial_edge(id_a, id_b, SpatialRel.ON)
                        added = True
                    elif abs(gap_b_on_a) < vertical_gap_thresh and b_cen[2] > a_cen[2]:
                        self.add_spatial_edge(id_b, id_a, SpatialRel.ON)
                        added = True

                horiz_dist = float(np.linalg.norm(a_cen[:2] - b_cen[:2]))
                if not added and horiz_dist < proximity_thresh:
                    vert_diff = float(a_cen[2] - b_cen[2])
                    if vert_diff > 0.3:
                        above_candidates.append((horiz_dist, id_a, id_b))
                        added = True
                    elif vert_diff < -0.3:
                        above_candidates.append((horiz_dist, id_b, id_a))
                        added = True

                if not added and max_next_to_per_node > 0:
                    dist_3d = float(np.linalg.norm(a_cen - b_cen))
                    if dist_3d < proximity_thresh:
                        next_to_candidates.append((dist_3d, id_a, id_b))

        if max_above_per_node > 0:
            self._add_capped_spatial_edges(
                above_candidates, SpatialRel.ABOVE, max_above_per_node
            )
        if max_next_to_per_node > 0:
            self._add_capped_spatial_edges(
                next_to_candidates, SpatialRel.NEXT_TO, max_next_to_per_node
            )

    def _add_capped_spatial_edges(
        self,
        candidates: list[tuple[float, str, str]],
        label: str,
        max_per_source: int,
    ) -> None:
        counts: dict[str, int] = defaultdict(int)
        seen_pairs: set[tuple[str, str, str]] = set()
        for _, src, dst in sorted(candidates, key=lambda item: item[0]):
            if counts[src] >= max_per_source:
                continue
            key = (src, dst, label)
            if key in seen_pairs:
                continue
            seen_pairs.add(key)
            counts[src] += 1
            self.add_spatial_edge(src, dst, label)

    def classify_hierarchy(self) -> None:
        """Classify nodes as Realman ASSET or OBJECT."""
        has_on_top: set[str] = set()
        sits_on: dict[str, list[str]] = defaultdict(list)

        for u, v, data in self.G.edges(data=True):
            if data.get("type") == EdgeType.SPATIAL and data.get("label") == SpatialRel.ON:
                has_on_top.add(v)
                sits_on[u].append(v)

        for nid, data in self.get_objects():
            label_lower = str(data.get("label", "")).lower()
            is_keyword = any(keyword in label_lower for keyword in self._ASSET_KEYWORDS)
            node_type = NodeType.ASSET if nid in has_on_top or is_keyword else NodeType.OBJECT
            self.G.nodes[nid]["type"] = node_type
            self.G.nodes[nid].setdefault("attributes", {})["role"] = node_type
            if nid in sits_on:
                self.G.nodes[nid]["attributes"]["supported_by"] = [
                    self.G.nodes[target]["label"] for target in sits_on[nid] if target in self.G
                ]

        n_assets = sum(
            1 for _, data in self.G.nodes(data=True) if data.get("type") == NodeType.ASSET
        )
        n_objects = sum(
            1 for _, data in self.G.nodes(data=True) if data.get("type") == NodeType.OBJECT
        )
        print(f"[SceneGraph] Hierarchy: {n_assets} assets, {n_objects} objects")

    def build_rooms(self, records: list[RoomRecord]) -> dict[str, str]:
        """Add Mosaic3D room nodes under the scene root."""
        room_to_node: dict[str, str] = {}
        for record in records:
            room_to_node[record.room_id] = self.add_room(record)
        return room_to_node

    def build_from_mosaic3d_instances(
        self,
        records: list[ObjectRecord],
        room_to_node: dict[str, str] | None = None,
    ) -> dict[int, str]:
        """Populate the graph from Mosaic3D instance records."""
        instance_to_node: dict[int, str] = {}
        for record in records:
            parent_id = None
            if room_to_node:
                room_id = record.attributes.get("room_id")
                parent_id = room_to_node.get(str(room_id)) if room_id is not None else None
            nid = self.add_object_from_geometry(
                record.label,
                record.centroid,
                record.bbox_min,
                record.bbox_max,
                color=record.color,
                parent_id=parent_id,
                attributes=dict(record.attributes),
            )
            instance_to_node[record.instance_id] = nid
        return instance_to_node

    def add_room_neighbor_edges(
        self,
        room_records: list[RoomRecord],
        room_to_node: dict[str, str],
        neighbor_thresh: float,
    ) -> int:
        """Add undirected room-neighbor relationships as paired spatial edges."""
        edge_count = 0
        for i, room_a in enumerate(room_records):
            node_a = room_to_node.get(room_a.room_id)
            if node_a is None:
                continue
            for room_b in room_records[i + 1 :]:
                node_b = room_to_node.get(room_b.room_id)
                if node_b is None:
                    continue
                if room_distance_xy(room_a, room_b, neighbor_thresh) <= neighbor_thresh:
                    self.add_spatial_edge(node_a, node_b, "near")
                    self.add_spatial_edge(node_b, node_a, "near")
                    edge_count += 2
        return edge_count

    def to_dict(self) -> dict[str, Any]:
        nodes = {}
        for nid, data in self.G.nodes(data=True):
            nodes[nid] = dict(data)

        edges = []
        for u, v, data in self.G.edges(data=True):
            edges.append({"src": u, "dst": v, **data})

        return {
            "root_id": self.root_id,
            "next_id": self._next_id,
            "nodes": nodes,
            "edges": edges,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any], n_points: int | None = None) -> "SceneGraph":
        sg = cls.__new__(cls)
        sg.G = nx.DiGraph()
        sg._masks = {}
        sg._next_id = payload["next_id"]
        sg.root_id = payload["root_id"]

        for nid, node_data in payload["nodes"].items():
            node_data = dict(node_data)
            mask_indices = node_data.pop("mask_indices", None)
            sg.G.add_node(nid, **node_data)
            if mask_indices is not None:
                indices = np.asarray(mask_indices, dtype=np.int32)
                if n_points is None:
                    sg._masks[nid] = indices
                else:
                    mask = np.zeros(n_points, dtype=bool)
                    mask[indices] = True
                    sg._masks[nid] = mask

        for edge_data in payload["edges"]:
            edge_data = dict(edge_data)
            src = edge_data.pop("src")
            dst = edge_data.pop("dst")
            sg.G.add_edge(src, dst, **edge_data)
        return sg

    def save_json(self, path: Path, readable_format: str = "room-list") -> None:
        """Save Realman-style JSON, sparse mask NPZ, and readable scene JSON."""
        path = Path(path)
        payload = self.to_dict()
        data = json.dumps(payload, indent=2)
        if str(path).endswith(".gz"):
            with gzip.open(path, "wt", encoding="utf-8") as handle:
                handle.write(data)
            stem = path.name.replace(".json.gz", "")
        else:
            path.write_text(data, encoding="utf-8")
            stem = path.stem

        if self._masks:
            mask_arrays = {
                nid: np.asarray(indices, dtype=np.int32) for nid, indices in self._masks.items()
            }
            np.savez_compressed(path.with_name(f"{stem}_masks.npz"), **mask_arrays)

        readable_path = path.with_name(f"{stem}_readable.json")
        readable = (
            self.to_room_scene_format()
            if readable_format == "room-list"
            else self.to_scene_format()
        )
        readable_path.write_text(
            json.dumps(readable, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load_json(cls, path: Path, n_points: int | None = None) -> "SceneGraph":
        path = Path(path)
        if str(path).endswith(".gz"):
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                payload = json.load(handle)
        else:
            payload = load_json(path)
        return cls.from_dict(payload, n_points=n_points)

    def to_scene_format(self) -> list[dict[str, Any]]:
        result = []
        scene_data = self.G.nodes[self.root_id]
        object_nodes = self.get_objects()

        child_entries = []
        for nid, data in object_nodes:
            neighbors = []
            for u, v, edge_data in self.G.edges(data=True):
                if edge_data.get("type") != EdgeType.SPATIAL:
                    continue
                if u == nid:
                    neighbors.append(
                        {
                            "node": self.G.nodes[v].get("label", v),
                            "relation": edge_data.get("label", "near"),
                        }
                    )
                elif v == nid:
                    rel = edge_data.get("label", "near")
                    if rel == SpatialRel.ON:
                        rel = SpatialRel.BELOW
                    elif rel == SpatialRel.ABOVE:
                        rel = SpatialRel.BELOW
                    elif rel == SpatialRel.BELOW:
                        rel = SpatialRel.ABOVE
                    elif rel == SpatialRel.INSIDE:
                        rel = "contains"
                    neighbors.append(
                        {
                            "node": self.G.nodes[u].get("label", u),
                            "relation": rel,
                        }
                    )

            part_entries = []
            for child_id in self.get_children(nid):
                child_data = self.G.nodes[child_id]
                if child_data.get("type") != NodeType.PART:
                    continue
                part_entries.append(
                    {
                        "node_name": child_data["label"],
                        "node_type": "part",
                        "description": child_data.get("attributes", {}).get(
                            "description", ""
                        ),
                    }
                )

            child_entries.append(
                {
                    "node_name": data["label"],
                    "node_type": "object",
                    "description": data.get("attributes", {}).get("description", ""),
                    "neighbor_nodes": neighbors,
                    "child_nodes": part_entries,
                }
            )

        scene_entry = {
            "node_name": scene_data.get("label", "Scene"),
            "node_type": "scene",
            "description": "",
            "neighbor_nodes": [data["label"] for _, data in object_nodes],
            "child_nodes": child_entries,
        }
        result.append(scene_entry)

        for nid, data in self.G.nodes(data=True):
            if data.get("type") != NodeType.REGION:
                continue
            region_children = self.get_children(nid)
            region_entry = {
                "node_name": data["label"],
                "node_type": "region",
                "description": data.get("attributes", {}).get("description", ""),
                "neighbor_nodes": [
                    self.G.nodes[child_id].get("label", child_id)
                    for child_id in region_children
                ],
                "child_nodes": [
                    {
                        "node_name": self.G.nodes[child_id]["label"],
                        "node_type": self.G.nodes[child_id].get("type", NodeType.OBJECT),
                        "description": self.G.nodes[child_id]
                        .get("attributes", {})
                        .get("description", ""),
                        "child_nodes": [],
                    }
                    for child_id in region_children
                ],
            }
            result.append(region_entry)

        return result

    def to_room_scene_format(self) -> list[dict[str, Any]]:
        """Export as top-level rooms with nested assets/objects.

        Shape:
        [
          {
            "node_name": "room_001",
            "node_type": "room",
            "neighbor_nodes": ["room_002"],
            "child_nodes": [
              {
                "node_name": "table_1",
                "node_type": "asset",
                "child_nodes": [
                  {"node_name": "computer_1", "node_type": "object"}
                ]
              }
            ]
          }
        ]
        """
        rooms = self.get_rooms()
        if not rooms:
            return self.to_scene_format()

        room_order = {
            nid: int(data.get("attributes", {}).get("room_order", idx))
            for idx, (nid, data) in enumerate(rooms)
        }
        room_neighbors: dict[str, set[str]] = defaultdict(set)
        for src, dst, edge_data in self.G.edges(data=True):
            if edge_data.get("type") != EdgeType.SPATIAL:
                continue
            if self.G.nodes[src].get("type") == NodeType.ROOM and self.G.nodes[dst].get(
                "type"
            ) == NodeType.ROOM:
                room_neighbors[src].add(dst)

        room_children: dict[str, list[str]] = defaultdict(list)
        object_room: dict[str, str] = {}
        for room_id, _ in rooms:
            for child_id in self.get_children(room_id):
                child_type = self.G.nodes[child_id].get("type")
                if child_type in {NodeType.OBJECT, NodeType.ASSET}:
                    room_children[room_id].append(child_id)
                    object_room[child_id] = room_id

        support_children: dict[str, list[str]] = defaultdict(list)
        supported_by: dict[str, str] = {}
        for src, dst, edge_data in self.G.edges(data=True):
            if edge_data.get("type") != EdgeType.SPATIAL:
                continue
            if edge_data.get("label") != SpatialRel.ON:
                continue
            if src not in object_room or dst not in object_room:
                continue
            if object_room[src] != object_room[dst]:
                continue
            if src not in supported_by:
                supported_by[src] = dst
                support_children[dst].append(src)

        def node_name(nid: str) -> str:
            data = self.G.nodes[nid]
            attrs = data.get("attributes", {})
            raw = (
                attrs.get("edited_label")
                or attrs.get("original_label")
                or attrs.get("room_id")
                or data.get("label")
                or nid
            )
            return re.sub(r"\s+", "_", str(raw).strip())

        def sort_key(nid: str) -> list[Any]:
            text = node_name(nid)
            parts = re.split(r"(\d+)", text)
            return [int(part) if part.isdigit() else part.lower() for part in parts]

        def object_entry(nid: str, seen: set[str] | None = None) -> dict[str, Any]:
            seen = set(seen or ())
            if nid in seen:
                return {
                    "node_name": node_name(nid),
                    "node_type": self.G.nodes[nid].get("type", NodeType.OBJECT),
                }
            seen.add(nid)

            data = self.G.nodes[nid]
            node_type = (
                "asset" if data.get("type") == NodeType.ASSET else "object"
            )
            entry: dict[str, Any] = {
                "node_name": node_name(nid),
                "node_type": node_type,
            }
            description = (
                data.get("attributes", {}).get("description")
                or data.get("attributes", {}).get("caption")
            )
            if description:
                entry["description"] = description

            children = [
                child
                for child in sorted(support_children.get(nid, []), key=sort_key)
                if child not in seen
            ]
            if children or node_type == "asset":
                entry["child_nodes"] = [
                    object_entry(child, seen=set(seen)) for child in children
                ]
            return entry

        result = []
        for room_id, room_data in sorted(rooms, key=lambda item: room_order[item[0]]):
            direct_children = [
                child
                for child in room_children.get(room_id, [])
                if supported_by.get(child) not in object_room
                or object_room.get(supported_by.get(child)) != room_id
            ]
            room_entry = {
                "node_name": node_name(room_id),
                "node_type": "room",
                "neighbor_nodes": [
                    node_name(neighbor)
                    for neighbor in sorted(room_neighbors.get(room_id, []), key=sort_key)
                ],
                "child_nodes": [
                    object_entry(child) for child in sorted(direct_children, key=sort_key)
                ],
            }
            result.append(room_entry)

        return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Mosaic3D floor instances to Realman_SG scene graph files."
    )
    parser.add_argument(
        "run_dir",
        type=Path,
        help="Mosaic3D floor run directory, e.g. runs/building_1B_floor",
    )
    parser.add_argument(
        "--source",
        choices=["latest-edit", "stitched"],
        default="latest-edit",
        help="Use the newest 05_interactive_edits session when available, or 04_stitched.",
    )
    parser.add_argument(
        "--edit-dir",
        type=Path,
        default=None,
        help="Explicit 05_interactive_edits/session_* directory to use.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to write outputs. Default: <run_dir>/04_stitched.",
    )
    parser.add_argument(
        "--scene-label",
        default="Scene",
        help="Root scene label for the Realman-style graph.",
    )
    parser.add_argument(
        "--rooms-json",
        type=Path,
        default=None,
        help="Room metadata JSON. Default: <run_dir>/02_rooms_refined/rooms.json, then 02_rooms/rooms.json.",
    )
    parser.add_argument(
        "--room-neighbor-thresh",
        type=float,
        default=0.50,
        help="Maximum XY polygon gap in meters for room neighbor links.",
    )
    parser.add_argument(
        "--readable-format",
        choices=["room-list", "scene-root"],
        default="room-list",
        help="Format for scene_graph_readable.json. room-list matches room/asset/object nesting.",
    )
    parser.add_argument(
        "--include-inactive",
        action="store_true",
        help="Include instances whose active flag is false.",
    )
    parser.add_argument(
        "--overlap-thresh",
        type=float,
        default=0.30,
        help="Horizontal IoU threshold for 'on' relationships.",
    )
    parser.add_argument(
        "--proximity-thresh",
        type=float,
        default=2.50,
        help="Distance threshold for nearby/above/next-to relationships.",
    )
    parser.add_argument(
        "--vertical-gap-thresh",
        type=float,
        default=0.15,
        help="Maximum vertical gap for 'on' relationships.",
    )
    parser.add_argument(
        "--max-next-to-per-node",
        type=int,
        default=3,
        help="Cap 'next-to' edges per source node to avoid huge floor graphs. Use 0 to disable.",
    )
    parser.add_argument(
        "--max-above-per-node",
        type=int,
        default=8,
        help="Cap geometry-only 'above' edges per source node. Use 0 to disable.",
    )
    parser.add_argument(
        "--backup-existing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Backup existing scene_graph* files before writing.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build and report counts without writing outputs.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def latest_edit_dir(run_dir: Path) -> Path | None:
    edits_root = run_dir / "05_interactive_edits"
    if not edits_root.exists():
        return None
    sessions = sorted(
        [
            path
            for path in edits_root.iterdir()
            if path.is_dir() and (path / "floor_instances.json").exists()
        ]
    )
    return sessions[-1] if sessions else None


def resolve_source(run_dir: Path, source: str, edit_dir: Path | None) -> tuple[Path, Path, Path]:
    stitched = run_dir / "04_stitched"
    if edit_dir is not None:
        src = edit_dir.expanduser().resolve()
    elif source == "latest-edit":
        src = latest_edit_dir(run_dir) or stitched
    else:
        src = stitched

    instances_json = src / "floor_instances.json"
    instances_npz = src / "floor_instances.npz"
    if not instances_json.exists():
        raise FileNotFoundError(f"missing instances JSON: {instances_json}")
    if not instances_npz.exists():
        raise FileNotFoundError(f"missing instances NPZ: {instances_npz}")

    return src, instances_json, instances_npz


def resolve_rooms_json(run_dir: Path, rooms_json: Path | None) -> Path | None:
    if rooms_json is not None:
        path = rooms_json.expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"rooms_json not found: {path}")
        return path

    candidates = [
        run_dir / "02_rooms_refined" / "rooms.json",
        run_dir / "02_rooms" / "rooms.json",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def as_float_list(value: Any, *, size: int = 3) -> list[float] | None:
    if value is None:
        return None
    if len(value) != size:
        return None
    out = []
    for item in value:
        try:
            item_float = float(item)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(item_float):
            return None
        out.append(item_float)
    return out


def normalize_label(raw: Any, fallback: str) -> str:
    text = str(raw or "").strip() or fallback
    text = text.replace("-", " ")
    match = re.match(r"^(.*?)[_ ]+(\d+)$", text)
    if match:
        base = re.sub(r"[_]+", " ", match.group(1)).strip()
        suffix = match.group(2)
        return f"{base.title()} {suffix}"
    return re.sub(r"[_]+", " ", text).strip().title()


def ensure_unique_labels(records: list[dict[str, Any]]) -> list[str]:
    counts: dict[str, int] = defaultdict(int)
    labels = []
    for inst in records:
        raw = (
            inst.get("edited_label")
            or inst.get("original_label")
            or inst.get("semantic_label")
            or f"instance_{inst.get('instance_id', len(labels))}"
        )
        label = normalize_label(raw, f"Instance {inst.get('instance_id', len(labels))}")
        counts[label] += 1
        if counts[label] == 1:
            labels.append(label)
        else:
            labels.append(f"{label} {counts[label]}")
    return labels


def color_for_index(index: int, total: int) -> list[float]:
    h = (index / max(total, 1) * 0.61803398875) % 1.0
    s = 0.90
    v = 0.95
    i = int(h * 6.0)
    f = h * 6.0 - i
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    i %= 6
    if i == 0:
        rgb = (v, t, p)
    elif i == 1:
        rgb = (q, v, p)
    elif i == 2:
        rgb = (p, v, t)
    elif i == 3:
        rgb = (p, q, v)
    elif i == 4:
        rgb = (t, p, v)
    else:
        rgb = (v, p, q)
    return [float(channel) for channel in rgb]


def build_room_records(
    rooms_payload: dict[str, Any] | None,
    object_records: list[ObjectRecord],
) -> list[RoomRecord]:
    raw_rooms = list((rooms_payload or {}).get("rooms", []))
    records: list[RoomRecord] = []
    seen_room_ids: set[str] = set()

    for idx, room in enumerate(raw_rooms):
        room_id = str(room.get("room_id") or room.get("id") or room.get("name") or "").strip()
        if not room_id:
            continue
        polygon_xy = parse_polygon_xy(room.get("polygon_xy"))
        bbox_min_xy = as_float_list(room.get("bbox_min_xy"), size=2)
        bbox_max_xy = as_float_list(room.get("bbox_max_xy"), size=2)
        if (bbox_min_xy is None or bbox_max_xy is None) and polygon_xy:
            bbox_min_xy, bbox_max_xy = polygon_bbox_xy(polygon_xy)
        if bbox_min_xy is None or bbox_max_xy is None:
            bbox_min_xy = [0.0, 0.0]
            bbox_max_xy = [0.0, 0.0]

        position = room.get("position") or {}
        if isinstance(position, dict) and "x" in position and "y" in position:
            centroid = [
                float(position["x"]),
                float(position["y"]),
                float(position.get("z", 0.0)),
            ]
        elif polygon_xy:
            centroid = polygon_centroid_xyz(polygon_xy)
        else:
            centroid = [
                (bbox_min_xy[0] + bbox_max_xy[0]) / 2.0,
                (bbox_min_xy[1] + bbox_max_xy[1]) / 2.0,
                0.0,
            ]

        label = str(room.get("name") or room_id)
        attributes = {
            "room_order": idx,
            "room_type": room.get("room_type") or room.get("type") or "room",
            "source": room.get("source"),
            "bbox_min_xy": bbox_min_xy,
            "bbox_max_xy": bbox_max_xy,
        }
        attributes = {key: value for key, value in attributes.items() if value is not None}
        records.append(
            RoomRecord(
                room_id=room_id,
                label=label,
                centroid=centroid,
                bbox_min=[bbox_min_xy[0], bbox_min_xy[1], 0.0],
                bbox_max=[bbox_max_xy[0], bbox_max_xy[1], 0.0],
                polygon_xy=polygon_xy,
                attributes=attributes,
            )
        )
        seen_room_ids.add(room_id)

    missing_rooms: dict[str, list[ObjectRecord]] = defaultdict(list)
    for record in object_records:
        room_id = record.attributes.get("room_id")
        if room_id is None:
            room_id = "unassigned"
        room_id = str(room_id)
        if room_id not in seen_room_ids:
            missing_rooms[room_id].append(record)

    next_order = len(records)
    for room_id in sorted(missing_rooms):
        grouped = missing_rooms[room_id]
        mins = np.asarray([record.bbox_min for record in grouped], dtype=float)
        maxs = np.asarray([record.bbox_max for record in grouped], dtype=float)
        centroid = np.asarray([record.centroid for record in grouped], dtype=float).mean(axis=0)
        bbox_min = mins.min(axis=0).tolist()
        bbox_max = maxs.max(axis=0).tolist()
        records.append(
            RoomRecord(
                room_id=room_id,
                label=room_id,
                centroid=centroid.tolist(),
                bbox_min=[bbox_min[0], bbox_min[1], 0.0],
                bbox_max=[bbox_max[0], bbox_max[1], 0.0],
                polygon_xy=[],
                attributes={
                    "room_order": next_order,
                    "room_type": "room",
                    "synthetic": True,
                    "bbox_min_xy": [bbox_min[0], bbox_min[1]],
                    "bbox_max_xy": [bbox_max[0], bbox_max[1]],
                },
            )
        )
        next_order += 1

    return records


def parse_polygon_xy(value: Any) -> list[list[float]]:
    if not value:
        return []
    polygon = []
    for point in value:
        if point is None or len(point) < 2:
            continue
        try:
            x = float(point[0])
            y = float(point[1])
        except (TypeError, ValueError):
            continue
        if math.isfinite(x) and math.isfinite(y):
            polygon.append([x, y])
    return polygon


def polygon_bbox_xy(polygon_xy: list[list[float]]) -> tuple[list[float], list[float]]:
    coords = np.asarray(polygon_xy, dtype=float)
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    return mins.tolist(), maxs.tolist()


def polygon_centroid_xyz(polygon_xy: list[list[float]]) -> list[float]:
    coords = np.asarray(polygon_xy, dtype=float)
    centroid = coords.mean(axis=0)
    return [float(centroid[0]), float(centroid[1]), 0.0]


def room_distance_xy(room_a: RoomRecord, room_b: RoomRecord, limit: float) -> float:
    gap = bbox_gap_xy(room_a, room_b)
    if gap > limit:
        return math.inf
    if len(room_a.polygon_xy) < 2 or len(room_b.polygon_xy) < 2:
        return gap
    return polygon_distance_xy(room_a.polygon_xy, room_b.polygon_xy, limit)


def bbox_gap_xy(room_a: RoomRecord, room_b: RoomRecord) -> float:
    a_min = room_a.bbox_min
    a_max = room_a.bbox_max
    b_min = room_b.bbox_min
    b_max = room_b.bbox_max
    dx = max(0.0, max(a_min[0], b_min[0]) - min(a_max[0], b_max[0]))
    dy = max(0.0, max(a_min[1], b_min[1]) - min(a_max[1], b_max[1]))
    return math.hypot(dx, dy)


def polygon_distance_xy(
    polygon_a: list[list[float]],
    polygon_b: list[list[float]],
    limit: float,
) -> float:
    best = math.inf
    for seg_a in polygon_segments_xy(polygon_a):
        for seg_b in polygon_segments_xy(polygon_b):
            distance = segment_distance_xy(seg_a[0], seg_a[1], seg_b[0], seg_b[1])
            if distance < best:
                best = distance
            if best <= limit:
                return best
    return best


def polygon_segments_xy(polygon_xy: list[list[float]]) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    points = [(float(point[0]), float(point[1])) for point in polygon_xy]
    if len(points) < 2:
        return []
    if points[0] != points[-1]:
        points.append(points[0])
    return list(zip(points, points[1:]))


def segment_distance_xy(
    a: tuple[float, float],
    b: tuple[float, float],
    c: tuple[float, float],
    d: tuple[float, float],
) -> float:
    if segments_intersect_xy(a, b, c, d):
        return 0.0
    return min(
        point_segment_distance_xy(a, c, d),
        point_segment_distance_xy(b, c, d),
        point_segment_distance_xy(c, a, b),
        point_segment_distance_xy(d, a, b),
    )


def point_segment_distance_xy(
    point: tuple[float, float],
    start: tuple[float, float],
    end: tuple[float, float],
) -> float:
    vx = end[0] - start[0]
    vy = end[1] - start[1]
    wx = point[0] - start[0]
    wy = point[1] - start[1]
    length_sq = vx * vx + vy * vy
    if length_sq <= 1e-12:
        return math.hypot(point[0] - start[0], point[1] - start[1])
    t = max(0.0, min(1.0, (wx * vx + wy * vy) / length_sq))
    closest = (start[0] + t * vx, start[1] + t * vy)
    return math.hypot(point[0] - closest[0], point[1] - closest[1])


def segments_intersect_xy(
    a: tuple[float, float],
    b: tuple[float, float],
    c: tuple[float, float],
    d: tuple[float, float],
) -> bool:
    o1 = orientation_xy(a, b, c)
    o2 = orientation_xy(a, b, d)
    o3 = orientation_xy(c, d, a)
    o4 = orientation_xy(c, d, b)
    if o1 * o2 < 0.0 and o3 * o4 < 0.0:
        return True
    return (
        almost_on_segment_xy(a, b, c, o1)
        or almost_on_segment_xy(a, b, d, o2)
        or almost_on_segment_xy(c, d, a, o3)
        or almost_on_segment_xy(c, d, b, o4)
    )


def orientation_xy(
    a: tuple[float, float],
    b: tuple[float, float],
    c: tuple[float, float],
) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def almost_on_segment_xy(
    a: tuple[float, float],
    b: tuple[float, float],
    p: tuple[float, float],
    orientation: float,
) -> bool:
    return (
        abs(orientation) < 1e-9
        and min(a[0], b[0]) - 1e-9 <= p[0] <= max(a[0], b[0]) + 1e-9
        and min(a[1], b[1]) - 1e-9 <= p[1] <= max(a[1], b[1]) + 1e-9
    )


def build_object_records(
    instances_payload: dict[str, Any],
    include_inactive: bool,
) -> list[ObjectRecord]:
    raw_instances = list(instances_payload.get("instances", []))
    active_records = [
        inst
        for inst in raw_instances
        if include_inactive or bool(inst.get("active", True))
    ]
    labels = ensure_unique_labels(active_records)
    objects: list[ObjectRecord] = []
    total = len(active_records)

    for idx, (inst, label) in enumerate(zip(active_records, labels), start=1):
        instance_id = int(inst["instance_id"])
        centroid = as_float_list(inst.get("centroid"))
        bbox_min = as_float_list(inst.get("bbox_min"))
        bbox_max = as_float_list(inst.get("bbox_max"))
        if centroid is None or bbox_min is None or bbox_max is None:
            raise ValueError(
                f"instance {instance_id} lacks usable centroid/bbox; "
                "this converter expects floor_instances.json geometry"
            )

        attrs: dict[str, Any] = {
            "instance_id": instance_id,
            "semantic_label": inst.get("semantic_label"),
            "original_label": inst.get("original_label"),
            "edited_label": inst.get("edited_label"),
            "room_id": inst.get("room_id"),
            "points": inst.get("points"),
            "mean_score": inst.get("mean_score"),
            "object_label_score": inst.get("object_label_score"),
            "object_label_margin": inst.get("object_label_margin"),
            "source": inst.get("source"),
            "qa_status": inst.get("qa_status"),
            "qa_reasons": inst.get("qa_reasons"),
        }
        attrs = {key: value for key, value in attrs.items() if value is not None}
        objects.append(
            ObjectRecord(
                instance_id=instance_id,
                label=label,
                centroid=centroid,
                bbox_min=bbox_min,
                bbox_max=bbox_max,
                color=color_for_index(idx - 1, total),
                attributes=attrs,
            )
        )
    return objects


def build_mask_arrays(
    instance_index: np.ndarray,
    instance_to_node: dict[int, str],
) -> dict[str, np.ndarray]:
    assigned_points = np.flatnonzero(instance_index >= 0)
    assigned_ids = instance_index[assigned_points]
    keep_mask = np.isin(assigned_ids, np.fromiter(instance_to_node.keys(), dtype=np.int32))
    assigned_points = assigned_points[keep_mask].astype(np.int32, copy=False)
    assigned_ids = assigned_ids[keep_mask].astype(np.int32, copy=False)

    if len(assigned_points) == 0:
        return {}

    order = np.argsort(assigned_ids, kind="stable")
    assigned_ids = assigned_ids[order]
    assigned_points = assigned_points[order]

    masks: dict[str, np.ndarray] = {}
    split_positions = np.flatnonzero(np.diff(assigned_ids)) + 1
    starts = np.concatenate(([0], split_positions))
    ends = np.concatenate((split_positions, [len(assigned_ids)]))
    for start, end in zip(starts, ends):
        instance_id = int(assigned_ids[start])
        node_id = instance_to_node.get(instance_id)
        if node_id:
            masks[node_id] = assigned_points[start:end].astype(np.int32, copy=False)
    return masks


def backup_outputs(output_dir: Path) -> None:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    for name in [
        "scene_graph.json",
        "scene_graph_masks.npz",
        "scene_graph_readable.json",
        "scene_graph_conversion_summary.json",
    ]:
        path = output_dir / name
        if path.exists():
            backup = output_dir / f"{name}.bak_{stamp}"
            shutil.copy2(path, backup)
            print(f"[backup] {path.name} -> {backup.name}")


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"run_dir not found: {run_dir}")

    source_dir, instances_json, instances_npz = resolve_source(
        run_dir, args.source, args.edit_dir
    )
    output_dir = (
        args.output_dir.expanduser().resolve() if args.output_dir else run_dir / "04_stitched"
    )
    rooms_json = resolve_rooms_json(run_dir, args.rooms_json)
    predictions_npz = run_dir / "04_stitched" / "floor_mosaic3d_stitched_predictions.npz"
    if not predictions_npz.exists():
        raise FileNotFoundError(f"missing stitched predictions: {predictions_npz}")

    print(f"[input] run_dir={run_dir}")
    print(f"[input] source_dir={source_dir}")
    print(f"[input] instances_json={instances_json}")
    print(f"[input] instances_npz={instances_npz}")
    print(f"[input] rooms_json={rooms_json}")
    print(f"[output] output_dir={output_dir}")

    instances_payload = load_json(instances_json)
    records = build_object_records(instances_payload, include_inactive=args.include_inactive)
    rooms_payload = load_json(rooms_json) if rooms_json else None
    room_records = build_room_records(rooms_payload, records)
    print(f"[build] active objects={len(records)}")
    print(f"[build] rooms={len(room_records)}")

    graph = SceneGraph(scene_label=args.scene_label)
    room_to_node = graph.build_rooms(room_records)
    room_neighbor_edge_count = graph.add_room_neighbor_edges(
        room_records,
        room_to_node,
        neighbor_thresh=args.room_neighbor_thresh,
    )
    instance_to_node = graph.build_from_mosaic3d_instances(records, room_to_node=room_to_node)
    contains_edges = [
        edge
        for edge in graph.to_dict()["edges"]
        if edge.get("type") == EdgeType.CONTAINS
    ]

    graph.compute_spatial_edges(
        overlap_thresh=args.overlap_thresh,
        proximity_thresh=args.proximity_thresh,
        vertical_gap_thresh=args.vertical_gap_thresh,
        max_next_to_per_node=args.max_next_to_per_node,
        max_above_per_node=args.max_above_per_node,
    )
    graph.classify_hierarchy()

    index_npz = np.load(instances_npz)
    instance_index = np.asarray(index_npz["instance_index"], dtype=np.int32)
    pred = np.load(predictions_npz)
    coord_shape = tuple(pred["coord"].shape)
    if instance_index.shape[0] != coord_shape[0]:
        raise ValueError(
            f"instance_index length {instance_index.shape[0]} != coord length {coord_shape[0]}"
        )
    mask_arrays = build_mask_arrays(instance_index, instance_to_node)
    for node_id, indices in mask_arrays.items():
        graph.set_mask_indices(node_id, indices)

    missing_masks = sorted(set(instance_to_node.values()) - set(mask_arrays))
    if missing_masks:
        print(f"[warn] missing masks for {len(missing_masks)} nodes")

    graph_payload = graph.to_dict()
    spatial_edges = [
        edge
        for edge in graph_payload["edges"]
        if edge.get("type") == EdgeType.SPATIAL
    ]
    object_spatial_edges = [
        edge
        for edge in spatial_edges
        if graph_payload["nodes"][edge["src"]].get("type") in {NodeType.OBJECT, NodeType.ASSET}
        and graph_payload["nodes"][edge["dst"]].get("type") in {NodeType.OBJECT, NodeType.ASSET}
    ]
    summary = {
        "converted_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_dir": str(source_dir),
        "instances_json": str(instances_json),
        "instances_npz": str(instances_npz),
        "rooms_json": str(rooms_json) if rooms_json else None,
        "predictions_npz": str(predictions_npz),
        "room_count": len(room_records),
        "object_count": len(records),
        "node_count": len(graph_payload["nodes"]),
        "contains_edge_count": len(contains_edges),
        "spatial_edge_count": len(spatial_edges),
        "object_spatial_edge_count": len(object_spatial_edges),
        "room_neighbor_edge_count": room_neighbor_edge_count,
        "mask_count": len(mask_arrays),
        "point_count": int(instance_index.shape[0]),
        "relationship_params": {
            "overlap_thresh": args.overlap_thresh,
            "proximity_thresh": args.proximity_thresh,
            "vertical_gap_thresh": args.vertical_gap_thresh,
            "max_next_to_per_node": args.max_next_to_per_node,
            "max_above_per_node": args.max_above_per_node,
            "room_neighbor_thresh": args.room_neighbor_thresh,
        },
        "readable_format": args.readable_format,
        "format": "Realman_SG SceneGraph JSON + room/asset/object readable scene format",
        "implementation": "Mosaic3D SceneGraph class modeled after Realman_SG/scene_graph.py",
    }

    print(
        "[build] nodes={node_count} contains={contains_edge_count} "
        "spatial={object_spatial_edge_count} room_neighbors={room_neighbor_edge_count} "
        "masks={mask_count}".format(**summary)
    )

    if args.dry_run:
        print("[dry-run] no files written")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    if args.backup_existing:
        backup_outputs(output_dir)

    graph.save_json(output_dir / "scene_graph.json", readable_format=args.readable_format)
    write_json(output_dir / "scene_graph_conversion_summary.json", summary)

    print(f"[write] {output_dir / 'scene_graph.json'}")
    print(f"[write] {output_dir / 'scene_graph_masks.npz'}")
    print(f"[write] {output_dir / 'scene_graph_readable.json'}")
    print(f"[write] {output_dir / 'scene_graph_conversion_summary.json'}")


if __name__ == "__main__":
    main()
