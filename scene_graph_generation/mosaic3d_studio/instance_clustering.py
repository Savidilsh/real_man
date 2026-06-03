"""Spatial instance clustering for Mosaic3D semantic point labels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


NEIGHBOR_OFFSETS = tuple(
    (dx, dy, dz)
    for dx in (-1, 0, 1)
    for dy in (-1, 0, 1)
    for dz in (-1, 0, 1)
    if (dx, dy, dz) != (0, 0, 0)
)


@dataclass
class ClusterResult:
    point_instance_index: np.ndarray
    instances: list[dict]


def cluster_semantic_instances(
    coord: np.ndarray,
    pred_index: np.ndarray,
    score: np.ndarray,
    semantic_labels: list[str],
    label_indices: Iterable[int],
    voxel_size: float,
    min_points: int,
    max_instances_per_label: int,
) -> ClusterResult:
    """Split selected semantic labels into connected voxel components.

    This is intentionally dependency-free. Points are quantized to a voxel grid,
    occupied voxels are grouped by 26-neighbor connectivity, then all points in
    each large enough component become one instance.
    """

    coord = np.asarray(coord, dtype=np.float32)
    pred_index = np.asarray(pred_index, dtype=np.int64)
    score = np.asarray(score, dtype=np.float32)

    if coord.ndim != 2 or coord.shape[1] != 3:
        raise ValueError("coord must have shape (N, 3)")
    if len(pred_index) != len(coord) or len(score) != len(coord):
        raise ValueError("coord, pred_index, and score must have the same length")

    voxel_size = float(voxel_size)
    min_points = int(min_points)
    max_instances_per_label = int(max_instances_per_label)
    if voxel_size <= 0:
        raise ValueError("voxel_size must be greater than 0")
    if min_points <= 0:
        raise ValueError("min_points must be greater than 0")
    if max_instances_per_label <= 0:
        raise ValueError("max_instances_per_label must be greater than 0")

    point_instance_index = np.full(len(coord), -1, dtype=np.int32)
    instances: list[dict] = []

    for label_index in label_indices:
        label_index = int(label_index)
        if label_index < 0 or label_index >= len(semantic_labels):
            continue

        label_point_indices = np.flatnonzero(pred_index == label_index)
        if len(label_point_indices) < min_points:
            continue

        local_coord = coord[label_point_indices]
        local_component, component_counts = _connected_voxel_components(local_coord, voxel_size)
        valid_components = np.flatnonzero(component_counts >= min_points)
        if len(valid_components) == 0:
            continue

        valid_components = sorted(
            (int(cid) for cid in valid_components),
            key=lambda cid: int(component_counts[cid]),
            reverse=True,
        )[:max_instances_per_label]

        for rank, component_id in enumerate(valid_components, start=1):
            local_mask = local_component == component_id
            local_indices = np.flatnonzero(local_mask)
            if len(local_indices) < min_points:
                continue

            global_indices = label_point_indices[local_indices]
            instance_id = len(instances)
            point_instance_index[global_indices] = instance_id

            pts = coord[global_indices]
            label_name = semantic_labels[label_index]
            instance_name = f"{label_name}_{rank}"
            instances.append(
                {
                    "instance_id": instance_id,
                    "semantic_label_index": label_index,
                    "semantic_label": label_name,
                    "original_label": instance_name,
                    "edited_label": instance_name,
                    "active": True,
                    "points": int(len(global_indices)),
                    "mean_score": float(np.mean(score[global_indices])) if len(global_indices) else 0.0,
                    "centroid": [float(x) for x in np.mean(pts, axis=0).tolist()],
                    "bbox_min": [float(x) for x in np.min(pts, axis=0).tolist()],
                    "bbox_max": [float(x) for x in np.max(pts, axis=0).tolist()],
                }
            )

    return ClusterResult(point_instance_index=point_instance_index, instances=instances)


def _connected_voxel_components(points: np.ndarray, voxel_size: float) -> tuple[np.ndarray, np.ndarray]:
    if len(points) == 0:
        return np.empty((0,), dtype=np.int32), np.empty((0,), dtype=np.int64)

    origin = np.min(points, axis=0)
    voxels = np.floor((points - origin) / voxel_size).astype(np.int64)
    unique_voxels, inverse = np.unique(voxels, axis=0, return_inverse=True)

    voxel_lookup = {tuple(cell.tolist()): idx for idx, cell in enumerate(unique_voxels)}
    voxel_component = np.full(len(unique_voxels), -1, dtype=np.int32)
    component_id = 0

    for start in range(len(unique_voxels)):
        if voxel_component[start] >= 0:
            continue
        voxel_component[start] = component_id
        queue = [start]
        cursor = 0

        while cursor < len(queue):
            current = queue[cursor]
            cursor += 1
            cx, cy, cz = unique_voxels[current]
            for dx, dy, dz in NEIGHBOR_OFFSETS:
                neighbor = voxel_lookup.get((int(cx + dx), int(cy + dy), int(cz + dz)))
                if neighbor is None or voxel_component[neighbor] >= 0:
                    continue
                voxel_component[neighbor] = component_id
                queue.append(neighbor)

        component_id += 1

    point_component = voxel_component[inverse]
    component_counts = np.bincount(point_component, minlength=component_id).astype(np.int64)
    return point_component.astype(np.int32), component_counts
