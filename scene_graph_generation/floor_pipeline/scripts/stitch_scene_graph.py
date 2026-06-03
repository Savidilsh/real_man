#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

from common import palette, points_in_polygon_xy, read_json, write_json, write_point_cloud


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "mosaic3d_studio"))
from instance_clustering import cluster_semantic_instances  # noqa: E402


DEFAULT_INSTANCE_LABELS = (
    "door,window,chair,table,desk,workbench,countertop,cabinet,shelf,equipment rack,"
    "cart,box,whiteboard,computer monitor,computer,couch,robot,robot arm,robot base,"
    "mobile robot,wheeled rover,humanoid robot,microwave,trash bin,sink,water dispenser,"
    "refrigerator,pipe,cable,solar panel"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stitch chunk predictions into one global floor prediction and scene graph."
    )
    parser.add_argument("--preprocess-manifest", required=True)
    parser.add_argument("--chunks-manifest", required=True)
    parser.add_argument("--batch-manifest", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--instance-labels", default=DEFAULT_INSTANCE_LABELS)
    parser.add_argument("--instance-voxel-size", type=float, default=0.18)
    parser.add_argument("--instance-min-points", type=int, default=250)
    parser.add_argument(
        "--instance-min-mean-score",
        type=float,
        default=0.0,
        help="Reject final object instances below this mean stitched score. Use 0 to disable.",
    )
    parser.add_argument(
        "--object-relabel",
        action="store_true",
        help=(
            "After instance masks are built, relabel each object mask from aggregated "
            "per-chunk Mosaic3D logits instead of only inheriting the top-1 point label."
        ),
    )
    parser.add_argument(
        "--object-relabel-candidates",
        default="all",
        help=(
            "Labels considered during object-level relabeling: 'all', 'instance', "
            "or a comma-separated label list."
        ),
    )
    parser.add_argument(
        "--object-relabel-top-k",
        type=int,
        default=5,
        help="Number of object-level label candidates to store per instance.",
    )
    parser.add_argument(
        "--object-relabel-min-score",
        type=float,
        default=0.0,
        help="Only apply a relabel if the best object-level label score is at least this value.",
    )
    parser.add_argument(
        "--object-relabel-min-margin",
        type=float,
        default=0.0,
        help="Only apply a relabel if best score minus second-best score is at least this value.",
    )
    parser.add_argument(
        "--object-relabel-top-point-fraction",
        type=float,
        default=1.0,
        help=(
            "Use only the top fraction of reliable point observations inside each mask "
            "for object-level relabeling. Reliability is point confidence times chunk center weight."
        ),
    )
    parser.add_argument(
        "--object-relabel-min-point-score",
        type=float,
        default=0.0,
        help="Ignore point observations below this Mosaic3D top-1 confidence during object relabeling.",
    )
    parser.add_argument(
        "--object-relabel-min-center-weight",
        type=float,
        default=0.0,
        help=(
            "Ignore point observations below this chunk center weight during object relabeling. "
            "Use this to discard halo/boundary observations."
        ),
    )
    parser.add_argument(
        "--object-relabel-geometry-filter",
        action="store_true",
        help=(
            "Before accepting an object-level label candidate, reject labels whose simple "
            "3D bounding-box geometry is implausible for the mask."
        ),
    )
    parser.add_argument(
        "--object-relabel-temperature",
        type=float,
        default=1.0,
        help="Temperature for softmaxing saved per-point logits during object-level relabeling.",
    )
    parser.add_argument(
        "--object-relabel-unknown-label",
        default="",
        help="Optional label to assign when the best object-level candidate is weak or near-tied.",
    )
    parser.add_argument(
        "--object-relabel-unknown-min-score",
        type=float,
        default=0.0,
        help="If >0, send object labels below this best score to --object-relabel-unknown-label.",
    )
    parser.add_argument(
        "--object-relabel-unknown-min-margin",
        type=float,
        default=0.0,
        help="If >0, send object labels with best-second margin below this to the unknown label.",
    )
    parser.add_argument("--max-instances-per-label", type=int, default=200)
    parser.add_argument("--near-threshold", type=float, default=1.0)
    parser.add_argument("--support-z-tolerance", type=float, default=0.10)
    parser.add_argument("--room-polygons", default="")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def load_labels_from_batch(batch_manifest: dict[str, Any]) -> list[str]:
    for entry in batch_manifest["chunks"]:
        if entry.get("status") != "done":
            continue
        pred_path = Path(entry["prediction_path"])
        if pred_path.exists():
            data = np.load(pred_path, allow_pickle=False)
            return [str(x) for x in data["labels"]]
    raise RuntimeError("No completed chunk predictions found.")


def reduce_votes(
    vote_keys: list[np.ndarray],
    vote_values: list[np.ndarray],
    point_count: int,
    label_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    if not vote_keys:
        raise RuntimeError("No votes were collected from chunk predictions.")

    keys = np.concatenate(vote_keys).astype(np.uint64)
    values = np.concatenate(vote_values).astype(np.float32)
    order = np.argsort(keys)
    keys = keys[order]
    values = values[order]
    unique_keys, starts = np.unique(keys, return_index=True)
    sums = np.add.reduceat(values, starts).astype(np.float32)

    point_ids = (unique_keys // np.uint64(label_count)).astype(np.int64)
    label_ids = (unique_keys % np.uint64(label_count)).astype(np.int32)
    final_label = np.full(point_count, -1, dtype=np.int32)
    final_score = np.zeros(point_count, dtype=np.float32)

    point_starts = np.r_[0, np.flatnonzero(np.diff(point_ids)) + 1]
    point_ends = np.r_[point_starts[1:], len(point_ids)]
    for start, end in zip(point_starts, point_ends):
        best_local = start + int(np.argmax(sums[start:end]))
        point_id = int(point_ids[best_local])
        final_label[point_id] = int(label_ids[best_local])
        final_score[point_id] = float(sums[best_local])

    return final_label, final_score


def collect_votes(batch_manifest: dict[str, Any], label_count: int) -> tuple[list[np.ndarray], list[np.ndarray], int]:
    vote_keys: list[np.ndarray] = []
    vote_values: list[np.ndarray] = []
    total_predictions = 0
    for entry in batch_manifest["chunks"]:
        if entry.get("status") != "done":
            continue
        pred_path = Path(entry["prediction_path"])
        meta_path = Path(entry["meta_path"])
        if not pred_path.exists() or not meta_path.exists():
            continue

        pred = np.load(pred_path, allow_pickle=False)
        meta = np.load(meta_path, allow_pickle=False)
        sample_index = np.asarray(pred["sample_index"], dtype=np.int64)
        global_index = np.asarray(meta["global_index"], dtype=np.int64)[sample_index]
        center_weight = np.asarray(meta["center_weight"], dtype=np.float32)[sample_index]
        pred_index = np.asarray(pred["pred_index"], dtype=np.int64)
        score = np.asarray(pred["score"], dtype=np.float32)
        weight = np.clip(score, 0.0, 1.0) * np.clip(center_weight, 0.01, 1.0)
        key = global_index.astype(np.uint64) * np.uint64(label_count) + pred_index.astype(np.uint64)
        vote_keys.append(key)
        vote_values.append(weight.astype(np.float32))
        total_predictions += int(len(pred_index))
    return vote_keys, vote_values, total_predictions


def parse_instance_label_indices(labels: list[str], requested: str) -> list[int]:
    include = {x.strip().lower() for x in requested.replace("\n", ",").split(",") if x.strip()}
    return [i for i, label in enumerate(labels) if label.strip().lower() in include]


def parse_relabel_candidate_indices(labels: list[str], requested: str, instance_labels: str) -> list[int]:
    mode = str(requested or "all").strip().lower()
    if mode in {"all", "*"}:
        return list(range(len(labels)))
    if mode in {"instance", "instances", "object", "objects"}:
        return parse_instance_label_indices(labels, instance_labels)

    include = {x.strip().lower() for x in str(requested).replace("\n", ",").split(",") if x.strip()}
    return [i for i, label in enumerate(labels) if label.strip().lower() in include]


def softmax_np(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float32)
    logits = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(logits).astype(np.float32, copy=False)
    denom = np.sum(exp, axis=1, keepdims=True)
    return exp / np.maximum(denom, 1e-12)


def compute_instance_reliability_thresholds(
    batch_manifest: dict[str, Any],
    point_instance_index: np.ndarray,
    instance_count: int,
    top_point_fraction: float,
    min_point_score: float,
    min_center_weight: float,
    bins: int = 256,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Approximate per-instance reliability thresholds without storing all samples."""

    top_point_fraction = float(np.clip(top_point_fraction, 0.0, 1.0))
    min_point_score = float(min_point_score)
    min_center_weight = float(min_center_weight)
    thresholds = np.zeros(instance_count, dtype=np.float32)
    eligible_counts = np.zeros(instance_count, dtype=np.int64)
    stats = {
        "reliable_point_filter_enabled": top_point_fraction < 1.0
        or min_point_score > 0.0
        or min_center_weight > 0.0,
        "top_point_fraction": top_point_fraction,
        "min_point_score": min_point_score,
        "min_center_weight": min_center_weight,
        "threshold_bins": int(bins),
        "threshold_chunks_used": 0,
        "threshold_samples_seen": 0,
        "threshold_samples_assigned_to_instances": 0,
        "threshold_samples_after_base_filter": 0,
        "instances_with_base_evidence": 0,
    }
    if instance_count <= 0:
        return thresholds, eligible_counts, stats

    hist = np.zeros((instance_count, int(bins)), dtype=np.int64)
    bin_scale = float(bins - 1)
    for entry in batch_manifest["chunks"]:
        if entry.get("status") != "done":
            continue
        pred_path = Path(entry["prediction_path"])
        meta_path = Path(entry["meta_path"])
        if not pred_path.exists() or not meta_path.exists():
            continue

        pred = np.load(pred_path, allow_pickle=False)
        meta = np.load(meta_path, allow_pickle=False)
        sample_index = np.asarray(pred["sample_index"], dtype=np.int64)
        global_index = np.asarray(meta["global_index"], dtype=np.int64)[sample_index]
        instance_ids = point_instance_index[global_index]
        valid = instance_ids >= 0
        stats["threshold_samples_seen"] += int(len(sample_index))
        stats["threshold_samples_assigned_to_instances"] += int(np.count_nonzero(valid))
        if not np.any(valid):
            continue

        score = np.asarray(pred["score"], dtype=np.float32)[valid]
        center_weight = np.asarray(meta["center_weight"], dtype=np.float32)[sample_index][valid]
        valid_instance_ids = np.asarray(instance_ids[valid], dtype=np.int64)
        base = (score >= min_point_score) & (center_weight >= min_center_weight)
        if not np.any(base):
            continue

        valid_instance_ids = valid_instance_ids[base]
        reliability = np.clip(score[base], 0.0, 1.0) * np.clip(center_weight[base], 0.0, 1.0)
        bin_index = np.minimum((reliability * bin_scale).astype(np.int64), bins - 1)
        np.add.at(hist, (valid_instance_ids, bin_index), 1)
        np.add.at(eligible_counts, valid_instance_ids, 1)
        stats["threshold_samples_after_base_filter"] += int(len(valid_instance_ids))
        stats["threshold_chunks_used"] += 1

    stats["instances_with_base_evidence"] = int(np.count_nonzero(eligible_counts > 0))
    if top_point_fraction >= 1.0:
        return thresholds, eligible_counts, stats

    for instance_id, count in enumerate(eligible_counts):
        if count <= 0:
            continue
        target = max(1, int(np.ceil(float(count) * top_point_fraction)))
        reverse_cum = np.cumsum(hist[instance_id, ::-1])
        reverse_index = int(np.searchsorted(reverse_cum, target, side="left"))
        threshold_bin = max(0, bins - 1 - reverse_index)
        thresholds[instance_id] = float(threshold_bin / bin_scale)

    nonzero_thresholds = thresholds[eligible_counts > 0]
    if len(nonzero_thresholds):
        stats["threshold_min"] = float(np.min(nonzero_thresholds))
        stats["threshold_median"] = float(np.median(nonzero_thresholds))
        stats["threshold_max"] = float(np.max(nonzero_thresholds))
    else:
        stats["threshold_min"] = 0.0
        stats["threshold_median"] = 0.0
        stats["threshold_max"] = 0.0
    return thresholds, eligible_counts, stats


def collect_object_label_evidence(
    batch_manifest: dict[str, Any],
    point_instance_index: np.ndarray,
    instance_count: int,
    label_count: int,
    candidate_indices: list[int],
    top_point_fraction: float,
    min_point_score: float,
    min_center_weight: float,
    temperature: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    candidate_count = len(candidate_indices)
    label_sums = np.zeros((instance_count, candidate_count), dtype=np.float64)
    weight_sums = np.zeros(instance_count, dtype=np.float64)
    sample_counts = np.zeros(instance_count, dtype=np.int64)
    reliable_thresholds, eligible_counts, threshold_stats = compute_instance_reliability_thresholds(
        batch_manifest=batch_manifest,
        point_instance_index=point_instance_index,
        instance_count=instance_count,
        top_point_fraction=float(top_point_fraction),
        min_point_score=float(min_point_score),
        min_center_weight=float(min_center_weight),
    )
    candidate_array = np.asarray(candidate_indices, dtype=np.int64)
    stats = {
        "chunks_used": 0,
        "chunks_missing_logits": 0,
        "samples_seen": 0,
        "samples_assigned_to_instances": 0,
        "samples_after_base_filter": 0,
        "samples_after_reliable_filter": 0,
        "object_relabel_temperature": float(temperature),
    }
    stats.update(threshold_stats)

    for entry in batch_manifest["chunks"]:
        if entry.get("status") != "done":
            continue
        pred_path = Path(entry["prediction_path"])
        meta_path = Path(entry["meta_path"])
        if not pred_path.exists() or not meta_path.exists():
            continue

        pred = np.load(pred_path, allow_pickle=False)
        if "logits" not in pred:
            stats["chunks_missing_logits"] += 1
            continue
        meta = np.load(meta_path, allow_pickle=False)
        sample_index = np.asarray(pred["sample_index"], dtype=np.int64)
        global_index = np.asarray(meta["global_index"], dtype=np.int64)[sample_index]
        instance_ids = point_instance_index[global_index]
        valid = instance_ids >= 0
        stats["samples_seen"] += int(len(sample_index))
        if not np.any(valid):
            continue

        logits = np.asarray(pred["logits"], dtype=np.float32)
        if logits.ndim != 2 or logits.shape[1] != label_count:
            raise ValueError(f"Unexpected logits shape in {pred_path}: {logits.shape}")

        valid_instance_ids = np.asarray(instance_ids[valid], dtype=np.int64)
        score = np.asarray(pred["score"], dtype=np.float32)[valid]
        center_weight = np.asarray(meta["center_weight"], dtype=np.float32)[sample_index][valid]
        base = (score >= float(min_point_score)) & (center_weight >= float(min_center_weight))
        stats["samples_after_base_filter"] += int(np.count_nonzero(base))
        if not np.any(base):
            continue

        valid_instance_ids = valid_instance_ids[base]
        score = score[base]
        center_weight = center_weight[base]
        reliability = np.clip(score, 0.0, 1.0) * np.clip(center_weight, 0.0, 1.0)
        keep = reliability >= reliable_thresholds[valid_instance_ids]
        stats["samples_after_reliable_filter"] += int(np.count_nonzero(keep))
        if not np.any(keep):
            continue

        valid_instance_ids = valid_instance_ids[keep]
        score = score[keep]
        center_weight = np.clip(center_weight[keep], 0.01, 1.0).astype(np.float64)
        sample_weight = center_weight * np.clip(score.astype(np.float64), 0.0, 1.0)
        temp = max(float(temperature), 1e-6)
        probabilities = softmax_np(logits[valid][base][keep] / temp)[:, candidate_array].astype(
            np.float64, copy=False
        )

        np.add.at(label_sums, valid_instance_ids, probabilities * sample_weight[:, None])
        np.add.at(weight_sums, valid_instance_ids, sample_weight)
        np.add.at(sample_counts, valid_instance_ids, 1)
        stats["chunks_used"] += 1
        stats["samples_assigned_to_instances"] += int(np.count_nonzero(valid))

    return label_sums, weight_sums, sample_counts, eligible_counts, stats


def geometry_label_key(label: str) -> str:
    return " ".join(str(label).strip().lower().replace("_", " ").split())


def geometry_rejection_reason(label: str, item: dict[str, Any]) -> str:
    """Return a reason when a label is geometrically implausible for this mask."""

    key = geometry_label_key(label)
    bbox_min = np.asarray(item.get("bbox_min", [0.0, 0.0, 0.0]), dtype=np.float32)
    bbox_max = np.asarray(item.get("bbox_max", [0.0, 0.0, 0.0]), dtype=np.float32)
    dims = np.maximum(bbox_max - bbox_min, 0.0)
    xy_long = float(max(dims[0], dims[1]))
    xy_short = float(min(dims[0], dims[1]))
    height = float(dims[2])
    xy_area = float(max(dims[0], 1e-6) * max(dims[1], 1e-6))

    if key == "door":
        if height < 1.20:
            return f"door too short: height {height:.2f}m"
        if height > 4.20:
            return f"door too tall: height {height:.2f}m"
        if xy_long < 0.35:
            return f"door too narrow: width {xy_long:.2f}m"
        if xy_long > 3.50:
            return f"door too wide: width {xy_long:.2f}m"
        if xy_short > 1.30:
            return f"door too deep/thick: depth {xy_short:.2f}m"
        if xy_area > 7.00:
            return f"door footprint too large: {xy_area:.2f}m2"

    elif key == "elevator":
        if height < 1.60:
            return f"elevator too short: height {height:.2f}m"
        if height > 4.20:
            return f"elevator too tall: height {height:.2f}m"
        if xy_long < 0.70:
            return f"elevator too narrow: width {xy_long:.2f}m"
        if xy_long > 5.00:
            return f"elevator too wide: width {xy_long:.2f}m"
        if xy_short > 2.00:
            return f"elevator too deep/thick: depth {xy_short:.2f}m"
        if xy_area > 10.00:
            return f"elevator footprint too large: {xy_area:.2f}m2"

    elif key in {"water dispenser", "water dispensor"}:
        if height < 0.50:
            return f"water dispenser too short: height {height:.2f}m"
        if height > 3.00:
            return f"water dispenser too tall: height {height:.2f}m"
        if xy_long < 0.20:
            return f"water dispenser too narrow: width {xy_long:.2f}m"
        if xy_long > 2.50:
            return f"water dispenser too wide: width {xy_long:.2f}m"
        if xy_short > 1.20:
            return f"water dispenser too deep: depth {xy_short:.2f}m"
        if xy_area > 3.00:
            return f"water dispenser footprint too large: {xy_area:.2f}m2"

    elif key in {"whiteboard", "noticeboard", "notice board", "projector screen"}:
        if height < 0.25:
            return f"{key} too short: height {height:.2f}m"
        if height > 3.50:
            return f"{key} too tall: height {height:.2f}m"
        if xy_long < 0.30:
            return f"{key} too narrow: width {xy_long:.2f}m"
        if xy_long > 8.00:
            return f"{key} too wide: width {xy_long:.2f}m"
        if xy_short > 1.40:
            return f"{key} too deep/thick: depth {xy_short:.2f}m"
        if xy_area > 12.00:
            return f"{key} footprint too large: {xy_area:.2f}m2"

    return ""


def relabel_instances_from_object_logits(
    instances: list[dict[str, Any]],
    point_instance_index: np.ndarray,
    batch_manifest: dict[str, Any],
    labels: list[str],
    instance_labels: str,
    candidate_spec: str,
    top_k: int,
    min_score: float,
    min_margin: float,
    top_point_fraction: float,
    min_point_score: float,
    min_center_weight: float,
    geometry_filter: bool,
    temperature: float,
    unknown_label: str,
    unknown_min_score: float,
    unknown_min_margin: float,
) -> dict[str, Any]:
    candidate_indices = parse_relabel_candidate_indices(labels, candidate_spec, instance_labels)
    candidate_indices = [idx for idx in candidate_indices if 0 <= int(idx) < len(labels)]
    summary: dict[str, Any] = {
        "enabled": True,
        "candidate_spec": candidate_spec,
        "candidate_labels": [labels[i] for i in candidate_indices],
        "top_k": int(top_k),
        "min_score": float(min_score),
        "min_margin": float(min_margin),
        "top_point_fraction": float(top_point_fraction),
        "min_point_score": float(min_point_score),
        "min_center_weight": float(min_center_weight),
        "geometry_filter": bool(geometry_filter),
        "temperature": float(temperature),
        "unknown_label": str(unknown_label or ""),
        "unknown_min_score": float(unknown_min_score),
        "unknown_min_margin": float(unknown_min_margin),
        "unknown_label_found": False,
        "instance_count": len(instances),
        "instances_with_evidence": 0,
        "changed_instances": 0,
        "unchanged_instances": 0,
        "uncertain_to_unknown_instances": 0,
        "geometry_corrected_instances": 0,
        "geometry_rejected_top_candidates": 0,
        "blocked_low_score": 0,
        "blocked_low_margin": 0,
        "no_evidence": 0,
    }
    if not instances or not candidate_indices:
        summary["enabled"] = False
        summary["disabled_reason"] = "no_instances_or_candidates"
        return summary

    label_sums, weight_sums, sample_counts, eligible_counts, stats = collect_object_label_evidence(
        batch_manifest=batch_manifest,
        point_instance_index=point_instance_index,
        instance_count=len(instances),
        label_count=len(labels),
        candidate_indices=candidate_indices,
        top_point_fraction=float(top_point_fraction),
        min_point_score=float(min_point_score),
        min_center_weight=float(min_center_weight),
        temperature=float(temperature),
    )
    summary.update(stats)
    candidate_array = np.asarray(candidate_indices, dtype=np.int64)
    top_k = max(1, int(top_k))
    min_score = float(min_score)
    min_margin = float(min_margin)
    unknown_min_score = float(unknown_min_score)
    unknown_min_margin = float(unknown_min_margin)
    unknown_key = geometry_label_key(unknown_label)
    unknown_label_index = -1
    if unknown_key:
        for label_index, label in enumerate(labels):
            if geometry_label_key(label) == unknown_key:
                unknown_label_index = int(label_index)
                summary["unknown_label_found"] = True
                break

    for item in instances:
        instance_id = int(item.get("instance_id", -1))
        item["pre_relabel_semantic_label_index"] = int(item.get("semantic_label_index", -1))
        item["pre_relabel_semantic_label"] = str(item.get("semantic_label", ""))
        item["pre_relabel_original_label"] = str(item.get("original_label", ""))
        item["pre_relabel_edited_label"] = str(item.get("edited_label", ""))
        item["pre_relabel_mean_score"] = float(item.get("mean_score", 0.0))
        item["relabel_changed"] = False
        item["object_relabel_applied"] = False
        item["object_relabel_blocked_reason"] = ""
        item["unknown_relabel_applied"] = False
        item["unknown_relabel_reason"] = ""
        item["object_label_sample_count"] = int(sample_counts[instance_id]) if 0 <= instance_id < len(sample_counts) else 0
        item["object_label_eligible_sample_count"] = (
            int(eligible_counts[instance_id]) if 0 <= instance_id < len(eligible_counts) else 0
        )

        if instance_id < 0 or instance_id >= len(instances) or weight_sums[instance_id] <= 0:
            item["label_candidates"] = []
            item["object_label_score"] = 0.0
            item["object_label_margin"] = 0.0
            item["object_relabel_blocked_reason"] = "no_evidence"
            summary["no_evidence"] += 1
            continue

        scores = label_sums[instance_id] / max(float(weight_sums[instance_id]), 1e-12)
        order = np.argsort(-scores)
        top = order[: min(top_k, len(order))]
        rejected_by_geometry: list[dict[str, Any]] = []
        chosen_rank: int | None = None
        for rank in order:
            rank = int(rank)
            candidate_label_index = int(candidate_array[rank])
            candidate_label = str(labels[candidate_label_index])
            reason = geometry_rejection_reason(candidate_label, item) if geometry_filter else ""
            if reason:
                if len(rejected_by_geometry) < 10:
                    rejected_by_geometry.append(
                        {
                            "label": candidate_label,
                            "label_index": candidate_label_index,
                            "score": float(scores[rank]),
                            "reason": reason,
                        }
                    )
                continue
            chosen_rank = rank
            break

        raw_best_rank = int(order[0])
        raw_best_label_index = int(candidate_array[raw_best_rank])
        raw_best_label = str(labels[raw_best_label_index])
        raw_best_score = float(scores[raw_best_rank])
        if chosen_rank is None:
            chosen_rank = raw_best_rank

        candidates = []
        for rank in top:
            rank = int(rank)
            label_index = int(candidate_array[rank])
            label = str(labels[label_index])
            reason = geometry_rejection_reason(label, item) if geometry_filter else ""
            candidates.append(
                {
                    "label": label,
                    "label_index": label_index,
                    "score": float(scores[rank]),
                    "geometry_valid": not bool(reason),
                    "geometry_rejection_reason": reason,
                }
            )

        best_rank = int(chosen_rank)
        best_label_index = int(candidate_array[best_rank])
        best_label = str(labels[best_label_index])
        best_score = float(scores[best_rank])
        valid_order = [
            int(rank)
            for rank in order
            if not (geometry_filter and geometry_rejection_reason(str(labels[int(candidate_array[int(rank)])]), item))
        ]
        if len(valid_order) > 1:
            second_score = float(scores[valid_order[1]])
        else:
            second_score = 0.0
        margin = float(best_score - second_score)
        current_label_index = int(item.get("semantic_label_index", -1))
        current_score = 0.0
        current_positions = np.flatnonzero(candidate_array == current_label_index)
        if len(current_positions):
            current_score = float(scores[int(current_positions[0])])

        item["label_candidates"] = candidates
        item["object_label_score"] = best_score
        item["object_label_margin"] = margin
        item["pre_relabel_object_label_score"] = current_score
        item["pre_geometry_best_label"] = raw_best_label
        item["pre_geometry_best_label_index"] = raw_best_label_index
        item["pre_geometry_best_score"] = raw_best_score
        item["geometry_filter_applied"] = bool(geometry_filter)
        item["geometry_corrected_label"] = bool(geometry_filter and best_rank != raw_best_rank)
        item["geometry_rejected_candidates"] = rejected_by_geometry
        summary["instances_with_evidence"] += 1
        if geometry_filter and rejected_by_geometry and rejected_by_geometry[0]["label"] == raw_best_label:
            summary["geometry_rejected_top_candidates"] += 1
        if item["geometry_corrected_label"]:
            summary["geometry_corrected_instances"] += 1

        final_label_index = best_label_index
        final_label = best_label
        unknown_reasons = []
        if unknown_label_index >= 0 and best_label_index != unknown_label_index:
            if unknown_min_score > 0.0 and best_score < unknown_min_score:
                unknown_reasons.append(f"best_score {best_score:.3f} < {unknown_min_score:.3f}")
            if unknown_min_margin > 0.0 and margin < unknown_min_margin:
                unknown_reasons.append(f"margin {margin:.3f} < {unknown_min_margin:.3f}")
        if unknown_reasons:
            final_label_index = unknown_label_index
            final_label = str(labels[unknown_label_index])
            item["unknown_relabel_applied"] = True
            item["unknown_relabel_reason"] = "; ".join(unknown_reasons)
            item["pre_unknown_semantic_label"] = best_label
            item["pre_unknown_semantic_label_index"] = best_label_index
            item["pre_unknown_object_label_score"] = best_score
            item["pre_unknown_object_label_margin"] = margin
            summary["uncertain_to_unknown_instances"] += 1

        if not item["unknown_relabel_applied"] and best_score < min_score:
            item["object_relabel_blocked_reason"] = "best_score_below_threshold"
            summary["blocked_low_score"] += 1
            continue
        if not item["unknown_relabel_applied"] and margin < min_margin:
            item["object_relabel_blocked_reason"] = "margin_below_threshold"
            summary["blocked_low_margin"] += 1
            continue

        previous_label = str(item.get("semantic_label", ""))
        item["semantic_label_index"] = final_label_index
        item["semantic_label"] = final_label
        item["mean_score"] = best_score
        item["source"] = "mosaic3d_object_relabel"
        item["object_relabel_applied"] = True
        item["relabel_changed"] = final_label != previous_label
        if item["relabel_changed"]:
            summary["changed_instances"] += 1
        else:
            summary["unchanged_instances"] += 1

    per_label_count: dict[str, int] = {}
    for item in instances:
        label = str(item.get("semantic_label", "object"))
        per_label_count[label] = per_label_count.get(label, 0) + 1
        new_name = f"{label}_{per_label_count[label]}"
        item["original_label"] = new_name
        item["edited_label"] = new_name

    return summary


def instance_palette(count: int) -> np.ndarray:
    return palette(max(count, 1))


def write_instance_ply(path: Path, coord: np.ndarray, instance_index: np.ndarray, count: int) -> None:
    colors = np.full((len(coord), 3), 145, dtype=np.uint8)
    if count > 0:
        pal = instance_palette(count)
        assigned = instance_index >= 0
        colors[assigned] = pal[instance_index[assigned]]
    write_point_cloud(path, coord, colors)


def filter_instances_by_mean_score(
    instances: list[dict[str, Any]],
    point_instance_index: np.ndarray,
    min_mean_score: float,
) -> tuple[list[dict[str, Any]], np.ndarray, list[dict[str, Any]]]:
    min_mean_score = float(min_mean_score)
    if min_mean_score <= 0:
        return instances, point_instance_index, []

    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    old_to_new: dict[int, int] = {}
    per_label_count: dict[str, int] = {}

    for item in instances:
        old_id = int(item.get("instance_id", -1))
        mean_score = float(item.get("mean_score", 0.0))
        if mean_score < min_mean_score:
            row = dict(item)
            row["rejection_reason"] = "mean_score_below_threshold"
            row["min_mean_score"] = min_mean_score
            rejected.append(row)
            continue

        label = str(item.get("semantic_label", "object"))
        per_label_count[label] = per_label_count.get(label, 0) + 1
        new_id = len(accepted)
        new_name = f"{label}_{per_label_count[label]}"
        row = dict(item)
        row["instance_id"] = new_id
        row["original_label"] = new_name
        row["edited_label"] = new_name
        accepted.append(row)
        old_to_new[old_id] = new_id

    new_index = np.full_like(point_instance_index, -1, dtype=np.int32)
    assigned = point_instance_index >= 0
    if np.any(assigned) and old_to_new:
        old_ids = point_instance_index[assigned]
        max_old = int(old_ids.max(initial=-1))
        mapping = np.full(max_old + 1, -1, dtype=np.int32)
        for old_id, new_id in old_to_new.items():
            if 0 <= old_id <= max_old:
                mapping[old_id] = int(new_id)
        valid = old_ids <= max_old
        mapped = np.full(len(old_ids), -1, dtype=np.int32)
        mapped[valid] = mapping[old_ids[valid]]
        new_index[assigned] = mapped

    return accepted, new_index, rejected


def _label_key(label: str) -> str:
    return " ".join(str(label).strip().lower().replace("_", " ").split())


def validate_label_quality(instances: list[dict[str, Any]]) -> dict[str, Any]:
    """Attach geometry/context QA flags for labels that often get confused."""

    watched_labels = {"door", "elevator", "water dispenser", "water dispensor"}
    summary = {
        "enabled": True,
        "watched_labels": sorted(watched_labels),
        "likely_correct": 0,
        "review": 0,
        "likely_wrong": 0,
        "flagged_instances": [],
    }

    for item in instances:
        label = _label_key(str(item.get("semantic_label", "")))
        bbox_min = np.asarray(item.get("bbox_min", [0.0, 0.0, 0.0]), dtype=np.float32)
        bbox_max = np.asarray(item.get("bbox_max", [0.0, 0.0, 0.0]), dtype=np.float32)
        dims = np.maximum(bbox_max - bbox_min, 0.0)
        xy_long = float(max(dims[0], dims[1]))
        xy_short = float(min(dims[0], dims[1]))
        height = float(dims[2])
        xy_area = float(max(dims[0], 1e-6) * max(dims[1], 1e-6))
        mean_score = float(item.get("mean_score", 0.0))
        room_id = str(item.get("room_id", "floor"))
        reasons: list[str] = []
        severity = 0

        def flag(level: str, message: str) -> None:
            nonlocal severity
            reasons.append(message)
            severity = max(severity, 2 if level == "likely_wrong" else 1)

        if label in watched_labels:
            if mean_score < 0.020:
                flag("likely_wrong", f"very low mean_score {mean_score:.3f}")
            elif mean_score < 0.025:
                flag("review", f"low mean_score {mean_score:.3f}")

        if label == "door":
            if height < 1.20:
                flag("likely_wrong", f"too short for door: height {height:.2f}m")
            elif height < 1.60:
                flag("review", f"short for door: height {height:.2f}m")
            if height > 4.00:
                flag("review", f"too tall for normal door: height {height:.2f}m")
            if xy_long < 0.35:
                flag("review", f"too narrow for door: width {xy_long:.2f}m")
            if xy_long > 3.50:
                flag("likely_wrong", f"too wide for single door: width {xy_long:.2f}m")
            elif xy_long > 2.40:
                flag("review", f"wide door-like object: width {xy_long:.2f}m")
            if xy_short > 1.30:
                flag("review", f"too thick/deep for door: depth {xy_short:.2f}m")
            if xy_area > 7.00:
                flag("likely_wrong", f"door footprint too large: {xy_area:.2f}m2")

        elif label == "elevator":
            if mean_score < 0.035:
                flag("review", f"weak elevator confidence: mean_score {mean_score:.3f}")
            if height < 1.60:
                flag("likely_wrong", f"too short for elevator entrance: height {height:.2f}m")
            elif height < 1.90:
                flag("review", f"short for elevator entrance: height {height:.2f}m")
            if height > 4.20:
                flag("review", f"too tall for elevator entrance: height {height:.2f}m")
            if xy_long < 0.70:
                flag("review", f"too narrow for elevator entrance: width {xy_long:.2f}m")
            if xy_long > 5.00:
                flag("likely_wrong", f"too wide for elevator entrance: width {xy_long:.2f}m")
            elif xy_long > 3.50:
                flag("review", f"wide elevator-like surface: width {xy_long:.2f}m")
            if xy_short > 2.00:
                flag("review", f"too deep/thick for elevator face: depth {xy_short:.2f}m")

        elif label in {"water dispenser", "water dispensor"}:
            if height < 0.50:
                flag("likely_wrong", f"too short for water dispenser: height {height:.2f}m")
            elif height < 0.90:
                flag("review", f"short for water dispenser: height {height:.2f}m")
            if height > 3.00:
                flag("likely_wrong", f"too tall for water dispenser: height {height:.2f}m")
            elif height > 2.20:
                flag("review", f"tall for water dispenser: height {height:.2f}m")
            if xy_long < 0.20:
                flag("review", f"too narrow for water dispenser: width {xy_long:.2f}m")
            if xy_long > 2.50:
                flag("likely_wrong", f"too wide for water dispenser: width {xy_long:.2f}m")
            elif xy_long > 1.50:
                flag("review", f"wide for water dispenser: width {xy_long:.2f}m")
            if xy_short > 1.20:
                flag("review", f"too deep for water dispenser: depth {xy_short:.2f}m")

        status = "likely_wrong" if severity >= 2 else "review" if severity == 1 else "likely_correct"
        item["qa_status"] = status
        item["qa_reasons"] = reasons
        item["qa_metrics"] = {
            "height_m": height,
            "xy_long_m": xy_long,
            "xy_short_m": xy_short,
            "xy_area_m2": xy_area,
            "mean_score": mean_score,
            "room_id": room_id,
        }
        summary[status] += 1
        if status != "likely_correct":
            summary["flagged_instances"].append(
                {
                    "instance_id": int(item.get("instance_id", -1)),
                    "name": str(item.get("edited_label", "")),
                    "semantic_label": str(item.get("semantic_label", "")),
                    "room_id": room_id,
                    "points": int(item.get("points", 0)),
                    "mean_score": mean_score,
                    "qa_status": status,
                    "qa_reasons": reasons,
                    "qa_metrics": item["qa_metrics"],
                }
            )

    return summary


def bbox_xy_distance(a: dict[str, Any], b: dict[str, Any]) -> float:
    amin = np.asarray(a["bbox_min"], dtype=np.float32)
    amax = np.asarray(a["bbox_max"], dtype=np.float32)
    bmin = np.asarray(b["bbox_min"], dtype=np.float32)
    bmax = np.asarray(b["bbox_max"], dtype=np.float32)
    dx = max(0.0, float(max(bmin[0] - amax[0], amin[0] - bmax[0])))
    dy = max(0.0, float(max(bmin[1] - amax[1], amin[1] - bmax[1])))
    return float((dx * dx + dy * dy) ** 0.5)


def xy_overlap_fraction(a: dict[str, Any], b: dict[str, Any]) -> float:
    amin = np.asarray(a["bbox_min"], dtype=np.float32)
    amax = np.asarray(a["bbox_max"], dtype=np.float32)
    bmin = np.asarray(b["bbox_min"], dtype=np.float32)
    bmax = np.asarray(b["bbox_max"], dtype=np.float32)
    ix = max(0.0, float(min(amax[0], bmax[0]) - max(amin[0], bmin[0])))
    iy = max(0.0, float(min(amax[1], bmax[1]) - max(amin[1], bmin[1])))
    inter = ix * iy
    area_a = max(1e-6, float((amax[0] - amin[0]) * (amax[1] - amin[1])))
    area_b = max(1e-6, float((bmax[0] - bmin[0]) * (bmax[1] - bmin[1])))
    return float(inter / min(area_a, area_b))


def assign_rooms(
    instances: list[dict[str, Any]],
    room_polygons_path: str,
    coord: np.ndarray,
    instance_index: np.ndarray,
) -> list[dict[str, Any]]:
    if not room_polygons_path:
        for item in instances:
            item["room_id"] = "floor"
        return [{"room_id": "floor", "room_type": "floor", "polygon_xy": [], "object_ids": []}]

    room_path = Path(room_polygons_path).expanduser().resolve()
    payload = read_json(room_path)
    rooms = payload.get("rooms", payload if isinstance(payload, list) else [])
    normalized = []
    for i, room in enumerate(rooms):
        polygon = room.get("polygon_xy") or room.get("polygon") or []
        if len(polygon) < 3:
            continue
        npz_text = str(room.get("point_indices_npz") or "").strip()
        npz_path = Path(npz_text).expanduser() if npz_text else None
        if npz_path and not npz_path.is_absolute():
            npz_path = room_path.parent / npz_path
        normalized.append(
            {
                "room_id": str(room.get("room_id") or room.get("id") or f"room_{i + 1:03d}"),
                "room_type": str(room.get("room_type") or room.get("type") or "room"),
                "polygon_xy": [[float(x), float(y)] for x, y in polygon],
                "bbox_min_xy": room.get("bbox_min_xy", []),
                "bbox_max_xy": room.get("bbox_max_xy", []),
                "point_indices_npz": str(npz_path.resolve()) if npz_path else "",
                "point_indices_key": str(room.get("point_indices_key") or ""),
                "object_ids": [],
            }
        )

    point_room = np.full(len(coord), -1, dtype=np.int32)
    index_cache: dict[str, Any] = {}
    for room_number, room in enumerate(normalized):
        npz_path = room.get("point_indices_npz", "")
        npz_key = room.get("point_indices_key", "")
        if npz_path and npz_key:
            if npz_path not in index_cache:
                index_cache[npz_path] = np.load(npz_path)
            data = index_cache[npz_path]
            if npz_key in data:
                indices = np.asarray(data[npz_key], dtype=np.int64)
                indices = indices[(indices >= 0) & (indices < len(coord))]
                point_room[indices] = room_number
                continue
        mask = points_in_polygon_xy(coord[:, :2], room["polygon_xy"])
        point_room[mask] = room_number

    for item in instances:
        instance_id = int(item["instance_id"])
        item["room_id"] = "floor"
        instance_points = np.flatnonzero(instance_index == instance_id)
        room_votes = point_room[instance_points]
        room_votes = room_votes[room_votes >= 0]
        if len(room_votes):
            counts = np.bincount(room_votes, minlength=len(normalized))
            room_number = int(np.argmax(counts))
            item["room_id"] = normalized[room_number]["room_id"]
            normalized[room_number]["object_ids"].append(str(item["edited_label"]))
            continue

        centroid = np.asarray(item["centroid"], dtype=np.float32)[None, :2]
        for room_number, room in enumerate(normalized):
            if points_in_polygon_xy(centroid, room["polygon_xy"])[0]:
                item["room_id"] = room["room_id"]
                normalized[room_number]["object_ids"].append(str(item["edited_label"]))
                break
        if item["room_id"] != "floor":
            continue
    return normalized or [{"room_id": "floor", "room_type": "floor", "polygon_xy": [], "object_ids": []}]


def build_scene_graph(
    instances: list[dict[str, Any]],
    rooms: list[dict[str, Any]],
    near_threshold: float,
    support_z_tolerance: float,
) -> dict[str, Any]:
    nodes = [
        {
            "id": "floor",
            "name": "floor",
            "type": "floor",
            "label": "floor",
            "parent": None,
            "children": [room["room_id"] for room in rooms],
        }
    ]
    edges = []
    room_objects: dict[str, list[str]] = {room["room_id"]: [] for room in rooms}
    for room in rooms:
        polygon = np.asarray(room.get("polygon_xy", []), dtype=np.float32)
        if len(polygon):
            center_xy = np.mean(polygon, axis=0).astype(float).tolist()
        else:
            center_xy = [0.0, 0.0]
        node = {
            "id": room["room_id"],
            "name": room["room_id"],
            "type": "room",
            "label": room.get("room_type", "room"),
            "parent": "floor",
            "position": {"x": float(center_xy[0]), "y": float(center_xy[1]), "z": 0.0},
            "coordinates": {"x": float(center_xy[0]), "y": float(center_xy[1])},
            "polygon_xy": room.get("polygon_xy", []),
            "bbox_min_xy": room.get("bbox_min_xy", []),
            "bbox_max_xy": room.get("bbox_max_xy", []),
            "children": list(room.get("object_ids", [])),
        }
        nodes.append(node)
        edges.append({"source": "floor", "target": room["room_id"], "relation": "contains"})

    object_nodes = []
    for item in instances:
        obj_id = str(item["edited_label"])
        centroid = [float(x) for x in item["centroid"]]
        room_id = str(item.get("room_id", "floor"))
        node = {
            "id": obj_id,
            "name": obj_id,
            "type": "object",
            "label": str(item["semantic_label"]),
            "class_name": str(item["semantic_label"]),
            "object_name": obj_id,
            "room_id": room_id,
            "parent": room_id,
            "position": {"x": centroid[0], "y": centroid[1], "z": centroid[2]},
            "coordinates": {"x": centroid[0], "y": centroid[1]},
            "centroid": centroid,
            "bbox_min": [float(x) for x in item["bbox_min"]],
            "bbox_max": [float(x) for x in item["bbox_max"]],
            "points": int(item["points"]),
            "mean_score": float(item["mean_score"]),
            "object_label_score": float(item.get("object_label_score", item.get("mean_score", 0.0))),
            "object_label_margin": float(item.get("object_label_margin", 0.0)),
            "label_candidates": list(item.get("label_candidates", [])),
            "relabel_changed": bool(item.get("relabel_changed", False)),
            "pre_relabel_semantic_label": str(item.get("pre_relabel_semantic_label", "")),
            "pre_geometry_best_label": str(item.get("pre_geometry_best_label", "")),
            "geometry_filter_applied": bool(item.get("geometry_filter_applied", False)),
            "geometry_corrected_label": bool(item.get("geometry_corrected_label", False)),
            "geometry_rejected_candidates": list(item.get("geometry_rejected_candidates", [])),
            "unknown_relabel_applied": bool(item.get("unknown_relabel_applied", False)),
            "unknown_relabel_reason": str(item.get("unknown_relabel_reason", "")),
            "pre_unknown_semantic_label": str(item.get("pre_unknown_semantic_label", "")),
            "qa_status": str(item.get("qa_status", "likely_correct")),
            "qa_reasons": list(item.get("qa_reasons", [])),
            "qa_metrics": dict(item.get("qa_metrics", {})),
            "hierarchy": ["floor", room_id, obj_id],
        }
        room_objects.setdefault(room_id, []).append(obj_id)
        object_nodes.append(node)
        nodes.append(node)
        edges.append({"source": room_id, "target": obj_id, "relation": "contains"})

    for i, a in enumerate(instances):
        for b in instances[i + 1 :]:
            a_id = str(a["edited_label"])
            b_id = str(b["edited_label"])
            xy_dist = bbox_xy_distance(a, b)
            if xy_dist <= near_threshold:
                edges.append(
                    {
                        "source": a_id,
                        "target": b_id,
                        "relation": "near",
                        "distance_xy": xy_dist,
                    }
                )

            a_min = np.asarray(a["bbox_min"], dtype=np.float32)
            a_max = np.asarray(a["bbox_max"], dtype=np.float32)
            b_min = np.asarray(b["bbox_min"], dtype=np.float32)
            b_max = np.asarray(b["bbox_max"], dtype=np.float32)
            overlap = xy_overlap_fraction(a, b)
            if overlap > 0.2:
                if abs(float(a_min[2] - b_max[2])) <= support_z_tolerance:
                    edges.append({"source": a_id, "target": b_id, "relation": "on_top_of"})
                if abs(float(b_min[2] - a_max[2])) <= support_z_tolerance:
                    edges.append({"source": b_id, "target": a_id, "relation": "on_top_of"})

    hierarchy = {
        "floor": {
            "id": "floor",
            "rooms": [
                {
                    "room_id": room["room_id"],
                    "room_type": room.get("room_type", "room"),
                    "objects": sorted(room_objects.get(room["room_id"], [])),
                }
                for room in rooms
            ],
        }
    }

    return {
        "graph_type": "point2graph_style_scene_graph",
        "root": "floor",
        "hierarchy": hierarchy,
        "rooms": [node for node in nodes if node["type"] == "room"],
        "objects": object_nodes,
        "nodes": nodes,
        "edges": edges,
    }


def readable_node_name(raw: str) -> str:
    return re.sub(r"\s+", "_", str(raw).strip())


def natural_sort_key(text: str) -> list[Any]:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", str(text))]


def bbox_xy_gap(a: dict[str, Any], b: dict[str, Any]) -> float:
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


def room_neighbor_map(rooms: list[dict[str, Any]], max_gap: float = 0.75, max_neighbors: int = 6) -> dict[str, list[str]]:
    distances: dict[str, list[tuple[float, str]]] = {str(room["room_id"]): [] for room in rooms}
    for i, room_a in enumerate(rooms):
        room_a_id = str(room_a["room_id"])
        for room_b in rooms[i + 1 :]:
            room_b_id = str(room_b["room_id"])
            gap = bbox_xy_gap(room_a, room_b)
            if gap <= float(max_gap):
                distances[room_a_id].append((gap, room_b_id))
                distances[room_b_id].append((gap, room_a_id))

    neighbors = {}
    for room_id, items in distances.items():
        items = sorted(items, key=lambda item: (item[0], natural_sort_key(item[1])))[: int(max_neighbors)]
        neighbors[room_id] = [room_id for _, room_id in items]
    return neighbors


def is_asset_label(label: str) -> bool:
    key = geometry_label_key(label)
    asset_labels = {
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
    return key in asset_labels


def build_readable_scene_graph(
    graph: dict[str, Any],
    rooms: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    objects = {str(item["id"]): item for item in graph.get("objects", [])}
    room_ids = [str(room["room_id"]) for room in rooms]
    room_lookup = {str(room["room_id"]): room for room in rooms}
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

    def object_sort_key(obj_id: str) -> list[Any]:
        return natural_sort_key(readable_node_name(objects.get(obj_id, {}).get("name", obj_id)))

    def object_entry(obj_id: str, seen: set[str] | None = None) -> dict[str, Any]:
        seen = set(seen or ())
        item = objects.get(obj_id, {})
        name = readable_node_name(str(item.get("name") or item.get("object_name") or obj_id))
        node_type = "asset" if is_asset_label(str(item.get("label", ""))) else "object"
        if obj_id in seen:
            return {"node_name": name, "node_type": node_type}
        seen.add(obj_id)

        entry: dict[str, Any] = {"node_name": name, "node_type": node_type}
        children = [
            child
            for child in sorted(support_children.get(obj_id, []), key=object_sort_key)
            if child not in seen
        ]
        if children or node_type == "asset":
            entry["child_nodes"] = [object_entry(child, seen=set(seen)) for child in children]
        return entry

    readable = []
    for room in sorted(rooms, key=lambda item: natural_sort_key(str(item.get("room_id", "")))):
        room_id = str(room["room_id"])
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


def main() -> None:
    args = parse_args()
    work_dir = Path(args.work_dir).expanduser().resolve()
    stitch_dir = work_dir / "04_stitched"
    stitch_dir.mkdir(parents=True, exist_ok=True)

    preprocess = read_json(Path(args.preprocess_manifest))
    batch = read_json(Path(args.batch_manifest))
    cloud = np.load(Path(preprocess["cloud_npz"]), allow_pickle=False)
    coord = np.asarray(cloud["coord"], dtype=np.float32)
    color = np.asarray(cloud["color"], dtype=np.float32)
    point_count = int(len(coord))
    labels = load_labels_from_batch(batch)
    label_count = len(labels)

    vote_keys, vote_values, total_predictions = collect_votes(batch, label_count)
    pred_index, vote_score = reduce_votes(vote_keys, vote_values, point_count, label_count)

    unlabeled = pred_index < 0
    if np.any(unlabeled):
        fallback_label = labels.index("other") if "other" in labels else 0
        pred_index[unlabeled] = fallback_label
        vote_score[unlabeled] = 0.0

    pred_label = np.asarray([labels[int(i)] for i in pred_index])
    colors = palette(label_count)[pred_index]

    stitched_npz = stitch_dir / "floor_mosaic3d_stitched_predictions.npz"
    stitched_ply = stitch_dir / "floor_mosaic3d_stitched_colorized.ply"
    np.savez_compressed(
        stitched_npz,
        coord=coord,
        color=color,
        pred_index=pred_index.astype(np.int32),
        pred_label=pred_label,
        score=vote_score.astype(np.float32),
        labels=np.asarray(labels),
        source_index=np.asarray(cloud["source_index"], dtype=np.int64),
        source_input=np.asarray(preprocess["source_input"]),
    )
    write_point_cloud(stitched_ply, coord, colors)

    label_indices = parse_instance_label_indices(labels, args.instance_labels)
    cluster = cluster_semantic_instances(
        coord=coord,
        pred_index=pred_index,
        score=vote_score,
        semantic_labels=labels,
        label_indices=label_indices,
        voxel_size=float(args.instance_voxel_size),
        min_points=int(args.instance_min_points),
        max_instances_per_label=int(args.max_instances_per_label),
    )
    unfiltered_instance_count = len(cluster.instances)
    unfiltered_assigned_points = int(np.count_nonzero(cluster.point_instance_index >= 0))
    filtered_instances, filtered_instance_index, rejected_instances = filter_instances_by_mean_score(
        cluster.instances,
        cluster.point_instance_index,
        min_mean_score=float(args.instance_min_mean_score),
    )
    cluster.instances = filtered_instances
    cluster.point_instance_index = filtered_instance_index
    filtered_assigned_points = int(np.count_nonzero(cluster.point_instance_index >= 0))
    if args.object_relabel:
        object_relabel_summary = relabel_instances_from_object_logits(
            instances=cluster.instances,
            point_instance_index=cluster.point_instance_index,
            batch_manifest=batch,
            labels=labels,
            instance_labels=args.instance_labels,
            candidate_spec=args.object_relabel_candidates,
            top_k=int(args.object_relabel_top_k),
            min_score=float(args.object_relabel_min_score),
            min_margin=float(args.object_relabel_min_margin),
            top_point_fraction=float(args.object_relabel_top_point_fraction),
            min_point_score=float(args.object_relabel_min_point_score),
            min_center_weight=float(args.object_relabel_min_center_weight),
            geometry_filter=bool(args.object_relabel_geometry_filter),
            temperature=float(args.object_relabel_temperature),
            unknown_label=str(args.object_relabel_unknown_label),
            unknown_min_score=float(args.object_relabel_unknown_min_score),
            unknown_min_margin=float(args.object_relabel_unknown_min_margin),
        )
    else:
        object_relabel_summary = {
            "enabled": False,
            "candidate_spec": str(args.object_relabel_candidates),
            "top_k": int(args.object_relabel_top_k),
            "min_score": float(args.object_relabel_min_score),
            "min_margin": float(args.object_relabel_min_margin),
            "top_point_fraction": float(args.object_relabel_top_point_fraction),
            "min_point_score": float(args.object_relabel_min_point_score),
            "min_center_weight": float(args.object_relabel_min_center_weight),
            "geometry_filter": bool(args.object_relabel_geometry_filter),
            "temperature": float(args.object_relabel_temperature),
            "unknown_label": str(args.object_relabel_unknown_label),
            "unknown_min_score": float(args.object_relabel_unknown_min_score),
            "unknown_min_margin": float(args.object_relabel_unknown_min_margin),
        }
    rooms = assign_rooms(
        cluster.instances,
        args.room_polygons,
        coord=coord,
        instance_index=cluster.point_instance_index,
    )
    label_qa_summary = validate_label_quality(cluster.instances)

    instances_json = stitch_dir / "floor_instances.json"
    instances_npz = stitch_dir / "floor_instances.npz"
    instances_ply = stitch_dir / "floor_instances_colorized.ply"
    scene_graph_json = stitch_dir / "scene_graph.json"
    scene_graph_readable_json = stitch_dir / "scene_graph_readable.json"
    scene_graph_detailed_json = stitch_dir / "scene_graph_detailed.json"
    summary_json = stitch_dir / "summary.json"
    qa_json = stitch_dir / "qa_flags.json"
    rejected_json = stitch_dir / "rejected_instances.json"

    write_json(
        instances_json,
        {
            "instance_count": len(cluster.instances),
            "params": {
                "instance_labels": args.instance_labels,
                "voxel_size": float(args.instance_voxel_size),
                "min_points": int(args.instance_min_points),
                "min_mean_score": float(args.instance_min_mean_score),
                "object_relabel": bool(args.object_relabel),
                "object_relabel_candidates": str(args.object_relabel_candidates),
                "object_relabel_top_k": int(args.object_relabel_top_k),
                "object_relabel_min_score": float(args.object_relabel_min_score),
                "object_relabel_min_margin": float(args.object_relabel_min_margin),
                "object_relabel_top_point_fraction": float(args.object_relabel_top_point_fraction),
                "object_relabel_min_point_score": float(args.object_relabel_min_point_score),
                "object_relabel_min_center_weight": float(args.object_relabel_min_center_weight),
                "object_relabel_geometry_filter": bool(args.object_relabel_geometry_filter),
                "object_relabel_temperature": float(args.object_relabel_temperature),
                "object_relabel_unknown_label": str(args.object_relabel_unknown_label),
                "object_relabel_unknown_min_score": float(args.object_relabel_unknown_min_score),
                "object_relabel_unknown_min_margin": float(args.object_relabel_unknown_min_margin),
            },
            "instances": cluster.instances,
        },
    )
    write_json(
        rejected_json,
        {
            "rejected_count": len(rejected_instances),
            "min_mean_score": float(args.instance_min_mean_score),
            "instances": rejected_instances,
        },
    )
    np.savez_compressed(instances_npz, instance_index=cluster.point_instance_index.astype(np.int32))
    write_instance_ply(instances_ply, coord, cluster.point_instance_index, len(cluster.instances))

    graph = build_scene_graph(
        cluster.instances,
        rooms,
        near_threshold=float(args.near_threshold),
        support_z_tolerance=float(args.support_z_tolerance),
    )
    readable_graph = build_readable_scene_graph(graph, rooms)
    write_json(scene_graph_json, readable_graph)
    write_json(scene_graph_readable_json, readable_graph)
    write_json(scene_graph_detailed_json, graph)

    unique, counts = np.unique(pred_label, return_counts=True)
    label_counts = {str(label): int(count) for label, count in zip(unique, counts)}
    summary = {
        "point_count": point_count,
        "total_chunk_predictions": total_predictions,
        "unlabeled_points_filled": int(np.count_nonzero(unlabeled)),
        "labels": labels,
        "label_counts": label_counts,
        "instance_filter": {
            "min_mean_score": float(args.instance_min_mean_score),
            "unfiltered_instance_count": int(unfiltered_instance_count),
            "kept_instance_count": int(len(cluster.instances)),
            "rejected_instance_count": int(len(rejected_instances)),
            "unfiltered_assigned_points": int(unfiltered_assigned_points),
            "kept_assigned_points": int(filtered_assigned_points),
            "rejected_assigned_points": int(unfiltered_assigned_points - filtered_assigned_points),
            "rejected_instances_json": str(rejected_json),
        },
        "object_relabel": object_relabel_summary,
        "label_quality": label_qa_summary,
        "stitched_npz": str(stitched_npz),
        "stitched_ply": str(stitched_ply),
        "instances_json": str(instances_json),
        "instances_ply": str(instances_ply),
        "scene_graph_json": str(scene_graph_json),
        "scene_graph_readable_json": str(scene_graph_readable_json),
        "scene_graph_detailed_json": str(scene_graph_detailed_json),
    }
    write_json(summary_json, summary)
    write_json(
        qa_json,
        {
            "failed_chunks": [
                item for item in batch["chunks"] if item.get("status") not in {"done", "printed"}
            ],
            "unlabeled_points_filled": int(np.count_nonzero(unlabeled)),
            "low_vote_points": int(np.count_nonzero(vote_score < 0.05)),
            "instance_min_mean_score": float(args.instance_min_mean_score),
            "rejected_low_score_instances": rejected_instances,
            "object_relabel": object_relabel_summary,
            "label_quality": label_qa_summary,
            "limited_chunks": [
                item["chunk_id"] for item in batch["chunks"] if item.get("limited_to_max_points")
            ],
        },
    )

    print(f"[stitch] wrote {stitched_ply}")
    print(f"[stitch] wrote {instances_json}")
    print(f"[stitch] wrote {scene_graph_json}")
    print(f"[stitch] wrote {scene_graph_detailed_json}")


if __name__ == "__main__":
    main()
