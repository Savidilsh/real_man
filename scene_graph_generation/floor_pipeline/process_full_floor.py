#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


PIPELINE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PIPELINE_ROOT.parents[0]


DEFAULT_CONFIG: dict[str, Any] = {
    "voxel_size": 0.03,
    "halo_m": 0.8,
    "room_full_max_points": 900000,
    "max_points_per_chunk": 900000,
    "mosaic3d_max_points": 1000000,
    "min_points_per_chunk": 1500,
    "min_split_points_per_chunk": 0,
    "min_tile_size_m": 1.0,
    "runner": "docker",
    "docker_image": "mosaic3d:latest",
    "device": "cuda",
    "grid_size": 0.02,
    "condition": "ScanNet",
    "prompt_template": "a photo of a {} in a robotics laboratory",
    "prompt_ensemble_file": "",
    "instance_labels": (
        "door,window,chair,table,desk,workbench,countertop,cabinet,shelf,equipment rack,"
        "cart,box,whiteboard,computer monitor,computer,couch,robot,robot arm,robot base,"
        "mobile robot,wheeled rover,humanoid robot,microwave,trash bin,sink,water dispenser,"
        "refrigerator,pipe,cable,solar panel"
    ),
    "instance_voxel_size": 0.18,
    "instance_min_points": 250,
    "instance_min_mean_score": 0.0,
    "object_relabel": False,
    "object_relabel_candidates": "all",
    "object_relabel_top_k": 5,
    "object_relabel_min_score": 0.0,
    "object_relabel_min_margin": 0.0,
    "object_relabel_top_point_fraction": 1.0,
    "object_relabel_min_point_score": 0.0,
    "object_relabel_min_center_weight": 0.0,
    "object_relabel_geometry_filter": False,
    "object_relabel_temperature": 1.0,
    "object_relabel_unknown_label": "",
    "object_relabel_unknown_min_score": 0.0,
    "object_relabel_unknown_min_margin": 0.0,
    "max_instances_per_label": 200,
    "near_threshold_m": 1.0,
    "support_z_tolerance_m": 0.10,
    "room_polygons": "",
    "room_detection": "point2graph",
    "room_grid_size": 0.10,
    "room_slice_count": 16,
    "room_min_component_area": 3.0,
    "room_wall_persistence": 0.28,
    "room_footprint_dilate_cells": 3,
    "room_wall_dilate_cells": 2,
    "room_free_close_cells": 1,
    "room_assignment_dilate_cells": 3,
    "room_wall_min_height": 0.35,
    "room_wall_max_height": 2.30,
    "room_refinement": True,
    "room_refine_slice_count": 8,
    "room_refine_wall_persistence": 0.32,
    "room_refine_wall_dilate_cells": 1,
    "room_refine_cleanup_close_cells": 1,
    "room_refine_split_wall_dilate_cells": 1,
    "room_refine_split_assignment_dilate_cells": 2,
    "room_refine_split_min_area": 8.0,
    "room_refine_merge_gap_cells": 2,
    "room_refine_merge_min_open_boundary": 0.75,
    "room_refine_merge_max_wall_ratio": 0.18,
    "room_refine_min_room_area": 3.0,
    "room_refine_assignment_dilate_cells": 2,
    "room_refine_snap_distance": 0.30,
    "room_refine_snap_min_support": 1.0,
    "room_refine_simplify_epsilon": 0.20,
    "room_assign_all_points": True,
    "room_leftover_component_min_area": 8.0,
    "room_leftover_corridor_ratio": 2.5,
    "room_leftover_corridor_max_width": 3.0,
    "preview_ply": True,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Automatic full-floor Mosaic3D pipeline: preprocess, chunk, infer, stitch, graph."
    )
    parser.add_argument("--input", required=True, help="Full-floor point cloud.")
    parser.add_argument("--work-dir", required=True, help="Output/work directory for this full-floor run.")
    parser.add_argument("--config", default="", help="Optional JSON config. See config.example.json.")
    parser.add_argument("--ckpt", default=str(REPO_ROOT / "ckpt_raw" / "converted" / "spunet34c.ckpt"))
    parser.add_argument("--labels-file", default=str(REPO_ROOT / "labels" / "robotics_lab_recommended_labels.txt"))
    parser.add_argument("--runner", choices=["docker", "python", "print"], default="")
    parser.add_argument("--room-polygons", default="", help="Optional room polygons JSON from a room detector.")
    parser.add_argument(
        "--room-detection",
        choices=["point2graph", "none"],
        default="",
        help="Run the built-in Point2Graph-style room detector after preprocessing.",
    )
    parser.add_argument("--resume", action="store_true", help="Reuse completed stage outputs.")
    parser.add_argument("--force-preprocess", action="store_true")
    parser.add_argument("--force-rooms", action="store_true")
    parser.add_argument("--force-chunks", action="store_true")
    parser.add_argument("--force-stitch", action="store_true")
    parser.add_argument(
        "--start-at",
        choices=["preprocess", "rooms", "chunks", "batch", "stitch"],
        default="preprocess",
        help="Resume from a specific stage.",
    )
    return parser.parse_args()


def load_config(path: str) -> dict[str, Any]:
    cfg = dict(DEFAULT_CONFIG)
    if path:
        payload = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
        cfg.update(payload)
    return cfg


def run_step(cmd: list[str]) -> None:
    print("[pipeline]", " ".join(cmd))
    subprocess.run(cmd, cwd=str(REPO_ROOT), check=True)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.runner:
        cfg["runner"] = args.runner
    if args.room_polygons:
        cfg["room_polygons"] = args.room_polygons
    if args.room_detection:
        cfg["room_detection"] = args.room_detection

    work_dir = Path(args.work_dir).expanduser().resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    resolved_config = work_dir / "resolved_config.json"
    resolved_config.write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    preprocess_manifest = work_dir / "01_preprocess" / "preprocess_manifest.json"
    detected_rooms_json = work_dir / "02_rooms" / "rooms.json"
    refined_rooms_json = work_dir / "02_rooms_refined" / "rooms.json"
    chunks_manifest = work_dir / "02_chunks" / "chunks_manifest.json"
    batch_manifest = work_dir / "03_inference" / "batch_manifest.json"

    stages = ["preprocess", "rooms", "chunks", "batch", "stitch"]
    start_index = stages.index(args.start_at)

    if start_index <= stages.index("preprocess"):
        cmd = [
            sys.executable,
            str(PIPELINE_ROOT / "scripts" / "preprocess_floor.py"),
            "--input",
            str(Path(args.input).expanduser().resolve()),
            "--work-dir",
            str(work_dir),
            "--voxel-size",
            str(cfg["voxel_size"]),
        ]
        if cfg.get("preview_ply", True):
            cmd.append("--preview-ply")
        if args.force_preprocess or not args.resume:
            cmd.append("--force")
        run_step(cmd)

    if start_index <= stages.index("rooms"):
        if not cfg.get("room_polygons") and str(cfg.get("room_detection", "none")).lower() == "point2graph":
            cmd = [
                sys.executable,
                str(PIPELINE_ROOT / "scripts" / "detect_rooms_point2graph.py"),
                "--preprocess-manifest",
                str(preprocess_manifest),
                "--work-dir",
                str(work_dir),
                "--grid-size",
                str(cfg["room_grid_size"]),
                "--slice-count",
                str(cfg["room_slice_count"]),
                "--wall-min-height",
                str(cfg["room_wall_min_height"]),
                "--wall-max-height",
                str(cfg["room_wall_max_height"]),
                "--min-component-area",
                str(cfg["room_min_component_area"]),
                "--wall-persistence",
                str(cfg["room_wall_persistence"]),
                "--footprint-dilate-cells",
                str(cfg["room_footprint_dilate_cells"]),
                "--wall-dilate-cells",
                str(cfg["room_wall_dilate_cells"]),
                "--free-close-cells",
                str(cfg["room_free_close_cells"]),
                "--assignment-dilate-cells",
                str(cfg["room_assignment_dilate_cells"]),
                "--preview-ply",
            ]
            if args.force_rooms or not args.resume:
                cmd.append("--force")
            run_step(cmd)

        if (
            not cfg.get("room_polygons")
            and str(cfg.get("room_detection", "none")).lower() == "point2graph"
            and bool(cfg.get("room_refinement", True))
            and detected_rooms_json.exists()
        ):
            cmd = [
                sys.executable,
                str(PIPELINE_ROOT / "scripts" / "refine_room_proposals.py"),
                "--preprocess-manifest",
                str(preprocess_manifest),
                "--rooms-json",
                str(detected_rooms_json),
                "--work-dir",
                str(work_dir),
                "--slice-count",
                str(cfg["room_refine_slice_count"]),
                "--wall-min-height",
                str(cfg["room_wall_min_height"]),
                "--wall-max-height",
                str(cfg["room_wall_max_height"]),
                "--wall-persistence",
                str(cfg["room_refine_wall_persistence"]),
                "--wall-dilate-cells",
                str(cfg["room_refine_wall_dilate_cells"]),
                "--cleanup-close-cells",
                str(cfg["room_refine_cleanup_close_cells"]),
                "--split-wall-dilate-cells",
                str(cfg["room_refine_split_wall_dilate_cells"]),
                "--split-assignment-dilate-cells",
                str(cfg["room_refine_split_assignment_dilate_cells"]),
                "--split-min-area",
                str(cfg["room_refine_split_min_area"]),
                "--merge-gap-cells",
                str(cfg["room_refine_merge_gap_cells"]),
                "--merge-min-open-boundary",
                str(cfg["room_refine_merge_min_open_boundary"]),
                "--merge-max-wall-ratio",
                str(cfg["room_refine_merge_max_wall_ratio"]),
                "--min-room-area",
                str(cfg["room_refine_min_room_area"]),
                "--assignment-dilate-cells",
                str(cfg["room_refine_assignment_dilate_cells"]),
                "--snap-distance",
                str(cfg["room_refine_snap_distance"]),
                "--snap-min-support",
                str(cfg["room_refine_snap_min_support"]),
                "--simplify-epsilon",
                str(cfg["room_refine_simplify_epsilon"]),
                "--leftover-component-min-area",
                str(cfg["room_leftover_component_min_area"]),
                "--leftover-corridor-ratio",
                str(cfg["room_leftover_corridor_ratio"]),
                "--leftover-corridor-max-width",
                str(cfg["room_leftover_corridor_max_width"]),
            ]
            if bool(cfg.get("room_assign_all_points", True)):
                cmd.append("--assign-all-points")
            if cfg.get("preview_ply", True):
                cmd.append("--preview-ply")
            if args.force_rooms or not args.resume:
                cmd.append("--force")
            run_step(cmd)

    rooms_for_chunking = str(cfg.get("room_polygons") or "")
    if (
        not rooms_for_chunking
        and str(cfg.get("room_detection", "none")).lower() == "point2graph"
        and bool(cfg.get("room_refinement", True))
        and refined_rooms_json.exists()
    ):
        rooms_for_chunking = str(refined_rooms_json)
    if not rooms_for_chunking and str(cfg.get("room_detection", "none")).lower() == "point2graph" and detected_rooms_json.exists():
        rooms_for_chunking = str(detected_rooms_json)

    if start_index <= stages.index("chunks"):
        cmd = [
            sys.executable,
            str(PIPELINE_ROOT / "scripts" / "make_chunks.py"),
            "--preprocess-manifest",
            str(preprocess_manifest),
            "--work-dir",
            str(work_dir),
            "--halo",
            str(cfg["halo_m"]),
            "--max-points",
            str(cfg["max_points_per_chunk"]),
            "--room-full-max-points",
            str(cfg["room_full_max_points"]),
            "--min-points",
            str(cfg["min_points_per_chunk"]),
            "--min-split-points",
            str(cfg["min_split_points_per_chunk"]),
            "--min-tile-size",
            str(cfg["min_tile_size_m"]),
        ]
        if rooms_for_chunking:
            cmd.extend(["--room-polygons", str(Path(rooms_for_chunking).expanduser().resolve())])
        if args.force_chunks or not args.resume:
            cmd.append("--force")
        run_step(cmd)

    if start_index <= stages.index("batch"):
        cmd = [
            sys.executable,
            str(PIPELINE_ROOT / "scripts" / "run_mosaic3d_batch.py"),
            "--chunks-manifest",
            str(chunks_manifest),
            "--work-dir",
            str(work_dir),
            "--ckpt",
            str(Path(args.ckpt).expanduser().resolve()),
            "--labels-file",
            str(Path(args.labels_file).expanduser().resolve()),
            "--runner",
            str(cfg["runner"]),
            "--docker-image",
            str(cfg["docker_image"]),
            "--device",
            str(cfg["device"]),
            "--max-points",
            str(cfg["mosaic3d_max_points"]),
            "--grid-size",
            str(cfg["grid_size"]),
            "--condition",
            str(cfg["condition"]),
            "--prompt-template",
            str(cfg["prompt_template"]),
        ]
        if cfg.get("prompt_ensemble_file"):
            cmd.extend(["--prompt-ensemble-file", str(Path(cfg["prompt_ensemble_file"]).expanduser().resolve())])
        if args.resume:
            cmd.append("--resume")
        run_step(cmd)
        if cfg["runner"] == "print":
            print("[pipeline] runner=print; Mosaic3D commands were printed, so stitching is skipped.")
            return

    if start_index <= stages.index("stitch"):
        cmd = [
            sys.executable,
            str(PIPELINE_ROOT / "scripts" / "stitch_scene_graph.py"),
            "--preprocess-manifest",
            str(preprocess_manifest),
            "--chunks-manifest",
            str(chunks_manifest),
            "--batch-manifest",
            str(batch_manifest),
            "--work-dir",
            str(work_dir),
            "--instance-labels",
            str(cfg["instance_labels"]),
            "--instance-voxel-size",
            str(cfg["instance_voxel_size"]),
            "--instance-min-points",
            str(cfg["instance_min_points"]),
            "--instance-min-mean-score",
            str(cfg["instance_min_mean_score"]),
            "--object-relabel-candidates",
            str(cfg.get("object_relabel_candidates", "all")),
            "--object-relabel-top-k",
            str(cfg.get("object_relabel_top_k", 5)),
            "--object-relabel-min-score",
            str(cfg.get("object_relabel_min_score", 0.0)),
            "--object-relabel-min-margin",
            str(cfg.get("object_relabel_min_margin", 0.0)),
            "--object-relabel-top-point-fraction",
            str(cfg.get("object_relabel_top_point_fraction", 1.0)),
            "--object-relabel-min-point-score",
            str(cfg.get("object_relabel_min_point_score", 0.0)),
            "--object-relabel-min-center-weight",
            str(cfg.get("object_relabel_min_center_weight", 0.0)),
            "--object-relabel-temperature",
            str(cfg.get("object_relabel_temperature", 1.0)),
            "--object-relabel-unknown-label",
            str(cfg.get("object_relabel_unknown_label", "")),
            "--object-relabel-unknown-min-score",
            str(cfg.get("object_relabel_unknown_min_score", 0.0)),
            "--object-relabel-unknown-min-margin",
            str(cfg.get("object_relabel_unknown_min_margin", 0.0)),
            "--max-instances-per-label",
            str(cfg["max_instances_per_label"]),
            "--near-threshold",
            str(cfg["near_threshold_m"]),
            "--support-z-tolerance",
            str(cfg["support_z_tolerance_m"]),
        ]
        if bool(cfg.get("object_relabel", False)):
            cmd.append("--object-relabel")
        if bool(cfg.get("object_relabel_geometry_filter", False)):
            cmd.append("--object-relabel-geometry-filter")
        if rooms_for_chunking:
            cmd.extend(["--room-polygons", str(Path(rooms_for_chunking).expanduser().resolve())])
        if args.force_stitch:
            cmd.append("--force")
        run_step(cmd)

    print("[pipeline] done")
    print(f"[pipeline] stitched output: {work_dir / '04_stitched' / 'floor_mosaic3d_stitched_colorized.ply'}")
    print(f"[pipeline] scene graph: {work_dir / '04_stitched' / 'scene_graph.json'}")


if __name__ == "__main__":
    main()
