#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from common import (
    prepare_pointcloud_input,
    read_point_cloud,
    safe_stem,
    voxel_downsample_representatives,
    write_json,
    write_point_cloud,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess a full-floor point cloud for chunked Mosaic3D inference."
    )
    parser.add_argument("--input", required=True, help="Full-floor .ply/.las/.laz/.npy/.bin point cloud.")
    parser.add_argument("--work-dir", required=True, help="Pipeline working directory.")
    parser.add_argument("--voxel-size", type=float, default=0.03, help="Inference voxel size in meters. Use 0 to skip.")
    parser.add_argument("--preview-ply", action="store_true", help="Write a downsampled preview PLY.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing preprocess outputs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    work_dir = Path(args.work_dir).expanduser().resolve()
    preprocess_dir = work_dir / "01_preprocess"
    converted_dir = work_dir / "converted_inputs"
    preprocess_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = preprocess_dir / "preprocess_manifest.json"
    cloud_npz = preprocess_dir / "floor_infer.npz"
    if manifest_path.exists() and cloud_npz.exists() and not args.force:
        print(f"[preprocess] Reusing {manifest_path}")
        return

    source_input = Path(args.input).expanduser().resolve()
    source_suffix = source_input.suffix.lower()
    large_cloud_bytes = 2 * 1024 * 1024 * 1024
    use_cloudcompare_subsample = (
        args.voxel_size > 0
        and (
            source_suffix == ".bin"
            or (
                source_suffix in {".ply", ".las", ".laz", ".e57"}
                and source_input.stat().st_size >= large_cloud_bytes
            )
        )
    )
    cloudcompare_spatial_subsample = float(args.voxel_size) if use_cloudcompare_subsample else None
    prepared_input = prepare_pointcloud_input(
        source_input,
        converted_dir,
        spatial_subsample=cloudcompare_spatial_subsample,
    )
    coord, color = read_point_cloud(prepared_input)
    source_count = int(len(coord))
    if cloudcompare_spatial_subsample is not None:
        infer_coord = coord.astype(np.float32)
        infer_color = color.astype(np.float32)
        source_index = np.arange(len(coord), dtype=np.int64)
    else:
        infer_coord, infer_color, source_index = voxel_downsample_representatives(
            coord, color, float(args.voxel_size)
        )

    np.savez_compressed(
        cloud_npz,
        coord=infer_coord.astype(np.float32),
        color=infer_color.astype(np.float32),
        source_index=source_index.astype(np.int64),
        source_input=str(source_input),
        prepared_input=str(prepared_input),
        voxel_size=float(args.voxel_size),
    )

    preview_path = ""
    if args.preview_ply:
        preview = preprocess_dir / f"{safe_stem(source_input)}_infer_preview.ply"
        write_point_cloud(preview, infer_coord, infer_color)
        preview_path = str(preview)

    bbox_min = infer_coord.min(axis=0).astype(float).tolist() if len(infer_coord) else [0, 0, 0]
    bbox_max = infer_coord.max(axis=0).astype(float).tolist() if len(infer_coord) else [0, 0, 0]
    manifest = {
        "source_input": str(source_input),
        "prepared_input": str(prepared_input),
        "cloud_npz": str(cloud_npz),
        "preview_ply": preview_path,
        "source_point_count": source_count,
        "infer_point_count": int(len(infer_coord)),
        "voxel_size": float(args.voxel_size),
        "bin_spatial_subsample": cloudcompare_spatial_subsample,
        "cloudcompare_spatial_subsample": cloudcompare_spatial_subsample,
        "bbox_min": bbox_min,
        "bbox_max": bbox_max,
    }
    write_json(manifest_path, manifest)
    print(f"[preprocess] source_points={source_count:,}")
    print(f"[preprocess] infer_points={len(infer_coord):,}")
    print(f"[preprocess] wrote {manifest_path}")


if __name__ == "__main__":
    main()
