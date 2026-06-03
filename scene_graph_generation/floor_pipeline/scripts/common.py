from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np


PIPELINE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[2]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def safe_stem(path: Path) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in path.stem)


def palette(num_labels: int) -> np.ndarray:
    import colorsys

    hues = (np.arange(num_labels) * 0.618033988749895) % 1.0
    colors = [colorsys.hsv_to_rgb(float(hue), 0.65, 0.95) for hue in hues]
    return (np.asarray(colors, dtype=np.float32) * 255).astype(np.uint8)


def find_cloudcompare_executable() -> str:
    env_value = (
        os.environ.get("CLOUDCOMPARE_BIN")
        or os.environ.get("CLOUDCOMPARE_EXECUTABLE")
        or os.environ.get("CLOUDCOMPARE_PATH")
    )
    candidates = [
        env_value,
        "CloudCompare",
        "cloudcompare",
        "cloudcompare.CloudCompare",
        "/snap/bin/cloudcompare.CloudCompare",
        "/usr/bin/CloudCompare",
        "/usr/bin/cloudcompare",
        "/usr/local/bin/CloudCompare",
        "/usr/local/bin/cloudcompare",
    ]
    for candidate in candidates:
        if not candidate:
            continue
        resolved = shutil.which(candidate)
        if resolved:
            return resolved
        path = Path(candidate).expanduser()
        if path.exists() and os.access(path, os.X_OK):
            return str(path)
    raise FileNotFoundError(
        "CloudCompare executable was not found. Install CloudCompare or set "
        "CLOUDCOMPARE_BIN=/path/to/CloudCompare. On this machine it is often "
        "/snap/bin/cloudcompare.CloudCompare."
    )


def parse_binary_ply_header(path: Path) -> tuple[int, int, np.dtype]:
    type_map = {
        "char": "i1",
        "int8": "i1",
        "uchar": "u1",
        "uint8": "u1",
        "short": "<i2",
        "int16": "<i2",
        "ushort": "<u2",
        "uint16": "<u2",
        "int": "<i4",
        "int32": "<i4",
        "uint": "<u4",
        "uint32": "<u4",
        "float": "<f4",
        "float32": "<f4",
        "double": "<f8",
        "float64": "<f8",
    }
    vertex_count = 0
    vertex_properties: list[tuple[str, str]] = []
    in_vertex = False
    with path.open("rb") as handle:
        first = handle.readline().decode("ascii", errors="replace").strip()
        if first != "ply":
            raise ValueError(f"{path} is not a PLY file.")
        while True:
            line_bytes = handle.readline()
            if not line_bytes:
                raise ValueError(f"{path} ended before end_header.")
            line = line_bytes.decode("ascii", errors="replace").strip()
            if line == "end_header":
                header_size = handle.tell()
                break
            parts = line.split()
            if not parts:
                continue
            if parts[:2] == ["format", "binary_little_endian"]:
                continue
            if parts[0] == "format" and parts[1] != "binary_little_endian":
                raise ValueError(f"Only binary_little_endian PLY is supported; got: {line}")
            if parts[0] == "element":
                in_vertex = len(parts) >= 3 and parts[1] == "vertex"
                if in_vertex:
                    vertex_count = int(parts[2])
                continue
            if in_vertex and parts[0] == "property":
                if len(parts) >= 2 and parts[1] == "list":
                    raise ValueError("PLY list properties are not supported for streaming fallback.")
                if len(parts) >= 3:
                    prop_type = parts[1]
                    prop_name = parts[2]
                    if prop_type not in type_map:
                        raise ValueError(f"Unsupported PLY property type: {prop_type}")
                    vertex_properties.append((prop_name, type_map[prop_type]))
    required = {"x", "y", "z"}
    names = [name for name, _ in vertex_properties]
    if not required.issubset(set(names)):
        raise ValueError("PLY must contain x, y, z vertex properties.")
    dtype = np.dtype(vertex_properties)
    return header_size, vertex_count, dtype


def pack_voxel_keys(voxels: np.ndarray) -> np.ndarray:
    offset = np.int64(1 << 20)
    shifted = voxels.astype(np.int64, copy=False) + offset
    if shifted.min(initial=0) < 0 or shifted.max(initial=0) >= (1 << 21):
        raise ValueError("Voxel coordinate range is too large for packed-key streaming fallback.")
    return (
        (shifted[:, 0].astype(np.uint64) << np.uint64(42))
        | (shifted[:, 1].astype(np.uint64) << np.uint64(21))
        | shifted[:, 2].astype(np.uint64)
    )


def stream_subsample_binary_ply_to_npy(
    input_path: Path,
    output_path: Path,
    voxel_size: float,
    chunk_vertices: int = 5_000_000,
) -> None:
    header_size, vertex_count, dtype = parse_binary_ply_header(input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path = output_path.with_suffix(output_path.suffix + ".raw")
    seen: set[int] = set()
    selected_count = 0
    mmap = np.memmap(input_path, mode="r", offset=header_size, dtype=dtype, shape=(vertex_count,))
    print(
        f"[preprocess] CloudCompare not found; streaming {vertex_count:,} PLY vertices "
        f"with spatial step {voxel_size:g}"
    )
    with tempfile.NamedTemporaryFile(
        dir=str(output_path.parent),
        prefix=output_path.name + ".",
        suffix=".raw",
        delete=False,
    ) as raw_handle:
        raw_path = Path(raw_handle.name)
        for start in range(0, vertex_count, chunk_vertices):
            end = min(start + chunk_vertices, vertex_count)
            block = mmap[start:end]
            coord = np.column_stack(
                (
                    np.asarray(block["x"], dtype=np.float32),
                    np.asarray(block["y"], dtype=np.float32),
                    np.asarray(block["z"], dtype=np.float32),
                )
            )
            voxels = np.floor(coord / float(voxel_size)).astype(np.int64)
            keys = pack_voxel_keys(voxels)
            unique_keys, first = np.unique(keys, return_index=True)
            keep_local: list[int] = []
            for key, index in zip(unique_keys, first):
                key_int = int(key)
                if key_int in seen:
                    continue
                seen.add(key_int)
                keep_local.append(int(index))
            if keep_local:
                keep = np.asarray(keep_local, dtype=np.int64)
                if {"red", "green", "blue"}.issubset(dtype.names or ()):
                    color = np.column_stack(
                        (
                            np.asarray(block["red"][keep], dtype=np.float32),
                            np.asarray(block["green"][keep], dtype=np.float32),
                            np.asarray(block["blue"][keep], dtype=np.float32),
                        )
                    )
                elif "scalar_Intensity" in (dtype.names or ()):
                    intensity = np.asarray(block["scalar_Intensity"][keep], dtype=np.float32)
                    if intensity.max(initial=0) > 0:
                        intensity = intensity / intensity.max(initial=0) * 255.0
                    color = np.repeat(intensity[:, None], 3, axis=1)
                else:
                    color = np.full((len(keep), 3), 127.5, dtype=np.float32)
                rows = np.hstack([coord[keep], color]).astype(np.float32, copy=False)
                rows.tofile(raw_handle)
                selected_count += int(len(rows))
            print(
                f"[preprocess] streamed {end:,}/{vertex_count:,}; kept {selected_count:,}",
                flush=True,
            )

    try:
        with output_path.open("wb") as npy_handle, raw_path.open("rb") as raw_handle:
            np.lib.format.write_array_header_2_0(
                npy_handle,
                {
                    "descr": np.dtype("<f4").str,
                    "fortran_order": False,
                    "shape": (selected_count, 6),
                },
            )
            shutil.copyfileobj(raw_handle, npy_handle, length=64 * 1024 * 1024)
    finally:
        raw_path.unlink(missing_ok=True)
    print(f"[preprocess] wrote streamed subsample: {output_path} ({selected_count:,} points)")


def prepare_pointcloud_input(
    path: Path,
    output_dir: Path,
    spatial_subsample: float | None = None,
) -> Path:
    path = Path(path).expanduser().resolve()
    suffix_lower = path.suffix.lower()
    should_prepare = suffix_lower == ".bin" or (
        spatial_subsample is not None
        and spatial_subsample > 0
        and suffix_lower in {".ply", ".las", ".laz", ".e57"}
    )
    if not should_prepare:
        return path

    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = f"{path.stat().st_size}_{int(path.stat().st_mtime)}"
    suffix = ""
    if spatial_subsample is not None and spatial_subsample > 0:
        suffix = f"_ss{float(spatial_subsample):.3f}".replace(".", "p")
    output_path = output_dir / f"{safe_stem(path)}_cc_{stamp}{suffix}.ply"
    if output_path.exists() and output_path.stat().st_size > 0:
        return output_path

    try:
        cloudcompare_bin = find_cloudcompare_executable()
    except FileNotFoundError:
        if suffix_lower == ".ply" and spatial_subsample is not None and spatial_subsample > 0:
            fallback_path = output_dir / f"{safe_stem(path)}_stream_{stamp}{suffix}.npy"
            if fallback_path.exists() and fallback_path.stat().st_size > 0:
                return fallback_path
            stream_subsample_binary_ply_to_npy(path, fallback_path, float(spatial_subsample))
            return fallback_path
        raise
    cmd = [
        cloudcompare_bin,
        "-SILENT",
        "-NO_TIMESTAMP",
        "-AUTO_SAVE",
        "OFF",
        "-O",
        str(path),
        "-MERGE_CLOUDS",
    ]
    if spatial_subsample is not None and spatial_subsample > 0:
        cmd.extend(["-SS", "SPATIAL", str(float(spatial_subsample))])
    cmd.extend(
        [
        "-C_EXPORT_FMT",
        "PLY",
        "-PLY_EXPORT_FMT",
        "BINARY_LE",
        "-SAVE_CLOUDS",
        "FILE",
        str(output_path),
        ]
    )
    env = os.environ.copy()
    env["QT_QPA_PLATFORM"] = "offscreen"
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), env=env, text=True, capture_output=True)
    if not output_path.exists():
        candidates = [
            candidate
            for candidate in output_dir.glob(f"{output_path.name}*")
            if candidate.is_file() and candidate.stat().st_size > 0
        ]
        if candidates:
            newest = max(candidates, key=lambda item: item.stat().st_mtime)
            newest.replace(output_path)
    if proc.returncode != 0 or not output_path.exists():
        details = "\n".join(x for x in (proc.stderr, proc.stdout) if x).strip()
        raise RuntimeError(
            "Could not convert CloudCompare .bin input to .ply. "
            "Use a .ply/.las/.laz/.npy input or install CloudCompare.\n"
            f"{details[-2000:]}"
        )
    return output_path


def read_point_cloud(path: Path) -> tuple[np.ndarray, np.ndarray]:
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Point cloud not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".npy":
        array = np.load(path)
        if array.ndim != 2 or array.shape[1] not in (3, 6):
            raise ValueError(".npy input must have shape [N, 3] or [N, 6].")
        coord = np.asarray(array[:, :3], dtype=np.float32)
        color = np.asarray(array[:, 3:6], dtype=np.float32) if array.shape[1] == 6 else None
    elif suffix in (".las", ".laz"):
        coord, color = read_las_point_cloud(path)
    else:
        import open3d as o3d

        pcd = o3d.io.read_point_cloud(str(path))
        if pcd.is_empty():
            raise ValueError(f"Could not read a point cloud from {path}")
        coord = np.asarray(pcd.points, dtype=np.float32)
        color = np.asarray(pcd.colors, dtype=np.float32) if pcd.has_colors() else None

    if color is None or len(color) == 0:
        color = np.full((len(coord), 3), 127.5, dtype=np.float32)
    elif color.max(initial=0) <= 1.0:
        color = color * 255.0
    color = np.clip(color, 0, 255).astype(np.float32)
    return coord, color


def read_las_point_cloud(path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    try:
        import laspy
    except ImportError as exc:
        raise ImportError("Reading .las/.laz requires: pip install laspy lazrs") from exc

    las = laspy.read(path)
    coord = np.column_stack((las.x, las.y, las.z)).astype(np.float32)
    dimensions = set(las.point_format.dimension_names)
    color = None
    if {"red", "green", "blue"}.issubset(dimensions):
        color = np.column_stack((las.red, las.green, las.blue)).astype(np.float32)
        if color.max(initial=0) > 255.0:
            color = color / 65535.0 * 255.0
    elif "intensity" in dimensions:
        intensity = np.asarray(las.intensity, dtype=np.float32)
        denom = float(intensity.max(initial=0) - intensity.min(initial=0))
        if denom > 0:
            intensity = (intensity - intensity.min()) / denom * 255.0
        color = np.repeat(intensity[:, None], 3, axis=1)
    return coord, color


def write_point_cloud(path: Path, coord: np.ndarray, color: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import open3d as o3d

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.asarray(coord, dtype=np.float64))
        normalized = np.asarray(color, dtype=np.float32)
        if normalized.max(initial=0) > 1.0:
            normalized = normalized / 255.0
        pcd.colors = o3d.utility.Vector3dVector(np.clip(normalized, 0.0, 1.0).astype(np.float64))
        o3d.io.write_point_cloud(str(path), pcd)
        return
    except Exception as exc:
        print(f"[common] Open3D writer unavailable for {path}: {exc}. Falling back to binary PLY.")

    write_binary_ply(path, coord, color)


def write_binary_ply(path: Path, coord: np.ndarray, color: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    coord = np.asarray(coord, dtype=np.float32)
    color = np.asarray(color, dtype=np.float32)
    if color.max(initial=0) <= 1.0:
        color = color * 255.0
    color = np.clip(color, 0.0, 255.0).astype(np.uint8)
    if len(coord) != len(color):
        raise ValueError("coord and color must have the same length")

    vertex = np.empty(
        len(coord),
        dtype=[
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ],
    )
    vertex["x"] = coord[:, 0]
    vertex["y"] = coord[:, 1]
    vertex["z"] = coord[:, 2]
    vertex["red"] = color[:, 0]
    vertex["green"] = color[:, 1]
    vertex["blue"] = color[:, 2]
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {len(vertex)}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    )
    with path.open("wb") as handle:
        handle.write(header.encode("ascii"))
        vertex.tofile(handle)


def voxel_downsample_representatives(
    coord: np.ndarray,
    color: np.ndarray,
    voxel_size: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return one representative point per voxel and its source index.

    The source index allows the final inference cloud to be traced back to the
    original dense cloud. This keeps preprocessing deterministic and cheap.
    """

    if voxel_size <= 0 or len(coord) == 0:
        source_index = np.arange(len(coord), dtype=np.int64)
        return coord.astype(np.float32), color.astype(np.float32), source_index

    origin = np.min(coord, axis=0)
    voxels = np.floor((coord - origin) / float(voxel_size)).astype(np.int64)
    _, first = np.unique(voxels, axis=0, return_index=True)
    first = np.sort(first.astype(np.int64))
    return coord[first].astype(np.float32), color[first].astype(np.float32), first


def save_chunk_npy(path: Path, coord: np.ndarray, color: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.hstack(
        [np.asarray(coord, dtype=np.float32), np.asarray(color, dtype=np.float32)]
    )
    np.save(path, array.astype(np.float32))


def read_labels_file(path: Path) -> list[str]:
    labels = [
        line.strip()
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    if len(labels) < 2:
        raise ValueError(f"Need at least two labels in {path}")
    return labels


def load_room_polygons(path: str | Path | None) -> list[dict[str, Any]]:
    if not path:
        return []
    json_path = Path(path).expanduser().resolve()
    payload = read_json(json_path)
    rooms = payload.get("rooms", payload if isinstance(payload, list) else [])
    normalized = []
    for i, room in enumerate(rooms):
        polygon = room.get("polygon_xy") or room.get("polygon") or []
        if len(polygon) < 3:
            continue
        point_indices_npz = str(room.get("point_indices_npz") or "").strip()
        if point_indices_npz:
            point_indices_path = Path(point_indices_npz).expanduser()
            if not point_indices_path.is_absolute():
                point_indices_path = json_path.parent / point_indices_path
            point_indices_npz = str(point_indices_path.resolve())
        normalized.append(
            {
                "room_id": str(room.get("room_id") or room.get("id") or f"room_{i + 1:03d}"),
                "room_type": str(room.get("room_type") or room.get("type") or "room"),
                "polygon_xy": [[float(x), float(y)] for x, y in polygon],
                "point_indices_npz": point_indices_npz,
                "point_indices_key": str(room.get("point_indices_key") or ""),
            }
        )
    return normalized


def points_in_polygon_xy(xy: np.ndarray, polygon: list[list[float]]) -> np.ndarray:
    """Vectorized ray-casting test for an XY polygon."""

    poly = np.asarray(polygon, dtype=np.float64)
    x = xy[:, 0].astype(np.float64)
    y = xy[:, 1].astype(np.float64)
    inside = np.zeros(len(xy), dtype=bool)
    j = len(poly) - 1
    for i in range(len(poly)):
        xi, yi = poly[i]
        xj, yj = poly[j]
        crosses = ((yi > y) != (yj > y)) & (
            x < (xj - xi) * (y - yi) / ((yj - yi) if abs(yj - yi) > 1e-12 else 1e-12) + xi
        )
        inside ^= crosses
        j = i
    return inside


def container_path(path: Path, repo_root: Path = REPO_ROOT) -> str:
    path = Path(path).expanduser().resolve()
    repo_root = repo_root.resolve()
    home = Path.home().resolve()
    try:
        return str(Path("/workspace") / path.relative_to(repo_root))
    except ValueError:
        pass
    try:
        return str(Path("/root") / path.relative_to(home))
    except ValueError:
        return str(path)
