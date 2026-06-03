#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

from common import REPO_ROOT, container_path, read_json, read_labels_file, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Mosaic3D inference for every generated chunk.")
    parser.add_argument("--chunks-manifest", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--labels-file", required=True)
    parser.add_argument("--runner", default="docker", choices=["docker", "python", "print"])
    parser.add_argument("--docker-image", default="mosaic3d:latest")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--max-points", type=int, default=250000)
    parser.add_argument("--grid-size", type=float, default=0.02)
    parser.add_argument("--condition", default="ScanNet")
    parser.add_argument("--prompt-template", default="a {} in a scene")
    parser.add_argument("--prompt-ensemble-file", default="")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def build_python_command(
    input_path: Path,
    output_dir: Path,
    args: argparse.Namespace,
) -> list[str]:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "infer_pointcloud.py"),
        "--input",
        str(input_path),
        "--ckpt",
        str(Path(args.ckpt).expanduser().resolve()),
        "--labels-file",
        str(Path(args.labels_file).expanduser().resolve()),
        "--output-dir",
        str(output_dir),
        "--max-points",
        str(args.max_points),
        "--grid-size",
        str(args.grid_size),
        "--condition",
        str(args.condition),
        "--prompt-template",
        str(args.prompt_template),
        "--device",
        str(args.device),
    ]
    if args.prompt_ensemble_file:
        cmd.extend(["--prompt-ensemble-file", str(Path(args.prompt_ensemble_file).expanduser().resolve())])
    return cmd


def build_docker_command(
    input_path: Path,
    output_dir: Path,
    args: argparse.Namespace,
) -> list[str]:
    home = Path.home().resolve()
    ckpt_path = Path(args.ckpt).expanduser().resolve()
    labels_file = Path(args.labels_file).expanduser().resolve()
    prompt_ensemble_file = Path(args.prompt_ensemble_file).expanduser().resolve() if args.prompt_ensemble_file else None
    infer_args = [
        "python",
        "scripts/infer_pointcloud.py",
        "--input",
        container_path(input_path),
        "--ckpt",
        container_path(ckpt_path),
        "--labels-file",
        container_path(labels_file),
        "--output-dir",
        container_path(output_dir),
        "--max-points",
        str(args.max_points),
        "--grid-size",
        str(args.grid_size),
        "--condition",
        str(args.condition),
        "--prompt-template",
        str(args.prompt_template),
        "--device",
        str(args.device),
    ]
    if prompt_ensemble_file:
        infer_args.extend(["--prompt-ensemble-file", container_path(prompt_ensemble_file)])
    container_output_dir = container_path(output_dir)
    shell_cmd = (
        "python -m pip install -q transformers==4.44.2 "
        ">/tmp/mosaic3d_transformers_pin.log 2>&1 && "
        f"{shlex.join(infer_args)}; "
        "status=$?; "
        f"chmod -R a+rwX {shlex.quote(container_output_dir)} >/dev/null 2>&1 || true; "
        "exit $status"
    )
    cmd = [
        "docker",
        "run",
        "--rm",
        "--gpus",
        "all",
        "--shm-size=32g",
        "--ipc",
        "host",
        "--net",
        "host",
        "-v",
        f"{REPO_ROOT.resolve()}:/workspace",
        "-v",
        f"{home}:/root",
        "--workdir",
        "/workspace",
        str(args.docker_image),
        "bash",
        "-lc",
        shell_cmd,
    ]
    datasets = home / "ws_demo" / "datasets"
    if datasets.exists():
        insert_at = cmd.index("--workdir")
        cmd[insert_at:insert_at] = ["-v", f"{datasets}:/datasets"]
    mount_paths = [input_path, output_dir, ckpt_path, labels_file]
    if prompt_ensemble_file:
        mount_paths.append(prompt_ensemble_file)
    for extra in extra_docker_mounts(mount_paths):
        insert_at = cmd.index("--workdir")
        cmd[insert_at:insert_at] = ["-v", f"{extra}:{extra}"]
    return cmd


def extra_docker_mounts(paths: list[Path]) -> list[str]:
    roots: list[Path] = []
    home = Path.home().resolve()
    repo = REPO_ROOT.resolve()
    for path in paths:
        resolved = Path(path).expanduser().resolve()
        if is_relative_to(resolved, repo) or is_relative_to(resolved, home):
            continue
        root = Path("/tmp") if is_relative_to(resolved, Path("/tmp")) else resolved.parent
        if root not in roots:
            roots.append(root)
    return [str(root) for root in roots]


def check_docker_gpu(args: argparse.Namespace) -> None:
    cmd = [
        "docker",
        "run",
        "--rm",
        "--gpus",
        "all",
        str(args.docker_image),
        "nvidia-smi",
    ]
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), text=True, capture_output=True)
    if proc.returncode == 0:
        return
    details = (proc.stderr or proc.stdout or "").strip()
    raise SystemExit(
        "Docker GPU preflight failed before Mosaic3D inference.\n"
        "Host nvidia-smi can work while Docker still cannot use the NVIDIA runtime.\n"
        "Test manually with:\n"
        f"  {' '.join(shlex.quote(x) for x in cmd)}\n"
        "If it fails, restart/configure the NVIDIA container runtime, then resume with --start-at batch --resume.\n\n"
        f"Docker output:\n{details}"
    )


def is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def expected_prediction_path(output_dir: Path, input_path: Path) -> Path:
    return output_dir / f"{input_path.stem}_mosaic3d_predictions.npz"


def labels_hash(labels: list[str]) -> str:
    payload = "\n".join(labels).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path | None) -> str:
    if not path:
        return ""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prediction_config_matches(prediction_path: Path, labels: list[str], prompt_ensemble_sha256: str) -> bool:
    if not prediction_path.exists():
        return False
    try:
        data = np.load(prediction_path, allow_pickle=False)
        pred_labels = [str(x) for x in data["labels"]]
        if pred_labels != labels:
            return False
        if prompt_ensemble_sha256:
            saved_hash = str(data["prompt_ensemble_sha256"]) if "prompt_ensemble_sha256" in data else ""
            return saved_hash == prompt_ensemble_sha256
        return True
    except Exception:
        return False


def main() -> None:
    args = parse_args()
    work_dir = Path(args.work_dir).expanduser().resolve()
    inference_dir = work_dir / "03_inference"
    inference_dir.mkdir(parents=True, exist_ok=True)
    labels_file = Path(args.labels_file).expanduser().resolve()
    labels = read_labels_file(labels_file)
    label_hash = labels_hash(labels)
    prompt_ensemble_file = Path(args.prompt_ensemble_file).expanduser().resolve() if args.prompt_ensemble_file else None
    prompt_ensemble_sha256 = file_sha256(prompt_ensemble_file)

    chunks_manifest = read_json(Path(args.chunks_manifest))
    batch_entries: list[dict[str, Any]] = []
    failures = 0

    if args.runner == "docker":
        check_docker_gpu(args)

    for i, chunk in enumerate(chunks_manifest["chunks"], start=1):
        chunk_id = chunk["chunk_id"]
        input_path = Path(chunk["cloud_path"]).expanduser().resolve()
        output_dir = inference_dir / chunk_id
        output_dir.mkdir(parents=True, exist_ok=True)
        prediction_path = expected_prediction_path(output_dir, input_path)
        stdout_log = output_dir / "stdout.log"
        stderr_log = output_dir / "stderr.log"

        entry = {
            **chunk,
            "output_dir": str(output_dir),
            "prediction_path": str(prediction_path),
            "stdout_log": str(stdout_log),
            "stderr_log": str(stderr_log),
            "status": "pending",
        }

        if prediction_path.exists() and args.resume and prediction_config_matches(
            prediction_path, labels, prompt_ensemble_sha256
        ):
            entry["status"] = "done"
            batch_entries.append(entry)
            print(f"[batch] {i}/{chunks_manifest['chunk_count']} reuse {chunk_id}")
            continue
        if prediction_path.exists() and args.resume:
            print(f"[batch] {i}/{chunks_manifest['chunk_count']} rerun {chunk_id} because labels/prompts changed")

        cmd = (
            build_python_command(input_path, output_dir, args)
            if args.runner == "python"
            else build_docker_command(input_path, output_dir, args)
        )
        entry["command"] = cmd
        if args.runner == "print":
            print(shlex.join(cmd))
            entry["status"] = "printed"
            batch_entries.append(entry)
            continue

        print(f"[batch] {i}/{chunks_manifest['chunk_count']} running {chunk_id} ({chunk['point_count']:,} points)")
        proc = subprocess.run(cmd, cwd=str(REPO_ROOT), text=True, capture_output=True)
        stdout_log.write_text(proc.stdout or "", encoding="utf-8")
        stderr_log.write_text(proc.stderr or "", encoding="utf-8")
        if proc.returncode == 0 and prediction_path.exists():
            entry["status"] = "done"
        else:
            failures += 1
            entry["status"] = "failed"
            entry["returncode"] = int(proc.returncode)
            entry["error_tail"] = ((proc.stderr or proc.stdout or "")[-2000:])
            print(f"[batch] FAILED {chunk_id}: see {stderr_log}")
        batch_entries.append(entry)

    batch_manifest = {
        "chunks_manifest": str(Path(args.chunks_manifest).resolve()),
        "inference_dir": str(inference_dir),
        "runner": args.runner,
        "docker_image": args.docker_image,
        "device": args.device,
        "labels_file": str(labels_file),
        "labels": labels,
        "labels_sha256": label_hash,
        "prompt_template": str(args.prompt_template),
        "prompt_ensemble_file": str(prompt_ensemble_file or ""),
        "prompt_ensemble_sha256": str(prompt_ensemble_sha256 or ""),
        "max_points": int(args.max_points),
        "chunk_count": len(batch_entries),
        "failed_count": failures,
        "chunks": batch_entries,
    }
    manifest_path = inference_dir / "batch_manifest.json"
    write_json(manifest_path, batch_manifest)
    print(f"[batch] wrote {manifest_path}")
    if failures:
        raise SystemExit(f"{failures} chunk(s) failed")


if __name__ == "__main__":
    main()
