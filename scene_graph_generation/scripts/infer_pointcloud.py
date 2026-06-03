import argparse
import colorsys
import hashlib
import json
from pathlib import Path

import hydra
import numpy as np
import open3d as o3d
import rootutils
import torch
import torch.nn.functional as F
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.utils.caption_utils import forward_text_encoder  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Mosaic3D open-vocabulary semantic segmentation on one point cloud."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input .ply, .pcd, .xyz, .xyzn, .xyzrgb, .las, .laz, or .npy file.",
    )
    parser.add_argument("--ckpt", required=True, help="Path to a Mosaic3D encoder checkpoint.")
    parser.add_argument(
        "--labels",
        help="Comma-separated labels, for example 'floor,wall,chair,table,door'.",
    )
    parser.add_argument("--labels-file", help="Text file with one label per line.")
    parser.add_argument("--output-dir", default="outputs/custom_inference")
    parser.add_argument("--data-config", default="sc+ar+sc++")
    parser.add_argument("--model-config", default="spunet34c+ppt")
    parser.add_argument("--experiment", default="train_spunet_multidata_ppt")
    parser.add_argument("--condition", default="ScanNet")
    parser.add_argument("--grid-size", type=float, default=0.02)
    parser.add_argument("--max-points", type=int, default=250000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--prompt-template",
        default="a {} in a scene",
        help="Prompt wrapped around each label. Use an empty string to pass labels directly.",
    )
    parser.add_argument(
        "--prompt-ensemble-file",
        default="",
        help=(
            "Optional JSON mapping labels to multiple prompts. The normalized text embeddings "
            "for each label are averaged and normalized again."
        ),
    )
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    return parser.parse_args()


def read_labels(labels, labels_file):
    if labels_file:
        label_list = [
            line.strip()
            for line in Path(labels_file).read_text().splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
    elif labels:
        label_list = [label.strip() for label in labels.split(",") if label.strip()]
    else:
        raise ValueError("Provide --labels or --labels-file.")

    if len(label_list) < 2:
        raise ValueError(
            "Provide at least two labels so the open-vocabulary classifier can compare them."
        )
    return label_list


def read_point_cloud(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Input point cloud does not exist: {path}. "
            "If you are inside Docker, use the path as mounted inside the container."
        )

    suffix = path.suffix.lower()

    if suffix == ".npy":
        array = np.load(path)
        if array.ndim != 2 or array.shape[1] not in (3, 6):
            raise ValueError(".npy input must have shape [N, 3] or [N, 6].")
        coord = array[:, :3].astype(np.float32)
        color = array[:, 3:6].astype(np.float32) if array.shape[1] == 6 else None
    elif suffix in (".las", ".laz"):
        coord, color = read_las_point_cloud(path)
    else:
        pcd = o3d.io.read_point_cloud(str(path))
        if pcd.is_empty():
            raise ValueError(f"Could not read a point cloud from {path}.")
        coord = np.asarray(pcd.points, dtype=np.float32)
        color = np.asarray(pcd.colors, dtype=np.float32) if pcd.has_colors() else None

    if color is None or len(color) == 0:
        color = np.full_like(coord, 127.5, dtype=np.float32)
    elif color.max() <= 1.0:
        color = color * 255.0
    color = np.clip(color, 0, 255).astype(np.float32)
    return coord, color


def read_las_point_cloud(path):
    try:
        import laspy
    except ImportError as exc:
        raise ImportError(
            "Reading .las/.laz input requires laspy. Install it inside the Mosaic3D "
            "environment with: pip install laspy lazrs"
        ) from exc

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


def sample_points(coord, color, max_points, seed):
    num_points = coord.shape[0]
    if max_points <= 0 or num_points <= max_points:
        return coord, color, np.arange(num_points)

    rng = np.random.default_rng(seed)
    sample_idx = np.sort(rng.choice(num_points, max_points, replace=False))
    return coord[sample_idx], color[sample_idx], sample_idx


def center_shift(coord):
    coord = coord.copy()
    x_min, y_min, z_min = coord.min(axis=0)
    x_max, y_max, _ = coord.max(axis=0)
    shift = np.array([(x_min + x_max) / 2.0, (y_min + y_max) / 2.0, z_min], dtype=np.float32)
    coord -= shift
    return coord, shift


def build_batch(coord, color, condition, grid_size, device):
    shifted_coord, shift = center_shift(coord)
    normalized_color = color / 127.5 - 1.0
    num_points = coord.shape[0]
    return (
        {
            "coord": torch.from_numpy(shifted_coord).float().to(device),
            "origin_coord": torch.from_numpy(coord).float().to(device),
            "color": torch.from_numpy(normalized_color).float().to(device),
            "feat": torch.from_numpy(normalized_color).float().to(device),
            "offset": torch.tensor([0, num_points], dtype=torch.long, device=device),
            "grid_size": grid_size,
            "condition": [condition],
        },
        shift,
    )


def load_config(args):
    config_dir = Path(__file__).resolve().parents[1] / "configs"
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
        return compose(
            config_name="eval.yaml",
            overrides=[
                f"experiment={args.experiment}",
                f"data={args.data_config}",
                f"model={args.model_config}",
                f"ckpt_path={args.ckpt}",
                "trainer.devices=1",
            ],
        )


def load_model(cfg, ckpt_path, device):
    model = hydra.utils.instantiate(cfg.model)
    model.configure_model()

    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    state_dict = dict(state_dict)
    for key in list(state_dict.keys()):
        if "emb_target" in key:
            del state_dict[key]
    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    model.to(device)
    if hasattr(model, "clip_encoder"):
        model.clip_encoder.to(device)
    model.eval()
    return model, missing, unexpected


def make_prompts(labels, prompt_template):
    if not prompt_template:
        return labels
    return [label if label.lower() == "other" else prompt_template.format(label) for label in labels]


def label_key(label):
    return " ".join(str(label).strip().lower().replace("_", " ").split())


def file_sha256(path):
    path = Path(path)
    if not path:
        return ""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_prompt_ensemble_file(path):
    if not path:
        return {}, ""

    prompt_path = Path(path).expanduser().resolve()
    payload = json.loads(prompt_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("--prompt-ensemble-file must contain a JSON object mapping labels to prompt lists.")

    prompts_by_label = {}
    for label, prompts in payload.items():
        if isinstance(prompts, str):
            prompts = [prompts]
        if not isinstance(prompts, list):
            raise ValueError(f"Prompt ensemble for {label!r} must be a string or a list of strings.")
        clean_prompts = [str(prompt).strip() for prompt in prompts if str(prompt).strip()]
        if clean_prompts:
            prompts_by_label[label_key(label)] = clean_prompts
    return prompts_by_label, file_sha256(prompt_path)


def make_prompt_groups(labels, prompt_template, prompt_ensemble_file):
    prompts_by_label, prompt_ensemble_sha256 = read_prompt_ensemble_file(prompt_ensemble_file)
    fallback_prompts = make_prompts(labels, prompt_template)
    prompt_groups = []
    for label, fallback in zip(labels, fallback_prompts):
        group = prompts_by_label.get(label_key(label), [fallback])
        prompt_groups.append([prompt.format(label) if "{}" in prompt else prompt for prompt in group])
    prompts = [" || ".join(group) for group in prompt_groups]
    return prompt_groups, prompts, prompt_ensemble_sha256


def encode_prompt_groups(prompt_groups, clip_encoder, device):
    flat_prompts = [prompt for group in prompt_groups for prompt in group]
    if not flat_prompts:
        raise ValueError("No prompts were generated for the label set.")

    offsets = np.cumsum([0] + [len(group) for group in prompt_groups])
    flat_features = forward_text_encoder(flat_prompts, clip_encoder, normalize=True, device=device)
    text_features = []
    for start, end in zip(offsets[:-1], offsets[1:]):
        group_features = flat_features[int(start) : int(end)]
        group_feature = F.normalize(group_features.mean(dim=0, keepdim=True), p=2, dim=1)
        text_features.append(group_feature)
    return torch.cat(text_features, dim=0)


def palette(num_labels):
    hues = (np.arange(num_labels) * 0.618033988749895) % 1.0
    return np.array([colorsys.hsv_to_rgb(float(hue), 0.65, 0.95) for hue in hues], dtype=np.float32)


@torch.inference_mode()
def run_inference(model, cfg, batch, labels, prompt_template, prompt_ensemble_file, device):
    output = model(batch)
    point_features = output["clip_feat"]

    normalize_input = bool(
        OmegaConf.select(cfg, "model.eval_cfg.seg_eval.normalize_input", default=False)
    )
    if normalize_input:
        point_features = F.normalize(point_features, p=2, dim=1)

    prompt_groups, prompts, prompt_ensemble_sha256 = make_prompt_groups(
        labels, prompt_template, prompt_ensemble_file
    )
    text_features = encode_prompt_groups(prompt_groups, model.clip_encoder, device=device)
    logits = point_features @ text_features.t()
    probabilities = logits.softmax(dim=1)
    scores, pred_index = probabilities.max(dim=1)
    return (
        pred_index.cpu().numpy(),
        scores.cpu().numpy(),
        logits.cpu().numpy(),
        prompts,
        prompt_groups,
        prompt_ensemble_sha256,
    )


def save_outputs(
    output_dir,
    input_path,
    coord,
    pred_index,
    scores,
    logits,
    labels,
    prompts,
    prompt_groups,
    prompt_ensemble_file,
    prompt_ensemble_sha256,
    sample_idx,
    shift,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(input_path).stem

    npz_path = output_dir / f"{stem}_mosaic3d_predictions.npz"
    ply_path = output_dir / f"{stem}_mosaic3d_colorized.ply"
    json_path = output_dir / f"{stem}_mosaic3d_summary.json"

    pred_labels = np.array([labels[i] for i in pred_index])
    np.savez_compressed(
        npz_path,
        coord=coord,
        pred_index=pred_index,
        pred_label=pred_labels,
        score=scores,
        logits=logits,
        labels=np.array(labels),
        prompts=np.array(prompts),
        prompt_groups_json=np.asarray(json.dumps(prompt_groups)),
        prompt_ensemble_file=np.asarray(str(prompt_ensemble_file or "")),
        prompt_ensemble_sha256=np.asarray(str(prompt_ensemble_sha256 or "")),
        sample_index=sample_idx,
        center_shift=shift,
    )

    colors = palette(len(labels))[pred_index]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coord.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
    o3d.io.write_point_cloud(str(ply_path), pcd)

    unique, counts = np.unique(pred_labels, return_counts=True)
    summary = {
        "input": str(input_path),
        "num_points": int(coord.shape[0]),
        "labels": labels,
        "prompts": prompts,
        "prompt_groups": prompt_groups,
        "prompt_ensemble_file": str(prompt_ensemble_file or ""),
        "prompt_ensemble_sha256": str(prompt_ensemble_sha256 or ""),
        "counts": {str(label): int(count) for label, count in zip(unique, counts)},
        "npz": str(npz_path),
        "colorized_ply": str(ply_path),
    }
    json_path.write_text(json.dumps(summary, indent=2))
    return npz_path, ply_path, json_path, summary


def main():
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false.")

    labels = read_labels(args.labels, args.labels_file)
    coord, color = read_point_cloud(args.input)
    coord, color, sample_idx = sample_points(coord, color, args.max_points, args.seed)

    cfg = load_config(args)
    conditions = list(OmegaConf.select(cfg, "data.train_dataset.conditions", default=[]))
    if conditions and args.condition not in conditions:
        raise ValueError(f"--condition must be one of {conditions}; got {args.condition!r}.")

    model, missing, unexpected = load_model(cfg, args.ckpt, device)
    batch, shift = build_batch(coord, color, args.condition, args.grid_size, device)
    pred_index, scores, logits, prompts, prompt_groups, prompt_ensemble_sha256 = run_inference(
        model, cfg, batch, labels, args.prompt_template, args.prompt_ensemble_file, device
    )
    npz_path, ply_path, json_path, summary = save_outputs(
        args.output_dir,
        args.input,
        coord,
        pred_index,
        scores,
        logits,
        labels,
        prompts,
        prompt_groups,
        args.prompt_ensemble_file,
        prompt_ensemble_sha256,
        sample_idx,
        shift,
    )

    print(f"Saved predictions: {npz_path}")
    print(f"Saved colorized point cloud: {ply_path}")
    print(f"Saved summary: {json_path}")
    print(f"Counts: {summary['counts']}")
    if missing:
        print(f"Missing checkpoint keys loaded with strict=False: {len(missing)}")
    if unexpected:
        print(f"Unexpected checkpoint keys loaded with strict=False: {len(unexpected)}")


if __name__ == "__main__":
    main()
