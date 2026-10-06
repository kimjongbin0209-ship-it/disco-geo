"""Held-out orbit metrics, with explicit coordinate-registration status."""

import hashlib
import json
import time
from pathlib import Path
import numpy as np
from PIL import Image
from disco_geo.data.bundle import RGBBundle, load_evaluation_labels
from disco_geo.generative.render import render
from disco_geo.io import sha256, write_json


def evaluate_generation(
    generation,
    dataset,
    scene,
    source_view,
    output,
    alignment=None,
    allow_unregistered=False,
    evaluation_split="test",
):
    start = time.monotonic()
    if evaluation_split not in {"test", "probe_val"}:
        raise ValueError("Use test or probe_val identities")
    bundle = RGBBundle(dataset, "test")
    if bundle.manifest["source"] != "analytic_voxel_objects":
        raise ValueError(
            "This evaluator requires the analytic benchmark's known camera conventions"
        )
    if scene not in bundle.manifest["splits"][evaluation_split]:
        raise ValueError(f"Generative evaluation must use {evaluation_split} identities")
    if not 0 <= source_view < bundle.images.shape[1]:
        raise ValueError("Source view out of bounds")
    metadata = json.loads((Path(generation) / "generation.json").read_text())
    if metadata["mode"] == "native":
        source_digest = hashlib.sha256(bundle.images[scene, source_view].tobytes()).hexdigest()
        if metadata.get("input_rgb_sha256") != source_digest:
            raise ValueError("Native generation did not use this scene's source RGB")
    elif (
        metadata.get("dataset_manifest_sha256") != sha256(bundle.root / "manifest.json")
        or metadata.get("scene_id") != bundle.scene_ids[scene]
        or metadata.get("view") != source_view
    ):
        raise ValueError("Generation and evaluation source identities disagree")
    if alignment is None and not allow_unregistered:
        raise ValueError(
            "Provide a verified alignment JSON, or explicitly request diagnostic --allow-unregistered"
        )
    transform = np.asarray(json.loads(Path(alignment).read_text())) if alignment else None
    with np.load(Path(generation) / "gaussians.npz", allow_pickle=False) as f:
        gaussians = dict(f)
    labels = load_evaluation_labels(dataset)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    metrics = []
    for view, yaw in enumerate(labels["yaw"][scene]):
        if view == source_view:
            continue
        predicted, alpha = render(gaussians, yaw, bundle.images.shape[2], transform=transform)
        truth = bundle.images[scene, view].astype(float) / 255
        mask = np.asarray(labels["mask"][scene, view])
        mse = float(np.mean((predicted - truth) ** 2))
        fg_mse = float(np.mean((predicted[mask] - truth[mask]) ** 2))
        pred_mask = alpha > 0.5
        union = np.logical_or(mask, pred_mask).sum()
        metrics.append(
            {
                "view": view,
                "psnr": float(-10 * np.log10(max(mse, 1e-12))),
                "foreground_psnr": float(-10 * np.log10(max(fg_mse, 1e-12))),
                "silhouette_iou": float(np.logical_and(mask, pred_mask).sum() / max(union, 1)),
            }
        )
        Image.fromarray((predicted * 255).astype(np.uint8)).save(output / f"view_{view:03d}.png")
    result = {
        "evaluation_split": evaluation_split,
        "wall_seconds": time.monotonic() - start,
        "scene": scene,
        "source_view_excluded": source_view,
        "coordinate_alignment_supplied": alignment is not None,
        "interpretation": "diagnostic unregistered"
        if alignment is None
        else "registered held-out orbit",
        "renderer": "CPU approximation; no exact gsplat parity claim",
        "views": metrics,
        **{
            f"mean_{key}": float(np.mean([m[key] for m in metrics]))
            for key in ("psnr", "foreground_psnr", "silhouette_iou")
        },
    }
    write_json(output / "metrics.json", result)
    return result
