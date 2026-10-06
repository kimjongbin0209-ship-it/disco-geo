from pathlib import Path
import time
import jax
import numpy as np
from disco_geo.config import TrainConfig
from disco_geo.data.bundle import RGBBundle
from disco_geo.io import load_checkpoint, sha256, write_json
from disco_geo.models.encoder import feature_transform


def extract(checkpoint, dataset, output, batch_size=128, max_scenes=None):
    start_time = time.monotonic()
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    saved = load_checkpoint(checkpoint)
    config = TrainConfig(**dict(saved["config"], channels=tuple(saved["config"]["channels"])))
    if sha256(Path(dataset) / "manifest.json") != saved["dataset_manifest_sha256"]:
        raise ValueError("Checkpoint and dataset manifests differ")
    bundle = RGBBundle(dataset, "test")
    images = bundle.images
    n = min(len(images), max_scenes) if max_scenes else len(images)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result = np.lib.format.open_memmap(
        output, mode="w+", dtype=np.float32, shape=(n, images.shape[1], config.feature_dim)
    )
    params = {k: v for k, v in saved["learner"].params.items() if k.startswith("visual_encoder/")}
    apply = jax.jit(feature_transform(config).apply)
    total = n * images.shape[1]
    for start in range(0, total, batch_size):
        rows = np.arange(start, min(start + batch_size, total))
        pixels = images[rows // images.shape[1], rows % images.shape[1]]
        result.reshape(-1, config.feature_dim)[rows] = np.asarray(apply(params, pixels))
    result.flush()
    write_json(
        output.with_suffix(".json"),
        {
            "checkpoint_sha256": sha256(checkpoint),
            "dataset_manifest_sha256": sha256(Path(dataset) / "manifest.json"),
            "method": config.method,
            "ablation": config.ablation,
            "seed": config.seed,
            "scene_ids": bundle.scene_ids[:n].tolist(),
            "frozen": True,
            "wall_seconds": time.monotonic() - start_time,
        },
    )
    return output
