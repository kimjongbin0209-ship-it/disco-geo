"""Small frozen-encoder snapshots for post-training probes and 3D monitoring."""

import json
from pathlib import Path
from types import SimpleNamespace
import jax
from disco_geo.io import load_checkpoint, save_checkpoint, write_json


def save_encoder_snapshot(output, config, params, dataset_hash, updates, transitions, policy=None):
    directory = Path(output) / "snapshots"
    root = directory / f"iter{updates:07d}"
    path = root / "encoder.pkl"
    metadata = dict(
        updates=updates,
        transitions=transitions,
        method=config.method,
        ablation=config.ablation,
        seed=config.seed,
        dataset_manifest_sha256=dataset_hash,
    )
    encoder = {k: v for k, v in params.items() if k.startswith("visual_encoder/")}
    if path.exists():
        previous = load_checkpoint(path)
        if any(previous.get(k) != v for k, v in metadata.items()):
            raise ValueError("Existing snapshot has different provenance")
    else:
        save_checkpoint(
            path,
            jax.device_get(
                {
                    "config": config.to_dict(),
                    "learner": SimpleNamespace(params=encoder),
                    "checkpoint_kind": "frozen_encoder_snapshot",
                    **metadata,
                }
            ),
        )
    if not (root / "iteration.json").exists():
        write_json(root / "iteration.json", {**metadata, "rl_validation": policy})
    manifest_path = directory / "manifest.json"
    items = json.loads(manifest_path.read_text())["items"] if manifest_path.exists() else []
    if not any(i["updates"] == updates for i in items):
        items.append({**metadata, "checkpoint": str(path.relative_to(directory))})
    write_json(
        manifest_path,
        {"iteration_unit": "learner update", "items": sorted(items, key=lambda x: x["updates"])},
    )
    return path
