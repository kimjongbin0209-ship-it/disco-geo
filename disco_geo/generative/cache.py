"""Cache image-only teacher conditioning for an explicitly downstream experiment."""

from pathlib import Path
import time
import numpy as np
from PIL import Image
import torch
from disco_geo.data.bundle import RGBBundle
from disco_geo.generative.external import official_module, weight_paths
from disco_geo.io import sha256, write_json


def cache_conditioning(dataset, weights, output, max_items=128, seed=0, device="cuda"):
    start = time.monotonic()
    if max_items < 3:
        raise ValueError("At least three items needed for train/validation/test")
    root = Path(output)
    if (root / "manifest.json").exists():
        raise FileExistsError(root)
    bundle = RGBBundle(dataset)
    api = official_module()
    paths = weight_paths(weights, {"dinov3_path", "flux2_vae_encoder_path", "rmbg_path"})
    dino = api.load_dinov3(paths["dinov3_path"], device=device, dtype=torch.bfloat16)
    vae = api.load_vae_encoder(paths["flux2_vae_encoder_path"], device=device, dtype=torch.bfloat16)
    rmbg = api.load_rmbg(paths["rmbg_path"], device=device, dtype=torch.float16)
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    items = []
    # Different identity sets for downstream fitting, model selection and generation.
    split_names = ("probe_train", "probe_val", "test")
    budgets = (max(1, int(max_items * 0.7)), max(1, int(max_items * 0.15)))
    budgets = (*budgets, max_items - sum(budgets))
    for split, budget in zip(split_names, budgets):
        scenes = rng.permutation(bundle.manifest["splits"][split])[:budget]
        for scene in scenes:
            view = int(rng.integers(bundle.images.shape[1]))
            source = Image.fromarray(bundle.images[scene, view])
            prepared = api.preprocess_image(source, rmbg)
            generator = torch.Generator(device=device).manual_seed(seed + len(items))
            cond = api.encode_image(prepared, dino, vae, generator=generator)
            name = f"{len(items):06d}"
            np.savez_compressed(
                root / f"{name}.npz",
                **{k: v.float().cpu().numpy().astype(np.float16) for k, v in cond.items()},
            )
            source.save(root / f"{name}.png")
            prepared.save(root / f"{name}_prepared.png")
            items.append(
                {
                    "scene": int(scene),
                    "view": view,
                    "scene_id": bundle.scene_ids[scene],
                    "split": split,
                    "conditioning": f"{name}.npz",
                    "image": f"{name}.png",
                }
            )
            print(f"Cached {len(items)} image-condition pairs", flush=True)
    write_json(
        root / "manifest.json",
        {
            "dataset_manifest_sha256": sha256(bundle.root / "manifest.json"),
            "items": items,
            "seed": seed,
            "wall_seconds": time.monotonic() - start,
            "device": device,
            "teacher": "TripoSplat native DINOv3 + FLUX.2 image conditioning",
            "teacher_weight_sha256": {k: sha256(v) for k, v in paths.items()},
            "stage": "downstream_only",
            "geometry_labels_read": False,
        },
    )
    return root
