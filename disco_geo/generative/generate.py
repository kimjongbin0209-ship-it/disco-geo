"""Run paired native/learned/zero/shuffled conditioning through official flow sampling."""

import hashlib
import json
from pathlib import Path
import time
import numpy as np
from PIL import Image
import torch
from disco_geo.generative.adapter import ConditioningAdapter
from disco_geo.generative.external import official_module, weight_paths
from disco_geo.io import sha256, write_json


def generate(
    weights,
    output,
    image=None,
    features=None,
    adapter=None,
    scene=0,
    view=0,
    mode="native",
    seed=42,
    steps=20,
    num_gaussians=32768,
    device="cuda",
    exports=("ply", "npz"),
):
    start = time.monotonic()
    if not exports or set(exports) - {"ply", "splat", "npz"}:
        raise ValueError("Choose ply, splat and/or npz exports")
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(device)
    if mode not in {"native", "learned", "zero", "shuffled"}:
        raise ValueError(mode)
    if steps < 1 or not 32768 <= num_gaussians <= 262144 or num_gaussians % 32:
        raise ValueError("Use positive steps and 32768–262144 Gaussians in multiples of 32")
    root = Path(output)
    if (root / "generation.json").exists():
        raise FileExistsError(root)
    api = official_module()
    gen = torch.Generator(device=device).manual_seed(seed)
    if mode == "native":
        if image is None:
            raise ValueError("Native baseline requires --image")
        pipe = api.TripoSplatPipeline(**weight_paths(weights), device=device)
        # Separate encoder randomness from flow noise for paired native/adapter runs.
        prepared = pipe.preprocess_image(image)
        cond = pipe.encode_image(
            prepared, torch.Generator(device=device).manual_seed(seed + 100_000)
        )
        if device.startswith("cuda"):
            torch.cuda.synchronize(device)
        sample_start = time.monotonic()
        sampled = pipe.sample_latent(
            cond, steps=steps, guidance_scale=3.0, shift=3.0, generator=gen
        )
        if device.startswith("cuda"):
            torch.cuda.synchronize(device)
        sampling_seconds = time.monotonic() - sample_start
        decode_start = time.monotonic()
        # Upstream octree point jitter uses the global RNG, not `generator`.
        torch.manual_seed(seed + 200_000)
        gaussian = pipe.decode_latent(sampled["latent"], num_gaussians=num_gaussians)
        pixels = np.asarray(Image.open(image).convert("RGB"))
        metadata = {
            "input_image_sha256": sha256(image),
            "input_rgb_sha256": hashlib.sha256(pixels.tobytes()).hexdigest(),
        }
    else:
        if adapter is None or features is None:
            raise ValueError("Learned/zero/shuffled modes require --adapter and --features")
        saved = torch.load(adapter, map_location="cpu", weights_only=True)
        meta = json.loads(Path(features).with_suffix(".json").read_text())
        if (
            not saved["encoder_frozen"]
            or sha256(features) != saved["features_sha256"]
            or meta != saved["feature_metadata"]
        ):
            raise ValueError("Features do not match the frozen encoder used for adapter training")
        h = np.load(features, mmap_mode="r", allow_pickle=False)
        if not 0 <= scene < len(h) or not 0 <= view < h.shape[1]:
            raise ValueError("Scene or view out of bounds")
        selected = scene
        if mode == "shuffled":
            if len(h) < 2:
                raise ValueError("Shuffled control needs at least two scene identities")
            selected = (scene + 1 + int(np.random.default_rng(seed).integers(len(h) - 1))) % len(h)
        bridge = ConditioningAdapter(**saved["settings"]).to(device).eval()
        bridge.load_state_dict(saved["state_dict"])
        with torch.no_grad():
            cond = bridge(torch.tensor(h[selected, view].copy(), device=device).unsqueeze(0))
        if mode == "zero":
            cond = {k: torch.zeros_like(v) for k, v in cond.items()}
        paths = weight_paths(weights, {"ckpt_path", "decoder_path"})
        flow = api.load_flow_model(paths["ckpt_path"], device=device, dtype=torch.float16)
        if device.startswith("cuda"):
            torch.cuda.synchronize(device)
        sample_start = time.monotonic()
        sampled = api.sample_latent(
            flow, cond, steps=steps, guidance_scale=3.0, shift=3.0, generator=gen
        )
        if device.startswith("cuda"):
            torch.cuda.synchronize(device)
        sampling_seconds = time.monotonic() - sample_start
        decode_start = time.monotonic()
        del flow
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
        decoder = api.load_decoder(paths["decoder_path"], device=device, dtype=torch.float16)
        torch.manual_seed(seed + 200_000)
        with torch.no_grad():
            gaussian = decoder.decode(sampled["latent"], num_gaussians=num_gaussians)
        prepared = None
        metadata = {
            "adapter_sha256": sha256(adapter),
            "features_sha256": sha256(features),
            "method": meta["method"],
            "encoder_seed": meta["seed"],
            "ablation": meta["ablation"],
            "dataset_manifest_sha256": meta["dataset_manifest_sha256"],
            "scene": scene,
            "scene_id": meta["scene_ids"][scene],
            "view": view,
            "conditioning_scene": selected,
            "conditioning_scene_id": meta["scene_ids"][selected],
        }
    if device.startswith("cuda"):
        torch.cuda.synchronize(device)
    decoding_and_decoder_load_seconds = time.monotonic() - decode_start
    root.mkdir(parents=True, exist_ok=True)
    if "ply" in exports:
        gaussian.save_ply(root / "object.ply")
    if "splat" in exports:
        gaussian.save_splat(root / "object.splat")
    if "camera" in sampled:
        np.save(root / "sampled_camera.npy", sampled["camera"].detach().float().cpu().numpy())
    if "npz" in exports:
        np.savez_compressed(
            root / "gaussians.npz",
            xyz=gaussian.get_xyz.detach().float().cpu().numpy(),
            scale=gaussian.get_scaling.detach().float().cpu().numpy(),
            rotation=(gaussian._rotation + gaussian.rots_bias[None]).detach().float().cpu().numpy(),
            opacity=gaussian.get_opacity.detach().float().cpu().numpy(),
            color=np.clip(
                gaussian._features_dc.detach().float().cpu().numpy()[:, 0] * 0.28209479177387814
                + 0.5,
                0,
                1,
            ),
        )
    if prepared is not None:
        prepared.save(root / "prepared.png")
    write_json(
        root / "generation.json",
        {
            "mode": mode,
            "exports": list(exports),
            "device": device,
            "device_name": torch.cuda.get_device_name(device)
            if device.startswith("cuda")
            else "CPU",
            "sampling_seconds": sampling_seconds,
            "decoding_and_decoder_load_seconds": decoding_and_decoder_load_seconds,
            "wall_seconds": time.monotonic() - start,
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device)
            if device.startswith("cuda")
            else None,
            "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(device)
            if device.startswith("cuda")
            else None,
            "seed": seed,
            "steps": steps,
            "num_gaussians": num_gaussians,
            "prior": "TripoSplat",
            "flow_matching": True,
            "encoder_updated": False,
            "flow_noise_seed": seed,
            "decoder_seed": seed + 200_000,
            "guidance_scale": 3.0,
            "schedule_shift": 3.0,
            "gaussian_coordinates": "TripoSplat internal frame, before PLY export transform",
            "geometry_claim": "Requires held-out quantitative evaluation; output is not evidence by itself",
            **metadata,
        },
    )
    return root
