"""Import the pinned official package without global `model` module collisions."""

import importlib
import importlib.util
from pathlib import Path
import sys
from disco_geo.upstream import ROOT


def official_module(name="triposplat"):
    namespace = "_discogeo_official_triposplat"
    directory = ROOT / "third_party/triposplat"
    if namespace not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            namespace, directory / "__init__.py", submodule_search_locations=[str(directory)]
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[namespace] = module
        spec.loader.exec_module(module)
    return importlib.import_module(f"{namespace}.{name}")


WEIGHTS = {
    "ckpt_path": "diffusion_models/triposplat_fp16.safetensors",
    "decoder_path": "vae/triposplat_vae_decoder_fp16.safetensors",
    "dinov3_path": "clip_vision/dino_v3_vit_h.safetensors",
    "flux2_vae_encoder_path": "vae/flux2-vae.safetensors",
    "rmbg_path": "background_removal/birefnet.safetensors",
}


def weight_paths(root, keys=None):
    result = {k: str(Path(root) / v) for k, v in WEIGHTS.items() if keys is None or k in keys}
    missing = [v for v in result.values() if not Path(v).is_file()]
    if missing:
        raise FileNotFoundError(
            "Download the official TripoSplat weights first. Missing: " + ", ".join(missing)
        )
    return result
