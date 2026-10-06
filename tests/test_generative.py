import numpy as np
import pytest
import torch
from disco_geo.generative.adapter import ConditioningAdapter, conditioning_loss
from disco_geo.generative.external import official_module
from disco_geo.generative.render import render
from disco_geo.generative.train_adapter import train_adapter
from disco_geo.generative.evaluate import evaluate_generation
from disco_geo.data.bundle import RGBBundle
from disco_geo.io import sha256, write_json


def test_conditioning_adapter_never_backpropagates_into_encoder():
    torch.manual_seed(0)
    model = ConditioningAdapter(
        feature_dim=8, token_count=6, channels1=4, channels2=2, rank=2, hidden_dim=16
    )
    feature = torch.randn(3, 8, requires_grad=True)
    teacher = {
        "feature1": torch.randn(3, 6, 4, requires_grad=True),
        "feature2": torch.randn(3, 6, 2, requires_grad=True),
    }
    loss = conditioning_loss(model(feature), teacher)
    loss.backward()
    assert feature.grad is None and teacher["feature1"].grad is None
    assert model.tokens.grad is not None and torch.isfinite(model.tokens.grad).all()
    assert model(feature, torch.tensor([1, 4]))["feature1"].shape == (3, 2, 4)


def test_adapter_conditioning_is_accepted_by_actual_official_flow_model():
    # The actual upstream DiT at reduced width, rather than a mocked API.
    official = official_module("model")
    torch.manual_seed(0)
    flow = official.LatentSeqMMFlowModel(
        q_token_length=8,
        in_channels=4,
        model_channels=32,
        cond_channels=6,
        cond2_channels=2,
        out_channels=4,
        cam_channels=5,
        num_blocks=1,
        num_refiner_blocks=2,
        num_heads=2,
        num_head_channels=16,
    )
    bridge = ConditioningAdapter(
        feature_dim=8, token_count=9, channels1=6, channels2=2, rank=2, hidden_dim=16
    )
    cond = bridge(torch.randn(1, 8))
    out = flow(
        {"latent": torch.randn(1, 8, 4), "camera": torch.randn(1, 1, 5)},
        torch.tensor([500.0]),
        cond,
    )
    assert out["latent"].shape == (1, 8, 4)
    assert out["camera"].shape == (1, 1, 5)
    assert torch.isfinite(out["latent"]).all()


def test_cpu_gaussian_rendering_returns_visible_foreground():
    g = {
        "xyz": np.array([[0.0, 0.0, 0.0]]),
        "scale": np.array([[0.15, 0.15, 0.15]]),
        "rotation": np.array([[1.0, 0.0, 0.0, 0.0]]),
        "opacity": np.array([[0.9]]),
        "color": np.array([[1.0, 0.0, 0.0]]),
    }
    image, alpha = render(g, 0, size=32, elevation=0)
    assert image.shape == (32, 32, 3)
    assert alpha.max() > 0.5
    assert image[16, 16, 0] > image[16, 16, 1]


def test_actual_flow_sampling_and_gaussian_decoding_exports(tmp_path):
    api = official_module()
    model_api = official_module("model")
    flow = model_api.LatentSeqMMFlowModel(
        q_token_length=8,
        in_channels=4,
        model_channels=32,
        cond_channels=6,
        cond2_channels=2,
        out_channels=4,
        cam_channels=5,
        num_blocks=1,
        num_refiner_blocks=2,
        num_heads=2,
        num_head_channels=16,
    )
    cond = {"feature1": torch.zeros(1, 9, 6), "feature2": torch.zeros(1, 9, 2)}
    latent = api.sample_latent(flow, cond, steps=2, generator=torch.Generator().manual_seed(7))
    octree = dict(
        api.OCTREE_DECODER_ARGS, model_channels=32, cond_channels=4, num_blocks=1, num_heads=2
    )
    splats = dict(
        api.GS_DECODER_ARGS, model_channels=32, cond_channels=4, num_blocks=1, num_heads=2
    )
    decoder = model_api.OctreeGaussianDecoder(octree, splats)
    torch.manual_seed(7)
    gaussian = decoder.decode(latent["latent"], num_gaussians=32)
    assert gaussian.get_xyz.shape == (32, 3) and torch.isfinite(gaussian.get_xyz).all()
    gaussian.save_ply(tmp_path / "tiny.ply")
    gaussian.save_splat(tmp_path / "tiny.splat")
    assert (tmp_path / "tiny.ply").read_bytes().startswith(b"ply\n")
    assert (tmp_path / "tiny.splat").stat().st_size == 32 * 32
    torch.manual_seed(7)
    replay = decoder.decode(latent["latent"], num_gaussians=32)
    torch.testing.assert_close(gaussian.get_xyz, replay.get_xyz)


def test_adapter_fit_uses_scene_disjoint_cache_and_keeps_features_fixed(dataset, tmp_path):
    bundle = RGBBundle(dataset)
    features = tmp_path / "features.npy"
    np.save(features, np.random.default_rng(0).normal(size=(20, 6, 8)).astype(np.float32))
    write_json(
        features.with_suffix(".json"),
        {
            "frozen": True,
            "dataset_manifest_sha256": sha256(bundle.root / "manifest.json"),
            "scene_ids": bundle.scene_ids.tolist(),
            "method": "untrained",
            "seed": 0,
            "ablation": "none",
        },
    )
    digest = sha256(features)
    cache = tmp_path / "cache"
    cache.mkdir()
    items = []
    # Small synthetic native-format tensors test the fitting path, not pretrained quality.
    for index, split in enumerate(("probe_train", "probe_val", "test")):
        scene = bundle.manifest["splits"][split][0]
        name = f"{index}.npz"
        np.savez(
            cache / name,
            feature1=np.zeros((1, 9, 6), np.float16),
            feature2=np.zeros((1, 9, 2), np.float16),
        )
        items.append(
            {
                "scene": scene,
                "scene_id": str(bundle.scene_ids[scene]),
                "view": 0,
                "split": split,
                "conditioning": name,
            }
        )
    write_json(
        cache / "manifest.json",
        {
            "dataset_manifest_sha256": sha256(bundle.root / "manifest.json"),
            "items": items,
        },
    )
    result = train_adapter(features, cache, tmp_path / "fit", epochs=2, token_batch=4)
    saved = torch.load(result, weights_only=True)
    assert saved["encoder_frozen"] and sha256(features) == digest
    assert saved["settings"]["token_count"] == 9
    model = ConditioningAdapter(**saved["settings"])
    model.load_state_dict(saved["state_dict"])
    assert torch.isfinite(model(torch.zeros(1, 8))["feature1"]).all()


def test_generation_evaluation_rejects_wrong_source_and_affine_alignment(dataset, tmp_path):
    bundle = RGBBundle(dataset)
    scene = bundle.manifest["splits"]["test"][0]
    generated = tmp_path / "generated"
    generated.mkdir()
    metadata = {
        "mode": "learned",
        "dataset_manifest_sha256": sha256(bundle.root / "manifest.json"),
        "scene_id": str(bundle.scene_ids[scene]),
        "view": 0,
    }
    write_json(generated / "generation.json", metadata)
    np.savez(
        generated / "gaussians.npz",
        xyz=np.array([[0.0, 0.0, 0.0]]),
        scale=np.full((1, 3), 0.2),
        rotation=np.array([[1.0, 0.0, 0.0, 0.0]]),
        opacity=np.array([[0.9]]),
        color=np.array([[0.6, 0.6, 0.6]]),
    )
    with pytest.raises(ValueError, match="source identities"):
        evaluate_generation(
            generated, dataset, scene, 1, tmp_path / "wrong", allow_unregistered=True
        )
    result = evaluate_generation(
        generated, dataset, scene, 0, tmp_path / "eval", allow_unregistered=True
    )
    assert len(result["views"]) == 5 and all(v["view"] != 0 for v in result["views"])
    assert np.isfinite(result["mean_psnr"])
    alignment = tmp_path / "bad.json"
    write_json(alignment, np.diag([1.0, 2.0, 1.0, 1.0]).tolist())
    with pytest.raises(ValueError, match="uniform positive scale"):
        evaluate_generation(generated, dataset, scene, 0, tmp_path / "bad", alignment=alignment)
