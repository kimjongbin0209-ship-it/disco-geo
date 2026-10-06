import json
from pathlib import Path
import time
import numpy as np
import torch
from disco_geo.generative.adapter import ConditioningAdapter, conditioning_loss
from disco_geo.io import sha256, write_json


def train_adapter(
    features, cache, output, epochs=30, learning_rate=1e-3, token_batch=128, seed=0, device="cpu"
):
    start = time.monotonic()
    if epochs <= 0 or token_batch <= 0:
        raise ValueError("epochs and token_batch must be positive")
    root = Path(cache)
    manifest = json.loads((root / "manifest.json").read_text())
    meta = json.loads(Path(features).with_suffix(".json").read_text())
    if (
        not meta.get("frozen")
        or meta["dataset_manifest_sha256"] != manifest["dataset_manifest_sha256"]
    ):
        raise ValueError("Adapter requires frozen features from the conditioning cache's dataset")
    h = np.load(features, mmap_mode="r", allow_pickle=False)
    for item in manifest["items"]:
        if meta["scene_ids"][item["scene"]] != item["scene_id"]:
            raise ValueError("Conditioning and feature scene identities disagree")
    train = [i for i in manifest["items"] if i["split"] == "probe_train"]
    val = [i for i in manifest["items"] if i["split"] == "probe_val"]
    if not train or not val:
        raise ValueError("Missing adapter train or validation items")
    output = Path(output)
    if (output / "best.pt").exists():
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    with np.load(root / train[0]["conditioning"], allow_pickle=False) as sample:
        settings = dict(
            feature_dim=h.shape[-1],
            token_count=sample["feature1"].shape[1],
            channels1=sample["feature1"].shape[2],
            channels2=sample["feature2"].shape[2],
        )
    torch.manual_seed(seed)
    model = ConditioningAdapter(**settings).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    rng = np.random.default_rng(seed)

    def item_loss(item, indices=None):
        feature = torch.tensor(h[item["scene"], item["view"]].copy(), device=device).unsqueeze(0)
        with np.load(root / item["conditioning"], allow_pickle=False) as data:
            teacher = {
                k: torch.tensor(data[k].astype(np.float32), device=device)
                for k in ("feature1", "feature2")
            }
        if indices is not None:
            teacher = {k: v[:, indices] for k, v in teacher.items()}
        return conditioning_loss(model(feature, indices), teacher)

    best = float("inf")
    with (output / "metrics.jsonl").open("w") as logs:
        for epoch in range(epochs):
            model.train()
            losses = []
            for i in rng.permutation(len(train)):
                indices = torch.tensor(
                    rng.choice(
                        settings["token_count"],
                        min(token_batch, settings["token_count"]),
                        replace=False,
                    ),
                    device=device,
                )
                optimizer.zero_grad(set_to_none=True)
                loss = item_loss(train[i], indices)
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach()))
            model.eval()
            with torch.no_grad():
                validation = float(np.mean([float(item_loss(item)) for item in val]))
            if not np.isfinite(validation):
                raise FloatingPointError("Nonfinite conditioning loss")
            record = dict(
                epoch=epoch + 1,
                train_mse=float(np.mean(losses)),
                validation_mse=validation,
                wall_seconds=time.monotonic() - start,
            )
            logs.write(json.dumps(record) + "\n")
            logs.flush()
            print(json.dumps(record), flush=True)
            if validation < best:
                best = validation
                torch.save(
                    {
                        "settings": model.settings,
                        "state_dict": model.state_dict(),
                        "features_sha256": sha256(features),
                        "feature_metadata": meta,
                        "cache_manifest_sha256": sha256(root / "manifest.json"),
                        "encoder_frozen": True,
                        "loss": "native conditioning MSE",
                    },
                    output / "best.pt",
                )
    write_json(
        output / "result.json",
        {
            "best_validation_mse": best,
            "encoder_frozen": True,
            "method": meta["method"],
            "seed": seed,
            "epochs": epochs,
            "device": device,
            "wall_seconds": time.monotonic() - start,
            "loss": "single MSE on pretrained image conditioning",
        },
    )
    return output / "best.pt"
