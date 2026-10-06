"""Small held-out probes. No probe gradient can reach the RL encoder."""

import json
from pathlib import Path
import time
import numpy as np
from disco_geo.data.bundle import RGBBundle, load_evaluation_labels
from disco_geo.io import sha256, write_json


def ridge_fit(x, y, alpha=1):
    mean = x.mean(0)
    std = np.maximum(x.std(0), 1e-6)
    z = np.c_[(x - mean) / std, np.ones(len(x))]
    penalty = np.eye(z.shape[1]) * alpha
    penalty[-1, -1] = 0
    weight = np.linalg.solve(z.T @ z + penalty, z.T @ y)
    return mean, std, weight


def ridge_predict(model, x):
    mean, std, weight = model
    return np.c_[(x - mean) / std, np.ones(len(x))] @ weight


def angular_error(prediction, yaw):
    angle = np.arctan2(prediction[..., 0], prediction[..., 1])
    delta = np.arctan2(np.sin(angle - yaw), np.cos(angle - yaw))
    return np.rad2deg(np.abs(delta))


def run_probes(dataset, features, output, label_budget=5000, seed=0, evaluation_split="test"):
    start_time = time.monotonic()
    if label_budget <= 0:
        raise ValueError("label_budget must be positive")
    if evaluation_split not in {"test", "probe_val"}:
        raise ValueError("Use test for final results or probe_val for iteration monitoring")
    bundle = RGBBundle(dataset, "test")
    feature_meta = json.loads(Path(features).with_suffix(".json").read_text())
    if not feature_meta.get("frozen") or feature_meta["scene_ids"] != bundle.scene_ids.tolist():
        raise ValueError("Probing needs frozen features for every scene in manifest order")
    if feature_meta["dataset_manifest_sha256"] != sha256(bundle.root / "manifest.json"):
        raise ValueError("Feature dataset mismatch")
    h = np.load(features, allow_pickle=False)
    labels = load_evaluation_labels(dataset)
    yaw = np.asarray(labels["yaw"])
    if h.shape[:2] != yaw.shape:
        raise ValueError("Feature and camera dimensions disagree")
    rng = np.random.default_rng(seed)
    split = bundle.manifest["splits"]
    train_indices = np.array(split["probe_train"])
    order = rng.permutation(len(train_indices) * h.shape[1])[:label_budget]
    x_train = h[train_indices].reshape(-1, h.shape[-1])[order]
    theta_train = yaw[train_indices].reshape(-1)[order]
    y_train = np.c_[np.sin(theta_train), np.cos(theta_train)]
    val = np.array(split["probe_val"])
    test = np.array(split[evaluation_split])
    x_val = h[val].reshape(-1, h.shape[-1])
    x_test = h[test].reshape(-1, h.shape[-1])
    candidates = []
    for alpha in (1e-4, 1e-2, 1.0, 100.0, 10000.0):
        model = ridge_fit(x_train, y_train, alpha)
        error = angular_error(ridge_predict(model, x_val), yaw[val].reshape(-1)).mean()
        candidates.append((error, alpha, model))
    _, alpha, model = min(candidates, key=lambda item: item[0])
    errors = angular_error(ridge_predict(model, x_test), yaw[test].reshape(-1))
    metrics = {
        "evaluation_split": evaluation_split,
        "evaluation_scene_ids": bundle.scene_ids[test].tolist(),
        "method": feature_meta["method"],
        "seed": feature_meta["seed"],
        "ablation": feature_meta["ablation"],
        "label_budget_requested": label_budget,
        "label_budget_used": len(order),
        "pose_ridge_alpha": alpha,
        "pose_mean_degrees": float(errors.mean()),
        "pose_median_degrees": float(np.median(errors)),
        "pose_per_scene_mean_degrees": errors.reshape(len(test), -1).mean(1).tolist(),
    }
    from disco_geo.evaluation.relative import relative_probe

    metrics.update(
        relative_probe(h, yaw, dict(split, test=split[evaluation_split]), label_budget, seed)
    )
    predictions = {
        "pose_predicted_radians": np.arctan2(
            ridge_predict(model, x_test)[:, 0], ridge_predict(model, x_test)[:, 1]
        ),
        "pose_target_radians": yaw[test].reshape(-1),
        "scene_ids": bundle.scene_ids[test],
    }
    # Depth is a strictly post-RL dense linear probe from the global feature.
    if "depth" in labels:
        d = np.asarray(labels["depth"])[..., ::4, ::4].astype(np.float64)
        shape = d.shape[-2:]
        train_depth = d[train_indices].reshape(-1, np.prod(shape))[order]
        val_depth = d[val].reshape(-1, np.prod(shape))
        choices = []
        for a in (1.0, 100.0, 10000.0):
            m = ridge_fit(x_train, train_depth, a)
            prediction = ridge_predict(m, x_val)
            mask = val_depth > 0
            choices.append((np.mean((prediction[mask] - val_depth[mask]) ** 2), a, m))
        _, a, m = min(choices, key=lambda item: item[0])
        target = d[test].reshape(-1, np.prod(shape))
        pred = np.maximum(ridge_predict(m, x_test), 1e-6)
        mask = target > 0
        truth, prediction = target[mask], pred[mask]
        metrics.update(
            depth_ridge_alpha=a,
            depth_rmse=float(np.sqrt(np.mean((truth - prediction) ** 2))),
            depth_abs_rel=float(np.mean(np.abs(truth - prediction) / truth)),
            depth_delta_125=float(
                np.mean(np.maximum(truth / prediction, prediction / truth) < 1.25)
            ),
            depth_probe_resolution=list(shape),
            depth_scale_alignment=False,
        )
        predictions.update(
            depth_prediction=pred.reshape(-1, *shape), depth_target=target.reshape(-1, *shape)
        )
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(Path(output).with_suffix(".predictions.npz"), **predictions)
    metrics["wall_seconds"] = time.monotonic() - start_time
    write_json(output, metrics)
    return metrics
