"""Frozen-representation ability suite over common-history features.

Every readout is fitted on probe-train, its ridge alpha chosen on the
layout-disjoint probe-validation split, and scored on probe-final-test.
Scores are reduced per test episode first; frames, cells and targets are
coverage counts, never independent samples. Navigation is evaluated separately
from the agent's own frozen-policy episodes (see ``navigation.py``).
"""

import json
from pathlib import Path

import numpy as np

from disco_spatial.evaluation.metrics import masked_map_iou, target_location_rmse
from disco_spatial.evaluation.probes import ProbeSplit, evaluate_ridge_probe, fit_ridge_probe
from disco_spatial.io import sha256


PARTITIONS = ("probe_train", "probe_validation", "probe_final_test")
REPRESENTATIONS = ("belief", "image_features", "action_history")
ACTION_TRACE_HORIZONS = (4, 16, 64, 256)
TIME_BINS = (0, 100, 300, 1000000)


def load_partition(directory, partition):
    """Load per-episode feature/label files after manifest and checksum checks."""
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    if not manifest.get("frozen") or manifest["partition"] != partition:
        raise ValueError("Feature directory has the wrong frozen split")
    episodes = []
    for entry in manifest["entries"]:
        path = directory / entry["file"]
        if sha256(path) != entry["sha256"]:
            raise ValueError("Feature checksum changed")
        with np.load(path, allow_pickle=False) as data:
            episode = {key: data[key] for key in data.files}
        episode["action_history"] = action_history_features(episode["previous_action"])
        episodes.append(episode)
    if not episodes:
        raise ValueError(f"No episodes in {partition}")
    return manifest["checkpoint_sha256"], episodes


def action_history_features(previous_action, horizons=ACTION_TRACE_HORIZONS, num_actions=6):
    """Dead-reckoning control: decayed action traces and normalized counts, no RGB.

    START (previous_action == num_actions) is the all-zero action.
    """
    actions = np.asarray(previous_action)
    one_hot = np.zeros((len(actions), num_actions), np.float32)
    valid = actions < num_actions
    one_hot[np.flatnonzero(valid), actions[valid]] = 1
    columns = [one_hot]
    for horizon in horizons:
        decay, trace = np.exp(-1 / horizon), np.zeros(num_actions, np.float32)
        traces = np.zeros_like(one_hot)
        for t, row in enumerate(one_hot):
            trace = decay * trace + row
            traces[t] = trace / horizon
        columns.append(traces)
    time = np.arange(len(actions), dtype=np.float32)[:, None]
    columns.append(np.cumsum(one_hot, 0) / np.maximum(time, 1))
    columns.append(time / 1000)
    return np.concatenate(columns, axis=1)


def _split(episodes, representation, target, select=None, stride=1, lag=None):
    features, targets, layouts, ids = [], [], [], []
    for episode in episodes:
        x = episode[representation]
        y = episode[f"target_{target}"].reshape(len(x), -1).astype(float)
        keep = np.arange(len(x)) % stride == 0
        if select is not None:
            keep &= select(episode)
        if lag is not None:
            keep &= np.arange(len(x)) >= lag
            x = np.concatenate((x, np.roll(x, lag, axis=0)), axis=1)
        if not keep.any():
            continue
        features.append(x[keep])
        targets.append(y[keep])
        layouts.extend([str(episode["layout_ids"][0])] * int(keep.sum()))
        ids.extend([str(episode["episode_ids"][0])] * int(keep.sum()))
    return ProbeSplit(np.concatenate(features), np.concatenate(targets), tuple(layouts), tuple(ids))


def _fit(partitions, representation, target, stride, select=None, lag=None):
    train, validation, test = (
        _split(partitions[name], representation, target, select, stride, lag)
        for name in PARTITIONS)
    model = fit_ridge_probe(train, validation)
    overall = evaluate_ridge_probe(model, test)
    overall.pop("prediction")
    return model, overall


def _episode_prediction(model, episode, representation, lag=None):
    x = episode[representation]
    if lag is not None:
        x = np.concatenate((x, np.roll(x, lag, axis=0)), axis=1)
    return model.predict(x)


def _record(episode, **values):
    return {"layout_id": str(episode["layout_ids"][0]),
            "episode_id": str(episode["episode_ids"][0]), **values}


def _mean(records, key):
    values = [row[key] for row in records if row.get(key) is not None]
    return float(np.mean(values)) if values else None


def spatial_memory(partitions, representation, stride):
    """Primary 1: egocentric target location by causal observability group."""
    model, overall = _fit(partitions, representation, "targets_vec", stride)
    mean = model.train_target_mean.reshape(-1, 2)
    records = []
    for episode in partitions["probe_final_test"]:
        truth = episode["target_targets_vec"]
        masks = {name: episode[f"target_targets_{name}"]
                 for name in ("visible", "seen_hidden", "never_seen")}
        prediction = _episode_prediction(model, episode, representation).reshape(truth.shape)
        scores = target_location_rmse(prediction, truth, masks)
        baseline = target_location_rmse(np.broadcast_to(mean, truth.shape), truth, masks)
        records.append(_record(
            episode, **{f"{group}_rmse": scores[group]["rmse"] for group in scores},
            **{f"{group}_n": scores[group]["n_target_observations"] for group in scores},
            **{f"{group}_train_mean_rmse": baseline[group]["rmse"] for group in scores}))
    return {
        "seen_hidden_target_location_rmse": _mean(records, "seen_hidden_rmse"),
        "visible_target_location_rmse": _mean(records, "visible_rmse"),
        "never_seen_target_location_rmse": _mean(records, "never_seen_rmse"),
        "seen_hidden_train_mean_rmse": _mean(records, "seen_hidden_train_mean_rmse"),
        "alpha": model.alpha, "overall": overall, "episodes": records,
        "unit": "maze cells, (right, forward) egocentric; Euclidean RMSE",
    }


def map_topology(partitions, representation, stride, frame_stride=10):
    """Primary 2: balanced wall/free IoU over causally known origin-canvas cells."""
    model, overall = _fit(partitions, representation, "origin_walls", stride)
    records = []
    for episode in partitions["probe_final_test"]:
        walls, valid, known = (episode["target_origin_walls"], episode["target_origin_map_valid"],
                               episode["target_origin_map_known"])
        prediction = np.clip(_episode_prediction(model, episode, representation), 0, 1)
        prediction = prediction.reshape(walls.shape)
        known_scores, unknown_scores = [], []
        for t in range(0, len(walls), frame_stride):
            score = masked_map_iou(prediction[t], walls[t], known[t])["balanced_iou"]
            if score is not None:
                known_scores.append(score)
            unseen = valid[t] & ~known[t]
            score = masked_map_iou(prediction[t], walls[t], unseen)["balanced_iou"]
            if score is not None:
                unknown_scores.append(score)
        records.append(_record(
            episode,
            known_balanced_iou=float(np.mean(known_scores)) if known_scores else None,
            unknown_balanced_iou=float(np.mean(unknown_scores)) if unknown_scores else None,
            final_known_fraction=float(known[-1][valid[-1]].mean())))
    return {
        "known_wall_balanced_iou": _mean(records, "known_balanced_iou"),
        "unknown_cell_balanced_iou": _mean(records, "unknown_balanced_iou"),
        "final_known_fraction": _mean(records, "final_known_fraction"),
        "alpha": model.alpha, "overall": overall, "episodes": records,
        "gauge": "start cell, start heading rounded to the nearest grid axis; outer ring is wall",
    }


def self_localization(partitions, representation, stride):
    model, overall = _fit(partitions, representation, "origin_relative_position", stride)
    heading, heading_overall = _fit(partitions, representation, "origin_relative_heading", stride)
    records = []
    for episode in partitions["probe_final_test"]:
        truth = episode["target_origin_relative_position"]
        error = np.linalg.norm(_episode_prediction(model, episode, representation) - truth, axis=-1)
        predicted = _episode_prediction(heading, episode, representation)
        true_heading = episode["target_origin_relative_heading"]
        angle = np.abs(np.angle(np.exp(1j * (np.arctan2(predicted[:, 0], predicted[:, 1])
                                              - np.arctan2(true_heading[:, 0], true_heading[:, 1])))))
        time = episode["target_time_index"]
        row = {"position_rmse": float(np.sqrt(np.mean(error ** 2))),
               "heading_error_degrees": float(np.degrees(angle).mean())}
        for low, high in zip(TIME_BINS[:-1], TIME_BINS[1:]):
            inside = (time >= low) & (time < high)
            row[f"position_rmse_t{low}"] = (float(np.sqrt(np.mean(error[inside] ** 2)))
                                            if inside.any() else None)
        records.append(_record(episode, **row))
    result = {
        "origin_position_rmse": _mean(records, "position_rmse"),
        "origin_heading_error_degrees": _mean(records, "heading_error_degrees"),
        "drift": {f"t{low}": _mean(records, f"position_rmse_t{low}") for low in TIME_BINS[:-1]},
        "alpha": model.alpha, "heading_alpha": heading.alpha, "overall": overall,
        "heading_overall": heading_overall, "episodes": records,
    }
    for lag in (1, 8):
        name = f"relative_motion_{lag}"
        motion, _ = _fit(partitions, representation, name, stride,
                         select=lambda e, n=name: e[f"target_{n}_valid"], lag=lag)
        translation, rotation, baseline = [], [], []
        for episode in partitions["probe_final_test"]:
            valid = episode[f"target_{name}_valid"]
            truth = episode[f"target_{name}"][valid]
            predicted = _episode_prediction(motion, episode, representation, lag=lag)[valid]
            translation.append(np.sqrt(np.mean(np.sum((predicted[:, :2] - truth[:, :2]) ** 2, -1))))
            rotation.append(np.sqrt(np.mean(np.sum((predicted[:, 2:] - truth[:, 2:]) ** 2, -1))))
            baseline.append(np.sqrt(np.mean(np.sum(
                (motion.train_target_mean[:2] - truth[:, :2]) ** 2, -1))))
        result[f"{name}_translation_rmse"] = float(np.mean(translation))
        result[f"{name}_heading_vector_rmse"] = float(np.mean(rotation))
        result[f"{name}_translation_train_mean_rmse"] = float(np.mean(baseline))
    return result


def loop_closure(partitions, representation, gap=50, radius=0.5, heading_degrees=45.0,
                 query_stride=5):
    """Place retrieval recall@1 with temporally separated revisits; no fitted readout."""
    train = np.concatenate([e[representation] for e in partitions["probe_train"]])
    mean, std = train.mean(0), np.maximum(train.std(0), 1e-6)
    records = []
    for episode in partitions["probe_final_test"]:
        x = (episode[representation] - mean) / std
        x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-9)
        position = episode["target_origin_relative_position"]
        angle = episode["target_agent_heading_angle"]
        hits, chance, queries = [], [], 0
        for t in range(gap, len(x), query_stride):
            candidates = np.arange(t - gap + 1)
            near = np.linalg.norm(position[candidates] - position[t], axis=1) < radius
            aligned = np.abs(np.angle(np.exp(1j * (angle[candidates] - angle[t])))) < np.deg2rad(heading_degrees)
            positive = near & aligned
            if not positive.any() or positive.all():
                continue
            queries += 1
            nearest = candidates[np.argmax(x[candidates] @ x[t])]
            hits.append(bool(positive[nearest - candidates[0]]))
            chance.append(float(positive.mean()))
        records.append(_record(episode, recall_at_1=float(np.mean(hits)) if hits else None,
                               chance_recall=float(np.mean(chance)) if chance else None,
                               n_queries=queries))
    return {"place_retrieval_recall_at_1": _mean(records, "recall_at_1"),
            "chance_recall_at_1": _mean(records, "chance_recall"),
            "gap_steps": gap, "radius_cells": radius, "heading_degrees": heading_degrees,
            "episodes": records,
            "method": "cosine nearest neighbor on probe-train-standardized features"}


def _average_precision(score, positive):
    order = np.argsort(-score, kind="stable")
    positive = np.asarray(positive, bool)[order]
    if not positive.any():
        return None
    precision = np.cumsum(positive) / np.arange(1, len(positive) + 1)
    return float(precision[positive].mean())


def local_geometry(partitions, representation, stride):
    depth, depth_overall = _fit(partitions, representation, "depth_coarse", stride)
    rays, ray_overall = _fit(partitions, representation, "free_space_rays", stride)
    records = []
    for episode in partitions["probe_final_test"]:
        truth = episode["target_free_space_rays"]
        predicted = _episode_prediction(rays, episode, representation)
        depth_error = _episode_prediction(depth, episode, representation) - episode[
            "target_depth_coarse"].reshape(len(truth), -1)
        records.append(_record(
            episode, depth_rmse=float(np.sqrt(np.mean(depth_error ** 2))),
            forward_free_rmse=float(np.sqrt(np.mean((predicted[:, 1] - truth[:, 1]) ** 2))),
            traversable_ap=_average_precision(predicted[:, 1], truth[:, 1] > 1.0),
            traversable_base_rate=float(np.mean(truth[:, 1] > 1.0))))
    return {"depth_rmse_cells": _mean(records, "depth_rmse"),
            "forward_free_space_rmse_cells": _mean(records, "forward_free_rmse"),
            "traversable_forward_average_precision": _mean(records, "traversable_ap"),
            "traversable_base_rate": _mean(records, "traversable_base_rate"),
            "depth_alpha": depth.alpha, "ray_alpha": rays.alpha,
            "depth_overall": depth_overall, "ray_overall": ray_overall, "episodes": records}


def run_suite(train_directory, validation_directory, test_directory, stride=5,
              representations=REPRESENTATIONS):
    """All frozen-representation assays; navigation lives in navigation.py."""
    checkpoints, partitions = [], {}
    for directory, partition in zip((train_directory, validation_directory, test_directory),
                                    PARTITIONS):
        checkpoint, partitions[partition] = load_partition(directory, partition)
        checkpoints.append(checkpoint)
    if len(set(checkpoints)) != 1:
        raise ValueError("All probe partitions must come from the same frozen checkpoint")
    first = partitions["probe_final_test"][0]
    if "target_targets_seen_hidden" not in first or "target_origin_map_known" not in first:
        raise ValueError("Collections lack render telemetry; recollect with schema_version 2")
    families = {"spatial_memory": {}, "map_topology": {}, "self_localization": {},
                "local_geometry": {}, "loop_closure": {}}
    for representation in representations:
        families["spatial_memory"][representation] = spatial_memory(partitions, representation, stride)
        families["map_topology"][representation] = map_topology(partitions, representation, stride)
        families["self_localization"][representation] = self_localization(
            partitions, representation, stride)
        if representation != "action_history":
            families["local_geometry"][representation] = local_geometry(
                partitions, representation, stride)
            families["loop_closure"][representation] = loop_closure(partitions, representation)
    primary = {
        "seen_hidden_target_location_rmse":
            families["spatial_memory"]["belief"]["seen_hidden_target_location_rmse"]
            if "belief" in representations else None,
        "known_wall_balanced_iou":
            families["map_topology"]["belief"]["known_wall_balanced_iou"]
            if "belief" in representations else None,
        "matched_revisit_efficiency_gain": None,
    }
    return {
        "schema_version": 1, "checkpoint_sha256": checkpoints[0], "stride": stride,
        "n_episodes": {name: len(episodes) for name, episodes in partitions.items()},
        "primary": primary, "families": families,
        "navigation": "Run the navigation command on frozen-policy episodes for primary 3.",
        "interpretation": "Diagnostic linear readouts on one checkpoint. Aggregate over training "
                          "seeds and layouts before any comparison claim.",
    }
