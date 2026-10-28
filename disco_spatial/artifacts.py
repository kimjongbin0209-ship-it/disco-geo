"""Common-history collection and frozen readouts, separate from policy learning."""

import json
from pathlib import Path

import numpy as np

from disco_spatial.envs.memory_maze import make_memory_maze
from disco_spatial.io import write_json, sha256
from disco_spatial.trajectory import save_trajectory, load_trajectory


def layout_fingerprint(layout):
    """Conservative wall-geometry identity, equal under grid rotations/reflections."""
    from disco_spatial.evaluation.metrics import canonical_layout_fingerprint

    return canonical_layout_fingerprint(layout)


REFERENCE_POLICIES = {
    # Uniform actions mostly spin in place; "explore" is a fixed forward-biased,
    # sticky random walk that covers corridors. Neither is a learned policy.
    "uniform": {"actions": [0, 1, 2, 3, 4, 5], "probabilities": [1 / 6] * 6, "stickiness": 0.0},
    "explore": {"actions": [1, 2, 3, 4, 5], "probabilities": [0.5, 0.15, 0.15, 0.1, 0.1],
                "stickiness": 0.7},
}


def reference_actions(rng, policy, steps):
    spec = REFERENCE_POLICIES[policy]
    actions, current = [], int(rng.choice(spec["actions"], p=spec["probabilities"]))
    for _ in range(steps):
        if rng.random() >= spec["stickiness"]:
            current = int(rng.choice(spec["actions"], p=spec["probabilities"]))
        actions.append(current)
    return actions


def collect_common_history(output, seed=3000000, episodes=1, max_steps=1000,
                           maze_size=9, resolution=64, partition="probe_train", policy="explore"):
    namespaces = {"probe_train": (3000000, 4000000),
                  "probe_validation": (4000000, 5000000),
                  "probe_final_test": (5000000, 6000000)}
    if partition not in namespaces:
        raise ValueError("Choose a declared probe partition")
    if policy not in REFERENCE_POLICIES:
        raise ValueError(f"Choose a reference policy from {sorted(REFERENCE_POLICIES)}")
    lower, upper = namespaces[partition]
    if not lower <= seed < upper or seed + episodes > upper or episodes <= 0 or max_steps <= 0:
        raise ValueError("Invalid collection budget or seed namespace")
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Common-history output must be new")
    output.mkdir(parents=True, exist_ok=True)
    # Late import: protocol and dependency inspection need no JAX/MuJoCo load.
    from disco_spatial.training import EpisodeRecord

    rng = np.random.default_rng(seed)
    entries = []
    for index in range(episodes):
        split = "test" if partition == "probe_final_test" else "validation"
        env = make_memory_maze(maze_size, seed + index, split, resolution, evaluation=True,
                               telemetry=True)
        try:
            timestep = env.reset()
            record = EpisodeRecord(env, timestep, f"reference_{policy}")
            sidecars = [env.last_sidecar]
            for action in reference_actions(rng, policy, max_steps):
                timestep = env.step(action)
                record.append(action, np.zeros(6, np.float32), 0, timestep)
                sidecars.append(env.last_sidecar)
                if timestep.last():
                    break
            name = f"episode{index:05d}"
            trajectory = record.trajectory()
            trajectory_path = output / f"{name}.npz"
            labels_path = output / f"{name}.labels.npz"
            save_trajectory(trajectory, trajectory_path)
            keys = set(sidecars[0])
            if "maze_layout" not in keys or any(set(row) != keys for row in sidecars):
                raise ValueError("Evaluation sidecar keys are missing or unstable")
            labels = {key: np.stack([row[key] for row in sidecars]) for key in sorted(keys)}
            fingerprint = layout_fingerprint(labels["maze_layout"][0])
            if any(layout_fingerprint(grid) != fingerprint for grid in labels["maze_layout"]):
                raise ValueError("A layout unexpectedly changed within an episode")
            np.savez_compressed(labels_path, **labels)
            floor = int(labels["floor_pixels"].sum())
            entries.append({
                "trajectory": trajectory_path.name, "sidecar": labels_path.name,
                "trajectory_sha256": sha256(trajectory_path), "sidecar_sha256": sha256(labels_path),
                "episode_id": trajectory.episode_id, "layout_id": fingerprint,
                "steps": trajectory.n_steps, "complete_episode": trajectory.complete,
                "environment": env.metadata,
                "oracle_floor_pixels_in_walls_fraction":
                    float(labels["floor_pixels_in_walls"].sum() / floor) if floor else None,
            })
        finally:
            env.close()
    manifest = {
        "schema_version": 2, "partition": partition, "seed": seed,
        "policy": f"reference_{policy}_for_evaluation_only",
        "reference_policy": REFERENCE_POLICIES[policy],
        "learner_training_data": False, "entries": entries,
        "layout_identity": "D4-canonical wall occupancy, conservative geometry split",
        "visibility_telemetry_available": True,
        "telemetry": "Re-rendered actor camera: segmentation target pixels, depth back-projected "
                     "observed cells, coarse depth. Evaluation sidecar only.",
    }
    write_json(output / "manifest.json", manifest)
    return manifest


def extract_common_features(checkpoint, collection, output, chunk_size=128):
    """Run only raw causal history through a frozen model; join labels afterwards."""
    import jax
    from disco_spatial.agents import RecurrentAgent
    from disco_spatial.config import TrainConfig
    from disco_spatial.io import load_checkpoint
    from disco_spatial.training import trajectory_observations
    from disco_spatial.evaluation.labels import episode_labels
    from dm_env import specs

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    collection = Path(collection)
    manifest = json.loads((collection / "manifest.json").read_text())
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Feature output must be new")
    output.mkdir(parents=True, exist_ok=True)
    saved = load_checkpoint(checkpoint)
    config = TrainConfig(**dict(saved["config"], channels=tuple(saved["config"]["channels"]),
                               validation_seeds=tuple(saved["config"]["validation_seeds"])))

    class Specification:
        def single_observation_spec(self):
            return {"rgb": specs.Array((config.camera_resolution, config.camera_resolution, 3), np.uint8),
                    "previous_action": specs.Array((), np.int32),
                    "previous_reward": specs.Array((), np.float32)}

        def single_action_spec(self):
            return specs.BoundedArray((), np.int32, 0, 5)

    agent = RecurrentAgent(config, Specification())
    unroll = jax.jit(agent.policy_network.unroll)
    entries = []
    for index, item in enumerate(manifest["entries"]):
        trajectory_path, sidecar_path = collection / item["trajectory"], collection / item["sidecar"]
        if (sha256(trajectory_path) != item["trajectory_sha256"]
                or sha256(sidecar_path) != item["sidecar_sha256"]):
            raise ValueError("Common-history source checksum changed")
        trajectory = load_trajectory(trajectory_path)
        if trajectory.rgb.shape[1:3] != (config.camera_resolution,) * 2:
            raise ValueError("Collection/checkpoint RGB resolution differs")
        observations = trajectory_observations(trajectory)
        state = agent.initial_actor_state(jax.random.PRNGKey(0), batch_size=1)
        features, beliefs = [], []
        for start in range(0, trajectory.n_steps + 1, chunk_size):
            part = jax.tree.map(lambda x: x[start:start + chunk_size], observations)
            out, state = unroll(saved["learner"].params, state, part, part["is_first"])
            features.append(np.asarray(out["image_features"][:, 0]))
            beliefs.append(np.asarray(out["belief"][:, 0]))
        n = trajectory.n_steps + 1
        data = {"image_features": np.concatenate(features), "belief": np.concatenate(beliefs),
                "layout_ids": np.full(n, item["layout_id"]),
                "episode_ids": np.full(n, trajectory.episode_id)}
        # No label array is passed to the model above. These are readout-only artifacts.
        with np.load(sidecar_path, allow_pickle=False) as stored:
            sidecar = {key: stored[key] for key in stored.files}
        if layout_fingerprint(sidecar["maze_layout"][0]) != item["layout_id"]:
            raise ValueError("Wrong layout fingerprint")
        for key, value in episode_labels(sidecar).items():
            data[f"target_{key}"] = value
        data["previous_action"] = np.asarray(observations["previous_action"][:, 0])
        name = f"episode{index:05d}.features.npz"
        np.savez_compressed(output / name, **data)
        entries.append({"file": name, "sha256": sha256(output / name),
                        "layout_id": item["layout_id"], "episode_id": trajectory.episode_id})
    result = {"schema_version": 1, "frozen": True, "partition": manifest["partition"],
              "checkpoint_sha256": sha256(checkpoint),
              "collection_manifest_sha256": sha256(collection / "manifest.json"),
              "entries": entries, "student_feedback_used": False,
              "collection_schema_version": manifest.get("schema_version", 1),
              "primary_visibility_metrics_ready": bool(manifest.get("visibility_telemetry_available")),
              "interpretation": "Frozen features joined to evaluation labels after the forward pass."}
    write_json(output / "manifest.json", result)
    return result


def probe_features(train_directory, validation_directory, test_directory, output,
                   representation="belief", target="origin_relative_position", stride=1):
    """A generic supervised readout, deliberately not an automatic emergence claim."""
    from disco_spatial.evaluation.probes import ProbeSplit, fit_ridge_probe, evaluate_ridge_probe

    if stride < 1 or representation not in {"belief", "image_features"}:
        raise ValueError("Invalid feature selection")
    expected = ["probe_train", "probe_validation", "probe_final_test"]
    checkpoints = []

    def load(directory, partition):
        directory = Path(directory)
        manifest = json.loads((directory / "manifest.json").read_text())
        if not manifest.get("frozen") or manifest["partition"] != partition:
            raise ValueError("Probe directory has the wrong frozen split")
        checkpoints.append(manifest["checkpoint_sha256"])
        x, y, layouts, episodes = [], [], [], []
        for entry in manifest["entries"]:
            path = directory / entry["file"]
            if sha256(path) != entry["sha256"]:
                raise ValueError("Probe feature checksum changed")
            with np.load(path, allow_pickle=False) as data:
                x.append(data[representation][::stride])
                y.append(data[f"target_{target}"][::stride])
                layouts.extend(data["layout_ids"][::stride].tolist())
                episodes.extend(data["episode_ids"][::stride].tolist())
        return ProbeSplit(np.concatenate(x), np.concatenate(y), tuple(layouts), tuple(episodes))

    train, validation, test = [load(path, partition) for path, partition in zip(
        (train_directory, validation_directory, test_directory), expected)]
    if len(set(checkpoints)) != 1:
        raise ValueError("Probe splits must use the same frozen checkpoint")
    model = fit_ridge_probe(train, validation)
    metrics = evaluate_ridge_probe(model, test)
    prediction = metrics.pop("prediction")
    output = Path(output)
    if output.exists() or output.with_suffix(".predictions.npz").exists():
        raise FileExistsError("Use a new probe output")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output.with_suffix(".predictions.npz"), prediction=prediction,
                        target=test.targets, layout_ids=np.asarray(test.layout_ids),
                        episode_ids=np.asarray(test.episode_ids))
    metrics.update(representation=representation, target=target, checkpoint_sha256=checkpoints[0],
                   primary_endpoint=False, confidence_interval=None,
                   warning="Per-coordinate generic readout; use seed/layout aggregation and observability for research claims.")
    write_json(output, metrics)
    return metrics
