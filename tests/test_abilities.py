"""Ability labels, telemetry geometry, navigation segments and the readout suite."""

import json
from pathlib import Path

import numpy as np
import pytest

from disco_spatial.evaluation import labels as L
from disco_spatial.evaluation import telemetry as T
from disco_spatial.evaluation.navigation import goal_segments
from disco_spatial.evaluation.metrics import revisit_efficiency
from disco_spatial.evaluation.suite import action_history_features, run_suite
from disco_spatial.io import sha256


RING = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], np.uint8)  # 1 = free, center wall


def test_palette_matches_upstream_memory_maze_source():
    import importlib.util
    import re

    spec = importlib.util.find_spec("memory_maze")
    if spec is None:
        pytest.skip("memory-maze is not installed")
    # Read the source rather than importing it: import initializes a renderer.
    source = (Path(spec.origin).parent / "maze.py").read_text()
    start = source.index("TARGET_COLORS = [")
    block = source[start:source.index("\n]", start)]
    upstream = [[int(v) for v in row.split(",")] for row in re.findall(r"np\.array\(\[([^\]]+)\]\)", block)]
    np.testing.assert_allclose(L.TARGET_PALETTE, np.asarray(upstream) / 220)


def test_origin_canvas_rotates_with_rounded_start_heading_and_keeps_ring_walls():
    known = np.zeros((2, 5, 5), bool)
    known[1, 1, 3] = True  # ring cell (x=2, y=0) observed at t=1 only
    walls, valid, mask = L.wall_canvas(RING, (0.5, 0.5), (0.1, 0.9), known)
    span = 4
    assert walls[span, span] == 0 and walls[span + 1, span + 1] == 1  # start free, wall ahead-right
    assert walls[span, span - 1] == 1  # ring cell to the left is wall
    assert walls[span, span + 2] == 0  # (2, 0) is free, two cells to the right
    assert not mask[0].any() and mask[1, span, span + 2]  # causal accumulation only
    assert valid.sum() == 25
    west, _, _ = L.wall_canvas(RING, (0.5, 0.5), (-1.0, 0.05), known)
    assert west[span - 1, span + 1] == 1  # facing -x, the wall at (1,1) is behind-right


def test_rays_distances_and_relative_frames():
    assert L.ray_distance(L.ring_walls(RING), (0.5, 0.5), (1, 0)) == pytest.approx(2.5, abs=0.03)
    assert L.ray_distance(L.ring_walls(RING), (0.5, 1.5), (1, 0)) == pytest.approx(0.5, abs=0.03)
    assert L.grid_distance(RING, (0, 0), (2, 2)) == 4
    assert L.grid_distance(RING, (0, 0), (0, 0)) == 0
    assert L.grid_distance(np.array([[1, 0, 1]]), (0, 0), (2, 0)) is None
    np.testing.assert_allclose(L.egocentric(np.array([[1.0, 0.0]]), np.array([[0.0, 1.0]])),
                               [[1.0, 0.0]])  # facing +y, +x is to the right


def test_camera_backprojection_and_observed_cells():
    rays = T.camera_rays(4, 4, 90.0)
    np.testing.assert_allclose(rays[..., 2], -1)
    matrix = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float)  # looking along +y
    points = T.backproject(np.full((4, 4), 2.0), np.zeros(3), matrix, 90.0)
    np.testing.assert_allclose(points[..., 1], 2.0)
    wall = np.array([[[3.999, 1.0, 0.5]]])  # just short of the boundary x = 4 (cell 2 at scale 2)
    cells = T.observed_ring_cells(wall, np.ones((1, 1), bool), np.array([0.0, 1.0, 0.5]), 2.0,
                                  np.zeros(2), 3, nudge=0.02)
    assert cells[1, 3]  # ring index [y + 1, x + 1] for cell (2, 0)
    image = np.arange(64, dtype=float).reshape(8, 8)
    depth = T.coarse_depth(image, np.ones((8, 8), bool), grid=2)
    expected = np.minimum(image[:4, :4], T.MAX_DEPTH_CELLS).mean()  # clipped per pixel
    assert depth.shape == (2, 2) and depth[0, 0] == pytest.approx(expected)
    assert T.border_width(64) == 2 and T.border_width(32) == 0


def episode_sidecar():
    positions = np.array([[0.5, .5], [1.0, .5], [0.5, .5], [1.5, .5], [2.5, .5],
                          [2.5, 1.5], [2.5, 2.5], [2.5, 1.5], [2.5, .5]])
    requested = [0, 0, 0, 0, 1, 1, 0, 0, 1]
    targets = np.array([[2.5, 0.5], [2.5, 2.5]])
    n = len(positions)
    direction = np.tile([1.0, 0.0], (n, 1))
    return {
        "agent_pos": positions, "agent_dir": direction,
        "maze_layout": np.broadcast_to(RING, (n, 3, 3)),
        "targets_pos": np.broadcast_to(targets, (n, 2, 2)),
        "targets_vec": L.egocentric(targets[None] - positions[:, None], direction[:, None]),
        "target_color": L.TARGET_PALETTE[requested],
        "target_visible_pixels": np.zeros((n, 2), np.int32),
    }


def test_goal_segments_match_first_visit_and_revisit_in_the_same_stratum():
    episode = {"sidecar": episode_sidecar(), "episode_id": "test:e0",
               "rewards": np.array([0, 0, 0, 1, 0, 1, 0, 1], np.float32)}
    rows, excluded = goal_segments(episode, seed=2000000, layout_id="layout")
    assert excluded == 0 and [row["kind"] for row in rows] == ["first_visit", "first_visit", "revisit"]
    assert [row["oracle_distance"] for row in rows] == [2, 2, 2]
    assert rows[0]["path_length"] == pytest.approx(3.0)
    result = revisit_efficiency(rows)
    assert result["n_matched_pairs"] == 1
    assert result["mean_efficiency_gain"] == pytest.approx(1 - 2 / 3)


def test_action_history_control_contains_no_visual_information():
    actions = np.array([6, 1, 1, 2, 5])
    features = action_history_features(actions)
    assert features.shape[0] == 5 and not features[0, :6].any()
    np.testing.assert_array_equal(features[1, :6], np.eye(6)[1])


def random_walk_sidecar(rng, steps=60):
    layout = np.ones((5, 5), np.uint8)
    layout[rng.integers(5, size=4), rng.integers(5, size=4)] = 0
    layout[0, 0] = 1
    position = np.array([0.5, 0.5])
    heading = rng.uniform(-np.pi, np.pi)
    positions, directions = [], []
    for _ in range(steps):
        positions.append(position.copy())
        directions.append([np.cos(heading), np.sin(heading)])
        heading += rng.normal(0, 0.3)
        position = np.clip(position + 0.2 * np.array(directions[-1]), 0.1, 4.9)
    positions, directions = np.array(positions), np.array(directions)
    targets = rng.uniform(0.5, 4.5, (2, 2))
    visible = rng.random((steps, 2)) < 0.2
    return layout, {
        "agent_pos": positions, "agent_dir": directions,
        "maze_layout": np.broadcast_to(layout, (steps,) + layout.shape),
        "targets_pos": np.broadcast_to(targets, (steps, 2, 2)),
        "targets_vec": L.egocentric(targets[None] - positions[:, None], directions[:, None]),
        "target_color": L.TARGET_PALETTE[rng.integers(2, size=steps)],
        "target_visible_pixels": visible.astype(np.int32),
        "observed_cells": rng.random((steps, 7, 7)) < 0.3,
        "depth_coarse": rng.uniform(0, 5, (steps, 8, 8)).astype(np.float32),
    }


def write_partition(root, partition, rng, episodes=3):
    root.mkdir()
    entries = []
    for index in range(episodes):
        layout, sidecar = random_walk_sidecar(rng)
        labels = L.episode_labels(sidecar)
        steps = len(sidecar["agent_pos"])
        informative = np.c_[labels["targets_vec"].reshape(steps, -1),
                            labels["origin_relative_position"], labels["origin_relative_heading"]]
        data = {"belief": informative + 0.01 * rng.normal(size=informative.shape),
                "image_features": rng.normal(size=(steps, 6)),
                "layout_ids": np.full(steps, f"{partition}-{index}-{layout.tobytes().hex()}"),
                "episode_ids": np.full(steps, f"{partition}:{index}"),
                "previous_action": rng.integers(0, 7, steps)}
        data.update({f"target_{key}": value for key, value in labels.items()})
        path = root / f"episode{index:05d}.features.npz"
        np.savez_compressed(path, **data)
        entries.append({"file": path.name, "sha256": sha256(path)})
    manifest = {"frozen": True, "partition": partition, "checkpoint_sha256": "same", "entries": entries}
    (root / "manifest.json").write_text(json.dumps(manifest))
    return root


def test_suite_runs_every_family_and_informative_beliefs_beat_controls(tmp_path):
    rng = np.random.default_rng(0)
    dirs = [write_partition(tmp_path / name, name, rng)
            for name in ("probe_train", "probe_validation", "probe_final_test")]
    result = run_suite(*dirs, stride=1)
    families = result["families"]
    assert set(families) == {"spatial_memory", "map_topology", "self_localization",
                             "local_geometry", "loop_closure"}
    memory = families["spatial_memory"]
    assert memory["belief"]["visible_target_location_rmse"] < memory["image_features"][
        "visible_target_location_rmse"]
    position = families["self_localization"]
    assert position["belief"]["origin_position_rmse"] < position["action_history"][
        "origin_position_rmse"]
    assert result["primary"]["matched_revisit_efficiency_gain"] is None
    assert result["primary"]["known_wall_balanced_iou"] is not None
    json.dumps(result, allow_nan=False)


def test_report_pairs_training_seeds_and_reduces_revisit_pairs_per_episode(tmp_path):
    from disco_spatial.evaluation.report import build_report, sign_flip_p

    assert sign_flip_p([1.0, 1.0, 1.0, 1.0, 1.0]) == 0.0625  # resolution limit at N=5
    assert sign_flip_p([1.0, -1.0]) == 1.0

    def write(name, seed, rmse, iou, gains):
        abilities = {"families": {
            "spatial_memory": {"belief": {"episodes": [
                {"layout_id": f"L{i}", "episode_id": f"e{i}", "seen_hidden_rmse": rmse + i}
                for i in range(2)]}},
            "map_topology": {"belief": {"episodes": [
                {"layout_id": f"L{i}", "episode_id": f"e{i}", "known_balanced_iou": iou}
                for i in range(2)]}}}}
        episodes = [{"seed": 2000000 + i, "layout_id": f"L{i}", "episode_id": f"e{i}",
                     "return": 3.0 + i} for i in range(2)]
        navigation = {"interventions": {
            "none": {"pairs": [{"layout_id": "L0", "episode_id": "e0", "efficiency_gain": g}
                               for g in gains], "episodes": episodes},
            "reset_every_step": {"pairs": [], "episodes": [dict(row, **{"return": 1.0})
                                                           for row in episodes]}}}
        a, n = tmp_path / f"{name}{seed}a.json", tmp_path / f"{name}{seed}n.json"
        a.write_text(json.dumps(abilities))
        n.write_text(json.dumps(navigation))
        return f"{name}:{seed}:{a}:{n}"

    runs = [write("ac", s, 2.0, 0.5, [0.0]) for s in (0, 1)]
    runs += [write("disco", s, 1.0, 0.6, [0.2, 0.4]) for s in (0, 1)]
    report = build_report(runs, "ac", bootstrap_samples=50)
    memory = report["conditions"]["disco"]["seen_hidden_target_location_rmse"]
    assert memory["paired_vs_reference"]["mean_difference"] == pytest.approx(-1.0)
    revisit = report["conditions"]["disco"]["matched_revisit_efficiency_gain"]
    assert revisit["aggregate"]["n_episode_records"] == 2  # two pairs -> one episode value
    assert revisit["aggregate"]["mean"] == pytest.approx(0.3)
    assert report["status"] == "pilot_descriptive_not_confirmatory"
    drop = report["conditions"]["disco"]["memory_reset_return_drop"]["aggregate"]
    assert drop["mean"] == pytest.approx(2.5)  # (3-1 + 4-1) / 2 per seed, paired by env seed
    assert "task_return (secondary)" in report["markdown"]
    navigation_only = build_report([f"random:0:-:{runs[0].split(':')[3]}"] + runs, "ac",
                                   bootstrap_samples=10)
    assert "seen_hidden_target_location_rmse" not in navigation_only["conditions"]["random"]
    with pytest.raises(ValueError, match="Duplicate"):
        build_report(runs + [runs[0]], "ac")
