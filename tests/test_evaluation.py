"""Research-validity checks for the new spatial evaluation primitives."""

import copy
import json

import numpy as np
import pytest

from disco_spatial.evaluation import (
    ProbeSplit,
    aggregate_seed_clusters,
    build_manifest,
    causal_observability_masks,
    canonical_layout_fingerprint,
    default_protocol,
    evaluate_ridge_probe,
    fit_ridge_probe,
    holm_primary_tests,
    masked_map_iou,
    observability_masks,
    revisit_efficiency,
    target_location_rmse,
    validate_protocol,
)


def split(features, targets, layout, episode="episode0"):
    return ProbeSplit(np.asarray(features), np.asarray(targets),
                      (layout,) * len(features), (episode,) * len(features))


def test_observability_uses_causal_partition_and_rejects_impossible_history():
    masks = observability_masks([True, False, False], [True, True, False])
    assert masks["visible"].tolist() == [True, False, False]
    assert masks["seen_hidden"].tolist() == [False, True, False]
    assert masks["never_seen"].tolist() == [False, False, True]
    with pytest.raises(ValueError, match="causal ever_seen"):
        observability_masks([True], [False])
    with pytest.raises(ValueError, match="different shapes"):
        observability_masks([True], [[True]])


def test_geometry_fingerprint_removes_d4_symmetries_but_not_geometry_changes():
    grid = np.array([[1, 0, 1, 1], [0, 0, 1, 0], [1, 1, 0, 0]], dtype=np.uint8)
    fingerprint = canonical_layout_fingerprint(grid)
    assert len(fingerprint) == 64
    for reflected in (grid, np.fliplr(grid)):
        for turns in range(4):
            assert canonical_layout_fingerprint(np.rot90(reflected, turns)) == fingerprint
    assert canonical_layout_fingerprint(grid.astype(bool)) == fingerprint
    changed = grid.copy()
    changed[0, 0] = 0
    assert canonical_layout_fingerprint(changed) != fingerprint
    with pytest.raises(ValueError, match="0/1"):
        canonical_layout_fingerprint([[0, 1, 2]])
    with pytest.raises(ValueError, match="nonempty"):
        canonical_layout_fingerprint(np.empty((0, 2)))


def test_location_error_is_euclidean_masked_without_alignment():
    masks = observability_masks([True, False, False], [True, True, False])
    result = target_location_rmse([[3, 4], [0, 2], [0, 0]], np.zeros((3, 2)), masks)
    assert result["visible"]["rmse"] == 5.0
    assert result["seen_hidden"]["rmse"] == 2.0
    assert result["never_seen"]["n_target_observations"] == 1
    shifted = target_location_rmse(np.full((3, 2), 10), np.zeros((3, 2)), masks)
    assert shifted["seen_hidden"]["rmse"] == pytest.approx(np.sqrt(200))
    empty = target_location_rmse([[0, 0]], [[0, 0]], observability_masks([True], [True]))
    assert empty["seen_hidden"]["rmse"] is None
    bad_masks = {name: np.ones(3, bool) for name in masks}
    with pytest.raises(ValueError, match="disjoint partition"):
        target_location_rmse(np.zeros((3, 2)), np.zeros((3, 2)), bad_masks)


def test_future_visibility_cannot_mark_past_targets_as_seen():
    masks = causal_observability_masks([[False], [True], [False]])
    assert masks["never_seen"].ravel().tolist() == [True, False, False]
    assert masks["seen_hidden"].ravel().tolist() == [False, False, True]


def test_map_iou_balances_classes_and_excludes_never_observed_cells():
    result = masked_map_iou([0.9, 0.1, 0.9, np.nan], [1, 0, 0, 1], [1, 1, 1, 0])
    assert result["wall_iou"] == 0.5
    assert result["free_iou"] == 0.5
    assert result["balanced_iou"] == 0.5
    assert result["n_known_cells"] == 3
    assert masked_map_iou([0.0], [0], [0])["balanced_iou"] is None
    assert masked_map_iou([0.0], [0], [1])["balanced_iou"] == 1.0
    with pytest.raises(ValueError, match="probabilities"):
        masked_map_iou([1.1], [1], [1])


def segment(segment_id, kind, target="red", match_bin="distance2_visible0", **kwargs):
    return {"seed": 0, "layout_id": "maze0", "episode_id": "ep0", "target_id": target,
            "segment_id": segment_id, "kind": kind, "match_bin": match_bin,
            "success": True, "oracle_distance": 2.0, "path_length": 4.0, **kwargs}


def test_revisit_pairs_only_same_goal_episode_and_predeclared_stratum():
    rows = [segment(0, "first_visit"), segment(1, "revisit", path_length=2.0),
            segment(2, "revisit", target="blue"),
            segment(3, "revisit", match_bin="different_distance")]
    result = revisit_efficiency(rows)
    assert result["n_matched_pairs"] == 1
    assert result["n_unmatched_segments"] == 2
    assert result["mean_efficiency_gain"] == 0.5
    assert result["pairs"][0]["target_id"] == "red"
    with pytest.raises(ValueError, match="Duplicate"):
        revisit_efficiency([rows[0], rows[0]])
    with pytest.raises(ValueError, match="Ambiguous"):
        revisit_efficiency([segment(0, "first_visit"), segment(1, "first_visit")])
    failed = revisit_efficiency([segment(0, "first_visit"), segment(1, "revisit", success=False)])
    assert failed["mean_efficiency_gain"] == -0.5
    with pytest.raises(ValueError, match="occur after"):
        revisit_efficiency([segment(1, "first_visit"), segment(0, "revisit")])
    with pytest.raises(ValueError, match="prior successful"):
        revisit_efficiency([segment(0, "first_visit", success=False), segment(1, "revisit")])


def test_hierarchical_aggregation_weights_layouts_and_seeds_not_episode_counts():
    rows = [
        {"seed": 0, "layout_id": "a", "episode_id": f"ep{i}", "score": 0.0}
        for i in range(10)
    ] + [
        {"seed": 0, "layout_id": "b", "episode_id": "ep0", "score": 2.0},
        {"seed": 1, "layout_id": "a", "episode_id": "ep0", "score": 2.0},
        {"seed": 1, "layout_id": "b", "episode_id": "ep0", "score": 4.0},
    ]
    result = aggregate_seed_clusters(rows, "score", bootstrap_samples=200, seed=9)
    assert result["per_seed"] == [1.0, 3.0]
    assert result["mean"] == 2.0
    assert result["std_across_training_seeds"] == pytest.approx(np.sqrt(2))
    assert result["n_training_seeds"] == 2
    assert result["n_layout_clusters_per_seed"] == [2, 2]
    assert result == aggregate_seed_clusters(rows, "score", bootstrap_samples=200, seed=9)
    lo, hi = result["hierarchical_bootstrap_ci95"]
    assert lo <= 2 <= hi
    with pytest.raises(ValueError, match="Duplicate episode"):
        aggregate_seed_clusters([rows[0], rows[0]], "score")


def test_one_seed_does_not_fabricate_across_training_uncertainty():
    result = aggregate_seed_clusters([
        {"seed": 0, "layout_id": "a", "episode_id": "e", "score": 1.0},
    ], "score", bootstrap_samples=10)
    assert result["std_across_training_seeds"] is None
    assert result["hierarchical_bootstrap_ci95"] is None


def test_holm_correction_uses_fixed_family_and_monotone_adjustment():
    keys = default_protocol()["primary_endpoints"]
    result = holm_primary_tests(dict(zip(keys, [0.01, 0.03, 0.04])))
    assert [result["tests"][key]["holm_adjusted_p"] for key in keys] == [0.03, 0.06, 0.06]
    assert [result["tests"][key]["reject"] for key in keys] == [True, False, False]
    with pytest.raises(ValueError, match="exactly the three"):
        holm_primary_tests({keys[0]: 0.01})


def test_probe_train_only_scaling_validation_selection_and_unseen_test():
    x = np.arange(12, dtype=float).reshape(6, 2)
    y = 3 * x[:, :1] - 2
    train = split(x, y, "train")
    validation = split(x + 2, y + 6, "val")
    model = fit_ridge_probe(train, validation, alphas=(10000, 0.0001))
    np.testing.assert_allclose(model.feature_mean, x.mean(0))
    np.testing.assert_allclose(model.feature_std, x.std(0))
    assert model.alpha == 0.0001
    test = split(x + 4, y + 12, "test")
    result = evaluate_ridge_probe(model, test)
    assert result["mse"] < 1e-6
    assert result["train_target_mean_baseline_mse"] > 1
    assert result["n_test_layouts"] == 1
    json.dumps({k: v for k, v in result.items() if k != "prediction"}, allow_nan=False)


def test_probe_layout_leak_rejected_at_fit_and_final_test():
    train = split([[0], [1], [2]], [[0], [1], [2]], "train")
    with pytest.raises(ValueError, match="Layout leakage"):
        fit_ridge_probe(train, train)
    validation = split([[1], [2]], [[1], [2]], "val")
    model = fit_ridge_probe(train, validation)
    with pytest.raises(ValueError, match="Final-test layout"):
        evaluate_ridge_probe(model, validation)
    with pytest.raises(ValueError, match="Final-test layout"):
        evaluate_ridge_probe(model, train)


def test_probe_copies_inputs_and_requires_provenance():
    x = np.array([[0.0], [1.0]])
    data = split(x, [[0], [1]], "train")
    x[:] = 99
    assert data.features[0, 0] == 0
    with pytest.raises(ValueError, match="provenance"):
        ProbeSplit(np.ones((2, 1)), np.ones((2, 1)), ("a",), ("e", "e"))


def test_protocol_is_deterministic_from_config_and_explicit_about_gates():
    config = {"method": "disco", "maze_size": 9, "steps": 100, "seed": 0}
    manifest = build_manifest(config)
    assert manifest == build_manifest(copy.deepcopy(config))
    assert len(manifest["protocol"]["ability_families"]) == 6
    assert len(manifest["protocol"]["primary_endpoints"]) == 3
    assert manifest["claims"]["sdpo_teacher_implemented"]
    assert not manifest["claims"]["sdpo_teacher_validated"]
    assert not manifest["claims"]["benchmark_training_completed"]
    assert not manifest["claims"]["runtime_end_to_end_verified"]
    assert not manifest["protocol"]["teacher"]["validated"]
    assert not manifest["protocol"]["teacher"]["privileged_inputs"]
    statistics = manifest["protocol"]["statistics"]
    assert statistics["planned_seed_role"] == "pilot_descriptive_only"
    assert statistics["confirmatory_seed_count"] is None
    assert statistics["confirmatory_seed_ids"] is None
    assert not statistics["confirmatory_design_locked"]
    test_plan = statistics["paired_test_candidate"]
    assert test_plan["minimum_possible_p_n5"] == 0.0625
    assert test_plan["minimum_possible_p_n8"] == 0.0078125
    json.dumps(manifest, allow_nan=False)
    changed = build_manifest({**config, "steps": 200})
    assert changed["config_sha256"] != manifest["config_sha256"]
    assert changed["protocol_sha256"] == manifest["protocol_sha256"]
    protocol = default_protocol()
    protocol["teacher"]["validated"] = True
    with pytest.raises(ValueError, match="run evidence"):
        validate_protocol(protocol)
    protocol = default_protocol()
    protocol["teacher"]["inputs"].append("maze_layout")
    with pytest.raises(ValueError, match="guided-study"):
        validate_protocol(protocol)
    protocol = default_protocol()
    protocol["teacher"]["enabled_purposes"].append("confirmatory")
    with pytest.raises(ValueError, match="confirmatory"):
        validate_protocol(protocol)
    assert manifest["protocol"]["training"]["step_cost"] == 0.0
    with pytest.raises(ValueError, match="Teacher training is gated"):
        build_manifest({"recipe": "sdpo"})
    with pytest.raises(ValueError, match="development/pilot"):
        build_manifest({"purpose": "confirmatory"})


@pytest.mark.parametrize("mutation, message", [
    (lambda p: p["training"]["allowed_inputs"].append("pose"), "Privileged geometry"),
    (lambda p: p["splits"].update(standardization="all_data"), "standardization"),
    (lambda p: p["splits"].update(final_test_used_for_selection=True), "final tests"),
    (lambda p: p["primary_endpoints"].append("best_posthoc_metric"), "fixed before"),
    (lambda p: p["statistics"].update(no_frame_pseudoreplication=False), "Frame-level"),
])
def test_protocol_rejects_leakage_and_posthoc_primary_selection(mutation, message):
    protocol = default_protocol()
    mutation(protocol)
    with pytest.raises(ValueError, match=message):
        validate_protocol(protocol)
