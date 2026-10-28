"""Versioned, machine-checkable research declarations, without model imports."""

import copy
import hashlib
import json


FAMILIES = {
    "local_geometry", "self_localization", "spatial_memory",
    "loop_closure", "map_topology", "navigation",
}
PRIMARY_ENDPOINTS = {
    "seen_hidden_target_location_rmse", "known_wall_balanced_iou",
    "matched_revisit_efficiency_gain",
}
FORBIDDEN_TRAINING_INPUTS = {
    "depth", "pose", "agent_pos", "agent_dir", "target_pos", "targets_pos",
    "target_vec", "targets_vec", "maze_layout", "wall_map", "shortest_path",
    "expert_action", "top_down_map",
}
TEACHER_INPUTS = {
    "stopgrad_student_belief", "own_reward_outcome_within_horizon",
    "own_rgb_frame_of_previous_same_color_success", "elapsed_steps",
}


def default_protocol():
    """Return a fresh declaration; entries describe plans, never completed results."""
    return {
        "schema_version": 1,
        "status": "development_no_research_result",
        "question": "Which spatial abilities arise from limited-reward RGB RL, and which update changes induce them?",
        "training": {
            "one_task_per_study": True,
            "allowed_inputs": ["rgb", "previous_action", "previous_reward"],
            "geometry_supervision": False,
            "privileged_teacher_feedback": False,
            "reward": "Unmodified Memory Maze: +1 on requested-target contact, otherwise 0; no step cost or distance shaping.",
            "step_cost": 0.0,
            "checkpoint_selection": "training_task_validation_return_only",
        },
        "benchmark": {
            "preferred": "memory_maze_rgb",
            "alternative": "robothor_rgb_objectnav",
            "preserve_official_task_and_report_reward_variant": True,
            "caveat": "Memory Maze rewards each target contact, not only episode termination; strict terminal-only studies must be identified separately.",
            "sources": [
                "https://github.com/jurgisp/memory-maze",
                "https://arxiv.org/abs/2210.13383",
                "https://github.com/allenai/robothor-challenge",
                "https://allenact.org/projects/objectnav_baselines/",
            ],
        },
        "hypotheses": [
            {"id": "H1", "comparison": "Disco versus AC+next-policy auxiliary",
             "test": "Do matched recurrent Disco features improve the three fixed spatial endpoints?"},
            {"id": "H2", "comparison": "reward reuse versus trajectory-only self-distillation",
             "test": "Does a distinct teacher loss add value beyond corrected reward replay at matched data and compute?"},
            {"id": "H3", "comparison": "teacher-feedback utility",
             "test": "When an implemented teacher is validated, separate trajectory-derived feedback from new privileged information.",
             "status": "gated"},
        ],
        "ability_families": [
            {"id": "local_geometry", "metrics": ["depth_rmse", "forward_free_space_rmse",
                                                 "traversable_average_precision"],
             "origin": "proposed_custom_diagnostic",
             "implementation": "render_depth_and_layout_ray_readouts",
             "unit": "single RGB frame; depth/free-space are one related family"},
            {"id": "self_localization", "metrics": ["origin_position_rmse", "heading_error",
                                                    "relative_motion_rmse", "drift_by_time"],
             "origin": "proposed_custom_diagnostic",
             "implementation": "origin_gauge_readouts_with_action_history_control",
             "unit": "history state or paired frames in episode-origin gauge"},
            {"id": "spatial_memory", "metrics": ["seen_hidden_target_location_rmse"],
             "origin": "official_object_location_probe_concept_custom_masked_metric",
             "implementation": "segmentation_visibility_masked_location_readout",
             "unit": "previously observed but currently invisible targets"},
            {"id": "loop_closure", "metrics": ["place_retrieval_recall_at_1"],
             "origin": "proposed_custom_diagnostic",
             "implementation": "unsupervised_feature_retrieval_with_chance_rate",
             "unit": "temporally separated places with heading control"},
            {"id": "map_topology", "metrics": ["known_wall_balanced_iou", "unknown_cell_balanced_iou"],
             "origin": "official_wall_probe_concept_custom_known_mask_and_topology_metrics",
             "implementation": "depth_backprojected_known_mask_origin_canvas_readout",
             "unit": "causally observed map cells; unseen cells scored separately"},
            {"id": "navigation", "metrics": ["task_return", "matched_revisit_efficiency_gain",
                                             "memory_intervention_return_drop"],
             "origin": "official_return_custom_matched_revisit",
             "implementation": "frozen_policy_episodes_with_segment_telemetry",
             "unit": "frozen-policy episodes on policy-final-test seeds"},
        ],
        "primary_endpoints": sorted(PRIMARY_ENDPOINTS),
        "observability": {
            "masks": ["visible", "seen_hidden", "never_seen"],
            "history_cutoff": "current_time_inclusive_no_future_visibility",
            "gauge": "ego-centric targets; position relative to episode origin; no test-label registration",
            "known_map": "only cells causally observed through evaluation time",
        },
        "splits": {
            "unit": "layout_and_episode",
            "names": ["rl_train", "rl_validation", "probe_train", "probe_validation", "final_test"],
            "layout_disjoint": True,
            "standardization": "probe_train_only",
            "ridge_selection": "probe_validation_only",
            "final_test_used_for_selection": False,
            "seed_namespaces": {
                "rl_train": [0, 999999],
                "rl_validation": [1000000, 1999999],
                "policy_final_test": [2000000, 2999999],
                "probe_train": [3000000, 3999999],
                "probe_validation": [4000000, 4999999],
                "probe_final_test": [5000000, 5999999],
            },
            "identity_check": "Canonical maze-layout fingerprints must be disjoint; different RNG seeds alone do not prove distinct geometry.",
        },
        "evaluation_levels": {
            "diagnostic": "Frozen encoder/state plus supervised readout; labels used after RL only.",
            "readout_transfer": "New controller/head fitted on frozen features; not zero-shot.",
            "functional_zero_shot": "Frozen actor and encoder; no task-specific training or test-time weight updates.",
        },
        "controls": [
            "untrained_encoder", "frame_only_no_memory", "action_only_dead_reckoning",
            "hidden_state_reset", "history_truncation", "train_mean_predictor",
            "common_evaluation_trajectories", "own_policy_functional_episodes",
        ],
        "ablations": [
            {"id": "A0", "method": "actor_critic_aux", "new_feedback": False},
            {"id": "A1", "method": "disco", "new_feedback": False},
            {"id": "A2", "method": "disco_corrected_reward_reuse", "new_feedback": False},
            {"id": "A2s", "method": "positive_advantage_self_imitation", "new_feedback": False,
             "status": "implemented_objective_change"},
            {"id": "A3", "method": "outcome_feedback_self_distillation", "new_feedback": False,
             "status": "implemented_gated_by_per_run_evidence"},
            {"id": "A3m", "method": "outcome_plus_success_memory_self_distillation",
             "new_feedback": False, "status": "implemented_gated_by_per_run_evidence"},
            {"id": "A3c", "method": "shuffled_feedback_control", "new_feedback": False,
             "status": "implemented_control"},
            {"id": "A4", "method": "privileged_guided_teacher", "new_feedback": True,
             "status": "separate_guided_study_not_emergence_not_implemented"},
        ],
        "budget": {
            "matched": ["encoder_and_memory_capacity", "environment_seeds", "reward", "initialization"],
            "two_comparisons": ["matched_fresh_transitions", "matched_total_learner_compute"],
            "report": ["fresh_transitions", "replayed_transitions", "learner_updates",
                       "teacher_compute", "wall_seconds", "peak_memory"],
            "warning": "Equal environment steps alone do not make extra replay or teacher compute free.",
        },
        "statistics": {
            "planned_training_seeds": [0, 1, 2, 3, 4],
            "planned_seed_role": "pilot_descriptive_only",
            "confirmatory_seed_count": None,
            "confirmatory_seed_ids": None,
            "confirmatory_design_locked": False,
            "confirmatory_lock_requires": [
                "pilot_variance_and_paired_effect_estimates",
                "scientifically_meaningful_effect_size",
                "power_plan_including_three_primary_tests",
                "independent_confirmatory_seed_and_layout_banks",
            ],
            "independent_unit": "training_seed_and_layout_cluster",
            "aggregation": "episode_then_layout_then_training_seed",
            "interval": "descriptive_hierarchical_seed_layout_bootstrap_not_a_p_value",
            "paired_test_candidate": {
                "method": "two_sided_exact_paired_training_seed_sign_flip",
                "assumption": "Paired seed differences are sign-exchangeable under the null; frame-level permutations are invalid.",
                "statistic": "mean paired difference after matched-layout averaging",
                "minimum_possible_p_n5": 2 / 2**5,
                "minimum_possible_p_n8": 2 / 2**8,
                "candidate_seed_count_for_power_planning": 8,
                "caveat": "Five seeds cannot reject at 0.05 even before Holm; eight or more permit finer resolution but do not guarantee power.",
            },
            "multiple_comparisons": "After confirmatory lock: Holm at 0.05 across H1's three fixed primary endpoints for the single Disco-versus-AC+aux contrast. Pilot statistics and all other contrasts remain descriptive/exploratory.",
            "comparison_families": {
                "H1": {"contrast": "Disco versus AC+aux", "endpoints": sorted(PRIMARY_ENDPOINTS),
                       "adjustment": "one_Holm_family_of_three_after_confirmatory_lock"},
                "H2": "Gated separate preregistration; reuse-versus-distillation is not included in H1 correction.",
                "H3": "Teacher pilot gated; feedback variants need a separate endpoint/contrast family before confirmation.",
                "secondary_and_ood": "Exploratory; no per-ability-family resets of alpha and no expanded confirmatory claims.",
            },
            "paired_method_comparison": True,
            "no_frame_pseudoreplication": True,
            "ood_is_axis_not_separate_ability": True,
        },
        "teacher": {
            "name": "SDPO-style feedback self-teacher for a recurrent RGB student",
            "paper": "https://arxiv.org/abs/2601.20802",
            "status": "implemented_experimental_unvalidated",
            "implemented": True,
            "validated": False,
            "enabled_purposes": ["development", "pilot"],
            "inputs": sorted(TEACHER_INPUTS),
            "privileged_inputs": False,
            "form": "base logits + delta(stopgrad belief, feedback); delta trained by behavior "
                    "cloning on observed outcomes; EMA delta for KL(student || stopgrad(teacher))",
            "per_run_gate": "distill only after held-out information gain beats the shuffled-"
                            "feedback control, local improvement is positive and copy rate is low",
            "required_evidence": ["causal_student_inputs", "teacher_input_provenance",
                                  "nontrivial_teacher_advantage", "loss_gradient_test",
                                  "reward_replay_control", "compute_matched_control",
                                  "shuffled_feedback_control", "teacher_acting_validation"],
        },
        "gates": [
            {"id": "G0", "requires": "observation boundary, split/probe and metric tests"},
            {"id": "G1", "requires": "actual official benchmark reset/step plus recurrent learner integration"},
            {"id": "G2", "requires": "pilot learns above random and benefits from history"},
            {"id": "G3", "requires": "pilot-informed power plan and locked independent confirmatory seed count, endpoint family and layout bank"},
            {"id": "G4", "requires": "H2/H3 only after teacher evidence and feedback classification"},
            {"id": "G4a", "requires": "teacher information gain exceeds shuffled control on held-out chunks"},
            {"id": "G4b", "requires": "causal success-query teacher acts at least as well as the student"},
        ],
    }


def validate_protocol(protocol):
    """Validate declarations, not implementation completion or scientific success."""
    if not isinstance(protocol, dict) or protocol.get("schema_version") != 1:
        raise ValueError("Expected protocol mapping with schema_version=1")
    training = protocol["training"]
    allowed = training["allowed_inputs"]
    if (set(allowed) & FORBIDDEN_TRAINING_INPUTS or
            not set(allowed).issubset({"rgb", "previous_action", "previous_reward"})):
        raise ValueError("Privileged geometry cannot be an input in the emergence protocol")
    if "rgb" not in allowed or len(set(allowed)) != len(allowed):
        raise ValueError("Declare a unique input whitelist including RGB")
    if training["geometry_supervision"] or training["privileged_teacher_feedback"]:
        raise ValueError("GT-supervised/guided studies need a separate protocol")
    if training.get("step_cost") != 0.0:
        raise ValueError("The primary Memory Maze condition preserves upstream reward without step cost")
    if training["checkpoint_selection"] != "training_task_validation_return_only":
        raise ValueError("Encoder checkpoints must not be selected on spatial test probes")
    families = [family["id"] for family in protocol["ability_families"]]
    if len(families) != len(FAMILIES) or set(families) != FAMILIES:
        raise ValueError("Declare exactly the six nonredundant ability families")
    primary = protocol["primary_endpoints"]
    if len(primary) != len(PRIMARY_ENDPOINTS) or set(primary) != PRIMARY_ENDPOINTS:
        raise ValueError("The three primary endpoints are fixed before evaluation")
    splits = protocol["splits"]
    if not splits["layout_disjoint"] or splits["final_test_used_for_selection"]:
        raise ValueError("Layouts must be disjoint and final tests excluded from selection")
    if splits["standardization"] != "probe_train_only":
        raise ValueError("Feature standardization must use probe training data only")
    if splits["ridge_selection"] != "probe_validation_only":
        raise ValueError("Ridge alpha must be selected on probe validation only")
    if not protocol["statistics"]["no_frame_pseudoreplication"]:
        raise ValueError("Frame-level sample counts are not independent experimental N")
    if protocol["statistics"]["planned_seed_role"] != "pilot_descriptive_only":
        raise ValueError("The current five-seed plan is pilot/descriptive, not confirmatory")
    teacher = protocol["teacher"]
    if teacher.get("validated"):
        raise ValueError("Teacher validation comes from logged run evidence, not a declaration")
    if teacher.get("privileged_inputs") or not set(teacher.get("inputs", ())) <= TEACHER_INPUTS:
        raise ValueError("Privileged teacher feedback needs a separate guided-study protocol")
    if set(teacher.get("enabled_purposes", ())) - {"development", "pilot"}:
        raise ValueError("The teacher remains gated for confirmatory use")
    json.dumps(protocol, allow_nan=False)
    return {"valid": True, "schema_version": 1, "ability_families": len(families),
            "primary_endpoints": len(primary), "teacher_validated": False,
            "scope": "declaration validation only; not evidence of completed benchmark experiments"}


def build_manifest(config=None, protocol=None):
    """Build a deterministic manifest from a config dictionary (no model imports)."""
    protocol = copy.deepcopy(default_protocol() if protocol is None else protocol)
    validation = validate_protocol(protocol)
    config = copy.deepcopy({} if config is None else config)
    if not isinstance(config, dict):
        raise ValueError("Experiment config must be a dictionary")
    if config.get("purpose", "development") not in {"development", "pilot"}:
        raise ValueError("Only development/pilot manifests are enabled before protocol validation")
    if config.get("recipe", "fresh") not in {"fresh", "recent_replay"}:
        raise ValueError("Teacher training is gated; only fresh/recent_replay recipes are enabled")
    def canonical(value):
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return {
        "schema_version": 1,
        "status": "protocol_manifest_not_experiment_results",
        "config": config, "protocol": protocol,
        "config_sha256": hashlib.sha256(canonical(config)).hexdigest(),
        "protocol_sha256": hashlib.sha256(canonical(protocol)).hexdigest(),
        "validation": validation,
        "claims": {
            "runtime_end_to_end_verified": False,
            "benchmark_training_completed": False,
            "six_ability_suite_implemented": True,
            "six_ability_suite_end_to_end_completed": False,
            "sdpo_teacher_implemented": True,
            "sdpo_teacher_validated": False,
        },
    }
