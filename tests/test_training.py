"""Collector/reset/replay integration, independent of a native renderer."""

from dataclasses import replace
import json

import jax
import numpy as np
import pytest

from disco_spatial.agents import RecurrentAgent
from disco_spatial.config import TrainConfig
from disco_spatial.replay import ReplayChunk
from disco_spatial.training import BatchedEnvironment, collect_rollout, replay_rollout, train
from fake_env import ShortTask


def small_config(tmp_path, **changes):
    settings = dict(output=str(tmp_path / "run"), method="actor_critic",
                    camera_resolution=64, steps=8, num_envs=1, rollout_length=2,
                    channels=(2,), feature_dim=4, hidden_size=4,
                    validation_seeds=(), save_trajectories=False)
    return replace(TrainConfig(), **dict(settings, **changes))


def test_chunks_end_at_last_and_reset_without_dummy_actions(tmp_path):
    config = small_config(tmp_path)
    env = BatchedEnvironment([ShortTask()], config.method)
    agent = RecurrentAgent(config, env)
    key = jax.random.PRNGKey(3)
    learner = agent.initial_learner_state(key)
    state = agent.initial_actor_state(key)
    first, state, _, key, count = collect_rollout(agent, learner.params, state, env, key, 2, 0)
    assert count == 2 and first.actions.shape[0] == 3
    second, state, _, key, count = collect_rollout(agent, learner.params, state, env, key, 2, 0)
    assert count == 1 and env.timesteps[0].last()
    assert float(second.discounts[-1, 0]) == 1  # time limit bootstraps
    assert not np.asarray(second.observations["is_first"]).any()
    third, _, _, _, count = collect_rollout(agent, learner.params, state, env, key, 1, 0)
    assert count == 1 and bool(third.observations["is_first"][0, 0])
    assert env.records[0].trajectory().n_steps == 1  # reset was not an action


def test_replay_restores_current_prefix_and_keeps_behavior(tmp_path):
    config = small_config(tmp_path)
    env = BatchedEnvironment([ShortTask()], config.method)
    agent = RecurrentAgent(config, env)
    key = jax.random.PRNGKey(0)
    learner = agent.initial_learner_state(key)
    state = agent.initial_actor_state(key)
    original, _, _, _, _ = collect_rollout(agent, learner.params, state, env, key, 2, 0)
    trajectory = env.records[0].trajectory()
    replay, start_state, diagnostics = replay_rollout(
        agent, learner.params, ReplayChunk(trajectory, 1, 2, 0))
    expected = jax.tree.map(lambda x: x[1], original.states)
    for actual, wanted in zip(jax.tree.leaves(start_state), jax.tree.leaves(expected)):
        np.testing.assert_allclose(actual, wanted, atol=1e-6)
    np.testing.assert_array_equal(replay.agent_outs["logits"][0, 0], trajectory.behavior_logits[1])
    assert diagnostics["replay_prefix_steps"] == 1
    np.testing.assert_allclose(diagnostics["replay_rho_mean"], 1, atol=1e-6)


def test_real_learner_driver_uses_replay_but_not_teacher(tmp_path):
    config = small_config(tmp_path, recipe="recent_replay")
    result = train(config, env_factory=ShortTask)
    assert result["transitions"] == 8 and result["updates"] == 5
    assert not result["scientific_result"] and not result["sdpo_teacher_trained"]
    records = [json.loads(row) for row in (tmp_path / "run/metrics.jsonl").read_text().splitlines()]
    assert not records[0]["replay_used"]
    assert any(row["replay_used"] for row in records[1:])
    assert (tmp_path / "run/latest.pkl").is_file()
    assert not (tmp_path / "run/best.pkl").exists()  # no validation => no best claim
    with pytest.raises(FileExistsError):
        train(config, env_factory=ShortTask)


def test_budget_and_teacher_gates():
    with pytest.raises(ValueError, match="divisible"):
        TrainConfig(steps=3, num_envs=2)
    with pytest.raises(ValueError, match="SDPO"):
        TrainConfig(recipe="sdpo")
    with pytest.raises(ValueError, match="Confirmatory"):
        TrainConfig(purpose="confirmatory")


def test_block_prefix_reconstruction_matches_step_by_step_recurrence(tmp_path):
    from disco_spatial.trajectory import Trajectory
    from disco_spatial.training import reconstruct_prefix_state

    config = small_config(tmp_path)
    agent = RecurrentAgent(config, BatchedEnvironment([ShortTask()], config.method))
    params = agent.initial_learner_state(jax.random.PRNGKey(1)).params
    n = 300
    rng = np.random.default_rng(0)
    types = np.ones(n + 1, np.int8)
    types[0] = 0
    trajectory = Trajectory(
        episode_id="train:p", instance_id="p", split="train",
        rgb=rng.integers(0, 256, (n + 1, 64, 64, 3), dtype=np.uint8),
        actions=rng.integers(0, 6, n).astype(np.int32), rewards=np.zeros(n, np.float32),
        discounts=np.ones(n, np.float32), behavior_logits=np.zeros((n, 6), np.float32),
        policy_versions=np.zeros(n, np.int64), step_types=types,
        goal_segments=np.zeros(n + 1, np.int32), terminated=np.zeros(n, bool),
        truncated=np.zeros(n, bool), metadata={})
    step = jax.jit(agent.policy_network.one_step)
    state = agent.initial_actor_state(jax.random.PRNGKey(0), batch_size=1)
    for stop in (0, 1, 5, 300):
        expected = agent.initial_actor_state(jax.random.PRNGKey(0), batch_size=1)
        for t in range(stop):
            obs = {"rgb": trajectory.rgb[t:t + 1],
                   "previous_action": np.array([6 if t == 0 else trajectory.actions[t - 1]], np.int32),
                   "previous_reward": np.zeros(1, np.float32), "is_first": np.array([t == 0])}
            expected = step(params, expected, obs, obs["is_first"])[1]
        state = reconstruct_prefix_state(agent, params, trajectory, stop)
        for a, b in zip(jax.tree.leaves(state), jax.tree.leaves(expected)):
            np.testing.assert_allclose(a, b, atol=2e-5)


def test_matured_chunks_wait_for_the_whole_outcome_window():
    from disco_spatial.training import MaturedChunks
    from test_feedback import BLUE, RED, trajectory

    queue = MaturedChunks(length=2, horizon=3, limit=10)
    partial = trajectory([0, 1, 0, 0], [RED] * 5, complete=False)
    queue.observe([partial])
    # [0,2) needs 0+2+3-1 = 4 <= 4 observed transitions; [2,4) is not mature yet.
    assert [start for _, start in queue.pop(1)] == [0] and queue.pop(1) is None
    full = trajectory([0, 1, 0, 0, 1], [RED] * 3 + [BLUE] * 3, complete=True)
    queue.observe([full])
    assert [start for _, start in queue.pop(1)] == [2] and len(queue) == 0


def test_evidence_gate_opens_only_past_shuffled_control_and_can_close(tmp_path):
    from disco_spatial.training import TeacherGate

    config = small_config(tmp_path, teacher="outcome", distillation_weight=0.5,
                          teacher_warmup_updates=2, teacher_min_information_gain=0.05,
                          teacher_min_success_steps=10)
    gate = TeacherGate(config, decay=0.0)
    good = {"teacher_information_gain_success": 0.2,
            "teacher_shuffled_information_gain_success": 0.0,
            "teacher_local_improvement": 0.1, "teacher_copy_rate": 0.1,
            "teacher_success_steps": 12.0, "teacher_known_steps": 64.0}
    uninformative = dict(good, teacher_shuffled_information_gain_success=0.2)
    gate.observe(uninformative, 1)
    gate.observe(uninformative, 2)
    assert not gate.open and gate.distillation_weight() == 0
    gate.observe(dict(good, teacher_success_steps=3.0), 3)
    assert not gate.open  # too few successful steps behind the evidence
    gate.observe(good, 4)
    assert gate.open and gate.distillation_weight() == 0.5
    gate.observe(dict(good, teacher_local_improvement=-1.0), 5)
    assert not gate.open and gate.transitions == [("open", 4), ("close", 5)]
    sparse = TeacherGate(config, decay=1.0)
    for update in range(4):  # chunks without success must not dilute success evidence
        sparse.observe(dict(good, teacher_success_steps=0.0, teacher_information_gain_success=0.0), update)
    sparse.observe(good, 4)
    assert sparse.evidence()["teacher_information_gain_success"] == pytest.approx(0.2)
    never = TeacherGate(replace(config, teacher_gate="never"), decay=0.0)
    for update in range(5):
        never.observe(good, update)
    assert never.distillation_weight() == 0


@pytest.mark.parametrize("settings", [
    {"self_imitation_weight": 0.5},
    {"teacher": "outcome", "distillation_weight": 0.5, "teacher_gate": "always",
     "teacher_horizon": 2},
    {"teacher": "outcome_memory", "teacher_query": "counterfactual_success",
     "teacher_feedback": "shuffled", "distillation_weight": 0.5, "teacher_horizon": 2,
     "recipe": "recent_replay"},
])
def test_auxiliary_recipes_train_end_to_end_with_explicit_logs(tmp_path, settings):
    config = small_config(tmp_path, steps=16, **settings)
    result = train(config, env_factory=ShortTask)
    rows = [json.loads(row) for row in (tmp_path / "run/metrics.jsonl").read_text().splitlines()]
    assert result["transitions"] == 16 and (tmp_path / "run/init.pkl").is_file()
    assert all(np.isfinite(row["loss"]) for row in rows)
    if config.self_imitation_weight:
        assert any("self_imitation_loss" in row for row in rows)
        assert not result["sdpo_teacher_trained"]
    else:
        assert result["sdpo_teacher_trained"]
        trained = [row for row in rows if "teacher_nll" in row]
        assert trained and all(np.isfinite(row["teacher_nll"]) for row in trained)
        applied = sum(row.get("distillation_weight_applied", 0) > 0 for row in rows)
        assert result["distillation_updates"] == applied
        assert (applied > 0) == (config.teacher_gate == "always")  # 16 steps < evidence warmup


def test_teacher_configuration_is_explicit():
    with pytest.raises(ValueError, match="teacher variant"):
        TrainConfig(distillation_weight=0.1)
    with pytest.raises(ValueError, match="teacher must be"):
        TrainConfig(teacher="oracle_map")
    with pytest.raises(ValueError, match="teacher_query"):
        TrainConfig(teacher="outcome", teacher_query="future_pose")
    with pytest.raises(ValueError, match="linear head"):
        TrainConfig(teacher="outcome", teacher_hidden=-1)
    with pytest.raises(ValueError, match="teacher_l2"):
        TrainConfig(teacher="outcome", teacher_l2=-0.1)
    assert TrainConfig(teacher="outcome", teacher_hidden=0, teacher_l2=0.01).teacher_hidden == 0


def test_process_workers_match_in_process_environments_and_report_failures():
    from disco_spatial.envs.workers import ProcessEnvironment
    from fake_env import BrokenTask

    local, remote = ShortTask(seed=5), ProcessEnvironment(ShortTask, 9, 5, "train", 64)
    try:
        assert remote.metadata == local.metadata and remote.split == "train"
        np.testing.assert_array_equal(remote.reset().observation["rgb"],
                                      local.reset().observation["rgb"])
        for action in (1, 2, 3):
            expected, actual = local.step(action), remote.step(action)
            assert (actual.step_type, actual.reward, actual.discount) == (
                expected.step_type, expected.reward, expected.discount)
            np.testing.assert_array_equal(actual.observation["rgb"], expected.observation["rgb"])
        assert remote.episode_id == local.episode_id and remote.goal_segment == 1
        remote.step_async(0)  # ShortTask asserts after its last step: the worker fails
        with pytest.raises(RuntimeError, match="AssertionError"):
            remote.wait()
    finally:
        remote.close()
    with pytest.raises(RuntimeError, match="deliberately broken"):
        ProcessEnvironment(BrokenTask, 9, 0, "train", 64)


def test_training_with_worker_processes_is_identical_to_in_process(tmp_path):
    rows = {}
    for processes in (False, True):
        config = small_config(tmp_path / str(processes), num_envs=2, steps=12,
                              recipe="recent_replay", env_processes=processes)
        train(config, env_factory=ShortTask)
        rows[processes] = [json.loads(row) for row in
                           (tmp_path / str(processes) / "run/metrics.jsonl").read_text().splitlines()]
    for left, right in zip(rows[False], rows[True]):
        for key in ("transitions", "loss", "action_counts", "reward_events_this_chunk",
                    "replay_used"):
            assert left[key] == right[key], key


def test_batched_prefix_states_equal_individual_reconstruction(tmp_path):
    from disco_spatial.trajectory import Trajectory
    from disco_spatial.training import reconstruct_prefix_state, reconstruct_prefix_states

    config = small_config(tmp_path)
    agent = RecurrentAgent(config, BatchedEnvironment([ShortTask()], config.method))
    params = agent.initial_learner_state(jax.random.PRNGKey(1)).params
    rng = np.random.default_rng(3)

    def episode(n=40):
        types = np.ones(n + 1, np.int8)
        types[0] = 0
        return Trajectory(
            episode_id=f"train:{rng.integers(1 << 30)}", instance_id="p", split="train",
            rgb=rng.integers(0, 256, (n + 1, 64, 64, 3), dtype=np.uint8),
            actions=rng.integers(0, 6, n).astype(np.int32), rewards=np.zeros(n, np.float32),
            discounts=np.ones(n, np.float32), behavior_logits=np.zeros((n, 6), np.float32),
            policy_versions=np.zeros(n, np.int64), step_types=types,
            goal_segments=np.zeros(n + 1, np.int32), terminated=np.zeros(n, bool),
            truncated=np.zeros(n, bool), metadata={})

    episodes = [episode(), episode(), episode()]
    batched = reconstruct_prefix_states(agent, params, episodes, 37)
    for index, trajectory in enumerate(episodes):
        single = reconstruct_prefix_state(agent, params, trajectory, 37)
        for a, b in zip(jax.tree.leaves(batched), jax.tree.leaves(single)):
            np.testing.assert_allclose(a[index:index + 1], b, atol=2e-5)


def test_disco_with_several_environments_trains_end_to_end(tmp_path):
    config = small_config(tmp_path, method="disco", num_envs=2, steps=12)
    result = train(config, env_factory=ShortTask)
    assert result["transitions"] == 12 and result["learner_network_steps"] > 0


def test_batched_replay_rollouts_equal_individual_chunks(tmp_path):
    from disco_spatial.training import replay_rollouts

    config = small_config(tmp_path, num_envs=2)
    envs = BatchedEnvironment([ShortTask(seed=1), ShortTask(seed=2)], config.method)
    agent = RecurrentAgent(config, envs)
    key = jax.random.PRNGKey(0)
    learner = agent.initial_learner_state(key)
    state = agent.initial_actor_state(key)
    collect_rollout(agent, learner.params, state, envs, key, 3, 0)
    chunks = [ReplayChunk(record.trajectory(), 1, 3, 0) for record in envs.records]
    batched, batched_state, _ = replay_rollouts(agent, learner.params, chunks)
    for index, chunk in enumerate(chunks):
        single, single_state, _ = replay_rollout(agent, learner.params, chunk)
        for a, b in zip(jax.tree.leaves(batched), jax.tree.leaves(single)):
            np.testing.assert_allclose(np.asarray(a)[:, index:index + 1], b, atol=2e-5)
        for a, b in zip(jax.tree.leaves(batched_state), jax.tree.leaves(single_state)):
            np.testing.assert_allclose(np.asarray(a)[index:index + 1], b, atol=2e-5)
    with pytest.raises(ValueError, match="share start"):
        replay_rollouts(agent, learner.params, [chunks[0], ReplayChunk(chunks[1].trajectory, 0, 2, 0)])


def test_rollout_keeps_each_steps_previous_action_and_reward(tmp_path):
    from disco_spatial.envs.control import CueRecall

    config = small_config(tmp_path, num_envs=2)
    env = BatchedEnvironment([CueRecall(seed=1), CueRecall(seed=2)], config.method)
    agent = RecurrentAgent(config, env)
    key = jax.random.PRNGKey(0)
    learner = agent.initial_learner_state(key)
    state = agent.initial_actor_state(key)
    rollout = collect_rollout(agent, learner.params, state, env, key, 6, 0)[0]
    observations = rollout.observations
    # Later environment steps must not rewrite earlier observations (CPU buffer aliasing).
    np.testing.assert_array_equal(observations["previous_action"][0], [6, 6])
    np.testing.assert_array_equal(observations["previous_action"][1:], rollout.actions[:-1])
    np.testing.assert_array_equal(observations["previous_reward"][1:], rollout.rewards[1:])
