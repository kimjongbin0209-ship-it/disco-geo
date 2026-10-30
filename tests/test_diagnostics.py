"""P1d diagnostics: positive-control task, credit logging, references and stage tooling."""

from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from disco_spatial.agents import RecurrentAgent
from disco_spatial.config import TrainConfig
from disco_spatial.envs.control import CueRecall, make_cue_recall
from disco_spatial.evaluation.scripted import action_probabilities, scripted_returns
from disco_spatial.methods import reward_proximity
from disco_spatial.training import BatchedEnvironment, collect_rollout, train


ROOT = Path(__file__).resolve().parents[1]


def control_config(tmp_path, **changes):
    settings = dict(output=str(tmp_path / "stage/run"), method="actor_critic",
                    environment="cue_recall", control_delay=1, control_episode_steps=10,
                    steps=32, num_envs=2, rollout_length=4, channels=(2,), feature_dim=4,
                    hidden_size=4, credit_window=2, validation_seeds=(1000000,), eval_every=2,
                    validation_policy="sampled", save_trajectories=False)
    return replace(TrainConfig(), **dict(settings, **changes))


def test_reward_proximity_marks_reward_windows_and_leaves_unknown_tails_out():
    rewards = jnp.asarray([[0], [0], [1], [0], [0], [0]], jnp.float32)
    proximal, distal = reward_proximity(rewards, 2)
    assert proximal[:, 0].tolist() == [False, True, True, False, False, False]
    # t=0 sees rewards[0:2] = no reward; t=5's window runs past the chunk end.
    assert distal[:, 0].tolist() == [True, False, False, True, True, False]


def test_cue_recall_scores_only_the_go_step_and_ends_in_a_time_limit():
    env = CueRecall(seed=3, delay=2, episode_steps=7)
    timestep, rewards, cues = env.reset(), [], []
    while not timestep.last():
        phase = env._t % 3
        cue = env._cue
        action = cue if phase == 2 else (cue + 1) % 6  # wrong actions before the go step
        cues.append(cue)
        timestep = env.step(action)
        rewards.append(timestep.reward)
    assert rewards == [0, 0, 1, 0, 0, 1, 0] and env.goal_segment == 2
    assert timestep.discount == 1.0 and set(timestep.observation) == {"rgb"}
    again = CueRecall(seed=3, delay=2, episode_steps=7)
    again.reset()
    assert again._cue == cues[0]  # the cue sequence is fixed by seed and episode
    reactive = make_cue_recall(9, 1, "validation", 64, delay=0, episode_steps=5)
    reactive.reset()
    assert reactive.step(reactive._cue).reward == 1.0
    bandit = make_cue_recall(9, 1, "train", 64, delay=0, episode_steps=20, fixed_cue=3)
    bandit.reset()
    assert [bandit.step(3).reward for _ in range(5)] == [1.0] * 5
    with pytest.raises(ValueError, match="control_fixed_cue"):
        TrainConfig(environment="cue_recall", control_fixed_cue=6)
    assert reactive.last_sidecar == {} and reactive.observation_spec()["rgb"].shape == (64, 64, 3)


def test_diagnostic_settings_are_validated():
    with pytest.raises(ValueError, match="development diagnostic"):
        TrainConfig(environment="cue_recall", purpose="pilot")
    with pytest.raises(ValueError, match="environment must"):
        TrainConfig(environment="grid_world")
    with pytest.raises(ValueError, match="credit_window"):
        TrainConfig(credit_window=-1)
    with pytest.raises(ValueError, match="control_episode_steps"):
        TrainConfig(environment="cue_recall", control_delay=5, control_episode_steps=5)
    with pytest.raises(ValueError, match="validation_policy"):
        TrainConfig(validation_policy="softmax")


@pytest.mark.parametrize("method", ["actor_critic_aux", "disco"])
def test_credit_diagnostics_are_zero_without_an_update(tmp_path, method):
    config = control_config(tmp_path, method=method, control_delay=0, num_envs=1)
    env = BatchedEnvironment([CueRecall(seed=1, delay=0, episode_steps=10)], method)
    agent = RecurrentAgent(config, env)
    key = jax.random.PRNGKey(0)
    learner = agent.initial_learner_state(key)
    state = agent.initial_actor_state(key)
    _, state, _, key, _ = collect_rollout(agent, learner.params, state, env, key, 3, 0)
    rollout, _, start, _, _ = collect_rollout(agent, learner.params, state, env, key, 4, 0)
    assert any(float(jnp.abs(leaf).max()) > 0 for leaf in jax.tree.leaves(start))
    shape = rollout.actions[:-1].shape
    signal = jnp.zeros(shape + (6,)) if method == "disco" else jnp.ones(shape)
    sums = agent.credit_diagnostics(learner.params, start, rollout, signal, 2)
    # The same weights from the rollout's own start state reproduce the behaviour logits.
    for side in ("proximal", "distal"):
        np.testing.assert_allclose(sums[f"update_logp_gain_{side}"], 0, atol=1e-5)
    np.testing.assert_allclose(sums["update_policy_kl"], 0, atol=1e-6)
    if method != "disco":
        np.testing.assert_allclose(sums["rule_signal_proximal"], sums["proximal_steps"])


@pytest.mark.parametrize("method", ["actor_critic_aux", "disco"])
def test_positive_control_trains_with_credit_logs_and_sampled_validation(tmp_path, method):
    config = control_config(tmp_path, method=method)
    result = train(config)
    rows = [json.loads(row) for row in (tmp_path / "stage/run/metrics.jsonl").read_text().splitlines()]
    assert result["transitions"] == 32
    provenance = json.loads((tmp_path / "stage/run/provenance.json").read_text())
    assert provenance["environment"]["environment"] == "cue_recall_positive_control"
    assert all(row["credit_window"] == 2 and row["update_policy_kl"] >= -1e-6 for row in rows)
    assert sum(row["credit_proximal_steps"] + row["credit_distal_steps"] for row in rows) > 0
    assert "rule_entropy" in rows[0]
    if method == "disco":
        assert "rule_pi_hat_entropy" in rows[0] and rows[0]["credit_rule_signal"] == "target_logp_gain"
    validations = [row["validation"] for row in rows if "validation" in row]
    assert validations and all(v["policy"] == "sampled" and v["lengths"] == [10] for v in validations)


def test_scripted_reference_policies():
    np.testing.assert_allclose(action_probabilities("uniform"), np.full(6, 1 / 6))
    biased = action_probabilities("forward_left:0.5")
    assert biased[4] == pytest.approx(0.5 + 0.5 / 6) and biased.sum() == pytest.approx(1)
    with pytest.raises(ValueError):
        action_probabilities("teleport:0.5")
    factory = lambda size, seed, split, resolution: CueRecall(seed, split, resolution, 0, 60)  # noqa: E731
    result = scripted_returns(factory, [2000000, 2000001], "uniform")
    assert result["total_steps"] == 120 and 0 < result["total_reward"] < 60


def test_diagnostic_stage_expands_and_summarizes(tmp_path):
    env = dict(os.environ, PYTHONPATH=str(ROOT))
    expanded = subprocess.run(
        [sys.executable, str(ROOT / "scripts/expand_conditions.py"), "--stage", "P1d_diagnostics",
         "--output", str(tmp_path / "configs")], env=env, capture_output=True, text=True, check=True)
    names = sorted(Path(p).stem for p in json.loads(expanded.stdout)["configs"])
    assert len(names) == 45 and "disco-diag_entropy_0-seed0" not in names
    assert "actor_critic_aux-diag_lr_1e-2-seed0" not in names
    config = json.loads((tmp_path / "configs/disco-control_memory-seed1.json").read_text())
    assert (config["control_delay"], config["steps"], config["credit_window"]) == (8, 2000000, 1)
    assert config["output"] == "runs/diagnostics/disco-control_memory-seed1"
    all_stage = subprocess.run(
        [sys.executable, str(ROOT / "scripts/expand_conditions.py"), "--stage", "P2_pilot_matrix",
         "--output", str(tmp_path / "p2")], env=env, capture_output=True, text=True, check=True)
    p2 = [Path(p).stem for p in json.loads(all_stage.stdout)["configs"]]
    assert len(p2) == 80 and not any("diag" in p or "control" in p for p in p2)

    train(control_config(tmp_path))
    subprocess.run([sys.executable, str(ROOT / "scripts/diagnose_stage.py"), str(tmp_path / "stage")],
                   env=env, cwd=tmp_path, capture_output=True, text=True, check=True)
    summary = json.loads((tmp_path / "runs/evaluation/stage/diagnostics.json").read_text())
    run = summary["runs"][0]
    assert run["environment"] == "cue_recall" and run["reward"]["criterion"] == 0.5
    assert run["credit"]["window"] == 2 and run["credit"]["rule_signal_kind"] == "advantage"
