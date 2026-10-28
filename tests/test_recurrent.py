"""Temporal semantics and a real released-rule update, independent of a benchmark."""

from types import SimpleNamespace

import dm_env
from dm_env import specs
import jax
import jax.numpy as jnp
import numpy as np

from disco_spatial.agents import RecurrentAgent
from disco_rl import types


class SpecEnvironment:
    def single_observation_spec(self):
        return {
            "rgb": specs.Array((8, 8, 3), np.uint8),
            "previous_action": specs.Array((), np.int32),
            "previous_reward": specs.Array((), np.float32),
        }

    def single_action_spec(self):
        return specs.BoundedArray((), np.int32, 0, 5)


def small_agent(method="actor_critic"):
    config = SimpleNamespace(
        method=method, channels=(4,), feature_dim=8, hidden_size=8,
        batch_size=2, learning_rate=3e-4,
    )
    return RecurrentAgent(config, SpecEnvironment())


def observations(length=4, batch_size=2):
    rng = np.random.default_rng(42)
    first = np.zeros((length, batch_size), dtype=bool)
    first[0] = True
    first[2, 0] = True
    return {
        "rgb": jnp.array(rng.integers(0, 256, (length, batch_size, 8, 8, 3), dtype=np.uint8)),
        "previous_action": jnp.array(rng.integers(0, 7, (length, batch_size)), jnp.int32),
        "previous_reward": jnp.array(rng.normal(size=(length, batch_size)), jnp.float32),
        "is_first": jnp.array(first),
    }


def assert_tree_close(first, second, atol=2e-6):
    for a, b in zip(jax.tree.leaves(first), jax.tree.leaves(second)):
        np.testing.assert_allclose(a, b, atol=atol, rtol=2e-6)


def test_online_and_scan_match_with_midsequence_first_and_authoritative_masks():
    agent = small_agent()
    learner = agent.initial_learner_state(jax.random.PRNGKey(0))
    start = agent.initial_actor_state(jax.random.PRNGKey(1))
    for value in jax.tree.leaves(start):
        np.testing.assert_array_equal(value, np.zeros_like(value))
    obs = observations()
    state, online = start, []
    for t in range(4):
        item = jax.tree.map(lambda x: x[t], obs)
        out, state = agent.policy_network.one_step(learner.params, state, item, item["is_first"])
        online.append(out)
    expected = jax.tree.map(lambda *x: jnp.stack(x), *online)
    # Deliberately wrong supplied masks must not override saved FIRST flags.
    actual, final_state = agent.policy_network.unroll(
        learner.params, start, obs, jnp.ones_like(obs["is_first"])
    )
    assert_tree_close(expected, actual)
    assert_tree_close(state, final_state)
    assert actual["logits"].shape == (4, 2, 6)


def test_first_resets_hidden_and_previous_action_reward_without_terminal_guessing():
    agent = small_agent()
    learner = agent.initial_learner_state(jax.random.PRNGKey(0))
    start = agent.initial_actor_state(jax.random.PRNGKey(1))
    poisoned = jax.tree.map(lambda x: jnp.ones_like(x) * 100, start)
    obs = jax.tree.map(lambda x: x[0], observations())
    out, state = agent.policy_network.one_step(learner.params, poisoned, obs, None)
    clean_obs = dict(
        obs, previous_action=jnp.full((2,), 6, jnp.int32), previous_reward=jnp.zeros((2,))
    )
    expected, expected_state = agent.policy_network.one_step(learner.params, start, clean_obs, None)
    assert_tree_close(out, expected)
    assert_tree_close(state, expected_state)
    # Without FIRST, an older observation must affect the present belief.
    continuing = dict(clean_obs, is_first=jnp.zeros((2,), dtype=bool))
    continued, _ = agent.policy_network.one_step(learner.params, poisoned, continuing, None)
    assert not np.allclose(out["belief"], continued["belief"])


def test_last_preserves_truncation_bootstrap_discount_and_first_is_independent():
    agent = small_agent()
    key = jax.random.PRNGKey(0)
    learner = agent.initial_learner_state(key)
    start = agent.initial_actor_state(key)
    item = {k: v[0] for k, v in observations().items() if k != "is_first"}
    timestep = SimpleNamespace(
        observation=item,
        step_type=jnp.full((2,), dm_env.StepType.LAST, jnp.int32),
        reward=jnp.zeros((2,), jnp.float32),
        discount=jnp.array([0.0, 1.0], jnp.float32),
    )
    actor, _ = agent.actor_step(learner.params, key, timestep, start)
    np.testing.assert_array_equal(actor.discounts, [0.0, 1.0])
    np.testing.assert_array_equal(actor.observations["is_first"], [False, False])
    timestep.step_type = jnp.full((2,), dm_env.StepType.FIRST, jnp.int32)
    actor, _ = agent.actor_step(learner.params, key, timestep, start)
    np.testing.assert_array_equal(actor.discounts, [1.0, 1.0])
    np.testing.assert_array_equal(actor.observations["is_first"], [True, True])


def test_scan_propagates_credit_through_history_and_stops_at_first():
    agent = small_agent()
    learner = agent.initial_learner_state(jax.random.PRNGKey(0))
    start = agent.initial_actor_state(jax.random.PRNGKey(1))
    obs = observations()
    obs["is_first"] = jnp.array([[True, True], [False, False], [False, False], [False, False]])

    def objective(rgb, first):
        out, _ = agent.policy_network.unroll(learner.params, start, dict(obs, rgb=rgb, is_first=first), None)
        return out["belief"][-1, 0].sum()

    pixels = obs["rgb"].astype(jnp.float32)
    gradient = jax.grad(objective)(pixels, obs["is_first"])
    assert np.linalg.norm(np.asarray(gradient[0, 0])) > 1e-9
    first = obs["is_first"].at[2, 0].set(True)
    cut = jax.grad(objective)(pixels, first)
    np.testing.assert_array_equal(np.asarray(cut[:2, 0]), 0)
    assert np.linalg.norm(np.asarray(cut[2:, 0])) > 1e-9


def scripted_rollout(agent, params, start):
    obs = observations()
    # Terminal RGB is followed by a separate FIRST, matching dm_env semantics.
    step_types = [dm_env.StepType.FIRST, dm_env.StepType.MID, dm_env.StepType.LAST, dm_env.StepType.FIRST]
    obs["is_first"] = jnp.array([[True, True], [False, False], [False, False], [True, True]])
    rewards = [0.0, -0.01, 1.0, 0.0]
    state, rows = start, []
    for t, step_type in enumerate(step_types):
        item = {key: value[t] for key, value in obs.items() if key != "is_first"}
        timestep = types.EnvironmentTimestep(
            observation=item, step_type=jnp.full((2,), step_type, jnp.int32),
            reward=jnp.full((2,), rewards[t], jnp.float32),
        )
        row, state = agent.actor_step(params, jax.random.PRNGKey(t + 10), timestep, state)
        rows.append(row)
    stacked = jax.tree.map(lambda *items: jnp.stack(items), *rows)
    return types.ActorRollout.from_timestep(stacked)


def test_released_disco_update_is_finite_changes_cnn_and_preserves_rule():
    agent = small_agent("disco")
    key = jax.random.PRNGKey(3)
    learner = agent.initial_learner_state(key)
    start = agent.initial_actor_state(key)
    rollout = scripted_rollout(agent, learner.params, start)
    output, _ = agent.unroll_net(learner.params, start, rollout)
    assert output["y"].shape == (4, 2, 600)
    assert output["z"].shape == (4, 2, 6, 600)
    assert output["q"].shape == (4, 2, 6, 601)
    assert output["aux_pi"].shape == (4, 2, 6, 6)
    # Recomputed online probabilities match recorded behavior on the fresh data.
    np.testing.assert_allclose(output["logits"], rollout.logits, atol=2e-6)
    rule = agent.rule_params()
    before = jax.tree.map(lambda x: np.array(x, copy=True), rule)
    update = jax.jit(agent.learner_step, static_argnames=("is_meta_training",))
    new, _, logs = update(key, rollout, learner, start, rule, False)
    assert np.isfinite(logs["total_loss"])
    assert all(np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves(new.params))
    changes = [
        np.max(np.abs(np.asarray(new.params[name]["w"] - values["w"])))
        for name, values in learner.params.items()
        if name.startswith("visual_encoder/") and "w" in values
    ]
    assert changes and max(changes) > 0
    for old, current in zip(jax.tree.leaves(before), jax.tree.leaves(rule)):
        np.testing.assert_array_equal(old, current)


def test_base_rules_share_recurrent_backbone_initialization_and_aux_update_runs():
    encoders = []
    for method in ("actor_critic", "actor_critic_aux", "disco"):
        agent = small_agent(method)
        key = jax.random.PRNGKey(7)
        learner = agent.initial_learner_state(key)
        backbone = {k: v for k, v in learner.params.items()
                    if k.startswith(("visual_encoder/", "memory", "policy_torso/"))}
        assert any(name.startswith("memory") for name in backbone)
        encoders.append(backbone)
        if method == "actor_critic_aux":
            start = agent.initial_actor_state(key)
            rollout = scripted_rollout(agent, learner.params, start)
            new, _, logs = jax.jit(agent.learner_step, static_argnames=("is_meta_training",))(
                key, rollout, learner, start, agent.rule_params(), False
            )
            assert np.isfinite(logs["total_loss"])
            assert new.params.keys() == learner.params.keys()
    for encoder in encoders[1:]:
        assert_tree_close(encoders[0], encoder, atol=0)


def test_zero_replay_weight_is_exact_released_update():
    agent = small_agent("actor_critic")
    key = jax.random.PRNGKey(3)
    learner = agent.initial_learner_state(key)
    start = agent.initial_actor_state(key)
    rollout = scripted_rollout(agent, learner.params, start)
    rule = agent.rule_params()
    original = jax.jit(agent.learner_step, static_argnames=("is_meta_training",))(
        key, rollout, learner, start, rule, False
    )
    mixed = jax.jit(agent.learner_step_mixed, static_argnames=("replay_weight",))(
        key, rollout, None, learner, start, None, rule, replay_weight=0.0
    )
    for expected, actual in zip(jax.tree.leaves(original), jax.tree.leaves(mixed)):
        np.testing.assert_array_equal(expected, actual)


def test_mixed_disco_gradient_uses_one_update_and_fresh_meta_state_only():
    agent = small_agent("disco")
    key = jax.random.PRNGKey(3)
    learner = agent.initial_learner_state(key)
    start = agent.initial_actor_state(key)
    fresh = scripted_rollout(agent, learner.params, start)
    # Different T and B, with older behavior probabilities and changed reward.
    replay = jax.tree.map(lambda x: x[:3, :1], fresh)
    replay.logits = replay.logits + jnp.array([1.0, -1.0, 0.0, 0.0, 0.0, 0.0])
    replay.agent_outs = dict(replay.agent_outs, logits=replay.logits)
    replay.rewards = replay.rewards.at[2, 0].set(0.0)
    replay_start = agent.initial_actor_state(key, batch_size=1)
    rule = agent.rule_params()
    expected_fresh, _, _ = jax.jit(agent.learner_step, static_argnames=("is_meta_training",))(
        key, fresh, learner, start, rule, False
    )
    mixed, _, logs = jax.jit(agent.learner_step_mixed, static_argnames=("replay_weight",))(
        key, fresh, replay, learner, start, replay_start, rule, replay_weight=0.25
    )
    # Exact fresh-only lifetime RNN/EMA/target-state updates despite replay.
    for expected, actual in zip(
        jax.tree.leaves(expected_fresh.meta_state), jax.tree.leaves(mixed.meta_state)
    ):
        np.testing.assert_array_equal(expected, actual)
    assert np.isfinite(logs["fresh_loss"])
    assert np.isfinite(logs["replay_loss"])
    assert all(np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves(mixed.params))
    assert any(
        not np.array_equal(a, b)
        for a, b in zip(jax.tree.leaves(expected_fresh.params), jax.tree.leaves(mixed.params))
    )
    # The Adam step counter advances once, just as in one fresh update.
    np.testing.assert_array_equal(mixed.opt_state[0].count, expected_fresh.opt_state[0].count)


def test_explicit_ema_state_only_changes_target_forward_and_first_clears_both_states():
    agent = small_agent("disco")
    key = jax.random.PRNGKey(3)
    learner = agent.initial_learner_state(key)
    start = agent.initial_actor_state(key)
    rollout = scripted_rollout(agent, learner.params, start)
    # A chunk beginning inside an episode must use its separately reconstructed
    # EMA prefix, rather than assume online and target hidden states coincide.
    rollout.observations = dict(
        rollout.observations,
        is_first=rollout.observations["is_first"].at[0].set(False),
    )
    target_start = jax.tree.map(lambda x: jnp.ones_like(x), start)
    rule = agent.rule_params()
    branch = jax.jit(agent._rollout_gradient)
    _, _, online_final, _, legacy_meta = branch(key, rollout, learner, start, rule, start)
    _, _, explicit_online_final, _, explicit_meta = branch(
        key, rollout, learner, start, rule, target_start
    )
    expected, _ = agent.policy_network.unroll(
        learner.meta_state["target_params"], target_start,
        rollout.observations, rollout.observations["is_first"],
    )
    assert_tree_close(explicit_meta["target_out"], expected)
    assert not np.allclose(legacy_meta["target_out"]["q"], explicit_meta["target_out"]["q"])
    # Only target propagation changes; the student's final recurrent state does
    # not consume the target memory or any future/privileged observation.
    assert_tree_close(online_final, explicit_online_final, atol=0)
    rollout.observations = dict(
        rollout.observations,
        is_first=rollout.observations["is_first"].at[0].set(True),
    )
    _, _, _, _, reset_zero = branch(key, rollout, learner, start, rule, start)
    _, _, _, _, reset_one = branch(key, rollout, learner, start, rule, target_start)
    assert_tree_close(reset_zero["target_out"], reset_one["target_out"], atol=0)


def test_target_state_api_none_keeps_baseline_and_zero_mixture_delegates_explicit_state():
    agent = small_agent("disco")
    key = jax.random.PRNGKey(4)
    learner = agent.initial_learner_state(key)
    start = agent.initial_actor_state(key)
    rollout = scripted_rollout(agent, learner.params, start)
    rule = agent.rule_params()
    official = jax.jit(agent.learner_step, static_argnames=("is_meta_training",))(
        key, rollout, learner, start, rule, False
    )
    target_step = jax.jit(agent.learner_step_targets)
    absent = target_step(key, rollout, learner, start, None, rule)
    for expected, actual in zip(jax.tree.leaves(official), jax.tree.leaves(absent)):
        np.testing.assert_array_equal(expected, actual)
    rollout.observations = dict(
        rollout.observations,
        is_first=rollout.observations["is_first"].at[0].set(False),
    )
    target_start = jax.tree.map(lambda x: jnp.ones_like(x), start)
    explicit = target_step(key, rollout, learner, start, target_start, rule)
    mixed = jax.jit(agent.learner_step_mixed, static_argnames=("replay_weight",))(
        key, rollout, None, learner, start, None, rule,
        replay_weight=0.0, fresh_target_start_state=target_start,
    )
    for expected, actual in zip(jax.tree.leaves(explicit), jax.tree.leaves(mixed)):
        np.testing.assert_array_equal(expected, actual)
