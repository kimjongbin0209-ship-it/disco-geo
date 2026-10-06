import jax
import numpy as np
import dm_env
from disco_geo.config import TrainConfig
from disco_geo.agents.disco import GeoAgent
from disco_geo.models.encoder import feature_transform
from disco_geo.training.runner import make_env, make_collector


def small_config(dataset, method="disco", **kw):
    return TrainConfig(
        dataset=str(dataset),
        method=method,
        channels=(8, 16),
        feature_dim=16,
        batch_size=2,
        rollout_length=3,
        steps=8,
        **kw,
    )


def test_sparse_reward_and_terminal_reset(dataset):
    config = small_config(dataset)
    _, env = make_env(config)
    state, ts = env.reset(jax.random.PRNGKey(2))
    assert set(ts.observation) == {"rgb", "reference_rgb"}
    state = state._replace(reference_scene=state.scene)
    end, ts = env.step(state, np.array([2, 3], np.int32))
    np.testing.assert_array_equal(ts.reward, [1, 0])
    assert np.all(np.asarray(ts.step_type) == dm_env.StepType.LAST)
    fresh, ts = env.step(end, np.array([0, 0], np.int32))
    assert np.all(np.asarray(ts.step_type) == dm_env.StepType.FIRST)
    assert not np.any(np.asarray(fresh.done))
    np.testing.assert_array_equal(ts.reward, [0, 0])


def test_native_disco_outputs_gradients_and_frozen_rule(dataset):
    config = small_config(dataset)
    _, env = make_env(config)
    agent = GeoAgent(config, env)
    key = jax.random.PRNGKey(0)
    learner = agent.initial_learner_state(key)
    actor_state = agent.initial_actor_state(key)
    state, ts = env.reset(key)
    rule = agent.rule_params()
    rule_before = jax.tree.map(lambda x: np.array(x, copy=True), rule)
    out, _ = agent._network.one_step(learner.params, actor_state, ts.observation, state.done)
    assert out["y"].shape == (2, 600)
    assert out["z"].shape == (2, 4, 600)
    assert out["q"].shape == (2, 4, 601)
    assert out["aux_pi"].shape == (2, 4, 4)
    rollout, actor_state, *_ = make_collector(agent, env, 3)(
        learner.params, actor_state, state, ts, key
    )
    step = jax.jit(agent.learner_step, static_argnames=("is_meta_training",))
    new, _, logs = step(key, rollout, learner, actor_state, rule, False)
    assert np.isfinite(logs["total_loss"])
    deltas = [
        np.linalg.norm(np.asarray(new.params[k]["w"] - v["w"]))
        for k, v in learner.params.items()
        if k.startswith("visual_encoder/")
    ]
    assert deltas and max(deltas) > 0
    for a, b in zip(jax.tree.leaves(rule), jax.tree.leaves(rule_before)):
        np.testing.assert_array_equal(a, b)
    features = {k: v for k, v in new.params.items() if k.startswith("visual_encoder/")}
    result = feature_transform(config).apply(features, ts.observation["rgb"])
    assert result.shape == (2, 16)


def test_matched_encoder_initialization_and_aux_baseline(dataset):
    _, env = make_env(small_config(dataset))
    encoders = []
    for method in ("disco", "actor_critic", "actor_critic_aux"):
        agent = GeoAgent(small_config(dataset, method), env)
        state = agent.initial_learner_state(jax.random.PRNGKey(10))
        encoders.append({k: v for k, v in state.params.items() if k.startswith("visual_encoder/")})
        if method == "actor_critic_aux":
            env_state, ts = env.reset(jax.random.PRNGKey(10))
            actor_state = agent.initial_actor_state(jax.random.PRNGKey(10))
            rollout, actor_state, *_ = make_collector(agent, env, 3)(
                state.params, actor_state, env_state, ts, jax.random.PRNGKey(10)
            )
            new, _, metrics = jax.jit(agent.learner_step, static_argnames=("is_meta_training",))(
                jax.random.PRNGKey(11), rollout, state, actor_state, agent.rule_params(), False
            )
            assert np.isfinite(metrics["total_loss"])
    for other in encoders[1:]:
        for a, b in zip(jax.tree.leaves(encoders[0]), jax.tree.leaves(other)):
            np.testing.assert_array_equal(a, b)
