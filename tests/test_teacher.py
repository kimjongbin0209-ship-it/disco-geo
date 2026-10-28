"""Self-teacher: shared initialization, stop-gradient backbone, Bayes-posterior target."""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from disco_spatial import teacher as teacher_lib
from disco_spatial.agents import RecurrentAgent
from test_recurrent import SpecEnvironment, assert_tree_close, scripted_rollout


def agent(method="actor_critic", teacher="outcome_memory"):
    config = SimpleNamespace(method=method, channels=(4,), feature_dim=8, hidden_size=8,
                             batch_size=2, learning_rate=3e-4, teacher=teacher, teacher_hidden=16)
    return RecurrentAgent(config, SpecEnvironment())


def feedback_batch(student, rollout, start, outcome):
    length, batch = rollout.actions.shape
    tree = {
        "outcome": jnp.asarray(outcome, jnp.float32),
        "memory_rgb": jnp.zeros((length, batch, 8, 8, 3), jnp.uint8),
        "memory_valid": jnp.zeros((length, batch), bool),
        "memory_elapsed": jnp.zeros((length, batch)),
    }
    known = jnp.ones((length - 1, batch), bool)
    return {"rollout": rollout, "start_state": start,
            "behavior_logits": rollout.logits[:-1], "train": tree, "query": tree,
            "shuffled": tree, "known": known, "success": tree["outcome"][:-1] > 0.5,
            "distill_mask": known, "returns": jnp.zeros((length - 1, batch))}


def test_teacher_head_leaves_backbone_initialization_unchanged():
    for method in ("actor_critic", "disco"):
        plain = agent(method, "none").initial_learner_state(jax.random.PRNGKey(5))
        with_teacher = agent(method).initial_learner_state(jax.random.PRNGKey(5))
        head = teacher_lib.teacher_params(with_teacher.params)
        assert head and all(name.startswith("teacher_") for name in head)
        assert_tree_close(plain.params, teacher_lib.backbone_params(with_teacher.params), atol=0)


def test_initial_teacher_equals_base_and_backbone_gets_no_teacher_gradient():
    student = agent()
    key = jax.random.PRNGKey(0)
    learner = student.initial_learner_state(key)
    start = student.initial_actor_state(key)
    rollout = scripted_rollout(student, learner.params, start)
    batch = feedback_batch(student, rollout, start, np.ones((4, 2)))
    head = teacher_lib.teacher_params(learner.params)

    def objective(params, distill):
        return student.feedback_objective(params, head, batch, distill, 1.0)

    loss, logs = objective(learner.params, 1.0)
    assert float(logs["distillation_kl"]) < 1e-7  # zero-initialized delta
    assert abs(float(logs["teacher_information_gain_success"])) < 1e-6
    gradient = jax.grad(lambda p: objective(p, 0.0)[0])(learner.params)
    for name, values in teacher_lib.backbone_params(gradient).items():
        for leaf in jax.tree.leaves(values):
            np.testing.assert_array_equal(leaf, 0, err_msg=name)
    assert any(np.abs(leaf).max() > 0 for leaf in jax.tree.leaves(teacher_lib.teacher_params(gradient)))


@pytest.mark.parametrize("hidden", [32, 0])
def test_trained_delta_recovers_success_posterior_over_actions(hidden):
    """Behavior pi uniform; P(success | a) = p_a. Teacher(o=1) must approach pi * p."""
    transform, _ = teacher_lib.teacher_network(
        SimpleNamespace(teacher="outcome", teacher_hidden=hidden), num_actions=3)
    rng = np.random.default_rng(0)
    p = np.array([0.8, 0.2, 0.05])
    actions = rng.integers(3, size=4096)
    outcome = (rng.random(4096) < p[actions]).astype(np.float32)
    belief = jnp.ones((1, 4096, 4))
    feedback = {"outcome": jnp.asarray(outcome)[None]}
    params = transform.init(jax.random.PRNGKey(0), belief, feedback)
    optimizer = optax.adam(3e-3)
    state = optimizer.init(params)

    @jax.jit
    def step(params, state):
        def nll(params):
            logp = jax.nn.log_softmax(transform.apply(params, belief, feedback)[0])
            return -jnp.mean(logp[jnp.arange(4096), actions])
        grads = jax.grad(nll)(params)
        updates, state = optimizer.update(grads, state)
        return optax.apply_updates(params, updates), state

    for _ in range(600):
        params, state = step(params, state)
    query = transform.apply(params, belief[:, :1], {"outcome": jnp.ones((1, 1))})[0, 0]
    expected = p / p.sum()
    np.testing.assert_allclose(np.asarray(jax.nn.softmax(query)), expected, atol=0.04)


def test_composite_without_auxiliary_batches_is_exactly_the_mixed_update():
    student = agent("disco", "none")
    key = jax.random.PRNGKey(3)
    learner = student.initial_learner_state(key)
    start = student.initial_actor_state(key)
    rollout = scripted_rollout(student, learner.params, start)
    rule = student.rule_params()
    mixed = jax.jit(student.learner_step_mixed, static_argnames=("replay_weight",))(
        key, rollout, None, learner, start, None, rule, replay_weight=0.0)
    composite = jax.jit(student.learner_step_composite, static_argnames=("replay_weight",))(
        key, rollout, None, learner, start, None, rule, replay_weight=0.0)
    for expected, actual in zip(jax.tree.leaves(mixed), jax.tree.leaves(composite)):
        np.testing.assert_array_equal(expected, actual)


def test_auxiliary_objectives_share_one_step_and_leave_meta_state_fresh_only():
    student = agent("disco")
    key = jax.random.PRNGKey(3)
    learner = student.initial_learner_state(key)
    start = student.initial_actor_state(key)
    rollout = scripted_rollout(student, learner.params, start)
    rule = student.rule_params()
    plain = jax.jit(student.learner_step_mixed, static_argnames=("replay_weight",))(
        key, rollout, None, learner, start, None, rule, replay_weight=0.0)[0]
    weights = {"self_imitation": jnp.float32(1.0), "distillation": jnp.float32(1.0),
               "teacher": jnp.float32(1.0)}
    new, _, logs = jax.jit(student.learner_step_composite, static_argnames=("replay_weight",))(
        key, rollout, None, learner, start, None, rule, replay_weight=0.0,
        self_imitation={"rollout": rollout, "start_state": start},
        feedback=feedback_batch(student, rollout, start, np.ones((4, 2))),
        teacher_ema=teacher_lib.teacher_params(learner.params), weights=weights)
    for expected, actual in zip(jax.tree.leaves(plain.meta_state["rnn_state"]),
                                jax.tree.leaves(new.meta_state["rnn_state"])):
        np.testing.assert_array_equal(expected, actual)
    np.testing.assert_array_equal(new.opt_state[0].count, plain.opt_state[0].count)
    assert np.isfinite(float(logs["self_imitation_loss"]))
    assert np.isfinite(float(logs["teacher_nll"]))
    assert float(logs["auxiliary_gradient_norm"]) > 0
    head_before = teacher_lib.teacher_params(learner.params)
    head_after = teacher_lib.teacher_params(new.params)
    assert any(not np.array_equal(a, b) for a, b in
               zip(jax.tree.leaves(head_before), jax.tree.leaves(head_after)))
