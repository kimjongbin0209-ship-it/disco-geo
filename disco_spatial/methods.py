"""Functional loss components shared by self-imitation and feedback distillation.

Distillation is opt-in through an explicit static ``teacher_ready`` flag. The
feedback teacher itself lives in ``teacher.py``; whether its targets are used
is decided by logged runtime evidence in the training loop, not here.
"""

import jax
import jax.numpy as jnp


def masked_mean(values, valid):
    weights = jnp.asarray(valid, dtype=values.dtype)
    return jnp.sum(jnp.where(weights > 0, values, 0) * weights) / jnp.maximum(weights.sum(), 1)


_masked_mean = masked_mean


def reverse_kl_distillation(student_logits, teacher_logits, valid, *, teacher_ready=False):
    """Mean KL(student || stopgrad(teacher)); full discrete action distribution.

    Passing an unvalidated/random feedback branch as a self-teacher is a caller
    error. ``teacher_ready`` is a static Python gate, deliberately not inferred
    from training steps or a teacher loss. No external teacher is constructed.
    """
    if teacher_ready is not True:
        raise ValueError("A validated feedback teacher is required before enabling distillation")
    if student_logits.shape != teacher_logits.shape or student_logits.shape[:-1] != valid.shape:
        raise ValueError("Student/teacher logits and valid mask must have matching leading shapes")
    student_logp = jax.nn.log_softmax(student_logits, axis=-1)
    teacher_logp = jax.lax.stop_gradient(jax.nn.log_softmax(teacher_logits, axis=-1))
    divergence = jnp.sum(jnp.exp(student_logp) * (student_logp - teacher_logp), axis=-1)
    return _masked_mean(divergence, valid)


def reward_proximity(rewards, window):
    """[T, B] masks: steps followed by a reward within ``window`` steps, and steps not.

    ``rewards[t]`` is the reward after action t. A step is proximal when any of
    rewards[t : t + window] is positive, distal when that whole window lies
    inside the chunk without a reward. Windows that run past the chunk end with
    no reward have an unknown outcome and are in neither mask.
    """
    t = rewards.shape[0]
    cumulative = jnp.concatenate((jnp.zeros_like(rewards[:1]), jnp.cumsum(rewards > 0, axis=0)))
    ends = jnp.minimum(jnp.arange(t) + window, t)
    counts = cumulative[ends] - cumulative[:t]
    complete = (jnp.arange(t) + window <= t)[:, None]
    return counts > 0, (counts == 0) & complete


def self_imitation_loss(logits, actions, returns, baseline, valid):
    """Positive-return-advantage behavior cloning; distinct from SDPO/replay RL.

    Returns are observed discounted returns. This intentionally changes the
    objective and does not claim an unbiased off-policy policy-gradient update.
    """
    if logits.shape[:-1] != actions.shape:
        raise ValueError("Actions must match logits' leading dimensions")
    if any(value.shape != actions.shape for value in (returns, baseline, valid)):
        raise ValueError("Returns, baseline and mask must match actions")
    advantage = jax.lax.stop_gradient(jnp.maximum(returns - baseline, 0))
    logp = jnp.take_along_axis(jax.nn.log_softmax(logits), actions[..., None], axis=-1)[..., 0]
    return _masked_mean(-advantage * logp, valid)
