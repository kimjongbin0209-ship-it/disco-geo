"""Feedback-conditioned self-teacher (SDPO-style) for a recurrent RGB student.

SDPO uses the current model, with feedback in its context, as the teacher and
distills KL(student || stopgrad(teacher)) on the student's own attempt. A
scratch CNN/GRU cannot read feedback in context, so the feedback pathway is an
explicit head on the shared, stop-gradient belief:

    teacher_logits = base_logits + delta(belief, feedback)

``delta`` is trained by behavior cloning on observed outcomes with the
behavior logits as base. By Bayes' rule its optimum is
log P(feedback | history, action) up to a per-step constant, so the teacher is
the posterior over actions given the feedback. Two consequences shape the design:

* On-policy actions carry no information about causal feedback beyond the
  history, so a teacher without an outcome channel learns delta = 0. Every
  teacher here therefore conditions on the outcome.
* Querying with outcome = success gives pi(a) P(success | h, a) / P(success | h),
  a soft policy-improvement target. On successful steps its expectation equals
  that of success-conditioned imitation (self-imitation), so the teacher is the
  Rao-Blackwellized form; any further benefit must come from generalization.

The distillation target uses an exponential moving average of the delta head,
as in SDPO's EMA-regularized teacher. The encoder used for remembered frames
is the student's own encoder, behind a stop-gradient.
"""

import haiku as hk
import jax
import jax.numpy as jnp

from disco_spatial.models import RGBEncoder, setting


TEACHER_PREFIX = "teacher_"
CHANNELS = {"outcome": ("outcome",), "outcome_memory": ("outcome", "success_memory")}


def teacher_params(params):
    return {key: value for key, value in params.items() if key.startswith(TEACHER_PREFIX)}


def backbone_params(params):
    return {key: value for key, value in params.items() if not key.startswith(TEACHER_PREFIX)}


def teacher_network(config, num_actions):
    """Return a Haiku transform for delta logits; [T, B] leading axes."""
    variant = setting(config, "teacher", "none")
    if variant not in CHANNELS:
        raise ValueError(f"Unknown teacher feedback variant: {variant}")
    channels = CHANNELS[variant]
    hidden = setting(config, "teacher_hidden", 128)  # 0 selects the linear head

    def forward(belief, feedback):
        parts = [belief, feedback["outcome"][..., None]]
        if "success_memory" in channels:
            frames = feedback["memory_rgb"]
            leading = frames.shape[:2]
            # Same module name, hence the student's own encoder weights.
            encoded = RGBEncoder(setting(config, "channels", (32, 64, 64)),
                                 setting(config, "feature_dim", 128))(
                frames.reshape((-1,) + frames.shape[2:])).reshape(leading + (-1,))
            valid = feedback["memory_valid"].astype(jnp.float32)[..., None]
            parts += [encoded * valid, valid, feedback["memory_elapsed"][..., None]]
        x = jnp.concatenate(parts, axis=-1)
        if hidden:
            x = jax.nn.silu(hk.Linear(hidden, name="teacher_delta_0")(x))
            x = jax.nn.silu(hk.Linear(hidden, name="teacher_delta_1")(x))
        else:
            # Low-capacity variant: delta is linear in [belief, feedback] plus their
            # outcome interaction, so feedback can still change the action ranking.
            x = jnp.concatenate((x, x * feedback["outcome"][..., None]), axis=-1)
        # Zero output layer: the initial teacher equals its base distribution.
        return hk.Linear(num_actions, w_init=jnp.zeros, name="teacher_delta_out")(x)

    return hk.without_apply_rng(hk.transform(forward)), channels


def dummy_feedback(belief_shape, rgb_shape):
    leading = belief_shape[:2]
    return {
        "outcome": jnp.zeros(leading),
        "memory_rgb": jnp.zeros(leading + tuple(rgb_shape), jnp.uint8),
        "memory_valid": jnp.zeros(leading, bool),
        "memory_elapsed": jnp.zeros(leading),
    }


def init_teacher(transform, rng, hidden_size, rgb_shape):
    belief = jnp.zeros((1, 1, hidden_size))
    return teacher_params(transform.init(rng, belief, dummy_feedback(belief.shape, rgb_shape)))


def delta_logits(transform, params, delta_head, belief, feedback):
    """Delta logits with a stop-gradient backbone; only ``delta_head`` is trainable."""
    frozen = jax.lax.stop_gradient(backbone_params(params))
    return transform.apply(dict(frozen, **delta_head), jax.lax.stop_gradient(belief), feedback)


def l2_penalty(delta_head):
    return sum(jnp.sum(jnp.square(leaf)) for leaf in jax.tree.leaves(delta_head))


def ema_update(average, current, rate):
    return jax.tree.map(lambda old, new: (1 - rate) * old + rate * new, average, current)
