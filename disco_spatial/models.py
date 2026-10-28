"""Causal RGB/previous-action/previous-reward policy with explicit FIRST resets.

The GRU carries observation history. The released Disco action-model LSTM is
separate: it predicts action-conditional outputs from the current GRU belief.
"""

from pathlib import Path
import sys

import haiku as hk
import jax
import jax.numpy as jnp


_UPSTREAM = Path(__file__).resolve().parents[1] / "third_party" / "disco_rl"
if str(_UPSTREAM) not in sys.path:
    sys.path.insert(0, str(_UPSTREAM))
from disco_rl import types  # noqa: E402
from disco_rl.networks import action_models  # noqa: E402


def setting(config, name, default):
    """Accept a config object or mapping without importing the runner config."""
    return config.get(name, default) if isinstance(config, dict) else getattr(config, name, default)


class RGBEncoder(hk.Module):
    """A signed, normalized image feature, before temporal memory."""

    def __init__(self, channels=(32, 64, 64), feature_dim=128, name="visual_encoder"):
        super().__init__(name=name)
        self.channels = tuple(channels)
        self.feature_dim = feature_dim

    def __call__(self, rgb):
        x = rgb.astype(jnp.float32) / 255.0
        for index, channels in enumerate(self.channels):
            x = hk.Conv2D(channels, 3, stride=2, padding="SAME", name=f"conv_{index}")(x)
            x = jax.nn.silu(x)
        x = hk.Linear(self.feature_dim, name="projection")(hk.Flatten()(x))
        return hk.LayerNorm(-1, create_scale=True, create_offset=True, name="feature_norm")(x)


def network(config, out_spec, model_out_spec, action_spec):
    """Return official-compatible one-step and genuinely sequential unroll APIs.

    Observations may contain ``is_first``. When present it is authoritative, so
    the released learner's legacy reset masks cannot invert FIRST semantics.
    Otherwise the explicit ``should_reset`` argument means reset *before* the
    observation, and is useful for stand-alone policy evaluation.
    """
    channels = setting(config, "channels", (32, 64, 64))
    feature_dim = setting(config, "feature_dim", 128)
    hidden_size = setting(config, "hidden_size", 128)
    num_actions = int(action_spec.maximum) - int(action_spec.minimum) + 1

    def forward(obs, should_reset=None):
        rgb = obs["rgb"]
        batch_size = rgb.shape[0]
        first = obs.get("is_first", should_reset)
        if first is None:
            first = jnp.zeros((batch_size,), dtype=bool)
        first = jnp.asarray(first, dtype=bool)
        hidden = hk.get_state("actor_hidden", (batch_size, hidden_size), jnp.float32, init=jnp.zeros)
        hidden = jnp.where(first[:, None], jnp.zeros_like(hidden), hidden)
        features = RGBEncoder(channels, feature_dim)(rgb)
        previous_action = obs.get("previous_action", obs.get("prev_action"))
        previous_reward = obs.get("previous_reward", obs.get("prev_reward"))
        if previous_action is None or previous_reward is None:
            raise ValueError("Policy observations require previous_action and previous_reward")
        # START=num_actions (or -1) maps to the all-zero one-hot. On FIRST the
        # previous episode's action/reward are discarded even if a caller errs.
        action = jax.nn.one_hot(previous_action, num_actions, dtype=jnp.float32)
        action = jnp.where(first[:, None], jnp.zeros_like(action), action)
        reward = jnp.asarray(previous_reward, dtype=jnp.float32).reshape(batch_size, 1)
        reward = jnp.where(first[:, None], jnp.zeros_like(reward), reward)
        belief, new_hidden = hk.GRU(hidden_size, name="memory")(
            jnp.concatenate((features, action, reward), axis=-1), hidden
        )
        hk.set_state("actor_hidden", new_hidden)
        torso = hk.nets.MLP((hidden_size,), activate_final=True, name="policy_torso")(belief)
        out = {
            key: hk.Linear(
                int(spec.shape[0]),
                w_init=hk.initializers.TruncatedNormal(0.01),
                name=f"head_{key}",
            )(torso)
            for key, spec in out_spec.items()
        }
        if model_out_spec:
            action_model = action_models.LSTMModel(action_spec, model_out_spec, (128,), 128)
            out.update(action_model.model_step(action_model.root_embedding(torso)))
        out["image_features"] = features
        out["belief"] = belief
        return out

    def sequential(obs, should_reset=None):
        if should_reset is None:
            should_reset = jnp.zeros(obs["rgb"].shape[:2], dtype=bool)

        def step(carry, inputs):
            item, reset = inputs
            return carry, forward(item, reset)

        _, outputs = hk.scan(step, (), (obs, should_reset))
        return outputs

    one = hk.without_apply_rng(hk.transform_with_state(forward))
    sequence = hk.without_apply_rng(hk.transform_with_state(sequential))
    return types.PolicyNetwork(one.init, one.apply, sequence.apply)


def feature_transform(config):
    """Apply the same image encoder independently of the causal GRU."""
    return hk.without_apply_rng(
        hk.transform(
            lambda rgb: RGBEncoder(
                setting(config, "channels", (32, 64, 64)), setting(config, "feature_dim", 128)
            )(rgb)
        )
    )
