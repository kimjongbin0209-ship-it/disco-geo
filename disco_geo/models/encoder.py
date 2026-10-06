import haiku as hk
import jax
import jax.numpy as jnp
from disco_geo.upstream import ensure_disco

ensure_disco()
from disco_rl import types  # noqa: E402
from disco_rl.networks import action_models  # noqa: E402


class RGBEncoder(hk.Module):
    def __init__(self, channels=(32, 64, 64), feature_dim=256, name="visual_encoder"):
        super().__init__(name=name)
        self.channels = channels
        self.feature_dim = feature_dim

    def __call__(self, rgb):
        x = rgb.astype(jnp.float32) / 255
        for i, channels in enumerate(self.channels):
            x = jax.nn.relu(hk.Conv2D(channels, 3, stride=2, name=f"conv_{i}")(x))
        return jax.nn.relu(hk.Linear(self.feature_dim, name="projection")(hk.Flatten()(x)))


def network(config, out_spec, model_out_spec, action_spec):
    def forward(obs, should_reset=None):
        del should_reset
        # This root module naming is identical to the standalone feature extractor.
        encoder = RGBEncoder(config.channels, config.feature_dim)
        current = encoder(obs["rgb"])
        reference = encoder(obs["reference_rgb"])
        torso = hk.nets.MLP((256,), activate_final=True, name="policy_torso")(
            jnp.concatenate((current, reference, current - reference, current * reference), -1)
        )
        out = {
            key: hk.Linear(
                int(spec.shape[0]), w_init=hk.initializers.TruncatedNormal(0.01), name=f"head_{key}"
            )(torso)
            for key, spec in out_spec.items()
        }
        if model_out_spec:
            model = action_models.LSTMModel(action_spec, model_out_spec, (128,), 128)
            out.update(model.model_step(model.root_embedding(torso)))
        return out

    one = hk.without_apply_rng(hk.transform_with_state(forward))
    unroll = hk.without_apply_rng(hk.transform_with_state(hk.BatchApply(forward)))
    return types.PolicyNetwork(one.init, one.apply, unroll.apply)


def feature_transform(config):
    return hk.without_apply_rng(
        hk.transform(lambda rgb: RGBEncoder(config.channels, config.feature_dim)(rgb))
    )
