import jax
import jax.numpy as jnp
import rlax
from disco_geo.models.encoder import network
from disco_geo.upstream import ensure_disco, load_rule

ensure_disco()
from disco_rl import agent as official  # noqa: E402
from disco_rl import utils  # noqa: E402
from disco_rl.update_rules.actor_critic import ActorCritic  # noqa: E402


class ActorCriticAux(ActorCritic):
    """Conventional V-trace plus the same next-policy prediction as Disco."""

    def model_output_spec(self, single_action_spec):
        return {"aux_pi": utils.get_logits_specs(single_action_spec)}

    def agent_loss(self, rollout, meta_out, hyper_params, backprop):
        loss, logs = super().agent_loss(rollout, meta_out, hyper_params, backprop)
        chosen = utils.batch_lookup(rollout.agent_out["aux_pi"][:-1], rollout.actions[:-1])
        target = jax.lax.stop_gradient(rollout.agent_out["logits"][1:])
        aux = rlax.categorical_kl_divergence(target, chosen) * (1 - rollout.is_terminal)
        return loss + aux, dict(logs, aux_kl_loss=jnp.mean(aux))


class GeoAgent(official.Agent):
    def __init__(self, config, env):
        settings = (
            official.get_settings_disco()
            if config.method == "disco"
            else official.get_settings_actor_critic()
        )
        settings.learning_rate = config.learning_rate
        if config.method == "disco":
            if config.ablation == "no_yz_loss":
                settings.hyper_params.y_cost = settings.hyper_params.z_cost = 0.0
            elif config.ablation == "no_aux_policy":
                settings.hyper_params.aux_policy_cost = 0.0
        else:
            # Equal reward discount/trace; a modest entropy coefficient for four actions.
            settings.hyper_params.entropy_cost = 0.01
        super().__init__(
            single_observation_spec=env.single_observation_spec(),
            single_action_spec=env.single_action_spec(),
            agent_settings=settings,
            batch_axis_name=None,
        )
        if config.method == "actor_critic_aux":
            self.update_rule = ActorCriticAux(**settings.update_rule)
        self._network = network(
            config,
            self.update_rule.flat_output_spec(self.single_action_spec),
            self.update_rule.model_output_spec(self.single_action_spec),
            self.single_action_spec,
        )

    def rule_params(self):
        params = (
            load_rule() if self.settings.update_rule_name == "disco" else {"dummy": jnp.array(0.0)}
        )
        if self.settings.update_rule_name == "disco":
            expected, _ = self.update_rule.init_params(jax.random.PRNGKey(0))
            import chex

            chex.assert_trees_all_equal_shapes_and_dtypes(expected, params)
        return params
