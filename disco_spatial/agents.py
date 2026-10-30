"""Recurrent agents trained with the released Disco103 or V-trace rules."""

import hashlib
from pathlib import Path
import sys

import chex
import dm_env
from dm_env import specs
import jax
import jax.numpy as jnp
import numpy as np
import optax
import rlax

from disco_spatial import teacher as teacher_lib
from disco_spatial.methods import (masked_mean, reverse_kl_distillation, reward_proximity,
                                   self_imitation_loss)
from disco_spatial.models import network, setting


_UPSTREAM = Path(__file__).resolve().parents[1] / "third_party" / "disco_rl"
if str(_UPSTREAM) not in sys.path:
    sys.path.insert(0, str(_UPSTREAM))
from disco_rl import agent as official  # noqa: E402
from disco_rl import types, utils  # noqa: E402
from disco_rl.update_rules.actor_critic import ActorCritic  # noqa: E402
from disco_rl.value_fns import value_utils  # noqa: E402


DISCO_COMMIT = "9059a29f7121d60948f25ef165e08e050e9399c8"
RULE_HASH = "dac840a15b81e8805326363458c33979626172618f620d9e6e170315388530aa"


def load_rule():
    path = _UPSTREAM / "disco_rl/update_rules/weights/disco_103.npz"
    if hashlib.sha256(path.read_bytes()).hexdigest() != RULE_HASH:
        raise ValueError("Released Disco103 checkpoint checksum mismatch")
    with np.load(path, allow_pickle=False) as weights:
        result = {}
        for key in weights.files:
            module, name = key.rsplit("/", 1)
            result.setdefault(module, {})[name] = np.array(weights[key], copy=True)
        return result


class ActorCriticAux(ActorCritic):
    """The same action-conditional next-policy prediction, alongside V-trace."""

    def model_output_spec(self, single_action_spec):
        return {"aux_pi": utils.get_logits_specs(single_action_spec)}

    def agent_loss(self, rollout, meta_out, hyper_params, backprop):
        loss, logs = super().agent_loss(rollout, meta_out, hyper_params, backprop)
        prediction = utils.batch_lookup(rollout.agent_out["aux_pi"][:-1], rollout.actions[:-1])
        target = jax.lax.stop_gradient(rollout.agent_out["logits"][1:])
        auxiliary = rlax.categorical_kl_divergence(target, prediction) * (1 - rollout.is_terminal)
        return loss + auxiliary, dict(logs, aux_kl_loss=jnp.mean(auxiliary))


class RecurrentAgent(official.Agent):
    """Official agent API with explicit FIRST and correct recurrent start state.

    ``env`` supplies single_observation_spec() and single_action_spec(). Its
    observation spec includes RGB, previous_action and previous_reward; START
    action may equal num_actions. The actor injects is_first into saved rollouts.
    Pass the rollout's *initial* actor state to learner_step, never its end state.
    """

    def __init__(self, config, env, method=None):
        aliases = {
            "ac": "actor_critic", "fresh_ac": "actor_critic",
            "ac_aux": "actor_critic_aux", "fresh_ac_aux": "actor_critic_aux",
            "fresh_disco": "disco",
        }
        selected = method or setting(config, "base_method", None) or setting(config, "method", "disco")
        self.method = aliases.get(selected, selected)
        if self.method not in {"actor_critic", "actor_critic_aux", "disco"}:
            raise ValueError(f"Unsupported base update rule: {self.method}")
        self.config = config
        self.batch_size = setting(config, "batch_size", 1)
        settings = official.get_settings_disco() if self.method == "disco" else official.get_settings_actor_critic()
        settings.learning_rate = setting(config, "learning_rate", 3e-4)
        if self.method != "disco":
            settings.hyper_params.entropy_cost = setting(config, "entropy_cost", 0.01)
        observations = dict(env.single_observation_spec())
        observations["is_first"] = specs.Array((), np.bool_, name="is_first")
        super().__init__(
            single_observation_spec=observations,
            single_action_spec=env.single_action_spec(),
            agent_settings=settings,
            batch_axis_name=None,
        )
        if self.method == "actor_critic_aux":
            self.update_rule = ActorCriticAux(**settings.update_rule)
        self._network = network(
            config,
            self.update_rule.flat_output_spec(self.single_action_spec),
            self.update_rule.model_output_spec(self.single_action_spec),
            self.single_action_spec,
        )
        self.teacher, self.teacher_channels = None, ()
        if setting(config, "teacher", "none") != "none":
            self.teacher, self.teacher_channels = teacher_lib.teacher_network(
                config, int(self.single_action_spec.maximum) + 1)

    @property
    def policy_network(self):
        return self._network

    def jitted(self, name, function, **kwargs):
        """Per-agent cache of compiled helpers, reused across evaluation episodes."""
        cache = self.__dict__.setdefault("_jit_cache", {})
        if name not in cache:
            cache[name] = jax.jit(function, **kwargs)
        return cache[name]

    def initial_learner_state(self, rng_key):
        """Upstream initialization; a teacher head is added without changing the backbone.

        The backbone, optimizer moments and meta-state come from the same RNG
        splits as the teacher-free agent, so conditions share initialization.
        """
        state = super().initial_learner_state(rng_key)
        if self.teacher is None:
            return state
        rgb_shape = self.single_observation_spec["rgb"].shape
        head = teacher_lib.init_teacher(self.teacher, jax.random.fold_in(rng_key, 7),
                                        setting(self.config, "hidden_size", 128), rgb_shape)
        params = dict(state.params, **head)
        _, state_rng = jax.random.split(rng_key)
        return official.LearnerState(
            params=params, opt_state=self._optimizer.init(params),
            meta_state=self.update_rule.init_meta_state(rng=state_rng, params=params))

    def scalar_values(self, agent_out):
        """[T, B] state values from the rule's own categorical value outputs."""
        if self.method == "disco":
            values, _ = value_utils.extract_scalar_values_from_net_out(
                None, agent_out["q"], agent_out["logits"], categorical_value=True,
                max_abs_value=self.update_rule._max_abs_value, nonlinear_transform=True)
        else:
            rule = self.update_rule
            values, _ = value_utils.extract_scalar_values_from_net_out(
                agent_out["v"], None, agent_out["logits"], categorical_value=rule._categorical_value,
                max_abs_value=rule._max_abs_value, nonlinear_transform=rule._nonlinear_value_transform)
        return values

    def initial_actor_state(self, rng, batch_size=None):
        size = self.batch_size if batch_size is None else batch_size
        if size < 1:
            raise ValueError("batch_size must be positive")
        dummy = jax.tree.map(
            lambda spec: np.zeros((size,) + spec.shape, dtype=spec.dtype),
            self.single_observation_spec,
        )
        _, state = self._network.init(rng, dummy, jnp.ones((size,), dtype=bool))
        # Haiku init executes the forward pass: do not retain a dummy image's
        # GRU output as the initial memory of the real episode.
        return jax.tree.map(jnp.zeros_like, state)

    def actor_step(self, actor_params, rng, timestep, actor_state):
        first = timestep.step_type == dm_env.StepType.FIRST
        observations = dict(timestep.observation, is_first=first)
        output, state = self._network.one_step(actor_params, actor_state, observations, first)
        import distrax

        actions = distrax.Softmax(logits=output["logits"]).sample(seed=rng)
        discount = getattr(timestep, "discount", None)
        if discount is None:
            # The old official EnvironmentTimestep omits discount altogether.
            # Benchmark adapters should supply it so time-limit LAST can keep
            # continuation=1 while a true terminal has continuation=0.
            discount = jnp.asarray(timestep.step_type != dm_env.StepType.LAST, jnp.float32)
        else:
            discount = jnp.asarray(discount, jnp.float32)
            discount = jnp.where(first, jnp.ones_like(discount), discount)
        discount = jnp.broadcast_to(discount, first.shape)
        actor = types.ActorTimestep(
            observations=observations,
            actions=actions,
            rewards=timestep.reward,
            discounts=discount,
            agent_outs=output,
            states=actor_state,
            logits=output["logits"],
        )
        return actor, state

    def unroll_net(self, agent_net_params, agent_net_state, rollout):
        observations = rollout.observations
        if "is_first" not in observations:
            raise ValueError("Recurrent rollouts must preserve explicit is_first flags")
        return self._network.unroll(
            agent_net_params, agent_net_state, observations, observations["is_first"]
        )

    def rule_params(self):
        if self.method != "disco":
            return {"dummy": jnp.array(0.0)}
        params = load_rule()
        expected, _ = self.update_rule.init_params(jax.random.PRNGKey(0))
        chex.assert_trees_all_equal_shapes_and_dtypes(expected, params)
        return params

    def _rollout_gradient(
        self, rng, rollout, learner_state, start_state, rule_params, target_start_state=None
    ):
        """Released loss and target calculation, without applying an optimizer."""
        output, _ = self.unroll_net(learner_state.params, start_state, rollout)
        inputs = types.UpdateRuleInputs(
            observations=rollout.observations,
            actions=rollout.actions,
            rewards=rollout.rewards[1:],
            is_terminal=rollout.discounts[1:] == 0,
            behaviour_agent_out=rollout.agent_outs,
            agent_out=output,
            value_out=None,
        )
        # The conventional AC rule mutates its meta_state dictionary. Distinct
        # tree copies prevent either branch from altering the other's EMA input.
        meta_state = jax.tree.map(lambda x: x, learner_state.meta_state)

        def target_unroll(params, inherited_state, observations, reset):
            # The released rule supplies EMA parameters but knows nothing about
            # their separately reconstructed recurrent prefix. Override only
            # this target forward pass, never the online student forward pass.
            state = inherited_state if target_start_state is None else target_start_state
            return self._network.unroll(params, state, observations, reset)

        meta_out, updated_meta_state = self.update_rule.unroll_meta_net(
            meta_params=rule_params,
            params=learner_state.params,
            state=start_state,
            meta_state=meta_state,
            rollout=inputs,
            hyper_params=self.settings.hyper_params.to_dict(),
            unroll_policy_fn=target_unroll,
            rng=rng,
            axis_name=None,
        )
        grads, (_, final_state, logs) = jax.grad(self._loss, has_aux=True)(
            learner_state.params,
            agent_net_state=start_state,
            meta_state=meta_state,
            rollout=rollout,
            meta_out=meta_out,
            is_meta_training=False,
        )
        return grads, updated_meta_state, final_state, logs, meta_out

    def learner_step_targets(
        self, rng, rollout, learner_state, start_state, target_start_state, rule_params
    ):
        """One released-rule update with an independently reconstructed EMA state.

        Recompute ``target_start_state`` from the raw causal episode prefix
        under ``learner_state.meta_state['target_params']``. Online actor state
        remains ``start_state``. No recurrent state or annotation is added to
        learner observations. ``None`` delegates to the original learner API.
        """
        if target_start_state is None:
            return self.learner_step(rng, rollout, learner_state, start_state, rule_params, False)
        gradients, meta_state, final_state, logs, meta_out = self._rollout_gradient(
            rng, rollout, learner_state, start_state, rule_params, target_start_state
        )
        updates, optimizer_state = self._optimizer.update(
            gradients, learner_state.opt_state, learner_state.params
        )
        params = optax.apply_updates(learner_state.params, updates)
        return (
            official.LearnerState(params=params, opt_state=optimizer_state, meta_state=meta_state),
            final_state,
            dict(
                logs,
                global_gradient_norm=optax.global_norm(gradients),
                global_update_norm=optax.global_norm(updates),
                meta_out=meta_out,
            ),
        )

    def learner_step_mixed(
        self,
        rng,
        fresh_rollout,
        replay_rollout,
        learner_state,
        fresh_start_state,
        replay_start_state,
        rule_params,
        replay_weight,
        fresh_target_start_state=None,
        replay_target_start_state=None,
    ):
        """One mixed-gradient update; only fresh lifetime statistics advance.

        ``replay_weight`` is a static Python float in [0,1); mark it static when
        JIT compiling this method. Each released loss normalizes its own valid
        timesteps, so sequence lengths and batch sizes may differ. Replay must
        retain its actual behavior logits and observed rewards. Recompute its
        current hidden start state from a causal prefix before calling here.
        For Disco, supply each optional target start state recomputed with the
        current EMA target parameters over the same raw causal prefix.

        This is an explicit extension of the released rule, not a guarantee of
        off-policy correctness for arbitrary stale replay. At weight zero the
        call delegates directly to the unmodified released learner step.
        """
        if not 0 <= replay_weight < 1:
            raise ValueError("replay_weight must be in [0, 1) and leave a fresh component")
        if replay_weight == 0:
            return self.learner_step_targets(
                rng, fresh_rollout, learner_state, fresh_start_state,
                fresh_target_start_state, rule_params,
            )
        gradients, meta_state, final_state, logs = self._rule_gradient(
            rng, fresh_rollout, replay_rollout, learner_state, fresh_start_state,
            replay_start_state, rule_params, replay_weight, fresh_target_start_state,
            replay_target_start_state,
        )
        return self._apply(gradients, learner_state, meta_state, final_state, logs)

    def _rule_gradient(self, rng, fresh_rollout, replay_rollout, learner_state, fresh_start_state,
                       replay_start_state, rule_params, replay_weight,
                       fresh_target_start_state=None, replay_target_start_state=None):
        """Released-rule gradient on fresh data, optionally mixed with replay."""
        fresh, meta_state, final_state, fresh_logs, fresh_meta_out = self._rollout_gradient(
            rng, fresh_rollout, learner_state, fresh_start_state, rule_params,
            fresh_target_start_state,
        )
        if replay_rollout is None or replay_weight == 0:
            return fresh, meta_state, final_state, dict(fresh_logs, meta_out=fresh_meta_out)
        replay, _, _, replay_logs, _ = self._rollout_gradient(
            jax.random.fold_in(rng, 1), replay_rollout, learner_state, replay_start_state,
            rule_params, replay_target_start_state,
        )
        gradients = jax.tree.map(
            lambda fresh_grad, replay_grad: (
                (1 - replay_weight) * fresh_grad + replay_weight * replay_grad
            ),
            fresh,
            replay,
        )
        logs = dict(
            fresh_logs,
            total_loss=(1 - replay_weight) * fresh_logs["total_loss"]
            + replay_weight * replay_logs["total_loss"],
            fresh_loss=fresh_logs["total_loss"],
            replay_loss=replay_logs["total_loss"],
            meta_out=fresh_meta_out,
        )
        return gradients, meta_state, final_state, logs

    def _apply(self, gradients, learner_state, meta_state, final_state, logs):
        updates, optimizer_state = self._optimizer.update(
            gradients, learner_state.opt_state, learner_state.params
        )
        params = optax.apply_updates(learner_state.params, updates)
        logs = dict(logs, global_gradient_norm=optax.global_norm(gradients),
                    global_update_norm=optax.global_norm(updates))
        return (
            official.LearnerState(params=params, opt_state=optimizer_state, meta_state=meta_state),
            final_state,
            logs,
        )

    def credit_diagnostics(self, params, start_state, rollout, rule_signal, window):
        """Sums and step counts: does an update favour actions that preceded a reward?

        ``params`` are the weights after one update on the fresh ``rollout``,
        whose behaviour logits come from the weights before it. The update's
        effect is the change in the taken action's log-probability. The rule's
        own signal is, for Disco, the log-probability its policy target gives
        the taken action relative to the behaviour policy (``rule_signal`` is
        meta_out["pi"]); for actor-critic it is the advantage (meta_out["raw_advs"]).
        Both are split into steps followed by a reward within ``window`` steps
        and steps that are not. Diagnostic only; nothing here trains.
        """
        output, _ = self.unroll_net(params, start_state, rollout)
        actions = rollout.actions[:-1]
        before = jax.nn.log_softmax(rollout.agent_outs["logits"][:-1])
        after = jax.nn.log_softmax(output["logits"][:-1])

        def taken(log_probabilities):
            return jnp.take_along_axis(log_probabilities, actions[..., None], axis=-1)[..., 0]

        if self.method == "disco":
            signal = taken(jax.nn.log_softmax(rule_signal)) - taken(before)
        else:
            signal = rule_signal
        gain = taken(after) - taken(before)
        proximal, distal = reward_proximity(rollout.rewards[1:], window)
        sums = {"update_policy_kl": jnp.mean(jnp.sum(jnp.exp(before) * (before - after), -1))}
        for name, mask in (("proximal", proximal), ("distal", distal)):
            weights = mask.astype(jnp.float32)
            sums[f"{name}_steps"] = weights.sum()
            sums[f"update_logp_gain_{name}"] = jnp.sum(gain * weights)
            sums[f"rule_signal_{name}"] = jnp.sum(signal * weights)
        return sums

    def self_imitation_objective(self, params, batch):
        """n-step positive-advantage imitation on a replayed chunk (stop-gradient V).

        Returns use observed rewards inside the chunk and bootstrap with the
        current value of its final observation. This changes the objective; it is
        not an off-policy-corrected policy gradient.
        """
        rollout = batch["rollout"]
        output, _ = self.unroll_net(params, batch["start_state"], rollout)
        values = jax.lax.stop_gradient(self.scalar_values(output))
        discount = self.settings.hyper_params.discount_factor

        def backward(future, step):
            reward, continuation = step
            value = reward + discount * continuation * future
            return value, value

        _, returns = jax.lax.scan(backward, values[-1],
                                  (rollout.rewards[1:], rollout.discounts[1:]), reverse=True)
        valid = jnp.ones(returns.shape, bool)
        loss = self_imitation_loss(output["logits"][:-1], rollout.actions[:-1], returns,
                                   values[:-1], valid)
        return loss, {"self_imitation_loss": loss,
                      "self_imitation_positive_fraction": jnp.mean(returns > values[:-1]),
                      "self_imitation_mean_advantage": jnp.mean(jnp.maximum(returns - values[:-1], 0))}

    def feedback_objective(self, params, teacher_ema, batch, distillation_weight, teacher_weight):
        """Teacher behavior cloning plus KL(student || stopgrad(EMA teacher)).

        ``batch`` holds a rollout of the student's own recent attempt, its
        behavior logits, three feedback trees (train: actual or control inputs;
        query: what the distillation target is conditioned on; shuffled: a
        diagnostic permutation) and masks. Diagnostics use pre-update weights on
        chunks the teacher has not been trained on yet.
        """
        rollout = batch["rollout"]
        output, _ = self.unroll_net(params, batch["start_state"], rollout)
        logits, belief = output["logits"][:-1], output["belief"][:-1]
        actions = rollout.actions[:-1]
        behavior = batch["behavior_logits"]
        head = teacher_lib.teacher_params(params)

        def body(tree):
            return jax.tree.map(lambda x: x[:-1], tree)

        def taken(log_probabilities):
            return jnp.take_along_axis(log_probabilities, actions[..., None], axis=-1)[..., 0]

        def delta(weights, feedback):
            return teacher_lib.delta_logits(self.teacher, params, weights, belief, body(feedback))

        known, success = batch["known"], batch["known"] & batch["success"]
        behavior_logp = jax.nn.log_softmax(behavior)
        teacher_logp = jax.nn.log_softmax(behavior + delta(head, batch["train"]))
        teacher_nll = masked_mean(-taken(teacher_logp), known)
        regularization = setting(self.config, "teacher_l2", 0.0) * teacher_lib.l2_penalty(head)
        query = jax.lax.stop_gradient(delta(teacher_ema, batch["query"]))
        target = jax.lax.stop_gradient(logits) + query
        distillation = reverse_kl_distillation(logits, target, batch["distill_mask"],
                                               teacher_ready=True)
        loss = teacher_weight * (teacher_nll + regularization) + distillation_weight * distillation
        shuffled_logp = jax.nn.log_softmax(behavior + jax.lax.stop_gradient(
            delta(head, batch["shuffled"])))
        gain = jax.lax.stop_gradient(taken(teacher_logp) - taken(behavior_logp))
        shuffled_gain = taken(shuffled_logp) - taken(behavior_logp)
        query_ratio = jnp.exp(taken(jax.nn.log_softmax(behavior + query)) - taken(behavior_logp))
        query_probabilities = jax.nn.softmax(target)
        logs = {
            "teacher_nll": teacher_nll, "distillation_kl": distillation,
            "teacher_information_gain_success": masked_mean(gain, success),
            "teacher_information_gain_known": masked_mean(gain, known),
            "teacher_shuffled_information_gain_success": masked_mean(shuffled_gain, success),
            "teacher_local_improvement": masked_mean((query_ratio - 1) * batch["returns"], known),
            "teacher_copy_rate": masked_mean(
                (jnp.exp(jax.lax.stop_gradient(taken(teacher_logp))) > 0.95).astype(jnp.float32),
                known),
            "teacher_query_entropy": jnp.mean(
                -jnp.sum(query_probabilities * jnp.log(jnp.maximum(query_probabilities, 1e-8)), -1)),
            "teacher_known_fraction": jnp.mean(known.astype(jnp.float32)),
            "teacher_success_fraction": jnp.mean(success.astype(jnp.float32)),
            "teacher_known_steps": jnp.sum(known.astype(jnp.float32)),
            "teacher_success_steps": jnp.sum(success.astype(jnp.float32)),
            "distill_mask_fraction": jnp.mean(batch["distill_mask"].astype(jnp.float32)),
        }
        return loss, logs

    def learner_step_composite(
        self,
        rng,
        fresh_rollout,
        replay_rollout,
        learner_state,
        fresh_start_state,
        replay_start_state,
        rule_params,
        replay_weight,
        fresh_target_start_state=None,
        replay_target_start_state=None,
        self_imitation=None,
        feedback=None,
        teacher_ema=None,
        weights=None,
    ):
        """Released-rule gradient plus optional auxiliary objectives, one optimizer step.

        ``replay_weight`` is static. ``weights`` holds dynamic scalars
        ``self_imitation``, ``distillation`` and ``teacher``, so an evidence gate
        can switch distillation on without recompiling. Without auxiliary
        batches this is exactly ``learner_step_mixed``. Only fresh data advance
        the rule's meta-state; auxiliary objectives never touch it.
        """
        if self_imitation is None and feedback is None:
            return self.learner_step_mixed(
                rng, fresh_rollout, replay_rollout, learner_state, fresh_start_state,
                replay_start_state, rule_params, replay_weight, fresh_target_start_state,
                replay_target_start_state)
        if feedback is not None and (self.teacher is None or teacher_ema is None):
            raise ValueError("Feedback distillation needs a configured teacher and its EMA head")
        gradients, meta_state, final_state, logs = self._rule_gradient(
            rng, fresh_rollout, replay_rollout, learner_state, fresh_start_state,
            replay_start_state, rule_params, replay_weight, fresh_target_start_state,
            replay_target_start_state)

        def auxiliary(params):
            total, aux_logs = 0.0, {}
            if self_imitation is not None:
                loss, extra = self.self_imitation_objective(params, self_imitation)
                total = total + weights["self_imitation"] * loss
                aux_logs.update(extra)
            if feedback is not None:
                loss, extra = self.feedback_objective(params, teacher_ema, feedback,
                                                      weights["distillation"], weights["teacher"])
                total = total + loss
                aux_logs.update(extra)
            return total, aux_logs

        extra, aux_logs = jax.grad(auxiliary, has_aux=True)(learner_state.params)
        logs = dict(logs, **aux_logs, rule_gradient_norm=optax.global_norm(gradients),
                    auxiliary_gradient_norm=optax.global_norm(extra))
        gradients = jax.tree.map(jnp.add, gradients, extra)
        return self._apply(gradients, learner_state, meta_state, final_state, logs)

