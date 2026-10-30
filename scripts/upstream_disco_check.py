"""Run the unmodified upstream Agent on upstream Catch: is the released rule working here?

Usage: python scripts/upstream_disco_check.py --method disco --learning-rate 1e-2 --updates 400

No project code is involved except the checksum-verified Disco103 loader. The setup
follows the upstream evaluation colab (MLP agent, 64 Catch streams, 29-step rollouts),
without its replay buffer: one on-policy update per rollout. The colab uses learning
rate 1e-2; P1 used 3e-4. Prints the mean terminal reward of recent episodes (Catch
pays +1 or -1 once per episode; a uniform policy scores about -0.75 on 8x8).
"""

import argparse
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party" / "disco_rl"))
sys.path.insert(0, str(ROOT))
from disco_rl import agent as official, types  # noqa: E402
from disco_rl.environments import jittable_envs  # noqa: E402
from disco_spatial.agents import load_rule  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=("disco", "actor_critic"), default="disco")
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--updates", type=int, default=400)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--length", type=int, default=29)
    parser.add_argument("--seed", type=int, default=0)
    options = parser.parse_args()
    if options.method == "disco":
        settings = official.get_settings_disco()
        settings.net_settings.name = "mlp"
    else:
        settings = official.get_settings_actor_critic()
    settings.learning_rate = options.learning_rate
    env = jittable_envs.CatchJittableEnvironment(batch_size=options.batch,
                                                 env_settings=jittable_envs.get_config_catch())
    agent = official.Agent(single_observation_spec=env.single_observation_spec(),
                           single_action_spec=env.single_action_spec(), agent_settings=settings,
                           batch_axis_name=None)
    rule = load_rule() if options.method == "disco" else {"dummy": jnp.array(0.0)}
    key = jax.random.PRNGKey(options.seed)
    learner = agent.initial_learner_state(key)
    state = jax.tree.map(lambda x: jnp.repeat(x, options.batch, 0), agent.initial_actor_state(key))
    env_state, timestep = env.reset(jax.random.fold_in(key, 1))

    @jax.jit
    def collect(params, rng, env_state, timestep, state):
        def body(carry, step_rng):
            env_state, timestep, state = carry
            actor, state = agent.actor_step(params, step_rng, timestep, state)
            env_state, timestep = env.step(env_state, actor.actions)
            return (env_state, timestep, state), actor

        carry, actors = jax.lax.scan(body, (env_state, timestep, state),
                                     jax.random.split(rng, options.length + 1))
        return (actors,) + carry

    learner_step = jax.jit(agent.learner_step, static_argnums=5)
    returns = []
    for update in range(options.updates):
        start = state
        key, collect_key, update_key = jax.random.split(key, 3)
        actors, env_state, timestep, state = collect(learner.params, collect_key, env_state,
                                                     timestep, state)
        rollout = types.ActorRollout.from_timestep(actors)
        learner, _, logs = learner_step(update_key, rollout, learner, start, rule, False)
        rewards, discounts = np.asarray(rollout.rewards[1:]), np.asarray(rollout.discounts[1:])
        returns.extend(rewards[discounts == 0].tolist())
        if update % 50 == 0 or update == options.updates - 1:
            recent = returns[-50 * options.batch * 4:]
            print(f"{options.method} lr {options.learning_rate:g} update {update:4d} "
                  f"mean episode return {np.mean(recent):+.3f} entropy {float(logs['entropy']):.3f}",
                  flush=True)


if __name__ == "__main__":
    main()
