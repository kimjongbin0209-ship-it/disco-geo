"""Offline check that the outcome teacher extracts information from real reward timing.

A fixed, known behavior policy (the sticky "explore" random walk) collects real
Memory Maze episodes on reserved RL-train seeds. Its exact per-step action
probabilities are the teacher's base logits. Beliefs come from a fixed,
untrained encoder/GRU. The outcome teacher is fitted on some episodes and its
information gain is measured on held-out episodes against the shuffled-feedback
control. This validates the teacher pathway on real frames and rewards; it says
nothing about whether distillation helps an RL learner.

Usage: MUJOCO_GL=glfw python scripts/validate_teacher_offline.py --episodes 36 \
          --cache runs/teacher_validation/episodes --variants 128:0 0:0 0:0.01 \
          --output runs/teacher_validation/variants.json

Each variant is teacher_hidden:teacher_l2. Cached episodes are reused, so all
variants are compared on identical data.
"""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import time

import numpy as np


def behavior_logits(actions, policy):
    """Exact log-probabilities of the sticky reference policy for each taken step."""
    spec = policy
    base = np.zeros(6)
    base[spec["actions"]] = spec["probabilities"]
    logits = np.zeros((len(actions), 6))
    previous = None
    for t, action in enumerate(actions):
        if previous is None:
            probabilities = base
        else:
            probabilities = (1 - spec["stickiness"]) * base
            probabilities[previous] += spec["stickiness"]
        logits[t] = np.log(np.maximum(probabilities, 1e-12))
        previous = action
    return logits


def main():
    import jax
    import jax.numpy as jnp
    import optax
    from dm_env import specs

    from disco_spatial import feedback as fb
    from disco_spatial import teacher as teacher_lib
    from disco_spatial.agents import RecurrentAgent
    from disco_spatial.artifacts import REFERENCE_POLICIES, reference_actions
    from disco_spatial.envs.memory_maze import make_memory_maze
    from disco_spatial.io import write_json
    from disco_spatial.training import EpisodeRecord, trajectory_observations
    from disco_spatial.trajectory import load_trajectory, save_trajectory

    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=24)
    parser.add_argument("--seed", type=int, default=900000)
    parser.add_argument("--horizon", type=int, default=64)
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--variants", nargs="+", default=["128:0"])
    parser.add_argument("--cache")
    parser.add_argument("--output", required=True)
    options = parser.parse_args()
    if options.seed < 0 or options.seed + options.episodes > 1000000 or options.episodes < 3:
        raise ValueError("Use at least three episodes inside the RL-train seed namespace")
    output = Path(options.output)
    if output.exists():
        raise FileExistsError("Use a new output")
    policy = REFERENCE_POLICIES["explore"]
    started = time.monotonic()
    cache = Path(options.cache) if options.cache else None
    cached = sorted(cache.glob("episode*.npz")) if cache and cache.exists() else []
    trajectories = [load_trajectory(path) for path in cached[:options.episodes]]
    for index in range(len(trajectories), options.episodes):
        env = make_memory_maze(9, options.seed + index, "train", 64)
        try:
            timestep = env.reset()
            record = EpisodeRecord(env, timestep, "reference_explore")
            actions = reference_actions(np.random.default_rng(options.seed + index), "explore", 1000)
            logits = behavior_logits(actions, policy)
            for action, row in zip(actions, logits):
                timestep = env.step(action)
                record.append(action, row.astype(np.float32), 0, timestep)
                if timestep.last():
                    break
            trajectories.append(record.trajectory())
            if cache:
                save_trajectory(trajectories[-1], cache / f"episode{index:05d}.npz")
        finally:
            env.close()
    collection_seconds = time.monotonic() - started

    def run_variant(hidden, l2):
        config = SimpleNamespace(method="actor_critic", channels=(32, 64, 64), feature_dim=128,
                                 hidden_size=128, batch_size=1, learning_rate=3e-4,
                                 teacher="outcome", teacher_hidden=hidden)

        class Specification:
            def single_observation_spec(self):
                return {"rgb": specs.Array((64, 64, 3), np.uint8),
                        "previous_action": specs.Array((), np.int32),
                        "previous_reward": specs.Array((), np.float32)}

            def single_action_spec(self):
                return specs.BoundedArray((), np.int32, 0, 5)

        agent = RecurrentAgent(config, Specification())
        params = agent.initial_learner_state(jax.random.PRNGKey(0)).params
        unroll = jax.jit(agent.policy_network.unroll)

        def episode_arrays(trajectory):
            observations = trajectory_observations(trajectory)
            state = agent.initial_actor_state(jax.random.PRNGKey(0), batch_size=1)
            out, _ = unroll(params, state, observations, observations["is_first"])
            success, _, censored = fb.outcome_labels(trajectory.rewards, options.horizon, True)
            return {"belief": np.asarray(out["belief"][:-1, 0]), "actions": trajectory.actions,
                    "behavior": trajectory.behavior_logits, "outcome": success.astype(np.float32),
                    "known": ~censored}

        data = [episode_arrays(t) for t in trajectories]
        split = len(data) * 2 // 3

        def stack(rows):
            return {key: np.concatenate([row[key] for row in rows]) for key in rows[0]}

        train, held = stack(data[:split]), stack(data[split:])
        transform = agent.teacher
        head = teacher_lib.teacher_params(params)

        def delta(head, belief, outcome):
            return teacher_lib.delta_logits(transform, params, head, belief[None],
                                            {"outcome": outcome[None]})[0]

        def nll(head, batch):
            logp = jax.nn.log_softmax(batch["behavior"] + delta(head, batch["belief"], batch["outcome"]))
            taken = jnp.take_along_axis(logp, batch["actions"][:, None], axis=1)[:, 0]
            return -jnp.sum(taken * batch["known"]) / jnp.maximum(batch["known"].sum(), 1)

        optimizer = optax.adam(1e-3)
        state = optimizer.init(head)
        train_j = jax.tree.map(jnp.asarray, train)

        @jax.jit
        def step(head, state):
            loss, grads = jax.value_and_grad(
                lambda h: nll(h, train_j) + l2 * teacher_lib.l2_penalty(h))(head)
            updates, state = optimizer.update(grads, state)
            return optax.apply_updates(head, updates), state, loss

        def gains(head, batch, outcome):
            base = jax.nn.log_softmax(batch["behavior"])
            teacher = jax.nn.log_softmax(batch["behavior"] + delta(head, batch["belief"], outcome))
            index = batch["actions"][:, None]
            return np.asarray(jnp.take_along_axis(teacher - base, index, axis=1)[:, 0])

        held_j = jax.tree.map(jnp.asarray, held)
        shuffled_outcome = jnp.asarray(np.random.default_rng(1).permutation(held["outcome"]))
        success = (held["outcome"] > 0.5) & held["known"]
        checkpoints = sorted({1, 100, 250, 500, 1000, options.iterations} - {0})
        curve = []
        for iteration in range(1, options.iterations + 1):
            head, state, loss = step(head, state)
            if iteration in checkpoints:
                actual = gains(head, held_j, held_j["outcome"])
                control = gains(head, held_j, shuffled_outcome)
                curve.append({
                    "iteration": iteration, "train_objective": float(loss),
                    "held_out_information_gain_success": float(actual[success].mean()),
                    "held_out_shuffled_information_gain_success": float(control[success].mean()),
                    "held_out_information_gain_known": float(actual[held["known"]].mean()),
                })
                print(json.dumps({"hidden": hidden, "l2": l2, **curve[-1]}), flush=True)
        return {"teacher_hidden": hidden, "teacher_l2": l2, "curve": curve,
                "best_held_out_margin": max(r["held_out_information_gain_success"]
                                            - r["held_out_shuffled_information_gain_success"]
                                            for r in curve),
                "train_episodes": split, "held_out_episodes": len(data) - split,
                "held_out_success_steps": int(success.sum()),
                "held_out_known_steps": int(held["known"].sum())}

    variants = []
    for spec in options.variants:
        hidden, l2 = spec.split(":")
        variants.append(run_variant(int(hidden), float(l2)))
    result = {
        "schema_version": 1, "status": "offline_teacher_pathway_check_not_rl_result",
        "behavior_policy": "reference_explore (exact sticky-mixture logits as teacher base)",
        "representation": "untrained encoder/GRU belief, initialization PRNGKey(0)",
        "episodes": options.episodes, "seeds": [options.seed, options.seed + options.episodes - 1],
        "horizon": options.horizon, "rewards": int(sum(t.rewards.sum() for t in trajectories)),
        "variants": variants, "collection_seconds": collection_seconds,
        "interpretation": "Positive held-out gain above the shuffled control means outcome feedback "
                          "carries action information beyond the history for this behavior policy.",
    }
    write_json(output, result)


if __name__ == "__main__":
    main()
