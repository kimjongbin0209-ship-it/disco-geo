"""Fixed reactive reference policies: what return needs no learning at all.

"uniform" samples the six actions uniformly; "forward_left:0.5" plays that
action with probability 0.5 and otherwise a uniform action. P1 showed a policy
drifting to about half forward_left roughly doubled the training reward rate,
so these references calibrate "above random" and the reward available to a
memoryless action bias. Sampling is seeded per episode.
"""

import time

import numpy as np

from disco_spatial.envs.memory_maze import ACTION_NAMES


def action_probabilities(policy):
    if policy == "uniform":
        return np.full(6, 1 / 6)
    name, _, value = policy.partition(":")
    if name not in ACTION_NAMES or not value:
        raise ValueError("Policy must be 'uniform' or '<action name>:<probability>'")
    probability = float(value)
    if not 0 <= probability <= 1:
        raise ValueError("Bias probability must be in [0, 1]")
    probabilities = np.full(6, (1 - probability) / 6)
    probabilities[ACTION_NAMES.index(name)] += probability
    return probabilities


def scripted_returns(env_factory, seeds, policy, *, split="test", maze_size=9, resolution=64,
                     max_steps=None):
    """Whole episodes of a fixed policy on fresh environments, one per seed."""
    probabilities = action_probabilities(policy)
    started, episodes = time.monotonic(), []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        env = env_factory(maze_size, seed, split, resolution)
        try:
            timestep, rewards = env.reset(), []
            while not timestep.last() and (max_steps is None or len(rewards) < max_steps):
                timestep = env.step(int(rng.choice(6, p=probabilities)))
                rewards.append(float(timestep.reward or 0))
        finally:
            env.close()
        episodes.append({"seed": int(seed), "return": float(sum(rewards)), "steps": len(rewards)})
    steps = sum(row["steps"] for row in episodes)
    reward = sum(row["return"] for row in episodes)
    return {"schema_version": 1, "policy": policy, "action_probabilities": probabilities.tolist(),
            "split": split, "episodes": episodes, "total_reward": reward, "total_steps": steps,
            "reward_per_1000_steps": 1000 * reward / steps if steps else None,
            "mean_return": float(np.mean([row["return"] for row in episodes])) if episodes else None,
            "wall_seconds": time.monotonic() - started,
            "interpretation": "Memoryless fixed policy; a reference, not a learned result."}
