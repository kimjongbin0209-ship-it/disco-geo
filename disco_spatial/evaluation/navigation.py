"""Frozen-policy navigation episodes: return, matched revisits and memory interventions.

The actor sees only RGB, previous action and previous reward. Sidecars and
render telemetry are recorded for scoring after each step and never reach the
policy. Interventions alter only the actor's recurrent state.
"""

import time

import numpy as np

from disco_spatial.evaluation.labels import grid_distance, target_indices
from disco_spatial.evaluation.metrics import revisit_efficiency


INTERVENTIONS = ("none", "reset_every_step", "truncate_16", "truncate_64")
DISTANCE_BIN_CELLS = 3
MAX_DISTANCE_BIN = 3


def _truncation(intervention):
    if intervention == "none":
        return None
    if intervention == "reset_every_step":
        return 1
    if intervention.startswith("truncate_"):
        return int(intervention.split("_", 1)[1])
    raise ValueError(f"Unknown intervention {intervention}; choose from {INTERVENTIONS}")


def run_episode(agent, params, env, *, intervention="none", greedy=True, rng=None,
                actor=None, max_steps=None):
    """One whole episode. ``actor(output, history)`` may replace student logits."""
    import jax
    import jax.numpy as jnp

    period = _truncation(intervention)
    step = agent.jitted("one_step", agent.policy_network.one_step)
    state = agent.initial_actor_state(jax.random.PRNGKey(0), batch_size=1)
    rng = rng or np.random.default_rng(0)
    timestep = env.reset()
    history = {"rgb": [timestep.observation["rgb"]], "actions": [], "rewards": []}
    sidecars = [env.last_sidecar]
    previous_action, previous_reward, t = 6, 0.0, 0
    while not timestep.last() and (max_steps is None or t < max_steps):
        if period is not None and t % period == 0:
            state = jax.tree.map(jnp.zeros_like, state)
        observation = {
            "rgb": jnp.asarray(timestep.observation["rgb"][None]),
            "previous_action": jnp.asarray([previous_action], jnp.int32),
            "previous_reward": jnp.asarray([previous_reward], jnp.float32),
            "is_first": jnp.asarray([timestep.first()]),
        }
        output, state = step(params, state, observation, observation["is_first"])
        logits = np.asarray(output["logits"][0] if actor is None else actor(output, history))
        if greedy:
            action = int(np.argmax(logits))
        else:
            p = np.exp(logits - logits.max())
            action = int(rng.choice(len(p), p=p / p.sum()))
        timestep = env.step(action)
        previous_action, previous_reward = action, float(timestep.reward or 0)
        history["rgb"].append(timestep.observation["rgb"])
        history["actions"].append(action)
        history["rewards"].append(previous_reward)
        sidecars.append(env.last_sidecar)
        t += 1
    return {
        "actions": np.asarray(history["actions"], np.int32),
        "rewards": np.asarray(history["rewards"], np.float32),
        "sidecar": {key: np.stack([row[key] for row in sidecars]) for key in sidecars[0]},
        "episode_id": env.episode_id, "complete": bool(timestep.last()),
    }


def goal_segments(episode, seed, layout_id):
    """Segment records for revisit_efficiency; distance-0 segments are excluded."""
    sidecar, rewards = episode["sidecar"], episode["rewards"]
    positions = sidecar["agent_pos"]
    layout = sidecar["maze_layout"][0]
    n_targets = sidecar["targets_vec"].shape[1]
    requested = target_indices(sidecar["target_color"], n_targets)
    visible = sidecar.get("target_visible_pixels")
    ends = list(np.flatnonzero(rewards == 1) + 1) + [len(positions) - 1]
    records, excluded, seen, start = [], 0, set(), 0
    for index, end in enumerate(ends):
        if end <= start:
            continue
        success = index < len(ends) - 1
        target = int(requested[start])
        goal = np.floor(sidecar["targets_pos"][start, target]).astype(int)
        distance = grid_distance(layout, np.floor(positions[start]).astype(int), goal)
        path = float(np.sum(np.linalg.norm(np.diff(positions[start:end + 1], axis=0), axis=1)))
        kind = "revisit" if target in seen else "first_visit"
        seen.add(target)
        if not distance:
            excluded += 1
        else:
            in_view = bool(visible[start, target] > 0) if visible is not None else False
            records.append({
                "seed": seed, "layout_id": layout_id, "episode_id": episode["episode_id"],
                "target_id": target, "segment_id": index, "kind": kind,
                "match_bin": f"d{min(distance // DISTANCE_BIN_CELLS, MAX_DISTANCE_BIN)}_v{int(in_view)}",
                "success": bool(success), "oracle_distance": float(distance),
                "path_length": path, "steps": int(end - start),
            })
        start = end
    return records, excluded


def evaluate_navigation(agent, params, maze_size, resolution, seeds, *, interventions=("none",),
                        greedy=True, actor_factory=None, max_steps=None):
    """Frozen-policy episodes on policy-final-test seeds; telemetry is scoring-only."""
    from disco_spatial.artifacts import layout_fingerprint
    from disco_spatial.envs.memory_maze import make_memory_maze

    if any(not 2000000 <= seed < 3000000 for seed in seeds):
        raise ValueError("Navigation evaluation uses the policy-final-test seed namespace")
    started = time.monotonic()
    results = {}
    for intervention in interventions:
        episodes, segments, excluded = [], [], 0
        for seed in seeds:
            env = make_memory_maze(maze_size, seed, "test", resolution, evaluation=True,
                                   telemetry=True)
            try:
                actor = actor_factory() if actor_factory is not None else None
                episode = run_episode(agent, params, env, intervention=intervention, greedy=greedy,
                                      rng=np.random.default_rng(seed), actor=actor,
                                      max_steps=max_steps)
            finally:
                env.close()
            layout_id = layout_fingerprint(episode["sidecar"]["maze_layout"][0])
            rows, dropped = goal_segments(episode, seed, layout_id)
            segments.extend(rows)
            excluded += dropped
            rewards = episode["rewards"]
            half = len(rewards) // 2
            episodes.append({
                "seed": seed, "layout_id": layout_id, "episode_id": episode["episode_id"],
                "return": float(rewards.sum()), "steps": int(len(rewards)),
                "first_half_return": float(rewards[:half].sum()),
                "second_half_return": float(rewards[half:].sum()),
                "complete": episode["complete"],
            })
        revisits = revisit_efficiency(segments)
        by_kind = {kind: [row["steps"] for row in segments if row["kind"] == kind and row["success"]]
                   for kind in ("first_visit", "revisit")}
        results[intervention] = {
            "mean_return": float(np.mean([row["return"] for row in episodes])),
            "matched_revisit_efficiency_gain": revisits["mean_efficiency_gain"],
            "n_matched_pairs": revisits["n_matched_pairs"],
            "n_unmatched_segments": revisits["n_unmatched_segments"],
            "n_distance_zero_segments_excluded": excluded,
            "mean_successful_steps": {kind: float(np.mean(v)) if v else None
                                      for kind, v in by_kind.items()},
            "within_episode_return_change": float(np.mean(
                [row["second_half_return"] - row["first_half_return"] for row in episodes])),
            "episodes": episodes, "pairs": revisits["pairs"],
        }
    return {"schema_version": 1, "policy": "greedy" if greedy else "sampled",
            "seeds": list(seeds), "interventions": results,
            "wall_seconds": time.monotonic() - started,
            "interpretation": "Functional frozen-actor evaluation; matched pairs are correlated "
                              "within episodes and are not independent N."}
