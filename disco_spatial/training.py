"""Synchronous recurrent collection and conservative recent-sequence replay.

Chunks never include an ignored reset action. A LAST observation is used only
for bootstrap; its continuation discount is preserved. Replay reconstructs the
current recurrent state from the entire raw episode prefix.
"""

from pathlib import Path
from functools import partial
import json
import math
import time

import dm_env
from dm_env import specs
import jax
import jax.numpy as jnp
import numpy as np

from disco_spatial.agents import RecurrentAgent, DISCO_COMMIT, RULE_HASH
from disco_spatial.envs.memory_maze import make_memory_maze
from disco_spatial.evaluation.protocol import build_manifest
from disco_spatial.io import save_checkpoint, write_json, sha256
from disco_spatial.replay import SequenceReplay
from disco_spatial.trajectory import Trajectory, save_trajectory
from disco_rl import types


class EpisodeRecord:
    def __init__(self, env, timestep, method):
        self.env = env
        self.episode_id, self.instance_id = env.episode_id, env.instance_id
        self.rgb = [timestep.observation["rgb"].copy()]
        self.step_types = [int(timestep.step_type)]
        self.goal_segments = [env.goal_segment]
        self.actions, self.rewards, self.discounts = [], [], []
        self.logits, self.versions, self.terminated, self.truncated = [], [], [], []
        self.metadata = dict(env.metadata, collector="disco_spatial.training", method=method)
        self.saved = False

    def append(self, action, logits, version, timestep):
        discount = float(timestep.discount if timestep.discount is not None else 1)
        self.actions.append(int(action))
        self.logits.append(np.asarray(logits).copy())
        self.versions.append(int(version))
        self.rewards.append(float(timestep.reward or 0))
        self.discounts.append(discount)
        self.terminated.append(bool(timestep.last() and discount == 0))
        self.truncated.append(bool(timestep.last() and discount > 0))
        self.rgb.append(timestep.observation["rgb"].copy())
        self.step_types.append(int(timestep.step_type))
        self.goal_segments.append(self.env.goal_segment)

    def trajectory(self):
        return Trajectory(
            episode_id=self.episode_id, instance_id=self.instance_id, split=self.env.split,
            rgb=np.asarray(self.rgb, dtype=np.uint8), actions=np.asarray(self.actions, np.int32),
            rewards=np.asarray(self.rewards, np.float32),
            discounts=np.asarray(self.discounts, np.float32),
            behavior_logits=np.asarray(self.logits, np.float32).reshape(-1, 6),
            policy_versions=np.asarray(self.versions, np.int64),
            step_types=np.asarray(self.step_types, np.int8),
            goal_segments=np.asarray(self.goal_segments, np.int32),
            terminated=np.asarray(self.terminated, bool), truncated=np.asarray(self.truncated, bool),
            metadata=self.metadata,
        )


class BatchedEnvironment:
    """Add causal action/reward inputs to the strict RGB-only task boundary."""

    def __init__(self, envs, method="disco"):
        if not envs:
            raise ValueError("At least one environment is required")
        self.envs, self.method = list(envs), method
        self.batch_size, self.num_actions = len(envs), 6
        self.timesteps = [None] * len(envs)
        self.records = [None] * len(envs)
        self.previous_actions = np.full(len(envs), 6, np.int32)
        self.previous_rewards = np.zeros(len(envs), np.float32)

    def single_observation_spec(self):
        return {
            "rgb": self.envs[0].observation_spec()["rgb"],
            "previous_action": specs.Array((), np.int32),
            "previous_reward": specs.Array((), np.float32),
        }

    def single_action_spec(self):
        return specs.BoundedArray((), np.int32, 0, 5)

    def reset_finished(self):
        for i, env in enumerate(self.envs):
            if self.timesteps[i] is None or self.timesteps[i].last():
                self.timesteps[i] = env.reset()
                self.records[i] = EpisodeRecord(env, self.timesteps[i], self.method)
                self.previous_actions[i], self.previous_rewards[i] = 6, 0

    def timestep(self):
        return dm_env.TimeStep(
            step_type=jnp.asarray([int(ts.step_type) for ts in self.timesteps]),
            reward=jnp.asarray([float(ts.reward or 0) for ts in self.timesteps]),
            discount=jnp.asarray([float(ts.discount) if ts.discount is not None else 1.
                                  for ts in self.timesteps]),
            observation={
                "rgb": jnp.asarray(np.stack([ts.observation["rgb"] for ts in self.timesteps])),
                "previous_action": jnp.asarray(self.previous_actions),
                "previous_reward": jnp.asarray(self.previous_rewards),
            },
        )

    def step(self, actions, logits, version):
        for i, env in enumerate(self.envs):
            if self.timesteps[i].last():
                raise RuntimeError("Reset transitions cannot enter a learner rollout")
            next_ts = env.step(int(actions[i]))
            self.records[i].append(actions[i], logits[i], version, next_ts)
            self.timesteps[i] = next_ts
            self.previous_actions[i] = actions[i]
            self.previous_rewards[i] = float(next_ts.reward or 0)

    def close(self):
        for env in self.envs:
            env.close()


def collect_rollout(agent, params, state, env, key, length, policy_version, actor_step=None):
    env.reset_finished()
    initial_state = jax.tree.map(lambda x: x, state)
    call = actor_step or agent.actor_step
    records = []
    for _ in range(length):
        key, action_key = jax.random.split(key)
        actor, state = call(params, action_key, env.timestep(), state)
        env.step(np.asarray(actor.actions), np.asarray(actor.logits), policy_version)
        records.append(actor)
        if any(ts.last() for ts in env.timesteps):
            break
    key, bootstrap_key = jax.random.split(key)
    # Bootstrap is evaluated but is not consumed as the next actor's memory.
    bootstrap, _ = call(params, bootstrap_key, env.timestep(), state)
    stacked = jax.tree.map(lambda *xs: jnp.stack(xs), *records, bootstrap)
    return (types.ActorRollout.from_timestep(stacked), state, initial_state,
            key, len(records) * env.batch_size)


def trajectory_observations(trajectory):
    t = trajectory.n_steps
    return {
        "rgb": jnp.asarray(trajectory.rgb[:, None]),
        "previous_action": jnp.asarray(np.r_[6, trajectory.actions][:, None], jnp.int32),
        "previous_reward": jnp.asarray(np.r_[0., trajectory.rewards][:, None], jnp.float32),
        "is_first": jnp.asarray((np.arange(t + 1) == 0)[:, None]),
    }


PREFIX_BLOCKS = (256, 64, 16, 4, 1)


@partial(jax.jit, static_argnums=(0,))
def _prefix_unroll(agent, params, state, observations):
    # Returning only state lets XLA eliminate the unused prediction/value heads.
    return agent.policy_network.unroll(params, state, observations, observations["is_first"])[1]


def _observation_arrays(trajectory):
    t = trajectory.n_steps
    return {
        "rgb": trajectory.rgb[:, None],
        "previous_action": np.r_[6, trajectory.actions].astype(np.int32)[:, None],
        "previous_reward": np.r_[0., trajectory.rewards].astype(np.float32)[:, None],
        "is_first": (np.arange(t + 1) == 0)[:, None],
    }


def reconstruct_prefix_state(agent, params, trajectory, stop):
    """Hidden state immediately before observation[stop], under these weights.

    The raw prefix is unrolled in a few fixed block sizes so compilation is
    bounded while the recurrence is identical to step-by-step processing.
    """
    if not 0 <= stop <= trajectory.n_steps:
        raise ValueError("Prefix end is outside the episode")
    state = agent.initial_actor_state(jax.random.PRNGKey(0), batch_size=1)
    observations, t = _observation_arrays(trajectory), 0
    for block in PREFIX_BLOCKS:
        while stop - t >= block:
            part = {key: jnp.asarray(value[t:t + block]) for key, value in observations.items()}
            state = _prefix_unroll(agent, params, state, part)
            t += block
    return state


def replay_rollout(agent, params, chunk):
    """Recompute current state/outputs; keep immutable actual behavior logits."""
    trajectory = chunk.trajectory
    observations = trajectory_observations(trajectory)
    state = reconstruct_prefix_state(agent, params, trajectory, chunk.start)
    selected = jax.tree.map(lambda x: x[chunk.start:chunk.stop + 1], observations)
    current, _ = agent.policy_network.unroll(params, state, selected, selected["is_first"])
    behavior = jnp.asarray(chunk.behavior_logits[:, None])
    old_logits = jnp.concatenate((behavior, current["logits"][-1:]), axis=0)
    behavior_out = dict(current, logits=old_logits)
    s, e = chunk.start, chunk.stop
    incoming_reward = np.r_[0., trajectory.rewards][s:e + 1]
    incoming_discount = np.r_[1., trajectory.discounts][s:e + 1]
    rollout = types.ActorRollout(
        observations=selected,
        actions=jnp.asarray(np.r_[chunk.actions, 0][:, None], jnp.int32),
        rewards=jnp.asarray(incoming_reward[:, None], jnp.float32),
        discounts=jnp.asarray(incoming_discount[:, None], jnp.float32),
        agent_outs=behavior_out, logits=old_logits, states={},
    )
    log_pi = jax.nn.log_softmax(current["logits"][:-1], axis=-1)
    log_mu = jax.nn.log_softmax(behavior, axis=-1)
    action = jnp.asarray(chunk.actions[:, None, None])
    log_ratio = jnp.take_along_axis(log_pi - log_mu, action, axis=-1)[..., 0]
    # Clipping is for finite diagnostic reporting only. Eligibility below skips
    # excessive drift; the released learner still receives actual behavior logits.
    ratio = jnp.exp(jnp.clip(log_ratio, -20., 20.))
    diagnostics = {
        "replay_rho_mean": float(ratio.mean()), "replay_rho_max": float(ratio.max()),
        "replay_current_behavior_kl": float((jnp.exp(log_pi) * (log_pi - log_mu)).sum(-1).mean()),
        "replay_effective_sample_size": float(ratio.sum() ** 2 / jnp.maximum((ratio**2).sum(), 1e-8)),
        "replay_prefix_steps": chunk.start,
        "replay_abs_log_ratio_max": float(jnp.max(jnp.abs(log_ratio))),
        "replay_rho_reporting_clipped": bool(jnp.any(jnp.abs(log_ratio) > 20)),
    }
    return rollout, state, diagnostics


def stack_rollouts(parts):
    """Concatenate single-environment replay rollouts and states on the batch axis."""
    rollouts, states = zip(*parts)
    return (jax.tree.map(lambda *xs: jnp.concatenate(xs, axis=1), *rollouts),
            jax.tree.map(lambda *xs: jnp.concatenate(xs, axis=0), *states))


class MaturedChunks:
    """Each recent attempt chunk once, after its whole outcome window is observed.

    A chunk [s, s + L) matures when s + L + H - 1 <= T, or when its episode has
    ended. Waiting for the full window rather than the first reward avoids
    releasing successes earlier than failures. This is the near-on-policy
    "fresh rollout with lookahead" used by the feedback teacher; its age in
    policy versions is logged.
    """

    def __init__(self, length, horizon, limit):
        self.length, self.horizon, self.limit = length, horizon, limit
        self._cursor, self._ready, self.dropped = {}, [], 0

    def __len__(self):
        return len(self._ready)

    def observe(self, trajectories):
        for trajectory in trajectories:
            cursor = self._cursor.get(trajectory.episode_id, 0)
            while cursor + self.length <= trajectory.n_steps and (
                    trajectory.complete
                    or cursor + self.length + self.horizon - 1 <= trajectory.n_steps):
                self._ready.append((trajectory, cursor))
                cursor += self.length
            self._cursor[trajectory.episode_id] = cursor
        while len(self._ready) > self.limit:
            self._ready.pop(0)
            self.dropped += 1
        live = {t.episode_id for t in trajectories}
        self._cursor = {k: v for k, v in self._cursor.items()
                        if k in live or any(t.episode_id == k for t, _ in self._ready)}

    def pop(self, count):
        """Oldest ``count`` chunks with the newest trajectory snapshot, or None."""
        if len(self._ready) < count:
            return None
        taken, self._ready = self._ready[:count], self._ready[count:]
        return taken


def feedback_batch(agent, params, items, config, rng):
    """Jittable feedback batch for matured chunks of the student's own attempts."""
    from disco_spatial import feedback as fb
    from disco_spatial.replay import ReplayChunk

    parts, feedbacks, behavior = [], [], []
    for trajectory, start in items:
        stop = start + config.rollout_length
        rollout, state, _ = replay_rollout(agent, params, ReplayChunk(trajectory, start, stop, 0))
        parts.append((rollout, state))
        behavior.append(trajectory.behavior_logits[start:stop])
        feedbacks.append(fb.chunk_feedback(
            trajectory, start, stop, config.teacher_horizon,
            memory="success_memory" in agent.teacher_channels))
    rollout, state = stack_rollouts(parts)
    actual = fb.stack_feedback(feedbacks)
    train_inputs = fb.shuffled(actual, rng) if config.teacher_feedback == "shuffled" else actual

    def inputs(feedback, outcome=None):
        return {"outcome": jnp.asarray(feedback.outcome if outcome is None else outcome, jnp.float32),
                "memory_rgb": jnp.asarray(feedback.memory_rgb),
                "memory_valid": jnp.asarray(feedback.memory_valid),
                "memory_elapsed": jnp.asarray(feedback.memory_elapsed, jnp.float32)}

    known = actual.known[:-1]
    success = actual.outcome[:-1] > 0.5
    if config.teacher_query == "counterfactual_success":
        query = inputs(train_inputs, np.ones_like(train_inputs.outcome))
        mask = np.ones_like(known)  # outcome=1 is a constant query: causal on every step
    else:
        query, mask = inputs(train_inputs), known & success
    return {
        "rollout": rollout, "start_state": state,
        "behavior_logits": jnp.asarray(np.stack(behavior, axis=1), jnp.float32),
        "train": inputs(train_inputs), "query": query,
        "shuffled": inputs(fb.shuffled(actual, rng)),
        "known": jnp.asarray(known), "success": jnp.asarray(success),
        "distill_mask": jnp.asarray(mask), "returns": jnp.asarray(actual.returns[:-1], jnp.float32),
    }


class TeacherGate:
    """Evidence gate: distill only after the teacher beats its own shuffled control.

    Diagnostics are held out (computed before the teacher trains on each chunk)
    and accumulated weighted by the steps they describe: information gain of
    feedback and its shuffled control over successful steps, local improvement
    and copy rate over known steps. Sums decay by ``decay`` per update so the
    evidence tracks the current teacher. Successes are rare early in Memory
    Maze; a per-update average would be dominated by chunks without any success,
    so opening also requires ``teacher_min_success_steps`` effective successful
    steps. The gate closes again if the margin or improvement turns negative.
    """

    WEIGHTS = {"teacher_information_gain_success": "teacher_success_steps",
               "teacher_shuffled_information_gain_success": "teacher_success_steps",
               "teacher_local_improvement": "teacher_known_steps",
               "teacher_copy_rate": "teacher_known_steps"}

    def __init__(self, config, decay=0.98):
        self.config, self.decay = config, decay
        self.mode, self.observed = config.teacher_gate, 0
        self.sums = dict.fromkeys(self.WEIGHTS, 0.0)
        self.weights = dict.fromkeys(self.WEIGHTS, 0.0)
        self.open = self.mode == "always"
        self.opened_at, self.transitions = None, []

    def evidence(self):
        return {key: self.sums[key] / self.weights[key] if self.weights[key] > 0 else None
                for key in self.WEIGHTS}

    def observe(self, logs, updates):
        for key, weight_key in self.WEIGHTS.items():
            weight = float(logs[weight_key])
            self.sums[key] = self.decay * self.sums[key] + weight * float(logs[key])
            self.weights[key] = self.decay * self.weights[key] + weight
        self.observed += 1
        evidence = self.evidence()
        if self.mode != "evidence" or None in evidence.values():
            return
        gain = evidence["teacher_information_gain_success"]
        margin = gain - evidence["teacher_shuffled_information_gain_success"]
        improvement = evidence["teacher_local_improvement"]
        copy = evidence["teacher_copy_rate"]
        threshold = self.config.teacher_min_information_gain
        ready = (self.observed >= self.config.teacher_warmup_updates
                 and self.weights["teacher_information_gain_success"]
                 >= self.config.teacher_min_success_steps
                 and gain >= threshold and margin >= threshold and improvement > 0
                 and copy <= self.config.teacher_max_copy_rate)
        failing = margin < 0 or improvement < 0 or copy > self.config.teacher_max_copy_rate
        if not self.open and ready:
            self.open = True
            self.opened_at = self.opened_at if self.opened_at is not None else updates
            self.transitions.append(("open", updates))
        elif self.open and failing:
            self.open = False
            self.transitions.append(("close", updates))

    def distillation_weight(self):
        return self.config.distillation_weight if self.open and self.mode != "never" else 0.0

    def state(self):
        return {"teacher_gate_mode": self.mode, "teacher_gate_open": self.open,
                "teacher_gate_evidence": self.evidence(), "teacher_gate_observed": self.observed,
                "teacher_gate_effective_success_steps":
                    self.weights["teacher_information_gain_success"]}


def evaluate(agent, params, config, seeds=None, reset_memory=False, teacher_head=None):
    """Whole official episodes with no evaluation sidecars.

    With ``teacher_head`` the teacher acts: student logits plus delta with the
    constant query outcome=1 and causal success memory from the episode itself.
    This is valid only because that query uses no hindsight information.
    """
    from disco_spatial.evaluation.navigation import run_episode

    seeds = tuple(config.validation_seeds if seeds is None else seeds)
    returns, lengths = [], []
    started = time.monotonic()
    for seed in seeds:
        env = make_memory_maze(config.maze_size, seed, "validation", config.camera_resolution)
        try:
            actor = None if teacher_head is None else teacher_actor(agent, params, teacher_head)
            episode = run_episode(agent, params, env, actor=actor,
                                  intervention="reset_every_step" if reset_memory else "none")
        finally:
            env.close()
        returns.append(float(episode["rewards"].sum()))
        lengths.append(int(len(episode["rewards"])))
    return {"seeds": list(seeds), "returns": returns, "lengths": lengths,
            "mean_return": float(np.mean(returns)) if returns else None,
            "wall_seconds": time.monotonic() - started, "memory_reset": reset_memory,
            "policy": "greedy_teacher_success_query" if teacher_head is not None else "greedy",
            "sidecar_used": False}


def teacher_actor(agent, params, head):
    """Acting hook for run_episode: causal teacher with query outcome=1."""
    from disco_spatial import feedback as fb
    from disco_spatial import teacher as teacher_lib

    apply = agent.jitted("teacher_delta", lambda p, h, b, f: teacher_lib.delta_logits(
        agent.teacher, p, h, b, f))
    latest = {}

    def act(output, history):
        frames = history["rgb"]
        t = len(frames) - 1
        if t > 0 and history["rewards"][-1] == 1:
            latest[int(fb.border_color_keys(frames[t - 1][None])[0])] = (frames[t - 1], t - 1)
        remembered = latest.get(int(fb.border_color_keys(frames[t][None])[0]))
        memory = np.zeros((1, 1) + frames[t].shape, np.uint8)
        if remembered is not None:
            memory[0, 0] = remembered[0]
        feedback = {
            "outcome": jnp.ones((1, 1)), "memory_rgb": jnp.asarray(memory),
            "memory_valid": jnp.asarray([[remembered is not None]]),
            "memory_elapsed": jnp.asarray([[0.0 if remembered is None else
                                            (t - remembered[1]) / fb.MEMORY_ELAPSED_SCALE]]),
        }
        delta = apply(params, head, output["belief"][None], feedback)
        return np.asarray(output["logits"][0] + delta[0, 0])

    return act


def train(config, env_factory=make_memory_maze):
    from disco_spatial import teacher as teacher_lib

    output = Path(config.output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use a new dedicated output directory; exact environment resume is not implemented")
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "config.json", config.to_dict())
    write_json(output / "protocol.json", build_manifest(config.to_dict()))
    batch = BatchedEnvironment([
        env_factory(config.maze_size, config.seed * 1000 + worker,
                    "train", config.camera_resolution)
        for worker in range(config.num_envs)
    ], config.method)
    agent = RecurrentAgent(config, batch)
    key = jax.random.PRNGKey(config.seed)
    learner = agent.initial_learner_state(key)
    state = agent.initial_actor_state(key)
    rule = agent.rule_params()
    replay = SequenceReplay(config.replay_capacity_episodes, config.seed)
    teacher_on = agent.teacher is not None
    teacher_ema = teacher_lib.teacher_params(learner.params) if teacher_on else None
    matured = MaturedChunks(config.rollout_length, config.teacher_horizon, 4 * config.num_envs)
    gate = TeacherGate(config) if teacher_on else None
    feedback_rng = np.random.default_rng(config.seed + 17)
    actor_step = jax.jit(agent.actor_step)
    update = jax.jit(agent.learner_step_targets)
    mixed = jax.jit(agent.learner_step_mixed, static_argnames=("replay_weight",))
    composite = jax.jit(agent.learner_step_composite, static_argnames=("replay_weight",))
    transitions, updates, best = 0, 0, -float("inf")
    forward_steps, distilled_updates = 0, 0
    started = time.monotonic()
    source_root = Path(__file__).resolve().parents[1]
    source_hashes = {str(path.relative_to(source_root)): sha256(path)
                     for path in sorted((source_root / "disco_spatial").rglob("*.py"))}
    write_json(output / "provenance.json", {
        "disco_commit": DISCO_COMMIT, "rule_sha256": RULE_HASH,
        "devices": [str(device) for device in jax.devices()], "jax_version": jax.__version__,
        "environment": batch.envs[0].metadata,
        "training_seed_namespace": [config.seed * 1000 + i for i in range(config.num_envs)],
        "learner_observations": ["rgb", "previous_action", "previous_reward", "is_first"],
        "geometry_supervision": False, "feedback_teacher_enabled": teacher_on,
        "teacher_inputs": (["stopgrad_student_belief", "own_reward_outcome_within_horizon"]
                           + (["own_rgb_frame_of_previous_same_color_success", "elapsed_steps"]
                              if "success_memory" in agent.teacher_channels else [])
                           if teacher_on else []),
        "teacher_privileged_inputs": False,
        "replay_state": "current parameters, full raw episode-prefix recomputation",
        "online_state": "truncated-BPTT carry from actual collector; FIRST resets",
        "recurrent_target_state": "EMA weights independently reprocess each raw episode prefix",
        "exact_recurrent_target_state": True,
        "chunk_boundary": "stop at any LAST; bootstrap with original continuation discount",
        "exact_environment_resume": False,
        "source_sha256": source_hashes,
    })

    def checkpoint(name):
        save_checkpoint(output / name, jax.device_get({
            "schema_version": 2, "config": config.to_dict(), "learner": learner,
            "teacher_ema": teacher_ema, "gate": gate.state() if gate else None,
            "updates": updates, "transitions": transitions, "rng": key,
            "rule_sha256": RULE_HASH, "resume_supported": False,
            "checkpoint_scope": "policy/optimizer evaluation snapshot; simulator/replay state not included",
        }))

    def save_record(record, partial=False):
        if not config.save_trajectories or record.saved or not record.actions:
            return
        import hashlib

        tag = hashlib.sha256(record.episode_id.encode()).hexdigest()[:16]
        name = f"{'partial' if partial else 'episode'}-{tag}.npz"
        save_trajectory(record.trajectory(), output / "trajectories" / name)
        record.saved = True

    checkpoint("init.pkl")  # untrained-encoder control with the same initialization
    try:
        with (output / "metrics.jsonl").open("w") as log:
            while transitions < config.steps:
                length = min(config.rollout_length,
                             max(1, (config.steps - transitions) // config.num_envs))
                fresh, state, start_state, key, count = collect_rollout(
                    agent, learner.params, state, batch, key, length, updates, actor_step)
                chunk_length = count // config.num_envs
                forward_steps += 2 * (chunk_length + 1) * config.num_envs
                diagnostics = {"replay_used": False}
                old, old_state = None, None
                if config.recipe == "recent_replay" and config.replay_weight > 0 and len(replay):
                    try:
                        chunk = replay.sample(
                            chunk_length, current_policy_version=updates,
                            max_replay_age=config.replay_max_age)[0]
                    except ValueError:
                        chunk = None
                    if chunk is not None:
                        old, old_state, replay_logs = replay_rollout(agent, learner.params, chunk)
                        forward_steps += chunk.start + 2 * (chunk.n_steps + 1)
                        finite = all(math.isfinite(v) for v in replay_logs.values()
                                     if isinstance(v, (int, float)))
                        if not finite:
                            old = None
                            diagnostics["replay_skip_reason"] = "nonfinite_policy_diagnostics"
                        elif replay_logs["replay_abs_log_ratio_max"] > config.replay_max_abs_log_ratio:
                            old = None
                            diagnostics.update(replay_logs, replay_skip_reason="excessive_policy_drift")
                        else:
                            diagnostics.update(replay_logs, replay_used=True)
                            diagnostics["replay_mean_age"] = float((updates - chunk.policy_versions).mean())
                key, update_key = jax.random.split(key)
                if agent.method == "disco":
                    ema_params = learner.meta_state["target_params"]
                    target_states = [reconstruct_prefix_state(
                        agent, ema_params, record.trajectory(),
                        len(record.actions) - chunk_length) for record in batch.records]
                    forward_steps += sum(len(r.actions) - chunk_length for r in batch.records)
                    target_start = jax.tree.map(
                        lambda *parts: jnp.concatenate(parts, axis=0), *target_states)
                    replay_target_start = None
                    if old is not None:
                        replay_target_start = reconstruct_prefix_state(
                            agent, ema_params, chunk.trajectory, chunk.start)
                        forward_steps += chunk.start
                else:
                    target_start, replay_target_start = start_state, None
                imitation = None
                if config.self_imitation_weight > 0 and len(replay):
                    try:
                        sil_chunk = replay.sample(
                            chunk_length, current_policy_version=updates,
                            max_replay_age=config.self_imitation_max_age, require_reward=True)[0]
                    except ValueError:
                        sil_chunk = None
                    if sil_chunk is not None:
                        sil_rollout, sil_state, _ = replay_rollout(agent, learner.params, sil_chunk)
                        imitation = {"rollout": sil_rollout, "start_state": sil_state}
                        forward_steps += sil_chunk.start + 2 * (sil_chunk.n_steps + 1)
                        diagnostics["self_imitation_age"] = float(
                            (updates - sil_chunk.policy_versions).mean())
                feedback, distill_weight = None, 0.0
                if teacher_on:
                    items = matured.pop(config.num_envs)
                    if items is not None:
                        feedback = feedback_batch(agent, learner.params, items, config, feedback_rng)
                        distill_weight = gate.distillation_weight()
                        forward_steps += sum(start + 2 * (config.rollout_length + 1)
                                             for _, start in items)
                        diagnostics["feedback_chunk_age"] = float(np.mean(
                            [updates - t.policy_versions[s] for t, s in items]))
                weights = {"self_imitation": jnp.float32(config.self_imitation_weight),
                           "distillation": jnp.float32(distill_weight),
                           "teacher": jnp.float32(config.teacher_learning_weight)}
                if imitation is not None or feedback is not None:
                    learner, _, metrics = composite(
                        update_key, fresh, old, learner, start_state, old_state, rule,
                        replay_weight=config.replay_weight if old is not None else 0.0,
                        fresh_target_start_state=target_start,
                        replay_target_start_state=replay_target_start,
                        self_imitation=imitation, feedback=feedback,
                        teacher_ema=teacher_ema, weights=weights)
                elif old is None:
                    learner, _, metrics = update(
                        update_key, fresh, learner, start_state, target_start, rule)
                else:
                    learner, _, metrics = mixed(
                        update_key, fresh, old, learner, start_state, old_state, rule,
                        replay_weight=config.replay_weight,
                        fresh_target_start_state=target_start,
                        replay_target_start_state=replay_target_start)
                if teacher_on:
                    teacher_ema = teacher_lib.ema_update(
                        teacher_ema, teacher_lib.teacher_params(learner.params),
                        config.teacher_ema_rate)
                transitions += count
                updates += 1
                loss = float(metrics["total_loss"])
                if not np.isfinite(loss):
                    raise FloatingPointError("Nonfinite learner loss")
                snapshots = []
                for record, ts in zip(batch.records, batch.timesteps):
                    snapshot = record.trajectory()
                    replay.add(snapshot)
                    snapshots.append(snapshot)
                    if ts.last():
                        save_record(record)
                if teacher_on:
                    matured.observe(snapshots)
                probabilities = jax.nn.softmax(fresh.agent_outs["logits"][:-1], axis=-1)
                record = {
                    "updates": updates, "transitions": transitions, "loss": loss,
                    "gradient_norm": float(metrics["global_gradient_norm"]),
                    "policy_entropy": float((-probabilities * jnp.log(jnp.maximum(probabilities, 1e-8))).sum(-1).mean()),
                    "action_counts": np.bincount(np.asarray(fresh.actions[:-1]).ravel(), minlength=6).tolist(),
                    "reward_events_this_chunk": int(np.asarray(fresh.rewards[1:]).sum()),
                    "replayed_transitions_this_update": 0 if old is None else int(old.actions.shape[0] - 1),
                    "learner_network_steps": forward_steps,
                    "wall_seconds": time.monotonic() - started, **diagnostics,
                }
                for name in ("self_imitation_loss", "self_imitation_positive_fraction",
                             "rule_gradient_norm", "auxiliary_gradient_norm"):
                    if name in metrics:
                        record[name] = float(metrics[name])
                if feedback is not None:
                    record.update({name: float(value) for name, value in metrics.items()
                                   if name.startswith(("teacher_", "distill"))})
                    gate.observe(metrics, updates)
                    record["distillation_weight_applied"] = distill_weight
                    distilled_updates += int(distill_weight > 0)
                if gate is not None:
                    record.update(gate.state())
                    record["matured_chunks_dropped"] = matured.dropped
                due = updates % config.eval_every == 0 or transitions >= config.steps
                if due and config.validation_seeds:
                    validation = evaluate(agent, learner.params, config)
                    record["validation"] = validation
                    if teacher_on:
                        record["teacher_validation"] = evaluate(
                            agent, learner.params, config, teacher_head=teacher_ema)
                    if validation["mean_return"] > best:
                        best = validation["mean_return"]
                        checkpoint("best.pkl")
                log.write(json.dumps(record, allow_nan=False) + "\n")
                log.flush()
                print(json.dumps(record, allow_nan=False), flush=True)
                checkpoint("latest.pkl")
            for record in batch.records:
                save_record(record, partial=record.step_types[-1] != dm_env.StepType.LAST)
        result = {
            "purpose": config.purpose, "transitions": transitions, "updates": updates,
            "wall_seconds": time.monotonic() - started,
            "validation_performed": bool(config.validation_seeds),
            "best_validation_return": best if np.isfinite(best) else None,
            "latest_checkpoint_sha256": sha256(output / "latest.pkl"),
            "init_checkpoint_sha256": sha256(output / "init.pkl"),
            "learner_network_steps": forward_steps,
            "learner_network_steps_definition": "approximate count of per-observation network "
                                                "evaluations (forward or forward+backward) by the learner",
            "scientific_result": False,
            "sdpo_teacher_trained": teacher_on,
            "distillation_updates": distilled_updates,
            "teacher_gate": gate.state() if gate else None,
            "teacher_gate_transitions": gate.transitions if gate else [],
        }
        write_json(output / "completed.json", result)
        return result
    finally:
        batch.close()
