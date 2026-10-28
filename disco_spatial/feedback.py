"""Hindsight feedback built from the agent's own RGB and reward stream only.

Two channels feed the SDPO-style self-teacher:

``outcome``         Whether a +1 target reward arrives within ``horizon`` steps
                    of observation t. It is hindsight about the attempt itself.
                    Steps whose window is not yet observed are censored.
``success_memory``  The frame from the most recent earlier success on the
                    currently requested target in the same episode, and how
                    long ago it was. The requested target is read from the
                    color border of the RGB the student also sees. It is causal.

No simulator state, map, pose or target position is used. Memory Maze builds a
new maze at every reset, so "the same task instance" is one episode and one
target color. An attempt is never its own feedback.
"""

from dataclasses import dataclass

import numpy as np

from disco_spatial.evaluation.telemetry import border_width


MEMORY_ELAPSED_SCALE = 1000.0


def border_color_keys(rgb):
    """Integer key for the requested target color, from the RGB border only."""
    rgb = np.asarray(rgb)
    width = max(1, border_width(rgb.shape[1]))
    border = np.concatenate((rgb[:, :width].reshape(len(rgb), -1, 3),
                             rgb[:, -width:].reshape(len(rgb), -1, 3)), axis=1)
    quantized = np.round(np.median(border, axis=1) / 16).astype(np.int64)
    return quantized[:, 0] * 10000 + quantized[:, 1] * 100 + quantized[:, 2]


def outcome_labels(rewards, horizon, complete):
    """Per observation t < T: success within horizon, steps to reward, censored flag.

    Transition t earns rewards[t]; reward at transition t + k - 1 means the
    target is reached k steps after observation t. If no reward is seen and the
    remaining horizon extends past the recorded data of an unfinished episode,
    the outcome is unknown (censored). A finished episode's shortfall is a
    genuine failure: the time limit ended the attempt.
    """
    rewards = np.asarray(rewards)
    steps = len(rewards)
    success = np.zeros(steps, bool)
    distance = np.full(steps, horizon + 1, np.int32)
    censored = np.zeros(steps, bool)
    reward_times = np.flatnonzero(rewards == 1)
    for t in range(steps):
        later = reward_times[(reward_times >= t) & (reward_times < t + horizon)]
        if later.size:
            success[t], distance[t] = True, int(later[0] - t + 1)
        elif not complete and t + horizon > steps:
            censored[t] = True
    return success, distance, censored


def success_memory(rgb, rewards):
    """Index of the remembered success frame per observation, and elapsed steps.

    A success on transition s was for the target shown in frame s; frame s is
    remembered (the approach view). It becomes available from observation s + 1.
    """
    keys = border_color_keys(rgb)
    rewards = np.asarray(rewards)
    frames = len(rgb)
    index = np.full(frames, -1, np.int64)
    elapsed = np.zeros(frames, np.float32)
    latest = {}
    for t in range(frames):
        if t > 0 and rewards[t - 1] == 1:
            latest[int(keys[t - 1])] = t - 1
        remembered = latest.get(int(keys[t]))
        if remembered is not None:
            index[t], elapsed[t] = remembered, t - remembered
    return index, elapsed


@dataclass(frozen=True)
class Feedback:
    """Arrays for one chunk of T transitions plus its bootstrap observation."""

    outcome: np.ndarray        # [T+1] float, 1 if success within horizon
    known: np.ndarray          # [T+1] bool, outcome observed (not censored, not bootstrap)
    returns: np.ndarray        # [T+1] float, horizon-truncated discounted return
    memory_rgb: np.ndarray     # [T+1, H, W, 3] uint8 (zeros where absent)
    memory_valid: np.ndarray   # [T+1] bool
    memory_elapsed: np.ndarray  # [T+1] float, scaled


def chunk_feedback(trajectory, start, stop, horizon, discount=0.997, memory=True):
    """Feedback for observations start..stop of a raw episode prefix."""
    rewards = trajectory.rewards
    success, _, censored = outcome_labels(rewards, horizon, trajectory.complete)
    length = stop - start + 1
    outcome = np.zeros(length, np.float32)
    known = np.zeros(length, bool)
    returns = np.zeros(length, np.float32)
    weights = discount ** np.arange(horizon)
    for i, t in enumerate(range(start, stop)):
        outcome[i] = success[t]
        known[i] = not censored[t]
        window = rewards[t:t + horizon]
        returns[i] = float(np.dot(window, weights[:len(window)]))
    shape = trajectory.rgb.shape[1:]
    memory_rgb = np.zeros((length,) + shape, np.uint8)
    memory_valid = np.zeros(length, bool)
    memory_elapsed = np.zeros(length, np.float32)
    if memory:
        index, elapsed = success_memory(trajectory.rgb[:stop + 1], rewards[:stop])
        for i, t in enumerate(range(start, stop + 1)):
            if index[t] >= 0 and t < stop:
                memory_rgb[i] = trajectory.rgb[index[t]]
                memory_valid[i] = True
                memory_elapsed[i] = elapsed[t] / MEMORY_ELAPSED_SCALE
    return Feedback(outcome, known, returns, memory_rgb, memory_valid, memory_elapsed)


def stack_feedback(items):
    """Stack per-chunk feedback along a new batch axis: [T+1, B, ...]."""
    return Feedback(*(np.stack([getattr(item, name) for item in items], axis=1)
                      for name in Feedback.__dataclass_fields__))


TEACHER_INPUTS = ("outcome", "memory_rgb", "memory_valid", "memory_elapsed")


def shuffled(feedback, rng):
    """Control: permute teacher inputs across all (time, batch) positions jointly.

    The marginal feedback distribution is kept while its pairing with the
    student's history and actions is destroyed. Censoring masks and returns
    describe the step itself and stay in place, so the supervised positions
    and the diagnostic returns are unchanged.
    """
    flat = feedback.outcome[:-1].size  # the bootstrap row carries no feedback
    order = rng.permutation(flat)

    def permute(array):
        body = array[:-1]
        mixed = body.reshape((flat,) + body.shape[2:])[order].reshape(body.shape)
        return np.concatenate((mixed, array[-1:]), axis=0)

    return Feedback(**{name: permute(getattr(feedback, name)) if name in TEACHER_INPUTS
                       else getattr(feedback, name) for name in Feedback.__dataclass_fields__})
