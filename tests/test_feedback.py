"""Hindsight feedback is built from own rewards/RGB, is censored honestly and is causal."""

import numpy as np

from disco_spatial import feedback as fb
from disco_spatial.trajectory import Trajectory


def frames(colors, size=64):
    """Uniform scene with a colored border per frame, as Memory Maze draws it."""
    rgb = np.full((len(colors), size, size, 3), 50, np.uint8)
    width = fb.border_width(size)
    for i, color in enumerate(colors):
        rgb[i, :, :width] = rgb[i, :, -width:] = color
        rgb[i, :width] = rgb[i, -width:] = color
        rgb[i, size // 2, size // 2] = i  # distinguish frames
    return rgb


def trajectory(rewards, colors, complete=True):
    n = len(rewards)
    rewards = np.asarray(rewards, np.float32)
    types = np.ones(n + 1, np.int8)
    types[0] = 0
    if complete:
        types[-1] = 2
    truncated = np.zeros(n, bool)
    truncated[-1] = complete
    return Trajectory(
        episode_id="train:e", instance_id="i", split="train", rgb=frames(colors),
        actions=np.zeros(n, np.int32), rewards=rewards, discounts=np.ones(n, np.float32),
        behavior_logits=np.zeros((n, 6), np.float32), policy_versions=np.zeros(n, np.int64),
        step_types=types, goal_segments=np.r_[0, np.cumsum(rewards)].astype(np.int32),
        terminated=np.zeros(n, bool), truncated=truncated, metadata={})


RED, BLUE = (170, 30, 20), (20, 90, 200)


def test_border_keys_read_only_the_requested_color_border():
    rgb = frames([RED, RED, BLUE])
    keys = fb.border_color_keys(rgb)
    assert keys[0] == keys[1] != keys[2]
    changed = rgb.copy()
    changed[:, 10:50, 10:50] = 255  # scene content does not change the key
    np.testing.assert_array_equal(fb.border_color_keys(changed), keys)


def test_outcome_labels_censor_unobserved_windows_but_not_finished_episodes():
    rewards = np.array([0, 0, 1, 0, 0, 0])
    success, distance, censored = fb.outcome_labels(rewards, horizon=3, complete=False)
    assert success.tolist() == [True, True, True, False, False, False]
    assert distance[:3].tolist() == [3, 2, 1]
    # t=3 sees rewards 3..5 (all observed) -> genuine failure; t=4,5 lack data.
    assert censored.tolist() == [False, False, False, False, True, True]
    _, _, finished = fb.outcome_labels(rewards, horizon=3, complete=True)
    assert not finished.any()


def test_success_memory_is_causal_same_color_and_never_the_attempt_itself():
    # Transition 1 is rewarded for RED (frame 1 shows RED); frames 2.. request BLUE, then RED.
    colors = [RED, RED, BLUE, BLUE, RED, RED]
    rewards = [0, 1, 0, 1, 0]
    index, elapsed = fb.success_memory(frames(colors), np.array(rewards))
    assert index[:2].tolist() == [-1, -1]  # nothing remembered before the success is observed
    assert index[2] == -1 and index[3] == -1  # BLUE has no earlier success
    assert index[4] == 1 and elapsed[4] == 3  # RED remembers frame 1 (approach view)
    assert index[5] == 1
    # Changing the future cannot change what is remembered at earlier times.
    alternative = fb.success_memory(frames(colors[:4] + [BLUE, BLUE]), np.array(rewards))[0]
    np.testing.assert_array_equal(alternative[:4], index[:4])


def test_chunk_feedback_and_shuffle_keep_step_masks_and_returns():
    t = trajectory([0, 1, 0, 0, 1, 0], [RED, RED, BLUE, BLUE, BLUE, RED, RED], complete=False)
    item = fb.chunk_feedback(t, 0, 4, horizon=2)
    assert item.outcome.tolist() == [1, 1, 0, 1, 0]  # bootstrap row has no feedback
    assert item.known.tolist() == [True, True, True, True, False]
    np.testing.assert_allclose(item.returns[:4], [0.997, 1, 0, 0.997], rtol=1e-6)
    batch = fb.stack_feedback([item, fb.chunk_feedback(t, 1, 5, horizon=2)])
    mixed = fb.shuffled(batch, np.random.default_rng(0))
    np.testing.assert_array_equal(mixed.known, batch.known)
    np.testing.assert_array_equal(mixed.returns, batch.returns)
    assert sorted(mixed.outcome[:-1].ravel()) == sorted(batch.outcome[:-1].ravel())
    np.testing.assert_array_equal(mixed.outcome[-1], batch.outcome[-1])
