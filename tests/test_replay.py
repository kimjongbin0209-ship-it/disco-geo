from dataclasses import replace
import numpy as np
import pytest
from disco_spatial.trajectory import Trajectory, save_trajectory, load_trajectory
from disco_spatial.replay import SequenceReplay, FeedbackMemory


def episode(episode_id="e0", instance_id="layout0", split="train", n=5, complete=True):
    reward = np.zeros(n, np.float32)
    reward[1] = 1
    goal = np.r_[0, np.cumsum(reward).astype(np.int32)]
    types = np.ones(n + 1, np.int8)
    types[0] = 0
    if complete:
        types[-1] = 2
    trunc = np.zeros(n, bool)
    trunc[-1] = complete
    return Trajectory(episode_id=episode_id, instance_id=instance_id, split=split,
                      rgb=np.broadcast_to(np.arange(n + 1, dtype=np.uint8)[:, None, None, None],
                                          (n + 1, 8, 8, 3)),
                      actions=np.arange(n, dtype=np.int32) % 6, rewards=reward,
                      discounts=np.ones(n, np.float32),
                      behavior_logits=np.arange(n * 6, dtype=np.float64).reshape(n, 6) / 7,
                      policy_versions=np.arange(n, dtype=np.int64), step_types=types,
                      goal_segments=goal, terminated=np.zeros(n, bool), truncated=trunc,
                      metadata={"environment": "memory-maze", "seed": 10})


def test_raw_trajectory_safe_roundtrip_and_behavior_is_immutable(tmp_path):
    trajectory = episode()
    original = trajectory.behavior_probabilities().copy()
    with pytest.raises(ValueError):
        trajectory.behavior_logits[0, 0] = 1000
    with pytest.raises(ValueError):
        trajectory.behavior_logits.setflags(write=True)
    path = save_trajectory(trajectory, tmp_path / "trajectory.npz")
    saved = load_trajectory(path)
    np.testing.assert_array_equal(original, saved.behavior_probabilities())
    assert saved.behavior_logits.dtype == np.float64
    assert saved.complete and saved.truncated[-1] and saved.discounts[-1] == 1
    with pytest.raises(FileExistsError):
        save_trajectory(trajectory, path)


def test_schema_rejects_object_pickle_and_extra_eval_labels(tmp_path):
    original = episode()
    with pytest.raises(ValueError, match="sidecar"):
        replace(original, metadata={"maze_layout": [[1, 2]]})
    with pytest.raises(ValueError, match="flat provenance"):
        replace(original, metadata={"collector": {"agent_pos": [1, 2]}})
    with pytest.raises(ValueError, match="non-object"):
        replace(original, actions=np.array([None] * 5, object))
    path = tmp_path / "unsafe.npz"
    np.savez(path, header=np.array({"schema_version": 1}, object))
    with pytest.raises(ValueError):
        load_trajectory(path)
    path2 = tmp_path / "extra.npz"
    save_trajectory(original, tmp_path / "good.npz")
    with np.load(tmp_path / "good.npz", allow_pickle=False) as f:
        np.savez(path2, **dict(f), agent_pos=np.array([1, 2]))
    with pytest.raises(ValueError, match="Unexpected NPZ"):
        load_trajectory(path2)


def test_schema_validates_state_action_alignment_and_terminal_semantics():
    original = episode()
    with pytest.raises(ValueError, match="T\\+1"):
        replace(original, rgb=original.rgb[:-1])
    with pytest.raises(ValueError, match="episode boundaries"):
        replace(original, step_types=np.array([0, 1, 2, 0, 1, 2]))
    with pytest.raises(ValueError, match="zero continuation"):
        replace(original, truncated=np.zeros(5, bool), terminated=np.array([0, 0, 0, 0, 1], bool))
    with pytest.raises(ValueError, match="advance"):
        replace(original, goal_segments=np.zeros(6, int))


def test_sequence_replay_is_contiguous_has_causal_prefix_and_age_filter():
    replay = SequenceReplay(2, seed=4)
    replay.add(episode())
    chunks = replay.sample(2, batch_size=10, burn_in=2, current_policy_version=4,
                           max_replay_age=2)
    for chunk in chunks:
        assert chunk.start >= 2 and chunk.stop - chunk.start == 2
        assert len(chunk.rgb) == len(chunk.actions) + 1
        np.testing.assert_array_equal(chunk.rgb[:-1, 0, 0, 0], chunk.actions)
        assert chunk.prefix[-1, 0, 0, 0] == chunk.start
        assert len(chunk.prefix_actions) == len(chunk.prefix_rewards) == chunk.start
        assert len(chunk.burn_in_actions) <= 2
        assert chunk.burn_in_rgb[-1, 0, 0, 0] == chunk.start
    with pytest.raises(ValueError, match="No contiguous"):
        replay.sample(2, current_policy_version=20, max_replay_age=1)
    with pytest.raises(ValueError, match="train"):
        replay.add(episode(split="test"))


def test_episode_prefix_extension_preserves_raw_history_and_old_behavior():
    full = episode(complete=False)
    fields = {name: getattr(full, name)[:3 if name in ("rgb", "step_types", "goal_segments") else 2]
              for name in ("rgb", "step_types", "goal_segments", "actions", "rewards", "discounts",
                           "behavior_logits", "policy_versions", "terminated", "truncated")}
    short = replace(full, **fields)
    replay = SequenceReplay(2)
    replay.add(short)
    replay.add(full)
    changed = full.behavior_logits.copy()
    changed[0] += 1
    other = replace(full, behavior_logits=changed)
    replay2 = SequenceReplay(2)
    replay2.add(short)
    with pytest.raises(ValueError, match="behavior"):
        replay2.add(other)


def test_feedback_is_same_instance_same_goal_and_never_unseen_test():
    memory = FeedbackMemory()
    memory.add(episode())
    found = memory.query(instance_id="layout0", goal_segment=0)
    assert len(found) == 1
    assert found[0].rewards[-1] == 1
    assert len(found[0].rgb) == len(found[0].actions) + 1
    assert memory.query(instance_id="new-unseen-layout", goal_segment=0) == []
    assert memory.query(instance_id="layout0", goal_segment=1) == []
    with pytest.raises(ValueError, match="validation/test"):
        memory.query(instance_id="layout0", goal_segment=0, split="test")
    with pytest.raises(ValueError, match="train"):
        memory.add(episode(split="validation"))
    assert memory.retrieve(episode(), 0) == []  # own successful attempt excluded by default
    other = episode(episode_id="retry", instance_id="layout0")
    assert len(memory.retrieve(other, 0)) == 1


def test_feedback_rejects_id_collision_with_different_rgb_history():
    original = episode()
    memory = FeedbackMemory()
    memory.add(original)
    altered = original.rgb.copy()
    altered[0] += 1
    with pytest.raises(ValueError, match="collision"):
        memory.add(replace(original, rgb=altered))
