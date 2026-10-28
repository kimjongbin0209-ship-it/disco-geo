"""Contiguous raw episode replay and guarded same-instance feedback memory.

Replay is storage and sampling only: it does not silently add replay updates or
correct an on-policy objective. Recompute recurrent states from raw prefixes.
"""

from collections import OrderedDict
from dataclasses import dataclass
import numpy as np
from disco_spatial.trajectory import ARRAY_FIELDS, Trajectory


@dataclass(frozen=True)
class ReplayChunk:
    trajectory: Trajectory
    start: int
    stop: int
    burn_in_start: int

    @property
    def n_steps(self):
        return self.stop - self.start

    @property
    def burn_in(self):
        return self.start - self.burn_in_start

    @property
    def episode_id(self):
        return self.trajectory.episode_id

    @property
    def instance_id(self):
        return self.trajectory.instance_id

    @property
    def split(self):
        return self.trajectory.split

    @property
    def rgb(self):
        return self.trajectory.rgb[self.start:self.stop + 1]

    @property
    def actions(self):
        return self.trajectory.actions[self.start:self.stop]

    @property
    def rewards(self):
        return self.trajectory.rewards[self.start:self.stop]

    @property
    def discounts(self):
        return self.trajectory.discounts[self.start:self.stop]

    @property
    def behavior_logits(self):
        return self.trajectory.behavior_logits[self.start:self.stop]

    @property
    def policy_versions(self):
        return self.trajectory.policy_versions[self.start:self.stop]

    @property
    def step_types(self):
        return self.trajectory.step_types[self.start:self.stop + 1]

    @property
    def goal_segments(self):
        return self.trajectory.goal_segments[self.start:self.stop + 1]

    @property
    def terminated(self):
        return self.trajectory.terminated[self.start:self.stop]

    @property
    def truncated(self):
        return self.trajectory.truncated[self.start:self.stop]

    @property
    def prefix(self):
        """Full raw history through the first loss observation; no future RGB."""
        return self.trajectory.rgb[:self.start + 1]

    @property
    def burn_in_rgb(self):
        return self.trajectory.rgb[self.burn_in_start:self.start + 1]

    @property
    def burn_in_actions(self):
        return self.trajectory.actions[self.burn_in_start:self.start]

    @property
    def burn_in_rewards(self):
        return self.trajectory.rewards[self.burn_in_start:self.start]

    @property
    def prefix_actions(self):
        return self.trajectory.actions[:self.start]

    @property
    def prefix_rewards(self):
        return self.trajectory.rewards[:self.start]


class SequenceReplay:
    def __init__(self, capacity_episodes=100, seed=0):
        if not isinstance(capacity_episodes, int) or capacity_episodes <= 0:
            raise ValueError("capacity_episodes must be positive")
        self.capacity_episodes = capacity_episodes
        self._episodes = OrderedDict()
        self._rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self._episodes)

    def add(self, trajectory):
        if not isinstance(trajectory, Trajectory):
            raise TypeError("Replay accepts validated raw Trajectory objects only")
        if trajectory.split != "train":
            raise ValueError("Only train episodes are accepted; replay rejects validation/test")
        trajectory.validate()
        if not trajectory.n_steps:
            raise ValueError("Replay needs at least one transition")
        old = self._episodes.get(trajectory.episode_id)
        if old is not None:
            if (old.instance_id != trajectory.instance_id or old.split != trajectory.split
                    or dict(old.metadata) != dict(trajectory.metadata)
                    or old.n_steps >= trajectory.n_steps or old.complete):
                raise ValueError("An episode may only be extended by an identical contiguous prefix")
            for name in ARRAY_FIELDS:
                length = old.n_steps + int(name in ("rgb", "step_types", "goal_segments"))
                if not np.array_equal(getattr(old, name), getattr(trajectory, name)[:length]):
                    raise ValueError("Episode extension changed raw history or behavior probabilities")
        self._episodes[trajectory.episode_id] = trajectory
        while len(self._episodes) > self.capacity_episodes:
            self._episodes.popitem(last=False)

    def sample(self, sequence_length, batch_size=1, burn_in=0, current_policy_version=None,
               max_replay_age=None, require_reward=False):
        """Uniform over eligible windows; ``require_reward`` keeps successful chunks only.

        Policy versions are monotonic within an episode (schema-validated), so a
        window's oldest version is at its start and its newest at its end.
        """
        if not isinstance(sequence_length, int) or sequence_length <= 0:
            raise ValueError("sequence_length must be positive")
        if (not isinstance(batch_size, int) or batch_size <= 0 or not isinstance(burn_in, int)
                or burn_in < 0):
            raise ValueError("batch_size must be positive and burn_in nonnegative")
        if max_replay_age is not None and (max_replay_age < 0 or current_policy_version is None):
            raise ValueError("Replay age filtering needs a nonnegative age and current policy version")
        eligible = []
        for episode in self._episodes.values():
            starts = np.arange(max(0, episode.n_steps - sequence_length + 1))
            stops = starts + sequence_length
            keep = np.ones(len(starts), bool)
            versions = episode.policy_versions.astype(np.int64)
            if current_policy_version is not None and len(starts):
                keep &= versions[stops - 1] <= current_policy_version
                if max_replay_age is not None:
                    keep &= current_policy_version - versions[starts] <= max_replay_age
            if require_reward and len(starts):
                cumulative = np.r_[0.0, np.cumsum(episode.rewards)]
                keep &= cumulative[stops] - cumulative[starts] > 0
            eligible.extend((episode, int(s), int(s) + sequence_length) for s in starts[keep])
        if not eligible:
            raise ValueError("No contiguous train sequence meets length/replay-age constraints")
        chosen = self._rng.integers(len(eligible), size=batch_size)
        return [ReplayChunk(eligible[i][0], eligible[i][1], eligible[i][2],
                            max(0, eligible[i][1] - burn_in)) for i in chosen]


@dataclass(frozen=True)
class FeedbackSegment:
    trajectory: Trajectory
    goal_segment: int
    start: int
    stop: int

    @property
    def rgb(self):
        return self.trajectory.rgb[self.start:self.stop + 1]

    @property
    def actions(self):
        return self.trajectory.actions[self.start:self.stop]

    @property
    def rewards(self):
        return self.trajectory.rewards[self.start:self.stop]


class FeedbackMemory:
    """Guarded storage only, not an operational SDPO feedback/teacher pipeline.

    No nearest-layout or cross-goal fallback is performed. A successful segment
    is retrospective teacher context, never an actor observation or an oracle.

    Keys are exact instance IDs and monotonically increasing goal-segment
    counters, not semantic target colors. The current Memory Maze wrapper gives
    each episode its own instance ID. Consequently, ``retrieve(exclude_self=True)``
    finds no other attempt for trajectories from that collector. Repeated-attempt
    feedback requires a separately verified logical-instance / attempt-ID design;
    this class does not infer that two resets share a layout or target.
    """

    def __init__(self, capacity_segments=1000):
        if capacity_segments <= 0:
            raise ValueError("capacity_segments must be positive")
        self.capacity_segments = capacity_segments
        self._segments = OrderedDict()

    def add(self, trajectory):
        if not isinstance(trajectory, Trajectory) or trajectory.split != "train":
            raise ValueError("Feedback memory accepts train trajectories only")
        trajectory.validate()
        for end in np.flatnonzero(trajectory.rewards == 1):
            goal = int(trajectory.goal_segments[end])
            starts = np.flatnonzero(trajectory.goal_segments[:end + 1] == goal)
            segment = FeedbackSegment(trajectory, goal, int(starts[0]), int(end) + 1)
            key = (trajectory.instance_id, trajectory.episode_id, goal)
            old = self._segments.get(key)
            if old is not None:
                for name in ("rgb", "actions", "rewards"):
                    if not np.array_equal(getattr(old, name), getattr(segment, name)):
                        raise ValueError("Feedback episode/goal collision changed successful history")
            self._segments[key] = segment
        while len(self._segments) > self.capacity_segments:
            self._segments.popitem(last=False)

    def query(self, *, instance_id, goal_segment, split="train", exclude_episode_id=None):
        if split != "train":
            raise ValueError("No feedback retrieval for validation/test or unseen-layout evaluation")
        if not isinstance(instance_id, str) or not instance_id or goal_segment < 0:
            raise ValueError("Explicit instance_id and nonnegative goal_segment are required")
        return [s for (instance, episode, goal), s in self._segments.items()
                if instance == instance_id and goal == goal_segment
                and episode != exclude_episode_id]

    def retrieve(self, trajectory, goal_segment, exclude_self=True):
        return self.query(instance_id=trajectory.instance_id, goal_segment=goal_segment,
                          split=trajectory.split,
                          exclude_episode_id=trajectory.episode_id if exclude_self else None)
