"""Canonical, immutable raw experience with T actions and T+1 observations.

The transition at t is rgb[t] --actions[t]--> rgb[t+1], earning
rewards[t] and discounts[t]. A time limit is LAST with truncated=True and
an upstream continuation discount (normally 1), not an artificial terminal.
Evaluation labels belong in separate sidecars and are rejected here.
"""

from dataclasses import dataclass, field
from collections.abc import Mapping
import json
from pathlib import Path
from types import MappingProxyType
import numpy as np


SCHEMA_VERSION = 1
FIRST, MID, LAST = 0, 1, 2
ARRAY_FIELDS = (
    "rgb", "actions", "rewards", "discounts", "behavior_logits", "policy_versions",
    "step_types", "goal_segments", "terminated", "truncated",
)
METADATA_KEYS = frozenset({
    "environment", "environment_version", "dm_control_version", "mujoco_version",
    "labmaze_version", "task", "maze_size", "camera_resolution", "control_freq", "seed",
    "reward", "source_url", "wrapper_version", "seed_semantics", "episode_index",
    "action_names", "schema_version", "collector", "policy_id", "method",
})


def _frozen_array(value, name):
    array = np.array(value, copy=True)
    if array.dtype.hasobject or array.dtype.kind not in "buif":
        raise ValueError(f"{name} must be a numeric non-object array")
    # bytes-backed arrays cannot have writeability re-enabled by a consumer.
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


@dataclass(frozen=True)
class Trajectory:
    episode_id: str
    instance_id: str
    split: str
    rgb: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    discounts: np.ndarray
    behavior_logits: np.ndarray
    policy_versions: np.ndarray
    step_types: np.ndarray
    goal_segments: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        for key in ("episode_id", "instance_id"):
            if not isinstance(getattr(self, key), str) or not getattr(self, key):
                raise ValueError(f"{key} must be a nonempty string")
        if self.split not in {"train", "validation", "test"}:
            raise ValueError("split must be train, validation, or test")
        for name in ARRAY_FIELDS:
            object.__setattr__(self, name, _frozen_array(getattr(self, name), name))
        if not isinstance(self.metadata, Mapping) or set(self.metadata) - METADATA_KEYS:
            raise ValueError("Unknown trajectory metadata; evaluation labels need a sidecar")
        # Validate JSON now, not during a later checkpoint save. Deep copy all values.
        meta = json.loads(json.dumps(dict(self.metadata), allow_nan=False))
        for name, value in meta.items():
            if isinstance(value, dict) or (isinstance(value, list) and
                                          (name != "action_names" or
                                           not all(isinstance(v, str) for v in value))):
                raise ValueError("Metadata must be flat provenance; labels need a sidecar")
            if isinstance(value, list):
                meta[name] = tuple(value)
        object.__setattr__(self, "metadata", MappingProxyType(meta))
        self.validate()

    @property
    def n_steps(self):
        return len(self.actions)

    @property
    def complete(self):
        return bool(self.step_types[-1] == LAST)

    @property
    def continuation_discounts(self):
        return self.discounts

    def validate(self):
        t = self.n_steps
        if self.rgb.dtype != np.uint8 or self.rgb.ndim != 4 or self.rgb.shape[-1] != 3:
            raise ValueError("rgb must be uint8 [T+1,H,W,3]")
        if self.rgb.shape[0] != t + 1 or min(self.rgb.shape[1:3]) < 1:
            raise ValueError("rgb/action alignment requires T+1 states for T transitions")
        for name in ("actions", "policy_versions", "goal_segments", "step_types"):
            if getattr(self, name).dtype.kind not in "iu":
                raise ValueError(f"{name} must have integer dtype")
        for name in ("actions", "rewards", "discounts", "policy_versions", "terminated",
                     "truncated"):
            if getattr(self, name).shape != (t,):
                raise ValueError(f"{name} must have shape [T]")
        if self.step_types.shape != (t + 1,) or self.goal_segments.shape != (t + 1,):
            raise ValueError("step_types and goal_segments must have shape [T+1]")
        if self.terminated.dtype != np.bool_ or self.truncated.dtype != np.bool_:
            raise ValueError("terminated/truncated must be bool arrays")
        if self.behavior_logits.ndim != 2 or self.behavior_logits.shape[0] != t:
            raise ValueError("behavior_logits must contain the full action distribution [T,A]")
        if self.behavior_logits.shape[1] < 2 or self.behavior_logits.dtype.kind != "f":
            raise ValueError("behavior_logits must be floating [T,A], A>=2")
        for name in ("rewards", "discounts", "behavior_logits"):
            if not np.isfinite(getattr(self, name)).all():
                raise ValueError(f"{name} must be finite")
        if np.any(self.actions < 0) or np.any(self.actions >= self.behavior_logits.shape[1]):
            raise ValueError("Recorded action is outside the behavior distribution")
        if (np.any(self.policy_versions < 0)
                or np.any(self.policy_versions > np.iinfo(np.int64).max)
                or np.any(self.policy_versions[1:] < self.policy_versions[:-1])):
            raise ValueError("policy_versions must be nonnegative and monotonic within an episode")
        if np.any(self.discounts < 0) or np.any(self.discounts > 1):
            raise ValueError("Continuation discounts must be in [0,1]")
        if self.step_types[0] != FIRST or np.any(self.step_types[1:-1] != MID):
            raise ValueError("An episode starts with FIRST and cannot cross episode boundaries")
        if t and self.step_types[-1] not in (MID, LAST):
            raise ValueError("Final state must be MID (open prefix) or LAST")
        if np.any(self.terminated & self.truncated):
            raise ValueError("A transition cannot be both terminated and truncated")
        ended = self.terminated | self.truncated
        if np.any(ended != (self.step_types[1:] == LAST)):
            raise ValueError("LAST must match exactly one terminal/truncation flag")
        if np.any(self.discounts[self.terminated] != 0):
            raise ValueError("True terminals must have zero continuation discount")
        if (np.any(self.goal_segments < 0)
                or np.any(self.goal_segments[1:] < self.goal_segments[:-1])):
            raise ValueError("goal_segments must be nonnegative and monotonic")
        if np.any((self.rewards != 0) & (self.rewards != 1)):
            raise ValueError("Canonical Memory Maze rewards are upstream 0/+1, without shaping")
        if np.any(np.diff(self.goal_segments) != (self.rewards == 1).astype(int)):
            raise ValueError("Goal segments must advance at each upstream +1 target reward")
        return self

    def behavior_probabilities(self):
        logits = self.behavior_logits.astype(np.float64)
        logits -= logits.max(axis=-1, keepdims=True)
        p = np.exp(logits)
        return p / p.sum(axis=-1, keepdims=True)


def save_trajectory(trajectory, path):
    """Write a typed NPZ plus a scalar JSON header; never serialize Python objects."""
    trajectory.validate()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = {
        "schema_version": SCHEMA_VERSION, "episode_id": trajectory.episode_id,
        "instance_id": trajectory.instance_id, "split": trajectory.split,
        "metadata": dict(trajectory.metadata),
    }
    with path.open("xb") as stream:
        np.savez_compressed(stream, header=np.asarray(json.dumps(header, allow_nan=False)),
                            **{k: getattr(trajectory, k) for k in ARRAY_FIELDS})
    return path


def load_trajectory(path):
    with np.load(path, allow_pickle=False) as data:
        if len(data.files) != len(ARRAY_FIELDS) + 1 or set(data.files) != {"header", *ARRAY_FIELDS}:
            raise ValueError("Unexpected NPZ keys; evaluation sidecars cannot enter experience")
        header_array = data["header"]
        if header_array.shape != () or header_array.dtype.kind != "U":
            raise ValueError("NPZ header must be a scalar Unicode JSON string")
        header = json.loads(str(header_array))
        if set(header) != {"schema_version", "episode_id", "instance_id", "split", "metadata"}:
            raise ValueError("Invalid trajectory header fields")
        if header.pop("schema_version") != SCHEMA_VERSION:
            raise ValueError("Unsupported trajectory schema version")
        return Trajectory(**header, **{k: data[k] for k in ARRAY_FIELDS})
