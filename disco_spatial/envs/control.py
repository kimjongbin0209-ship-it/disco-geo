"""Synthetic positive-control task for the learner pipeline; never a Memory Maze result.

Cue recall: each trial shows a full-frame cue colour on its first step, then
``delay`` grey/white frames; only the action on the trial's last step is
scored (+1 when it equals the cue's action). With ``delay=0`` the cue frame is
itself scored, so the task is reactive and dense. Larger delays need the GRU to
hold the cue and the update rule to assign credit to a sparse, delayed reward.
A uniform policy earns 1/6 per trial and the optimum 1 per trial. Episodes end
in a time limit with continuation discount 1, as Memory Maze's do. A fixed cue
turns the task into a one-answer bandit that needs no perception at all: the
sharpest check that an update rule follows reward.

The task exposes the same boundary as RGBOnlyMemoryMaze (RGB only, six
actions, episode bookkeeping, empty sidecar), so the unmodified collector,
learner and validation loop run on it. This module imports no JAX.
"""

import dm_env
from dm_env import specs
import numpy as np


CUE_COLORS = np.array([(220, 40, 40), (40, 200, 40), (40, 70, 230),
                       (230, 210, 40), (200, 40, 210), (40, 210, 220)], np.uint8)
BLANK, GO = 96, 255


class CueRecall:
    def __init__(self, seed=0, split="train", resolution=64, delay=0, episode_steps=240,
                 fixed_cue=None):
        if split not in {"train", "validation", "test"}:
            raise ValueError(f"Unknown split: {split}")
        if not isinstance(delay, int) or delay < 0 or episode_steps <= delay:
            raise ValueError("Choose delay >= 0 and an episode longer than one trial")
        if fixed_cue is not None and not 0 <= fixed_cue < 6:
            raise ValueError("fixed_cue must be an action index or None")
        self.fixed_cue = fixed_cue
        self.seed, self.split, self.resolution = int(seed), split, resolution
        self.delay, self.episode_steps = delay, episode_steps
        self.goal_segment, self.episode_id, self.instance_id = 0, None, None
        self._episode_index, self._t, self._cue, self._active = -1, 0, 0, False
        self._rng = None
        self._metadata = {
            "environment": "cue_recall_positive_control",
            "task": f"cue_recall_delay{delay}" + ("" if fixed_cue is None else f"_fixed{fixed_cue}"),
            "seed": self.seed, "camera_resolution": resolution, "wrapper_version": 1,
            "reward": "+1 when the scored trial step's action equals the cue; no shaping",
            "seed_semantics": "seed plus reset index identifies the cue sequence",
            "action_names": ["noop", "forward", "left", "right", "forward_left", "forward_right"],
        }

    @property
    def metadata(self):
        return dict(self._metadata)

    @property
    def last_sidecar(self):
        return {}

    def observation_spec(self):
        return {"rgb": specs.Array((self.resolution, self.resolution, 3), np.uint8, name="rgb")}

    def action_spec(self):
        return specs.BoundedArray((), np.int32, 0, 5)

    def _phase(self):
        return self._t % (self.delay + 1)

    def _image(self):
        phase = self._phase()
        if phase == 0:
            cue = int(self._rng.integers(6))
            self._cue = cue if self.fixed_cue is None else self.fixed_cue
            color = CUE_COLORS[self._cue]
        else:
            color = np.full(3, GO if phase == self.delay else BLANK, np.uint8)
        return {"rgb": np.broadcast_to(color, (self.resolution, self.resolution, 3)).copy()}

    def reset(self):
        self._episode_index += 1
        self.instance_id = f"cue-recall-{self.delay}:seed{self.seed}:episode{self._episode_index}"
        self.episode_id = f"{self.split}:{self.instance_id}"
        self._rng = np.random.default_rng((self.seed, self._episode_index))
        self._t, self.goal_segment, self._active = 0, 0, True
        return dm_env.restart(self._image())

    def step(self, action):
        if not self._active:
            raise RuntimeError("Call reset after construction or LAST")
        if not 0 <= int(action) < 6:
            raise ValueError("Action must be in [0, 5]")
        reward = float(self._phase() == self.delay and int(action) == self._cue)
        self.goal_segment += int(reward)
        self._t += 1
        observation = self._image()
        if self._t >= self.episode_steps:
            self._active = False
            return dm_env.truncation(reward, observation, discount=1.0)
        return dm_env.transition(reward, observation, discount=1.0)

    def close(self):
        self._active = False


def make_cue_recall(size=9, seed=0, split="train", camera_resolution=64, evaluation=False,
                    telemetry=False, *, delay=0, episode_steps=240, fixed_cue=-1):
    """Factory with make_memory_maze's positional signature; ``size`` is ignored.

    ``fixed_cue=-1`` draws cues at random; 0..5 always asks for that action.
    """
    del size
    if telemetry:
        raise ValueError("The positive control has no render telemetry")
    return CueRecall(seed, split, camera_resolution, delay, episode_steps,
                     None if fixed_cue < 0 else fixed_cue)
