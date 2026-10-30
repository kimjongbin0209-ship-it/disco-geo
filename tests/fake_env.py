"""Unit-test-only environments; importable by spawned workers without JAX."""

import dm_env
from dm_env import specs
import numpy as np


class ShortTask:
    """Unit-test-only sequence with a genuine time-limit continuation discount."""

    def __init__(self, size=9, seed=0, split="train", resolution=64):
        self.size, self.seed, self.split, self.resolution = size, seed, split, resolution
        self.metadata = {"environment": "unit_test_only", "seed": seed}
        self.episode_index, self.goal_segment, self.position = -1, 0, 0
        self.closed = False

    def observation_spec(self):
        return {"rgb": specs.Array((self.resolution, self.resolution, 3), np.uint8)}

    def action_spec(self):
        return specs.BoundedArray((), np.int32, 0, 5)

    def image(self):
        pixels = np.full((self.resolution, self.resolution, 3), self.position * 20, np.uint8)
        pixels[:, self.position % self.resolution, 0] = 200
        return {"rgb": pixels}

    def reset(self):
        self.episode_index += 1
        self.episode_id = f"train:{self.seed}:{self.episode_index}"
        self.instance_id = f"{self.seed}:{self.episode_index}"
        self.position, self.goal_segment = 0, 0
        return dm_env.restart(self.image())

    def step(self, action):
        assert self.position < 3
        self.position += 1
        reward = float(self.position == 2)
        self.goal_segment += int(reward)
        if self.position == 3:
            return dm_env.truncation(reward, self.image(), discount=1.)
        return dm_env.transition(reward, self.image(), discount=1.)

    def close(self):
        self.closed = True


class BrokenTask(ShortTask):
    """Fails on construction, to test that worker errors reach the parent."""

    def __init__(self, *args):
        raise ValueError("deliberately broken environment")
