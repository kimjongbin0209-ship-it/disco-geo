"""Fake dm_env boundary tests; these do not replace real-backend validation."""

from collections import namedtuple
from types import SimpleNamespace
import builtins
import numpy as np
import pytest
from disco_spatial.envs.memory_maze import RGBOnlyMemoryMaze, make_memory_maze


class TimeStep(namedtuple("TimeStep", "step_type reward discount observation")):
    def first(self):
        return self.step_type == 0

    def last(self):
        return self.step_type == 2


class FakeEnv:
    def __init__(self, privileged=10):
        self.privileged = privileged
        self.i = 0
        self.image = np.zeros((8, 8, 3), np.uint8)

    def observation_spec(self):
        return {"image": SimpleNamespace(shape=(8, 8, 3), dtype=np.uint8),
                "agent_pos": SimpleNamespace(shape=(2,), dtype=np.float64)}

    def action_spec(self):
        return SimpleNamespace(shape=(), minimum=0, maximum=5)

    def _obs(self):
        self.image[:] = self.i
        return {"image": self.image, "agent_pos": np.array([self.privileged, 0]),
                "maze_layout": np.full((9, 9), self.privileged),
                "unreviewed_future_label": np.array(888)}

    def reset(self):
        self.i = 0
        return TimeStep(0, None, None, self._obs())

    def step(self, action):
        self.i += 1
        return TimeStep(2 if self.i == 2 else 1, 1.0 if self.i == 1 else 0.0,
                        1.0, self._obs())


def test_rgb_whitelist_metadata_perturbation_and_reward_semantics():
    a = RGBOnlyMemoryMaze(FakeEnv(10), instance_id="seed1")
    b = RGBOnlyMemoryMaze(FakeEnv(999), instance_id="seed2")
    first = a.reset()
    other = b.reset()
    assert set(first.observation) == {"rgb"}
    np.testing.assert_array_equal(first.observation["rgb"], other.observation["rgb"])
    assert a.last_sidecar == {}
    assert set(a.observation_spec()) == {"rgb"}
    for action in [1, 2]:
        x, y = a.step(action), b.step(action)
        np.testing.assert_array_equal(x.observation["rgb"], y.observation["rgb"])
        assert x.reward == y.reward and x.discount == y.discount
    assert x.last() and x.discount == 1  # Official time limit is a truncation.
    assert a.goal_segment == 1
    assert not first.observation["rgb"].any()  # No reused-render-buffer corruption.
    with pytest.raises(RuntimeError, match="reset"):
        a.step(0)
    old = a.instance_id
    a.reset()
    assert a.instance_id != old and a.goal_segment == 0


def test_extra_observations_are_explicit_separate_copies():
    with pytest.raises(ValueError, match="reserved"):
        RGBOnlyMemoryMaze(FakeEnv(), evaluation=True)
    env = RGBOnlyMemoryMaze(FakeEnv(), split="test", evaluation=True)
    actor = env.reset()
    assert set(actor.observation) == {"rgb"}
    assert set(env.last_sidecar) == {"agent_pos", "maze_layout"}
    sidecar = env.last_sidecar
    sidecar["agent_pos"][:] = -1000
    assert env.last_sidecar["agent_pos"][0] == 10


def test_factory_is_lazy_and_missing_backend_is_not_a_fake_environment(monkeypatch):
    original = builtins.__import__

    def no_backend(name, *args, **kwargs):
        if name == "memory_maze":
            raise ModuleNotFoundError("not installed")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_backend)
    with pytest.raises(ImportError, match="no fake fallback"):
        make_memory_maze()


@pytest.mark.parametrize("resolution", [8, 32, 63])
def test_real_factory_rejects_zero_width_target_border_resolutions(resolution):
    # In upstream 1.0.3 B=int(2*sqrt(H//64)) is zero below 64. The -0
    # border slice then covers the whole frame, destroying spatial RGB input.
    # Reject before any backend import/construction; small fake wrapper tests
    # remain valid and do not stand in for a real rendered observation.
    with pytest.raises(ValueError, match="border can overwrite the entire image"):
        make_memory_maze(camera_resolution=resolution)


def test_factory_preserves_official_task_settings_and_reports_versions(monkeypatch):
    import sys
    calls = []

    def factory(**kwargs):
        calls.append(kwargs)
        return FakeEnv()

    monkeypatch.setitem(sys.modules, "memory_maze", SimpleNamespace(
        tasks=SimpleNamespace(memory_maze_9x9=factory)))
    train = make_memory_maze(seed=17)
    assert calls[-1]["image_only_obs"] is True
    assert calls[-1]["global_observables"] is False
    assert calls[-1]["seed"] == 17
    assert calls[-1]["target_color_in_image"] is True
    assert not calls[-1]["show_path"] and not calls[-1]["top_camera"]
    assert train.metadata["environment"] == "memory-maze"
    assert "environment_version" in train.metadata
    make_memory_maze(seed=18, split="validation", evaluation=True)
    assert calls[-1]["global_observables"] is True
    assert calls[-1]["image_only_obs"] is False
