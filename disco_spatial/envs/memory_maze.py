"""RGB-only boundary around the real, unmodified Memory Maze dm_env task.

ExtraObs is an evaluation sidecar, never part of a learner TimeStep.  The seed
and reset index identify a realization; they are bookkeeping, not observations.
"""

from importlib import metadata as package_metadata
from collections.abc import Mapping
import copy
import numpy as np


ACTION_NAMES = ("noop", "forward", "left", "right", "forward_left", "forward_right")
SIDECAR_KEYS = frozenset({
    "agent_pos", "agent_dir", "maze_layout", "target_color", "target_pos", "target_vec",
    "targets_pos", "targets_vec",
})
TELEMETRY_KEYS = frozenset({
    "target_visible_pixels", "observed_cells", "depth_coarse", "floor_pixels",
    "floor_pixels_in_walls",
})
SPLITS = frozenset({"train", "validation", "test"})


def _version(name):
    try:
        return package_metadata.version(name)
    except package_metadata.PackageNotFoundError:
        return None


class RGBOnlyMemoryMaze:
    """Whitelist RGB and preserve upstream rewards, step types, and discounts.

    There is deliberately no ``__getattr__`` delegation to the underlying task:
    physics/global observables are not available through the actor-facing API.
    Evaluation consumers must explicitly read ``last_sidecar``.
    """

    def __init__(self, env, *, split="train", instance_id="memory-maze", evaluation=False,
                 metadata=None, telemetry=None):
        if split not in SPLITS:
            raise ValueError(f"Unknown split: {split}")
        if evaluation and split == "train":
            raise ValueError("ExtraObs sidecars are reserved for validation/test environments")
        if telemetry is not None and not evaluation:
            raise ValueError("Render telemetry is an evaluation sidecar; enable evaluation first")
        self._env = env
        self._telemetry = telemetry
        self.split = split
        self.evaluation = bool(evaluation)
        self._instance_prefix = str(instance_id)
        self._episode_index = -1
        self._sidecar = {}
        self._active = False
        self.goal_segment = 0
        self.episode_id = None
        self.instance_id = None
        self._metadata = copy.deepcopy(metadata or {})
        spec = env.action_spec()
        if tuple(spec.shape) != () or int(spec.minimum) != 0 or int(spec.maximum) != 5:
            raise ValueError("Memory Maze must expose the official six discrete actions")
        self.num_actions = 6

    @property
    def metadata(self):
        return copy.deepcopy(self._metadata)

    @property
    def last_sidecar(self):
        return {k: np.array(v, copy=True) for k, v in self._sidecar.items()}

    def action_spec(self):
        return self._env.action_spec()

    def observation_spec(self):
        observation = self._env.observation_spec()
        image = observation["image"] if isinstance(observation, Mapping) else observation
        return {"rgb": image}

    def _clean(self, timestep):
        obs = timestep.observation
        if isinstance(obs, Mapping):
            if "image" not in obs:
                raise ValueError("Upstream Memory Maze observation has no image")
            image = obs["image"]
            self._sidecar = ({k: np.array(obs[k], copy=True) for k in SIDECAR_KEYS if k in obs}
                             if self.evaluation else {})
            if self._telemetry is not None:
                if "maze_layout" not in obs:
                    raise ValueError("Render telemetry needs the evaluation maze layout")
                self._sidecar.update(self._telemetry.frame(obs["maze_layout"]))
        else:
            image = obs
            self._sidecar = {}
        image = np.asarray(image)
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError("Memory Maze image must be uint8 HWC RGB")
        # Copy: upstream renderer buffers can be reused by the next env.step().
        clean = timestep._replace(observation={"rgb": image.copy()})
        if timestep.reward is not None:
            reward = float(timestep.reward)
            if reward not in (0.0, 1.0):
                raise ValueError("Unexpected upstream reward; reward shaping is not supported")
            self.goal_segment += int(reward == 1.0)
        self._active = not bool(timestep.last())
        return clean

    def reset(self):
        self._episode_index += 1
        # A new procedural realization, not a retry of a reusable logical task.
        # goal_segment counts reward-triggered requests; it is not a color ID.
        self.instance_id = f"{self._instance_prefix}:episode{self._episode_index}"
        self.episode_id = f"{self.split}:{self.instance_id}"
        self.goal_segment = 0
        timestep = self._env.reset()
        if not timestep.first():
            raise ValueError("Environment reset must produce FIRST")
        return self._clean(timestep)

    def step(self, action):
        if not self._active:
            raise RuntimeError("Call reset after construction or LAST; no implicit reset step")
        if isinstance(action, (bool, np.bool_)) or not np.isscalar(action):
            raise ValueError("Action must be a scalar integer")
        if int(action) != action or not 0 <= int(action) < self.num_actions:
            raise ValueError("Action must be in the official range [0, 5]")
        return self._clean(self._env.step(int(action)))

    def close(self):
        close = getattr(self._env, "close", None)
        if close is not None:
            close()


def make_memory_maze(size=9, seed=0, split="train", camera_resolution=64, evaluation=False,
                     telemetry=False):
    """Create a real official task lazily; never substitute a synthetic backend.

    Training uses upstream image_only_obs=True. Evaluation requests global
    observations, then strips them at the same boundary into a separate sidecar.
    ``telemetry`` re-renders the actor camera with depth/segmentation for exact
    visibility and observed-cell labels; it requires ``evaluation``.
    Renderer selection (MUJOCO_GL) is left to the caller before construction.
    """
    if size not in (9, 11, 13, 15):
        raise ValueError("Official maze sizes are 9, 11, 13, and 15")
    if split not in SPLITS or (evaluation and split == "train") or (telemetry and not evaluation):
        raise ValueError("Invalid split/evaluation/telemetry combination")
    if not isinstance(seed, (int, np.integer)) or isinstance(seed, (bool, np.bool_)):
        raise ValueError("An explicit integer seed is required")
    if int(seed) < 0 or int(seed) >= 2**32:
        raise ValueError("Seed must fit the upstream uint32 random state")
    if (isinstance(camera_resolution, bool) or not isinstance(camera_resolution, int)
            or camera_resolution < 64):
        raise ValueError(
            "camera_resolution must be an integer >= 64: Memory Maze's target-color "
            "border can overwrite the entire image below 64 pixels (zero-width -0 slicing)."
        )
    try:
        from memory_maze import tasks
    except ImportError as exc:
        raise ImportError(
            "The real Memory Maze backend is unavailable. Install this project's spatial "
            "environment dependencies (memory-maze, dm-control, mujoco, gym); "
            "no fake fallback is used."
        ) from exc
    constructor = getattr(tasks, f"memory_maze_{size}x{size}")
    env = constructor(seed=int(seed), image_only_obs=not evaluation,
                      global_observables=bool(evaluation), camera_resolution=camera_resolution,
                      discrete_actions=True, target_color_in_image=True, top_camera=False,
                      show_path=False, good_visibility=False)
    provenance = {
        "environment": "memory-maze", "environment_version": _version("memory-maze"),
        "dm_control_version": _version("dm-control"), "mujoco_version": _version("mujoco"),
        "labmaze_version": _version("labmaze"), "maze_size": size,
        "camera_resolution": camera_resolution, "control_freq": 4.0, "seed": int(seed),
        "reward": "upstream +1 on requested target; no shaping",
        "source_url": "https://github.com/jurgisp/memory-maze", "wrapper_version": 1,
        "action_names": list(ACTION_NAMES),
        "seed_semantics": "seed plus reset index identifies a procedural realization",
    }
    probe = None
    if telemetry:
        from disco_spatial.evaluation.telemetry import RenderTelemetry

        probe = RenderTelemetry(env, camera_resolution)
    return RGBOnlyMemoryMaze(env, split=split, instance_id=f"memory-maze-{size}:seed{seed}",
                             evaluation=evaluation, metadata=provenance, telemetry=probe)
