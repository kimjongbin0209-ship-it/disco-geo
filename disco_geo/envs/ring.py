"""Batched JAX camera ring with either recognition or reference-view navigation."""

from typing import NamedTuple
import numpy as np
import jax
import jax.numpy as jnp
import dm_env
from dm_env import specs
from disco_geo.upstream import ensure_disco

ensure_disco()
from disco_rl import types  # noqa: E402


class RingState(NamedTuple):
    scene: jax.Array
    view: jax.Array
    reference_scene: jax.Array
    reference_view: jax.Array
    moves: jax.Array
    done: jax.Array
    rng: jax.Array


class RingEnvironment:
    """Indices remain private state. The learner receives exactly two RGB arrays.

    LEFT=0, RIGHT=1; navigation STOP=2; recognition SAME=2, DIFFERENT=3.
    A terminal timestep contains terminal RGB. The following step ignores its
    action and resets, returning FIRST with zero reward (dm_env convention).
    """

    def __init__(
        self, images, transitions, batch_size=16, task="recognition", max_moves=12, move_cost=0.02
    ):
        if task not in {"navigation", "recognition"}:
            raise ValueError(task)
        if len(images) < 2 or images.shape[1] < 3:
            raise ValueError("At least two scenes and three views required")
        self.images = jnp.asarray(images, dtype=jnp.uint8)
        self.transitions = jnp.asarray(transitions, dtype=jnp.int32)
        self.batch_size = batch_size
        self.task = task
        self.max_moves = max_moves
        self.move_cost = move_cost
        self.num_actions = 3 if task == "navigation" else 4

    def single_observation_spec(self):
        return {
            key: specs.Array(self.images.shape[2:], np.uint8) for key in ("rgb", "reference_rgb")
        }

    def single_action_spec(self):
        return specs.BoundedArray((), np.int32, 0, self.num_actions - 1)

    def observation(self, state):
        return {
            "rgb": self.images[state.scene, state.view],
            "reference_rgb": self.images[state.reference_scene, state.reference_view],
        }

    def reset(self, key):
        keys = jax.random.split(key, 6)
        n, views = self.images.shape[:2]
        shape = (self.batch_size,)
        scene = jax.random.randint(keys[0], shape, 0, n)
        view = jax.random.randint(keys[1], shape, 0, views)
        ref_view = jax.random.randint(keys[2], shape, 0, views)
        if self.task == "navigation":
            ref_scene = scene
            ref_view = (view + 1 + jax.random.randint(keys[2], shape, 0, views - 1)) % views
        else:
            same = jax.random.bernoulli(keys[3], 0.5, shape)
            other = (scene + 1 + jax.random.randint(keys[4], shape, 0, n - 1)) % n
            ref_scene = jnp.where(same, scene, other)
        state = RingState(
            scene,
            view,
            ref_scene,
            ref_view,
            jnp.zeros(shape, jnp.int32),
            jnp.zeros(shape, bool),
            keys[5],
        )
        ts = types.EnvironmentTimestep(
            observation=self.observation(state),
            step_type=jnp.full(shape, dm_env.StepType.FIRST, jnp.int32),
            reward=jnp.zeros(shape, jnp.float32),
        )
        return state, ts

    def step(self, state, actions):
        fresh, _ = self.reset(state.rng)
        moving = actions < 2
        view = jnp.where(
            moving, self.transitions[state.scene, state.view, jnp.minimum(actions, 1)], state.view
        )
        moves = state.moves + moving.astype(jnp.int32)
        if self.task == "navigation":
            success = (actions == 2) & (view == state.reference_view)
        else:
            same = state.scene == state.reference_scene
            success = ((actions == 2) & same) | ((actions == 3) & ~same)
        done = ~moving | (moves >= self.max_moves)
        reward = jnp.where(moving, -self.move_cost, success.astype(jnp.float32))
        old_done = state.done
        next_state = RingState(
            jnp.where(old_done, fresh.scene, state.scene),
            jnp.where(old_done, fresh.view, view),
            jnp.where(old_done, fresh.reference_scene, state.reference_scene),
            jnp.where(old_done, fresh.reference_view, state.reference_view),
            jnp.where(old_done, 0, moves),
            jnp.where(old_done, False, done),
            fresh.rng,
        )
        step_type = jnp.where(
            old_done,
            dm_env.StepType.FIRST,
            jnp.where(done, dm_env.StepType.LAST, dm_env.StepType.MID),
        ).astype(jnp.int32)
        ts = types.EnvironmentTimestep(
            observation=self.observation(next_state),
            step_type=step_type,
            reward=jnp.where(old_done, 0, reward).astype(jnp.float32),
        )
        return next_state, ts
