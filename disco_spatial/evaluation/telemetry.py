"""Evaluation-only render telemetry for the actor's own egocentric camera.

The same MuJoCo camera that produced the RGB observation is rendered again with
depth and segmentation. Target visibility therefore counts real, unoccluded
target pixels, and observed map cells come from back-projected depth, not from
an idealized field-of-view guess. Nothing here may enter a learner TimeStep.

Grid convention follows upstream Memory Maze: ``agent_pos`` is (x, y) in cell
units, ``maze_layout[y, x]`` is 1 for free and 0 for wall. Observed cells use a
ring grid with the outer wall: index [y + 1, x + 1] for x, y in [-1, N].
"""

import numpy as np


CAMERA = "walker/egocentric"
GEOM_OBJECT_TYPE = 5  # mjOBJ_GEOM in MuJoCo segmentation renders
DEPTH_GRID = 8
MAX_DEPTH_CELLS = 10.0


def border_width(resolution):
    """Upstream TargetColorAsBorderWrapper width; zero below 64 pixels."""
    return int(2 * np.sqrt(resolution // 64))


def camera_rays(height, width, fovy_degrees):
    """Camera-frame rays scaled so that z = -1 (MuJoCo looks along -z)."""
    focal = 0.5 * height / np.tan(np.deg2rad(fovy_degrees) / 2)
    x = (np.arange(width) + 0.5 - width / 2) / focal
    y = -(np.arange(height) + 0.5 - height / 2) / focal
    return np.stack(np.broadcast_arrays(x[None, :], y[:, None], -np.ones((height, 1))), axis=-1)


def backproject(depth, camera_position, camera_matrix, fovy_degrees):
    """World points for a z-depth image; columns of camera_matrix are camera axes."""
    rays = camera_rays(*depth.shape, fovy_degrees)
    return camera_position + (depth[..., None] * rays) @ np.asarray(camera_matrix).reshape(3, 3).T


def observed_ring_cells(points, valid, camera_position, xy_scale, center, size, nudge=0.02):
    """Mark ring-grid cells hit by valid pixels.

    A wall-surface point lies on a cell boundary, so each point is pushed a small
    distance further along its horizontal ray before taking the cell. Floor points
    and target spheres fall inside free cells either way.
    """
    grid = points[..., :2] / xy_scale + center
    eye = np.asarray(camera_position)[:2] / xy_scale + center
    direction = grid - eye
    norm = np.linalg.norm(direction, axis=-1, keepdims=True)
    grid = grid + nudge * direction / np.maximum(norm, 1e-9)
    cells = np.floor(grid[valid]).astype(int)
    inside = np.all((cells >= -1) & (cells <= size), axis=-1)
    observed = np.zeros((size + 2, size + 2), bool)
    observed[cells[inside, 1] + 1, cells[inside, 0] + 1] = True
    return observed


def coarse_depth(depth_cells, valid, grid=DEPTH_GRID):
    """Block mean of valid pixels in cell units; empty blocks are NaN-free maxima."""
    height, width = depth_cells.shape
    if height % grid or width % grid:
        raise ValueError("Depth resolution must be divisible by the coarse grid")
    values = np.where(valid, np.minimum(depth_cells, MAX_DEPTH_CELLS), 0.0)
    shape = (grid, height // grid, grid, width // grid)
    total = values.reshape(shape).sum(axis=(1, 3))
    count = valid.reshape(shape).sum(axis=(1, 3))
    return np.where(count > 0, total / np.maximum(count, 1), MAX_DEPTH_CELLS).astype(np.float32)


class RenderTelemetry:
    """Re-render the egocentric camera of an evaluation-only upstream environment.

    Composer recompiles physics when a new maze is generated, so physics and
    target geom IDs are looked up on every frame rather than cached at reset.
    """

    def __init__(self, upstream, resolution):
        self._env = upstream
        self.resolution = int(resolution)
        self.border = border_width(self.resolution)
        task = upstream.task
        arena = task._maze_arena
        self.xy_scale = float(arena.xy_scale)
        self.size = int(arena.maze.width) - 2
        self.center = np.array([arena.maze.width - 2.0, arena.maze.height - 2.0]) / 2.0

    def frame(self, layout):
        """Telemetry for the current physics state; ``layout`` is the 1=free grid."""
        physics, task = self._env.physics, self._env.task
        r = self.resolution
        depth = np.asarray(physics.render(r, r, camera_id=CAMERA, depth=True), float)
        segmentation = np.asarray(physics.render(r, r, camera_id=CAMERA, segmentation=True))
        camera = physics.model.name2id(CAMERA, "camera")
        position = np.array(physics.data.cam_xpos[camera], copy=True)
        matrix = np.array(physics.data.cam_xmat[camera], copy=True).reshape(3, 3)
        fovy = float(physics.model.cam_fovy[camera])
        far = float(physics.model.stat.extent * physics.model.vis.map.zfar)
        valid = np.zeros((r, r), bool)
        b = self.border
        valid[b:r - b, b:r - b] = True  # the agent sees the color border, not the scene
        scene = valid & (segmentation[..., 0] >= 0) & (depth < 0.99 * far)
        points = backproject(depth, position, matrix, fovy)
        observed = observed_ring_cells(points, scene, position, self.xy_scale, self.center,
                                       self.size)
        geoms = np.where(segmentation[..., 1] == GEOM_OBJECT_TYPE, segmentation[..., 0], -1)
        target_ids = [physics.bind(target.geom).element_id for target in task._targets]
        pixels = np.array([int(np.sum(valid & (geoms == gid))) for gid in target_ids], np.int32)
        floor = scene & (np.abs(points[..., 2]) < 0.02)
        in_walls, total = floor_cells_in_walls(
            np.floor(points[floor][:, :2] / self.xy_scale + self.center).astype(int), layout)
        return {
            "target_visible_pixels": pixels,
            "observed_cells": observed,
            "depth_coarse": coarse_depth(depth / self.xy_scale, scene),
            # Oracle self-check: back-projected floor pixels must land in free cells.
            "floor_pixels": np.int32(total),
            "floor_pixels_in_walls": np.int32(in_walls),
        }


def floor_cells_in_walls(cells, layout):
    """Count back-projected floor pixels that land outside free inner cells."""
    cells = np.asarray(cells, int).reshape(-1, 2)
    size_y, size_x = np.shape(layout)
    inside = (cells[:, 0] >= 0) & (cells[:, 0] < size_x) & (cells[:, 1] >= 0) & (cells[:, 1] < size_y)
    free = np.zeros(len(cells), bool)
    free[inside] = np.asarray(layout)[cells[inside, 1], cells[inside, 0]] == 1
    return int(np.sum(~free)), int(len(cells))
