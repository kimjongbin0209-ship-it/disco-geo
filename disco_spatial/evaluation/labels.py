"""Readout labels for the six ability families, built only from evaluation sidecars.

Coordinates are (right, forward), matching upstream ``targets_vec``. The
self-localization gauge is the exact start pose. The allocentric map gauge is
the start cell with the start heading rounded to the nearest grid axis: the
grid is axis-aligned, so this is a deterministic function of the agent's own
start view. No label is aligned to predictions after the fact.
"""

from collections import deque

import numpy as np

from disco_spatial.evaluation.metrics import causal_observability_masks


GRID_AXES = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]])
RAY_ANGLES = (-30.0, 0.0, 30.0)
MOTION_LAGS = (1, 8)
# memory_maze.maze.TARGET_COLORS (1.0.3), copied so labels need no MuJoCo import.
TARGET_PALETTE = np.array([
    [170, 38, 30], [99, 170, 88], [39, 140, 217], [93, 105, 199], [220, 193, 59],
    [220, 128, 107],
]) / 220


def _unit(vectors):
    vectors = np.asarray(vectors, float)
    return vectors / np.maximum(np.linalg.norm(vectors, axis=-1, keepdims=True), 1e-9)


def egocentric(vectors, direction):
    """(right, forward) components of world-grid vectors for a heading."""
    forward = _unit(direction)
    right = np.stack((forward[..., 1], -forward[..., 0]), axis=-1)
    return np.stack((np.sum(vectors * right, -1), np.sum(vectors * forward, -1)), axis=-1)


def ring_walls(layout):
    """Wall occupancy on the ring grid [y + 1, x + 1]; the outer ring is wall."""
    layout = np.asarray(layout)
    walls = np.ones((layout.shape[0] + 2, layout.shape[1] + 2), np.uint8)
    walls[1:-1, 1:-1] = layout == 0
    return walls


def origin_canvas(layout, start_position, start_direction):
    """Map the ring grid into a start-centered, start-heading-up canvas.

    Returns integer canvas indices [ring_y, ring_x, 2] and the canvas size.
    Canvas row grows with forward offset, column with rightward offset.
    """
    size_y, size_x = np.shape(layout)
    span = max(size_x, size_y) + 1
    canvas = 2 * span + 1
    axis = GRID_AXES[int(np.argmax(GRID_AXES @ _unit(start_direction)))]
    right = np.array([axis[1], -axis[0]])
    x0, y0 = np.floor(start_position).astype(int)
    ys, xs = np.mgrid[-1:size_y + 1, -1:size_x + 1]
    offset = np.stack((xs - x0, ys - y0), axis=-1)
    index = np.stack((offset @ axis + span, offset @ right + span), axis=-1)
    return index, canvas


def wall_canvas(layout, start_position, start_direction, observed_cells):
    """Constant wall map, valid mask and causal known masks in the origin canvas."""
    index, canvas = origin_canvas(layout, start_position, start_direction)
    walls = np.zeros((canvas, canvas), np.uint8)
    valid = np.zeros((canvas, canvas), bool)
    walls[index[..., 0], index[..., 1]] = ring_walls(layout)
    valid[index[..., 0], index[..., 1]] = True
    observed = np.asarray(observed_cells, bool)
    known_ring = np.logical_or.accumulate(observed, axis=0)
    known = np.zeros((len(observed), canvas, canvas), bool)
    known[:, index[..., 0], index[..., 1]] = known_ring
    return walls, valid, known


def ray_distance(walls, position, direction, step=0.02, limit=12.0):
    """Distance in cells along a ray until the first wall cell on the ring grid."""
    direction = _unit(direction)
    distances = np.arange(step, limit, step)
    points = np.asarray(position) + distances[:, None] * direction
    cells = np.floor(points).astype(int) + 1
    inside = np.all((cells >= 0) & (cells < np.array(walls.shape)[::-1]), axis=-1)
    hit = ~inside
    hit[inside] = walls[cells[inside, 1], cells[inside, 0]] == 1
    return float(distances[np.argmax(hit)]) if hit.any() else float(limit)


def free_space_rays(layout, positions, directions):
    walls = ring_walls(layout)
    rays = np.zeros((len(positions), len(RAY_ANGLES)), np.float32)
    for t, (position, direction) in enumerate(zip(positions, directions)):
        for k, angle in enumerate(RAY_ANGLES):
            c, s = np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))
            # Positive angles rotate clockwise, toward the agent's right.
            rotated = np.array([c * direction[0] + s * direction[1],
                                -s * direction[0] + c * direction[1]])
            rays[t, k] = ray_distance(walls, position, rotated)
    return rays


def grid_distance(layout, start, goal):
    """4-connected shortest path length in cells over free cells; None if unreachable."""
    layout = np.asarray(layout)
    start, goal = tuple(int(v) for v in start), tuple(int(v) for v in goal)
    if start == goal:
        return 0
    seen = {start}
    queue = deque([(start, 0)])
    while queue:
        (x, y), distance = queue.popleft()
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nxt = (x + dx, y + dy)
            if (0 <= nxt[0] < layout.shape[1] and 0 <= nxt[1] < layout.shape[0]
                    and layout[nxt[1], nxt[0]] == 1 and nxt not in seen):
                if nxt == goal:
                    return distance + 1
                seen.add(nxt)
                queue.append((nxt, distance + 1))
    return None


def target_indices(target_colors, n_targets):
    """Requested target index from the sidecar color (upstream fixed palette order)."""
    palette = TARGET_PALETTE[:n_targets]
    colors = np.asarray(target_colors, float)
    return np.argmin(np.linalg.norm(colors[:, None] - palette[None], axis=-1), axis=1)


def episode_labels(sidecar, visibility_pixels=1):
    """Build all readout labels for one episode of stacked sidecars [T+1, ...]."""
    positions, directions = np.asarray(sidecar["agent_pos"], float), np.asarray(sidecar["agent_dir"], float)
    layout = np.asarray(sidecar["maze_layout"][0])
    steps = len(positions)
    start, heading = positions[0], directions[0]
    labels = {
        "origin_relative_position": egocentric(positions - start, heading).astype(np.float32),
        "origin_relative_heading": egocentric(_unit(directions), heading).astype(np.float32),
        "targets_vec": np.asarray(sidecar["targets_vec"], np.float32),
        "free_space_rays": free_space_rays(layout, positions, directions),
        "agent_cell": np.floor(positions).astype(np.int16),
        "agent_heading_angle": np.arctan2(directions[:, 1], directions[:, 0]).astype(np.float32),
        "time_index": np.arange(steps, dtype=np.int32),
    }
    for lag in MOTION_LAGS:
        motion = np.zeros((steps, 4), np.float32)
        if steps > lag:
            before = slice(0, steps - lag)
            motion[lag:, :2] = egocentric(positions[lag:] - positions[before], directions[before])
            motion[lag:, 2:] = egocentric(_unit(directions[lag:]), directions[before])
        labels[f"relative_motion_{lag}"] = motion
        labels[f"relative_motion_{lag}_valid"] = np.arange(steps) >= lag
    if "target_visible_pixels" in sidecar:
        visible = np.asarray(sidecar["target_visible_pixels"]) >= visibility_pixels
        masks = causal_observability_masks(visible)
        for name, mask in masks.items():
            labels[f"targets_{name}"] = mask
    if "observed_cells" in sidecar:
        walls, valid, known = wall_canvas(layout, start, heading, sidecar["observed_cells"])
        labels.update(origin_walls=np.broadcast_to(walls, known.shape).copy(),
                      origin_map_valid=np.broadcast_to(valid, known.shape).copy(),
                      origin_map_known=known)
    if "depth_coarse" in sidecar:
        labels["depth_coarse"] = np.asarray(sidecar["depth_coarse"], np.float32)
    if "target_color" in sidecar:
        labels["requested_target"] = target_indices(
            sidecar["target_color"], np.shape(sidecar["targets_vec"])[1]).astype(np.int8)
    return labels
