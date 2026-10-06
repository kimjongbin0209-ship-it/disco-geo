"""Small perspective-rendered voxel objects with exact, hidden surface labels.

This is an analytic ray-box renderer, not Kubric or a photorealistic benchmark.
Shape identities are disjoint up to proper cube rotations and translations.
"""

import hashlib
import itertools
from pathlib import Path
import numpy as np
from disco_geo.data.bundle import make_splits, ring_transitions
from disco_geo.io import write_json


def canonical_shape(cells):
    cells = np.asarray(cells, np.int32)
    candidates = []
    for perm in itertools.permutations(range(3)):
        mat = np.eye(3, dtype=int)[:, perm]
        for signs in itertools.product((-1, 1), repeat=3):
            rot = mat * np.asarray(signs)
            if round(np.linalg.det(rot)) != 1:
                continue
            points = cells @ rot
            points -= points.min(axis=0)
            candidates.append(tuple(sorted(map(tuple, points))))
    return min(candidates)


def random_shape(rng, parts):
    cells = {(0, 0, 0)}
    moves = np.r_[np.eye(3, dtype=int), -np.eye(3, dtype=int)]
    while len(cells) < parts:
        source = sorted(cells)[int(rng.integers(len(cells)))]
        cells.add(tuple(np.asarray(source) + moves[rng.integers(6)]))
    return np.asarray(sorted(cells), np.float32)


def render_object(cells, yaw, size=64, elevation=0.25):
    """Return RGB, Euclidean ray depth, world normals, mask, and camera position."""
    span = np.ptp(cells, axis=0) + 1
    scale = 1.0 / span.max()
    center = (cells.min(axis=0) + cells.max(axis=0)) / 2
    centers = (cells - center) * scale
    lo, hi = centers - scale / 2, centers + scale / 2
    camera = np.array([2 * np.cos(yaw), 2 * elevation, 2 * np.sin(yaw)])
    forward = -camera / np.linalg.norm(camera)
    right = np.cross(forward, [0, 1, 0])
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    uv = (np.arange(size) + 0.5) / size * 2 - 1
    xx, yy = np.meshgrid(uv, -uv)
    focal = np.tan(np.deg2rad(35) / 2)
    rays = forward + focal * (xx[..., None] * right + yy[..., None] * up)
    rays /= np.linalg.norm(rays, axis=-1, keepdims=True)
    safe = np.where(np.abs(rays) < 1e-8, 1e-8, rays)
    t0 = (lo[:, None, None, :] - camera) / safe[None]
    t1 = (hi[:, None, None, :] - camera) / safe[None]
    near_axes, far_axes = np.minimum(t0, t1), np.maximum(t0, t1)
    near, far = near_axes.max(-1), far_axes.min(-1)
    hits = (far >= np.maximum(near, 0)) & (near > 0)
    distances = np.where(hits, near, np.inf)
    cube = distances.argmin(0)
    depth = distances.min(0)
    mask = np.isfinite(depth)
    chosen_axes = np.take_along_axis(near_axes, cube[None, ..., None], axis=0)[0]
    axis = chosen_axes.argmax(-1)
    normal = np.eye(3)[axis] * -np.sign(rays)
    normal[~mask] = 0
    light = np.array([0.4, 0.8, 0.6])
    light /= np.linalg.norm(light)
    shade = 0.35 + 0.65 * np.maximum(normal @ light, 0)
    rgb = np.full((size, size, 3), 235, np.uint8)
    # The same material for all identities avoids identity-specific color labels.
    rgb[mask] = np.clip(shade[mask, None] * np.array([165, 180, 200]), 0, 255)
    depth[~mask] = 0
    return rgb, depth.astype(np.float32), normal.astype(np.float16), mask, camera


def generate(root, scenes=256, views=12, size=64, parts=7, seed=0):
    if scenes < 10 or views < 3 or size < 8 or parts < 5:
        raise ValueError("Use >=10 scenes, >=3 views, >=8 pixels and >=5 parts")
    root = Path(root)
    if (root / "manifest.json").exists():
        raise FileExistsError(f"Dataset already exists: {root}")
    root.mkdir(parents=True, exist_ok=True)
    labels = root / "eval_labels"
    labels.mkdir(exist_ok=True)
    shape = (scenes, views, size, size)
    rgb = np.lib.format.open_memmap(root / "rgb.npy", mode="w+", dtype=np.uint8, shape=(*shape, 3))
    depth = np.lib.format.open_memmap(
        labels / "depth.npy", mode="w+", dtype=np.float32, shape=shape
    )
    normal = np.lib.format.open_memmap(
        labels / "normals.npy", mode="w+", dtype=np.float16, shape=(*shape, 3)
    )
    masks = np.lib.format.open_memmap(labels / "mask.npy", mode="w+", dtype=np.bool_, shape=shape)
    rng = np.random.default_rng(seed)
    yaw = np.broadcast_to(np.linspace(0, 2 * np.pi, views, endpoint=False), (scenes, views)).copy()
    ids, seen, occupancy = [], set(), []
    for s in range(scenes):
        for _ in range(10_000):
            cells = random_shape(rng, parts)
            key = canonical_shape(cells)
            if key not in seen:
                break
        else:
            raise ValueError("Too many unique objects requested; increase parts")
        seen.add(key)
        ids.append(hashlib.sha256(repr(key).encode()).hexdigest()[:16])
        occupancy.append(cells)
        for v in range(views):
            rgb[s, v], depth[s, v], normal[s, v], masks[s, v], _ = render_object(
                cells, yaw[s, v], size
            )
        if (s + 1) % 32 == 0:
            print(f"Rendered {s + 1}/{scenes} objects", flush=True)
    for arr in (rgb, depth, normal, masks):
        arr.flush()
    np.save(root / "transitions.npy", ring_transitions(yaw))
    np.save(labels / "yaw.npy", yaw.astype(np.float32))
    np.save(labels / "occupancy.npy", np.asarray(occupancy))
    write_json(
        root / "manifest.json",
        {
            "format_version": 1,
            "source": "analytic_voxel_objects",
            "seed": seed,
            "scene_ids": ids,
            "splits": make_splits(ids, seed),
            "label_conventions": {
                "depth": "Euclidean distance along unit camera ray; background=0",
                "normals": "world-space outward surface normals",
                "yaw": "radians around world Y",
            },
            "recognition": "same identity across views; balanced same/different reward",
        },
    )
    return root
