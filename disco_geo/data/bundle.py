"""A strict separation between environment data and evaluation annotations."""

import json
from pathlib import Path
import numpy as np

SPLITS = ("train", "rl_val", "probe_train", "probe_val", "test")


def make_splits(scene_ids, seed=0):
    if len(scene_ids) < 10 or len(set(scene_ids)) != len(scene_ids):
        raise ValueError("At least ten unique scene identities are required")
    order = np.random.default_rng(seed).permutation(len(scene_ids))
    boundaries = np.round(np.array([0, 0.5, 0.6, 0.8, 0.9, 1]) * len(order)).astype(int)
    return {
        name: order[boundaries[i] : boundaries[i + 1]].tolist() for i, name in enumerate(SPLITS)
    }


def ring_transitions(yaw):
    yaw = np.asarray(yaw)
    if yaw.ndim != 2 or yaw.shape[1] < 3 or not np.isfinite(yaw).all():
        raise ValueError("yaw must contain at least three finite views per scene")
    order = np.argsort(np.mod(yaw, 2 * np.pi), axis=1)
    result = np.empty((*yaw.shape, 2), np.int32)
    for s, indices in enumerate(order):
        sorted_yaw = np.mod(yaw[s, indices], 2 * np.pi)
        if np.min(np.diff(np.r_[sorted_yaw, sorted_yaw[0] + 2 * np.pi])) < 1e-5:
            raise ValueError("Duplicate camera angles cannot define a ring")
        result[s, indices, 0] = np.roll(indices, 1)
        result[s, indices, 1] = np.roll(indices, -1)
    return result


class RGBBundle:
    """Training reads only RGB, transitions, scene identities, and split indices."""

    def __init__(self, root, split="train"):
        self.root = Path(root)
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        self._validate_manifest()
        if split not in SPLITS:
            raise ValueError(f"Unknown split: {split}")
        self.indices = np.asarray(self.manifest["splits"][split], np.int32)
        self.images = np.load(self.root / "rgb.npy", mmap_mode="r", allow_pickle=False)
        self.transitions = np.load(self.root / "transitions.npy", allow_pickle=False)
        self.scene_ids = np.asarray(self.manifest["scene_ids"])
        if self.images.dtype != np.uint8 or self.images.ndim != 5 or self.images.shape[-1] != 3:
            raise ValueError("RGB must be uint8 [scene, view, height, width, 3]")
        n, views = self.images.shape[:2]
        if n != len(self.scene_ids) or self.transitions.shape != (n, views, 2):
            raise ValueError("Image/identity/transition dimensions disagree")
        if np.any(self.transitions < 0) or np.any(self.transitions >= views):
            raise ValueError("Transition index out of bounds")

    def _validate_manifest(self):
        ids = self.manifest["scene_ids"]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate scene identities")
        parts = [self.manifest["splits"][name] for name in SPLITS]
        all_indices = [i for part in parts for i in part]
        if any(not part for part in parts):
            raise ValueError("Every scene-level split must be nonempty")
        if sorted(all_indices) != list(range(len(ids))):
            raise ValueError("Splits overlap, omit scenes, or contain invalid indices")

    def training_arrays(self):
        return np.asarray(self.images[self.indices]), self.transitions[self.indices]


def load_evaluation_labels(root):
    """Call only after freezing the encoder; never used by representation training."""
    root = Path(root) / "eval_labels"
    return {p.stem: np.load(p, mmap_mode="r", allow_pickle=False) for p in root.glob("*.npy")}
