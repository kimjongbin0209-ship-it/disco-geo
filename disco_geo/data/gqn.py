"""One-time conversion of a bounded local GQN TFRecord subset."""

from io import BytesIO
from pathlib import Path
import numpy as np
from PIL import Image
from disco_geo.data.bundle import make_splits, ring_transitions
from disco_geo.io import sha256, write_json


def convert(files, root, max_scenes=10_000, seed=0):
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError("GQN conversion needs the optional gqn dependencies") from exc
    # Keep TensorFlow from reserving the JAX training GPU.
    tf.config.set_visible_devices([], "GPU")
    files = [Path(p) for p in files]
    root = Path(root)
    if (root / "manifest.json").exists():
        raise FileExistsError(root)
    frames, cameras, ids = [], [], []
    for file in files:
        file_digest = sha256(file)
        for index, record in enumerate(tf.data.TFRecordDataset(str(file))):
            example = tf.train.Example.FromString(bytes(record.numpy()))
            f = example.features.feature
            images = np.stack(
                [
                    np.asarray(Image.open(BytesIO(b)).convert("RGB"))
                    for b in f["frames"].bytes_list.value
                ]
            )
            raw_pose = np.asarray(f["cameras"].float_list.value, np.float32).reshape(-1, 5)
            if len(images) != len(raw_pose):
                raise ValueError("GQN image and camera counts disagree")
            frames.append(images)
            cameras.append(raw_pose)
            ids.append(f"{file_digest}:{index}")
            if len(frames) >= max_scenes:
                break
        if len(frames) >= max_scenes:
            break
    poses = np.stack(cameras)
    root.mkdir(parents=True, exist_ok=True)
    (root / "eval_labels").mkdir(exist_ok=True)
    np.save(root / "rgb.npy", np.stack(frames))
    np.save(root / "transitions.npy", ring_transitions(poses[..., 3]))
    np.save(root / "eval_labels/yaw.npy", poses[..., 3])
    np.save(root / "eval_labels/camera.npy", poses)
    write_json(
        root / "manifest.json",
        {
            "format_version": 1,
            "source": "gqn",
            "scene_ids": ids,
            "splits": make_splits(ids, seed),
            "source_files": [str(p) for p in files],
            "warning": "Angular graph hops need not be equal camera rotations. No depth ground truth.",
        },
    )
    return root
