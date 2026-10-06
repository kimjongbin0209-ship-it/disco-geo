import json
import numpy as np
import pytest
from disco_geo.data.bundle import RGBBundle, ring_transitions
from disco_geo.data.procedural import canonical_shape, render_object


def test_rotated_and_translated_shapes_share_identity():
    cells = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [1, 1, 1]])
    rotated = np.c_[cells[:, 2], cells[:, 0], cells[:, 1]] + [4, -2, 9]
    assert canonical_shape(cells) == canonical_shape(rotated)


def test_training_does_not_open_annotation_files(dataset, monkeypatch):
    original = np.load

    def checked(path, *args, **kwargs):
        assert "eval_labels" not in str(path)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(np, "load", checked)
    images, transitions = RGBBundle(dataset).training_arrays()
    assert images.dtype == np.uint8
    assert transitions.shape == (*images.shape[:2], 2)


def test_scene_splits_are_disjoint_and_exhaustive(dataset):
    bundle = RGBBundle(dataset)
    parts = list(bundle.manifest["splits"].values())
    assert len({i for part in parts for i in part}) == sum(map(len, parts)) == 20
    assert len(set(bundle.scene_ids)) == 20


def test_overlap_is_rejected(dataset, tmp_path):
    manifest = json.loads((dataset / "manifest.json").read_text())
    manifest["splits"]["test"] = manifest["splits"]["train"][:2]
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="overlap"):
        RGBBundle(tmp_path)


def test_unsorted_angles_define_inverse_neighbors():
    yaw = np.array([[3.0, 0.0, 5.0, 1.0]])
    graph = ring_transitions(yaw)
    for view in range(4):
        assert graph[0, graph[0, view, 0], 1] == view
    assert graph[0, 1, 1] == 3
    with pytest.raises(ValueError, match="Duplicate"):
        ring_transitions([[0.0, 0.0, 1.0]])


def test_renderer_depth_and_normals_are_physical():
    rgb, depth, normals, mask, camera = render_object(np.zeros((1, 3)), 0, size=32, elevation=0)
    # Front face is x=+0.5 with camera x=2: central unit ray distance ~1.5.
    assert abs(depth[16, 16] - 1.5) < 0.01
    np.testing.assert_allclose(normals[16, 16], [1, 0, 0])
    assert np.all(depth[~mask] == 0)
    assert rgb.dtype == np.uint8
