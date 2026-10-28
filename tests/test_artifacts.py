"""Readout artifacts cannot silently mix checkpoints, partitions or layouts."""

import json

import numpy as np
import pytest

from disco_spatial.artifacts import probe_features
from disco_spatial.io import sha256


def make_features(root, partition, offset, checkpoint="same-checkpoint", layout=None):
    root.mkdir()
    x = np.arange(12, dtype=float).reshape(6, 2) + offset
    path = root / "features.npz"
    np.savez_compressed(
        path, belief=x, image_features=x,
        target_origin_relative_position=np.c_[2 * x[:, 0], -x[:, 1]],
        layout_ids=np.full(6, layout or partition), episode_ids=np.full(6, partition + "-episode"),
    )
    manifest = {"frozen": True, "partition": partition, "checkpoint_sha256": checkpoint,
                "entries": [{"file": path.name, "sha256": sha256(path)}]}
    (root / "manifest.json").write_text(json.dumps(manifest))
    return root


def test_readout_is_explicitly_diagnostic_and_cannot_overwrite(tmp_path):
    train = make_features(tmp_path / "train", "probe_train", 0)
    val = make_features(tmp_path / "val", "probe_validation", 1)
    test = make_features(tmp_path / "test", "probe_final_test", 2)
    output = tmp_path / "result.json"
    result = probe_features(train, val, test, output)
    assert not result["primary_endpoint"] and result["confidence_interval"] is None
    assert result["mse"] < result["train_target_mean_baseline_mse"]
    assert output.with_suffix(".predictions.npz").is_file()
    with pytest.raises(FileExistsError):
        probe_features(train, val, test, output)


@pytest.mark.parametrize("violation", ["checkpoint", "partition", "layout", "checksum"])
def test_readout_provenance_rejects_mixed_sources(tmp_path, violation):
    train = make_features(tmp_path / "train", "probe_train", 0)
    val = make_features(tmp_path / "val", "probe_validation", 1)
    test = make_features(tmp_path / "test", "probe_final_test", 2,
                         checkpoint="other" if violation == "checkpoint" else "same-checkpoint",
                         layout="probe_train" if violation == "layout" else None)
    if violation in {"partition", "checksum"}:
        path = test / "manifest.json"
        manifest = json.loads(path.read_text())
        if violation == "partition":
            manifest["partition"] = "probe_validation"
        else:
            manifest["entries"][0]["sha256"] = "wrong-digest"
        path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        probe_features(train, val, test, tmp_path / "invalid-result.json")
    assert not (tmp_path / "invalid-result.json").exists()
