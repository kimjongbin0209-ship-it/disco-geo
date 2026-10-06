"""Resolve the audited source bundled with this checkout."""

import hashlib
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
DISCO_COMMIT = "9059a29f7121d60948f25ef165e08e050e9399c8"
RULE_HASH = "dac840a15b81e8805326363458c33979626172618f620d9e6e170315388530aa"


def ensure_disco():
    path = ROOT / "third_party/disco_rl"
    if not (path / "disco_rl/agent.py").exists():
        raise RuntimeError("Use an editable install from the complete DiscoGeo checkout")
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def load_rule():
    import numpy as np

    path = ROOT / "third_party/disco_rl/disco_rl/update_rules/weights/disco_103.npz"
    if hashlib.sha256(path.read_bytes()).hexdigest() != RULE_HASH:
        raise ValueError("Released Disco103 checkpoint checksum mismatch")
    with np.load(path, allow_pickle=False) as flat:
        return {
            key.rsplit("/", 1)[0]: {
                "w": flat[key.rsplit("/", 1)[0] + "/w"],
                "b": flat[key.rsplit("/", 1)[0] + "/b"],
            }
            for key in flat.files
        }
