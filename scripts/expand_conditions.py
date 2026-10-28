"""Write one config per (stage condition, base method, seed) from experiments/conditions.json.

Usage: python scripts/expand_conditions.py --stage P1_learning_gate --output runs/pilot/configs
Each config gets its own run directory; existing files are never overwritten.
"""

import argparse
import json
from pathlib import Path

from disco_spatial.config import TrainConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--runs-root")
    options = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    plan = json.loads((root / "experiments/conditions.json").read_text())
    stage = next(s for s in plan["stages"] if s["id"] == options.stage)
    base = json.loads((root / plan["base_config"][stage["profile"]]).read_text())
    names = ([c["id"] for c in plan["conditions"]] if stage["conditions"] == "all"
             else stage["conditions"])
    conditions = {c["id"]: c for c in plan["conditions"]}
    output = Path(options.output)
    output.mkdir(parents=True, exist_ok=True)
    runs_root = Path(options.runs_root or f"runs/{stage['profile']}")
    written = []
    for name in names:
        for method in plan["base_methods"]:
            for seed in stage["seeds"]:
                label = f"{method}-{name}-seed{seed}"
                data = dict(base, **conditions[name]["overrides"], method=method, seed=seed,
                            output=str(runs_root / label))
                TrainConfig(**dict(data, channels=tuple(data["channels"]),
                                   validation_seeds=tuple(data["validation_seeds"])))
                path = output / f"{label}.json"
                if path.exists():
                    raise FileExistsError(path)
                path.write_text(json.dumps(data, indent=2) + "\n")
                written.append(str(path))
    print(json.dumps({"stage": stage["id"], "configs": written}, indent=2))


if __name__ == "__main__":
    main()
