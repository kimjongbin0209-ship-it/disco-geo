"""Write one config per (stage condition, base method, seed) from experiments/conditions.json.

Usage: python scripts/expand_conditions.py --stage P1_learning_gate --output runs/pilot/configs
Each config gets its own run directory; existing files are never overwritten.
--skip-finished leaves out runs whose output already holds completed.json, so a later
stage (e.g. P2 after P1) reuses finished runs instead of failing on their directories.
Settings apply in order: profile config, stage "overrides", condition "overrides". A
condition's "methods" limits its base methods; "all" leaves out diagnostic_only conditions.
A stage's "runs_root" replaces the default runs/<profile>.
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
    parser.add_argument("--skip-finished", action="store_true")
    options = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    plan = json.loads((root / "experiments/conditions.json").read_text())
    stage = next(s for s in plan["stages"] if s["id"] == options.stage)
    base = json.loads((root / plan["base_config"][stage["profile"]]).read_text())
    names = ([c["id"] for c in plan["conditions"] if not c.get("diagnostic_only")]
             if stage["conditions"] == "all" else stage["conditions"])
    conditions = {c["id"]: c for c in plan["conditions"]}
    output = Path(options.output)
    output.mkdir(parents=True, exist_ok=True)
    runs_root = Path(options.runs_root or stage.get("runs_root") or f"runs/{stage['profile']}")
    written, reused = [], []
    for name in names:
        for method in conditions[name].get("methods", plan["base_methods"]):
            for seed in stage["seeds"]:
                label = f"{method}-{name}-seed{seed}"
                data = dict(base)
                data.update(stage.get("overrides", {}))
                data.update(conditions[name]["overrides"])
                data.update(method=method, seed=seed, output=str(runs_root / label))
                TrainConfig(**dict(data, channels=tuple(data["channels"]),
                                   validation_seeds=tuple(data["validation_seeds"])))
                if options.skip_finished and (runs_root / label / "completed.json").exists():
                    reused.append(label)
                    continue
                path = output / f"{label}.json"
                if path.exists():
                    raise FileExistsError(path)
                path.write_text(json.dumps(data, indent=2) + "\n")
                written.append(str(path))
    print(json.dumps({"stage": stage["id"], "configs": written, "reused_finished_runs": reused},
                     indent=2))


if __name__ == "__main__":
    main()
