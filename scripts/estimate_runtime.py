"""Conditional budget from actual host measurements; not a published L40S benchmark."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from disco_geo.io import sha256, write_json  # noqa: E402


def estimate(args):
    profiles = {}
    for file in args.profiles:
        data = json.loads(Path(file).read_text())
        if data["platform"] != "gpu" and not args.allow_cpu:
            raise ValueError(
                "GPU estimates require GPU measurements; use --allow-cpu only for CPU budgets"
            )
        profiles[data["config"]["method"]] = data
    methods = ["actor_critic", "actor_critic_aux", "disco"]
    if args.ablations:
        methods += ["disco", "disco"]
    if set(methods) - set(profiles):
        raise ValueError("Provide benchmark JSONs for actor_critic, actor_critic_aux and disco")
    rl_hours = args.seeds * sum(profiles[m]["target_rl_plus_eval_hours"] for m in methods)
    # Initial + periodic + final points vary with policy; supply the observed/planned count.
    adapters = len(methods) * args.seeds * args.snapshots_per_trained_run + args.seeds
    generations = adapters * args.items * args.modes
    adapter = json.loads(Path(args.adapter_result).read_text()) if args.adapter_result else None
    generation = (
        json.loads(Path(args.generation_metadata).read_text()) if args.generation_metadata else None
    )
    adapter_hours = adapters * adapter["wall_seconds"] / 3600 if adapter else None
    generation_hours = generations * generation["wall_seconds"] / 3600 if generation else None
    total = rl_hours + adapter_hours + generation_hours if adapter and generation else None
    result = {
        "rl_training_plus_validation_hours": rl_hours,
        "trained_runs": len(methods) * args.seeds,
        "adapter_jobs": adapters,
        "generation_jobs": generations,
        "adapter_hours": adapter_hours,
        "generation_hours": generation_hours,
        "measured_stage_subtotal_hours": total,
        "devices": {m: profiles[m]["devices"] for m in set(methods)},
        "input_sha256": {
            file: sha256(file)
            for file in [*args.profiles, args.adapter_result, args.generation_metadata]
            if file
        },
        "assumptions": [
            "Disco loss ablations reuse Disco throughput; measure separately if substantially different.",
            "Each trained run uses the target step budget recorded in its benchmark JSON.",
            "Snapshot count includes initial and final; untrained encoders have one snapshot each.",
            "Adapters and generations reuse the measured settings, input sizes, and host.",
            "Subtotal excludes installation/download, cache building, feature extraction, probes, CPU orbit rendering, disk I/O and per-process RL recompilation.",
            "Not an observed complete experiment runtime; workload/policy changes can change throughput.",
        ],
    }
    write_json(args.output, result)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--profiles", nargs="+", required=True)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--ablations", action="store_true")
    p.add_argument("--snapshots-per-trained-run", type=int, default=5)
    p.add_argument("--items", type=int, default=2)
    p.add_argument("--modes", type=int, default=1)
    p.add_argument("--adapter-result")
    p.add_argument("--generation-metadata")
    p.add_argument("--allow-cpu", action="store_true")
    p.add_argument("--output", default="runs/runtime_estimate.json")
    args = p.parse_args()
    if min(args.seeds, args.snapshots_per_trained_run, args.items, args.modes) < 1:
        p.error("Counts must be positive")
    print(json.dumps(estimate(args), indent=2))
