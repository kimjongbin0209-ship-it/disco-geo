"""Monitor saved RL iterations on fixed validation identities, in separate processes."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from disco_geo.io import sha256, write_json  # noqa: E402


def build_plan(args):
    run = Path(args.run)
    config = json.loads((run / "config.json").read_text())
    manifest = json.loads((run / "snapshots/manifest.json").read_text())
    selected = [
        item
        for item in manifest["items"]
        if args.iterations is None or item["updates"] in args.iterations
    ]
    if not selected or (
        args.iterations and set(args.iterations) - {i["updates"] for i in selected}
    ):
        raise ValueError("Requested iterations must exist in snapshots/manifest.json")
    items = []
    cache_hash = None
    weight_hashes = None
    if not args.probes_only:
        if not args.cache or not args.weights or args.items < 1:
            raise ValueError("3D monitoring requires --cache, --weights and positive --items")
        cache = json.loads((Path(args.cache) / "manifest.json").read_text())
        cache_hash = sha256(Path(args.cache) / "manifest.json")
        if cache["dataset_manifest_sha256"] != sha256(Path(config["dataset"]) / "manifest.json"):
            raise ValueError("Conditioning cache and RL dataset differ")
        items = [i for i in cache["items"] if i["split"] == "probe_val"][: args.items]
        if len(items) != args.items:
            raise ValueError("Not enough probe_val cache items for the requested fixed monitor set")
        if not args.plan_only:
            from disco_geo.generative.external import weight_paths

            weight_hashes = {key: sha256(file) for key, file in weight_paths(args.weights).items()}
    return {
        "dataset": config["dataset"],
        "snapshots": selected,
        "monitor_inputs": items,
        "cache_manifest_sha256": cache_hash,
        "modes": [] if args.probes_only else args.modes,
        "generator_weight_sha256": weight_hashes,
        "adapter_epochs": args.adapter_epochs,
        "adapter_seed": args.adapter_seed,
        "sampling_seed": args.sampling_seed,
        "sampling_steps": args.steps,
        "num_gaussians": args.num_gaussians,
        "label_budget": args.label_budget,
        "device": args.device,
        "adapter_device": args.adapter_device,
        "evaluation_split": "probe_val",
        "test_identities_used": False,
        "iteration_unit": "learner update",
        "generation_jobs": len(selected) * len(items) * len(args.modes),
        "adapter_jobs": len(selected) if items and any(m != "native" for m in args.modes) else 0,
        "stage": "postprocess frozen RL snapshots; does not change RL losses",
    }


def execute(args):
    plan = build_plan(args)
    if args.plan_only:
        print(json.dumps(plan, indent=2))
        return plan
    dataset = plan["dataset"]
    run = Path(args.run)

    def call(*arguments):
        subprocess.run([sys.executable, "-m", "disco_geo", *map(str, arguments)], check=True)

    def stage(target, marker, action):
        # Only completed stages become visible; temporary files are owned by this call.
        if (target / marker).exists():
            if args.resume:
                return
            raise FileExistsError(f"Stage exists; use --resume: {target}")
        if target.exists():
            raise FileExistsError(f"Incomplete/unknown stage exists: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".stage-", dir=target.parent) as temporary:
            scratch = Path(temporary) / "output"
            action(scratch)
            if not (scratch / marker).exists():
                raise RuntimeError(f"Stage did not finish: {target}")
            scratch.rename(target)

    for item in plan["snapshots"]:
        start = time.monotonic()
        root = run / "snapshots" / Path(item["checkpoint"]).parent
        settings = {k: v for k, v in plan.items() if k != "snapshots"}
        settings["encoder_sha256"] = sha256(root / "encoder.pkl")
        plan_file = root / "monitor_plan.json"
        if plan_file.exists():
            if not args.resume or json.loads(plan_file.read_text()) != settings:
                raise ValueError(f"Use --resume with identical monitoring settings: {root}")
        else:
            write_json(plan_file, settings)
        if (root / "monitor_completed.json").exists():
            continue
        features = root / "features.npy"
        if not args.resume or not features.with_suffix(".json").exists():
            call(
                "extract",
                "--checkpoint",
                root / "encoder.pkl",
                "--dataset",
                dataset,
                "--output",
                features,
            )
        if not args.resume or not (root / "probes.json").exists():
            call(
                "probe",
                "--dataset",
                dataset,
                "--features",
                features,
                "--output",
                root / "probes.json",
                "--label-budget",
                args.label_budget,
                "--evaluation-split",
                "probe_val",
            )
        if plan["adapter_jobs"]:
            stage(
                root / "adapter",
                "result.json",
                lambda scratch: call(
                    "train-adapter",
                    "--features",
                    features,
                    "--cache",
                    args.cache,
                    "--output",
                    scratch,
                    "--epochs",
                    args.adapter_epochs,
                    "--seed",
                    args.adapter_seed,
                    "--device",
                    args.adapter_device,
                ),
            )
        for monitor in plan["monitor_inputs"]:
            for mode in plan["modes"]:
                target = (
                    root / "generation" / f"scene{monitor['scene']}_view{monitor['view']}" / mode
                )
                source = (
                    ("--image", Path(args.cache) / monitor["image"])
                    if mode == "native"
                    else (
                        "--features",
                        features,
                        "--adapter",
                        root / "adapter/best.pt",
                        "--scene",
                        monitor["scene"],
                        "--view",
                        monitor["view"],
                    )
                )
                stage(
                    target,
                    "generation.json",
                    lambda scratch, mode=mode, source=source: call(
                        "generate",
                        "--weights",
                        args.weights,
                        "--output",
                        scratch,
                        "--mode",
                        mode,
                        "--seed",
                        args.sampling_seed,
                        "--steps",
                        args.steps,
                        "--num-gaussians",
                        args.num_gaussians,
                        "--device",
                        args.device,
                        "--exports",
                        "ply",
                        "npz",
                        *source,
                    ),
                )
                stage(
                    target / "evaluation",
                    "metrics.json",
                    lambda scratch: call(
                        "evaluate-generation",
                        "--generation",
                        target,
                        "--dataset",
                        dataset,
                        "--scene",
                        monitor["scene"],
                        "--source-view",
                        monitor["view"],
                        "--output",
                        scratch,
                        "--evaluation-split",
                        "probe_val",
                        "--allow-unregistered",
                    ),
                )
        write_json(
            root / "monitor_completed.json",
            {
                "updates": item["updates"],
                "wall_seconds_this_attempt": time.monotonic() - start,
                "plan": settings,
            },
        )
    return plan


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True)
    p.add_argument("--cache")
    p.add_argument("--weights")
    p.add_argument("--probes-only", action="store_true")
    p.add_argument("--iterations", nargs="+", type=int)
    p.add_argument("--items", type=int, default=2)
    p.add_argument(
        "--modes", nargs="+", choices=("learned", "zero", "shuffled", "native"), default=["learned"]
    )
    p.add_argument("--adapter-epochs", type=int, default=30)
    p.add_argument("--adapter-seed", type=int, default=0)
    p.add_argument("--sampling-seed", type=int, default=42)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--num-gaussians", type=int, default=32768)
    p.add_argument("--label-budget", type=int, default=5000)
    p.add_argument("--device", default="cuda")
    p.add_argument("--adapter-device", default="cuda")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--plan-only", action="store_true")
    return p


if __name__ == "__main__":
    execute(parser().parse_args())
