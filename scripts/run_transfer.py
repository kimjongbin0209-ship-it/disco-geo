"""Fit matched frozen adapters and generate paired controls on fixed test inputs."""

import argparse
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from disco_geo.generative.external import weight_paths  # noqa: E402
from disco_geo.io import sha256, write_json  # noqa: E402


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", required=True)
    p.add_argument("--weights", required=True)
    p.add_argument("--features", nargs="+", required=True, metavar="LABEL=FEATURES.npy")
    p.add_argument("--output", default="runs/transfer")
    p.add_argument("--test-items", type=int, default=1)
    p.add_argument("--adapter-epochs", type=int, default=30)
    p.add_argument("--adapter-seed", type=int, default=0)
    p.add_argument("--sampling-seed", type=int, default=42)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--num-gaussians", type=int, default=32768)
    p.add_argument("--device", default="cuda")
    p.add_argument("--adapter-device", default="cpu")
    p.add_argument("--diagnostic-dataset", help="Report UNREGISTERED orbit diagnostics")
    p.add_argument("--resume", action="store_true")
    return p


def execute(args):
    if args.test_items <= 0 or args.adapter_epochs <= 0:
        raise ValueError("Test item count and adapter epochs must be positive")
    cache = Path(args.cache)
    manifest = json.loads((cache / "manifest.json").read_text())
    test = [item for item in manifest["items"] if item["split"] == "test"][: args.test_items]
    if len(test) != args.test_items:
        raise ValueError("The cache has fewer test inputs than requested")
    encoders, metadata = {}, {}
    for spec in args.features:
        label, separator, file = spec.partition("=")
        if not separator or not re.fullmatch(r"[A-Za-z0-9_-]+", label) or label in encoders:
            raise ValueError("Provide unique safe labels as LABEL=FEATURES.npy")
        meta = json.loads(Path(file).with_suffix(".json").read_text())
        if (
            not meta.get("frozen")
            or meta["dataset_manifest_sha256"] != manifest["dataset_manifest_sha256"]
        ):
            raise ValueError(f"Dataset provenance mismatch for {label}")
        encoders[label] = file
        metadata[label] = {key: meta[key] for key in ("method", "ablation", "seed")}
    settings = {key: value for key, value in vars(args).items() if key != "resume"}
    plan = {
        **settings,
        "selected_test_inputs": test,
        "encoders": metadata,
        "features_sha256": {label: sha256(file) for label, file in encoders.items()},
        "cache_manifest_sha256": sha256(cache / "manifest.json"),
        "generator_weight_sha256": {
            key: sha256(file) for key, file in weight_paths(args.weights).items()
        },
        "control_modes": ["native", "learned", "zero", "shuffled"],
        "encoder_updated": False,
    }
    output = Path(args.output)
    plan_file = output / "plan.json"
    if plan_file.exists():
        if not args.resume or json.loads(plan_file.read_text()) != plan:
            raise ValueError(
                "Existing transfer requires --resume with identical settings and features; otherwise choose a new output"
            )
    elif output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Unrecognized output directory: {output}")
    else:
        write_json(plan_file, plan)

    def call(*arguments):
        subprocess.run([sys.executable, "-m", "disco_geo", *map(str, arguments)], check=True)

    def stage(target, marker, action):
        if (target / marker).exists():
            if args.resume:
                return
            raise FileExistsError(target)
        if target.exists():
            raise FileExistsError(f"Incomplete/unknown stage: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".stage-", dir=target.parent) as temporary:
            scratch = Path(temporary) / "output"
            action(scratch)
            if not (scratch / marker).exists():
                raise RuntimeError(f"Stage did not complete: {target}")
            scratch.rename(target)

    start = time.monotonic()
    for label, file in encoders.items():
        stage(
            output / "adapters" / label,
            "result.json",
            lambda scratch, file=file: call(
                "train-adapter",
                "--features",
                file,
                "--cache",
                cache,
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
    for item in test:
        item_root = output / f"scene{item['scene']}_view{item['view']}"
        common = (
            "--weights",
            args.weights,
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
        )
        jobs = [("native", "native", None, None)]
        jobs.extend(
            (f"{label}-{mode}", mode, file, output / "adapters" / label / "best.pt")
            for label, file in encoders.items()
            for mode in ("learned", "zero", "shuffled")
        )
        for name, mode, file, adapter in jobs:
            target = item_root / name
            source = (
                ("--image", cache / item["image"])
                if mode == "native"
                else (
                    "--features",
                    file,
                    "--adapter",
                    adapter,
                    "--scene",
                    item["scene"],
                    "--view",
                    item["view"],
                )
            )
            stage(
                target,
                "generation.json",
                lambda scratch, mode=mode, source=source: call(
                    "generate", *common, "--mode", mode, "--output", scratch, *source
                ),
            )
            if args.diagnostic_dataset:
                stage(
                    target / "evaluation",
                    "metrics.json",
                    lambda scratch: call(
                        "evaluate-generation",
                        "--generation",
                        target,
                        "--dataset",
                        args.diagnostic_dataset,
                        "--scene",
                        item["scene"],
                        "--source-view",
                        item["view"],
                        "--output",
                        scratch,
                        "--allow-unregistered",
                    ),
                )
    write_json(
        output / "completed.json",
        {
            "wall_seconds_this_attempt": time.monotonic() - start,
            "encoder_count": len(encoders),
            "test_items": len(test),
            "generation_count": len(test) * (1 + 3 * len(encoders)),
        },
    )
    return output


if __name__ == "__main__":
    execute(parser().parse_args())
