"""Lazy imports keep JAX and PyTorch workloads in separate processes."""

import argparse
import json
from pathlib import Path


def parser():
    p = argparse.ArgumentParser(
        prog="disco-geo", description="RGB-only Disco103 research + frozen TripoSplat transfer"
    )
    sub = p.add_subparsers(dest="command", required=True)
    q = sub.add_parser("prepare", help="Render a small object benchmark with hidden labels")
    q.add_argument("--output", default="datasets/objects")
    q.add_argument("--scenes", type=int, default=256)
    q.add_argument("--views", type=int, default=12)
    q.add_argument("--size", type=int, default=64)
    q.add_argument("--parts", type=int, default=7)
    q.add_argument("--seed", type=int, default=0)
    q = sub.add_parser(
        "convert-gqn", help="Convert local TFRecords; does not download entire datasets"
    )
    q.add_argument("files", nargs="+")
    q.add_argument("--output", required=True)
    q.add_argument("--max-scenes", type=int, default=10000)
    q.add_argument("--seed", type=int, default=0)
    q = sub.add_parser("train")
    q.add_argument("--config", default="configs/objects.json")
    for name in ("dataset", "output", "method", "task", "ablation"):
        q.add_argument(f"--{name}")
    for name in ("seed", "steps", "batch-size", "rollout-length", "snapshot-every"):
        q.add_argument(f"--{name}", type=int)
    q.add_argument("--resume", action="store_true")
    q = sub.add_parser("benchmark-rl", help="Measure warmed-up throughput on the actual host")
    q.add_argument("--config", default="configs/objects.json")
    q.add_argument("--dataset")
    q.add_argument("--method")
    q.add_argument("--steps", type=int)
    q.add_argument("--updates", type=int, default=20)
    q.add_argument("--warmup", type=int, default=3)
    q.add_argument("--output", required=True)
    q = sub.add_parser("report", help="Rebuild CSV/JSON tables and PNG/PDF plots")
    q.add_argument("--study", default="runs/study")
    q.add_argument("--output")
    q = sub.add_parser("report-transfer", help="Compare final paired test generations")
    q.add_argument("--transfer", required=True)
    q.add_argument("--output")
    q = sub.add_parser("extract")
    q.add_argument("--checkpoint", required=True)
    q.add_argument("--dataset", required=True)
    q.add_argument("--output", required=True)
    q.add_argument("--batch-size", type=int, default=128)
    q = sub.add_parser("probe")
    q.add_argument("--dataset", required=True)
    q.add_argument("--features", required=True)
    q.add_argument("--output", required=True)
    q.add_argument("--label-budget", type=int, default=5000)
    q.add_argument("--seed", type=int, default=0)
    q.add_argument("--evaluation-split", choices=("test", "probe_val"), default="test")
    q = sub.add_parser("evaluate-policy")
    q.add_argument("--checkpoint", required=True)
    q.add_argument("--dataset", required=True)
    q.add_argument("--episodes", type=int, default=256)
    q.add_argument("--output", required=True)
    q.add_argument("--random", action="store_true")
    q = sub.add_parser(
        "cache-conditioning", help="Precompute native image features; never updates RL encoder"
    )
    q.add_argument("--dataset", required=True)
    q.add_argument("--weights", required=True)
    q.add_argument("--output", required=True)
    q.add_argument("--max-items", type=int, default=128)
    q.add_argument("--seed", type=int, default=0)
    q.add_argument("--device", default="cuda")
    q = sub.add_parser("train-adapter")
    q.add_argument("--features", required=True)
    q.add_argument("--cache", required=True)
    q.add_argument("--output", required=True)
    q.add_argument("--epochs", type=int, default=30)
    q.add_argument("--token-batch", type=int, default=128)
    q.add_argument("--seed", type=int, default=0)
    q.add_argument("--device", default="cpu")
    q = sub.add_parser("generate")
    q.add_argument("--weights", required=True)
    q.add_argument("--output", required=True)
    q.add_argument("--mode", choices=("native", "learned", "zero", "shuffled"), default="native")
    q.add_argument("--image")
    q.add_argument("--features")
    q.add_argument("--adapter")
    q.add_argument("--scene", type=int, default=0)
    q.add_argument("--view", type=int, default=0)
    q.add_argument("--seed", type=int, default=42)
    q.add_argument("--steps", type=int, default=20)
    q.add_argument("--num-gaussians", type=int, default=32768)
    q.add_argument("--device", default="cuda")
    q.add_argument("--exports", nargs="+", choices=("ply", "splat", "npz"), default=["ply", "npz"])
    q = sub.add_parser("evaluate-generation")
    q.add_argument("--generation", required=True)
    q.add_argument("--dataset", required=True)
    q.add_argument("--scene", type=int, required=True)
    q.add_argument("--source-view", type=int, required=True)
    q.add_argument("--output", required=True)
    q.add_argument(
        "--alignment", help="Verified 4x4 similarity transform JSON; estimated without test views"
    )
    q.add_argument("--allow-unregistered", action="store_true")
    q.add_argument("--evaluation-split", choices=("test", "probe_val"), default="test")
    q = sub.add_parser(
        "download-triposplat", help="Explicitly download pretrained weights (several GB)"
    )
    q.add_argument("--output", default="checkpoints/triposplat")
    return p


def main(argv=None):
    args = vars(parser().parse_args(argv))
    command = args.pop("command")
    if command == "prepare":
        from disco_geo.data.procedural import generate

        args["root"] = args.pop("output")
        result = generate(**args)
    elif command == "convert-gqn":
        from disco_geo.data.gqn import convert

        args["root"] = args.pop("output")
        result = convert(**args)
    elif command == "train":
        from disco_geo.config import TrainConfig
        from disco_geo.training.runner import train

        path = args.pop("config")
        resume = args.pop("resume")
        result = train(TrainConfig.load(path, **args), resume=resume)
    elif command == "benchmark-rl":
        from disco_geo.config import TrainConfig
        from disco_geo.training.benchmark import benchmark_rl

        path = args.pop("config")
        options = {key: args.pop(key) for key in ("output", "updates", "warmup")}
        result = benchmark_rl(TrainConfig.load(path, **args), **options)
    elif command == "report":
        from disco_geo.evaluation.report import create_report

        result = create_report(**args)
    elif command == "report-transfer":
        from disco_geo.evaluation.transfer_report import create_transfer_report

        result = create_transfer_report(**args)
    elif command == "extract":
        from disco_geo.evaluation.features import extract

        result = extract(**args)
    elif command == "probe":
        from disco_geo.evaluation.probes import run_probes

        result = run_probes(**args)
    elif command == "evaluate-policy":
        from disco_geo.agents.disco import GeoAgent
        from disco_geo.config import TrainConfig
        from disco_geo.io import load_checkpoint, write_json
        from disco_geo.training.runner import make_env, evaluate_policy

        saved = load_checkpoint(args["checkpoint"])
        config = TrainConfig(**dict(saved["config"], dataset=args["dataset"]))
        _, env = make_env(config, "test")
        result = evaluate_policy(
            GeoAgent(config, env),
            saved["learner"].params,
            config,
            split="test",
            episodes=args["episodes"],
            random=args["random"],
        )
        write_json(args["output"], result)
    elif command == "cache-conditioning":
        from disco_geo.generative.cache import cache_conditioning

        result = cache_conditioning(**args)
    elif command == "train-adapter":
        from disco_geo.generative.train_adapter import train_adapter

        result = train_adapter(**args)
    elif command == "generate":
        from disco_geo.generative.generate import generate

        result = generate(**args)
    elif command == "evaluate-generation":
        from disco_geo.generative.evaluate import evaluate_generation

        result = evaluate_generation(**args)
    elif command == "download-triposplat":
        from huggingface_hub import snapshot_download
        from disco_geo.generative.external import WEIGHTS

        result = snapshot_download(
            repo_id="VAST-AI/TripoSplat",
            local_dir=args["output"],
            allow_patterns=list(WEIGHTS.values()),
        )
    else:
        raise AssertionError(command)
    print(
        json.dumps(result, indent=2) if isinstance(result, dict) else str(Path(result)), flush=True
    )
