"""Reproducible seeds and controls, each in its own process to release GPU memory."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

# Direct script execution works even when the runtime suppresses editable .pth hooks.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from disco_geo.evaluation.report import create_report  # noqa: E402
from disco_geo.io import sha256, write_json  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/objects.json")
    p.add_argument("--output", default="runs/study")
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--steps", type=int, default=500000)
    p.add_argument("--label-budget", type=int, default=5000)
    p.add_argument("--ablations", action="store_true")
    p.add_argument("--snapshot-every", type=int, default=0, help="Learner updates; zero disables")
    p.add_argument("--snapshot-cache", help="Generate 3D after each run using this native cache")
    p.add_argument("--snapshot-weights")
    p.add_argument("--snapshot-items", type=int, default=2)
    p.add_argument("--snapshot-epochs", type=int, default=30)
    p.add_argument(
        "--snapshot-probes", action="store_true", help="Probe each saved iteration without 3D"
    )
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    if args.snapshot_cache and (not args.snapshot_weights or not args.snapshot_every):
        p.error("--snapshot-cache requires --snapshot-weights and --snapshot-every")
    if args.snapshot_probes and not args.snapshot_every:
        p.error("--snapshot-probes requires --snapshot-every")
    dataset = json.loads(Path(args.config).read_text())["dataset"]
    methods = [
        ("untrained", "none"),
        ("actor_critic", "none"),
        ("actor_critic_aux", "none"),
        ("disco", "none"),
    ]
    if args.ablations:
        methods.extend([("disco", "no_yz_loss"), ("disco", "no_aux_policy")])

    def call(*arguments):
        subprocess.run([sys.executable, "-m", "disco_geo", *map(str, arguments)], check=True)

    for seed in args.seeds:
        for method, ablation in methods:
            start = time.monotonic()
            root = Path(args.output) / f"{method}-{ablation}" / f"seed{seed}"
            old_best = sha256(root / "best.pkl") if (root / "best.pkl").exists() else None
            resume = ["--resume"] if args.resume and (root / "latest.pkl").exists() else []
            call(
                "train",
                "--config",
                args.config,
                "--method",
                method,
                "--ablation",
                ablation,
                "--seed",
                seed,
                "--steps",
                args.steps,
                "--output",
                root,
                "--snapshot-every",
                args.snapshot_every,
                *resume,
            )
            changed = old_best != sha256(root / "best.pkl")
            feature_meta = (
                json.loads((root / "features.json").read_text())
                if (root / "features.json").exists()
                else {}
            )
            refresh = changed or feature_meta.get("checkpoint_sha256") != sha256(root / "best.pkl")
            if not args.resume or refresh:
                call(
                    "extract",
                    "--checkpoint",
                    root / "best.pkl",
                    "--dataset",
                    dataset,
                    "--output",
                    root / "features.npy",
                )
            probe_file = root / "probes.json"
            probe_budget = (
                json.loads(probe_file.read_text()).get("label_budget_requested")
                if probe_file.exists()
                else None
            )
            if not args.resume or refresh or probe_budget != args.label_budget:
                call(
                    "probe",
                    "--dataset",
                    dataset,
                    "--features",
                    root / "features.npy",
                    "--output",
                    root / "probes.json",
                    "--label-budget",
                    args.label_budget,
                )
            if not args.resume or changed or not (root / "policy.json").exists():
                call(
                    "evaluate-policy",
                    "--checkpoint",
                    root / "best.pkl",
                    "--dataset",
                    dataset,
                    "--output",
                    root / "policy.json",
                )
            if method == "untrained" and (
                not args.resume or not (root / "random_policy.json").exists()
            ):
                call(
                    "evaluate-policy",
                    "--checkpoint",
                    root / "best.pkl",
                    "--dataset",
                    dataset,
                    "--output",
                    root / "random_policy.json",
                    "--random",
                )

            if args.snapshot_cache or args.snapshot_probes:
                options = (
                    [
                        "--cache",
                        args.snapshot_cache,
                        "--weights",
                        args.snapshot_weights,
                        "--items",
                        args.snapshot_items,
                        "--adapter-epochs",
                        args.snapshot_epochs,
                    ]
                    if args.snapshot_cache
                    else ["--probes-only"]
                )
                subprocess.run(
                    [
                        sys.executable,
                        "scripts/run_snapshots.py",
                        "--run",
                        str(root),
                        "--label-budget",
                        str(args.label_budget),
                        *map(str, options),
                        *(["--resume"] if args.resume else []),
                    ],
                    check=True,
                )
            timing_file = root / "study_timing.json"
            attempts = (
                json.loads(timing_file.read_text())["attempts"] if timing_file.exists() else []
            )
            attempts.append({"wall_seconds_this_attempt": time.monotonic() - start})
            write_json(timing_file, {"attempts": attempts})
            create_report(args.output)
    report = create_report(args.output)
    write_json(
        Path(args.output) / "summary.json", json.loads((report / "summary.json").read_text())
    )


if __name__ == "__main__":
    main()
