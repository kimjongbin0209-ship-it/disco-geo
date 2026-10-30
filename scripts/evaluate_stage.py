"""Evaluate every finished run of a training stage, then pack the results.

Usage (repository root, environment prepared, e.g. `. ./server_env.sh`):
    python scripts/evaluate_stage.py runs/pilot --parallel 4 --cores 8

Steps, each skipped when its output already exists (so the script can resume):
1. Collect the common probe histories once (train / validation / final test).
2. Per finished run: extract frozen features of its best checkpoint and run the
   ability suite; run frozen-policy navigation with memory interventions on the
   policy-final-test seeds; for teacher runs, let the causal teacher act.
3. Per (method, seed): the same suite on the untrained init.pkl, and a
   random-policy baseline (init.pkl with sampled actions).
4. Reports per method (reference: fresh) and for H1 (reference: AC+aux fresh).
5. One archive with results, reports, logs, run records and checkpoints.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import queue
import subprocess
import sys
import tarfile
import time


PARTITIONS = {"probe_train": 3000000, "probe_validation": 4000000, "probe_final_test": 5000000}


def log(message):
    print(f"[{time.strftime('%Y-%m-%dT%H:%M:%S')}] {message}", flush=True)


class Slots:
    """Core blocks for concurrent jobs, so each job sees only its own cores."""

    def __init__(self, parallel, cores):
        self.cores, self.free = cores, queue.Queue()
        for slot in range(parallel):
            self.free.put(slot)

    def run(self, name, command, logs):
        slot = self.free.get()
        try:
            first = slot * self.cores
            pinned = ["taskset", "-c", f"{first}-{first + self.cores - 1}"] if _has_taskset() else []
            log(f"start {name}")
            with open(logs / f"{name}.log", "w") as stream:
                code = subprocess.call(pinned + command, stdout=stream, stderr=subprocess.STDOUT)
            log(f"{'done' if code == 0 else 'FAILED'} {name}")
            return code == 0
        finally:
            self.free.put(slot)


def _has_taskset():
    return subprocess.call(["which", "taskset"], stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL) == 0


def cli(*args):
    return [sys.executable, "-m", "disco_spatial", *map(str, args)]


def finished_runs(stage):
    runs = []
    for directory in sorted(stage.iterdir()):
        if not (directory / "completed.json").exists():
            continue
        config = json.loads((directory / "config.json").read_text())
        if config.get("environment", "memory_maze") != "memory_maze":
            continue  # positive controls have no Memory Maze ability or navigation evaluation
        method, seed = config["method"], config["seed"]
        condition = directory.name[len(method) + 1:].rsplit(f"-seed{seed}", 1)[0]
        checkpoint = directory / ("best.pkl" if (directory / "best.pkl").exists() else "latest.pkl")
        runs.append({"directory": directory, "method": method, "seed": seed,
                     "condition": condition, "checkpoint": checkpoint,
                     "teacher": config.get("teacher", "none") != "none"})
    return runs


def abilities_job(checkpoint, tag, data, out):
    """Extract three partitions and run the suite; one shell-free command list."""
    steps = []
    for partition in PARTITIONS:
        target = out / "features" / tag / partition
        if not (target / "manifest.json").exists():
            if target.exists():  # interrupted extraction: keep it aside, never overwrite
                target.rename(target.with_name(f"{partition}.incomplete-{int(time.time())}"))
            steps.append(cli("extract", "--checkpoint", checkpoint, "--collection", data / partition,
                             "--output", target))
    steps.append(cli("abilities", "--train", out / "features" / tag / "probe_train",
                     "--validation", out / "features" / tag / "probe_validation",
                     "--test", out / "features" / tag / "probe_final_test",
                     "--output", out / "abilities" / f"{tag}.json"))
    return steps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage")
    parser.add_argument("--parallel", type=int, default=4)
    parser.add_argument("--cores", type=int, default=8)
    parser.add_argument("--probe-train-episodes", type=int, default=24)
    parser.add_argument("--probe-eval-episodes", type=int, default=8)
    parser.add_argument("--test-seeds", type=int, default=8)
    parser.add_argument("--policy", choices=("greedy", "sampled"), default="greedy",
                        help="Trained-policy actions; 'sampled' matches the random baseline")
    parser.add_argument("--max-steps", type=int, help="Shorten episodes (smoke tests only)")
    options = parser.parse_args()
    stage = Path(options.stage)
    name = stage.name
    out = Path("runs/evaluation") / name
    data = Path("datasets") / name
    logs = out / "logs"
    for sub in ("features", "abilities", "navigation", "reports"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    slots = Slots(options.parallel, options.cores)
    runs = finished_runs(stage)
    if not runs:
        raise SystemExit(f"No finished runs in {stage}")
    log(f"{len(runs)} finished runs in {stage}")

    # 1. Common histories, collected once and shared by every checkpoint.
    collect = []
    for partition, seed in PARTITIONS.items():
        target = data / partition
        if (target / "manifest.json").exists():
            continue
        if target.exists():  # an interrupted collection: keep it aside, never overwrite
            target.rename(target.with_name(f"{partition}.incomplete-{int(time.time())}"))
        episodes = options.probe_train_episodes if partition == "probe_train" else options.probe_eval_episodes
        command = cli("collect", "--partition", partition, "--seed", seed, "--episodes", episodes,
                      "--output", target)
        if options.max_steps:
            command += ["--max-steps", str(options.max_steps)]
        collect.append((f"collect-{partition}", command))
    with ThreadPoolExecutor(options.parallel) as pool:
        if not all(pool.map(lambda job: slots.run(job[0], job[1], logs), collect)):
            raise SystemExit("Common-history collection failed; see the logs")

    # 2-3. Per-run and per-(method, seed) jobs; each job is a sequence of commands.
    test_seeds = [str(2000000 + i) for i in range(options.test_seeds)]
    steps = ["--max-steps", str(options.max_steps)] if options.max_steps else []
    sample = ["--sample"] if options.policy == "sampled" else []
    jobs = []
    for run in runs:
        tag = run["directory"].name
        if not (out / "abilities" / f"{tag}.json").exists():
            jobs.append((f"abilities-{tag}", abilities_job(run["checkpoint"], tag, data, out)))
        target = out / "navigation" / f"{tag}.json"
        if not target.exists():
            jobs.append((f"navigate-{tag}", [cli(
                "navigate", "--checkpoint", run["checkpoint"], "--seeds", *test_seeds,
                "--interventions", "none", "reset_every_step", "truncate_16",
                "--output", target, *sample, *steps)]))
        target = out / "navigation" / f"{tag}.teacher.json"
        if run["teacher"] and not target.exists():
            jobs.append((f"teacher-{tag}", [cli(
                "navigate", "--checkpoint", run["checkpoint"], "--seeds", *test_seeds,
                "--interventions", "none", "--teacher", "--output", target, *sample, *steps)]))
    initial = {}
    for run in runs:  # one untrained control per (method, seed); identical across conditions
        initial.setdefault((run["method"], run["seed"]), run["directory"] / "init.pkl")
    for (method, seed), checkpoint in sorted(initial.items()):
        tag = f"{method}-init-seed{seed}"
        if not (out / "abilities" / f"{tag}.json").exists():
            jobs.append((f"abilities-{tag}", abilities_job(checkpoint, tag, data, out)))
        target = out / "navigation" / f"{method}-random-seed{seed}.json"
        if not target.exists():
            jobs.append((f"random-{method}-seed{seed}", [cli(
                "navigate", "--checkpoint", checkpoint, "--seeds", *test_seeds,
                "--interventions", "none", "reset_every_step", "--sample",
                "--output", target, *steps)]))

    def run_job(job):
        name, commands = job
        return all(slots.run(f"{name}-{i}" if len(commands) > 1 else name, command, logs)
                   for i, command in enumerate(commands))

    with ThreadPoolExecutor(options.parallel) as pool:
        results = list(pool.map(run_job, jobs))
    failed = [job[0] for job, ok in zip(jobs, results) if not ok]

    # 4. Reports: per method against its fresh runs, and H1 across methods.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from disco_spatial.evaluation.report import build_report

    def entry(label, seed, abilities, navigation):
        a = abilities if abilities.name != "-" and abilities.exists() else None
        n = navigation if navigation.exists() else None
        if a is None and n is None:
            return None
        return f"{label}:{seed}:{a or '-'}" + (f":{n}" if n else "")

    reports = {}
    for method in sorted({run["method"] for run in runs}):
        entries = []
        for run in runs:
            if run["method"] == method:
                tag = run["directory"].name
                entries.append(entry(run["condition"], run["seed"], out / "abilities" / f"{tag}.json",
                                     out / "navigation" / f"{tag}.json"))
                if run["teacher"]:  # G4b: the causal teacher acting on the same test seeds
                    entries.append(entry(f"{run['condition']}_teacher_acting", run["seed"],
                                         Path("-"), out / "navigation" / f"{tag}.teacher.json"))
        for (m, seed) in initial:
            if m == method:
                entries.append(entry("untrained_init", seed,
                                     out / "abilities" / f"{method}-init-seed{seed}.json",
                                     out / "navigation" / f"{method}-random-seed{seed}.json"))
        reports[method] = (entries, "fresh")
    h1 = [entry(run["method"], run["seed"], out / "abilities" / f"{run['directory'].name}.json",
                out / "navigation" / f"{run['directory'].name}.json")
          for run in runs if run["condition"] == "fresh"]
    if len({run["method"] for run in runs if run["condition"] == "fresh"}) > 1:
        reports["h1_disco_vs_acaux"] = (h1, "actor_critic_aux")
    for title, (entries, reference) in reports.items():
        entries = [e for e in entries if e]
        try:
            report = build_report(entries, reference)
        except ValueError as error:
            log(f"report {title} skipped: {error}")
            continue
        markdown = report.pop("markdown")
        (out / "reports" / f"{title}.json").write_text(json.dumps(report, indent=2))
        (out / "reports" / f"{title}.md").write_text(f"# {title} (reference: {reference})\n\n{markdown}")
        log(f"report {title} written")

    # 5. Archive for download (the instance has no persistent volume).
    archive = Path("runs") / f"{name}_results.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        for sub in ("abilities", "navigation", "reports", "logs"):
            tar.add(out / sub, arcname=f"{name}_results/evaluation/{sub}")
        for run in finished_runs(stage):
            for file in ("config.json", "provenance.json", "completed.json", "metrics.jsonl",
                         "protocol.json", "best.pkl", "init.pkl"):
                path = run["directory"] / file
                if path.exists():
                    tar.add(path, arcname=f"{name}_results/runs/{run['directory'].name}/{file}")
        for manifest in data.glob("*/manifest.json"):
            tar.add(manifest, arcname=f"{name}_results/datasets/{manifest.parent.name}/manifest.json")
    log(f"archive {archive} ({archive.stat().st_size / 1e6:.1f} MB)")
    if failed:
        log(f"FAILED jobs: {failed}")
        raise SystemExit(1)
    log("evaluation finished")


if __name__ == "__main__":
    main()
