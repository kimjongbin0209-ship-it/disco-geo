"""Progress of every run in a stage directory; standard library only.

Usage (on the server, from the repository root):
    python3 scripts/stage_status.py runs/pilot
Prints, per run: steps done / budget, recent speed, ETA, latest and best validation
return, teacher acting return and gate state, and whether it finished or failed.
Queue logs under runs/logs/*_queue.log are summarized at the end.
"""

import json
from pathlib import Path
import sys


def load_rows(path):
    rows = []
    for line in path.read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass  # a line being written right now
    return rows


def run_status(directory):
    config = json.loads((directory / "config.json").read_text())
    budget = config["steps"]
    metrics = directory / "metrics.jsonl"
    rows = load_rows(metrics) if metrics.exists() else []
    done = rows[-1]["transitions"] if rows else 0
    recent = [row for row in rows[-30:]]
    rate = None
    if len(recent) > 2:
        span = recent[-1]["wall_seconds"] - recent[0]["wall_seconds"]
        rate = (recent[-1]["transitions"] - recent[0]["transitions"]) / span if span > 0 else None
    validations = [row["validation"]["mean_return"] for row in rows if "validation" in row]
    teacher = [row["teacher_validation"]["mean_return"] for row in rows if "teacher_validation" in row]
    if (directory / "completed.json").exists():
        state = "finished"
    else:
        log = Path("runs/logs") / f"{directory.name}.log"
        text = log.read_text(errors="replace") if log.exists() else ""
        state = "FAILED" if "Traceback" in text else ("running" if rows else "starting")
    return {
        "run": directory.name, "state": state, "steps": done, "budget": budget,
        "percent": 100 * done / budget, "rate": rate,
        "eta_hours": (budget - done) / rate / 3600 if rate and state == "running" else None,
        "validation_last": validations[-1] if validations else None,
        "validation_best": max(validations) if validations else None,
        "teacher_last": teacher[-1] if teacher else None,
        "gate_open": rows[-1].get("teacher_gate_open") if rows else None,
        "train_rewards_recent": sum(row["reward_events_this_chunk"] for row in rows[-50:]),
    }


def fmt(value, pattern="{:.2f}"):
    return "-" if value is None else pattern.format(value)


def main():
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "runs/pilot")
    runs = [run_status(d) for d in sorted(root.iterdir())
            if d.is_dir() and (d / "config.json").exists()]
    header = f"{'run':40s} {'state':9s} {'steps':>9s} {'%':>6s} {'step/s':>7s} {'ETA h':>6s} " \
             f"{'val last':>8s} {'val best':>8s} {'teacher':>8s} {'gate':>5s} {'rew/50upd':>9s}"
    print(header)
    print("-" * len(header))
    for row in runs:
        print(f"{row['run']:40s} {row['state']:9s} {row['steps']:9d} {row['percent']:6.1f} "
              f"{fmt(row['rate'], '{:.0f}'):>7s} {fmt(row['eta_hours'], '{:.1f}'):>6s} "
              f"{fmt(row['validation_last']):>8s} {fmt(row['validation_best']):>8s} "
              f"{fmt(row['teacher_last']):>8s} {fmt(row['gate_open'], '{}'):>5s} "
              f"{row['train_rewards_recent']:9d}")
    counts = {state: sum(row["state"] == state for row in runs)
              for state in ("finished", "running", "starting", "FAILED")}
    print("\n" + ", ".join(f"{state}: {count}" for state, count in counts.items()))
    for queue in sorted(Path("runs/logs").glob("*_queue.log")):
        lines = queue.read_text(errors="replace").splitlines()
        pending = [line for line in lines if line.startswith("start")]
        finished = [line for line in lines if line.startswith(("done", "FAILED"))]
        status = "finished" if any(line.startswith("stage finished") for line in lines) else "active"
        print(f"{queue.name}: {status}, started {len(pending)}, ended {len(finished)}")


if __name__ == "__main__":
    main()
