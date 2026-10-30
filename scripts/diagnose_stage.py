"""Summarize a diagnostic stage (P1d) from its training logs and scripted reference policies.

Usage (repository root, environment prepared, e.g. `. ./server_env.sh`):
    python scripts/diagnose_stage.py runs/diagnostics --references --parallel 8

Per run, from metrics.jsonl (finished or still running):
* on-policy training reward: every training step is a sampled action of the current
  policy, so the reward rate over the last quarter of training is a far larger sample
  than a few validation episodes. Memory Maze runs are compared with the scripted
  uniform reference as a rate ratio (approximate CI; reward events treated as Poisson).
  Cue-recall positive controls are scored per trial against chance (1/6) and optimum (1).
* credit gap: how much more an update raises the taken action's log-probability on
  steps followed by a reward within credit_window steps than on steps that are not,
  with a standard error over blocks of 100 updates; the same for the rule's own signal
  (Disco: policy-target log-prob gain; AC: advantage).
* entropy, and for Disco the entropy of its policy target minus the policy's entropy.

--references first runs the scripted reference policies on the policy-final-test
seeds (split into parallel parts, merged here). Outputs go to
runs/evaluation/<stage name>/diagnostics.{json,md}.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np


BLOCK = 100


def rows_of(directory):
    with (directory / "metrics.jsonl").open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def quarter(rows):
    return rows[len(rows) - max(1, len(rows) // 4):]


def chunk_sizes(rows):
    """Transitions collected in each update, from the cumulative counter."""
    sizes, last = [], 0
    for row in rows:
        sizes.append(row["transitions"] - last)
        last = row["transitions"]
    return sizes


def rate(rows, sizes, selection):
    events = sum(rows[i]["reward_events_this_chunk"] for i in selection)
    return events, sum(sizes[i] for i in selection)


def weighted_gap(rows, key):
    """Count-weighted proximal and distal means, their gap, and a block standard error."""
    def means(part):
        out = {}
        for side in ("proximal", "distal"):
            pairs = [(r[f"credit_{key}_{side}"], r[f"credit_{side}_steps"]) for r in part
                     if r.get(f"credit_{key}_{side}") is not None]
            steps = sum(n for _, n in pairs)
            out[side] = sum(v * n for v, n in pairs) / steps if steps else None
            out[f"{side}_steps"] = steps
        return out

    total = means(rows)
    if total["proximal"] is None or total["distal"] is None:
        return dict(total, gap=None, gap_se=None)
    gaps = []
    for start in range(0, len(rows), BLOCK):
        block = means(rows[start:start + BLOCK])
        if block["proximal"] is not None and block["distal"] is not None:
            gaps.append(block["proximal"] - block["distal"])
    se = float(np.std(gaps, ddof=1) / math.sqrt(len(gaps))) if len(gaps) > 1 else None
    return dict(total, gap=total["proximal"] - total["distal"], gap_se=se, blocks=len(gaps))


def rate_ratio(a, ta, b, tb):
    if not (a and b and ta and tb):
        return None
    ratio = (a / ta) / (b / tb)
    half = 1.96 * math.sqrt(1 / a + 1 / b)
    return {"ratio": ratio, "ci95": [ratio * math.exp(-half), ratio * math.exp(half)]}


def summarize(directory, references):
    config = json.loads((directory / "config.json").read_text())
    rows = rows_of(directory)
    if not rows:
        return None
    sizes = chunk_sizes(rows)
    n = len(rows)
    early = range(0, max(1, n // 20))
    late = range(n - max(1, n // 4), n)
    events, steps = rate(rows, sizes, late)
    early_events, early_steps = rate(rows, sizes, early)
    entropy = [row["policy_entropy"] for row in rows]
    result = {
        "run": directory.name, "method": config["method"],
        "environment": config.get("environment", "memory_maze"), "seed": config["seed"],
        "entropy_cost": config.get("entropy_cost") if config["method"] != "disco" else None,
        "learning_rate": config.get("learning_rate"),
        "finished": (directory / "completed.json").exists(), "updates": n,
        "transitions": rows[-1]["transitions"], "planned_transitions": config["steps"],
        "entropy": {"first_5pct": float(np.mean([entropy[i] for i in early])),
                    "min": float(min(entropy)),
                    "last_quarter": float(np.mean([entropy[i] for i in late]))},
        "reward": {"last_quarter_events": events, "last_quarter_steps": steps,
                   "first_5pct_events": early_events, "first_5pct_steps": early_steps},
    }
    if result["environment"] == "cue_recall":
        trial = config.get("control_delay", 0) + 1
        per_trial = events / steps * trial if steps else None
        threshold = 0.8 if trial == 1 else 0.5
        result["reward"].update(per_trial_last_quarter=per_trial, chance=1 / 6, optimum=1.0,
                                criterion=threshold,
                                passed=per_trial is not None and per_trial >= threshold)
    else:
        result["reward"]["per_1000_last_quarter"] = 1000 * events / steps if steps else None
        result["reward"]["per_1000_first_5pct"] = (1000 * early_events / early_steps
                                                   if early_steps else None)
        comparisons = {}
        for name, reference in references.items():
            comparisons[name] = rate_ratio(events, steps, reference["total_reward"],
                                           reference["total_steps"])
        result["reward"]["vs_reference"] = comparisons
        uniform = comparisons.get("uniform")
        result["reward"]["above_uniform"] = (None if uniform is None
                                             else uniform["ci95"][0] > 1)
    if any("credit_window" in row for row in rows):
        diagnostic = [row for row in rows if "credit_window" in row]
        quarter_rows = diagnostic[:max(1, len(diagnostic) // 4)]
        result["credit"] = {
            "window": diagnostic[0]["credit_window"],
            "rule_signal_kind": diagnostic[0]["credit_rule_signal"],
            "update_logp_gain": weighted_gap(diagnostic, "update_logp_gain"),
            "update_logp_gain_first_quarter": weighted_gap(quarter_rows, "update_logp_gain"),
            "rule_signal": weighted_gap(diagnostic, "rule_signal"),
            "update_policy_kl_mean": float(np.mean([r["update_policy_kl"] for r in diagnostic])),
        }
        if config["method"] == "disco":
            def excess(part):
                values = [r["rule_pi_hat_entropy"] - r["rule_entropy"] for r in part
                          if "rule_pi_hat_entropy" in r and "rule_entropy" in r]
                return float(np.mean(values)) if values else None
            result["credit"]["target_minus_policy_entropy"] = {
                "first_10pct": excess(diagnostic[:max(1, len(diagnostic) // 10)]),
                "last_quarter": excess(quarter(diagnostic))}
    validations = [row["validation"] for row in rows if "validation" in row]
    if validations:
        result["validation_last"] = {key: validations[-1][key]
                                     for key in ("mean_return", "policy", "returns")}
    return result


def reference_jobs(stage_out, policies, episodes, parts):
    jobs = []
    for policy in policies:
        folder = stage_out / "references" / policy.replace(":", "_")
        seeds = [2000000 + i for i in range(episodes)]
        for index, chunk in enumerate(np.array_split(seeds, parts)):
            target = folder / f"part{index}.json"
            if len(chunk) and not target.exists():
                jobs.append((target, [sys.executable, "-m", "disco_spatial", "scripted",
                                      "--policy", policy, "--seeds", *map(str, chunk),
                                      "--output", str(target)]))
    return jobs


def load_references(stage_out):
    merged = {}
    for folder in sorted((stage_out / "references").glob("*")):
        parts = [json.loads(path.read_text()) for path in sorted(folder.glob("part*.json"))]
        if parts:
            merged[folder.name] = {
                "policy": parts[0]["policy"],
                "episodes": sum(len(p["episodes"]) for p in parts),
                "total_reward": sum(p["total_reward"] for p in parts),
                "total_steps": sum(p["total_steps"] for p in parts)}
    for reference in merged.values():
        reference["reward_per_1000_steps"] = 1000 * reference["total_reward"] / reference["total_steps"]
    return merged


def fmt(value, digits=3):
    return "-" if value is None else f"{value:.{digits}f}"


def markdown(results, references):
    lines = ["## Scripted references (Memory Maze 9x9, policy-final-test seeds)", "",
             "| policy | episodes | reward / 1000 steps |", "| --- | --- | --- |"]
    for reference in references.values():
        lines.append(f"| {reference['policy']} | {reference['episodes']} | "
                     f"{fmt(reference['reward_per_1000_steps'])} |")
    lines += ["", "## Positive controls (cue recall)", "",
              "| run | done | per-trial reward, last quarter | criterion | pass | entropy last |",
              "| --- | --- | --- | --- | --- | --- |"]
    for r in results:
        if r["environment"] == "cue_recall":
            lines.append(f"| {r['run']} | {r['transitions']}/{r['planned_transitions']} | "
                         f"{fmt(r['reward']['per_trial_last_quarter'])} | "
                         f">= {r['reward']['criterion']} | {r['reward']['passed']} | "
                         f"{fmt(r['entropy']['last_quarter'])} |")
    lines += ["", "## Memory Maze", "",
              "| run | done | reward/1k first 5% | reward/1k last quarter | vs uniform (95% CI) | "
              "entropy first/min/last | credit gap +- SE | rule-signal gap +- SE | "
              "target - policy entropy (first/last) |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in results:
        if r["environment"] != "memory_maze":
            continue
        uniform = r["reward"]["vs_reference"].get("uniform")
        versus = ("-" if uniform is None else
                  f"{uniform['ratio']:.2f} [{uniform['ci95'][0]:.2f}, {uniform['ci95'][1]:.2f}]")
        credit = r.get("credit", {})
        gain, signal = credit.get("update_logp_gain", {}), credit.get("rule_signal", {})
        excess = credit.get("target_minus_policy_entropy", {})
        lines.append(
            f"| {r['run']} | {r['transitions']}/{r['planned_transitions']} | "
            f"{fmt(r['reward']['per_1000_first_5pct'])} | {fmt(r['reward']['per_1000_last_quarter'])} | "
            f"{versus} | {fmt(r['entropy']['first_5pct'], 2)}/{fmt(r['entropy']['min'], 2)}/"
            f"{fmt(r['entropy']['last_quarter'], 2)} | "
            f"{fmt(gain.get('gap'), 4)} +- {fmt(gain.get('gap_se'), 4)} | "
            f"{fmt(signal.get('gap'), 4)} +- {fmt(signal.get('gap_se'), 4)} | "
            f"{fmt(excess.get('first_10pct'))}/{fmt(excess.get('last_quarter'))} |")
    lines += ["", "Credit gap: mean change of log pi(a_t) from one update on steps followed by a "
              "reward within the window minus steps that are not. A learner that uses the reward "
              "should show a positive gap; a rule that drifts regardless of reward shows ~0. "
              "Diagnostic only; not a study result."]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage")
    parser.add_argument("--references", action="store_true",
                        help="Run missing scripted reference policies first")
    parser.add_argument("--reference-policies", nargs="+",
                        default=["uniform", "forward_left:0.5", "forward:0.5"])
    parser.add_argument("--reference-episodes", type=int, default=32)
    parser.add_argument("--reference-parts", type=int, default=4)
    parser.add_argument("--parallel", type=int, default=4)
    options = parser.parse_args()
    stage = Path(options.stage)
    out = Path("runs/evaluation") / stage.name
    out.mkdir(parents=True, exist_ok=True)
    if options.references:
        jobs = reference_jobs(out, options.reference_policies, options.reference_episodes,
                              options.reference_parts)

        def run(job):
            target, command = job
            target.parent.mkdir(parents=True, exist_ok=True)
            with open(target.with_suffix(".log"), "w") as stream:
                return subprocess.call(command, stdout=stream, stderr=subprocess.STDOUT) == 0

        with ThreadPoolExecutor(options.parallel) as pool:
            failed = [str(job[0]) for job, ok in zip(jobs, pool.map(run, jobs)) if not ok]
        if failed:
            raise SystemExit(f"Reference policies failed: {failed}")
    references = load_references(out) if (out / "references").exists() else {}
    results = [result for directory in sorted(stage.iterdir())
               if (directory / "metrics.jsonl").exists()
               for result in [summarize(directory, references)] if result]
    (out / "diagnostics.json").write_text(json.dumps(
        {"references": references, "runs": results}, indent=2, allow_nan=False))
    (out / "diagnostics.md").write_text(f"# {stage.name} diagnostics\n\n"
                                        + markdown(results, references))
    print((out / "diagnostics.md").read_text())


if __name__ == "__main__":
    main()
