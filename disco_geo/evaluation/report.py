"""Rebuild numerical tables and exportable plots from saved experiment artifacts."""

import csv
import json
from pathlib import Path
import numpy as np
from disco_geo.io import write_json

METRICS = {
    "success_rate": "Task success rate",
    "mean_return": "Episode return",
    "mean_moves": "Camera moves",
    "pose_mean_degrees": "Absolute yaw error (degrees)",
    "relative_mean_degrees": "Relative yaw error (degrees)",
    "depth_abs_rel": "Depth absolute relative error",
    "depth_rmse": "Depth RMSE",
    "depth_delta_125": "Depth delta < 1.25",
}
TRAIN_METRICS = {
    **METRICS,
    "loss": "Learner loss",
    "gradient_norm": "Gradient norm",
    "wall_seconds": "Elapsed seconds",
}


def read_jsonl(path):
    lines = Path(path).read_text().splitlines()
    result = []
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            result.append(json.loads(line))
        except json.JSONDecodeError:
            if i != len(lines) - 1:
                raise
            # A running/crashed writer may leave only its final line incomplete.
    return result


def create_report(study, output=None):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = Path(study)
    output = Path(output) if output else root / "report"
    output.mkdir(parents=True, exist_ok=True)
    records, curves, snapshots, generation = [], [], [], []
    for config_path in sorted(root.glob("*/seed*/config.json")):
        run = config_path.parent
        cfg = json.loads(config_path.read_text())
        identity = {k: cfg[k] for k in ("method", "ablation", "seed")}
        values = {}
        for name in ("policy.json", "probes.json"):
            if (run / name).exists():
                data = json.loads((run / name).read_text())
                values.update({k: data[k] for k in METRICS if k in data})
        if values:
            records.append({**identity, "split": "test", **values})
        if (run / "random_policy.json").exists():
            data = json.loads((run / "random_policy.json").read_text())
            records.append(
                {
                    "method": "random",
                    "ablation": "none",
                    "seed": cfg["seed"],
                    "split": "test",
                    **{k: data[k] for k in METRICS if k in data},
                }
            )
        if (run / "metrics.jsonl").exists():
            curves.extend(
                {**identity, "split": "rl_val", **row} for row in read_jsonl(run / "metrics.jsonl")
            )
        for file in sorted((run / "snapshots").glob("iter*/probes.json")):
            point = json.loads((file.parent / "iteration.json").read_text())
            probe = json.loads(file.read_text())
            snapshots.append(
                {
                    **identity,
                    "split": probe["evaluation_split"],
                    "updates": point["updates"],
                    "transitions": point["transitions"],
                    **{k: probe[k] for k in METRICS if k in probe},
                }
            )
        for file in sorted(
            (run / "snapshots").glob("iter*/generation/*/*/evaluation/metrics.json")
        ):
            data = json.loads(file.read_text())
            gen_path = file.parent.parent
            point = json.loads((gen_path.parents[2] / "iteration.json").read_text())
            generation.append(
                {
                    **identity,
                    "updates": point["updates"],
                    "transitions": point["transitions"],
                    "mode": gen_path.name,
                    "scene": data["scene"],
                    "split": data["evaluation_split"],
                    "interpretation": data["interpretation"],
                    **{
                        k: data[k]
                        for k in ("mean_psnr", "mean_foreground_psnr", "mean_silhouette_iou")
                    },
                }
            )
    if not records and not curves:
        raise ValueError("No study artifacts found; expected METHOD/seedN/config.json")

    def label(row):
        return row["method"] + ("" if row["ablation"] == "none" else ":" + row["ablation"])

    summary = {}
    for name in sorted({label(row) for row in records}):
        group = [row for row in records if label(row) == name]
        summary[name] = {"seeds": [r["seed"] for r in group], "metrics": {}}
        for key in METRICS:
            values = [r[key] for r in group if key in r and r[key] is not None]
            if values:
                summary[name]["metrics"][key] = {
                    "mean": float(np.mean(values)),
                    "n": len(values),
                    "per_seed": values,
                    "std_across_seeds": float(np.std(values, ddof=1)) if len(values) > 1 else None,
                }
    write_json(output / "summary.json", summary)

    def table(path, rows, metrics=METRICS):
        fields = [
            "method",
            "ablation",
            "seed",
            "split",
            "updates",
            "transitions",
            "mode",
            "scene",
            "interpretation",
            "metric",
            "value",
        ]
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fields, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                for key in metrics:
                    if key in row and row[key] is not None:
                        writer.writerow({**row, "metric": key, "value": row[key]})

    table(output / "benchmarks.csv", records)
    table(output / "learning_curves.csv", curves, TRAIN_METRICS)
    table(output / "snapshot_probes.csv", snapshots)
    write_json(output / "generation_metrics.json", generation)
    table(
        output / "generation_metrics.csv",
        generation,
        ("mean_psnr", "mean_foreground_psnr", "mean_silhouette_iou"),
    )

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(output / f"{name}.png", dpi=160)
        fig.savefig(output / f"{name}.pdf")
        plt.close(fig)

    available = [key for key in METRICS if any(key in row for row in records)]
    if available:
        fig, axes = plt.subplots(
            (len(available) + 2) // 3,
            3,
            figsize=(14, 4 * ((len(available) + 2) // 3)),
            squeeze=False,
        )
        for ax, key in zip(axes.flat, available):
            names = [n for n in summary if key in summary[n]["metrics"]]
            stats = [summary[n]["metrics"][key] for n in names]
            ax.bar(range(len(names)), [s["mean"] for s in stats])
            repeated = [i for i, s in enumerate(stats) if s["std_across_seeds"] is not None]
            if repeated:
                ax.errorbar(
                    repeated,
                    [stats[i]["mean"] for i in repeated],
                    yerr=[stats[i]["std_across_seeds"] for i in repeated],
                    fmt="none",
                    ecolor="black",
                    capsize=4,
                )
            ax.set_xticks(range(len(names)), names, rotation=35, ha="right", fontsize=8)
            ax.set_title(METRICS[key])
            ax.grid(axis="y", alpha=0.2)
        for ax in list(axes.flat)[len(available) :]:
            ax.set_visible(False)
        fig.suptitle(
            "Final test means; error bars = seed std when n >= 2 (not confidence intervals)"
        )
        save(fig, "benchmark_comparison")

    def plot_curves(rows, keys, name, title):
        if not rows:
            return
        fig, axes = plt.subplots(1, len(keys), figsize=(5 * len(keys), 4), squeeze=False)
        groups = sorted({(label(r), r["seed"]) for r in rows})
        for ax, key in zip(axes.flat, keys):
            for group_name, seed in groups:
                points = sorted(
                    [r for r in rows if label(r) == group_name and r["seed"] == seed and key in r],
                    key=lambda r: r["transitions"],
                )
                if points:
                    ax.plot(
                        [r["transitions"] for r in points],
                        [r[key] for r in points],
                        marker=".",
                        label=f"{group_name}/s{seed}",
                    )
            ax.set_xlabel("Valid environment transitions")
            ax.set_title(METRICS.get(key, key))
            ax.grid(alpha=0.2)
        handles, labels = axes.flat[0].get_legend_handles_labels()
        if handles:
            axes.flat[0].legend(handles, labels, fontsize=7)
        fig.suptitle(title)
        save(fig, name)

    plot_curves(
        curves,
        ["success_rate", "mean_return", "mean_moves"],
        "learning_curves",
        "RL validation; individual seeds",
    )
    plot_curves(
        curves,
        ["loss", "gradient_norm"],
        "optimization_curves",
        "Optimization diagnostics; loss definitions differ across methods",
    )
    plot_curves(
        snapshots,
        ["pose_mean_degrees", "relative_mean_degrees", "depth_abs_rel"],
        "snapshot_probes",
        "Iteration monitoring on probe validation identities",
    )
    if generation:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        for ax, key in zip(axes, ("mean_psnr", "mean_foreground_psnr", "mean_silhouette_iou")):
            groups = sorted({(label(r), r["seed"], r["mode"]) for r in generation})
            for method, seed, mode in groups:
                rows = [
                    r
                    for r in generation
                    if (label(r), r["seed"], r["mode"]) == (method, seed, mode)
                ]
                steps = sorted({r["transitions"] for r in rows})
                ax.plot(
                    steps,
                    [np.mean([r[key] for r in rows if r["transitions"] == t]) for t in steps],
                    marker=".",
                    label=f"{method}/s{seed}/{mode}",
                )
            ax.set_title(key)
            ax.set_xlabel("Valid environment transitions")
        axes[0].legend(fontsize=6)
        status = (
            "UNREGISTERED diagnostics"
            if any(r["interpretation"] == "diagnostic unregistered" for r in generation)
            else "Registered orbit metrics"
        )
        fig.suptitle(status + "; per-iteration image means, no 3D emergence claim")
        save(fig, "generation_curves")
    # Save one compact truth/prediction panel per final run for interpretation.
    for file in sorted(root.glob("*/seed*/probes.predictions.npz")):
        with np.load(file, allow_pickle=False) as data:
            if "depth_prediction" not in data:
                continue
            truth, pred = data["depth_target"][:4], data["depth_prediction"][:4]
        vmax = max(float(truth.max()), 1e-6)
        fig, axes = plt.subplots(2, len(truth), figsize=(3 * len(truth), 5), squeeze=False)
        for j in range(len(truth)):
            mask = truth[j] > 0
            for i, array in enumerate((truth[j], pred[j])):
                axes[i, j].imshow(
                    np.ma.masked_where(~mask, array), vmin=0, vmax=vmax, cmap="viridis"
                )
                axes[i, j].set_title("Ground truth" if i == 0 else "Frozen linear probe")
                axes[i, j].axis("off")
        save(fig, f"depth_{file.parent.parent.name}_{file.parent.name}")
    return output
