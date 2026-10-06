"""Compare final paired generations; average images within seed before seed statistics."""

import csv
import json
from pathlib import Path
import numpy as np
from PIL import Image
from disco_geo.io import write_json

METRICS = {
    "mean_psnr": "Orbit PSNR (dB)",
    "mean_foreground_psnr": "Foreground PSNR (dB)",
    "mean_silhouette_iou": "Silhouette IoU",
}


def create_transfer_report(transfer, output=None):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = Path(transfer)
    output = Path(output) if output else root / "report"
    plan = json.loads((root / "plan.json").read_text())
    output.mkdir(parents=True, exist_ok=True)
    rows, view_rows = [], []
    for item in plan["selected_test_inputs"]:
        item_root = root / f"scene{item['scene']}_view{item['view']}"
        jobs = [("native", None, "native")]
        jobs += [
            (f"{label}-{mode}", label, mode)
            for label in plan["encoders"]
            for mode in ("learned", "zero", "shuffled")
        ]
        for name, label, mode in jobs:
            path = item_root / name
            metrics = json.loads((path / "evaluation/metrics.json").read_text())
            generated = json.loads((path / "generation.json").read_text())
            identity = (
                plan["encoders"][label]
                if label
                else {"method": "TripoSplat-native", "ablation": "none", "seed": None}
            )
            if (
                metrics["evaluation_split"] != "test"
                or metrics["scene"] != item["scene"]
                or metrics["source_view_excluded"] != item["view"]
            ):
                raise ValueError(
                    "Final comparison requires the planned test scene and excluded source view"
                )
            base = {
                **identity,
                "label": label or "native",
                "mode": mode,
                "scene": item["scene"],
                "scene_id": item["scene_id"],
                "source_view": item["view"],
                "interpretation": metrics["interpretation"],
            }
            rows.append(
                {
                    **base,
                    **{key: metrics[key] for key in METRICS},
                    "generation_wall_seconds": generated.get("wall_seconds"),
                    "peak_cuda_allocated_bytes": generated.get("peak_cuda_allocated_bytes"),
                }
            )
            view_rows.extend({**base, **view} for view in metrics["views"])
    interpretations = {r["interpretation"] for r in rows}
    if len(interpretations) != 1:
        raise ValueError("Do not aggregate mixed registered and unregistered evaluations")
    status = (
        "UNREGISTERED diagnostics"
        if "diagnostic unregistered" in interpretations
        else "Registered orbit comparison"
    )
    write_json(output / "per_image.json", rows)

    def table(file, records):
        fields = list(dict.fromkeys(key for row in records for key in row))
        with (output / file).open("w", newline="") as f:
            writer = csv.DictWriter(f, fields)
            writer.writeheader()
            writer.writerows(records)

    table("per_image.csv", rows)
    table("per_view.csv", view_rows)
    seed_rows, summary = [], []
    groups = sorted({(r["method"], r["ablation"], r["mode"]) for r in rows})
    for method, ablation, mode in groups:
        selected = [
            r for r in rows if (r["method"], r["ablation"], r["mode"]) == (method, ablation, mode)
        ]
        seeds = sorted({r["seed"] for r in selected}, key=lambda v: -1 if v is None else v)
        seed_means = []
        for seed in seeds:
            images = [r for r in selected if r["seed"] == seed]
            means = {key: float(np.mean([r[key] for r in images])) for key in METRICS}
            seed_row = {
                "method": method,
                "ablation": ablation,
                "mode": mode,
                "seed": seed,
                "n_images": len(images),
                **means,
            }
            seed_rows.append(seed_row)
            seed_means.append(seed_row)
        summary.append(
            {
                "method": method,
                "ablation": ablation,
                "mode": mode,
                "n_seeds": len(seeds) if mode != "native" else 0,
                "n_images_per_seed": len(plan["selected_test_inputs"]),
                "metrics": {
                    key: {
                        "mean": float(np.mean([r[key] for r in seed_means])),
                        "std_across_seeds": float(np.std([r[key] for r in seed_means], ddof=1))
                        if len(seeds) > 1
                        else None,
                        "seed_means": [r[key] for r in seed_means],
                    }
                    for key in METRICS
                },
            }
        )
    table("per_seed.csv", seed_rows)
    write_json(
        output / "summary.json",
        {
            "interpretation": status,
            "groups": summary,
            "aggregation": "Orbit views averaged per image, images averaged per RL seed, then mean/std across RL seeds. Native is one shared baseline without an RL-seed std.",
            "test_items": plan["selected_test_inputs"],
            "renderer": "CPU approximation; no exact gsplat parity claim",
        },
    )

    def name(method, ablation):
        return method + ("" if ablation == "none" else ":" + ablation)

    names = sorted({name(s["method"], s["ablation"]) for s in summary})
    modes = ("native", "learned", "zero", "shuffled")
    fig, axes = plt.subplots(1, 3, figsize=(19, 6))
    for ax, key in zip(axes, METRICS):
        for i, mode in enumerate(modes):
            stats = [s for s in summary if s["mode"] == mode]
            x = [names.index(name(s["method"], s["ablation"])) + (i - 1.5) * 0.2 for s in stats]
            ax.bar(x, [s["metrics"][key]["mean"] for s in stats], width=0.19, label=mode)
            indices = [
                j for j, s in enumerate(stats) if s["metrics"][key]["std_across_seeds"] is not None
            ]
            if indices:
                ax.errorbar(
                    [x[j] for j in indices],
                    [stats[j]["metrics"][key]["mean"] for j in indices],
                    yerr=[stats[j]["metrics"][key]["std_across_seeds"] for j in indices],
                    fmt="none",
                    ecolor="black",
                    capsize=3,
                )
        ax.set_xticks(range(len(names)), names, rotation=35, ha="right", fontsize=8)
        ax.set_title(METRICS[key])
        ax.grid(axis="y", alpha=0.2)
    axes[0].legend(fontsize=8)
    fig.suptitle(
        status
        + "; final test, paired inputs/noise; error bars = RL seed std, not confidence intervals"
    )
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"generation_comparison.{suffix}", dpi=160)
    plt.close(fig)

    # One fixed held-out view per object, compared across all methods and controls.
    dataset = plan.get("diagnostic_dataset")
    if dataset:
        from disco_geo.data.bundle import RGBBundle

        bundle = RGBBundle(dataset, "test")
        seeds = sorted({meta["seed"] for meta in plan["encoders"].values()})
        for item in plan["selected_test_inputs"]:
            item_root = root / f"scene{item['scene']}_view{item['view']}"
            native = json.loads((item_root / "native/evaluation/metrics.json").read_text())
            view = native["views"][0]["view"]
            for seed in seeds:
                labels = [label for label, meta in plan["encoders"].items() if meta["seed"] == seed]
                fig, axes = plt.subplots(
                    3, len(labels) + 2, figsize=(2 * (len(labels) + 2), 6), squeeze=False
                )
                for row, mode in enumerate(("learned", "zero", "shuffled")):
                    axes[row, 0].imshow(bundle.images[item["scene"], view])
                    axes[row, 0].set_title("Ground truth", fontsize=8)
                    axes[row, 1].imshow(
                        Image.open(item_root / f"native/evaluation/view_{view:03d}.png")
                    )
                    axes[row, 1].set_title("Native RGB", fontsize=8)
                    for col, label in enumerate(labels, 2):
                        axes[row, col].imshow(
                            Image.open(item_root / f"{label}-{mode}/evaluation/view_{view:03d}.png")
                        )
                        meta = plan["encoders"][label]
                        axes[row, col].set_title(
                            name(meta["method"], meta["ablation"]) + "/" + mode, fontsize=7
                        )
                for ax in axes.flat:
                    ax.axis("off")
                fig.suptitle(
                    f"{status}: scene {item['scene']}, held-out view {view}, source {item['view']}, RL seed {seed}"
                )
                fig.tight_layout()
                fig.savefig(output / f"qualitative_scene{item['scene']}_seed{seed}.png", dpi=160)
                plt.close(fig)
    return output
