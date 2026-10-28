"""Export small, deterministic result tables without loading model dependencies.

The saved reports are checked against their per-seed/per-image/per-view tables.
This verifies artifact consistency; it does not rerun training or validate the
scientific interpretation of the original experiment.
"""

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import statistics


RL_METRICS = {
    "success_rate": "higher",
    "mean_return": "higher",
    "mean_moves": "descriptive",
    "pose_mean_degrees": "lower",
    "relative_mean_degrees": "lower",
    "depth_abs_rel": "lower",
    "depth_rmse": "lower",
    "depth_delta_125": "higher",
}
TRANSFER_METRICS = {
    "mean_psnr": "higher",
    "mean_foreground_psnr": "higher",
    "mean_silhouette_iou": "higher",
}
CONDITIONS = {
    ("untrained", "none"),
    ("actor_critic", "none"),
    ("actor_critic_aux", "none"),
    ("disco", "none"),
    ("disco", "no_yz_loss"),
    ("disco", "no_aux_policy"),
}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def condition(label):
    method, separator, ablation = label.partition(":")
    return method, ablation if separator else "none"


def seed_number(value):
    return None if value in (None, "") else int(value)


def close(actual, expected, context, checks):
    if not math.isfinite(float(actual)) or not math.isfinite(float(expected)):
        raise ValueError(f"Nonfinite number: {context}")
    if not math.isclose(float(actual), float(expected), rel_tol=1e-9, abs_tol=1e-12):
        raise ValueError(f"Inconsistent {context}: {actual!r} != {expected!r}")
    checks[context.split(":", 1)[0]] += 1


def aggregate(values, saved, context, checks):
    if not values:
        raise ValueError(f"No values: {context}")
    close(saved["mean"], statistics.mean(values), f"aggregate_mean:{context}", checks)
    expected_sd = statistics.stdev(values) if len(values) > 1 else None
    if expected_sd is None:
        if saved["std_across_seeds"] is not None:
            raise ValueError(f"Single baseline must not have an RL-seed SD: {context}")
    else:
        close(saved["std_across_seeds"], expected_sd, f"aggregate_sd:{context}", checks)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def write_csv(path, fields, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def summarize(run, output):
    sources = {}
    checks = Counter()

    def source(relative):
        path = run / relative
        if not path.is_file():
            raise ValueError(f"Required source is missing: {path}")
        sources[relative] = {"sha256": sha256(path), "size_bytes": path.stat().st_size}
        return path

    def read_json(relative):
        return json.loads(source(relative).read_text())

    def read_csv(relative):
        with source(relative).open(newline="", encoding="utf-8") as stream:
            return list(csv.DictReader(stream))

    study = read_json("study/report/summary.json")
    transfer = read_json("transfer/report/summary.json")
    pipeline = read_json("pipeline.json")
    completion = read_json("completed.json")
    diagnostics = read_json("audit/diagnostics.json")
    benchmark = read_csv("study/report/benchmarks.csv")
    seed_table = read_csv("transfer/report/per_seed.csv")
    images = read_csv("transfer/report/per_image.csv")
    views = read_csv("transfer/report/per_view.csv")

    cache_manifest = Path(pipeline["cache"]) / "manifest.json"
    if not cache_manifest.is_file():
        # Runs copied from a GPU host retain that host's absolute paths.
        cache_manifest = Path(__file__).resolve().parents[1] / "datasets" / Path(pipeline["cache"]).name / "manifest.json"
    cache_coverage = {
        "requested_max_items": pipeline["cache_items"],
        "final_selected_test_items": len(transfer["test_items"]),
        "manifest_available": cache_manifest.is_file(),
    }
    if cache_manifest.is_file():
        cache_relative = Path(os.path.relpath(cache_manifest, run.resolve())).as_posix()
        cache = read_json(cache_relative)
        split_counts = dict(sorted(Counter(i["split"] for i in cache["items"]).items()))
        cache_coverage.update({
            "actual_manifest_items": len(cache["items"]), "items_by_split": split_counts,
            "teacher": cache["teacher"],
            "dataset_manifest_sha256": cache["dataset_manifest_sha256"],
            "teacher_weight_sha256": cache["teacher_weight_sha256"],
        })
        if diagnostics["adapter_train_images"] != split_counts.get("probe_train"):
            raise ValueError("Audit adapter training count differs from cache manifest")
        if diagnostics["adapter_validation_images"] != split_counts.get("probe_val"):
            raise ValueError("Audit adapter validation count differs from cache manifest")
        cache_tests = {(i["scene"], i["view"], i["scene_id"])
                       for i in cache["items"] if i["split"] == "test"}
        if any((i["scene"], i["view"], i["scene_id"]) not in cache_tests
               for i in transfer["test_items"]):
            raise ValueError("Selected transfer tests are absent from cache test split")

    seeds = pipeline["seeds"]
    if len(seeds) != len(set(seeds)) or not seeds:
        raise ValueError("Pipeline seeds must be nonempty and unique")
    found_conditions = {condition(label) for label in study if label != "random"}
    if found_conditions != CONDITIONS or "random" not in study:
        raise ValueError("Expected six encoder conditions plus the random policy control")
    if len(transfer["groups"]) != 1 + 3 * len(CONDITIONS):
        raise ValueError("Expected native plus learned/zero/shuffled for six encoders")
    if len(transfer["test_items"]) != pipeline["test_items"]:
        raise ValueError("Transfer test image count differs from pipeline configuration")

    benchmark_values = {}
    for row in benchmark:
        key = (row["method"], row["ablation"], int(row["seed"]), row["metric"])
        if key in benchmark_values or row["split"] != "test":
            raise ValueError(f"Duplicate or non-test benchmark row: {key}")
        benchmark_values[key] = float(row["value"])

    rl_rows = []
    for label, group in sorted(study.items()):
        method, ablation = condition(label)
        if group["seeds"] != seeds:
            raise ValueError(f"RL seed order differs from pipeline: {label}")
        expected_metrics = set(RL_METRICS) if method != "random" else set(list(RL_METRICS)[:3])
        if set(group["metrics"]) != expected_metrics:
            raise ValueError(f"Incomplete RL metrics: {label}")
        for metric, saved in sorted(group["metrics"].items()):
            values = saved["per_seed"]
            if saved["n"] != len(seeds) or len(values) != len(seeds):
                raise ValueError(f"RL sample count differs from seeds: {label}/{metric}")
            aggregate(values, saved, f"rl/{label}/{metric}", checks)
            for seed, value in zip(seeds, values):
                key = (method, ablation, seed, metric)
                close(value, benchmark_values.pop(key), f"rl_per_seed:{key}", checks)
            rl_rows.append({
                "method": method, "ablation": ablation, "metric": metric,
                "direction": RL_METRICS[metric], "n_seeds": saved["n"],
                "mean": saved["mean"], "std_across_seeds": saved["std_across_seeds"],
                **{f"seed_{seed}": value for seed, value in zip(seeds, values)},
            })
    if benchmark_values:
        raise ValueError("Benchmark CSV has rows absent from the summary")

    def transfer_key(row):
        return row["method"], row["ablation"], row["mode"], seed_number(row["seed"])

    expected_groups = {("TripoSplat-native", "none", "native")}
    expected_groups.update((method, ablation, mode) for method, ablation in CONDITIONS
                           for mode in ("learned", "zero", "shuffled"))
    actual_groups = {(g["method"], g["ablation"], g["mode"]) for g in transfer["groups"]}
    if actual_groups != expected_groups:
        raise ValueError("Transfer conditions differ from the six-encoder design")
    image_groups = defaultdict(list)
    view_groups = defaultdict(list)
    expected_inputs = {(str(i["scene"]), str(i["view"])) for i in transfer["test_items"]}
    for row in views:
        key = (*transfer_key(row), row["scene"], row["source_view"])
        view_groups[key].append(row)
    for row in images:
        key = transfer_key(row)
        if (row["scene"], row["source_view"]) not in expected_inputs:
            raise ValueError("Unexpected transfer test input")
        view_key = (*key, row["scene"], row["source_view"])
        orbit = view_groups.pop(view_key)
        if len(orbit) != pipeline["prepare"]["views"] - 1:
            raise ValueError(f"Wrong held-out orbit view count: {view_key}")
        orbit_ids = [int(v["view"]) for v in orbit]
        expected_views = set(range(pipeline["prepare"]["views"])) - {int(row["source_view"])}
        if set(orbit_ids) != expected_views or len(set(orbit_ids)) != len(orbit_ids):
            raise ValueError(f"Incorrect held-out view IDs: {view_key}")
        for metric in TRANSFER_METRICS:
            close(float(row[metric]), statistics.mean(float(v[metric[5:]]) for v in orbit),
                  f"transfer_per_image:{view_key}/{metric}", checks)
        image_groups[key].append(row)
    if view_groups:
        raise ValueError("Transfer view rows lack matching image records")

    saved_seed_values = {}
    for row in seed_table:
        key = transfer_key(row)
        if key in saved_seed_values:
            raise ValueError(f"Duplicate transfer seed row: {key}")
        group_images = image_groups.pop(key)
        actual_inputs = [(r["scene"], r["source_view"]) for r in group_images]
        if len(actual_inputs) != len(expected_inputs) or set(actual_inputs) != expected_inputs:
            raise ValueError(f"Transfer images incomplete or duplicated: {key}")
        if int(row["n_images"]) != len(group_images):
            raise ValueError(f"Transfer per-seed image count mismatch: {key}")
        for metric in TRANSFER_METRICS:
            close(float(row[metric]), statistics.mean(float(r[metric]) for r in group_images),
                  f"transfer_per_seed:{key}/{metric}", checks)
        saved_seed_values[key] = row
    if image_groups:
        raise ValueError("Transfer image records lack matching seed rows")
    expected_seed_keys = {
        (*key, seed)
        for key in expected_groups
        for seed in ([None] if key[2] == "native" else seeds)
    }
    if set(saved_seed_values) != expected_seed_keys:
        raise ValueError("Transfer tables have missing or unexpected condition/seed rows")

    transfer_rows = []
    for group in sorted(transfer["groups"], key=lambda g: (g["method"], g["ablation"], g["mode"])):
        identity = {k: group[k] for k in ("method", "ablation", "mode")}
        group_seeds = [None] if group["mode"] == "native" else seeds
        expected_n_seeds = 0 if group["mode"] == "native" else len(seeds)
        if group["n_seeds"] != expected_n_seeds:
            raise ValueError(f"Incorrect transfer seed count: {identity}")
        if group["n_images_per_seed"] != len(expected_inputs):
            raise ValueError(f"Incorrect transfer image count: {identity}")
        if set(group["metrics"]) != set(TRANSFER_METRICS):
            raise ValueError(f"Incomplete transfer metrics: {identity}")
        for metric, saved in sorted(group["metrics"].items()):
            values = saved["seed_means"]
            if len(values) != len(group_seeds):
                raise ValueError(f"Missing transfer seed means: {identity}/{metric}")
            aggregate(values, saved, f"transfer/{identity}/{metric}", checks)
            for seed, value in zip(group_seeds, values):
                key = (*identity.values(), seed)
                close(value, float(saved_seed_values[key][metric]),
                      f"transfer_summary_seed:{key}/{metric}", checks)
            transfer_rows.append({
                **identity, "metric": metric, "direction": TRANSFER_METRICS[metric],
                "n_seeds": group["n_seeds"], "n_images_per_seed": group["n_images_per_seed"],
                "mean": saved["mean"], "std_across_seeds": saved["std_across_seeds"],
                **{("shared_native" if seed is None else f"seed_{seed}"): value
                   for seed, value in zip(group_seeds, values)},
            })

    validation = {
        "status": "passed", "numeric_comparisons": dict(sorted(checks.items())),
        "numeric_comparisons_total": sum(checks.values()),
        "tolerance": {"relative": 1e-9, "absolute": 1e-12},
        "encoder_conditions": len(CONDITIONS), "rl_seed_count": len(seeds),
        "rl_metric_rows": len(rl_rows), "transfer_groups": len(transfer["groups"]),
        "transfer_metric_rows": len(transfer_rows), "transfer_image_rows": len(images),
        "transfer_view_rows": len(views),
        "scope": "Report arithmetic, source table agreement, seed/condition/input/view coverage only; no model inference or scientific-validity test.",
    }
    metrics = {
        "schema_version": 1, "pipeline": pipeline, "completion": completion,
        "rl": study, "transfer": transfer, "audit": diagnostics,
        "conditioning_cache": cache_coverage,
        "validation": validation,
        "interpretation_notes": [
            "RL values are final-test results from RL-validation-return-selected encoders; six encoder conditions plus a random-policy control.",
            "Sample SD is computed across RL seeds (ddof=1), not a confidence interval. Random-policy rows repeat one fixed evaluation RNG; their zero SD is not independent-seed evidence.",
            "Transfer averages orbit views within image, then images within RL seed. Native conditioning is shared, so n_seeds=0 and seed SD is null.",
            "Zero-conditioning generations are repeated controls with identical paired settings, not independent evidence for six encoders or three seeds.",
            "Transfer metrics are unregistered orbit diagnostics from an approximate CPU renderer; they do not establish geometric reconstruction quality.",
            "Audit numbers are copied from post-hoc diagnostics. Adapter diagnostics cover only three methods at seed 0 on validation data; teacher/depth constants use training labels or targets, not final-test fitting.",
        ],
    }
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "metrics.json", metrics)
    seed_fields = [f"seed_{seed}" for seed in seeds]
    write_csv(output / "rl_metrics.csv", [
        "method", "ablation", "metric", "direction", "n_seeds", "mean",
        "std_across_seeds", *seed_fields,
    ], rl_rows)
    write_csv(output / "transfer_metrics.csv", [
        "method", "ablation", "mode", "metric", "direction", "n_seeds", "n_images_per_seed",
        "mean", "std_across_seeds", *seed_fields, "shared_native",
    ], transfer_rows)
    provenance = {
        "schema_version": 1,
        "source_root": str(run),
        "sources": dict(sorted(sources.items())),
        "generator": {"path": "scripts/summarize_results.py", "sha256": sha256(Path(__file__))},
        "command": f"python3 scripts/summarize_results.py --run {run} --output {output}",
        "validation": validation,
        "outputs": {name: {"sha256": sha256(output / name), "size_bytes": (output / name).stat().st_size}
                    for name in ("metrics.json", "rl_metrics.csv", "transfer_metrics.csv")},
        "determinism": "Sorted records/JSON keys, stable float serialization and LF line endings; no timestamps or runtime-specific model dependencies.",
    }
    write_json(output / "provenance.json", provenance)
    print(f"Verified {validation['numeric_comparisons_total']} numeric comparisons; exported "
          f"{len(rl_rows)} RL metric rows and {len(transfer_rows)} transfer metric rows "
          f"({len(transfer['groups'])} groups) to {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("runs/full-experiment"))
    parser.add_argument("--output", type=Path, default=Path("results/full-experiment"))
    args = parser.parse_args()
    summarize(args.run, args.output)


if __name__ == "__main__":
    main()
