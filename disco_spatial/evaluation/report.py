"""Cross-seed reports for the primary endpoints; descriptive until the design is locked.

Each run contributes episode records. ``aggregate_seed_clusters`` reduces them
episode -> layout -> training seed. Paired differences use training seeds shared
with the reference condition. The exact sign-flip p-value is reported for
transparency only; with five seeds its minimum is 0.0625.
"""

from collections import defaultdict
from itertools import product
import json
from pathlib import Path

import numpy as np

from disco_spatial.evaluation.metrics import aggregate_seed_clusters


ENDPOINTS = {
    "seen_hidden_target_location_rmse": "lower",
    "known_wall_balanced_iou": "higher",
    "matched_revisit_efficiency_gain": "higher",
}


def parse_run(entry):
    parts = entry.split(":")
    if len(parts) not in (3, 4):
        raise ValueError("Run entries are label:seed:abilities.json[:navigation.json]")
    label, seed = parts[0], int(parts[1])
    return label, seed, Path(parts[2]), Path(parts[3]) if len(parts) == 4 else None


def episode_records(seed, abilities, navigation):
    records = defaultdict(list)
    if abilities is not None:
        families = abilities["families"]
        for row in families["spatial_memory"]["belief"]["episodes"]:
            if row["seen_hidden_rmse"] is not None:
                records["seen_hidden_target_location_rmse"].append(
                    {"seed": seed, "layout_id": row["layout_id"], "episode_id": row["episode_id"],
                     "value": row["seen_hidden_rmse"]})
        for row in families["map_topology"]["belief"]["episodes"]:
            if row["known_balanced_iou"] is not None:
                records["known_wall_balanced_iou"].append(
                    {"seed": seed, "layout_id": row["layout_id"], "episode_id": row["episode_id"],
                     "value": row["known_balanced_iou"]})
    if navigation is not None:
        gains = defaultdict(list)
        for pair in navigation["interventions"]["none"]["pairs"]:
            gains[(pair["layout_id"], pair["episode_id"])].append(pair["efficiency_gain"])
        for (layout, episode), values in gains.items():
            # Pairs inside one episode are correlated: reduce to one episode value.
            records["matched_revisit_efficiency_gain"].append(
                {"seed": seed, "layout_id": layout, "episode_id": episode,
                 "value": float(np.mean(values))})
    return records


def sign_flip_p(differences):
    """Exact two-sided paired sign-flip p-value on the mean difference."""
    differences = np.asarray(differences, float)
    if len(differences) == 0 or len(differences) > 16:
        return None
    observed = abs(differences.mean())
    flips = np.array(list(product((1, -1), repeat=len(differences))))
    statistics = np.abs((flips * differences).mean(axis=1))
    return float(np.mean(statistics >= observed - 1e-12))


def build_report(entries, reference, bootstrap_samples=2000):
    by_label = defaultdict(lambda: defaultdict(list))
    seeds = defaultdict(set)
    for entry in entries:
        label, seed, abilities_path, navigation_path = parse_run(entry)
        if seed in seeds[label]:
            raise ValueError(f"Duplicate training seed {seed} for {label}")
        seeds[label].add(seed)
        abilities = json.loads(abilities_path.read_text())
        navigation = json.loads(navigation_path.read_text()) if navigation_path else None
        for endpoint, rows in episode_records(seed, abilities, navigation).items():
            by_label[label][endpoint].extend(rows)
    if reference not in by_label:
        raise ValueError("The reference label has no runs")
    summary, lines = {}, ["| condition | endpoint | mean | per-seed | CI95 | paired diff vs reference | sign-flip p |",
                          "| --- | --- | --- | --- | --- | --- | --- |"]
    for label in sorted(by_label):
        summary[label] = {}
        for endpoint, direction in ENDPOINTS.items():
            rows = by_label[label].get(endpoint)
            if not rows:
                continue
            aggregate = aggregate_seed_clusters(rows, "value", bootstrap_samples=bootstrap_samples)
            per_seed = dict(zip(aggregate["seeds"], aggregate["per_seed"]))
            reference_rows = by_label[reference].get(endpoint) or []
            paired = None
            if label != reference and reference_rows:
                base = aggregate_seed_clusters(reference_rows, "value", bootstrap_samples=1)
                base_seed = dict(zip(base["seeds"], base["per_seed"]))
                shared = sorted(set(per_seed) & set(base_seed))
                differences = [per_seed[s] - base_seed[s] for s in shared]
                paired = {"shared_seeds": shared, "differences": differences,
                          "mean_difference": float(np.mean(differences)) if differences else None,
                          "better_direction": direction,
                          "sign_flip_p_descriptive": sign_flip_p(differences)}
            summary[label][endpoint] = {"aggregate": aggregate, "paired_vs_reference": paired}
            interval = aggregate["hierarchical_bootstrap_ci95"]
            lines.append("| {} | {} | {:.4f} | {} | {} | {} | {} |".format(
                label, endpoint, aggregate["mean"],
                ", ".join(f"{v:.3f}" for v in aggregate["per_seed"]),
                "-" if interval is None else f"[{interval[0]:.3f}, {interval[1]:.3f}]",
                "-" if not paired or paired["mean_difference"] is None
                else f"{paired['mean_difference']:+.4f} (n={len(paired['shared_seeds'])})",
                "-" if not paired or paired["sign_flip_p_descriptive"] is None
                else f"{paired['sign_flip_p_descriptive']:.4f}"))
    return {
        "schema_version": 1, "reference": reference, "conditions": summary,
        "seeds": {label: sorted(values) for label, values in seeds.items()},
        "status": "pilot_descriptive_not_confirmatory",
        "caveat": "No Holm-adjusted confirmatory claim: the design, seed bank and power plan are not locked.",
        "markdown": "\n".join(lines) + "\n",
    }
