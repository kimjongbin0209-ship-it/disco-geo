"""Numeric spatial assays, with explicit observation and sampling units.

These are evaluation utilities, not reward functions. Counts of targets/cells
describe coverage and must not be used as independent statistical sample sizes.
"""

from collections import defaultdict
from collections.abc import Mapping
import hashlib
from numbers import Integral

import numpy as np


OBSERVABILITY_GROUPS = ("visible", "seen_hidden", "never_seen")


def _boolean(value, name):
    array = np.asarray(value)
    if not np.isin(array, [0, 1]).all():
        raise ValueError(f"{name} must contain only boolean/0/1 values")
    return array.astype(bool)


def canonical_layout_fingerprint(grid):
    """SHA256 of binary maze geometry modulo all eight D4 symmetries.

    Input is a nonempty 2-D 0/1 wall/free grid. Rotations and reflections count
    as the same layout; texture, target placement, RNG seed and episode ID are
    deliberately excluded. Rectangular grids are supported. This does not
    remove translations/crop outer padding or prove other topological identity.
    Use only in evaluator inventories, never as an actor input.
    """
    array = np.asarray(grid)
    if array.ndim != 2 or not all(array.shape) or array.dtype.kind not in "biuf":
        raise ValueError("Layout must be a nonempty 2-D binary wall/free grid")
    array = _boolean(array, "layout").astype(np.uint8)
    candidates = []
    for reflection in (array, np.fliplr(array)):
        for turns in range(4):
            transformed = np.ascontiguousarray(np.rot90(reflection, turns))
            shape = f"{transformed.shape[0]}x{transformed.shape[1]}:".encode("ascii")
            candidates.append(shape + transformed.tobytes())
    return hashlib.sha256(b"maze-layout-d4-v1\0" + min(candidates)).hexdigest()


def observability_masks(visible, ever_seen):
    """Partition targets using visibility accumulated only through current time.

    The caller constructs ``ever_seen`` causally, never from future frames.
    Current visibility must be included in that history. Masks may have any
    shape, conventionally [time, target] or [sample, target].
    """
    visible = _boolean(visible, "visible")
    ever_seen = _boolean(ever_seen, "ever_seen")
    if visible.shape != ever_seen.shape:
        raise ValueError("Visibility masks have different shapes")
    if np.any(visible & ~ever_seen):
        raise ValueError("Current visibility must be included in causal ever_seen")
    return {
        "visible": visible,
        "seen_hidden": ever_seen & ~visible,
        "never_seen": ~ever_seen,
    }


def causal_observability_masks(visibility_history):
    """Construct [time, target, ...] masks using only each frame's prefix.

    Unlike a caller-supplied ever_seen mask, this helper makes the temporal
    cutoff executable. Each array must begin at the actual episode reset; never
    concatenate independent episodes before applying this function.
    """
    visible = _boolean(visibility_history, "visibility_history")
    if visible.ndim < 2 or not visible.shape[0]:
        raise ValueError("Visibility history must be nonempty [time, target, ...]")
    return observability_masks(visible, np.logical_or.accumulate(visible, axis=0))


def target_location_rmse(prediction, target, masks):
    """Euclidean location RMSE per observability group, without alignment.

    Prediction/target shape is [..., coordinate]. Masks have shape [...].
    Empty groups return None rather than a fabricated zero error. Coordinate
    conventions and units must already match; no test-time gauge fitting occurs.
    """
    prediction, target = np.asarray(prediction, float), np.asarray(target, float)
    if prediction.shape != target.shape or target.ndim < 2 or target.shape[-1] < 1:
        raise ValueError("Locations must have matching [..., coordinate] shapes")
    if set(masks) != set(OBSERVABILITY_GROUPS):
        raise ValueError(f"Expected masks for {OBSERVABILITY_GROUPS}")
    masks = {key: _boolean(value, key) for key, value in masks.items()}
    if any(mask.shape != target.shape[:-1] for mask in masks.values()):
        raise ValueError("Location mask shape does not match targets")
    if not np.all(sum(mask.astype(int) for mask in masks.values()) == 1):
        raise ValueError("Observability groups must be an exhaustive disjoint partition")
    if not np.isfinite(prediction).all() or not np.isfinite(target).all():
        raise ValueError("Locations must be finite")
    squared_distance = np.sum((prediction - target) ** 2, axis=-1)
    result = {}
    for name in OBSERVABILITY_GROUPS:
        errors = squared_distance[masks[name]]
        result[name] = {
            "rmse": float(np.sqrt(errors.mean())) if errors.size else None,
            "mean_euclidean_error": float(np.sqrt(errors).mean()) if errors.size else None,
            "n_target_observations": int(errors.size),
        }
    return result


def masked_map_iou(prediction, target, known_mask, threshold=0.5):
    """Macro IoU of wall/free classes over a causally known region only.

    ``target=True`` denotes walls; predictions are wall probabilities. Classes
    absent in both prediction and truth have undefined IoU and are omitted from
    the macro average. Never-observed cells are excluded by ``known_mask``.
    This is a proposed assay, not an asserted official Memory Maze metric.
    """
    prediction = np.asarray(prediction, float)
    target = _boolean(target, "target map")
    known = _boolean(known_mask, "known_mask")
    if prediction.shape != target.shape or known.shape != target.shape:
        raise ValueError("Map predictions, labels and known mask must have the same shape")
    if not 0 < threshold < 1:
        raise ValueError("Probability threshold must be strictly between zero and one")
    probabilities = prediction[known]
    if not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("Known-cell predictions must be finite probabilities")
    predicted, truth = probabilities >= threshold, target[known]
    scores = {}
    for label, positive in (("wall", True), ("free", False)):
        p, t = predicted == positive, truth == positive
        union = np.logical_or(p, t).sum()
        scores[f"{label}_iou"] = float(np.logical_and(p, t).sum() / union) if union else None
    valid = [value for value in scores.values() if value is not None]
    return {
        **scores,
        "balanced_iou": float(np.mean(valid)) if valid else None,
        "n_known_cells": int(known.sum()),
        "known_fraction": float(known.mean()) if known.size else 0.0,
    }


def revisit_efficiency(records):
    """Pair first-visit and revisit goal segments with explicit matching strata.

    Each mapping requires seed/layout_id/episode_id/target_id/segment_id,
    kind (first_visit or revisit), match_bin, success, oracle_distance and
    path_length. ``match_bin`` is a predeclared initial-distance/visibility
    stratum; the function never invents matches across targets or episodes.
    Efficiency is success * oracle/max(path, oracle), a proposed segment assay.
    Multiple revisits can share one first visit; they are correlated records.
    """
    groups, seen = defaultdict(list), set()
    for row in records:
        if not isinstance(row["segment_id"], Integral) or row["segment_id"] < 0:
            raise ValueError("segment_id must be a nonnegative chronological integer")
        identity = (row["seed"], row["layout_id"], row["episode_id"], row["segment_id"])
        if identity in seen:
            raise ValueError("Duplicate goal segment record")
        seen.add(identity)
        if row["kind"] not in {"first_visit", "revisit"}:
            raise ValueError("Segment kind must be first_visit or revisit")
        if not isinstance(row["success"], (bool, np.bool_)):
            raise ValueError("Segment success must be boolean")
        oracle, path = float(row["oracle_distance"]), float(row["path_length"])
        if not np.isfinite([oracle, path]).all() or oracle <= 0 or path < 0:
            raise ValueError("Segment distances must be finite with oracle>0 and path>=0")
        if row["match_bin"] in (None, ""):
            raise ValueError("Explicit predeclared match_bin is required")
        key = (row["seed"], row["layout_id"], row["episode_id"], row["target_id"],
               row["match_bin"])
        efficiency = float(row["success"]) * oracle / max(path, oracle)
        groups[key].append({**row, "efficiency": efficiency})
    pairs, unmatched = [], 0
    for key in sorted(groups, key=str):
        first = [row for row in groups[key] if row["kind"] == "first_visit"]
        repeat = [row for row in groups[key] if row["kind"] == "revisit"]
        if len(first) > 1:
            raise ValueError("Ambiguous match: multiple first visits in one matching stratum")
        if not first or not repeat:
            unmatched += len(groups[key])
            continue
        if not first[0]["success"]:
            raise ValueError("A revisit requires a prior successful first visit")
        for row in repeat:
            if row["segment_id"] <= first[0]["segment_id"]:
                raise ValueError("Revisit segments must occur after the first visit")
            pairs.append({
                **dict(zip(("seed", "layout_id", "episode_id", "target_id", "match_bin"), key)),
                "first_segment_id": first[0]["segment_id"],
                "revisit_segment_id": row["segment_id"],
                "first_efficiency": first[0]["efficiency"],
                "revisit_efficiency": row["efficiency"],
                "efficiency_gain": row["efficiency"] - first[0]["efficiency"],
            })
    return {
        "n_matched_pairs": len(pairs), "n_unmatched_segments": unmatched, "pairs": pairs,
        "mean_efficiency_gain": float(np.mean([p["efficiency_gain"] for p in pairs]))
        if pairs else None,
        "interpretation": "Proposed matched-segment diagnostic; pair count is not independent N.",
    }


def aggregate_seed_clusters(records, metric, bootstrap_samples=2000, seed=0):
    """Average episodes within layout, layouts within seed; bootstrap both levels.

    Records require seed/layout_id/episode_id and one finite scalar metric.
    Repeated frame or target rows for one episode must be aggregated beforehand.
    Training seeds, not frames or cells, determine sample SD. A single training
    seed gives no across-training uncertainty interval.
    """
    if not isinstance(bootstrap_samples, int) or bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be a positive integer")
    if not records:
        raise ValueError("No episode records to aggregate")
    by_layout, seen = defaultdict(list), set()
    for row in records:
        key = (row["seed"], row["layout_id"], row["episode_id"])
        if key in seen:
            raise ValueError("Duplicate episode: aggregate frames/targets before clustering")
        seen.add(key)
        if isinstance(row[metric], Mapping):
            raise ValueError("Metric must be a scalar episode summary")
        value = float(row[metric])
        if not np.isfinite(value):
            raise ValueError("Episode metric must be finite")
        by_layout[key[:2]].append(value)
    by_seed = defaultdict(dict)
    for (training_seed, layout), values in by_layout.items():
        by_seed[training_seed][layout] = float(np.mean(values))
    seed_ids = sorted(by_seed, key=str)
    layout_values = [np.array([values[k] for k in sorted(values, key=str)])
                     for values in (by_seed[s] for s in seed_ids)]
    seed_means = np.array([values.mean() for values in layout_values])
    mean = float(seed_means.mean())
    interval = None
    if len(seed_ids) > 1:
        # If the same evaluation layouts are shared, preserve their pairing when
        # resampling. Otherwise sample layout clusters separately within seeds.
        sets = [set(by_seed[s]) for s in seed_ids]
        paired_layouts = all(s == sets[0] for s in sets)
        rng, samples = np.random.default_rng(seed), []
        for _ in range(bootstrap_samples):
            selected_seeds = rng.integers(len(seed_ids), size=len(seed_ids))
            shared = rng.integers(len(layout_values[0]), size=len(layout_values[0]))
            samples.append(np.mean([
                layout_values[i][shared if paired_layouts else
                                 rng.integers(len(layout_values[i]), size=len(layout_values[i]))].mean()
                for i in selected_seeds
            ]))
        interval = [float(v) for v in np.quantile(samples, [0.025, 0.975])]
    return {
        "metric": metric, "mean": mean,
        "std_across_training_seeds": float(seed_means.std(ddof=1)) if len(seed_ids) > 1 else None,
        "seeds": seed_ids, "per_seed": seed_means.tolist(), "n_training_seeds": len(seed_ids),
        "n_layout_clusters_per_seed": [len(v) for v in layout_values],
        "n_episode_records": len(records), "hierarchical_bootstrap_ci95": interval,
        "bootstrap_samples": bootstrap_samples if interval is not None else 0,
        "aggregation": "episodes within layout; layouts within training seed; training seeds equally weighted",
        "caveat": "Intervals are descriptive with few seeds; no frame-level pseudo-replication.",
    }


def holm_primary_tests(p_values, alpha=0.05):
    """Adjust supplied valid p-values over exactly the fixed three H1 endpoints.

    This function does not manufacture p-values from few-seed descriptive CIs.
    The test generating these p-values must honor the seed/layout pairing.
    """
    from .protocol import PRIMARY_ENDPOINTS

    if set(p_values) != PRIMARY_ENDPOINTS or not 0 < alpha < 1:
        raise ValueError("Provide exactly the three predeclared primary p-values and 0<alpha<1")
    ordered = sorted((float(value), key) for key, value in p_values.items())
    if any(not np.isfinite(value) or not 0 <= value <= 1 for value, _ in ordered):
        raise ValueError("P-values must be finite and between zero and one")
    adjusted, previous = {}, 0.0
    for index, (value, key) in enumerate(ordered):
        previous = max(previous, min(1.0, (len(ordered) - index) * value))
        adjusted[key] = {"p_value": value, "holm_adjusted_p": previous,
                         "reject": previous <= alpha}
    return {"alpha": alpha, "family": "H1_fixed_three_spatial_endpoints",
            "tests": {key: adjusted[key] for key in sorted(adjusted)}}
