"""Frozen-feature ridge probes with layout-disjoint validation and final tests."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ProbeSplit:
    features: np.ndarray
    targets: np.ndarray
    layout_ids: tuple
    episode_ids: tuple

    def __post_init__(self):
        features, targets = np.asarray(self.features, float), np.asarray(self.targets, float)
        if features.ndim != 2 or targets.ndim < 1 or not len(features):
            raise ValueError("Probe features must be a nonempty matrix; targets need a sample axis")
        if len(targets) != len(features):
            raise ValueError("Probe feature and target counts differ")
        if len(self.layout_ids) != len(features) or len(self.episode_ids) != len(features):
            raise ValueError("Every probe sample needs layout and episode provenance")
        if not np.isfinite(features).all() or not np.isfinite(targets).all():
            raise ValueError("Probe inputs and labels must be finite")
        if any(value in (None, "") for value in (*self.layout_ids, *self.episode_ids)):
            raise ValueError("Layout and episode IDs cannot be missing")
        object.__setattr__(self, "features", features.copy())
        object.__setattr__(self, "targets", targets.copy())
        object.__setattr__(self, "layout_ids", tuple(self.layout_ids))
        object.__setattr__(self, "episode_ids", tuple(self.episode_ids))


def _assert_disjoint(left, right, context):
    if set(left.layout_ids) & set(right.layout_ids):
        raise ValueError(f"Layout leakage between {context}")
    # Episode IDs may be local counters; their layout-qualified identity matters.
    if set(zip(left.layout_ids, left.episode_ids)) & set(zip(right.layout_ids, right.episode_ids)):
        raise ValueError(f"Episode leakage between {context}")


@dataclass(frozen=True)
class RidgeProbe:
    feature_mean: np.ndarray
    feature_std: np.ndarray
    weights: np.ndarray
    target_shape: tuple
    train_target_mean: np.ndarray
    alpha: float
    validation_mse: float
    train_layouts: frozenset
    validation_layouts: frozenset

    def predict(self, features):
        features = np.asarray(features, float)
        if features.ndim != 2 or features.shape[1] != len(self.feature_mean):
            raise ValueError("Probe feature dimensions differ from fitted model")
        if not np.isfinite(features).all():
            raise ValueError("Probe features must be finite")
        design = np.c_[(features - self.feature_mean) / self.feature_std, np.ones(len(features))]
        return (design @ self.weights).reshape((len(features), *self.target_shape))


def fit_ridge_probe(train, validation, alphas=(1e-4, 1e-2, 1.0, 100.0, 10000.0)):
    """Fit only on probe-train; choose alpha only on layout-disjoint probe-val.

    The encoder must already be frozen by the caller. This NumPy utility has no
    connection to the policy optimizer or any autodiff framework. No final-test
    split is accepted during fitting.
    """
    _assert_disjoint(train, validation, "probe train/validation")
    if train.features.shape[1:] != validation.features.shape[1:]:
        raise ValueError("Train and validation feature dimensions differ")
    if train.targets.shape[1:] != validation.targets.shape[1:]:
        raise ValueError("Train and validation target dimensions differ")
    alphas = tuple(float(a) for a in alphas)
    if not alphas or any(not np.isfinite(a) or a <= 0 for a in alphas):
        raise ValueError("Ridge alpha candidates must be finite and strictly positive")
    mean = train.features.mean(axis=0)
    std = np.maximum(train.features.std(axis=0), 1e-6)
    design = np.c_[(train.features - mean) / std, np.ones(len(train.features))]
    targets = train.targets.reshape(len(train.targets), -1)
    penalty = np.eye(design.shape[1])
    penalty[-1, -1] = 0.0
    fitted = []
    for alpha in sorted(set(alphas)):
        weights = np.linalg.solve(design.T @ design + alpha * penalty, design.T @ targets)
        model = RidgeProbe(
            mean, std, weights, train.targets.shape[1:], train.targets.mean(axis=0),
            alpha, float("inf"), frozenset(train.layout_ids), frozenset(validation.layout_ids),
        )
        error = float(np.mean((model.predict(validation.features) - validation.targets) ** 2))
        fitted.append((error, alpha, weights))
    error, alpha, weights = min(fitted, key=lambda row: (row[0], row[1]))
    return RidgeProbe(
        mean.copy(), std.copy(), weights, train.targets.shape[1:], train.targets.mean(axis=0),
        alpha, error, frozenset(train.layout_ids), frozenset(validation.layout_ids),
    )


def evaluate_ridge_probe(model, test):
    """Final-test scalar diagnostics; uncertainty needs separate cluster reduction."""
    if (model.train_layouts | model.validation_layouts) & set(test.layout_ids):
        raise ValueError("Final-test layout overlaps probe training or validation")
    if test.targets.shape[1:] != model.target_shape:
        raise ValueError("Final-test target dimensions differ")
    prediction = model.predict(test.features)
    mse = float(np.mean((prediction - test.targets) ** 2))
    baseline = float(np.mean((model.train_target_mean - test.targets) ** 2))
    return {
        "mse": mse, "rmse_per_coordinate": float(np.sqrt(mse)),
        "train_target_mean_baseline_mse": baseline,
        "improvement_over_train_mean": 1 - mse / baseline if baseline > 0 else None,
        "alpha": model.alpha, "validation_mse": model.validation_mse,
        "n_samples": len(test.features), "n_test_layouts": len(set(test.layout_ids)),
        "prediction": prediction,
        "interpretation": "Frozen-feature supervised readout; not zero-shot policy performance.",
    }
