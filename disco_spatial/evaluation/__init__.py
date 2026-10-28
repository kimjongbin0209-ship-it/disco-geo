"""Post-training evaluation; simulator ground truth never enters the learner."""

from .metrics import (
    aggregate_seed_clusters,
    causal_observability_masks,
    canonical_layout_fingerprint,
    holm_primary_tests,
    masked_map_iou,
    observability_masks,
    revisit_efficiency,
    target_location_rmse,
)
from .probes import ProbeSplit, evaluate_ridge_probe, fit_ridge_probe
from .protocol import build_manifest, default_protocol, validate_protocol

__all__ = [
    "ProbeSplit",
    "aggregate_seed_clusters",
    "build_manifest",
    "causal_observability_masks",
    "canonical_layout_fingerprint",
    "default_protocol",
    "evaluate_ridge_probe",
    "fit_ridge_probe",
    "holm_primary_tests",
    "masked_map_iou",
    "observability_masks",
    "revisit_efficiency",
    "target_location_rmse",
    "validate_protocol",
]
