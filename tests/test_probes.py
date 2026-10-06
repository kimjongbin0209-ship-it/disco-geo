import numpy as np
from disco_geo.evaluation.probes import angular_error, ridge_fit, ridge_predict


def test_circular_error_handles_wraparound():
    angles = np.deg2rad([359, 1])
    predicted = np.c_[np.sin(angles), np.cos(angles)]
    truth = np.deg2rad([1, 359])
    np.testing.assert_allclose(angular_error(predicted, truth), [2, 2])


def test_linear_probe_generalizes_on_synthetic_known_latent():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(200, 8))
    y = x @ rng.normal(size=(8, 2))
    model = ridge_fit(x[:150], y[:150], alpha=1e-4)
    prediction = ridge_predict(model, x[150:])
    assert np.mean((prediction - y[150:]) ** 2) < 1e-8
