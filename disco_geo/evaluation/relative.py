"""A fixed 64-unit MLP for pairwise viewpoint, fitted only after encoder freezing."""

import jax
import jax.numpy as jnp
import numpy as np
import optax
from disco_geo.evaluation.probes import angular_error


def relative_probe(h, yaw, splits, label_budget, seed=0, epochs=100):
    rng = np.random.default_rng(seed)

    def pairs(indices, limit):
        scenes = rng.choice(indices, size=limit)
        i = rng.integers(h.shape[1], size=limit)
        j = (i + 1 + rng.integers(h.shape[1] - 1, size=limit)) % h.shape[1]
        x = np.concatenate((h[scenes, i], h[scenes, j]), -1)
        angle = yaw[scenes, j] - yaw[scenes, i]
        return x, angle, np.c_[np.sin(angle), np.cos(angle)]

    train, train_angle, target = pairs(splits["probe_train"], label_budget)
    val, val_angle, _ = pairs(splits["probe_val"], min(label_budget, 2000))
    test, test_angle, _ = pairs(splits["test"], min(label_budget, 5000))
    mean, std = train.mean(0), np.maximum(train.std(0), 1e-6)
    train, val, test = [(x - mean) / std for x in (train, val, test)]
    r = np.random.default_rng(seed)
    params = {
        "w1": jnp.array(r.normal(size=(train.shape[1], 64)) / np.sqrt(train.shape[1]), jnp.float32),
        "b1": jnp.zeros(64),
        "w2": jnp.array(r.normal(size=(64, 2)) / 8, jnp.float32),
        "b2": jnp.zeros(2),
    }
    optimizer = optax.adam(1e-3)
    state = optimizer.init(params)

    def predict(p, x):
        return jax.nn.relu(x @ p["w1"] + p["b1"]) @ p["w2"] + p["b2"]

    @jax.jit
    def step(p, state, x, y):
        loss, grads = jax.value_and_grad(lambda p: jnp.mean((predict(p, x) - y) ** 2))(p)
        updates, state = optimizer.update(grads, state)
        return optax.apply_updates(p, updates), state, loss

    best, best_params = float("inf"), params
    for _ in range(epochs):
        order = rng.permutation(len(train))
        for start in range(0, len(train), 128):
            rows = order[start : start + 128]
            params, state, _ = step(params, state, train[rows], target[rows])
        error = angular_error(np.asarray(predict(params, val)), val_angle).mean()
        if error < best:
            best, best_params = error, params
    error = angular_error(np.asarray(predict(best_params, test)), test_angle)
    return {
        "relative_probe": "64-unit MLP, validation-selected, no action history",
        "relative_mean_degrees": float(error.mean()),
        "relative_median_degrees": float(np.median(error)),
    }
