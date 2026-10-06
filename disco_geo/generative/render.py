"""Small CPU anisotropic Gaussian renderer for diagnostics, not exact gsplat parity."""

import numpy as np


def quaternion_matrix(q):
    q = q / np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1e-8)
    w, x, y, z = q.T
    return np.stack(
        [
            1 - 2 * (y * y + z * z),
            2 * (x * y - w * z),
            2 * (x * z + w * y),
            2 * (x * y + w * z),
            1 - 2 * (x * x + z * z),
            2 * (y * z - w * x),
            2 * (x * z - w * y),
            2 * (y * z + w * x),
            1 - 2 * (x * x + y * y),
        ],
        -1,
    ).reshape(-1, 3, 3)


def render(gaussians, yaw, size=64, elevation=0.25, transform=None, background=235 / 255):
    xyz, scales = gaussians["xyz"], gaussians["scale"]
    rot = quaternion_matrix(gaussians["rotation"])
    cov = (rot * scales[:, None, :] ** 2) @ rot.transpose(0, 2, 1)
    if transform is not None:
        transform = np.asarray(transform)
        if transform.shape != (4, 4) or not np.isfinite(transform).all():
            raise ValueError("Alignment must be a finite 4x4 similarity matrix")
        linear = transform[:3, :3]
        gram = linear.T @ linear
        scale2 = np.trace(gram) / 3
        if (
            scale2 <= 0
            or np.linalg.det(linear) <= 0
            or not np.allclose(gram, np.eye(3) * scale2, atol=1e-6, rtol=1e-5)
            or not np.allclose(transform[3], [0, 0, 0, 1])
        ):
            raise ValueError("Alignment must contain uniform positive scale and a proper rotation")
        xyz = xyz @ transform[:3, :3].T + transform[:3, 3]
        cov = transform[:3, :3] @ cov @ transform[:3, :3].T
    camera = np.array([2 * np.cos(yaw), 2 * elevation, 2 * np.sin(yaw)])
    forward = -camera / np.linalg.norm(camera)
    right = np.cross(forward, [0, 1, 0])
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    basis = np.stack([right, -up, forward])
    position = (xyz - camera) @ basis.T
    cov = basis @ cov @ basis.T
    focal = size / (2 * np.tan(np.deg2rad(35) / 2))
    z = np.maximum(position[:, 2], 1e-6)
    uv = position[:, :2] / z[:, None] * focal + size / 2 - 0.5
    jac = np.zeros((len(xyz), 2, 3))
    jac[:, 0, 0] = jac[:, 1, 1] = focal / z
    jac[:, 0, 2], jac[:, 1, 2] = -focal * position[:, 0] / z**2, -focal * position[:, 1] / z**2
    screen_cov = jac @ cov @ jac.transpose(0, 2, 1) + np.eye(2) * 0.3
    inv = np.linalg.inv(screen_cov)
    radius = np.ceil(3 * np.sqrt(np.maximum(screen_cov[:, [0, 1], [0, 1]], 1e-8))).astype(int)
    image = np.zeros((size, size, 3))
    transmittance = np.ones((size, size))
    opacity = gaussians["opacity"].reshape(-1)
    for i in np.argsort(position[:, 2]):
        if position[i, 2] <= 0 or opacity[i] < 1 / 255:
            continue
        x0, y0 = np.maximum(np.floor(uv[i] - radius[i]).astype(int), 0)
        x1, y1 = np.minimum(np.ceil(uv[i] + radius[i]).astype(int) + 1, size)
        if x0 >= x1 or y0 >= y1:
            continue
        xx, yy = np.meshgrid(np.arange(x0, x1), np.arange(y0, y1))
        delta = np.stack((xx - uv[i, 0], yy - uv[i, 1]), -1)
        alpha = np.minimum(
            0.99, opacity[i] * np.exp(-0.5 * np.einsum("...i,ij,...j->...", delta, inv[i], delta))
        )
        region = np.s_[y0:y1, x0:x1]
        image[region] += (transmittance[region] * alpha)[..., None] * gaussians["color"][i]
        transmittance[region] *= 1 - alpha
    image += transmittance[..., None] * background
    return np.clip(image, 0, 1), 1 - transmittance
