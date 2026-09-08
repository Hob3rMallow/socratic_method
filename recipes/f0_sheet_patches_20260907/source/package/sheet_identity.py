"""Local normalized CT-profile samples; identical to development helper."""

import numpy as np

from . import sheet_chart as chart


def normalized_profiles(ct, coordinates, normal):
    sampled = []
    for offset in (-6, -4, -2, 0, 2, 4, 6):
        points = coordinates.copy()
        points[..., 1:] += offset * normal[None, :, None, :]
        sampled.append(chart.sample(ct, points))
    values = np.stack(sampled, axis=-1).astype(float)
    values -= values.mean(axis=-1, keepdims=True)
    return values / np.maximum(5.0, np.linalg.norm(values, axis=-1, keepdims=True))
