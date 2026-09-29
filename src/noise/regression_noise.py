from typing import Tuple

import numpy as np


def add_regression_noise(
    y: np.ndarray,
    noise_level: float,
    seed: int,
    noise_scale: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:

    if not 0.0 <= noise_level <= 1.0:
        raise ValueError("noise_level must be between 0 and 1.")

    rng = np.random.default_rng(seed)

    y_clean = np.asarray(y, dtype=np.float32)
    y_noisy = y_clean.copy()

    n_samples = len(y_clean)
    n_noisy = int(round(noise_level * n_samples))

    noise_mask = np.zeros(n_samples, dtype=bool)

    if n_noisy == 0:
        return y_noisy, noise_mask

    noisy_indices = rng.choice(
        n_samples,
        size=n_noisy,
        replace=False,
    )

    noise_mask[noisy_indices] = True

    target_std = np.std(y_clean)

    added_noise = rng.normal(
        loc=0.0,
        scale=noise_scale * target_std,
        size=n_noisy,
    )

    y_noisy[noisy_indices] += added_noise.astype(np.float32)

    return y_noisy, noise_mask