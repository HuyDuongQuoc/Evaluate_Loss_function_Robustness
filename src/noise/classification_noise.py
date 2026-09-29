from typing import Tuple

import numpy as np


def add_symmetric_label_noise(
    y: np.ndarray,
    noise_level: float,
    n_classes: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:

    if not 0.0 <= noise_level <= 1.0:
        raise ValueError("noise_level must be between 0 and 1.")

    if n_classes < 2:
        raise ValueError("n_classes must be at least 2.")

    rng = np.random.default_rng(seed)

    y_clean = np.asarray(y, dtype=np.int64)
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

    all_classes = np.arange(n_classes)

    for idx in noisy_indices:

        original_class = y_clean[idx]

        possible_classes = all_classes[
            all_classes != original_class
        ]

        y_noisy[idx] = rng.choice(possible_classes)

    return y_noisy, noise_mask