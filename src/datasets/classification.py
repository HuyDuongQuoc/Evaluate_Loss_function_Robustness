from typing import Dict

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def create_classification_dataset(
    n_samples: int = 5000,
    n_features: int = 20,
    n_informative: int = 15,
    n_redundant: int = 5,
    n_classes: int = 3,
    class_sep: float = 1.0,
    train_size: float = 0.70,
    val_size: float = 0.15,
    test_size: float = 0.15,
    seed: int = 42,
) -> Dict[str, np.ndarray]:

    if not np.isclose(train_size + val_size + test_size, 1.0):
        raise ValueError(
            "train_size + val_size + test_size must equal 1."
        )

    # 1. Generate balanced clean classification data.
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=n_redundant,
        n_classes=n_classes,
        weights=None,
        class_sep=class_sep,
        flip_y=0.0,
        random_state=seed,
    )

    X = X.astype(np.float32)
    y = y.astype(np.int64)

    # 2. Split test set.
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )

    relative_val_size = val_size / (train_size + val_size)

    # 3. Train / validation split.
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=relative_val_size,
        random_state=seed,
        stratify=y_train_val,
    )

    # 4. Fit scaler ONLY on training features.
    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    return {
        "X_train": X_train.astype(np.float32),
        "y_train": y_train.astype(np.int64),
        "X_val": X_val.astype(np.float32),
        "y_val": y_val.astype(np.int64),
        "X_test": X_test.astype(np.float32),
        "y_test": y_test.astype(np.int64),
    }