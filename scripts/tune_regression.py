import copy
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from src.datasets.regression import create_regression_dataset

from src.noise.regression_noise import add_regression_noise

from src.models.mlp import RegressionMLP

from src.losses.regression_losses import get_regression_loss

from src.training.data import create_dataloader

from src.training.trainer import fit

from src.training.evaluator import evaluate_regression

from src.experiments.io import load_yaml

from src.experiments.runner import regression_monitor

from src.utils import (
    set_seed,
    get_device,
)

TUNING_NOISE = 0.1

DELTA_FACTORS = [
    0.1,
    0.5,
    1.0,
    2.0,
]

def main():

    config = load_yaml("configs/regression.yaml" )

    training_config = config["training"]
    dataset_config = config["dataset"]
    split_config = config["split"]
    model_config = config["model"]

    device = get_device(
        training_config["device"]
    )

    results = []

    for seed in config["experiment"]["seeds"]:

        print(f"\n==============================")
        print(f"HUBER TUNING | seed={seed}")
        print(f"==============================")

        data = create_regression_dataset(
            n_samples=dataset_config[
                "n_samples"
            ],
            n_features=dataset_config[
                "n_features"
            ],
            n_informative=dataset_config[
                "n_informative"
            ],
            bias=dataset_config[
                "bias"
            ],
            train_size=split_config[
                "train_size"
            ],
            val_size=split_config[
                "val_size"
            ],
            test_size=split_config[
                "test_size"
            ],
            seed=seed,
        )

        y_train_noisy, _ = (
            add_regression_noise(
                data["y_train"],
                noise_level=TUNING_NOISE,
                seed=seed,
                noise_scale=config[
                    "noise"
                ]["scale"],
            )
        )

        target_std = float(
            np.std(data["y_train"])
        )

        set_seed(seed)

        base_model = RegressionMLP(
            input_dim=model_config[
                "input_dim"
            ],
            hidden_dims=tuple(
                model_config[
                    "hidden_dims"
                ]
            ),
        )

        initial_state = copy.deepcopy(base_model.state_dict())

        for delta_factor in DELTA_FACTORS:

            delta = delta_factor * target_std
            
            print(
                f"\nseed={seed}"
                f" | factor={delta_factor}"
                f" | delta={delta:.4f}"
            )

            train_loader = (
                create_dataloader(
                    X=data["X_train"],
                    y=y_train_noisy,
                    task="regression",
                    batch_size=training_config["batch_size"],
                    shuffle=True,
                    seed=seed,
                    num_workers=training_config["num_workers"],
                )
            )

            val_loader = (
                create_dataloader(
                    X=data["X_val"],
                    y=data["y_val"],
                    task="regression",
                    batch_size=training_config["batch_size"],
                    shuffle=False,
                    seed=seed,
                    num_workers=training_config["num_workers"],
                )
            )

            model = RegressionMLP(
                input_dim=model_config[
                    "input_dim"
                ],
                hidden_dims=tuple(
                    model_config[
                        "hidden_dims"
                    ]
                ),
            )

            model.load_state_dict(
                initial_state
            )

            criterion = (
                get_regression_loss(
                    "huber",
                    huber_delta=delta,
                )
            )

            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=training_config["learning_rate"],
                weight_decay=training_config["weight_decay"],
            )

            early_config = training_config["early_stopping"]

            history, training_info = fit(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                max_epochs=training_config["max_epochs"],
                validation_metric_fn=regression_monitor,
                monitor_name="val_mae",
                monitor_mode=early_config["mode"],
                patience=early_config["patience"],
                min_delta=early_config["min_delta"],
                restore_best_weights=early_config["restore_best_weights"],
                verbose=False,
            )

            metrics = evaluate_regression(
                model=model,
                dataloader=val_loader,
                device=device,
            )

            results.append(
                {
                    "seed": seed,
                    "noise_level": TUNING_NOISE,
                    "delta_factor": delta_factor,
                    "delta": delta,
                    "val_mae": metrics["mae"],
                    "val_rmse": metrics["rmse"],
                    "best_epoch": training_info["best_epoch"],
                    "stopped_epoch": training_info["stopped_epoch"],
                    "epochs_trained": training_info["epochs_trained"],
                }
            )

    output_dir = Path("results/tuning")

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = pd.DataFrame(results)

    df.to_csv(
        output_dir
        / "huber_tuning_raw.csv",
        index=False,
    )

    summary = (
        df
        .groupby(
            "delta_factor",
            as_index=False,
        )
        .agg(
            val_mae_mean=(
                "val_mae",
                "mean",
            ),
            val_mae_std=(
                "val_mae",
                "std",
            ),
            val_rmse_mean=(
                "val_rmse",
                "mean",
            ),
            val_rmse_std=(
                "val_rmse",
                "std",
            ),
        )
        .sort_values(
            [
                "val_mae_mean",
                "val_rmse_mean",
            ]
        )
    )

    summary.to_csv(
        output_dir
        / "huber_tuning_summary.csv",
        index=False,
    )

    print("\n==============================")
    print("HUBER TUNING SUMMARY")
    print("==============================")

    print(summary)

    best = summary.iloc[0]

    print(
        "\nSelected delta_factor:",
        best["delta_factor"],
    )


if __name__ == "__main__":
    main()