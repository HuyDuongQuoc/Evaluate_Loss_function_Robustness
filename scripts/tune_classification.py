import copy
from itertools import product
from pathlib import Path
import pandas as pd
import torch

from src.datasets.classification import create_classification_dataset

from src.noise.classification_noise import add_symmetric_label_noise

from src.models.mlp import ClassificationMLP

from src.losses.classification_losses import get_classification_loss

from src.training.data import create_dataloader

from src.training.trainer import fit

from src.training.evaluator import evaluate_classification

from src.experiments.io import load_yaml

from src.experiments.runner import classification_monitor

from src.utils import (
    set_seed,
    get_device,
)

TUNING_NOISE = 0.2

GCE_Q_VALUES = [
    0.3,
    0.5,
    0.7,
    0.9,
]

SCE_ALPHA_VALUES = [
    0.01,
    0.1,
    0.5,
    1.0,
]

SCE_BETA_VALUES = [
    0.1,
    1.0,
]

def train_candidate(
    data,
    y_train_noisy,
    initial_state,
    config,
    seed,
    criterion,
    device,
):

    training_config = config["training"]

    model_config = config["model"]

    dataset_config = config["dataset"]

    train_loader = create_dataloader(
        X=data["X_train"],
        y=y_train_noisy,
        task="classification",
        batch_size=training_config["batch_size"],
        shuffle=True,
        seed=seed,
        num_workers=training_config["num_workers"],
    )

    val_loader = create_dataloader(
        X=data["X_val"],
        y=data["y_val"],
        task="classification",
        batch_size=training_config["batch_size"],
        shuffle=False,
        seed=seed,
        num_workers=training_config["num_workers"],
    )

    model = ClassificationMLP(
        input_dim=model_config[
            "input_dim"
        ],
        hidden_dims=tuple(
            model_config[
                "hidden_dims"
            ]
        ),
        num_classes=dataset_config["n_classes"],
    )

    model.load_state_dict(
        initial_state
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
        max_epochs=training_config[ "max_epochs"],
        validation_metric_fn=classification_monitor,
        monitor_name="val_macro_f1",
        monitor_mode=early_config["mode"],
        patience=early_config["patience"],
        min_delta=early_config["min_delta"],
        restore_best_weights=early_config["restore_best_weights"],
        verbose=False,
    )

    metrics = evaluate_classification(
        model=model,
        dataloader=val_loader,
        device=device,
    )

    return (
        metrics,
        training_info,
    )

def main():

    config = load_yaml("configs/classification.yaml")

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
        print(
            f"CLASSIFICATION TUNING "
            f"| seed={seed}"
        )
        print(f"==============================")

        data = (
            create_classification_dataset(
                n_samples=dataset_config[
                    "n_samples"
                ],
                n_features=dataset_config[
                    "n_features"
                ],
                n_informative=dataset_config[
                    "n_informative"
                ],
                n_redundant=dataset_config[
                    "n_redundant"
                ],
                n_classes=dataset_config[
                    "n_classes"
                ],
                class_sep=dataset_config[
                    "class_sep"
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
        )

        y_train_noisy, _ = (
            add_symmetric_label_noise(
                data["y_train"],
                noise_level=TUNING_NOISE,
                n_classes=dataset_config["n_classes"],
                seed=seed,
            )
        )

        set_seed(seed)

        base_model = ClassificationMLP(
            input_dim=model_config[
                "input_dim"
            ],
            hidden_dims=tuple(
                model_config[
                    "hidden_dims"
                ]
            ),
            num_classes=dataset_config["n_classes"],
        )

        initial_state = copy.deepcopy(
            base_model.state_dict()
        )

        for q in GCE_Q_VALUES:

            print(
                f"GCE | seed={seed}"
                f" | q={q}"
            )

            criterion = (
                get_classification_loss(
                    "gce",
                    num_classes=dataset_config["n_classes"],
                    gce_q=q,
                )
            )

            metrics, training_info = (
                train_candidate(
                    data=data,
                    y_train_noisy=y_train_noisy,
                    initial_state=initial_state,
                    config=config,
                    seed=seed,
                    criterion=criterion,
                    device=device,
                )
            )

            results.append(
                {
                    "loss": "gce",
                    "seed": seed,
                    "noise_level":TUNING_NOISE,
                    "q": q,
                    "alpha": None,
                    "beta": None,
                    "val_accuracy":
                        metrics["accuracy"],
                    "val_macro_f1":
                        metrics["macro_f1"],
                    "best_epoch":
                        training_info["best_epoch"],
                    "stopped_epoch":
                        training_info["stopped_epoch"],
                    "epochs_trained":
                        training_info["epochs_trained"],
                }
            )

        for alpha, beta in product(
            SCE_ALPHA_VALUES,
            SCE_BETA_VALUES,
        ):

            print(
                f"SCE | seed={seed}"
                f" | alpha={alpha}"
                f" | beta={beta}"
            )

            criterion = (
                get_classification_loss(
                    "sce",
                    num_classes=dataset_config["n_classes"],
                    sce_alpha=alpha,
                    sce_beta=beta,
                )
            )

            metrics, training_info = (
                train_candidate(
                    data=data,
                    y_train_noisy=y_train_noisy,
                    initial_state=initial_state,
                    config=config,
                    seed=seed,
                    criterion=criterion,
                    device=device,
                )
            )

            results.append(
                {
                    "loss": "sce",
                    "seed": seed,
                    "noise_level":TUNING_NOISE,
                    "q": None,
                    "alpha": alpha,
                    "beta": beta,
                    "val_accuracy":
                        metrics["accuracy"],
                    "val_macro_f1":
                        metrics["macro_f1"],
                    "best_epoch":
                        training_info["best_epoch"],
                    "stopped_epoch":
                        training_info["stopped_epoch"],
                    "epochs_trained":
                        training_info["epochs_trained"],
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
        / "classification_tuning_raw.csv",
        index=False,
    )

    gce_summary = (
        df[df["loss"] == "gce"]
        .groupby(
            "q",
            as_index=False,
        )
        .agg(
            macro_f1_mean=(
                "val_macro_f1",
                "mean",
            ),
            macro_f1_std=(
                "val_macro_f1",
                "std",
            ),
            accuracy_mean=(
                "val_accuracy",
                "mean",
            ),
            accuracy_std=(
                "val_accuracy",
                "std",
            ),
        )
        .sort_values(
            [
                "macro_f1_mean",
                "accuracy_mean",
            ],
            ascending=[
                False,
                False,
            ],
        )
    )

    gce_summary.to_csv(
        output_dir
        / "gce_tuning_summary.csv",
        index=False,
    )
    
    sce_summary = (
        df[
            df["loss"] == "sce"
        ]
        .groupby(
            [
                "alpha",
                "beta",
            ],
            as_index=False,
        )
        .agg(
            macro_f1_mean=(
                "val_macro_f1",
                "mean",
            ),
            macro_f1_std=(
                "val_macro_f1",
                "std",
            ),
            accuracy_mean=(
                "val_accuracy",
                "mean",
            ),
            accuracy_std=(
                "val_accuracy",
                "std",
            ),
        )
        .sort_values(
            [
                "macro_f1_mean",
                "accuracy_mean",
            ],
            ascending=[
                False,
                False,
            ],
        )
    )

    sce_summary.to_csv(
        output_dir
        / "sce_tuning_summary.csv",
        index=False,
    )

    print("\n==============================")
    print("GCE SUMMARY")
    print("==============================")

    print(gce_summary)

    print(
        "\nSelected GCE q:",
        gce_summary.iloc[0]["q"],
    )

    print("\n==============================")
    print("SCE SUMMARY")
    print("==============================")

    print(sce_summary)

    print(
        "\nSelected SCE alpha:",
        sce_summary.iloc[0]["alpha"],
    )

    print(
        "Selected SCE beta:",
        sce_summary.iloc[0]["beta"],
    )


if __name__ == "__main__":
    main()