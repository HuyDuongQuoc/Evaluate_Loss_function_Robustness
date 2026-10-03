import copy
from pathlib import Path
import numpy as np
import torch

from src.datasets.regression import create_regression_dataset
from src.datasets.classification import create_classification_dataset

from src.noise.regression_noise import add_regression_noise
from src.noise.classification_noise import add_symmetric_label_noise

from src.models.mlp import (
    RegressionMLP,
    ClassificationMLP,
)

from src.losses.regression_losses import get_regression_loss
from src.losses.classification_losses import get_classification_loss

from src.training.trainer import fit

from src.training.data import create_dataloader

from src.training.evaluator import (
    evaluate_classification,
    evaluate_regression,
)

from src.utils import (
    set_seed,
    get_device,
)

from src.experiments.io import(
    save_checkpoint,
    save_json,
)

def create_optimizer(
    model,
    training_config,
):
    
    optimizer_name = (
        training_config["optimizer"].lower()
    )
    
    if optimizer_name == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=training_config["learning_rate"],
            weight_decay=training_config["weight_decay"],
        )
        
    raise ValueError(
        f"Unsupported optimizer: "
        f"{optimizer_name}"
    )
    
def make_noise_tag(noise_level: float) -> str:
    percentage = round(noise_level*100)
    return f"noise{percentage:02d}"

def make_run_id(
    task: str,
    seed: int,
    noise_level: float,
    loss_name: str,
) -> str:
    noise_tag = make_noise_tag(noise_level)
    
    return (
        f"{task}"
        f"_seed{seed}"
        f"_{noise_tag}"
        f"_{loss_name}"
    )
    
def run_regression_group(
    config: dict,
    seed: int,
    noise_level: float,
    final_evaluation: bool = False,
    verbose: bool = False,
) -> list[dict]:

    training_config = config["training"]
    dataset_config = config["dataset"]
    split_config = config["split"]
    model_config = config["model"]
    loss_config = config["loss"]
    output_config = config["output"]

    root_dir = Path(output_config["root_dir"])

    device = get_device(training_config["device"])

    data = create_regression_dataset(
        n_samples=dataset_config["n_samples"],
        n_features=dataset_config["n_features"],
        n_informative=dataset_config["n_informative"],
        bias=dataset_config["bias"],
        train_size=split_config["train_size"],
        val_size=split_config["val_size"],
        test_size=split_config["test_size"],
        seed=seed,
    )

    y_train_noisy, noise_mask = (
        add_regression_noise(
            data["y_train"],
            noise_level=noise_level,
            seed=seed,
            noise_scale=config["noise"]["scale"],
        )
    )

    actual_noise_ratio = float(noise_mask.mean())

    noise_path = (
        root_dir
        / "noise"
        / (
            f"regression"
            f"_seed{seed}"
            f"_{make_noise_tag(noise_level)}"
            f".npz"
        )
    )

    noise_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        noise_path,
        y_train_clean=data["y_train"],
        y_train_noisy=y_train_noisy,
        noise_mask=noise_mask,
    )

    set_seed(seed)

    base_model = RegressionMLP(
        input_dim=model_config["input_dim"],
        hidden_dims=tuple(
            model_config[
                "hidden_dims"
            ]
        ),
    )

    initial_state = copy.deepcopy(base_model.state_dict())

    results = []

    for loss_name in config["experiment"]["losses"]:

        run_id = make_run_id(
            task="regression",
            seed=seed,
            noise_level=noise_level,
            loss_name=loss_name,
        )

        print(f"\nRunning: {run_id}")

        train_loader = create_dataloader(
            X=data["X_train"],
            y=y_train_noisy,
            task="regression",
            batch_size=training_config["batch_size"],
            shuffle=True,
            seed=seed,
            num_workers=training_config["num_workers"],
        )

        val_loader = create_dataloader(
            X=data["X_val"],
            y=data["y_val"],
            task="regression",
            batch_size=training_config["batch_size"],
            shuffle=False,
            seed=seed,
            num_workers=training_config["num_workers"],
        )

        model = RegressionMLP(
            input_dim=model_config["input_dim"],
            hidden_dims=tuple(
                model_config[
                    "hidden_dims"
                ]
            ),
        )

        model.load_state_dict(initial_state)

        criterion = get_regression_loss(
            loss_name,
            huber_delta=loss_config["huber_delta"],
        )

        optimizer = create_optimizer(
            model=model,
            training_config=training_config,
        )

        history = fit(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epochs=training_config["epochs"],
            verbose=verbose,
        )

        val_metrics = evaluate_regression(
            model=model,
            dataloader=val_loader,
            device=device,
        )

        test_metrics = None

        if final_evaluation:

            test_loader = (
                create_dataloader(
                    X=data["X_test"],
                    y=data["y_test"],
                    task="regression",
                    batch_size=training_config["batch_size"],
                    shuffle=False,
                    seed=seed,
                    num_workers=training_config["num_workers"],
                )
            )

            test_metrics = (
                evaluate_regression(
                    model=model,
                    dataloader=test_loader,
                    device=device,
                )
            )

        run_result = {
            "run_id": run_id,
            "task": "regression",
            "seed": seed,
            "noise_level": noise_level,
            "actual_noise_ratio": actual_noise_ratio,
            "loss": loss_name,
            "validation": val_metrics,
            "test": test_metrics,
            "history": history,
            "config": config,
        }

        save_json(
            run_result,
            root_dir
            / "logs"
            / f"{run_id}.json",
        )

        if output_config["save_checkpoint"]:
            save_checkpoint(
                model=model,
                metadata={
                    "run_id": run_id,
                    "seed": seed,
                    "noise_level":
                        noise_level,
                    "loss":
                        loss_name,
                },
                path=(
                    root_dir
                    / "checkpoints"
                    / f"{run_id}.pt"
                ),
            )

        row = {
            "run_id": run_id,
            "task": "regression",
            "seed": seed,
            "noise_level":
                noise_level,
            "actual_noise_ratio":
                actual_noise_ratio,
            "loss": loss_name,
            "val_mae":
                val_metrics["mae"],
            "val_rmse":
                val_metrics["rmse"],
        }

        if test_metrics is not None:

            row["test_mae"] = (
                test_metrics["mae"]
            )

            row["test_rmse"] = (
                test_metrics["rmse"]
            )

        results.append(row)

    return results

def run_classification_group(
    config: dict,
    seed: int,
    noise_level: float,
    final_evaluation: bool = False,
    verbose: bool = False,
) -> list[dict]:
    
    training_config = config["training"]
    dataset_config = config["dataset"]
    split_config = config["split"]
    model_config = config["model"]
    loss_config = config["loss"]
    output_config = config["output"]

    root_dir = Path(output_config["root_dir"])

    device = get_device(training_config["device"])
    
    data = create_classification_dataset(
            n_samples=dataset_config["n_samples"],
            n_features=dataset_config["n_features"],
            n_informative=dataset_config["n_informative"],
            n_redundant=dataset_config["n_redundant"],
            n_classes=dataset_config["n_classes"],
            class_sep=dataset_config["class_sep"],
            train_size=split_config["train_size"],
            val_size=split_config["val_size"],
            test_size=split_config["test_size"],
            seed=seed,
        )
    
    y_train_noisy, noise_mask = (
            add_symmetric_label_noise(
                data["y_train"],
                noise_level=noise_level,
                n_classes=dataset_config["n_classes"],
                seed=seed,
            )
        )
    
    actual_noise_ratio = float(noise_mask.mean())
    
    noise_path = (
        root_dir
        / "noise"
        / (
            f"classification"
            f"_seed{seed}"
            f"_{make_noise_tag(noise_level)}"
            f".npz"
        )
    )

    noise_path.parent.mkdir(parents=True, exist_ok=True)
    
    np.savez_compressed(
        noise_path,
        y_train_clean=data["y_train"],
        y_train_noisy=y_train_noisy,
        noise_mask=noise_mask,
    )
    
    set_seed(seed)

    base_model = ClassificationMLP(
        input_dim=model_config["input_dim"],
        hidden_dims=tuple(
            model_config["hidden_dims"]
        ),
        num_classes=dataset_config["n_classes"],
    )

    initial_state = copy.deepcopy(base_model.state_dict())

    results = []
    
    for loss_name in config["experiment"]["losses"]:

        run_id = make_run_id(
            task="classification",
            seed=seed,
            noise_level=noise_level,
            loss_name=loss_name,
        )

        print(f"\nRunning: {run_id}")

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
            input_dim=model_config["input_dim"],
            hidden_dims=tuple(
                model_config["hidden_dims"]
            ),
            num_classes=dataset_config["n_classes"],
        )

        model.load_state_dict(initial_state)

        criterion = (
            get_classification_loss(
                loss_name,
                num_classes=dataset_config["n_classes"],
                gce_q=loss_config["gce_q"],
                sce_alpha=loss_config["sce_alpha"],
                sce_beta=loss_config["sce_beta"],
            )
        )

        optimizer = create_optimizer(
            model=model,
            training_config=training_config,
        )

        history = fit(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epochs=training_config["epochs"],
            verbose=verbose,
        )

        val_metrics = (
            evaluate_classification(
                model=model,
                dataloader=val_loader,
                device=device,
            )
        )

        test_metrics = None

        if final_evaluation:

            test_loader = (
                create_dataloader(
                    X=data["X_test"],
                    y=data["y_test"],
                    task="classification",
                    batch_size=training_config["batch_size"],
                    shuffle=False,
                    seed=seed,
                    num_workers=training_config["num_workers"],
                )
            )

            test_metrics = (
                evaluate_classification(
                    model=model,
                    dataloader=test_loader,
                    device=device,
                )
            )

        run_result = {
            "run_id": run_id,
            "task": "classification",
            "seed": seed,
            "noise_level": noise_level,
            "actual_noise_ratio": actual_noise_ratio,
            "loss": loss_name,
            "validation": val_metrics,
            "test": test_metrics,
            "history": history,
            "config": config,
        }

        save_json(
            run_result,
            root_dir
            / "logs"
            / f"{run_id}.json",
        )

        if output_config["save_checkpoint"]:
            save_checkpoint(
                model=model,
                metadata={
                    "run_id": run_id,
                    "seed": seed,
                    "noise_level": noise_level,
                    "loss": loss_name,
                },
                path=(
                    root_dir
                    / "checkpoints"
                    / f"{run_id}.pt"
                ),
            )

        row = {
            "run_id": run_id,
            "task": "classification",
            "seed": seed,
            "noise_level": noise_level,
            "actual_noise_ratio": actual_noise_ratio,
            "loss": loss_name,
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
        }

        if test_metrics is not None:

            row["test_accuracy"] = (
                test_metrics[
                    "accuracy"
                ]
            )

            row["test_macro_f1"] = (
                test_metrics[
                    "macro_f1"
                ]
            )

        results.append(row)

    return results