import copy

import torch

from src.datasets.regression import (
    create_regression_dataset,
)

from src.datasets.classification import (
    create_classification_dataset,
)

from src.noise.regression_noise import (
    add_regression_noise,
)

from src.noise.classification_noise import (
    add_symmetric_label_noise,
)

from src.models.mlp import (
    RegressionMLP,
    ClassificationMLP,
)

from src.losses.regression_losses import (
    get_regression_loss,
)

from src.losses.classification_losses import (
    get_classification_loss,
)

from src.training.data import (
    create_dataloader,
)

from src.training.trainer import fit

from src.training.evaluator import (
    evaluate_regression,
    evaluate_classification,
)

from src.utils import (
    set_seed,
    get_device,
)

SEED = 42

BATCH_SIZE = 64

SMOKE_EPOCHS = 3

LEARNING_RATE = 0.001

INPUT_DIM = 20

NUM_CLASSES = 3

def smoke_test_regression():

    print("\n==============================")
    print("REGRESSION SMOKE TEST")
    print("==============================")

    set_seed(SEED)

    device = get_device()

    data = create_regression_dataset(
        seed=SEED
    )

    y_train_noisy, noise_mask = (
        add_regression_noise(
            data["y_train"],
            noise_level=0.1,
            seed=SEED,
            noise_scale=1.0,
        )
    )

    print(
        "Actual noise ratio:",
        noise_mask.mean(),
    )

    # Create ONE initial model.
    base_model = RegressionMLP(
        input_dim=INPUT_DIM
    )

    initial_state = copy.deepcopy(
        base_model.state_dict()
    )

    for loss_name in [
        "mse",
        "mae",
        "huber",
    ]:

        print(
            f"\n--- {loss_name.upper()} ---"
        )

        # Fresh DataLoader for every loss.
        train_loader = create_dataloader(
            X=data["X_train"],
            y=y_train_noisy,
            task="regression",
            batch_size=BATCH_SIZE,
            shuffle=True,
            seed=SEED,
        )

        val_loader = create_dataloader(
            X=data["X_val"],
            y=data["y_val"],
            task="regression",
            batch_size=BATCH_SIZE,
            shuffle=False,
            seed=SEED,
        )

        model = RegressionMLP(
            input_dim=INPUT_DIM
        )

        # Identical initialization.
        model.load_state_dict(
            initial_state
        )

        criterion = get_regression_loss(
            loss_name
        )

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=LEARNING_RATE,
        )

        history = fit(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epochs=SMOKE_EPOCHS,
        )

        metrics = evaluate_regression(
            model=model,
            dataloader=val_loader,
            device=device,
        )

        print(
            "Validation metrics:",
            metrics,
        )
        
def smoke_test_classification():

    print("\n==============================")
    print("CLASSIFICATION SMOKE TEST")
    print("==============================")

    set_seed(SEED)

    device = get_device()

    data = create_classification_dataset(
        seed=SEED
    )

    y_train_noisy, noise_mask = (
        add_symmetric_label_noise(
            data["y_train"],
            noise_level=0.2,
            n_classes=NUM_CLASSES,
            seed=SEED,
        )
    )

    print(
        "Actual noise ratio:",
        noise_mask.mean(),
    )

    base_model = ClassificationMLP(
        input_dim=INPUT_DIM,
        num_classes=NUM_CLASSES,
    )

    initial_state = copy.deepcopy(
        base_model.state_dict()
    )

    for loss_name in [
        "ce",
        "gce",
        "sce",
    ]:

        print(
            f"\n--- {loss_name.upper()} ---"
        )

        train_loader = create_dataloader(
            X=data["X_train"],
            y=y_train_noisy,
            task="classification",
            batch_size=BATCH_SIZE,
            shuffle=True,
            seed=SEED,
        )

        val_loader = create_dataloader(
            X=data["X_val"],
            y=data["y_val"],
            task="classification",
            batch_size=BATCH_SIZE,
            shuffle=False,
            seed=SEED,
        )

        model = ClassificationMLP(
            input_dim=INPUT_DIM,
            num_classes=NUM_CLASSES,
        )

        model.load_state_dict(
            initial_state
        )

        criterion = get_classification_loss(
            loss_name,
            num_classes=NUM_CLASSES,
        )

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=LEARNING_RATE,
        )

        history = fit(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epochs=SMOKE_EPOCHS,
        )

        metrics = evaluate_classification(
            model=model,
            dataloader=val_loader,
            device=device,
        )

        print(
            "Validation Accuracy:",
            metrics["accuracy"],
        )

        print(
            "Validation Macro-F1:",
            metrics["macro_f1"],
        )

        print(
            "Confusion Matrix:"
        )

        print(
            metrics["confusion_matrix"]
        )
        
if __name__ == "__main__":

    smoke_test_regression()

    smoke_test_classification()

    print(
        "\nPhase 3 smoke test completed."
    )