import copy

import torch

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


SEED = 42
BATCH_SIZE = 32
INPUT_DIM = 20
NUM_CLASSES = 3


def gradients_are_finite(model):
    for parameter in model.parameters():

        if parameter.grad is None:
            continue

        if not torch.isfinite(parameter.grad).all():
            return False

    return True


def test_regression():

    print("\n=== REGRESSION ===")

    torch.manual_seed(SEED)

    base_model = RegressionMLP(
        input_dim=INPUT_DIM
    )

    initial_state = copy.deepcopy(
        base_model.state_dict()
    )

    x = torch.randn(
        BATCH_SIZE,
        INPUT_DIM,
    )

    y = torch.randn(BATCH_SIZE)

    for loss_name in [
        "mse",
        "mae",
        "huber",
    ]:

        model = RegressionMLP(
            input_dim=INPUT_DIM
        )

        # Every loss starts from identical weights.
        model.load_state_dict(initial_state)

        criterion = get_regression_loss(
            loss_name
        )

        model.zero_grad(
            set_to_none=True
        )

        predictions = model(x)

        loss = criterion(
            predictions,
            y,
        )

        loss.backward()

        print(
            f"{loss_name.upper():6s} "
            f"| output={tuple(predictions.shape)} "
            f"| loss={loss.item():.6f} "
            f"| finite_grad={gradients_are_finite(model)}"
        )


def test_classification():

    print("\n=== CLASSIFICATION ===")

    torch.manual_seed(SEED)

    base_model = ClassificationMLP(
        input_dim=INPUT_DIM,
        num_classes=NUM_CLASSES,
    )

    initial_state = copy.deepcopy(
        base_model.state_dict()
    )

    x = torch.randn(
        BATCH_SIZE,
        INPUT_DIM,
    )

    y = torch.randint(
        low=0,
        high=NUM_CLASSES,
        size=(BATCH_SIZE,),
    )

    for loss_name in [
        "ce",
        "gce",
        "sce",
    ]:

        model = ClassificationMLP(
            input_dim=INPUT_DIM,
            num_classes=NUM_CLASSES,
        )

        # Every loss starts from identical weights.
        model.load_state_dict(initial_state)

        criterion = get_classification_loss(
            loss_name,
            num_classes=NUM_CLASSES,
        )

        model.zero_grad(
            set_to_none=True
        )

        logits = model(x)

        loss = criterion(
            logits,
            y,
        )

        loss.backward()

        print(
            f"{loss_name.upper():6s} "
            f"| output={tuple(logits.shape)} "
            f"| loss={loss.item():.6f} "
            f"| finite_grad={gradients_are_finite(model)}"
        )


if __name__ == "__main__":

    test_regression()
    test_classification()

    print("\nPhase 2 sanity check completed.")