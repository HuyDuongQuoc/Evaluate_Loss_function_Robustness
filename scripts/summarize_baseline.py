from pathlib import Path

import pandas as pd


def summarize_regression():

    path = Path(
        "results/metrics/"
        "regression_baseline.csv"
    )

    df = pd.read_csv(path)

    summary = (
        df
        .groupby(
            "loss",
            as_index=False,
        )
        .agg(
            mae_mean=(
                "val_mae",
                "mean",
            ),
            mae_std=(
                "val_mae",
                "std",
            ),
            rmse_mean=(
                "val_rmse",
                "mean",
            ),
            rmse_std=(
                "val_rmse",
                "std",
            ),
        )
    )

    summary.to_csv(
        "results/metrics/"
        "regression_baseline_summary.csv",
        index=False,
    )

    print("\nREGRESSION BASELINE")

    print(summary)


def summarize_classification():

    path = Path(
        "results/metrics/"
        "classification_baseline.csv"
    )

    df = pd.read_csv(path)

    summary = (
        df
        .groupby(
            "loss",
            as_index=False,
        )
        .agg(
            accuracy_mean=(
                "val_accuracy",
                "mean",
            ),
            accuracy_std=(
                "val_accuracy",
                "std",
            ),
            macro_f1_mean=(
                "val_macro_f1",
                "mean",
            ),
            macro_f1_std=(
                "val_macro_f1",
                "std",
            ),
        )
    )

    summary.to_csv(
        "results/metrics/"
        "classification_baseline_summary.csv",
        index=False,
    )

    print("\nCLASSIFICATION BASELINE")

    print(summary)


if __name__ == "__main__":

    summarize_regression()

    summarize_classification()