import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd


EXPECTED_SEEDS = (42, 123, 2026)

TASK_SPECS = {
    "regression": {
        "losses": ("mse", "mae", "huber"),
        "noise_levels": (0.0, 0.1, 0.2),
    },
    "classification": {
        "losses": ("ce", "gce", "sce"),
        "noise_levels": (0.0, 0.2, 0.4),
    },
}

LOSS_LABELS = {
    "mse": "MSE",
    "mae": "MAE",
    "huber": "Huber",
    "ce": "CE",
    "gce": "GCE",
    "sce": "SCE",
}

LOSS_COLORS = {
    "mse": "#0072B2",
    "mae": "#009E73",
    "huber": "#D55E00",
    "ce": "#0072B2",
    "gce": "#009E73",
    "sce": "#CC79A7",
}

RESULT_COLUMNS = {
    "run_id",
    "task",
    "loss",
    "noise_level",
    "seed",
    "mae",
    "rmse",
    "accuracy",
    "macro_f1",
    "best_epoch",
    "final_train_loss",
    "final_val_loss",
}

SUMMARY_COLUMNS = {
    "task",
    "loss",
    "noise_level",
    "count",
    "mae_mean",
    "mae_std",
    "rmse_mean",
    "rmse_std",
    "accuracy_mean",
    "accuracy_std",
    "macro_f1_mean",
    "macro_f1_std",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create publication-ready experiment figures from the master "
            "results, summary, and per-run JSON logs."
        )
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("results/metrics/results.csv"),
        help="Path to the 54-row master results CSV.",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("results/metrics/summary.csv"),
        help="Path to the 18-row aggregated summary CSV.",
    )
    parser.add_argument(
        "--logs-dir",
        type=Path,
        default=Path("results/logs"),
        help="Directory containing the 54 per-run JSON logs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/figures"),
        help="Directory in which figures are written.",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=("png", "pdf"),
        default=("png", "pdf"),
        help="One or more output formats (default: png pdf).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Raster resolution used for PNG files (default: 300).",
    )
    parser.add_argument(
        "--representative-seed",
        type=int,
        default=42,
        help="Seed used for representative learning curves (default: 42).",
    )
    return parser.parse_args()


def _normalise_noise(value: object) -> float:
    return round(float(value), 10)


def _expected_combinations() -> set[tuple[str, str, float, int]]:
    return {
        (task, loss, noise, seed)
        for task, spec in TASK_SPECS.items()
        for loss in spec["losses"]
        for noise in spec["noise_levels"]
        for seed in EXPECTED_SEEDS
    }


def _expected_summary_groups() -> set[tuple[str, str, float]]:
    return {
        (task, loss, noise)
        for task, spec in TASK_SPECS.items()
        for loss in spec["losses"]
        for noise in spec["noise_levels"]
    }


def _format_combinations(combinations: set[tuple]) -> str:
    return "\n".join(f"  - {item}" for item in sorted(combinations))


def _require_columns(
    frame: pd.DataFrame,
    required: set[str],
    source: Path,
) -> None:
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(
            f"{source} is missing required columns: "
            f"{', '.join(sorted(missing))}"
        )


def load_results(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Results CSV not found: {path}")

    results = pd.read_csv(path)
    _require_columns(results, RESULT_COLUMNS, path)

    numeric_columns = (
        "noise_level",
        "seed",
        "mae",
        "rmse",
        "accuracy",
        "macro_f1",
        "best_epoch",
        "final_train_loss",
        "final_val_loss",
    )
    for column in numeric_columns:
        results[column] = pd.to_numeric(results[column], errors="coerce")

    return results


def load_summary(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Summary CSV not found: {path}")

    summary = pd.read_csv(path)
    _require_columns(summary, SUMMARY_COLUMNS, path)

    numeric_columns = tuple(SUMMARY_COLUMNS - {"task", "loss"})
    for column in numeric_columns:
        summary[column] = pd.to_numeric(summary[column], errors="coerce")

    return summary


def validate_results(results: pd.DataFrame) -> None:
    expected = _expected_combinations()

    if len(results) != len(expected):
        raise ValueError(
            f"results.csv must contain {len(expected)} rows; "
            f"found {len(results)}."
        )

    duplicate_ids = results.loc[
        results["run_id"].duplicated(keep=False), "run_id"
    ].tolist()
    if duplicate_ids:
        raise ValueError(
            "results.csv contains duplicate run_id values: "
            + ", ".join(sorted(set(duplicate_ids)))
        )

    if results[["task", "loss", "noise_level", "seed"]].isna().any().any():
        raise ValueError(
            "results.csv contains missing task/loss/noise_level/seed values."
        )

    actual = {
        (
            str(row.task),
            str(row.loss),
            _normalise_noise(row.noise_level),
            int(row.seed),
        )
        for row in results.itertuples(index=False)
    }

    missing = expected - actual
    unexpected = actual - expected
    if missing or unexpected:
        details = []
        if missing:
            details.append("Missing combinations:\n" + _format_combinations(missing))
        if unexpected:
            details.append(
                "Unexpected combinations:\n" + _format_combinations(unexpected)
            )
        raise ValueError("Invalid results experiment matrix.\n" + "\n".join(details))

    regression = results[results["task"] == "regression"]
    classification = results[results["task"] == "classification"]

    if regression[["mae", "rmse"]].isna().any().any():
        raise ValueError("Regression rows contain missing MAE or RMSE values.")
    if classification[["accuracy", "macro_f1"]].isna().any().any():
        raise ValueError(
            "Classification rows contain missing accuracy or macro-F1 values."
        )
    if not classification["accuracy"].between(0.0, 1.0).all():
        raise ValueError("Classification accuracy values must be between 0 and 1.")
    if not classification["macro_f1"].between(0.0, 1.0).all():
        raise ValueError("Classification macro-F1 values must be between 0 and 1.")


def validate_summary(summary: pd.DataFrame) -> None:
    expected = _expected_summary_groups()

    if len(summary) != len(expected):
        raise ValueError(
            f"summary.csv must contain {len(expected)} rows; "
            f"found {len(summary)}."
        )

    if summary[["task", "loss", "noise_level"]].isna().any().any():
        raise ValueError("summary.csv contains missing group identifiers.")

    actual = {
        (str(row.task), str(row.loss), _normalise_noise(row.noise_level))
        for row in summary.itertuples(index=False)
    }
    missing = expected - actual
    unexpected = actual - expected
    if missing or unexpected:
        details = []
        if missing:
            details.append("Missing groups:\n" + _format_combinations(missing))
        if unexpected:
            details.append("Unexpected groups:\n" + _format_combinations(unexpected))
        raise ValueError("Invalid summary experiment matrix.\n" + "\n".join(details))

    duplicate_groups = summary.duplicated(
        subset=["task", "loss", "noise_level"],
        keep=False,
    )
    if duplicate_groups.any():
        raise ValueError("summary.csv contains duplicate task/loss/noise groups.")

    invalid_counts = summary[summary["count"] != len(EXPECTED_SEEDS)]
    if not invalid_counts.empty:
        groups = {
            (row.task, row.loss, row.noise_level, row.count)
            for row in invalid_counts.itertuples(index=False)
        }
        raise ValueError(
            "Every summary group must contain exactly three runs. "
            "Invalid groups (task, loss, noise, count):\n"
            + _format_combinations(groups)
        )

    metric_columns = {
        "regression": ("mae_mean", "mae_std", "rmse_mean", "rmse_std"),
        "classification": (
            "accuracy_mean",
            "accuracy_std",
            "macro_f1_mean",
            "macro_f1_std",
        ),
    }
    for task, columns in metric_columns.items():
        task_rows = summary[summary["task"] == task]
        if task_rows[list(columns)].isna().any().any():
            raise ValueError(f"{task} summary rows contain missing metrics.")


def validate_summary_against_results(
    results: pd.DataFrame,
    summary: pd.DataFrame,
) -> None:
    metric_map = {
        "regression": ("mae", "rmse"),
        "classification": ("accuracy", "macro_f1"),
    }

    indexed_summary = summary.set_index(["task", "loss", "noise_level"])

    for task, metrics in metric_map.items():
        task_results = results[results["task"] == task]
        grouped = task_results.groupby(["task", "loss", "noise_level"])

        for group_key, group in grouped:
            summary_row = indexed_summary.loc[group_key]
            if int(summary_row["count"]) != len(group):
                raise ValueError(
                    f"Summary count does not match results for group {group_key}."
                )

            for metric in metrics:
                expected_mean = group[metric].mean()
                expected_std = group[metric].std(ddof=1)
                actual_mean = summary_row[f"{metric}_mean"]
                actual_std = summary_row[f"{metric}_std"]

                if not np.isclose(actual_mean, expected_mean, rtol=1e-9, atol=1e-12):
                    raise ValueError(
                        f"Summary mean mismatch for {group_key}, metric {metric}: "
                        f"expected {expected_mean}, found {actual_mean}."
                    )
                if not np.isclose(actual_std, expected_std, rtol=1e-9, atol=1e-12):
                    raise ValueError(
                        f"Summary std mismatch for {group_key}, metric {metric}: "
                        f"expected {expected_std}, found {actual_std}."
                    )


def _expected_run_id(
    task: str,
    seed: int,
    noise_level: float,
    loss: str,
) -> str:
    noise_percentage = round(noise_level * 100)
    return f"{task}_seed{seed}_noise{noise_percentage:02d}_{loss}"


def load_and_validate_logs(
    logs_dir: Path,
    expected_result_ids: set[str],
) -> dict[str, dict]:
    if not logs_dir.is_dir():
        raise FileNotFoundError(f"Logs directory not found: {logs_dir}")

    paths = sorted(logs_dir.glob("*.json"))
    if len(paths) != len(_expected_combinations()):
        raise ValueError(
            f"{logs_dir} must contain exactly 54 JSON logs; found {len(paths)}."
        )

    logs: dict[str, dict] = {}
    for path in paths:
        try:
            with path.open("r", encoding="utf-8") as file:
                run = json.load(file)
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"Could not read JSON log {path}: {error}") from error

        required_keys = {
            "run_id",
            "task",
            "seed",
            "noise_level",
            "loss",
            "test",
            "history",
        }
        missing_keys = required_keys - set(run)
        if missing_keys:
            raise ValueError(
                f"{path} is missing keys: {', '.join(sorted(missing_keys))}"
            )

        run_id = str(run["run_id"])
        if run_id in logs:
            raise ValueError(f"Duplicate run_id in JSON logs: {run_id}")
        if not run["test"]:
            raise ValueError(f"Log was not created with --final: {path}")

        task = str(run["task"])
        loss = str(run["loss"])
        seed = int(run["seed"])
        noise_level = _normalise_noise(run["noise_level"])
        canonical_id = _expected_run_id(task, seed, noise_level, loss)
        if run_id != canonical_id:
            raise ValueError(
                f"Log metadata and run_id disagree in {path}: "
                f"expected {canonical_id}, found {run_id}."
            )

        history = run["history"]
        train_history = history.get("train_loss", [])
        val_history = history.get("val_loss", [])
        if not train_history or not val_history:
            raise ValueError(f"Missing learning-curve history in {path}.")
        if len(train_history) != len(val_history):
            raise ValueError(f"Train/validation history lengths differ in {path}.")

        if task == "classification":
            matrix = np.asarray(
                run["test"].get("confusion_matrix"),
                dtype=float,
            )
            if matrix.shape != (3, 3):
                raise ValueError(
                    f"Classification confusion matrix in {path} must be 3x3; "
                    f"found {matrix.shape}."
                )
            if not np.isfinite(matrix).all() or (matrix < 0).any():
                raise ValueError(f"Invalid confusion matrix values in {path}.")

        logs[run_id] = run

    log_ids = set(logs)
    missing_logs = expected_result_ids - log_ids
    unexpected_logs = log_ids - expected_result_ids
    if missing_logs or unexpected_logs:
        details = []
        if missing_logs:
            details.append("Missing logs:\n" + "\n".join(sorted(missing_logs)))
        if unexpected_logs:
            details.append(
                "Unexpected logs:\n" + "\n".join(sorted(unexpected_logs))
            )
        raise ValueError("JSON logs do not match results.csv.\n" + "\n".join(details))

    return logs


def configure_plot_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.family": "sans-serif",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "--",
            "legend.frameon": False,
            "lines.linewidth": 2.0,
            "lines.markersize": 6,
        }
    )


def save_figure(
    figure: plt.Figure,
    output_dir: Path,
    stem: str,
    formats: tuple[str, ...],
    dpi: int,
) -> list[Path]:
    paths = []
    for output_format in formats:
        path = output_dir / f"{stem}.{output_format}"
        save_options = {"bbox_inches": "tight"}
        if output_format == "png":
            save_options["dpi"] = dpi
        figure.savefig(path, **save_options)
        paths.append(path)
    plt.close(figure)
    return paths


def plot_performance_metric(
    summary: pd.DataFrame,
    task: str,
    metric: str,
    title: str,
    y_label: str,
    percent: bool = False,
) -> plt.Figure:
    figure, axis = plt.subplots(figsize=(7.2, 4.8))
    task_summary = summary[summary["task"] == task]

    for loss in TASK_SPECS[task]["losses"]:
        data = (
            task_summary[task_summary["loss"] == loss]
            .sort_values("noise_level")
        )
        x = data["noise_level"].to_numpy() * 100.0
        y = data[f"{metric}_mean"].to_numpy(dtype=float)
        y_error = data[f"{metric}_std"].to_numpy(dtype=float)

        if percent:
            y = y * 100.0
            y_error = y_error * 100.0

        axis.errorbar(
            x,
            y,
            yerr=y_error,
            label=LOSS_LABELS[loss],
            color=LOSS_COLORS[loss],
            marker="o",
            capsize=4,
        )

    axis.set_title(title)
    axis.set_xlabel("Training-label noise (%)")
    axis.set_ylabel(y_label)
    axis.set_xticks(
        np.asarray(TASK_SPECS[task]["noise_levels"], dtype=float) * 100.0
    )
    if percent:
        axis.set_ylim(0.0, 100.0)
        axis.yaxis.set_major_formatter(PercentFormatter(xmax=100.0))
    else:
        axis.set_ylim(bottom=0.0)
    axis.legend(title="Training loss")
    figure.tight_layout()
    return figure


def calculate_paired_robustness(
    results: pd.DataFrame,
    task: str,
    metric: str,
    baseline_minus_noisy: bool,
) -> pd.DataFrame:
    task_results = results[results["task"] == task].copy()
    baseline = (
        task_results[task_results["noise_level"] == 0.0]
        .set_index(["loss", "seed"])[metric]
    )

    rows = []
    for loss in TASK_SPECS[task]["losses"]:
        for noise_level in TASK_SPECS[task]["noise_levels"]:
            group = task_results[
                (task_results["loss"] == loss)
                & (task_results["noise_level"] == noise_level)
            ].set_index("seed")

            deltas = []
            for seed in EXPECTED_SEEDS:
                baseline_value = float(baseline.loc[(loss, seed)])
                noisy_value = float(group.loc[seed, metric])
                delta = (
                    baseline_value - noisy_value
                    if baseline_minus_noisy
                    else noisy_value - baseline_value
                )
                deltas.append(delta)

            rows.append(
                {
                    "loss": loss,
                    "noise_level": noise_level,
                    "mean": float(np.mean(deltas)),
                    "std": float(np.std(deltas, ddof=1)),
                }
            )

    return pd.DataFrame(rows)


def plot_robustness(
    robustness: pd.DataFrame,
    task: str,
    title: str,
    y_label: str,
    percentage_points: bool = False,
) -> plt.Figure:
    figure, axis = plt.subplots(figsize=(7.2, 4.8))

    for loss in TASK_SPECS[task]["losses"]:
        data = robustness[robustness["loss"] == loss].sort_values(
            "noise_level"
        )
        x = data["noise_level"].to_numpy() * 100.0
        y = data["mean"].to_numpy(dtype=float)
        y_error = data["std"].to_numpy(dtype=float)

        if percentage_points:
            y = y * 100.0
            y_error = y_error * 100.0

        axis.errorbar(
            x,
            y,
            yerr=y_error,
            label=LOSS_LABELS[loss],
            color=LOSS_COLORS[loss],
            marker="o",
            capsize=4,
        )

    axis.axhline(0.0, color="#555555", linewidth=1.0)
    axis.set_title(title)
    axis.set_xlabel("Training-label noise (%)")
    axis.set_ylabel(y_label)
    axis.set_xticks(
        np.asarray(TASK_SPECS[task]["noise_levels"], dtype=float) * 100.0
    )
    axis.legend(title="Training loss")
    figure.tight_layout()
    return figure


def plot_confusion_matrices(logs: dict[str, dict]) -> plt.Figure:
    figure = plt.figure(figsize=(12.8, 4.4), layout="constrained")
    grid = figure.add_gridspec(
        1,
        4,
        width_ratios=(1.0, 1.0, 1.0, 0.055),
        wspace=0.08,
    )
    axes = np.asarray(
        [figure.add_subplot(grid[0, index]) for index in range(3)]
    )
    colorbar_axis = figure.add_subplot(grid[0, 3])
    image = None

    for axis, loss in zip(axes, TASK_SPECS["classification"]["losses"]):
        combined = np.zeros((3, 3), dtype=float)
        for seed in EXPECTED_SEEDS:
            run_id = _expected_run_id(
                "classification",
                seed,
                0.4,
                loss,
            )
            combined += np.asarray(
                logs[run_id]["test"]["confusion_matrix"],
                dtype=float,
            )

        row_totals = combined.sum(axis=1, keepdims=True)
        if (row_totals == 0).any():
            raise ValueError(
                f"Cannot normalise the {loss} confusion matrix: an actual "
                "class has zero samples."
            )
        normalised = combined / row_totals
        if not np.allclose(normalised.sum(axis=1), 1.0, atol=1e-9):
            raise ValueError(f"Row-normalised {loss} confusion matrix is invalid.")

        image = axis.imshow(
            normalised,
            cmap="Blues",
            vmin=0.0,
            vmax=1.0,
        )
        axis.set_title(LOSS_LABELS[loss])
        axis.set_xlabel("Predicted class")
        axis.set_ylabel("Actual class")
        axis.set_xticks(range(3), labels=("Class 0", "Class 1", "Class 2"))
        axis.set_yticks(range(3), labels=("Class 0", "Class 1", "Class 2"))
        axis.grid(False)

        for row in range(3):
            for column in range(3):
                value = normalised[row, column]
                count = int(round(combined[row, column]))
                axis.text(
                    column,
                    row,
                    f"{value:.1%}\n(n={count})",
                    ha="center",
                    va="center",
                    color="white" if value >= 0.5 else "black",
                    fontsize=9,
                )

    figure.suptitle(
        "Classification Confusion Matrices at 40% Label Noise\n"
        "Aggregated across three seeds and normalised by actual class",
    )
    colorbar = figure.colorbar(image, cax=colorbar_axis)
    colorbar.set_label("Share of actual class")
    colorbar.ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    return figure


def plot_learning_curves(
    logs: dict[str, dict],
    task: str,
    seed: int,
) -> plt.Figure:
    spec = TASK_SPECS[task]
    noise_levels = (spec["noise_levels"][0], spec["noise_levels"][-1])
    figure, axes = plt.subplots(2, 3, figsize=(13.0, 7.2), squeeze=False)

    for row_index, noise_level in enumerate(noise_levels):
        for column_index, loss in enumerate(spec["losses"]):
            axis = axes[row_index, column_index]
            run_id = _expected_run_id(task, seed, noise_level, loss)
            history = logs[run_id]["history"]
            train_loss = np.asarray(history["train_loss"], dtype=float)
            val_loss = np.asarray(history["val_loss"], dtype=float)
            epochs = np.arange(1, len(train_loss) + 1)

            axis.plot(
                epochs,
                train_loss,
                color="#0072B2",
                label="Training loss",
            )
            axis.plot(
                epochs,
                val_loss,
                color="#D55E00",
                label="Validation loss",
            )
            axis.set_title(
                f"{LOSS_LABELS[loss]} — {noise_level * 100:.0f}% noise"
            )
            axis.set_xlabel("Epoch")
            axis.set_ylabel("Loss")
            axis.set_xlim(left=1)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        bbox_to_anchor=(0.5, 0.96),
    )
    task_label = task.capitalize()
    figure.suptitle(
        f"{task_label} Learning Curves — Representative Seed {seed}",
        y=1.0,
        fontsize=14,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.91))
    return figure


def create_all_figures(
    results: pd.DataFrame,
    summary: pd.DataFrame,
    logs: dict[str, dict],
    output_dir: Path,
    formats: tuple[str, ...],
    dpi: int,
    representative_seed: int,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    written_paths = []

    figures = [
        (
            "01_regression_mae",
            plot_performance_metric(
                summary,
                task="regression",
                metric="mae",
                title="Regression Test MAE under Training-Target Noise",
                y_label="Test MAE",
            ),
        ),
        (
            "02_regression_rmse",
            plot_performance_metric(
                summary,
                task="regression",
                metric="rmse",
                title="Regression Test RMSE under Training-Target Noise",
                y_label="Test RMSE",
            ),
        ),
        (
            "03_classification_accuracy",
            plot_performance_metric(
                summary,
                task="classification",
                metric="accuracy",
                title="Classification Test Accuracy under Label Noise",
                y_label="Test accuracy",
                percent=True,
            ),
        ),
        (
            "04_classification_macro_f1",
            plot_performance_metric(
                summary,
                task="classification",
                metric="macro_f1",
                title="Classification Test Macro-F1 under Label Noise",
                y_label="Test macro-F1",
                percent=True,
            ),
        ),
    ]

    regression_robustness = calculate_paired_robustness(
        results,
        task="regression",
        metric="mae",
        baseline_minus_noisy=False,
    )
    figures.append(
        (
            "05_regression_robustness_delta_mae",
            plot_robustness(
                regression_robustness,
                task="regression",
                title="Regression Robustness: Paired Increase in Test MAE",
                y_label="Test MAE increase (ΔMAE)",
            ),
        )
    )

    classification_robustness = calculate_paired_robustness(
        results,
        task="classification",
        metric="accuracy",
        baseline_minus_noisy=True,
    )
    figures.append(
        (
            "06_classification_robustness_delta_accuracy",
            plot_robustness(
                classification_robustness,
                task="classification",
                title="Classification Robustness: Paired Drop in Test Accuracy",
                y_label="Test accuracy drop (percentage points)",
                percentage_points=True,
            ),
        )
    )

    figures.extend(
        [
            (
                "07_classification_confusion_matrices_noise40",
                plot_confusion_matrices(logs),
            ),
            (
                f"08_regression_learning_curves_seed{representative_seed}",
                plot_learning_curves(
                    logs,
                    task="regression",
                    seed=representative_seed,
                ),
            ),
            (
                f"09_classification_learning_curves_seed{representative_seed}",
                plot_learning_curves(
                    logs,
                    task="classification",
                    seed=representative_seed,
                ),
            ),
        ]
    )

    for stem, figure in figures:
        written_paths.extend(
            save_figure(
                figure,
                output_dir=output_dir,
                stem=stem,
                formats=formats,
                dpi=dpi,
            )
        )

    return written_paths


def main() -> None:
    args = parse_args()

    if args.dpi <= 0:
        raise ValueError("--dpi must be a positive integer.")
    if args.representative_seed not in EXPECTED_SEEDS:
        raise ValueError(
            "--representative-seed must be one of: "
            + ", ".join(map(str, EXPECTED_SEEDS))
        )

    formats = tuple(dict.fromkeys(args.formats))

    results = load_results(args.results)
    summary = load_summary(args.summary)
    validate_results(results)
    validate_summary(summary)
    validate_summary_against_results(results, summary)

    logs = load_and_validate_logs(
        args.logs_dir,
        expected_result_ids=set(results["run_id"].astype(str)),
    )

    configure_plot_style()
    written_paths = create_all_figures(
        results=results,
        summary=summary,
        logs=logs,
        output_dir=args.output_dir,
        formats=formats,
        dpi=args.dpi,
        representative_seed=args.representative_seed,
    )

    print(f"Created {len(written_paths)} figure files in {args.output_dir}:")
    for path in written_paths:
        print(f"  {path}")


if __name__ == "__main__":
    main()
