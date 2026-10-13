import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


EXPECTED_SEEDS = (42, 123, 2026)

TASK_SPECS = {
    "regression": {
        "metric": "mae",
        "losses": ("mse", "mae", "huber"),
        "noise_levels": (0.1, 0.2),
    },
    "classification": {
        "metric": "accuracy",
        "losses": ("ce", "gce", "sce"),
        "noise_levels": (0.2, 0.4),
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

SEED_STYLES = {
    42: {"linestyle": "-", "marker": "o"},
    123: {"linestyle": "--", "marker": "s"},
    2026: {"linestyle": ":", "marker": "^"},
}

RAW_COLUMNS = {
    "task",
    "metric",
    "loss",
    "noise_level",
    "seed",
    "baseline_run_id",
    "noisy_run_id",
    "baseline_value",
    "noisy_value",
    "delta",
}

SUMMARY_COLUMNS = {
    "task",
    "metric",
    "loss",
    "noise_level",
    "count",
    "baseline_mean",
    "baseline_std",
    "noisy_mean",
    "noisy_std",
    "delta_mean",
    "delta_std",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a dedicated suite of robustness comparison figures from "
            "paired-seed robustness results."
        )
    )
    parser.add_argument(
        "--raw",
        type=Path,
        default=Path("results/metrics/robustness_results.csv"),
        help="Path to the 36-row paired robustness CSV.",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("results/metrics/robustness_summary.csv"),
        help="Path to the 12-row robustness summary CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/figures/robustness"),
        help="Directory in which robustness figures are written.",
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
    return parser.parse_args()


def _normalise_noise(value: object) -> float:
    return round(float(value), 10)


def _expected_raw_combinations() -> set[tuple[str, str, float, int]]:
    return {
        (task, loss, noise_level, seed)
        for task, spec in TASK_SPECS.items()
        for loss in spec["losses"]
        for noise_level in spec["noise_levels"]
        for seed in EXPECTED_SEEDS
    }


def _expected_summary_combinations() -> set[tuple[str, str, float]]:
    return {
        (task, loss, noise_level)
        for task, spec in TASK_SPECS.items()
        for loss in spec["losses"]
        for noise_level in spec["noise_levels"]
    }


def _format_combinations(combinations: set[tuple]) -> str:
    return "\n".join(f"  - {combination}" for combination in sorted(combinations))


def _require_columns(
    frame: pd.DataFrame,
    required: set[str],
    source: Path,
) -> None:
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(
            f"{source} is missing required columns: "
            + ", ".join(sorted(missing))
        )


def load_raw(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Raw robustness CSV not found: {path}")

    raw = pd.read_csv(path)
    _require_columns(raw, RAW_COLUMNS, path)

    for column in (
        "noise_level",
        "seed",
        "baseline_value",
        "noisy_value",
        "delta",
    ):
        raw[column] = pd.to_numeric(raw[column], errors="coerce")

    return raw


def load_summary(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Robustness summary CSV not found: {path}")

    summary = pd.read_csv(path)
    _require_columns(summary, SUMMARY_COLUMNS, path)

    for column in SUMMARY_COLUMNS - {"task", "metric", "loss"}:
        summary[column] = pd.to_numeric(summary[column], errors="coerce")

    return summary


def validate_raw(raw: pd.DataFrame) -> None:
    expected = _expected_raw_combinations()

    if len(raw) != len(expected):
        raise ValueError(
            f"Raw robustness data must contain {len(expected)} rows; "
            f"found {len(raw)}."
        )

    required_non_null = [
        "task",
        "metric",
        "loss",
        "noise_level",
        "seed",
        "baseline_run_id",
        "noisy_run_id",
        "baseline_value",
        "noisy_value",
        "delta",
    ]
    if raw[required_non_null].isna().any().any():
        raise ValueError("Raw robustness data contains missing or invalid values.")

    duplicate_pairs = raw.duplicated(
        subset=["task", "loss", "noise_level", "seed"],
        keep=False,
    )
    if duplicate_pairs.any():
        duplicates = raw.loc[
            duplicate_pairs,
            ["task", "loss", "noise_level", "seed"],
        ]
        raise ValueError(
            "Raw robustness data contains duplicate paired rows:\n"
            + duplicates.to_string(index=False)
        )

    actual = {
        (
            str(row.task),
            str(row.loss),
            _normalise_noise(row.noise_level),
            int(row.seed),
        )
        for row in raw.itertuples(index=False)
    }
    missing = expected - actual
    unexpected = actual - expected
    if missing or unexpected:
        messages = []
        if missing:
            messages.append("Missing pairs:\n" + _format_combinations(missing))
        if unexpected:
            messages.append(
                "Unexpected pairs:\n" + _format_combinations(unexpected)
            )
        raise ValueError("Invalid raw robustness matrix.\n" + "\n".join(messages))

    for task, spec in TASK_SPECS.items():
        task_rows = raw[raw["task"] == task]
        if not (task_rows["metric"] == spec["metric"]).all():
            raise ValueError(
                f"All {task} rows must use metric={spec['metric']}."
            )

    numeric_values = raw[
        ["baseline_value", "noisy_value", "delta"]
    ].to_numpy(dtype=float)
    if not np.isfinite(numeric_values).all():
        raise ValueError("Raw robustness values must all be finite.")

    classification = raw[raw["task"] == "classification"]
    for column in ("baseline_value", "noisy_value"):
        if not classification[column].between(0.0, 1.0).all():
            raise ValueError(
                f"Classification {column} values must be between 0 and 1."
            )

    regression = raw[raw["task"] == "regression"]
    if not np.allclose(
        regression["delta"],
        regression["noisy_value"] - regression["baseline_value"],
        rtol=1e-10,
        atol=1e-12,
    ):
        raise ValueError(
            "Regression delta must equal noisy_value - baseline_value."
        )
    if not np.allclose(
        classification["delta"],
        classification["baseline_value"] - classification["noisy_value"],
        rtol=1e-10,
        atol=1e-12,
    ):
        raise ValueError(
            "Classification delta must equal baseline_value - noisy_value."
        )


def validate_summary(summary: pd.DataFrame) -> None:
    expected = _expected_summary_combinations()

    if len(summary) != len(expected):
        raise ValueError(
            f"Robustness summary must contain {len(expected)} rows; "
            f"found {len(summary)}."
        )

    if summary[list(SUMMARY_COLUMNS)].isna().any().any():
        raise ValueError("Robustness summary contains missing or invalid values.")

    duplicate_groups = summary.duplicated(
        subset=["task", "loss", "noise_level"],
        keep=False,
    )
    if duplicate_groups.any():
        duplicates = summary.loc[
            duplicate_groups,
            ["task", "loss", "noise_level"],
        ]
        raise ValueError(
            "Robustness summary contains duplicate groups:\n"
            + duplicates.to_string(index=False)
        )

    actual = {
        (str(row.task), str(row.loss), _normalise_noise(row.noise_level))
        for row in summary.itertuples(index=False)
    }
    missing = expected - actual
    unexpected = actual - expected
    if missing or unexpected:
        messages = []
        if missing:
            messages.append("Missing groups:\n" + _format_combinations(missing))
        if unexpected:
            messages.append(
                "Unexpected groups:\n" + _format_combinations(unexpected)
            )
        raise ValueError(
            "Invalid robustness summary matrix.\n" + "\n".join(messages)
        )

    if not (summary["count"] == len(EXPECTED_SEEDS)).all():
        invalid = summary.loc[
            summary["count"] != len(EXPECTED_SEEDS),
            ["task", "loss", "noise_level", "count"],
        ]
        raise ValueError(
            "Every summary group must contain three seeds:\n"
            + invalid.to_string(index=False)
        )

    for task, spec in TASK_SPECS.items():
        task_rows = summary[summary["task"] == task]
        if not (task_rows["metric"] == spec["metric"]).all():
            raise ValueError(
                f"All {task} summary rows must use metric={spec['metric']}."
            )

    numeric_columns = [
        "baseline_mean",
        "baseline_std",
        "noisy_mean",
        "noisy_std",
        "delta_mean",
        "delta_std",
    ]
    if not np.isfinite(summary[numeric_columns].to_numpy(dtype=float)).all():
        raise ValueError("Summary statistics must all be finite.")


def validate_summary_against_raw(
    raw: pd.DataFrame,
    summary: pd.DataFrame,
) -> None:
    indexed_summary = summary.set_index(
        ["task", "metric", "loss", "noise_level"]
    )
    grouped = raw.groupby(
        ["task", "metric", "loss", "noise_level"],
        sort=False,
    )

    statistic_map = {
        "baseline_mean": ("baseline_value", "mean"),
        "baseline_std": ("baseline_value", "std"),
        "noisy_mean": ("noisy_value", "mean"),
        "noisy_std": ("noisy_value", "std"),
        "delta_mean": ("delta", "mean"),
        "delta_std": ("delta", "std"),
    }

    for group_key, group in grouped:
        if group_key not in indexed_summary.index:
            raise ValueError(f"Summary is missing raw group {group_key}.")

        summary_row = indexed_summary.loc[group_key]
        if int(summary_row["count"]) != len(group):
            raise ValueError(f"Summary count mismatch for {group_key}.")

        for summary_column, (raw_column, operation) in statistic_map.items():
            expected = getattr(group[raw_column], operation)()
            actual = float(summary_row[summary_column])
            if not np.isclose(actual, expected, rtol=1e-9, atol=1e-12):
                raise ValueError(
                    f"{summary_column} mismatch for {group_key}: "
                    f"expected {expected}, found {actual}."
                )


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
    written = []
    for output_format in formats:
        path = output_dir / f"{stem}.{output_format}"
        options = {"bbox_inches": "tight"}
        if output_format == "png":
            options["dpi"] = dpi
        figure.savefig(path, **options)
        written.append(path)
    plt.close(figure)
    return written


def plot_degradation_curve(
    summary: pd.DataFrame,
    task: str,
) -> plt.Figure:
    spec = TASK_SPECS[task]
    percentage_points = task == "classification"
    figure, axis = plt.subplots(figsize=(7.2, 4.8))

    for loss in spec["losses"]:
        data = summary[
            (summary["task"] == task) & (summary["loss"] == loss)
        ].sort_values("noise_level")

        x = np.concatenate(([0.0], data["noise_level"].to_numpy() * 100.0))
        y = np.concatenate(([0.0], data["delta_mean"].to_numpy(dtype=float)))
        y_error = np.concatenate(
            ([0.0], data["delta_std"].to_numpy(dtype=float))
        )

        if percentage_points:
            y = y * 100.0
            y_error = y_error * 100.0

        axis.errorbar(
            x,
            y,
            yerr=y_error,
            color=LOSS_COLORS[loss],
            marker="o",
            capsize=4,
            label=LOSS_LABELS[loss],
        )

    if task == "regression":
        title = "Regression Robustness: Paired Increase in Test MAE"
        y_label = "ΔMAE (lower is more robust)"
    else:
        title = "Classification Robustness: Paired Drop in Test Accuracy"
        y_label = "ΔAccuracy (percentage points; lower is more robust)"

    axis.axhline(0.0, color="#555555", linewidth=1.0)
    axis.set_title(title)
    axis.set_xlabel("Training-label noise (%)")
    axis.set_ylabel(y_label)
    axis.set_xticks(
        np.concatenate(([0.0], np.asarray(spec["noise_levels"]) * 100.0))
    )
    axis.legend(title="Training loss")
    figure.tight_layout()
    return figure


def plot_degradation_heatmap(
    summary: pd.DataFrame,
    task: str,
) -> plt.Figure:
    spec = TASK_SPECS[task]
    percentage_points = task == "classification"
    values = np.zeros((len(spec["losses"]), len(spec["noise_levels"])))
    errors = np.zeros_like(values)

    for row, loss in enumerate(spec["losses"]):
        for column, noise_level in enumerate(spec["noise_levels"]):
            matched = summary[
                (summary["task"] == task)
                & (summary["loss"] == loss)
                & np.isclose(summary["noise_level"], noise_level)
            ]
            values[row, column] = float(matched.iloc[0]["delta_mean"])
            errors[row, column] = float(matched.iloc[0]["delta_std"])

    if percentage_points:
        values = values * 100.0
        errors = errors * 100.0

    figure, axis = plt.subplots(figsize=(6.8, 4.5))
    lower = min(0.0, float(values.min()))
    upper = float(values.max())
    if np.isclose(lower, upper):
        upper = lower + 1.0
    image = axis.imshow(values, cmap="YlOrRd", vmin=lower, vmax=upper)
    axis.grid(False)
    axis.set_xticks(
        range(len(spec["noise_levels"])),
        labels=[f"{noise * 100:.0f}%" for noise in spec["noise_levels"]],
    )
    axis.set_yticks(
        range(len(spec["losses"])),
        labels=[LOSS_LABELS[loss] for loss in spec["losses"]],
    )
    axis.set_xlabel("Training-label noise")
    axis.set_ylabel("Training loss")

    metric_label = "ΔMAE" if task == "regression" else "ΔAccuracy (pp)"
    task_label = task.capitalize()
    axis.set_title(
        f"{task_label} Robustness Heatmap — {metric_label}\n"
        "Lower values indicate greater robustness"
    )

    midpoint = lower + (upper - lower) / 2.0
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = values[row, column]
            axis.text(
                column,
                row,
                f"{value:.3f} ± {errors[row, column]:.3f}",
                ha="center",
                va="center",
                color="white" if value > midpoint else "black",
                fontsize=10,
            )

    colorbar = figure.colorbar(image, ax=axis, fraction=0.045, pad=0.04)
    colorbar.set_label(metric_label)
    figure.tight_layout()
    return figure


def plot_seed_level_degradation(
    raw: pd.DataFrame,
    summary: pd.DataFrame,
    task: str,
) -> plt.Figure:
    spec = TASK_SPECS[task]
    percentage_points = task == "classification"
    figure, axes = plt.subplots(1, 3, figsize=(13.0, 4.3), squeeze=False)

    for column, loss in enumerate(spec["losses"]):
        axis = axes[0, column]
        loss_raw = raw[(raw["task"] == task) & (raw["loss"] == loss)]
        loss_summary = summary[
            (summary["task"] == task) & (summary["loss"] == loss)
        ].sort_values("noise_level")

        for shade_index, seed in enumerate(EXPECTED_SEEDS):
            data = loss_raw[loss_raw["seed"] == seed].sort_values(
                "noise_level"
            )
            x = np.concatenate(([0.0], data["noise_level"].to_numpy() * 100.0))
            y = np.concatenate(([0.0], data["delta"].to_numpy(dtype=float)))
            if percentage_points:
                y = y * 100.0

            style = SEED_STYLES[seed]
            axis.plot(
                x,
                y,
                color=str(0.45 + 0.15 * shade_index),
                linewidth=1.3,
                alpha=0.9,
                linestyle=style["linestyle"],
                marker=style["marker"],
                label=f"Seed {seed}",
            )

        mean_x = np.concatenate(
            ([0.0], loss_summary["noise_level"].to_numpy() * 100.0)
        )
        mean_y = np.concatenate(
            ([0.0], loss_summary["delta_mean"].to_numpy(dtype=float))
        )
        if percentage_points:
            mean_y = mean_y * 100.0

        axis.plot(
            mean_x,
            mean_y,
            color=LOSS_COLORS[loss],
            linewidth=3.0,
            marker="D",
            label="Mean",
            zorder=5,
        )
        axis.set_title(LOSS_LABELS[loss])
        axis.set_xlabel("Training-label noise (%)")
        axis.set_xticks(
            np.concatenate(([0.0], np.asarray(spec["noise_levels"]) * 100.0))
        )
        axis.set_ylabel(
            "ΔMAE" if task == "regression" else "ΔAccuracy (percentage points)"
        )

    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        ncol=4,
        bbox_to_anchor=(0.5, 1.02),
    )
    task_label = task.capitalize()
    figure.suptitle(
        f"{task_label} Robustness across Individual Seeds",
        y=1.12,
        fontsize=14,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))
    return figure


def plot_loss_robustness_comparison(
    summary: pd.DataFrame,
    task: str,
) -> plt.Figure:
    spec = TASK_SPECS[task]
    figure, axis = plt.subplots(figsize=(7.6, 5.2))

    for loss in spec["losses"]:
        data = summary[
            (summary["task"] == task) & (summary["loss"] == loss)
        ].sort_values("noise_level")
        noise = np.concatenate(
            ([0.0], data["noise_level"].to_numpy(dtype=float) * 100.0)
        )
        means = np.concatenate(
            ([0.0], data["delta_mean"].to_numpy(dtype=float))
        )
        errors = np.concatenate(
            ([0.0], data["delta_std"].to_numpy(dtype=float))
        )
        if task == "classification":
            means = means * 100.0
            errors = errors * 100.0

        axis.errorbar(
            noise,
            means,
            yerr=errors,
            color=LOSS_COLORS[loss],
            marker="o",
            markersize=8,
            capsize=5,
            linewidth=2.2,
            label=LOSS_LABELS[loss],
        )

    axis.set_xticks(
        np.concatenate(
            ([0.0], np.asarray(spec["noise_levels"], dtype=float) * 100.0)
        )
    )
    axis.set_xlabel("Noise Level")
    axis.set_ylim(bottom=0.0)
    axis.margins(x=0.04, y=0.08)

    if task == "regression":
        axis.set_title("Regression Robustness Comparison")
        axis.set_ylabel("ΔMAE")
    else:
        axis.set_title("Classification Robustness Comparison")
        axis.set_ylabel("ΔAccuracy Percentage Points")
    axis.legend(title="Loss Function")
    figure.tight_layout()
    return figure


def create_all_figures(
    raw: pd.DataFrame,
    summary: pd.DataFrame,
    output_dir: Path,
    formats: tuple[str, ...],
    dpi: int,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    figures = [
        (
            "01_regression_delta_mae_curve",
            plot_degradation_curve(summary, task="regression"),
        ),
        (
            "02_classification_delta_accuracy_curve",
            plot_degradation_curve(summary, task="classification"),
        ),
        (
            "03_regression_delta_mae_heatmap",
            plot_degradation_heatmap(summary, task="regression"),
        ),
        (
            "04_classification_delta_accuracy_heatmap",
            plot_degradation_heatmap(summary, task="classification"),
        ),
        (
            "05_regression_seed_level_degradation",
            plot_seed_level_degradation(
                raw,
                summary,
                task="regression",
            ),
        ),
        (
            "06_classification_seed_level_degradation",
            plot_seed_level_degradation(
                raw,
                summary,
                task="classification",
            ),
        ),
        (
            "07_regression_loss_robustness_comparison",
            plot_loss_robustness_comparison(
                summary,
                task="regression",
            ),
        ),
        (
            "08_classification_loss_robustness_comparison",
            plot_loss_robustness_comparison(
                summary,
                task="classification",
            ),
        ),
    ]

    written = []
    for stem, figure in figures:
        written.extend(
            save_figure(
                figure,
                output_dir=output_dir,
                stem=stem,
                formats=formats,
                dpi=dpi,
            )
        )

    return written


def _validate_output_path(
    raw_path: Path,
    summary_path: Path,
    output_dir: Path,
) -> None:
    resolved_output = output_dir.resolve()
    for source in (raw_path, summary_path):
        resolved_source = source.resolve()
        if resolved_output == resolved_source:
            raise ValueError("--output-dir must be a directory, not an input CSV.")


def main() -> None:
    args = parse_args()
    if args.dpi <= 0:
        raise ValueError("--dpi must be a positive integer.")

    formats = tuple(dict.fromkeys(args.formats))
    _validate_output_path(args.raw, args.summary, args.output_dir)

    raw = load_raw(args.raw)
    summary = load_summary(args.summary)
    validate_raw(raw)
    validate_summary(summary)
    validate_summary_against_raw(raw, summary)

    configure_plot_style()
    written = create_all_figures(
        raw=raw,
        summary=summary,
        output_dir=args.output_dir,
        formats=formats,
        dpi=args.dpi,
    )

    print(f"Created {len(written)} robustness figure files in {args.output_dir}:")
    for path in written:
        print(f"  {path}")


if __name__ == "__main__":
    main()
