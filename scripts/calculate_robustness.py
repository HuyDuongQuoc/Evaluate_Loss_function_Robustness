import argparse
from pathlib import Path

import numpy as np
import pandas as pd


EXPECTED_SEEDS = (42, 123, 2026)

TASK_SPECS = {
    "regression": {
        "metric": "mae",
        "losses": ("mse", "mae", "huber"),
        "noise_levels": (0.0, 0.1, 0.2),
        "baseline_minus_noisy": False,
    },
    "classification": {
        "metric": "accuracy",
        "losses": ("ce", "gce", "sce"),
        "noise_levels": (0.0, 0.2, 0.4),
        "baseline_minus_noisy": True,
    },
}

REQUIRED_COLUMNS = {
    "run_id",
    "task",
    "loss",
    "noise_level",
    "seed",
    "mae",
    "accuracy",
}

RAW_COLUMNS = (
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
)

SUMMARY_COLUMNS = (
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
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Calculate paired-seed robustness metrics from the master "
            "experiment results."
        )
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("results/metrics/results.csv"),
        help="Path to the 54-row master results CSV.",
    )
    parser.add_argument(
        "--raw-output",
        type=Path,
        default=Path("results/metrics/robustness_results.csv"),
        help="Output path for per-seed paired robustness results.",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("results/metrics/robustness_summary.csv"),
        help="Output path for aggregated robustness statistics.",
    )
    return parser.parse_args()


def _normalise_noise(value: object) -> float:
    return round(float(value), 10)


def _expected_combinations() -> set[tuple[str, str, float, int]]:
    return {
        (task, loss, noise_level, seed)
        for task, spec in TASK_SPECS.items()
        for loss in spec["losses"]
        for noise_level in spec["noise_levels"]
        for seed in EXPECTED_SEEDS
    }


def _format_combinations(combinations: set[tuple]) -> str:
    return "\n".join(f"  - {combination}" for combination in sorted(combinations))


def load_results(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Results CSV not found: {path}")

    results = pd.read_csv(path)
    missing_columns = REQUIRED_COLUMNS - set(results.columns)
    if missing_columns:
        raise ValueError(
            f"{path} is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    for column in ("noise_level", "seed", "mae", "accuracy"):
        results[column] = pd.to_numeric(results[column], errors="coerce")

    return results


def validate_results(results: pd.DataFrame) -> None:
    expected = _expected_combinations()

    if len(results) != len(expected):
        raise ValueError(
            f"results.csv must contain {len(expected)} runs; found {len(results)}."
        )

    if results["run_id"].isna().any():
        raise ValueError("results.csv contains a missing run_id.")

    duplicate_ids = results.loc[
        results["run_id"].duplicated(keep=False), "run_id"
    ].astype(str)
    if not duplicate_ids.empty:
        raise ValueError(
            "results.csv contains duplicate run_id values: "
            + ", ".join(sorted(set(duplicate_ids)))
        )

    identifier_columns = ["task", "loss", "noise_level", "seed"]
    if results[identifier_columns].isna().any().any():
        raise ValueError(
            "results.csv contains missing or invalid task/loss/noise_level/seed "
            "values."
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
        messages = []
        if missing:
            messages.append("Missing runs:\n" + _format_combinations(missing))
        if unexpected:
            messages.append(
                "Unexpected runs:\n" + _format_combinations(unexpected)
            )
        raise ValueError("Invalid experiment matrix.\n" + "\n".join(messages))

    regression = results[results["task"] == "regression"]
    classification = results[results["task"] == "classification"]

    if regression["mae"].isna().any():
        bad_runs = regression.loc[regression["mae"].isna(), "run_id"].astype(str)
        raise ValueError(
            "Regression runs contain missing or invalid MAE values: "
            + ", ".join(bad_runs)
        )
    if classification["accuracy"].isna().any():
        bad_runs = classification.loc[
            classification["accuracy"].isna(), "run_id"
        ].astype(str)
        raise ValueError(
            "Classification runs contain missing or invalid Accuracy values: "
            + ", ".join(bad_runs)
        )
    if not classification["accuracy"].between(0.0, 1.0).all():
        raise ValueError("Classification Accuracy values must be between 0 and 1.")


def _select_single_run(
    results: pd.DataFrame,
    task: str,
    loss: str,
    noise_level: float,
    seed: int,
) -> pd.Series:
    matched = results[
        (results["task"] == task)
        & (results["loss"] == loss)
        & np.isclose(results["noise_level"], noise_level)
        & (results["seed"] == seed)
    ]

    if len(matched) != 1:
        raise ValueError(
            "Expected exactly one run for "
            f"task={task}, loss={loss}, noise_level={noise_level}, seed={seed}; "
            f"found {len(matched)}."
        )

    return matched.iloc[0]


def calculate_paired_robustness(results: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for task, spec in TASK_SPECS.items():
        metric = str(spec["metric"])
        baseline_minus_noisy = bool(spec["baseline_minus_noisy"])

        for loss in spec["losses"]:
            for noise_level in spec["noise_levels"][1:]:
                for seed in EXPECTED_SEEDS:
                    baseline = _select_single_run(
                        results,
                        task=task,
                        loss=loss,
                        noise_level=0.0,
                        seed=seed,
                    )
                    noisy = _select_single_run(
                        results,
                        task=task,
                        loss=loss,
                        noise_level=noise_level,
                        seed=seed,
                    )

                    baseline_value = float(baseline[metric])
                    noisy_value = float(noisy[metric])
                    delta = (
                        baseline_value - noisy_value
                        if baseline_minus_noisy
                        else noisy_value - baseline_value
                    )

                    rows.append(
                        {
                            "task": task,
                            "metric": metric,
                            "loss": loss,
                            "noise_level": noise_level,
                            "seed": seed,
                            "baseline_run_id": str(baseline["run_id"]),
                            "noisy_run_id": str(noisy["run_id"]),
                            "baseline_value": baseline_value,
                            "noisy_value": noisy_value,
                            "delta": delta,
                        }
                    )

    raw = pd.DataFrame(rows, columns=RAW_COLUMNS)
    if len(raw) != 36:
        raise RuntimeError(
            f"Expected 36 paired robustness rows; calculated {len(raw)}."
        )
    if not np.isfinite(
        raw[["baseline_value", "noisy_value", "delta"]].to_numpy(dtype=float)
    ).all():
        raise ValueError("Calculated robustness rows contain non-finite values.")

    return raw


def create_robustness_summary(raw: pd.DataFrame) -> pd.DataFrame:
    summary = (
        raw.groupby(
            ["task", "metric", "loss", "noise_level"],
            as_index=False,
            sort=False,
        )
        .agg(
            count=("seed", "count"),
            baseline_mean=("baseline_value", "mean"),
            baseline_std=("baseline_value", "std"),
            noisy_mean=("noisy_value", "mean"),
            noisy_std=("noisy_value", "std"),
            delta_mean=("delta", "mean"),
            delta_std=("delta", "std"),
        )
        .loc[:, SUMMARY_COLUMNS]
    )

    if len(summary) != 12:
        raise RuntimeError(
            f"Expected 12 robustness summary groups; calculated {len(summary)}."
        )
    if not (summary["count"] == len(EXPECTED_SEEDS)).all():
        invalid = summary.loc[
            summary["count"] != len(EXPECTED_SEEDS),
            ["task", "loss", "noise_level", "count"],
        ]
        raise ValueError(
            "Every robustness group must contain three paired seeds:\n"
            + invalid.to_string(index=False)
        )

    return summary


def validate_calculations(
    raw: pd.DataFrame,
    summary: pd.DataFrame,
) -> None:
    regression = raw[raw["task"] == "regression"]
    classification = raw[raw["task"] == "classification"]

    if not np.allclose(
        regression["delta"],
        regression["noisy_value"] - regression["baseline_value"],
    ):
        raise RuntimeError("Regression delta validation failed.")
    if not np.allclose(
        classification["delta"],
        classification["baseline_value"] - classification["noisy_value"],
    ):
        raise RuntimeError("Classification delta validation failed.")

    grouped = raw.groupby(
        ["task", "metric", "loss", "noise_level"],
        sort=False,
    )
    indexed_summary = summary.set_index(
        ["task", "metric", "loss", "noise_level"]
    )

    for group_key, group in grouped:
        summary_row = indexed_summary.loc[group_key]
        expected_mean = group["delta"].mean()
        expected_std = group["delta"].std(ddof=1)

        if not np.isclose(
            summary_row["delta_mean"],
            expected_mean,
            rtol=1e-12,
            atol=1e-12,
        ):
            raise RuntimeError(f"Delta mean validation failed for {group_key}.")
        if not np.isclose(
            summary_row["delta_std"],
            expected_std,
            rtol=1e-12,
            atol=1e-12,
        ):
            raise RuntimeError(f"Delta std validation failed for {group_key}.")


def _validate_output_paths(
    results_path: Path,
    raw_output: Path,
    summary_output: Path,
) -> None:
    resolved_results = results_path.resolve()
    resolved_raw = raw_output.resolve()
    resolved_summary = summary_output.resolve()

    if resolved_raw == resolved_summary:
        raise ValueError("--raw-output and --summary-output must be different files.")
    if resolved_results in {resolved_raw, resolved_summary}:
        raise ValueError("Output paths must not overwrite the input results CSV.")


def write_outputs(
    raw: pd.DataFrame,
    summary: pd.DataFrame,
    raw_output: Path,
    summary_output: Path,
) -> None:
    raw_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.parent.mkdir(parents=True, exist_ok=True)

    raw.to_csv(raw_output, index=False)
    summary.to_csv(summary_output, index=False)


def main() -> None:
    args = parse_args()
    _validate_output_paths(args.results, args.raw_output, args.summary_output)

    results = load_results(args.results)
    validate_results(results)

    raw = calculate_paired_robustness(results)
    summary = create_robustness_summary(raw)
    validate_calculations(raw, summary)

    write_outputs(
        raw=raw,
        summary=summary,
        raw_output=args.raw_output,
        summary_output=args.summary_output,
    )

    print(f"Saved {len(raw)} paired rows to {args.raw_output}")
    print(f"Saved {len(summary)} summary rows to {args.summary_output}")
    print("\nRobustness summary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
