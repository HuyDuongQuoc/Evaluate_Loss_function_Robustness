from pathlib import Path

import pandas as pd


input_path = Path("results/metrics/results.csv")
output_path = Path("results/metrics/summary.csv")

results = pd.read_csv(input_path)

summary = (
    results
    .groupby(
        ["task", "loss", "noise_level"],
        as_index=False,
    )
    .agg(
        count=("run_id", "count"),
        mae_mean=("mae", "mean"),
        mae_std=("mae", "std"),
        rmse_mean=("rmse", "mean"),
        rmse_std=("rmse", "std"),
        accuracy_mean=("accuracy", "mean"),
        accuracy_std=("accuracy", "std"),
        macro_f1_mean=("macro_f1", "mean"),
        macro_f1_std=("macro_f1", "std"),
    )
    .sort_values(
        ["task", "noise_level", "loss"]
    )
    .reset_index(drop=True)
)

summary.to_csv(output_path, index=False)

print(summary.to_string(index=False))
print(f"\nSaved: {output_path}")
