import json
from pathlib import Path

import pandas as pd


log_dir = Path("results/logs")
output_path = Path("results/metrics/results.csv")

rows = []

for path in sorted(log_dir.glob("*.json")):
    with path.open("r", encoding="utf-8") as file:
        run = json.load(file)

    task = run.get("task")
    noise_level = float(run.get("noise_level", -1))

    # Chỉ lấy đúng 54 main-study combinations.
    valid_run = (
        task == "regression"
        and noise_level in {0.0, 0.1, 0.2}
    ) or (
        task == "classification"
        and noise_level in {0.0, 0.2, 0.4}
    )

    if not valid_run:
        continue

    # Bỏ qua log được chạy không có --final.
    test = run.get("test")
    if not test:
        continue

    history = run.get("history", {})
    train_history = history.get("train_loss", [])
    val_history = history.get("val_loss", [])
    training_info = run.get("training_info", {})

    row = {
        "run_id": run["run_id"],
        "task": task,
        "loss": run["loss"],
        "noise_level": noise_level,
        "seed": int(run["seed"]),
        "mae": test.get("mae"),
        "rmse": test.get("rmse"),
        "accuracy": test.get("accuracy"),
        "macro_f1": test.get("macro_f1"),
        "best_epoch": training_info.get("best_epoch"),
        "final_train_loss": (
            train_history[-1] if train_history else None
        ),
        "final_val_loss": (
            val_history[-1] if val_history else None
        ),
    }

    rows.append(row)

results = pd.DataFrame(rows)

if results.empty:
    raise RuntimeError(
        "Không tìm thấy final log. "
        "Hãy chắc rằng experiments đã chạy với --final."
    )

duplicates = results[
    results.duplicated(subset=["run_id"], keep=False)
]

if not duplicates.empty:
    raise RuntimeError(
        "Phát hiện run_id trùng:\n"
        + duplicates[["run_id"]].to_string(index=False)
    )

results = results.sort_values(
    ["task", "noise_level", "loss", "seed"]
).reset_index(drop=True)

results.to_csv(output_path, index=False)

print(results)
print(f"\nSaved: {output_path}")
print(f"Total rows: {len(results)}")
print(
    results.groupby(
        ["task", "loss", "noise_level"]
    ).size().rename("count")
)
