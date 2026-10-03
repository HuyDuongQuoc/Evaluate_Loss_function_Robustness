import argparse
from pathlib import Path
import pandas as pd

from src.experiments.io import load_yaml

from src.experiments.runner import run_classification_group

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        default=(
            "configs/"
            "classification.yaml"
        ),
    )

    parser.add_argument(
        "--mode",
        choices=[
            "single",
            "baseline",
            "full",
        ],
        default="single",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--noise",
        type=float,
        default=0.2,
    )

    parser.add_argument(
        "--final",
        action="store_true",
    )

    return parser.parse_args()


def main():

    args = parse_args()

    config = load_yaml(args.config)

    if args.mode == "single":
        pairs = [
            (args.seed, args.noise)
        ]

    elif args.mode == "baseline":
        pairs = [
            (seed, 0.0)
            for seed in config["experiment"]["seeds"]
        ]

    else:

        pairs = [
            (seed, noise)
            for seed in config["experiment"]["seeds"]
            for noise in config["noise"]["levels"]
        ]

    all_results = []

    for seed, noise in pairs:
        rows = (
            run_classification_group(
                config=config,
                seed=seed,
                noise_level=noise,
                final_evaluation=args.final,
                verbose=(
                    args.mode == "single"
                ),
            )
        )

        all_results.extend(rows)

        output_dir = Path(
            config["output"]["root_dir"]
        ) / "metrics"

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_name = (
            "classification_final.csv"
            if args.final
            else (
                f"classification_"
                f"{args.mode}.csv"
            )
        )

        pd.DataFrame(all_results).to_csv(
            output_dir
            / output_name,
            index=False,
        )

    print("\nClassification experiments completed.")


if __name__ == "__main__":
    main()