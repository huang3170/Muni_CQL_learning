from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from . import TrainingConfig, run_experiment
from .demo import make_synthetic_data


def main():
    parser = argparse.ArgumentParser(description="Train a first-fill arrival model from interval CSV/parquet.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--data", type=Path)
    source.add_argument("--demo", action="store_true", help="Use synthetic data; not trading performance")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validation-start")
    parser.add_argument("--test-start")
    parser.add_argument("--ipcw", action="store_true", help="Optional censor-adjusted Brier; read assumptions first")
    args = parser.parse_args()
    if args.demo:
        df = make_synthetic_data()
        print("SYNTHETIC DEMO ONLY. Results do not measure your trading data.")
    elif args.data.suffix.lower() == ".csv":
        # Deserialize CSV timestamps without adding or converting timezones.
        df = pd.read_csv(args.data, dtype={"cusip": "string", "episode_id": "string"},
                         parse_dates=["cycle_time", "quote_end_time", "cep_time"])
    else:
        df = pd.read_parquet(args.data)
    config = TrainingConfig(validation_start=args.validation_start, test_start=args.test_start,
                            compute_ipcw=args.ipcw)
    result = run_experiment(df, args.output, config)
    print(result.split_audit.to_string(index=False))
    print(result.metrics[["model", "split", "events", "event_time_nll_per_30_exposure_minutes", "integrated_hazard_to_events"]].to_string(index=False))
    print("Validation-selected model:", result.selected_model_name)
    print("Outputs:", args.output.resolve())


if __name__ == "__main__":
    main()
