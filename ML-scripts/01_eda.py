#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import json
import pandas as pd


TARGET_COL = "RainTomorrow"


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def main():
    ap = argparse.ArgumentParser(description="EDA for weatherAUS.csv")
    ap.add_argument("--data-path", type=str, required=True, help="Path to weatherAUS.csv")
    ap.add_argument("--out-dir", type=str, default="artifacts/eda", help="Output directory for EDA artifacts")
    args = ap.parse_args()

    ensure_dir(args.out_dir)

    df = pd.read_csv(args.data_path)

    # Basic info
    info = {
        "shape": [int(df.shape[0]), int(df.shape[1])],
        "columns": df.columns.tolist(),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
    }

    # Missing values
    missing = (df.isna().mean().sort_values(ascending=False) * 100).round(2)
    missing_df = missing.reset_index()
    missing_df.columns = ["column", "missing_pct"]

    # Target distribution (if present)
    target_stats = {}
    if TARGET_COL in df.columns:
        vc = df[TARGET_COL].value_counts(dropna=False)
        target_stats = {str(k): int(v) for k, v in vc.items()}

    # Write artifacts
    with open(os.path.join(args.out_dir, "eda_summary.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "info": info,
                "target_distribution": target_stats,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    df.head(50).to_csv(os.path.join(args.out_dir, "head_50.csv"), index=False)
    missing_df.to_csv(os.path.join(args.out_dir, "missing_values_pct.csv"), index=False)

    # Describe numeric
    num_desc = df.select_dtypes(include=["number"]).describe().T
    num_desc.to_csv(os.path.join(args.out_dir, "numeric_describe.csv"))

    print(f"[OK] EDA artifacts written to: {args.out_dir}")


if __name__ == "__main__":
    main()
