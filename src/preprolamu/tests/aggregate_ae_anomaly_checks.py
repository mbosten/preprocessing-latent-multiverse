from pathlib import Path

import pandas as pd

BASE_DATA_DIR = Path("data")
OUTDIR = BASE_DATA_DIR / "tests"
SUMMARY_DIR = OUTDIR / "ae_debug_summaries"
MASTER_TABLE_PATH = OUTDIR / "ae_debug_master_table.parquet"
MASTER_TABLE_CSV_PATH = OUTDIR / "ae_debug_master_table.csv"


def aggregate_summary_rows(
    summary_dir: Path, master_parquet_path: Path, master_csv_path: Path
):
    files = sorted(summary_dir.glob("ae_debug_summary_*.parquet"))
    if not files:
        print("No summary files found.")
        return

    dfs = [pd.read_parquet(p) for p in files]
    master = pd.concat(dfs, ignore_index=True)

    if "dataset_id" in master.columns and "universe_id" in master.columns:
        master = master.sort_values(["dataset_id", "universe_id"]).reset_index(
            drop=True
        )

    master.to_parquet(master_parquet_path, index=False)
    master.to_csv(master_csv_path, index=False)

    print(f"Wrote: {master_parquet_path}")
    print(f"Wrote: {master_csv_path}")


if __name__ == "__main__":
    aggregate_summary_rows(SUMMARY_DIR, MASTER_TABLE_PATH, MASTER_TABLE_CSV_PATH)
