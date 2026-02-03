import csv
from pathlib import Path

results_dir = Path("data/experiments/sample_size_experiment")
out_path = results_dir / "ALL_landscape_norms_sample_sizes.csv"

files = sorted(results_dir.glob("landscape_norm_sample_size_universe_*.csv"))

rows = []
header = None

for p in files:
    with p.open(newline="") as f:
        r = csv.reader(f)
        h = next(r)
        if header is None:
            header = h
        for row in r:
            rows.append(row)

# optional: sort by universe_id then sample_size
rows.sort(key=lambda r: (int(r[0]), r[3]))  # r[0]=universe_id, r[3]=sample_size

with out_path.open("w", newline="") as f:
    w = csv.writer(f)
    w.writerow(header)
    w.writerows(rows)

print(f"Wrote {len(rows)} rows to {out_path}")
