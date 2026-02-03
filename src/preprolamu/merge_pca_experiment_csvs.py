import csv
from pathlib import Path

results_dir = Path("data/experiments/pca_dim_experiment")
out_path = results_dir / "ALL_landscape_norms_pca_dims.csv"

files = sorted(results_dir.glob("landscape_norm_pca_dims_universe_*.csv"))

header = None
rows = []

for p in files:
    with p.open(newline="") as f:
        r = csv.reader(f)
        h = next(r)
        header = header or h
        rows.extend(list(r))

# sort: universe_id then pca_components
# rows.sort(key=lambda r: (int(r[0]), int(r[4])))

with out_path.open("w", newline="") as f:
    w = csv.writer(f)
    w.writerow(header)
    w.writerows(rows)

print(f"Wrote {len(rows)} rows to {out_path}")
