import csv
import numpy as np
import os
import glob
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Compute seed statistics from metrics CSV files")
    parser.add_argument(
        "--dataset",
        default=None,
        help="Optional dataset filter (e.g., PSM, WADI). If omitted, include all datasets.",
    )
    return parser.parse_args()


    args = parse_args()
    dataset_filter = args.dataset.upper() if args.dataset else None
for dataset_filter in ["SMD","SMAP","SWAT","PSM", "MSL", "WADI"]:
    # Directory containing the metrics CSV files
    results_dir = os.path.dirname(os.path.abspath(__file__))
    csv_files = []
    for seed in range(1000, 1005):
        fpath = os.path.join(results_dir, f"{seed}metrics_by_dataset.csv")
        if os.path.exists(fpath):
            csv_files.append(fpath)
    csv_files = sorted(csv_files, key=os.path.getmtime)

    # Aggregate data: data[dataset][metric] = list of values
    data = {}
    metrics = set()

    REQUESTED_METRICS = [
        "AUC-ROC",
        "Standard-F1",
        "VUS-PR",
        "VUS-ROC",
        "AUC-PR",
        "Affiliation-F",
        "PA-F1",
        "R_AUC_PR",
        "R_AUC_ROC"
    ]
    data=[]
    def parse_file(fpath):
        with open(fpath, "r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            # Search from the last row backwards and append the first row
            # whose dataset matches the optional dataset_filter (if provided).
            if rows:
                found = False
                for row in reversed(rows):
                    # support both 'Dataset' and 'Data' header names
                    ds = row.get("Dataset") or row.get("Data")
                    if ds is None:
                        # no dataset field — accept the last row
                        data.append(row)
                        found = True
                        break
                    if dataset_filter is None or ds.upper() == dataset_filter:
                        data.append(row)
                        found = True
                        break
                if not found:
                    # no matching row found; skip this file
                    return



    for fpath in csv_files:
        parse_file(fpath)

    # Write aggregated statistics
    stat_file = os.path.join(results_dir, f"metrics_seed_stats_{dataset_filter or 'all'}.csv")
    strout=None
    with open(stat_file, "w", newline="") as wf:
        writer = csv.writer(wf)
        writer.writerow(["Dataset", "Metric", "Mean", "Std", "Min", "Max"])
        for metric in REQUESTED_METRICS:
            valueslist = []
            for dataset in data:
                values = np.array(dataset[metric], dtype=float)
                valueslist.append(values)
            if len(valueslist) > 0:
                mean = np.mean(valueslist)
                std = np.std(valueslist)
                minv = np.min(valueslist)
                maxv = np.max(valueslist)
                writer.writerow([
                    dataset_filter,metric,
                    f"{mean:.3f}", f"{std:.4f}", f"{minv:.3f}", f"{maxv:.3f}"
                ])
                if strout is None:  
                    strout=f"{mean:.3f}""±"f"{std:.4f}"
                else:
                    strout+=f" & {mean:.3f}""±"f"{std:.4f}"

    print(f"Saved statistics to {stat_file}")
    print(strout)

