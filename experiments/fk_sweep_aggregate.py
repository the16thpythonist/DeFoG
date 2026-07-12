"""
Aggregate the FK-SMC beta x warmup sweep (results/fk_steer__aqsoldb/*/).

Reads every completed run's experiment_data.json, builds the comparison table over
the three objectives -- mean-to-target bias, MAE, and diversity (# unique) -- marks
the Pareto-optimal configs, and renders beta x warmup heatmaps. Safe to run while
the sweep is still going (only completed runs are read).

    python experiments/fk_sweep_aggregate.py
"""
import glob
import json
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS = "experiments/results/fk_steer__aqsoldb"
OUTDIR = "experiments/_fk_sweep_report"
LEVELS = ["low", "med", "high"]


def load_runs():
    rows = []
    for f in sorted(glob.glob(os.path.join(RESULTS, "*", "experiment_data.json"))):
        if os.path.basename(os.path.dirname(f)) == "debug":
            continue  # stale __DEBUG__ archive, not a sweep config
        try:
            d = json.load(open(f))
        except Exception:
            continue
        # pycomex stores e["a/b"]=x as nested {"a": {"b": x}}
        results = d.get("results", {})
        s = results.get("summary")
        if not s:  # run not finished
            continue
        row = {"beta": float(s["BETA"]), "warmup": float(s["WARMUP_FRAC"]),
               "avg_abs_bias": s["avg_abs_bias"], "avg_mae": s["avg_mae"],
               "total_unique": s["total_unique"], "min_unique": s.get("min_unique"),
               "path": os.path.dirname(f)}
        for lvl in LEVELS:
            r = results.get(lvl, {})
            row[f"{lvl}_bias"] = r.get("bias")
            row[f"{lvl}_mae"] = r.get("mae")
            row[f"{lvl}_unique"] = r.get("n_unique")
            row[f"{lvl}_size"] = r.get("size_mean")
        rows.append(row)
    return pd.DataFrame(rows)


def pareto_mask(df):
    """Pareto-optimal minimizing avg_abs_bias & avg_mae, maximizing total_unique."""
    b = df["avg_abs_bias"].values
    m = df["avg_mae"].values
    u = df["total_unique"].values
    n = len(df)
    opt = np.ones(n, bool)
    for i in range(n):
        for j in range(n):
            if j == i:
                continue
            # j dominates i?
            if (b[j] <= b[i] and m[j] <= m[i] and u[j] >= u[i] and
                    (b[j] < b[i] or m[j] < m[i] or u[j] > u[i])):
                opt[i] = False
                break
    return opt


def heatmap(df, col, title, path, better="low"):
    piv = df.pivot(index="beta", columns="warmup", values=col)
    fig, ax = plt.subplots(figsize=(6, 5))
    cmap = "viridis_r" if better == "low" else "viridis"
    im = ax.imshow(piv.values, cmap=cmap, aspect="auto", origin="lower")
    ax.set_xticks(range(len(piv.columns))); ax.set_xticklabels([f"{w:.2f}" for w in piv.columns])
    ax.set_yticks(range(len(piv.index))); ax.set_yticklabels([f"{b:.1f}" for b in piv.index])
    ax.set_xlabel("warmup_frac"); ax.set_ylabel("beta")
    ax.set_title(title)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}" if abs(v) < 100 else f"{v:.0f}",
                        ha="center", va="center", color="w", fontsize=9)
    fig.colorbar(im, ax=ax)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    df = load_runs()
    if df.empty:
        print("no completed runs yet."); return
    df = df.sort_values(["avg_abs_bias", "avg_mae"]).reset_index(drop=True)
    df["pareto"] = pareto_mask(df)
    print(f"completed runs: {len(df)}/20\n")
    cols = ["beta", "warmup", "avg_abs_bias", "avg_mae", "total_unique", "min_unique",
            "low_bias", "med_bias", "high_bias", "high_size", "pareto"]
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(df[cols].to_string(index=False))
    print("\nPareto-optimal (min bias, min MAE, max diversity):")
    print(df[df["pareto"]][["beta", "warmup", "avg_abs_bias", "avg_mae", "total_unique"]].to_string(index=False))

    df.to_csv(os.path.join(OUTDIR, "sweep_table.csv"), index=False)
    if df["beta"].nunique() > 1 and df["warmup"].nunique() > 1:
        heatmap(df, "avg_abs_bias", "avg |mean - target|  (lower=better)",
                os.path.join(OUTDIR, "heat_bias.png"), better="low")
        heatmap(df, "avg_mae", "avg MAE  (lower=better)",
                os.path.join(OUTDIR, "heat_mae.png"), better="low")
        heatmap(df, "total_unique", "total unique SMILES  (higher=better)",
                os.path.join(OUTDIR, "heat_diversity.png"), better="high")
        print(f"\nheatmaps + table -> {OUTDIR}")


if __name__ == "__main__":
    main()
