#!/usr/bin/env python3

import argparse
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression

plt.style.use('seaborn-v0_8-paper')
CB_PALETTE = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#56B4E9', '#F0E442']


def read_csv_if_exists(path):
    if path and os.path.exists(path):
        return pd.read_csv(path)
    return None


def read_pickle_if_exists(path):
    if path and os.path.exists(path):
        with open(path, "rb") as handle:
            return pickle.load(handle)
    return None


def fig_delta_var_vs_delta_pearson(
    variance_csv="variance_raw_vs_ranked.csv",
    codabench_csv=None,
    out_png="fig_delta_var_vs_delta_pearson.png",
):
    df_var = read_csv_if_exists(variance_csv)
    if df_var is None:
        print(f"Missing variance CSV: {variance_csv}")
        return

    merged = df_var.copy()
    if "delta_var" not in merged.columns:
        merged["delta_var"] = merged["var_rank"] - merged["var_raw"]

    if codabench_csv and os.path.exists(codabench_csv):
        cb = pd.read_csv(codabench_csv)
        cb.columns = [col.strip() for col in cb.columns]
        if {"pearson_raw", "pearson_rank"}.issubset(cb.columns):
            cbm = cb[["langpair", "pearson_raw", "pearson_rank"]].copy()
            cbm["delta_pearson"] = cbm["pearson_rank"] - cbm["pearson_raw"]
            merged = merged.merge(cbm[["langpair", "delta_pearson"]], on="langpair", how="left")
        else:
            pear_cols = [col for col in cb.columns if "pearson" in col.lower()]
            if len(pear_cols) == 1:
                cbm = cb[["langpair", pear_cols[0]]].copy()
                cbm = cbm.rename(columns={pear_cols[0]: "pearson_snapshot"})
                merged = merged.merge(cbm, on="langpair", how="left")
                merged["delta_pearson"] = merged["pearson_snapshot"] - merged["pearson_raw_vs_ranked"]

    if "delta_pearson" not in merged.columns or merged["delta_pearson"].isna().all():
        merged["delta_pearson"] = 1.0 - pd.to_numeric(
            merged["pearson_raw_vs_ranked"], errors="coerce"
        )

    merged["delta_var"] = pd.to_numeric(merged["delta_var"], errors="coerce")
    merged["delta_pearson"] = pd.to_numeric(merged["delta_pearson"], errors="coerce")

    mask = merged["delta_var"].notna() & merged["delta_pearson"].notna()
    if mask.sum() < 3:
        print("Not enough data to produce Figure A")
        return

    x = merged.loc[mask, "delta_var"].to_numpy().reshape(-1, 1)
    y = merged.loc[mask, "delta_pearson"].to_numpy().reshape(-1, 1)

    model = LinearRegression().fit(x, y)
    r_value = pearsonr(x.ravel(), y.ravel())[0]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(x, y, s=30, alpha=0.85, color=CB_PALETTE[0], edgecolors="white", linewidth=0.4)
    xs = np.linspace(x.min(), x.max(), 100).reshape(-1, 1)
    ax.plot(xs, model.predict(xs), color=CB_PALETTE[1], linewidth=1.8)
    ax.set_xlabel("Variance Change (ranked − raw)")
    ax.set_ylabel("Pearson Change")
    ax.set_title(f"r = {r_value:.3f}")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_png}")


def fig_hist_pearson_raw_vs_ranked(
    raw_vs_ranked_csv="raw_vs_ranked_stats.csv",
    out_png="fig_hist_pearson_raw_vs_ranked.png",
):
    df = read_csv_if_exists(raw_vs_ranked_csv)
    if df is None:
        print(f"Missing raw_vs_ranked_stats.csv: {raw_vs_ranked_csv}")
        return

    pearson_col = None
    for candidate in ["pearson_raw_vs_ranked", "pearson_raw_vs_ranked.1", "pearson"]:
        if candidate in df.columns:
            pearson_col = candidate
            break
    if pearson_col is None:
        pear_cols = [col for col in df.columns if "pearson" in col.lower()]
        if pear_cols:
            pearson_col = pear_cols[0]
    if pearson_col is None:
        print(f"No Pearson column found in {raw_vs_ranked_csv}")
        return

    vals = pd.to_numeric(df[pearson_col], errors="coerce").dropna()
    if vals.empty:
        print("No numeric Pearson values available for Figure B")
        return

    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.hist(vals, bins=20, color=CB_PALETTE[0], edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Pearson(raw, ranked)")
    ax.set_ylabel("Language pairs")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_png}")


def fig_ties_prepost(
    segments_tsv="segments_fifth.tsv",
    raw_pkl="raw_scores_backup.pkl",
    out_png="fig_ties_prepost.png",
    top_k=30,
):
    if not os.path.exists(segments_tsv):
        print(f"Missing segments file: {segments_tsv}")
        return

    df = pd.read_csv(segments_tsv, sep="\t", dtype=str)
    df.columns = [col.strip() for col in df.columns]
    df["overall"] = pd.to_numeric(df["overall"], errors="coerce")
    df["langpair"] = (
        df["source_lang"].astype(str).str.strip()
        + "-"
        + df["target_lang"].astype(str).str.strip()
    )

    counts = df["langpair"].value_counts()
    top_langpairs = counts.index.tolist()[:top_k]

    raw_scores = read_pickle_if_exists(raw_pkl)
    use_raw = raw_scores is not None and len(raw_scores) == len(df)
    if use_raw:
        df["raw_score"] = pd.Series(raw_scores)

    rows = []
    for langpair in top_langpairs:
        subset = df[df["langpair"] == langpair]
        size = len(subset)
        if size == 0:
            continue
        ranked_vals = subset["overall"].dropna().to_numpy()
        pct_ranked = np.unique(ranked_vals).size / size
        pct_raw = np.nan
        if use_raw:
            raw_vals = pd.to_numeric(subset["raw_score"], errors="coerce").dropna().to_numpy()
            pct_raw = np.unique(raw_vals).size / size if raw_vals.size else np.nan
        rows.append(
            {
                "langpair": langpair,
                "pct_unique_rank": pct_ranked,
                "pct_unique_raw": pct_raw,
            }
        )

    if not rows:
        print("No data available for Figure C")
        return

    df_stats = pd.DataFrame(rows).sort_values("pct_unique_rank").reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(8, 10))
    if use_raw:
        ax.hlines(
            y=df_stats.index,
            xmin=df_stats["pct_unique_raw"],
            xmax=df_stats["pct_unique_rank"],
            color="grey",
            alpha=0.6,
            linewidth=1.5,
        )
        ax.scatter(
            df_stats["pct_unique_raw"],
            df_stats.index,
            color=CB_PALETTE[0],
            s=40,
            label="Raw",
        )
    ax.scatter(
        df_stats["pct_unique_rank"],
        df_stats.index,
        color=CB_PALETTE[1],
        s=40,
        label="Ranked",
    )
    ax.set_yticks(df_stats.index)
    ax.set_yticklabels(df_stats["langpair"], fontsize=10)
    ax.set_xlabel("Fraction of unique predictions")
    ax.grid(axis="x", linestyle="--", alpha=0.5)
    ax.set_xlim(0, 1.02)
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_png}")


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--fig", choices=["A", "B", "C", "all"], default="all")
    parser.add_argument("--variance", default="variance_raw_vs_ranked.csv")
    parser.add_argument("--codabench", default=None)
    parser.add_argument("--raw_vs_ranked", default="raw_vs_ranked_stats.csv")
    parser.add_argument("--segments", default="segments_fifth.tsv")
    parser.add_argument("--raw_pkl", default="raw_scores_backup.pkl")
    parser.add_argument("--outdir", default="figures_camera_ready")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    if args.fig in ("A", "all"):
        fig_delta_var_vs_delta_pearson(
            variance_csv=args.variance,
            codabench_csv=args.codabench,
            out_png=os.path.join(args.outdir, "fig_A_delta_var_vs_pearson.png"),
        )
    if args.fig in ("B", "all"):
        fig_hist_pearson_raw_vs_ranked(
            raw_vs_ranked_csv=args.raw_vs_ranked,
            out_png=os.path.join(args.outdir, "fig_B_hist_pearson.png"),
        )
    if args.fig in ("C", "all"):
        fig_ties_prepost(
            segments_tsv=args.segments,
            raw_pkl=args.raw_pkl,
            out_png=os.path.join(args.outdir, "fig_C_unique_fractions.png"),
        )


if __name__ == "__main__":
    main()
