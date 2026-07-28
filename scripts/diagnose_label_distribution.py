#!/usr/bin/env python
"""Diagnose ODIAC label distributions and NTL-ODIAC coupling per city.

Reads the grid-month panel only; no model artifacts required. Outputs CSV
tables and figures under reports/stage1/diagnostics/.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kurtosis, pearsonr, skew, spearmanr

ROOT = Path(__file__).resolve().parents[1]

CITY_ORDER = ["chicago", "nyc", "singapore", "tokyo"]
CITY_LABELS = {"chicago": "Chicago", "nyc": "NYC", "singapore": "Singapore", "tokyo": "Tokyo"}
# Okabe-Ito, colorblind-safe, fixed assignment per city.
CITY_COLORS = {"chicago": "#0072B2", "nyc": "#E69F00", "singapore": "#009E73", "tokyo": "#CC79A7"}

CORRELATION_FEATURES = {
    "log1p_ntl_sum": lambda df: np.log1p(df["ntl_radiance_sum"]),
    "ntl_log_radiance_mean": lambda df: df["ntl_log_radiance_mean"],
    "log1p_poi_total": lambda df: np.log1p(
        df[[c for c in df.columns if c.startswith("poi_")]].sum(axis=1)
    ),
    "modis_ndvi_mean": lambda df: df["modis_ndvi_mean"],
}


def gini(values: np.ndarray) -> float:
    values = np.sort(np.asarray(values, dtype=float))
    if values.sum() <= 0:
        return float("nan")
    n = len(values)
    ranks = np.arange(1, n + 1)
    return float((2 * (ranks * values).sum()) / (n * values.sum()) - (n + 1) / n)


def label_stats(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for city in CITY_ORDER:
        sub = panel[panel["city_id"] == city]
        y = sub["log1p_emission"].to_numpy(dtype=float)
        tc = sub["emission_tc"].to_numpy(dtype=float)
        grid_mean = sub.groupby("cell_id")["emission_tc"].mean().sort_values(ascending=False)
        total = grid_mean.sum()
        top1 = max(1, int(round(len(grid_mean) * 0.01)))
        rows.append({
            "city": city,
            "grids": sub["cell_id"].nunique(),
            "samples": len(sub),
            "zero_share": float((tc == 0).mean()),
            "log_mean": float(y.mean()),
            "log_std": float(y.std(ddof=0)),
            "log_skew": float(skew(y)),
            "log_kurtosis": float(kurtosis(y)),
            "log_p50": float(np.percentile(y, 50)),
            "log_p90": float(np.percentile(y, 90)),
            "log_p99": float(np.percentile(y, 99)),
            "log_max": float(y.max()),
            "tc_month_city_total_mean": float(sub.groupby("period")["emission_tc"].sum().mean()),
            "top1pct_grid_share": float(grid_mean.head(top1).sum() / total),
            "top10_grid_share": float(grid_mean.head(10).sum() / total),
            "gini_grid_mean": gini(grid_mean.to_numpy()),
        })
    return pd.DataFrame(rows)


def monthly_correlations(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for city in CITY_ORDER:
        sub = panel[panel["city_id"] == city]
        for name, fn in CORRELATION_FEATURES.items():
            feature = fn(sub)
            frame = pd.DataFrame({
                "period": sub["period"].to_numpy(),
                "x": feature.to_numpy(dtype=float),
                "y": sub["log1p_emission"].to_numpy(dtype=float),
            }).dropna()
            spearmans, pearsons = [], []
            for _, group in frame.groupby("period"):
                if group["x"].nunique() < 2 or group["y"].nunique() < 2:
                    continue
                spearmans.append(spearmanr(group["x"], group["y"]).statistic)
                pearsons.append(pearsonr(group["x"], group["y"]).statistic)
            rows.append({
                "city": city,
                "feature": name,
                "months": len(spearmans),
                "spearman_mean": float(np.mean(spearmans)),
                "spearman_std": float(np.std(spearmans, ddof=1)),
                "spearman_min": float(np.min(spearmans)),
                "pearson_mean": float(np.mean(pearsons)),
                "pearson_std": float(np.std(pearsons, ddof=1)),
            })
    return pd.DataFrame(rows)


def monthly_ntl_series(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for city in CITY_ORDER:
        sub = panel[panel["city_id"] == city]
        x = np.log1p(sub["ntl_radiance_sum"])
        frame = pd.DataFrame({"period": sub["period"], "x": x, "y": sub["log1p_emission"]}).dropna()
        for period, group in frame.groupby("period"):
            if group["x"].nunique() < 2:
                continue
            rows.append({
                "city": city, "period": str(period),
                "spearman": float(spearmanr(group["x"], group["y"]).statistic),
            })
    return pd.DataFrame(rows)


def singapore_top_grids(panel: pd.DataFrame, top: int = 15) -> pd.DataFrame:
    sub = panel[panel["city_id"] == "singapore"]
    grid = (
        sub.groupby(["cell_id", "admin_name"], dropna=False)
        .agg(emission_tc_mean=("emission_tc", "mean"), lon=("lon", "first"), lat=("lat", "first"))
        .reset_index()
        .sort_values("emission_tc_mean", ascending=False)
    )
    total = grid["emission_tc_mean"].sum()
    grid["share_of_city_total"] = grid["emission_tc_mean"] / total
    grid["cumulative_share"] = grid["share_of_city_total"].cumsum()
    return grid.head(top)


def plot_histograms(panel: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    bins = np.linspace(0, float(panel["log1p_emission"].max()) + 0.5, 60)
    for ax, city in zip(axes.ravel(), CITY_ORDER):
        y = panel.loc[panel["city_id"] == city, "log1p_emission"]
        ax.hist(y, bins=bins, color=CITY_COLORS[city], edgecolor="white", linewidth=0.3)
        ax.set_title(f"{CITY_LABELS[city]}  (std={y.std(ddof=0):.2f}, skew={skew(y):.2f})", fontsize=11)
        ax.set_yscale("log")
        ax.grid(axis="y", alpha=0.25, linewidth=0.5)
        ax.spines[["top", "right"]].set_visible(False)
    for ax in axes[1]:
        ax.set_xlabel("log1p(emission_tc)")
    for ax in axes[:, 0]:
        ax.set_ylabel("grid-months (log scale)")
    fig.suptitle("ODIAC label distribution by city, 2021-01 to 2023-12", fontsize=13)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_ntl_scatter(panel: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for ax, city in zip(axes.ravel(), CITY_ORDER):
        sub = panel[panel["city_id"] == city]
        grid = sub.groupby("cell_id").agg(
            ntl=("ntl_radiance_sum", "mean"), tc=("emission_tc", "mean")
        )
        x, y = np.log1p(grid["ntl"]), np.log1p(grid["tc"])
        rho = spearmanr(x, y).statistic
        ax.scatter(x, y, s=8, alpha=0.35, color=CITY_COLORS[city], edgecolors="none")
        ax.set_title(f"{CITY_LABELS[city]}  (Spearman={rho:.3f})", fontsize=11)
        ax.set_xlabel("log1p(NTL radiance sum), grid mean")
        ax.set_ylabel("log1p(emission_tc), grid mean")
        ax.grid(alpha=0.25, linewidth=0.5)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("NTL vs ODIAC coupling (grid-level 36-month means)", fontsize=13)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_monthly_spearman(series: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.5))
    periods = sorted(series["period"].unique())
    positions = {p: i for i, p in enumerate(periods)}
    for city in CITY_ORDER:
        sub = series[series["city"] == city].sort_values("period")
        ax.plot(
            [positions[p] for p in sub["period"]], sub["spearman"],
            color=CITY_COLORS[city], linewidth=2, label=CITY_LABELS[city],
        )
        last = sub.iloc[-1]
        ax.annotate(
            CITY_LABELS[city], (positions[last["period"]], last["spearman"]),
            xytext=(6, 0), textcoords="offset points",
            color=CITY_COLORS[city], fontsize=9, va="center",
        )
    ticks = [i for i, p in enumerate(periods) if p.endswith(("01", "07"))]
    ax.set_xticks(ticks)
    ax.set_xticklabels([periods[i] for i in ticks], fontsize=8)
    ax.set_xlim(0, len(periods) + 4)
    ax.set_ylabel("Spearman(log NTL sum, log ODIAC)")
    ax.set_title("Monthly within-city NTL-ODIAC rank correlation", fontsize=12)
    ax.grid(axis="y", alpha=0.25, linewidth=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=9, ncol=4, loc="lower left")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", default="data/features/grid_month_panel.parquet")
    parser.add_argument("--output-dir", default="reports/stage1/diagnostics")
    args = parser.parse_args()

    out_dir = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    panel = pd.read_parquet(ROOT / args.panel)

    stats = label_stats(panel)
    stats.to_csv(out_dir / "label_stats.csv", index=False)
    correlations = monthly_correlations(panel)
    correlations.to_csv(out_dir / "feature_label_correlations.csv", index=False)
    series = monthly_ntl_series(panel)
    series.to_csv(out_dir / "ntl_spearman_monthly.csv", index=False)
    top_grids = singapore_top_grids(panel)
    top_grids.to_csv(out_dir / "singapore_top_grids.csv", index=False)

    plot_histograms(panel, out_dir / "label_histograms.png")
    plot_ntl_scatter(panel, out_dir / "ntl_odiac_scatter.png")
    plot_monthly_spearman(series, out_dir / "ntl_spearman_monthly.png")

    print(stats.to_string(index=False))
    print()
    print(correlations.to_string(index=False))
    print()
    print(top_grids.to_string(index=False))
    print(f"\nOutputs written to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
