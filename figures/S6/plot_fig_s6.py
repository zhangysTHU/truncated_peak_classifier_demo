"""Replot Supplementary Fig. S6B-F from the recorded, unchanged source tables."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path(__file__).resolve().parent / "source_data"
ABLATIONS = ROOT / "ablations" / "outputs"
COLORS = ["#2B6CB0", "#4C9A7A", "#7C8A35", "#B87C3B", "#9A5B6F"]
GROUP_COLORS = dict(zip(
    ["Full model", "Coverage", "Modification", "Sequence", "Structure"],
    [COLORS[0], COLORS[3], COLORS[1], COLORS[2], COLORS[4]],
))
SERIES = {
    "C": ("DL_ablation_posencoded_results.csv", "attention_tailfirst_ablation_feature_map.csv",
          "Attention-model tail-first feature ablation", "FigS6C_attention_tailfirst", True),
    "D": ("cross_ablation_reveresd_results.csv", "attention_reverse_ablation_feature_map.csv",
          "Attention-model reverse feature ablation", "FigS6D_attention_reverse", True),
    "E": ("LR_ablation_results.csv", "lr_tailfirst_ablation_feature_map.csv",
          "Logistic-regression tail-first feature ablation", "FigS6E_lr_tailfirst", False),
    "F": ("LR_ablation_reveresd_results.csv", "lr_reverse_ablation_feature_map.csv",
          "Logistic-regression reverse feature ablation", "FigS6F_lr_reverse", False),
}


def apply_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
        "svg.fonttype": "none", "pdf.fonttype": 42, "font.size": 10,
        "text.color": "#1F1F1F", "axes.labelcolor": "#1F1F1F",
        "xtick.color": "#1F1F1F", "ytick.color": "#1F1F1F",
        "axes.spines.right": False, "axes.spines.top": False,
        "axes.linewidth": 1.2, "legend.frameon": False,
    })


def style_axes(ax):
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.8)
    ax.spines["left"].set_linewidth(1.2)
    ax.spines["bottom"].set_linewidth(1.2)
    ax.tick_params(width=1.1, length=4)


def load_comparison():
    data = pd.read_csv(SOURCE / "reviewer11_model_comparison_metrics.csv")
    expected = ["Attention DL", "LightGBM", "Random Forest", "SVM", "Logistic Regression"]
    if data["Model"].tolist() != expected:
        raise ValueError("Model-comparison rows must follow the five recorded models.")
    values = data["AUROC"].to_numpy(float)
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("Invalid AUROC values.")
    return data


def load_ablation(panel):
    results, mapping, _, _, _ = SERIES[panel]
    data = pd.read_csv(ABLATIONS / results)
    feature_map = pd.read_csv(SOURCE / mapping)
    expected = list(range(640, 0, -40))
    if data["features_num"].tolist() != expected or feature_map["features_num"].tolist() != expected:
        raise ValueError(f"Panel {panel} must contain all 16 feature levels in order.")
    data = data.merge(feature_map, on="features_num", validate="one_to_one", how="left")
    if data["newly_removed"].isna().any() or not set(data["feature_group"]).issubset(GROUP_COLORS):
        raise ValueError(f"Incomplete feature mapping for panel {panel}.")
    values = data[["mean_auroc", "auroc_variance"]].to_numpy(float)
    if not np.isfinite(values).all() or (values[:, 1] < 0).any():
        raise ValueError(f"Invalid mean/variance for panel {panel}.")
    if ((values[:, 0] < 0) | (values[:, 0] > 1)).any():
        raise ValueError(f"AUROC outside [0, 1] for panel {panel}.")
    data["auroc_sd"] = np.sqrt(data["auroc_variance"])
    return data


def plot_comparison():
    data = load_comparison()
    values = data["AUROC"].to_numpy(float)
    labels = ["Attention\nDL", "LightGBM", "Random\nForest", "SVM", "Logistic\nRegression"]
    fig, ax = plt.subplots(figsize=(5.6, 4.45))
    bars = ax.bar(np.arange(5), values, width=0.68, color=COLORS,
                  edgecolor="black", linewidth=0.8, zorder=3)
    ax.set_ylim(0.68, 0.90)
    ax.set_yticks(np.arange(0.70, 0.901, 0.025))
    ax.set_xticks(np.arange(5), labels, fontsize=9)
    ax.set_ylabel("AUROC on held-out test set", fontsize=11)
    ax.set_title("Model selection by AUROC", fontsize=14, pad=11)
    style_axes(ax)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.004, f"{value:.3f}",
                ha="center", va="bottom", fontsize=10)
    fig.tight_layout(pad=1.2)
    return fig


def plot_ablation(panel):
    data = load_ablation(panel)
    _, _, title, _, attention = SERIES[panel]
    if attention:
        counts = data["retained_channels"].astype(int)
        suffix = "ch"
    else:
        counts = data["features_num"].astype(int)
        suffix = "features"
    labels = [
        f"Full model\n({count} {suffix})" if removed == "Full model"
        else f"−{removed}\n({count} {suffix})"
        for removed, count in zip(data["newly_removed"], counts)
    ]
    fig, ax = plt.subplots(figsize=(9.7, 5.2))
    x = np.arange(len(data))
    ax.bar(x, data["mean_auroc"], yerr=data["auroc_sd"], width=0.72,
           color=[GROUP_COLORS[g] for g in data["feature_group"]], alpha=0.88,
           edgecolor="black", linewidth=0.7, capsize=2.5,
           error_kw={"elinewidth": 0.9, "capthick": 0.9, "ecolor": "black"}, zorder=3)
    ax.set_ylim((0.45, 0.92) if attention else (0.55, 0.85))
    ax.set_yticks(np.arange(0.50, 0.901, 0.05) if attention else np.arange(0.55, 0.851, 0.05))
    ax.set_xticks(x, labels, rotation=52, ha="right", fontsize=7.5)
    ax.set_ylabel("Mean AUROC (5-fold CV)", fontsize=11)
    style_axes(ax)
    groups = [g for g in GROUP_COLORS if g in set(data["feature_group"])]
    handles = [Patch(facecolor=GROUP_COLORS[g], edgecolor="black", linewidth=0.6, label=g)
               for g in groups]
    fig.suptitle(title, fontsize=14, y=0.985)
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.885),
               ncol=len(groups), fontsize=12, handlelength=1.1, columnspacing=1.25)
    fig.tight_layout(pad=1.2)
    fig.subplots_adjust(top=0.79)
    return fig


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panels", nargs="+", choices=list("BCDEF"), default=list("BCDEF"))
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "output")
    parser.add_argument("--formats", nargs="+", choices=["pdf", "svg", "png"], default=["pdf", "svg"])
    parser.add_argument("--dpi", type=int, default=600)
    args = parser.parse_args(argv)
    if args.dpi <= 0:
        parser.error("--dpi must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    apply_style()
    for panel in dict.fromkeys(args.panels):
        fig = plot_comparison() if panel == "B" else plot_ablation(panel)
        stem = "FigS6B_model_comparison" if panel == "B" else SERIES[panel][3]
        for fmt in dict.fromkeys(args.formats):
            destination = args.output_dir / f"{stem}.{fmt}"
            fig.savefig(destination, dpi=args.dpi, bbox_inches="tight")
            print(destination)
        plt.close(fig)


if __name__ == "__main__":
    main()
