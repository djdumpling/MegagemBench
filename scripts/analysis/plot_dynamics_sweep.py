#!/usr/bin/env python3
"""Render the recovered E1 dynamics-simulator sweep."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Patch, Rectangle

from megagem.assets import asset_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(asset_path("dynamics_sim_sweep_recovered_200.json")))
    ap.add_argument("--output", default="results/analysis/dynamics_sim_sweep_recovered_200.png")
    args = ap.parse_args()

    data = json.loads(Path(args.input).read_text())
    sns.set_theme(style="white", context="notebook", font_scale=1.1)
    reports = data["reports"]
    lambdas = [0.0, 0.5, 1.0, 1.6, 2.5]
    deltas = [0.0, 2.0, 5.0]
    main_grid = {(r["lam"], r["delta"]): r for r in reports if r["gate_min"] == 1.0}
    gate_variants = [r for r in reports if r["gate_min"] != 1.0]

    values = np.array([[main_grid[(lam, delta)]["paired_margin"]["mean"]
                        for lam in lambdas] for delta in deltas])
    ses = np.array([[main_grid[(lam, delta)]["paired_margin"]["se"]
                     for lam in lambdas] for delta in deltas])

    fig, (ax, gate_ax) = plt.subplots(1, 2, figsize=(13, 5.8),
                                      gridspec_kw={"width_ratios": [1.65, 1]})
    labels = np.array([[f"{values[row, col]:+.1f}\n±{ses[row, col]:.1f}"
                        for col in range(len(lambdas))] for row in range(len(deltas))])
    sns.heatmap(values, ax=ax, cmap="RdYlGn", vmin=-8, vmax=28,
                annot=labels, fmt="", annot_kws={"fontsize": 10, "fontweight": "semibold"},
                xticklabels=[str(x) for x in lambdas], yticklabels=[str(x) for x in deltas],
                cbar_kws={"label": "paired Δmargin"}, linewidths=0.8, linecolor="white")
    ax.set_xlabel("Pacing λ")
    ax.set_ylabel("V̂ de-bias δ")
    ax.set_title("Gate = 1.0: paired Δmargin")
    # The historical interpretation rejects λ>=1.0: the fitted-law proxy starts
    # vetoing its own sampled overbids, yielding an unrealistically large
    # deviation footprint that the live LLM does not have.
    for col, lam in enumerate(lambdas):
        if lam >= 1.0:
            for row in range(len(deltas)):
                ax.add_patch(Rectangle((col, row), 1, 1,
                                       facecolor="none", edgecolor="#252525",
                                       hatch="///", linewidth=0.0, alpha=0.6))
    ax.legend(handles=[Patch(facecolor="white", edgecolor="#252525", hatch="///",
                             label="λ ≥ 1.0: proxy-instability region (exclude)")],
              loc="upper center", bbox_to_anchor=(0.5, -0.16), frameon=False, fontsize=9)
    labels = [f"λ={r['lam']:g}, δ={r['delta']:g}\ngate={r['gate_min']:g}"
              for r in gate_variants]
    vals = [r["paired_margin"]["mean"] for r in gate_variants]
    errs = [r["paired_margin"]["se"] for r in gate_variants]
    colors = ["#4f8f5b" if x >= 0 else "#bb4e4e" for x in vals]
    bars = gate_ax.bar(range(len(vals)), vals, yerr=errs, capsize=5, color=colors)
    gate_ax.axhline(0, color="#333", linewidth=0.8)
    gate_ax.set_xticks(range(len(vals)), labels, fontsize=9)
    gate_ax.set_ylabel("paired Δmargin")
    gate_ax.set_title("Gate-only variants")
    for bar, val in zip(bars, vals):
        gate_ax.text(bar.get_x() + bar.get_width() / 2, val + (0.7 if val >= 0 else -0.7),
                     f"{val:+.1f}", ha="center", va="bottom" if val >= 0 else "top",
                     fontweight="semibold")

    fig.suptitle("Recovered E1 dynamics-simulator sweep (200 paired seeds)",
                 fontsize=14, fontweight="bold", y=0.98)
    fig.text(0.5, 0.01,
             "Simulation result; use the documented credible-region / live-confirmation criteria when interpreting it.",
             ha="center", color="#555", fontsize=9)
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180, bbox_inches="tight")
    print(out)


if __name__ == "__main__":
    main()
