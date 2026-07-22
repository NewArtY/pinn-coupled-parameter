"""Shared plotting style and output helpers for the paper figures."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

# Allow "python paper_figures/figNN_*.py" to import the top-level modules.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

REPO_ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = Path(os.environ.get("FIG_OUTDIR", REPO_ROOT / "figures"))
DATA_DIR = Path(os.environ.get("DATA_OUTDIR", REPO_ROOT / "data"))

# Regime colours used consistently across every figure of the paper.
REGIME_COLORS = {"LT": "#2ca02c", "WNL": "#ff7f0e", "UR": "#1f77b4"}
REGIME_A0 = {"LT": 0.85, "WNL": 8.5, "UR": 85.0}
REGIME_INTENSITY = {"LT": "10^{18}", "WNL": "10^{20}", "UR": "10^{22}"}


def apply_style():
    matplotlib.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.size": 9,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "legend.fontsize": 8,
        "axes.grid": False,
        "axes.axisbelow": True,
        "lines.linewidth": 1.6,
        "mathtext.fontset": "dejavusans",
    })


def save(fig, stem: str, also_png: bool = True):
    """Write ``stem`` as PDF (and PNG) into the figure directory."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    pdf = FIG_DIR / f"{stem}.pdf"
    fig.savefig(pdf)
    if also_png:
        fig.savefig(FIG_DIR / f"{stem}.png")
    plt.close(fig)
    print(f"  wrote {pdf}")
    return pdf


def save_data(stem: str, **columns):
    """Write the numerical content of a figure as a CSV next to the figure."""
    import numpy as np

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    keys = list(columns)
    arr = np.column_stack([np.asarray(columns[k]).ravel() for k in keys])
    path = DATA_DIR / f"{stem}.csv"
    np.savetxt(path, arr, delimiter=",", header=",".join(keys),
               comments="", fmt="%.8g")
    print(f"  wrote {path}")
    return path


def panel_label(ax, text, loc=(0.03, 0.94), **kw):
    ax.text(*loc, text, transform=ax.transAxes, fontweight="bold",
            va="top", ha="left", **kw)
