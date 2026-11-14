"""Shared matplotlib style and a save helper so every figure looks the same."""
from __future__ import annotations

import matplotlib.pyplot as plt

from .config import FIGURES

NHS_BLUE = "#005EB8"
NHS_DARK = "#003087"
NHS_LIGHT = "#41B6E6"
NHS_RED = "#DA291C"
NHS_GREY = "#768692"
LEVEL_COLOURS = {"England": NHS_DARK, "Region": NHS_BLUE, "ICB": NHS_LIGHT, "Trust": NHS_GREY}


def set_style() -> None:
    plt.rcParams.update({
        "figure.figsize": (9, 4.2), "figure.dpi": 110, "savefig.dpi": 140, "savefig.bbox": "tight",
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25,
        "axes.titleweight": "bold", "axes.titlesize": 12, "font.size": 10, "legend.frameon": False,
    })


def save(fig, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f"{name}.png")


def covid_band(ax) -> None:
    import pandas as pd
    ax.axvspan(pd.Timestamp("2020-03-15"), pd.Timestamp("2021-06-15"), color=NHS_GREY, alpha=0.12, lw=0)
