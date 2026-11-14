"""Small result tables for the README and dashboard (keeps the large backtest parquet files out of the app)."""
from __future__ import annotations

import pandas as pd

from .config import RESULTS

LEVELS = ["England", "Region", "ICB", "Trust"]


def main():
    for measure in ["attendances", "emergency_admissions"]:
        frames = []
        for f in sorted(RESULTS.glob(f"backtest_{measure}_*.parquet")):
            covid = f.stem.rsplit("_", 1)[-1]
            b = pd.read_parquet(f)
            hz = (b.groupby(["model", "recon", "level", "h"])
                  .agg(MASE=("ase", "mean"), MAPE=("ape", "mean"), CRPS=("crps_scaled", "mean")).reset_index())
            hz["covid"] = covid
            frames.append(hz)
        if not frames:
            continue
        hz = pd.concat(frames)
        hz.to_csv(RESULTS / f"backtest_by_horizon_{measure}.csv", index=False)

        summ = pd.concat([pd.read_csv(f).assign(covid=f.stem.rsplit("_", 1)[-1])
                          for f in RESULTS.glob(f"backtest_summary_{measure}_*.csv")])
        summ.to_csv(RESULTS / f"backtest_summary_{measure}.csv", index=False)
        piv = summ.pivot_table(index=["covid", "model", "recon"], columns="level", values="MASE")[LEVELS]
        piv["Average"] = piv.mean(axis=1)
        print(f"\n== {measure}: top 8 by average MASE")
        print(piv.sort_values("Average").head(8).round(3).to_string())
        print(f"\n== {measure}: best reconciliation per model & COVID treatment")
        print(piv.reset_index().sort_values("Average").groupby(["covid", "model"]).head(1)
              .set_index(["covid", "model"]).round(3).to_string())


if __name__ == "__main__":
    main()
