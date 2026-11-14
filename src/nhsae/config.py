"""Project paths and constants shared by every module."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
FIGURES = ROOT / "docs" / "figures"

MONTHLY_BUNDLE = RAW / "nhs_ae_monthly_bundle.txt"
ICB_MAPPING = RAW / "nhs_ae_System-Mapping-Oct-26.xls"
PANEL = PROC / "trust_month_panel.csv"
HIERARCHY = PROC / "hierarchy.csv"

LEVELS = ["England", "Region", "ICB", "Trust"]
