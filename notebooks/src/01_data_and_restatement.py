# %% [markdown]
# # 01 · The NHS A&E data, and restating eight years of history onto today's NHS
#
# **Question:** what is in NHS England's monthly A&E files, and what has to be fixed before
# anything can be forecast?
#
# **Source:** NHS England, *A&E Attendances and Emergency Admissions* monthly situation reports
# (Accredited Official Statistics, Open Government Licence v3.0). One CSV per month, April 2018 to
# October 2026, plus the October 2026 provider-to-ICB mapping.
#
# Three things break a naive model:
# 1. **file formats drift** (column names change across years)
# 2. **trusts merge** (21 Type 1 providers disappear)
# 3. **trusts change how they record Type 1 attendances** when they open or close a co-located
#    urgent treatment centre (32 recording changes at 25 trusts)

# %%
import matplotlib.pyplot as plt
import pandas as pd

from nhsae import viz
from nhsae.config import PANEL, PROC
from nhsae.ingest import load_icb_mapping, load_monthly
from nhsae.panel import MERGERS

viz.set_style()
pd.set_option("display.width", 140)
raw = load_monthly()
print(f"{len(raw):,} provider-months, {raw.month.nunique()} months "
      f"({raw.month.min():%b %Y} – {raw.month.max():%b %Y}), {raw.org_code.nunique()} organisation codes")

# %% [markdown]
# ## 1. What a provider-month looks like
# Every A&E provider reports attendances by department type, breaches of the 4-hour standard and
# emergency admissions. **Type 1** = consultant-led 24-hour major emergency departments, the focus here.
# Type 2 = single-specialty (eye, dental); "Other" = urgent treatment centres, walk-in centres, minor injury units.

# %%
cols = ["month", "org_code", "org_name", "att_type1", "att_other", "over4h_type1", "emadm_type1"]
raw.loc[raw.month == raw.month.max(), cols].sort_values("att_type1", ascending=False).head(5)

# %% [markdown]
# The loader (`nhsae.ingest`) maps every historical header variant (for example `A&E attendances Type 1` vs
# `Number of A&E attendances Type 1`) onto one schema, drops TOTAL rows, and runs a **data contract**:
# no missing months, no duplicate provider-months, no negative counts.

# %%
by_type = raw.groupby("month")[["att_type1", "att_type2", "att_other"]].sum()
share = by_type.div(by_type.sum(axis=1), axis=0)
print("Share of all A&E attendances, latest month:")
print(share.iloc[-1].round(3).to_string())

# %% [markdown]
# ## 2. Mergers: restating history on today's trusts
# 21 Type 1 providers stopped reporting under their old code because they merged. Each is mapped to
# its successor. The check is that predecessor + successor is **continuous** across the merger month.

# %%
m = pd.DataFrame([{"retired": k, "successor": v[0], "from": v[1], "note": v[2]} for k, v in MERGERS.items()])
m.head(8)

# %%
t1 = raw.pivot_table(index="month", columns="org_code", values="att_type1", aggfunc="sum").fillna(0)
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=False)
for ax, (old, new, title) in zip(axes, [("RD3", "R0D", "Poole + Bournemouth → UH Dorset (Oct 2020)"),
                                        ("RVJ", "RA7", "North Bristol → Bristol NHS FT (Jul 2026)")]):
    olds = [k for k, v in MERGERS.items() if v[0] == new and (k == old or new == "R0D")]
    win = t1.loc[pd.Timestamp(MERGERS[old][1]) - pd.DateOffset(months=14): pd.Timestamp(MERGERS[old][1]) + pd.DateOffset(months=6)]
    win[olds + [new]].plot.area(ax=ax, alpha=0.75, color=[viz.NHS_LIGHT, viz.NHS_GREY, viz.NHS_BLUE][: len(olds) + 1], lw=0)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("")
    ax.set_ylabel("Type 1 attendances / month")
fig.suptitle("Mergers: the old codes stop, the combined series is continuous", fontweight="bold")
viz.save(fig, "01_mergers")

# %% [markdown]
# ## 3. Recording changes: the problem the data contract found
# After the mergers were fixed, a contract check (no trust's Type 1 daily rate should jump more than 60%
# in a month) still failed for two trusts. They hadn't merged. They had **reclassified attendances**:
# patients streamed to a co-located urgent treatment centre move from "Type 1" to "Other".
# Total activity barely moves; the Type 1 series steps.
#
# `nhsae.reclass.detect` finds every such step: Type 1 moves by more than 15%, "Other" moves the opposite
# way, and the two changes largely cancel.

# %%
events = pd.read_csv(PROC / "type1_reclassifications.csv", parse_dates=["month"])
print(f"{len(events)} recording changes at {events.trust_code.nunique()} trusts")
events[["trust_name", "month", "t1_change", "share_pre", "share_post", "factor"]].round(3).head(10)

# %%
rjl = raw[raw.org_code == "RJL"].set_index("month")[["att_type1", "att_other"]].loc["2023-06":"2025-03"]
rjl = rjl.div(rjl.index.days_in_month, axis=0)
panel = pd.read_csv(PANEL, parse_dates=["month"])
rest = panel[panel.trust_code == "RJL"].set_index("month").attendances.loc["2023-06":"2025-03"]
fig, ax = plt.subplots(figsize=(9, 3.8))
ax.plot(rjl.index, rjl.att_type1, color=viz.NHS_BLUE, lw=2, label="Type 1 as reported")
ax.plot(rjl.index, rjl.att_other, color=viz.NHS_LIGHT, lw=2, label="Other A&E (UTC) as reported")
ax.plot(rjl.index, rjl.att_type1 + rjl.att_other, color=viz.NHS_GREY, ls=":", label="total")
ax.plot(rest.index, rest / rest.index.days_in_month, color=viz.NHS_DARK, ls="--", lw=2,
        label="Type 1 restated on current basis")
ax.axvline(pd.Timestamp("2024-05-01"), color=viz.NHS_RED, lw=1)
ax.set_title("Northern Lincolnshire & Goole: ~210 patients/day moved from Type 1 to the UTC in May 2024")
ax.set_ylabel("attendances per day")
ax.legend(ncol=4, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.1))
viz.save(fig, "01_reclassification")

# %% [markdown]
# Without the restatement, a model sees a 42% collapse in demand at this trust and forecasts it to keep
# falling. With it, the series is continuous. The 4-hour model keeps the **reported** figures, because
# breaches and attendances must be on the same basis to give a valid percentage.

# %% [markdown]
# ## 4. The hierarchy
# 120 trusts with a Type 1 department in October 2026 and at least 3 years of history, mapped to
# today's 36 ICBs and 7 NHS England regions.

# %%
icb = load_icb_mapping()
hier = pd.read_csv(PROC / "hierarchy.csv")
summary = hier.groupby("region_name").agg(ICBs=("icb_code", "nunique"), trusts=("trust_code", "size"))
summary.loc["England"] = summary.sum()
summary

# %%
eng = panel.groupby("month")[["attendances", "attendances_reported"]].sum()
print("Restatement changes the England total by "
      f"{(eng.attendances / eng.attendances_reported - 1).abs().max():.1%} at most in any month; "
      "it matters at trust level, not nationally.")
