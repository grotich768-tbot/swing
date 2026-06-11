"""
monthly_breakdown.py  —  Monthly PnL breakdown for all symbols and years
Run from your project root:  python monthly_breakdown.py
"""
import sys
import os
sys.path.insert(0, ".")

import pandas as pd
import numpy as np
from pathlib import Path

RESULTS_DIR = Path("results")
SYMBOLS     = ["GOLD", "SILVER", "EURUSD", "GBPUSD"]
YEARS       = [2022, 2023, 2024, 2025]


def load_tradelog(symbol, year):
    path = RESULTS_DIR / f"tradelog_{symbol}_{year}-01-01_{year}-12-31.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    return df


def monthly_summary(symbol, year):
    df = load_tradelog(symbol, year)
    if df is None:
        return None
    monthly = df["pnl_usd"].resample("ME").sum()
    return monthly


# ─────────────────────────────────────────────────────────────────────────────
# Per-symbol monthly breakdown
# ─────────────────────────────────────────────────────────────────────────────
for sym in SYMBOLS:
    print(f"\n{'='*55}")
    print(f"  {sym}  —  Monthly PnL by Year")
    print(f"{'='*55}")
    print(f"  {'Month':<6}", end="")
    for yr in YEARS:
        print(f"  {yr:>10}", end="")
    print()
    print(f"  {'──────':<6}", end="")
    for _ in YEARS:
        print(f"  {'──────────':>10}", end="")
    print()

    month_names = ["Jan","Feb","Mar","Apr","May","Jun",
                   "Jul","Aug","Sep","Oct","Nov","Dec"]

    yearly_data = {}
    for yr in YEARS:
        m = monthly_summary(sym, yr)
        if m is not None:
            # Build month → pnl dict
            yearly_data[yr] = {d.month: v for d, v in m.items()}

    for mo_num, mo_name in enumerate(month_names, 1):
        print(f"  {mo_name:<6}", end="")
        for yr in YEARS:
            pnl = yearly_data.get(yr, {}).get(mo_num, None)
            if pnl is None:
                print(f"  {'—':>10}", end="")
            else:
                sign  = "+" if pnl >= 0 else ""
                col   = "" if pnl >= 0 else ""
                print(f"  {sign}${pnl:>8,.0f}", end="")
        print()

    # Totals row
    print(f"  {'──────':<6}", end="")
    for _ in YEARS:
        print(f"  {'──────────':>10}", end="")
    print()
    print(f"  {'TOTAL':<6}", end="")
    for yr in YEARS:
        m = monthly_summary(sym, yr)
        if m is not None:
            total = m.sum()
            sign  = "+" if total >= 0 else ""
            print(f"  {sign}${total:>8,.0f}", end="")
        else:
            print(f"  {'N/A':>10}", end="")
    print()

    # Win rate row
    print(f"  {'GREEN':<6}", end="")
    for yr in YEARS:
        m = monthly_summary(sym, yr)
        if m is not None:
            wins = (m > 0).sum()
            total_mo = len(m)
            print(f"  {wins:>3}/{total_mo} months", end="")
        else:
            print(f"  {'N/A':>10}", end="")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Portfolio monthly totals (all 4 symbols combined)
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'='*55}")
print(f"  PORTFOLIO  —  Combined Monthly PnL (all 4 symbols)")
print(f"{'='*55}")
print(f"  {'Month':<6}", end="")
for yr in YEARS:
    print(f"  {yr:>10}", end="")
print()
print(f"  {'──────':<6}", end="")
for _ in YEARS:
    print(f"  {'──────────':>10}", end="")
print()

for mo_num, mo_name in enumerate(month_names, 1):
    print(f"  {mo_name:<6}", end="")
    for yr in YEARS:
        total = 0.0
        found = False
        for sym in SYMBOLS:
            m = monthly_summary(sym, yr)
            if m is not None:
                pnl = {d.month: v for d, v in m.items()}.get(mo_num, 0.0)
                total += pnl
                found = True
        if found:
            sign = "+" if total >= 0 else ""
            mark = "▲" if total >= 0 else "▼"
            print(f"  {sign}${total:>8,.0f}", end="")
        else:
            print(f"  {'—':>10}", end="")
    print()

# Portfolio totals
print(f"  {'──────':<6}", end="")
for _ in YEARS:
    print(f"  {'──────────':>10}", end="")
print()
print(f"  {'TOTAL':<6}", end="")
yr_totals = []
for yr in YEARS:
    total = 0.0
    for sym in SYMBOLS:
        m = monthly_summary(sym, yr)
        if m is not None:
            total += m.sum()
    sign = "+" if total >= 0 else ""
    print(f"  {sign}${total:>8,.0f}", end="")
    yr_totals.append(total)
print()

print(f"\n  4-year average : ${np.mean(yr_totals):>10,.0f}/year")
print(f"  Best year      : ${max(yr_totals):>10,.0f}")
print(f"  Worst year     : ${min(yr_totals):>10,.0f}")
print(f"  Year variance  : ${np.std(yr_totals):>10,.0f} std\n")
