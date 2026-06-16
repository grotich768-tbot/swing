"""
session_backtest.py  —  Session-by-Session Performance Analyser
──────────────────────────────────────────────────────────────────────────────
Loads trade log CSVs and breaks performance down by:
  • Trading session  (Asian / London / NY Overlap / New York / Dead Zone)
  • Hour of day      (24-hour heatmap)
  • Day of week      (Mon–Fri consistency)
  • Flip quality     (which session produces the best entry points)

Sessions (UTC):
  Asian      :  22:00 – 07:00   (overnight liquidity, thin)
  London     :  07:00 – 12:00   (high liquidity, trending)
  Overlap    :  12:00 – 16:00   (highest volatility, London+NY active)
  New York   :  16:00 – 21:00   (good liquidity, USD-driven)
  Dead Zone  :  21:00 – 22:00   (pre-Asian, very thin)

Usage:
    python session_backtest.py
    python session_backtest.py --years 2024 2025
    python session_backtest.py --symbol GOLD
    python session_backtest.py --show
"""

import sys
import os
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
from matplotlib.colors import TwoSlopeNorm

sys.path.insert(0, ".")
from config import SYMBOLS, RESULT_DIR

# ─────────────────────────────────────────────────────────────────────────────
# Session definitions (UTC hours)
# ─────────────────────────────────────────────────────────────────────────────
SESSIONS = {
    "Asian":     (22, 7),    # wraps midnight
    "London":    (7,  12),
    "Overlap":   (12, 16),
    "New York":  (16, 21),
    "Dead Zone": (21, 22),
}

SESSION_COLOURS = {
    "Asian":     "#4A90D9",
    "London":    "#F39C12",
    "Overlap":   "#E74C3C",
    "New York":  "#2ECC71",
    "Dead Zone": "#95A5A6",
}

DAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


# ─────────────────────────────────────────────────────────────────────────────
# Session tagger
# ─────────────────────────────────────────────────────────────────────────────
def tag_session(hour: int) -> str:
    """Return session name for a UTC hour (0–23)."""
    if 7  <= hour < 12: return "London"
    if 12 <= hour < 16: return "Overlap"
    if 16 <= hour < 21: return "New York"
    if 21 <= hour < 22: return "Dead Zone"
    return "Asian"   # 22–24 and 0–7


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────
def load_logs(symbols: list, years: list) -> dict:
    """Returns {symbol: combined_DataFrame} for all years merged."""
    data = {}
    for sym in symbols:
        frames = []
        for yr in years:
            # Try both possible filename formats
            for pattern in [
                f"tradelog_{sym}_{yr}-01-01_{yr}-12-31.csv",
                f"tradelog_{sym}_{yr}*.csv",
            ]:
                matches = list(RESULT_DIR.glob(pattern)) if "*" in pattern \
                    else ([RESULT_DIR / pattern]
                          if (RESULT_DIR / pattern).exists() else [])
                for path in matches:
                    try:
                        df = pd.read_csv(path, index_col=0, parse_dates=True)
                        if df.index.tz is None:
                            df.index = df.index.tz_localize("UTC")
                        frames.append(df)
                        break
                    except Exception as e:
                        print(f"  ⚠  {path.name}: {e}")

        if frames:
            combined = pd.concat(frames).sort_index()
            # Drop duplicate timestamps that might occur from overlapping backtests
            combined = combined[~combined.index.duplicated(keep='first')]
            
            combined["hour"]    = combined.index.hour
            combined["dow"]     = combined.index.dayofweek
            combined["month"]   = combined.index.month
            combined["session"] = combined["hour"].map(tag_session)
            combined["is_flip"] = combined["action"] == 1
            data[sym] = combined
            print(f"  ✓  {sym}: {len(combined):,} bars across {len(frames)} file(s)")
        else:
            print(f"  ✗  {sym}: no trade logs found")

    return data


# ─────────────────────────────────────────────────────────────────────────────
# Metrics helpers
# ─────────────────────────────────────────────────────────────────────────────
def session_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-session performance metrics."""
    rows = []
    for sess in SESSIONS:
        sub = df[df["session"] == sess]
        if len(sub) == 0:
            continue
        pnl      = sub["pnl_usd"].values
        flips    = sub["is_flip"].sum()
        total    = pnl.sum()
        win_rate = (pnl > 0).mean()
        avg_bar  = pnl.mean()
        pf       = (pnl[pnl>0].sum() / abs(pnl[pnl<0].sum() + 1e-10))
        rows.append({
            "Session":    sess,
            "Bars":       len(sub),
            "Flips":      int(flips),
            "Total $":    total,
            "Avg $/bar":  avg_bar,
            "Win Rate":   win_rate,
            "Prof Factor":pf,
        })
    return pd.DataFrame(rows).set_index("Session")


def hourly_pnl(df: pd.DataFrame) -> pd.Series:
    return df.groupby("hour")["pnl_usd"].mean()


def daily_pnl(df: pd.DataFrame) -> pd.Series:
    return df.groupby("dow")["pnl_usd"].mean()


def flip_session_quality(df: pd.DataFrame, lookahead: int = 5) -> pd.DataFrame:
    """
    For each flip, measure PnL over the next N bars.
    Shows which session produces the best entry timing.
    """
    flips = df[df["is_flip"]].copy()
    if len(flips) == 0:
        return pd.DataFrame()

    future_pnl = []
    pnl_arr    = df["pnl_usd"].values
    
    is_flip_arr = df["is_flip"].values
    flip_positions = np.where(is_flip_arr)[0]

    for pos in flip_positions:
        end = min(pos + lookahead + 1, len(pnl_arr))
        future = pnl_arr[pos+1 : end].sum()
        future_pnl.append(future)

    # Convert to array just in case there's an index mismatch, but they align perfectly
    flips["future_pnl"] = np.array(future_pnl)
    return flips.groupby("session")["future_pnl"].agg(["mean","count","std"])


# ─────────────────────────────────────────────────────────────────────────────
# Printing
# ─────────────────────────────────────────────────────────────────────────────
def print_session_table(sym: str, df_metrics: pd.DataFrame):
    print(f"\n{'='*65}")
    print(f"  {sym}  —  Session Performance")
    print(f"{'='*65}")
    print(f"  {'Session':<12} {'Bars':>6} {'Flips':>6} "
          f"{'Total $':>10} {'$/bar':>8} {'WinRate':>8} {'PF':>6}")
    print(f"  {'-'*63}")
    for sess, row in df_metrics.iterrows():
        sign = "+" if row["Total $"] >= 0 else ""
        print(
            f"  {sess:<12} "
            f"{row['Bars']:>6,} "
            f"{row['Flips']:>6} "
            f"  {sign}${row['Total $']:>8,.2f} "
            f"{row['Avg $/bar']:>+8.3f} "
            f"{row['Win Rate']:>8.1%} "
            f"{row['Prof Factor']:>6.2f}"
        )
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Charts
# ─────────────────────────────────────────────────────────────────────────────
def plot_all(data: dict, save_dir: Path, show: bool):
    symbols = list(data.keys())
    n = len(symbols)

    # ── Chart 1: Session PnL bars per symbol ─────────────────────────────────
    fig, axes = plt.subplots(1, n, figsize=(5*n, 5), sharey=False)
    if n == 1: axes = [axes]
    fig.suptitle("Total PnL by Trading Session", fontsize=13, fontweight="bold")

    for ax, sym in zip(axes, symbols):
        df   = data[sym]
        sess_pnl = df.groupby("session")["pnl_usd"].sum().reindex(list(SESSIONS.keys()))
        colours  = [SESSION_COLOURS[s] for s in sess_pnl.index]
        bars     = ax.bar(sess_pnl.index, sess_pnl.values, color=colours,
                          edgecolor="white", linewidth=0.5)
        for bar, val in zip(bars, sess_pnl.values):
            if np.isnan(val): continue
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + abs(bar.get_height())*0.02,
                    f"${val:,.0f}", ha="center", va="bottom",
                    fontsize=7.5, fontweight="bold")
        ax.axhline(0, color="black", linewidth=0.7, alpha=0.4)
        ax.set_title(sym, fontsize=10, fontweight="bold")
        ax.set_ylabel("Total PnL ($)")
        ax.tick_params(axis="x", rotation=30, labelsize=8)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"${x:,.0f}"))
        ax.grid(axis="y", alpha=0.25)
        ax.spines[["top","right"]].set_visible(False)

    plt.tight_layout()
    _save(fig, save_dir / "session_pnl_bars.png", show)

    # ── Chart 2: Hourly PnL heatmap (symbols × hours) ────────────────────────
    matrix = np.zeros((n, 24))
    for i, sym in enumerate(symbols):
        hp = hourly_pnl(data[sym])
        for h in range(24):
            matrix[i, h] = hp.get(h, 0.0)

    fig, ax = plt.subplots(figsize=(16, max(3, n*1.2)))
    fig.suptitle("Average PnL per Bar by Hour (UTC)", fontsize=13, fontweight="bold")
    vmax = np.abs(matrix).max() or 1.0
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    im   = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", norm=norm,
                     interpolation="nearest")
    plt.colorbar(im, ax=ax, format="$%.3f", shrink=0.8)

    # Annotate
    for i in range(n):
        for h in range(24):
            val = matrix[i, h]
            tc  = "white" if abs(val) > vmax*0.5 else "black"
            ax.text(h, i, f"{val:+.2f}", ha="center", va="center",
                    fontsize=6, color=tc)

    # Session bands
    session_bands = [
        ("Asian",    [22,23,0,1,2,3,4,5,6],   "#4A90D9"),
        ("London",   list(range(7,12)),         "#F39C12"),
        ("Overlap",  list(range(12,16)),        "#E74C3C"),
        ("New York", list(range(16,21)),        "#2ECC71"),
        ("Dead Zone",[21],                      "#95A5A6"),
    ]
    for sess, hours, col in session_bands:
        if hours:
            ax.axvline(hours[0]-0.5, color=col, linewidth=1.5, alpha=0.5)
            mid = np.mean(hours)
            ax.text(mid, -0.7, sess, ha="center", va="top",
                    fontsize=7, color=col, fontweight="bold",
                    transform=ax.get_xaxis_transform())

    ax.set_yticks(range(n))
    ax.set_yticklabels(symbols, fontsize=9)
    ax.set_xticks(range(24))
    ax.set_xticklabels([f"{h:02d}" for h in range(24)], fontsize=7)
    ax.set_xlabel("Hour (UTC)", fontsize=9)
    plt.tight_layout()
    _save(fig, save_dir / "hourly_heatmap.png", show)

    # ── Chart 3: Day-of-week performance ─────────────────────────────────────
    fig, axes = plt.subplots(1, n, figsize=(5*n, 4), sharey=False)
    if n == 1: axes = [axes]
    fig.suptitle("Average PnL per Bar by Day of Week", fontsize=13, fontweight="bold")

    for ax, sym in zip(axes, symbols):
        dp   = daily_pnl(data[sym])
        vals = [dp.get(d, 0.0) for d in range(7)]
        cols = ["#2ECC71" if v >= 0 else "#E74C3C" for v in vals]
        bars = ax.bar(DAY_NAMES, vals, color=cols, edgecolor="white", linewidth=0.4)
        ax.axhline(0, color="black", linewidth=0.7, alpha=0.4)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    val + abs(val)*0.05 if val >= 0 else val - abs(val)*0.05,
                    f"{val:+.3f}", ha="center",
                    va="bottom" if val >= 0 else "top",
                    fontsize=7.5)
        ax.set_title(sym, fontsize=10, fontweight="bold")
        ax.set_ylabel("Avg PnL/bar ($)")
        ax.grid(axis="y", alpha=0.25)
        ax.spines[["top","right"]].set_visible(False)
        ax.tick_params(axis="x", labelsize=8)

    plt.tight_layout()
    _save(fig, save_dir / "day_of_week.png", show)

    # ── Chart 4: Win rate by session ─────────────────────────────────────────
    fig, axes = plt.subplots(1, n, figsize=(5*n, 4), sharey=True)
    if n == 1: axes = [axes]
    fig.suptitle("Win Rate by Session vs Random Baseline (52.5%)",
                 fontsize=13, fontweight="bold")

    for ax, sym in zip(axes, symbols):
        df    = data[sym]
        slist = list(SESSIONS.keys())
        wrs   = [df[df["session"]==s]["pnl_usd"].pipe(lambda x: (x>0).mean())
                 for s in slist]
        cols  = [SESSION_COLOURS[s] for s in slist]
        bars  = ax.barh(slist, wrs, color=cols, edgecolor="white", linewidth=0.4)
        ax.axvline(0.525, color="black", linewidth=1.2,
                   linestyle="--", alpha=0.6, label="Random baseline")
        for bar, wr in zip(bars, wrs):
            ax.text(wr + 0.003, bar.get_y() + bar.get_height()/2,
                    f"{wr:.1%}", va="center", fontsize=8)
        ax.set_title(sym, fontsize=10, fontweight="bold")
        ax.set_xlabel("Win Rate")
        ax.set_xlim(0.45, 0.75)
        ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
        ax.grid(axis="x", alpha=0.25)
        ax.spines[["top","right"]].set_visible(False)
        ax.legend(fontsize=7)

    plt.tight_layout()
    _save(fig, save_dir / "session_winrate.png", show)

    # ── Chart 5: Flip entry quality by session ────────────────────────────────
    fig, axes = plt.subplots(1, n, figsize=(5*n, 4), sharey=False)
    if n == 1: axes = [axes]
    fig.suptitle("Average PnL in 5 Bars After Flip — by Session\n"
                 "(positive = flip was well-timed, negative = premature)",
                 fontsize=11, fontweight="bold")

    for ax, sym in zip(axes, symbols):
        fq   = flip_session_quality(data[sym], lookahead=5)
        if fq.empty:
            ax.text(0.5, 0.5, "No flips", ha="center", transform=ax.transAxes)
            continue
        fq   = fq.reindex([s for s in SESSIONS if s in fq.index])
        cols = [SESSION_COLOURS[s] for s in fq.index]
        bars = ax.bar(fq.index, fq["mean"], color=cols,
                      edgecolor="white", linewidth=0.4)
        ax.axhline(0, color="black", linewidth=0.7, alpha=0.4)
        for bar, (_, row) in zip(bars, fq.iterrows()):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.5,
                    f"${row['mean']:+.1f}\n(n={int(row['count'])})",
                    ha="center", va="bottom", fontsize=7)
        ax.set_title(sym, fontsize=10, fontweight="bold")
        ax.set_ylabel("Avg PnL next 5 bars ($)")
        ax.tick_params(axis="x", rotation=30, labelsize=8)
        ax.grid(axis="y", alpha=0.25)
        ax.spines[["top","right"]].set_visible(False)

    plt.tight_layout()
    _save(fig, save_dir / "flip_entry_quality.png", show)

    # ── Chart 6: Cumulative PnL by session over time (portfolio) ─────────────
    fig, ax = plt.subplots(figsize=(14, 5))
    fig.suptitle("Cumulative PnL by Session  (all symbols combined)",
                 fontsize=13, fontweight="bold")

    all_bars = pd.concat(data.values()).sort_index()
    for sess in SESSIONS:
        sub  = all_bars[all_bars["session"] == sess]["pnl_usd"]
        cum  = sub.cumsum()
        ax.plot(cum.index, cum.values,
                label=sess, color=SESSION_COLOURS[sess],
                linewidth=1.5, alpha=0.85)

    ax.axhline(0, color="black", linewidth=0.7, alpha=0.4)
    ax.set_ylabel("Cumulative PnL ($)")
    ax.legend(loc="upper left", fontsize=9, framealpha=0.7)
    ax.grid(alpha=0.25)
    ax.spines[["top","right"]].set_visible(False)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"${x:,.0f}"))
    import matplotlib.dates as mdates
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")
    plt.tight_layout()
    _save(fig, save_dir / "session_cumulative.png", show)

    print(f"\n  Charts saved to {save_dir}/")
    print("    session_pnl_bars.png     — total PnL per session per symbol")
    print("    hourly_heatmap.png       — avg PnL per bar at each hour of day")
    print("    day_of_week.png          — best/worst days of the week")
    print("    session_winrate.png      — win rate vs random baseline per session")
    print("    flip_entry_quality.png   — post-flip PnL by session (timing quality)")
    print("    session_cumulative.png   — cumulative PnL per session over all time")


def _save(fig, path: Path, show: bool):
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"  → {path.name}")
    if show:
        plt.show()
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Session-by-session backtest performance analyser"
    )
    parser.add_argument("--symbols", nargs="+", default=SYMBOLS)
    parser.add_argument("--years",   nargs="+", type=int, default=[2022,2023,2024,2025])
    parser.add_argument("--show",    action="store_true", help="Display charts interactively")
    args = parser.parse_args()

    if not args.show:
        matplotlib.use("Agg")

    save_dir = RESULT_DIR
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nSession Backtest Analyser")
    print(f"Symbols : {args.symbols}")
    print(f"Years   : {args.years}")
    print(f"Results : {save_dir}\n")

    print("Loading trade logs...")
    data = load_logs(args.symbols, args.years)

    if not data:
        print(
            "\nNo trade logs found. Generate them first:\n"
            "  python backtest.py --from 2022-01-01 --to 2022-12-31 --export-csv --no-plots\n"
            "  python backtest.py --from 2023-01-01 --to 2023-12-31 --export-csv --no-plots\n"
            "  python backtest.py --from 2024-01-01 --to 2024-12-31 --export-csv --no-plots\n"
            "  python backtest.py --from 2025-01-01 --to 2025-12-31 --export-csv --no-plots"
        )
        return

    # Print session tables
    for sym, df in data.items():
        metrics = session_metrics(df)
        print_session_table(sym, metrics)

        # Flip quality
        fq = flip_session_quality(df)
        if not fq.empty:
            print(f"  Flip entry quality (avg PnL next 5 bars after flip):")
            for sess, row in fq.iterrows():
                print(f"    {sess:<12}  ${row['mean']:+.2f}  (n={int(row['count'])})")
            print()

    # Portfolio summary
    all_bars = pd.concat(data.values()).sort_index()
    print(f"\n{'='*65}")
    print(f"  PORTFOLIO — All Symbols Combined")
    print(f"{'='*65}")
    metrics = session_metrics(all_bars)
    print_session_table("PORTFOLIO", metrics)

    print("Drawing charts...")
    plot_all(data, save_dir, args.show)


if __name__ == "__main__":
    main()