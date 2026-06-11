"""
monthly_charts.py  —  Monthly PnL charts across all backtest periods
─────────────────────────────────────────────────────────────────────
Reads trade log CSVs from results/ and draws:
  1. Monthly bar chart per symbol (4 panels, one per year)
  2. Portfolio combined monthly bar chart
  3. Cumulative equity curves per symbol
  4. Year-over-year monthly comparison heatmap

Usage:
    python monthly_charts.py                      # saves to results/
    python monthly_charts.py --show               # display interactively
    python monthly_charts.py --years 2022 2024    # specific years only
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
import matplotlib.cm as cm

sys.path.insert(0, ".")
from config import SYMBOLS, RESULT_DIR

RESULTS_DIR = RESULT_DIR   # absolute path — works from any directory
MONTH_NAMES = ["Jan","Feb","Mar","Apr","May","Jun",
               "Jul","Aug","Sep","Oct","Nov","Dec"]

SYMBOL_COLOURS = {
    "GOLD":   "#FFD700",
    "SILVER": "#A8A9AD",
    "EURUSD": "#4A90D9",
    "GBPUSD": "#E74C3C",
}


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────
def find_tradelog(symbol: str, year: int) -> Path | None:
    """
    Find a trade log CSV for a symbol and year.
    Accepts any filename that contains the symbol and year,
    e.g. tradelog_GOLD_2022-01-01_2022-12-31.csv
    """
    if not RESULTS_DIR.exists():
        return None
    for f in RESULTS_DIR.glob(f"tradelog_{symbol}_{year}*.csv"):
        return f   # return first match
    return None


def load_monthly(symbol: str, year: int) -> "pd.Series | None":
    path = find_tradelog(symbol, year)
    if path is None:
        return None
    try:
        df = pd.read_csv(path, index_col=0, parse_dates=True)

        # Show columns on first load to help diagnose issues
        if not hasattr(load_monthly, "_cols_shown"):
            load_monthly._cols_shown = True
            print(f"    CSV columns: {list(df.columns)}")
            print(f"    Index dtype: {df.index.dtype}")
            print(f"    Rows: {len(df):,}")

        if "pnl_usd" not in df.columns:
            # Try fallback column names from older backtest versions
            for alt in ["pnl", "step_pnl", "PnL"]:
                if alt in df.columns:
                    df["pnl_usd"] = df[alt]
                    print(f"    ⚠  Using '{alt}' as pnl_usd for {path.name}")
                    break
            else:
                print(f"    ✗  No pnl_usd column in {path.name}")
                return None

        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        else:
            df.index = df.index.tz_convert("UTC")

        monthly = df["pnl_usd"].resample("ME").sum()
        monthly.index = monthly.index.month   # 1–12

        total = monthly.sum()
        print(f"    ✓  {symbol} {year}  →  {len(monthly)} months  total=${total:,.0f}")
        return monthly

    except Exception as e:
        print(f"    ✗  Failed loading {path.name}: {e}")
        return None


def load_all(symbols: list, years: list) -> dict:
    """Returns {symbol: {year: pd.Series(month→pnl)}}"""
    print("\nLoading data:")
    data = {}
    for sym in symbols:
        data[sym] = {}
        for yr in years:
            m = load_monthly(sym, yr)
            if m is not None:
                data[sym][yr] = m
    return data


# ─────────────────────────────────────────────────────────────────────────────
# Chart 1 — Monthly bars per symbol across years
# ─────────────────────────────────────────────────────────────────────────────
def plot_monthly_bars(data: dict, years: list, save_dir: Path, show: bool):
    symbols = [s for s in SYMBOLS if s in data and data[s]]
    n_syms  = len(symbols)
    n_years = len(years)

    fig, axes = plt.subplots(
        n_syms, n_years,
        figsize=(5 * n_years, 3.5 * n_syms),
        sharey="row",
    )
    if n_syms == 1: axes = [axes]
    if n_years == 1: axes = [[ax] for ax in axes]

    fig.suptitle(
        "Monthly PnL by Symbol and Year  (Fixed 2% Sizing, $10k Balance)",
        fontsize=13, fontweight="bold", y=1.01,
    )

    for row, sym in enumerate(symbols):
        colour = SYMBOL_COLOURS.get(sym, "#7F8C8D")
        for col, yr in enumerate(years):
            ax     = axes[row][col]
            series = data[sym].get(yr)

            if series is None:
                ax.text(0.5, 0.5, "No data", ha="center", va="center",
                        transform=ax.transAxes, color="grey")
                ax.set_title(f"{sym}  {yr}", fontsize=9)
                continue

            # Build full 12-month array (NaN for missing months)
            months = np.arange(1, 13)
            values = np.array([series.get(m, np.nan) for m in months])

            bar_colours = [
                colour if (not np.isnan(v) and v >= 0) else "#E74C3C"
                for v in values
            ]
            bars = ax.bar(
                np.arange(12), values,
                color=bar_colours, edgecolor="white", linewidth=0.4, width=0.75,
            )

            # Value labels on bars
            for bar, val in zip(bars, values):
                if np.isnan(val):
                    continue
                h   = bar.get_height()
                yp  = h + abs(h) * 0.03 if h >= 0 else h - abs(h) * 0.03
                va  = "bottom" if h >= 0 else "top"
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    yp, f"${val/1000:.1f}k",
                    ha="center", va=va, fontsize=6.5, fontweight="bold",
                    color="black",
                )

            ax.axhline(0, color="black", linewidth=0.6, alpha=0.4)
            ax.set_xticks(np.arange(12))
            ax.set_xticklabels(MONTH_NAMES, fontsize=7, rotation=45, ha="right")
            ax.set_title(f"{sym}  {yr}", fontsize=9, fontweight="bold")
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(
                lambda x, _: f"${x/1000:.0f}k"
            ))
            ax.tick_params(axis="y", labelsize=7)
            ax.grid(axis="y", alpha=0.25, linewidth=0.5)
            ax.spines[["top","right"]].set_visible(False)

            # Annual total annotation
            total = np.nansum(values)
            ax.text(
                0.98, 0.97,
                f"Total\n${total:,.0f}",
                transform=ax.transAxes,
                ha="right", va="top",
                fontsize=7.5, fontweight="bold",
                color=colour if total >= 0 else "#E74C3C",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7, lw=0),
            )

    plt.tight_layout()
    _save_or_show(fig, save_dir / "monthly_bars_by_symbol.png", show)


# ─────────────────────────────────────────────────────────────────────────────
# Chart 2 — Portfolio combined monthly bars
# ─────────────────────────────────────────────────────────────────────────────
def plot_portfolio_monthly(data: dict, years: list, save_dir: Path, show: bool):
    fig, axes = plt.subplots(
        1, len(years),
        figsize=(5 * len(years), 4.5),
        sharey=True,
    )
    if len(years) == 1:
        axes = [axes]

    fig.suptitle(
        "Portfolio Monthly PnL  (All 4 Symbols Combined)",
        fontsize=13, fontweight="bold",
    )

    for ax, yr in zip(axes, years):
        portfolio = np.zeros(12)
        has_data  = False
        for sym in SYMBOLS:
            series = data.get(sym, {}).get(yr)
            if series is not None:
                has_data = True
                for mo in range(1, 13):
                    portfolio[mo - 1] += series.get(mo, 0.0)

        if not has_data:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes, color="grey")
            ax.set_title(str(yr))
            continue

        colours = ["#2ECC71" if v >= 0 else "#E74C3C" for v in portfolio]
        bars    = ax.bar(np.arange(12), portfolio, color=colours,
                         edgecolor="white", linewidth=0.4, width=0.75)

        for bar, val in zip(bars, portfolio):
            if val == 0:
                continue
            h  = bar.get_height()
            yp = h + abs(h) * 0.03 if h >= 0 else h - abs(h) * 0.03
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                yp, f"${val/1000:.1f}k",
                ha="center", va="bottom" if h >= 0 else "top",
                fontsize=7, fontweight="bold",
            )

        ax.axhline(0, color="black", linewidth=0.6, alpha=0.4)
        ax.set_xticks(np.arange(12))
        ax.set_xticklabels(MONTH_NAMES, fontsize=8, rotation=45, ha="right")
        ax.set_title(f"{yr}", fontsize=11, fontweight="bold")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(
            lambda x, _: f"${x/1000:.0f}k"
        ))
        ax.tick_params(axis="y", labelsize=8)
        ax.grid(axis="y", alpha=0.25, linewidth=0.5)
        ax.spines[["top","right"]].set_visible(False)

        total    = portfolio.sum()
        pos_mos  = (portfolio > 0).sum()
        ax.text(
            0.02, 0.97,
            f"Total  : ${total:,.0f}\n"
            f"Green  : {pos_mos}/12 months\n"
            f"Avg/mo : ${total/12:,.0f}",
            transform=ax.transAxes,
            ha="left", va="top", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8, lw=0.5),
        )

    plt.tight_layout()
    _save_or_show(fig, save_dir / "portfolio_monthly_bars.png", show)


# ─────────────────────────────────────────────────────────────────────────────
# Chart 3 — Cumulative equity curves per symbol
# ─────────────────────────────────────────────────────────────────────────────
def plot_equity_curves(data: dict, years: list, save_dir: Path, show: bool):
    symbols = [s for s in SYMBOLS if s in data and data[s]]
    fig, axes = plt.subplots(
        len(symbols), 1,
        figsize=(14, 3.5 * len(symbols)),
        sharex=False,
    )
    if len(symbols) == 1:
        axes = [axes]

    fig.suptitle(
        "Cumulative Monthly Equity  (rebased to $0 each year)",
        fontsize=13, fontweight="bold",
    )

    for ax, sym in zip(axes, symbols):
        colour   = SYMBOL_COLOURS.get(sym, "#7F8C8D")
        linestyles = ["-", "--", "-.", ":"]

        for i, yr in enumerate(years):
            series = data[sym].get(yr)
            if series is None:
                continue
            months   = sorted(series.index)
            cumulative = np.cumsum([series.get(m, 0.0) for m in months])
            xs = [f"{MONTH_NAMES[m-1]}" for m in months]

            ax.plot(
                xs, cumulative,
                color=colour,
                linestyle=linestyles[i % len(linestyles)],
                linewidth=1.8,
                marker="o", markersize=4,
                label=str(yr),
                alpha=0.85,
            )
            # End-of-year annotation
            ax.annotate(
                f"${cumulative[-1]:,.0f}",
                xy=(len(xs) - 1, cumulative[-1]),
                xytext=(5, 0), textcoords="offset points",
                fontsize=7, color=colour,
                va="center",
            )

        ax.axhline(0, color="black", linewidth=0.8, alpha=0.4, linestyle="--")
        ax.fill_between(
            range(12), 0,
            [max(0, v) for v in [0]*12],
            alpha=0.05, color=colour,
        )
        ax.set_title(sym, fontsize=10, fontweight="bold", color=colour)
        ax.legend(loc="upper left", fontsize=8, framealpha=0.7)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(
            lambda x, _: f"${x:,.0f}"
        ))
        ax.tick_params(axis="x", labelsize=8, rotation=30)
        ax.tick_params(axis="y", labelsize=8)
        ax.grid(alpha=0.25, linewidth=0.5)
        ax.spines[["top","right"]].set_visible(False)

    plt.tight_layout()
    _save_or_show(fig, save_dir / "cumulative_equity_curves.png", show)


# ─────────────────────────────────────────────────────────────────────────────
# Chart 4 — Month × Year heatmap per symbol
# ─────────────────────────────────────────────────────────────────────────────
def plot_heatmap(data: dict, years: list, save_dir: Path, show: bool):
    symbols = [s for s in SYMBOLS if s in data and data[s]]
    n       = len(symbols)
    fig, axes = plt.subplots(
        n, 1, figsize=(max(8, len(years) * 2.5), 2.5 * n)
    )
    if n == 1:
        axes = [axes]

    fig.suptitle(
        "Monthly PnL Heatmap  (green = profit, red = loss)",
        fontsize=13, fontweight="bold",
    )

    for ax, sym in zip(axes, symbols):
        # Build matrix: rows=months, cols=years
        matrix = np.full((12, len(years)), np.nan)
        for col_i, yr in enumerate(years):
            series = data[sym].get(yr)
            if series is None:
                continue
            for mo in range(1, 13):
                matrix[mo - 1, col_i] = series.get(mo, np.nan)

        # Normalise colour around 0
        vmax = np.nanmax(np.abs(matrix)) if not np.all(np.isnan(matrix)) else 1.0
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

        im = ax.imshow(
            matrix, aspect="auto", norm=norm,
            cmap="RdYlGn", interpolation="nearest",
        )
        plt.colorbar(im, ax=ax, format="$%.0f", shrink=0.8, pad=0.01)

        # Annotate cells
        for row_i in range(12):
            for col_i in range(len(years)):
                val = matrix[row_i, col_i]
                if np.isnan(val):
                    continue
                text_col = "black" if abs(val) < vmax * 0.6 else "white"
                ax.text(
                    col_i, row_i,
                    f"${val/1000:.1f}k",
                    ha="center", va="center",
                    fontsize=7.5, fontweight="bold",
                    color=text_col,
                )

        ax.set_yticks(np.arange(12))
        ax.set_yticklabels(MONTH_NAMES, fontsize=8)
        ax.set_xticks(np.arange(len(years)))
        ax.set_xticklabels(years, fontsize=9, fontweight="bold")
        ax.set_title(sym, fontsize=10, fontweight="bold",
                     color=SYMBOL_COLOURS.get(sym, "black"))
        ax.tick_params(length=0)

        # Year totals on top
        for col_i, yr in enumerate(years):
            col_vals = matrix[:, col_i]
            total    = np.nansum(col_vals)
            pos_mos  = int(np.nansum(col_vals > 0))
            ax.text(
                col_i, -0.8,
                f"${total/1000:.0f}k\n{pos_mos}/12",
                ha="center", va="bottom",
                fontsize=7, fontweight="bold",
                color="#2ECC71" if total >= 0 else "#E74C3C",
            )

    plt.tight_layout()
    _save_or_show(fig, save_dir / "monthly_heatmap.png", show)


# ─────────────────────────────────────────────────────────────────────────────
# Chart 5 — Year comparison: stacked monthly portfolio
# ─────────────────────────────────────────────────────────────────────────────
def plot_year_comparison(data: dict, years: list, save_dir: Path, show: bool):
    fig, ax = plt.subplots(figsize=(13, 5))
    fig.suptitle(
        "Portfolio Monthly PnL — Year-over-Year Comparison",
        fontsize=13, fontweight="bold",
    )

    x      = np.arange(12)
    n_yrs  = len(years)
    width  = 0.8 / n_yrs
    cmap   = cm.get_cmap("tab10", n_yrs)

    for i, yr in enumerate(years):
        portfolio = np.zeros(12)
        for sym in SYMBOLS:
            series = data.get(sym, {}).get(yr)
            if series is not None:
                for mo in range(1, 13):
                    portfolio[mo - 1] += series.get(mo, 0.0)

        offset = (i - n_yrs / 2 + 0.5) * width
        colour = cmap(i)
        bars   = ax.bar(
            x + offset, portfolio,
            width=width * 0.9,
            color=colour,
            label=str(yr),
            edgecolor="white",
            linewidth=0.3,
            alpha=0.88,
        )

    ax.axhline(0, color="black", linewidth=0.8, alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(MONTH_NAMES, fontsize=9)
    ax.set_ylabel("Portfolio PnL (USD)", fontsize=9)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda v, _: f"${v/1000:.0f}k"
    ))
    ax.legend(title="Year", fontsize=9, title_fontsize=9, loc="upper right")
    ax.grid(axis="y", alpha=0.25, linewidth=0.5)
    ax.spines[["top","right"]].set_visible(False)
    ax.tick_params(axis="y", labelsize=8)

    # Jan is always the weakest — annotate it
    ax.annotate(
        "Jan: consistent\nlow-liquidity month",
        xy=(0, 5000), xytext=(0.5, 0.85),
        textcoords="axes fraction",
        fontsize=7.5, color="grey",
        arrowprops=dict(arrowstyle="->", color="grey", lw=0.8),
        ha="center",
    )

    plt.tight_layout()
    _save_or_show(fig, save_dir / "year_comparison.png", show)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _save_or_show(fig, path: Path, show: bool):
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"  Saved → {path}")
    if show:
        plt.show()
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Draw monthly PnL charts from backtest trade logs."
    )
    parser.add_argument(
        "--show", action="store_true",
        help="Display charts interactively instead of saving"
    )
    parser.add_argument(
        "--years", nargs="+", type=int,
        default=[2022, 2023, 2024, 2025],
        help="Years to include (default: 2022 2023 2024 2025)"
    )
    parser.add_argument(
        "--symbols", nargs="+",
        default=SYMBOLS,
        help="Symbols to include"
    )
    args = parser.parse_args()

    save_dir = RESULT_DIR  # absolute path from config — always correct
    save_dir.mkdir(parents=True, exist_ok=True)

    # Always use Agg (file backend) unless --show is explicitly passed
    # on a machine with a display. Agg works everywhere.
    if not args.show:
        matplotlib.use("Agg")

    years   = sorted(args.years)
    symbols = args.symbols

    print(f"\nScanning {RESULTS_DIR}/ for trade logs...")
    if not RESULTS_DIR.exists():
        print(f"  ✗  results/ folder not found at {RESULTS_DIR.resolve()}")
        return

    all_csvs = sorted(RESULTS_DIR.glob("tradelog_*.csv"))
    if not all_csvs:
        print(
            "  ✗  No tradelog_*.csv files found.\n"
            "  Run these first:\n"
            "    python backtest.py --from 2022-01-01 --to 2022-12-31 --export-csv --no-plots\n"
            "    python backtest.py --from 2023-01-01 --to 2023-12-31 --export-csv --no-plots\n"
            "    python backtest.py --from 2024-01-01 --to 2024-12-31 --export-csv --no-plots\n"
            "    python backtest.py --from 2025-01-01 --to 2025-12-31 --export-csv --no-plots"
        )
        return

    print(f"  Found {len(all_csvs)} file(s):")
    for f in all_csvs:
        size = f.stat().st_size / 1024
        print(f"    ✓  {f.name}  ({size:.0f} KB)")

    data = load_all(symbols, years)

    found = sum(1 for sym in symbols for yr in years if data.get(sym, {}).get(yr) is not None)
    if found == 0:
        print(
            "\nNo trade logs found. Run first:\n"
            "  python backtest.py --from 2022-01-01 --to 2022-12-31 --export-csv --no-plots\n"
            "  python backtest.py --from 2023-01-01 --to 2023-12-31 --export-csv --no-plots\n"
            "  python backtest.py --from 2024-01-01 --to 2024-12-31 --export-csv --no-plots\n"
            "  python backtest.py --from 2025-01-01 --to 2025-12-31 --export-csv --no-plots\n"
        )
        return

    print(f"Found {found} symbol-year combinations. Drawing charts...\n")

    plot_monthly_bars(data, years, save_dir, args.show)
    plot_portfolio_monthly(data, years, save_dir, args.show)
    plot_equity_curves(data, years, save_dir, args.show)
    plot_heatmap(data, years, save_dir, args.show)
    plot_year_comparison(data, years, save_dir, args.show)

    print(f"\nAll charts saved to {save_dir}/")
    print("  monthly_bars_by_symbol.png   — monthly bars per symbol × year")
    print("  portfolio_monthly_bars.png   — combined portfolio monthly bars")
    print("  cumulative_equity_curves.png — running equity per symbol")
    print("  monthly_heatmap.png          — month × year heatmap (green/red)")
    print("  year_comparison.png          — all years side by side per month")


if __name__ == "__main__":
    main()
