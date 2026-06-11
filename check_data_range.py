"""
check_data_range.py  —  Shows exact date range and bar counts for all Parquet files
Run from your project root:  python check_data_range.py
"""
import sys
sys.path.insert(0, ".")
from data.data_loader import DataLoader
from config import SYMBOLS

loader = DataLoader()

print("\n" + "="*70)
print("  Data Range Report")
print("="*70)

timeframes = ["M15", "H1", "H4", "D1"]

for sym in SYMBOLS:
    print(f"\n  {sym}")
    print(f"  {'─'*60}")
    for tf in timeframes:
        df = loader.load(sym, tf, "1900-01-01", "2099-12-31")
        if df is None or df.empty:
            print(f"    {tf:5s}  ✗  not found")
        else:
            print(
                f"    {tf:5s}  ✓  "
                f"{df.index[0].strftime('%Y-%m-%d')}  →  "
                f"{df.index[-1].strftime('%Y-%m-%d')}  "
                f"({len(df):,} bars)"
            )

print("\n" + "="*70)
print("  Recommended config.py settings based on your data:")
print("="*70)

# Find the common usable range across all symbols on H1
starts, ends = [], []
for sym in SYMBOLS:
    df = loader.load(sym, "H1", "1900-01-01", "2099-12-31")
    if df is not None and not df.empty:
        starts.append(df.index[0])
        ends.append(df.index[-1])

if starts and ends:
    common_start = max(starts).strftime("%Y-%m-%d")
    common_end   = min(ends).strftime("%Y-%m-%d")

    # Suggest 80/20 train/test split
    import pandas as pd
    total_days = (pd.Timestamp(common_end) - pd.Timestamp(common_start)).days
    split_days = int(total_days * 0.8)
    train_end  = (pd.Timestamp(common_start) + pd.Timedelta(days=split_days)).strftime("%Y-%m-%d")
    test_start = (pd.Timestamp(common_start) + pd.Timedelta(days=split_days + 14)).strftime("%Y-%m-%d")

    print(f"""
  Your full H1 range  : {common_start}  →  {common_end}
  Total days          : {total_days:,}

  Suggested split (80/20 with 2-week purge gap):
    TRAIN_START = "{common_start}"
    TRAIN_END   = "{train_end}"
    TEST_START  = "{test_start}"
    TEST_END    = "{common_end}"
""")

