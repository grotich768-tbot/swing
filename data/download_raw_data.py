import MetaTrader5 as mt5
import pandas as pd
import os
from datetime import datetime

# 1. Configuration
SYMBOLS = [
    "NETH25Cash", "US500Cash", "US2000Cash", "US100Cash",
    "UK100Cash",  "GER40Cash", "EU50Cash",   "US30Cash",
    "CA60Cash",   "AUS200Cash","TaiwanCash"
]

# Map string timeframes to MT5 timeframe constants
TIMEFRAMES = {
    "M1":  mt5.TIMEFRAME_M1,
    "M5":  mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15,
    "M30": mt5.TIMEFRAME_M30,
    "H1":  mt5.TIMEFRAME_H1,
    "H4":  mt5.TIMEFRAME_H4,
    "D1":  mt5.TIMEFRAME_D1,
    "W1":  mt5.TIMEFRAME_W1,
    "MN1": mt5.TIMEFRAME_MN1
}

# Number of bars logic removed to fetch ALL available data
OUTPUT_DIR = "raw"

def main():
    # Ensure the output directory exists
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created directory: {OUTPUT_DIR}/")

    # Initialize MT5 connection
    if not mt5.initialize():
        print("Failed to initialize MT5. Ensure the terminal is running.")
        return

    print("Successfully connected to MetaTrader 5")

    # Define a date safely in the past to capture all history
    # Using 2000-01-01 avoids Windows OS "Invalid argument" errors with 1970 epoch dates
    date_from = datetime(2000, 1, 1)
    date_to = datetime.now()

    # Loop through all symbols and timeframes
    for sym in SYMBOLS:
        for tf_name, tf_val in TIMEFRAMES.items():
            print(f"Fetching ALL available data for {sym} on {tf_name}...")
            
            # Request rates from 1970 to now
            rates = mt5.copy_rates_range(sym, tf_val, date_from, date_to)
            
            if rates is None or len(rates) == 0:
                print(f"  -> WARNING: No data retrieved for {sym} ({tf_name})")
                continue
            
            # Convert to Pandas DataFrame
            df = pd.DataFrame(rates)
            
            # Convert time in seconds to datetime format
            df['time'] = pd.to_datetime(df['time'], unit='s')
            
            # Set the time column as the index
            df.set_index('time', inplace=True)
            
            # Create a clean parquet filename
            filename = os.path.join(OUTPUT_DIR, f"{sym}_{tf_name}.parquet")
            
            # Save to Parquet format
            df.to_parquet(filename, engine='pyarrow')
            
            print(f"  -> Saved {len(df)} rows to {filename}")

    # Shut down MT5 connection
    mt5.shutdown()
    print("\nData extraction complete! All files saved in the 'raw/' folder.")

if __name__ == "__main__":
    main()
