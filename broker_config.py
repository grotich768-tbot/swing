"""
broker_config.py  —  Per-broker instrument specifications
──────────────────────────────────────────────────────────────────────────────
Each broker has different:
  - Pip USD values (contract sizes differ)
  - Spreads (variable by broker)
  - Symbol names (XAUUSD vs GOLD vs XAUUSDm)

Usage:
  1. Set ACTIVE_BROKER to your broker name below
  2. Or set environment variable: set BROKER=ICMarkets
  3. Or pass at runtime: python train.py --broker ICMarkets

Adding a new broker:
  Copy an existing broker block, rename it, update values.
  Values come from: MT5 → Market Watch → right-click → Specification

Auto-detect from MT5:
  Run: python broker_config.py --detect
  This connects to MT5 and reads live specs automatically.
"""

import os
import sys
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Set your broker here (or use env var BROKER=BrokerName)
# ─────────────────────────────────────────────────────────────────────────────
ACTIVE_BROKER = os.environ.get("BROKER", "Eightcap")


# ─────────────────────────────────────────────────────────────────────────────
# Broker specifications
# Format: { symbol: { pip_usd, spread, pip_size } }
#   pip_usd  — USD per pip per 1.0 standard lot
#   spread   — typical spread in pips
#   pip_size — price movement per pip (PIP_VALUE in config)
# ─────────────────────────────────────────────────────────────────────────────
BROKERS = {

    # ── Eightcap (your current broker) ───────────────────────────────────────
    "Eightcap": {
        "GOLD":   {"pip_usd": 0.10, "spread": 6.0,   "pip_size": 0.01},
        "SILVER": {"pip_usd": 0.50, "spread": 10.0,  "pip_size": 0.001},
        "EURUSD": {"pip_usd": 0.10, "spread": 1.0,   "pip_size": 0.0001},
        "GBPUSD": {"pip_usd": 0.10, "spread": 1.1,   "pip_size": 0.0001},
        "USDJPY": {"pip_usd": 0.06, "spread": 1.1,   "pip_size": 0.01},
        "BTCUSD": {"pip_usd": 0.10, "spread": 297.6, "pip_size": 0.1},
        "ETHUSD": {"pip_usd": 0.10, "spread": 49.8,  "pip_size": 0.1},
        "US30":   {"pip_usd": 1.00, "spread": 3.90,  "pip_size": 1.0},
        "US100":  {"pip_usd": 1.00, "spread": 1.95,  "pip_size": 1.0},
        "US500":  {"pip_usd": 1.00, "spread": 0.55,  "pip_size": 1.0},
        "UK100":  {"pip_usd": 1.00, "spread": 1.60,  "pip_size": 1.0},
        "GER40":  {"pip_usd": 1.00, "spread": 1.95,  "pip_size": 1.0},
        "AUS200": {"pip_usd": 1.00, "spread": 5.54,  "pip_size": 1.0},
        "JP225":  {"pip_usd": 1.00, "spread": 8.00,  "pip_size": 1.0},
    },

    # ── ICMarkets (Raw/ECN account) ───────────────────────────────────────────
    "ICMarkets": {
        "GOLD":   {"pip_usd": 1.00, "spread": 2.0,   "pip_size": 0.01},
        "SILVER": {"pip_usd": 5.00, "spread": 3.0,   "pip_size": 0.001},
        "EURUSD": {"pip_usd": 10.0, "spread": 0.1,   "pip_size": 0.0001},
        "GBPUSD": {"pip_usd": 10.0, "spread": 0.4,   "pip_size": 0.0001},
        "USDJPY": {"pip_usd": 9.09, "spread": 0.3,   "pip_size": 0.01},
        "BTCUSD": {"pip_usd": 1.00, "spread": 50.0,  "pip_size": 0.1},
        "ETHUSD": {"pip_usd": 1.00, "spread": 10.0,  "pip_size": 0.1},
        "US30":   {"pip_usd": 1.00, "spread": 1.00,  "pip_size": 1.0},
        "US100":  {"pip_usd": 1.00, "spread": 0.50,  "pip_size": 1.0},
        "US500":  {"pip_usd": 1.00, "spread": 0.10,  "pip_size": 1.0},
        "UK100":  {"pip_usd": 1.00, "spread": 0.80,  "pip_size": 1.0},
        "GER40":  {"pip_usd": 1.00, "spread": 0.50,  "pip_size": 1.0},
        "AUS200": {"pip_usd": 1.00, "spread": 1.80,  "pip_size": 1.0},
        "JP225":  {"pip_usd": 1.00, "spread": 5.00,  "pip_size": 1.0},
    },

    # ── Pepperstone (Razor account) ───────────────────────────────────────────
    "Pepperstone": {
        "GOLD":   {"pip_usd": 1.00, "spread": 1.5,   "pip_size": 0.01},
        "SILVER": {"pip_usd": 5.00, "spread": 2.5,   "pip_size": 0.001},
        "EURUSD": {"pip_usd": 10.0, "spread": 0.1,   "pip_size": 0.0001},
        "GBPUSD": {"pip_usd": 10.0, "spread": 0.4,   "pip_size": 0.0001},
        "USDJPY": {"pip_usd": 9.09, "spread": 0.2,   "pip_size": 0.01},
        "BTCUSD": {"pip_usd": 1.00, "spread": 40.0,  "pip_size": 0.1},
        "ETHUSD": {"pip_usd": 1.00, "spread": 8.0,   "pip_size": 0.1},
        "US30":   {"pip_usd": 1.00, "spread": 1.50,  "pip_size": 1.0},
        "US100":  {"pip_usd": 1.00, "spread": 0.50,  "pip_size": 1.0},
        "US500":  {"pip_usd": 1.00, "spread": 0.10,  "pip_size": 1.0},
        "UK100":  {"pip_usd": 1.00, "spread": 0.80,  "pip_size": 1.0},
        "GER40":  {"pip_usd": 1.00, "spread": 0.50,  "pip_size": 1.0},
        "AUS200": {"pip_usd": 1.00, "spread": 1.80,  "pip_size": 1.0},
        "JP225":  {"pip_usd": 1.00, "spread": 5.00,  "pip_size": 1.0},
    },

    # ── Exness ────────────────────────────────────────────────────────────────
    "Exness": {
        "GOLD":   {"pip_usd": 1.00, "spread": 1.0,   "pip_size": 0.01},
        "SILVER": {"pip_usd": 5.00, "spread": 2.0,   "pip_size": 0.001},
        "EURUSD": {"pip_usd": 10.0, "spread": 0.3,   "pip_size": 0.0001},
        "GBPUSD": {"pip_usd": 10.0, "spread": 0.5,   "pip_size": 0.0001},
        "USDJPY": {"pip_usd": 9.09, "spread": 0.3,   "pip_size": 0.01},
        "BTCUSD": {"pip_usd": 1.00, "spread": 100.0, "pip_size": 0.1},
        "ETHUSD": {"pip_usd": 1.00, "spread": 20.0,  "pip_size": 0.1},
        "US30":   {"pip_usd": 1.00, "spread": 2.00,  "pip_size": 1.0},
        "US100":  {"pip_usd": 1.00, "spread": 1.00,  "pip_size": 1.0},
        "US500":  {"pip_usd": 1.00, "spread": 0.50,  "pip_size": 1.0},
        "UK100":  {"pip_usd": 1.00, "spread": 1.50,  "pip_size": 1.0},
        "GER40":  {"pip_usd": 1.00, "spread": 1.00,  "pip_size": 1.0},
        "AUS200": {"pip_usd": 1.00, "spread": 3.00,  "pip_size": 1.0},
        "JP225":  {"pip_usd": 1.00, "spread": 8.00,  "pip_size": 1.0},
    },

    # ── Add your broker here ──────────────────────────────────────────────────
    # "MyBroker": {
    #     "GOLD":   {"pip_usd": ?, "spread": ?, "pip_size": 0.01},
    #     ...
    # },
}


# ─────────────────────────────────────────────────────────────────────────────
# Public API — used by config.py, backtest.py, live_utils.py
# ─────────────────────────────────────────────────────────────────────────────
def get_broker_specs(broker: str = None) -> dict:
    """Return full spec dict for the active broker."""
    broker = broker or ACTIVE_BROKER
    if broker not in BROKERS:
        available = list(BROKERS.keys())
        raise ValueError(
            f"Broker '{broker}' not found. Available: {available}\n"
            f"Add it to broker_config.py or run: python broker_config.py --detect"
        )
    return BROKERS[broker]


def get_pip_usd(symbol: str, broker: str = None) -> float:
    """USD per pip per 1.0 lot for symbol at broker."""
    specs = get_broker_specs(broker)
    if symbol not in specs:
        return 1.0   # safe default
    return specs[symbol]["pip_usd"]


def get_spread(symbol: str, broker: str = None) -> float:
    """Typical spread in pips for symbol at broker."""
    specs = get_broker_specs(broker)
    if symbol not in specs:
        return 5.0   # safe default
    return specs[symbol]["spread"]


def get_pip_size(symbol: str, broker: str = None) -> float:
    """Price per pip for symbol at broker."""
    specs = get_broker_specs(broker)
    if symbol not in specs:
        return 0.0001
    return specs[symbol]["pip_size"]


def apply_to_config():
    """
    Apply active broker specs to config.py values at runtime.
    Call this at the top of train.py and live_run.py.
    """
    import config as cfg
    specs = get_broker_specs()

    for symbol, vals in specs.items():
        cfg.SPREAD_PIPS[symbol] = vals["spread"]
        cfg.PIP_VALUE[symbol]   = vals["pip_size"]

    from loguru import logger
    logger.info(
        f"Broker specs applied: {ACTIVE_BROKER}  "
        f"({len(specs)} symbols)"
    )


# ─────────────────────────────────────────────────────────────────────────────
# MT5 auto-detect — reads live specs from connected MT5 terminal
# ─────────────────────────────────────────────────────────────────────────────
def detect_from_mt5(symbols: list = None) -> dict:
    """
    Auto-detect broker specs by reading from live MT5 terminal.
    Prints a broker_config.py block you can copy in.

    Usage:
        python broker_config.py --detect
        python broker_config.py --detect --broker MyBroker
    """
    try:
        import MetaTrader5 as mt5
    except ImportError:
        print("MetaTrader5 package not installed. Run: pip install MetaTrader5")
        return {}

    if not mt5.initialize():
        print(f"MT5 not running or not connected: {mt5.last_error()}")
        return {}

    from config import MT5_SYMBOLS
    symbols = symbols or list(MT5_SYMBOLS.keys())

    detected = {}
    print(f"\nDetecting broker specs from MT5...")
    print(f"{'Symbol':<10} {'pip_usd':>10} {'spread':>10} {'pip_size':>12}")
    print("-" * 45)

    for sym in symbols:
        # Try all alias names for this symbol
        mt5_names = MT5_SYMBOLS.get(sym, [sym])
        info = None
        for name in mt5_names:
            info = mt5.symbol_info(name)
            if info:
                break

        if not info:
            print(f"{sym:<10}  not found in MT5")
            continue

        pip_size = info.point * 10 if info.digits <= 3 else info.point * 10
        pip_usd  = info.trade_contract_size * pip_size / (info.trade_tick_size / info.trade_tick_value) if info.trade_tick_value else 0
        spread   = info.spread * info.point / pip_size if pip_size > 0 else info.spread

        detected[sym] = {
            "pip_usd":  round(pip_usd, 4),
            "spread":   round(spread, 1),
            "pip_size": pip_size,
        }
        print(f"{sym:<10} {pip_usd:>10.4f} {spread:>10.1f} {pip_size:>12.5f}")

    mt5.shutdown()

    # Print copy-pasteable block
    print(f"\n# Paste this into BROKERS dict in broker_config.py:")
    print(f'    "YourBrokerName": {{')
    for sym, vals in detected.items():
        print(f'        "{sym}": {{"pip_usd": {vals["pip_usd"]}, "spread": {vals["spread"]}, "pip_size": {vals["pip_size"]}}},')
    print(f'    }},')

    return detected


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Broker config tool")
    parser.add_argument("--detect", action="store_true",
                        help="Auto-detect specs from live MT5 terminal")
    parser.add_argument("--broker", type=str, default=None,
                        help="Broker name to use")
    parser.add_argument("--list",   action="store_true",
                        help="List all configured brokers")
    parser.add_argument("--show",   action="store_true",
                        help="Show active broker specs")
    args = parser.parse_args()

    if args.list:
        print("Configured brokers:")
        for b in BROKERS:
            marker = " ← ACTIVE" if b == ACTIVE_BROKER else ""
            print(f"  {b}{marker}")

    elif args.show:
        broker = args.broker or ACTIVE_BROKER
        print(f"\nSpecs for: {broker}")
        specs  = get_broker_specs(broker)
        print(f"{'Symbol':<10} {'pip_usd':>10} {'spread':>10} {'pip_size':>12}")
        print("-" * 45)
        for sym, vals in specs.items():
            print(f"{sym:<10} {vals['pip_usd']:>10.4f} {vals['spread']:>10.1f} {vals['pip_size']:>12.5f}")

    elif args.detect:
        detect_from_mt5()

    else:
        parser.print_help()
