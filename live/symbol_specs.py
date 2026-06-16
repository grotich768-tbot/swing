"""
live/symbol_specs.py  —  Auto-fetch symbol specifications from MT5
──────────────────────────────────────────────────────────────────────────────
Reads live symbol properties directly from the connected MT5 terminal:
  - Contract size
  - Tick size / tick value  
  - Current spread
  - Pip value in USD
  - Digits / point

Zero manual configuration when switching brokers.
Values read once on startup and cached for the session.

Usage:
    from live.symbol_specs import SymbolSpecs
    specs = SymbolSpecs()
    specs.apply_to_config()     # override config.py values automatically

    pip_usd = specs.pip_usd("GOLD")
    spread  = specs.spread("GOLD")
    specs.print_table()

    # Save detected values permanently to broker_config.py
    python live/symbol_specs.py --save --broker MyBroker
"""

import sys
from pathlib import Path
from typing import Dict, Optional
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import MT5_SYMBOLS


class SymbolSpecs:
    """
    Reads and caches symbol specifications from MT5.

    Parameters
    ----------
    symbols          : list of internal symbol names e.g. ["GOLD", "SILVER"]
                       None = all symbols from config.MT5_SYMBOLS
    account_currency : account base currency (default "USD")
    """

    def __init__(self, symbols: list = None, account_currency: str = "USD", settings=None):
        self._specs:    Dict[str, dict] = {}
        self._currency: str             = account_currency
        self._symbols:  list            = symbols or list(MT5_SYMBOLS.keys())
        self._mt5_map:  Dict[str, str]  = {}   # internal name → actual MT5 name
        self._broker:   str             = "Unknown"
        self._settings                  = settings
        self._fetch_all()

    # ── Public API ────────────────────────────────────────────────────────────
    def pip_usd(self, symbol: str) -> float:
        """USD value per pip per 1.0 standard lot."""
        return self._specs.get(symbol, {}).get("pip_usd", 1.0)

    def spread(self, symbol: str) -> float:
        """Current spread in pips — fetches live value if MT5 available."""
        live = self._live_spread(symbol)
        if live is not None:
            return live
        return self._specs.get(symbol, {}).get("spread", 5.0)

    def pip_size(self, symbol: str) -> float:
        """Price movement per 1 pip."""
        return self._specs.get(symbol, {}).get("pip_size", 0.0001)

    def contract_size(self, symbol: str) -> float:
        """Standard lot size."""
        return self._specs.get(symbol, {}).get("contract_size", 100000.0)

    def mt5_name(self, symbol: str) -> Optional[str]:
        """Actual MT5 symbol name used."""
        return self._mt5_map.get(symbol)

    def is_loaded(self, symbol: str) -> bool:
        return symbol in self._specs

    def apply_to_config(self):
        """
        Override config.py SPREAD_PIPS and PIP_VALUE with live MT5 values.
        Call once on startup — affects all training and backtesting.
        """
        import config as cfg
        for sym, vals in self._specs.items():
            cfg.SPREAD_PIPS[sym] = round(vals["spread"],   4)
            cfg.PIP_VALUE[sym]   = round(vals["pip_size"], 6)
        logger.info(
            f"[SymbolSpecs] Config updated from MT5 ({self._broker})  "
            f"symbols={list(self._specs.keys())}"
        )

    def apply_to_backtest(self, backtest_module):
        """Override PIP_USD_PER_LOT and SPREAD_PIPS in a loaded backtest module."""
        for sym, vals in self._specs.items():
            if hasattr(backtest_module, "PIP_USD_PER_LOT"):
                backtest_module.PIP_USD_PER_LOT[sym] = vals["pip_usd"]
            if hasattr(backtest_module, "SPREAD_PIPS"):
                backtest_module.SPREAD_PIPS[sym]     = vals["spread"]
        logger.info("[SymbolSpecs] Backtest module updated from MT5")

    def to_broker_config_block(self, broker_name: str = "AutoDetected") -> str:
        """Return a broker_config.py-compatible block ready to paste."""
        lines = [f'    "{broker_name}": {{']
        for sym, vals in self._specs.items():
            lines.append(
                f'        "{sym}": {{'
                f'"pip_usd": {vals["pip_usd"]}, '
                f'"spread": {vals["spread"]}, '
                f'"pip_size": {vals["pip_size"]}'
                f'}},'
            )
        lines.append("    },")
        return "\n".join(lines)

    def print_table(self):
        """Print detected specs as a rich table."""
        from rich.table import Table
        from rich.console import Console
        from rich import print as rprint

        console = Console()
        rprint(f"\n[bold cyan]Symbol Specs — {self._broker}[/bold cyan]\n")

        table = Table(show_lines=True)
        table.add_column("Symbol",        style="cyan",   justify="left")
        table.add_column("MT5 Name",      style="dim",    justify="left")
        table.add_column("pip_usd",       style="yellow", justify="right")
        table.add_column("Spread (pips)", style="white",  justify="right")
        table.add_column("pip_size",      style="white",  justify="right")
        table.add_column("Lot Size",      style="white",  justify="right")
        table.add_column("Status",        style="bold",   justify="center")

        for sym in self._symbols:
            if sym in self._specs:
                v = self._specs[sym]
                table.add_row(
                    sym,
                    self._mt5_map.get(sym, "?"),
                    f"{v['pip_usd']:.4f}",
                    f"{v['spread']:.2f}",
                    f"{v['pip_size']:.6f}",
                    f"{v['contract_size']:,.0f}",
                    "[green][OK][/green]",
                )
            else:
                table.add_row(sym, "-", "-", "-", "-", "-", "[red][X] not found[/red]")

        console.print(table)

    # ── Fetch from MT5 ────────────────────────────────────────────────────────
    def _fetch_all(self):
        try:
            import MetaTrader5 as mt5
        except ImportError:
            logger.warning("[SymbolSpecs] MetaTrader5 not installed — falling back to broker_config.py")
            self._fallback_to_broker_config()
            return

        kwargs = {}
        if self._settings:
            if self._settings.mt5_path:
                kwargs["path"] = self._settings.mt5_path
                
        if not mt5.initialize(**kwargs):
            logger.warning(f"[SymbolSpecs] MT5 not running ({mt5.last_error()}) — falling back to broker_config.py")
            self._fallback_to_broker_config()
            return
            
        if self._settings and self._settings.mt5_login:
            mt5.login(
                self._settings.mt5_login,
                password=self._settings.mt5_password or "",
                server=self._settings.mt5_server or ""
            )

        account = mt5.account_info()
        if account:
            self._currency = account.currency
            self._broker   = account.company
            logger.info(
                f"[SymbolSpecs] Connected  broker={account.company}  "
                f"login={account.login}  currency={account.currency}"
            )

        loaded = 0
        for sym in self._symbols:
            for mt5_name in MT5_SYMBOLS.get(sym, [sym]):
                info = mt5.symbol_info(mt5_name)
                if info is None:
                    continue
                if not info.visible:
                    mt5.symbol_select(mt5_name, True)
                    info = mt5.symbol_info(mt5_name)
                parsed = self._parse(info, sym)
                if parsed:
                    self._specs[sym]   = parsed
                    self._mt5_map[sym] = mt5_name
                    loaded += 1
                    logger.debug(
                        f"[SymbolSpecs] {sym} ({mt5_name})  "
                        f"pip_usd={parsed['pip_usd']:.4f}  "
                        f"spread={parsed['spread']:.2f}pips  "
                        f"pip_size={parsed['pip_size']}"
                    )
                    break
            else:
                logger.warning(f"[SymbolSpecs] {sym} not found in MT5")

        mt5.shutdown()
        logger.info(f"[SymbolSpecs] Loaded {loaded}/{len(self._symbols)} symbols")

    def _parse(self, info, sym: str) -> Optional[dict]:
        """Compute pip_usd and pip_size from raw MT5 symbol_info."""
        try:
            from config import PIP_VALUE
            
            point         = info.point
            digits        = info.digits
            tick_size     = info.trade_tick_size
            tick_value    = info.trade_tick_value
            contract_size = info.trade_contract_size

            # Use PIP_VALUE from config if defined, else fallback to standard MT5 heuristic
            pip_size = PIP_VALUE.get(sym)
            if pip_size is None:
                pip_size = point * 10 if digits in (3, 5) else point

            # pip_usd = USD per pip per standard lot
            pip_usd = (tick_value * pip_size / tick_size) if tick_size > 0 else 0.0

            # spread in pips
            conversion_ratio = pip_size / point if point > 0 else 1.0
            spread_pips = (info.spread / conversion_ratio) if pip_size > 0 else info.spread

            return {
                "pip_usd":       round(pip_usd,     6),
                "spread":        round(spread_pips, 2),
                "pip_size":      pip_size,
                "contract_size": contract_size,
                "digits":        digits,
                "point":         point,
                "tick_size":     tick_size,
                "tick_value":    tick_value,
            }
        except Exception as e:
            logger.warning(f"[SymbolSpecs] Parse error: {e}")
            return None

    def _live_spread(self, symbol: str) -> Optional[float]:
        """Fetch current live spread from MT5."""
        try:
            import MetaTrader5 as mt5
            mt5_name = self._mt5_map.get(symbol)
            if not mt5_name:
                return None
            info     = mt5.symbol_info(mt5_name)
            pip_size = self._specs.get(symbol, {}).get("pip_size", 0.0001)
            if info and pip_size > 0:
                return round(info.spread * info.point / pip_size, 2)
        except Exception:
            pass
        return None

    def _fallback_to_broker_config(self):
        """Use broker_config.py when MT5 unavailable."""
        try:
            from broker_config import get_broker_specs, ACTIVE_BROKER
            self._broker = ACTIVE_BROKER
            for sym, vals in get_broker_specs().items():
                self._specs[sym] = {
                    "pip_usd":       vals["pip_usd"],
                    "spread":        vals["spread"],
                    "pip_size":      vals["pip_size"],
                    "contract_size": 100000.0,
                }
            logger.info(
                f"[SymbolSpecs] Fallback: {len(self._specs)} symbols "
                f"from broker_config ({ACTIVE_BROKER})"
            )
        except Exception as e:
            logger.warning(f"[SymbolSpecs] broker_config fallback failed: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Singleton
# ─────────────────────────────────────────────────────────────────────────────
_instance: Optional[SymbolSpecs] = None

def get_specs(symbols: list = None, settings=None) -> SymbolSpecs:
    global _instance
    if _instance is None:
        _instance = SymbolSpecs(symbols, settings=settings)
    return _instance

def reset_specs():
    global _instance
    _instance = None


def patch_live_trader_pip_usd(live_trader_module=None):
    """
    Patch PIP_USD_PER_LOT in live_trader.py module at runtime
    with values read from live MT5 connection.

    Call this once after SymbolSpecs loads:
        from live.symbol_specs import get_specs, patch_live_trader_pip_usd
        specs = get_specs()
        patch_live_trader_pip_usd()
    """
    specs = get_specs()
    if not specs._specs:
        logger.warning("[SymbolSpecs] No specs loaded — cannot patch live_trader")
        return

    try:
        if live_trader_module is None:
            import live.live_trader as live_trader_module

        patched = []
        for sym, vals in specs._specs.items():
            if sym in live_trader_module.PIP_USD_PER_LOT:
                old_val = live_trader_module.PIP_USD_PER_LOT[sym]
                new_val = vals["pip_usd"]
                if abs(old_val - new_val) > 0.001:
                    live_trader_module.PIP_USD_PER_LOT[sym] = new_val
                    patched.append(f"{sym}: {old_val:.4f}→{new_val:.4f}")

        if patched:
            logger.info(f"[SymbolSpecs] PIP_USD_PER_LOT patched: {patched}")
        else:
            logger.info("[SymbolSpecs] PIP_USD_PER_LOT already correct — no changes")

    except Exception as e:
        logger.warning(f"[SymbolSpecs] Could not patch live_trader: {e}")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Fetch symbol specs from MT5")
    parser.add_argument("--symbols", nargs="+", default=None)
    parser.add_argument("--save",    action="store_true",
                        help="Print broker_config.py block to save permanently")
    parser.add_argument("--broker",  type=str, default="AutoDetected",
                        help="Broker name for --save output")
    args = parser.parse_args()

    specs = SymbolSpecs(args.symbols)
    specs.print_table()

    if args.save:
        print(f"\n# Paste into BROKERS dict in broker_config.py:\n")
        print(specs.to_broker_config_block(args.broker))
