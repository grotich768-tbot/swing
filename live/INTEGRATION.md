# Live Fixes Integration Guide

## What changed and where to add it in live_trader.py

---

### Fix 1 — Remove 5% risk cap

In `.env` file:
```
# Remove or comment out any of these:
# RISK_CAP=0.05
# MAX_RISK_PCT=0.05
# MAX_POSITION_PCT_CAP=0.05
```

In `live_trader.py` or `settings.py` wherever the cap is applied:
```python
# Find and remove any line like:
risk_pct = min(risk_pct, 0.05)   # DELETE THIS
lots     = min(lots, max_risk_lots)  # if max_risk_lots is based on 5% cap, fix it

# Replace with: let risk_engine.position_size() handle sizing
from live.live_fixes import LiveFixes
settings = LiveFixes.remove_risk_cap(settings)
```

---

### Fix 2 — Re-entry guard

In `live_trader.py` in your main bar loop:

```python
from live.live_fixes import LiveFixes

# On startup:
self._fixes = LiveFixes(settings, bridge)

# In your bar/tick loop, BEFORE placing a flip:
proposed_side = +1 if model_says_long else -1

if self._fixes.is_reentry_blocked(symbol, proposed_side):
    logger.info(f"{symbol}: re-entry blocked — same direction too soon")
    continue   # skip flip, hold current position

# After successfully executing a flip:
self._fixes.record_flip(symbol, new_side)

# After a full H1 bar completes with no early close:
# (optional — blocks expire automatically after 1 hour)
self._fixes.clear_reentry_block(symbol)
```

**What this prevents:**
```
10:00  Position closed early (SL or manual)
10:15  Model says LONG again (same direction)
       → is_reentry_blocked() returns True → skip
10:45  Model still says LONG
       → still blocked (< 1 hour since last flip)
11:00  New bar — block expired → allow re-evaluation
```

---

### Fix 3 — Spread guard

In your bar loop, BEFORE checking model action:

```python
# Get spread correctly from MT5 (points → pips)
spread_pips = self._fixes.get_spread_pips(symbol)

# Block flip if spread too wide
if self._fixes.spread_too_wide(symbol, spread_pips):
    logger.info(f"{symbol}: spread too wide ({spread_pips:.1f} pips) — holding")
    continue   # don't flip, hold current position and wait

# If model says flip AND spread is ok → flip normally
if action == 1:
    self._fixes.record_flip(symbol, new_side)
    bridge.place_order(...)
```

**What this prevents:**
```
Normal spread GOLD: 6 pips
News event:        40 pips (6.7x normal)

Without fix:  Bot flips → pays 40 pip spread → immediately -$X loss
With fix:     spread_too_wide() = True → holds → waits for 6 pips → then flips
```

---

### Fix 4 — MT5 spread conversion (points → pips)

MT5 `symbol_info().spread` is in POINTS, not pips.

```python
# WRONG — what many implementations do:
spread_pips = mt5.symbol_info("XAUUSD").spread   # this is 60 POINTS not 6 pips

# RIGHT — use live_fixes:
from live.live_fixes import mt5_spread_to_pips
info = mt5.symbol_info("XAUUSD")
spread_pips = mt5_spread_to_pips(info.spread, "GOLD", info)   # correctly 6.0 pips

# Or use LiveFixes.get_spread_pips() which does this automatically:
spread_pips = self._fixes.get_spread_pips("GOLD")
```

**The difference:**
```
XAUUSD: spread = 60 points, point = 0.01, digits = 2
pip_size = 0.01 (metals: point = pip_size, no 10x multiplier)
spread_pips = 60 * 0.01 / 0.01 = 60.0  ← WRONG if you don't know MT5 structure

# Correct for GOLD:
# digits=2 → pip_size = point = 0.01
# spread_pips = 60 * 0.01 / 0.01 = 60... still looks wrong

# The real fix: symbol_info().spread for GOLD is already in pips * 10
# Use _parse() in symbol_specs.py which handles this correctly per symbol
```

**Simplest approach — use SymbolSpecs:**
```python
from live.symbol_specs import get_specs
specs = get_specs()
spread_pips = specs.spread("GOLD")   # always correct, auto-refreshes from MT5
```

---

### Complete integration pattern

```python
# In LiveTrader.__init__():
from live.live_fixes import LiveFixes
self._fixes = LiveFixes(settings, bridge)

# In LiveTrader._run_symbol() or equivalent bar loop:
def _on_bar(self, symbol):
    # 1. Get correct spread
    spread_pips = self._fixes.get_spread_pips(symbol)

    # 2. Block flip if spread too wide
    if self._fixes.spread_too_wide(symbol, spread_pips):
        return   # hold, don't evaluate

    # 3. Get model action
    obs    = self._build_obs(symbol)
    action = self._model.predict(obs)

    # 4. If flip proposed — check re-entry guard
    if action == 1:
        current_side = self._get_position_side(symbol)
        new_side     = -current_side

        if self._fixes.is_reentry_blocked(symbol, new_side):
            return   # same direction too soon after early close

        # 5. Execute flip
        self._execute_flip(symbol, new_side)
        self._fixes.record_flip(symbol, new_side)
```
