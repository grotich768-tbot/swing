#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# install.sh  —  Installer for Always-In Bot (Linux / GitHub Codespaces)
# ─────────────────────────────────────────────────────────────────────────────
set -e

echo ""
echo "============================================================"
echo "  Always-In Bot  —  Dependency Installer (Linux/Codespaces)"
echo "============================================================"
echo ""

# Step 1: Upgrade pip
echo "[1/3] Upgrading pip..."
python -m pip install --upgrade pip -q
echo "      Done."
echo ""

# Step 2: Core packages (strip MetaTrader5 — Linux only)
echo "[2/3] Installing core packages..."
grep -v "MetaTrader5" requirements.txt > /tmp/requirements_linux.txt
pip install -r /tmp/requirements_linux.txt -q
echo "      Done."
echo ""

# Step 3: PyTorch CPU
echo "[3/3] Installing PyTorch (CPU)..."
pip install torch --index-url https://download.pytorch.org/whl/cpu -q
echo "      Done."
echo ""

# Verify
echo "Verifying imports..."
python -c "import numpy, pandas, torch, gymnasium, stable_baselines3, loguru; print('  All imports OK')"

echo ""
echo "============================================================"
echo "  Installation complete!"
echo ""
echo "  To train (yfinance fetches GOLD/SILVER automatically):"
echo "    python train.py --timesteps 50000"
echo "============================================================"
echo ""
