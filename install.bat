@echo off
:: ─────────────────────────────────────────────────────────────────────────────
:: install.bat  —  One-click installer for Always-In Trading Bot (Windows)
:: Works with Python 3.11, 3.12, and 3.13
:: ─────────────────────────────────────────────────────────────────────────────

echo.
echo ============================================================
echo   Always-In Bot  —  Dependency Installer
echo ============================================================
echo.

:: Step 1: Upgrade pip to latest
echo [1/3] Upgrading pip...
python -m pip install --upgrade pip --quiet
if %errorlevel% neq 0 (
    echo ERROR: pip upgrade failed. Is Python installed and on your PATH?
    pause
    exit /b 1
)
echo       Done.
echo.

:: Step 2: Install all packages except torch
echo [2/3] Installing core packages (numpy, pandas, sklearn, SB3, yfinance...)
echo       This may take a few minutes on first run.
echo.
pip install -r requirements.txt --quiet
if %errorlevel% neq 0 (
    echo.
    echo ERROR: Some packages failed to install.
    echo        Check the output above for details.
    pause
    exit /b 1
)
echo.
echo       Core packages installed.
echo.

:: Step 3: Install PyTorch (CPU build — no CUDA needed)
echo [3/3] Installing PyTorch (CPU build)...
echo       Downloading from pytorch.org — may take a few minutes.
echo.
pip install torch --index-url https://download.pytorch.org/whl/cpu --quiet
if %errorlevel% neq 0 (
    echo.
    echo ERROR: PyTorch install failed.
    echo        Try manually:  pip install torch --index-url https://download.pytorch.org/whl/cpu
    pause
    exit /b 1
)
echo.
echo       PyTorch installed.
echo.

:: Verify key imports
echo Verifying installation...
python -c "import numpy, pandas, torch, gymnasium, stable_baselines3, MetaTrader5, loguru, dateutil; print('  All imports OK')"
if %errorlevel% neq 0 (
    echo.
    echo WARNING: One or more imports failed — check the output above.
) else (
    echo.
    echo ============================================================
    echo   Installation complete!
    echo.
    echo   Next step — fetch data from your open MT5 terminal:
    echo     python data\mt5_fetcher.py
    echo.
    echo   Then train the bot:
    echo     python train.py
    echo ============================================================
)
echo.
pause
