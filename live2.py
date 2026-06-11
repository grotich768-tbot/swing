import os
import sys
from loguru import logger

# Force the Terminal UI to be disabled for this simple runner
os.environ["TERMINAL_UI_ENABLED"] = "false"

import live_run

# Custom logging override to keep the console clean and silent
def silent_setup_logging(level, to_file, rotation, ui=None):
    logger.remove()
    
    def log_filter(record):
        msg = record["message"]
        # Hide spammy setup logs
        if "LSTM device:" in msg: return False
        if "No LSTM found" in msg: return False
        if "RegimeRouter loaded" in msg: return False
        if "Ensemble loaded" in msg: return False
        if "Single model loaded" in msg: return False
        if "Loaded .env" in msg: return False
        return True

    # Add console logger with our filter
    logger.add(
        sys.stdout,
        level=level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level:<7}</level> | <level>{message}</level>",
        colorize=True,
        filter=log_filter
    )
    
    # Still write absolutely everything to the log file for safety
    if to_file:
        from pathlib import Path
        log_dir = Path(__file__).parent / "logs"
        log_dir.mkdir(exist_ok=True)
        logger.add(str(log_dir / "live_{time}.log"), rotation=rotation, level="DEBUG")

# Inject our silent logger into the run process
live_run.setup_logging = silent_setup_logging

if __name__ == "__main__":
    """
    Simple no-UI wrapper to launch the live trading bot for Terminal 2 (Crypto/Valetax).
    Simply run: python live2.py
    """
    # Automatically force the terminal argument to "2"
    if "--terminal" not in sys.argv:
        sys.argv.extend(["--terminal", "2"])
        
    sys.exit(live_run.main())
