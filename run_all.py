import subprocess
import sys
import threading
import time

def run_bot(terminal_id, prefix):
    # Construct the command
    cmd = [sys.executable, "live_run.py"]
    if terminal_id == "2":
        cmd.extend(["--terminal", "2"])
        
    print(f"\n{prefix} Starting Bot for Terminal {terminal_id}...\n")
    
    # Start the process
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True
    )
    
    # Read and print the output in real-time
    for line in iter(process.stdout.readline, ''):
        # Print with a prefix so we know which bot is talking
        print(f"{prefix} {line}", end="")
        sys.stdout.flush()
        
    process.wait()

if __name__ == "__main__":
    print("============================================================")
    print("   MULTI-TERMINAL RUNNER (Trading Both Brokers at Once)")
    print("============================================================")
    
    # Create two separate threads so both bots run simultaneously in the background
    t1 = threading.Thread(target=run_bot, args=("", "[TERM 1 - TPXM]"))
    t2 = threading.Thread(target=run_bot, args=("2", "[TERM 2 - VALETAX]"))
    
    t1.start()
    
    # Give Terminal 1 a few seconds to print its table cleanly before Terminal 2 starts
    time.sleep(5) 
    
    t2.start()
    
    # Keep the main script alive while the bots run
    try:
        t1.join()
        t2.join()
    except KeyboardInterrupt:
        print("\nShutting down both bots...")
