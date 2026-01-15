"""
GEE Task Watchdog
- Monitors running NDVI tasks in Google Earth Engine
- Logs failures and completions to data/gee_tasks/task_failures.log
- Can be launched directly or from a parent script
"""

import ee
import time
import sys
import os
import subprocess
from pathlib import Path
from datetime import datetime
import psutil
import traceback

CHECK_INTERVAL = 60  # seconds

# === Determine script and project directories ===
SCRIPT_DIR = Path(__file__).resolve().parent  # Tools/
PROJECT_DIR = SCRIPT_DIR.parent                # Parent folder (where project.py lives)
DATA_DIR = SCRIPT_DIR / 'data' / 'gee_tasks'
DATA_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = DATA_DIR / 'task_failures.log'
PID_FILE = DATA_DIR / 'watchdog.pid'


# === Watchdog helpers ===
def is_watchdog_running():
    """Check if watchdog is already running using PID file."""
    if not PID_FILE.exists():
        return False
    try:
        pid = int(PID_FILE.read_text().strip())
        return psutil.pid_exists(pid)
    except Exception:
        return False


def launch_watchdog():
    """
    Launch this script in a new console window.
    Safe to call from project.py or from the Tools folder directly.
    """
    if is_watchdog_running():
        print("Watchdog already running.")
        return False

    WATCHDOG_SCRIPT = Path(__file__).resolve()

    if sys.platform == 'win32':
        subprocess.Popen(
            [sys.executable, str(WATCHDOG_SCRIPT), '--monitor'],
            creationflags=subprocess.CREATE_NEW_CONSOLE,
            cwd=str(WATCHDOG_SCRIPT.parent)
        )
    else:
        subprocess.Popen(
            [sys.executable, str(WATCHDOG_SCRIPT), '--monitor'],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            cwd=str(WATCHDOG_SCRIPT.parent)
        )

    time.sleep(2)
    print("Watchdog launched in new window.")
    return True


# === Monitoring function ===
def monitor_tasks():
    """Monitor GEE NDVI tasks and log failures/completions."""
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    PID_FILE.write_text(str(os.getpid()))

    try:
        # Initialize GEE
        try:
            ee.Initialize(project='ee-litepc')
        except Exception as e:
            err_text = f"Failed to initialize GEE: {e}\n{traceback.format_exc()}"
            print(err_text)
            with open(LOG_FILE, 'a') as f:
                f.write(f"{datetime.now().isoformat()},ERROR,GEE_INIT,{err_text}\n")
            return

        # Load previously seen tasks
        seen_tasks = set()
        if LOG_FILE.exists():
            with open(LOG_FILE, 'r') as f:
                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) >= 4:
                        seen_tasks.add(parts[3])
            print(f"Loaded {len(seen_tasks)} previously seen tasks from log")

        print(f"Watchdog started (PID {os.getpid()})")

        # Show currently running NDVI tasks
        try:
            tasks = ee.data.getTaskList()
            running_tasks = [t for t in tasks if t.get('state') == 'RUNNING' and t.get('description','').startswith('NDVI_')]
            if running_tasks:
                print(f"Currently running tasks ({len(running_tasks)}):")
                for t in running_tasks:
                    print(f"  ▶ {t.get('description', 'Unknown')}")
            else:
                print("No NDVI tasks currently running")
        except Exception as e:
            print(f"Could not fetch initial task status: {e}")

        # Main loop
        while True:
            try:
                tasks = ee.data.getTaskList()
            except Exception as e:
                err_text = f"Failed to fetch tasks: {e}\n{traceback.format_exc()}"
                print(err_text)
                with open(LOG_FILE, 'a') as f:
                    f.write(f"{datetime.now().isoformat()},ERROR,TASK_FETCH,{err_text}\n")
                time.sleep(CHECK_INTERVAL)
                continue

            for task in tasks:
                task_id = task.get('id')
                if task_id in seen_tasks:
                    continue

                state = task.get('state')
                desc = task.get('description','Unknown')
                if not desc.startswith('NDVI_'):
                    continue

                timestamp = datetime.now().isoformat()

                if state in ['FAILED','CANCELLED']:
                    error = task.get('error_message','No error message')
                    with open(LOG_FILE,'a') as f:
                        f.write(f"{timestamp},{desc},{state},{task_id},{error}\n")
                    print(f"✗ {desc}: {state} - {error}")
                    seen_tasks.add(task_id)

                elif state == 'COMPLETED':
                    with open(LOG_FILE,'a') as f:
                        f.write(f"{timestamp},{desc},{state},{task_id},\n")
                    print(f"✓ {desc}: COMPLETED")
                    seen_tasks.add(task_id)

            time.sleep(CHECK_INTERVAL)

    finally:
        if PID_FILE.exists():
            PID_FILE.unlink()


# === Entry point ===
if __name__ == '__main__':
    # Auto-launch in new console if not monitoring
    if '--monitor' not in sys.argv:
        launch_watchdog()
        sys.exit(0)

    # Otherwise, run the monitoring loop
    print("Starting GEE Task Watchdog (CTRL+C to exit)...")
    monitor_tasks()
