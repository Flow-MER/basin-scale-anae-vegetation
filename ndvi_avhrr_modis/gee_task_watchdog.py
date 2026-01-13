"""
GEE Task Monitor - Watches running tasks and logs failures
Auto-launched by main NDVI script or run standalone.
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

# Absolute paths
BASE_DIR = Path(__file__).parent.resolve()
LOG_FILE = BASE_DIR / 'task_failures.log'
PID_FILE = BASE_DIR / 'watchdog.pid'
CHECK_INTERVAL = 60  # seconds


def is_watchdog_running():
    """Check if watchdog is already running."""
    if not PID_FILE.exists():
        return False
    try:
        pid = int(PID_FILE.read_text().strip())
        return psutil.pid_exists(pid)
    except Exception:
        return False


def launch_watchdog():
    """Launch watchdog in new console window."""
    if is_watchdog_running():
        return False

    if sys.platform == 'win32':
        subprocess.Popen(
            [sys.executable, str(BASE_DIR / 'gee_task_watchdog.py'), '--monitor'],
            creationflags=subprocess.CREATE_NEW_CONSOLE,
            cwd=str(BASE_DIR)
        )
    else:
        subprocess.Popen(
            [sys.executable, str(BASE_DIR / 'gee_task_watchdog.py'), '--monitor'],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            cwd=str(BASE_DIR)
        )

    time.sleep(2)
    return True


def monitor_tasks():
    """Monitor GEE tasks and log failures."""
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

    # Write PID file
    PID_FILE.write_text(str(os.getpid()))

    try:
        try:
            ee.Initialize(project='ee-litepc')
        except Exception as e:
            err_text = f"Failed to initialize GEE: {e}\n{traceback.format_exc()}"
            print(err_text)
            with open(LOG_FILE, 'a') as f:
                f.write(f"{datetime.now().isoformat()},ERROR,GEE_INIT,{err_text}\n")
            return  # Cannot continue without EE

        # Load previously seen tasks from log
        seen_tasks = set()
        if LOG_FILE.exists():
            with open(LOG_FILE, 'r') as f:
                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) >= 4:
                        task_id = parts[3]  # task_id is 4th column
                        seen_tasks.add(task_id)
            print(f"Loaded {len(seen_tasks)} previously seen tasks from log")
        
        print(f"Watchdog started (PID: {os.getpid()})")
        
        # Show currently running tasks on startup
        try:
            tasks = ee.data.getTaskList()
            running_tasks = [t for t in tasks if t.get('state') == 'RUNNING' and t.get('description', '').startswith('NDVI_')]
            if running_tasks:
                print(f"Currently running tasks ({len(running_tasks)}):")
                for task in running_tasks:
                    print(f"  ▶ {task.get('description', 'Unknown')}")
            else:
                print("No NDVI tasks currently running")
        except Exception as e:
            print(f"Could not fetch initial task status: {e}")

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
                desc = task.get('description', 'Unknown')

                if not desc.startswith('NDVI_'):
                    continue

                if state in ['FAILED', 'CANCELLED']:
                    error = task.get('error_message', 'No error message')
                    timestamp = datetime.now().isoformat()
                    with open(LOG_FILE, 'a') as f:
                        f.write(f"{timestamp},{desc},{state},{task_id},{error}\n")
                    print(f"✗ {desc}: {state} - {error}")
                    seen_tasks.add(task_id)

                elif state == 'COMPLETED':
                    timestamp = datetime.now().isoformat()
                    with open(LOG_FILE, 'a') as f:
                        f.write(f"{timestamp},{desc},{state},{task_id},\n")
                    print(f"✓ {desc}: COMPLETED")
                    seen_tasks.add(task_id)

            time.sleep(CHECK_INTERVAL)

    finally:
        if PID_FILE.exists():
            PID_FILE.unlink()


if __name__ == '__main__':
    # Check if this is a direct launch (not already in new window)
    if '--monitor' not in sys.argv:
        if is_watchdog_running():
            print("Watchdog is already running.")
            sys.exit(0)
        # Relaunch in new window and exit
        print("Launching watchdog in new window...")
        launch_watchdog()
        sys.exit(0)
    
    # This is the monitoring process
    print("Starting GEE task watchdog (CTRL+C to exit)...")
    monitor_tasks()
