#!/usr/bin/env python
"""
Kill all stuck training runs in the database.
Useful for cleaning up after crashes or force-quits.
"""

import os
import sys
import signal
from datetime import datetime

try:
    import psutil  # type: ignore
except Exception:  # pragma: no cover
    psutil = None  # type: ignore

# Add trainer to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'trainer'))

from db import ExperimentDB


def kill_stuck_runs(auto_mode: bool = False, db_path: str = "experiments.db"):
    """Kill all training runs marked as 'running' in the database."""
    db = ExperimentDB(db_path)

    # Get all running runs
    runs = db.get_runs(status='running')

    if not runs:
        print("✓ No stuck runs found in database")
        return

    print(f"\n⚠️  Found {len(runs)} stuck training runs:")
    print("-" * 60)

    for run in runs:
        run_id = run['id']
        timestamp = run['timestamp']
        model = run['model']
        dataset = run['dataset']

        # Check how old the run is
        try:
            run_time = datetime.fromisoformat(timestamp)
            age = datetime.now() - run_time
            age_str = (
                f"{age.days}d {age.seconds//3600}h"
                if age.days > 0
                else f"{age.seconds//3600}h {(age.seconds%3600)//60}m"
            )
        except Exception:
            age_str = "unknown"

        print(f"  Run #{run_id}: {model} on {dataset}")
        print(f"    Started: {timestamp}")
        print(f"    Age: {age_str}")

    print("-" * 60)

    if not auto_mode:
        response = input("\n[?] Kill all stuck runs? (yes/no): ")
        if response.lower() not in ['yes', 'y']:
            print("Cancelled.")
            return

    # Kill all stuck runs
    print("\n[*] Killing stuck runs...")

    for run in runs:
        run_id = run['id']
        pid = run.get('pid')
        try:
            if pid:
                if psutil:
                    try:
                        proc = psutil.Process(pid)
                        proc.terminate()
                        try:
                            proc.wait(timeout=1)
                        except psutil.TimeoutExpired:
                            proc.kill()
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
                else:
                    try:
                        os.kill(pid, signal.SIGTERM)
                    except OSError:
                        pass

            # Update database status and clear pid
            db.update_run(run_id,
                         status='failed',
                         notes='Killed by cleanup script - process was stuck',
                         pid=None)
            print(f"  ✓ Killed run #{run_id}")
        except Exception as e:
            print(f"  ✗ Error killing run #{run_id}: {e}")

    print("\n✓ Cleanup complete!")

    # Also try to kill any orphaned Python processes running train_worker.py
    if psutil:
        print("\n[*] Checking for orphaned training processes...")
        killed_processes = 0

        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                cmdline = proc.info.get('cmdline', [])
                if cmdline and any('train_worker.py' in arg for arg in cmdline):
                    print(f"  Found orphaned process PID {proc.info['pid']}")
                    proc.terminate()
                    killed_processes += 1
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        if killed_processes > 0:
            print(f"  ✓ Terminated {killed_processes} orphaned processes")
        else:
            print("  ✓ No orphaned processes found")
    else:
        print("\n[!] psutil not available - skipping orphaned process check")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='Kill stuck training runs')
    parser.add_argument('--auto', '-a', action='store_true',
                       help='Automatically kill all stuck runs without confirmation')
    parser.add_argument('--dry-run', '-d', action='store_true',
                       help='Show what would be killed without actually doing it')
    parser.add_argument('--db-path', default='experiments.db',
                       help='Path to experiments database')

    args = parser.parse_args()

    if args.dry_run:
        print("DRY RUN MODE - No changes will be made")
        db = ExperimentDB(args.db_path)
        runs = db.get_runs(status='running')

        if not runs:
            print("No stuck runs found")
        else:
            print(f"Would kill {len(runs)} stuck runs:")
            for run in runs:
                print(f"  - Run #{run['id']}: {run['model']} on {run['dataset']}")
    else:
        kill_stuck_runs(auto_mode=args.auto, db_path=args.db_path)


if __name__ == "__main__":
    main()
