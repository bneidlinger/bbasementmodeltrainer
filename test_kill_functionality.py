"""Tests for killing stuck training runs."""

import os
import sys
import subprocess

# Add trainer to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'trainer'))

from db import ExperimentDB
from kill_stuck_runs import kill_stuck_runs


def test_kill_stuck_run_updates_db_and_terminates(tmp_path):
    """Ensure kill_stuck_runs terminates processes and updates database."""
    db_path = tmp_path / "test_experiments.db"
    db = ExperimentDB(str(db_path))

    # Create a dummy running process and corresponding DB entry
    run_id = db.create_run("TestModel", "TestData", epochs=1, batch_size=1, lr=0.001)
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    db.update_run(run_id, pid=proc.pid)

    # Kill stuck runs
    kill_stuck_runs(auto_mode=True, db_path=str(db_path))

    # Verify process was terminated
    proc.wait(timeout=1)
    assert proc.poll() is not None

    # Verify database status updated
    run = db.get_run(run_id)
    assert run["status"] == "failed"
    assert run["pid"] is None

