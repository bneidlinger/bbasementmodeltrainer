#!/usr/bin/env python
"""Test the kill functionality"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'trainer'))

from db import ExperimentDB

def test_kill():
    db = ExperimentDB()
    
    # Get all runs
    all_runs = db.get_runs(limit=10)
    print(f"\nTotal runs in database: {len(all_runs)}")
    
    for run in all_runs:
        print(f"  Run #{run['id']}: {run['model']} on {run['dataset']} - Status: {run['status']}")
    
    # Get stuck runs
    stuck_runs = db.get_runs(status='running')
    print(f"\nStuck runs (status='running'): {len(stuck_runs)}")
    
    if stuck_runs:
        print("\nStuck runs details:")
        for run in stuck_runs:
            print(f"  Run #{run['id']}: {run['model']} on {run['dataset']}")
            print(f"    Started: {run['timestamp']}")
            print(f"    Status: {run['status']}")
            print(f"    Notes: {run.get('notes', 'None')}")
        
        # Test updating one
        test_id = stuck_runs[0]['id']
        print(f"\nTesting update on run #{test_id}...")
        
        try:
            db.update_run(test_id, status='failed', notes='Test kill')
            print(f"  ✓ Successfully updated run #{test_id}")
            
            # Verify the update
            updated_run = db.get_run(test_id)
            print(f"  Verification: Run #{test_id} status is now '{updated_run['status']}'")
            
        except Exception as e:
            print(f"  ✗ Error updating run: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("\nNo stuck runs to test with.")
        print("All runs have status: completed or failed")

if __name__ == "__main__":
    test_kill()