"""
Automatic cleanup script for failed runs (non-interactive).
"""

import os
import sys

# Add trainer to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'trainer'))

from trainer.db import ExperimentDB

def auto_fix_database():
    """Automatically fix stuck runs."""
    db = ExperimentDB()
    
    # Update all 'running' status to 'failed'
    with db.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE runs 
            SET status = 'failed', 
                notes = 'Marked as failed by auto-cleanup - process likely crashed'
            WHERE status = 'running'
        """)
        affected = cursor.rowcount
    
    print(f"Fixed {affected} stuck run(s)")
    return affected

if __name__ == "__main__":
    print("Running automatic cleanup...")
    auto_fix_database()
    print("Done!")