#!/usr/bin/env python
"""
Force clean all stuck runs directly in the database.
Simple and direct approach.
"""

import sqlite3
import os
from datetime import datetime

def force_clean_runs():
    """Directly update all running status to failed in database."""
    
    db_path = "experiments.db"
    
    if not os.path.exists(db_path):
        print("Database not found!")
        return
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # First, check what we have
    cursor.execute("SELECT id, model, dataset, status FROM runs WHERE status = 'running'")
    stuck_runs = cursor.fetchall()
    
    if not stuck_runs:
        print("✓ No stuck runs found")
        conn.close()
        return
    
    print(f"\nFound {len(stuck_runs)} stuck runs:")
    for run_id, model, dataset, status in stuck_runs:
        print(f"  Run #{run_id}: {model} on {dataset} - {status}")
    
    response = input("\n[?] Force clean all these runs? (yes/no): ")
    if response.lower() not in ['yes', 'y']:
        print("Cancelled.")
        conn.close()
        return
    
    # Update all running to failed
    cursor.execute("""
        UPDATE runs 
        SET status = 'failed', 
            notes = 'Force cleaned - stuck process' 
        WHERE status = 'running'
    """)
    
    affected = cursor.rowcount
    conn.commit()
    conn.close()
    
    print(f"\n✓ Force cleaned {affected} stuck runs")
    print("Database updated successfully!")

if __name__ == "__main__":
    force_clean_runs()