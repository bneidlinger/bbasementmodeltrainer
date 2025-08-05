"""
Utility script to clean up failed training runs and zombie processes.
"""

import os
import sys
import psutil
import time

# Add trainer to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'trainer'))

from trainer.db import ExperimentDB

def cleanup_zombie_processes():
    """Find and terminate zombie Python processes from failed training runs."""
    print("Checking for zombie training processes...")
    
    current_pid = os.getpid()
    killed_count = 0
    
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            # Look for Python processes
            if proc.info['name'] and 'python' in proc.info['name'].lower():
                cmdline = proc.info.get('cmdline', [])
                
                # Check if it's a training worker process
                if any('train_worker.py' in str(arg) for arg in cmdline):
                    if proc.pid != current_pid:
                        print(f"Found training process PID {proc.pid}")
                        proc.terminate()
                        killed_count += 1
                        
                        # Wait a bit and force kill if needed
                        time.sleep(0.5)
                        if proc.is_running():
                            proc.kill()
                            print(f"Force killed process {proc.pid}")
                        else:
                            print(f"Terminated process {proc.pid}")
                            
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    
    if killed_count > 0:
        print(f"\nKilled {killed_count} zombie training process(es)")
    else:
        print("No zombie training processes found")
    
    return killed_count

def fix_database_status():
    """Fix any training runs stuck in 'running' status."""
    print("\nChecking database for stuck runs...")
    
    db = ExperimentDB()
    
    # Get all runs with 'running' status
    with db.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, model, dataset, timestamp 
            FROM runs 
            WHERE status = 'running'
            ORDER BY timestamp DESC
        """)
        stuck_runs = cursor.fetchall()
    
    if not stuck_runs:
        print("No stuck runs found in database")
        return 0
    
    print(f"\nFound {len(stuck_runs)} run(s) stuck in 'running' status:")
    for run in stuck_runs:
        print(f"  Run {run['id']}: {run['model']} on {run['dataset']} (started {run['timestamp']})")
    
    # Ask for confirmation
    response = input("\nMark these as 'failed'? (y/n): ")
    
    if response.lower() == 'y':
        for run in stuck_runs:
            db.update_run(
                run['id'], 
                status='failed', 
                notes='Marked as failed by cleanup script - process likely crashed'
            )
        print(f"Updated {len(stuck_runs)} run(s) to 'failed' status")
        return len(stuck_runs)
    else:
        print("No changes made to database")
        return 0

def cleanup_orphaned_models():
    """Check for model files without corresponding database entries."""
    print("\nChecking for orphaned model files...")
    
    models_dir = os.path.join(os.path.dirname(__file__), 'saved_models')
    
    if not os.path.exists(models_dir):
        print("No saved_models directory found")
        return
    
    db = ExperimentDB()
    
    # Get all model files
    model_files = [f for f in os.listdir(models_dir) if f.endswith('.pth')]
    
    orphaned = []
    for model_file in model_files:
        # Extract run_id from filename (run_X_model.pth)
        if model_file.startswith('run_'):
            try:
                run_id = int(model_file.split('_')[1])
                run = db.get_run(run_id)
                if not run:
                    orphaned.append(model_file)
            except (IndexError, ValueError):
                orphaned.append(model_file)
    
    if orphaned:
        print(f"\nFound {len(orphaned)} orphaned model file(s):")
        for f in orphaned:
            print(f"  {f}")
        
        response = input("\nDelete orphaned files? (y/n): ")
        if response.lower() == 'y':
            for f in orphaned:
                os.remove(os.path.join(models_dir, f))
                # Also remove _full.pth version if it exists
                full_path = f.replace('.pth', '_full.pth')
                if os.path.exists(os.path.join(models_dir, full_path)):
                    os.remove(os.path.join(models_dir, full_path))
            print(f"Deleted {len(orphaned)} orphaned file(s)")
    else:
        print("No orphaned model files found")

def main():
    print("ModelBuilder Cleanup Utility")
    print("=" * 50)
    
    # 1. Kill zombie processes
    killed = cleanup_zombie_processes()
    
    # 2. Fix database status
    fixed = fix_database_status()
    
    # 3. Clean up orphaned models
    cleanup_orphaned_models()
    
    print("\n" + "=" * 50)
    print("Cleanup complete!")
    
    if killed > 0 or fixed > 0:
        print("\nRecommendation: Restart ModelBuilder for a clean state")

if __name__ == "__main__":
    # Check if psutil is installed
    try:
        import psutil
    except ImportError:
        print("Installing psutil for process management...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "psutil"])
        import psutil
    
    main()