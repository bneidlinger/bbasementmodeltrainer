import sqlite3
import os
from datetime import datetime
from contextlib import contextmanager
from typing import Dict, List, Optional, Any

class ExperimentDB:
    def __init__(self, db_path: str = "experiments.db"):
        self.db_path = db_path
        self.init_database()
    
    @contextmanager
    def get_connection(self):
        """Context manager for database connections."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Enable column access by name
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()
    
    def init_database(self):
        """Initialize database tables if they don't exist."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Create runs table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT DEFAULT (datetime('now')),
                    model TEXT NOT NULL,
                    dataset TEXT NOT NULL,
                    epochs INTEGER NOT NULL,
                    batch_size INTEGER NOT NULL,
                    lr REAL NOT NULL,
                    optimizer TEXT DEFAULT 'Adam',
                    train_loss REAL,
                    val_loss REAL,
                    val_acc REAL,
                    best_val_acc REAL,
                    training_time REAL,  -- in seconds
                    status TEXT DEFAULT 'running',  -- running, completed, failed
                    notes TEXT,
                    config TEXT  -- JSON string for additional config
                )
            """)
            
            # Create datasets table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS datasets (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE NOT NULL,
                    source TEXT,  -- builtin, kaggle, openml, custom
                    sha1 TEXT,
                    file_size INTEGER,  -- in bytes
                    rows INTEGER,
                    features INTEGER,
                    classes INTEGER,
                    license TEXT,
                    path TEXT,
                    downloaded_at TEXT DEFAULT (datetime('now'))
                )
            """)
            
            # Create metrics table for tracking training progress
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id INTEGER NOT NULL,
                    epoch INTEGER NOT NULL,
                    train_loss REAL,
                    val_loss REAL,
                    val_acc REAL,
                    learning_rate REAL,
                    timestamp TEXT DEFAULT (datetime('now')),
                    FOREIGN KEY (run_id) REFERENCES runs(id)
                )
            """)
            
            # Create index for faster queries
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_metrics_run_id 
                ON metrics(run_id)
            """)
    
    def create_run(self, model: str, dataset: str, epochs: int, 
                   batch_size: int, lr: float, optimizer: str = "Adam",
                   notes: str = "", config: Dict[str, Any] = None) -> int:
        """Create a new training run and return its ID."""
        import json
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO runs (model, dataset, epochs, batch_size, lr, 
                                optimizer, notes, config, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'running')
            """, (model, dataset, epochs, batch_size, lr, optimizer, 
                  notes, json.dumps(config) if config else None))
            return cursor.lastrowid
    
    def update_run(self, run_id: int, **kwargs):
        """Update run information."""
        allowed_fields = {
            'train_loss', 'val_loss', 'val_acc', 'best_val_acc',
            'training_time', 'status', 'notes'
        }
        
        # Filter out non-allowed fields
        updates = {k: v for k, v in kwargs.items() if k in allowed_fields}
        
        if not updates:
            return
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            set_clause = ", ".join([f"{k} = ?" for k in updates.keys()])
            values = list(updates.values()) + [run_id]
            
            cursor.execute(f"""
                UPDATE runs 
                SET {set_clause}
                WHERE id = ?
            """, values)
    
    def add_metric(self, run_id: int, epoch: int, train_loss: float = None,
                   val_loss: float = None, val_acc: float = None, 
                   learning_rate: float = None):
        """Add a metric entry for a specific epoch."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO metrics (run_id, epoch, train_loss, val_loss, 
                                   val_acc, learning_rate)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (run_id, epoch, train_loss, val_loss, val_acc, learning_rate))
    
    def get_run(self, run_id: int) -> Optional[Dict[str, Any]]:
        """Get a specific run by ID."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM runs WHERE id = ?", (run_id,))
            row = cursor.fetchone()
            return dict(row) if row else None
    
    def get_runs(self, limit: int = 50, offset: int = 0, status: str = None) -> List[Dict[str, Any]]:
        """Get recent runs with pagination and optional status filter."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if status:
                cursor.execute("""
                    SELECT * FROM runs 
                    WHERE status = ?
                    ORDER BY timestamp DESC 
                    LIMIT ? OFFSET ?
                """, (status, limit, offset))
            else:
                cursor.execute("""
                    SELECT * FROM runs 
                    ORDER BY timestamp DESC 
                    LIMIT ? OFFSET ?
                """, (limit, offset))
            return [dict(row) for row in cursor.fetchall()]
    
    def get_metrics(self, run_id: int) -> List[Dict[str, Any]]:
        """Get all metrics for a specific run."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM metrics 
                WHERE run_id = ? 
                ORDER BY epoch
            """, (run_id,))
            return [dict(row) for row in cursor.fetchall()]
    
    def add_dataset(self, name: str, source: str, path: str, 
                    sha1: str = None, file_size: int = None,
                    rows: int = None, features: int = None, 
                    classes: int = None, license: str = None):
        """Add or update dataset information."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO datasets 
                (name, source, sha1, file_size, rows, features, classes, 
                 license, path)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (name, source, sha1, file_size, rows, features, classes, 
                  license, path))
    
    def get_dataset(self, name: str) -> Optional[Dict[str, Any]]:
        """Get dataset information by name."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM datasets WHERE name = ?", (name,))
            row = cursor.fetchone()
            return dict(row) if row else None
    
    def get_datasets(self) -> List[Dict[str, Any]]:
        """Get all cached datasets."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM datasets ORDER BY downloaded_at DESC")
            return [dict(row) for row in cursor.fetchall()]


# Convenience functions for common operations
def create_experiment_db(db_path: str = "experiments.db") -> ExperimentDB:
    """Create and return an ExperimentDB instance."""
    return ExperimentDB(db_path)


if __name__ == "__main__":
    # Test the database
    db = create_experiment_db("test_experiments.db")
    
    # Create a test run
    run_id = db.create_run(
        model="ResNet18",
        dataset="CIFAR-10",
        epochs=10,
        batch_size=32,
        lr=0.001,
        notes="Test run"
    )
    
    print(f"Created run with ID: {run_id}")
    
    # Add some metrics
    for epoch in range(3):
        db.add_metric(
            run_id=run_id,
            epoch=epoch,
            train_loss=0.5 - epoch * 0.1,
            val_loss=0.6 - epoch * 0.1,
            val_acc=0.7 + epoch * 0.05
        )
    
    # Update the run
    db.update_run(run_id, status="completed", best_val_acc=0.85)
    
    # Retrieve and display
    run = db.get_run(run_id)
    print(f"\nRun details: {run}")
    
    metrics = db.get_metrics(run_id)
    print(f"\nMetrics: {metrics}")
    
    # Clean up test file
    import os
    os.remove("test_experiments.db")