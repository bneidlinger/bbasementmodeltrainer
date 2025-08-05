import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import traceback
from typing import Dict, Any, Optional
import multiprocessing as mp
from queue import Empty
import json
import os

def train_loop(config: Dict[str, Any], queue: mp.Queue):
    """
    Training worker process that runs the PyTorch training loop.
    
    Args:
        config: Configuration dictionary containing:
            - model_class: The model class to instantiate
            - model_params: Parameters for model initialization
            - dataset: Dataset object
            - epochs: Number of epochs
            - batch_size: Batch size
            - lr: Learning rate
            - optimizer_name: Name of optimizer
            - device: 'cuda' or 'cpu'
            - run_id: Database run ID
            - db_path: Path to database
        queue: Multiprocessing queue for communication with main process
    """
    try:
        # Import here to avoid issues with multiprocessing
        from db import ExperimentDB
        import numpy as np
        
        # Extract configuration
        model_class = config['model_class']
        model_params = config.get('model_params', {})
        train_dataset = config['train_dataset']
        val_dataset = config.get('val_dataset')
        epochs = config['epochs']
        batch_size = config['batch_size']
        lr = config['lr']
        optimizer_name = config.get('optimizer_name', 'Adam')
        device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        run_id = config.get('run_id')
        db_path = config.get('db_path', 'experiments.db')
        
        # Initialize database connection
        db = ExperimentDB(db_path) if run_id else None
        
        # Send initial status
        queue.put({
            'type': 'status',
            'message': f'Initializing training on {device}...'
        })
        
        # Create model
        model = model_class(**model_params).to(device)
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=0  # Avoid multiprocessing issues on Windows
        )
        
        val_loader = None
        if val_dataset:
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0
            )
        
        # Create loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        
        if optimizer_name == 'Adam':
            optimizer = optim.Adam(model.parameters(), lr=lr)
        elif optimizer_name == 'SGD':
            optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        elif optimizer_name == 'AdamW':
            optimizer = optim.AdamW(model.parameters(), lr=lr)
        else:
            optimizer = optim.Adam(model.parameters(), lr=lr)
        
        # Training metrics
        best_val_acc = 0.0
        start_time = time.time()
        
        # Send ready status
        queue.put({
            'type': 'status',
            'message': 'Training started!'
        })
        
        # Training loop
        for epoch in range(epochs):
            # Check for stop signal
            try:
                signal = queue.get_nowait()
                if signal.get('type') == 'stop':
                    queue.put({
                        'type': 'status',
                        'message': 'Training stopped by user'
                    })
                    break
            except Empty:
                pass
            
            # Training phase
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_idx, (inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += targets.size(0)
                train_correct += predicted.eq(targets).sum().item()
                
                # Send batch update every 10 batches
                if batch_idx % 10 == 0:
                    queue.put({
                        'type': 'batch_update',
                        'epoch': epoch,
                        'batch': batch_idx,
                        'total_batches': len(train_loader),
                        'loss': loss.item()
                    })
            
            # Calculate training metrics
            avg_train_loss = train_loss / len(train_loader)
            train_acc = 100. * train_correct / train_total
            
            # Validation phase
            val_loss = 0.0
            val_acc = 0.0
            
            if val_loader:
                model.eval()
                val_correct = 0
                val_total = 0
                
                with torch.no_grad():
                    for inputs, targets in val_loader:
                        inputs, targets = inputs.to(device), targets.to(device)
                        outputs = model(inputs)
                        loss = criterion(outputs, targets)
                        
                        val_loss += loss.item()
                        _, predicted = outputs.max(1)
                        val_total += targets.size(0)
                        val_correct += predicted.eq(targets).sum().item()
                
                val_loss = val_loss / len(val_loader)
                val_acc = 100. * val_correct / val_total
                
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
            
            # Send epoch update
            epoch_data = {
                'type': 'epoch_complete',
                'epoch': epoch,
                'train_loss': avg_train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss if val_loader else None,
                'val_acc': val_acc if val_loader else None,
                'lr': optimizer.param_groups[0]['lr']
            }
            queue.put(epoch_data)
            
            # Update database if available
            if db and run_id:
                db.add_metric(
                    run_id=run_id,
                    epoch=epoch,
                    train_loss=avg_train_loss,
                    val_loss=val_loss if val_loader else None,
                    val_acc=val_acc/100.0 if val_loader else None,
                    learning_rate=optimizer.param_groups[0]['lr']
                )
        
        # Training complete
        training_time = time.time() - start_time
        
        # Save the trained model
        model_save_path = None
        try:
            # Create models directory if it doesn't exist
            models_dir = os.path.join(os.path.dirname(db_path) if db_path else '.', 'saved_models')
            os.makedirs(models_dir, exist_ok=True)
            
            # Save model checkpoint
            model_filename = f"run_{run_id}_model.pth" if run_id else f"model_{int(time.time())}.pth"
            model_save_path = os.path.join(models_dir, model_filename)
            
            # Save model state dict along with metadata
            torch.save({
                'model_state_dict': model.state_dict(),
                'model_class': model.__class__.__name__,
                'model_params': model_params,
                'dataset_name': config.get('dataset_name', 'unknown'),
                'num_classes': model_params.get('num_classes', model_params.get('output_size', 10)),
                'input_shape': config.get('input_shape'),
                'best_val_acc': best_val_acc,
                'final_train_loss': avg_train_loss,
                'final_val_loss': val_loss if val_loader else None,
                'epochs_trained': epoch + 1,
                'optimizer_name': optimizer_name,
                'learning_rate': lr,
                'run_id': run_id
            }, model_save_path)
            
            # Also save a full model for easier loading
            full_model_path = model_save_path.replace('.pth', '_full.pth')
            torch.save(model, full_model_path)
            
        except Exception as e:
            print(f"Error saving model: {e}")
            model_save_path = None
        
        # Update database with final results
        if db and run_id:
            db.update_run(
                run_id,
                status='completed',
                train_loss=avg_train_loss,
                val_loss=val_loss if val_loader else None,
                val_acc=val_acc/100.0 if val_loader else None,
                best_val_acc=best_val_acc/100.0 if val_loader else None,
                training_time=training_time,
                notes=f"Model saved to: {model_save_path}" if model_save_path else None
            )
        
        # Send completion message
        queue.put({
            'type': 'complete',
            'message': 'Training completed successfully!',
            'training_time': training_time,
            'best_val_acc': best_val_acc if val_loader else None,
            'model_path': model_save_path
        })
        
    except Exception as e:
        # Send error message
        error_msg = f"Training error: {str(e)}\n{traceback.format_exc()}"
        queue.put({
            'type': 'error',
            'message': error_msg
        })
        
        # Update database status if available
        try:
            if 'db' in locals() and 'run_id' in locals() and db and run_id:
                db.update_run(run_id, status='failed', notes=f"Error: {str(e)[:200]}")
        except:
            pass  # Don't let database errors prevent error reporting


def create_dummy_dataset(num_samples=1000, input_size=784, num_classes=10):
    """Create a dummy dataset for testing."""
    import torch.utils.data as data
    
    class DummyDataset(data.Dataset):
        def __init__(self, num_samples, input_size, num_classes):
            self.data = torch.randn(num_samples, input_size)
            self.targets = torch.randint(0, num_classes, (num_samples,))
            
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            return self.data[idx], self.targets[idx]
    
    return DummyDataset(num_samples, input_size, num_classes)


if __name__ == "__main__":
    # Test the training worker
    from models.mlp import MLP
    
    # Create a queue for communication
    queue = mp.Queue()
    
    # Create dummy datasets
    train_dataset = create_dummy_dataset(1000, 784, 10)
    val_dataset = create_dummy_dataset(200, 784, 10)
    
    # Configuration
    config = {
        'model_class': MLP,
        'model_params': {'input_size': 784, 'hidden_size': 128, 'output_size': 10},
        'train_dataset': train_dataset,
        'val_dataset': val_dataset,
        'epochs': 3,
        'batch_size': 32,
        'lr': 0.001,
        'optimizer_name': 'Adam',
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    # Start training in a separate process
    process = mp.Process(target=train_loop, args=(config, queue))
    process.start()
    
    # Monitor the queue
    while True:
        try:
            message = queue.get(timeout=1)
            print(f"[{message['type']}] {message}")
            
            if message['type'] in ['complete', 'error']:
                break
                
        except Empty:
            pass
    
    process.join()
    print("Test completed!")