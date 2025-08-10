import dearpygui.dearpygui as dpg
import torch
import multiprocessing as mp
from queue import Empty
import threading
import time
import os
import sys
from typing import Dict, Any, Optional
import psutil

# Add the trainer directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from db import ExperimentDB
from train_worker import train_loop
from models import MODEL_REGISTRY
from data.core import DATASET_REGISTRY
from test_ui import TestingUI
from model_comparison_ui import ModelComparisonUI
from llm_ui import LLMTrainingUI
from retro_theme import create_retro_theme, create_button_themes, COLORS
from ascii_blocks import ASCII_BOLD_BLOCKS, ASCII_MATRIX_BLOCKS, ASCII_PRO_BLOCKS

class ModelBuilderApp:
    def __init__(self):
        self.db = ExperimentDB()
        self.training_process: Optional[mp.Process] = None
        self.message_queue: Optional[mp.Queue] = None
        self.monitor_thread: Optional[threading.Thread] = None
        self.is_training = False
        
        # Training metrics for plotting
        self.epochs = []
        self.train_losses = []
        self.val_losses = []
        self.val_accs = []
        
        # GUI element tags
        self.tags = {
            'dataset_combo': 'dataset_combo',
            'model_combo': 'model_combo',
            'epochs_input': 'epochs_input',
            'batch_size_input': 'batch_size_input',
            'lr_input': 'lr_input',
            'optimizer_combo': 'optimizer_combo',
            'gpu_check': 'gpu_check',
            'start_button': 'start_button',
            'stop_button': 'stop_button',
            'progress_bar': 'progress_bar',
            'status_text': 'status_text',
            'loss_plot': 'loss_plot',
            'loss_x_axis': 'main_loss_x_axis',
            'loss_y_axis': 'main_loss_y_axis',
            'acc_plot': 'acc_plot',
            'acc_x_axis': 'main_acc_x_axis',
            'acc_y_axis': 'main_acc_y_axis',
            'train_loss_series': 'train_loss_series',
            'val_loss_series': 'val_loss_series',
            'val_acc_series': 'val_acc_series',
            'history_table': 'history_table'
        }
        
    def setup_gui(self):
        """Initialize the Dear PyGui interface."""
        dpg.create_context()
        
        # Set up retro theme
        global_theme = create_retro_theme()
        self.button_themes = create_button_themes()
                
        # Create main window
        with dpg.window(label="B's BasementBrewAI - Industrial ML Terminal", tag="main_window"):
            # Add ASCII title
            with dpg.group(horizontal=False):
                # ASCII art header - using professional blocks style
                lines = ASCII_PRO_BLOCKS.strip().split('\n')
                for i, line in enumerate(lines):
                    if line.strip():  # Only show non-empty lines
                        # Color scheme: borders green, blocks orange, text green
                        if '-' in line and not '+' in line:
                            dpg.add_text(line, color=COLORS['green_dim'])
                        elif '#' in line:
                            # Main block letters in orange
                            dpg.add_text(line, color=COLORS['orange_bright'])
                        else:
                            # Regular text in green
                            dpg.add_text(line, color=COLORS['green_normal'])
                
                dpg.add_separator()
                dpg.add_spacer(height=5)
                
                # Create tab bar
                with dpg.tab_bar(label="MainTabs"):
                    # Training tab
                    with dpg.tab(label="> TRAINING"):
                        self.create_training_tab()
                    
                    # Testing tab  
                    with dpg.tab(label="* TESTING"):
                        self.testing_ui = TestingUI(self)
                        self.testing_ui.create_ui(dpg.last_item())
                    
                    # Comparison tab
                    with dpg.tab(label="[] COMPARE"):
                        self.comparison_ui = ModelComparisonUI(self)
                        self.comparison_ui.create_ui(dpg.last_item())
                    
                    # LLM tab
                    with dpg.tab(label="@ LLM LAB"):
                        self.llm_ui = LLMTrainingUI(self)
                        self.llm_ui.create_ui(dpg.last_item())
        
        # Set up viewport
        dpg.create_viewport(title="B's BasementBrewAI - Industrial ML Terminal v1.0", width=1300, height=900)
        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.set_primary_window("main_window", True)
        dpg.bind_theme(global_theme)
    
    def create_training_tab(self):
        """Create the training interface tab."""
        with dpg.group():
            with dpg.group(horizontal=True):
                # Left panel - Configuration
                with dpg.child_window(width=300, height=-1):
                    dpg.add_text("[ CONFIGURATION ]", color=COLORS['orange_bright'])
                    dpg.add_separator()
                    
                    # Dataset selection
                    dpg.add_text("Dataset:", color=COLORS['green_normal'])
                    datasets = list(DATASET_REGISTRY.keys())
                    dpg.add_combo(
                        datasets, 
                        default_value=datasets[0] if datasets else "",
                        tag=self.tags['dataset_combo'],
                        width=-1
                    )
                    
                    # Model selection
                    dpg.add_text("Model:", color=COLORS['green_normal'])
                    models = list(MODEL_REGISTRY.keys())
                    dpg.add_combo(
                        models,
                        default_value=models[0] if models else "",
                        tag=self.tags['model_combo'],
                        width=-1
                    )
                    
                    dpg.add_separator()
                    
                    # Hyperparameters
                    dpg.add_text("[ HYPERPARAMETERS ]", color=COLORS['orange_bright'])
                    
                    dpg.add_text("Epochs:", color=COLORS['green_normal'])
                    dpg.add_input_int(
                        default_value=10,
                        min_value=1,
                        max_value=1000,
                        tag=self.tags['epochs_input'],
                        width=-1
                    )
                    
                    dpg.add_text("Batch Size:", color=COLORS['green_normal'])
                    dpg.add_input_int(
                        default_value=32,
                        min_value=1,
                        max_value=512,
                        tag=self.tags['batch_size_input'],
                        width=-1
                    )
                    
                    dpg.add_text("Learning Rate:", color=COLORS['green_normal'])
                    dpg.add_input_float(
                        default_value=0.001,
                        min_value=0.00001,
                        max_value=1.0,
                        format="%.5f",
                        tag=self.tags['lr_input'],
                        width=-1
                    )
                    
                    dpg.add_text("Optimizer:", color=COLORS['green_normal'])
                    dpg.add_combo(
                        ["Adam", "SGD", "AdamW"],
                        default_value="Adam",
                        tag=self.tags['optimizer_combo'],
                        width=-1
                    )
                    
                    dpg.add_separator()
                    
                    # GPU selection
                    gpu_available = torch.cuda.is_available()
                    dpg.add_checkbox(
                        label=f"Use GPU {'(Available)' if gpu_available else '(Not Available)'}",
                        default_value=gpu_available,
                        tag=self.tags['gpu_check'],
                        enabled=gpu_available
                    )
                    
                    dpg.add_separator()
                    
                    # Control buttons
                    start_btn = dpg.add_button(
                        label="> START TRAINING",
                        callback=self.start_training,
                        tag=self.tags['start_button'],
                        width=-1,
                        height=35
                    )
                    dpg.bind_item_theme(start_btn, self.button_themes['start'])
                    
                    stop_btn = dpg.add_button(
                        label="[] STOP TRAINING",
                        callback=self.stop_training,
                        tag=self.tags['stop_button'],
                        width=-1,
                        height=35,
                        enabled=False
                    )
                    dpg.bind_item_theme(stop_btn, self.button_themes['stop'])
                    
                    dpg.add_separator()
                    
                    # Status
                    dpg.add_text("[ STATUS ]", color=COLORS['orange_bright'])
                    dpg.add_text("System Ready", tag=self.tags['status_text'], wrap=280, color=COLORS['green_bright'])
                    
                    # Progress bar
                    dpg.add_progress_bar(
                        default_value=0.0,
                        tag=self.tags['progress_bar'],
                        width=-1
                    )
                
                # Right panel - Metrics and visualization
                with dpg.child_window(width=-1, height=-1):
                    dpg.add_text("[ TRAINING METRICS ]", color=COLORS['orange_bright'])
                    dpg.add_separator()
                    
                    # Create plots
                    with dpg.plot(label="Loss", height=250, width=-1, tag=self.tags['loss_plot']):
                        dpg.add_plot_legend()
                        dpg.add_plot_axis(dpg.mvXAxis, label="Epoch", tag=self.tags['loss_x_axis'])
                        dpg.add_plot_axis(dpg.mvYAxis, label="Loss", tag=self.tags['loss_y_axis'])
                        
                        dpg.add_line_series([], [], 
                                          label="Train Loss",
                                          parent=self.tags['loss_y_axis'],
                                          tag=self.tags['train_loss_series'])
                        dpg.add_line_series([], [], 
                                          label="Val Loss",
                                          parent=self.tags['loss_y_axis'],
                                          tag=self.tags['val_loss_series'])
                    
                    with dpg.plot(label="Validation Accuracy", height=250, width=-1, tag=self.tags['acc_plot']):
                        dpg.add_plot_legend()
                        dpg.add_plot_axis(dpg.mvXAxis, label="Epoch", tag=self.tags['acc_x_axis'])
                        dpg.add_plot_axis(dpg.mvYAxis, label="Accuracy (%)", tag=self.tags['acc_y_axis'])
                        
                        dpg.add_line_series([], [],
                                          label="Val Accuracy",
                                          parent=self.tags['acc_y_axis'],
                                          tag=self.tags['val_acc_series'])
                    
                    dpg.add_separator()
                    
                    # Training history button
                    dpg.add_button(
                        label="View Training History",
                        callback=self.show_history_modal,
                        width=200
                    )
    
    def start_training(self):
        """Start the training process."""
        if self.is_training:
            return
        
        # Get configuration from GUI
        dataset_name = dpg.get_value(self.tags['dataset_combo'])
        model_name = dpg.get_value(self.tags['model_combo'])
        epochs = dpg.get_value(self.tags['epochs_input'])
        batch_size = dpg.get_value(self.tags['batch_size_input'])
        lr = dpg.get_value(self.tags['lr_input'])
        optimizer_name = dpg.get_value(self.tags['optimizer_combo'])
        use_gpu = dpg.get_value(self.tags['gpu_check'])
        
        # Validate inputs
        if not dataset_name or not model_name:
            dpg.set_value(self.tags['status_text'], "[WARNING] Dataset and model selection required")
            return
        
        # Clear previous metrics
        self.epochs.clear()
        self.train_losses.clear()
        self.val_losses.clear()
        self.val_accs.clear()
        self.update_plots()
        
        # Load dataset
        dpg.set_value(self.tags['status_text'], "[LOADING] Initializing dataset module...")
        dataset_class = DATASET_REGISTRY[dataset_name]
        dataset_loader = dataset_class()
        
        try:
            train_dataset, val_dataset = dataset_loader.load()
            dataset_info = dataset_loader.get_info()
        except Exception as e:
            dpg.set_value(self.tags['status_text'], f"[ERROR] Dataset module failure: {str(e)}")
            return
        
        # Create model
        model_class = MODEL_REGISTRY[model_name]
        
        # Determine model parameters based on dataset
        model_params = {}
        if hasattr(model_class, 'get_config_options'):
            config_options = model_class.get_config_options()
            
            # Auto-configure based on dataset info
            if 'input_size' in config_options:
                input_shape = dataset_info.get('input_shape', (784,))
                if len(input_shape) == 1:
                    model_params['input_size'] = int(input_shape[0])
                else:
                    model_params['input_size'] = int(np.prod(input_shape))
            
            if 'in_channels' in config_options:
                input_shape = dataset_info.get('input_shape', (3, 32, 32))
                if len(input_shape) == 3:
                    model_params['in_channels'] = int(input_shape[0])
            
            if 'num_classes' in config_options:
                model_params['num_classes'] = int(dataset_info.get('num_classes', 10))
            
            if 'output_size' in config_options:
                model_params['output_size'] = int(dataset_info.get('num_classes', 10))
        
        # Create database entry
        run_id = self.db.create_run(
            model=model_name,
            dataset=dataset_name,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            optimizer=optimizer_name,
            config=model_params
        )
        self.current_run_id = run_id  # Store for timeout handling
        
        # Save dataset info to database
        self.db.add_dataset(
            name=dataset_name,
            source=dataset_info.get('source', 'unknown'),
            path=dataset_loader.root,
            rows=dataset_info.get('train_samples', 0),
            features=np.prod(dataset_info.get('input_shape', [0])),
            classes=dataset_info.get('num_classes', 0),
            license=dataset_info.get('license', 'unknown')
        )
        
        # Prepare configuration for training worker
        config = {
            'model_class': model_class,
            'model_params': model_params,
            'train_dataset': train_dataset,
            'val_dataset': val_dataset,
            'epochs': epochs,
            'batch_size': batch_size,
            'lr': lr,
            'optimizer_name': optimizer_name,
            'device': 'cuda' if use_gpu and torch.cuda.is_available() else 'cpu',
            'run_id': run_id,
            'db_path': self.db.db_path,
            'dataset_name': dataset_name,
            'input_shape': dataset_info.get('input_shape')
        }
        
        # Create communication queue
        self.message_queue = mp.Queue()
        
        # Start training process
        self.training_process = mp.Process(
            target=train_loop,
            args=(config, self.message_queue)
        )
        self.training_process.start()
        # Record process ID in database for later management
        self.db.update_run(run_id, pid=self.training_process.pid)
        
        # Start monitoring thread
        self.is_training = True
        self.monitor_thread = threading.Thread(target=self.monitor_training)
        self.monitor_thread.start()
        
        # Update GUI state
        dpg.configure_item(self.tags['start_button'], enabled=False)
        dpg.configure_item(self.tags['stop_button'], enabled=True)
        dpg.set_value(self.tags['status_text'], "[ACTIVE] Training process initialized...")
    
    def stop_training(self):
        """Stop the training process."""
        if not self.is_training:
            return
        
        # Send stop signal
        if self.message_queue:
            self.message_queue.put({'type': 'stop'})
        
        dpg.set_value(self.tags['status_text'], "[TERMINATING] Sending interrupt signal...")
        
        # Force cleanup after a timeout
        def force_cleanup():
            time.sleep(3)  # Give it 3 seconds to stop gracefully
            if self.training_process and self.training_process.is_alive():
                self.training_process.terminate()
                self.is_training = False
                dpg.set_value(self.tags['status_text'], "[TERMINATED] Process force-killed")
                dpg.configure_item(self.tags['start_button'], enabled=True)
                dpg.configure_item(self.tags['stop_button'], enabled=False)
        
        cleanup_thread = threading.Thread(target=force_cleanup)
        cleanup_thread.daemon = True
        cleanup_thread.start()
    
    def monitor_training(self):
        """Monitor training progress in a separate thread."""
        last_message_time = time.time()
        timeout_seconds = 300  # 5 minute timeout for no messages
        
        while self.is_training:
            try:
                message = self.message_queue.get(timeout=0.1)
                last_message_time = time.time()  # Reset timeout on any message
                
                if message['type'] == 'status':
                    dpg.set_value(self.tags['status_text'], message['message'])
                
                elif message['type'] == 'batch_update':
                    # Update progress bar
                    progress = (message['batch'] + 1) / message['total_batches']
                    overall_progress = (message['epoch'] + progress) / dpg.get_value(self.tags['epochs_input'])
                    dpg.set_value(self.tags['progress_bar'], overall_progress)
                
                elif message['type'] == 'epoch_complete':
                    # Update metrics
                    self.epochs.append(message['epoch'])
                    self.train_losses.append(message['train_loss'])
                    
                    if message['val_loss'] is not None:
                        self.val_losses.append(message['val_loss'])
                    if message['val_acc'] is not None:
                        self.val_accs.append(message['val_acc'])
                    
                    # Update plots
                    self.update_plots()
                    
                    # Update status
                    status = f"Epoch {message['epoch']+1}: "
                    status += f"Train Loss: {message['train_loss']:.4f}"
                    if message['val_loss'] is not None:
                        status += f", Val Loss: {message['val_loss']:.4f}"
                    if message['val_acc'] is not None:
                        status += f", Val Acc: {message['val_acc']:.2f}%"
                    dpg.set_value(self.tags['status_text'], status)
                
                elif message['type'] in ['complete', 'error']:
                    dpg.set_value(self.tags['status_text'], message['message'])
                    self.is_training = False
                    
                    # Reset GUI state
                    dpg.configure_item(self.tags['start_button'], enabled=True)
                    dpg.configure_item(self.tags['stop_button'], enabled=False)
                    
                    if message['type'] == 'complete':
                        dpg.set_value(self.tags['progress_bar'], 1.0)
                        
                        # Show training completion modal
                        self.show_training_complete_modal(message)
            
            except Empty:
                # Check for timeout
                if time.time() - last_message_time > timeout_seconds:
                    dpg.set_value(self.tags['status_text'], 
                                "Training timeout - no response from worker")
                    self.is_training = False
                    
                    # Update database
                    if hasattr(self, 'current_run_id'):
                        self.db.update_run(self.current_run_id, 
                                         status='failed', 
                                         notes='Training timeout - no response from worker')
                    
                    # Terminate the process
                    if self.training_process and self.training_process.is_alive():
                        self.training_process.terminate()
                    
                    # Reset GUI
                    dpg.configure_item(self.tags['start_button'], enabled=True)
                    dpg.configure_item(self.tags['stop_button'], enabled=False)
                    break
                    
            except Exception as e:
                print(f"Monitor error: {e}")
        
        # Clean up
        if self.training_process and self.training_process.is_alive():
            self.training_process.terminate()
            self.training_process.join()
    
    def update_plots(self):
        """Update the training plots."""
        if self.epochs:
            dpg.set_value(self.tags['train_loss_series'], [self.epochs, self.train_losses])
            
            if self.val_losses:
                dpg.set_value(self.tags['val_loss_series'], [self.epochs, self.val_losses])
            
            if self.val_accs:
                dpg.set_value(self.tags['val_acc_series'], [self.epochs, self.val_accs])
    
    def show_history_modal(self):
        """Show training history in a modal window."""
        if dpg.does_item_exist("history_modal"):
            dpg.delete_item("history_modal")
        
        with dpg.window(label="Training History", modal=True, tag="history_modal",
                       width=800, height=600, pos=[200, 100]):
            
            # Get recent runs
            runs = self.db.get_runs(limit=20)
            
            if not runs:
                dpg.add_text("No training runs found.")
                return
            
            # Create table
            with dpg.table(header_row=True, resizable=True, tag=self.tags['history_table'],
                          borders_innerH=True, borders_outerH=True, 
                          borders_innerV=True, borders_outerV=True):
                
                # Add columns
                dpg.add_table_column(label="ID")
                dpg.add_table_column(label="Timestamp")
                dpg.add_table_column(label="Model")
                dpg.add_table_column(label="Dataset")
                dpg.add_table_column(label="Epochs")
                dpg.add_table_column(label="Status")
                dpg.add_table_column(label="Best Val Acc")
                dpg.add_table_column(label="Action")  # New column for kill button
                
                # Add rows
                for run in runs:
                    with dpg.table_row():
                        dpg.add_text(str(run['id']))
                        dpg.add_text(run['timestamp'][:16])
                        dpg.add_text(run['model'])
                        dpg.add_text(run['dataset'])
                        dpg.add_text(str(run['epochs']))
                        
                        # Status with color
                        status = run['status']
                        color = (100, 255, 100) if status == 'completed' else \
                               (255, 100, 100) if status == 'failed' else \
                               (255, 255, 100)
                        dpg.add_text(status, color=color)
                        
                        # Best validation accuracy
                        best_acc = run.get('best_val_acc')
                        if best_acc is not None:
                            dpg.add_text(f"{best_acc*100:.2f}%")
                        else:
                            dpg.add_text("N/A")
                        
                        # Add kill button for running processes
                        if status == 'running':
                            dpg.add_button(
                                label="[X] KILL",
                                callback=lambda s, a, u: self.kill_training_run(u),
                                user_data=run['id'],
                                small=True
                            )
                        else:
                            dpg.add_text("")  # Empty cell for completed/failed runs
            
            # Button group at the bottom
            dpg.add_separator()
            with dpg.group(horizontal=True):
                dpg.add_button(label="Close", callback=lambda: dpg.delete_item("history_modal"))
                dpg.add_button(label="[R] Refresh", callback=self.refresh_history_table)
                
                # Check if there are any stuck runs
                stuck_runs = [r for r in runs if r['status'] == 'running']
                if stuck_runs:
                    dpg.add_button(
                        label=f"[X] Kill All Stuck ({len(stuck_runs)})",
                        callback=self.kill_all_stuck_runs,
                        small=False
                    )
    
    def kill_training_run(self, run_id: int):
        """Kill a stuck training run with confirmation."""
        # Show confirmation dialog
        if dpg.does_item_exist("kill_confirm_modal"):
            dpg.delete_item("kill_confirm_modal")
        
        with dpg.window(label="Confirm Kill", modal=True, tag="kill_confirm_modal",
                       width=400, height=200, pos=[450, 300]):
            dpg.add_text(f"Are you sure you want to kill training run #{run_id}?",
                        color=COLORS['orange_bright'])
            dpg.add_text("This action cannot be undone!", 
                        color=(255, 100, 100))
            dpg.add_spacer(height=20)
            
            with dpg.group(horizontal=True):
                dpg.add_button(
                    label="[X] YES, KILL IT",
                    callback=lambda: self.confirm_kill_run(run_id),
                    width=150
                )
                dpg.add_button(
                    label="Cancel",
                    callback=lambda: dpg.delete_item("kill_confirm_modal"),
                    width=150
                )
    
    def confirm_kill_run(self, run_id: int):
        """Actually kill the training run and update database."""
        try:
            print(f"Killing run {run_id}...")  # Debug

            run_info = self.db.get_run(run_id)
            pid = run_info.get('pid') if run_info else None

            # Update database status first and clear stored pid
            self.db.update_run(run_id,
                             status='failed',
                             notes='Manually killed by user',
                             pid=None)
            print(f"Database updated for run {run_id}")  # Debug

            # Then check if this is the current running process
            if hasattr(self, 'current_run_id') and self.current_run_id == run_id:
                print(f"Killing current process for run {run_id}")  # Debug
                # Kill the current training process
                if self.training_process and self.training_process.is_alive():
                    self.training_process.terminate()
                    # Don't wait too long
                    self.training_process.join(timeout=1)

                    # Force kill if still alive
                    if self.training_process.is_alive():
                        print(f"Force killing process for run {run_id}")  # Debug
                        self.training_process.kill()

                # Stop monitoring thread
                self.is_training = False

                # Reset GUI
                dpg.configure_item(self.tags['start_button'], enabled=True)
                dpg.configure_item(self.tags['stop_button'], enabled=False)
            elif pid:
                # Kill external process if pid is known
                try:
                    proc = psutil.Process(pid)
                    proc.terminate()
                    try:
                        proc.wait(timeout=1)
                    except psutil.TimeoutExpired:
                        proc.kill()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            
            # Close confirmation dialog
            if dpg.does_item_exist("kill_confirm_modal"):
                dpg.delete_item("kill_confirm_modal")
            
            # Refresh the history table
            self.refresh_history_table()
            
            # Show success message
            dpg.set_value(self.tags['status_text'], 
                         f"[TERMINATED] Run #{run_id} has been killed")
            print(f"Successfully killed run {run_id}")  # Debug
            
        except Exception as e:
            print(f"Error killing run {run_id}: {e}")
            import traceback
            traceback.print_exc()
            dpg.set_value(self.tags['status_text'], 
                         f"Error killing run #{run_id}: {str(e)}")
    
    def refresh_history_table(self):
        """Refresh the training history table."""
        # Clear and rebuild the table content
        if dpg.does_item_exist(self.tags['history_table']):
            dpg.delete_item(self.tags['history_table'], children_only=True)
            
            # Get fresh data
            runs = self.db.get_runs(limit=20)
            
            if runs:
                # Rebuild table columns
                dpg.add_table_column(label="ID", parent=self.tags['history_table'])
                dpg.add_table_column(label="Timestamp", parent=self.tags['history_table'])
                dpg.add_table_column(label="Model", parent=self.tags['history_table'])
                dpg.add_table_column(label="Dataset", parent=self.tags['history_table'])
                dpg.add_table_column(label="Epochs", parent=self.tags['history_table'])
                dpg.add_table_column(label="Status", parent=self.tags['history_table'])
                dpg.add_table_column(label="Best Val Acc", parent=self.tags['history_table'])
                dpg.add_table_column(label="Action", parent=self.tags['history_table'])
                
                # Rebuild rows
                for run in runs:
                    with dpg.table_row(parent=self.tags['history_table']):
                        dpg.add_text(str(run['id']))
                        dpg.add_text(run['timestamp'][:16])
                        dpg.add_text(run['model'])
                        dpg.add_text(run['dataset'])
                        dpg.add_text(str(run['epochs']))
                        
                        # Status with color
                        status = run['status']
                        color = (100, 255, 100) if status == 'completed' else \
                               (255, 100, 100) if status == 'failed' else \
                               (255, 255, 100)
                        dpg.add_text(status, color=color)
                        
                        # Best validation accuracy
                        best_acc = run.get('best_val_acc')
                        if best_acc is not None:
                            dpg.add_text(f"{best_acc*100:.2f}%")
                        else:
                            dpg.add_text("N/A")
                        
                        # Add kill button for running processes
                        if status == 'running':
                            dpg.add_button(
                                label="[X] KILL",
                                callback=lambda s, a, u: self.kill_training_run(u),
                                user_data=run['id'],
                                small=True
                            )
                        else:
                            dpg.add_text("")
    
    def kill_all_stuck_runs(self):
        """Kill all stuck training runs at once."""
        # Get all running runs
        stuck_runs = self.db.get_runs(status='running')
        
        if not stuck_runs:
            return
        
        # Show confirmation dialog
        if dpg.does_item_exist("kill_all_confirm_modal"):
            dpg.delete_item("kill_all_confirm_modal")
        
        with dpg.window(label="Confirm Kill All", modal=True, tag="kill_all_confirm_modal",
                       width=450, height=250, pos=[425, 275]):
            dpg.add_text(f"⚠️ Kill ALL {len(stuck_runs)} stuck training runs?",
                        color=COLORS['orange_bright'])
            dpg.add_separator()
            
            # List the runs that will be killed
            dpg.add_text("The following runs will be terminated:", 
                        color=COLORS['green_normal'])
            for run in stuck_runs[:5]:  # Show first 5
                dpg.add_text(f"  • Run #{run['id']}: {run['model']} on {run['dataset']}", 
                           color=(200, 200, 200))
            if len(stuck_runs) > 5:
                dpg.add_text(f"  ... and {len(stuck_runs) - 5} more", 
                           color=(200, 200, 200))
            
            dpg.add_spacer(height=10)
            dpg.add_text("This action cannot be undone!", 
                        color=(255, 100, 100))
            dpg.add_spacer(height=10)
            
            with dpg.group(horizontal=True):
                dpg.add_button(
                    label="[X] KILL ALL",
                    callback=lambda: self.confirm_kill_all_runs(stuck_runs),
                    width=150
                )
                dpg.add_button(
                    label="Cancel",
                    callback=lambda: dpg.delete_item("kill_all_confirm_modal"),
                    width=150
                )
    
    def confirm_kill_all_runs(self, stuck_runs):
        """Actually kill all stuck runs."""
        killed_count = 0

        for run in stuck_runs:
            try:
                run_id = run['id']
                run_info = self.db.get_run(run_id)
                pid = run_info.get('pid') if run_info else None

                # Check if this is the current running process
                if hasattr(self, 'current_run_id') and self.current_run_id == run_id:
                    # Kill the current training process
                    if self.training_process and self.training_process.is_alive():
                        self.training_process.terminate()
                        self.training_process.join(timeout=1)
                        if self.training_process.is_alive():
                            self.training_process.kill()

                    # Stop monitoring
                    self.is_training = False

                    # Reset GUI
                    dpg.configure_item(self.tags['start_button'], enabled=True)
                    dpg.configure_item(self.tags['stop_button'], enabled=False)
                elif pid:
                    try:
                        proc = psutil.Process(pid)
                        proc.terminate()
                        try:
                            proc.wait(timeout=1)
                        except psutil.TimeoutExpired:
                            proc.kill()
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass

                # Update database
                self.db.update_run(run_id,
                                 status='failed',
                                 notes='Batch killed - stuck process',
                                 pid=None)
                killed_count += 1
                
            except Exception as e:
                print(f"Error killing run {run['id']}: {e}")
        
        # Close confirmation dialog
        if dpg.does_item_exist("kill_all_confirm_modal"):
            dpg.delete_item("kill_all_confirm_modal")
        
        # Refresh history table
        self.refresh_history_table()
        
        # Show success message
        dpg.set_value(self.tags['status_text'], 
                     f"[TERMINATED] Killed {killed_count} stuck training runs")
    
    def show_training_complete_modal(self, message):
        """Show modal dialog when training completes with options to save or delete model."""
        if dpg.does_item_exist("training_complete_modal"):
            dpg.delete_item("training_complete_modal")
        
        # Extract info from message
        training_time = message.get('training_time', 0)
        best_val_acc = message.get('best_val_acc', None)
        model_path = message.get('model_path', None)
        
        # Format training time
        minutes = int(training_time // 60)
        seconds = int(training_time % 60)
        time_str = f"{minutes}m {seconds}s"
        
        with dpg.window(label="Training Complete!", modal=True, tag="training_complete_modal",
                       width=500, height=400, pos=[400, 200], no_close=True):
            
            # Success header
            dpg.add_text("[ TRAINING COMPLETED SUCCESSFULLY ]", color=COLORS['green_bright'])
            dpg.add_separator()
            dpg.add_spacer(height=10)
            
            # Training statistics
            dpg.add_text("Training Statistics:", color=COLORS['orange_bright'])
            dpg.add_spacer(height=5)
            
            # Model info
            model_name = dpg.get_value(self.tags['model_combo'])
            dataset_name = dpg.get_value(self.tags['dataset_combo'])
            epochs = dpg.get_value(self.tags['epochs_input'])
            
            dpg.add_text(f"  Model: {model_name}", color=COLORS['green_normal'])
            dpg.add_text(f"  Dataset: {dataset_name}", color=COLORS['green_normal'])
            dpg.add_text(f"  Epochs: {epochs}", color=COLORS['green_normal'])
            dpg.add_text(f"  Training Time: {time_str}", color=COLORS['green_normal'])
            
            if best_val_acc is not None:
                dpg.add_text(f"  Best Validation Accuracy: {best_val_acc:.2f}%", 
                           color=COLORS['green_bright'])
            
            if self.train_losses:
                final_loss = self.train_losses[-1]
                dpg.add_text(f"  Final Training Loss: {final_loss:.4f}", 
                           color=COLORS['green_normal'])
            
            dpg.add_spacer(height=15)
            dpg.add_separator()
            dpg.add_spacer(height=15)
            
            # Model save info
            if model_path:
                dpg.add_text("Model Location:", color=COLORS['orange_bright'])
                dpg.add_text(f"  {os.path.basename(model_path)}", 
                           color=COLORS['green_normal'], wrap=450)
                dpg.add_spacer(height=15)
            
            # Action buttons
            dpg.add_text("What would you like to do?", color=COLORS['orange_bright'])
            dpg.add_spacer(height=10)
            
            with dpg.group(horizontal=True):
                # Keep model button
                dpg.add_button(
                    label="[+] KEEP MODEL", 
                    width=150, height=30,
                    callback=lambda: self._handle_keep_model(model_path),
                    user_data=model_path
                )
                
                dpg.add_spacer(width=10)
                
                # Delete model button
                dpg.add_button(
                    label="[X] DELETE MODEL", 
                    width=150, height=30,
                    callback=lambda: self._handle_delete_model(model_path),
                    user_data=model_path
                )
                
                dpg.add_spacer(width=10)
                
                # Train again button
                dpg.add_button(
                    label="[>] TRAIN AGAIN", 
                    width=150, height=30,
                    callback=lambda: self._handle_train_again()
                )
    
    def _handle_keep_model(self, model_path):
        """Handle keeping the trained model."""
        dpg.delete_item("training_complete_modal")
        
        # Update status
        dpg.set_value(self.tags['status_text'], 
                     f"Model saved successfully!\nLocation: {os.path.basename(model_path)}")
        
        # Refresh testing tab model list if it exists
        if hasattr(self, 'testing_ui'):
            self.testing_ui.refresh_model_list()
    
    def _handle_delete_model(self, model_path):
        """Handle deleting the trained model."""
        if model_path and os.path.exists(model_path):
            try:
                # Delete both the state dict and full model files
                os.remove(model_path)
                full_model_path = model_path.replace('.pth', '_full.pth')
                if os.path.exists(full_model_path):
                    os.remove(full_model_path)
                
                dpg.set_value(self.tags['status_text'], "Model deleted.")
            except Exception as e:
                dpg.set_value(self.tags['status_text'], f"Error deleting model: {e}")
        
        dpg.delete_item("training_complete_modal")
    
    def _handle_train_again(self):
        """Handle training again with same settings."""
        dpg.delete_item("training_complete_modal")
        
        # Clear previous training data
        self.epochs = []
        self.train_losses = []
        self.val_losses = []
        self.val_accs = []
        self.update_plots()
        
        # Start training again
        self.start_training()
    
    def run(self):
        """Run the application."""
        self.setup_gui()
        
        # Main loop
        while dpg.is_dearpygui_running():
            dpg.render_dearpygui_frame()
        
        # Cleanup
        if self.is_training:
            self.stop_training()
            time.sleep(2)  # Give more time for cleanup
        
        # Force terminate if still alive
        if self.training_process and self.training_process.is_alive():
            self.training_process.terminate()
            self.training_process.join(timeout=1)
            if self.training_process.is_alive():
                # If still alive, force kill
                import signal
                os.kill(self.training_process.pid, signal.SIGTERM)
        
        dpg.destroy_context()


if __name__ == "__main__":
    # Enable multiprocessing support on Windows
    mp.freeze_support()
    
    # Import numpy for auto-configuration
    import numpy as np
    
    # Run the application
    app = ModelBuilderApp()
    app.run()