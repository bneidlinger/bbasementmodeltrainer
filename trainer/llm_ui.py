"""
LLM Training UI for BasementBrewAI
Industrial-themed interface for fine-tuning language models
"""

import dearpygui.dearpygui as dpg
import os
import json
import multiprocessing as mp
from queue import Empty
import threading
import time
from typing import Optional, Dict, Any
from tkinter import filedialog
import tkinter as tk

from retro_theme import COLORS
from llm.training.trainer import LLMTrainingConfig, train_llm_worker
from llm.training.qlora import QLoRAConfig
from llm.safety.danger_mode import DangerModeController
from llm import list_available_llms
from llm.inference import LLMInference, InferenceConfig

class LLMTrainingUI:
    """UI for LLM training and fine-tuning."""
    
    def __init__(self, parent_app):
        self.app = parent_app
        self.training_process: Optional[mp.Process] = None
        self.message_queue: Optional[mp.Queue] = None
        self.monitor_thread: Optional[threading.Thread] = None
        self.is_training = False
        self.danger_controller = DangerModeController()
        
        # Inference components
        self.inference_engine: Optional[LLMInference] = None
        self.conversation_history = []
        self.current_model_path = None
        
        # Training metrics
        self.training_steps = []
        self.training_losses = []
        self.learning_rates = []
        
        # UI tags
        self.tags = {
            # Model selection
            'model_combo': 'llm_model_combo',
            'model_info': 'llm_model_info',
            
            # Dataset configuration
            'dataset_path': 'llm_dataset_path',
            'browse_dataset': 'llm_browse_dataset',
            'scrape_data': 'llm_scrape_data',
            'max_seq_length': 'llm_max_seq_length',
            
            # Training parameters
            'epochs_input': 'llm_epochs',
            'batch_size': 'llm_batch_size',
            'learning_rate': 'llm_learning_rate',
            'gradient_accum': 'llm_gradient_accum',
            
            # QLoRA settings
            'use_qlora': 'llm_use_qlora',
            'lora_rank': 'llm_lora_rank',
            'lora_alpha': 'llm_lora_alpha',
            
            # Safety settings
            'danger_mode': 'llm_danger_mode',
            'safety_status': 'llm_safety_status',
            
            # Control buttons
            'start_button': 'llm_start_button',
            'stop_button': 'llm_stop_button',
            
            # Progress monitoring
            'progress_bar': 'llm_progress_bar',
            'status_text': 'llm_status_text',
            'loss_plot': 'llm_loss_plot',
            'loss_x_axis': 'llm_loss_x_axis',
            'loss_y_axis': 'llm_loss_y_axis',
            'loss_series': 'llm_loss_series',
            
            # Chat interface
            'chat_input': 'llm_chat_input',
            'chat_output': 'llm_chat_output',
            'chat_send': 'llm_chat_send',
            'load_model_button': 'llm_load_model',
            'model_status': 'llm_model_status',
        }
    
    def create_ui(self, parent):
        """Create the LLM training interface."""
        with dpg.group(parent=parent):
            # Header
            dpg.add_text("[ LLM FINE-TUNING LABORATORY ]", color=COLORS['orange_bright'])
            dpg.add_text("GPT-OSS Training & Experimentation", color=COLORS['green_dim'])
            dpg.add_separator()
            
            with dpg.group(horizontal=True):
                # Left panel - Configuration
                with dpg.child_window(width=400, height=-200):
                    self._create_model_section()
                    dpg.add_separator()
                    self._create_dataset_section()
                    dpg.add_separator()
                    self._create_training_params_section()
                    dpg.add_separator()
                    self._create_safety_section()
                
                # Middle panel - Monitoring
                with dpg.child_window(width=400, height=-200):
                    self._create_monitoring_section()
                
                # Right panel - Chat/Inference
                with dpg.child_window(width=-1, height=-200):
                    self._create_chat_section()
            
            # Bottom - Controls and status
            dpg.add_separator()
            self._create_control_section()
    
    def _create_model_section(self):
        """Create model selection section."""
        dpg.add_text("[ MODEL SELECTION ]", color=COLORS['orange_bright'])
        
        # Get available models
        available_models = list_available_llms()
        
        # Filter for GPT-OSS models
        gpt_models = [m for m in available_models if 'gpt-oss' in m]
        if not gpt_models:
            gpt_models = ['gpt-oss-20b', 'gpt-oss-120b']  # Defaults
        
        dpg.add_text("Model:", color=COLORS['green_normal'])
        dpg.add_combo(
            gpt_models,
            default_value=gpt_models[0],
            tag=self.tags['model_combo'],
            width=-1,
            callback=self._on_model_selected
        )
        
        dpg.add_text(
            "Model: GPT-OSS-20B\n"
            "Parameters: 21B (3.6B active)\n"
            "VRAM Required: ~16GB (4-bit)\n"
            "Status: ✓ Compatible with your GPU",
            tag=self.tags['model_info'],
            color=COLORS['green_dim'],
            wrap=380
        )
    
    def _create_dataset_section(self):
        """Create dataset configuration section."""
        dpg.add_text("[ DATASET CONFIGURATION ]", color=COLORS['orange_bright'])
        
        dpg.add_text("Training Data:", color=COLORS['green_normal'])
        
        with dpg.group(horizontal=True):
            dpg.add_input_text(
                tag=self.tags['dataset_path'],
                width=250,
                hint="Path to dataset...",
                readonly=True
            )
            dpg.add_button(
                label="Browse",
                callback=self._browse_dataset,
                width=70
            )
        
        dpg.add_button(
            label="[↓] Scrape Data from Web",
            tag=self.tags['scrape_data'],
            callback=self._open_scraper,
            width=-1
        )
        
        dpg.add_text("Max Sequence Length:", color=COLORS['green_normal'])
        dpg.add_input_int(
            tag=self.tags['max_seq_length'],
            default_value=2048,
            min_value=128,
            max_value=128000,
            width=150
        )
    
    def _create_training_params_section(self):
        """Create training parameters section."""
        dpg.add_text("[ TRAINING PARAMETERS ]", color=COLORS['orange_bright'])
        
        # Basic parameters
        with dpg.group(horizontal=True):
            with dpg.group():
                dpg.add_text("Epochs:", color=COLORS['green_normal'])
                dpg.add_input_int(
                    tag=self.tags['epochs_input'],
                    default_value=3,
                    min_value=1,
                    max_value=100,
                    width=100
                )
            
            with dpg.group():
                dpg.add_text("Batch Size:", color=COLORS['green_normal'])
                dpg.add_input_int(
                    tag=self.tags['batch_size'],
                    default_value=2,
                    min_value=1,
                    max_value=32,
                    width=100
                )
        
        dpg.add_text("Learning Rate:", color=COLORS['green_normal'])
        dpg.add_input_float(
            tag=self.tags['learning_rate'],
            default_value=0.0002,
            format="%.6f",
            width=150
        )
        
        dpg.add_text("Gradient Accumulation:", color=COLORS['green_normal'])
        dpg.add_input_int(
            tag=self.tags['gradient_accum'],
            default_value=8,
            min_value=1,
            max_value=64,
            width=150
        )
        
        # QLoRA settings
        dpg.add_separator()
        dpg.add_checkbox(
            label="Use QLoRA (4-bit training)",
            tag=self.tags['use_qlora'],
            default_value=True,
            callback=self._toggle_qlora
        )
        
        with dpg.group(horizontal=True):
            with dpg.group():
                dpg.add_text("LoRA Rank:", color=COLORS['green_normal'])
                dpg.add_input_int(
                    tag=self.tags['lora_rank'],
                    default_value=64,
                    min_value=8,
                    max_value=256,
                    width=100
                )
            
            with dpg.group():
                dpg.add_text("LoRA Alpha:", color=COLORS['green_normal'])
                dpg.add_input_int(
                    tag=self.tags['lora_alpha'],
                    default_value=128,
                    min_value=8,
                    max_value=512,
                    width=100
                )
    
    def _create_safety_section(self):
        """Create safety/danger mode section."""
        dpg.add_text("[ SAFETY CONTROLS ]", color=COLORS['orange_bright'])
        
        dpg.add_text(
            "✓ Safety Enabled - Content filtering active",
            tag=self.tags['safety_status'],
            color=COLORS['green_bright']
        )
        
        dpg.add_separator()
        
        dpg.add_checkbox(
            label="⚠️ DANGER MODE - Disable all safety",
            tag=self.tags['danger_mode'],
            default_value=False,
            callback=self._toggle_danger_mode
        )
        
        dpg.add_text(
            "Warning: Danger mode removes ALL content\n"
            "filtering and safety checks. Use with caution!",
            color=(255, 100, 100),
            wrap=380,
            show=False,
            tag="danger_warning"
        )
    
    def _create_monitoring_section(self):
        """Create training monitoring section."""
        dpg.add_text("[ TRAINING MONITOR ]", color=COLORS['orange_bright'])
        
        # Status
        dpg.add_text("Status: Ready", tag=self.tags['status_text'], 
                    color=COLORS['green_normal'])
        
        # Progress bar
        dpg.add_progress_bar(
            tag=self.tags['progress_bar'],
            width=-1,
            default_value=0.0
        )
        
        dpg.add_separator()
        
        # Loss plot
        dpg.add_text("Training Loss:", color=COLORS['green_normal'])
        
        # Check if plot already exists and delete if so
        if dpg.does_alias_exist(self.tags['loss_plot']):
            dpg.delete_item(self.tags['loss_plot'])
        
        with dpg.plot(label="Loss", height=300, width=-1, tag=self.tags['loss_plot']):
            dpg.add_plot_axis(dpg.mvXAxis, label="Steps", tag=self.tags['loss_x_axis'])
            dpg.add_plot_axis(dpg.mvYAxis, label="Loss", tag=self.tags['loss_y_axis'])
            dpg.add_line_series(
                [], [],
                label="Training Loss",
                parent=self.tags['loss_y_axis'],
                tag=self.tags['loss_series']
            )
    
    def _create_chat_section(self):
        """Create chat/inference section."""
        dpg.add_text("[ MODEL CHAT INTERFACE ]", color=COLORS['orange_bright'])
        
        # Model loading controls
        with dpg.group(horizontal=True):
            dpg.add_button(
                label="[↓] Load Model",
                tag=self.tags['load_model_button'],
                callback=self._open_model_loader,
                width=120
            )
            dpg.add_text(
                "No model loaded",
                tag=self.tags['model_status'],
                color=COLORS['green_dim']
            )
        
        dpg.add_separator()
        
        # Chat output
        dpg.add_input_text(
            tag=self.tags['chat_output'],
            multiline=True,
            readonly=True,
            width=-1,
            height=320,
            default_value="Model not loaded. Click 'Load Model' to load a trained model.\n"
        )
        
        # Chat input
        dpg.add_separator()
        with dpg.group(horizontal=True):
            dpg.add_input_text(
                tag=self.tags['chat_input'],
                width=-100,
                hint="Enter prompt...",
                on_enter=True,
                callback=self._send_chat
            )
            dpg.add_button(
                label="Send",
                tag=self.tags['chat_send'],
                callback=self._send_chat,
                width=80,
                enabled=False
            )
    
    def _create_control_section(self):
        """Create control buttons section."""
        with dpg.group(horizontal=True):
            dpg.add_button(
                label="[>] START TRAINING",
                tag=self.tags['start_button'],
                callback=self._start_training,
                width=200,
                height=40
            )
            
            dpg.add_button(
                label="[X] STOP",
                tag=self.tags['stop_button'],
                callback=self._stop_training,
                width=150,
                height=40,
                enabled=False
            )
            
            dpg.add_button(
                label="[L] Load Model",
                callback=self._load_selected_model,
                width=150,
                height=40
            )
    
    def _on_model_selected(self, sender, app_data):
        """Handle model selection."""
        model = app_data
        
        if 'gpt-oss-20b' in model:
            info = (
                "Model: GPT-OSS-20B\n"
                "Parameters: 21B (3.6B active)\n"
                "VRAM Required: ~16GB (4-bit)\n"
                "Status: ✓ Compatible with your GPU"
            )
            dpg.set_value(self.tags['model_info'], info)
        elif 'gpt-oss-120b' in model:
            info = (
                "Model: GPT-OSS-120B\n"
                "Parameters: 117B (5.1B active)\n"
                "VRAM Required: ~80GB (4-bit)\n"
                "Status: ⚠️ Requires more VRAM"
            )
            dpg.set_value(self.tags['model_info'], info)
    
    def _toggle_qlora(self, sender, app_data):
        """Toggle QLoRA settings."""
        enabled = app_data
        dpg.configure_item(self.tags['lora_rank'], enabled=enabled)
        dpg.configure_item(self.tags['lora_alpha'], enabled=enabled)
    
    def _toggle_danger_mode(self, sender, app_data):
        """Toggle danger mode with confirmation."""
        if app_data:  # Turning ON danger mode
            # Show warning
            dpg.configure_item("danger_warning", show=True)
            
            # Show confirmation dialog
            if dpg.does_item_exist("danger_confirm_modal"):
                dpg.delete_item("danger_confirm_modal")
            
            with dpg.window(label="⚠️ DANGER MODE WARNING ⚠️", 
                          modal=True, 
                          tag="danger_confirm_modal",
                          width=600, height=400, pos=[350, 200]):
                
                warning_text = self.danger_controller.show_warning_dialog()
                dpg.add_text(warning_text, color=(255, 150, 50))
                
                dpg.add_separator()
                dpg.add_input_text(
                    tag="danger_consent",
                    hint="Type consent phrase here...",
                    width=-1
                )
                
                with dpg.group(horizontal=True):
                    dpg.add_button(
                        label="ACTIVATE DANGER MODE",
                        callback=self._confirm_danger_mode,
                        width=200
                    )
                    dpg.add_button(
                        label="Cancel",
                        callback=lambda: self._cancel_danger_mode(),
                        width=200
                    )
        else:  # Turning OFF danger mode
            self.danger_controller.deactivate_danger_mode()
            dpg.set_value(self.tags['safety_status'], 
                         "✓ Safety Enabled - Content filtering active")
            dpg.configure_item(self.tags['safety_status'], 
                              color=COLORS['green_bright'])
            dpg.configure_item("danger_warning", show=False)
    
    def _confirm_danger_mode(self):
        """Confirm danger mode activation."""
        consent = dpg.get_value("danger_consent")
        
        if self.danger_controller.activate_danger_mode(consent):
            dpg.set_value(self.tags['safety_status'], 
                         "⚠️ DANGER MODE ACTIVE - No safety filters!")
            dpg.configure_item(self.tags['safety_status'], 
                              color=(255, 100, 100))
            dpg.delete_item("danger_confirm_modal")
        else:
            dpg.set_value(self.tags['danger_mode'], False)
            dpg.delete_item("danger_confirm_modal")
    
    def _cancel_danger_mode(self):
        """Cancel danger mode activation."""
        dpg.set_value(self.tags['danger_mode'], False)
        dpg.configure_item("danger_warning", show=False)
        dpg.delete_item("danger_confirm_modal")
    
    def _browse_dataset(self):
        """Browse for dataset file/folder."""
        root = tk.Tk()
        root.withdraw()
        
        path = filedialog.askopenfilename(
            title="Select dataset file",
            filetypes=[("Text files", "*.txt"), ("JSON files", "*.json"), 
                      ("All files", "*.*")]
        )
        
        root.destroy()
        
        if path:
            dpg.set_value(self.tags['dataset_path'], path)
    
    def _open_scraper(self):
        """Open data scraping dialog."""
        if dpg.does_item_exist("scraper_modal"):
            dpg.delete_item("scraper_modal")
        
        with dpg.window(label="Data Scraper", modal=True, tag="scraper_modal",
                       width=600, height=500, pos=[350, 150]):
            dpg.add_text("[ WEB DATA SCRAPER ]", color=COLORS['orange_bright'])
            dpg.add_separator()
            
            dpg.add_text("Data Sources:", color=COLORS['green_normal'])
            dpg.add_checkbox(label="Web Pages", default_value=True, tag="scrape_web")
            dpg.add_checkbox(label="Reddit", tag="scrape_reddit")
            dpg.add_checkbox(label="ArXiv Papers", tag="scrape_arxiv")
            
            dpg.add_separator()
            
            dpg.add_text("Search Query:", color=COLORS['green_normal'])
            dpg.add_input_text(tag="scrape_query", width=-1, 
                             hint="e.g., 'machine learning tutorials'")
            
            dpg.add_text("Max Items:", color=COLORS['green_normal'])
            dpg.add_input_int(tag="scrape_max", default_value=100, 
                            min_value=10, max_value=1000)
            
            dpg.add_separator()
            
            with dpg.group(horizontal=True):
                dpg.add_button(label="Start Scraping", 
                             callback=self._start_scraping, width=150)
                dpg.add_button(label="Cancel", 
                             callback=lambda: dpg.delete_item("scraper_modal"), 
                             width=150)
            
            dpg.add_separator()
            dpg.add_text("Status: Ready", tag="scrape_status", 
                        color=COLORS['green_dim'])
    
    def _start_scraping(self):
        """Start data scraping process."""
        # This would implement actual scraping
        dpg.set_value("scrape_status", "Scraping... (not implemented yet)")
    
    def _start_training(self):
        """Start LLM training."""
        if self.is_training:
            return
        
        # Gather configuration
        config = LLMTrainingConfig(
            model_name=dpg.get_value(self.tags['model_combo']),
            model_type=dpg.get_value(self.tags['model_combo']),
            dataset_path=dpg.get_value(self.tags['dataset_path']) or None,
            max_seq_length=dpg.get_value(self.tags['max_seq_length']),
            num_epochs=dpg.get_value(self.tags['epochs_input']),
            batch_size=dpg.get_value(self.tags['batch_size']),
            learning_rate=dpg.get_value(self.tags['learning_rate']),
            gradient_accumulation_steps=dpg.get_value(self.tags['gradient_accum']),
            use_qlora=dpg.get_value(self.tags['use_qlora']),
            danger_mode=dpg.get_value(self.tags['danger_mode']),
        )
        
        if config.use_qlora:
            config.qlora_config = QLoRAConfig(
                r=dpg.get_value(self.tags['lora_rank']),
                lora_alpha=dpg.get_value(self.tags['lora_alpha'])
            )
        
        # Create message queue
        self.message_queue = mp.Queue()
        
        # Start training process
        self.training_process = mp.Process(
            target=train_llm_worker,
            args=(config.__dict__, self.message_queue)
        )
        self.training_process.start()
        
        # Start monitoring thread
        self.is_training = True
        self.monitor_thread = threading.Thread(target=self._monitor_training)
        self.monitor_thread.start()
        
        # Update UI
        dpg.configure_item(self.tags['start_button'], enabled=False)
        dpg.configure_item(self.tags['stop_button'], enabled=True)
        dpg.set_value(self.tags['status_text'], "Starting training...")
    
    def _stop_training(self):
        """Stop LLM training."""
        if self.training_process and self.training_process.is_alive():
            self.training_process.terminate()
            self.training_process.join(timeout=5)
            
            if self.training_process.is_alive():
                self.training_process.kill()
        
        self.is_training = False
        
        # Update UI
        dpg.configure_item(self.tags['start_button'], enabled=True)
        dpg.configure_item(self.tags['stop_button'], enabled=False)
        dpg.set_value(self.tags['status_text'], "Training stopped")
    
    def _monitor_training(self):
        """Monitor training progress."""
        while self.is_training:
            try:
                message = self.message_queue.get(timeout=1)
                
                if message['type'] == 'status':
                    dpg.set_value(self.tags['status_text'], message['data'])
                
                elif message['type'] == 'metrics':
                    metrics = message['data']
                    
                    # Update loss plot
                    self.training_steps.append(metrics['step'])
                    self.training_losses.append(metrics['loss'])
                    
                    dpg.set_value(self.tags['loss_series'], 
                                [self.training_steps, self.training_losses])
                    
                    # Update progress
                    if metrics.get('epoch'):
                        progress = metrics['epoch'] / dpg.get_value(self.tags['epochs_input'])
                        dpg.set_value(self.tags['progress_bar'], progress)
                
                elif message['type'] == 'complete':
                    self.is_training = False
                    dpg.set_value(self.tags['status_text'], "Training complete!")
                    dpg.configure_item(self.tags['chat_send'], enabled=True)
                
                elif message['type'] == 'error':
                    self.is_training = False
                    dpg.set_value(self.tags['status_text'], 
                                f"Error: {message['data']}")
                
            except Empty:
                pass
            except Exception as e:
                print(f"Monitor error: {e}")
    
    def _open_model_loader(self):
        """Open model loading dialog."""
        if dpg.does_item_exist("model_loader_modal"):
            dpg.delete_item("model_loader_modal")
        
        # Get available models from llm_outputs directory
        model_dirs = []
        output_dir = "llm_outputs"
        if os.path.exists(output_dir):
            for item in os.listdir(output_dir):
                item_path = os.path.join(output_dir, item)
                if os.path.isdir(item_path):
                    # Check if it contains model files
                    if any(f in os.listdir(item_path) for f in ['adapter_config.json', 'pytorch_model.bin', 'model.safetensors']):
                        model_dirs.append(item)
        
        # Also check saved_models directory for regular models
        saved_dir = "saved_models"
        if os.path.exists(saved_dir):
            for item in os.listdir(saved_dir):
                if item.endswith(('.pth', '.pt', '.bin')):
                    model_dirs.append(f"saved_models/{item}")
        
        with dpg.window(label="Load Model", modal=True, tag="model_loader_modal",
                       width=500, height=400, pos=[400, 200]):
            dpg.add_text("[ MODEL LOADER ]", color=COLORS['orange_bright'])
            dpg.add_separator()
            
            if model_dirs:
                dpg.add_text("Available Models:", color=COLORS['green_normal'])
                dpg.add_listbox(
                    model_dirs,
                    tag="model_list",
                    width=-1,
                    num_items=8
                )
            else:
                dpg.add_text("No trained models found.", color=COLORS['green_dim'])
                dpg.add_text("Train a model first or browse for a model file.", 
                           color=COLORS['green_dim'], wrap=480)
            
            dpg.add_separator()
            
            # Browse option
            with dpg.group(horizontal=True):
                dpg.add_input_text(
                    tag="custom_model_path",
                    width=350,
                    hint="Or enter custom model path..."
                )
                dpg.add_button(
                    label="Browse",
                    callback=self._browse_model,
                    width=80
                )
            
            # Load in 4-bit option
            dpg.add_checkbox(
                label="Load in 4-bit (reduces VRAM usage)",
                tag="load_4bit",
                default_value=True
            )
            
            dpg.add_separator()
            
            with dpg.group(horizontal=True):
                dpg.add_button(
                    label="Load Model",
                    callback=self._load_selected_model,
                    width=150
                )
                dpg.add_button(
                    label="Cancel",
                    callback=lambda: dpg.delete_item("model_loader_modal"),
                    width=150
                )
            
            dpg.add_text("", tag="load_status", color=COLORS['green_dim'])
    
    def _browse_model(self):
        """Browse for model file or directory."""
        root = tk.Tk()
        root.withdraw()
        
        path = filedialog.askdirectory(
            title="Select model directory"
        )
        
        root.destroy()
        
        if path:
            dpg.set_value("custom_model_path", path)
    
    def _load_selected_model(self):
        """Load the selected model."""
        # Get selected model
        model_path = None
        
        if dpg.does_item_exist("model_list"):
            selected = dpg.get_value("model_list")
            if selected:
                if selected.startswith("saved_models/"):
                    model_path = selected
                else:
                    model_path = os.path.join("llm_outputs", selected)
        
        # Check custom path
        custom_path = dpg.get_value("custom_model_path") if dpg.does_item_exist("custom_model_path") else None
        if custom_path and os.path.exists(custom_path):
            model_path = custom_path
        
        if not model_path:
            dpg.set_value("load_status", "Please select a model")
            return
        
        dpg.set_value("load_status", "Loading model... This may take a moment.")
        
        # Load model in a separate thread to avoid blocking UI
        threading.Thread(target=self._load_model_thread, args=(model_path,)).start()
    
    def _load_model_thread(self, model_path: str):
        """Load model in background thread."""
        try:
            # Create inference config
            load_4bit = dpg.get_value("load_4bit") if dpg.does_item_exist("load_4bit") else True
            
            config = InferenceConfig(
                model_path=model_path,
                load_in_4bit=load_4bit,
                temperature=0.7,
                top_p=0.9,
                max_new_tokens=512,
                apply_safety_filter=not self.danger_controller.danger_mode_active
            )
            
            # Create inference engine
            self.inference_engine = LLMInference(config)
            
            # Load model
            self.inference_engine.load_model()
            
            # Update UI on success
            self.current_model_path = model_path
            model_name = os.path.basename(model_path)
            
            dpg.set_value(self.tags['model_status'], f"Loaded: {model_name}")
            dpg.configure_item(self.tags['model_status'], color=COLORS['green_bright'])
            dpg.configure_item(self.tags['chat_send'], enabled=True)
            dpg.set_value(self.tags['chat_output'], 
                         f"Model loaded successfully: {model_name}\n"
                         f"You can now start chatting!\n\n")
            
            # Clear conversation history
            self.conversation_history = []
            
            # Close dialog
            if dpg.does_item_exist("model_loader_modal"):
                dpg.delete_item("model_loader_modal")
            
        except Exception as e:
            error_msg = f"Failed to load model: {str(e)}"
            dpg.set_value("load_status", error_msg)
            dpg.set_value(self.tags['model_status'], "Failed to load")
            dpg.configure_item(self.tags['model_status'], color=(255, 100, 100))
            print(f"Model loading error: {e}")
    
    def _send_chat(self):
        """Send chat message to model."""
        if not self.inference_engine or not self.inference_engine.is_loaded:
            dpg.set_value(self.tags['chat_output'], 
                         dpg.get_value(self.tags['chat_output']) + 
                         "\n[System] Please load a model first.\n")
            return
        
        prompt = dpg.get_value(self.tags['chat_input'])
        if not prompt:
            return
        
        # Add user message to output
        current = dpg.get_value(self.tags['chat_output'])
        current += f"\n>> User: {prompt}\n"
        dpg.set_value(self.tags['chat_output'], current)
        dpg.set_value(self.tags['chat_input'], "")
        
        # Disable send button during generation
        dpg.configure_item(self.tags['chat_send'], enabled=False)
        
        # Generate response in thread
        threading.Thread(target=self._generate_response, args=(prompt,)).start()
    
    def _generate_response(self, prompt: str):
        """Generate model response in background."""
        try:
            # Add thinking indicator
            current = dpg.get_value(self.tags['chat_output'])
            dpg.set_value(self.tags['chat_output'], current + "\n<< Model: [Generating...]\n")
            
            # Generate response
            response = self.inference_engine.chat(prompt, self.conversation_history)
            
            # Update conversation history
            self.conversation_history.append({"role": "user", "content": prompt})
            self.conversation_history.append({"role": "assistant", "content": response})
            
            # Keep only last 20 messages
            if len(self.conversation_history) > 20:
                self.conversation_history = self.conversation_history[-20:]
            
            # Update output
            current = dpg.get_value(self.tags['chat_output'])
            # Remove the generating indicator
            current = current.replace("\n<< Model: [Generating...]\n", "")
            current += f"\n<< Model: {response}\n"
            dpg.set_value(self.tags['chat_output'], current)
            
        except Exception as e:
            current = dpg.get_value(self.tags['chat_output'])
            current = current.replace("\n<< Model: [Generating...]\n", "")
            current += f"\n<< Model: [Error: {str(e)}]\n"
            dpg.set_value(self.tags['chat_output'], current)
        
        finally:
            # Re-enable send button
            dpg.configure_item(self.tags['chat_send'], enabled=True)