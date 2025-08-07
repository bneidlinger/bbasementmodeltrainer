"""
Danger Mode Controller
Manages safety features and experimental mode toggles
⚠️ Use with extreme caution!
"""

import logging
import hashlib
from datetime import datetime
from typing import Optional, Dict, Any
import json
import os

logger = logging.getLogger(__name__)

class DangerModeController:
    """
    Controls safety features and danger mode activation.
    Requires explicit user consent and logs all activations.
    """
    
    def __init__(self, log_dir: str = "danger_logs"):
        self.safety_enabled = True
        self.warning_accepted = False
        self.danger_session_id = None
        self.log_dir = log_dir
        self.filters = {
            'toxicity': True,
            'pii': True,
            'prompt_injection': True,
            'illegal_content': True,
            'bias_mitigation': True
        }
        
        # Create log directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
        
        # Load danger mode configuration
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load danger mode configuration."""
        config_path = os.path.join(self.log_dir, "danger_config.json")
        
        default_config = {
            'require_password': False,
            'password_hash': None,
            'max_session_hours': 24,
            'log_all_outputs': True,
            'allowed_ips': [],  # Empty means all IPs allowed
            'blocked_keywords': [],  # Keywords to always block
            'auto_disable_after_errors': 3
        }
        
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return json.load(f)
        
        return default_config
    
    def show_warning_dialog(self) -> str:
        """Return warning text for UI display."""
        warning = """
╔══════════════════════════════════════════════════════════════╗
║                  ⚠️  DANGER MODE WARNING ⚠️                   ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  ACTIVATING DANGER MODE WILL:                               ║
║                                                              ║
║  ⛔ DISABLE all content filtering                           ║
║  ⛔ REMOVE safety guardrails                                ║
║  ⛔ ALLOW potentially harmful outputs                       ║
║  ⛔ PERMIT uncensored model responses                       ║
║  ⛔ BYPASS ethical guidelines                               ║
║                                                              ║
║  YOU ARE RESPONSIBLE FOR:                                   ║
║                                                              ║
║  ✓ All generated content                                    ║
║  ✓ Any consequences of model outputs                        ║
║  ✓ Compliance with local laws and regulations               ║
║  ✓ Ethical use of the system                               ║
║                                                              ║
║  This mode is intended ONLY for:                            ║
║  • Research and experimentation                             ║
║  • Testing model capabilities                               ║
║  • Academic study                                           ║
║                                                              ║
║  NOT for production or public-facing applications!          ║
║                                                              ║
║  All activities will be logged for accountability.          ║
║                                                              ║
║  Type 'I ACCEPT FULL RESPONSIBILITY' to continue           ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
        """
        return warning
    
    def activate_danger_mode(self, consent_text: str, password: Optional[str] = None) -> bool:
        """
        Activate danger mode with explicit consent.
        
        Args:
            consent_text: User must type exact consent phrase
            password: Optional password if configured
        
        Returns:
            True if activated, False otherwise
        """
        # Check consent text
        if consent_text != "I ACCEPT FULL RESPONSIBILITY":
            logger.warning("Danger mode activation failed: Invalid consent text")
            return False
        
        # Check password if required
        if self.config['require_password']:
            if not password:
                logger.warning("Danger mode activation failed: Password required")
                return False
            
            password_hash = hashlib.sha256(password.encode()).hexdigest()
            if password_hash != self.config['password_hash']:
                logger.warning("Danger mode activation failed: Invalid password")
                return False
        
        # Generate session ID
        self.danger_session_id = hashlib.md5(
            f"{datetime.now().isoformat()}".encode()
        ).hexdigest()
        
        # Log activation
        self._log_activation()
        
        # Disable all safety features
        self.safety_enabled = False
        self.warning_accepted = True
        
        for filter_name in self.filters:
            self.filters[filter_name] = False
        
        logger.warning("=" * 60)
        logger.warning("DANGER MODE ACTIVATED")
        logger.warning(f"Session ID: {self.danger_session_id}")
        logger.warning("All safety features DISABLED")
        logger.warning("User has accepted full responsibility")
        logger.warning("=" * 60)
        
        return True
    
    def deactivate_danger_mode(self):
        """Deactivate danger mode and re-enable safety features."""
        if self.danger_session_id:
            self._log_deactivation()
        
        self.safety_enabled = True
        self.warning_accepted = False
        self.danger_session_id = None
        
        # Re-enable all filters
        for filter_name in self.filters:
            self.filters[filter_name] = True
        
        logger.info("Danger mode deactivated. Safety features restored.")
    
    def is_content_allowed(self, content: str) -> bool:
        """
        Check if content is allowed based on current safety settings.
        
        Args:
            content: Text to check
        
        Returns:
            True if content is allowed, False if blocked
        """
        if not self.safety_enabled:
            # In danger mode, only check blocked keywords
            for keyword in self.config.get('blocked_keywords', []):
                if keyword.lower() in content.lower():
                    logger.warning(f"Content blocked due to keyword: {keyword}")
                    return False
            return True
        
        # In safe mode, apply all filters
        # This is where you'd integrate actual safety checks
        # For now, this is a placeholder
        return True
    
    def log_generation(self, prompt: str, output: str, metadata: Optional[Dict] = None):
        """Log generation in danger mode."""
        if not self.danger_session_id:
            return
        
        if not self.config.get('log_all_outputs', True):
            return
        
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'session_id': self.danger_session_id,
            'prompt': prompt,
            'output': output,
            'metadata': metadata or {}
        }
        
        log_file = os.path.join(
            self.log_dir, 
            f"danger_session_{self.danger_session_id}.jsonl"
        )
        
        with open(log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
    
    def _log_activation(self):
        """Log danger mode activation."""
        log_entry = {
            'event': 'activation',
            'timestamp': datetime.now().isoformat(),
            'session_id': self.danger_session_id,
            'filters_disabled': list(self.filters.keys())
        }
        
        activation_log = os.path.join(self.log_dir, "activations.jsonl")
        with open(activation_log, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
    
    def _log_deactivation(self):
        """Log danger mode deactivation."""
        log_entry = {
            'event': 'deactivation',
            'timestamp': datetime.now().isoformat(),
            'session_id': self.danger_session_id
        }
        
        activation_log = os.path.join(self.log_dir, "activations.jsonl")
        with open(activation_log, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
    
    def get_status(self) -> Dict[str, Any]:
        """Get current danger mode status."""
        return {
            'safety_enabled': self.safety_enabled,
            'danger_mode_active': not self.safety_enabled,
            'session_id': self.danger_session_id,
            'active_filters': {k: v for k, v in self.filters.items() if v},
            'warning_accepted': self.warning_accepted
        }


class SafetyFilter:
    """Basic safety filtering when not in danger mode."""
    
    def __init__(self):
        # Basic list of concerning keywords (expand as needed)
        self.blocked_patterns = [
            # Add actual patterns here
            # This is a placeholder
        ]
    
    def check_prompt(self, prompt: str) -> Dict[str, Any]:
        """
        Check prompt for safety issues.
        
        Returns:
            Dict with 'safe' boolean and 'reasons' list
        """
        reasons = []
        
        # Check for prompt injection attempts
        injection_patterns = [
            "ignore previous instructions",
            "disregard all prior",
            "forget everything"
        ]
        
        for pattern in injection_patterns:
            if pattern.lower() in prompt.lower():
                reasons.append(f"Potential prompt injection: {pattern}")
        
        return {
            'safe': len(reasons) == 0,
            'reasons': reasons
        }
    
    def sanitize_output(self, output: str) -> str:
        """
        Sanitize model output when safety is enabled.
        
        Args:
            output: Raw model output
        
        Returns:
            Sanitized output
        """
        # This is a placeholder for actual content filtering
        # In production, you'd use more sophisticated methods
        
        # Remove potential PII (very basic example)
        import re
        
        # Remove email addresses
        output = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', output)
        
        # Remove phone numbers (US format)
        output = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '[PHONE]', output)
        
        # Remove SSN-like patterns
        output = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', '[SSN]', output)
        
        return output