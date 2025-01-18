import os
from datetime import datetime
import logging
import json
from typing import Dict, Any
from .config import Config

logger = logging.getLogger(__name__)

class UserLogger:
    """User activity logger with improved error handling."""
    
    def __init__(self):
        try:
            self.enabled = Config.POSTHOG_ENABLED
            self.fallback_log = []
            
            # Create analytics log directory
            self.log_dir = os.path.join('logs', 'analytics')
            os.makedirs(self.log_dir, exist_ok=True)
            
            logger.info("UserLogger initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing UserLogger: {e}")
            self.enabled = False
            self.fallback_log = []
    
    def log_user_activity(self, data: Dict[str, Any], event_type: str) -> bool:
        """Log user activity with fallback to local storage."""
        if not self.enabled:
            return self._log_locally(data, event_type)
            
        try:
            import posthog
            
            # Attempt to send to PostHog
            posthog.capture(
                distinct_id=data.get('email', 'anonymous'),
                event=event_type,
                properties=data,
                timestamp=datetime.now().isoformat()
            )
            
            logger.info(f"Successfully logged {event_type} event")
            return True
            
        except Exception as e:
            logger.warning(f"Failed to log to PostHog: {e}")
            return self._log_locally(data, event_type)
    
    def _log_locally(self, data: Dict[str, Any], event_type: str) -> bool:
        """Fallback logging to local storage."""
        try:
            log_entry = {
                'timestamp': datetime.now().isoformat(),
                'event_type': event_type,
                'data': data
            }
            
            self.fallback_log.append(log_entry)
            
            # Save to local file
            log_file = os.path.join(
                self.log_dir, 
                f'user_activity_{datetime.now().strftime("%Y%m%d")}.json'
            )
            
            with open(log_file, 'a') as f:
                json.dump(log_entry, f)
                f.write('\n')
            
            logger.info(f"Logged event locally: {event_type}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to log locally: {e}")
            return False 