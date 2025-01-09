import os
from datetime import datetime
import logging

class UserLogger:
    def __init__(self):
        """Initialize UserLogger."""
        self.log_dir = 'logs'
        self.user_log_file = os.path.join(self.log_dir, 'user_activity.log')
        self._ensure_log_dir()
        
        # Setup logger
        self.logger = logging.getLogger('user_activity')
        self.logger.setLevel(logging.INFO)
        
        # Create file handler
        handler = logging.FileHandler(self.user_log_file)
        handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        
        # Add handler to logger
        self.logger.addHandler(handler)

    def _ensure_log_dir(self):
        """Ensure log directory exists."""
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

    def log_user_activity(self, user_data, activity_type):
        """Log user activity."""
        try:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            message = f"[{activity_type}] {str(user_data)}"
            self.logger.info(message)
        except Exception as e:
            print(f"Error logging user activity: {str(e)}") 