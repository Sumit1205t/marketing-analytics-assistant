import logging
import json
import os
from datetime import datetime

class MetricsLogger:
    def __init__(self, log_level=logging.INFO):
        # Initialize logger with a customizable log level
        self.logger = logging.getLogger('metrics')
        self.logger.setLevel(log_level)
        self.metrics_history = []
        
        # Create a handler to log to console
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        
        # Create a formatter and set it for the console handler
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        
        # Add the handler to the logger
        self.logger.addHandler(console_handler)

    def log_metrics(self, metrics_dict):
        """Log evaluation metrics with a structured format"""
        timestamp = datetime.now().isoformat()
        
        # Validate and log metrics
        if not isinstance(metrics_dict, dict):
            self.logger.error("Provided metrics are not in dictionary format.")
            return
        
        # Log the full metrics dictionary in structured format
        metrics_entry = {
            'timestamp': timestamp,
            'metrics': metrics_dict
        }
        
        # Add to history
        self.metrics_history.append(metrics_entry)
        
        # Log in a structured JSON format
        metrics_json = json.dumps(metrics_dict, indent=2)
        self.logger.info(f"Evaluation Metrics at {timestamp}:\n{metrics_json}")
        
        # Log individual metrics for easier parsing
        for metric_name, value in metrics_dict.items():
            if isinstance(value, dict):
                for sub_name, sub_value in value.items():
                    self.logger.info(f"{metric_name}.{sub_name}: {sub_value}")
            else:
                self.logger.info(f"{metric_name}: {value}")

    def get_metrics_history(self):
        """Return the history of logged metrics"""
        return self.metrics_history

    def export_metrics(self, filepath):
        """Export metrics history to JSON file"""
        try:
            # Ensure the directory exists
            directory = os.path.dirname(filepath)
            if directory and not os.path.exists(directory):
                os.makedirs(directory)

            # Write to JSON file
            with open(filepath, 'w') as f:
                json.dump(self.metrics_history, f, indent=2)
            self.logger.info(f"Metrics exported successfully to {filepath}")
        
        except Exception as e:
            self.logger.error(f"Error exporting metrics to {filepath}: {str(e)}")

