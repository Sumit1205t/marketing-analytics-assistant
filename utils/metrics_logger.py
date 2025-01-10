import logging
import json
from datetime import datetime

class MetricsLogger:
    def __init__(self):
        self.logger = logging.getLogger('metrics')
        self.metrics_history = []

    def log_metrics(self, metrics_dict):
        """Log evaluation metrics with structured format"""
        timestamp = datetime.now().isoformat()
        
        metrics_entry = {
            'timestamp': timestamp,
            'metrics': metrics_dict
        }
        
        # Add to history
        self.metrics_history.append(metrics_entry)
        
        # Log in a structured format
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
        with open(filepath, 'w') as f:
            json.dump(self.metrics_history, f, indent=2) 