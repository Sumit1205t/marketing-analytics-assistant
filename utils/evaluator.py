from .metrics_logger import MetricsLogger

class ResponseEvaluator:
    def __init__(self):
        self.metrics_logger = MetricsLogger()

    def evaluate_response(self, response, reference):
        # Your existing evaluation code here
        metrics = {
            'ROUGE-L': {
                'precision': rouge_l_precision,
                'recall': rouge_l_recall,
                'f1': rouge_l_f1
            },
            'ROUGE-2': {
                'precision': rouge_2_precision,
                'recall': rouge_2_recall,
                'f1': rouge_2_f1
            },
            'perplexity': perplexity_score
        }
        
        # Log the metrics
        self.metrics_logger.log_metrics(metrics)
        
        return metrics 