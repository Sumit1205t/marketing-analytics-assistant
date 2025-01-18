from .metrics_logger import MetricsLogger
from rouge_score import rouge_scorer
import numpy as np

class ResponseEvaluator:
    def __init__(self):
        self.metrics_logger = MetricsLogger()
        self.scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    def evaluate_response(self, response: str, reference: str) -> Dict[str, Any]:
        """Evaluate response quality using ROUGE scores and other metrics"""
        try:
            # Calculate ROUGE scores
            scores = self.scorer.score(reference, response)
            
            metrics = {
                'ROUGE-L': {
                    'precision': scores['rougeL'].precision,
                    'recall': scores['rougeL'].recall,
                    'f1': scores['rougeL'].fmeasure
                },
                'ROUGE-2': {
                    'precision': scores['rouge2'].precision,
                    'recall': scores['rouge2'].recall,
                    'f1': scores['rouge2'].fmeasure
                }
            }
            
            # Add additional metrics
            metrics['response_length'] = len(response.split())
            metrics['reference_length'] = len(reference.split())
            
            # Log the metrics
            self.metrics_logger.log_metrics(metrics)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating response: {str(e)}")
            return {
                'ROUGE-L': {'precision': 0, 'recall': 0, 'f1': 0},
                'ROUGE-2': {'precision': 0, 'recall': 0, 'f1': 0}
            } 