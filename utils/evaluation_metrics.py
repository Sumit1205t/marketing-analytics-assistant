from rouge_score import rouge_scorer
import numpy as np
from math import exp
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)

class ResponseEvaluator:
    def __init__(self):
        """Initialize the evaluator with ROUGE scorer and NLTK downloads."""
        try:
            # Download required NLTK data
            nltk.download('punkt', quiet=True)
            
            # Initialize ROUGE scorer with ROUGE-L and ROUGE-2 (instead of ROUGE-S)
            self.scorer = rouge_scorer.RougeScorer(
                ['rougeL', 'rouge2'],  # Changed from 'rougeS' to 'rouge2'
                use_stemmer=True
            )
            
            logger.info("ResponseEvaluator initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize ResponseEvaluator: {e}")
            raise

    def evaluate_response(self, 
                         generated_text: str, 
                         reference_text: str) -> Dict[str, Dict[str, float]]:
        """
        Evaluate generated text against reference using multiple metrics.
        
        Args:
            generated_text: The model-generated response
            reference_text: The reference/ground truth text
            
        Returns:
            Dictionary containing ROUGE scores and perplexity
        """
        try:
            # Calculate ROUGE scores
            rouge_scores = self.calculate_rouge_scores(generated_text, reference_text)
            
            # Calculate perplexity
            perplexity = self.calculate_perplexity(generated_text)
            
            # Combine all metrics
            evaluation_results = {
                'rouge_scores': rouge_scores,
                'perplexity': perplexity
            }
            
            return evaluation_results
            
        except Exception as e:
            logger.error(f"Error in evaluate_response: {e}")
            return {
                'error': str(e)
            }

    def calculate_rouge_scores(self, 
                             generated_text: str, 
                             reference_text: str) -> Dict[str, Dict[str, float]]:
        """Calculate ROUGE-L and ROUGE-2 scores."""
        try:
            # Get ROUGE scores
            scores = self.scorer.score(reference_text, generated_text)
            
            # Format scores
            rouge_scores = {
                'rouge_l': {
                    'precision': scores['rougeL'].precision,
                    'recall': scores['rougeL'].recall,
                    'f1': scores['rougeL'].fmeasure
                },
                'rouge_2': {  # Changed from 'rouge_s' to 'rouge_2'
                    'precision': scores['rouge2'].precision,
                    'recall': scores['rouge2'].recall,
                    'f1': scores['rouge2'].fmeasure
                }
            }
            
            return rouge_scores
            
        except Exception as e:
            logger.error(f"Error calculating ROUGE scores: {e}")
            raise

    def calculate_perplexity(self, text: str) -> float:
        """
        Calculate perplexity score for the generated text.
        Lower perplexity indicates more fluent text.
        """
        try:
            # Tokenize text into sentences and words
            sentences = sent_tokenize(text)
            words = word_tokenize(text)
            
            if not words:
                return float('inf')
            
            # Calculate n-gram probabilities (using bigrams)
            bigrams = list(nltk.bigrams(words))
            word_freq = nltk.FreqDist(words)
            bigram_freq = nltk.FreqDist(bigrams)
            
            # Calculate log probability
            log_prob = 0.0
            N = len(words)
            
            for bigram in bigrams:
                # Use bigram probability with smoothing
                bigram_count = bigram_freq[bigram]
                word_count = word_freq[bigram[0]]
                
                # Add-1 smoothing
                prob = (bigram_count + 1) / (word_count + len(word_freq))
                log_prob += np.log2(prob)
            
            # Calculate perplexity
            perplexity = exp(-log_prob / N)
            
            return perplexity
            
        except Exception as e:
            logger.error(f"Error calculating perplexity: {e}")
            return float('inf')

    def get_evaluation_summary(self, 
                             evaluation_results: Dict[str, Dict[str, float]]) -> str:
        """Generate a human-readable summary of evaluation metrics."""
        try:
            summary = []
            
            if 'rouge_scores' in evaluation_results:
                # Add ROUGE-L scores
                rouge_l = evaluation_results['rouge_scores']['rouge_l']
                summary.append("ROUGE-L Scores:")
                summary.append(f"- Precision: {rouge_l['precision']:.4f}")
                summary.append(f"- Recall: {rouge_l['recall']:.4f}")
                summary.append(f"- F1: {rouge_l['f1']:.4f}")
                
                # Add ROUGE-2 scores
                rouge_2 = evaluation_results['rouge_scores']['rouge_2']
                summary.append("\nROUGE-2 Scores:")
                summary.append(f"- Precision: {rouge_2['precision']:.4f}")
                summary.append(f"- Recall: {rouge_2['recall']:.4f}")
                summary.append(f"- F1: {rouge_2['f1']:.4f}")
            
            if 'perplexity' in evaluation_results:
                # Add Perplexity
                summary.append(f"\nPerplexity Score: {evaluation_results['perplexity']:.2f}")
            
            # Add interpretation
            summary.append("\nInterpretation:")
            summary.append("- ROUGE scores range from 0 to 1 (higher is better)")
            summary.append("- Perplexity: lower scores indicate more fluent text")
            
            return "\n".join(summary)
            
        except Exception as e:
            logger.error(f"Error generating evaluation summary: {e}")
            return f"Error generating summary: {str(e)}" 