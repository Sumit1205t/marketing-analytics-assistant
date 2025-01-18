import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import re

logger = logging.getLogger(__name__)

try:
    from rouge_score import rouge_scorer
except ImportError:
    logger.warning("rouge-score not available, using simplified evaluation")
    rouge_scorer = None

try:
    from nltk.translate.bleu_score import sentence_bleu
except ImportError:
    logger.warning("nltk.translate not available, using simplified evaluation")
    sentence_bleu = None

class MetricType(Enum):
    """Enum for different types of metrics"""
    MONETARY = 'monetary'
    ENGAGEMENT = 'engagement'
    CONVERSION = 'conversion'
    EFFICIENCY = 'efficiency'

@dataclass
class MetricConfig:
    """Configuration for a metric"""
    name: str
    type: MetricType
    aggregations: List[str]
    description: str
    threshold_low: float = 0
    threshold_high: float = float('inf')

class MarketingAnalyzer:
    """Enhanced Marketing Analyzer with ML capabilities and evaluation metrics."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize with optional configuration."""
        self.metrics = {
            'revenue': MetricConfig(
                name='revenue',
                type=MetricType.MONETARY,
                aggregations=['sum', 'mean', 'growth'],
                description='Total revenue generated',
                threshold_low=1000
            ),
            'hits': MetricConfig(
                name='hits',
                type=MetricType.ENGAGEMENT,
                aggregations=['sum', 'mean'],
                description='Number of customer interactions',
                threshold_low=100
            ),
            'subs': MetricConfig(
                name='subs',
                type=MetricType.CONVERSION,
                aggregations=['sum', 'growth'],
                description='Number of subscriptions',
                threshold_low=10
            ),
            'ctr': MetricConfig(
                name='ctr',
                type=MetricType.EFFICIENCY,
                aggregations=['mean'],
                description='Click-through rate',
                threshold_low=0.02,
                threshold_high=0.15
            )
        }
        
        self.dimensions = {
            'customertype': 'Customer segmentation',
            'tenure_bkt': 'Customer tenure bucket',
            'datapackpreference': 'Preferred data package',
            'arpu_bucket': 'Average revenue per user bucket',
            'usggrid': 'Behaviour Segmentation',
            'Campaign_ID':'Campaign Name'
        }
        
        # Update with custom config if provided
        if config:
            self.update_config(config)
        
        # Add ML components
        self.query_classifier = Pipeline([
            ('vectorizer', TfidfVectorizer(ngram_range=(1, 3))),
            ('classifier', MultinomialNB())
        ])
        
        # Training data for query understanding
        self.training_data = {
            'queries': [
                # Metric queries
                "show total revenue",
                "what is the total hits",
                "how many subscriptions",
                "revenue analysis",
                "hits breakdown",
                
                # Trend queries
                "show revenue trend of last 7 days",
                "how are daywise hits trending",
                "subscription growth",
                "average monthly hits trend",
                "performance over time",
                
                # Segment queries
                "revenue by customer type",
                "hits per segment",
                "subscription breakdown by segment",
                "segment performance",
                "customer type analysis",
                "usggrid type analysis",
                
                # Comparison queries
                "compare segments",
                "segment comparison",
                "which segment performs best",
                "top performing segments",
                "top performing campaigns",
                "segment revenue comparison"
            ],
            'intents': [
                'metric', 'metric', 'metric', 'metric', 'metric',
                'trend', 'trend', 'trend', 'trend', 'trend',
                'segment', 'segment', 'segment', 'segment', 'segment', 'segment',
                'comparison', 'comparison', 'comparison', 'comparison', 'comparison', 'comparison'
            ]
        }
        
        # Train the classifier
        self._train_classifier()
        
        # Add evaluation components
        if rouge_scorer:
            self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        else:
            self.rouge_scorer = None
            
        self.quality_thresholds = {
            'rouge_f1_min': 0.3,
            'perplexity_max': 200.0,
            'coherence_min': 0.5
        }

    def update_config(self, config: Dict[str, Any]) -> None:
        """Update configuration with custom settings."""
        if 'metrics' in config:
            self.metrics.update(config['metrics'])
        if 'dimensions' in config:
            self.dimensions.update(config['dimensions'])

    def _train_classifier(self) -> None:
        """Train the query classifier."""
        self.query_classifier.fit(
            self.training_data['queries'],
            self.training_data['intents']
        )

    def _analyze_query_intent(self, query: str) -> Dict[str, Any]:
        """Analyze query intent using ML classifier."""
        try:
            intent = self.query_classifier.predict([query])[0]
            
            # Extract time range if present
            time_range = None
            if 'last 7 days' in query.lower():
                time_range = 'last_7_days'
            elif 'last 30 days' in query.lower():
                time_range = 'last_30_days'
            elif 'last 90 days' in query.lower():
                time_range = 'last_90_days'
            
            # Extract dimensions
            dimensions = []
            for dim in self.dimensions:
                if dim.lower() in query.lower() or self.dimensions[dim].lower() in query.lower():
                    dimensions.append(dim)
            
            # Extract metrics
            metrics = []
            for metric in self.metrics:
                if metric in query.lower():
                    metrics.append(metric)
            
            return {
                'intent': intent,
                'time_range': time_range,
                'dimensions': dimensions,
                'metrics': metrics
            }
            
        except Exception as e:
            logger.error(f"Error analyzing query intent: {e}")
            return {
                'intent': 'unknown',
                'time_range': None,
                'dimensions': [],
                'metrics': []
            }

    def process_query(self, query: str, df: pd.DataFrame) -> Dict[str, Any]:
        """Process a query with dimension-based analysis."""
        try:
            logger.info("="*50)
            logger.info("STARTING QUERY ANALYSIS")
            logger.info(f"Query: {query}")
            logger.info(f"Input data shape: {df.shape}")

            # 1. Query Intent Analysis
            query_intent = self._analyze_query_intent(query)
            logger.info(f"Query intent: {query_intent}")

            # 2. Data Preprocessing
            df = self.preprocess_data(df)
            
            # 3. Extract Analysis Parameters
            dimensions = query_intent.get('dimensions', [])
            metrics = query_intent.get('metrics', [])
            time_range = query_intent.get('time_range')
            
            # Validate dimensions and metrics
            if not dimensions:
                return {
                    'status': 'error',
                    'message': 'No dimensions identified in query. Please specify what you want to analyze (e.g., customertype, tenure, etc.)'
                }
            
            if not metrics:
                return {
                    'status': 'error',
                    'message': 'No metrics identified in query. Please specify what you want to measure (e.g., hits, revenue, etc.)'
                }

            # 4. Data Aggregation
            analysis_results = self._aggregate_data(df, dimensions, metrics, time_range)
            if analysis_results is None or analysis_results.empty:
                return {
                    'status': 'error',
                    'message': 'No data found for the specified dimensions and metrics'
                }

            # 5. Generate Visualizations
            visualizations = self._generate_visualizations(analysis_results, dimensions, metrics)

            # 6. Generate Insights
            insights = self._generate_insights(analysis_results, dimensions, metrics)

            # 7. Generate Recommendations
            recommendations = self._generate_recommendations(analysis_results, dimensions, metrics)

            response = {
                'status': 'success',
                'analysis': {
                    'data': analysis_results.to_dict('records'),
                    'summary': insights,
                    'visualizations': visualizations,
                    'recommendations': recommendations
                }
            }
            
            # 8. Evaluate Response Quality
            if self.rouge_scorer:
                response = self.evaluate_response(response)
            
            logger.info("Analysis completed successfully")
            return response

        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {
                'status': 'error',
                'message': f"Analysis error: {str(e)}"
            }

    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess the data for analysis."""
        try:
            # Create a copy to avoid modifying original
            df = df.copy()
            
            logger.info(f"Initial column types: {df.dtypes.to_dict()}")
            
            # Handle date columns first
            date_cols = [col for col in df.columns if 'date' in col.lower()]
            for col in date_cols:
                try:
                    # Try parsing with different formats
                    if df[col].dtype == 'O':  # Object type
                        df[col] = pd.to_datetime(df[col], errors='coerce')
                        logger.info(f"Converted {col} using flexible parser")
                    
                    logger.info(f"Column {col} type after conversion: {df[col].dtype}")
                    
                except Exception as e:
                    logger.error(f"Error converting date column {col}: {e}")
                    continue
            
            # Handle numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Handle categorical columns
            categorical_cols = df.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                if col not in date_cols:
                    df[col] = df[col].astype('category')
            
            logger.info(f"Final column types: {df.dtypes.to_dict()}")
            logger.info(f"Analyzing data with shape: {df.shape}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error preprocessing data: {e}")
            return df

    def _aggregate_data(self, df: pd.DataFrame, dimensions: List[str], 
                       metrics: List[str], time_range: Optional[str] = None) -> pd.DataFrame:
        """Aggregate data based on dimensions and metrics."""
        try:
            # Ensure all required columns are present
            missing_cols = [col for col in dimensions + metrics if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            # Apply time range filter if specified
            if time_range and 'reward_sent_date' in df.columns:
                df = self._filter_by_time_range(df, time_range)
            
            # Perform aggregation for each metric
            agg_dfs = []
            for metric in metrics:
                # Create aggregation for current metric with observed=True
                agg_df = df.groupby(dimensions, observed=True)[metric].agg([
                    ('sum', 'sum'),
                    ('avg', 'mean'),
                    ('count', 'count')
                ]).reset_index()
                
                # Rename columns to include metric name
                agg_df.columns = [
                    col if col in dimensions 
                    else f'{metric}_{col}'
                    for col in agg_df.columns
                ]
                
                # Calculate percentage
                total = agg_df[f'{metric}_sum'].sum()
                agg_df[f'{metric}_pct'] = (agg_df[f'{metric}_sum'] / total * 100).round(2)
                
                agg_dfs.append(agg_df)
            
            # Merge all aggregated dataframes
            if len(agg_dfs) > 1:
                result = agg_dfs[0]
                for df_right in agg_dfs[1:]:
                    result = result.merge(df_right, on=dimensions)
            else:
                result = agg_dfs[0]
            
            # Sort by the first metric's sum in descending order
            result = result.sort_values(f'{metrics[0]}_sum', ascending=False)
            
            logger.info(f"Aggregation complete. Result shape: {result.shape}")
            return result
            
        except Exception as e:
            logger.error(f"Error in data aggregation: {e}")
            return None

    def _generate_visualizations(self, df: pd.DataFrame, dimensions: List[str], 
                               metrics: List[str]) -> List[Dict[str, Any]]:
        """Generate visualizations based on analysis results."""
        visualizations = []
        
        for metric in metrics:
            # Bar chart for each metric
            visualizations.append({
                'type': 'bar',
                'title': f'{metric.title()} by {dimensions[0].title()}',
                'data': {
                    'x': df[dimensions[0]].tolist(),
                    'y': df[f'{metric}_sum'].tolist()
                }
            })
            
            # Pie chart for percentage distribution
            visualizations.append({
                'type': 'pie',
                'title': f'{metric.title()} Distribution',
                'data': {
                    'labels': df[dimensions[0]].tolist(),
                    'values': df[f'{metric}_pct'].tolist()
                }
            })
        
        return visualizations

    def _generate_insights(self, df: pd.DataFrame, dimensions: List[str], 
                          metrics: List[str]) -> str:
        """Generate insights from analysis results."""
        insights = []
        
        for dimension in dimensions:
            insights.append(f"\n{dimension.title()} Analysis:")
            for metric in metrics:
                total = df[f'{metric}_sum'].sum()
                top_segment = df.iloc[0]
                bottom_segment = df.iloc[-1]
                
                insights.extend([
                    f"• Total {metric}: {total:,.0f}",
                    f"• Top performing {dimension}: {top_segment[dimension]}",
                    f"  - {metric}: {top_segment[f'{metric}_sum']:,.0f} ({top_segment[f'{metric}_pct']:.1f}%)",
                    f"• Lowest performing {dimension}: {bottom_segment[dimension]}",
                    f"  - {metric}: {bottom_segment[f'{metric}_sum']:,.0f} ({bottom_segment[f'{metric}_pct']:.1f}%)"
                ])
        
        return '\n'.join(insights)

    def _generate_recommendations(self, df: pd.DataFrame, dimensions: List[str], 
                                metrics: List[str]) -> List[str]:
        """Generate recommendations based on analysis."""
        recommendations = []
        
        for dimension in dimensions:
            top_segment = df.iloc[0]
            bottom_segment = df.iloc[-1]
            
            recommendations.extend([
                f"• Replicate success factors from top {dimension}: {top_segment[dimension]}",
                f"• Develop improvement plan for {dimension}: {bottom_segment[dimension]}",
                f"• Focus on segments with high potential but lower performance",
                f"• Consider targeted campaigns based on {dimension} performance"
            ])
        
        return recommendations

    def _filter_by_time_range(self, df: pd.DataFrame, time_range: str) -> pd.DataFrame:
        """Filter DataFrame by time range."""
        try:
            end_date = df['reward_sent_date'].max()
            
            if time_range == 'last_7_days':
                start_date = end_date - pd.Timedelta(days=7)
            elif time_range == 'last_30_days':
                start_date = end_date - pd.Timedelta(days=30)
            elif time_range == 'last_90_days':
                start_date = end_date - pd.Timedelta(days=90)
            else:
                return df
            
            return df[df['reward_sent_date'] >= start_date]
            
        except Exception as e:
            logger.error(f"Error filtering by time range: {e}")
            return df

    def evaluate_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate response quality using multiple metrics."""
        try:
            if response['status'] != 'success':
                return response
            
            analysis = response['analysis']
            
            # Calculate evaluation metrics
            metrics = {
                'rouge_scores': self._calculate_rouge_scores(analysis['summary']),
                'perplexity': self._calculate_perplexity(analysis['summary']),
                'coherence': self._calculate_coherence(analysis)
            }
            
            # Add quality indicators
            quality_indicators = self._assess_quality(metrics)
            
            # Add evaluation results to response
            response['evaluation'] = {
                'metrics': metrics,
                'quality_indicators': quality_indicators,
                'suggestions': self._generate_improvement_suggestions(metrics)
            }
            
            logger.info("Response evaluation metrics:")
            logger.info(f"ROUGE Scores: {metrics['rouge_scores']}")
            logger.info(f"Perplexity: {metrics['perplexity']:.2f}")
            logger.info(f"Coherence: {metrics['coherence']:.2f}")
            
            return response
            
        except Exception as e:
            logger.error(f"Error evaluating response: {e}")
            return response

    def _calculate_rouge_scores(self, text: str) -> Dict[str, float]:
        """Calculate ROUGE scores for text quality."""
        try:
            if not self.rouge_scorer:
                return {'rouge1_f1': 0.5, 'rouge2_f1': 0.5, 'rougeL_f1': 0.5}
            
            # Use reference text from training examples
            reference = self._get_reference_text(text)
            scores = self.rouge_scorer.score(reference, text)
            
            return {
                'rouge1_f1': scores['rouge1'].fmeasure,
                'rouge2_f1': scores['rouge2'].fmeasure,
                'rougeL_f1': scores['rougeL'].fmeasure
            }
        except Exception as e:
            logger.error(f"Error calculating ROUGE scores: {e}")
            return {'rouge1_f1': 0.0, 'rouge2_f1': 0.0, 'rougeL_f1': 0.0}

    def _calculate_perplexity(self, text: str) -> float:
        """Calculate perplexity as a measure of text fluency."""
        try:
            # Simple n-gram based perplexity
            words = text.split()
            if len(words) < 2:
                return 0.0
            
            bigrams = list(zip(words[:-1], words[1:]))
            prob = 1.0 / len(set(bigrams))
            return -np.log(prob)
            
        except Exception as e:
            logger.error(f"Error calculating perplexity: {e}")
            return 0.0

    def _calculate_coherence(self, analysis: Dict[str, Any]) -> float:
        """Calculate coherence between insights and data."""
        try:
            # Check alignment between summary and data
            summary = analysis['summary'].lower()
            data = analysis.get('data', [])
            
            # Count mentioned metrics that appear in data
            mentioned_metrics = sum(1 for metric in self.metrics 
                                  if metric in summary and 
                                  any(metric in str(d).lower() for d in data))
            
            coherence = mentioned_metrics / max(1, len(self.metrics))
            return min(1.0, coherence)
            
        except Exception as e:
            logger.error(f"Error calculating coherence: {e}")
            return 0.0

    def _assess_quality(self, metrics: Dict[str, Any]) -> Dict[str, str]:
        """Assess overall response quality."""
        return {
            'rouge': 'good' if metrics['rouge_scores']['rougeL_f1'] > self.quality_thresholds['rouge_f1_min'] else 'needs_improvement',
            'perplexity': 'good' if metrics['perplexity'] < self.quality_thresholds['perplexity_max'] else 'needs_improvement',
            'coherence': 'good' if metrics['coherence'] > self.quality_thresholds['coherence_min'] else 'needs_improvement'
        }

    def _generate_improvement_suggestions(self, metrics: Dict[str, Any]) -> List[str]:
        """Generate suggestions for improving response quality."""
        suggestions = []
        
        if metrics['rouge_scores']['rougeL_f1'] < self.quality_thresholds['rouge_f1_min']:
            suggestions.append("Consider adding more specific details from the data")
        
        if metrics['perplexity'] > self.quality_thresholds['perplexity_max']:
            suggestions.append("Simplify language for better clarity")
        
        if metrics['coherence'] < self.quality_thresholds['coherence_min']:
            suggestions.append("Ensure insights directly relate to the analyzed metrics")
        
        return suggestions

    def _get_reference_text(self, text: str) -> str:
        """Get reference text for ROUGE score calculation."""
        # Use similar examples from training data
        similar_examples = [
            "Analysis shows significant variation in performance across segments.",
            "Key metrics indicate clear patterns in customer behavior.",
            "Data reveals important insights about segment performance."
        ]
        return similar_examples[0]  # For simplicity, use first example