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
from dateutil import parser

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
            'campaign_id':'Campaign Name'
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
        """Analyze query intent with enhanced weekday and date pattern detection."""
        try:
            intent = self.query_classifier.predict([query])[0]
            query_lower = query.lower()

            # Enhanced time and weekday patterns
            time_patterns = {
                'last_7_days': [
                    'last 7 days', 'past week', 'last week', 
                    '7 days', 'seven days', 'weekly'
                ],
                'last_30_days': [
                    'last 30 days', 'past month', 'last month', 
                    '30 days', 'thirty days', 'monthly'
                ],
                'last_90_days': [
                    'last 90 days', 'past quarter', 'last quarter',
                    '90 days', 'ninety days', 'quarterly'
                ],
                'last_365_days': [
                    'last 365 days', 'past year', 'last year',
                    'yearly', 'annual', 'this year'
                ]
            }

            weekday_patterns = {
                'monday': ['monday', 'mon'],
                'tuesday': ['tuesday', 'tue'],
                'wednesday': ['wednesday', 'wed'],
                'thursday': ['thursday', 'thu'],
                'friday': ['friday', 'fri'],
                'saturday': ['saturday', 'sat'],
                'sunday': ['sunday', 'sun']
            }
            
            # Check for specific date in query
            date = None
            date_pattern = r'(\d{1,2}-[a-zA-Z]{3}-\d{4})'  # Pattern for date like "2-Jan-2025"
            date_match = re.search(date_pattern, query)
            
            if date_match:
                try:
                    date = parser.parse(date_match.group(0)).date()  # Parse the date to a datetime object
                except Exception as e:
                    logger.error(f"Error parsing date from query: {e}")
                    date = None
            
            # Check for weekday patterns
            weekday = None
            for day, patterns in weekday_patterns.items():
                if any(pattern in query_lower for pattern in patterns):
                    weekday = day
                    break
            
            # Check for time patterns
            time_range = None
            for range_key, patterns in time_patterns.items():
                if any(pattern in query_lower for pattern in patterns):
                    time_range = range_key
                    break
            
            # Set default time range if none specified
            if not time_range:
                time_range = 'last_30_days'
                logger.info("No time range specified, using default: last_30_days")
            
            # Extract dimensions and metrics
            dimensions = []
            for dim in self.dimensions:
                if dim.lower() in query_lower or self.dimensions[dim].lower() in query_lower:
                    dimensions.append(dim)
            
            # Add weekday as dimension if specified
            if weekday:
                date_col = next((col for col in df.columns if 'date' in col.lower()), None)
                if date_col:
                    dimensions.append(f'{date_col}_weekday')
            
            # Extract metrics
            metrics = []
            for metric in self.metrics:
                if metric in query_lower:
                    metrics.append(metric)
            
            # Result formation
            result = {
                'intent': intent,
                'time_range': time_range,
                'dimensions': dimensions,
                'metrics': metrics,
                'weekday': weekday,
                'specific_date': date,  # New field for the specific date
                'is_default_time_range': time_range == 'last_30_days' and not any(
                    pattern in query_lower 
                    for patterns in time_patterns.values() 
                    for pattern in patterns
                )
            }

            logger.info(f"Query analysis result: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing query intent: {e}")
            return {
                'intent': 'unknown',
                'time_range': 'last_30_days',
                'dimensions': [],
                'metrics': [],
                'weekday': None,
                'specific_date': None,  # Default to None if no date is found
                'is_default_time_range': True
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
            
            # Validate metrics
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

            logger.info("Analysis completed successfully")
            return response

        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {
                'status': 'error',
                'message': f"Analysis error: {str(e)}"
            }

    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess the data for analysis with enhanced date features."""
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
                        
                        # Add date-based features
                        df[f'{col}_weekday'] = df[col].dt.day_name()
                        df[f'{col}_day'] = df[col].dt.day
                        df[f'{col}_month'] = df[col].dt.month_name()
                        df[f'{col}_year'] = df[col].dt.year
                        
                        logger.info(f"Added date features for {col}")
                    
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
                if col not in date_cols and not col.endswith(('weekday', 'month')):
                    df[col] = df[col].astype('category')
            
            logger.info(f"Final column types: {df.dtypes.to_dict()}")
            logger.info(f"Analyzing data with shape: {df.shape}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error preprocessing data: {e}")
            return df

    def _aggregate_data(self, df: pd.DataFrame, dimensions: List[str], 
                       metrics: List[str], time_range: str) -> pd.DataFrame:
        """Aggregate data based on dimensions, metrics, and time range."""
        try:
            # Ensure all required columns are present
            required_cols = dimensions + metrics
            if 'reward_sent_date' in df.columns:
                required_cols.append('reward_sent_date')
            
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                logger.error(f"Missing required columns: {missing_cols}")
                return None

            # Apply time range filter
            if time_range and 'reward_sent_date' in df.columns:
                df = self._filter_by_time_range(df, time_range)
                if df.empty:
                    logger.warning("No data after time range filter")
                    return None

            # If no dimensions provided, aggregate by date
            if not dimensions and 'reward_sent_date' in df.columns:
                df['date'] = pd.to_datetime(df['reward_sent_date']).dt.date
                dimensions = ['date']

            # Perform aggregation
            agg_config = {}
            for metric in metrics:
                agg_config[metric] = ['sum', 'mean', 'count']

            agg_df = df.groupby(dimensions, observed=True).agg(agg_config)
            agg_df.columns = [f"{col[0]}_{col[1]}" for col in agg_df.columns]
            agg_df = agg_df.reset_index()

            # Sort by date if it's a time series
            if 'date' in dimensions:
                agg_df = agg_df.sort_values('date')

            logger.info(f"Aggregated data shape: {agg_df.shape}")
            return agg_df

        except Exception as e:
            logger.error(f"Error in data aggregation: {e}")
            return None

    def _generate_visualizations(self, df: pd.DataFrame, dimensions: List[str], 
                               metrics: List[str]) -> List[Dict[str, Any]]:
        """Generate visualizations based on analysis results."""
        visualizations = []
        
        try:
            # Time series visualization
            if 'date' in df.columns:
                for metric in metrics:
                    # Line chart for trend
                    visualizations.append({
                        'type': 'line',
                        'title': f'{metric.title()} Trend Over Time',
                        'data': {
                            'x': df['date'].astype(str).tolist(),
                            'y': df[f'{metric}_sum'].tolist(),
                            'labels': {
                                'x': 'Date',
                                'y': f'Total {metric.title()}'
                            }
                        }
                    })
                    
                    # Moving average for trend smoothing
                    if len(df) > 3:
                        df[f'{metric}_ma'] = df[f'{metric}_sum'].rolling(window=3).mean()
                        visualizations.append({
                            'type': 'line',
                            'title': f'{metric.title()} Trend (3-day Moving Average)',
                            'data': {
                                'x': df['date'].astype(str).tolist(),
                                'y': df[f'{metric}_ma'].tolist(),
                                'labels': {
                                    'x': 'Date',
                                    'y': f'{metric.title()} (Moving Average)'
                                }
                            }
                        })

            # Dimension-based visualizations
            for dimension in dimensions:
                if dimension != 'date':
                    for metric in metrics:
                        # Bar chart
                        visualizations.append({
                            'type': 'bar',
                            'title': f'{metric.title()} by {dimension.title()}',
                            'data': {
                                'x': df[dimension].tolist(),
                                'y': df[f'{metric}_sum'].tolist()
                            }
                        })
                        
                        # Pie chart for distribution
                        visualizations.append({
                            'type': 'pie',
                            'title': f'{metric.title()} Distribution by {dimension.title()}',
                            'data': {
                                'labels': df[dimension].tolist(),
                                'values': df[f'{metric}_sum'].tolist()
                            }
                        })

            logger.info(f"Generated {len(visualizations)} visualizations")
            return visualizations

        except Exception as e:
            logger.error(f"Error generating visualizations: {e}")
            return []

    def _generate_insights(self, df: pd.DataFrame, dimensions: List[str], 
                          metrics: List[str]) -> str:
        """Generate insights from analysis results."""
        try:
            insights = []
            
            # If no dimensions, generate time-based insights
            if not dimensions or (len(dimensions) == 1 and dimensions[0] == 'date'):
                for metric in metrics:
                    total = df[f'{metric}_sum'].sum()
                    avg = df[f'{metric}_mean'].mean()
                    max_val = df[f'{metric}_sum'].max()
                    min_val = df[f'{metric}_sum'].min()
                    
                    insights.extend([
                        f"\nTime-based {metric.title()} Analysis:",
                        f"• Total {metric}: {total:,.0f}",
                        f"• Average {metric} per period: {avg:,.2f}",
                        f"• Highest {metric}: {max_val:,.0f}",
                        f"• Lowest {metric}: {min_val:,.0f}"
                    ])
                    
                    # Calculate period-over-period change
                    if len(df) > 1:
                        change = ((df[f'{metric}_sum'].iloc[-1] - df[f'{metric}_sum'].iloc[0]) 
                                 / df[f'{metric}_sum'].iloc[0] * 100)
                        insights.append(f"• Overall change: {change:+.1f}%")
            
            # Generate dimension-based insights
            else:
                for dimension in dimensions:
                    if dimension != 'date':
                        insights.append(f"\n{dimension.title()} Analysis:")
                        for metric in metrics:
                            total = df[f'{metric}_sum'].sum()
                            
                            # Calculate percentages
                            df[f'{metric}_pct'] = (df[f'{metric}_sum'] / total * 100)
                            
                            # Get top and bottom segments
                            top_segment = df.nlargest(1, f'{metric}_sum').iloc[0]
                            bottom_segment = df.nsmallest(1, f'{metric}_sum').iloc[0]
                            
                            insights.extend([
                                f"• Total {metric}: {total:,.0f}",
                                f"• Top performing {dimension}: {top_segment[dimension]}",
                                f"  - {metric}: {top_segment[f'{metric}_sum']:,.0f} "
                                f"({top_segment[f'{metric}_pct']:.1f}%)",
                                f"• Lowest performing {dimension}: {bottom_segment[dimension]}",
                                f"  - {metric}: {bottom_segment[f'{metric}_sum']:,.0f} "
                                f"({bottom_segment[f'{metric}_pct']:.1f}%)"
                            ])
                            
                            # Add distribution insight
                            if len(df) > 1:
                                std_dev = df[f'{metric}_sum'].std()
                                cv = (std_dev / df[f'{metric}_sum'].mean()) * 100
                                insights.append(
                                    f"• Distribution: {cv:.1f}% coefficient of variation "
                                    f"({'High' if cv > 50 else 'Moderate' if cv > 25 else 'Low'} variability)"
                                )
            
            return '\n'.join(insights)
            
        except Exception as e:
            logger.error(f"Error generating insights: {e}")
            return "Unable to generate insights due to an error."

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