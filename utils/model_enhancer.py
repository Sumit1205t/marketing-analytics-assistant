import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import logging
from typing import Dict, List, Any
import re
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from datetime import datetime
import logging

# Try importing plotly, use alternative if not available
try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    logging.warning("Plotly not available. Visualizations will be limited.")

class MarketingModelEnhancer:
    def __init__(self):
        # Set up the logger to track and debug queries
        self.logger = logging.getLogger('model_enhancer')
        logging.basicConfig(level=logging.INFO)

        # Scaler for normalization of continuous marketing metrics
        self.scaler = StandardScaler()

        # Label encoders for marketing categorical features (e.g., ad platforms, campaign names)
        self.label_encoders = {}

        # Define a more comprehensive set of regex patterns for marketing queries
        self.query_patterns = {
            'trend': r'(?i)(trend|growth|change|performance)\s+(?:of|in|for)?\s+([a-zA-Z\s]+)',  # Trend in metrics
            'comparison': r'(?i)(compare|difference|versus|vs)\s+([a-zA-Z\s]+)',  # Comparison of metrics
            'breakdown': r'(?i)(breakdown|split|segment|group)\s+by\s+([a-zA-Z\s]+)',  # Breakdown by category (e.g., channel, region)
            'top_bottom': r'(?i)(top|bottom|best|worst)\s+(\d+)?\s*([a-zA-Z\s]+)',  # Top/Bottom N items (e.g., top 3 channels)
            'correlation': r'(?i)(correlation|relationship|impact)\s+between\s+([a-zA-Z\s]+)',  # Correlation between variables
            'campaign_performance': r'(?i)(performance|effectiveness|ROI)\s+of\s+([a-zA-Z\s]+)',  # Specific campaign performance
            'ad_channel_comparison': r'(?i)(compare|difference|vs)\s+(google|facebook|instagram|linkedin|tiktok|twitter)',  # Specific channel comparison
            'customer_behavior': r'(?i)(engagement|behavior|conversion)\s+(of|for)\s+([a-zA-Z\s]+)',  # Customer behavior by segment
            'kpi_performance': r'(?i)(performance|trends)\s+of\s+([a-zA-Z\s]+)\s+over\s+(the\s+last\s+\d+\s+(days|weeks|months))',  # KPI performance over time
        }

    def process_query(self, query):
        """
        Process and classify a marketing query using regex patterns.
        
        Args:
        - query (str): The user query to be processed.
        
        Returns:
        - dict: A dictionary with query type and identified parameters (if any).
        """
        query_info = {'query': query, 'type': None, 'parameters': None}

        for query_type, pattern in self.query_patterns.items():
            match = re.search(pattern, query)
            if match:
                query_info['type'] = query_type
                query_info['parameters'] = match.groups()
                self.logger.info(f"Matched query: {query} -> Type: {query_type}")
                return query_info
        
        self.logger.warning(f"No match found for query: {query}")
        return query_info

    def encode_data(self, data, column_name):
        """
        Encodes categorical features (e.g., campaign names or ad platforms) for analysis.
        
        Args:
        - data (pd.DataFrame): The dataset containing the categorical column.
        - column_name (str): The name of the column to be encoded.
        
        Returns:
        - pd.DataFrame: Data with the encoded column.
        """
        if column_name not in self.label_encoders:
            encoder = LabelEncoder()
            data[column_name] = encoder.fit_transform(data[column_name])
            self.label_encoders[column_name] = encoder
        else:
            data[column_name] = self.label_encoders[column_name].transform(data[column_name])
        
        return data

    def scale_data(self, data, columns):
        """
        Scales continuous numerical features (e.g., spend, revenue, conversions).
        
        Args:
        - data (pd.DataFrame): The dataset with the numerical columns to be scaled.
        - columns (list): A list of columns to scale.
        
        Returns:
        - pd.DataFrame: Data with scaled columns.
        """
        data[columns] = self.scaler.fit_transform(data[columns])
        return data


    def analyze_marketing_data(self, df: pd.DataFrame, query: str) -> Dict[str, Any]:
        """Enhanced analysis based on specific query intent"""
        try:
            # First, understand query intent and extract relevant parts
            query_intent = self._analyze_query_intent(query)
            
            # Refine DataFrame based on query
            refined_df = self._refine_data_for_query(df, query_intent)
            
            # Process with timeout using ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(self._process_analysis, refined_df, query_intent)
                try:
                    result = future.result(timeout=300)  # 5 minutes timeout
                    return result
                except TimeoutError:
                    self.logger.error("Analysis timeout: exceeded 5 minutes limit")
                    raise TimeoutError("Analysis took too long to complete")
                
        except Exception as e:
            self.logger.error(f"Error in analyze_marketing_data: {str(e)}")
            raise

    def _analyze_query_intent(self, query: str) -> Dict[str, Any]:
        """Analyze query to understand user intent and extract relevant components"""
        intent = {
            'type': None,
            'metrics': [],
            'dimensions': [],
            'time_range': None,
            'filters': {}
        }
        
        # Check each pattern type
        for pattern_type, pattern in self.query_patterns.items():
            matches = re.finditer(pattern, query)
            for match in matches:
                if pattern_type == 'trend':
                    intent['type'] = 'trend_analysis'
                    intent['metrics'].append(match.group(2).strip())
                elif pattern_type == 'comparison':
                    intent['type'] = 'comparative_analysis'
                    intent['dimensions'].append(match.group(2).strip())
                elif pattern_type == 'breakdown':
                    intent['type'] = 'segment_analysis'
                    intent['dimensions'].append(match.group(2).strip())
                elif pattern_type == 'correlation':
                    intent['type'] = 'correlation_analysis'
                    metrics = match.group(2).split('and')
                    intent['metrics'].extend([m.strip() for m in metrics])
        
        return intent

    def _refine_data_for_query(self, df: pd.DataFrame, query_intent: Dict) -> pd.DataFrame:
        """Refine DataFrame based on query intent"""
        try:
            refined_df = df.copy()
            
            # Filter columns based on metrics and dimensions in query
            if query_intent['metrics'] or query_intent['dimensions']:
                relevant_cols = []
                
                # Find relevant metric columns
                for metric in query_intent['metrics']:
                    matching_cols = [col for col in df.columns 
                                   if metric.lower() in col.lower()]
                    relevant_cols.extend(matching_cols)
                
                # Find relevant dimension columns
                for dim in query_intent['dimensions']:
                    matching_cols = [col for col in df.columns 
                                   if dim.lower() in col.lower()]
                    relevant_cols.extend(matching_cols)
                
                # Always include date columns for trend analysis
                date_cols = [col for col in df.columns 
                           if pd.api.types.is_datetime64_any_dtype(df[col])]
                relevant_cols.extend(date_cols)
                
                # Keep only relevant columns
                if relevant_cols:
                    refined_df = refined_df[list(set(relevant_cols))]
            
            return refined_df
            
        except Exception as e:
            self.logger.error(f"Error refining data: {str(e)}")
            return df

    def _process_analysis(self, df: pd.DataFrame, query_intent: Dict) -> Dict[str, Any]:
        """Process analysis based on specific user query only"""
        try:
            insights = {}
            
            # Identify metrics and dimensions from query
            metrics = self._identify_metrics(df, query_intent.get('metrics', []))
            dimensions = self._identify_dimensions(df, query_intent.get('dimensions', []))
            
            # Only process the specific analysis type requested
            if query_intent['type'] == 'trend_analysis':
                # Only analyze trends for metrics mentioned in query
                trends = self._analyze_trends(df, metrics)
                if trends:
                    insights['trend_analysis'] = trends
                    insights['visualizations'] = self._create_query_specific_visualizations(
                        df, query_intent, metrics, dimensions
                    )
                    insights['recommendations'] = self._generate_trend_recommendations(
                        df, metrics, trends
                    )
                    
            elif query_intent['type'] == 'comparative_analysis':
                # Only compare dimensions mentioned in query
                comparisons = self._analyze_comparisons(df, metrics, dimensions)
                if comparisons:
                    insights['comparison_analysis'] = comparisons
                    insights['visualizations'] = self._create_query_specific_visualizations(
                        df, query_intent, metrics, dimensions
                    )
                    insights['recommendations'] = self._generate_comparison_recommendations(
                        comparisons
                    )
            
            # Add LLM analysis focused on query
            insights['llm_analysis'] = self._generate_query_specific_analysis(
                query_intent, insights
            )
            
            return insights
            
        except Exception as e:
            self.logger.error(f"Error in analysis processing: {str(e)}")
            raise

    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess the data for analysis"""
        try:
            processed_df = df.copy()
            
            # Convert date columns
            for col in processed_df.columns:
                if any(date_indicator in col.lower() for date_indicator in ['date', 'time', 'period']):
                    try:
                        processed_df[col] = pd.to_datetime(processed_df[col])
                    except:
                        pass
            
            # Handle numeric columns
            numeric_cols = processed_df.select_dtypes(include=['float64', 'int64']).columns
            for col in numeric_cols:
                processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce')
                processed_df[col] = processed_df[col].fillna(processed_df[col].mean())
            
            return processed_df
            
        except Exception as e:
            self.logger.error(f"Error in preprocessing: {str(e)}")
            raise

    def _create_visualizations(self, df: pd.DataFrame, metrics: List[str], dimensions: List[str]) -> Dict:
        """Create various visualizations for the data"""
        if not PLOTLY_AVAILABLE:
            return {}
            
        try:
            visualizations = {}
            
            # Time series plots for metrics
            date_cols = [col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col])]
            if date_cols:
                main_date_col = date_cols[0]
                for metric in metrics:
                    fig = px.line(df, x=main_date_col, y=metric, title=f'{metric} Over Time')
                    visualizations[f'{metric}_time_series'] = fig.to_json()
            
            # Bar charts for dimensional analysis
            for dim in dimensions:
                for metric in metrics:
                    agg_data = df.groupby(dim)[metric].mean().reset_index()
                    fig = px.bar(agg_data, x=dim, y=metric, title=f'{metric} by {dim}')
                    visualizations[f'{metric}_by_{dim}'] = fig.to_json()
            
            return visualizations
            
        except Exception as e:
            self.logger.error(f"Error creating visualizations: {str(e)}")
            return {}

    def _identify_metrics(self, df: pd.DataFrame, query_metrics: List[str] = None) -> List[str]:
        """Identify metrics from DataFrame based on query and data types"""
        try:
            # Get numeric columns
            numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
            
            if not query_metrics:
                # If no specific metrics requested, return all numeric columns
                return numeric_cols
            
            # Find columns that match requested metrics
            matched_metrics = []
            for metric in query_metrics:
                metric_lower = metric.lower()
                matches = [col for col in numeric_cols 
                          if metric_lower in col.lower()]
                matched_metrics.extend(matches)
            
            # If no matches found, return all numeric columns
            return matched_metrics if matched_metrics else numeric_cols
            
        except Exception as e:
            self.logger.error(f"Error identifying metrics: {str(e)}")
            return []

    def _identify_dimensions(self, df: pd.DataFrame, query_dimensions: List[str] = None) -> List[str]:
        """Identify dimensions from DataFrame based on query and data types"""
        try:
            # Get categorical and datetime columns
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            date_cols = [col for col in df.columns 
                        if pd.api.types.is_datetime64_any_dtype(df[col])]
            
            all_dimension_cols = categorical_cols + date_cols
            
            if not query_dimensions:
                # If no specific dimensions requested, return all dimension columns
                return all_dimension_cols
            
            # Find columns that match requested dimensions
            matched_dims = []
            for dim in query_dimensions:
                dim_lower = dim.lower()
                matches = [col for col in all_dimension_cols 
                          if dim_lower in col.lower()]
                matched_dims.extend(matches)
            
            # If no matches found, return all dimension columns
            return matched_dims if matched_dims else all_dimension_cols
            
        except Exception as e:
            self.logger.error(f"Error identifying dimensions: {str(e)}")
            return []

    def _generate_summary_stats(self, df: pd.DataFrame, metrics: List[str]) -> Dict:
        """Generate summary statistics for key metrics"""
        summary = {}
        for metric in metrics:
            summary[metric] = {
                'mean': df[metric].mean(),
                'median': df[metric].median(),
                'std': df[metric].std(),
                'min': df[metric].min(),
                'max': df[metric].max(),
                'trend': 'increasing' if df[metric].iloc[-1] > df[metric].iloc[0] else 'decreasing'
            }
        return summary

    def _analyze_trends(self, df: pd.DataFrame, metrics: List[str]) -> Dict:
        """Analyze trends only for requested metrics"""
        trends = {}
        try:
            date_cols = [col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col])]
            if date_cols and metrics:
                main_date_col = date_cols[0]
                
                for metric in metrics:
                    try:
                        # Keep original values, no currency conversion
                        df_sorted = df.sort_values(by=main_date_col)
                        
                        # Only calculate basic trend data
                        trend_data = df_sorted.groupby(main_date_col)[metric].mean()
                        
                        # Store raw trend data
                        trends[metric] = {
                            date.strftime('%Y-%m-%d'): value 
                            for date, value in trend_data.items()
                        }
                        
                        # Basic growth calculation if needed
                        if len(trend_data) >= 2:
                            growth_rate = ((trend_data.iloc[-1] - trend_data.iloc[0]) 
                                         / trend_data.iloc[0] * 100)
                            trends[f'{metric}_growth_rate'] = round(growth_rate, 2)
                            
                    except Exception as e:
                        self.logger.warning(f"Error analyzing trend for {metric}: {str(e)}")
                        continue
            
            return trends
            
        except Exception as e:
            self.logger.error(f"Error in trend analysis: {str(e)}")
            return {}

    def _analyze_correlations(self, df: pd.DataFrame, metrics: List[str]) -> Dict:
        """Analyze correlations between metrics"""
        if len(metrics) > 1:
            corr_matrix = df[metrics].corr()
            return {
                'correlation_matrix': corr_matrix.to_dict(),
                'strong_correlations': self._get_strong_correlations(corr_matrix)
            }
        return {}

    def _generate_recommendations(self, df: pd.DataFrame, metrics: List[str]) -> List[Dict]:
        """Generate actionable recommendations based on data analysis"""
        recommendations = []
        
        # Performance recommendations
        for metric in metrics:
            recent_trend = df[metric].iloc[-3:].mean() - df[metric].iloc[-6:-3].mean()
            if recent_trend < 0:
                recommendations.append({
                    'type': 'performance_alert',
                    'metric': metric,
                    'message': f"Declining trend in {metric}. Consider reviewing recent changes."
                })
        
        # Channel recommendations
        if 'channel' in df.columns and 'revenue' in metrics:
            top_channels = df.groupby('channel')['revenue'].mean().nlargest(3)
            recommendations.append({
                'type': 'channel_optimization',
                'message': f"Focus on top performing channels: {', '.join(top_channels.index)}"
            })
        
        return recommendations

    def _get_strong_correlations(self, corr_matrix: pd.DataFrame, threshold: float = 0.7) -> List[Dict]:
        """Extract strong correlations from correlation matrix"""
        strong_corr = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if abs(corr_matrix.iloc[i,j]) >= threshold:
                    strong_corr.append({
                        'metric1': corr_matrix.columns[i],
                        'metric2': corr_matrix.columns[j],
                        'correlation': corr_matrix.iloc[i,j]
                    })
        return strong_corr 

    def _analyze_segments(self, df: pd.DataFrame, metrics: List[str], dimensions: List[str]) -> Dict:
        """Analyze performance across different segments"""
        segment_analysis = {}
        
        try:
            for dimension in dimensions:
                segment_metrics = {}
                
                # Calculate metrics for each segment
                for metric in metrics:
                    # Basic statistics by segment
                    segment_stats = df.groupby(dimension)[metric].agg([
                        'mean', 'median', 'std', 'count'
                    ]).round(2)
                    
                    # Calculate segment performance
                    total_mean = df[metric].mean()
                    segment_stats['performance_vs_avg'] = (
                        (segment_stats['mean'] - total_mean) / total_mean * 100
                    ).round(2)
                    
                    # Identify top and bottom performers
                    top_segments = segment_stats.nlargest(3, 'mean')
                    bottom_segments = segment_stats.nsmallest(3, 'mean')
                    
                    segment_metrics[metric] = {
                        'statistics': segment_stats.to_dict('index'),
                        'top_performers': top_segments.index.tolist(),
                        'bottom_performers': bottom_segments.index.tolist()
                    }
                
                segment_analysis[dimension] = segment_metrics
            
            return segment_analysis
            
        except Exception as e:
            self.logger.error(f"Error in segment analysis: {str(e)}")
            return {} 

    def _handle_missing_data(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Handle missing data in specified columns"""
        try:
            df_clean = df.copy()
            
            for col in columns:
                if col not in df_clean.columns:
                    continue
                    
                if pd.api.types.is_numeric_dtype(df_clean[col]):
                    # Fill numeric columns with median
                    df_clean[col] = df_clean[col].fillna(df_clean[col].median())
                else:
                    # Fill categorical columns with mode or 'Unknown'
                    mode_value = df_clean[col].mode().iloc[0] if not df_clean[col].mode().empty else 'Unknown'
                    df_clean[col] = df_clean[col].fillna(mode_value)
            
            return df_clean
            
        except Exception as e:
            self.logger.error(f"Error handling missing data: {str(e)}")
            return df 

    def _create_query_specific_visualizations(self, df: pd.DataFrame, query_intent: Dict, 
                                            metrics: List[str], dimensions: List[str]) -> Dict:
        """Create visualizations specific to the query type and intent"""
        if not PLOTLY_AVAILABLE:
            return {}
        
        try:
            visualizations = {}
            
            # Get the main date column if it exists
            date_cols = [col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col])]
            main_date_col = date_cols[0] if date_cols else None
            
            if query_intent['type'] == 'trend_analysis' and main_date_col:
                for metric in metrics:
                    try:
                        # Create time series plot with original values
                        fig = px.line(df.sort_values(main_date_col), 
                                    x=main_date_col, 
                                    y=metric,
                                    title=f'{metric} Trend Over Time')
                        fig.update_layout(
                            xaxis_title="Date",
                            yaxis_title=metric.replace('_', ' ').title(),
                            showlegend=True
                        )
                        visualizations[f'{metric}_trend'] = fig.to_json()
                        
                        # Add moving average if enough data points
                        if len(df) > 7:
                            df_ma = df.sort_values(main_date_col).copy()
                            df_ma[f'{metric}_MA7'] = df_ma[metric].rolling(window=7).mean()
                            
                            fig = px.line(df_ma, 
                                        x=main_date_col,
                                        y=[metric, f'{metric}_MA7'],
                                        title=f'{metric} with 7-day Moving Average',
                                        labels={metric: 'Actual', f'{metric}_MA7': '7-day MA'})
                            fig.update_layout(
                                xaxis_title="Date",
                                yaxis_title=metric.replace('_', ' ').title(),
                                showlegend=True
                            )
                            visualizations[f'{metric}_ma'] = fig.to_json()
                            
                    except Exception as e:
                        self.logger.warning(f"Error creating trend visualization for {metric}: {str(e)}")
            
            elif query_intent['type'] == 'comparative_analysis':
                for metric in metrics:
                    for dim in dimensions:
                        try:
                            # Create bar chart with original values
                            agg_data = df.groupby(dim)[metric].agg(['mean', 'std']).reset_index()
                            fig = px.bar(agg_data, 
                                       x=dim, 
                                       y='mean',
                                       error_y='std',
                                       title=f'{metric} by {dim}')
                            fig.update_layout(
                                xaxis_title=dim.replace('_', ' ').title(),
                                yaxis_title=metric.replace('_', ' ').title(),
                                showlegend=True
                            )
                            visualizations[f'{metric}_by_{dim}'] = fig.to_json()
                        except Exception as e:
                            self.logger.warning(f"Error creating comparison visualization: {str(e)}")
            
            return visualizations
            
        except Exception as e:
            self.logger.error(f"Error creating visualizations: {str(e)}")
            return {} 

    def _generate_trend_recommendations(self, df: pd.DataFrame, metrics: List[str], 
                                      trend_analysis: Dict) -> List[Dict]:
        """Generate recommendations based on trend analysis"""
        recommendations = []
        try:
            for metric in metrics:
                # Get trend info
                growth_rate = trend_analysis.get(f"{metric}_growth_rate")
                mom_change = trend_analysis.get(f"{metric}_mom_change")
                trend = trend_analysis.get(f"{metric}_trend")
                
                if growth_rate is not None:
                    if growth_rate < 0:
                        recommendations.append({
                            'message': f"Investigate decline in {metric} (Overall decline: {abs(growth_rate):.1f}%). "
                                     f"Consider reviewing marketing strategy and identifying potential issues.",
                            'priority': 'high' if growth_rate < -10 else 'medium'
                        })
                    elif growth_rate > 20:
                        recommendations.append({
                            'message': f"Strong growth in {metric} ({growth_rate:.1f}%). "
                                     f"Analyze successful factors and consider scaling these strategies.",
                            'priority': 'high'
                        })
                
                if mom_change is not None:
                    if mom_change < -5:
                        recommendations.append({
                            'message': f"Recent decline in {metric} (MoM change: {mom_change:.1f}%). "
                                     f"Take immediate action to address this downward trend.",
                            'priority': 'high'
                        })
                    elif mom_change > 10:
                        recommendations.append({
                            'message': f"Recent improvement in {metric} (MoM change: +{mom_change:.1f}%). "
                                     f"Document and replicate successful recent initiatives.",
                            'priority': 'medium'
                        })
            
            return recommendations[:3]  # Return top 3 recommendations
            
        except Exception as e:
            self.logger.error(f"Error generating trend recommendations: {str(e)}")
            return []

    def _generate_comparison_recommendations(self, comparison_analysis: Dict) -> List[Dict]:
        """Generate recommendations based on comparison analysis"""
        recommendations = []
        try:
            for dimension, metrics in comparison_analysis.items():
                for metric, data in metrics.items():
                    if 'variance' in data and data['variance'] > 0.5:
                        recommendations.append({
                            'message': f"High variance in {metric} across {dimension}s. "
                                     f"Consider standardizing practices to reduce performance gaps.",
                            'priority': 'medium'
                        })
                    
                    if 'top_performer' in data:
                        recommendations.append({
                            'message': f"Best performing {dimension} for {metric}: {data['top_performer']}. "
                                     f"Analyze and replicate successful strategies.",
                            'priority': 'high'
                        })
                    
                    if 'underperforming' in data:
                        recommendations.append({
                            'message': f"Underperforming {dimension}s for {metric}: {', '.join(data['underperforming'])}. "
                                     f"Review and optimize these segments.",
                            'priority': 'high'
                        })
            
            return sorted(recommendations, key=lambda x: x['priority'] == 'high', reverse=True)[:3]
            
        except Exception as e:
            self.logger.error(f"Error generating comparison recommendations: {str(e)}")
            return []

    def _generate_correlation_recommendations(self, correlation_analysis: Dict) -> List[Dict]:
        """Generate recommendations based on correlation analysis"""
        recommendations = []
        try:
            if 'strong_correlations' in correlation_analysis:
                for corr in correlation_analysis['strong_correlations']:
                    if corr['correlation'] > 0.7:
                        recommendations.append({
                            'message': f"Strong positive correlation between {corr['metric1']} and {corr['metric2']} "
                                     f"({corr['correlation']:.2f}). Consider leveraging this relationship in strategy.",
                            'priority': 'high'
                        })
                    elif corr['correlation'] < -0.7:
                        recommendations.append({
                            'message': f"Strong negative correlation between {corr['metric1']} and {corr['metric2']} "
                                     f"({corr['correlation']:.2f}). Investigate potential trade-offs.",
                            'priority': 'high'
                        })
            
            if 'weak_correlations' in correlation_analysis:
                for metric_pair in correlation_analysis['weak_correlations'][:2]:
                    recommendations.append({
                        'message': f"Weak relationship between {metric_pair[0]} and {metric_pair[1]}. "
                                 f"Consider investigating other factors affecting these metrics.",
                        'priority': 'medium'
                    })
            
            return recommendations[:3]
            
        except Exception as e:
            self.logger.error(f"Error generating correlation recommendations: {str(e)}")
            return []

    def _generate_segment_recommendations(self, df: pd.DataFrame, metrics: List[str], 
                                       dimensions: List[str]) -> List[Dict]:
        """Generate recommendations based on segment analysis"""
        recommendations = []
        try:
            for dimension in dimensions:
                for metric in metrics:
                    try:
                        # Calculate segment statistics
                        segment_stats = df.groupby(dimension)[metric].agg(['mean', 'std', 'count'])
                        
                        # Identify top and bottom segments
                        top_segment = segment_stats.nlargest(1, 'mean').index[0]
                        bottom_segment = segment_stats.nsmallest(1, 'mean').index[0]
                        
                        # Calculate overall variance
                        variance = segment_stats['std'].mean() / segment_stats['mean'].mean()
                        
                        if variance > 0.3:
                            recommendations.append({
                                'message': f"High variance in {metric} across {dimension}s. "
                                         f"Focus on reducing performance gaps.",
                                'priority': 'high'
                            })
                        
                        recommendations.append({
                            'message': f"Best performing {dimension} for {metric}: {top_segment}. "
                                     f"Analyze success factors.",
                            'priority': 'medium'
                        })
                        
                        recommendations.append({
                            'message': f"Opportunity to improve {metric} in {bottom_segment} {dimension}. "
                                     f"Review and optimize strategy.",
                            'priority': 'high'
                        })
                        
                    except Exception as e:
                        self.logger.warning(f"Error analyzing {metric} by {dimension}: {str(e)}")
                        continue
            
            return sorted(recommendations, key=lambda x: x['priority'] == 'high', reverse=True)[:3]
            
        except Exception as e:
            self.logger.error(f"Error generating segment recommendations: {str(e)}")
            return [] 

    def _generate_query_specific_analysis(self, query_intent: Dict, insights: Dict) -> str:
        """Generate analysis text focused only on what was asked"""
        try:
            analysis_text = []
            
            if query_intent['type'] == 'trend_analysis' and 'trend_analysis' in insights:
                for metric in query_intent['metrics']:
                    metric_data = insights['trend_analysis'].get(metric, {})
                    if not metric_data:
                        continue
                        
                    # Get actual dates from the data
                    dates = sorted(metric_data.keys())
                    if len(dates) >= 2:
                        start_date = dates[0]
                        end_date = dates[-1]
                        start_value = metric_data[start_date]
                        end_value = metric_data[end_date]
                        
                        # Calculate percentage change
                        pct_change = ((end_value - start_value) / start_value * 100)
                        
                        # Format dates for readability
                        start_date = pd.to_datetime(start_date).strftime('%B %Y')
                        end_date = pd.to_datetime(end_date).strftime('%B %Y')
                        
                        analysis_text.append(
                            f"{metric.title()} {'increased' if pct_change > 0 else 'decreased'} "
                            f"by {abs(pct_change):.1f}% from {start_date} to {end_date} "
                            f"({start_value:,.0f} to {end_value:,.0f})."
                        )
                        
                        # Add trend context if available
                        if len(dates) > 2:
                            # Calculate recent trend (last 3 periods)
                            recent_values = [metric_data[d] for d in dates[-3:]]
                            recent_trend = all(recent_values[i] > recent_values[i-1] 
                                            for i in range(1, len(recent_values)))
                            
                            if recent_trend:
                                analysis_text.append(
                                    f"The metric shows consistent growth in recent months."
                                )
                        
            elif query_intent['type'] == 'comparative_analysis' and 'comparison_analysis' in insights:
                for dim in query_intent['dimensions']:
                    if dim not in insights['comparison_analysis']:
                        continue
                        
                    for metric in query_intent['metrics']:
                        data = insights['comparison_analysis'][dim].get(metric, {})
                        if not data:
                            continue
                            
                        # Get detailed comparison data
                        if 'statistics' in data:
                            stats = data['statistics']
                            top_performer = stats.nlargest(1, 'value').iloc[0]
                            bottom_performer = stats.nsmallest(1, 'value').iloc[0]
                            
                            analysis_text.append(
                                f"For {metric.title()}, {top_performer.name} leads with "
                                f"{top_performer['value']:,.0f}, while {bottom_performer.name} "
                                f"shows {bottom_performer['value']:,.0f}."
                            )
                            
                            # Add performance gap context
                            gap_pct = ((top_performer['value'] - bottom_performer['value']) 
                                     / bottom_performer['value'] * 100)
                            analysis_text.append(
                                f"Performance gap is {gap_pct:.1f}% between top and bottom {dim}s."
                            )
            
            # If no specific analysis could be generated
            if not analysis_text:
                return "Could not find relevant data for your specific query. Please try rephrasing or check if the requested metrics are available."
            
            return "\n".join(analysis_text)
            
        except Exception as e:
            self.logger.error(f"Error generating query analysis: {str(e)}")
            return "Could not generate analysis for your query. Please try again." 

    def _analyze_comparisons(self, df: pd.DataFrame, metrics: List[str], dimensions: List[str]) -> Dict:
        """Analyze comparisons between dimensions for specified metrics"""
        comparisons = {}
        try:
            for dim in dimensions:
                comparisons[dim] = {}
                for metric in metrics:
                    try:
                        # Aggregate data by dimension
                        agg_data = df.groupby(dim)[metric].agg(['mean', 'std', 'count']).reset_index()
                        
                        # Identify top and bottom performers
                        top_performer = agg_data.nlargest(1, 'mean').iloc[0]
                        bottom_performer = agg_data.nsmallest(1, 'mean').iloc[0]
                        
                        # Calculate variance
                        variance = agg_data['std'].mean() / agg_data['mean'].mean() if agg_data['mean'].mean() != 0 else 0
                        
                        # Store comparison data
                        comparisons[dim][metric] = {
                            'statistics': agg_data.to_dict(orient='records'),
                            'top_performer': top_performer[dim],
                            'bottom_performer': bottom_performer[dim],
                            'variance': variance
                        }
                        
                    except Exception as e:
                        self.logger.warning(f"Error analyzing comparison for {metric} by {dim}: {str(e)}")
                        continue
            
            return comparisons
            
        except Exception as e:
            self.logger.error(f"Error in comparison analysis: {str(e)}")
            return {} 