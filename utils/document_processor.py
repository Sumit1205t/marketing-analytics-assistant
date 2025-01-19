import logging
import pandas as pd
import json
from typing import Dict, List, Any
import os
from datetime import datetime
import re
import warnings
import fitz  # PyMuPDF
import concurrent.futures
import time
from PIL import Image
import pytesseract
from pathlib import Path
from utils.data_parser import DataParser
from utils.marketing_analyzer import MarketingAnalyzer
from utils.config import Config
import glob

logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Document processing and analysis handler."""
    
    def __init__(self):
        """Initialize DocumentProcessor."""
        try:
            # Setup logging
            self._setup_logging()
            
            # Initialize analyzer
            self.marketing_analyzer = MarketingAnalyzer()
            
            # Initialize data parser
            self.data_parser = DataParser()
            
            logger.info("DocumentProcessor initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing DocumentProcessor: {e}")
            raise

    def process_document(self, file) -> Dict[str, Any]:
        """Process uploaded document."""
        try:
            if not hasattr(file, 'name'):
                raise ValueError("Invalid file object - missing filename")
                
            logger.info(f"Processing file: {file.name}")
            
            # Get file extension
            file_extension = file.name.split('.')[-1].lower()
            
            # Process based on file type
            if file_extension in ['xlsx', 'xls']:
                # Read Excel file with explicit engine
                df = pd.read_excel(file, engine='openpyxl')
                file_type = 'excel'
            elif file_extension == 'csv':
                # Read CSV file
                df = pd.read_csv(file)
                file_type = 'csv'
            else:
                return {
                    'status': 'error',
                    'message': f'Unsupported file type: {file_extension}'
                }
            
            logger.info(f"Loaded {file_type} file with shape: {df.shape}")
            
            # Basic data validation
            if df.empty:
                return {
                    'status': 'error',
                    'message': 'File contains no data'
                }
            
            # Create processed data structure
            processed_data = {
                'data': df,
                'filename': str(file.name),  # Ensure string
                'file_type': file_type,
                'shape': tuple(df.shape),  # Ensure tuple
                'columns': list(df.columns),  # Ensure list
                'Top summary':df.describe(),
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"Successfully processed {processed_data['filename']}")
            return {
                'status': 'success',
                'data': processed_data
            }
            
        except Exception as e:
            error_msg = f"Error processing file: {str(e)}"
            logger.error(error_msg)
            return {
                'status': 'error',
                'message': error_msg
            }

    def process_query(self, query: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process a natural language query and return relevant analysis."""
        try:
            logger.info(f"Processing query: {query}")
            
            # Extract DataFrame from data
            if isinstance(data, dict) and 'data' in data:
                df = data['data']
            elif isinstance(data, pd.DataFrame):
                df = data
            else:
                raise ValueError("Invalid data format. Expected DataFrame or dict with 'data' key.")
            
            if not isinstance(df, pd.DataFrame):
                raise ValueError("Data must be a pandas DataFrame")
            
            logger.info(f"Processing DataFrame with shape: {df.shape}")
            logger.info(f"Available columns: {df.columns.tolist()}")
            
            # Extract required columns based on query
            required_columns = self._get_required_columns(query, df.columns)
            logger.info(f"Required columns: {required_columns}")
            
            if not required_columns:
                return {
                    'status': 'error',
                    'message': 'Could not determine required columns from query. Please specify metrics or dimensions.'
                }
            
            # Validate required columns exist
            missing_cols = [col for col in required_columns if col not in df.columns]
            if missing_cols:
                return {
                    'status': 'error',
                    'message': f'Missing required columns: {", ".join(missing_cols)}'
                }
            
            # Select required columns
            selected_df = df[required_columns].copy()
            logger.info(f"Selected columns: {selected_df.columns.tolist()}")
            logger.info(f"Found {len(selected_df)} matching records")
            
            # Process through marketing analyzer
            results = self.marketing_analyzer.process_query(query, selected_df)
            
            return results
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {
                'status': 'error',
                'message': f"Query processing error: {str(e)}"
            }

    def _setup_logging(self):
        """Setup logging configuration."""
        try:
            # Create logs directory if it doesn't exist
            log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs')
            os.makedirs(log_dir, exist_ok=True)
            
            # Configure logging
            current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_file = os.path.join(log_dir, f'document_processor_{current_time}.log')
            
            logger.info(f"Logging initialized. Log file: {log_file}")
            
        except Exception as e:
            print(f"Error setting up logging: {e}")

    def _get_required_columns(self, query: str, available_columns: pd.Index) -> List[str]:
        """Determine required columns based on query."""
        required_columns = set()
        query = query.lower()
        
        # Dimension mappings
        dimension_patterns = {
            'customertype': ['customertype', 'customer type', 'customer segment'],
            'tenure_bkt': ['tenure', 'tenure bucket', 'tenure_bkt'],
            'datapackpreference': ['data pack', 'pack preference', 'datapackpreference'],
            'usggrid':['Behaviour Segment','usggrid'],
            'arpu_bucket': ['arpu', 'revenue bucket', 'arpu_bucket'],
            'campaign_id': ['campaign name']
        }
        
        # Metric mappings
        metric_patterns = {
            'hits': ['hits', 'interactions', 'visits'],
            'revenue': ['revenue', 'earnings', 'income'],
            'subs': ['subs', 'subscriptions', 'subscribers']
        }
        
        # Add dimensions from query
        for dim, patterns in dimension_patterns.items():
            if any(pattern in query for pattern in patterns) and dim in available_columns:
                required_columns.add(dim)
                logger.info(f"Found dimension in query: {dim}")
        
        # Add metrics from query
        for metric, patterns in metric_patterns.items():
            if any(pattern in query for pattern in patterns) and metric in available_columns:
                required_columns.add(metric)
                logger.info(f"Found metric in query: {metric}")
        
        # Always include date column for time-based analysis if available
        if 'reward_sent_date' in available_columns:
            required_columns.add('reward_sent_date')
        
        logger.info(f"Identified required columns: {required_columns}")
        return list(required_columns)