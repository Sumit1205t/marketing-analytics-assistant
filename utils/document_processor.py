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
            
            # Initialize analyzer with default config
            default_config = {
                'metrics': {
                    'revenue': {'type': 'monetary'},
                    'hits': {'type': 'engagement'},
                    'subs': {'type': 'conversion'}
                },
                'dimensions': {
                    'customertype': 'Customer segmentation',
                    'tenure_bkt': 'Customer tenure bucket',
                    'datapackpreference': 'Preferred data package'
                }
            }
            self.marketing_analyzer = MarketingAnalyzer(config=default_config)
            
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
                'Top summary': df.describe(),
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

    def process_query(self, query: str, content: Dict[str, Any]) -> Dict[str, Any]:
        """Process a query against document content."""
        try:
            # Validate query
            if not query or not isinstance(query, str):
                return {
                    'status': 'error',
                    'message': 'Invalid query format'
                }
            
            # Get the DataFrame from content
            if isinstance(content, dict) and 'data' in content:
                df = content['data']
            else:
                return {
                    'status': 'error',
                    'message': 'Invalid data format'
                }
            
            if not isinstance(df, pd.DataFrame):
                return {
                    'status': 'error',
                    'message': 'Data must be a pandas DataFrame'
                }
            
            # Process query using MarketingAnalyzer
            response = self.marketing_analyzer.process_query(query, df)
            
            if response['status'] == 'error':
                logger.error(f"Query processing error: {response['message']}")
            else:
                logger.info("Query processed successfully")
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {
                'status': 'error',
                'message': f'Query processing error: {str(e)}'
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
            'usggrid': ['Behaviour Segment', 'usggrid'],
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