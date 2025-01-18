import pandas as pd
import json
from typing import Dict, List, Union, Optional
import logging
import re
import io

logger = logging.getLogger(__name__)

pd.set_option('display.float_format', lambda x: '%.2f' % x)  # For consistent number formatting

class DataParser:
    """Utility class for parsing and structuring data from various sources"""
    
    @staticmethod
    def parse_excel(file) -> Dict:
        """Parse Excel file into structured format"""
        try:
            # Read all sheets with explicit engine
            excel_data = pd.read_excel(file, sheet_name=None, engine='openpyxl')
            structured_data = {}
            
            for sheet_name, df in excel_data.items():
                if not df.empty:
                    # Clean and standardize the dataframe
                    df = DataParser._standardize_dataframe(df)
                    
                    # Store the processed data
                    structured_data[sheet_name] = {
                        'data': df.to_dict(orient='records'),
                        'metadata': {
                            'columns': list(df.columns),
                            'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
                            'shape': df.shape,
                            'Sample data':df.head()
                        }
                    }
            
            return {
                'type': 'excel',
                'sheets': structured_data,
                'total_sheets': len(structured_data)
            }
            
        except Exception as e:
            logger.error(f"Error parsing Excel file: {str(e)}")
            raise

    @staticmethod
    def _standardize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """Standardize DataFrame columns and data types."""
        try:
            # Clean column names
            df.columns = df.columns.str.strip().str.lower()
            
            logger.info(f"Initial columns: {df.columns.tolist()}")
            
            # Define date formats with examples
            date_formats = [
                {'format': '%d-%b-%Y', 'example': '01-Jan-2025'},  # 01-Jan-2025
                {'format': '%Y-%m-%d', 'example': '2025-01-01'},   # 2025-01-01
                {'format': '%d/%m/%Y', 'example': '01/01/2025'},   # 01/01/2025
                {'format': '%Y/%m/%d', 'example': '2025/01/01'},   # 2025/01/01
                {'format': '%b-%y', 'example': 'Jan-25'},          # Jan-25
                {'format': '%d-%m-%Y', 'example': '01-01-2025'},   # 01-01-2025
                {'format': '%m-%d-%Y', 'example': '01-25-2025'},    # 01-25-2025
                {'format': '%y%m%d', 'example': '20250112'}    # 20250112
            ]

            # Process each column
            for col in df.columns:
                try:
                    if 'date' in col.lower():
                        # Get sample of non-null values
                        sample = df[col].dropna().iloc[:5]
                        logger.info(f"Sample dates from {col}: {sample.tolist()}")
                        
                        # Try each format
                        format_found = False
                        for date_spec in date_formats:
                            try:
                                df[col] = pd.to_datetime(df[col], format=date_spec['format'])
                                logger.info(f"Successfully parsed {col} using format: {date_spec['format']}")
                                format_found = True
                                break
                            except ValueError:
                                continue
                        
                        if not format_found:
                            # Fallback to flexible parser with explicit dayfirst
                            logger.warning(f"Using flexible date parser for {col}")
                            # Check if dates look like they start with day
                            sample_str = str(sample.iloc[0])
                            day_first = bool(re.match(r'\d{1,2}[-/]', sample_str))
                            df[col] = pd.to_datetime(df[col], dayfirst=day_first, errors='coerce')
                        
                    elif any(metric in col.lower() for metric in ['revenue', 'amount', 'cost']):
                        # Handle numeric columns
                        df[col] = pd.to_numeric(df[col].astype(str).str.replace(r'[^\d.-]', '', regex=True), errors='coerce')
                    
                    elif any(metric in col.lower() for metric in ['hits', 'subs', 'count']):
                        # Handle integer columns
                        df[col] = pd.to_numeric(df[col].astype(str).str.replace(r'[^\d]', '', regex=True), errors='coerce')
                    
                    elif any(cat in col.lower() for cat in ['type', 'category', 'status']):
                        # Handle categorical columns
                        df[col] = df[col].astype('category')

                except Exception as e:
                    logger.error(f"Error processing column {col}: {e}")
                    continue

            return df
            
        except Exception as e:
            logger.error(f"Error standardizing DataFrame: {e}")
            raise

    @staticmethod
    def parse_csv(file) -> Dict:
        """Parse CSV file into structured format"""
        try:
            df = pd.read_csv(file)
            
            # Clean column names
            df.columns = df.columns.str.strip().str.lower()
            
            # Convert to records format
            records = df.to_dict(orient='records')
            
            # Get column metadata
            column_types = {
                col: str(dtype) for col, dtype in df.dtypes.items()
            }
            
            return {
                'type': 'csv',
                'data': records,
                'metadata': {
                    'columns': list(df.columns),
                    'types': column_types,
                    'row_count': len(df)
                }
            }
            
        except Exception as e:
            logger.error(f"Error parsing CSV file: {str(e)}")
            raise

    @staticmethod
    def to_dataframe(parsed_data: Dict) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """Convert parsed data back to DataFrame format"""
        try:
            if parsed_data['type'] == 'csv':
                return pd.DataFrame(parsed_data['data'])
            
            elif parsed_data['type'] == 'excel':
                dataframes = {}
                for sheet_name, sheet_data in parsed_data['sheets'].items():
                    df = pd.DataFrame(sheet_data['data'])
                    
                    # Restore data types
                    for col, dtype_str in sheet_data['metadata']['dtypes'].items():
                        if 'datetime' in dtype_str:
                            df[col] = pd.to_datetime(df[col])
                        elif 'float' in dtype_str:
                            df[col] = df[col].astype(float)
                        elif 'int' in dtype_str:
                            df[col] = df[col].astype(float).astype('Int64')  # nullable integer type
                    
                    dataframes[sheet_name] = df
                
                return dataframes
            
        except Exception as e:
            logger.error(f"Error converting to DataFrame: {str(e)}")
            raise

    @staticmethod
    def validate_data(df: pd.DataFrame) -> bool:
        """Validate DataFrame structure and content"""
        try:
            # Basic validation checks
            if df.empty:
                logger.warning("Empty DataFrame")
                return False
            
            logger.info(f"Validating DataFrame with columns: {df.columns.tolist()}")
            logger.info(f"Data types: {df.dtypes.to_dict()}")
            
            # Make column names lowercase for case-insensitive comparison
            df.columns = df.columns.str.lower().str.strip()
            
            # Define patterns for different types of data
            patterns = {
                'date': ['date', 'day', 'time', 'period', 'year'],
                'numeric': ['amount', 'hits', 'subs', 'count', 'total', 'value', 
                           'price', 'revenue', 'cost', 'quantity', 'number'],
                'text': ['name', 'description', 'category', 'type', 'status']
            }
            
            # Check for date columns
            date_cols = df.select_dtypes(include=['datetime64']).columns
            if not len(date_cols):
                date_cols = [col for col in df.columns 
                            if any(pat in col for pat in patterns['date'])]
            
            # Check for numeric columns
            numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
            if not len(numeric_cols):
                numeric_cols = [col for col in df.columns 
                              if any(pat in col for pat in patterns['numeric'])]
            
            logger.info(f"Found date columns: {date_cols}")
            logger.info(f"Found numeric columns: {numeric_cols}")
            
            # Validation criteria:
            # 1. Must have at least 1 row of data
            if len(df) < 1:
                logger.warning("No data rows found")
                return False
            
            # 2. Must have either:
            #    a) At least one date column and one numeric column, or
            #    b) At least two numeric columns
            has_date_and_numeric = len(date_cols) > 0 and len(numeric_cols) > 0
            has_multiple_numeric = len(numeric_cols) >= 2
            
            if not (has_date_and_numeric or has_multiple_numeric):
                logger.warning("Insufficient column types for analysis")
                return False
            
            # 3. Check for excessive missing values
            missing_pct = df[numeric_cols].isnull().mean()
            if (missing_pct > 0.8).any():
                logger.warning("Some numeric columns have excessive missing values")
                # Don't fail validation, just warn
            
            logger.info("Data validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Error in validate_data: {str(e)}")
            return False

    @staticmethod
    def clean_data(df: pd.DataFrame) -> pd.DataFrame:
        """Clean and prepare DataFrame for analysis"""
        try:
            # Remove completely empty rows and columns
            df = df.dropna(how='all', axis=0).dropna(how='all', axis=1)
            
            # Handle date columns
            date_columns = [col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col])]
            for col in date_columns:
                # Create year and month columns
                df[f'{col}_year'] = df[col].dt.year
                df[f'{col}_month'] = df[col].dt.month
            
            # Handle numeric columns
            numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
            for col in numeric_columns:
                # Fill missing values with median
                df[col] = df[col].fillna(df[col].median())
            
            # Handle categorical columns
            categorical_columns = df.select_dtypes(include=['object']).columns
            for col in categorical_columns:
                # Fill missing values with mode
                df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown')
            
            return df
            
        except Exception as e:
            logger.error(f"Error cleaning data: {str(e)}")
            raise 

    def extract_tables_from_text(self, text: str) -> List[pd.DataFrame]:
        """Extract tables from text content using pattern recognition"""
        try:
            tables = []
            lines = text.split('\n')
            current_table = []
            in_table = False
            
            # Patterns that might indicate table content
            table_patterns = {
                'header': r'^[\s]*(?:[A-Za-z]+[\s]*){2,}$',  # Multiple words in a line
                'data': r'^[\s]*(?:\d+[.,\s]*)+$',  # Multiple numbers in a line
                'mixed': r'^[\s]*(?:[A-Za-z]+[\s]*\d+[.,\s]*)+$'  # Mix of words and numbers
            }
            
            for line in lines:
                line = line.strip()
                if not line:
                    if in_table and current_table:
                        # Try to convert current table to DataFrame
                        df = self._convert_to_dataframe(current_table)
                        if df is not None:
                            tables.append(df)
                        current_table = []
                        in_table = False
                    continue
                
                # Check if line matches table patterns
                is_table_row = any(re.match(pattern, line) for pattern in table_patterns.values())
                
                if is_table_row:
                    in_table = True
                    current_table.append(line)
                elif in_table:
                    # Check if this might be part of the table
                    if re.search(r'\d', line) or len(line.split()) > 1:
                        current_table.append(line)
                    else:
                        # End of table
                        if current_table:
                            df = self._convert_to_dataframe(current_table)
                            if df is not None:
                                tables.append(df)
                        current_table = []
                        in_table = False
            
            # Process any remaining table
            if current_table:
                df = self._convert_to_dataframe(current_table)
                if df is not None:
                    tables.append(df)
            
            return tables
            
        except Exception as e:
            logger.error(f"Error extracting tables: {str(e)}")
            return []

    def extract_structured_data_from_text(self, text: str) -> Optional[List[Dict]]:
        """Extract structured data from text using pattern matching"""
        try:
            data_points = []
            current_record = {}
            
            # Patterns for different types of data
            patterns = {
                'date': r'(?i)(?:date|day|time):\s*(\d{1,4}[-/]\d{1,2}[-/]\d{1,4})',
                'numeric': r'(?i)(?:amount|total|value|revenue|cost):\s*[\$£€]?([\d,]+(?:\.\d{2})?)',
                'percentage': r'(?i)(?:percentage|rate|share):\s*(\d+(?:\.\d+)?)\s*%',
                'category': r'(?i)(?:category|type|segment|brand):\s*([A-Za-z\s]+)',
                'metric': r'(?i)(?:metric|measure|kpi):\s*([A-Za-z\s]+)'
            }
            
            # Process text line by line
            lines = text.split('\n')
            for line in lines:
                line = line.strip()
                if not line:
                    if current_record:
                        data_points.append(current_record.copy())
                        current_record = {}
                    continue
                
                # Try to match patterns
                for key, pattern in patterns.items():
                    matches = re.finditer(pattern, line)
                    for match in matches:
                        value = match.group(1)
                        # Convert to appropriate type
                        if key == 'numeric':
                            value = float(value.replace(',', ''))
                        elif key == 'percentage':
                            value = float(value) / 100
                        
                        # Use the pattern type as the column name if not already present
                        col_name = key if key not in current_record else f"{key}_{len([k for k in current_record.keys() if k.startswith(key)])}"
                        current_record[col_name] = value
            
            # Add final record if exists
            if current_record:
                data_points.append(current_record)
            
            if data_points:
                logger.info(f"Extracted {len(data_points)} records from text")
                return data_points
            return None
            
        except Exception as e:
            logger.error(f"Error extracting structured data: {str(e)}")
            return None

    def _convert_to_dataframe(self, table_lines: List[str]) -> Optional[pd.DataFrame]:
        """Convert table lines to DataFrame"""
        try:
            if len(table_lines) < 2:  # Need at least header and one data row
                return None
            
            # Try to determine delimiter and structure
            delimiter = self._detect_delimiter(table_lines)
            
            # Convert to DataFrame
            df = pd.read_csv(io.StringIO('\n'.join(table_lines)), 
                            sep=delimiter,
                            engine='python',
                            error_bad_lines=False)
            
            # Basic validation
            if df.empty or df.shape[1] < 2:
                return None
            
            return df
            
        except Exception as e:
            logger.debug(f"Error converting table lines: {str(e)}")
            return None 

    def _detect_date_format(self, sample_dates: pd.Series) -> str:
        """
        Detect the date format from a sample of dates.
        Returns the most likely format string.
        """
        try:
            # Drop any null values and convert to string
            sample_dates = sample_dates.dropna().astype(str)
            if sample_dates.empty:
                return None
            
            # Get a few samples for analysis
            samples = sample_dates.head(5).tolist()
            logger.info(f"Analyzing date samples: {samples}")
            
            # Common date format patterns
            format_patterns = [
                # MM/DD/YY
                {'pattern': r'^\d{2}/\d{2}/\d{2}$', 'format': '%m/%d/%y'},
                # MM/DD/YYYY
                {'pattern': r'^\d{2}/\d{2}/\d{4}$', 'format': '%m/%d/%Y'},
                # DD/MM/YY
                {'pattern': r'^\d{2}/\d{2}/\d{2}$', 'format': '%d/%m/%y'},
                # DD/MM/YYYY
                {'pattern': r'^\d{2}/\d{2}/\d{4}$', 'format': '%d/%m/%Y'},
                # YYYY-MM-DD
                {'pattern': r'^\d{4}-\d{2}-\d{2}$', 'format': '%Y-%m-%d'},
                # DD-MM-YYYY
                {'pattern': r'^\d{2}-\d{2}-\d{4}$', 'format': '%d-%m-%Y'},
                # YYYYMMDD
                {'pattern': r'^\d{8}$', 'format': '%Y%m%d'}
            ]
            
            import re
            for sample in samples:
                for pattern in format_patterns:
                    if re.match(pattern['pattern'], sample):
                        # Validate if this format works for all samples
                        try:
                            for date_str in samples:
                                pd.to_datetime(date_str, format=pattern['format'])
                            logger.info(f"Detected date format: {pattern['format']}")
                            return pattern['format']
                        except:
                            continue
            
            logger.warning("Could not detect specific date format, using general parser")
            return None
            
        except Exception as e:
            logger.error(f"Error detecting date format: {e}")
            return None

    def process_dates(self, df: pd.DataFrame, date_columns: List[str]) -> pd.DataFrame:
        """Process date columns with explicit format detection."""
        date_formats = [
            '%Y-%m-%d', '%d/%m/%Y', '%m/%d/%Y', 
            '%Y/%m/%d', '%d-%m-%Y', '%m-%d-%Y',
            '%Y%m%d', '%d%m%Y', '%m%d%Y'
        ]
        
        for col in date_columns:
            # Try each format until one works
            detected_format = self._detect_date_format(df[col])
            if detected_format:
                try:
                    df[col] = pd.to_datetime(df[col], format=detected_format)
                    continue
                except:
                    pass
            
            # If no specific format detected, try common formats
            for date_format in date_formats:
                try:
                    df[col] = pd.to_datetime(df[col], format=date_format)
                    break
                except:
                    continue
            
            # If no format works, use the parser with coerce
            if df[col].dtype != 'datetime64[ns]':
                df[col] = pd.to_datetime(df[col], errors='coerce')
            
        return df

    def _validate_date_values(self, dates: pd.Series, format_str: str = None) -> bool:
        """
        Validate if the dates make logical sense.
        """
        try:
            if format_str:
                dates = pd.to_datetime(dates, format=format_str, errors='coerce')
            else:
                dates = pd.to_datetime(dates, errors='coerce')
            
            valid_dates = dates.dropna()
            
            if len(valid_dates) == 0:
                logger.warning("No valid dates found")
                return False
            
            # Check if dates are within reasonable range
            current_year = pd.Timestamp.now().year
            years = valid_dates.dt.year
            
            if (years < current_year - 100).any() or (years > current_year + 10).any():
                logger.warning(f"Found dates outside reasonable range: {years.min()} - {years.max()}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating dates: {e}")
            return False 