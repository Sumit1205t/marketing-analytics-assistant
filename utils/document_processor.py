import fitz  # PyMuPDF
import pandas as pd
import logging
from datetime import datetime
import os
import pytesseract
from PIL import Image
import io
import json
import numpy as np
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from pathlib import Path
import mimetypes  # Replace magic with mimetypes

logger = logging.getLogger(__name__)

@dataclass
class ProcessingError:
    """Class for storing processing errors."""
    error_type: str
    message: str
    context: Dict[str, Any]
    timestamp: datetime = datetime.now()

@dataclass
class ColumnSchema:
    """Schema for data columns."""
    name: str
    aliases: List[str]
    data_type: str
    required: bool = False

class DocumentProcessor:
    def __init__(self, vector_store=None):
        """Initialize DocumentProcessor with supported formats and vector store."""
        self.supported_formats = {
            'pdf': self._process_pdf,
            'csv': self._process_csv,
            'xlsx': self._process_excel
        }
        
        # Define column schemas (can be customized based on your needs)
        self.column_schemas = {
            'date': ColumnSchema(
                name='Date',
                aliases=['date', 'datetime', 'period'],
                data_type='datetime'
            ),
            'revenue': ColumnSchema(
                name='Revenue',
                aliases=['revenue', 'revenue (bn)', 'revenue_bn'],
                data_type='float'
            )
        }
        
        # Initialize mimetypes
        mimetypes.init()
        
        self.valid_mimetypes = {
            '.pdf': 'application/pdf',
            '.csv': 'text/csv',
            '.xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        }
        
        self.vector_store = vector_store
        self.errors: List[ProcessingError] = []
        
        # Configure OCR
        if os.name == 'nt':
            pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

    def process_document(self, file) -> Dict[str, Any]:
        """Synchronous process document method."""
        try:
            # Validate file
            if not self._validate_file(file):
                raise ValueError(f"Invalid or corrupt file: {file.name}")
            
            # Get file extension
            file_extension = Path(file.name).suffix.lower().lstrip('.')
            
            if file_extension not in self.supported_formats:
                raise ValueError(f"Unsupported file format: {file_extension}")
            
            # Process the file using the appropriate method
            content = self.supported_formats[file_extension](file)
            
            # Create document record
            doc_record = {
                'status': 'success',
                'filename': file.name,
                'content': content,
                'doc_type': file_extension,
                'processed_at': datetime.now().isoformat()
            }
            
            # Store in vector database if available
            if self.vector_store:
                if not self.vector_store.add_documents([doc_record]):
                    logger.warning(f"Failed to add document to vector store: {file.name}")
            
            return doc_record
            
        except Exception as e:
            error = ProcessingError(
                error_type=type(e).__name__,
                message=str(e),
                context={'filename': file.name}
            )
            self.errors.append(error)
            logger.error(f"Error processing document {file.name}: {str(e)}")
            return {
                'status': 'error',
                'filename': file.name,
                'error': str(e)
            }

    def _validate_file(self, file) -> bool:
        """Validate file using mimetypes and basic checks."""
        try:
            # Get file extension
            extension = Path(file.name).suffix.lower()
            
            # Check if extension is supported
            if extension not in self.valid_mimetypes:
                logger.error(f"Unsupported file extension: {extension}")
                return False
            
            # Basic file size check (adjust limits as needed)
            file_size = len(file.getvalue())
            if file_size == 0:
                logger.error("Empty file detected")
                return False
            
            # Maximum file size (e.g., 100MB)
            if file_size > 100 * 1024 * 1024:
                logger.error("File too large")
                return False
            
            # For PDFs, try to open with PyMuPDF
            if extension == '.pdf':
                try:
                    doc = fitz.open(stream=file.getvalue())
                    doc.close()
                except Exception as e:
                    logger.error(f"Invalid PDF file: {e}")
                    return False
            
            # For CSV/Excel, try to read first few rows
            elif extension in ['.csv', '.xlsx']:
                try:
                    file.seek(0)
                    if extension == '.csv':
                        pd.read_csv(file, nrows=5)
                    else:
                        pd.read_excel(file, nrows=5)
                except Exception as e:
                    logger.error(f"Invalid spreadsheet file: {e}")
                    return False
                finally:
                    file.seek(0)
            
            return True
            
        except Exception as e:
            logger.error(f"File validation error: {e}")
            return False

    def _process_pdf(self, file) -> str:
        """Process PDF files with parallel page processing."""
        try:
            # Save file temporarily
            temp_path = f"temp_{file.name}"
            with open(temp_path, 'wb') as f:
                f.write(file.getvalue())
            
            # Process PDF pages in parallel
            pdf = fitz.open(temp_path)
            
            with ThreadPoolExecutor() as executor:
                # Process pages in parallel
                futures = [
                    executor.submit(self._process_pdf_page, page)
                    for page in pdf
                ]
                
                # Gather results
                pages_content = []
                for idx, future in enumerate(futures):
                    try:
                        content = future.result()
                        if content:
                            pages_content.append({
                                'page': idx + 1,
                                'content': content
                            })
                    except Exception as e:
                        logger.error(f"Error processing page {idx + 1}: {e}")
            
            # Clean up
            pdf.close()
            os.remove(temp_path)
            
            return json.dumps({
                'document_type': 'pdf',
                'pages': pages_content,
                'total_pages': len(pages_content)
            })
            
        except Exception as e:
            logger.error(f"Error processing PDF {file.name}: {e}")
            raise

    def _process_pdf_page(self, page) -> Optional[str]:
        """Process a single PDF page."""
        try:
            # Try text extraction
            text = page.get_text()
            
            # If no text found, try OCR
            if not text.strip():
                pix = page.get_pixmap()
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                text = pytesseract.image_to_string(img)
            
            return self._clean_text(text)
            
        except Exception as e:
            logger.error(f"Error processing PDF page: {e}")
            return None

    def _process_csv(self, file) -> str:
        """Process CSV files while preserving original structure."""
        try:
            # Read the CSV file
            df = pd.read_csv(file)
            original_columns = list(df.columns)
            
            # Create a copy for processing while preserving original
            processed_df = df.copy()
            
            # Identify and process date columns
            date_columns = self._identify_date_columns(processed_df)
            for col in date_columns:
                processed_df[col] = pd.to_datetime(processed_df[col], errors='coerce')
            
            # Identify and process numeric columns
            numeric_columns = self._identify_numeric_columns(processed_df)
            for col in numeric_columns:
                processed_df[col] = self._convert_to_numeric(processed_df[col])
                
                # Add a formatted column for display
                if col in numeric_columns:
                    processed_df[f"{col}_formatted"] = processed_df[col].apply(
                        lambda x: f"{x/1e9:.2f}B" if pd.notnull(x) else ''
                    )
            
            # Create structured output
            result = {
                'document_type': 'csv',
                'original_structure': {
                    'columns': original_columns,
                    'rows': len(df)
                },
                'processed_data': {
                    'data': processed_df.to_dict(orient='records'),
                    'column_types': {
                        col: str(processed_df[col].dtype) for col in processed_df.columns
                    }
                },
                'metadata': {
                    'date_columns': date_columns,
                    'numeric_columns': numeric_columns,
                    'processed_at': datetime.now().isoformat()
                }
            }
            
            return json.dumps(result, default=str)
            
        except Exception as e:
            logger.error(f"Error processing CSV: {str(e)}")
            raise

    def _identify_date_columns(self, df: pd.DataFrame) -> List[str]:
        """Identify potential date columns."""
        date_columns = []
        
        for col in df.columns:
            col_lower = col.lower()
            # Check if column name matches date patterns
            if any(keyword in col_lower for keyword in ['date', 'time', 'period', 'day']):
                date_columns.append(col)
            # Check if column contains date-like values
            elif df[col].dtype == 'object':
                sample = df[col].dropna().head(10)
                try:
                    pd.to_datetime(sample, errors='raise')
                    date_columns.append(col)
                except:
                    continue
                
        return date_columns

    def _identify_numeric_columns(self, df: pd.DataFrame) -> List[str]:
        """Identify numeric columns including those with currency or percentage symbols."""
        numeric_columns = []
        
        for col in df.columns:
            col_lower = col.lower()
            # Check column name patterns
            if any(keyword in col_lower for keyword in ['revenue', 'amount', 'price', 'cost', 'value']):
                numeric_columns.append(col)
            # Check content patterns
            elif df[col].dtype == 'object':
                sample = df[col].dropna().head(10)
                if sample.str.contains(r'[\d.,]+%?$|^\$[\d.,]+').any():
                    numeric_columns.append(col)
                
        return numeric_columns

    def _convert_to_numeric(self, series: pd.Series) -> pd.Series:
        """Convert string numbers to numeric while handling currency and percentages."""
        if series.dtype in ['int64', 'float64']:
            return series
            
        def clean_number(val):
            if pd.isna(val):
                return val
            if isinstance(val, (int, float)):
                return val
            
            # Convert string to processable format
            val = str(val).strip()
            
            # Remove currency symbols, commas, and whitespace
            val = val.replace('$', '').replace(',', '').strip()
            
            # Handle percentages
            if '%' in val:
                val = val.replace('%', '')
                try:
                    return float(val) / 100
                except:
                    return np.nan
                    
            # Handle 'bn' or 'billion' indicators
            if any(suffix in val.lower() for suffix in ['bn', 'billion']):
                val = val.lower()
                for suffix in ['bn', 'billion']:
                    val = val.replace(suffix, '').strip()
                try:
                    return float(val) * 1e9
                except:
                    return np.nan
                    
            # Handle 'mn' or 'million' indicators
            if any(suffix in val.lower() for suffix in ['mn', 'million', 'm']):
                val = val.lower()
                for suffix in ['mn', 'million', 'm']:
                    val = val.replace(suffix, '').strip()
                try:
                    return float(val) * 1e6
                except:
                    return np.nan
                    
            # Handle 'k' or 'thousand' indicators
            if any(suffix in val.lower() for suffix in ['k', 'thousand']):
                val = val.lower()
                for suffix in ['k', 'thousand']:
                    val = val.replace(suffix, '').strip()
                try:
                    return float(val) * 1e3
                except:
                    return np.nan
            
            # If the number is very large (>1e8), assume it's in raw form
            try:
                num = float(val)
                if num > 1e8:  # If number is greater than 100 million
                    return num  # Return as is
                else:
                    # For smaller numbers, assume they're in billions
                    return num * 1e9
            except:
                return np.nan
                
        return series.apply(clean_number)

    def _process_excel(self, file) -> str:
        """Process Excel files while preserving original structure."""
        try:
            excel_data = pd.read_excel(file, sheet_name=None)
            
            if not excel_data:
                raise ValueError("Excel file contains no valid sheets")
            
            sheets_content = {}
            for sheet_name, df in excel_data.items():
                if df.empty:
                    logger.warning(f"Empty sheet found: {sheet_name}")
                    continue
                
                original_columns = list(df.columns)
                processed_df = df.copy()
                
                # Process date and numeric columns
                date_columns = self._identify_date_columns(processed_df)
                numeric_columns = self._identify_numeric_columns(processed_df)
                
                for col in date_columns:
                    processed_df[col] = pd.to_datetime(processed_df[col], errors='coerce')
                for col in numeric_columns:
                    processed_df[col] = self._convert_to_numeric(processed_df[col])
                
                sheets_content[sheet_name] = {
                    'original_structure': {
                        'columns': original_columns,
                        'rows': len(df)
                    },
                    'processed_data': {
                        'data': processed_df.to_dict(orient='records'),
                        'column_types': {
                            col: str(processed_df[col].dtype) for col in processed_df.columns
                        }
                    },
                    'metadata': {
                        'date_columns': date_columns,
                        'numeric_columns': numeric_columns
                    }
                }
            
            result = {
                'document_type': 'excel',
                'sheets': sheets_content,
                'total_sheets': len(sheets_content),
                'processed_at': datetime.now().isoformat()
            }
            
            return json.dumps(result, default=str)
            
        except Exception as e:
            logger.error(f"Error processing Excel: {str(e)}")
            raise

    def _clean_text(self, text):
        """Clean and structure extracted text."""
        if not text:
            return ""
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Remove common headers/footers patterns
        text = self._remove_headers_footers(text)
        
        return text

    def _clean_dataframe(self, df):
        """Clean and transform DataFrame."""
        # Remove empty rows and columns
        df = df.dropna(how='all')
        df = df.dropna(axis=1, how='all')
        
        # Clean column headers
        df.columns = df.columns.str.strip().str.lower()
        
        # Handle missing values
        df = df.fillna('')
        
        return df

    def _extract_text_from_dataframe(self, df):
        """Extract and concatenate text content from DataFrame."""
        text_content = []
        
        # Convert all columns to string and concatenate
        for _, row in df.iterrows():
            row_text = ' '.join(row.astype(str).values)
            if row_text.strip():
                text_content.append(row_text)
        
        return '\n'.join(text_content)

    def _remove_headers_footers(self, text):
        """Remove common headers and footers patterns."""
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            # Skip common header/footer patterns
            if any(pattern in line.lower() for pattern in ['page', 'confidential', 'all rights reserved']):
                continue
            cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines) 

    def get_processing_errors(self) -> List[Dict[str, Any]]:
        """Get list of processing errors."""
        return [
            {
                'type': error.error_type,
                'message': error.message,
                'context': error.context,
                'timestamp': error.timestamp.isoformat()
            }
            for error in self.errors
        ] 