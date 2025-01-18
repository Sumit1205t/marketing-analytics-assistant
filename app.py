import streamlit as st
import os
from dotenv import load_dotenv
import logging
from datetime import datetime
from utils.validators import InputValidator, BusinessRelevanceValidator
from utils.document_processor import DocumentProcessor
from utils.api_handler import AIHandler
from utils.logger import UserLogger
from prompts.system_prompts import SYSTEM_PROMPTS
from prompts.user_prompts import USER_PROMPTS
from prompts.human_prompts import HUMAN_PROMPTS
from utils.vector_store import VectorStore
import pandas as pd
import json
from utils.data_parser import DataParser
from typing import Dict
from utils.config import Config

# Load environment variables first
load_dotenv()

# Initialize components globally
user_logger = None
ai_handler = None
document_processor = None

logger = logging.getLogger(__name__)

def setup_logging():
    """Setup logging configuration."""
    try:
        log_dir = 'logs'
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        log_file = os.path.join(log_dir, 'app.log')
        logging.basicConfig(
            level=getattr(logging, os.getenv('LOG_LEVEL', 'INFO')),
            filename=log_file,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return True
    except Exception as e:
        st.error(f"Failed to setup logging: {str(e)}")
        return False

def initialize_components():
    """Initialize application components."""
    global user_logger, ai_handler, document_processor
    
    try:
        # Initialize Config first
        if not Config.init():
            raise Exception("Failed to initialize configuration")
        logger.info("Config initialized successfully")
        
        # Initialize components
        user_logger = UserLogger()
        ai_handler = AIHandler()
        document_processor = DocumentProcessor()
        
        logger.info("All components initialized successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize components: {e}")
        return False

def user_details_page():
    """Render the user details collection page."""
    st.title(HUMAN_PROMPTS['welcome']['title'])
    st.write(HUMAN_PROMPTS['welcome']['subtitle'])
    
    with st.form(key='user_details_form'):
        name = st.text_input('Name (required)')
        designation = st.text_input('Designation (required)')
        company = st.text_input('Company (required)')
        domain = st.selectbox('Domain', ['Marketing','Finance', 'Human Resource', 'Analytix', 'Others'])
        email = st.text_input('Email ID (required)')
        mobile = st.text_input('Mobile No (required)')

        if st.form_submit_button('Confirm'):
            validator = InputValidator()
            
            if not all([name, designation, company, email, mobile]):
                st.error("All fields are required.")
            elif not validator.validate_name(name):
                st.error("Please enter a valid name.")
            elif not validator.validate_email(email):
                st.error("Please enter a valid email ID.")
            elif not validator.validate_phone(mobile):
                st.error("Please enter a valid phone number.")
            else:
                st.session_state.user_data = {
                    'name': name,
                    'designation': designation,
                    'company': company,
                    'domain': domain,
                    'email': email,
                    'mobile': mobile,
                    'timestamp': datetime.now().isoformat()
                }
                
                user_logger.log_user_activity(
                    st.session_state.user_data,
                    "user_registration"
                )
                
                st.session_state.current_page = 'upload_files'
                st.rerun()

def upload_files_page():
    """Render the file upload page."""
    st.title("Upload Your Files")
    st.write("Please upload your marketing data files (Excel, CSV)")
    
    uploaded_file = st.file_uploader("Choose a file", type=['xlsx', 'xls', 'csv'])
    
    if uploaded_file is not None:
        try:
            # Process the uploaded file
            result = document_processor.process_document(uploaded_file)
            
            if result['status'] == 'success':
                processed_doc = result['data']
                
                # Initialize processed_documents if not exists
                if 'processed_documents' not in st.session_state:
                    st.session_state.processed_documents = []
                
                # Store only necessary information
                doc_info = {
                    'filename': processed_doc['filename'],
                    'file_type': processed_doc['file_type'],
                    'shape': processed_doc['shape'],
                    'columns': processed_doc['columns'],
                    'data': processed_doc['data']  # Store the DataFrame
                }
                
                # Check if file already exists
                existing_files = [doc['filename'] for doc in st.session_state.processed_documents]
                if doc_info['filename'] not in existing_files:
                    st.session_state.processed_documents.append(doc_info)
                    st.success(f"Successfully processed {doc_info['filename']}")
                else:
                    st.warning(f"File {doc_info['filename']} already processed")
                
                # Display processed documents
                if st.session_state.processed_documents:
                    st.write("### Processed Documents:")
                    for doc in st.session_state.processed_documents:
                        st.write(f"- {doc['filename']} ({doc['file_type'].upper()})")
                        st.write(f"  Columns: {', '.join(doc['columns'])}")
                        st.write(f"  Rows: {doc['shape'][0]}")
                
                # Show continue button if at least one document is processed
                if len(st.session_state.processed_documents) > 0:
                    if st.button('Continue to Analysis'):
                        st.session_state.current_page = 'query'
                        st.rerun()
            else:
                st.error(result['message'])
                
        except Exception as e:
            logger.error(f"Error processing file: {str(e)}")
            st.error("Error processing file. Please try again.")

def process_document_content(doc):
    """Process document content into DataFrame using DataParser"""
    try:
        # Validate input document structure
        if not doc or not isinstance(doc, dict):
            logger.error("Invalid document format")
            return []
            
        if 'content' not in doc or 'filename' not in doc or 'doc_type' not in doc:
            logger.error(f"Missing required document fields: {doc.keys() if isinstance(doc, dict) else 'not a dict'}")
            return []

        # Safely parse JSON content
        try:
            content = json.loads(doc['content']) if isinstance(doc['content'], str) else doc['content']
            if not isinstance(content, dict):
                logger.error(f"Invalid content format: expected dict, got {type(content)}")
                return []
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse document content as JSON: {str(e)}")
            return []
        
        parser = DataParser()
        dfs = []
        
        logger.info(f"Processing document: {doc['filename']}, type: {doc['doc_type']}")
        logger.debug(f"Content structure: {list(content.keys())}")
        
        if doc['doc_type'] == 'pdf':
            dfs.extend(process_pdf_content(content, doc['filename'], parser))
        elif doc['doc_type'] == 'csv':
            dfs.extend(process_csv_content(content, doc['filename'], parser))
        elif doc['doc_type'] in ['xlsx', 'xls', 'excel']:
            dfs.extend(process_excel_content(content, doc['filename'], parser))
        else:
            logger.warning(f"Unsupported document type: {doc['doc_type']}")
            
        if not dfs:
            logger.warning(f"No valid data extracted from {doc['filename']}")
            return []
        
        logger.info(f"Successfully extracted {len(dfs)} dataframes from {doc['filename']}")
        return dfs
        
    except Exception as e:
        logger.error(f"Error in process_document_content: {str(e)}")
        logger.exception("Detailed error trace:")
        return []

def process_pdf_content(content, filename, parser):
    """Process PDF content with robust error handling"""
    dfs = []
    try:
        # Validate pages content
        pages_content = content.get('pages', [])
        if not isinstance(pages_content, list):
            logger.error(f"Invalid PDF pages format in {filename}: expected list, got {type(pages_content)}")
            return []
            
        logger.info(f"Processing {len(pages_content)} PDF pages from {filename}")
        
        # Extract tables from PDF content
        tables_data = []
        for page_num, page in enumerate(pages_content, 1):
            if not isinstance(page, dict):
                logger.warning(f"Skipping invalid page format on page {page_num}")
                continue
                
            page_content = page.get('content', '')
            if not isinstance(page_content, str):
                logger.warning(f"Skipping invalid page content type on page {page_num}")
                continue
                
            try:
                # Extract tables using parser
                tables = parser.extract_tables_from_text(page_content)
                if tables:
                    tables_data.extend(tables)
            except Exception as e:
                logger.error(f"Error extracting tables from page {page_num}: {str(e)}")
                continue
        
        # Process extracted tables
        if tables_data:
            logger.info(f"Found {len(tables_data)} tables in PDF")
            for i, table_df in enumerate(tables_data):
                try:
                    if isinstance(table_df, pd.DataFrame) and not table_df.empty:
                        # Standardize and validate table data
                        table_df = parser._standardize_dataframe(table_df)
                        if parser.validate_data(table_df):
                            table_df['source'] = f"{filename} - Table {i+1}"
                            dfs.append(table_df)
                            logger.info(f"Successfully processed table {i+1}")
                except Exception as e:
                    logger.error(f"Error processing table {i+1}: {str(e)}")
                    continue
        
        # Try extracting structured data if no valid tables found
        if not dfs:
            try:
                logger.info("No valid tables found, attempting to extract structured data from text")
                all_text = ' '.join(
                    page.get('content', '') 
                    for page in pages_content 
                    if isinstance(page, dict) and isinstance(page.get('content', ''), str)
                )
                structured_data = parser.extract_structured_data_from_text(all_text)
                if structured_data is not None:
                    df = pd.DataFrame(structured_data)
                    if not df.empty and parser.validate_data(df):
                        df['source'] = filename
                        dfs.append(df)
                        logger.info("Successfully extracted structured data from text")
            except Exception as e:
                logger.error(f"Error extracting structured data: {str(e)}")
                
    except Exception as e:
        logger.error(f"Error processing PDF content: {str(e)}")
        logger.exception("Detailed error trace:")
        
    return dfs

def process_csv_content(content, filename, parser):
    """Process CSV content with robust error handling"""
    dfs = []
    try:
        # Validate CSV data structure
        data = content.get('data')
        if not isinstance(data, (list, dict)):
            logger.error(f"Invalid CSV data format in {filename}")
            return []
            
        try:
            df = pd.DataFrame(data)
            if not df.empty:
                logger.info(f"CSV data shape before validation: {df.shape}")
                logger.debug(f"CSV columns: {df.columns.tolist()}")
                
                if parser.validate_data(df):
                    df['source'] = filename
                    logger.info(f"Successfully validated CSV data: {df.shape}")
                    dfs.append(df)
                else:
                    logger.warning(f"CSV validation failed for {filename}")
        except Exception as e:
            logger.error(f"Error creating DataFrame from CSV data: {str(e)}")
            
    except Exception as e:
        logger.error(f"Error processing CSV content: {str(e)}")
        logger.exception("Detailed error trace:")
        
    return dfs

def process_excel_content(content, filename, parser):
    """Process Excel content with robust error handling"""
    dfs = []
    try:
        # Validate sheets structure
        sheets = content.get('sheets', {})
        if not isinstance(sheets, dict):
            logger.error(f"Invalid Excel sheets format in {filename}")
            return []
            
        for sheet_name, sheet_data in sheets.items():
            try:
                if not isinstance(sheet_data, dict) or 'data' not in sheet_data:
                    logger.warning(f"Invalid sheet data format for sheet {sheet_name}")
                    continue
                    
                df = pd.DataFrame(sheet_data['data'])
                if not df.empty:
                    logger.info(f"Processing sheet {sheet_name}, shape: {df.shape}")
                    logger.debug(f"Sheet columns: {df.columns.tolist()}")
                    
                    if parser.validate_data(df):
                        df['source'] = f"{filename} - {sheet_name}"
                        logger.info(f"Successfully validated sheet {sheet_name}")
                        dfs.append(df)
                    else:
                        logger.warning(f"Validation failed for sheet {sheet_name}")
            except Exception as e:
                logger.error(f"Error processing sheet {sheet_name}: {str(e)}")
                continue
                
    except Exception as e:
        logger.error(f"Error processing Excel content: {str(e)}")
        logger.exception("Detailed error trace:")
        
    return dfs

def display_visualizations(visualizations: Dict):
    """Display plotly visualizations in Streamlit with error handling"""
    try:
        if not visualizations:
            st.info("No visualizations available for this query.")
            return
            
        import plotly.io as pio
        
        # Create columns for better layout
        col1, col2 = st.columns(2)
        
        for viz_name, viz_json in visualizations.items():
            try:
                with col1:
                    fig = pio.from_json(viz_json)
                    st.plotly_chart(fig, use_container_width=True)
                    col1, col2 = col2, col1  # Alternate columns
            except Exception as e:
                st.warning(f"Could not display visualization {viz_name}")
                logger.error(f"Error displaying visualization {viz_name}: {str(e)}")
                continue
                
    except Exception as e:
        st.error("Error displaying visualizations. Please try a different query.")
        logger.error(f"Error in display_visualizations: {str(e)}")

def query_page():
    """Render the query page."""
    st.title("Ask Questions About Your Data")
    
    if not st.session_state.processed_documents:
        st.warning("Please upload some documents first.")
        if st.button("Go to Upload Page"):
            st.session_state.current_page = 'upload_files'
            st.rerun()
        return
    
    # Get the first document's data (assuming single file for now)
    current_doc = st.session_state.processed_documents[0]
    
    # Display query input
    query = st.text_input("Enter your query:", 
                         placeholder="e.g., Show me customertype wise hits and revenue")
    
    if query:
        try:
            # Process the query
            results = document_processor.process_query(query, current_doc)
            
            if results['status'] == 'success':
                analysis = results['analysis']
                
                # Display summary
                st.write("### Analysis Summary")
                st.write(analysis['summary'])
                
                # Display visualizations
                st.write("### Visualizations")
                for viz in analysis['visualizations']:
                    if viz['type'] == 'bar':
                        st.bar_chart(
                            data=pd.DataFrame({
                                'x': viz['data']['x'],
                                'y': viz['data']['y']
                            }).set_index('x')
                        )
                    elif viz['type'] == 'pie':
                        # Use plotly for pie charts
                        import plotly.express as px
                        fig = px.pie(
                            values=viz['data']['values'],
                            names=viz['data']['labels'],
                            title=viz['title']
                        )
                        st.plotly_chart(fig)
                
                # Display recommendations
                st.write("### Recommendations")
                for rec in analysis['recommendations']:
                    st.write(f"• {rec}")
                    
            else:
                st.error(results['message'])
                
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            st.error("Error processing your query. Please try a different question.")

def preview_data(df: pd.DataFrame, max_rows: int = 5) -> None:
    """Preview DataFrame contents with detailed information"""
    try:
        st.write("#### Data Sample")
        st.dataframe(df.head(max_rows))
        
        st.write("#### Column Information")
        info_df = pd.DataFrame({
            'Type': df.dtypes,
            'Non-Null Count': df.count(),
            'Null Count': df.isnull().sum(),
            'Unique Values': [df[col].nunique() for col in df.columns]
        })
        st.dataframe(info_df)
        
        # Show sample values for each column
        st.write("#### Sample Values by Column")
        for col in df.columns:
            unique_values = df[col].dropna().unique()[:5]
            st.write(f"**{col}:** {', '.join(map(str, unique_values))}")
            
    except Exception as e:
        logger.error(f"Error in data preview: {e}")
        st.error("Error displaying data preview")

def display_analysis_results(response: Dict, query: str):
    """Display analysis results in an organized format"""
    try:
        # Create tabs for organized display
        analysis_tab, viz_tab, insights_tab = st.tabs([
            "📊 Analysis", "📈 Visualizations", "💡 Insights & Actions"
        ])
        
        with analysis_tab:
            if response.get('llm_analysis'):
                st.markdown(response['llm_analysis'])
            
            # Show relevant data tables based on query
            if 'segment_analysis' in response.get('ml_insights', {}):
                relevant_segments = response['ml_insights']['segment_analysis']
                if relevant_segments:
                    st.write("#### 📋 Relevant Data")
                    for dim, metrics in relevant_segments.items():
                        if any(keyword in query.lower() for keyword in [dim.lower(), 'segment', 'breakdown']):
                            st.write(f"**{dim.title()} Analysis:**")
                            for metric, data in metrics.items():
                                if 'statistics' in data:
                                    df = pd.DataFrame(data['statistics']).round(2)
                                    st.dataframe(df)
        
        with viz_tab:
            # Display relevant visualizations
            if response.get('visualizations'):
                display_visualizations(response['visualizations'])
            
            # Show trend analysis if relevant
            if 'trend_analysis' in response.get('ml_insights', {}):
                trends = response['ml_insights']['trend_analysis']
                if any(keyword in query.lower() for keyword in ['trend', 'over time', 'growth']):
                    st.write("#### 📈 Trend Analysis")
                    for metric, values in trends.items():
                        if isinstance(values, dict) and not metric.endswith(('_growth_rate', '_mom_change', '_trend')):
                            df = pd.DataFrame(list(values.items()), columns=['Date', 'Value'])
                            st.line_chart(df.set_index('Date'))
        
        with insights_tab:
            st.write("### 🎯 Key Insights & Actions")
            
            # Display correlations if relevant
            if 'correlation_analysis' in response.get('ml_insights', {}):
                correlations = response['ml_insights']['correlation_analysis']
                if correlations.get('strong_correlations'):
                    st.write("#### 🔗 Key Relationships")
                    for corr in correlations['strong_correlations']:
                        st.write(f"- Strong relationship between {corr['metric1']} and {corr['metric2']} "
                                f"(correlation: {corr['correlation']:.2f})")
            
            # Display recommendations
            if response.get('ml_insights', {}).get('recommendations'):
                st.write("#### 📝 Recommended Actions")
                for rec in response['ml_insights']['recommendations']:
                    st.write(f"- {rec['message']}")
            
            # Display trend insights
            if 'trend_analysis' in response.get('ml_insights', {}):
                trends = response['ml_insights']['trend_analysis']
                trend_metrics = [k for k in trends.keys() if k.endswith('_trend')]
                if trend_metrics:
                    st.write("#### 📈 Trend Insights")
                    for metric in trend_metrics:
                        base_metric = metric.replace('_trend', '')
                        growth_rate = trends.get(f"{base_metric}_growth_rate")
                        mom_change = trends.get(f"{base_metric}_mom_change")
                        
                        st.write(f"- **{base_metric.title()}**: {trends[metric].title()}")
                        if growth_rate is not None:
                            st.write(f"  - Overall growth: {growth_rate:.1f}%")
                        if mom_change is not None:
                            st.write(f"  - Month-over-month change: {mom_change:.1f}%")
    
    except Exception as e:
        logger.error(f"Error displaying analysis results: {str(e)}")
        st.error("Error displaying analysis results. Please try again.")

def handle_continue_analysis():
    if not st.session_state.get('processed_content'):
        st.error("Please upload and process a document first")
        return False
        
    try:
        # Validate processed content
        if not isinstance(st.session_state.processed_content, dict):
            st.error("Invalid processed content format")
            return False
            
        # Store current page state
        st.session_state.current_page = 'analysis'
        return True
        
    except Exception as e:
        st.error(f"Error transitioning to analysis: {str(e)}")
        return False

def handle_query(query: str):
    if not st.session_state.get('processed_content'):
        st.error("Please upload a document first")
        return
        
    try:
        # Process query against document content only
        response = document_processor.process_query(query, st.session_state.processed_content)
        
        if response['status'] == 'error':
            st.error(response['message'])
            return
            
        if response['response']['source'] == 'no_relevant_content':
            st.warning(response['response']['answer'])
            return
            
        # Display response with context
        st.write("Answer:", response['response']['answer'])
        
        # Optionally show source context
        with st.expander("View Source Context"):
            st.json(response['response']['context'])
            
    except Exception as e:
        st.error(f"Error processing query: {str(e)}")

def main():
    """Main application entry point."""
    st.title("CMO Assistant")
    
    # Setup logging
    if not setup_logging():
        st.error("Failed to setup logging. Check file permissions.")
        return

    # Initialize components
    if not initialize_components():
        st.error("Failed to initialize components. Please check the errors above.")
        return

    # Initialize session state
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 'user_details'
    if 'user_data' not in st.session_state:
        st.session_state.user_data = {}
    if 'processed_documents' not in st.session_state:
        st.session_state.processed_documents = []

    # Display current page
    pages = {
        'user_details': user_details_page,
        'upload_files': upload_files_page,
        'query': query_page
    }
    
    current_page = st.session_state.current_page
    if current_page in pages:
        pages[current_page]()
    else:
        st.error(f"Unknown page: {current_page}")

if __name__ == "__main__":
    main() 