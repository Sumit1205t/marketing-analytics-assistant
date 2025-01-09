import streamlit as st
import os
from dotenv import load_dotenv
import logging
from datetime import datetime
from utils.validators import InputValidator
from utils.document_processor import DocumentProcessor
from utils.api_handler import AIHandler
from utils.logger import UserLogger
from prompts.system_prompts import SYSTEM_PROMPTS
from prompts.user_prompts import USER_PROMPTS
from prompts.human_prompts import HUMAN_PROMPTS
from utils.vector_store import VectorStore

# Load environment variables first
load_dotenv()

# Initialize components globally
user_logger = None
ai_handler = None

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
    global user_logger, ai_handler
    
    try:
        # Initialize UserLogger
        user_logger = UserLogger()
        st.write("UserLogger initialized successfully")
    except Exception as e:
        st.error(f"Failed to initialize UserLogger: {str(e)}")
        return False

    try:
        # Initialize AIHandler
        ai_handler = AIHandler()
        st.write("AIHandler initialized successfully")
    except Exception as e:
        st.error(f"Failed to initialize AIHandler: {str(e)}")
        return False

    return True

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
                st.experimental_rerun()

def upload_files_page():
    """Render the file upload page."""
    st.title('Upload Your Documents')
    st.write(HUMAN_PROMPTS['file_upload']['instructions'])
    st.info(HUMAN_PROMPTS['file_upload']['tips'])
    
    uploaded_files = st.file_uploader(
        "Choose files",
        type=eval(os.getenv('ALLOWED_FILE_TYPES', '["pdf", "csv", "xlsx"]')),
        accept_multiple_files=True
    )
    
    if uploaded_files:
        progress_bar = st.progress(0)
        processor = DocumentProcessor()
        
        for i, file in enumerate(uploaded_files):
            try:
                result = processor.process_document(file)
                if result['status'] == 'success':
                    st.session_state.processed_documents.append(result)
                progress_bar.progress((i + 1) / len(uploaded_files))
                st.write(f"✓ Processed: {file.name}")
            except Exception as e:
                st.error(f"Error processing {file.name}: {str(e)}")
        
        if st.session_state.processed_documents:
            st.success("Files processed successfully!")
            if st.button("Click here to process documents"):
                user_logger.log_user_activity(
                    {"files": [doc['filename'] for doc in st.session_state.processed_documents]},
                    "document_processing"
                )
                st.session_state.current_page = 'query'
                st.experimental_rerun()

def query_page():
    """Render the query page."""
    st.title('Ask Your Questions')
    st.write(HUMAN_PROMPTS['query_help']['examples'])
    
    if st.session_state.processed_documents:
        st.write("Processed Documents:")
        for doc in st.session_state.processed_documents:
            st.write(f"- {doc['filename']}")
    
    st.write("Suggested queries:")
    for suggestion in USER_PROMPTS['query_suggestions']:
        if st.button(suggestion):
            st.session_state.current_query = suggestion
    
    query = st.text_input("Enter your query:", 
                         value=st.session_state.get('current_query', ''))
    
    if query:
        user_logger.log_user_activity(
            {"query": query},
            "user_query"
        )
        
        context = "\n".join([doc['content'] for doc in st.session_state.processed_documents])
        
        with st.spinner("Processing your query..."):
            try:
                # Get response from AI handler
                response = ai_handler.get_response(query, context)
                
                if isinstance(response, tuple):
                    response_text, df = response
                else:
                    response_text = response
                    df = None
                
                if response_text.startswith("Error"):
                    st.error(response_text)
                else:
                    # Display the text response
                    st.markdown(response_text)
                    
                    # If we have a DataFrame, create visualizations
                    if df is not None and not df.empty:
                        st.subheader("Data Visualization")
                        
                        # Determine if we should create a time series plot
                        if 'date' in df.columns.str.lower():
                            date_col = df.columns[df.columns.str.lower() == 'date'][0]
                            numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
                            
                            for col in numeric_cols:
                                # Line chart
                                st.line_chart(df.set_index(date_col)[col])
                                
                                # Bar chart
                                st.bar_chart(df.set_index(date_col)[col])
                        
                        # Show the raw data
                        st.subheader("Raw Data")
                        st.dataframe(df)
                        
            except Exception as e:
                st.error(f"Error processing query: {str(e)}")
                logger.error(f"Error in query_page: {str(e)}")

def main():
    """Main application entry point."""
    st.title("CMO Assistant")
    
    # Setup logging
    if not setup_logging():
        st.error("Failed to setup logging. Check file permissions.")
        return

    # Initialize components
    vector_store = VectorStore()
    document_processor = DocumentProcessor(vector_store=vector_store)
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