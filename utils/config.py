import os
import ssl
import certifi
import nltk
import urllib3
import logging
from pathlib import Path
import subprocess
import sys

logger = logging.getLogger(__name__)

class Config:
    """Application configuration."""
    
    # Configuration settings
    POSTHOG_ENABLED = False
    POSTHOG_CONFIG = {
        'timeout': 30,
        'max_retries': 3,
        'backoff_factor': 0.5,
        'retry_statuses': [408, 429, 500, 502, 503, 504]
    }
    
    BASE_DIR = os.path.dirname(os.path.dirname(__file__))
    NLTK_DATA_PATH = os.path.join(BASE_DIR, 'nltk_data')
    NLTK_PACKAGES = ['punkt']
    
    @classmethod
    def setup_dependencies(cls):
        """Setup all required dependencies."""
        try:
            # Install required packages if missing
            required_packages = [
                'google-generativeai',
                'pyOpenSSL',
                'certifi',
                'urllib3<2.0.0',
                'cryptography',
                'rouge-score',
                'nltk',
                'scikit-learn',
                'plotly'
            ]
            
            for package in required_packages:
                try:
                    __import__(package.split('>=')[0].split('<')[0].replace('-', '_'))
                except ImportError:
                    logger.info(f"Installing {package}...")
                    subprocess.check_call([
                        sys.executable, 
                        "-m", 
                        "pip", 
                        "install", 
                        package
                    ])
            
            # Setup SSL context
            cls.setup_ssl()
            return True
            
        except Exception as e:
            logger.error(f"Error setting up dependencies: {e}")
            return False

    @classmethod
    def setup_ssl(cls):
        """Setup SSL configuration."""
        try:
            # Use system certificates
            ssl_context = ssl.create_default_context(cafile=certifi.where())
            
            # Configure urllib3 to use system certificates
            urllib3.util.ssl_.DEFAULT_CERTS = certifi.where()
            
            # Disable SSL warnings
            urllib3.disable_warnings()
            
            logger.info("SSL configuration completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error setting up SSL: {e}")
            return False

    @classmethod
    def setup_nltk(cls):
        """Setup NLTK data."""
        try:
            # Create NLTK data directory
            Path(cls.NLTK_DATA_PATH).mkdir(parents=True, exist_ok=True)
            nltk.data.path.append(cls.NLTK_DATA_PATH)
            
            # Download required packages
            for package in cls.NLTK_PACKAGES:
                try:
                    nltk.data.find(f'tokenizers/{package}')
                    logger.info(f"NLTK package '{package}' already exists")
                except LookupError:
                    nltk.download(package, 
                                download_dir=cls.NLTK_DATA_PATH, 
                                quiet=True)
            
            return True
            
        except Exception as e:
            logger.error(f"Error setting up NLTK: {e}")
            return False

    @classmethod
    def setup_posthog(cls):
        """Configure PostHog with better error handling."""
        try:
            if not cls.POSTHOG_ENABLED:
                logger.info("PostHog is disabled")
                return True
                
            import posthog
            from requests.adapters import HTTPAdapter
            from urllib3.util.retry import Retry
            
            # Configure retry strategy
            retry_strategy = Retry(
                total=cls.POSTHOG_CONFIG['max_retries'],
                backoff_factor=cls.POSTHOG_CONFIG['backoff_factor'],
                status_forcelist=cls.POSTHOG_CONFIG['retry_statuses']
            )
            
            # Create HTTP adapter with retry strategy
            adapter = HTTPAdapter(max_retries=retry_strategy)
            
            # Configure PostHog client
            posthog.http.session.mount("https://", adapter)
            posthog.http.session.mount("http://", adapter)
            
            # Set timeout
            posthog.http.session.timeout = cls.POSTHOG_CONFIG['timeout']
            
            logger.info("PostHog configured successfully")
            return True
            
        except Exception as e:
            logger.warning(f"Failed to configure PostHog: {e}. Analytics will be disabled.")
            cls.POSTHOG_ENABLED = False
            return False

    @staticmethod
    def init():
        """Initialize all configurations."""
        try:
            # Setup dependencies
            Config.setup_dependencies()
            
            # Setup NLTK
            Config.setup_nltk()
            
            # Setup PostHog if enabled
            if Config.POSTHOG_ENABLED:
                Config.setup_posthog()
            
            return True
            
        except Exception as e:
            logger.error(f"Error in initialization: {e}")
            return False 