import re
import logging
from typing import Optional, Dict, Any, List, Set, Tuple
import json
import os
from pathlib import Path
import yaml
from difflib import SequenceMatcher
import phonenumbers
from unidecode import unidecode

logger = logging.getLogger(__name__)

class InputValidator:
    """Enhanced input validation with international support."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize with optional custom config."""
        self.config = self._load_config(config_path)
        
    def _load_config(self, config_path: Optional[str] = None) -> Dict:
        """Load validation configuration from file."""
        try:
            default_config = {
                'name': {
                    'min_length': 2,
                    'max_length': 50,
                    'allowed_special_chars': "'-.",
                    'allowed_scripts': ['LATIN', 'DEVANAGARI', 'CJK']
                },
                'phone': {
                    'default_region': 'IN',
                    'allowed_regions': ['IN', 'US', 'GB', 'AU', 'CA', 'ID','SG'],
                    'min_length': 8,
                    'max_length': 15
                },
                'email': {
                    'allowed_domains': ['gmail.com', 'yahoo.com', 'outlook.com'],
                    'min_length': 5,
                    'max_length': 254
                }
            }
            
            if config_path and os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    custom_config = yaml.safe_load(f)
                    return {**default_config, **custom_config}
            
            return default_config
            
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return {}

    def validate_name(self, name: str) -> bool:
        """
        Validate name with support for international characters and scripts.
        """
        try:
            if not name or not isinstance(name, str):
                return False
                
            name = name.strip()
            config = self.config.get('name', {})
            
            # Check length
            if not (config['min_length'] <= len(name) <= config['max_length']):
                return False
            
            # Allow international characters and specified special characters
            allowed_chars = config['allowed_special_chars']
            pattern = f"^[a-zA-Z{re.escape(allowed_chars)}\\s]+$"
            
            if not re.match(pattern, name):
                # Try with unidecode for normalized comparison
                normalized_name = unidecode(name)
                if not re.match(pattern, normalized_name):
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating name: {e}")
            return False

    def validate_phone(self, phone: str, region: Optional[str] = None) -> bool:
        """
        Validate phone numbers with international format support.
        """
        try:
            if not phone:
                return False
                
            config = self.config.get('phone', {})
            region = region or config.get('default_region', 'IN')
            
            # Parse phone number
            try:
                parsed_number = phonenumbers.parse(phone, region)
            except phonenumbers.NumberParseException:
                # Try adding region code if not present
                try:
                    parsed_number = phonenumbers.parse(f"+{phone}" if phone.startswith('91') else phone, region)
                except phonenumbers.NumberParseException:
                    return False
            
            # Validate number
            if not phonenumbers.is_valid_number(parsed_number):
                return False
                
            # Check length
            number_length = len(phonenumbers.format_number(parsed_number, phonenumbers.PhoneNumberFormat.E164))
            if not (config['min_length'] <= number_length <= config['max_length']):
                return False
            
            # Check region
            number_region = phonenumbers.region_code_for_number(parsed_number)
            if number_region not in config['allowed_regions']:
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating phone: {e}")
            return False

    def validate_email(self, email: str) -> bool:
        """
        Enhanced email validation with domain checking.
        """
        try:
            if not email or not isinstance(email, str):
                return False
                
            config = self.config.get('email', {})
            
            # Check length
            if not (config['min_length'] <= len(email) <= config['max_length']):
                return False
            
            # Basic format check
            pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
            if not re.match(pattern, email):
                return False
            
            # Domain check if specified
            if config.get('allowed_domains'):
                domain = email.split('@')[1].lower()
                if domain not in config['allowed_domains']:
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating email: {e}")
            return False

    @staticmethod
    def validate_file_size(file_size: int, max_size_mb: int = 10) -> bool:
        """Validate file size."""
        max_size_bytes = max_size_mb * 1024 * 1024
        return file_size <= max_size_bytes

    @staticmethod
    def validate_file_type(filename: str, allowed_types: list) -> bool:
        """Validate file type."""
        return filename.lower().split('.')[-1] in allowed_types 

class BusinessRelevanceValidator:
    """Enhanced business relevance validation with configurable rules."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize with optional custom config."""
        self.config = self._load_config(config_path)
        self._initialize_keywords()
        
    def _load_config(self, config_path: Optional[str] = None) -> Dict:
        """Load validation configuration from file."""
        try:
            default_config = {
                'weights': {
                    'metrics': 0.35,
                    'dimensions': 0.35,
                    'analysis_types': 0.30
                },
                'thresholds': {
                    'relevance_score': 0.6,
                    'similarity_score': 0.7
                },
                'bonus': {
                    'query_length': 0.1,
                    'specific_terms': 0.05
                }
            }
            
            if config_path and os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    custom_config = yaml.safe_load(f)
                    return {**default_config, **custom_config}
            
            return default_config
            
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return {}

    def _initialize_keywords(self):
        """Initialize keyword sets with synonyms."""
        self.keyword_sets = {
            'metrics': {
                'revenue': ['income', 'earnings', 'sales'],
                'hits': ['visits', 'interactions', 'engagements'],
                'conversion': ['conversions', 'sales', 'purchases'],
                'ctr': ['click-through', 'clickthrough', 'click rate'],
                'roi': ['return', 'investment return', 'profit'],
                'arpu': ['average revenue', 'revenue per user']
            },
            'dimensions': {
                'customertype': ['segment', 'customer segment', 'user type', 'customer type'],
                'tenure': ['duration', 'membership', 'loyalty'],
                'location': ['region', 'area', 'geography'],
                'campaign': ['promotion', 'marketing effort', 'initiative'],
                'product': ['item', 'service', 'offering'],
                'channel': ['medium', 'platform', 'source']
            },
            'analysis_types': {
                'trend': ['trend analysis', 'over time', 'historical'],
                'comparison': ['compare', 'versus', 'against'],
                'distribution': ['breakdown', 'split', 'composition'],
                'correlation': ['relationship', 'connection', 'association'],
                'forecast': ['prediction', 'projection', 'future']
            }
        }

    def validate_query(self, query: str) -> Dict:
        """
        Enhanced query validation with detailed feedback.
        """
        try:
            if not query:
                return self._format_response(0.0, "Empty query")
            
            # Normalize query
            query_terms = set(query.lower().split())
            
            # Calculate scores for each category
            scores = {}
            matches = {}
            
            for category, keywords in self.keyword_sets.items():
                category_score = 0
                category_matches = []
                
                for key_term, synonyms in keywords.items():
                    # Check for exact matches and synonyms
                    all_terms = {key_term} | set(synonyms)
                    
                    for term in query_terms:
                        max_similarity = max(
                            SequenceMatcher(None, term, key).ratio()
                            for key in all_terms
                        )
                        
                        if max_similarity >= self.config['thresholds']['similarity_score']:
                            category_score += max_similarity
                            category_matches.append(key_term)
                
                scores[category] = category_score / len(keywords)
                matches[category] = category_matches
            
            # Calculate weighted score
            weights = self.config['weights']
            total_score = sum(
                scores[cat] * weights[cat]
                for cat in scores
            )
            
            # Apply bonuses
            if len(query_terms) >= 5:
                total_score += self.config['bonus']['query_length']
            
            if len(set.union(*[set(m) for m in matches.values()])) >= 3:
                total_score += self.config['bonus']['specific_terms']
            
            # Generate suggestions
            suggestions = self._generate_suggestions(matches)
            
            return self._format_response(
                total_score,
                "Query validated",
                matches=matches,
                suggestions=suggestions
            )
            
        except Exception as e:
            logger.error(f"Error validating query: {e}")
            return self._format_response(0.0, f"Error: {str(e)}")

    def _generate_suggestions(self, matches: Dict) -> List[str]:
        """Generate improvement suggestions based on matches."""
        suggestions = []
        
        # Check for missing categories
        for category, keywords in self.keyword_sets.items():
            if not matches.get(category):
                suggestions.append(
                    f"Consider adding {category} terms like: "
                    f"{', '.join(list(keywords.keys())[:3])}"
                )
        
        # Suggest combinations
        if len(matches.get('metrics', [])) == 1 and len(matches.get('dimensions', [])) == 1:
            suggestions.append(
                "Try comparing multiple metrics or dimensions for richer insights"
            )
        
        return suggestions

    def _format_response(self, score: float, message: str, **kwargs) -> Dict:
        """Format validation response."""
        response = {
            'score': round(score, 2),
            'is_relevant': score >= self.config['thresholds']['relevance_score'],
            'message': message
        }
        response.update(kwargs)
        return response

    def validate_columns(self, columns: List[str]) -> Dict[str, Any]:
        """Validate if Excel/CSV columns are business relevant"""
        columns_lower = [col.lower() for col in columns]
        
        matches = {
            category: [word for word in words 
                      if any(word in col for col in columns_lower)]
            for category, words in self.keyword_sets.items()
        }
        
        relevance_score = sum(len(words) for words in matches.values()) / len(columns)
        
        return {
            'is_relevant': relevance_score >= 0.3,
            'relevance_score': relevance_score,
            'matches': matches,
            'irrelevant_columns': [col for col in columns_lower 
                                 if not any(word in col 
                                          for words in self.keyword_sets.values() 
                                          for word in words)]
        } 