import re
from typing import Optional, Dict, Any, List

class InputValidator:
    @staticmethod
    def validate_email(email: str) -> bool:
        """Validate email format."""
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return bool(re.match(pattern, email))

    @staticmethod
    def validate_phone(phone: str) -> bool:
        """Validate phone number format."""
        # Remove any spaces or special characters except +
        phone = ''.join(filter(lambda x: x.isdigit() or x == '+', phone))
        # Check if it starts with + and contains 10-15 digits
        pattern = r'^\+?[0-9]{10,15}$'
        return bool(re.match(pattern, phone))

    @staticmethod
    def validate_name(name: str) -> bool:
        """Validate name format."""
        # Allow letters, spaces, and common name characters
        pattern = r'^[a-zA-Z\s\'-]+$'
        return bool(re.match(pattern, name))

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
    def __init__(self):
        self.marketing_keywords = {
            'metrics': [
                'revenue', 'sales', 'conversion', 'roi', 'ctr', 'cpc', 'cpa',
                'impressions', 'clicks', 'engagement', 'reach', 'leads',
                'acquisition', 'retention', 'churn', 'ltv', 'arpu', 'income',
                'profit', 'cost', 'spending', 'budget', 'return', 'value',
                'amount', 'total', 'average', 'rate', 'number', 'count'
            ],
            'dimensions': [
                'channel', 'campaign', 'segment', 'audience', 'product',
                'market', 'region', 'platform', 'source', 'medium',
                'category', 'customer', 'demographic', 'location', 'device',
                'education', 'marital_status', 'age', 'gender', 'group',
                'type', 'status', 'level', 'tier', 'period', 'month', 'year'
            ],
            'analysis_types': [
                'trend', 'performance', 'comparison', 'breakdown',
                'attribution', 'forecast', 'segment', 'cohort', 'analyze',
                'show', 'tell', 'explain', 'what', 'how', 'compare',
                'view', 'display', 'see', 'find', 'get', 'calculate',
                'measure', 'track', 'monitor', 'check', 'look', 'give'
            ]
        }

    def validate_query(self, query: str) -> Dict[str, Any]:
        """Validate if query is business/marketing relevant with improved matching"""
        query_lower = query.lower()
        words = query_lower.split()
        
        # Enhanced matching with word boundaries and partial matches
        matches = {}
        for category, keywords in self.marketing_keywords.items():
            matches[category] = []
            for word in keywords:
                # Check for exact word match
                if word in words:
                    matches[category].append(word)
                    continue
                    
                # Check for word as part of another word
                for query_word in words:
                    if (word in query_word or 
                        query_word in word or 
                        self._calculate_similarity(word, query_word) > 0.8):
                        matches[category].append(word)
                        break
        
        # Calculate weighted relevance score with adjusted weights
        weights = {'metrics': 0.35, 'dimensions': 0.3, 'analysis_types': 0.35}
        relevance_score = sum(
            weights[category] * min(len(matches[category]) / 2, 1.0)  # Cap at 1.0
            for category in weights
        )
        
        # Add bonus for longer, more specific queries
        if len(words) >= 4:
            relevance_score += 0.1
        
        return {
            'is_relevant': relevance_score >= 0.1,  # Very lenient threshold
            'relevance_score': relevance_score,
            'matches': matches,
            'suggestions': self._generate_suggestions(matches) if relevance_score < 0.1 else []
        }

    def _calculate_similarity(self, word1: str, word2: str) -> float:
        """Calculate simple string similarity"""
        if len(word1) == 0 or len(word2) == 0:
            return 0.0
        
        # Convert to sets of characters
        set1 = set(word1)
        set2 = set(word2)
        
        # Calculate Jaccard similarity
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        # Ensure union is not zero before division
        return intersection / union if union > 0 else 0.0

    def validate_columns(self, columns: List[str]) -> Dict[str, Any]:
        """Validate if Excel/CSV columns are business relevant"""
        columns_lower = [col.lower() for col in columns]
        
        matches = {
            category: [word for word in words 
                      if any(word in col for col in columns_lower)]
            for category, words in self.marketing_keywords.items()
        }
        
        relevance_score = sum(len(words) for words in matches.values()) / len(columns)
        
        return {
            'is_relevant': relevance_score >= 0.3,
            'relevance_score': relevance_score,
            'matches': matches,
            'irrelevant_columns': [col for col in columns_lower 
                                 if not any(word in col 
                                          for words in self.marketing_keywords.values() 
                                          for word in words)]
        }

    def _generate_suggestions(self, matches: Dict[str, List[str]]) -> List[str]:
        """Generate more helpful suggestions for query improvement"""
        suggestions = []
        
        if not matches['metrics']:
            suggestions.append("Include specific metrics (e.g., revenue, sales, clicks, conversions)")
        if not matches['dimensions']:
            suggestions.append("Add business dimensions (e.g., by channel, by campaign, by region)")
        if not matches['analysis_types']:
            suggestions.append("Specify what you want to know (e.g., show, compare, analyze)")
            
        # Add example queries if no matches found
        if not any(matches.values()):
            suggestions.extend([
                "Example queries:",
                "- Show me the revenue trend over time",
                "- Compare sales by channel",
                "- What is the conversion rate by campaign?"
            ])
            
        return suggestions 