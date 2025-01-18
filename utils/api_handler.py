import os
import google.generativeai as genai
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class AIHandler:
    """Handler for AI model interactions."""
    
    def __init__(self):
        """Initialize the AI handler with Google's Gemini model."""
        try:
            # Get API key from environment
            api_key = os.getenv('GOOGLE_API_KEY')
            if not api_key:
                raise ValueError("GOOGLE_API_KEY is not set in environment variables")
            
            # Configure Google Gemini
            genai.configure(api_key=api_key)
            
            # Set up the model
            self.model = genai.GenerativeModel('gemini-pro')
            logger.info("Successfully initialized Google Gemini model")
            
        except Exception as e:
            logger.error(f"Failed to initialize Google Gemini configuration: {e}")
            raise

    async def generate_response(self, 
                              prompt: str, 
                              context: Optional[Dict[str, Any]] = None, 
                              **kwargs) -> Dict[str, Any]:
        """Generate a response using Google's Gemini model."""
        try:
            # Prepare the prompt with context if provided
            full_prompt = self._prepare_prompt(prompt, context)
            
            # Generate response
            response = self.model.generate_content(
                full_prompt,
                generation_config={
                    'temperature': kwargs.get('temperature', 0.7),
                    'top_p': kwargs.get('top_p', 0.95),
                    'top_k': kwargs.get('top_k', 40),
                    'max_output_tokens': kwargs.get('max_tokens', 1024)
                }
            )
            
            return {
                'content': response.text,
                'model': 'gemini-pro',
                'status': 'success'
            }
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return {
                'content': str(e),
                'model': 'gemini-pro',
                'status': 'error'
            }

    def _prepare_prompt(self, prompt: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Prepare the prompt with context."""
        if not context:
            return prompt
            
        # Format context into a string
        context_str = "\nContext:\n"
        for key, value in context.items():
            context_str += f"{key}: {value}\n"
            
        return f"{context_str}\n\nQuery: {prompt}"
