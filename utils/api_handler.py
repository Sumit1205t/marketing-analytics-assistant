import os
import logging
import httpx
import json
from .evaluation_metrics import ResponseEvaluator

logger = logging.getLogger(__name__)

class AIHandler:
    def __init__(self):
        self.setup_api_client()
        self.evaluator = ResponseEvaluator()

    def setup_api_client(self):
        """Setup Groq client."""
        try:
            self.api_key = os.getenv('GROQ_API_KEY')
            if not self.api_key:
                raise ValueError("GROQ_API_KEY is not set in environment variables")
            
            self.base_url = "https://api.groq.com/openai/v1"
            self.model = os.getenv('MODEL_NAME', 'mixtral-8x7b-32768')
            logger.info("Groq configuration initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Groq configuration: {str(e)}")
            raise

    def get_marketing_prompt(self, context_type="data_analysis"):
        """Get contextual marketing prompt based on type."""
        prompts = {
            "data_analysis": """You are an Executive Marketing Analytics Consultant with over 20 years of experience in data-driven marketing strategy. Your expertise lies in transforming raw data into actionable marketing insights.

ROLE AND EXPERTISE:
- Senior Marketing Analytics Expert
- Data-Driven Decision Making Specialist
- Strategic Marketing Consultant

ANALYSIS APPROACH:
1. Data Overview
- Begin with a clear summary of the key metrics
- Identify patterns and anomalies in the data

2. Detailed Analysis
- Break down each metric's significance
- Compare with industry benchmarks when relevant
- Calculate derived metrics for deeper insights

3. Strategic Insights
- Connect data points to business impact
- Identify opportunities and risks
- Provide context for decision-making

4. Actionable Recommendations
- Prioritize suggestions based on impact
- Include both quick wins and long-term strategies
- Provide specific, measurable actions

FORMAT YOUR RESPONSE AS FOLLOWS:

### Executive Summary
[Provide a brief, high-level overview of key findings]

### Detailed Analysis
[Step-by-step breakdown of the data]
- Metric 1: [Analysis]
- Metric 2: [Analysis]
[Include calculations and reasoning]

### Strategic Insights
[Connect analysis to business impact]
- Finding 1: [Impact + Reasoning]
- Finding 2: [Impact + Reasoning]

### Recommendations
1. [Primary recommendation with specific action steps]
2. [Secondary recommendation with implementation details]
3. [Additional recommendations prioritized by impact]

### Next Steps
[Outline immediate actions and timeline]

Remember to:
- Use clear, professional language
- Support all conclusions with data
- Provide specific, actionable steps
- Include relevant calculations
- Consider both short and long-term implications""",

            "trend_analysis": """You are a Senior Marketing Trend Analyst specializing in revenue and growth patterns. Your expertise is in identifying meaningful trends and providing strategic recommendations for optimization.

ANALYSIS FRAMEWORK:
1. Trend Identification
- Pattern recognition in time-series data
- Growth rate calculations
- Seasonality analysis

2. Comparative Analysis
- Period-over-period changes
- Year-over-year growth
- Market context and external factors

3. Impact Assessment
- Revenue implications
- Market share analysis
- Performance against targets

FORMAT YOUR RESPONSE AS FOLLOWS:

### Trend Summary
[Overview of key trends identified]

### Detailed Analysis
[Break down each trend with supporting data]
- Trend 1: [Analysis with calculations]
- Trend 2: [Analysis with calculations]

### Strategic Implications
[Business impact of identified trends]

### Recommendations
1. [Specific actions to capitalize on trends]
2. [Risk mitigation strategies]
3. [Growth opportunities]

Remember to:
- Show all calculations
- Explain reasoning clearly
- Provide actionable insights
- Consider market context
- Include both risks and opportunities"""
        }
        
        return prompts.get(context_type, prompts["data_analysis"])

    def get_response(self, query, context):
        """Get response from Groq API with enhanced prompting."""
        try:
            if not query.strip():
                return "Please provide a valid query."

            # Determine context type based on query content
            context_type = "trend_analysis" if any(
                keyword in query.lower() 
                for keyword in ['trend', 'growth', 'revenue', 'sales']
            ) else "data_analysis"

            # Get appropriate prompt
            system_prompt = self.get_marketing_prompt(context_type)

            try:
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
                
                data = {
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"Context: {context}\n\nQuery: {query}"}
                    ],
                    "temperature": 0.5,
                    "max_tokens": 1000
                }

                with httpx.Client() as client:
                    response = client.post(
                        f"{self.base_url}/chat/completions",
                        headers=headers,
                        json=data,
                        timeout=30.0
                    )
                
                response_json = response.json()
                
                if response.status_code == 200:
                    generated_response = response_json['choices'][0]['message']['content']
                    
                    # Evaluate response quality
                    evaluation_results = self.evaluator.evaluate_response(
                        generated_response,
                        context  # or some reference text
                    )
                    
                    # Log evaluation metrics
                    logger.info(
                        "Response evaluation metrics:\n%s", 
                        self.evaluator.get_evaluation_summary(evaluation_results)
                    )
                    
                    return generated_response
                else:
                    error_msg = response_json.get('error', {}).get('message', 'Unknown error')
                    logger.error(f"Groq API error: {error_msg}")
                    return f"Error from Groq API: {error_msg}"
            
            except httpx.RequestError as api_error:
                logger.error(f"Groq API call error: {str(api_error)}")
                if "api_key" in str(api_error).lower():
                    return ("⚠️ Groq API key is invalid or not set properly. Please:\n"
                           "1. Check your API key at https://console.groq.com\n"
                           "2. Update your .env file with a valid key\n"
                           "3. Restart the application")
                else:
                    return f"Error calling Groq API: {str(api_error)}"

        except Exception as e:
            error_msg = str(e)
            logger.error(f"General error: {error_msg}")
            return f"Error processing query: {error_msg}"

    def validate_api_key(self):
        """Validate Groq API key configuration."""
        try:
            return bool(self.api_key and self.api_key != 'your_groq_api_key_here')
        except Exception:
            return False 