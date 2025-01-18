HUMAN_PROMPTS = {
    'welcome': {
        'title': "Welcome to the Marketing Assistant!",
        'subtitle': "I'm here to help you analyze your marketing data and deliver valuable insights to drive your business forward. How can I assist you today?"
    },
    
    'file_upload': {
        'instructions': (
            "To get started, please upload your marketing documents. I can process the following formats:\n"
            "- **PDF reports** (e.g., monthly performance reports)\n"
            "- **Excel spreadsheets** (e.g., campaign data or revenue logs)\n"
            "- **CSV data files** (e.g., click-through or conversion data)"
        ),
        'tips': (
            "Tip: For the best results, ensure your files are well-structured, clearly labeled, and contain data relevant to marketing performance."
            " This will help me provide more accurate and actionable insights."
        )
    },
    
    'query_help': {
        'examples': (
            "You can ask questions like:\n"
            '- "What was the best performing marketing channel last quarter?"\n'
            '- "Show me the revenue trends for the last 7 days."\n'
            '- "Summarize the results of the latest campaign."\n'
            '- "Compare performance across regions for this year."'
        ),
        'guidance': (
            "To get the most accurate answers, be as specific as possible with your queries."
            " For example, include details like time periods, metrics (e.g., revenue, conversions), or campaign names."
        )
    },
    
    'error_messages': {
        'invalid_file': "Oops! This file type is not supported. Please upload a **PDF**, **Excel**, or **CSV** file.",
        'processing_error': "An error occurred while processing your request. Please try again or contact support if the issue persists.",
        'empty_query': "It looks like you haven't entered a query yet. Please type a question to continue."
    },
    
    'success_messages': {
        'file_upload': "Your files have been successfully uploaded! Ready to get started? You can ask any marketing-related questions now.",
        'processing_complete': "Document processing is complete! You can now ask questions based on your data, and I'll provide insights.",
        'query_processed': "Analysis complete! Here are the insights based on your query:"
    }
}
