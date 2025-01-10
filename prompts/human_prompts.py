HUMAN_PROMPTS = {
    'welcome': {
        'title': "Welcome to the Marketing Assistant!",
        'subtitle': "I'm here to help you analyze your marketing data and provide valuable insights."
    },
    
    'file_upload': {
        'instructions': (
            "Please upload your marketing documents. I can process:\n"
            "- PDF reports\n"
            "- Excel spreadsheets\n"
            "- CSV data files"
        ),
        'tips': "For best results, ensure your documents are properly formatted and contain clear data."
    },
    
    'query_help': {
        'examples': (
            "You can ask questions like:\n"
            '- "What was our best performing channel?"\n'
            '- "Show me the overall revenue trends of last 7 days"\n'
            '- "Summarize the campaign results"\n'
            '- "Compare performance across regions"'
        ),
        'guidance': "Be specific in your questions for more accurate answers."
    },
    
    'error_messages': {
        'invalid_file': "Sorry, this file type is not supported.",
        'processing_error': "An error occurred while processing your request.",
        'empty_query': "Please enter a query to continue."
    },
    
    'success_messages': {
        'file_upload': "Your files have been successfully uploaded!",
        'processing_complete': "Document processing complete. You can now ask questions.",
        'query_processed': "Analysis complete! Here are the insights:"
    }
} 