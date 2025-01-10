USER_PROMPTS = {
    'document_upload': {
        'success': "Your document has been successfully processed. What would you like to know about it?",
        'failure': "I couldn't process this document. Please ensure it's in a supported format (PDF, CSV, or Excel).",
        'partial': "I've processed part of your document. Some sections may need attention."
    },
    
    'query_suggestions': [
        "What were the top performing campaigns?",
        "Can you summarize the ROI for all campaigns?",
        "What are the key trends in customer engagement?",
        "How does this performance compare to previous periods?",
        "What are the main areas for improvement?"
    ],
    
    'clarification_requests': [
        "Could you specify the time period you're interested in?",
        "Would you like to focus on specific metrics?",
        "Should I include all campaigns or focus on specific ones?",
        "Would you prefer a detailed analysis or a high-level overview?"
    ]
}