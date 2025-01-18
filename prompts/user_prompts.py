USER_PROMPTS = {
    'document_upload': {
        'success': "Your document has been successfully processed. How would you like to proceed with the analysis? Please specify any questions or areas of interest.",
        'failure': "I couldn't process this document. Please ensure it's in a supported format (PDF, CSV, or Excel). If the issue persists, try re-uploading the file.",
        'partial': "I've processed part of your document, but some sections may need further attention. Would you like to review or refine those parts?"
    },
    
    'query_suggestions': [
        "What were the top-performing campaigns during the last quarter? Please provide specifics about the metrics you're interested in (e.g., revenue, ROI, conversions).",
        "Can you summarize the ROI across all campaigns for the specified time period? Feel free to provide any preferences on the types of campaigns to include.",
        "What are the key trends in customer engagement over the last 3 months? Would you like to focus on any specific segments or channels?",
        "How does the current campaign performance compare to previous periods? Should I include comparisons for specific campaigns or metrics?",
        "What are the main areas for improvement based on the data? If you'd like, I can focus on particular performance metrics to identify gaps."
    ],
    
    'clarification_requests': [
        "Could you specify the exact time period you're interested in? For example, last month, last quarter, or a custom date range?",
        "Would you like to focus on specific metrics or a general overview? Please provide details on which metrics are most important to you (e.g., revenue, clicks, ROI).",
        "Should I include all campaigns in the analysis, or would you prefer to focus on specific campaigns or segments?",
        "Would you prefer a detailed analysis with all insights or a high-level summary with key takeaways and recommendations?"
    ]
}
