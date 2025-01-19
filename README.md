### *Chatbot Model - Smart Business Analist* 

This repository contains a chatbot model designed to validate inputs (e.g., email, phone numbers, names, files) and evaluate business/marketing relevance in queries and columns. It is aimed at applications where user inputs and data need to be analyzed for their relevance to business metrics, dimensions, and analysis types.

##Features

#1. **Input Validation**
    - **Email Validation**: Validates that the email follows the correct format.
    - **Phone Number Validation**: Checks if the phone number contains only valid characters and fits within the acceptable length (10-15 digits).
    - **Name Validation**: Ensures that the name contains only letters, spaces, apostrophes, and hyphens.
    - **File Size Validation**: Checks whether the file size is within an acceptable range (default 10MB).
    - **File Type Validation**: Validates the file extension against a list of allowed types.

# 2. **Business Relevance Validation**
    - **Query Relevance**: Analyzes whether a given query contains business/marketing-related terms (metrics, dimensions, analysis types).
    - **Column Relevance**: Evaluates if columns in data (e.g., CSV or Excel files) are relevant to business analytics.
    - **Similarity Scoring**: Uses a simple string similarity metric (Jaccard) to detect partial matches.
    - **Suggestions for Improvement**: Provides helpful suggestions if the query or columns are not sufficiently relevant.

## Requirements

    - Python 3.6+
    - `re` (Regular Expressions)
    - `json`
    - `os`
    - `datetime`
    - `collections`

## Installation

    Clone the repository to your local machine:

    git clone https://github.com/Sumit1205t/marketing-analytics-assistant.git

## USAGE
    Input Validator Example
        code:
            from input_validator import InputValidator

            email = "sumit.tapadia@gmail.com"
            is_valid_email = InputValidator.validate_email(email)
            print(is_valid_email)  # True if valid, False if not

            phone = "+6285695398121"
            is_valid_phone = InputValidator.validate_phone(phone)
            print(is_valid_phone)  # True if valid, False if not

            name = "John Doe"
            is_valid_name = InputValidator.validate_name(name)
            print(is_valid_name)  # True if valid, False if not

            file_size = 5000000  # in bytes
            is_valid_size = InputValidator.validate_file_size(file_size, max_size_mb=5)
            print(is_valid_size)  # True if file size is valid

            file_type = "image.jpg"
            is_valid_type = InputValidator.validate_file_type(file_type, allowed_types=["jpg", "png"])
            print(is_valid_type)  # True if file type is valid

## Business Relevance Validator Example
    from business_relevance_validator import BusinessRelevanceValidator

            query = "What is the conversion rate by campaign?"
            validator = BusinessRelevanceValidator()

            result = validator.validate_query(query)
            print(result['is_relevant'])  # True if relevant, False if not
            print(result['relevance_score'])  # Relevance score (0.0 to 1.0)
            print(result['matches'])  # Matches to marketing terms (metrics, dimensions, analysis types)
            print(result['suggestions'])  # Suggestions for improving the query

            columns = ["Channel", "Revenue", "Region", "Clicks"]
            column_result = validator.validate_columns(columns)
            print(column_result['is_relevant'])  # True if relevant, False if not
            print(column_result['relevance_score'])  # Relevance score

            Log Exporting (Optional)
            from metrics_logger import MetricsLogger

# Initialize logger
            logger = MetricsLogger()

# Log evaluation metrics
            metrics = {
                "conversion_rate": 0.25,
                "impressions": 5000,
                "clicks": 1250,
                "roi": 5.0
            }
            logger.log_metrics(metrics)

# Export metrics to file
            logger.export_metrics("metrics_log.json")
            Customization
    You can easily extend the functionality of the input validation and business relevance checks by:

Adding new validation methods to InputValidator.
    Expanding the marketing_keywords dictionary in BusinessRelevanceValidator for more comprehensive checks.
    Enhancing the similarity scoring algorithm for more precise relevance detection (e.g., using NLP models).

Contributing

We welcome contributions to improve and extend the functionality of the chatbot model. If you'd like to contribute, please fork the repository, create a new branch, and submit a pull request with your changes.

License
This project is licensed under the MIT License - see the LICENSE file for details.

Other Relevant document:
        1. Application Architecture
        2. Setup Instructions
        3. Usage Guidelines
        4. Results & Evaluations