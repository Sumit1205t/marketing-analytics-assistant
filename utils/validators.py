import re
from typing import Optional

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