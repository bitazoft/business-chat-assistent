import re
from typing import Optional

# File validation constants
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
ALLOWED_FILE_TYPES = {
    'image/jpeg': ['.jpg', '.jpeg'],
    'image/png': ['.png'],
    'image/gif': ['.gif'],
    'image/webp': ['.webp'],
    'application/pdf': ['.pdf']
}

def validate_file(file_type: str, file_size: int) -> bool:
    """
    Validate file type and size
    
    Args:
        file_type (str): MIME type of the file
        file_size (int): Size of the file in bytes
        
    Returns:
        bool: True if valid, raises exception if invalid
        
    Raises:
        ValueError: If file type or size is invalid
    """
    # Check file size
    if file_size > MAX_FILE_SIZE:
        raise ValueError(f"File size {file_size} bytes exceeds maximum allowed size of {MAX_FILE_SIZE} bytes")
    
    # Check file type
    if file_type not in ALLOWED_FILE_TYPES:
        allowed_types = list(ALLOWED_FILE_TYPES.keys())
        raise ValueError(f"File type '{file_type}' not allowed. Allowed types: {allowed_types}")
    
    return True

def get_file_extension(file_name: str) -> str:
    """Get file extension from filename"""
    return '.' + file_name.split('.')[-1].lower() if '.' in file_name else ''

def validate_file_extension(file_name: str, file_type: str) -> bool:
    """Validate that file extension matches the MIME type"""
    extension = get_file_extension(file_name)
    allowed_extensions = ALLOWED_FILE_TYPES.get(file_type, [])
    
    if extension not in allowed_extensions:
        raise ValueError(f"File extension '{extension}' doesn't match MIME type '{file_type}'")
    
    return True
