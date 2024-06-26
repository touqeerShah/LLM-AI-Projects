import re

def sanitize_name(name):
    """
    Sanitize the provided name by replacing all special characters and whitespace
    with underscores, while also ensuring no leading or trailing underscores.

    Args:
        name (str): The string to be sanitized.

    Returns:
        str: The sanitized string, safe for use as an identifier.
    """
    # Replace all non-word characters (including whitespace) with underscores
    sanitized = re.sub(r'\W+', '_', name)
    
    # Remove leading/trailing underscores that may have been added
    sanitized = sanitized.strip('_')
    
    return sanitized

# Example usage:
column_name = "Identita's Data Processing: Purpose & Rights"
sanitized_name = sanitize_name(column_name)
print(sanitized_name)