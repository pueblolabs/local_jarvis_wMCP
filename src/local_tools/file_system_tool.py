import os
from agents import function_tool
from . import register_tool

# Sandboxed working directory within the user's home directory
SANDBOX_DIR = os.path.expanduser("~/local_jarvis_sandbox")

def _ensure_sandbox():
    """Ensures the sandbox directory exists."""
    if not os.path.exists(SANDBOX_DIR):
        os.makedirs(SANDBOX_DIR)

def _is_path_in_sandbox(path: str) -> bool:
    """Checks if a given path is safely within the sandbox."""
    _ensure_sandbox()
    # Resolve real paths to prevent '..' traversal attacks
    sandbox_real = os.path.realpath(SANDBOX_DIR)
    path_real = os.path.realpath(path)
    return os.path.commonpath([sandbox_real, path_real]) == sandbox_real

@register_tool
@function_tool
def create_file(relative_path: str, content: str, overwrite: bool = False) -> str:
    """
    Creates a file within a sandboxed directory ('~/local_jarvis_sandbox').
    Fails if the file already exists unless `overwrite` is set to True.
    
    Args:
        relative_path: The path of the file relative to the sandbox directory. e.g., 'my_document.txt' or 'reports/report.csv'.
        content: The text content to write into the file.
        overwrite: If True, will overwrite the file if it exists. Defaults to False.
    """
    _ensure_sandbox()
    full_path = os.path.join(SANDBOX_DIR, relative_path)
    
    if not _is_path_in_sandbox(full_path):
        return f"Error: Path ''{relative_path}'' is outside the allowed sandbox directory (''{SANDBOX_DIR}''."

    if os.path.exists(full_path) and not overwrite:
        return f"Error: File ''{full_path}'' already exists. Set overwrite=True to replace it."
        
    try:
        directory = os.path.dirname(full_path)
        if not os.path.exists(directory):
            os.makedirs(directory)
            
        with open(full_path, 'w', encoding='utf-8') as f:
            f.write(content)
            
        return f"Successfully created file at {full_path}"
    except Exception as e:
        return f"Error creating file: {e}"
