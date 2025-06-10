import os

from langchain.tools import tool

def find_file_recursive(filename: str, workspace_path: str) -> str:
    """
    Recursively searches for a file in the workspace_path.
    Returns the full path if found, else raises FileNotFoundError.

    Args:
        filename (str): The name of the file to search for.
        workspace_path (str): The base directory to start the search.

    Returns:
        str: The full path to the file if found.
    """
    for root, _, files in os.walk(workspace_path):
        if filename in files:
            return os.path.join(root, filename)
    raise FileNotFoundError(f"{filename} not found in {workspace_path}")


def resolve_path(file_path: str, workspace_path: str = None) -> str:
    """
    Resolves the full path for a file, searching recursively if file_path is relative.

    Args:
        file_path (str): The path to the file.
        workspace_path (str, optional): The base directory to prepend to file_path.

    Returns:
        str: The full path to the file.
    """
    # Remove '/path/to/' from the file_path if present
    if file_path.startswith("/path/to/"):
        file_path = file_path[len("/path/to/"):]
    if workspace_path is None or os.path.isabs(file_path):
        return file_path
    full_path = os.path.join(workspace_path, file_path)
    if os.path.exists(full_path):
        return full_path
    # If not found, search recursively
    return find_file_recursive(os.path.basename(file_path), workspace_path)


@tool
def read_file(file_path: str, workspace_path: str = None) -> str:
    """
    Reads the content of a file and returns it as a string.

    Args:
        file_path (str): The path to the file to be read.
        workspace_path (str, optional): The base directory to prepend to file_path.

    Returns:
        str: The content of the file.
    """
    full_path = resolve_path(file_path, workspace_path)
    with open(full_path, 'r', encoding='utf-8') as file:
        return file.read()


@tool
def read_multiple_files(file_paths: list, workspace_path: str = None) -> str:
    """
    Reads the content of multiple files and returns them as a single string.

    Args:
        file_paths (list): A list of paths to the files to be read.
        workspace_path (str, optional): The base directory to prepend to each file_path.

    Returns:
        str: The combined content of the files.
    """
    contents = []
    for file_path in file_paths:
        full_path = resolve_path(file_path, workspace_path)
        with open(full_path, 'r', encoding='utf-8') as file:
            contents.append(file.read())
    return "\n".join(contents)

@tool
def read_multiple_files_with_id(file_paths: list, workspace_path: str = None) -> str:
    """
    Reads the content of multiple files and returns them as a JSON string mapping file IDs to contents.

    Args:
        file_paths (list): A list of paths to the files to be read.
        workspace_path (str, optional): The base directory to prepend to each file_path.

    Returns:
        str: A JSON string mapping file IDs to their contents.
    """
    import json
    contents = {}
    for file_path in file_paths:
        full_path = resolve_path(file_path, workspace_path)
        with open(full_path, 'r', encoding='utf-8') as file:
            contents[file_path] = file.read()
    return json.dumps(contents, ensure_ascii=False, indent=2)

@tool
def list_files(directory: str, workspace_path: str = None) -> list:
    """
    Lists all files in a given directory.

    Args:
        directory (str): The path to the directory.
        workspace_path (str, optional): The base directory to prepend to directory.

    Returns:
        list: A list of file names in the directory.
    """
    full_dir = resolve_path(directory, workspace_path)
    return [f for f in os.listdir(full_dir) if os.path.isfile(os.path.join(full_dir, f))]


@tool
def write_file(file_path: str, content: str, workspace_path: str = None) -> str:
    """
    Writes the given content to a file. Creates directories if they don't exist.

    Args:
        file_path (str): The path to the file to be written.
        content (str): The content to write to the file.
        workspace_path (str, optional): The base directory to prepend to file_path.

    Returns:
        str: A message indicating the file has been written.
    """
    # For writing, we don't want to search recursively, just resolve the intended path
    full_path = os.path.join(workspace_path, file_path) if workspace_path and not os.path.isabs(file_path) else file_path
    directory = os.path.dirname(full_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
    with open(full_path, 'w', encoding='utf-8') as file:
        file.write(content)
    return f"Content written to {full_path}"


@tool
def write_file_lines(file_path: str, lines: dict, workspace_path: str = None) -> str:
    """
    Write lines content at the specified line numbers to a file. Creates directories if they don't exist.

    Args:
        file_path (str): The path to the file to be written.
        lines (dict): A dictionary where keys are line numbers and values are the content to write at those lines.
        workspace_path (str, optional): The base directory to prepend to file_path.
    
    Returns:
        str: A message indicating the lines have been written.
    """
    # For writing, we don't want to search recursively, just resolve the intended path
    full_path = os.path.join(workspace_path, file_path) if workspace_path and not os.path.isabs(file_path) else file_path
    directory = os.path.dirname(full_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

    # Read existing lines if the file exists
    existing_lines = []
    if os.path.exists(full_path):
        with open(full_path, 'r', encoding='utf-8') as file:
            existing_lines = file.readlines()

    # Write the specified lines at the given line numbers
    for line_number, content in lines.items():
        while len(existing_lines) <= line_number:
            existing_lines.append("\n")
        existing_lines[line_number] = content + "\n"


@tool
def append_file(file_path: str, content: str, workspace_path: str = None) -> str:
    """
    Appends the given content to a file. Creates directories if they don't exist.

    Args:
        file_path (str): The path to the file to be appended.
        content (str): The content to append to the file.
        workspace_path (str, optional): The base directory to prepend to file_path.

    Returns:
        str: A message indicating the content has been appended.
    """
    full_path = os.path.join(workspace_path, file_path) if workspace_path and not os.path.isabs(file_path) else file_path
    directory = os.path.dirname(full_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
    with open(full_path, 'a', encoding='utf-8') as file:
        file.write(content)
    return f"Content appended to {full_path}"


@tool
def append_file_lines(file_path: str, lines: dict, workspace_path: str = None) -> str:
    """
    Appends lines content at the specified line numbers to a file. Creates directories if they don't exist.

    Args:
        file_path (str): The path to the file to be appended.
        lines (dict): A dictionary where keys are line numbers and values are the content to append at those lines.
        workspace_path (str, optional): The base directory to prepend to file_path.
    
    Returns:
        str: A message indicating the lines have been appended.
    """
    # For appending, we don't want to search recursively, just resolve the intended path
    full_path = os.path.join(workspace_path, file_path) if workspace_path and not os.path.isabs(file_path) else file_path
    directory = os.path.dirname(full_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

    # Read existing lines if the file exists
    existing_lines = []
    if os.path.exists(full_path):
        with open(full_path, 'r', encoding='utf-8') as file:
            existing_lines = file.readlines()

    # Append the specified lines at the given line numbers
    for line_number, content in lines.items():
        while len(existing_lines) <= line_number:
            existing_lines.append("\n")
        existing_lines[line_number] += content + "\n"

    with open(full_path, 'w', encoding='utf-8') as file:
        file.writelines(existing_lines)

    return f"Lines appended to {full_path}"
