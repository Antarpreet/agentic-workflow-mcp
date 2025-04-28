import os

from langchain.tools import tool

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
    full_path = file_path if workspace_path is None else os.path.join(workspace_path, file_path)
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
        full_path = file_path if workspace_path is None else os.path.join(workspace_path, file_path)
        with open(full_path, 'r', encoding='utf-8') as file:
            contents.append(file.read())
    return "\n".join(contents)


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
    full_dir = directory if workspace_path is None else os.path.join(workspace_path, directory)
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
    full_path = file_path if workspace_path is None else os.path.join(workspace_path, file_path)
    directory = os.path.dirname(full_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
    with open(full_path, 'w', encoding='utf-8') as file:
        file.write(content)
    return f"Content written to {full_path}"
