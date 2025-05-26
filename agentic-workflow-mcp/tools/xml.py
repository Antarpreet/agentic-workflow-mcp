from langchain.tools import tool

from tools.file_system import resolve_path
from tools.shell_command import run_shell_command

@tool
def validate_xml(xml_file_path: str, xsd_file_path: str, workspace_path: str = None) -> str:
    """
    Validates an XML file against an XSD schema using xmllint.

    Args:
        xml_file_path (str): The path to the XML file to validate.
        xsd_file_path (str): The path to the XSD schema file.
        workspace_path (str, optional): The base directory to prepend to file paths if they are relative.

    Returns:
        str: A message indicating whether the validation was successful or if there were errors.
    """
    xml_full_path = resolve_path(xml_file_path, workspace_path)
    if not xml_full_path:
        return f"Error: XML file '{xml_file_path}' does not exist."

    xsd_full_path = resolve_path(xsd_file_path, workspace_path)
    if not xsd_full_path:
        return f"Error: XSD file '{xsd_file_path}' does not exist."

    try:
        # Run the xmllint command to validate the XML against the XSD
        result = run_shell_command(
            ["xmllint", "--noout", "--schema", xsd_full_path, xml_full_path]
        )
        return f"Validation successful: {result.stdout.strip()}"
    except Exception as e:
        return f"Validation failed:\n{e.stderr.strip()}"
    except FileNotFoundError:
        return "Error: 'xmllint' is not installed or not found in PATH."
