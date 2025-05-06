import os
from datetime import datetime

from core.model import DEFAULT_WORKFLOW_CONFIG

def log_message(logs, message):
    """
    Logs a message with a timestamp to the logs list and writes it to a log file.

    Args:
        logs (list): List to store log messages.
        message (str): The message to log.
    
    Returns:
        None
    """
    timestamped_message = f"{datetime.now().isoformat()} - {message}"
    logs.append(timestamped_message)
    # Get the log file path from the workflow configuration
    log_file = DEFAULT_WORKFLOW_CONFIG["log_file_path"]
    # Write the message to the log file
    with open(log_file, "a") as file:
        file.write(timestamped_message + "\n")
