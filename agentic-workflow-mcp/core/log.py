from datetime import datetime

def log_message(logs, message, log_file="agentic-workflow-mcp/logs.txt"):
    """
    Logs a message with a timestamp to the logs list and writes it to a log file.

    Args:
        logs (list): List to store log messages.
        message (str): The message to log.
        log_file (str): Path to the log file.
    
    Returns:
        None
    """
    timestamped_message = f"{datetime.now().isoformat()} - {message}"
    logs.append(timestamped_message)
    with open(log_file, "a") as file:
        file.write(timestamped_message + "\n")
