import json
import os
from typing_extensions import Optional, TypedDict

from core.model import DEFAULT_WORKFLOW_CONFIG, WorkflowConfig

# Define a TypedDict for the graph's state
# Utility to generate a TypedDict from a JSON schema
def typed_dict_from_json_schema(name: str, schema: dict) -> type:
    """
    Dynamically creates a TypedDict class from a JSON schema.

    Args:
        name (str): The name of the TypedDict class.
        schema (dict): The JSON schema dictionary.

    Returns:
        type: The generated TypedDict class.
    """

    type_map = {
        "string": str,
        "integer": int,
        "number": float,
        "boolean": bool,
        "object": dict,
        "array": list,
        # Add more mappings as needed
    }

    properties = schema.get("properties", {})
    required = set(schema.get("required", []))
    annotations = {}

    for prop, prop_schema in properties.items():
        json_type = prop_schema.get("type", "string")
        py_type = type_map.get(json_type, str)
        if prop not in required:
            py_type = Optional[py_type]
        annotations[prop] = py_type

    return TypedDict(name, annotations, total=False)


def load_workflow_config(config_path: str) -> WorkflowConfig:
    """
    Loads the workflow configuration from a JSON file.

    Args:
        config_path (str): Path to the workflow configuration file.
    
    Returns:
        dict: Parsed workflow configuration.
    """
    # Load the workflow config file path from the environment variable
    config_path = os.getenv("WORKFLOW_CONFIG_PATH")
    if not config_path:
        raise ValueError("WORKFLOW_CONFIG_PATH environment variable is not set.")

    workflow_config: WorkflowConfig = None
    # Load the workflow config file
    try:
        with open(config_path, "r") as file:
            workflow_config = json.load(file)
    except FileNotFoundError:
        raise FileNotFoundError(f"Error: Workflow config file not found at {config_path}.")
    except json.JSONDecodeError:
        raise ValueError(f"Error: Decoding JSON from the workflow config file at {config_path}.")
    except Exception as e:
        raise Exception(f"Error: Unexpected error loading workflow config: {str(e)}")

    return workflow_config


def get_full_schema(workflow_config: WorkflowConfig) -> dict:
    """
    Generates the full schema for the workflow configuration.

    Args:
        workflow_config (WorkflowConfig): The workflow configuration dictionary.

    Returns:
        dict: The full schema for the workflow configuration.
    """
    # Merge schema from workflow_config with the default schema if present
    schema = DEFAULT_WORKFLOW_CONFIG["state_schema"].copy()
    user_schema = workflow_config.get("state_schema")
    if user_schema:
        # Merge user_schema into schema (shallow merge for 'properties' and 'required')
        schema["properties"].update(user_schema.get("properties", {}))
        if "required" in user_schema:
            # Combine required fields, avoiding duplicates
            schema["required"] = list(set(schema.get("required", []) + user_schema["required"]))
        # Merge other top-level keys if present
        for k, v in user_schema.items():
            if k not in ("properties", "required"):
                schema[k] = v

    # Dynamically add agent_name_output properties for all agents in the config
    agents = workflow_config.get("agents", [])
    for agent in agents:
        agent_name = agent.get("name")
        if agent_name:
            output_key = f"{agent_name}_output"
            # Only add if not already present
            if output_key not in schema["properties"]:
                schema["properties"][output_key] = {"type": "string"}
    return schema


def ensure_utf8(val) -> str:
    """
    Ensures that the input value is UTF-8 encoded.

    Args:
        val: The input value to be encoded.
    
    Returns:
        str: The UTF-8 encoded string.
    """
    if isinstance(val, bytes):
        return val.decode("utf-8", errors="replace")
    elif isinstance(val, str):
        return val.encode("utf-8", errors="replace").decode("utf-8", errors="replace")
    else:
        return str(val)
