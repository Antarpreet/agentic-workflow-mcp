from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import List, Union

from mcp.server.fastmcp import FastMCP

from core.internal import initialize, display, process, embed, embed_visualize
from core.log import log_message
from core.model import AppContext, WorkflowConfig

@asynccontextmanager
async def app_lifespan(server: FastMCP) -> AsyncIterator[AppContext]:
    """
    Lifespan context manager for the FastMCP server.
    Initializes and yields application resources.

    Args:
        server (FastMCP): The FastMCP server instance.

    Returns:
        AsyncIterator[AppContext]: Yields the application context with initialized resources.
    """
    try:
        context = await initialize(server=server)
        # Yield the context with all initialized resources
        yield context
    except Exception as e:
        log_message([], f"Error: initializing app context: {str(e)} {repr(e)}")
        raise
    # Teardown code here (if needed, e.g., explicitly stopping services)
    # chroma_client.persist() # Chroma client might persist automatically depending on config


mcp = FastMCP("Local Agentic Workflow MCP Server", lifespan=app_lifespan)


@mcp.resource("resource://agentic-workflow-mcp/agents")
def get_agents() -> list:
    """
    Returns the list of agents defined in the workflow configuration.

    Args:
        None

    Returns:
        list: List of agent dictionaries with name and description.
    """
    ctx = mcp.get_context()
    workflow_config: WorkflowConfig = ctx.workflow_config
    agents = workflow_config.get("agents", [])
    return [{"name": agent["name"], "description": agent.get("description", "")} for agent in agents]


@mcp.tool()
async def display_graph(type: str = "image") -> str:
    """
    Generates a graph image from the workflow configuration and saves it to a file.
    The graph is generated using the LangGraph library and saved as a PNG image.

    Args:
        type (str): The type of graph to generate. Possible values are "image" or "mermaid".

    Returns:
        str: Path to the generated graph image file.
    """
    ctx = mcp.get_context()
    app_ctx = ctx.request_context.lifespan_context

    return await display(app_ctx, type=type)


@mcp.tool()
async def start_workflow(user_prompt: str) -> dict:
    """
    Starts a LangGraph workflow to process the user prompt.
    The workflow is defined in a config file existing in the workspace.
    The relative path to the config file is defined in the environment variable WORKFLOW_CONFIG_PATH in .env file in the workspace.

    Args:
        user_prompt (str): The user prompt to process.

    Returns:
        dict: The response from the workflow execution.
    """
    ctx = mcp.get_context()
    app_ctx = ctx.request_context.lifespan_context

    response = await process(app_ctx, user_prompt)
    return response["final_output"] if isinstance(response, dict) and "final_output" in response else str(response)


@mcp.tool()
async def embed_files(file_paths: List[str]) -> dict:
    """
    Tool to create embeddings for one or more files or folders and store them in the ChromaDB.

    Args:
        file_paths: List of file/folder paths to embed

    Returns:
        dict: Information about the embedding operation
    """
    ctx = mcp.get_context()
    app_ctx = ctx.request_context.lifespan_context
    
    return await embed(app_ctx, file_paths)


@mcp.tool()
async def visualize_embeddings(collection_name: str = None) -> str:
    """
    Visualize embeddings from a ChromaDB collection using PCA and Plotly.

    Args:
        collection_name (str, optional): Name of the ChromaDB collection to visualize. If not provided, uses the default from workflow config.

    Returns:
        str: Path to the generated visualization image file.
    """
    ctx = mcp.get_context()
    app_ctx = ctx.request_context.lifespan_context

    return await embed_visualize(app_ctx, collection_name)
