import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import List, Union

from chromadb import PersistentClient
from chromadb.config import Settings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama, OllamaEmbeddings
from mcp.server.fastmcp import FastMCP

from core.embedding import update_embeddings
from core.graph import generate_graph_from_workflow
from core.log import log_message
from core.model import DEFAULT_WORKFLOW_CONFIG, AppContext, WorkflowConfig
from core.util import load_workflow_config

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
        workspace_path = os.getenv('WORKSPACE_PATH')
        # Load the workflow configuration
        workflow_config: WorkflowConfig = load_workflow_config(os.getenv("WORKFLOW_CONFIG_PATH"))

        embedding_model_name = workflow_config.get("embedding_model", DEFAULT_WORKFLOW_CONFIG["embedding_model"])
        default_model_name = workflow_config.get("default_model", DEFAULT_WORKFLOW_CONFIG["default_model"])
        default_temperature = workflow_config.get("default_temperature", DEFAULT_WORKFLOW_CONFIG["default_temperature"])
        collection_name = workflow_config.get("collection_name", DEFAULT_WORKFLOW_CONFIG["collection_name"])

        log_message([], f"embedding_model_name: {embedding_model_name}")
        log_message([], f"default_model_name: {default_model_name}")
        log_message([], f"default_temperature: {default_temperature}")
        log_message([], f"collection_name: {collection_name}")
        log_message([], f"workspace_path: {workspace_path}")

        # Initialize Ollama embeddings
        embedding_model = OllamaEmbeddings(model=embedding_model_name)

        # Initialize ChromaDB client
        persist_directory = "chroma_vector_db"
        chroma_client = PersistentClient(
            path=os.path.join(workspace_path, persist_directory),
            settings=Settings(
                persist_directory=persist_directory,
                anonymized_telemetry=False
            )
        )

        # Initialize Chroma as a LangChain vectorstore using Ollama embeddings
        vectorstore = Chroma(
            client=chroma_client,
            collection_name=collection_name,
            embedding_function=embedding_model,
            persist_directory=persist_directory
        )

        # Initialize Ollama LLM
        llm = ChatOllama(model=default_model_name, temperature=default_temperature)

        # Set up a Retriever
        retriever = vectorstore.as_retriever()

        # Set up a Retrieval chain for RAG
        prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context:
            <context>
            {context}
            </context>
            Question: {input}""")
        document_chain = create_stuff_documents_chain(llm, prompt)
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        # Yield the context with all initialized resources
        yield AppContext(
            server=server,
            embedding_model=embedding_model,
            chroma_client=chroma_client,
            vectorstore=vectorstore,
            llm=llm,
            retriever=retriever,
            retrieval_chain=retrieval_chain,
            workflow_config=workflow_config
        )
    except Exception as e:
        log_message([], f"Error initializing app context: {str(e)} {repr(e)}")
        raise
    # Teardown code here (if needed, e.g., explicitly stopping services)
    # chroma_client.persist() # Chroma client might persist automatically depending on config


mcp = FastMCP("Local Agentic Workflow MCP Server", lifespan=app_lifespan)


@mcp.resource("resource://agentic-workflow-mcp/agents")
def get_agents() -> list:
    """
    Returns the list of agents defined in the workflow configuration.

    Args:
        ctx (Context): The MCP context containing application resources.

    Returns:
        list: List of agent dictionaries with name and description.
    """
    ctx = mcp.get_context()
    workflow_config: WorkflowConfig = ctx.request_context.lifespan_context.workflow_config
    agents = workflow_config.get("agents", [])
    return [{"name": agent["name"], "description": agent.get("description", "")} for agent in agents]


@mcp.tool()
def display_graph() -> str:
    """
    Generates a graph image from the workflow configuration and saves it to a file.
    The graph is generated using the LangGraph library and saved as a PNG image.

    Args:
        ctx (Context): The MCP context containing application resources.

    Returns:
        str: Path to the generated graph image file.
    """
    logs = []
    output_path = None
    ctx = mcp.get_context()
    workflow_config: WorkflowConfig = ctx.request_context.lifespan_context.workflow_config
    
    try:
        # Generate the graph from the workflow configuration
        graph = generate_graph_from_workflow(workflow_config, logs, ctx)
    except Exception as e:
        log_message(logs, f"Error generating graph: {str(e)} {repr(e)}")
        return

    try:
        # Generate the PNG image bytes
        image_bytes = graph.get_graph().draw_mermaid_png()
        # Define the output file path
        output_path = os.getenv("WORKSPACE_PATH", ".") + "/graph.png"
        # Write the image bytes to a local file
        with open(output_path, "wb") as f:
            f.write(image_bytes)
        log_message(logs, f"Graph image successfully saved to {output_path}")
        
    except Exception as e:
        log_message(logs, f"Error generating or saving graph image: {str(e)}")

    return output_path


@mcp.tool()
def start_workflow(user_prompt: str) -> dict:
    """
    Starts a LangGraph workflow to process the user prompt.
    The workflow is defined in a config file existing in the workspace.
    The relative path to the config file is defined in the environment variable WORKFLOW_CONFIG_PATH in .env file in the workspace.

    Args:
        user_prompt (str): The user prompt to process.
        ctx (Context): The MCP context containing application resources.

    Returns:
        dict: The response from the workflow execution.
    """
    logs = []
    ctx = mcp.get_context()
    log_message(logs, f"Starting workflow with user prompt: {user_prompt}")
    workflow_config: WorkflowConfig = ctx.request_context.lifespan_context.workflow_config

    try:
        # Generate the graph from the workflow configuration
        graph = generate_graph_from_workflow(workflow_config, logs, ctx)

        # Execute the graph with the user prompt
        response = graph.invoke({"input": user_prompt})
    except Exception as e:
        log_message(logs, f"Error running workflow: {str(e)} {repr(e)}")
        response = {"logs": logs}

    return response


@mcp.tool()
def embed_files(file_paths: Union[str, List[str]]) -> dict:
    """
    Tool to create embeddings for one or more files and store them in the ChromaDB.
    
    Args:
        file_paths: Path to a file or list of file paths to embed
        ctx: The MCP context containing application resources

    Returns:
        dict: Information about the embedding operation
    """
    ctx = mcp.get_context()
    log_message([], f"Embedding files: {file_paths}")
    workflow_config: WorkflowConfig = ctx.request_context.lifespan_context.workflow_config
    collection_name = workflow_config.get("collection_name", DEFAULT_WORKFLOW_CONFIG["collection_name"])
    delete_missing_embeddings = workflow_config.get("delete_missing_embeddings", DEFAULT_WORKFLOW_CONFIG["delete_missing_embeddings"])
    # Call the update_embeddings function to create or update embeddings
    return update_embeddings(
        file_paths=file_paths, 
        ctx=ctx, 
        collection_name=collection_name, 
        delete_missing=delete_missing_embeddings
    )
