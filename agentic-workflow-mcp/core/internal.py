import os
import time
from typing import List, Optional

from chromadb import PersistentClient
from chromadb.config import Settings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama, OllamaEmbeddings
from mcp.server.fastmcp import FastMCP

from core.embedding import update_embeddings, visualize
from core.graph import generate_graph_from_workflow
from core.log import log_message
from core.model import DEFAULT_WORKFLOW_CONFIG, AppContext, WorkflowConfig
from core.util import load_workflow_config

async def initialize(incoming_workflow_config: Optional[WorkflowConfig] = None, server: FastMCP = None) -> AppContext:
    """
    Initializes the application context with necessary resources.

    Args:
        None

    Returns:
        AppContext: An object containing initialized resources such as embedding model, vector store, and LLM.
    """
    workspace_path = os.getenv('WORKSPACE_PATH')
    # Load the workflow configuration
    workflow_config: WorkflowConfig = incoming_workflow_config or load_workflow_config(os.getenv("WORKFLOW_CONFIG_PATH"))

    embedding_model_name = workflow_config.get("embedding_model", DEFAULT_WORKFLOW_CONFIG["embedding_model"])
    default_model_name = workflow_config.get("default_model", DEFAULT_WORKFLOW_CONFIG["default_model"])
    default_temperature = workflow_config.get("default_temperature", DEFAULT_WORKFLOW_CONFIG["default_temperature"])
    collection_name = workflow_config.get("collection_name", DEFAULT_WORKFLOW_CONFIG["collection_name"])
    rag_prompt_template = workflow_config.get("rag_prompt_template", DEFAULT_WORKFLOW_CONFIG["rag_prompt_template"])
    persist_directory = workflow_config.get("vector_directory", DEFAULT_WORKFLOW_CONFIG["vector_directory"])

    log_message([], f"embedding_model_name: {embedding_model_name}")
    log_message([], f"default_model_name: {default_model_name}")
    log_message([], f"default_temperature: {default_temperature}")
    log_message([], f"collection_name: {collection_name}")
    log_message([], f"workspace_path: {workspace_path}")
    log_message([], f"persist_directory: {persist_directory}")

    # Initialize Ollama embeddings
    embedding_model = OllamaEmbeddings(model=embedding_model_name)

    # Initialize ChromaDB client
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

    # Set up a Retriever
    retriever = vectorstore.as_retriever()

    # Initialize Ollama LLM
    llm = ChatOllama(model=default_model_name, temperature=default_temperature)

    # Set up a Retrieval chain for RAG
    prompt = ChatPromptTemplate.from_template(rag_prompt_template)
    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    return AppContext(
        embedding_model=embedding_model,
        chroma_client=chroma_client,
        vectorstore=vectorstore,
        llm=llm,
        retriever=retriever,
        retrieval_chain=retrieval_chain,
        workflow_config=workflow_config,
        server=server
    )


async def display(ctx: AppContext) -> str:
    """
    Displays the workflow configuration.

    Args:
        ctx (AppContext): The application context containing the workflow configuration.

    Returns:
        str: The workflow configuration as a string.
    """
    logs = []
    output_path = None
    workflow_config: WorkflowConfig = ctx.workflow_config
    
    try:
        # Generate the graph from the workflow configuration
        graph = generate_graph_from_workflow(workflow_config, logs, ctx)
    except Exception as e:
        log_message(logs, f"Error: generating graph: {str(e)} {repr(e)}")
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
        log_message(logs, f"Error: generating or saving graph image: {str(e)}")

    return output_path


async def process(ctx: AppContext, user_prompt: str) -> dict:
    """
    Processes the user prompt using the workflow configuration.

    Args:
        ctx (AppContext): The application context containing the workflow configuration.
        user_prompt (str): The user prompt to process.

    Returns:
        dict: The result of processing the user prompt.
    """
    logs = []
    log_message(logs, f"Starting workflow with user prompt: {user_prompt}")
    workflow_config: WorkflowConfig = ctx.workflow_config
    recursion_limit = workflow_config.get("recursion_limit", DEFAULT_WORKFLOW_CONFIG["recursion_limit"])

    start_time = time.time()
    try:
        # Generate the graph from the workflow configuration
        graph = generate_graph_from_workflow(workflow_config, logs, ctx)

        # Execute the graph with the user prompt
        response = graph.invoke(input={"input": user_prompt}, config={"recursion_limit": recursion_limit})
    except Exception as e:
        log_message(logs, f"Error: running workflow: {str(e)} {repr(e)}")
        response = {"logs": logs}

    end_time = time.time()
    elapsed = end_time - start_time
    log_message(logs, f"Workflow processing time: {elapsed:.2f} seconds")

    for key, value in response.items():
        value_str = str(value)
        log_message(logs, f"State: {key}: {value_str[:100]}...")
    return response


async def embed(ctx: AppContext, file_paths: List[str]) -> dict:
    """
    Embeds the given files or folder using the embedding model.

    Args:
        ctx (AppContext): The application context containing the embedding model.
        file_paths (List[str]): A list of file paths or folders to embed.

    Returns:
        dict: The embedded text.
    """
    log_message([], f"Embedding files: {file_paths}")
    # Call the update_embeddings function to create or update embeddings
    return update_embeddings(file_paths=file_paths, ctx=ctx)


async def embed_visualize(ctx: AppContext, collection_name: str = None) -> str:
    """
    Visualizes the embeddings in the ChromaDB collection.

    Args:
        ctx (AppContext): The application context containing the embedding model.
        collection_name (str, optional): The name of the ChromaDB collection to visualize. If not provided, uses the default from workflow config.

    Returns:
        str: The path to the visualization image file.
    """
    workflow_config: WorkflowConfig = ctx.workflow_config
    workflow_collection_name = workflow_config.get("collection_name", DEFAULT_WORKFLOW_CONFIG["collection_name"])
    # Call the visualize_embeddings function to create the visualization
    return visualize(ctx, collection_name=(collection_name or workflow_collection_name))
