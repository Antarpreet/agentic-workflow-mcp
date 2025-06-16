"""
Qdrant Client Configuration
==========================

To use the Qdrant client securely, set the following environment variables for your MCP server:

- QDRANT_URL: The URL of your Qdrant instance (e.g., https://your-qdrant-url)
- QDRANT_API_KEY: The API key for your Qdrant instance

You can set these in your `.vscode/mcp.json` as follows:

{
    "servers": {
        "Agentic Workflow": {
            ...
            "env": {
                "QDRANT_URL": "https://your-qdrant-url",
                "QDRANT_API_KEY": "your-api-key"
            }
        }
    }
}

Or set them in your shell before starting VS Code:

export QDRANT_URL="https://your-qdrant-url"
export QDRANT_API_KEY="your-api-key"

This ensures your credentials are not hardcoded in source files and are managed securely.
"""

from typing import List, Union, Any
import os

from langchain.tools import tool
from qdrant_client import QdrantClient

from core.embedding import update_embeddings
from core.log import log_message

# Initialize Qdrant client
qdrant_url = os.getenv("QDRANT_URL")
qdrant_api_key = os.getenv("QDRANT_API_KEY")

qdrant_client = QdrantClient(
    url=qdrant_url,
    api_key=qdrant_api_key,
)


def ensure_collection_exists(collection_name: str, vector_size: int = 1536, distance: str = "Cosine"):
    """
    Ensures the specified collection exists in Qdrant. Creates it if it does not exist.
    Args:
        collection_name (str): Name of the Qdrant collection.
        vector_size (int): Size of the vectors for the collection.
        distance (str): Distance metric for the collection (e.g., "Cosine").
    """
    collections = qdrant_client.get_collections()
    if collection_name not in [c.name for c in collections.collections]:
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config={"size": vector_size, "distance": distance}
        )
        print(f"Created '{collection_name}' collection")
    else:
        print(f"'{collection_name}' collection already exists.")

@tool
def retrieve_embeddings(input: str, logs: list = [], retrieval_chain: Any = None, app_ctx: Any = None) -> str:
    """
    Answers a question using the Retrieval chain and local ChromaDB vectors.

    Args:
        input (str): The question to answer.
        logs (list): A list to store log messages.
        retrieval_chain (Any, optional): The Retrieval chain to use. If not provided, it will be retrieved from the context.
        app_ctx (Any, optional): The MCP context containing application resources.

    Returns:
        str: The answer from the Retrieval chain.
    """
    log_message(logs, f"Retrieval input: {input}")
    if retrieval_chain:
        local_retrieval_chain = retrieval_chain
    else:
        local_retrieval_chain = app_ctx.retrieval_chain
    response = local_retrieval_chain.invoke({"input": input})
    log_message(logs, f"Retrieval response: {response}")
    return response["result"] if isinstance(response, dict) and "result" in response else str(response)


@tool
def modify_embeddings(file_paths: Union[str, List[str]] = [], vectorstore: Any = None, collection_name: str = None, use_git_ignore: bool = False, exclude_file_paths: List[str] = None, app_ctx: Any = None) -> dict:
    """
    Updates the embeddings for the specified files.

    Args:
        file_paths (Union[str, List[str]]): A single file path or a list of file paths to update embeddings for.
        vectorstore (Any, optional): The vector store to use. If not provided, it will be retrieved from the context.
        collection_name (str, optional): The name of the collection to use. If not provided, it will be retrieved from the context. Do not provide this if you want to use the default collection name.
        use_git_ignore (bool, optional): If True, skips files that are ignored by .gitignore.
        exclude_file_paths (List[str], optional): A list of file paths to exclude from the update.
        app_ctx (Any, optional): The MCP context containing application resources.

    Returns:
        dict: Information about the update operation.
    """
    log_message([], f"Updating embeddings for files: {file_paths}")
    # Call the update_embeddings function to create or update embeddings
    return update_embeddings(file_paths=file_paths, app_ctx=app_ctx, vectorstore=vectorstore, collection_name=collection_name, use_git_ignore=use_git_ignore, exclude_file_paths=exclude_file_paths)

# Tool to store vectors in Qdrant
@tool
def store_vectors_in_qdrant(vectors: list, payloads: list = None, collection_name: str = "codebase") -> str:
    """
    Stores a list of vectors (and optional payloads) in the specified Qdrant collection.
    Args:
        vectors (list): List of vectors to store.
        payloads (list, optional): List of payloads (dicts) to store with vectors.
        collection_name (str): Name of the Qdrant collection.
    Returns:
        str: Status message.
    """
    ensure_collection_exists(collection_name)
    qdrant_client.upsert(
        collection_name=collection_name,
        points=[{"id": i, "vector": v, "payload": p if payloads else None} for i, (v, p) in enumerate(zip(vectors, payloads or [{}]*len(vectors)))]
    )
    return f"Stored {len(vectors)} vectors in '{collection_name}' collection."

# Tool to retrieve vectors from Qdrant
@tool
def retrieve_vectors_from_qdrant(query_vector: list, top_k: int = 5, collection_name: str = "codebase") -> list:
    """
    Retrieves the top_k most similar vectors from the specified Qdrant collection.
    Args:
        query_vector (list): The query vector.
        top_k (int): Number of results to return.
        collection_name (str): Name of the Qdrant collection.
    Returns:
        list: List of retrieved points (dicts).
    """
    ensure_collection_exists(collection_name)
    search_result = qdrant_client.search(
        collection_name=collection_name,
        query_vector=query_vector,
        limit=top_k
    )
    return [point.dict() for point in search_result]
