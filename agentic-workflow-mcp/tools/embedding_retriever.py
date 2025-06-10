from typing import List, Union, Any

from langchain.tools import tool

from core.embedding import update_embeddings
from core.log import log_message

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
