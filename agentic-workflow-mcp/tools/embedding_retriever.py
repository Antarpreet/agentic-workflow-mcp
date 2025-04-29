from langchain.tools import tool
from mcp.server.fastmcp import Context

from core.log import log_message

@tool
def retrieve_embeddings(input: str, logs: list = [], ctx: Context = None) -> str:
    """
    Answers a question using the Retrieval chain and local ChromaDB vectors.

    Args:
        input (str): The question to answer.
        logs (list): A list to store log messages.
        ctx (Context, optional): The MCP context containing application resources.

    Returns:
        str: The answer from the Retrieval chain.
    """
    log_message(logs, f"Retrieval input: {input}")
    retrieval_chain = ctx.request_context.lifespan_context.retrieval_chain
    response = retrieval_chain.invoke({"input": input})
    log_message(logs, f"Retrieval response: {response}")
    return response["result"] if isinstance(response, dict) and "result" in response else str(response)
