from langchain.tools import tool
from duckduckgo_search import DDGS

@tool
def web_search(query: str, max_results: int = 5) -> str:
    """
    Performs a web search using DuckDuckGo and returns the results.

    Args:
        query (str): The search query.
        max_results (int): The maximum number of results to return. Defaults to 5.

    Returns:
        str: A formatted string containing the search results.
    """
    try:
        with DDGS() as ddgs:
            results = ddgs.text(query, region="wt-wt", safesearch="Moderate", max_results=max_results)
            output = []
            for i, result in enumerate(results):
                title = result.get("title", "No title")
                body = result.get("body", "")
                href = result.get("href", "")
                output.append(f"{i+1}. {title}\n{body}\n{href}\n")
            return "\n".join(output) if output else "No results found."
    except Exception as e:
        return f"Error: during web search: {e}"
