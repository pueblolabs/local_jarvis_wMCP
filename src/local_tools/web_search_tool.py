from duckduckgo_search import DDGS
from agents import function_tool
from . import register_tool

@register_tool
@function_tool
def search_web(query: str, num_results: int = 3) -> list[dict]:
    """
    Searches the web and returns a list of results as structured data.
    
    Args:
        query: The search query.
        num_results: The number of results to return.
    
    Returns:
        A list of dictionaries, each with 'title', 'href', and 'body' keys.
    """
    try:
        with DDGS() as ddgs:
            results = [r for r in ddgs.text(query, max_results=num_results)]
            # We only pass title, href, and body to the model to avoid malicious content
            return [{"title": r['title'], "href": r['href'], "body": r['body']} for r in results]
    except Exception as e:
        return [{"error": f"An error occurred during web search: {e}"}]
