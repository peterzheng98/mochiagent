from duckduckgo_search import DDGS
import json

class WebSearchTool:
    def __init__(self):
        pass

    def search_and_reason(self, query, context):
        """
        Searches the web using DuckDuckGo and provides reasoning based on the query and context.
        """
        print(f"WebSearchTool: Searching for information related to: {query}")
        
        search_results = []
        try:
            with DDGS() as ddgs:
                # Search for the query, limit to 5 results
                results = list(ddgs.text(query, max_results=5))
                search_results = results
        except Exception as e:
            print(f"WebSearchTool: Error during search: {e}")
            return {
                "query": query,
                "reasoning": f"Search failed: {str(e)}",
                "search_results": []
            }

        # Synthesize a simple reasoning based on top results titles/snippets
        # In a full system, this would be fed into an LLM.
        summary = "Found relevant articles: " + "; ".join([r.get('title', 'No Title') for r in search_results])
        
        reasoning = (
            f"Based on web search results for '{query}', here is some context found:\n"
            f"{summary}\n\n"
            f"This information is being considered alongside the transformer context: {context}"
        )

        return {
            "query": query,
            "reasoning": reasoning,
            "search_results": search_results
        }
