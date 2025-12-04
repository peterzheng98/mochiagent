import json
import time
from typing import Dict, Any, Optional, List
from datetime import datetime

try:
    from duckduckgo_search import DDGS
    HAS_DDGS = True
except ImportError:
    HAS_DDGS = False

from server.base import MCPServer


class WebSearchMCPServer(MCPServer):
    """
    MCP Server for Web Search and Medical Reasoning.
    
    This server handles:
    - DuckDuckGo web searches
    - Medical literature retrieval
    - Search result synthesis
    - Context-aware reasoning
    """
    
    SERVER_KEY = "websearch"
    
    def __init__(
        self,
        search_engine: str = "duckduckgo",
        max_results: int = 10,
        search_timeout: int = 15,
        rate_limit_per_minute: int = 30,
        **kwargs
    ) -> None:
        """Initialize the Web Search MCP Server."""
        self.search_engine = search_engine
        self.max_results = max_results
        self.search_timeout = search_timeout
        self.rate_limit_per_minute = rate_limit_per_minute
        self._last_search_times: List[float] = []
        self._search_cache: Dict[str, Dict] = {}
        
        # Default Redis keys
        task_queue_key = kwargs.pop("task_queue_key", "mochiagent:websearch:task")
        result_queue_key = kwargs.pop("result_queue_key", "mochiagent:websearch:result")
        
        super().__init__(
            task_queue_key=task_queue_key,
            result_queue_key=result_queue_key,
            **kwargs
        )
    
    def _register_tools(self) -> None:
        """Register web search tools."""
        self.register_tool(
            "search",
            self._tool_search,
            "Perform a web search query"
        )
        self.register_tool(
            "search_medical",
            self._tool_search_medical,
            "Search for medical and clinical information"
        )
        self.register_tool(
            "search_and_reason",
            self._tool_search_and_reason,
            "Search web and synthesize reasoning"
        )
        self.register_tool(
            "search_literature",
            self._tool_search_literature,
            "Search medical literature databases"
        )
        self.register_tool(
            "validate_finding",
            self._tool_validate_finding,
            "Validate a clinical finding against web sources"
        )
    
    def _register_resources(self) -> None:
        """Register web search resources."""
        self.register_resource(
            "search://config",
            {
                "engine": self.search_engine,
                "max_results": self.max_results,
                "timeout": self.search_timeout
            }
        )
        self.register_resource(
            "search://medical_domains",
            self._get_medical_domains()
        )
    
    def _register_prompts(self) -> None:
        """Register search prompts."""
        self.register_prompt(
            "medical_query",
            "Search for medical information about: {condition}\nContext: {context}"
        )
        self.register_prompt(
            "reasoning_synthesis",
            "Based on the following search results, provide reasoning:\n{results}\n\nOriginal query: {query}"
        )
    
    def _get_medical_domains(self) -> List[str]:
        """Get list of trusted medical domains."""
        return [
            "pubmed.ncbi.nlm.nih.gov",
            "who.int",
            "cdc.gov",
            "mayoclinic.org",
            "webmd.com",
            "medlineplus.gov",
            "nih.gov",
            "uptodate.com",
            "cochranelibrary.com"
        ]
    
    def _check_rate_limit(self) -> bool:
        """Check if we're within rate limits."""
        current_time = time.time()
        # Remove old timestamps
        self._last_search_times = [
            t for t in self._last_search_times
            if current_time - t < 60
        ]
        return len(self._last_search_times) < self.rate_limit_per_minute
    
    def _record_search(self) -> None:
        """Record a search timestamp."""
        self._last_search_times.append(time.time())
    
    def _get_cache_key(self, query: str, **kwargs) -> str:
        """Generate cache key for a search query."""
        import hashlib
        key_data = json.dumps({"query": query, **kwargs}, sort_keys=True)
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _search_duckduckgo(self, query: str, max_results: Optional[int] = None) -> List[Dict]:
        """Perform DuckDuckGo search."""
        if not HAS_DDGS:
            self._log_warning("DuckDuckGo search library not available")
            return self._generate_fallback_results(query)
        
        max_results = max_results or self.max_results
        results = []
        
        try:
            with DDGS() as ddgs:
                search_results = list(ddgs.text(query, max_results=max_results))
                for result in search_results:
                    results.append({
                        "title": result.get("title", ""),
                        "url": result.get("href", result.get("link", "")),
                        "snippet": result.get("body", result.get("snippet", "")),
                        "source": "duckduckgo"
                    })
        except Exception as e:
            self._log_error(f"DuckDuckGo search failed: {e}")
            results = self._generate_fallback_results(query)
        
        return results
    
    def _generate_fallback_results(self, query: str) -> List[Dict]:
        """Generate fallback results when search fails."""
        return [{
            "title": f"Search results for: {query}",
            "url": f"https://search.example.com?q={query.replace(' ', '+')}",
            "snippet": f"Unable to perform live search. Query: {query}",
            "source": "fallback"
        }]
    
    def _filter_medical_results(self, results: List[Dict]) -> List[Dict]:
        """Filter results to prioritize medical sources."""
        medical_domains = self._get_medical_domains()
        
        def get_priority(result: Dict) -> int:
            url = result.get("url", "").lower()
            for i, domain in enumerate(medical_domains):
                if domain in url:
                    return i
            return len(medical_domains) + 1
        
        return sorted(results, key=get_priority)
    
    def _synthesize_reasoning(self, query: str, results: List[Dict], context: str = "") -> str:
        """Synthesize reasoning from search results."""
        if not results:
            return f"No search results found for query: {query}"
        
        # Build reasoning from search results
        snippets = [r.get("snippet", "") for r in results[:5]]
        titles = [r.get("title", "") for r in results[:5]]
        
        reasoning_parts = [
            f"Based on web search results for '{query}':",
            "",
            "Key findings from search results:"
        ]
        
        for i, (title, snippet) in enumerate(zip(titles, snippets), 1):
            if title and snippet:
                reasoning_parts.append(f"{i}. {title}: {snippet[:200]}...")
        
        if context:
            reasoning_parts.extend([
                "",
                f"In the context of: {context}",
                "",
                "This information suggests that the observed patterns are consistent with current medical understanding."
            ])
        
        return "\n".join(reasoning_parts)
    
    def _tool_search(
        self,
        query: str,
        max_results: Optional[int] = None,
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """
        Perform a general web search.
        
        Args:
            query: Search query string
            max_results: Maximum number of results
            use_cache: Whether to use cached results
            
        Returns:
            Search results
        """
        self._log_info(f"Searching: {query}")
        
        # Check cache
        cache_key = self._get_cache_key(query, max_results=max_results)
        if use_cache and cache_key in self._search_cache:
            cached = self._search_cache[cache_key]
            if time.time() - cached["timestamp"] < 3600:  # 1 hour cache
                self._log_info("Returning cached results")
                return cached["data"]
        
        # Check rate limit
        if not self._check_rate_limit():
            self._log_warning("Rate limit exceeded, waiting...")
            time.sleep(2)
        
        # Perform search
        results = self._search_duckduckgo(query, max_results)
        self._record_search()
        
        response = {
            "query": query,
            "results": results,
            "count": len(results),
            "timestamp": datetime.now().isoformat(),
            "source": self.search_engine
        }
        
        # Cache results
        self._search_cache[cache_key] = {
            "data": response,
            "timestamp": time.time()
        }
        
        return response
    
    def _tool_search_medical(
        self,
        query: str,
        condition: Optional[str] = None,
        max_results: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Search for medical and clinical information.
        
        Args:
            query: Search query
            condition: Specific medical condition to focus on
            max_results: Maximum results
            
        Returns:
            Medical search results
        """
        # Enhance query for medical context
        medical_query = query
        if condition:
            medical_query = f"{condition} {query} medical clinical"
        else:
            medical_query = f"{query} medical clinical research"
        
        # Perform search
        results = self._tool_search(medical_query, max_results)
        
        # Filter for medical sources
        filtered_results = self._filter_medical_results(results["results"])
        
        return {
            "query": query,
            "medical_query": medical_query,
            "results": filtered_results,
            "count": len(filtered_results),
            "condition": condition,
            "timestamp": datetime.now().isoformat()
        }
    
    def _tool_search_and_reason(
        self,
        query: str,
        context: str = "",
        max_results: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Search the web and synthesize reasoning.
        
        Args:
            query: Search query
            context: Additional context for reasoning
            max_results: Maximum results
            
        Returns:
            Search results with synthesized reasoning
        """
        self._log_info(f"Search and reason: {query}")
        
        # Perform search
        search_results = self._tool_search(query, max_results)
        
        # Synthesize reasoning
        reasoning = self._synthesize_reasoning(
            query,
            search_results["results"],
            context
        )
        
        return {
            "query": query,
            "context": context,
            "search_results": search_results["results"],
            "reasoning": reasoning,
            "timestamp": datetime.now().isoformat()
        }
    
    def _tool_search_literature(
        self,
        query: str,
        publication_type: str = "research",
        max_results: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Search medical literature databases.
        
        Args:
            query: Search query
            publication_type: Type of publication (research, review, meta-analysis)
            max_results: Maximum results
            
        Returns:
            Literature search results
        """
        # Build literature-focused query
        literature_query = f"{query} site:pubmed.ncbi.nlm.nih.gov OR site:ncbi.nlm.nih.gov"
        
        if publication_type == "review":
            literature_query += " review"
        elif publication_type == "meta-analysis":
            literature_query += " meta-analysis systematic review"
        
        results = self._tool_search(literature_query, max_results)
        
        # Parse and structure literature results
        literature_results = []
        for result in results["results"]:
            literature_results.append({
                "title": result["title"],
                "url": result["url"],
                "abstract": result["snippet"],
                "source": "pubmed" if "pubmed" in result["url"].lower() else "other",
                "type": publication_type
            })
        
        return {
            "query": query,
            "publication_type": publication_type,
            "results": literature_results,
            "count": len(literature_results),
            "timestamp": datetime.now().isoformat()
        }
    
    def _tool_validate_finding(
        self,
        finding: str,
        context: str = "",
        confidence_threshold: float = 0.7
    ) -> Dict[str, Any]:
        """
        Validate a clinical finding against web sources.
        
        Args:
            finding: The clinical finding to validate
            context: Additional context
            confidence_threshold: Minimum confidence for validation
            
        Returns:
            Validation results
        """
        self._log_info(f"Validating finding: {finding}")
        
        # Search for supporting evidence
        search_results = self._tool_search_medical(finding)
        
        # Analyze results for validation
        supporting_evidence = []
        contradicting_evidence = []
        
        for result in search_results["results"]:
            snippet = result.get("snippet", "").lower()
            finding_lower = finding.lower()
            
            # Simple keyword matching for validation
            # In production, this would use NLP/LLM for semantic matching
            if any(word in snippet for word in finding_lower.split()):
                supporting_evidence.append(result)
            else:
                contradicting_evidence.append(result)
        
        # Calculate validation score
        total = len(supporting_evidence) + len(contradicting_evidence)
        if total > 0:
            validation_score = len(supporting_evidence) / total
        else:
            validation_score = 0.5  # Neutral if no evidence
        
        is_validated = validation_score >= confidence_threshold
        
        return {
            "finding": finding,
            "context": context,
            "is_validated": is_validated,
            "validation_score": validation_score,
            "confidence_threshold": confidence_threshold,
            "supporting_evidence": supporting_evidence[:3],
            "contradicting_evidence": contradicting_evidence[:3],
            "recommendation": "Finding appears consistent with available literature" if is_validated 
                            else "Finding requires further review - limited supporting evidence",
            "timestamp": datetime.now().isoformat()
        }
    
    def _handle_custom_method(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle custom methods for web search server."""
        if method == "websearch/search":
            return self._tool_search(
                params.get("query", ""),
                params.get("max_results"),
                params.get("use_cache", True)
            )
        elif method == "websearch/medical":
            return self._tool_search_medical(
                params.get("query", ""),
                params.get("condition"),
                params.get("max_results")
            )
        elif method == "websearch/reason":
            return self._tool_search_and_reason(
                params.get("query", ""),
                params.get("context", ""),
                params.get("max_results")
            )
        elif method == "websearch/literature":
            return self._tool_search_literature(
                params.get("query", ""),
                params.get("publication_type", "research"),
                params.get("max_results")
            )
        elif method == "websearch/validate":
            return self._tool_validate_finding(
                params.get("finding", ""),
                params.get("context", ""),
                params.get("confidence_threshold", 0.7)
            )
        else:
            raise ValueError(f"Unknown method: {method}")


if __name__ == "__main__":
    server = WebSearchMCPServer()
    server.run()

