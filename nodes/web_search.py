# nodes/web_search.py
from ddgs import DDGS

def fallback_web_search(state: dict):
    """Executes a web search fallback when document context is insufficient."""
    sub_query = state["current_sub_query"]
    print(f"🌐 [NODE] Executing Web Search for: '{sub_query}'...")
    
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(sub_query, max_results=3))
            compiled = "\n".join([f"Title: {r.get('title')}\nSnippet: {r.get('body')}" for r in results]) if results else "No web results."
    except Exception as e:
        compiled = f"Web search error: {e}"

    existing = list(state.get("sub_query_results", []))
    existing.append({
        "sub_query": sub_query,
        "source": "Live Web Search",
        "context": compiled
    })
    return {"sub_query_results": existing}