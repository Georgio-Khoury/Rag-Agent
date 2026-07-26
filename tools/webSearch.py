from typing import List
from ddgs import DDGS
import os
from generateSearchQueries import generate_multiple_search_queries

def web_search_tool(query: str) -> str:
    """Dynamically generates multiple targeted search paths and aggregates all results cleanly."""
    # Step 1: Deconstruct the multi-part prompt into independent target searches
    search_list = generate_multiple_search_queries(query)
    print(f"🌐 [TOOL RUNNING] Deconstructed prompt into {len(search_list)} distinct searches: {search_list}")
    
    all_compiled_results = []
    
    # Step 2: Loop through each search query independently
    with DDGS() as ddgs:
        for search_phrase in search_list:
            print(f"   ├─ Executing sub-query: '{search_phrase}'")
            try:
                results = list(ddgs.text(search_phrase, max_results=2)) # 2 results per query to save space
                for res in results:
                    all_compiled_results.append(
                        f"Topic Search: {search_phrase}\nTitle: {res.get('title')}\nContent: {res.get('body')}\n"
                    )
            except Exception as sub_e:
                print(f"   ⚠️ Sub-query '{search_phrase}' failed: {sub_e}")
                
    if not all_compiled_results:
        return "No web search results could be retrieved."
        
    return "\n---\n".join(all_compiled_results)