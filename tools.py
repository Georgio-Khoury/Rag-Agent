# Upgraded section inside tools.py
from typing import List
from ddgs import DDGS
import os
from google import genai

def generate_multiple_search_queries(user_query: str) -> List[str]:
    """Asks Gemini to break a multi-part user question down into a clean list of individual search queries."""
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    
    prompt = f"""You are an advanced search coordinator. Analyze the user's input. If it asks for multiple distinct things, break it down into a list of separate, punchy search engine queries (maximum 3).
    Format your output strictly as a comma-separated list. Do not include bullet points, numbers, or quotes.
    
    User Query: {user_query}
    Search Queries:"""
    
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )
    
    # Split by comma and strip whitespace to get a clean Python list
    queries = [q.strip() for q in response.text.split(",") if q.strip()]
    return queries

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