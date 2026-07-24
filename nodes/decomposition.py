# nodes/decomposition.py
import os
from google import genai

# ==========================================
# Prompt Templates
# ==========================================
DECOMPOSITION_PROMPT_TEMPLATE = """You are a query decomposition assistant. Analyze the user query.
If it contains multiple distinct questions or topics, break it down into a list of standalone, focused sub-queries (maximum 3).
If it is already a single simple question, return it as a single item list.
Format strictly as a comma-separated list. No numbering, quotes, or markdown.

User Query: {query}
Sub-queries:"""

# ==========================================
# Helper Functions
# ==========================================
def get_gemini_client():
    return genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

# ==========================================
# Graph Nodes & Routers
# ==========================================
def decompose_user_query(state: dict):
    """Decomposes complex queries and sets the loop iterator to -1."""
    query = state["query"]
    print(f"\n🧩 [NODE] Decomposing User Query: '{query}'")
    
    client = get_gemini_client()
    prompt = DECOMPOSITION_PROMPT_TEMPLATE.format(query=query)

    response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
    sub_queries = [q.strip() for q in response.text.split(",") if q.strip()]
    if not sub_queries:
        sub_queries = [query]
        
    print(f"📋 Generated {len(sub_queries)} sub-queries: {sub_queries}")
    
    return {
        "sub_queries": sub_queries,
        "current_index": -1,  # Start at -1 so advance bumps it to 0
        "sub_query_results": []
    }

def advance_subquery(state: dict):
    """Acts as the iterator. Increments index and sets the active sub-query."""
    next_idx = state.get("current_index", -1) + 1
    sub_queries = state["sub_queries"]
    
    if next_idx < len(sub_queries):
        next_sub = sub_queries[next_idx]
        print(f"\n🔄 [LOOP] Advancing to Sub-query [{next_idx + 1}/{len(sub_queries)}]: '{next_sub}'")
        return {"current_index": next_idx, "current_sub_query": next_sub}
    else:
        print("\n✅ All sub-queries processed. Proceeding to final synthesis.")
        return {"current_index": next_idx}

def route_loop_check(state: dict) -> str:
    """Conditional router that checks if there are more sub-queries to process."""
    if state["current_index"] < len(state["sub_queries"]):
        return "loop"
    return "finish"