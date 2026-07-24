# nodes/router.py
import os
import chromadb
from google import genai

MASTER_COLLECTION = "knowledge_base"
DB_PATH = "./chroma_db"

# ==========================================
# Prompt Templates
# ==========================================
ROUTER_PROMPT_TEMPLATE = """You are a document routing assistant. 
Given the user's sub-query, choose the most relevant document source from the list below based on their summaries.
Reply ONLY with the exact filename string from the list, nothing else.

Available Documents:
{menu_str}

Sub-Query: {sub_query}
"""

# ==========================================
# Helper Functions
# ==========================================
def get_gemini_client():
    return genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

# ==========================================
# Graph Nodes
# ==========================================
def route_subquery_source(state: dict):
    """Inspecting available document summaries in ChromaDB and picking the best file for the sub-query."""
    sub_query = state["current_sub_query"]
    print(f"🧭 [NODE] Routing Sub-Query to Correct Document Source...")
    
    chroma_client = chromadb.PersistentClient(path=DB_PATH)
    try:
        collection = chroma_client.get_collection(name=MASTER_COLLECTION)
        all_data = collection.get(include=["metadatas"])
        metadatas = all_data.get("metadatas", [])
        
        doc_map = {}
        for meta in metadatas:
            src = meta.get("source")
            summary = meta.get("summary", "No summary available.")
            if src and src not in doc_map:
                doc_map[src] = summary
        
        available_sources = list(doc_map.keys())
    except Exception:
        available_sources = []
        doc_map = {}
        
    if not available_sources:
        print("⚠️ [ROUTER] No documents found in database.")
        return {"target_source": None}

    if len(available_sources) == 1:
        chosen = available_sources[0]
        print(f"📌 [ROUTER] Only one source available, auto-selecting: {chosen}")
        return {"target_source": chosen}

    client = get_gemini_client()
    menu_str = "\n".join([f"- Filename: {src}\n  Summary: {summary}" for src, summary in doc_map.items()])
    prompt = ROUTER_PROMPT_TEMPLATE.format(menu_str=menu_str, sub_query=sub_query)

    response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
    chosen_source = response.text.strip()
    
    if chosen_source not in available_sources:
        chosen_source = available_sources[0]
        
    print(f"🎯 [ROUTER] Matched sub-query to document: {chosen_source}")
    return {"target_source": chosen_source}