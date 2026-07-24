import os
import hashlib
from typing import TypedDict, List, Dict, Any
import chromadb
from google import genai
from ddgs import DDGS

# ==========================================
# 1. State Definition
# ==========================================
class AgentState(TypedDict):
    query: str
    collection_name: str
    sub_queries: List[str]
    current_index: int
    current_sub_query: str
    target_source: str          # <-- Added to store pre-filtered document source
    retrieved_chunks: List[str]
    is_pdf_sufficient: bool
    sub_query_results: List[Dict[str, Any]]
    final_response: str
    # Cache State
    is_cache_hit: bool
    cached_context: str

DB_PATH = "./chroma_db"
MASTER_COLLECTION = "knowledge_base" # <-- Updated to master collection
CACHE_COLLECTION_NAME = "semantic_cache"

def get_gemini_client():
    return genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

# ==========================================
# 2. Graph Nodes
# ==========================================

def decompose_user_query(state: AgentState):
    """Decomposes complex queries and sets the loop iterator to -1."""
    query = state["query"]
    print(f"\n🧩 [NODE] Decomposing User Query: '{query}'")
    
    client = get_gemini_client()
    prompt = f"""You are a query decomposition assistant. Analyze the user query.
If it contains multiple distinct questions or topics, break it down into a list of standalone, focused sub-queries (maximum 3).
If it is already a single simple question, return it as a single item list.
Format strictly as a comma-separated list. No numbering, quotes, or markdown.

User Query: {query}
Sub-queries:"""

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

def advance_subquery(state: AgentState):
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

def route_subquery_source(state: AgentState):
    """Routing Node: Inspects available document summaries in ChromaDB and picks the best file for the sub-query."""
    sub_query = state["current_sub_query"]
    print(f"🧭 [NODE] Routing Sub-Query to Correct Document Source...")
    
    chroma_client = chromadb.PersistentClient(path=DB_PATH)
    try:
        collection = chroma_client.get_collection(name=MASTER_COLLECTION)
        all_data = collection.get(include=["metadatas"])
        metadatas = all_data.get("metadatas", [])
        
        # Map unique sources to their generated summaries
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

    # If only one source exists, pick it automatically without an extra LLM call
    if len(available_sources) == 1:
        chosen = available_sources[0]
        print(f"📌 [ROUTER] Only one source available, auto-selecting: {chosen}")
        return {"target_source": chosen}

    # Ask Gemini to map the sub-query to the best matching file summary
    client = get_gemini_client()
    menu_str = "\n".join([f"- Filename: {src}\n  Summary: {summary}" for src, summary in doc_map.items()])
    
    prompt = f"""You are a document routing assistant. 
Given the user's sub-query, choose the most relevant document source from the list below based on their summaries.
Reply ONLY with the exact filename string from the list, nothing else.

Available Documents:
{menu_str}

Sub-Query: {sub_query}
"""

    response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
    chosen_source = response.text.strip()
    
    if chosen_source not in available_sources:
        chosen_source = available_sources[0] # Fallback safety check
        
    print(f"🎯 [ROUTER] Matched sub-query to document: {chosen_source}")
    return {"target_source": chosen_source}

def check_cache(state: AgentState):
    """Checks the semantic cache for the specific active sub-query."""
    sub_query = state["current_sub_query"]
    print(f"🧠 [NODE] Checking Semantic Cache for '{sub_query}'...")
    
    client = get_gemini_client()
    response = client.models.embed_content(model="gemini-embedding-2", contents=sub_query)
    query_vector = response.embeddings[0].values
    
    chroma_client = chromadb.PersistentClient(path=DB_PATH)
    cache_collection = chroma_client.get_or_create_collection(name=CACHE_COLLECTION_NAME)
    
    results = cache_collection.query(query_embeddings=[query_vector], n_results=1)
    
    if results['distances'] and results['distances'][0] and results['distances'][0][0] < 0.25:
        cached_text = results['documents'][0][0]
        print(f"⚡ CACHE HIT! Found identical previous question.")
        return {"is_cache_hit": True, "cached_context": cached_text}
    else:
        print(f"🐌 CACHE MISS. Proceeding to vector retrieval.")
        return {"is_cache_hit": False, "cached_context": ""}

def append_cache_result(state: AgentState):
    """Appends the cached answer to results, bypassing RAG."""
    existing_results = list(state.get("sub_query_results", []))
    existing_results.append({
        "sub_query": state["current_sub_query"],
        "source": "Semantic Cache (Instant)",
        "context": state["cached_context"]
    })
    return {"sub_query_results": existing_results}

def retrieve_from_chroma(state: AgentState):
    """Metadata-filtered vector retrieval for a Cache Miss from the master collection."""
    sub_query = state["current_sub_query"]
    target_source = state.get("target_source")
    
    client = get_gemini_client()
    response = client.models.embed_content(model="gemini-embedding-2", contents=sub_query)
    
    chroma_client = chromadb.PersistentClient(path=DB_PATH)
    try:
        collection = chroma_client.get_collection(name=MASTER_COLLECTION)
        
        query_args = {
            "query_embeddings": [response.embeddings[0].values],
            "n_results": 3
        }
        
        # Apply metadata filtering if a source was targeted by the router
        if target_source:
            query_args["where"] = {"source": target_source}
            
        results = collection.query(**query_args)
        chunks = results['documents'][0] if results['documents'] else []
    except Exception as e:
        print(f"❌ Retrieval error: {e}")
        chunks = []
        
    print(f"🔍 [NODE] Retrieved {len(chunks)} chunks from source: {target_source}")
    return {"retrieved_chunks": chunks}

def grade_retrieved_documents(state: AgentState):
    """Grades the retrieved PDF chunks."""
    if not state["retrieved_chunks"]:
        return {"is_pdf_sufficient": False}
        
    client = get_gemini_client()
    joined_context = "\n---\n".join(state["retrieved_chunks"])
    prompt = f"""Determine if the context answers this sub-query: {state['current_sub_query']}
Context: {joined_context}
Reply strictly YES or NO."""

    response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
    is_sufficient = "YES" in response.text.strip().upper()
    print(f"⚖️ [NODE] PDF Sufficient? -> {is_sufficient}")
    return {"is_pdf_sufficient": is_sufficient}

def save_pdf_result(state: AgentState):
    existing = list(state.get("sub_query_results", []))
    target_src = state.get("target_source", "PDF Document")
    existing.append({
        "sub_query": state["current_sub_query"],
        "source": f"PDF ({target_src})",
        "context": "\n".join(state["retrieved_chunks"])
    })
    return {"sub_query_results": existing}

def fallback_web_search(state: AgentState):
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

def cache_store(state: AgentState):
    """Saves newly minted context to ChromaDB for future users."""
    sub_query = state["current_sub_query"]
    latest_result = state["sub_query_results"][-1]["context"]
    print(f"📥 [NODE] Storing result into Semantic Cache...")
    
    client = get_gemini_client()
    response = client.models.embed_content(model="gemini-embedding-2", contents=sub_query)
    
    chroma_client = chromadb.PersistentClient(path=DB_PATH)
    cache_col = chroma_client.get_or_create_collection(name=CACHE_COLLECTION_NAME)
    
    doc_id = hashlib.md5(sub_query.encode()).hexdigest()
    cache_col.upsert(
        ids=[doc_id],
        embeddings=[response.embeddings[0].values],
        documents=[latest_result]
    )
    return {}

def generate_final_answer(state: AgentState):
    print("\n✍️ [NODE] Synthesizing Final Answer...")
    contexts = [f"--- Sub-Query: {r['sub_query']} (Source: {r['source']}) ---\n{r['context']}" for r in state.get("sub_query_results", [])]
    joined_evidence = "\n\n".join(contexts)
    prompt = f"""Synthesize a complete answer for the user query using the gathered evidence.
User Query: {state['query']}
Evidence:{joined_evidence}"""

    client = get_gemini_client()
    response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
    return {"final_response": response.text}

# ==========================================
# 3. Routers
# ==========================================
def route_loop_check(state: AgentState) -> str:
    if state["current_index"] < len(state["sub_queries"]):
        return "loop"
    return "finish"

def route_cache_check(state: AgentState) -> str:
    if state["is_cache_hit"]:
        return "append_cache"
    return "retrieve"

def route_after_grading(state: AgentState) -> str:
    if state["is_pdf_sufficient"]:
        return "save_pdf"
    return "web_search"