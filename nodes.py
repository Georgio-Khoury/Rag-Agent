# nodes.py
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
    retrieved_chunks: List[str]
    is_pdf_sufficient: bool
    sub_query_results: List[Dict[str, Any]]
    final_response: str
    # Cache State
    is_cache_hit: bool
    cached_context: str

DB_PATH = "./chroma_db"
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
        print(f"🐌 CACHE MISS. Proceeding to PDF retrieval.")
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
    """Standard vector retrieval for a Cache Miss."""
    sub_query = state["current_sub_query"]
    collection_target = state.get("collection_name", "attention_is_all_you_need")
    
    client = get_gemini_client()
    response = client.models.embed_content(model="gemini-embedding-2", contents=sub_query)
    
    chroma_client = chromadb.PersistentClient(path=DB_PATH)
    try:
        collection = chroma_client.get_collection(name=collection_target)
        results = collection.query(query_embeddings=[response.embeddings[0].values], n_results=3)
        chunks = results['documents'][0] if results['documents'] else []
    except Exception:
        chunks = []
        
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
    existing.append({
        "sub_query": state["current_sub_query"],
        "source": "PDF Document",
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