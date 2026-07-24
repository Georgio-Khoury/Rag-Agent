# nodes/cache.py
import os
import hashlib
import chromadb
from google import genai

DB_PATH = "./chroma_db"
CACHE_COLLECTION_NAME = "semantic_cache"

def get_gemini_client():
    return genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

def check_cache(state: dict):
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

def append_cache_result(state: dict):
    """Appends the cached answer to results, bypassing RAG."""
    existing_results = list(state.get("sub_query_results", []))
    existing_results.append({
        "sub_query": state["current_sub_query"],
        "source": "Semantic Cache (Instant)",
        "context": state["cached_context"]
    })
    return {"sub_query_results": existing_results}

def cache_store(state: dict):
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

def route_cache_check(state: dict) -> str:
    """Conditional router determining cache hit or miss."""
    if state["is_cache_hit"]:
        return "append_cache"
    return "retrieve"