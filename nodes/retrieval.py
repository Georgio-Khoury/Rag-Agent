# nodes/retrieval.py
import os
import chromadb
from google import genai

MASTER_COLLECTION = "knowledge_base"
DB_PATH = "./chroma_db"

# ==========================================
# Prompt Templates
# ==========================================
GRADER_PROMPT_TEMPLATE = """Determine if the context answers this sub-query: {sub_query}
Context: {joined_context}
Reply strictly YES or NO."""

# ==========================================
# Helper Functions
# ==========================================
def get_gemini_client():
    return genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

# ==========================================
# Graph Nodes & Routers
# ==========================================
def retrieve_from_chroma(state: dict):
    """Metadata-filtered vector retrieval from the master collection."""
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
        
        if target_source:
            query_args["where"] = {"source": target_source}
            
        results = collection.query(**query_args)
        chunks = results['documents'][0] if results['documents'] else []
    except Exception as e:
        print(f"❌ Retrieval error: {e}")
        chunks = []
        
    print(f"🔍 [NODE] Retrieved {len(chunks)} chunks from source: {target_source}")
    return {"retrieved_chunks": chunks}

def grade_retrieved_documents(state: dict):
    """Grades the retrieved PDF chunks."""
    if not state["retrieved_chunks"]:
        return {"is_pdf_sufficient": False}
        
    client = get_gemini_client()
    joined_context = "\n---\n".join(state["retrieved_chunks"])
    prompt = GRADER_PROMPT_TEMPLATE.format(sub_query=state['current_sub_query'], joined_context=joined_context)

    response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
    is_sufficient = "YES" in response.text.strip().upper()
    print(f"⚖️ [NODE] PDF Sufficient? -> {is_sufficient}")
    return {"is_pdf_sufficient": is_sufficient}

def route_after_grading(state: dict) -> str:
    """Conditional router based on PDF content sufficiency."""
    if state["is_pdf_sufficient"]:
        return "save_pdf"
    return "web_search"