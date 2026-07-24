# nodes/synthesis.py
import os
from google import genai

# ==========================================
# Prompt Templates
# ==========================================
SYNTHESIS_PROMPT_TEMPLATE = """Synthesize a complete answer for the user query using the gathered evidence.
User Query: {query}
Evidence:{joined_evidence}"""

# ==========================================
# Helper Functions
# ==========================================
def get_gemini_client():
    return genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

# ==========================================
# Graph Nodes
# ==========================================
def save_pdf_result(state: dict):
    """Saves successfully verified PDF retrieval results to state."""
    existing = list(state.get("sub_query_results", []))
    target_src = state.get("target_source", "PDF Document")
    existing.append({
        "sub_query": state["current_sub_query"],
        "source": f"PDF ({target_src})",
        "context": "\n".join(state["retrieved_chunks"])
    })
    return {"sub_query_results": existing}

def generate_final_answer(state: dict):
    """Synthesizes the final answer using all accumulated sub-query context."""
    print("\n✍️ [NODE] Synthesizing Final Answer...")
    contexts = [f"--- Sub-Query: {r['sub_query']} (Source: {r['source']}) ---\n{r['context']}" for r in state.get("sub_query_results", [])]
    joined_evidence = "\n\n".join(contexts)
    
    prompt = SYNTHESIS_PROMPT_TEMPLATE.format(query=state['query'], joined_evidence=joined_evidence)

    client = get_gemini_client()
    response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
    return {"final_response": response.text}