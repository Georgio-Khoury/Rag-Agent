# graph.py
from langgraph.graph import StateGraph, START, END
from nodes import (
    AgentState,
    decompose_user_query,
    advance_subquery,
    route_loop_check,
    route_subquery_source,
    check_cache,
    route_cache_check,
    append_cache_result,
    retrieve_from_chroma,
    grade_retrieved_documents,
    route_after_grading,
    save_pdf_result,
    fallback_web_search,
    cache_store,
    generate_final_answer
)

def create_agent_graph():
    print("🏗️ Compiling Modular Semantic Caching & Routing Graph...")
    workflow = StateGraph(AgentState)
    
    # 1. Register Nodes
    workflow.add_node("decompose", decompose_user_query)
    workflow.add_node("advance", advance_subquery)
    workflow.add_node("route_source", route_subquery_source) # <-- Pre-filters document source
    workflow.add_node("check_cache", check_cache)
    workflow.add_node("append_cache", append_cache_result)
    workflow.add_node("retrieve", retrieve_from_chroma)
    workflow.add_node("grade", grade_retrieved_documents)
    workflow.add_node("save_pdf", save_pdf_result)
    workflow.add_node("web_search", fallback_web_search)
    workflow.add_node("cache_store", cache_store)
    workflow.add_node("generate", generate_final_answer)
    
    # 2. Main Entry Path
    workflow.add_edge(START, "decompose")
    workflow.add_edge("decompose", "advance")
    
   # 3. The Central Loop Router (Points to Cache Check first)
    workflow.add_conditional_edges(
        "advance",
        route_loop_check,
        {
            "loop": "check_cache",   # Check cache first for the sub-query!
            "finish": "generate"      # No more sub-queries: generate final answer
        }
    )
    
    # 4. The Cache Router
    workflow.add_conditional_edges(
        "check_cache",
        route_cache_check,
        {
            "append_cache": "append_cache", # HIT: Bypass RAG completely
            "retrieve": "route_source"      # MISS: Now route to source & pull from Chroma
        }
    )
    
    # 5. Route source
    workflow.add_edge("route_source","retrieve")
    
    # 6. RAG Pipeline (Cache Miss)
    workflow.add_edge("retrieve", "grade")
    workflow.add_conditional_edges(
        "grade",
        route_after_grading,
        {
            "save_pdf": "save_pdf",
            "web_search": "web_search"
        }
    )
    
    # 7. Store new findings in Cache
    workflow.add_edge("save_pdf", "cache_store")
    workflow.add_edge("web_search", "cache_store")
    
    # 8. Loop Back to Advance
    workflow.add_edge("append_cache", "advance")  # Cache hit loops back
    workflow.add_edge("cache_store", "advance")   # Cache miss finishes storing, loops back
    
    # 9. Close Graph
    workflow.add_edge("generate", END)
    
    return workflow.compile()