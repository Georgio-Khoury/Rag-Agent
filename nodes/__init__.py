# nodes/__init__.py
from typing import TypedDict, List, Dict, Any

class AgentState(TypedDict):
    query: str
    collection_name: str
    sub_queries: List[str]
    current_index: int
    current_sub_query: str
    target_source: str
    retrieved_chunks: List[str]
    is_pdf_sufficient: bool
    sub_query_results: List[Dict[str, Any]]
    final_response: str
    is_cache_hit: bool
    cached_context: str

# Import all nodes and routers so graph.py can access them cleanly
from .decomposition import decompose_user_query, advance_subquery, route_loop_check
from .router import route_subquery_source
from .cache import check_cache, append_cache_result, cache_store, route_cache_check
from .retrieval import retrieve_from_chroma, grade_retrieved_documents, route_after_grading
from .web_search import fallback_web_search
from .synthesis import save_pdf_result, generate_final_answer